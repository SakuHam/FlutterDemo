// lib/ai/agent.dart
import 'dart:math' as math;
import '../engine/raycast.dart';
import 'nn_helper.dart' as nn;                 // helper (unchanged)
import '../engine/types.dart' as et;
import '../engine/game_engine.dart' as eng;

/* -------------------------------------------------------------------------- */
/*                                INTENT SET                                  */
/* -------------------------------------------------------------------------- */

enum Intent { hover, goLeft, goRight, descendSlow, brakeUp, brakeLeft, brakeRight }

const List<String> kIntentNames = [
  'hover',       // 0
  'goLeft',      // 1
  'goRight',     // 2
  'descendSlow', // 3
  'brakeUp',     // 4
  'brakeLeft',   // 5
  'brakeRight',  // 6
];

int intentToIndex(Intent i) => i.index;
Intent indexToIntent(int k) => Intent.values[k.clamp(0, Intent.values.length - 1)];

double _clip(double x, double a, double b) => x < a ? a : (x > b ? b : x);

/* -------------------------------------------------------------------------- */
/*                               RUNNING  NORM                                */
/* -------------------------------------------------------------------------- */

class RunningNorm {
  int dim;
  List<double> mean;
  List<double> var_; // variance, not std
  double momentum;
  bool inited;
  final double eps;

  RunningNorm(this.dim, {this.momentum = 0.995, this.eps = 1e-6})
      : mean = List<double>.filled(dim, 0.0),
        var_ = List<double>.filled(dim, 1.0),
        inited = false;

  void reset({double initVar = 1.0}) {
    for (int i = 0; i < dim; i++) {
      mean[i] = 0.0;
      var_[i] = initVar;
    }
    inited = false;
  }

  void _resizeTo(int newDim) {
    dim = newDim;
    mean = List<double>.filled(dim, 0.0);
    var_  = List<double>.filled(dim, 1.0);
    inited = false;
  }

  void observe(List<double> x) {
    if (x.length != dim) {
      _resizeTo(x.length);
    }
    if (!inited) {
      for (int i = 0; i < dim; i++) {
        mean[i] = x[i];
        var_[i] = 1.0;
      }
      inited = true;
      return;
    }
    final a = 1.0 - momentum;
    for (int i = 0; i < dim; i++) {
      final mPrev = mean[i];
      final xi = x[i];
      final m = momentum * mPrev + a * xi;
      final dev = xi - m;
      final devPrev = xi - mPrev;
      final v = momentum * var_[i] + a * (dev * devPrev);
      mean[i] = m;
      var_[i] = v <= 0 ? eps : v;
    }
  }

  List<double> normalize(List<double> x, {bool update = false}) {
    if (x.length != dim) _resizeTo(x.length);
    if (update) observe(x);
    final out = List<double>.filled(x.length, 0.0);
    for (int i = 0; i < x.length; i++) {
      out[i] = (x[i] - mean[i]) / math.sqrt(var_[i] + eps);
    }
    return out;
  }
}

/* -------------------------------------------------------------------------- */
/*                             FEATURE EXTRACTORS                              */
/* -------------------------------------------------------------------------- */

class FeatureExtractor {
  final int groundSamples;
  final double stridePx;

  FeatureExtractor({this.groundSamples = 3, this.stridePx = 48});

  // [px, py, vx, vy, ang, fuel, padCx, dxToPad, hAboveGround, slope, groundSamples...]
  int get inputSize => 10 + groundSamples;

  List<double> extract(eng.GameEngine env) {
    final L = env.lander;
    final T = env.terrain;
    final px = L.pos.x.toDouble();
    final py = L.pos.y.toDouble();
    final vx = L.vel.x.toDouble();
    final vy = L.vel.y.toDouble();
    final ang = L.angle.toDouble();
    final fuel = L.fuel;

    final padCx = T.padCenter.toDouble();
    final dxToPad = (px - padCx);

    final gy = T.heightAt(px);
    final hAbove = (gy - py);

    final hL = T.heightAt(_clip(px - 8.0, 0.0, env.cfg.worldW));
    final hR = T.heightAt(_clip(px + 8.0, 0.0, env.cfg.worldW));
    final slope = (hR - hL) / 16.0;

    final feats = <double>[
      px / env.cfg.worldW,
      py / env.cfg.worldH,
      vx / 200.0,
      vy / 200.0,
      ang / math.pi,
      fuel / (env.cfg.t.maxFuel > 0 ? env.cfg.t.maxFuel : 1.0),
      padCx / env.cfg.worldW,
      dxToPad / (0.5 * env.cfg.worldW),
      hAbove / 300.0,
      slope / 2.0,
    ];

    for (int i = 0; i < groundSamples; i++) {
      final off = (i - (groundSamples ~/ 2)) * stridePx;
      final gx = _clip(px + off, 0.0, env.cfg.worldW);
      final gyS = T.heightAt(gx);
      feats.add((gyS - py) / 300.0);
    }
    return feats;
  }
}

/// Ray-based features: lander scalars + per-ray channels.
class FeatureExtractorRays {
  final int rayCount;
  final bool kindsOneHot;
  FeatureExtractorRays({required this.rayCount, this.kindsOneHot = true});

  int get inputSize => 6 + rayCount * (kindsOneHot ? 4 : 1);

  List<double> extract(eng.GameEngine env) {
    final L = env.lander;
    final W = env.cfg.worldW.toDouble();
    final H = env.cfg.worldH.toDouble();

    final px = (L.pos.x.toDouble() / W);
    final py = (L.pos.y.toDouble() / H);
    final vx = (L.vel.x.toDouble() / 200.0).clamp(-3.0, 3.0);
    final vy = (L.vel.y.toDouble() / 200.0).clamp(-3.0, 3.0);
    final ang = (L.angle.toDouble() / math.pi).clamp(-2.0, 2.0);
    final fuel = (L.fuel / (env.cfg.t.maxFuel > 0 ? env.cfg.t.maxFuel : 1.0)).clamp(0.0, 1.0);

    final out = <double>[px, py, vx, vy, ang, fuel];
    final maxD = math.sqrt(W * W + H * H);
    final rays = env.rays;

    for (int i = 0; i < rayCount; i++) {
      RayHit? rh = (i < rays.length) ? rays[i] : null;

      double d;
      if (rh == null) {
        d = maxD;
      } else {
        final dx = rh.p.x - L.pos.x;
        final dy = rh.p.y - L.pos.y;
        d = math.sqrt(dx * dx + dy * dy).toDouble();
      }
      final dN = (d / maxD).clamp(0.0, 1.0);

      if (!kindsOneHot) {
        out.add(dN);
      } else {
        double tTerr = 0, tPad = 0, tWall = 0;
        if (rh != null) {
          switch (rh.kind) {
            case RayHitKind.terrain: tTerr = 1; break;
            case RayHitKind.pad:     tPad  = 1; break;
            case RayHitKind.wall:    tWall = 1; break;
          }
        }
        out.addAll([dN, tTerr, tPad, tWall]);
      }
    }

    return out;
  }
}

/* -------------------------------------------------------------------------- */
/*                                CONTROLLERS                                  */
/* -------------------------------------------------------------------------- */

int predictiveIntentLabelAdaptive(
    eng.GameEngine env, {
      double baseTauSec = 1.0,
      double minTauSec = 0.45,
      double maxTauSec = 2.20,
    }) {
  final L = env.lander;
  final T = env.terrain;
  final W = env.cfg.worldW.toDouble();
  final padCx = T.padCenter.toDouble();

  final px = L.pos.x.toDouble();
  final py = L.pos.y.toDouble();
  final vx = L.vel.x.toDouble();
  final vy = L.vel.y.toDouble();

  final gy = T.heightAt(px);
  final h  = (gy - py).toDouble();
  final dx = px - padCx;

  // Predict ~1 sec ahead depending on height
  final hNorm = (h / 320.0).clamp(0.0, 1.6);
  final tau = (baseTauSec * (0.7 + 0.5 * hNorm)).clamp(minTauSec, maxTauSec);

  final g = env.cfg.t.gravity;
  final dxF = dx + vx * tau;
  final vyF = vy + g * tau;

  final padEnter = 0.08 * W;
  final padExit  = 0.14 * W;

  // --- Vertical emergency 'brakeUp' (tight window) ---
  final vCapStrong = (0.07 * h + 6.0).clamp(6.0, 16.0);
  final tooLow      = h < 140.0;
  final tooFastDown = vyF > math.max(40.0, vCapStrong + 10.0);
  final nearPadLat  = dx.abs() <= 0.18 * W;

  if (tooLow && tooFastDown && nearPadLat) {
    return intentToIndex(Intent.brakeUp);
  }

  if (dx.abs() <= padEnter) {
    if (vx > 25.0)  return intentToIndex(Intent.brakeRight);
    if (vx < -25.0) return intentToIndex(Intent.brakeLeft);
    return intentToIndex(Intent.descendSlow);
  }

  if (dxF.abs() > padExit) {
    return dxF > 0 ? intentToIndex(Intent.goRight)
        : intentToIndex(Intent.goLeft);
  }

  final willExitSoon = (dxF.abs() > padEnter) && (dx.abs() <= padEnter);
  final vxIsOutward  = (dx.sign == vx.sign) && vx.abs() > 20.0;
  if ((willExitSoon || vxIsOutward) && h > 90) {
    return dx >= 0 ? intentToIndex(Intent.goLeft)
        : intentToIndex(Intent.goRight);
  }

  return intentToIndex(Intent.descendSlow);
}

bool _canStrafe(eng.GameEngine env, {double maxTilt = 0.10, double minH = 110.0, double maxVy = 35.0}) {
  final t = env.cfg.t;
  final hasRcs = (t as dynamic);
  final enabled = (hasRcs as dynamic).rcsEnabled ?? false;
  if (!enabled) return false;

  final L = env.lander;
  final gy = env.terrain.heightAt(L.pos.x);
  final h  = (gy - L.pos.y).toDouble();
  if (h <= minH) return false;
  if (L.vel.y.abs() >= maxVy) return false;
  if (L.angle.abs() > maxTilt) return false;
  return true;
}

et.ControlInput controllerForIntent(Intent intent, eng.GameEngine env) {
  final L = env.lander;
  final T = env.terrain;
  final px = L.pos.x.toDouble();
  final gy = T.heightAt(px);
  final h  = (gy - L.pos.y).toDouble();
  final vx = L.vel.x.toDouble();
  final vy = L.vel.y.toDouble();

  bool rcsLeft = false, rcsRight = false, down = false;

  switch (intent) {
    case Intent.brakeUp: {
      final vCap = (0.07 * h + 6.0).clamp(6.0, 16.0);
      final needUp = vy > vCap;
      return et.ControlInput(
        thrust: needUp, left: false, right: false,
        sideLeft: false, sideRight: false,
        downThrust: false,
      );
    }
    case Intent.descendSlow: {
      final vCap = (0.10 * h + 8.0).clamp(8.0, 26.0);
      final g = env.cfg.t.gravity;
      final lowG = g.abs() < 1e-6;
      final t = (env.cfg.t as dynamic);
      final downEnabled = (t.downThrEnabled ?? false) == true;
      final wantDown = lowG && downEnabled && (vy < 0.7 * vCap);
      final needUp   = vy > vCap || (!lowG && h < 110.0);
      down = wantDown;
      return et.ControlInput(
        thrust: needUp, left: false, right: false,
        sideLeft: false, sideRight: false,
        downThrust: down,
      );
    }
    case Intent.brakeLeft: {
      final wantTiltRight   = (vx < -4.0);
      final allowTranslate  = (h > 110 && h < 300) && (vy < 35);
      if (_canStrafe(env)) rcsLeft = true;
      return et.ControlInput(
        thrust: allowTranslate && !rcsLeft && wantTiltRight,
        left: false, right: (!rcsLeft && wantTiltRight),
        sideLeft: rcsLeft, sideRight: false,
        downThrust: false,
      );
    }
    case Intent.brakeRight: {
      final wantTiltLeft    = (vx >  4.0);
      final allowTranslate  = (h > 110 && h < 300) && (vy < 35);
      if (_canStrafe(env)) rcsRight = true;
      return et.ControlInput(
        thrust: allowTranslate && !rcsRight && wantTiltLeft,
        left: (!rcsRight && wantTiltLeft), right: false,
        sideLeft: false, sideRight: rcsRight,
        downThrust: false,
      );
    }
    case Intent.goLeft: {
      final translate = (h > 110 && h < 300) && (vy < 35);
      if (_canStrafe(env)) rcsRight = true;
      return et.ControlInput(
        thrust: translate && !rcsRight,
        left: !rcsRight, right: false,
        sideLeft: false, sideRight: rcsRight,
        downThrust: false,
      );
    }
    case Intent.goRight: {
      final translate = (h > 110 && h < 300) && (vy < 35);
      if (_canStrafe(env)) rcsLeft = true;
      return et.ControlInput(
        thrust: translate && !rcsLeft,
        left: false, right: !rcsLeft,
        sideLeft: rcsLeft, sideRight: false,
        downThrust: false,
      );
    }
    case Intent.hover:
    default: {
      final vHover = (0.06 * h + 6.0).clamp(6.0, 18.0);
      final needUp = vy > vHover;
      return et.ControlInput(
        thrust: needUp, left: false, right: false,
        sideLeft: false, sideRight: false,
        downThrust: false,
      );
    }
  }
}

/* -------------------------------------------------------------------------- */
/*                               POLICY NETWORK                                */
/* -------------------------------------------------------------------------- */

class ForwardCache {
  final List<double> x;
  final List<List<double>> acts; // a0..aL (tanh outputs)
  final List<double> intentLogits;
  final List<double> intentProbs;
  final List<double> turnLogits; // 3 logits
  final double thrLogit;
  final double thrProb;
  final double v; // value head
  ForwardCache({
    required this.x,
    required this.acts,
    required this.intentLogits,
    required this.intentProbs,
    required this.turnLogits,
    required this.thrLogit,
    required this.thrProb,
    required this.v,
  });
}

class PolicyNetwork {
  static const int kIntents = 7; // updated

  final int inputSize;
  final List<int> hidden;
  final nn.MLPTrunk trunk;
  final nn.PolicyHeads heads;

  PolicyNetwork({
    required this.inputSize,
    List<int> hidden = const [64, 64],
    int seed = 0,
  })  : hidden = List<int>.from(hidden),
        trunk  = nn.MLPTrunk(inputSize: inputSize, hiddenSizes: hidden, seed: seed ^ 0x7777),
        heads  = nn.PolicyHeads(hidden.isEmpty ? inputSize : hidden.last,
            intents: kIntents, seed: seed ^ 0x8888) {
    assert(heads.intent.b.length == kIntents, 'intent head outDim mismatch');
    assert(heads.intent.W.length == kIntents, 'intent head rows mismatch');
  }

  ForwardCache _forwardFull(List<double> x) {
    final acts = trunk.forwardAll(x);
    final h = acts.last;
    final intentLogits = heads.intent.forward(h);
    final intentProbs  = nn.Ops.softmax(intentLogits);
    final turnLogits   = heads.turn.forward(h);
    final thrLogit     = heads.thr.forward(h)[0];
    final thrProb      = nn.Ops.sigmoid(thrLogit);
    final v            = heads.val.forward(h)[0];

    return ForwardCache(
      x: List<double>.from(x),
      acts: acts,
      intentLogits: intentLogits,
      intentProbs: intentProbs,
      turnLogits: turnLogits,
      thrLogit: thrLogit,
      thrProb: thrProb,
      v: v,
    );
  }

  (int, List<double>, ForwardCache) actIntentGreedy(List<double> x) {
    final c = _forwardFull(x);
    int arg = 0;
    double best = c.intentProbs[0];
    for (int i = 1; i < c.intentProbs.length; i++) {
      if (c.intentProbs[i] > best) { best = c.intentProbs[i]; arg = i; }
    }
    return (arg, c.intentProbs, c);
  }

  (bool, bool, bool, List<double>, ForwardCache) actGreedy(List<double> x) {
    final c = _forwardFull(x);
    final thrust = c.thrProb >= 0.5;
    int tArg = 0;
    double best = c.turnLogits[0];
    for (int i = 1; i < 3; i++) {
      if (c.turnLogits[i] > best) { best = c.turnLogits[i]; tArg = i; }
    }
    final left = (tArg == 0);
    final right = (tArg == 2);
    final probs = <double>[ c.thrProb, ...nn.Ops.softmax(c.turnLogits) ];
    return (thrust, left, right, probs, c);
  }

  void updateFromEpisode({
    required List<ForwardCache> decisionCaches,     // for intent head
    required List<int> intentChoices,              // chosen intents
    required List<double> decisionReturns,         // advantages (can be signed)
    required List<int> alignLabels,                // teacher labels for intent
    required double alignWeight,                   // CE supervision
    required double intentPgWeight,                // PG on intent
    required double lr,
    required double entropyBeta,
    required double valueBeta,
    required double huberDelta,
    required bool intentMode,

    List<ForwardCache>? actionCaches,              // action supervision (turn/thr)
    List<int>? actionTurnTargets,
    List<bool>? actionThrustTargets,
    double actionAlignWeight = 0.0,
  }) {
    final int H = trunk.layers.isEmpty ? inputSize : trunk.layers.last.b.length;
    assert(heads.intent.b.length > 0 && heads.intent.W.isNotEmpty,
    'intent head not initialized');
    assert(heads.turn.b.length == 3 && heads.turn.W.length == 3,
    'turn head must have 3 logits');
    assert(heads.thr.b.length == 1 && heads.thr.W.length == 1,
    'thrust head must have 1 logit');
    assert(heads.intent.W[0].length == H &&
        heads.turn.W[0].length   == H &&
        heads.thr.W[0].length    == H,
    'head input dims must match trunk output');

    double _clipGrad(double g, [double c = 1.0]) {
      if (!g.isFinite) return 0.0;
      if (g > c) return c;
      if (g < -c) return -c;
      return g;
    }

    // Trunk accumulators
    List<List<List<double>>> gW_trunk = [
      for (final lin in trunk.layers)
        List.generate(lin.W.length, (_) => List<double>.filled(lin.W[0].length, 0.0))
    ];
    List<List<double>> gb_trunk = [
      for (final lin in trunk.layers) List<double>.filled(lin.b.length, 0.0)
    ];

    // Heads grads
    final gW_int = List.generate(heads.intent.W.length,
            (_) => List<double>.filled(heads.intent.W[0].length, 0.0));
    final gb_int = List<double>.filled(heads.intent.b.length, 0.0);

    final gW_turn = List.generate(heads.turn.W.length,
            (_) => List<double>.filled(heads.turn.W[0].length, 0.0));
    final gb_turn = List<double>.filled(heads.turn.b.length, 0.0);

    final gW_thr = List.generate(heads.thr.W.length,
            (_) => List<double>.filled(heads.thr.W[0].length, 0.0));
    final gb_thr = List<double>.filled(heads.thr.b.length, 0.0);

    // ----- Intent CE (optional) -----
    if (intentMode && alignWeight > 0 && decisionCaches.isNotEmpty) {
      final N = decisionCaches.length;
      for (int n = 0; n < N; n++) {
        final c = decisionCaches[n];
        final h = c.acts.last;
        final y = alignLabels[n].clamp(0, PolicyNetwork.kIntents - 1);
        final dLog = nn.Ops.crossEntropyGrad(c.intentProbs, y); // (p - y)
        final dH = heads.intent.backward(x: h, dOut: dLog, gW: gW_int, gb: gb_int);
        trunk.backwardFromTopGrad(
          dTop: dH, acts: c.acts, gW: gW_trunk, gb: gb_trunk, x0: c.x,
        );
      }
      final scale = lr * alignWeight / N;
      for (int i = 0; i < heads.intent.b.length; i++) {
        heads.intent.b[i] -= _clipGrad(scale * gb_int[i]);
        for (int j = 0; j < heads.intent.W[0].length; j++) {
          heads.intent.W[i][j] -= _clipGrad(scale * gW_int[i][j]);
        }
      }
      // zero CE accumulators so PG doesn't double-apply trunk grads
      for (int i = 0; i < gb_int.length; i++) gb_int[i] = 0.0;
      for (int i = 0; i < gW_int.length; i++) {
        for (int j = 0; j < gW_int[0].length; j++) gW_int[i][j] = 0.0;
      }
      for (int li = 0; li < gb_trunk.length; li++) {
        for (int j = 0; j < gb_trunk[li].length; j++) gb_trunk[li][j] = 0.0;
        for (int r = 0; r < gW_trunk[li].length; r++) {
          for (int c2 = 0; c2 < gW_trunk[li][0].length; c2++) gW_trunk[li][r][c2] = 0.0;
        }
      }
    }

    // ----- Intent PG (REINFORCE with advantage) -----
    if (intentPgWeight > 0 && decisionCaches.isNotEmpty) {
      final N = decisionCaches.length;
      for (int n = 0; n < N; n++) {
        final c = decisionCaches[n];
        final h = c.acts.last;
        final chosen = intentChoices[n].clamp(0, PolicyNetwork.kIntents - 1);
        final adv = decisionReturns[n]; // signed advantage
        final dLog = List<double>.generate(
          c.intentProbs.length,
              (i) => (c.intentProbs[i] - (i == chosen ? 1.0 : 0.0)) * (-adv),
        );
        final dH = heads.intent.backward(x: h, dOut: dLog, gW: gW_int, gb: gb_int);
        trunk.backwardFromTopGrad(
          dTop: dH, acts: c.acts, gW: gW_trunk, gb: gb_trunk, x0: c.x,
        );
      }

      final scale = lr * intentPgWeight / decisionCaches.length;
      for (int i = 0; i < heads.intent.b.length; i++) {
        heads.intent.b[i] -= _clipGrad(scale * gb_int[i]);
        for (int j = 0; j < heads.intent.W[0].length; j++) {
          heads.intent.W[i][j] -= _clipGrad(scale * gW_int[i][j]);
        }
      }
    }

    // ----- Action supervision (optional; unchanged) -----
    final hasAction = actionAlignWeight > 0.0 &&
        actionCaches != null &&
        actionTurnTargets != null &&
        actionThrustTargets != null &&
        actionCaches.isNotEmpty;

    if (hasAction) {
      final M = actionCaches!.length;
      double meanThrLogit = 0.0;
      double teacherThrRate = 0.0;

      for (int n = 0; n < M; n++) {
        final c = actionCaches![n];
        final h = c.acts.last;

        final turnProbs = nn.Ops.softmax(c.turnLogits);
        final yt = actionTurnTargets![n].clamp(0, 2);
        final dTurn = nn.Ops.crossEntropyGrad(turnProbs, yt, numClasses: 3);

        final yb = actionThrustTargets![n] ? 1.0 : 0.0;
        final dThr = nn.Ops.bceGradFromLogit(c.thrLogit, yb);
        teacherThrRate += yb;

        final dH_turn = heads.turn.backward(x: h, dOut: dTurn, gW: gW_turn, gb: gb_turn);
        final dH_thr  = heads.thr.backward (x: h, dOut: [dThr], gW: gW_thr , gb: gb_thr );

        final dH = List<double>.filled(h.length, 0.0);
        for (int i = 0; i < h.length; i++) dH[i] = dH_turn[i] + dH_thr[i];

        trunk.backwardFromTopGrad(
          dTop: dH, acts: c.acts, gW: gW_trunk, gb: gb_trunk, x0: c.x,
        );

        meanThrLogit += c.thrLogit;
      }

      final scale = lr * actionAlignWeight / actionCaches.length;
      for (int i = 0; i < heads.turn.b.length; i++) {
        heads.turn.b[i] -= _clipGrad(scale * gb_turn[i]);
        for (int j = 0; j < heads.turn.W[0].length; j++) {
          heads.turn.W[i][j] -= _clipGrad(scale * gW_turn[i][j]);
        }
      }
      heads.thr.b[0] -= _clipGrad(scale * gb_thr[0]);
      for (int j = 0; j < heads.thr.W[0].length; j++) {
        heads.thr.W[0][j] -= _clipGrad(scale * gW_thr[0][j]);
      }

      // quick thrust bias calibration
      meanThrLogit /= actionCaches.length;
      teacherThrRate /= actionCaches.length;
      final logitTarget = nn.Ops.logit(teacherThrRate);
      final calibStep = 0.25;
      heads.thr.b[0] += (logitTarget - meanThrLogit) * calibStep;
    }

    // ----- Apply trunk update (shared grads) -----
    final trunkScale = lr;
    for (int li = 0; li < trunk.layers.length; li++) {
      final L = trunk.layers[li];
      final gb = gb_trunk[li];
      final gW = gW_trunk[li];
      for (int i = 0; i < L.b.length; i++) L.b[i] -= _clipGrad(trunkScale * gb[i]);
      for (int i = 0; i < L.W.length; i++) {
        for (int j = 0; j < L.W[0].length; j++) {
          L.W[i][j] -= _clipGrad(trunkScale * gW[i][j]);
        }
      }
    }
  }
}

/* -------------------------------------------------------------------------- */
/*                                   TRAINER                                  */
/* -------------------------------------------------------------------------- */

class EpisodeResult {
  final int steps;
  final double totalCost;
  final bool landed;
  final double segMean; // here: mean PF reward (for logging/gating)
  EpisodeResult({
    required this.steps,
    required this.totalCost,
    required this.landed,
    required this.segMean,
  });
}

// external dense reward hook (potential-field reward)
typedef ExternalRewardHook = double Function({
required eng.GameEngine env,
required double dt,
required int tStep,
});

class Trainer {
  final eng.GameEngine env;
  final FeatureExtractorRays fe;
  final PolicyNetwork policy;
  final double dt;
  final double gamma;
  final int seed;
  final bool twoStage;
  final int planHold;
  final double tempIntent;
  final double intentEntropyBeta;
  final bool useLearnedController;
  final double blendPolicy; // probability-space blend for thrust
  final double intentAlignWeight;
  final double intentPgWeight;     // PG strength
  final double actionAlignWeight;
  final bool normalizeFeatures;

  // gating/logging
  final double gateScoreMin;       // now compares mean PF reward
  final bool gateOnlyLanded;
  final bool gateVerbose;

  final RunningNorm? norm;
  int _epCounter = 0;

  // PWM thrust state
  double _pwmA = 0.0;
  int _pwmCount = 0;
  int _pwmOn = 0;

  // external per-step reward hook (PF)
  final ExternalRewardHook? externalRewardHook;

  Trainer({
    required this.env,
    required this.fe,
    required this.policy,
    required this.dt,
    required this.gamma,
    required this.seed,
    required this.twoStage,
    required this.planHold,
    required this.tempIntent,
    required this.intentEntropyBeta,
    required this.useLearnedController,
    required this.blendPolicy,
    required this.intentAlignWeight,
    this.intentPgWeight = 0.6,
    required this.actionAlignWeight,
    required this.normalizeFeatures,

    this.gateScoreMin = -double.infinity,
    this.gateOnlyLanded = false,
    this.gateVerbose = true,

    this.externalRewardHook,
  }) : norm = RunningNorm(fe.inputSize, momentum: 0.995);

  int _sampleCategorical(List<double> probs, math.Random r, double temp) {
    if (temp <= 1e-6) {
      int arg = 0; double best = probs[0];
      for (int i = 1; i < probs.length; i++) if (probs[i] > best) { best = probs[i]; arg = i; }
      return arg;
    }
    final z = probs.map((p) => math.log(_clip(p, 1e-12, 1.0))).toList();
    for (int i = 0; i < z.length; i++) z[i] /= temp;
    final sm = nn.Ops.softmax(z);
    final u = r.nextDouble();
    double acc = 0.0;
    for (int i = 0; i < sm.length; i++) {
      acc += sm[i];
      if (u <= acc) return i;
    }
    return sm.length - 1;
  }

  EpisodeResult runEpisode({
    required bool train,
    required bool greedy,
    required bool scoreIsReward,   // kept for API compat; unused
    double lr = 3e-4,
    double valueBeta = 0.5,
    double huberDelta = 1.0,
  }) {
    final r = math.Random(seed ^ (_epCounter++));
    final decisionRewards = <double>[];  // per-decision reward (PF only)

    final actionCaches = <ForwardCache>[];
    final actionTurnTargets = <int>[];
    final actionThrustTargets = <bool>[];

    final decisionCaches = <ForwardCache>[];
    final intentChoices = <int>[];
    final decisionReturns = <double>[]; // advantages
    final alignLabels = <int>[];

    env.reset(seed: r.nextInt(1 << 30));
    double totalCost = 0.0;

    // PF logging (use seg* names to keep CLI the same)
    double segSum = 0.0;
    int segCount = 0;

    _pwmA = 0.0; _pwmCount = 0; _pwmOn = 0;

    int framesLeft = 0;
    int currentIntentIdx = 0;

    int steps = 0;
    bool landed = false;

    // accumulate PF reward until next decision boundary
    double pfAcc = 0.0;

    while (true) {
      if (framesLeft <= 0) {
        var x = fe.extract(env);
        final yTeacher = predictiveIntentLabelAdaptive(
            env, baseTauSec: 1.0, minTauSec: 0.45, maxTauSec: 1.35);

        if (normalizeFeatures) {
          norm?.observe(x);
          x = norm?.normalize(x, update: false) ?? x;
        }

        final (idxGreedy, p, cache) = policy.actIntentGreedy(x);
        final idx = greedy ? idxGreedy : _sampleCategorical(p, r, tempIntent);
        currentIntentIdx = idx;

        if (train) {
          // PF-only: push the accumulated PF reward for the last window
          decisionCaches.add(cache);
          intentChoices.add(idx);
          alignLabels.add(yTeacher);
          decisionRewards.add(pfAcc);
          pfAcc = 0.0;
        }

        // compute discounted returns (advantages)
        if (train && decisionCaches.isNotEmpty) {
          final T = decisionRewards.length;
          if (T != decisionCaches.length ||
              T != intentChoices.length ||
              T != alignLabels.length) {
            throw StateError('length mismatch in decision arrays');
          }

          final tmp = List<double>.filled(T, 0.0);
          double G = 0.0;
          for (int i = T - 1; i >= 0; i--) {
            G = decisionRewards[i] + gamma * G;
            tmp[i] = G;
          }

          double mean = 0.0;
          for (final v in tmp) mean += v;
          mean /= T;

          double var0 = 0.0;
          for (final v in tmp) { final d = v - mean; var0 += d * d; }
          var0 = (var0 / T).clamp(1e-9, double.infinity);
          final std = math.sqrt(var0);

          decisionReturns.clear();
          for (final v in tmp) decisionReturns.add((v - mean) / std);
        }

        // adaptive plan hold (unchanged)
        final padCx = env.terrain.padCenter.toDouble();
        final dxAbs = (env.lander.pos.x.toDouble() - padCx).abs();
        final vxAbs = env.lander.vel.x.toDouble().abs();
        final gy = env.terrain.heightAt(env.lander.pos.x.toDouble());
        final h = (gy - env.lander.pos.y).toDouble();
        final W = env.cfg.worldW.toDouble();

        int dynHold = 1;
        if (dxAbs > 0.12 * W || vxAbs > 60.0) dynHold = 1;
        if (dynHold == 1 && h > 320.0 && dxAbs < 0.04 * W && vxAbs < 25.0) dynHold = 2;
        framesLeft = dynHold;
      }

      final intent = indexToIntent(currentIntentIdx);
      final uTeacher = controllerForIntent(intent, env);

      var xAct = fe.extract(env);
      xAct = norm?.normalize(xAct, update: false) ?? xAct;
      final (thBool, lf, rt, probs, cAct) = policy.actGreedy(xAct);

      final groundY = env.terrain.heightAt(env.lander.pos.x);
      final height  = (groundY - env.lander.pos.y).toDouble();

      final pThrModel   = probs[0].clamp(0.0, 1.0);
      final pThrTeacher = uTeacher.thrust ? 1.0 : 0.0;
      final pThrExec = blendPolicy * pThrModel + (1.0 - blendPolicy) * pThrTeacher;

      _pwmA = (_pwmA + pThrExec).clamp(0.0, 10.0);

      bool thrustPWM = false;
      while (_pwmA >= 1.0) {
        thrustPWM = true;
        _pwmA -= 1.0;
      }
      if (height < 90.0 && !thrustPWM && pThrExec > 0.65) {
        thrustPWM = true;
        _pwmA = (_pwmA - 0.65).clamp(0.0, 0.999);
      }

      final execThrust = thrustPWM;
      final execLeft   = useLearnedController ? lf : uTeacher.left;
      final execRight  = useLearnedController ? rt : uTeacher.right;

      // pass teacher’s strafing & down-thrust to the environment
      final execSideLeft  = uTeacher.sideLeft;
      final execSideRight = uTeacher.sideRight;
      final execDown      = uTeacher.downThrust;

      // ----- PF-only reward accumulation -----
      final r_pf = externalRewardHook?.call(env: env, dt: dt, tStep: steps) ?? 0.0;
      pfAcc += r_pf;
      segSum += r_pf; // for logging/gating
      segCount++;

      actionCaches.add(cAct);
      actionTurnTargets.add(uTeacher.left ? 0 : (uTeacher.right ? 2 : 1));
      actionThrustTargets.add(uTeacher.thrust);

      _pwmCount++; if (execThrust) _pwmOn++;
      if ((_pwmCount % 240) == 0) {
        _pwmCount = 0; _pwmOn = 0;
      }

      final info = env.step(dt, et.ControlInput(
        thrust: execThrust,
        left: execLeft,
        right: execRight,
        sideLeft: execSideLeft,
        sideRight: execSideRight,
        downThrust: execDown,
        intentIdx: currentIntentIdx,
      ));
      totalCost += info.costDelta;
      steps++;
      framesLeft--;

      if (info.terminal) {
        landed = env.status == et.GameStatus.landed;

        // push any remaining PF reward into the final decision window
        if (train && decisionRewards.isNotEmpty && pfAcc.abs() > 0) {
          decisionRewards[decisionRewards.length - 1] += pfAcc;
          pfAcc = 0.0;
        }

        // IMPORTANT: no terminal bonus/penalty anymore (PF-only)
        break;
      }
      if (steps > 5000) break;
    }

    final segMean = (segCount > 0) ? (segSum / segCount) : 0.0;

    // ---- GATED UPDATE + LOGGING (PF mean) ----
    bool accept = true;
    if (gateOnlyLanded && !landed) accept = false;
    if (segMean < gateScoreMin) accept = false;

    if (train) {
      if (accept && (decisionCaches.isNotEmpty || actionCaches.isNotEmpty)) {
        policy.updateFromEpisode(
          decisionCaches: decisionCaches,
          intentChoices: intentChoices,
          decisionReturns: decisionReturns,   // advantages from PF-only rewards
          alignLabels: alignLabels,
          alignWeight: intentAlignWeight,
          intentPgWeight: intentPgWeight,
          lr: lr,
          entropyBeta: 0.0,
          valueBeta: valueBeta,
          huberDelta: huberDelta,
          intentMode: true,
          actionCaches: actionCaches,
          actionTurnTargets: actionTurnTargets,
          actionThrustTargets: actionThrustTargets,
          actionAlignWeight: actionAlignWeight,
        );
        if (gateVerbose) {
          print('[TRAIN] accepted | steps=$steps | pfMean=${segMean.toStringAsFixed(3)} | landed=${landed ? "Y" : "N"} '
              '| caches: dec=${decisionCaches.length}, act=${actionCaches.length}');
        }
      } else {
        if (gateVerbose) {
          print('[TRAIN] skipped  | steps=$steps | pfMean=${segMean.toStringAsFixed(3)} | landed=${landed ? "Y" : "N"}');
        }
      }
    }

    return EpisodeResult(steps: steps, totalCost: totalCost, landed: landed, segMean: segMean);
  }
}
