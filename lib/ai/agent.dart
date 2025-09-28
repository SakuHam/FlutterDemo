// lib/ai/agent.dart
import 'dart:math' as math;
import 'dart:convert' as convert;

import '../engine/raycast.dart';
import '../engine/types.dart' as et;
import '../engine/game_engine.dart' as eng;
import 'nn_helper.dart' as nn;

/* ============================== Intents ================================== */

enum Intent { hover, goLeft, goRight, descendSlow, brakeUp, brakeLeft, brakeRight }
const List<String> kIntentNames = [
  'hover', 'goLeft', 'goRight', 'descendSlow', 'brakeUp', 'brakeLeft', 'brakeRight'
];
int intentToIndex(Intent i) => i.index;
Intent indexToIntent(int k) => Intent.values[k.clamp(0, Intent.values.length - 1)];

/* ============================ RunningNorm ================================= */

class RunningNorm {
  int dim;
  List<double> mean;
  List<double> var_;
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
    var_ = List<double>.filled(dim, 1.0);
    inited = false;
  }

  void observe(List<double> x) {
    if (x.length != dim) _resizeTo(x.length);
    if (!inited) {
      for (int i = 0; i < dim; i++) { mean[i] = x[i]; var_[i] = 1.0; }
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
    for (int i = 0; i < x.length; i++) out[i] = (x[i] - mean[i]) / math.sqrt(var_[i] + eps);
    return out;
  }

  void copyFrom(RunningNorm other) {
    final m = math.min(mean.length, other.mean.length);
    for (int i = 0; i < m; i++) { mean[i] = other.mean[i]; var_[i] = other.var_[i]; }
    dim = other.dim; momentum = other.momentum; inited = other.inited;
  }

  Map<String, dynamic> toJson() => {
    'dim': dim, 'mean': mean, 'var': var_, 'momentum': momentum, 'inited': inited, 'eps': eps,
  };
  void fromJson(Map<String, dynamic> m) {
    dim = (m['dim'] as num).toInt();
    mean = (m['mean'] as List).map((e) => (e as num).toDouble()).toList();
    var_ = (m['var'] as List).map((e) => (e as num).toDouble()).toList();
    momentum = (m['momentum'] as num).toDouble();
    inited = m['inited'] == true;
  }

  String toJsonString() => convert.jsonEncode(toJson());
  void loadFromJson(String s) => fromJson(convert.jsonDecode(s));
}

/* ========================= Feature Extractors ============================= */

class FeatureExtractorRays {
  final int rayCount;
  final bool kindsOneHot;
  FeatureExtractorRays({required this.rayCount, this.kindsOneHot = true});

  int get inputSize => 5 + rayCount * (kindsOneHot ? 4 : 1);

  List<double> extract({
    required et.LanderState lander,
    required et.Terrain terrain,
    required double worldW,
    required double worldH,
    required List<RayHit>? rays,
    double uiMaxFuel = 100.0,
  }) {
    if (rays == null) {
      throw StateError('FeatureExtractorRays requires engine.rays.');
    }

    final ang = (lander.angle / math.pi).clamp(-2.0, 2.0);

    final gy = terrain.heightAt(lander.pos.x);
    final h  = (gy - lander.pos.y).toDouble().clamp(0.0, 1e9);
    double vCap = (0.10 * h + 8.0).clamp(8.0, 26.0);
    final hnVy  = (lander.vel.y.toDouble() / (vCap > 1e-6 ? vCap : 1.0)).clamp(-3.0, 3.0);

    final vx = lander.vel.x.toDouble();
    final vy = lander.vel.y.toDouble();
    double speed = math.sqrt(vx * vx + vy * vy);
    const sClip = 140.0;
    speed = (speed / sClip).clamp(0.0, 1.5);

    final padCx = (terrain.padX1 + terrain.padX2) * 0.5;
    final padCy = terrain.heightAt(padCx);
    double pxToPad = padCx - lander.pos.x;
    double pyToPad = padCy - lander.pos.y;

    double sx = 0.0, sy = 0.0, wsum = 0.0;
    for (final r in rays) {
      if (r.kind != RayHitKind.pad) continue;
      final dx = r.p.x - lander.pos.x;
      final dy = r.p.y - lander.pos.y;
      final d2 = dx*dx + dy*dy;
      if (d2 <= 1e-9) continue;
      final w = 1.0 / math.sqrt(d2 + 1e-6);
      sx += w * dx; sy += w * dy; wsum += w;
    }
    bool padVecValid = false;
    if (wsum > 0) { final inv = 1.0 / wsum; pxToPad = sx * inv; pyToPad = sy * inv; padVecValid = true; }

    final vLen = math.max(1e-9, math.sqrt(vx*vx + vy*vy)); final vnx = vx / vLen, vny = vy / vLen;
    final pLen = math.max(1e-9, math.sqrt(pxToPad*pxToPad + pyToPad*pyToPad));
    final pnx = pxToPad / pLen, pny = pyToPad / pLen;

    double _angWrap(double a) { const twoPi = math.pi * 2.0; a %= twoPi; if (a > math.pi) a -= twoPi; if (a < -math.pi) a += twoPi; return a; }
    final angDelta = _angWrap(math.atan2(pny, pnx) - math.atan2(vny, vnx));
    final angDeltaPi = (angDelta / math.pi).clamp(-1.0, 1.0);
    final padVis = padVecValid ? 1.0 : 0.0;

    final out = <double>[ speed, hnVy, ang, angDeltaPi, padVis ];

    final maxD = math.sqrt(worldW * worldW + worldH * worldH);
    for (int i = 0; i < rayCount; i++) {
      RayHit? rh = (i < rays.length) ? rays[i] : null;
      double d;
      if (rh == null) d = maxD; else { final dx = rh.p.x - lander.pos.x; final dy = rh.p.y - lander.pos.y; d = math.sqrt(dx*dx + dy*dy); }
      final dN = (d / maxD).clamp(0.0, 1.0);
      if (!kindsOneHot) {
        out.add(dN);
      } else {
        double tTerr = 0, tPad = 0, tWall = 0;
        if (rh != null) {
          switch (rh.kind) { case RayHitKind.terrain: tTerr = 1; break; case RayHitKind.pad: tPad = 1; break; case RayHitKind.wall: tWall = 1; break; }
        }
        out.addAll([dN, tTerr, tPad, tWall]);
      }
    }
    return out;
  }
}

/* ====================== Pad vector helper (rays) ========================= */

({double x, double y, bool valid}) _avgPadVectorFromRays({
  required List<RayHit> rays,
  required double px,
  required double py,
}) {
  double sx = 0.0, sy = 0.0, wsum = 0.0;
  for (final r in rays) {
    if (r.kind != RayHitKind.pad) continue;
    final dx = r.p.x - px;
    final dy = r.p.y - py;
    final d2 = dx * dx + dy * dy;
    if (d2 <= 1e-9) continue;
    final w = 1.0 / math.sqrt(d2 + 1e-6);
    sx += w * dx; sy += w * dy; wsum += w;
  }
  if (wsum <= 0) return (x: 0.0, y: 0.0, valid: false);
  final inv = 1.0 / wsum;
  return (x: sx * inv, y: sy * inv, valid: true);
}

/* ============================= Teacher Heuristic ========================== */

int predictiveIntentLabelAdaptive(
    eng.GameEngine env, {
      double baseTauSec = 1.0,
      double minTauSec = 0.45,
      double maxTauSec = 1.35,
    }) {
  final L = env.lander;
  final T = env.terrain;
  final cfg = env.cfg;

  final px = L.pos.x.toDouble();
  final py = L.pos.y.toDouble();
  final vx = L.vel.x.toDouble();
  final vy = L.vel.y.toDouble();

  final padCx = T.padCenter.toDouble();
  final gy = T.heightAt(px);
  final h  = (gy - py).toDouble().clamp(0.0, 1e9);
  final W  = cfg.worldW.toDouble();

  final hNorm = (h / 320.0).clamp(0.0, 1.6);
  final tau   = (baseTauSec * (0.7 + 0.5 * hNorm)).clamp(minTauSec, maxTauSec);

  final g   = cfg.t.gravity;
  final xF  = px + vx * tau;
  final vyF = vy + g * tau;

  double pdx = 0.0, pdy = 0.0; bool padVecValid = false;
  double sx = 0.0, sy = 0.0, wsum = 0.0;
  for (final r in env.rays) {
    if (r.kind != RayHitKind.pad) continue;
    final dx = r.p.x - px; final dy = r.p.y - py;
    final d2 = dx*dx + dy*dy; if (d2 <= 1e-9) continue;
    final w = 1.0 / math.sqrt(d2 + 1e-6);
    sx += w * dx; sy += w * dy; wsum += w;
  }
  if (wsum > 0.0) {
    final inv = 1.0 / wsum;
    pdx = sx * inv; pdy = sy * inv;
    padVecValid = true;
  } else {
    pdx = (padCx - px);
    pdy = 0.0;
  }

  double crossZ(double ax, double ay, double bx, double by) => ax*by - ay*bx;
  double dot   (double ax, double ay, double bx, double by) => ax*bx + ay*by;

  final padLen = math.sqrt(pdx*pdx + pdy*pdy);
  final pnx = padLen > 1e-6 ? (pdx / padLen) : 0.0;
  final pny = padLen > 1e-6 ? (pdy / padLen) : 0.0;

  final vFx = vx;
  final vFy = vyF;
  final vFmag = math.sqrt(vFx*vFx + vFy*vFy) + 1e-9;

  double _vCapBrakeUp(double hh)=> (0.07 * hh + 6.0).clamp(6.0, 16.0);

  // Emergency brake-up near pad center
  final vCapStrong   = _vCapBrakeUp(h);
  final tooLow       = h < 140.0;
  final tooFastDown  = vyF > math.max(40.0, vCapStrong + 10.0);
  final nearPadLat   = (px - padCx).abs() <= 0.18 * W;
  if (tooLow && tooFastDown && nearPadLat) {
    return intentToIndex(Intent.brakeUp);
  }

  // centered laterally → manage locally
  final padEnter = 0.08 * W;
  final dxNow    = px - padCx;
  if (dxNow.abs() <= padEnter) {
    if (vx >  25.0) return intentToIndex(Intent.brakeRight);
    if (vx < -25.0) return intentToIndex(Intent.brakeLeft);
    return intentToIndex(Intent.descendSlow);
  }

  // will cross center within τ?
  final dxF = xF - padCx;
  final crossesCenter = (dxNow * dxF) < 0.0;
  if (crossesCenter) {
    if (vx >  12.0) return intentToIndex(Intent.brakeRight);
    if (vx < -12.0) return intentToIndex(Intent.brakeLeft);
    return intentToIndex(Intent.descendSlow);
  }

  // drifting away?
  final driftingAway = dxF.abs() > dxNow.abs() + 2.0;
  if (driftingAway) {
    return (dxNow > 0.0) ? intentToIndex(Intent.goLeft)
        : intentToIndex(Intent.goRight);
  }

  final bool nearlyNoVx = vx.abs() < 6.0;
  if (!crossesCenter && !driftingAway && nearlyNoVx) {
    if (dxNow.abs() > padEnter) {
      return (dxNow > 0.0) ? intentToIndex(Intent.goLeft)
          : intentToIndex(Intent.goRight);
    }
  }

  if (padVecValid) {
    final cp = crossZ(vFx, vFy, pdx, pdy);   // >0: pad left of vF
    final dp = dot   (vFx, vFy, pdx, pdy);

    final cpThresh = 0.015 * vFmag * (padLen > 1.0 ? padLen : 1.0);
    final dpBad    = -0.030 * vFmag * (padLen > 1.0 ? padLen : 1.0);

    final misaligned = (cp.abs() > cpThresh) || (dp < dpBad);
    if (misaligned) {
      return (cp > 0.0) ? intentToIndex(Intent.goLeft)
          : intentToIndex(Intent.goRight);
    }
  } else {
    double _vCapHover(double hh)  => (0.06 * hh + 6.0).clamp(6.0, 18.0);
    final vCapHover = _vCapHover(h);
    final needUp    = (vy > vCapHover) || (vyF > 0.85 * vCapHover);
    if (needUp) return intentToIndex(Intent.brakeUp);
  }

  final willExitSoon = (dxF.abs() > padEnter) && (dxNow.abs() <= padEnter);
  final vxIsOutward  = (dxNow.sign == vx.sign) && vx.abs() > 18.0;
  if ((willExitSoon || vxIsOutward) && h > 90.0) {
    return (dxNow >= 0.0) ? intentToIndex(Intent.goLeft)
        : intentToIndex(Intent.goRight);
  }

  return intentToIndex(Intent.descendSlow);
}

/* ============================== Controllers =============================== */

double _vCapHover(double h)  => (0.06 * h + 6.0).clamp(6.0, 18.0);
double _vCapDesc(double h)   => (0.10 * h + 8.0).clamp(8.0, 26.0);
double _vCapBrakeUp(double h)=> (0.07 * h + 6.0).clamp(6.0, 16.0);

et.ControlInput controllerForIntent(Intent intent, eng.GameEngine env) {
  final L = env.lander;
  final T = env.terrain;
  final px = L.pos.x.toDouble();
  final gy = T.heightAt(px);
  final h  = (gy - L.pos.y).toDouble().clamp(0.0, 1e9);
  final vx = L.vel.x.toDouble();
  final vy = L.vel.y.toDouble();

  bool needUp(double cap, {double tau = 1.0, double warn = 0.80, double pad = 2.0}) {
    final g = env.cfg.t.gravity;
    final vyNext = vy + g * tau;
    final warnCap = cap * warn;
    return (vy > warnCap - pad) || (vyNext > warnCap);
  }

  bool thr=false, left=false, right=false, sL=false, sR=false, dT=false;

  switch (intent) {
    case Intent.brakeUp: {
      final vCap = _vCapBrakeUp(h);
      final needUpNow = vy > (0.9 * vCap) || (vy + env.cfg.t.gravity * 1.6) > 0.85 * vCap;
      return et.ControlInput(thrust: needUpNow, left: false, right: false, sideLeft: false, sideRight: false, downThrust: false);
    }
    case Intent.descendSlow: { thr = needUp(_vCapDesc(h), tau: 1.3, warn: 0.80, pad: 2.0); break; }
    case Intent.brakeLeft:  { right = (vx < -3.0); thr = needUp(_vCapHover(h), tau: 1.2); break; }
    case Intent.brakeRight: { left  = (vx >  3.0); thr = needUp(_vCapHover(h), tau: 1.2); break; }
    case Intent.goLeft:     { left  = true; thr = needUp(_vCapHover(h), tau: 1.2); break; }
    case Intent.goRight:    { right = true; thr = needUp(_vCapHover(h), tau: 1.2); break; }
    case Intent.hover:
    default: { thr = needUp(_vCapHover(h), tau: 1.2); break; }
  }

  return et.ControlInput(
    thrust: thr, left: left, right: right,
    sideLeft: sL, sideRight: sR, downThrust: dT,
  );
}

/* ============================= Policy Network ============================= */

class ForwardCache {
  final List<double> x;
  final List<List<double>> acts;
  final List<double> intentLogits;
  final List<double> intentProbs;
  final List<double> turnLogits;
  final double thrLogit;
  final double thrProb;
  final double v;
  final double durLogit;
  final double durFrames;
  ForwardCache({
    required this.x, required this.acts,
    required this.intentLogits, required this.intentProbs,
    required this.turnLogits, required this.thrLogit, required this.thrProb,
    required this.v, required this.durLogit, required this.durFrames,
  });
}

class _Lin {
  List<List<double>> W; List<double> b;
  _Lin(int outDim, int inDim, int seed)
      : W = List.generate(outDim, (_) {
    final r = math.Random(seed ^= 0x9E3779B9);
    final s = 1.0 / math.sqrt(inDim.toDouble());
    return List<double>.generate(inDim, (_) => (r.nextDouble()*2-1) * 0.05 * s);
  }),
        b = List<double>.filled(outDim, 0.0);
  List<double> forward(List<double> x) {
    final z = List<double>.filled(b.length, 0.0);
    for (int i = 0; i < W.length; i++) {
      double s = b[i]; final Wi = W[i];
      for (int j = 0; j < Wi.length; j++) s += Wi[j] * x[j];
      z[i] = s;
    }
    return z;
  }
  List<double> backward({
    required List<double> x,
    required List<double> dOut,
    required List<List<double>> gW,
    required List<double> gb,
  }) {
    for (int i = 0; i < W.length; i++) {
      gb[i] += dOut[i];
      for (int j = 0; j < W[0].length; j++) gW[i][j] += dOut[i] * x[j];
    }
    final dIn = List<double>.filled(W[0].length, 0.0);
    for (int j = 0; j < W[0].length; j++) {
      double s = 0.0; for (int i = 0; i < W.length; i++) s += dOut[i] * W[i][j];
      dIn[j] = s;
    }
    return dIn;
  }
}

class PolicyNetwork {
  static const int kIntents = 7;

  final int inputSize;
  final List<int> hidden;
  final nn.MLPTrunk trunk;
  final nn.PolicyHeads heads;
  late final _Lin durHead;

  bool _trainTrunk = true, _trainIntent = true, _trainAction = true, _trainValue = true;

  PolicyNetwork({
    required this.inputSize,
    List<int> hidden = const [64, 64],
    int seed = 0,
  }) : hidden = List<int>.from(hidden),
        trunk  = nn.MLPTrunk(
          inputSize: inputSize,
          hiddenSizes: hidden,
          seed: seed ^ 0x7777,
          activation: nn.Activation.silu,
          trainNoiseStd: 0.02,
          dropoutProb: 0.10,
          trainMode: false,
        ),
        heads  = nn.PolicyHeads(
          hidden.isEmpty ? inputSize : hidden.last, intents: kIntents, seed: seed ^ 0x8888,
        ) {
    final H = hidden.isEmpty ? inputSize : hidden.last;
    durHead = _Lin(1, H, seed ^ 0xDAA7);
  }

  void setTrunkTrainable(bool on) { _trainTrunk = on; }
  void setHeadsTrainable({bool intent = true, bool action = true, bool value = true}) {
    _trainIntent = intent; _trainAction = action; _trainValue = value;
  }

  ForwardCache forwardFull(List<double> x) {
    final acts = trunk.forwardAll(x);
    final h    = acts.last;
    final il   = heads.intent.forward(h);
    final ip   = nn.Ops.softmax(il);
    final tl   = heads.turn.forward(h);
    final thl  = heads.thr.forward(h)[0];
    final thp  = nn.Ops.sigmoid(thl);
    final v    = heads.val.forward(h)[0];
    final dLog = durHead.forward(h)[0];
    double dF  = nn.Ops.softplus(dLog).clamp(1.0, 24.0);
    return ForwardCache(
      x: List<double>.from(x), acts: acts,
      intentLogits: il, intentProbs: ip,
      turnLogits: tl, thrLogit: thl, thrProb: thp, v: v,
      durLogit: dLog, durFrames: dF,
    );
  }

  (int, List<double>, List<double>, ForwardCache) actIntent(List<double> x) {
    final c = forwardFull(x);
    int arg = 0; double best = c.intentProbs[0];
    for (int i = 1; i < c.intentProbs.length; i++) { if (c.intentProbs[i] > best) { best = c.intentProbs[i]; arg = i; } }
    return (arg, c.intentProbs, c.intentLogits, c);
  }

  (int, List<double>, ForwardCache) actIntentGreedy(List<double> x) {
    final c = forwardFull(x);
    int arg = 0; double best = c.intentProbs[0];
    for (int i = 1; i < c.intentProbs.length; i++) { if (c.intentProbs[i] > best) { best = c.intentProbs[i]; arg = i; } }
    return (arg, c.intentProbs, c);
  }

  (bool, bool, bool, List<double>, ForwardCache) actGreedy(List<double> x) {
    final c = forwardFull(x);
    final thrust = c.thrProb >= 0.5;
    int tArg = 0; double best = c.turnLogits[0];
    for (int i = 1; i < 3; i++) { if (c.turnLogits[i] > best) { best = c.turnLogits[i]; tArg = i; } }
    final left = (tArg == 0), right = (tArg == 2);
    final probs = <double>[ c.thrProb, ...nn.Ops.softmax(c.turnLogits) ];
    return (thrust, left, right, probs, c);
  }

  /// Pragmatic training hook used by curricula & Trainer.
  void updateFromEpisode({
    required List<ForwardCache> decisionCaches,
    required List<int> intentChoices,
    required List<double> decisionReturns,
    required List<int> alignLabels,
    required double alignWeight,
    required double intentPgWeight,
    required double lr,
    required double entropyBeta,
    required double valueBeta,
    required double huberDelta,
    required bool intentMode,
    List<ForwardCache>? actionCaches,
    List<int>? actionTurnTargets,
    List<bool>? actionThrustTargets,
    double actionAlignWeight = 0.0,
    List<ForwardCache>? durationCaches,
    List<double>? durationTargets,
    double durationAlignWeight = 0.0,
  }) {
    double _clip(double g, [double c = 1.0]) => g.isFinite ? g.clamp(-c, c) : 0.0;

    final gW_int  = List.generate(heads.intent.W.length, (_) => List<double>.filled(heads.intent.W[0].length, 0.0));
    final gb_int  = List<double>.filled(heads.intent.b.length, 0.0);
    final gW_turn = List.generate(heads.turn.W.length , (_) => List<double>.filled(heads.turn.W[0].length , 0.0));
    final gb_turn = List<double>.filled(heads.turn.b.length , 0.0);
    final gW_thr  = List.generate(heads.thr .W.length , (_) => List<double>.filled(heads.thr .W[0].length , 0.0));
    final gb_thr  = List<double>.filled(heads.thr .b.length , 0.0);

    final int H = trunk.layers.isEmpty ? inputSize : trunk.layers.last.b.length;
    final gW_dur = [ List<double>.filled(H, 0.0) ];
    final gb_dur = [ 0.0 ];

    List<List<List<double>>> gW_trunk = [
      for (final lin in trunk.layers)
        List.generate(lin.W.length, (_) => List<double>.filled(lin.W[0].length, 0.0))
    ];
    List<List<double>> gb_trunk = [
      for (final lin in trunk.layers) List<double>.filled(lin.b.length, 0.0)
    ];

    // Intent CE
    if (intentMode && alignWeight > 0 && decisionCaches.isNotEmpty) {
      final N = decisionCaches.length;
      for (int n = 0; n < N; n++) {
        final c = decisionCaches[n];
        final h = c.acts.last;
        final y = alignLabels[n].clamp(0, PolicyNetwork.kIntents - 1);
        final dLog = nn.Ops.crossEntropyGrad(c.intentProbs, y);
        if (entropyBeta > 0) {
          for (int i = 0; i < dLog.length; i++) dLog[i] += entropyBeta * c.intentProbs[i];
        }
        final dH = heads.intent.backward(x: h, dOut: dLog, gW: gW_int, gb: gb_int);
        if (_trainTrunk) trunk.backwardFromTopGrad(dTop: dH, acts: c.acts, gW: gW_trunk, gb: gb_trunk, x0: c.x);
      }
      final scale = lr * alignWeight / N;
      if (_trainIntent) {
        for (int i = 0; i < heads.intent.b.length; i++) {
          heads.intent.b[i] -= _clip(scale * gb_int[i]);
          for (int j = 0; j < heads.intent.W[0].length; j++) heads.intent.W[i][j] -= _clip(scale * gW_int[i][j]);
        }
      }
      // zero accumulators
      for (int li = 0; li < gb_trunk.length; li++) {
        for (int j = 0; j < gb_trunk[li].length; j++) gb_trunk[li][j] = 0.0;
        for (int r = 0; r < gW_trunk[li].length; r++) {
          for (int c2 = 0; c2 < gW_trunk[li][0].length; c2++) gW_trunk[li][r][c2] = 0.0;
        }
      }
      for (int i = 0; i < gb_int.length; i++) gb_int[i] = 0.0;
      for (int i = 0; i < gW_int.length; i++) { for (int j = 0; j < gW_int[0].length; j++) gW_int[i][j] = 0.0; }
    }

    // Intent PG
    if (intentPgWeight > 0 && decisionCaches.isNotEmpty) {
      final N = decisionCaches.length;
      for (int n = 0; n < N; n++) {
        final c = decisionCaches[n];
        final h = c.acts.last;
        final chosen = intentChoices[n].clamp(0, PolicyNetwork.kIntents - 1);
        final adv = decisionReturns[n];
        final dLog = List<double>.generate(c.intentProbs.length, (i) => (c.intentProbs[i] - (i == chosen ? 1.0 : 0.0)) * (-adv));
        final dH = heads.intent.backward(x: h, dOut: dLog, gW: gW_int, gb: gb_int);
        if (_trainTrunk) trunk.backwardFromTopGrad(dTop: dH, acts: c.acts, gW: gW_trunk, gb: gb_trunk, x0: c.x);
      }
      if (_trainIntent) {
        final scale = lr * intentPgWeight / decisionCaches.length;
        for (int i = 0; i < heads.intent.b.length; i++) {
          heads.intent.b[i] -= _clip(scale * gb_int[i]);
          for (int j = 0; j < heads.intent.W[0].length; j++) heads.intent.W[i][j] -= _clip(scale * gW_int[i][j]);
        }
      }
      for (int i = 0; i < gb_int.length; i++) gb_int[i] = 0.0;
      for (int i = 0; i < gW_int.length; i++) { for (int j = 0; j < gW_int[0].length; j++) gW_int[i][j] = 0.0; }
    }

    // Action supervision
    final hasAction = actionAlignWeight > 0.0 &&
        actionCaches != null && actionTurnTargets != null && actionThrustTargets != null &&
        actionCaches.isNotEmpty;

    if (hasAction) {
      final M = actionCaches!.length;
      double meanThrLogit = 0.0, teacherThrRate = 0.0;
      for (int n = 0; n < M; n++) {
        final c = actionCaches[n];
        final h = c.acts.last;
        final turnProbs = nn.Ops.softmax(c.turnLogits);
        final yt = actionTurnTargets![n].clamp(0, 2);
        final dTurn = nn.Ops.crossEntropyGrad(turnProbs, yt, numClasses: 3);
        final yb = actionThrustTargets![n] ? 1.0 : 0.0;
        final dThr = nn.Ops.bceGradFromLogit(c.thrLogit, yb);
        teacherThrRate += yb;

        final dH_turn = heads.turn.backward(x: h, dOut: dTurn, gW: gW_turn, gb: gb_turn);
        final dH_thr  = heads.thr .backward(x: h, dOut: [dThr], gW: gW_thr , gb: gb_thr );
        final dH = List<double>.filled(h.length, 0.0);
        for (int i = 0; i < h.length; i++) dH[i] = dH_turn[i] + dH_thr[i];
        if (_trainTrunk) trunk.backwardFromTopGrad(dTop: dH, acts: c.acts, gW: gW_trunk, gb: gb_trunk, x0: c.x);

        meanThrLogit += c.thrLogit;
      }
      final scale = lr * actionAlignWeight / M;
      if (_trainAction) {
        for (int i = 0; i < heads.turn.b.length; i++) {
          heads.turn.b[i] -= _clip(scale * gb_turn[i]);
          for (int j = 0; j < heads.turn.W[0].length; j++) heads.turn.W[i][j] -= _clip(scale * gW_turn[i][j]);
        }
        heads.thr.b[0] -= _clip(scale * gb_thr[0]);
        for (int j = 0; j < heads.thr.W[0].length; j++) heads.thr.W[0][j] -= _clip(scale * gW_thr[0][j]);
      }
      // bias calibration to match teacher thrust rate
      meanThrLogit /= M; teacherThrRate /= M;
      final logitTarget = nn.Ops.logit(teacherThrRate);
      if (_trainAction) heads.thr.b[0] += (logitTarget - meanThrLogit) * 0.25;
    }

    // Duration supervision (optional)
    final hasDur = durationAlignWeight > 0.0 && durationCaches != null && durationTargets != null && durationCaches.isNotEmpty;
    if (hasDur) {
      final N = durationCaches!.length;
      for (int n = 0; n < N; n++) {
        final c = durationCaches[n];
        final h = c.acts.last;
        final y = durationTargets[n].clamp(1.0, 24.0);
        final z = c.durLogit;
        final yhat = nn.Ops.softplus(z);
        final eps = 1e-6;
        final logDiff = (math.log(yhat + eps) - math.log(y + eps));
        final sig = nn.Ops.sigmoid(z);
        final dZ = logDiff * (sig / (yhat + eps));
        final dH = durHead.backward(x: h, dOut: [dZ], gW: gW_dur, gb: gb_dur);
        if (_trainTrunk) trunk.backwardFromTopGrad(dTop: dH, acts: c.acts, gW: gW_trunk, gb: gb_trunk, x0: c.x);
      }
      final scale = lr * durationAlignWeight / durationCaches.length;
      durHead.b[0] -= _clip(scale * gb_dur[0]);
      for (int j = 0; j < durHead.W[0].length; j++) durHead.W[0][j] -= _clip(scale * gW_dur[0][j]);
    }

    // Apply trunk update (SGD)
    if (_trainTrunk) {
      final trunkScale = lr;
      for (int li = 0; li < trunk.layers.length; li++) {
        final L = trunk.layers[li];
        final gb = gb_trunk[li];
        final gW = gW_trunk[li];
        for (int i = 0; i < L.b.length; i++) L.b[i] -= _clip(trunkScale * gb[i]);
        for (int i = 0; i < L.W.length; i++) {
          for (int j = 0; j < L.W[0].length; j++) L.W[i][j] -= _clip(trunkScale * gW[i][j]);
        }
      }
    }
  }
}

/* =============================== Trainer ================================== */

class EpisodeResult {
  final int steps; final double totalCost; final bool landed; final double segMean;
  EpisodeResult({required this.steps, required this.totalCost, required this.landed, required this.segMean});
}

typedef ExternalRewardHook = double Function({required eng.GameEngine env, required double dt, required int tStep});

class Trainer {
  final eng.GameEngine env;
  final FeatureExtractorRays fe;
  final PolicyNetwork policy;
  final double dt;
  final double gamma;
  final int seed;
  final bool twoStage;
  final int planHold;
  double tempIntent;
  double intentEntropyBeta;
  bool useLearnedController;
  final double blendPolicy;
  final double intentAlignWeight;
  final double intentPgWeight;
  final double actionAlignWeight;
  final bool normalizeFeatures;

  final double gateScoreMin;
  final bool gateOnlyLanded;
  final bool gateVerbose;

  final ExternalRewardHook? externalRewardHook;

  final RunningNorm? norm;
  int _epCounter = 0;

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

  int _sampleFromLogits(List<double> logits, math.Random r, double temp) {
    final T = temp.clamp(1e-6, 10.0);
    final z = List<double>.from(logits);
    for (int i = 0; i < z.length; i++) z[i] /= T;
    final maxZ = z.reduce((a,b) => a>b? a:b);
    double sum = 0.0; final exps = List<double>.filled(z.length, 0.0);
    for (int i = 0; i < z.length; i++) { final e = math.exp(z[i] - maxZ); exps[i] = e; sum += e; }
    final u = r.nextDouble() * sum; double acc = 0.0;
    for (int i = 0; i < exps.length; i++) { acc += exps[i]; if (u <= acc) return i; }
    return exps.length - 1;
  }

  EpisodeResult runEpisode({
    required bool train,
    required bool greedy,
    required bool scoreIsReward,
    double lr = 3e-4,
    double valueBeta = 0.5,
    double huberDelta = 1.0,
  }) {
    policy.trunk.trainMode = train;
    final r = math.Random(seed ^ (_epCounter++));
    final decisionRewards = <double>[];

    final actionCaches = <ForwardCache>[];
    final actionTurnTargets = <int>[];
    final actionThrustTargets = <bool>[];

    final decisionCaches = <ForwardCache>[];
    final intentChoices = <int>[];
    final decisionReturns = <double>[];
    final alignLabels = <int>[];

    final durationCaches = <ForwardCache>[];
    final durationTargets = <double>[];

    double totalCost = 0.0;
    double segSum = 0.0; int segCount = 0;

    int framesLeft = 0;
    int currentIntentIdx = 0;

    int steps = 0;
    bool landed = false;

    while (true) {
      if (framesLeft <= 0) {
        var x = fe.extract(
          lander: env.lander, terrain: env.terrain,
          worldW: env.cfg.worldW, worldH: env.cfg.worldH, rays: env.rays,
        );
        if (normalizeFeatures) { norm?.observe(x); x = norm?.normalize(x, update: false) ?? x; }

        // Teacher label (strong heuristic)
        int yTeacher = predictiveIntentLabelAdaptive(env);

        final (_idx, _p, logits, cache) = policy.actIntent(x);
        final idx = greedy ? _idx : _sampleFromLogits(logits, r, tempIntent);
        currentIntentIdx = idx;

        if (train) {
          decisionCaches.add(cache);
          intentChoices.add(idx);
          decisionRewards.add(0.0);
          alignLabels.add(yTeacher);
        }

        // use duration head prediction as hold target
        final teacherHold = cache.durFrames;
        framesLeft = teacherHold.round().clamp(1, 24);
        if (train) { durationCaches.add(cache); durationTargets.add(teacherHold); }
      }

      var xAct = fe.extract(
        lander: env.lander, terrain: env.terrain,
        worldW: env.cfg.worldW, worldH: env.cfg.worldH, rays: env.rays,
      );
      xAct = norm?.normalize(xAct, update: false) ?? xAct;

      final (thBool, lf, rt, probs, cAct) = policy.actGreedy(xAct);

      final intent = indexToIntent(currentIntentIdx);
      final uTeach = controllerForIntent(intent, env);

      final pThrModel = probs[0].clamp(0.0, 1.0);
      final pThrTeach = uTeach.thrust ? 1.0 : 0.0;
      final pThrExec  = blendPolicy * pThrModel + (1.0 - blendPolicy) * pThrTeach;
      final execThrust = pThrExec >= 0.5;
      final execLeft   = useLearnedController ? lf : uTeach.left;
      final execRight  = useLearnedController ? rt : uTeach.right;

      final r_pf = externalRewardHook?.call(env: env, dt: dt, tStep: steps) ?? 0.0;
      segSum += r_pf; segCount++;

      if (train) {
        actionCaches.add(cAct);
        actionTurnTargets.add(uTeach.left ? 0 : (uTeach.right ? 2 : 1));
        actionThrustTargets.add(uTeach.thrust);
      }

      final info = env.step(dt, et.ControlInput(
        thrust: execThrust, left: execLeft, right: execRight,
        sideLeft: uTeach.sideLeft, sideRight: uTeach.sideRight, downThrust: uTeach.downThrust,
        intentIdx: currentIntentIdx,
      ));
      totalCost += info.costDelta;
      steps++; framesLeft--;

      if (info.terminal) { landed = env.status == et.GameStatus.landed; break; }
      if (steps > 5000) break;
    }

    final segMean = (segCount > 0) ? (segSum / segCount) : 0.0;

    if (train && (decisionCaches.isNotEmpty || actionCaches.isNotEmpty)) {
      // simple zero-mean advantages for compatibility
      final returns = List<double>.filled(decisionCaches.length, 0.0);
      policy.updateFromEpisode(
        decisionCaches: decisionCaches,
        intentChoices: intentChoices,
        decisionReturns: returns,
        alignLabels: alignLabels,
        alignWeight: intentAlignWeight,
        intentPgWeight: intentPgWeight,
        lr: lr,
        entropyBeta: intentEntropyBeta,
        valueBeta: valueBeta,
        huberDelta: huberDelta,
        intentMode: true,
        actionCaches: actionCaches,
        actionTurnTargets: actionTurnTargets,
        actionThrustTargets: actionThrustTargets,
        actionAlignWeight: actionAlignWeight,
        durationCaches: durationCaches,
        durationTargets: durationTargets,
        durationAlignWeight: 0.25,
      );
    }

    if (gateVerbose) {
      final tag = landed ? 'L' : 'NL';
      // ignore: avoid_print
      print('[EP] steps=$steps segMean=${segMean.toStringAsFixed(3)} $tag');
    }

    return EpisodeResult(steps: steps, totalCost: totalCost, landed: landed, segMean: segMean);
  }
}

/* ========================== DualRuntimePolicy ============================= */

class DualRuntimePolicy {
  final PolicyNetwork _mono;
  bool _usePadPlanner = false;
  bool _stochasticPlanner = false;
  double _intentTemp = 1.0;

  DualRuntimePolicy({required int inputSize, List<int> hidden = const [64,64], int seed = 0})
      : _mono = PolicyNetwork(inputSize: inputSize, hidden: hidden, seed: seed);

  int get inputSize => _mono.inputSize;
  List<int> get hidden => _mono.hidden;
  RunningNorm? norm;

  void setStochasticPlanner(bool on) { _stochasticPlanner = on; }
  void setIntentTemperature(double t) { _intentTemp = t.clamp(0.05, 3.0); }
  void usePadAlignPlanner() { _usePadPlanner = true; }

  (bool thrust, bool left, bool right) act(eng.GameEngine env, FeatureExtractorRays fe) {
    if (_usePadPlanner) {
      final idx = predictiveIntentLabelAdaptive(env);
      final u = controllerForIntent(indexToIntent(idx), env);
      return (u.thrust, u.left, u.right);
    } else {
      var x = fe.extract(
        lander: env.lander, terrain: env.terrain,
        worldW: env.cfg.worldW, worldH: env.cfg.worldH, rays: env.rays,
      );
      final (thr, l, r, _p, _c) = _mono.actGreedy(x);
      return (thr, l, r);
    }
  }

  void setTrunkTrainable(bool on) => _mono.setTrunkTrainable(on);
  void setHeadsTrainable({bool intent = true, bool action = true, bool value = true}) =>
      _mono.setHeadsTrainable(intent: intent, action: action, value: value);

  PolicyNetwork asClassic() => _mono;
}
