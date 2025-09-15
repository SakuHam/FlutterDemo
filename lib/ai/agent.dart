// lib/ai/agent.dart
import 'dart:math' as math;
import '../engine/types.dart' as et;
import '../engine/game_engine.dart' as eng;

/* -------------------------------------------------------------------------- */
/*                                INTENT SET                                  */
/* -------------------------------------------------------------------------- */

enum Intent { hover, goLeft, goRight, descendSlow, brakeUp }

const List<String> kIntentNames = [
  'hover',
  'goLeft',
  'goRight',
  'descendSlow',
  'brakeUp',
];

int intentToIndex(Intent i) => i.index;
Intent indexToIntent(int k) => Intent.values[k.clamp(0, Intent.values.length - 1)];

double _clip(double x, double a, double b) => x < a ? a : (x > b ? b : x);

/* -------------------------------------------------------------------------- */
/*                               RUNNING  NORM                                */
/* -------------------------------------------------------------------------- */

class RunningNorm {
  final int dim;
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

  // Preferred way to update the statistics
  void observe(List<double> x) {
    if (x.length != dim) throw ArgumentError('RunningNorm dim mismatch: got ${x.length}, want $dim');
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

  // Normalize copy of x; optionally update first
  List<double> normalize(List<double> x, {bool update = false}) {
    if (update) observe(x);
    final out = List<double>.filled(x.length, 0.0);
    for (int i = 0; i < x.length; i++) {
      out[i] = (x[i] - mean[i]) / math.sqrt(var_[i] + eps);
    }
    return out;
  }
}

/* -------------------------------------------------------------------------- */
/*                             FEATURE EXTRACTOR                               */
/* -------------------------------------------------------------------------- */

class FeatureExtractor {
  final int groundSamples;
  final double stridePx;

  FeatureExtractor({this.groundSamples = 3, this.stridePx = 48});

  // Layout (must match training):
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

    // approximate slope via symmetric heights
    final hL = T.heightAt(_clip(px - 8.0, 0.0, env.cfg.worldW));
    final hR = T.heightAt(_clip(px + 8.0, 0.0, env.cfg.worldW));
    final slope = (hR - hL) / 16.0;

    final feats = <double>[
      px / env.cfg.worldW, // scale spatial to ~[0,1]
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

    // local ground samples ahead/back
    for (int i = 0; i < groundSamples; i++) {
      final off = (i - (groundSamples ~/ 2)) * stridePx;
      final gx = _clip(px + off, 0.0, env.cfg.worldW);
      final gyS = T.heightAt(gx);
      feats.add((gyS - py) / 300.0);
    }
    return feats;
  }
}

/* -------------------------------------------------------------------------- */
/*                                CONTROLLERS                                  */
/* -------------------------------------------------------------------------- */

// Simple heuristic teacher for intents
int predictiveIntentLabelAdaptive(
    eng.GameEngine env, {
      double baseTauSec = 1.0,
      double minTauSec = 0.45,
      double maxTauSec = 1.35,
    }) {
  final L = env.lander;
  final T = env.terrain;
  final padCx = T.padCenter.toDouble();
  final px = L.pos.x.toDouble();
  final py = L.pos.y.toDouble();
  final vx = L.vel.x.toDouble();
  final vy = L.vel.y.toDouble();

  final gy = T.heightAt(px);
  final height = gy - py;
  final dx = px - padCx;

  // emergency brake if falling fast near ground
  if (height < 80 && vy > 80) return intentToIndex(Intent.brakeUp);

  // high descent → descendSlow
  if (height > 120 && vy > 20 && dx.abs() < env.cfg.worldW * 0.15) {
    return intentToIndex(Intent.descendSlow);
  }

  // go left/right if far from pad
  if (dx > env.cfg.worldW * 0.08) return intentToIndex(Intent.goRight);
  if (dx < -env.cfg.worldW * 0.08) return intentToIndex(Intent.goLeft);

  // hover near target
  return intentToIndex(Intent.hover);
}

// Map an intent to a discrete control (heuristic teacher)
et.ControlInput controllerForIntent(Intent intent, eng.GameEngine env) {
  final L = env.lander;
  final T = env.terrain;
  final px = L.pos.x.toDouble();
  final gy = T.heightAt(px);
  final height = gy - L.pos.y;
  final dx = L.pos.x - T.padCenter;

  switch (intent) {
    case Intent.brakeUp:
      return const et.ControlInput(thrust: true, left: false, right: false);
    case Intent.descendSlow:
    // small thrust to moderate descent
      final need = (L.vel.y > 25 || height < 100);
      return et.ControlInput(thrust: need, left: false, right: false);
    case Intent.goLeft:
    // rotate and nudge thrust if low
      return et.ControlInput(thrust: height < 100, left: true, right: false);
    case Intent.goRight:
      return et.ControlInput(thrust: height < 100, left: false, right: true);
    case Intent.hover:
    default:
    // thrust if sinking fast
      return et.ControlInput(thrust: L.vel.y > 8, left: false, right: false);
  }
}

/* -------------------------------------------------------------------------- */
/*                               POLICY NETWORK                                */
/* -------------------------------------------------------------------------- */

class ForwardCache {
  final List<double> x;     // normalized input
  final List<double> h1;
  final List<double> h2;
  final List<double> intentLogits;
  final List<double> intentProbs;
  final List<double> turnLogits; // 3 logits: left, none, right
  final double thrLogit;
  final double thrProb;
  final double v; // value head
  ForwardCache({
    required this.x,
    required this.h1,
    required this.h2,
    required this.intentLogits,
    required this.intentProbs,
    required this.turnLogits,
    required this.thrLogit,
    required this.thrProb,
    required this.v,
  });
}

class PolicyNetwork {
  final int inputSize;
  final int h1;
  final int h2;
  static const int kIntents = 5;

  // Shared trunk
  List<List<double>> W1;
  List<double> b1;
  List<List<double>> W2;
  List<double> b2;

  // Heads
  List<List<double>> W_intent;
  List<double> b_intent;

  List<List<double>> W_turn; // 3 x h2 → softmax over [-1,0,1]
  List<double> b_turn;

  List<List<double>> W_thr; // 1 x h2 → sigmoid
  List<double> b_thr;

  List<List<double>> W_val; // 1 x h2 → scalar value
  List<double> b_val;

  PolicyNetwork({
    required this.inputSize,
    this.h1 = 64,
    this.h2 = 64,
    int seed = 0,
  })  : W1 = _xavier(h: 64, w: inputSize, seed: seed ^ 0x11),
        b1 = List<double>.filled(64, 0.0),
        W2 = _xavier(h: 64, w: 64, seed: seed ^ 0x22),
        b2 = List<double>.filled(64, 0.0),
        W_intent = _xavier(h: kIntents, w: 64, seed: seed ^ 0x33),
        b_intent = List<double>.filled(kIntents, 0.0),
        W_turn = _xavier(h: 3, w: 64, seed: seed ^ 0x44),
        b_turn = List<double>.filled(3, 0.0),
        W_thr = _xavier(h: 1, w: 64, seed: seed ^ 0x55),
        b_thr = List<double>.filled(1, 0.0),
        W_val = _xavier(h: 1, w: 64, seed: seed ^ 0x66),
        b_val = List<double>.filled(1, 0.0);

  // --- math helpers ---
  static List<List<double>> _xavier({required int h, required int w, int seed = 0}) {
    final r = math.Random(seed);
    final limit = math.sqrt(6.0 / (h + w));
    return List.generate(h, (_) => List<double>.generate(w, (_) => (r.nextDouble()*2-1)*limit));
    // good enough for small nets
  }

  static List<double> _matVec(List<List<double>> W, List<double> x, List<double> b) {
    final h = W.length, w = x.length;
    final y = List<double>.filled(h, 0.0);
    for (int i = 0; i < h; i++) {
      double s = b[i];
      final Wi = W[i];
      for (int j = 0; j < w; j++) s += Wi[j] * x[j];
      y[i] = s;
    }
    return y;
  }

  // --- math helpers (stable) ---
  static double _tanhScalar(double x) {
    // Stable tanh without overflow:
    // tanh(x) = sign(x) * (1 - e) / (1 + e), with e = exp(-2*|x|)
    final ax = x.abs();
    if (ax > 20.0) return x.isNegative ? -1.0 : 1.0; // avoid underflow/overflow
    final e = math.exp(-2.0 * ax);
    final t = (1.0 - e) / (1.0 + e);
    return x.isNegative ? -t : t;
  }

  static void _tanhInPlace(List<double> v) {
    for (int i = 0; i < v.length; i++) {
      v[i] = _tanhScalar(v[i]);
    }
  }

  static List<double> _softmax(List<double> z) {
    // subtract max for stability
    double m = z[0];
    for (int i = 1; i < z.length; i++) if (z[i] > m) m = z[i];
    double s = 0.0;
    final out = List<double>.filled(z.length, 0.0);
    for (int i = 0; i < z.length; i++) {
      final e = math.exp(z[i] - m);
      out[i] = e.isFinite ? e : 0.0;
      s += out[i];
    }
    final inv = (s > 0 && s.isFinite) ? (1.0 / s) : 0.0;
    for (int i = 0; i < z.length; i++) out[i] *= inv;
    return out;
  }

  static double _sigmoid(double x) {
    // stable sigmoid
    if (x >= 0) {
      final ex = math.exp(-x);
      return 1.0 / (1.0 + ex);
    } else {
      final ex = math.exp(x);
      return ex / (1.0 + ex);
    }
  }
  // Forward through trunk and all heads
  ForwardCache _forwardFull(List<double> x) {
    final h1v = _matVec(W1, x, b1);
    _tanhInPlace(h1v);
    final h2v = _matVec(W2, h1v, b2);
    _tanhInPlace(h2v);

    final intentLogits = _matVec(W_intent, h2v, b_intent);
    final intentProbs = _softmax(intentLogits);

    final turnLogits = _matVec(W_turn, h2v, b_turn);
    final thrLogit = _matVec(W_thr, h2v, b_thr)[0];
    final thrProb = _sigmoid(thrLogit);

    final v = _matVec(W_val, h2v, b_val)[0];

    return ForwardCache(
      x: List<double>.from(x),
      h1: h1v,
      h2: h2v,
      intentLogits: intentLogits,
      intentProbs: intentProbs,
      turnLogits: turnLogits,
      thrLogit: thrLogit,
      thrProb: thrProb,
      v: v,
    );
  }

  // Intent head (greedy)
  (int, List<double>, ForwardCache) actIntentGreedy(List<double> x) {
    final cache = _forwardFull(x);
    int arg = 0;
    double best = cache.intentProbs[0];
    for (int i = 1; i < cache.intentProbs.length; i++) {
      if (cache.intentProbs[i] > best) {
        best = cache.intentProbs[i];
        arg = i;
      }
    }
    return (arg, cache.intentProbs, cache);
  }

  // Full action (greedy): thrust boolean and left/right booleans
  (bool, bool, bool, List<double>, ForwardCache) actGreedy(List<double> x) {
    final c = _forwardFull(x);
    // thrust by sigmoid>0.5
    final thrust = c.thrProb >= 0.5;

    // turn by argmax over 3 logits: 0=left,1=none,2=right
    int tArg = 0;
    double best = c.turnLogits[0];
    for (int i = 1; i < 3; i++) {
      if (c.turnLogits[i] > best) {
        best = c.turnLogits[i];
        tArg = i;
      }
    }
    final left = (tArg == 0);
    final right = (tArg == 2);
    // expose a small probs list: [p_thr, p_turn_left, p_turn_none, p_turn_right]
    final probs = <double>[
      c.thrProb,
      // softmax over turn logits for readable probs
      ..._softmax(c.turnLogits),
    ];
    return (thrust, left, right, probs, c);
  }

  // Simple supervised CE update for intent head (used by pretrain)
  void updateFromEpisode({
    required List<ForwardCache> decisionCaches,
    required List<int> intentChoices,
    required List<double> decisionReturns, // ignored in intentMode
    required List<int> alignLabels, // intent labels
    required double alignWeight,
    required double lr,
    required double entropyBeta,
    required double valueBeta,
    required double huberDelta,
    required bool intentMode,
    List<int>? actionTurnTargets, // unused here
    List<bool>? actionThrustTargets, // unused here
    double actionAlignWeight = 0.0,
  }) {
    if (!intentMode) return; // this minimal build updates only intent for pretrain
    final N = decisionCaches.length;
    if (N == 0) return;

    // Accumulate grads
    final gW2 = List.generate(h2, (_) => List<double>.filled(h1, 0.0));
    final gb2 = List<double>.filled(h2, 0.0);
    final gW1 = List.generate(h1, (_) => List<double>.filled(inputSize, 0.0));
    final gb1 = List<double>.filled(h1, 0.0);

    final gW_int = List.generate(kIntents, (_) => List<double>.filled(h2, 0.0));
    final gb_int = List<double>.filled(kIntents, 0.0);

    for (int n = 0; n < N; n++) {
      final c = decisionCaches[n];
      final y = alignLabels[n].clamp(0, kIntents - 1);
      // dL/dlogits = p - onehot(y)
      final dLog = List<double>.from(c.intentProbs);
      dLog[y] -= 1.0; // CE gradient
      for (int i = 0; i < kIntents; i++) {
        gb_int[i] += dLog[i];
        for (int j = 0; j < h2; j++) {
          gW_int[i][j] += dLog[i] * c.h2[j];
        }
      }

      // backprop to h2 through intent head only (good enough for pretrain)
      final dh2 = List<double>.filled(h2, 0.0);
      for (int j = 0; j < h2; j++) {
        double s = 0.0;
        for (int i = 0; i < kIntents; i++) s += W_intent[i][j] * dLog[i];
        // tanh' on h2 pre-activation → we stored post-activation in c.h2
        final sech2 = 1.0 - c.h2[j] * c.h2[j];
        dh2[j] = s * sech2;
      }

      // backprop to h1 through W2
      final dh1 = List<double>.filled(h1, 0.0);
      for (int j = 0; j < h2; j++) {
        gb2[j] += dh2[j];
        for (int k = 0; k < h1; k++) {
          gW2[j][k] += dh2[j] * c.h1[k];
          dh1[k] += W2[j][k] * dh2[j];
        }
      }
      for (int k = 0; k < h1; k++) {
        final sech2 = 1.0 - c.h1[k] * c.h1[k];
        dh1[k] *= sech2;
      }

      // backprop to input through W1 (we don't use it but accumulate grads for W1/b1)
      for (int k = 0; k < h1; k++) {
        gb1[k] += dh1[k];
        for (int t = 0; t < inputSize; t++) {
          gW1[k][t] += dh1[k] * c.x[t];
        }
      }
    }

    final scale = lr * alignWeight / (N > 0 ? N : 1);

    // Apply grads
    for (int i = 0; i < kIntents; i++) {
      b_intent[i] -= scale * gb_int[i];
      for (int j = 0; j < h2; j++) {
        W_intent[i][j] -= scale * gW_int[i][j];
      }
    }
    for (int j = 0; j < h2; j++) {
      b2[j] -= scale * gb2[j];
      for (int k = 0; k < h1; k++) {
        W2[j][k] -= scale * gW2[j][k];
      }
    }
    for (int k = 0; k < h1; k++) {
      b1[k] -= scale * gb1[k];
      for (int t = 0; t < inputSize; t++) {
        W1[k][t] -= scale * gW1[k][t];
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
  EpisodeResult({required this.steps, required this.totalCost, required this.landed});
}

class Trainer {
  final eng.GameEngine env;
  final FeatureExtractor fe;
  final PolicyNetwork policy;
  final double dt;
  final double gamma;
  final int seed;
  final bool twoStage;
  final int planHold;
  final double tempIntent;
  final double intentEntropyBeta;
  final bool useLearnedController;
  final double blendPolicy; // blend between teacher and student actions
  final double intentAlignWeight;
  final double actionAlignWeight;
  final bool normalizeFeatures;

  final RunningNorm? norm;
  int _epCounter = 0; // <— add this

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
    required this.actionAlignWeight,
    required this.normalizeFeatures,
  }) : norm = RunningNorm(fe.inputSize, momentum: 0.995);

  // Softmax sampling with temperature
  int _sampleCategorical(List<double> probs, math.Random r, double temp) {
    if (temp <= 1e-6) {
      int arg = 0; double best = probs[0];
      for (int i = 1; i < probs.length; i++) if (probs[i] > best) { best = probs[i]; arg = i; }
      return arg;
    }
    // reweight logits by 1/temp
    final z = probs.map((p) => math.log(_clip(p, 1e-12, 1.0))).toList();
    for (int i = 0; i < z.length; i++) z[i] /= temp;
    final sm = PolicyNetwork._softmax(z);
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
    required bool scoreIsReward,
    double lr = 3e-4,
    double valueBeta = 0.5,
    double huberDelta = 1.0,
  }) {
    final r = math.Random(seed ^ (_epCounter++));

    // Trajectory buffers (intent-mode)
    final decisionCaches = <ForwardCache>[];
    final intentChoices = <int>[];
    final decisionReturns = <double>[];
    final alignLabels = <int>[];

    env.reset(seed: r.nextInt(1 << 30));
    double totalCost = 0.0;

    int framesLeft = 0;
    int currentIntentIdx = 0;

    int steps = 0;
    bool landed = false;

    while (true) {
      if (framesLeft <= 0) {
        var x = fe.extract(env);
        final yTeacher = predictiveIntentLabelAdaptive(env,
            baseTauSec: 1.0, minTauSec: 0.45, maxTauSec: 1.35);

        if (normalizeFeatures) {
          norm?.observe(x);
          x = norm?.normalize(x, update: false) ?? x;
        }

        final (idxGreedy, p, cache) = policy.actIntentGreedy(x);
        final idx = greedy ? idxGreedy : _sampleCategorical(p, r, tempIntent);
        currentIntentIdx = idx;

        if (train) {
          decisionCaches.add(cache);
          intentChoices.add(idx);
          alignLabels.add(yTeacher);
          // use value as a baseline-free return proxy (kept compatible with your trainer)
          decisionReturns.add(cache.v);
        }

        framesLeft = planHold;
      }

      // Decide controls
      final intent = indexToIntent(currentIntentIdx);
      final uTeacher = controllerForIntent(intent, env);

      bool thrust = uTeacher.thrust;
      bool left = uTeacher.left;
      bool right = uTeacher.right;

      if (useLearnedController || blendPolicy < 1.0) {
        var xAct = fe.extract(env);
        if (normalizeFeatures) xAct = norm?.normalize(xAct, update: false) ?? xAct;
        final (th, lf, rt, _probs, _c) = policy.actGreedy(xAct);

        if (blendPolicy >= 1.0) {
          thrust = th; left = lf; right = rt;
        } else if (blendPolicy <= 0.0) {
          // keep teacher
        } else {
          // blend: if student and teacher agree, keep; otherwise choose by weight
          bool pickStudent(bool a, bool b) =>
              (a == b) ? a : (r.nextDouble() < blendPolicy ? a : b);
          thrust = pickStudent(th, thrust);
          left = pickStudent(lf, left);
          right = pickStudent(rt, right);
          if (left && right) { // impossible; resolve
            if (r.nextBool()) left = false; else right = false;
          }
        }
      }

      final info = env.step(dt, et.ControlInput(
        thrust: thrust, left: left, right: right, intentIdx: currentIntentIdx,
      ));
      totalCost += info.costDelta;
      steps++;
      framesLeft--;

      if (info.terminal) {
        landed = env.status == et.GameStatus.landed;
        break;
      }
      // safety break
      if (steps > 5000) break;
    }

    // Simple intent supervised update during pretrain / train (intentMode path)
    if (train && decisionCaches.isNotEmpty) {
      policy.updateFromEpisode(
        decisionCaches: decisionCaches,
        intentChoices: intentChoices,
        decisionReturns: decisionReturns,
        alignLabels: alignLabels,
        alignWeight: intentAlignWeight,
        lr: lr,
        entropyBeta: 0.0,
        valueBeta: 0.0,
        huberDelta: huberDelta,
        intentMode: true,
        actionAlignWeight: actionAlignWeight,
      );
    }

    return EpisodeResult(steps: steps, totalCost: totalCost, landed: landed);
  }
}
