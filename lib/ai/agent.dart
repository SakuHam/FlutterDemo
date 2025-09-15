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

  void observe(List<double> x) {
    if (x.length != dim) {
      throw ArgumentError('RunningNorm dim mismatch: got ${x.length}, want $dim');
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

  // Look-ahead grows with height
  final hNorm = (h / 320.0).clamp(0.0, 1.6);
  final tau   = (baseTauSec * (0.7 + 0.5 * hNorm)).clamp(minTauSec, maxTauSec);

  // crude projection
  final g  = env.cfg.t.gravity;
  final dxF = dx + vx * tau;
  final vyF = vy + g * tau;

  // Commit bands (wider, with hysteresis)
  final padEnter = 0.08 * W;   // allow hover/descend inside this
  final padExit  = 0.14 * W;   // force lateral outside this

  // Emergency brake if we're about to slam
  if (h < 50 && vyF > 45) return intentToIndex(Intent.brakeUp);

  // Outside exit band → lateral
  if (dxF.abs() > padExit) {
    return dxF > 0 ? intentToIndex(Intent.goRight) : intentToIndex(Intent.goLeft);
  }

  // Near pad: if lateral drift is significant, brake it FIRST (anti ping-pong)
  // If vx would carry us *out* of padEnter, command opposite lateral to damp vx.
  final willExitSoon = (dxF.abs() > padEnter) && (dx.abs() <= padEnter);
  final vxIsBad = (dx.sign == vx.sign) && vx.abs() > 20.0; // drifting outward
  if ((willExitSoon || vxIsBad) && h > 90) {
    return dx >= 0 ? intentToIndex(Intent.goLeft) : intentToIndex(Intent.goRight);
  }

  // Otherwise, controlled descent inside commit band
  if (dxF.abs() <= padEnter) {
    return intentToIndex(Intent.descendSlow);
  }

  // Between enter/exit → gentle lateral nudge toward center
  return dxF > 0 ? intentToIndex(Intent.goRight) : intentToIndex(Intent.goLeft);
}

et.ControlInput controllerForIntent(Intent intent, eng.GameEngine env) {
  final L = env.lander;
  final T = env.terrain;
  final px = L.pos.x.toDouble();
  final gy = T.heightAt(px);
  final h  = (gy - L.pos.y).toDouble();
  final vy = L.vel.y.toDouble();

  switch (intent) {
    case Intent.brakeUp:
      return const et.ControlInput(thrust: true, left: false, right: false);

    case Intent.descendSlow: {
      // Track a gentle vertical target and *don’t* translate if drift is small
      final vTarget = (0.10 * h + 8.0).clamp(8.0, 26.0);
      final need = (vy > vTarget) || (h < 110);
      return et.ControlInput(thrust: need, left: false, right: false);
    }

    case Intent.goLeft: {
      // Translate only at mid heights; near ground do rotation-only nudges
      final translate = (h > 110 && h < 300) && (vy < 35);
      return et.ControlInput(thrust: translate, left: true, right: false);
    }

    case Intent.goRight: {
      final translate = (h > 110 && h < 300) && (vy < 35);
      return et.ControlInput(thrust: translate, left: false, right: true);
    }

    case Intent.hover:
    default: {
      // Light vertical support to avoid sink when hovering
      final vHover = (0.06 * h + 6.0).clamp(6.0, 18.0);
      return et.ControlInput(thrust: vy > vHover, left: false, right: false);
    }
  }
}

/* -------------------------------------------------------------------------- */
/*                               POLICY NETWORK                                */
/* -------------------------------------------------------------------------- */

class ForwardCache {
  final List<double> x;
  final List<double> h1;
  final List<double> h2;
  final List<double> intentLogits;
  final List<double> intentProbs;
  final List<double> turnLogits; // 3 logits
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

  List<List<double>> W1;
  List<double> b1;
  List<List<double>> W2;
  List<double> b2;

  List<List<double>> W_intent;
  List<double> b_intent;

  List<List<double>> W_turn;
  List<double> b_turn;

  List<List<double>> W_thr;
  List<double> b_thr;

  List<List<double>> W_val;
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

  static List<List<double>> _xavier({required int h, required int w, int seed = 0}) {
    final r = math.Random(seed);
    final limit = math.sqrt(6.0 / (h + w));
    return List.generate(h, (_) => List<double>.generate(w, (_) => (r.nextDouble()*2-1)*limit));
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

  static double _tanhScalar(double x) {
    final ax = x.abs();
    if (ax > 20.0) return x.isNegative ? -1.0 : 1.0;
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
    if (x >= 0) {
      final ex = math.exp(-x);
      return 1.0 / (1.0 + ex);
    } else {
      final ex = math.exp(x);
      return ex / (1.0 + ex);
    }
  }

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

  (int, List<double>, ForwardCache) actIntentGreedy(List<double> x) {
    final cache = _forwardFull(x);
    int arg = 0;
    double best = cache.intentProbs[0];
    for (int i = 1; i < cache.intentProbs.length; i++) {
      if (cache.intentProbs[i] > best) { best = cache.intentProbs[i]; arg = i; }
    }
    return (arg, cache.intentProbs, cache);
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
    final probs = <double>[ c.thrProb, ..._softmax(c.turnLogits) ];
    return (thrust, left, right, probs, c);
  }

  void updateFromEpisode({
    required List<ForwardCache> decisionCaches,
    required List<int> intentChoices,
    required List<double> decisionReturns,
    required List<int> alignLabels,
    required double alignWeight,
    required double lr,
    required double entropyBeta,
    required double valueBeta,
    required double huberDelta,
    required bool intentMode,

    List<ForwardCache>? actionCaches,
    List<int>? actionTurnTargets,
    List<bool>? actionThrustTargets,
    double actionAlignWeight = 0.0,
  }) {
    double _clipGrad(double g, [double c = 1.0]) {
      if (!g.isFinite) return 0.0;
      if (g > c) return c;
      if (g < -c) return -c;
      return g;
    }

    // ----- Intent CE (supervised) -----
    if (intentMode && alignWeight > 0 && decisionCaches.isNotEmpty) {
      final N = decisionCaches.length;

      final gW2 = List.generate(h2, (_) => List<double>.filled(h1, 0.0));
      final gb2 = List<double>.filled(h2, 0.0);
      final gW1 = List.generate(h1, (_) => List<double>.filled(inputSize, 0.0));
      final gb1 = List<double>.filled(h1, 0.0);

      final gW_int = List.generate(kIntents, (_) => List<double>.filled(h2, 0.0));
      final gb_int = List<double>.filled(kIntents, 0.0);

      for (int n = 0; n < N; n++) {
        final c = decisionCaches[n];
        final y = alignLabels[n].clamp(0, kIntents - 1);
        final dLog = List<double>.from(c.intentProbs);
        dLog[y] -= 1.0; // p - y

        for (int i = 0; i < kIntents; i++) {
          gb_int[i] += dLog[i];
          for (int j = 0; j < h2; j++) {
            gW_int[i][j] += dLog[i] * c.h2[j];
          }
        }

        final dh2 = List<double>.filled(h2, 0.0);
        for (int j = 0; j < h2; j++) {
          double s = 0.0;
          for (int i = 0; i < kIntents; i++) s += W_intent[i][j] * dLog[i];
          final sech2 = 1.0 - c.h2[j] * c.h2[j];
          dh2[j] = s * sech2;
        }

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

        for (int k = 0; k < h1; k++) {
          gb1[k] += dh1[k];
          for (int t = 0; t < inputSize; t++) {
            gW1[k][t] += dh1[k] * c.x[t];
          }
        }
      }

      final scale = lr * alignWeight / N;
      for (int i = 0; i < kIntents; i++) {
        b_intent[i] -= _clipGrad(scale * gb_int[i]);
        for (int j = 0; j < h2; j++) {
          W_intent[i][j] -= _clipGrad(scale * gW_int[i][j]);
        }
      }
      for (int j = 0; j < h2; j++) {
        b2[j] -= _clipGrad(scale * gb2[j]);
        for (int k = 0; k < h1; k++) {
          W2[j][k] -= _clipGrad(scale * gW2[j][k]);
        }
      }
      for (int k = 0; k < h1; k++) {
        b1[k] -= _clipGrad(scale * gb1[k]);
        for (int t = 0; t < inputSize; t++) {
          W1[k][t] -= _clipGrad(scale * gW1[k][t]);
        }
      }
    }

    // ----- Action supervision (turn CE + thrust BCE) -----
    final hasAction = actionAlignWeight > 0.0 &&
        actionCaches != null &&
        actionTurnTargets != null &&
        actionThrustTargets != null &&
        actionCaches.isNotEmpty;

    if (hasAction) {
      final M = actionCaches!.length;

      final gW2 = List.generate(h2, (_) => List<double>.filled(h1, 0.0));
      final gb2 = List<double>.filled(h2, 0.0);
      final gW1 = List.generate(h1, (_) => List<double>.filled(inputSize, 0.0));
      final gb1 = List<double>.filled(h1, 0.0);

      final gW_turn = List.generate(3, (_) => List<double>.filled(h2, 0.0));
      final gb_turn = List<double>.filled(3, 0.0);

      final gW_thr = List.generate(1, (_) => List<double>.filled(h2, 0.0));
      final gb_thr = List<double>.filled(1, 0.0);

      double meanThrLogit = 0.0;
      double teacherThrRate = 0.0;

      for (int n = 0; n < M; n++) {
        final c = actionCaches![n];

        // turn CE
        final turnProbs = _softmax(c.turnLogits);
        final yt = actionTurnTargets![n].clamp(0, 2);
        final dTurn = List<double>.from(turnProbs);
        dTurn[yt] -= 1.0;

        for (int i = 0; i < 3; i++) {
          gb_turn[i] += dTurn[i];
          for (int j = 0; j < h2; j++) {
            gW_turn[i][j] += dTurn[i] * c.h2[j];
          }
        }

        // thrust BCE
        final thrProb = c.thrProb;
        final yb = actionThrustTargets![n] ? 1.0 : 0.0;
        teacherThrRate += yb;
        final dThr = (thrProb - yb);
        gb_thr[0] += dThr;
        for (int j = 0; j < h2; j++) {
          gW_thr[0][j] += dThr * c.h2[j];
        }
        meanThrLogit += c.thrLogit;

        // backprop to shared trunk
        final dh2 = List<double>.filled(h2, 0.0);
        for (int j = 0; j < h2; j++) {
          double s = 0.0;
          for (int i = 0; i < 3; i++) s += W_turn[i][j] * dTurn[i];
          s += W_thr[0][j] * dThr;
          final sech2 = 1.0 - c.h2[j] * c.h2[j];
          dh2[j] = s * sech2;
        }
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
        for (int k = 0; k < h1; k++) {
          gb1[k] += dh1[k];
          for (int t = 0; t < inputSize; t++) {
            gW1[k][t] += dh1[k] * c.x[t];
          }
        }
      }

      final scale = lr * actionAlignWeight / M;
      for (int i = 0; i < 3; i++) {
        b_turn[i] -= _clipGrad(scale * gb_turn[i]);
        for (int j = 0; j < h2; j++) {
          W_turn[i][j] -= _clipGrad(scale * gW_turn[i][j]);
        }
      }
      b_thr[0] -= _clipGrad(scale * gb_thr[0]);
      for (int j = 0; j < h2; j++) {
        W_thr[0][j] -= _clipGrad(scale * gW_thr[0][j]);
      }
      for (int j = 0; j < h2; j++) {
        b2[j] -= _clipGrad(scale * gb2[j]);
        for (int k = 0; k < h1; k++) {
          W2[j][k] -= _clipGrad(scale * gW2[j][k]);
        }
      }
      for (int k = 0; k < h1; k++) {
        b1[k] -= _clipGrad(scale * gb1[k]);
        for (int t = 0; t < inputSize; t++) {
          W1[k][t] -= _clipGrad(scale * gW1[k][t]);
        }
      }

      // quick bias calibration for thrust fire-rate
      meanThrLogit /= M;
      teacherThrRate /= M;
      final eps = 1e-6;
      final logitTarget = math.log((teacherThrRate + eps) / (1 - teacherThrRate + eps));
      final calibStep = 0.25;
      b_thr[0] += (logitTarget - meanThrLogit) * calibStep;
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
  final double segMean; // ← mean per-frame segment score (raw; higher is better)
  EpisodeResult({
    required this.steps,
    required this.totalCost,
    required this.landed,
    required this.segMean,
  });
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
  final double blendPolicy; // probability-space blend for thrust
  final double intentAlignWeight;
  final double actionAlignWeight;
  final bool normalizeFeatures;

  // NEW: treat segment score as a cost (lower is better) when computing returns
  final bool segmentAsCost;

  final RunningNorm? norm;
  int _epCounter = 0;

  // Segment telemetry
  double _segEma = 0.0;
  int _segDecisions = 0;
  int _segPrintEvery = 400;

  double _prevAbsDx = double.nan;
  int _segNearGroundDec = 0;
  int _segPadZoneDec = 0;
  double _segMeanOverspeed = 0.0;
  int _segThrustNearGroundOn = 0;
  int _segThrustNearGroundTotal = 0;

  // PWM thrust state
  double _pwmA = 0.0;     // accumulator
  int _pwmCount = 0;
  int _pwmOn = 0;

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
    bool segmentAsCost = true, // ← optional param; defaults to "use segment as cost"
  })  : segmentAsCost = segmentAsCost,
        norm = RunningNorm(fe.inputSize, momentum: 0.995);

  void _segTelemetryReset() {
    _segEma = 0.0;
    _segDecisions = 0;
    _segNearGroundDec = 0;
    _segPadZoneDec = 0;
    _segMeanOverspeed = 0.0;
    _segThrustNearGroundOn = 0;
    _segThrustNearGroundTotal = 0;
  }

  void _segTelemetryTick(eng.GameEngine env, double segScore, bool thrustOn, {bool verbose=false}) {
    final L = env.lander;
    final padCx = env.terrain.padCenter.toDouble();
    final gy = env.terrain.heightAt(L.pos.x);
    final h = (gy - L.pos.y).toDouble();
    final W = env.cfg.worldW.toDouble();
    final dx = (L.pos.x - padCx).abs();
    final vy = L.vel.y.toDouble();

    final vyTarget = (0.10 * h + 8.0).clamp(8.0, 28.0);
    final over = vy - vyTarget;
    final overspeedPos = over > 0 ? over : 0.0;

    _segEma = (_segDecisions == 0) ? segScore : (0.99 * _segEma + 0.01 * segScore);
    _segMeanOverspeed = (_segDecisions == 0) ? overspeedPos : (0.99 * _segMeanOverspeed + 0.01 * overspeedPos);
    _segDecisions++;

    final nearGround = h < 200.0;
    final padZone = (dx <= 0.05 * W) && (h < 140.0);
    if (nearGround) _segNearGroundDec++;
    if (padZone) _segPadZoneDec++;

    if (nearGround) {
      _segThrustNearGroundTotal++;
      if (thrustOn) _segThrustNearGroundOn++;
    }

    if (verbose && (_segDecisions % _segPrintEvery == 0)) {
      final padPct = (_segPadZoneDec == 0) ? 0.0 : (100.0 * _segPadZoneDec / _segDecisions);
      final ngThr = (_segThrustNearGroundTotal == 0) ? 0.0
          : (100.0 * _segThrustNearGroundOn / _segThrustNearGroundTotal);
      /*
      print(
          'SEG ema=${_segEma.toStringAsFixed(3)} | meanOverspeed=${_segMeanOverspeed.toStringAsFixed(2)} '
              '| padZone%=${padPct.toStringAsFixed(1)} | thrustNG%=${ngThr.toStringAsFixed(1)} '
              '| h=${h.toStringAsFixed(1)} vy=${vy.toStringAsFixed(1)} dx=${dx.toStringAsFixed(1)}'
      );

       */
    }
  }

  void _segTelemetryFlush({String tag='SEG'}) {
    final padPct = (_segDecisions == 0) ? 0.0 : (100.0 * _segPadZoneDec / _segDecisions);
    final ngThr = (_segThrustNearGroundTotal == 0) ? 0.0
        : (100.0 * _segThrustNearGroundOn / _segThrustNearGroundTotal);
    print('$tag summary: decisions=$_segDecisions | ema=${_segEma.toStringAsFixed(3)} '
        '| meanOverspeed=${_segMeanOverspeed.toStringAsFixed(2)} '
        '| padZone%=${padPct.toStringAsFixed(1)} | thrustNearGround%=${ngThr.toStringAsFixed(1)}');
    _segTelemetryReset();
  }

  double _segmentScore(eng.GameEngine env) {
    final L = env.lander;
    final W = env.cfg.worldW.toDouble();
    final padCx = env.terrain.padCenter.toDouble();

    final gy = env.terrain.heightAt(L.pos.x);
    final h  = (gy - L.pos.y).toDouble();
    final dx = (L.pos.x - padCx).abs();
    final vx = L.vel.x.toDouble();
    final vy = L.vel.y.toDouble();

    final dxNorm = (dx / (0.5 * W)).clamp(0.0, 1.0);
    final rCenterGlobal = 1.0 - dxNorm;

    // Commit bands aligned with teacher:
    final tight = 0.08 * W;   // enter
    final soft  = 0.14 * W;   // exit
    double rCenterBand;
    if (dx <= tight) rCenterBand = 1.0;
    else if (dx <= soft) {
      final t = (dx - tight) / (soft - tight);
      rCenterBand = math.max(0.0, 1.0 - t * t);
    } else rCenterBand = 0.0;

    // Progress toward center
    double rProgress = 0.0;
    if (_prevAbsDx.isFinite) {
      final d = (_prevAbsDx - dx).clamp(-30.0, 30.0);
      rProgress = d / 30.0;
    }
    _prevAbsDx = dx;

    // Descent shaping
    final vyTarget = (0.10 * h + 8.0).clamp(8.0, 28.0);
    final err = vy - vyTarget;
    final sigmaUnder = 7.0, sigmaOver = 5.0;
    final eUnder = math.exp(-math.pow((math.min(0.0, err))/sigmaUnder, 2));
    final eOver  = math.exp(-math.pow((math.max(0.0, err))/sigmaOver , 2));
    final rDescent = 0.5 * eUnder + 0.5 * eOver;

    // NEW: inside tight band, prefer small |vx|
    final vxPenalty = (dx <= tight) ? (- (vx.abs().clamp(0.0, 60.0) / 60.0)) : 0.0;

    // Directional nudge (vx should reduce dx)
    final inward = (dx > 4.0) ? (((padCx - L.pos.x) * vx) > 0 ? 1.0 : -1.0) : 0.0;

    final ang = L.angle.toDouble();                 // radians

    double rLevel = 0.0;
    if (h < 160.0) {
      final angAbs = ang.abs().clamp(0.0, 0.35);    // up to ~20°
      // Penalize tilt near ground (scaled to ~[-2,0])
      rLevel = - 2.0 * (angAbs / 0.35);
      // bonus when *very* level
      if (angAbs < 0.05) rLevel += 0.25;            // small cherry on top
    }

// ...when you compose the final score:
    double score =
        5.0 * rCenterGlobal +
            4.0 * rDescent +
            3.0 * rCenterBand +
            2.0 * rProgress +
            0.6 * inward +
            1.5 * vxPenalty +
            1.2 * rLevel;                // << add this

    if (dx > soft) score -= 0.4;  // lingering far from pad
    if (h < 120.0 && vy > 38.0) score -= 2.5;

    return score;
  }

  int _sampleCategorical(List<double> probs, math.Random r, double temp) {
    if (temp <= 1e-6) {
      int arg = 0; double best = probs[0];
      for (int i = 1; i < probs.length; i++) if (probs[i] > best) { best = probs[i]; arg = i; }
      return arg;
    }
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

    final actionCaches = <ForwardCache>[];
    final actionTurnTargets = <int>[];
    final actionThrustTargets = <bool>[];

    final decisionCaches = <ForwardCache>[];
    final intentChoices = <int>[];
    final decisionReturns = <double>[];
    final alignLabels = <int>[];

    env.reset(seed: r.nextInt(1 << 30));
    double totalCost = 0.0;

    // per-episode accumulators for segment mean
    double segSum = 0.0;
    int segCount = 0;

    // reset per-episode scratch
    _pwmA = 0.0; _pwmCount = 0; _pwmOn = 0;
    _prevAbsDx = double.nan;

    int framesLeft = 0;
    int currentIntentIdx = 0;

    int steps = 0;
    bool landed = false;

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
          decisionCaches.add(cache);
          intentChoices.add(idx);
          alignLabels.add(yTeacher);
          // segment score ONLY at decision states; flip sign if treating as cost
          final segHere = _segmentScore(env);
          final segRL = segmentAsCost ? -segHere : segHere;
          decisionReturns.add(segRL);
        }

        // --- adaptive plan hold ---
        final padCx = env.terrain.padCenter.toDouble();
        final dxAbs = (env.lander.pos.x.toDouble() - padCx).abs();
        final vxAbs = env.lander.vel.x.toDouble().abs();
        final gy = env.terrain.heightAt(env.lander.pos.x.toDouble());
        final h = (gy - env.lander.pos.y).toDouble();
        final W = env.cfg.worldW.toDouble();

        // --- adaptive plan hold (be aggressive about updating intent) ---
        int dynHold = 1; // default: re-decide every frame

        // steer often when far or moving sideways fast
        if (dxAbs > 0.12 * W || vxAbs > 60.0) dynHold = 1;

        // allow 2 frames only if high and roughly centered
        if (dynHold == 1 && h > 320.0 && dxAbs < 0.04 * W && vxAbs < 25.0) dynHold = 2;

        framesLeft = dynHold;
      }

      // teacher control for chosen intent
      final intent = indexToIntent(currentIntentIdx);
      final uTeacher = controllerForIntent(intent, env);

      // student action (policy heads)
      var xAct = fe.extract(env);
      xAct = norm?.normalize(xAct, update: false) ?? xAct;
      final (thBool, lf, rt, probs, cAct) = policy.actGreedy(xAct);

// --- compute descent target and error
      final groundY = env.terrain.heightAt(env.lander.pos.x);
      final height  = (groundY - env.lander.pos.y).toDouble();
      double vyTarget = (0.10 * height + 8.0).clamp(8.0, 28.0); // same as segment score
      final vyNow = env.lander.vel.y.toDouble();                // + downward
      final errVy = vyNow - vyTarget;                           // >0 = too fast down

// model/teacher blend in probability space
      final pThrModel   = probs[0].clamp(0.0, 1.0);
      final pThrTeacher = uTeacher.thrust ? 1.0 : 0.0;
// Blend in probability-space
      final pThrExec = blendPolicy * pThrModel + (1.0 - blendPolicy) * pThrTeacher;

// Accumulate (bounded) and print AFTER increment
      _pwmA = (_pwmA + pThrExec).clamp(0.0, 10.0);

      /*
      if ((_pwmCount % 120) == 0) {
        print('pThr_teacher=${(100*pThrTeacher).toStringAsFixed(1)}%  '
            'pThr_exec=${(100*pThrExec).toStringAsFixed(1)}%  A=${_pwmA.toStringAsFixed(2)}');
      }

       */

// Emit pulse(s) if we’ve accumulated ≥ 1
      bool thrustPWM = false;
      while (_pwmA >= 1.0) {
        thrustPWM = true;
        _pwmA -= 1.0;
      }

// Near-ground flare bias (optional, tiny)
//      final groundY = env.terrain.heightAt(env.lander.pos.x);
//      final height = (groundY - env.lander.pos.y).toDouble();
      if (height < 90.0 && !thrustPWM && pThrExec > 0.65) {
        thrustPWM = true;
        _pwmA = (_pwmA - 0.65).clamp(0.0, 0.999);
      }

      final execThrust = thrustPWM;
      final execLeft   = useLearnedController ? lf : uTeacher.left;
      final execRight  = useLearnedController ? rt : uTeacher.right;

      // segment telemetry (frame-level) + accumulate mean
      final seg = _segmentScore(env);
      segSum += seg; segCount++;
      _segTelemetryTick(env, seg, execThrust, verbose:true);

      // store action caches for supervision (teacher labels)
      actionCaches.add(cAct);
      actionTurnTargets.add(uTeacher.left ? 0 : (uTeacher.right ? 2 : 1));
      actionThrustTargets.add(uTeacher.thrust);

      // PWM duty debug
      _pwmCount++; if (execThrust) _pwmOn++;
      if ((_pwmCount % 240) == 0) {
        final duty = 100.0 * _pwmOn / _pwmCount;
//        print('PWM duty=${duty.toStringAsFixed(1)}%  pThr_model=${(100*pThrModel).toStringAsFixed(1)}%  h=${height.toStringAsFixed(1)} vy=${env.lander.vel.y.toStringAsFixed(1)}');
        _pwmCount = 0; _pwmOn = 0;
      }

      final info = env.step(dt, et.ControlInput(
        thrust: execThrust, left: execLeft, right: execRight, intentIdx: currentIntentIdx,
      ));
      totalCost += info.costDelta;
      steps++;
      framesLeft--;

      if (info.terminal) {
        landed = env.status == et.GameStatus.landed;
        break;
      }
      if (steps > 5000) break;
    }

    if (train && (decisionCaches.isNotEmpty || actionCaches.isNotEmpty)) {
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
        actionCaches: actionCaches,
        actionTurnTargets: actionTurnTargets,
        actionThrustTargets: actionThrustTargets,
        actionAlignWeight: actionAlignWeight,
      );
    }

    final segMean = (segCount > 0) ? (segSum / segCount) : 0.0;
    return EpisodeResult(steps: steps, totalCost: totalCost, landed: landed, segMean: segMean);
  }
}
