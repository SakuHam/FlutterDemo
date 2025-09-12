// lib/ai/agent.dart
import 'dart:math' as math;

import '../engine/game_engine.dart' as eng;
import '../engine/types.dart' as et;

// If you don't have intent_bus.dart, you can delete these imports & the two publish blocks.
import 'intent_bus.dart';

/* ----------------------------- Feature Extraction ----------------------------- */

class FeatureExtractor {
  final int groundSamples; // number of local ground samples (even or odd)
  final double stridePx;
  FeatureExtractor({this.groundSamples = 3, this.stridePx = 48});

  List<double> extract(eng.GameEngine e) {
    final L = e.lander;
    final T = e.terrain;
    final cfg = e.cfg;

    final px = L.pos.x / cfg.worldW;                 // 0..1
    final py = L.pos.y / cfg.worldH;                 // 0..1
    final vx = (L.vel.x / 200.0).clamp(-2.0, 2.0);   // ~-1..1
    final vy = (L.vel.y / 200.0).clamp(-2.0, 2.0);
    final ang = (L.angle / math.pi).clamp(-1.5, 1.5);
    final fuel = (L.fuel / cfg.t.maxFuel).clamp(0.0, 1.0);

    final padCenter = T.padCenter / cfg.worldW;      // 0..1
    final dxCenter = ((L.pos.x - T.padCenter) / cfg.worldW).clamp(-1.0, 1.0);

    // Distance to ground & slope near feet
    final groundY = T.heightAt(L.pos.x);
    final dGround = ((groundY - L.pos.y) / cfg.worldH).clamp(-1.0, 1.0);
    final gyL = T.heightAt(math.max(0, L.pos.x - 20));
    final gyR = T.heightAt(math.min(cfg.worldW, L.pos.x + 20));
    final slope = (((gyR - gyL) / 40.0) / 0.5).clamp(-2.0, 2.0);

    // Local ground samples (exactly groundSamples, even/odd supported)
    final n = groundSamples;
    final samples = <double>[];
    final center = (n - 1) / 2.0; // symmetric offsets for even/odd n
    for (int k = 0; k < n; k++) {
      final relIndex = k - center; // e.g., n=4 -> [-1.5, -0.5, 0.5, 1.5]
      final sx = (L.pos.x + relIndex * stridePx).clamp(0.0, cfg.worldW);
      final sy = T.heightAt(sx);
      final rel = ((sy - L.pos.y) / cfg.worldH).clamp(-1.0, 1.0);
      samples.add(rel);
    }

    return [
      px, py, vx, vy, ang, fuel,
      padCenter, dxCenter,
      dGround, slope,
      ...samples,
    ];
  }

  int get inputSize => 10 + groundSamples;
}

/* ----------------------------- Tiny Linear Algebra ---------------------------- */

List<double> vecAdd(List<double> a, List<double> b) {
  final n = a.length; final out = List<double>.filled(n, 0);
  for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
  return out;
}

List<double> matVec(List<List<double>> W, List<double> x) {
  final m = W.length, n = W[0].length;
  final out = List<double>.filled(m, 0);
  for (int i = 0; i < m; i++) {
    double s = 0.0; final Wi = W[i];
    for (int j = 0; j < n; j++) s += Wi[j] * x[j];
    out[i] = s;
  }
  return out;
}

List<List<double>> zeros(int m, int n) =>
    List.generate(m, (_) => List<double>.filled(n, 0));

List<List<double>> outer(List<double> a, List<double> b) {
  final m = a.length, n = b.length;
  final out = zeros(m, n);
  for (int i = 0; i < m; i++) {
    final ai = a[i];
    for (int j = 0; j < n; j++) out[i][j] = ai * b[j];
  }
  return out;
}

void addInPlaceVec(List<double> a, List<double> b) {
  for (int i = 0; i < a.length; i++) a[i] += b[i];
}
void addInPlaceMat(List<List<double>> A, List<List<double>> B) {
  for (int i = 0; i < A.length; i++) {
    final Ai = A[i], Bi = B[i];
    for (int j = 0; j < Ai.length; j++) Ai[j] += Bi[j];
  }
}

/* -------------------------------- Nonlinearities ------------------------------ */

double relu(double x) => x > 0 ? x : 0.0;
double dRelu(double x) => x > 0 ? 1.0 : 0.0;
double sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));

List<double> reluVec(List<double> v) => v.map(relu).toList();

List<double> softmax(List<double> z) {
  final m = z.reduce(math.max);
  final exps = z.map((v) => math.exp(v - m)).toList();
  final s = exps.reduce((a,b)=>a+b);
  return exps.map((e)=> e / s).toList();
}

/* ----------------------- Intents & tiny low-level controller ------------------ */

enum Intent { hoverCenter, goLeft, goRight, descendSlow, brakeUp }

const List<String> kIntentNames = [
  'hover', 'goLeft', 'goRight', 'descendSlow', 'brakeUp'
];

int intentToIndex(Intent it) => it.index;
Intent indexToIntent(int i) => Intent.values[i];

/// Shared heuristic for supervised labels (correct left/right mapping).
int heuristicIntentLabel(eng.GameEngine env) {
  final padCx = env.terrain.padCenter;
  final dx = env.lander.pos.x - padCx;            // + if lander is RIGHT of pad
  final groundY = env.terrain.heightAt(env.lander.pos.x);
  final height = groundY - env.lander.pos.y;      // px above ground
  final vy = env.lander.vel.y;                    // +down

  final padHalfW = (((env.terrain.padX2 - env.terrain.padX1).abs()) * 0.5)
      .clamp(1.0, env.cfg.worldW.toDouble());
  final dxN = (dx.abs() / padHalfW);
  const dxDead = 0.25;  // within 1/4 pad halfwidth ~ treated as "centered"
  final high = height > 0.30 * env.cfg.worldH;

  if (dxN > dxDead) {
    // lander is off-center: if it's to the RIGHT (dx>0), we need to go LEFT, and vice versa.
    return dx > 0
        ? intentToIndex(Intent.goLeft)
        : intentToIndex(Intent.goRight);
  } else {
    if (high) {
      return intentToIndex(Intent.descendSlow);
    } else {
      return (vy > 55.0)
          ? intentToIndex(Intent.brakeUp)
          : intentToIndex(Intent.hoverCenter);
    }
  }
}

/// Predictive labeler: look ahead tau seconds and pick the intent
/// that will be correct *at that time*, not just now.
int predictiveIntentLabel(eng.GameEngine env, {double tauSec = 0.6}) {
  final L = env.lander;
  final padCx = env.terrain.padCenter.toDouble();

  // --- current state ---
  final dx = L.pos.x - padCx;      // + if right of pad
  final vx = L.vel.x;              // px/s
  final vy = L.vel.y;              // +down
  final groundY = env.terrain.heightAt(L.pos.x);
  final height = groundY - L.pos.y;    // px above ground

  // --- rough lookahead (constant-vel for x; clamp vy) ---
  final dxFuture = dx + vx * tauSec;
  final padHalfW = (((env.terrain.padX2 - env.terrain.padX1).abs()) * 0.5)
      .clamp(1.0, env.cfg.worldW.toDouble());
  final dxN  = (dx.abs() / padHalfW);
  final dxfN = (dxFuture.abs() / padHalfW);

  // crude time-to-contact; avoid div by zero
  final vySafe = math.max(10.0, vy); // assume at least slow descent
  final tGround = height / vySafe;   // s till ground if we kept vy (lower bound)

  // Hysteresis bands in pad units
  const double dxOuter = 0.30;  // clearly off-center
  const double dxInner = 0.18;  // treat as "centered"
  const double vxSmall = 18.0;

  // --- vertical logic first near touch ---
  // If we’ll hit ground sooner than our lookahead window and we're fast → brake
  if (tGround < tauSec * 0.7 && vy > 55.0) {
    return intentToIndex(Intent.brakeUp);
  }

  // If high above ground: descend gently unless way off-center in future
  final bool high = height > 0.30 * env.cfg.worldH;

  // --- horizontal logic with lookahead ---
  if (dxfN > dxOuter) {
    // In tau seconds we'll still be off-center → translate toward pad
    return (dxFuture > 0)
        ? intentToIndex(Intent.goLeft)
        : intentToIndex(Intent.goRight);
  }

  // If we'll be centered-ish and lateral motion small → prefer vertical intents
  if (dxfN < dxInner && vx.abs() < vxSmall) {
    if (high) return intentToIndex(Intent.descendSlow);
    return (vy > 55.0)
        ? intentToIndex(Intent.brakeUp)
        : intentToIndex(Intent.hoverCenter);
  }

  // In-between: bias to reducing future error (sign of dxFuture)
  if (dxFuture.abs() > 0.0) {
    return (dxFuture > 0)
        ? intentToIndex(Intent.goLeft)
        : intentToIndex(Intent.goRight);
  }

  // Fallback
  return intentToIndex(Intent.hoverCenter);
}

/// Predictive-but-adaptive: short lookahead when near center,
/// longer when far/high. Also velocity-aware.
int predictiveIntentLabelAdaptive(
    eng.GameEngine env, {
      double baseTauSec = 0.45,  // nominal horizon
      double minTauSec  = 0.20,  // never longer than needed near center
      double maxTauSec  = 0.90,  // still look ahead when high/far
    }) {
  final L = env.lander;
  final padCx = env.terrain.padCenter.toDouble();

  final dx = L.pos.x - padCx;         // + if right of pad
  final vx = L.vel.x;
  final vy = L.vel.y;                 // +down
  final groundY = env.terrain.heightAt(L.pos.x);
  final height = groundY - L.pos.y;

  final padHalfW = (((env.terrain.padX2 - env.terrain.padX1).abs()) * 0.5)
      .clamp(1.0, env.cfg.worldW.toDouble());
  final dxN = (dx.abs() / padHalfW);

  // --- adaptive horizon ---
  // shorter near center and when |vx| already small; longer if high/far
  final vxAbs = vx.abs();
  final high  = height > 0.35 * env.cfg.worldH;

  double tau = baseTauSec;
  // near center ⇒ shrink horizon
  if (dxN < 0.22) tau *= 0.55;
  else if (dxN < 0.30) tau *= 0.75;

  // if lateral speed is already small, don’t look too far
  if (vxAbs < 20.0) tau *= 0.75;

  // if we’re really high or far, allow a bit more lookahead
  if (high || dxN > 0.45) tau *= 1.25;

  tau = tau.clamp(minTauSec, maxTauSec);

  // --- future horizontal error (constant-vx approximation) ---
  final dxFuture = dx + vx * tau;
  final dxfN = (dxFuture.abs() / padHalfW);

  // --- vertical gating close to ground ---
  final vySafe   = math.max(10.0, vy);
  final tGround  = height / vySafe;
  if (tGround < tau * 0.6 && vy > 55.0) {
    return intentToIndex(Intent.brakeUp);
  }

  // thresholds (pad units)
  const double dxOuter = 0.30;
  const double dxInner = 0.18;
  const double vxSmall = 18.0;

  // If future still off-center → translate toward pad
  if (dxfN > dxOuter) {
    return (dxFuture > 0)
        ? intentToIndex(Intent.goLeft)
        : intentToIndex(Intent.goRight);
  }

  // If will be centered-ish and lateral motion small → prefer vertical intents
  if (dxfN < dxInner && vxAbs < vxSmall) {
    if (height > 0.30 * env.cfg.worldH) {
      return intentToIndex(Intent.descendSlow);
    }
    return (vy > 55.0) ? intentToIndex(Intent.brakeUp)
        : intentToIndex(Intent.hoverCenter);
  }

  // Middle band: bias to reducing FUTURE error but don’t over-commit
  // Use current dx sign if |vx| already small to avoid overshoot.
  if (vxAbs < 20.0) {
    return (dx > 0)
        ? intentToIndex(Intent.goLeft)
        : intentToIndex(Intent.goRight);
  }
  return (dxFuture > 0)
      ? intentToIndex(Intent.goLeft)
      : intentToIndex(Intent.goRight);
}

/// Deterministic low-level controller mapping an intent to (thrust/left/right).
et.ControlInput controllerForIntent(Intent intent, eng.GameEngine env) {
  final L = env.lander;
  final padCx = env.terrain.padCenter;
  final dx = L.pos.x - padCx;
  final vx = L.vel.x;
  final vy = L.vel.y;
  final angle = L.angle;

  bool left = false, right = false, thrust = false;

  // Angle PD toward desired angle (lean into correcting dx and vx)
  double targetAngle = 0.0;
  const kAngDx = 0.006;     // how much horizontal error affects target angle
  const kAngVx = 0.012;     // how much vx affects target angle
  const maxTilt = 15 * math.pi / 180;

  switch (intent) {
    case Intent.goLeft:
      targetAngle = (-kAngDx * dx - kAngVx * vx).clamp(-maxTilt, maxTilt);
      break;
    case Intent.goRight:
      targetAngle = (kAngDx * dx + kAngVx * vx).clamp(-maxTilt, maxTilt);
      break;
    case Intent.hoverCenter:
      targetAngle = (-0.5 * kAngDx * dx - 0.5 * kAngVx * vx).clamp(-maxTilt, maxTilt);
      break;
    case Intent.descendSlow:
    case Intent.brakeUp:
      targetAngle = 0.0;
      break;
  }

  const angDead = 3 * math.pi / 180;
  if (angle > targetAngle + angDead) left = true;
  if (angle < targetAngle - angDead) right = true;

  // Vertical PD for thrust target
  double targetVy = 60.0; // px/s downward normally
  if (intent == Intent.descendSlow) targetVy = 30.0;
  if (intent == Intent.brakeUp)     targetVy = -20.0;

  // Stricter near ground
  final groundY = env.terrain.heightAt(L.pos.x);
  final height = groundY - L.pos.y;
  if (height < 120) targetVy = math.min(targetVy, 20.0);
  if (height <  60) targetVy = math.min(targetVy, 10.0);

  final eVy = vy - targetVy;
  thrust = eVy > 0; // burn if falling faster than target

  // Avoid ceiling hover
  if (L.pos.y < env.cfg.ceilingMargin) {
    thrust = false;
  }

  return et.ControlInput(thrust: thrust, left: left, right: right);
}

/* -------------------- Policy Network (trunk + multiple heads) ----------------- */

class PolicyNetwork {
  final int inputSize;
  final int h1;
  final int h2;

  // Trunk
  late List<List<double>> W1, W2;
  late List<double> b1, b2;

  // Action Heads (single-stage)
  late List<List<double>> W_thr;   // (1, h2)
  late List<double> b_thr;         // (1)
  late List<List<double>> W_turn;  // (3, h2)  [none,left,right]
  late List<double> b_turn;        // (3)

  // Intent Head (two-stage)
  static const int kIntents = 5;   // keep in sync with Intent enum
  late List<List<double>> W_intent; // (K, h2)
  late List<double> b_intent;       // (K)

  // Value Head (critic)
  late List<List<double>> W_val;   // (1, h2)
  late List<double> b_val;         // (1)

  final math.Random rnd;

  PolicyNetwork({
    required this.inputSize,
    this.h1 = 64,
    this.h2 = 64,
    int seed = 1234,
  }) : rnd = math.Random(seed) {
    W1 = _init(h1, inputSize);
    W2 = _init(h2, h1);
    b1 = List<double>.filled(h1, 0);
    b2 = List<double>.filled(h2, 0);

    // single-stage heads
    W_thr = _init(1, h2);   b_thr = List<double>.filled(1, 0);
    W_turn = _init(3, h2);  b_turn = List<double>.filled(3, 0);

    // two-stage planner head
    W_intent = _init(kIntents, h2);
    b_intent = List<double>.filled(kIntents, 0);

    // critic
    W_val = _init(1, h2);   b_val = List<double>.filled(1, 0);
  }

  List<List<double>> _init(int rows, int cols) {
    final limit = math.sqrt(6.0 / (rows + cols));
    return List.generate(
      rows,
          (_) => List<double>.generate(cols, (_) => (rnd.nextDouble() * 2 - 1) * limit),
    );
  }
}

/* ----------------------------- Forward cache struct --------------------------- */

class _Forward {
  final List<double> x;
  final List<double> z1, h1;
  final List<double> z2, h2;

  // single-stage heads
  final double thrLogit, thrP;
  final List<double> turnLogits, turnP;

  // two-stage planner
  final List<double> intentLogits, intentP;

  // critic
  final double v;

  _Forward(
      this.x, this.z1, this.h1, this.z2, this.h2,
      this.thrLogit, this.thrP, this.turnLogits, this.turnP,
      this.intentLogits, this.intentP,
      this.v,
      );
}

/* ----------------------------- Policy operations ----------------------------- */

extension PolicyOps on PolicyNetwork {
  _Forward _forward(List<double> x) {
    final z1 = vecAdd(matVec(W1, x), b1);
    final h1v = reluVec(z1);
    final z2 = vecAdd(matVec(W2, h1v), b2);
    final h2v = reluVec(z2);

    // single-stage heads
    final thrLogit = matVec(W_thr, h2v)[0] + b_thr[0];
    final thrP = sigmoid(thrLogit);

    final turnLogits = matVec(W_turn, h2v);
    for (int i = 0; i < 3; i++) turnLogits[i] += b_turn[i];
    final turnP = softmax(turnLogits);

    // two-stage planner head
    final intentLogits = matVec(W_intent, h2v);
    for (int i = 0; i < PolicyNetwork.kIntents; i++) intentLogits[i] += b_intent[i];
    final intentP = softmax(intentLogits);

    // critic
    final v = matVec(W_val, h2v)[0] + b_val[0];

    return _Forward(
      x, z1, h1v, z2, h2v,
      thrLogit, thrP, turnLogits, turnP,
      intentLogits, intentP,
      v,
    );
  }

  int _argmax(List<double> v) {
    var bi = 0;
    var bv = v[0];
    for (int i = 1; i < v.length; i++) {
      if (v[i] > bv) { bv = v[i]; bi = i; }
    }
    return bi;
  }

  /* --------------------------- Single-stage (legacy) -------------------------- */

  (bool thrust, bool left, bool right, List<double> probs, _Forward cache)
  actGreedy(List<double> x) {
    final f = _forward(x);
    final thrust = f.thrP > 0.5;
    final cls = _argmax(f.turnLogits); // [none, left, right]
    final left = cls == 1;
    final right = cls == 2;
    return (thrust, left, right, [f.thrP, ...f.turnP], f);
  }

  (bool thrust, bool left, bool right, List<double> probs, _Forward cache) act(
      List<double> x,
      math.Random rnd, {
        double tempThr = 1.0,
        double tempTurn = 1.0,
        double epsilon = 0.0,
      }) {
    final f = _forward(x);

    final pThr = sigmoid(f.thrLogit / math.max(1e-6, tempThr));
    final scaled = [
      f.turnLogits[0] / math.max(1e-6, tempTurn),
      f.turnLogits[1] / math.max(1e-6, tempTurn),
      f.turnLogits[2] / math.max(1e-6, tempTurn),
    ];
    final pTurn = softmax(scaled);

    bool thrust = rnd.nextDouble() < pThr;
    final r = rnd.nextDouble();
    int cls;
    if (r < pTurn[0]) cls = 0;
    else if (r < pTurn[0] + pTurn[1]) cls = 1;
    else cls = 2;

    bool left = cls == 1;
    bool right = cls == 2;

    if (epsilon > 0.0 && rnd.nextDouble() < epsilon) {
      const combos = <List<bool>>[
        [false, false, false],
        [true,  false, false],
        [false, true,  false],
        [false, false, true ],
        [true,  true,  false],
        [true,  false, true ],
      ];
      final pick = combos[rnd.nextInt(combos.length)];
      thrust = pick[0]; left = pick[1]; right = pick[2];
    }

    return (thrust, left, right, [pThr, ...pTurn], f);
  }

  /* ---------------------------- Two-stage (planner) --------------------------- */

  (int intentIndex, List<double> probs, _Forward cache) actIntentGreedy(List<double> x) {
    final f = _forward(x);
    final idx = _argmax(f.intentLogits);
    return (idx, List<double>.from(f.intentP), f);
  }

  (int intentIndex, List<double> probs, _Forward cache) actIntent(
      List<double> x,
      math.Random rnd, {
        double tempIntent = 1.0,
        double epsilon = 0.0,
      }) {
    final f = _forward(x);
    final scaled = List<double>.generate(
      PolicyNetwork.kIntents,
          (i) => f.intentLogits[i] / math.max(1e-6, tempIntent),
    );
    final p = softmax(scaled);
    int idx;
    final r = rnd.nextDouble();
    double c = 0.0;
    for (idx = 0; idx < PolicyNetwork.kIntents; idx++) { c += p[idx]; if (r <= c) break; }
    if (idx >= PolicyNetwork.kIntents) idx = PolicyNetwork.kIntents - 1;

    if (epsilon > 0.0 && rnd.nextDouble() < epsilon) {
      idx = rnd.nextInt(PolicyNetwork.kIntents);
    }
    return (idx, p, f);
  }

  /* ------------------------ Updates: single & two-stage ----------------------- */

  void _updateSingleStage({
    required List<_Forward> caches,
    required List<List<int>> actions, // [th, left, right]
    required List<double> returns_,
    double lr = 3e-4,
    double l2 = 1e-6,
    double entropyBeta = 0.0,
    double valueBeta = 0.5,
    double huberDelta = 1.0,
  }) {
    final T = caches.length;
    if (T == 0) return;

    // Advantages: A = R - V
    final values = List<double>.generate(T, (t) => caches[t].v);
    final adv = List<double>.generate(T, (t) => returns_[t] - values[t]);
    final mean = adv.reduce((a,b)=>a+b) / T;
    double var0 = 0.0; for (final v in adv) var0 += (v-mean)*(v-mean);
    var0 /= T;
    final std = math.sqrt(var0 + 1e-8);
    for (int i=0;i<T;i++) adv[i] = (adv[i] - mean) / std;

    // Accumulators
    final dW1 = zeros(W1.length, W1[0].length);
    final dW2 = zeros(W2.length, W2[0].length);
    final db1 = List<double>.filled(b1.length, 0);
    final db2 = List<double>.filled(b2.length, 0);

    final dW_thr = zeros(W_thr.length, W_thr[0].length);
    final db_thr = List<double>.filled(b_thr.length, 0);
    final dW_turn = zeros(W_turn.length, W_turn[0].length);
    final db_turn = List<double>.filled(b_turn.length, 0);

    final dW_val = zeros(W_val.length, W_val[0].length);
    final db_val = List<double>.filled(b_val.length, 0);

    for (int t = 0; t < T; t++) {
      final f = caches[t];
      final a = actions[t];
      final A = adv[t];

      // logits grads = (p - a) * A
      final a_thr = a[0].toDouble();
      final a_turn = <double>[1 - a[1] - a[2] + 0.0, a[1] + 0.0, a[2] + 0.0];

      double dz_thr = (f.thrP - a_thr) * A;
      final dz_turn = List<double>.generate(3, (k) => (f.turnP[k] - a_turn[k]) * A);

      // entropy bonus
      if (entropyBeta > 0.0) {
        final p = f.thrP.clamp(1e-6, 1 - 1e-6);
        final dH_dz_thr = math.log((1 - p) / p) * p * (1 - p);
        dz_thr += -entropyBeta * dH_dz_thr;

        final p3 = f.turnP.map((x) => x.clamp(1e-8, 1.0)).toList();
        final g = List<double>.generate(3, (i) => -(math.log(p3[i]) + 1.0));
        double s = 0.0; for (int i = 0; i < 3; i++) s += p3[i] * g[i];
        for (int i = 0; i < 3; i++) {
          final dH_dz_i = p3[i] * g[i] - p3[i] * s; // (diag(p)-pp^T)g
          dz_turn[i] += -entropyBeta * dH_dz_i;
        }
      }

      // value head (Huber on v - R)
      final err = f.v - returns_[t];
      final dLdv = valueBeta * _huberGrad(err, huberDelta);

      // heads
      addInPlaceMat(dW_thr, outer([dz_thr], f.h2)); db_thr[0] += dz_thr;
      addInPlaceMat(dW_turn, outer(dz_turn, f.h2)); addInPlaceVec(db_turn, dz_turn);
      addInPlaceMat(dW_val, outer([dLdv], f.h2)); db_val[0] += dLdv;

      // backprop to trunk
      final dh2 = List<double>.filled(f.h2.length, 0);
      for (int j = 0; j < W_thr[0].length; j++) dh2[j] += W_thr[0][j] * dz_thr;
      for (int i = 0; i < 3; i++) {
        final row = W_turn[i];
        for (int j = 0; j < row.length; j++) dh2[j] += row[j] * dz_turn[i];
      }
      for (int j = 0; j < W_val[0].length; j++) dh2[j] += W_val[0][j] * dLdv;

      final dz2 = List<double>.generate(f.z2.length, (i) => dh2[i] * dRelu(f.z2[i]));
      addInPlaceMat(dW2, outer(dz2, f.h1)); addInPlaceVec(db2, dz2);

      final dh1 = List<double>.filled(f.h1.length, 0);
      for (int i = 0; i < W2.length; i++) {
        for (int j = 0; j < W2[0].length; j++) dh1[j] += W2[i][j] * dz2[i];
      }
      final dz1 = List<double>.generate(f.z1.length, (i) => dh1[i] * dRelu(f.z1[i]));
      addInPlaceMat(dW1, outer(dz1, f.x)); addInPlaceVec(db1, dz1);
    }

    // L2
    void addL2(List<List<double>> dW, List<List<double>> W) {
      for (int i = 0; i < dW.length; i++) {
        for (int j = 0; j < dW[0].length; j++) dW[i][j] += l2 * W[i][j];
      }
    }
    addL2(dW1, W1); addL2(dW2, W2);
    addL2(dW_thr, W_thr); addL2(dW_turn, W_turn);
    addL2(dW_val, W_val);

    // global norm clip
    double sq = 0.0;
    void accumM(List<List<double>> G){ for(final r in G){ for(final v in r) sq += v*v; } }
    void accumB(List<double> g){ for(final v in g) sq += v*v; }
    accumM(dW1); accumM(dW2); accumM(dW_thr); accumM(dW_turn); accumM(dW_val);
    accumB(db1); accumB(db2); accumB(db_thr); accumB(db_turn); accumB(db_val);
    final clip = 5.0;
    final nrm = math.sqrt(sq + 1e-12);
    final sc = nrm > clip ? (clip / nrm) : 1.0;
    if (sc != 1.0) {
      void scaleM(List<List<double>> G){ for(final r in G){ for(int j=0;j<r.length;j++) r[j]*=sc; } }
      void scaleB(List<double> g){ for(int j=0;j<g.length;j++) g[j]*=sc; }
      scaleM(dW1); scaleM(dW2); scaleM(dW_thr); scaleM(dW_turn); scaleM(dW_val);
      scaleB(db1); scaleB(db2); scaleB(db_thr); scaleB(db_turn); scaleB(db_val);
    }

    // SGD
    void sgdW(List<List<double>> W, List<List<double>> dW) {
      for (int i = 0; i < W.length; i++) {
        for (int j = 0; j < W[0].length; j++) W[i][j] -= lr * dW[i][j];
      }
    }
    void sgdB(List<double> b, List<double> db) { for (int i = 0; i < b.length; i++) b[i] -= lr * db[i]; }

    sgdW(W_thr, dW_thr); sgdB(b_thr, db_thr);
    sgdW(W_turn, dW_turn); sgdB(b_turn, db_turn);
    sgdW(W_val, dW_val); sgdB(b_val, db_val);
    sgdW(W2, dW2); sgdB(b2, db2);
    sgdW(W1, dW1); sgdB(b1, db1);
  }

  // Two-stage update (intent categorical + value + optional entropy)
  void _updateIntentStage({
    required List<_Forward> decisionCaches, // caches at decision times
    required List<int> intentChoices,       // indices 0..K-1
    required List<double> decisionReturns,  // rewards-to-go at decision times
    List<int>? alignLabels,                 // OPTIONAL heuristic labels for intent
    double alignWeight = 0.0,               // strength of auxiliary align loss
    double lr = 3e-4,
    double l2 = 1e-6,
    double entropyBeta = 0.0,
    double valueBeta = 0.5,
    double huberDelta = 1.0,
  }) {
    final T = decisionCaches.length;
    if (T == 0) return;
    assert(alignLabels == null || alignLabels!.length == T,
    'alignLabels length must match decisionCaches length when provided.');

    // Advantages: A = R_decision - V(s_decision)
    final values = List<double>.generate(T, (t) => decisionCaches[t].v);
    final adv = List<double>.generate(T, (t) => decisionReturns[t] - values[t]);
    final mean = adv.reduce((a,b)=>a+b) / T;
    double var0 = 0.0; for (final v in adv) var0 += (v-mean)*(v-mean);
    var0 /= T;
    final std = math.sqrt(var0 + 1e-8);
    for (int i=0;i<T;i++) adv[i] = (adv[i] - mean) / std;

    // Accumulators
    final dW1 = zeros(W1.length, W1[0].length);
    final dW2 = zeros(W2.length, W2[0].length);
    final db1 = List<double>.filled(b1.length, 0);
    final db2 = List<double>.filled(b2.length, 0);

    final dW_int = zeros(W_intent.length, W_intent[0].length);
    final db_int = List<double>.filled(b_intent.length, 0);

    final dW_val = zeros(W_val.length, W_val[0].length);
    final db_val = List<double>.filled(b_val.length, 0);

    for (int t = 0; t < T; t++) {
      final f = decisionCaches[t];
      final k = intentChoices[t];
      final A = adv[t];

      // logits grad for categorical intent (PG)
      final target = List<double>.filled(PolicyNetwork.kIntents, 0.0);
      target[k] = 1.0;
      final dz_int = List<double>.generate(
          PolicyNetwork.kIntents, (i) => (f.intentP[i] - target[i]) * A);

      // --- Auxiliary intent-alignment (OPTIONAL, small) ---
      if (alignWeight > 0.0 && alignLabels != null) {
        final yIdx = alignLabels[t];
        if (yIdx >= 0 && yIdx < PolicyNetwork.kIntents) {
          for (int i = 0; i < PolicyNetwork.kIntents; i++) {
            final y = (i == yIdx) ? 1.0 : 0.0;
            dz_int[i] += alignWeight * (f.intentP[i] - y);
          }
        }
      }

      // entropy on intents (encourage exploration)
      if (entropyBeta > 0.0) {
        final p = f.intentP.map((x) => x.clamp(1e-8, 1.0)).toList();
        final g = List<double>.generate(p.length, (i) => -(math.log(p[i]) + 1.0));
        double s = 0.0; for (int i = 0; i < p.length; i++) s += p[i] * g[i];
        for (int i = 0; i < p.length; i++) {
          final dH_dz_i = p[i] * g[i] - p[i] * s;
          dz_int[i] += -entropyBeta * dH_dz_i;
        }
      }

      // value at decision time
      final err = f.v - decisionReturns[t];
      final dLdv = valueBeta * _huberGrad(err, huberDelta);

      // heads
      addInPlaceMat(dW_int, outer(dz_int, f.h2)); addInPlaceVec(db_int, dz_int);
      addInPlaceMat(dW_val, outer([dLdv], f.h2)); db_val[0] += dLdv;

      // backprop to trunk
      final dh2 = List<double>.filled(f.h2.length, 0);
      for (int i = 0; i < W_intent.length; i++) {
        final row = W_intent[i];
        for (int j = 0; j < row.length; j++) dh2[j] += row[j] * dz_int[i];
      }
      for (int j = 0; j < W_val[0].length; j++) dh2[j] += W_val[0][j] * dLdv;

      final dz2 = List<double>.generate(f.z2.length, (i) => dh2[i] * dRelu(f.z2[i]));
      addInPlaceMat(dW2, outer(dz2, f.h1)); addInPlaceVec(db2, dz2);

      final dh1 = List<double>.filled(f.h1.length, 0);
      for (int i = 0; i < W2.length; i++) {
        for (int j = 0; j < W2[0].length; j++) dh1[j] += W2[i][j] * dz2[i];
      }
      final dz1 = List<double>.generate(f.z1.length, (i) => dh1[i] * dRelu(f.z1[i]));
      addInPlaceMat(dW1, outer(dz1, f.x)); addInPlaceVec(db1, dz1);
    }

    // L2
    void addL2(List<List<double>> dW, List<List<double>> W) {
      for (int i = 0; i < dW.length; i++) {
        for (int j = 0; j < dW[0].length; j++) dW[i][j] += l2 * W[i][j];
      }
    }
    addL2(dW1, W1); addL2(dW2, W2);
    addL2(dW_int, W_intent); addL2(dW_val, W_val);

    // clip
    double sq = 0.0;
    void accumM(List<List<double>> G){ for(final r in G){ for(final v in r) sq += v*v; } }
    void accumB(List<double> g){ for(final v in g) sq += v*v; }
    accumM(dW1); accumM(dW2); accumM(dW_int); accumM(dW_val);
    accumB(db1); accumB(db2); accumB(db_int); accumB(db_val);
    final clip = 5.0;
    final nrm = math.sqrt(sq + 1e-12);
    final sc = nrm > clip ? (clip / nrm) : 1.0;
    if (sc != 1.0) {
      void scaleM(List<List<double>> G){ for(final r in G){ for(int j=0;j<r.length;j++) r[j]*=sc; } }
      void scaleB(List<double> g){ for (int j=0;j<g.length;j++) g[j]*=sc; }
      scaleM(dW1); scaleM(dW2); scaleM(dW_int); scaleM(dW_val);
      scaleB(db1); scaleB(db2); scaleB(db_int); scaleB(db_val);
    }

    // SGD
    void sgdW(List<List<double>> W, List<List<double>> dW) {
      for (int i = 0; i < W.length; i++) {
        for (int j = 0; j < W[0].length; j++) W[i][j] -= lr * dW[i][j];
      }
    }
    void sgdB(List<double> b, List<double> db) { for (int i = 0; i < b.length; i++) b[i] -= lr * db[i]; }

    sgdW(W_intent, dW_int); sgdB(b_intent, db_int);
    sgdW(W_val, dW_val);    sgdB(b_val, db_val);
    sgdW(W2, dW2);          sgdB(b2, db2);
    sgdW(W1, dW1);          sgdB(b1, db1);
  }

  // shared helpers
  double _huberGrad(double error, double delta) {
    final ae = error.abs();
    if (ae <= delta) return error;
    return delta * (error.isNegative ? -1.0 : 1.0);
  }

  // Public: choose which update path to use
  void updateFromEpisode({
    // single-stage data
    List<_Forward>? caches,
    List<List<int>>? actions,
    List<double>? returns_,
    // two-stage data
    List<_Forward>? decisionCaches,
    List<int>? intentChoices,
    List<double>? decisionReturns,
    List<int>? alignLabels,           // heuristic intent labels
    double alignWeight = 0.0,         // auxiliary loss strength
    // common hyperparams
    double lr = 3e-4,
    double l2 = 1e-6,
    double entropyBeta = 0.0,
    double valueBeta = 0.5,
    double huberDelta = 1.0,
    // mode
    bool intentMode = true,
  }) {
    if (intentMode) {
      _updateIntentStage(
        decisionCaches: decisionCaches ?? const [],
        intentChoices: intentChoices ?? const [],
        decisionReturns: decisionReturns ?? const [],
        alignLabels: alignLabels,
        alignWeight: alignWeight,
        lr: lr, l2: l2, entropyBeta: entropyBeta,
        valueBeta: valueBeta, huberDelta: huberDelta,
      );
    } else {
      _updateSingleStage(
        caches: caches ?? const [],
        actions: actions ?? const [],
        returns_: returns_ ?? const [],
        lr: lr, l2: l2, entropyBeta: entropyBeta,
        valueBeta: valueBeta, huberDelta: huberDelta,
      );
    }
  }
}

/* --------------------------- Episode runner & trainer ------------------------- */

class EpisodeResult {
  final double totalCost;
  final int steps;
  final bool landed;

  // diagnostics (single-stage)
  final int turnSteps;
  final int leftSteps;
  final int rightSteps;
  final int thrustSteps;
  final double avgThrProb;

  // diagnostics (two-stage)
  final List<int> intentCounts;   // histogram over intents
  final int intentSwitches;

  EpisodeResult(
      this.totalCost,
      this.steps,
      this.landed, {
        this.turnSteps = 0,
        this.leftSteps = 0,
        this.rightSteps = 0,
        this.thrustSteps = 0,
        this.avgThrProb = 0.0,
        this.intentCounts = const [],
        this.intentSwitches = 0,
      });
}

class Trainer {
  final eng.GameEngine env;
  final FeatureExtractor fe;
  final PolicyNetwork policy;
  final math.Random rnd;

  final double dt;     // seconds per step
  final double gamma;  // discount

  // single-stage exploration knobs (train-time)
  double tempThr;
  double tempTurn;
  double epsilon;
  final double entropyBeta;

  // two-stage knobs
  final bool twoStage;                 // if true, use planner/controller
  final int planHold;                  // frames to hold an intent
  final double tempIntent;             // temperature for intent sampling
  final double intentEntropyBeta;      // entropy on intent head

  Trainer({
    required this.env,
    required this.fe,
    required this.policy,
    this.dt = 1/60.0,
    this.gamma = 0.99,
    int seed = 7,
    this.tempThr = 1.0,
    this.tempTurn = 1.0,
    this.epsilon = 0.0,
    this.entropyBeta = 0.0,
    // two-stage defaults (on by default here)
    this.twoStage = true,
    this.planHold = 1,
    this.tempIntent = 1.0,
    this.intentEntropyBeta = 0.0,
  }) : rnd = math.Random(seed);

  bool shouldBrakeNow(eng.GameEngine env, {double margin = 40.0}) {
    final L = env.lander;
    final padX = L.pos.x;
    final groundY = env.terrain.heightAt(padX);
    final height = groundY - L.pos.y;     // px above ground
    final vy = L.vel.y;                   // +down, px/s

    // Effective upward decel capability (px/s^2)
    final aThrust = env.cfg.t.thrustAccel;  // engine upward accel
    final g       = env.cfg.t.gravity;      // downward accel
    final a = (aThrust - g).clamp(1e-6, 1e9);

    // If we're already going up or essentially zero descent, no emergency brake.
    if (vy <= 8.0) return false;

    // Stopping distance with constant decel: d = vy^2 / (2a)
    final stopDist = (vy * vy) / (2.0 * a);

    // Brake if our stopping distance (plus a buffer) exceeds remaining height.
    return (stopDist + margin) >= height;
  }

  void debugMicroOverfitIntent({
    int perClass = 10,
    int steps = 600,
    double lr = 0.02,
    double alignWeight = 10.0,
    int seed = 42,
  }) {
    final rng = math.Random(seed);
    env.reset(seed: 123456);

    final xs = <List<double>>[];
    final ys = <int>[];

    void put(Intent it, {double? x, double? h, double? vx, double? vy}) {
      final padCx = env.terrain.padCenter.toDouble();
      final padHalfW = (((env.terrain.padX2 - env.terrain.padX1).abs()) * 0.5)
          .clamp(8.0, env.cfg.worldW.toDouble());

      // very separated prototypes
      switch (it) {
        case Intent.goLeft:      x ??= (padCx + 0.95 * padHalfW); h ??= 220; vx ??= 80;  vy ??= 40; break;
        case Intent.goRight:     x ??= (padCx - 0.95 * padHalfW); h ??= 220; vx ??= -80; vy ??= 40; break;
        case Intent.descendSlow: x ??= padCx;                     h ??= 0.7 * env.cfg.worldH; vx ??= 0;    vy ??= 35; break;
        case Intent.brakeUp:     x ??= padCx;                     h ??= 60;                    vx ??= 0;    vy ??= 140; break;
        case Intent.hoverCenter: x ??= padCx;                     h ??= 220;                   vx ??= 0;    vy ??= 12; break;
      }

      final gy = env.terrain.heightAt(x);
      env.lander.pos.x = x.clamp(10.0, env.cfg.worldW - 10.0);
      env.lander.pos.y = (gy - h).clamp(0.0, env.cfg.worldH - 10.0);
      env.lander.vel.x = vx;
      env.lander.vel.y = vy;
      env.lander.angle = 0.0;
      env.lander.fuel  = env.cfg.t.maxFuel;

      xs.add(fe.extract(env));
      ys.add(heuristicIntentLabel(env)); // label via the same heuristic
    }

    for (final it in Intent.values) {
      for (int i = 0; i < perClass; i++) put(it);
    }

    // train hard on the tiny set
    for (int t = 0; t < steps; t++) {
      final caches = <_Forward>[];
      final returns = <double>[];
      final intents = <int>[];
      final labels  = <int>[];
      for (int i = 0; i < xs.length; i++) {
        final res = policy.actIntentGreedy(xs[i]);
        caches.add(res.$3);
        returns.add(res.$3.v); // R=V → A=0 (no PG)
        intents.add(ys[i]);
        labels.add(ys[i]);
      }
      policy.updateFromEpisode(
        decisionCaches: caches,
        intentChoices: intents,
        decisionReturns: returns,
        alignLabels: labels,
        alignWeight: alignWeight,
        lr: lr,
        entropyBeta: 0.0,
        valueBeta: 0.0,
        intentMode: true,
      );
    }

    // confusion matrix
    final K = PolicyNetwork.kIntents;
    final conf = List.generate(K, (_) => List<int>.filled(K, 0));
    int ok = 0;
    for (int i = 0; i < xs.length; i++) {
      final pred = policy.actIntentGreedy(xs[i]).$1;
      conf[ys[i]][pred] += 1;
      if (pred == ys[i]) ok++;
    }
    final acc = ok / xs.length;

    print('MICRO-OVERFIT acc=${(acc*100).toStringAsFixed(1)}% '
        '(N=${xs.length}, steps=$steps, lr=$lr, align=$alignWeight)');
    final names = kIntentNames;
    for (int r = 0; r < K; r++) {
      print('${names[r].padRight(12)} | ' +
          List.generate(K, (c) => conf[r][c].toString().padLeft(3)).join(' '));
    }
  }

  /// Pretrain the intent head on synthetic snapshots (balanced, terrain inited).
  /// Returns stats: {'acc': accuracy, 'n': sampleCount, 'acc_class_k': per-class}
  Map<String, double> pretrainIntentOnSnapshots({
    int samples = 6000,
    int epochs = 2,
    double lr = 3e-4,          // recommend 3e-4..5e-4 for stability
    double alignWeight = 2.0,  // supervised CE weight
    int seed = 1337,
  }) {
    final rng = math.Random(seed);

    // 1) deterministic terrain so labels are stable
    env.reset(seed: 123456);

    // ------------ helpers (state + synth) ------------
    void _setState({
      required double x,
      required double height,
      required double vx,
      required double vy,
    }) {
      final gy = env.terrain.heightAt(x);
      final y = (gy - height).clamp(0.0, env.cfg.worldH - 10.0);
      env.lander.pos.x = x.clamp(10.0, env.cfg.worldW - 10.0);
      env.lander.pos.y = y;
      env.lander.vel.x = vx;
      env.lander.vel.y = vy;
      env.lander.angle = 0.0;
      env.lander.fuel  = env.cfg.t.maxFuel;
    }

    void _synthForIntent(Intent it) {
      final padCx = env.terrain.padCenter.toDouble();
      final padHalfW = (((env.terrain.padX2 - env.terrain.padX1).abs()) * 0.5)
          .clamp(12.0, env.cfg.worldW.toDouble());

      double x=padCx, h=200, vx=0, vy=20;
      const triesMax = 24; // strict; no fallback

      for (int tries=0; tries<triesMax; tries++) {
        switch (it) {
          case Intent.goLeft:
            x  = padCx + (0.92 + 0.06 * rng.nextDouble()) * padHalfW;
            h  = 160.0 + 120.0 * rng.nextDouble();
            vx = 60.0 + 40.0 * rng.nextDouble();
            vy = 30.0 + 40.0 * rng.nextDouble();
            break;
          case Intent.goRight:
            x  = padCx - (0.92 + 0.06 * rng.nextDouble()) * padHalfW;
            h  = 160.0 + 120.0 * rng.nextDouble();
            vx = -(60.0 + 40.0 * rng.nextDouble());
            vy = 30.0 + 40.0 * rng.nextDouble();
            break;
          case Intent.descendSlow:
            x  = padCx + (rng.nextDouble()*0.06 - 0.03) * padHalfW;
            h  = 0.65 * env.cfg.worldH + 0.15 * env.cfg.worldH * rng.nextDouble();
            vx = (rng.nextDouble()*30.0) - 15.0;
            vy = 28.0 + 12.0 * rng.nextDouble();
            break;
          case Intent.brakeUp:
            x  = padCx + (rng.nextDouble()*0.06 - 0.03) * padHalfW;
            h  = 40.0 + 40.0 * rng.nextDouble();      // 40–80 px above ground
            vx = (rng.nextDouble()*18.0) - 9.0;
            vy = 140.0 + 30.0 * rng.nextDouble();     // 140–170 px/s
            break;
          case Intent.hoverCenter:
          // clearly not "high", and very gentle descent
            x  = padCx + (rng.nextDouble()*0.04 - 0.02) * padHalfW;
            h  = 0.22 * env.cfg.worldH + 0.02 * env.cfg.worldH * rng.nextDouble(); // ~0.22–0.24 H
            vx = (rng.nextDouble()*14.0) - 7.0;
            vy = 4.0 + 4.0 * rng.nextDouble();        // 4–8 px/s
            break;
        }

        _setState(x: x, height: h, vx: vx, vy: vy);
        final lab = heuristicIntentLabel(env);
        if (lab == intentToIndex(it)) return; // accept only if label agrees
      }
    }

//    int _labelNow() => heuristicIntentLabel(env);
//    int _labelNow() => predictiveIntentLabel(env, tauSec: 0.6);
    int _labelNow() => predictiveIntentLabelAdaptive(env);

    // ------------ training buffers (mini-batch) ------------
    const int B = 32; // mini-batch size (IMPORTANT: not 1)
    final decisionCaches = <_Forward>[];
    final decisionReturns = <double>[];
    final intentChoices  = <int>[];
    final alignLabels    = <int>[];

    void _flush() {
      if (decisionCaches.isEmpty) return;
      policy.updateFromEpisode(
        decisionCaches: decisionCaches,
        intentChoices: intentChoices,
        decisionReturns: decisionReturns, // R = V → A = 0 (no PG term)
        alignLabels: alignLabels,
        alignWeight: alignWeight,         // supervised CE
        lr: lr,
        entropyBeta: 0.0,
        valueBeta: 0.0,                   // no critic in pretrain
        huberDelta: 1.0,
        intentMode: true,
      );
      decisionCaches.clear();
      decisionReturns.clear();
      intentChoices.clear();
      alignLabels.clear();
    }

    // small held-out train subset for a sanity probe
    final trainXs = <List<double>>[];
    final trainYs = <int>[];

    // ------------ snapshot / restore (checkpoint best) ------------
    List<List<List<double>>> _snapWeights() => [
      policy.W1.map((r) => List<double>.from(r)).toList(),
      policy.W2.map((r) => List<double>.from(r)).toList(),
      policy.W_intent.map((r) => List<double>.from(r)).toList(),
    ];
    List<List<double>> _snapBiases() => [
      List<double>.from(policy.b1),
      List<double>.from(policy.b2),
      List<double>.from(policy.b_intent),
    ];
    void _restoreWeights(List<List<List<double>>> Ws, List<List<double>> Bs) {
      policy.W1 = Ws[0].map((r) => List<double>.from(r)).toList();
      policy.W2 = Ws[1].map((r) => List<double>.from(r)).toList();
      policy.W_intent = Ws[2].map((r) => List<double>.from(r)).toList();
      policy.b1 = List<double>.from(Bs[0]);
      policy.b2 = List<double>.from(Bs[1]);
      policy.b_intent = List<double>.from(Bs[2]);
    }
    double _probeFreshAcc(int N) {
      int ok=0, tot=0;
      for (int i=0;i<N;i++) {
        final it = Intent.values[rng.nextInt(Intent.values.length)];
        _synthForIntent(it);
        final y = _labelNow();
        final pred = policy.actIntentGreedy(fe.extract(env)).$1;
        ok += (pred==y) ? 1 : 0; tot++;
      }
      return tot==0 ? 0.0 : ok/tot;
    }
    var bestAcc = 0.0;
    var bestW = _snapWeights();
    var bestB = _snapBiases();

    // ------------ balanced, interleaved training ------------
    final intents = Intent.values;                 // 5 classes
    final perClass = (samples / intents.length).ceil();

    int seen = 0, correctLive = 0;
    for (int e = 0; e < epochs; e++) {
      final baseOrder = intents.toList()..shuffle(rng);

      for (int i = 0; i < perClass; i++) {
        final order = baseOrder.toList()..shuffle(rng);
        for (final it in order) {
          // synth & label
          _synthForIntent(it);
          final x = fe.extract(env);
          final yIdx = _labelNow();

          if (trainXs.length < 2000) { trainXs.add(List<double>.from(x)); trainYs.add(yIdx); }

          // greedy pred BEFORE update (live acc)
          final res = policy.actIntentGreedy(x);
          final cache = res.$3;
          final pred = res.$1;
          seen++; if (pred == yIdx) correctLive++;

          // accumulate batch
          decisionCaches.add(cache);
          intentChoices.add(yIdx);
          alignLabels.add(yIdx);
          decisionReturns.add(cache.v); // makes A=0; CE-only

          if (decisionCaches.length >= B) {
            _flush();

            // checkpoint every ~2000 seen to avoid late collapse
            if (seen % 2000 == 0) {
              env.reset(seed: 123456);
              final acc = _probeFreshAcc(500);
              if (acc > bestAcc) { bestAcc = acc; bestW = _snapWeights(); bestB = _snapBiases(); }
            }
          }

          if (seen % 1000 == 0) {
            final pct = (100.0 * correctLive / seen).toStringAsFixed(1);
            print('pretrain live acc ~ $pct%  (seen=$seen)');
          }
        }
      }
      _flush();
    }

    // restore best checkpoint before final eval
    _restoreWeights(bestW, bestB);

    // ------------ evaluation ------------
    env.reset(seed: 123456);

    // (A) accuracy on a small stored train-subset (sanity only)
    int okTrain = 0;
    for (int i = 0; i < trainXs.length; i++) {
      final pred = policy.actIntentGreedy(trainXs[i]).$1;
      if (pred == trainYs[i]) okTrain++;
    }
    final accTrain = trainXs.isEmpty ? 0.0 : okTrain / trainXs.length;

    // (B) accuracy on a fresh synthetic set (same generator + rejection)
    final K = PolicyNetwork.kIntents;
    final countsFresh  = List<int>.filled(K, 0);
    final correctFresh = List<int>.filled(K, 0);
    final conf = List.generate(K, (_) => List<int>.filled(K, 0));

    int totalFresh = 0, okFresh = 0;
    final evalN = math.max(2000, samples ~/ 2);

    for (int i = 0; i < evalN; i++) {
      final it = intents[rng.nextInt(intents.length)];
      _synthForIntent(it);                // rejection-sampled
      final yIdx = _labelNow();
      final pred = policy.actIntentGreedy(fe.extract(env)).$1;

      countsFresh[yIdx] += 1;
      conf[yIdx][pred] += 1;
      totalFresh += 1;
      if (pred == yIdx) okFresh += 1;
    }

    final accFresh = totalFresh == 0 ? 0.0 : okFresh / totalFresh;

    print('Pretrain eval (train-subset) acc=${(accTrain*100).toStringAsFixed(1)}%  N=${trainXs.length}');
    print('Pretrain eval (fresh)        acc=${(accFresh*100).toStringAsFixed(1)}%  N=$evalN');

    final names = kIntentNames;
    for (int r = 0; r < K; r++) {
      final row = List.generate(K, (c) => conf[r][c].toString().padLeft(3)).join(' ');
      print('${names[r].padRight(12)} | $row   (n=${countsFresh[r]})');
    }

    // package metrics
    final out = <String, double>{ 'acc': accFresh, 'n': evalN.toDouble(), 'acc_train_subset': accTrain };
    for (int k = 0; k < K; k++) out['count_class_$k'] = countsFresh[k].toDouble();
    return out;
  }

  EpisodeResult runEpisode({
    bool train = true,
    double lr = 3e-4,
    bool greedy = false,
    bool scoreIsReward = false,
    double valueBeta = 0.5,
    double huberDelta = 1.0,
  }) {
    if (!twoStage) {
      // ----------------------- Single-stage path (legacy) -----------------------
      final caches = <_Forward>[];
      final actions = <List<int>>[];
      final costs = <double>[];

      int t = 0;
      int turnSteps = 0, leftSteps = 0, rightSteps = 0, thrustSteps = 0;
      double thrProbSum = 0.0;

      while (true) {
        final x = fe.extract(env);

        bool th, lf, rt; _Forward cache; List<double> probs;

        if (greedy) {
          final res = policy.actGreedy(x);
          th = res.$1; lf = res.$2; rt = res.$3; probs = res.$4; cache = res.$5;
        } else {
          final res = policy.act(
            x, rnd, tempThr: tempThr, tempTurn: tempTurn, epsilon: epsilon,
          );
          th = res.$1; lf = res.$2; rt = res.$3; probs = res.$4; cache = res.$5;
        }

        thrProbSum += probs[0];
        if (lf || rt) turnSteps++;
        if (lf) leftSteps++;
        if (rt) rightSteps++;
        if (th) thrustSteps++;

        final info = env.step(dt, et.ControlInput(thrust: th, left: lf, right: rt));

        // engine returns cost; convert to stored COST
        final s = info.scoreDelta;
        final stepCost = scoreIsReward ? -s : s;

        caches.add(cache);
        actions.add([th ? 1 : 0, lf ? 1 : 0, rt ? 1 : 0]);
        costs.add(stepCost);

        if (info.terminal || caches.length > 4000) break;
        t++;
      }

      // returns over reward = -cost
      final R = List<double>.filled(costs.length, 0.0);
      double running = 0.0;
      for (int i = costs.length - 1; i >= 0; i--) {
        final reward = -costs[i];
        running = reward + gamma * running;
        R[i] = running;
      }

      if (train && caches.isNotEmpty) {
        policy.updateFromEpisode(
          caches: caches,
          actions: actions,
          returns_: R,
          lr: lr,
          entropyBeta: entropyBeta,
          valueBeta: valueBeta,
          huberDelta: huberDelta,
          intentMode: false, // single-stage uses false
        );
      }

      final landed = env.status == et.GameStatus.landed;
      final totalCost = costs.fold(0.0, (a, b) => a + b);
      final avgThrProb = costs.isEmpty ? 0.0 : (thrProbSum / costs.length);

      return EpisodeResult(
        totalCost, costs.length, landed,
        turnSteps: turnSteps,
        leftSteps: leftSteps,
        rightSteps: rightSteps,
        thrustSteps: thrustSteps,
        avgThrProb: avgThrProb,
      );
    } else {
      // ----------------------- Two-stage path (planner/controller) --------------
      final decisionCaches = <_Forward>[];
      final intentChoices  = <int>[];
      final decisionTimes  = <int>[];
      final costs          = <double>[];

      // Heuristic intent labels (for auxiliary alignment)
      final heuristicLabels = <int>[];

      final intentCounts = List<int>.filled(PolicyNetwork.kIntents, 0);
      int intentSwitches = 0;

      int t = 0;
      int framesLeft = 0;
      int currentIntentIdx = -1;

      while (true) {
        if (framesLeft <= 0) {
          // (decide intent)
          final xPlan = fe.extract(env);
          int idx; List<double> probs; _Forward cache;

          if (greedy) {
            final res = policy.actIntentGreedy(xPlan);
            idx = res.$1; probs = res.$2; cache = res.$3;
          } else {
            final res = policy.actIntent(xPlan, rnd, tempIntent: tempIntent, epsilon: epsilon);
            idx = res.$1; probs = res.$2; cache = res.$3;
          }

          // ---------------- NEW: physics-based brake override ----------------
          if (shouldBrakeNow(env, margin: 50.0)) {
            idx = intentToIndex(Intent.brakeUp);
            // optional: make probs one-hot for logging/visualization
            probs = List<double>.filled(PolicyNetwork.kIntents, 0.0)
              ..[idx] = 1.0;
          }
          // -------------------------------------------------------------------

          currentIntentIdx = idx;
          decisionCaches.add(cache);
          intentChoices.add(idx);
          decisionTimes.add(t);
          framesLeft = planHold;

          // DIAGNOSTICS
          intentCounts[idx] += 1;
          if (intentChoices.length >= 2 &&
              intentChoices[intentChoices.length - 1] != intentChoices[intentChoices.length - 2]) {
            intentSwitches += 1;
          }

          // ---------- Heuristic intent label (for alignment loss) ----------
          heuristicLabels.add(heuristicIntentLabel(env));

          // ---------- (Optional) Publish intent to UI ----------
          // Remove this block if you don't use intent_bus.dart
          IntentBus.instance.publishIntent(IntentEvent(
            intent: kIntentNames[idx],
            probs: probs,
            step: t,
            meta: {'episode_step': t, 'plan_hold': planHold},
          ));
        }

        final intent = indexToIntent(currentIntentIdx);
        final u = controllerForIntent(intent, env);

        // ---------- (Optional) Publish control to UI ----------
        // Remove this block if you don't use intent_bus.dart
        IntentBus.instance.publishControl(ControlEvent(
          thrust: u.thrust, left: u.left, right: u.right, step: t,
          meta: {'intent': kIntentNames[currentIntentIdx]},
        ));

        final info = env.step(
          dt,
          et.ControlInput(
            thrust: u.thrust,
            left: u.left,
            right: u.right,
            intentIdx: currentIntentIdx,
          ),
        );
        final s = info.scoreDelta;
        costs.add(scoreIsReward ? -s : s);

        framesLeft -= 1;
        t += 1;
        if (info.terminal || t > 4000) break;
      }

      // reward-to-go for each step
      final R = List<double>.filled(costs.length, 0.0);
      double running = 0.0;
      for (int i = costs.length - 1; i >= 0; i--) {
        final reward = -costs[i];
        running = reward + gamma * running;
        R[i] = running;
      }

      // returns at decision times
      final decisionReturns = <double>[];
      for (final ti in decisionTimes) {
        final idx = ti.clamp(0, R.length - 1);
        decisionReturns.add(R[idx]);
      }

      if (train && decisionCaches.isNotEmpty) {
        policy.updateFromEpisode(
          decisionCaches: decisionCaches,
          intentChoices: intentChoices,
          decisionReturns: decisionReturns,
          alignLabels: heuristicLabels,    // supervised assist
          alignWeight: 0.08,               // small but steady guidance
          lr: lr,
          entropyBeta: intentEntropyBeta,
          valueBeta: valueBeta,
          huberDelta: huberDelta,
          intentMode: true,
        );
      }

      final landed = env.status == et.GameStatus.landed;
      final totalCost = costs.fold(0.0, (a, b) => a + b);

      return EpisodeResult(
        totalCost, costs.length, landed,
        intentCounts: intentCounts,
        intentSwitches: intentSwitches,
      );
    }
  }
}
