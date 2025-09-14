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

  // Acceleration finite-difference state (fallback if engine acc not available)
  double? _prevVx;
  double? _prevVy;
  bool _havePrev = false;

  FeatureExtractor({this.groundSamples = 3, this.stridePx = 48});

  /// Call at episode start to clear accel history.
  void reset() {
    _prevVx = null;
    _prevVy = null;
    _havePrev = false;
  }

  List<double> extract(eng.GameEngine e) {
    final L = e.lander;
    final T = e.terrain;
    final cfg = e.cfg;

    final px = L.pos.x / cfg.worldW;                 // 0..1
    final py = L.pos.y / cfg.worldH;                 // 0..1

    final vxRaw = L.vel.x;
    final vyRaw = L.vel.y;                            // +down
    final vx = (vxRaw / 200.0).clamp(-2.0, 2.0);      // ~-1..1
    final vy = (vyRaw / 200.0).clamp(-2.0, 2.0);

    final ang = (L.angle / math.pi).clamp(-1.5, 1.5);
    final fuel = (L.fuel / cfg.t.maxFuel).clamp(0.0, 1.0);

    // --- Acceleration (prefer engine-reported; else finite diff) ---
    // --- Acceleration via finite difference (no engine acc available) ---
    // Conventions: vy is +down, so ay will also be +down.
    final dt = 1.0 / cfg.stepScale;
    final pvx = _prevVx ?? vxRaw;
    final pvy = _prevVy ?? vyRaw;
    final axRaw = (vxRaw - pvx) / (dt > 0 ? dt : 1.0);
    final ayRaw = (vyRaw - pvy) / (dt > 0 ? dt : 1.0);
    _prevVx = vxRaw;
    _prevVy = vyRaw;
    _havePrev = true;

    // Normalize accelerations
    final aXscale = (cfg.t.thrustAccel.abs() > 1e-6) ? cfg.t.thrustAccel : 1.0;
    final aYscale = (cfg.t.thrustAccel + cfg.t.gravity).abs() > 1e-6
        ? (cfg.t.thrustAccel + cfg.t.gravity)
        : 1.0;
    final ax = (axRaw / aXscale).clamp(-2.0, 2.0);
    final ay = (ayRaw / aYscale).clamp(-2.0, 2.0);

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
      // accelerations (normalized)
      ax, ay,
      padCenter, dxCenter,
      dGround, slope,
      ...samples,
    ];
  }

  // Old was 10 + groundSamples. We added ax, ay → +2.
  int get inputSize => 12 + groundSamples;
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

/* ----------------------- Vertical speed envelope helpers ---------------------- */

double _safeDescentVy(double height) {
  if (height > 260) return 60;
  if (height > 160) return 40;
  if (height > 100) return 25;
  if (height >  50) return 14;
  return 8;
}

bool _shouldBrakeNow(eng.GameEngine env, {double margin = 50.0, double reactSec = 0.10}) {
  final L = env.lander;
  final vy = L.vel.y; // +down
  final groundY = env.terrain.heightAt(L.pos.x);
  final height = groundY - L.pos.y;

  if (vy <= 8.0) return false;

  final aThrust = env.cfg.t.thrustAccel; // upward accel
  final g       = env.cfg.t.gravity;     // downward accel
  final a       = (aThrust - g).clamp(1e-6, 1e9);   // effective up decel when level

  final dReact = vy * reactSec;           // extra fall during reaction
  final dStop  = (vy * vy) / (2.0 * a);   // constant-decel stopping distance
  return (dReact + dStop + margin) >= height;
}

/* ----------------------- Intents & low-level controller turns ----------------- */

enum Intent { hoverCenter, goLeft, goRight, descendSlow, brakeUp }

const List<String> kIntentNames = [
  'hover', 'goLeft', 'goRight', 'descendSlow', 'brakeUp'
];

int intentToIndex(Intent it) => it.index;
Intent indexToIntent(int i) => Intent.values[i];

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
    return dx > 0 ? intentToIndex(Intent.goLeft) : intentToIndex(Intent.goRight);
  } else {
    if (high) return intentToIndex(Intent.descendSlow);
    return (vy > 55.0) ? intentToIndex(Intent.brakeUp) : intentToIndex(Intent.hoverCenter);
  }
}

int predictiveIntentLabelAdaptive(
    eng.GameEngine env, {
      double baseTauSec = 1.00,
      double minTauSec  = 0.45,
      double maxTauSec  = 1.35,
    }) {
  final L = env.lander;
  final padCx = env.terrain.padCenter.toDouble();

  final dx = L.pos.x - padCx;         // + if right of pad
  final vx = L.vel.x;                  // px/s
  final vy = L.vel.y;                  // +down
  final groundY = env.terrain.heightAt(L.pos.x);
  final height = groundY - L.pos.y;

  final padHalfW = (((env.terrain.padX2 - env.terrain.padX1).abs()) * 0.5)
      .clamp(1.0, env.cfg.worldW.toDouble());
  final dxN = (dx.abs() / padHalfW);

  double tau = baseTauSec;
  if (dxN < 0.22) tau *= 0.60;
  else if (dxN < 0.30) tau *= 0.80;
  if (vx.abs() < 20.0) tau *= 0.80;
  final bool high = height > 0.35 * env.cfg.worldH;
  if (high || dxN > 0.45) tau *= 1.20;
  tau = tau.clamp(minTauSec, maxTauSec);

  final dxFuture = dx + vx * tau;
  final dxfN = (dxFuture.abs() / padHalfW);

  final vySafe  = math.max(10.0, vy);
  final tGround = height / vySafe;
  if (tGround < tau * 0.70 && vy > 55.0) {
    return intentToIndex(Intent.brakeUp);
  }

  const double dxOuter = 0.30; // off-center in future
  const double dxInner = 0.18; // centered-ish in future
  const double vxSmall = 18.0;

  if (dxfN > dxOuter) {
    return (dxFuture > 0)
        ? intentToIndex(Intent.goLeft)
        : intentToIndex(Intent.goRight);
  }
  if (dxfN < dxInner && vx.abs() < vxSmall) {
    if (height > 0.30 * env.cfg.worldH) {
      return intentToIndex(Intent.descendSlow);
    }
    return (vy > 55.0) ? intentToIndex(Intent.brakeUp)
        : intentToIndex(Intent.hoverCenter);
  }

  if (vx.abs() < 20.0) {
    return (dx > 0) ? intentToIndex(Intent.goLeft)
        : intentToIndex(Intent.goRight);
  }

  if (height > 0.30 * env.cfg.worldH && vy < -12.0) {
    return intentToIndex(Intent.descendSlow);
  }

  return (dxFuture > 0) ? intentToIndex(Intent.goLeft)
      : intentToIndex(Intent.goRight);
}

/* -------------------- Policy Network (trunk + multiple heads) ----------------- */

class PolicyNetwork {
  final int inputSize;
  final int h1;
  final int h2;

  // Trunk
  late List<List<double>> W1, W2;
  late List<double> b1, b2;

  // Legacy action Heads (optional)
  late List<List<double>> W_thr;   // (1, h2)
  late List<double> b_thr;         // (1)
  late List<List<double>> W_turn;  // (3, h2)  [none,left,right]
  late List<double> b_turn;        // (3)

  // Planner Heads
  static const int kIntents = 5;   // keep in sync with Intent enum
  late List<List<double>> W_intent; // (K, h2)
  late List<double> b_intent;       // (K)

  // Planner Thrust head (binary)
  late List<List<double>> W_thrplan; // (1, h2)
  late List<double> b_thrplan;       // (1)

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

    // legacy
    W_thr = _init(1, h2);   b_thr = List<double>.filled(1, 0);
    W_turn = _init(3, h2);  b_turn = List<double>.filled(3, 0);

    // planner
    W_intent = _init(kIntents, h2);  b_intent = List<double>.filled(kIntents, 0);
    W_thrplan = _init(1, h2);        b_thrplan = List<double>.filled(1, 0);

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

  // legacy heads (optional)
  final double thrLogit, thrP;
  final List<double> turnLogits, turnP;

  // planner heads
  final List<double> intentLogits, intentP;
  final double thrPlanLogit, thrPlanP;

  // critic
  final double v;

  _Forward(
      this.x, this.z1, this.h1, this.z2, this.h2,
      this.thrLogit, this.thrP, this.turnLogits, this.turnP,
      this.intentLogits, this.intentP,
      this.thrPlanLogit, this.thrPlanP,
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

    // legacy heads
    final thrLogit = matVec(W_thr, h2v)[0] + b_thr[0];
    final thrP = sigmoid(thrLogit);

    final turnLogits = matVec(W_turn, h2v);
    for (int i = 0; i < 3; i++) turnLogits[i] += b_turn[i];
    final turnP = softmax(turnLogits);

    // planner heads
    final intentLogits = matVec(W_intent, h2v);
    for (int i = 0; i < PolicyNetwork.kIntents; i++) intentLogits[i] += b_intent[i];
    final intentP = softmax(intentLogits);

    final thrPlanLogit = matVec(W_thrplan, h2v)[0] + b_thrplan[0];
    final thrPlanP     = sigmoid(thrPlanLogit);

    // critic
    final v = matVec(W_val, h2v)[0] + b_val[0];

    return _Forward(
      x, z1, h1v, z2, h2v,
      thrLogit, thrP, turnLogits, turnP,
      intentLogits, intentP,
      thrPlanLogit, thrPlanP,
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

  /* ------------------------ Planner: intent + thrust (separate) --------------- */

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

  (bool thrustPlan, double pThrPlan, _Forward cache) actThrustPlanGreedy(List<double> x) {
    final f = _forward(x);
    final thrustPlan = f.thrPlanP > 0.5;
    return (thrustPlan, f.thrPlanP, f);
  }

  (bool thrustPlan, double pThrPlan, _Forward cache) actThrustPlan(
      List<double> x, math.Random rnd, { double tempThrPlan = 1.0 }) {
    final f = _forward(x);
    final pThrPlan = sigmoid(f.thrPlanLogit / math.max(1e-6, tempThrPlan));
    final thrustPlan = rnd.nextDouble() < pThrPlan;
    return (thrustPlan, pThrPlan, f);
  }

  /* ---------------- Supervised update for planner thrust head (per frame) ----- */

  void updateThrustHeadSupervised({
    required List<_Forward> caches,   // per-frame forward caches
    required List<int> labels,        // 0/1 teacher per frame
    double lr = 3e-4,
    double l2 = 1e-6,
    double entropyBeta = 0.0,
    bool backpropTrunk = true,
  }) {
    final T = caches.length;
    if (T == 0) return;

    final dW1 = zeros(W1.length, W1[0].length);
    final dW2 = zeros(W2.length, W2[0].length);
    final db1 = List<double>.filled(b1.length, 0);
    final db2 = List<double>.filled(b2.length, 0);

    final dW_thrplan = zeros(W_thrplan.length, W_thrplan[0].length);
    final db_thrplan = List<double>.filled(b_thrplan.length, 0);

    for (int t = 0; t < T; t++) {
      final f = caches[t];
      final y = labels[t].clamp(0, 1).toDouble();

      // CE grad on logit: (p - y)
      double dz = (f.thrPlanP - y);

      if (entropyBeta > 0.0) {
        final p = f.thrPlanP.clamp(1e-6, 1 - 1e-6);
        final dH_dz = math.log((1 - p) / p) * p * (1 - p);
        dz += -entropyBeta * dH_dz;
      }

      addInPlaceMat(dW_thrplan, outer([dz], f.h2));
      db_thrplan[0] += dz;

      if (backpropTrunk) {
        final dh2 = List<double>.filled(f.h2.length, 0);
        for (int j = 0; j < W_thrplan[0].length; j++) dh2[j] += W_thrplan[0][j] * dz;

        final dz2 = List<double>.generate(f.z2.length, (i) => dh2[i] * dRelu(f.z2[i]));
        addInPlaceMat(dW2, outer(dz2, f.h1)); addInPlaceVec(db2, dz2);

        final dh1 = List<double>.filled(f.h1.length, 0);
        for (int i = 0; i < W2.length; i++) {
          for (int j = 0; j < W2[0].length; j++) dh1[j] += W2[i][j] * dz2[i];
        }
        final dz1 = List<double>.generate(f.z1.length, (i) => dh1[i] * dRelu(f.z1[i]));
        addInPlaceMat(dW1, outer(dz1, f.x)); addInPlaceVec(db1, dz1);
      }
    }

    void addL2(List<List<double>> dW, List<List<double>> W) {
      for (int i = 0; i < dW.length; i++) {
        for (int j = 0; j < dW[0].length; j++) dW[i][j] += l2 * W[i][j];
      }
    }
    addL2(dW_thrplan, W_thrplan);
    if (backpropTrunk) { addL2(dW1, W1); addL2(dW2, W2); }

    void sgdW(List<List<double>> W, List<List<double>> dW) {
      for (int i = 0; i < W.length; i++) {
        for (int j = 0; j < W[0].length; j++) W[i][j] -= lr * dW[i][j];
      }
    }
    void sgdB(List<double> b, List<double> db) { for (int i = 0; i < b.length; i++) b[i] -= lr * db[i]; }

    sgdW(W_thrplan, dW_thrplan); sgdB(b_thrplan, db_thrplan);
    if (backpropTrunk) { sgdW(W2, dW2); sgdB(b2, db2); sgdW(W1, dW1); sgdB(b1, db1); }
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

    final values = List<double>.generate(T, (t) => caches[t].v);
    final adv = List<double>.generate(T, (t) => returns_[t] - values[t]);
    final mean = adv.reduce((a,b)=>a+b) / T;
    double var0 = 0.0; for (final v in adv) var0 += (v-mean)*(v-mean);
    var0 /= T;
    final std = math.sqrt(var0 + 1e-8);
    for (int i=0;i<T;i++) adv[i] = (adv[i] - mean) / std;

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

      final a_thr = a[0].toDouble();
      final a_turn = <double>[1 - a[1] - a[2] + 0.0, a[1] + 0.0, a[2] + 0.0];

      double dz_thr = (f.thrP - a_thr) * A;
      final dz_turn = List<double>.generate(3, (k) => (f.turnP[k] - a_turn[k]) * A);

      if (entropyBeta > 0.0) {
        final p = f.thrP.clamp(1e-6, 1 - 1e-6);
        final dH_dz_thr = math.log((1 - p) / p) * p * (1 - p);
        dz_thr += -entropyBeta * dH_dz_thr;

        final p3 = f.turnP.map((x) => x.clamp(1e-8, 1.0)).toList();
        final g = List<double>.generate(3, (i) => -(math.log(p3[i]) + 1.0));
        double s = 0.0; for (int i = 0; i < 3; i++) s += p3[i] * g[i];
        for (int i = 0; i < 3; i++) {
          final dH_dz_i = p3[i] * g[i] - p3[i] * s;
          dz_turn[i] += -entropyBeta * dH_dz_i;
        }
      }

      final err = f.v - returns_[t];
      final dLdv = valueBeta * _huberGrad(err, huberDelta);

      addInPlaceMat(dW_thr, outer([dz_thr], f.h2)); db_thr[0] += dz_thr;
      addInPlaceMat(dW_turn, outer(dz_turn, f.h2)); addInPlaceVec(db_turn, dz_turn);
      addInPlaceMat(dW_val, outer([dLdv], f.h2)); db_val[0] += dLdv;

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

    void addL2(List<List<double>> dW, List<List<double>> W) {
      for (int i = 0; i < dW.length; i++) {
        for (int j = 0; j < dW[0].length; j++) dW[i][j] += l2 * W[i][j];
      }
    }
    addL2(dW1, W1); addL2(dW2, W2);
    addL2(dW_thr, W_thr); addL2(dW_turn, W_turn);
    addL2(dW_val, W_val);

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

  void _updateIntentStage({
    required List<_Forward> decisionCaches, // caches at decision times
    required List<int> intentChoices,       // indices 0..K-1
    required List<double> decisionReturns,  // rewards-to-go at decision times

    List<int>? alignLabels,                 // heuristic labels for intent
    double alignWeight = 0.0,

    List<List<int>>? actionLabels,          // [thr,left,right]
    double actionAlignWeight = 0.0,

    List<int>? thrPlanLabels,               // per-decision thrust teacher
    double thrAlignWeight = 0.0,

    double lr = 3e-4,
    double l2 = 1e-6,
    double entropyBeta = 0.0,
    double valueBeta = 0.5,
    double huberDelta = 1.0,
  }) {
    final T = decisionCaches.length;
    if (T == 0) return;

    final values = List<double>.generate(T, (t) => decisionCaches[t].v);
    final adv = List<double>.generate(T, (t) => decisionReturns[t] - values[t]);
    final mean = adv.reduce((a,b)=>a+b) / T;
    double var0 = 0.0; for (final v in adv) var0 += (v-mean)*(v-mean);
    var0 /= T;
    final std = math.sqrt(var0 + 1e-8);
    for (int i=0;i<T;i++) adv[i] = (adv[i] - mean) / std;

    final dW1 = zeros(W1.length, W1[0].length);
    final dW2 = zeros(W2.length, W2[0].length);
    final db1 = List<double>.filled(b1.length, 0);
    final db2 = List<double>.filled(b2.length, 0);

    final dW_int = zeros(W_intent.length, W_intent[0].length);
    final db_int = List<double>.filled(b_intent.length, 0);

    final dW_thr = zeros(W_thr.length, W_thr[0].length);
    final db_thr = List<double>.filled(b_thr.length, 0);
    final dW_turn = zeros(W_turn.length, W_turn[0].length);
    final db_turn = List<double>.filled(b_turn.length, 0);

    final dW_thrplan = zeros(W_thrplan.length, W_thrplan[0].length);
    final db_thrplan = List<double>.filled(b_thrplan.length, 0);

    final dW_val = zeros(W_val.length, W_val[0].length);
    final db_val = List<double>.filled(b_val.length, 0);

    for (int t = 0; t < T; t++) {
      final f = decisionCaches[t];
      final k = intentChoices[t];
      final A = adv[t];

      final target = List<double>.filled(PolicyNetwork.kIntents, 0.0);
      target[k] = 1.0;
      final dz_int = List<double>.generate(
          PolicyNetwork.kIntents, (i) => (f.intentP[i] - target[i]) * A);

      if (alignWeight > 0.0 && alignLabels != null) {
        final yIdx = alignLabels[t];
        if (yIdx >= 0 && yIdx < PolicyNetwork.kIntents) {
          for (int i = 0; i < PolicyNetwork.kIntents; i++) {
            final y = (i == yIdx) ? 1.0 : 0.0;
            dz_int[i] += alignWeight * (f.intentP[i] - y);
          }
        }
      }

      if (entropyBeta > 0.0) {
        final p = f.intentP.map((x) => x.clamp(1e-8, 1.0)).toList();
        final g = List<double>.generate(p.length, (i) => -(math.log(p[i]) + 1.0));
        double s = 0.0; for (int i = 0; i < p.length; i++) s += p[i] * g[i];
        for (int i = 0; i < p.length; i++) {
          final dH_dz_i = p[i] * g[i] - p[i] * s;
          dz_int[i] += -entropyBeta * dH_dz_i;
        }
      }

      final err = f.v - decisionReturns[t];
      final dLdv = valueBeta * _huberGrad(err, huberDelta);

      addInPlaceMat(dW_int, outer(dz_int, f.h2)); addInPlaceVec(db_int, dz_int);
      addInPlaceMat(dW_val, outer([dLdv], f.h2)); db_val[0] += dLdv;

      final dh2 = List<double>.filled(f.h2.length, 0);
      for (int i = 0; i < W_intent.length; i++) {
        final row = W_intent[i];
        for (int j = 0; j < row.length; j++) dh2[j] += row[j] * dz_int[i];
      }
      for (int j = 0; j < W_val[0].length; j++) dh2[j] += W_val[0][j] * dLdv;

      // Optional: legacy action distill at decision states
      if (actionAlignWeight > 0.0 && actionLabels != null) {
        final lab = actionLabels[t]; // [thr,left,right]
        final yThr = lab[0].toDouble();
        final yTurn = <double>[1 - lab[1] - lab[2] + 0.0, lab[1] + 0.0, lab[2] + 0.0];

        final dz_thr_sup = (f.thrP - yThr) * actionAlignWeight;
        final dz_turn_sup = List<double>.generate(3, (i) => (f.turnP[i] - yTurn[i]) * actionAlignWeight);

        addInPlaceMat(dW_thr, outer([dz_thr_sup], f.h2)); db_thr[0] += dz_thr_sup;
        addInPlaceMat(dW_turn, outer(dz_turn_sup, f.h2)); addInPlaceVec(db_turn, dz_turn_sup);

        for (int j = 0; j < W_thr[0].length; j++) dh2[j] += W_thr[0][j] * dz_thr_sup;
        for (int i = 0; i < 3; i++) {
          final row = W_turn[i];
          for (int j = 0; j < row.length; j++) dh2[j] += row[j] * dz_turn_sup[i];
        }
      }

      // Planner thrust supervision at decision states (coarse)
      if (thrAlignWeight > 0.0 && thrPlanLabels != null) {
        final yThr = thrPlanLabels[t].clamp(0, 1).toDouble();
        double dz_thrplan = thrAlignWeight * (f.thrPlanP - yThr);
        addInPlaceMat(dW_thrplan, outer([dz_thrplan], f.h2)); db_thrplan[0] += dz_thrplan;
        for (int j = 0; j < W_thrplan[0].length; j++) dh2[j] += W_thrplan[0][j] * dz_thrplan;
      }

      final dz2 = List<double>.generate(f.z2.length, (i) => dh2[i] * dRelu(f.z2[i]));
      addInPlaceMat(dW2, outer(dz2, f.h1)); addInPlaceVec(db2, dz2);

      final dh1 = List<double>.filled(f.h1.length, 0);
      for (int i = 0; i < W2.length; i++) {
        for (int j = 0; j < W2[0].length; j++) dh1[j] += W2[i][j] * dz2[i];
      }
      final dz1 = List<double>.generate(f.z1.length, (i) => dh1[i] * dRelu(f.z1[i]));
      addInPlaceMat(dW1, outer(dz1, f.x)); addInPlaceVec(db1, dz1);
    }

    void addL2(List<List<double>> dW, List<List<double>> W) {
      for (int i = 0; i < dW.length; i++) {
        for (int j = 0; j < dW[0].length; j++) dW[i][j] += l2 * W[i][j];
      }
    }
    addL2(dW1, W1); addL2(dW2, W2);
    addL2(dW_int, W_intent); addL2(dW_val, W_val);
    addL2(dW_thrplan, W_thrplan);
    addL2(dW_thr, W_thr); addL2(dW_turn, W_turn);

    double sq = 0.0;
    void accumM(List<List<double>> G){ for(final r in G){ for(final v in r) sq += v*v; } }
    void accumB(List<double> g){ for(final v in g) sq += v*v; }
    accumM(dW1); accumM(dW2); accumM(dW_int); accumM(dW_val);
    accumM(dW_thrplan); accumM(dW_thr); accumM(dW_turn);
    accumB(db1); accumB(db2); accumB(db_int); accumB(db_val);
    accumB(db_thrplan); accumB(db_thr); accumB(db_turn);
    final clip = 5.0;
    final nrm = math.sqrt(sq + 1e-12);
    final sc = nrm > clip ? (clip / nrm) : 1.0;
    if (sc != 1.0) {
      void scaleM(List<List<double>> G){ for(final r in G){ for(int j=0;j<r.length;j++) r[j]*=sc; } }
      void scaleB(List<double> g){ for (int j=0;j<g.length;j++) g[j]*=sc; }
      scaleM(dW1); scaleM(dW2); scaleM(dW_int); scaleM(dW_val);
      scaleM(dW_thrplan); scaleM(dW_thr); scaleM(dW_turn);
      scaleB(db1); scaleB(db2); scaleB(db_int); scaleB(db_val);
      scaleB(db_thrplan); scaleB(db_thr); scaleB(db_turn);
    }

    void sgdW(List<List<double>> W, List<List<double>> dW) {
      for (int i = 0; i < W.length; i++) {
        for (int j = 0; j < W[0].length; j++) W[i][j] -= lr * dW[i][j];
      }
    }
    void sgdB(List<double> b, List<double> db) { for (int i = 0; i < b.length; i++) b[i] -= lr * db[i]; }

    sgdW(W_intent, dW_int);     sgdB(b_intent, db_int);
    sgdW(W_val, dW_val);        sgdB(b_val, db_val);
    sgdW(W_thrplan, dW_thrplan);sgdB(b_thrplan, db_thrplan);
    sgdW(W_thr, dW_thr);        sgdB(b_thr, db_thr);
    sgdW(W_turn, dW_turn);      sgdB(b_turn, db_turn);
    sgdW(W2, dW2);              sgdB(b2, db2);
    sgdW(W1, dW1);              sgdB(b1, db1);
  }

  double _huberGrad(double error, double delta) {
    final ae = error.abs();
    if (ae <= delta) return error;
    return delta * (error.isNegative ? -1.0 : 1.0);
  }

  void updateFromEpisode({
    // single-stage data
    List<_Forward>? caches,
    List<List<int>>? actions,
    List<double>? returns_,

    // two-stage data
    List<_Forward>? decisionCaches,
    List<int>? intentChoices,
    List<double>? decisionReturns,

    // auxiliaries
    List<int>? alignLabels,
    double alignWeight = 0.0,

    List<List<int>>? actionLabels,
    double actionAlignWeight = 0.0,

    List<int>? thrPlanLabels,
    double thrAlignWeight = 0.0,

    // common hyperparams
    double lr = 3e-4,
    double l2 = 1e-6,
    double entropyBeta = 0.0,
    double valueBeta = 0.5,
    double huberDelta = 1.0,

    bool intentMode = true,
  }) {
    if (intentMode) {
      _updateIntentStage(
        decisionCaches: decisionCaches ?? const [],
        intentChoices: intentChoices ?? const [],
        decisionReturns: decisionReturns ?? const [],
        alignLabels: alignLabels,
        alignWeight: alignWeight,
        actionLabels: actionLabels,
        actionAlignWeight: actionAlignWeight,
        thrPlanLabels: thrPlanLabels,
        thrAlignWeight: thrAlignWeight,
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

  // Learned controller switch/blend (legacy turns head)
  final bool useLearnedController;     // use action heads to drive controls
  final double blendPolicy;            // 0=heuristic, 1=policy (turns)

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
    // two-stage defaults
    this.twoStage = true,
    this.planHold = 1,
    this.tempIntent = 1.0,
    this.intentEntropyBeta = 0.0,
    // learned controller defaults
    this.useLearnedController = false,
    this.blendPolicy = 1.0,
  }) : rnd = math.Random(seed);

  EpisodeResult runEpisode({
    bool train = true,
    double lr = 3e-4,
    bool greedy = false,
    bool scoreIsReward = false,
    double valueBeta = 0.5,
    double huberDelta = 1.0,
  }) {
    // Reset FE state so accel finite-diff starts clean
    fe.reset();

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

        final s = info.scoreDelta;
        final stepCost = scoreIsReward ? -s : s;

        caches.add(cache);
        actions.add([th ? 1 : 0, lf ? 1 : 0, rt ? 1 : 0]);
        costs.add(stepCost);

        if (info.terminal || caches.length > 4000) break;
        t++;
      }

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
      // ----------------------- Two-stage planner with thrust head ---------------
      final decisionCaches = <_Forward>[];
      final intentChoices  = <int>[];
      final decisionTimes  = <int>[];
      final costs          = <double>[];

      final heuristicLabels = <int>[];     // intent teacher (optional)
      final actionLabels = <List<int>>[];  // legacy distill (optional)
      final thrPlanLabels = <int>[];       // per-frame teacher for thrust (we'll downsample for decision-coupled CE)

      // Per-frame caches for thrust head supervised update
      final thrFrameCaches = <_Forward>[];
      final thrFrameLabels = <int>[];

      final intentCounts = List<int>.filled(PolicyNetwork.kIntents, 0);
      int intentSwitches = 0;

      int t = 0;
      int framesLeft = 0;
      int currentIntentIdx = -1;

      while (true) {
        // Decide (lateral) intent occasionally
        if (framesLeft <= 0) {
          final xPlan = fe.extract(env);
          int idx; List<double> probs; _Forward cache;

          if (greedy) {
            final res = policy.actIntentGreedy(xPlan);
            idx = res.$1; probs = res.$2; cache = res.$3;
          } else {
            final res = policy.actIntent(xPlan, rnd, tempIntent: tempIntent, epsilon: epsilon);
            idx = res.$1; probs = res.$2; cache = res.$3;
          }

          if (_shouldBrakeNow(env, margin: 50.0, reactSec: 0.10)) {
            idx = intentToIndex(Intent.brakeUp);
            probs = List<double>.filled(PolicyNetwork.kIntents, 0.0)..[idx] = 1.0;
          }

          if (currentIntentIdx == -1) currentIntentIdx = idx;
          if (idx == intentToIndex(Intent.descendSlow)) {
            final L = env.lander;
            final padCx = env.terrain.padCenter.toDouble();
            final padHalfW = (((env.terrain.padX2 - env.terrain.padX1).abs()) * 0.5)
                .clamp(1.0, env.cfg.worldW.toDouble());
            final dx = L.pos.x - padCx;
            final vx = L.vel.x;

            final tauHys = 1.0;
            final dxf = dx + vx * tauHys;
            final dxfN = (dxf.abs() / padHalfW);

            const dxInnerStrict = 0.15;
            const vxSmallStrict = 14.0;

            final wasLateral = (currentIntentIdx == intentToIndex(Intent.goLeft)) ||
                (currentIntentIdx == intentToIndex(Intent.goRight));
            final centeredEnough = (dxfN < dxInnerStrict) && (vx.abs() < vxSmallStrict);

            if (wasLateral && !centeredEnough) {
              idx = currentIntentIdx; // hold lateral one more hold
            }
          }

          currentIntentIdx = idx;
          decisionCaches.add(cache);
          intentChoices.add(idx);
          decisionTimes.add(t);
          framesLeft = planHold;

          intentCounts[idx] += 1;
          if (intentChoices.length >= 2 &&
              intentChoices.last != intentChoices[intentChoices.length - 2]) {
            intentSwitches += 1;
          }

          heuristicLabels.add(heuristicIntentLabel(env));

          // Publish intent to UI
          IntentBus.instance.publishIntent(IntentEvent(
            intent: kIntentNames[idx],
            probs: probs,
            step: t,
            meta: {'episode_step': t, 'plan_hold': planHold},
          ));

          // Distill legacy action heads once per decision (optional)
          final uTeach = controllerForIntent(indexToIntent(currentIntentIdx), env, thrustPlan: true);
          actionLabels.add([uTeach.thrust ? 1 : 0, uTeach.left ? 1 : 0, uTeach.right ? 1 : 0]);
        }

        // Every frame: planner thrust decision
        final xThr = fe.extract(env);
        bool thrustPlan; double pThrPlan; _Forward thrCache;
        if (greedy) {
          final pr = policy.actThrustPlanGreedy(xThr);
          thrustPlan = pr.$1; pThrPlan = pr.$2; thrCache = pr.$3;
        } else {
          final pr = policy.actThrustPlan(xThr, rnd, tempThrPlan: 1.0);
          thrustPlan = pr.$1; pThrPlan = pr.$2; thrCache = pr.$3;
        }

        // Conservative per-frame teacher
        final groundY = env.terrain.heightAt(env.lander.pos.x);
        final height  = groundY - env.lander.pos.y;
        final vy      = env.lander.vel.y; // +down
        final wantVy  = _safeDescentVy(height);

        final angAbsDeg = (env.lander.angle.abs() * 180.0 / math.pi);
        final tilting   = angAbsDeg > 8.0;  // "actively turning" threshold

        final needBrake = _shouldBrakeNow(env, margin: 40.0, reactSec: 0.08);
        final tooFast   = vy > (wantVy + 6.0);
        final climbing  = vy < -4.0 && height > 40.0;

        final yThrTeacher = (needBrake || (tooFast && !climbing) || (height < 30 && vy > 10))
            ? 1 : (tilting ? 0 : 0);

        thrPlanLabels.add(yThrTeacher);
        thrFrameCaches.add(thrCache);
        thrFrameLabels.add(yThrTeacher);

        // Compose control:
        final uHeurTurns = controllerForIntent(indexToIntent(currentIntentIdx), env, thrustPlan: thrustPlan);

        et.ControlInput u = uHeurTurns;
        if (useLearnedController) {
          final res = policy.actGreedy(xThr); // legacy heads for turns
          final uPol = et.ControlInput(thrust: thrustPlan, left: res.$2, right: res.$3);
          if (blendPolicy >= 1.0) {
            u = uPol;
          } else if (blendPolicy <= 0.0) {
            u = uHeurTurns;
          } else {
            final usePolicyTurns = blendPolicy >= 0.5;
            u = et.ControlInput(
              thrust: thrustPlan,
              left:  usePolicyTurns ? uPol.left  : uHeurTurns.left,
              right: usePolicyTurns ? uPol.right : uHeurTurns.right,
            );
          }
        } else {
          // keep safety overrides inside controllerForIntent
          u = et.ControlInput(thrust: uHeurTurns.thrust, left: uHeurTurns.left, right: uHeurTurns.right);
        }

        // Publish control to UI
        IntentBus.instance.publishControl(ControlEvent(
          thrust: u.thrust, left: u.left, right: u.right, step: t,
          meta: {'intent': kIntentNames[currentIntentIdx], 'pThrPlan': pThrPlan},
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

      // Downsample thrust teacher to decision times (coarse CE)
      final thrPlanLabelsDecision = <int>[];
      for (final ti in decisionTimes) {
        final idx = ti.clamp(0, thrPlanLabels.length - 1);
        thrPlanLabelsDecision.add(thrPlanLabels[idx]);
      }

      if (train && decisionCaches.isNotEmpty) {
        policy.updateFromEpisode(
          decisionCaches: decisionCaches,
          intentChoices: intentChoices,
          decisionReturns: decisionReturns,
          alignLabels: heuristicLabels,
          alignWeight: 0.08,
          actionLabels: actionLabels,
          actionAlignWeight: 0.0,
          thrPlanLabels: thrPlanLabelsDecision,
          thrAlignWeight: 0.8,
          lr: lr,
          entropyBeta: intentEntropyBeta,
          valueBeta: valueBeta,
          huberDelta: huberDelta,
          intentMode: true,
        );
      }

      // Per-frame supervised update for thrust head (strong signal)
      if (train && thrFrameCaches.isNotEmpty) {
        policy.updateThrustHeadSupervised(
          caches: thrFrameCaches,
          labels: thrFrameLabels,
          lr: lr,
          l2: 1e-6,
          entropyBeta: 0.0,
          backpropTrunk: true,
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

/* ------------------------- Controller: turns only + safety -------------------- */

et.ControlInput controllerForIntent(Intent intent, eng.GameEngine env, { required bool thrustPlan }) {
  final L = env.lander;
  final padCx = env.terrain.padCenter;
  final dx = L.pos.x - padCx;
  final vx = L.vel.x;
  final vy = L.vel.y;     // +down
  final angle = L.angle;

  bool left = false, right = false;

  // Angle PD toward desired angle (lean into correcting dx and vx)
  double targetAngle = 0.0;
  const kAngDx = 0.005;
  const kAngVx = 0.010;
  const maxTilt = 15 * math.pi / 180;

  switch (intent) {
    case Intent.goLeft:
      targetAngle = (-kAngDx * dx - kAngVx * vx).clamp(-maxTilt, maxTilt);
      break;
    case Intent.goRight:
      targetAngle = ( kAngDx * dx + kAngVx * vx).clamp(-maxTilt, maxTilt);
      break;
    case Intent.hoverCenter:
      targetAngle = (-0.5 * kAngDx * dx - 0.5 * kAngVx * vx).clamp(-maxTilt, maxTilt);
      break;
    case Intent.descendSlow:
    case Intent.brakeUp:
      targetAngle = 0.0;
      break;
  }

  // Soften lateral tilt if descending fast to avoid turning lift into climb
  if ((intent == Intent.goLeft || intent == Intent.goRight) && vy > 60.0) {
    final softMax = 10 * math.pi / 180; // 10°
    targetAngle = targetAngle.clamp(-softMax, softMax);
  }

  const angDead = 3 * math.pi / 180;
  if (angle > targetAngle + angDead) left = true;
  if (angle < targetAngle - angDead) right = true;

  // Thrust is decided by planner-thrust head; apply safety overrides here:
  bool thrust = thrustPlan;

  // Emergency brake (if required): force thrust on.
  if (_shouldBrakeNow(env, margin: 50.0, reactSec: 0.10)) {
    thrust = true;
  }

  // Avoid ceiling hover
  if (L.pos.y < env.cfg.ceilingMargin) {
    thrust = false;
  }

  return et.ControlInput(thrust: thrust, left: left, right: right);
}
