// lib/ai/agent.dart
import 'dart:math' as math;

import '../engine/game_engine.dart' as eng;
import '../engine/types.dart' as et;

// If you don't have intent_bus.dart, you can delete these imports & the two publish blocks.
import 'intent_bus.dart';

/* ----------------------------- Feature Extraction ----------------------------- */

/// Feature extractor with a tiny internal state so we can estimate acceleration
/// from velocity deltas (no engine changes required).
class FeatureExtractor {
  final int groundSamples; // number of local ground samples (even or odd)
  final double stridePx;
  final double dt;         // seconds per physics step (e.g. 1/60)

  // persistent tiny state (reset each episode)
  double _lastVx = 0.0;
  double _lastVy = 0.0;
  bool _haveLastVel = false;

  FeatureExtractor({
    this.groundSamples = 3,
    this.stridePx = 48,
    this.dt = 1 / 60.0,
  });

  void reset() {
    _haveLastVel = false;
    _lastVx = 0.0;
    _lastVy = 0.0;
  }

  List<double> extract(eng.GameEngine e) {
    final L = e.lander;
    final T = e.terrain;
    final cfg = e.cfg;

    // Normalize base kinematics
    final px = L.pos.x / cfg.worldW;                 // 0..1
    final py = L.pos.y / cfg.worldH;                 // 0..1
    final vx = (L.vel.x / 200.0).clamp(-2.0, 2.0);   // ~-1..1
    final vy = (L.vel.y / 200.0).clamp(-2.0, 2.0);
    final ang = (L.angle / math.pi).clamp(-1.5, 1.5);
    final fuel = (L.fuel / cfg.t.maxFuel).clamp(0.0, 1.0);

    // Estimated acceleration from velocity deltas (engine+gravity+tilt effect).
    double axEst = 0.0, ayEst = 0.0;
    if (_haveLastVel && dt > 0) {
      axEst = ((L.vel.x - _lastVx) / dt) / 400.0; // normalize a bit
      ayEst = ((L.vel.y - _lastVy) / dt) / 400.0;
      axEst = axEst.clamp(-2.0, 2.0);
      ayEst = ayEst.clamp(-2.0, 2.0);
    }
    _lastVx = L.vel.x;
    _lastVy = L.vel.y;
    _haveLastVel = true;

    // Pad info
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
      axEst, ayEst,       // NEW: engine+gravity proxy
      ...samples,
    ];
  }

  int get inputSize => 12 + groundSamples; // (10 base + 2 accel) + samples
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
    return dx > 0 ? intentToIndex(Intent.goLeft) : intentToIndex(Intent.goRight);
  } else {
    if (high) return intentToIndex(Intent.descendSlow);
    return (vy > 55.0) ? intentToIndex(Intent.brakeUp) : intentToIndex(Intent.hoverCenter);
  }
}

/// Predictive labeler: look ahead tau seconds and pick the intent that will be correct.
int predictiveIntentLabel(eng.GameEngine env, {double tauSec = 0.6}) {
  final L = env.lander;
  final padCx = env.terrain.padCenter.toDouble();

  final dx = L.pos.x - padCx;      // + if right of pad
  final vx = L.vel.x;              // px/s
  final vy = L.vel.y;              // +down
  final groundY = env.terrain.heightAt(L.pos.x);
  final height = groundY - L.pos.y;    // px above ground

  final dxFuture = dx + vx * tauSec;
  final padHalfW = (((env.terrain.padX2 - env.terrain.padX1).abs()) * 0.5)
      .clamp(1.0, env.cfg.worldW.toDouble());
  final dxfN = (dxFuture.abs() / padHalfW);

  final vySafe = math.max(10.0, vy);
  final tGround = height / vySafe;

  const double dxOuter = 0.30;
  const double dxInner = 0.18;
  const double vxSmall = 18.0;

  if (tGround < tauSec * 0.7 && vy > 55.0) return intentToIndex(Intent.brakeUp);

  final bool high = height > 0.30 * env.cfg.worldH;

  if (dxfN > dxOuter) {
    return (dxFuture > 0) ? intentToIndex(Intent.goLeft) : intentToIndex(Intent.goRight);
  }
  if (dxfN < dxInner && vx.abs() < vxSmall) {
    if (high) return intentToIndex(Intent.descendSlow);
    return (vy > 55.0) ? intentToIndex(Intent.brakeUp) : intentToIndex(Intent.hoverCenter);
  }
  return (dxFuture > 0) ? intentToIndex(Intent.goLeft) : intentToIndex(Intent.goRight);
}

/// Predictive-but-adaptive: ~1.0s lookahead, adapts with distance/height.
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

  const double dxOuter = 0.30;
  const double dxInner = 0.18;
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
  return (dxFuture > 0) ? intentToIndex(Intent.goLeft)
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
  const kAngDx = 0.005;     // how much horizontal error affects target angle
  const kAngVx = 0.010;     // how much vx affects target angle
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

  // Soften lateral tilt when descending fast to avoid thrust spikes
  if ((intent == Intent.goLeft || intent == Intent.goRight) && vy > 70.0) {
    final softMax = 10 * math.pi / 180; // 10°
    targetAngle = targetAngle.clamp(-softMax, softMax);
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

  // Action Heads (single-stage; also used for learned controller)
  late List<List<double>> W_thr;   // (1, h2)
  late List<double> b_thr;         // (1)
  late List<List<double>> W_turn;  // (3, h2)  [none,left,right]
  late List<double> b_turn;        // (3)

  // Intent Head (planner)
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

    // planner head
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

  // planner
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

    // action heads
    final thrLogit = matVec(W_thr, h2v)[0] + b_thr[0];
    final thrP = sigmoid(thrLogit);

    final turnLogits = matVec(W_turn, h2v);
    for (int i = 0; i < 3; i++) turnLogits[i] += b_turn[i];
    final turnP = softmax(turnLogits);

    // planner head
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

  // --- GAE helper (on decision points) ---
  List<double> _gaeAdvantages({
    required List<double> rewards,    // r_t at decision steps
    required List<double> values,     // V(s_t) at decision steps
    required double gamma,
    double lambda_ = 0.95,
    double bootstrapV = 0.0,
  }) {
    final T = rewards.length;
    final adv = List<double>.filled(T, 0.0);
    double nextAdv = 0.0;
    double nextV = bootstrapV;
    for (int t = T - 1; t >= 0; t--) {
      final delta = rewards[t] + gamma * nextV - values[t];
      nextAdv = delta + gamma * lambda_ * nextAdv;
      adv[t] = nextAdv;
      nextV = values[t];
    }
    // normalize
    if (T > 0) {
      final m = adv.reduce((a,b)=>a+b) / T;
      double v = 0.0; for (final a in adv) v += (a - m)*(a - m);
      v = math.sqrt((v / T) + 1e-8);
      if (v > 0) for (int i=0;i<T;i++) adv[i] = (adv[i] - m) / v;
    }
    return adv;
  }

  void _updateIntentStage({
    required List<_Forward> decisionCaches, // caches at decision times
    required List<int> intentChoices,       // indices 0..K-1
    required List<double> decisionRewards,  // per-decision summed rewards
    required double gamma,
    double gaeLambda = 0.95,
    bool bootstrap = true,        // if last step non-terminal, bootstrap V
    double bootstrapV = 0.0,      // supply when bootstrap==true

    List<int>? alignLabels,       // OPTIONAL heuristic labels for intent
    double alignWeight = 0.0,     // strength of auxiliary align loss

    // Optional supervised distillation of action heads at decision states
    List<List<int>>? actionLabels, // OPTIONAL: [thr,left,right]
    double actionAlignWeight = 0.0,

    // common
    double lr = 3e-4,
    double l2 = 1e-6,
    double entropyBeta = 0.0,
    double valueBeta = 0.5,
    double huberDelta = 1.0,
  }) {
    final T = decisionCaches.length;
    if (T == 0) return;
    if (intentChoices.length != T) return;
    if (decisionRewards.length != T) return;

    // Values at decision points
    final values = List<double>.generate(T, (t) => decisionCaches[t].v);

    // Compute GAE advantages on decisions
    final adv = _gaeAdvantages(
      rewards: decisionRewards,              // already aggregated per decision
      values: values,
      gamma: gamma,
      lambda_: gaeLambda,
      bootstrapV: bootstrap ? bootstrapV : 0.0,
    );

    // Returns for critic: V + A
    final decisionReturns = List<double>.generate(T, (i) => values[i] + adv[i]);

    // Accumulators
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

    final dW_val = zeros(W_val.length, W_val[0].length);
    final db_val = List<double>.filled(b_val.length, 0);

    for (int t = 0; t < T; t++) {
      final f = decisionCaches[t];
      final k = intentChoices[t];
      final A = adv[t];

      // logits grad for categorical intent (PG): (p - onehot) * A
      final target = List<double>.filled(PolicyNetwork.kIntents, 0.0);
      if (k >= 0 && k < PolicyNetwork.kIntents) target[k] = 1.0;
      final dz_int = List<double>.generate(
          PolicyNetwork.kIntents, (i) => (f.intentP[i] - target[i]) * A);

      // Auxiliary intent alignment
      if (alignWeight > 0.0 && alignLabels != null && alignLabels.length == T) {
        final yIdx = alignLabels[t];
        if (yIdx >= 0 && yIdx < PolicyNetwork.kIntents) {
          for (int i = 0; i < PolicyNetwork.kIntents; i++) {
            final y = (i == yIdx) ? 1.0 : 0.0;
            dz_int[i] += alignWeight * (f.intentP[i] - y);
          }
        }
      }

      // entropy on intents
      if (entropyBeta > 0.0) {
        final p = f.intentP.map((x) => x.clamp(1e-8, 1.0)).toList();
        final g = List<double>.generate(p.length, (i) => -(math.log(p[i]) + 1.0));
        double s = 0.0; for (int i = 0; i < p.length; i++) s += p[i] * g[i];
        for (int i = 0; i < p.length; i++) {
          final dH_dz_i = p[i] * g[i] - p[i] * s;
          dz_int[i] += -entropyBeta * dH_dz_i;
        }
      }

      // value head (Huber on v - R)
      final err = f.v - decisionReturns[t];
      double _huberGrad(double error, double delta) {
        final ae = error.abs();
        if (ae <= delta) return error;
        return delta * (error.isNegative ? -1.0 : 1.0);
      }
      final dLdv = valueBeta * _huberGrad(err, huberDelta);

      // heads: intent + value
      addInPlaceMat(dW_int, outer(dz_int, f.h2)); addInPlaceVec(db_int, dz_int);
      addInPlaceMat(dW_val, outer([dLdv], f.h2)); db_val[0] += dLdv;

      // backprop to trunk accumulates here
      final dh2 = List<double>.filled(f.h2.length, 0);
      for (int i = 0; i < W_intent.length; i++) {
        final row = W_intent[i];
        for (int j = 0; j < row.length; j++) dh2[j] += row[j] * dz_int[i];
      }
      for (int j = 0; j < W_val[0].length; j++) dh2[j] += W_val[0][j] * dLdv;

      // Optional: supervised action cloning grads at decision states
      if (actionAlignWeight > 0.0 && actionLabels != null && actionLabels.length == T) {
        final lab = actionLabels[t]; // [thr,left,right]
        if (lab.length == 3) {
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

    // L2
    void addL2(List<List<double>> dW, List<List<double>> W) {
      for (int i = 0; i < dW.length; i++) {
        for (int j = 0; j < dW[0].length; j++) dW[i][j] += l2 * W[i][j];
      }
    }
    addL2(dW1, W1); addL2(dW2, W2);
    addL2(dW_int, W_intent); addL2(dW_val, W_val);
    addL2(dW_thr, W_thr); addL2(dW_turn, W_turn);

    // clip
    double sq = 0.0;
    void accumM(List<List<double>> G){ for(final r in G){ for(final v in r) sq += v*v; } }
    void accumB(List<double> g){ for(final v in g) sq += v*v; }
    accumM(dW1); accumM(dW2); accumM(dW_int); accumM(dW_val); accumM(dW_thr); accumM(dW_turn);
    accumB(db1); accumB(db2); accumB(db_int); accumB(db_val); accumB(db_thr); accumB(db_turn);
    final clip = 5.0;
    final nrm = math.sqrt(sq + 1e-12);
    final sc = nrm > clip ? (clip / nrm) : 1.0;
    if (sc != 1.0) {
      void scaleM(List<List<double>> G){ for(final r in G){ for(int j=0;j<r.length;j++) r[j]*=sc; } }
      void scaleB(List<double> g){ for (int j=0;j<g.length;j++) g[j]*=sc; }
      scaleM(dW1); scaleM(dW2); scaleM(dW_int); scaleM(dW_val); scaleM(dW_thr); scaleM(dW_turn);
      scaleB(db1); scaleB(db2); scaleB(db_int); scaleB(db_val); scaleB(db_thr); scaleB(db_turn);
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
    sgdW(W_thr, dW_thr);    sgdB(b_thr, db_thr);
    sgdW(W_turn, dW_turn);  sgdB(b_turn, db_turn);
    sgdW(W2, dW2);          sgdB(b2, db2);
    sgdW(W1, dW1);          sgdB(b1, db1);
  }

  // Public: choose which update path to use
// AFTER
  void updateFromEpisode({
    // two-stage data
    List<dynamic>? decisionCaches,              // <— accept dynamic from train_agent.dart
    List<int>? intentChoices,
    List<double>? decisionRewards, // summed per decision interval
    required double gamma,
    double gaeLambda = 0.95,
    bool bootstrap = true,
    double bootstrapV = 0.0,
    List<int>? alignLabels,
    double alignWeight = 0.0,
    List<List<int>>? actionLabels,
    double actionAlignWeight = 0.0,
    // common hyperparams
    double lr = 3e-4,
    double l2 = 1e-6,
    double entropyBeta = 0.0,
    double valueBeta = 0.5,
    double huberDelta = 1.0,
  }) {
    // Safely cast the dynamic list to the private type used inside this file.
    final List<_Forward> _dc =
    (decisionCaches ?? const <dynamic>[]).cast<_Forward>();

    _updateIntentStage(
      decisionCaches: _dc,
      intentChoices: intentChoices ?? const [],
      decisionRewards: decisionRewards ?? const [],
      gamma: gamma,
      gaeLambda: gaeLambda,
      bootstrap: bootstrap,
      bootstrapV: bootstrapV,
      alignLabels: alignLabels,
      alignWeight: alignWeight,
      actionLabels: actionLabels,
      actionAlignWeight: actionAlignWeight,
      lr: lr, l2: l2, entropyBeta: entropyBeta,
      valueBeta: valueBeta, huberDelta: huberDelta,
    );
  }
}

/* --------------------------- Episode runner & trainer ------------------------- */

class EpisodeResult {
  final double totalCost;
  final int steps;
  final bool landed;

  // diagnostics (two-stage)
  final List<int> intentCounts;   // histogram over intents
  final int intentSwitches;

  EpisodeResult(
      this.totalCost,
      this.steps,
      this.landed, {
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

  // two-stage knobs
  final bool twoStage;                 // if true, use planner/controller
  final int planHold;                  // frames to hold an intent
  final double tempIntent;             // temperature for intent sampling
  final double intentEntropyBeta;      // entropy on intent head

  // learned controller switch/blend (usually off early in training)
  final bool useLearnedController;     // use action heads to drive controls
  final double blendPolicy;            // 0=heuristic, 1=policy (simple blend gate)

  // segmenting
  final int segHardMax;    // hard cap frames per segment
  final int segMin;        // minimum frames before allowing a boundary
  int _segmentsFlushed = 0;

  Trainer({
    required this.env,
    required this.fe,
    required this.policy,
    this.dt = 1/60.0,
    this.gamma = 0.99,
    int seed = 7,
    // two-stage defaults
    this.twoStage = true,
    this.planHold = 1,
    this.tempIntent = 1.0,
    this.intentEntropyBeta = 0.0,
    // learned controller defaults
    this.useLearnedController = false,
    this.blendPolicy = 1.0,
    // segment settings (works well in practice)
    this.segHardMax = 150,
    this.segMin = 50,
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

    if (vy <= 8.0) return false;

    // Stopping distance with constant decel: d = vy^2 / (2a)
    final stopDist = (vy * vy) / (2.0 * a);

    return (stopDist + margin) >= height;
  }

  // Height-based envelope (duplicate of scoring helper; local quick version)
  double _vyMaxForHeight(double h) {
    if (h > 260) return 60;
    if (h > 160) return 40;
    if (h > 100) return 25;
    if (h >  50) return 14;
    return 8;
  }

  EpisodeResult runEpisode({
    bool train = true,
    double lr = 3e-4,
    bool greedy = false,
    bool scoreIsReward = false,
    double valueBeta = 0.5,
    double huberDelta = 1.0,
  }) {
    // Reset per-episode extractor state
    fe.reset();

    // ----------------------- Two-stage path (planner/controller) --------------
    final decisionCaches = <_Forward>[];   // at decision points
    final intentChoices  = <int>[];
    final decisionRewards = <double>[];    // summed -cost between decisions
    final heuristicLabels = <int>[];       // optional small assist
    final actionLabels    = <List<int>>[]; // supervisor controls for distillation (optional)

    final intentCounts = List<int>.filled(PolicyNetwork.kIntents, 0);
    int intentSwitches = 0;

    // Per-frame accumulators for the *current* segment
    double segRewardAcc = 0.0;
    int segLen = 0;
    bool segTerminal = false;

    double totalCost = 0.0;

    int t = 0;
    int framesLeft = 0;
    int currentIntentIdx = -1;

    // helpers
    void _startNewDecision(int idx, _Forward cache, {int? heurLab, List<int>? actionLab}) {
      decisionCaches.add(cache);
      intentChoices.add(idx);
      if (heurLab != null) heuristicLabels.add(heurLab); else heuristicLabels.add(-1);
      if (actionLab != null) actionLabels.add(actionLab); else actionLabels.add(const [0,0,0]);
      // Close previous segment (if any) by pushing its summed reward
      if (decisionRewards.length < decisionCaches.length - 1) {
        decisionRewards.add(segRewardAcc);
        segRewardAcc = 0.0;
        segLen = 0;
      }
    }

    while (true) {
      final L = env.lander;

      // Decide intent when needed
      if (framesLeft <= 0) {
        final xPlan = fe.extract(env);
        int idx; List<double> probs; _Forward cache;

        if (greedy) {
          final res = policy.actIntentGreedy(xPlan);
          idx = res.$1; probs = res.$2; cache = res.$3;
        } else {
          final res = policy.actIntent(xPlan, rnd, tempIntent: tempIntent, epsilon: 0.0);
          idx = res.$1; probs = res.$2; cache = res.$3;
        }

        // physics-based brake override (safety)
        if (shouldBrakeNow(env, margin: 50.0)) {
          idx = intentToIndex(Intent.brakeUp);
          probs = List<double>.filled(PolicyNetwork.kIntents, 0.0)..[idx] = 1.0;
        }

        // Hysteresis: avoid switching to descendSlow too early if still off-center
        if (currentIntentIdx == -1) currentIntentIdx = idx; // guard first
        if (idx == intentToIndex(Intent.descendSlow)) {
          final padCx = env.terrain.padCenter.toDouble();
          final padHalfW = (((env.terrain.padX2 - env.terrain.padX1).abs()) * 0.5)
              .clamp(1.0, env.cfg.worldW.toDouble());
          final dx = L.pos.x - padCx;
          final vx = L.vel.x;

          final tauHys = 1.0; // match predictive horizon
          final dxf = dx + vx * tauHys;
          final dxfN = (dxf.abs() / padHalfW);

          const dxInnerStrict = 0.15;
          const vxSmallStrict = 14.0;

          final wasLateral = (currentIntentIdx == intentToIndex(Intent.goLeft)) ||
              (currentIntentIdx == intentToIndex(Intent.goRight));
          final centeredEnough = (dxfN < dxInnerStrict) && (vx.abs() < vxSmallStrict);

          if (wasLateral && !centeredEnough) {
            // hold previous lateral
            idx = currentIntentIdx;
          }
        }

        // Segment boundary by intent change: push accumulated reward for old segment
        if (decisionCaches.isNotEmpty && idx != currentIntentIdx) {
          decisionRewards.add(segRewardAcc);
          segRewardAcc = 0.0;
          segLen = 0;
          intentSwitches += 1;
        }

        currentIntentIdx = idx;
        framesLeft = planHold;

        // DIAGNOSTICS
        intentCounts[idx] += 1;

        // Optional: heuristic label for mild alignment
        final heurLab = heuristicIntentLabel(env);

        // Action label for distillation (use teacher controller at decision time)
        final teachU = controllerForIntent(indexToIntent(idx), env);
        final actLab = [teachU.thrust ? 1 : 0, teachU.left ? 1 : 0, teachU.right ? 1 : 0];

        _startNewDecision(idx, cache, heurLab: heurLab, actionLab: actLab);

        // (Optional) Publish intent to UI
        IntentBus.instance.publishIntent(IntentEvent(
          intent: kIntentNames[idx],
          probs: probs,
          step: t,
          meta: {'episode_step': t, 'plan_hold': planHold},
        ));
      }

      // Low-level control
      final intent = indexToIntent(currentIntentIdx);
      final uHeur = controllerForIntent(intent, env);

      // Policy action heads (student) control:
      et.ControlInput uPol;
      {
        final res = policy.actGreedy(fe.extract(env));
        uPol = et.ControlInput(thrust: res.$1, left: res.$2, right: res.$3);
      }

      // Choose control (blend/simple gate)
      et.ControlInput u = uHeur;
      if (useLearnedController) {
        if (blendPolicy >= 1.0) {
          u = uPol;
        } else if (blendPolicy <= 0.0) {
          u = uHeur;
        } else {
          // Simple gate blend: policy decides turns, heuristic decides thrust
          final usePolicyTurns = blendPolicy >= 0.5;
          final thrust = (blendPolicy >= 0.5) ? uPol.thrust : uHeur.thrust;
          final left   = usePolicyTurns ? uPol.left  : uHeur.left;
          final right  = usePolicyTurns ? uPol.right : uHeur.right;
          u = et.ControlInput(thrust: thrust, left: left, right: right);
        }
      }

      // (Optional) Publish control to UI
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

      final s = info.scoreDelta;            // >=0 (cost)
      final reward = scoreIsReward ? -s : -s; // reward is -cost
      segRewardAcc += reward;
      totalCost += s;

      // Segment boundary heuristics in addition to intent change:
      // 1) Hard cap length
      // 2) Violated descent envelope badly
      // 3) Thrust+turn sustained too long (ping-pong pattern)
      bool boundary = false;
      segLen += 1;

      final groundY = env.terrain.heightAt(env.lander.pos.x);
      final height = groundY - env.lander.pos.y;
      final vy = env.lander.vel.y;
      final vyMax = _vyMaxForHeight(height);
      final thrustTurn = u.thrust && (u.left || u.right);

      if (segLen >= segHardMax) boundary = true;
      if (vy > vyMax + 14.0 && segLen >= segMin) boundary = true;
      if (thrustTurn && segLen >= segMin && (segLen % 12 == 0)) boundary = true;

      framesLeft -= 1;
      t += 1;

      if (info.terminal || t > 4000) {
        segTerminal = true;
        // Close the final segment reward
        if (decisionRewards.length < decisionCaches.length) {
          decisionRewards.add(segRewardAcc);
          segRewardAcc = 0.0;
        }
        break;
      }

      if (boundary && decisionCaches.isNotEmpty) {
        decisionRewards.add(segRewardAcc);
        segRewardAcc = 0.0;
        segLen = 0;

        // force a new decision next frame
        framesLeft = 0;
      }
    }

    // === Update after the whole episode (but with *segment* rewards used) ===
    if (train && decisionCaches.isNotEmpty) {
      // Bootstrap if not terminal (rare because we break at terminal, but keep logic)
      final bool terminal = env.status != et.GameStatus.playing;
      final bool bootstrap = !terminal;
      final double bootstrapV = 0.0; // if you want, compute V(s_end) when !terminal

      // Keep supervised assists small during RL
      final double alignWeight = 0.02;
      final double actionAlignWeight = 0.0; // off by default; turn on later if desired

      policy.updateFromEpisode(
        decisionCaches: decisionCaches,
        intentChoices: intentChoices,
        decisionRewards: decisionRewards,
        gamma: gamma,
        gaeLambda: 0.95,
        bootstrap: bootstrap,
        bootstrapV: bootstrapV,
        alignLabels: heuristicLabels,
        alignWeight: alignWeight,
        actionLabels: actionLabels,
        actionAlignWeight: actionAlignWeight,
        lr: lr,
        l2: 1e-6,
        entropyBeta: intentEntropyBeta,
        valueBeta: valueBeta,
        huberDelta: huberDelta,
      );

      _segmentsFlushed += decisionCaches.length;
      // (You can print occasionally if you like.)
      // if ((_segmentsFlushed % 20) == 0) {
      //   print('seg/upd: total=$_segmentsFlushed  last=${decisionCaches.length}');
      // }
    }

    final landed = env.status == et.GameStatus.landed;

    return EpisodeResult(
      totalCost, t, landed,
      intentCounts: intentCounts,
      intentSwitches: intentSwitches,
    );
  }
}
