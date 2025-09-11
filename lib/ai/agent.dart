// lib/ai/agent.dart
import 'dart:math' as math;

import '../engine/game_engine.dart' as eng;
import '../engine/types.dart' as et;
import 'intent_bus.dart'; // ControlInput (adjust/remove if yours is in game_engine.dart)

/* ----------------------------- Feature Extraction ----------------------------- */

class FeatureExtractor {
  final int groundSamples; // total number of local ground samples (even or odd)
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

    // Local ground samples (generate EXACTLY groundSamples, even or odd)
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

/// Simple deterministic controller that converts a chosen intent into
/// (thrust,left,right) per physics step. Tune the gains here.
et.ControlInput controllerForIntent(Intent intent, eng.GameEngine env) {
  final L = env.lander;
  final padCx = env.terrain.padCenter;
  final dx = L.pos.x - padCx;
  final vx = L.vel.x;
  final vy = L.vel.y;
  final angle = L.angle;

  bool left = false, right = false, thrust = false;

  // -------- Horizontal guidance --------
  const double vxGoalAbs = 80.0;    // ↑ from 60 → snappier sideways moves
  const double kAngV     = 0.015;   // ↑ from 0.012 → quicker tilt toward vxDes
  const double kDxHover  = 0.40;    // hover centering gain
  const double maxTilt   = 20 * math.pi / 180; // ↑ allow a bit more lean

  double vxDes = 0.0;
  switch (intent) {
    case Intent.goLeft:       vxDes = -vxGoalAbs; break;
    case Intent.goRight:      vxDes = vxGoalAbs; break;
    case Intent.hoverCenter:  vxDes = -kDxHover * dx; break; // gentle drift to pad
    case Intent.descendSlow:
    case Intent.brakeUp:      vxDes = 0.0; break;
  }

  // target angle from horizontal velocity error
  final vxErr = (vxDes - vx);
  double targetAngle = (kAngV * vxErr).clamp(-maxTilt, maxTilt);

  // map desired angle to left/right jets with a deadzone
  const angDead = 3 * math.pi / 180;
  if (angle > targetAngle + angDead) left = true;
  if (angle < targetAngle - angDead) right = true;

  // -------- Vertical guidance (early braking) --------
  // Height-relative descent cap: tighter near ground, looser high up.
  final groundY = env.terrain.heightAt(L.pos.x);
  final height  = (groundY - L.pos.y).clamp(0.0, 1e9);

  // sqrt profile works well for moon-lander style units
  // Example: at 400px → ~30 px/s, at 100px → ~17 px/s, near ground → ~10 px/s
  double vyCapDown = 10.0 + 1.0 * math.sqrt(height.clamp(0.0, 9999.0));
  vyCapDown = vyCapDown.clamp(10.0, 45.0);

  // base targets
  double targetVy = vyCapDown;         // default: descend within the cap
  if (intent == Intent.descendSlow) targetVy = math.min(targetVy, 18.0);
  if (intent == Intent.brakeUp)     targetVy = -15.0;  // climb a bit

  // additional near-ground tightening
  if (height < 120) targetVy = math.min(targetVy, 18.0);
  if (height <  60) targetVy = math.min(targetVy, 10.0);

  // -------- Thrust logic --------
  // 1) Vertical safety: brake if falling faster than target
  final eVy = vy - targetVy;
  thrust = eVy > 0;

  // 2) Horizontal assist: if we are tilted toward the desired angle and
  //    still far from desired vx, burn to inject horizontal momentum.
  const double vxErrTh = 20.0;             // need meaningful horizontal error
  final bool tiltAligned =
      (targetAngle >  6 * math.pi / 180 && angle >  3 * math.pi / 180) || // tilted right
          (targetAngle < -6 * math.pi / 180 && angle < -3 * math.pi / 180);   // tilted left
  if ((intent == Intent.goLeft || intent == Intent.goRight) &&
      tiltAligned && vxErr.abs() > vxErrTh) {
    thrust = true;
  }

  // avoid ceiling hover
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

  /// Deterministic (greedy) action: no sampling, no temps/epsilon.
  /// Returns (thrust, left, right, probs[ p_thr, p_turn0, p_turn1, p_turn2 ], cache)
  (bool thrust, bool left, bool right, List<double> probs, _Forward cache)
  actGreedy(List<double> x) {
    final f = _forward(x);

    // throttle: STRICT tie-break → no thrust on tie
    final thrust = f.thrP > 0.5;

    // turn: argmax over raw logits: classes [none, left, right]
    final cls = _argmax(f.turnLogits);
    final left = cls == 1;
    final right = cls == 2;

    return (thrust, left, right, [f.thrP, ...f.turnP], f);
  }

  /// Tempered + epsilon-greedy action sampling (stochastic)
  (bool thrust, bool left, bool right, List<double> probs, _Forward cache) act(
      List<double> x,
      math.Random rnd, {
        double tempThr = 1.0,
        double tempTurn = 1.0,
        double epsilon = 0.0,
      }) {
    final f = _forward(x);

    // temperature scaling
    final pThr = sigmoid(f.thrLogit / math.max(1e-6, tempThr));

    final scaled = [
      f.turnLogits[0] / math.max(1e-6, tempTurn),
      f.turnLogits[1] / math.max(1e-6, tempTurn),
      f.turnLogits[2] / math.max(1e-6, tempTurn),
    ];
    final pTurn = softmax(scaled); // [none,left,right]

    // sample
    bool thrust = rnd.nextDouble() < pThr;
    final r = rnd.nextDouble();
    int cls;
    if (r < pTurn[0]) cls = 0;
    else if (r < pTurn[0] + pTurn[1]) cls = 1;
    else cls = 2;

    bool left = cls == 1;
    bool right = cls == 2;

    // epsilon-greedy kick
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

  /// Greedy intent selection (deterministic). Returns (intentIndex, probs[K], cache)
  (int intentIndex, List<double> probs, _Forward cache) actIntentGreedy(List<double> x) {
    final f = _forward(x);
    final idx = _argmax(f.intentLogits);
    return (idx, List<double>.from(f.intentP), f);
  }

  /// Sample an intent with temperature + epsilon.
  (int intentIndex, List<double> probs, _Forward cache) actIntent(
      List<double> x,
      math.Random rnd, {
        double tempIntent = 1.0,
        double epsilon = 0.0,
      }) {
    final f = _forward(x);
    final scaled = List<double>.generate(PolicyNetwork.kIntents, (i) => f.intentLogits[i] / math.max(1e-6, tempIntent));
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

  // Single-stage update (thr/turn + value + optional entropy)
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
    double lr = 3e-4,
    double l2 = 1e-6,
    double entropyBeta = 0.0,
    double valueBeta = 0.5,
    double huberDelta = 1.0,
  }) {
    final T = decisionCaches.length;
    if (T == 0) return;

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

      // logits grad for categorical intent
      final target = List<double>.filled(PolicyNetwork.kIntents, 0.0);
      target[k] = 1.0;
      final dz_int = List<double>.generate(PolicyNetwork.kIntents, (i) => (f.intentP[i] - target[i]) * A);

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
      // from intent head
      for (int i = 0; i < W_intent.length; i++) {
        final row = W_intent[i];
        for (int j = 0; j < row.length; j++) dh2[j] += row[j] * dz_int[i];
      }
      // from value head
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
    this.planHold = 12,
    this.tempIntent = 1.0,
    this.intentEntropyBeta = 0.0,
  }) : rnd = math.Random(seed);

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
          intentMode: false, // FIX: train the single-stage heads here
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

      final intentCounts = List<int>.filled(PolicyNetwork.kIntents, 0);
      int intentSwitches = 0;

      // NEW: actuation counters for diagnostics
      int turnSteps = 0, leftSteps = 0, rightSteps = 0, thrustSteps = 0;
      double thrProbSum = 0.0; // stays 0.0 in two-stage (no p_thr here)

      int t = 0;
      int framesLeft = 0;
      int currentIntentIdx = -1;

      while (true) {
        if (framesLeft <= 0) {
          // (decide intent as you already do)
          final xPlan = fe.extract(env);
          int idx; List<double> probs; _Forward cache;

          if (greedy) {
            final res = policy.actIntentGreedy(xPlan);
            idx = res.$1; probs = res.$2; cache = res.$3;
          } else {
            final res = policy.actIntent(xPlan, rnd, tempIntent: tempIntent, epsilon: epsilon);
            idx = res.$1; probs = res.$2; cache = res.$3;
          }

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

          // PUBLISH INTENT
          IntentBus.instance.publishIntent(IntentEvent(
            intent: kIntentNames[idx],
            probs: probs,
            step: t,
            meta: {
              'episode_step': t,
              'plan_hold': planHold,
            },
          ));
        } else {
          // heartbeat so late subscribers still see something
          IntentBus.instance.publishIntent(IntentEvent(
            intent: kIntentNames[currentIntentIdx],
            probs: const [], // same as prior, or omit
            step: t,
            meta: {'hold': true, 'framesLeft': framesLeft},
          ));
        }
        final intent = indexToIntent(currentIntentIdx);
        final u = controllerForIntent(intent, env);

        // PUBLISH CONTROL (what the controller actually outputs)
        IntentBus.instance.publishControl(ControlEvent(
          thrust: u.thrust, left: u.left, right: u.right, step: t,
          meta: {'intent': kIntentNames[currentIntentIdx]},
        ));

        // NEW: count actuation for diagnostics
        if (u.left || u.right) turnSteps++;
        if (u.left) leftSteps++;
        if (u.right) rightSteps++;
        if (u.thrust) thrustSteps++;

        final info = env.step(dt, et.ControlInput(thrust: u.thrust, left: u.left, right: u.right));
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
        // expose actuation diagnostics so your logs show non-zero %
        turnSteps: turnSteps,
        leftSteps: leftSteps,
        rightSteps: rightSteps,
        thrustSteps: thrustSteps,
        avgThrProb: thrProbSum,
        intentCounts: intentCounts,
        intentSwitches: intentSwitches,
      );
    }
  }
}
