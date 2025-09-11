// lib/ai/agent.dart
import 'dart:math' as math;

// Game engine (physics integrator)
import '../engine/game_engine.dart' as eng;
// Shared types/configs (ControlInput, GameStatus, etc.)
import '../engine/types.dart' as et;

/// -------- Feature Extraction (state -> input vector) --------
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

/// --------- Tiny Linear Algebra ----------
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

/// --------- Nonlinearities ----------
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

/// --------- Policy Network (2 hidden, 2 action heads + value head) ----------
class PolicyNetwork {
  final int inputSize;
  final int h1;
  final int h2;

  // Trunk
  late List<List<double>> W1, W2;
  late List<double> b1, b2;

  // Action Heads
  late List<List<double>> W_thr;  // (1, h2)
  late List<double> b_thr;        // (1)
  late List<List<double>> W_turn; // (3, h2)
  late List<double> b_turn;       // (3)

  // Value Head (critic)
  late List<List<double>> W_val;  // (1, h2)
  late List<double> b_val;        // (1)

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

    W_thr = _init(1, h2);
    b_thr = List<double>.filled(1, 0);
    W_turn = _init(3, h2);
    b_turn = List<double>.filled(3, 0);

    W_val = _init(1, h2);
    b_val = List<double>.filled(1, 0);
  }

  List<List<double>> _init(int rows, int cols) {
    final limit = math.sqrt(6.0 / (rows + cols));
    return List.generate(
      rows,
          (_) => List<double>.generate(cols, (_) => (rnd.nextDouble() * 2 - 1) * limit),
    );
  }
}

/// Forward cache
class _Forward {
  final List<double> x;
  final List<double> z1, h1;
  final List<double> z2, h2;
  final double thrLogit, thrP;
  final List<double> turnLogits, turnP;
  final double v; // value prediction
  _Forward(this.x, this.z1, this.h1, this.z2, this.h2,
      this.thrLogit, this.thrP, this.turnLogits, this.turnP, this.v);
}

extension PolicyOps on PolicyNetwork {
  _Forward _forward(List<double> x) {
    final z1 = vecAdd(matVec(W1, x), b1);
    final h1v = reluVec(z1);
    final z2 = vecAdd(matVec(W2, h1v), b2);
    final h2v = reluVec(z2);

    // Action heads
    final thrLogit = matVec(W_thr, h2v)[0] + b_thr[0];
    final thrP = sigmoid(thrLogit);

    final turnLogits = matVec(W_turn, h2v);
    for (int i = 0; i < 3; i++) turnLogits[i] += b_turn[i];
    final turnP = softmax(turnLogits);

    // Value head
    final v = matVec(W_val, h2v)[0] + b_val[0];

    return _Forward(x, z1, h1v, z2, h2v, thrLogit, thrP, turnLogits, turnP, v);
  }

  int _argmax(List<double> v) {
    var bi = 0;
    var bv = v[0];
    for (int i = 1; i < v.length; i++) {
      if (v[i] > bv) { bv = v[i]; bi = i; }
    }
    return bi;
  }

  /// Deterministic (greedy) action: no sampling, no temps/epsilon.
  /// Returns (thrust, left, right, probs[ p_thr, p_turn0, p_turn1, p_turn2 ], cache)
  (bool thrust, bool left, bool right, List<double> probs, _Forward cache)
  actGreedy(List<double> x) {
    final f = _forward(x);

    // throttle: threshold at 0 (equiv to p>=0.5)
    final thrust = f.thrLogit >= 0.0;

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
    final pThr = sigmoid(f.thrLogit / tempThr);

    final scaled = [
      f.turnLogits[0] / tempTurn,
      f.turnLogits[1] / tempTurn,
      f.turnLogits[2] / tempTurn,
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

  // ---------------- Value head loss (Huber) ----------------
  double _huberGrad(double error, double delta) {
    // returns d/d(error) huber(error)
    final ae = error.abs();
    if (ae <= delta) return error;              // derivative of 0.5*e^2
    return delta * (error.isNegative ? -1.0 : 1.0); // derivative of delta*(|e|-0.5*delta)
  }

  /// One REINFORCE step over an episode, with entropy bonus and a value head baseline.
  void updateFromEpisode({
    required List<_Forward> caches,
    required List<List<int>> actions,   // [th, left, right]
    required List<double> returns_,     // discounted sum of rewards
    double lr = 3e-4,
    double l2 = 1e-6,
    double entropyBeta = 0.0,
    double valueBeta = 0.5,             // weight for value loss
    double huberDelta = 1.0,            // Huber delta for value loss
  }) {
    final T = caches.length;
    if (T == 0) return;

    // ------- Compute advantages: A = R - V(s) -------
    final values = List<double>.generate(T, (t) => caches[t].v);
    final adv = List<double>.generate(T, (t) => returns_[t] - values[t]);

    // Normalize advantages for stability
    final mean = adv.reduce((a, b) => a + b) / T;
    double var0 = 0.0; for (final v in adv) { var0 += (v - mean) * (v - mean); }
    var0 /= T;
    final std = math.sqrt(var0 + 1e-8);
    for (int i = 0; i < adv.length; i++) adv[i] = (adv[i] - mean) / std;

    // Grad accumulators
    final dW1 = zeros(W1.length, W1[0].length);
    final dW2 = zeros(W2.length, W2[0].length);
    final db1 = List<double>.filled(b1.length, 0);
    final db2 = List<double>.filled(b2.length, 0);

    final dW_thr = zeros(W_thr.length, W_thr[0].length); // (1,h2)
    final db_thr = List<double>.filled(b_thr.length, 0); // (1)
    final dW_turn = zeros(W_turn.length, W_turn[0].length); // (3,h2)
    final db_turn = List<double>.filled(b_turn.length, 0);   // (3)

    final dW_val = zeros(W_val.length, W_val[0].length); // (1,h2)
    final db_val = List<double>.filled(b_val.length, 0); // (1)

    for (int t = 0; t < T; t++) {
      final f = caches[t];
      final a = actions[t];      // [th, left, right]
      final A = adv[t];

      // ---------- Policy grad: logits grads = (p - a) * A ----------
      final a_thr = a[0].toDouble();
      final a_turn = <double>[1 - a[1] - a[2] + 0.0, a[1] + 0.0, a[2] + 0.0]; // [none,left,right]

      double dz_thr = (f.thrP - a_thr) * A;
      final dz_turn = List<double>.generate(3, (k) => (f.turnP[k] - a_turn[k]) * A);

      // ---------- Entropy bonus ----------
      if (entropyBeta > 0.0) {
        // Bernoulli entropy grad
        final p = f.thrP.clamp(1e-6, 1 - 1e-6);
        final dH_dz_thr = math.log((1 - p) / p) * p * (1 - p);
        dz_thr += -entropyBeta * dH_dz_thr;

        // Categorical entropy grad
        final p3 = f.turnP.map((x) => x.clamp(1e-8, 1.0)).toList();
        final g = List<double>.generate(3, (i) => -(math.log(p3[i]) + 1.0)); // -∂H/∂p
        double s = 0.0; for (int i = 0; i < 3; i++) s += p3[i] * g[i];
        for (int i = 0; i < 3; i++) {
          final dH_dz_i = p3[i] * g[i] - p3[i] * s; // (diag(p)-pp^T)g
          dz_turn[i] += -entropyBeta * dH_dz_i;
        }
      }

      // ---------- Value loss (Huber) ----------
      final err = f.v - returns_[t];
      final dLdv = valueBeta * _huberGrad(err, huberDelta);

      // ---------- Head grads ----------
      // throttle head
      addInPlaceMat(dW_thr, outer([dz_thr], f.h2)); db_thr[0] += dz_thr;
      // turn head
      addInPlaceMat(dW_turn, outer(dz_turn, f.h2)); addInPlaceVec(db_turn, dz_turn);
      // value head
      addInPlaceMat(dW_val, outer([dLdv], f.h2)); db_val[0] += dLdv;

      // ---------- Backprop to h2 (sum contributions from all heads) ----------
      final dh2 = List<double>.filled(f.h2.length, 0);
      // from throttle head
      for (int j = 0; j < W_thr[0].length; j++) dh2[j] += W_thr[0][j] * dz_thr;
      // from turn head
      for (int i = 0; i < 3; i++) {
        final row = W_turn[i];
        for (int j = 0; j < row.length; j++) dh2[j] += row[j] * dz_turn[i];
      }
      // from value head
      for (int j = 0; j < W_val[0].length; j++) dh2[j] += W_val[0][j] * dLdv;

      final dz2 = List<double>.generate(f.z2.length, (i) => dh2[i] * dRelu(f.z2[i]));
      addInPlaceMat(dW2, outer(dz2, f.h1)); addInPlaceVec(db2, dz2);

      // Backprop to h1
      final dh1 = List<double>.filled(f.h1.length, 0);
      for (int i = 0; i < W2.length; i++) {
        for (int j = 0; j < W2[0].length; j++) dh1[j] += W2[i][j] * dz2[i];
      }

      final dz1 = List<double>.generate(f.z1.length, (i) => dh1[i] * dRelu(f.z1[i]));
      addInPlaceMat(dW1, outer(dz1, f.x)); addInPlaceVec(db1, dz1);
    }

    // ---- L2 weight decay ----
    void addL2(List<List<double>> dW, List<List<double>> W) {
      for (int i = 0; i < dW.length; i++) {
        for (int j = 0; j < dW[0].length; j++) dW[i][j] += l2 * W[i][j];
      }
    }
    addL2(dW1, W1); addL2(dW2, W2);
    addL2(dW_thr, W_thr); addL2(dW_turn, W_turn);
    addL2(dW_val, W_val);

    // ---- Global-norm gradient clipping ----
    double sqSum = 0.0;
    void accum(List<List<double>> G) { for (final r in G) { for (final v in r) sqSum += v*v; } }
    void accumB(List<double> g) { for (final v in g) sqSum += v*v; }
    accum(dW1); accum(dW2); accum(dW_thr); accum(dW_turn); accum(dW_val);
    accumB(db1); accumB(db2); accumB(db_thr); accumB(db_turn); accumB(db_val);
    final clip = 5.0;
    final norm = math.sqrt(sqSum + 1e-12);
    final scale = norm > clip ? (clip / norm) : 1.0;
    if (scale != 1.0) {
      void scaleM(List<List<double>> G) { for (final r in G) { for (int j=0;j<r.length;j++) r[j]*=scale; } }
      void scaleB(List<double> g) { for (int j=0;j<g.length;j++) g[j]*=scale; }
      scaleM(dW1); scaleM(dW2); scaleM(dW_thr); scaleM(dW_turn); scaleM(dW_val);
      scaleB(db1); scaleB(db2); scaleB(db_thr); scaleB(db_turn); scaleB(db_val);
    }

    // ---- SGD update ----
    void sgdW(List<List<double>> W, List<List<double>> dW) {
      for (int i = 0; i < W.length; i++) {
        for (int j = 0; j < W[0].length; j++) W[i][j] -= lr * dW[i][j];
      }
    }
    void sgdB(List<double> b, List<double> db) {
      for (int i = 0; i < b.length; i++) b[i] -= lr * db[i];
    }

    sgdW(W_thr, dW_thr); sgdB(b_thr, db_thr);
    sgdW(W_turn, dW_turn); sgdB(b_turn, db_turn);
    sgdW(W_val, dW_val); sgdB(b_val, db_val);
    sgdW(W2, dW2); sgdB(b2, db2);
    sgdW(W1, dW1); sgdB(b1, db1);
  }
}

/// --------- Episode runner & trainer ----------
class EpisodeResult {
  final double totalCost;
  final int steps;
  final bool landed;
  EpisodeResult(this.totalCost, this.steps, this.landed);
}

class Trainer {
  final eng.GameEngine env;
  final FeatureExtractor fe;
  final PolicyNetwork policy;
  final math.Random rnd;

  final double dt;     // seconds per step
  final double gamma;  // discount

  // exploration knobs (train-time)
  double tempThr;
  double tempTurn;
  double epsilon;
  final double entropyBeta;

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
  }) : rnd = math.Random(seed);

  /// Run one episode.
  /// - If `greedy == true`, uses argmax/thresholded actions (deterministic).
  /// - Otherwise, samples with temps/epsilon (stochastic).
  /// - If engine score is reward (not cost), pass scoreIsReward=true.
  EpisodeResult runEpisode({
    bool train = true,
    double lr = 3e-4,
    bool greedy = false,
    bool scoreIsReward = false,
    double valueBeta = 0.5,     // critic loss weight
    double huberDelta = 1.0,    // Huber delta for value loss
  }) {
    final caches = <_Forward>[];
    final actions = <List<int>>[];
    final costs = <double>[]; // store COST per step (positive = bad)

    while (true) {
      final x = fe.extract(env);

      bool th, lf, rt;
      _Forward cache;

      if (greedy) {
        final res = policy.actGreedy(x);
        th = res.$1; lf = res.$2; rt = res.$3; cache = res.$5;
      } else {
        final res = policy.act(
          x,
          rnd,
          tempThr: tempThr,
          tempTurn: tempTurn,
          epsilon: epsilon,
        );
        th = res.$1; lf = res.$2; rt = res.$3; cache = res.$5;
      }

      final info = env.step(dt, et.ControlInput(thrust: th, left: lf, right: rt));

      // Interpret engine signal
      final s = info.scoreDelta; // could be reward OR cost depending on engine
      final stepCost = scoreIsReward ? -s : s; // ensure we store COST here

      caches.add(cache);
      actions.add([th ? 1 : 0, lf ? 1 : 0, rt ? 1 : 0]);
      costs.add(stepCost);

      if (info.terminal || caches.length > 4000) break;
    }

    // Discounted returns over reward = -cost
    final R = List<double>.filled(costs.length, 0.0);
    double running = 0.0;
    for (int t = costs.length - 1; t >= 0; t--) {
      final reward = -costs[t];
      running = reward + gamma * running;
      R[t] = running;
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
      );
    }

    final landed = env.status == et.GameStatus.landed;
    final totalCost = costs.fold(0.0, (a, b) => a + b);
    return EpisodeResult(totalCost, costs.length, landed);
  }
}
