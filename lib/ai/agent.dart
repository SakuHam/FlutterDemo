// lib/ai/agent.dart
import 'dart:math' as math;
import '../engine/game_engine.dart' as eng;

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
    // center index = (n-1)/2.0 gives symmetric offsets for even/odd n
    final center = (n - 1) / 2.0;
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

/// --------- Policy Network (2 hidden, split heads) ----------
class PolicyNetwork {
  final int inputSize;
  final int h1;
  final int h2;

  // Trunk
  late List<List<double>> W1, W2;
  late List<double> b1, b2;

  // Heads
  late List<List<double>> W_thr; // (1, h2)
  late List<double> b_thr;       // (1)
  late List<List<double>> W_turn; // (3, h2)
  late List<double> b_turn;       // (3)

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
  _Forward(this.x, this.z1, this.h1, this.z2, this.h2,
      this.thrLogit, this.thrP, this.turnLogits, this.turnP);
}

extension PolicyOps on PolicyNetwork {
  _Forward _forward(List<double> x) {
    final z1 = vecAdd(matVec(W1, x), b1);
    final h1v = reluVec(z1);
    final z2 = vecAdd(matVec(W2, h1v), b2);
    final h2v = reluVec(z2);

    // heads
    final thrLogit = matVec(W_thr, h2v)[0] + b_thr[0];
    final thrP = sigmoid(thrLogit);

    final turnLogits = matVec(W_turn, h2v);
    for (int i = 0; i < 3; i++) turnLogits[i] += b_turn[i];
    final turnP = softmax(turnLogits);

    return _Forward(x, z1, h1v, z2, h2v, thrLogit, thrP, turnLogits, turnP);
  }

  /// Tempered + epsilon-greedy action sampling
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

  /// One REINFORCE step over an episode, with entropy bonus.
  void updateFromEpisode({
    required List<_Forward> caches,
    required List<List<int>> actions, // [th, left, right]
    required List<double> returns_,
    double lr = 3e-4,
    double l2 = 1e-6,
    double entropyBeta = 0.0,
  }) {
    // Normalize advantages
    final adv = List<double>.from(returns_);
    final mean = adv.reduce((a, b) => a + b) / adv.length;
    double var0 = 0.0; for (final v in adv) { var0 += (v - mean) * (v - mean); }
    var0 /= adv.length;
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

    for (int t = 0; t < caches.length; t++) {
      final f = caches[t];
      final a = actions[t];      // [th, left, right]
      final A = adv[t];

      // Targets
      final a_thr = a[0].toDouble();
      final a_turn = <double>[1 - a[1] - a[2] + 0.0, a[1] + 0.0, a[2] + 0.0]; // [none,left,right]

      // Policy-grad logits grads: (p - a) * A
      double dz_thr = (f.thrP - a_thr) * A;
      final dz_turn = List<double>.generate(3, (k) => (f.turnP[k] - a_turn[k]) * A);

      // ---- Entropy bonus: L = CE - beta * H  => dz += -(beta) * dH/dz ----
      if (entropyBeta > 0.0) {
        // Bernoulli
        final p = f.thrP.clamp(1e-6, 1 - 1e-6);
        final dH_dz_thr = math.log((1 - p) / p) * p * (1 - p);
        dz_thr += -entropyBeta * dH_dz_thr;

        // Categorical
        final p3 = f.turnP.map((x) => x.clamp(1e-8, 1.0)).toList();
        final g = List<double>.generate(3, (i) => -(math.log(p3[i]) + 1.0)); // -∂H/∂p
        double s = 0.0; for (int i = 0; i < 3; i++) s += p3[i] * g[i];
        for (int i = 0; i < 3; i++) {
          final dH_dz_i = p3[i] * g[i] - p3[i] * s; // (diag(p)-pp^T)g
          dz_turn[i] += -entropyBeta * dH_dz_i;
        }
      }

      // Head grads
      addInPlaceMat(dW_thr, outer([dz_thr], f.h2)); db_thr[0] += dz_thr;
      addInPlaceMat(dW_turn, outer(dz_turn, f.h2)); addInPlaceVec(db_turn, dz_turn);

      // Backprop to h2
      final dh2 = List<double>.filled(f.h2.length, 0);
      for (int j = 0; j < W_thr[0].length; j++) dh2[j] += W_thr[0][j] * dz_thr;
      for (int i = 0; i < 3; i++) {
        final row = W_turn[i];
        for (int j = 0; j < row.length; j++) dh2[j] += row[j] * dz_turn[i];
      }

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

    // L2 on grads (small)
    void reg(List<List<double>> W) {
      for (final row in W) {
        for (int j = 0; j < row.length; j++) row[j] += l2 * row[j];
      }
    }
    reg(dW1); reg(dW2); reg(dW_thr); reg(dW_turn);

    // SGD
    void sgdW(List<List<double>> W, List<List<double>> dW) {
      for (int i = 0; i < W.length; i++) {
        for (int j = 0; j < W[0].length; j++) W[i][j] -= lr * dW[i][j];
      }
    }
    void sgdB(List<double> b, List<double> db) { for (int i = 0; i < b.length; i++) b[i] -= lr * db[i]; }

    sgdW(W_thr, dW_thr); sgdB(b_thr, db_thr);
    sgdW(W_turn, dW_turn); sgdB(b_turn, db_turn);
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

  // exploration knobs
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

  EpisodeResult runEpisode({bool train = true, double lr = 3e-4}) {
    final caches = <_Forward>[];
    final actions = <List<int>>[];
    final costs = <double>[]; // per-step cost from engine

    while (true) {
      final x = fe.extract(env);
      final (th, lf, rt, _, cache) = policy.act(
        x,
        rnd,
        tempThr: tempThr,
        tempTurn: tempTurn,
        epsilon: epsilon,
      );

      final info = env.step(dt, eng.ControlInput(thrust: th, left: lf, right: rt));

      caches.add(cache);
      actions.add([th ? 1 : 0, lf ? 1 : 0, rt ? 1 : 0]);
      costs.add(info.scoreDelta);

      if (info.terminal || caches.length > 4000) break;
    }

    // Discounted returns over (-cost)
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
      );
    }

    final landed = env.status == eng.GameStatus.landed;
    final totalCost = costs.fold(0.0, (a, b) => a + b);
    return EpisodeResult(totalCost, costs.length, landed);
  }
}
