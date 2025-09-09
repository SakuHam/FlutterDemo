// lib/ai/agent.dart
import 'dart:math' as math;
import '../engine/game_engine.dart' as eng;

/// -------- Feature Extraction --------
class FeatureExtractor {
  final int groundSamples;
  final double stridePx;
  FeatureExtractor({this.groundSamples = 5, this.stridePx = 40});

  List<double> extract(eng.GameEngine e) {
    final L = e.lander;
    final T = e.terrain;
    final cfg = e.cfg;

    double clamp01(double v) => v < 0 ? 0 : (v > 1 ? 1 : v);

    final px = clamp01(L.pos.x / cfg.worldW);
    final py = clamp01(L.pos.y / cfg.worldH);
    final vx = (L.vel.x / 200.0).clamp(-1.5, 1.5);
    final vy = (L.vel.y / 200.0).clamp(-1.5, 1.5);
    final ang = (L.angle / math.pi).clamp(-1.0, 1.0);
    final fuel = clamp01(L.fuel / cfg.t.maxFuel);

    final padCenter = clamp01(T.padCenter / cfg.worldW);
    final dxCenter = ((L.pos.x - T.padCenter) / cfg.worldW).clamp(-1.0, 1.0);

    final groundY = T.heightAt(L.pos.x);
    final dGround = ((groundY - L.pos.y) / cfg.worldH).clamp(-1.0, 1.0);
    final gyL = T.heightAt(math.max(0, L.pos.x - 20));
    final gyR = T.heightAt(math.min(cfg.worldW, L.pos.x + 20));
    final slope = (((gyR - gyL) / 40.0) / 0.5).clamp(-1.5, 1.5);

    final n = groundSamples;
    final half = (n - 1) ~/ 2;
    final List<double> samples = [];
    for (int i = -half; i <= half; i++) {
      final sx = (L.pos.x + i * stridePx).clamp(0.0, cfg.worldW);
      final sy = T.heightAt(sx);
      final rel = ((sy - L.pos.y) / cfg.worldH).clamp(-1.0, 1.0);
      samples.add(rel);
    }

    return [px, py, vx, vy, ang, fuel, padCenter, dxCenter, dGround, slope, ...samples];
  }

  int get inputSize => 10 + groundSamples;
}

/// -------- Tiny math helpers --------
List<double> vecAdd(List<double> a, List<double> b) {
  final out = List<double>.filled(a.length, 0);
  for (int i = 0; i < a.length; i++) out[i] = a[i] + b[i];
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
double safeLog(double p) => math.log(p.clamp(1e-6, 1 - 1e-6));

List<double> reluVec(List<double> v) => v.map(relu).toList();
List<double> sigmoidVec(List<double> v) => v.map(sigmoid).toList();

/// --------- Policy Network ----------
class PolicyNetwork {
  final int inputSize;
  final int h1;
  final int h2;
  final int outputs; // 3 -> [thrust, left, right]

  late List<List<double>> W1, W2, W3;
  late List<double> b1, b2, b3;
  final math.Random rnd;

  PolicyNetwork({
    required this.inputSize,
    this.h1 = 128,
    this.h2 = 128,
    this.outputs = 3,
    int seed = 1234,
  }) : rnd = math.Random(seed) {
    W1 = _init(h1, inputSize);
    W2 = _init(h2, h1);
    W3 = _init(outputs, h2);
    b1 = List<double>.filled(h1, 0);
    b2 = List<double>.filled(h2, 0);
    b3 = List<double>.filled(outputs, 0);

    // Nudge thrust prior upward so it explores firing more early on.
    // Sigmoid(0.6) ≈ 0.645
    b3[0] = 0.6;  // thrust head bias
  }

  List<List<double>> _init(int rows, int cols) {
    final limit = math.sqrt(6.0 / (rows + cols));
    return List.generate(rows,
            (_) => List<double>.generate(cols, (_) => (rnd.nextDouble() * 2 - 1) * limit));
  }
}

class _Forward {
  final List<double> x;
  final List<double> z1, h1;
  final List<double> z2, h2;
  final List<double> z3, p;
  _Forward(this.x, this.z1, this.h1, this.z2, this.h2, this.z3, this.p);
}

extension PolicyOps on PolicyNetwork {
  _Forward _forward(List<double> x) {
    assert(W1[0].length == x.length, 'W1 cols (${W1[0].length}) != x length (${x.length})');
    final z1 = vecAdd(matVec(W1, x), b1);
    final h1v = reluVec(z1);

    assert(W2[0].length == h1v.length, 'W2 cols != h1 length');
    final z2 = vecAdd(matVec(W2, h1v), b2);
    final h2v = reluVec(z2);

    assert(W3[0].length == h2v.length, 'W3 cols != h2 length');
    final z3 = vecAdd(matVec(W3, h2v), b3);
    final p = sigmoidVec(z3);
    return _Forward(x, z1, h1v, z2, h2v, z3, p);
  }

  (bool thrust, bool left, bool right, List<double> probs, _Forward cache)
  act(List<double> x, math.Random rnd) {
    final f = _forward(x);
    bool thrust = rnd.nextDouble() < f.p[0];
    bool left   = rnd.nextDouble() < f.p[1];
    bool right  = rnd.nextDouble() < f.p[2];
    if (left && right) {
      if (f.p[1] >= f.p[2]) right = false; else left = false;
    }
    return (thrust, left, right, f.p, f);
  }

  void updateFromEpisode({
    required List<_Forward> caches,
    required List<List<int>> actions,
    required List<double> returns_,
    double lr = 3e-4,
    double l2 = 1e-6,
    double entropyBeta = 0.01, // encourages exploration
  }) {
    // Normalize advantages
    final adv = List<double>.from(returns_);
    final mean = adv.reduce((a,b)=>a+b)/adv.length;
    double var0 = 0.0; for (final v in adv) { var0 += (v-mean)*(v-mean); }
    var0 /= adv.length;
    final std = math.sqrt(var0 + 1e-8);
    for (int i=0; i<adv.length; i++) adv[i] = (adv[i]-mean)/std;

    final dW1 = zeros(W1.length, W1[0].length);
    final dW2 = zeros(W2.length, W2[0].length);
    final dW3 = zeros(W3.length, W3[0].length);
    final db1 = List<double>.filled(b1.length, 0);
    final db2 = List<double>.filled(b2.length, 0);
    final db3 = List<double>.filled(b3.length, 0);

    for (int t = 0; t < caches.length; t++) {
      final f = caches[t];
      final a = actions[t];
      final A = adv[t];

      // Policy gradient at logits: (p - a) * A  (for Bernoulli heads)
      final dz3 = List<double>.generate(3, (k) => (f.p[k] - a[k]) * A);

      // --- Entropy bonus gradient (encourage probabilities near 0.5) ---
      // H = sum_k [ -p log p - (1-p) log(1-p) ]
      // dH/dz = (log(1-p) - log(p)) * p * (1-p)  (chain from dp/dz = p(1-p))
      for (int k = 0; k < 3; k++) {
        final p = f.p[k].clamp(1e-6, 1-1e-6);
        final dH_dz = (safeLog(1-p) - safeLog(p)) * p * (1-p);
        dz3[k] -= entropyBeta * dH_dz; // subtract because we *maximize* entropy
      }

      // W3, b3
      addInPlaceMat(dW3, outer(dz3, f.h2));
      addInPlaceVec(db3, dz3);

      // backprop
      final dh2 = List<double>.filled(f.h2.length, 0);
      for (int i = 0; i < W3.length; i++) {
        for (int j = 0; j < W3[0].length; j++) dh2[j] += W3[i][j] * dz3[i];
      }
      final dz2 = List<double>.generate(f.z2.length, (i) => dh2[i] * dRelu(f.z2[i]));

      addInPlaceMat(dW2, outer(dz2, f.h1));
      addInPlaceVec(db2, dz2);

      final dh1 = List<double>.filled(f.h1.length, 0);
      for (int i = 0; i < W2.length; i++) {
        for (int j = 0; j < W2[0].length; j++) dh1[j] += W2[i][j] * dz2[i];
      }
      final dz1 = List<double>.generate(f.z1.length, (i) => dh1[i] * dRelu(f.z1[i]));
      addInPlaceMat(dW1, outer(dz1, f.x));
      addInPlaceVec(db1, dz1);
    }

    // Weight decay
    void reg(List<List<double>> W) {
      for (final row in W) {
        for (int j = 0; j < row.length; j++) row[j] += l2 * row[j];
      }
    }
    reg(dW1); reg(dW2); reg(dW3);

    // SGD
    void sgd(List<List<double>> W, List<List<double>> dW, double lr) {
      for (int i = 0; i < W.length; i++) {
        for (int j = 0; j < W[0].length; j++) W[i][j] -= lr * dW[i][j];
      }
    }
    void sgdB(List<double> b, List<double> db, double lr) {
      for (int i = 0; i < b.length; i++) b[i] -= lr * db[i];
    }

    sgd(W3, dW3, lr); sgdB(b3, db3, lr);
    sgd(W2, dW2, lr); sgdB(b2, db2, lr);
    sgd(W1, dW1, lr); sgdB(b1, db1, lr);
  }
}

/// --------- Episode runner & trainer ----------
class EpisodeResult {
  final double totalCost; // 0 is best
  final int steps;
  final bool landed;
  EpisodeResult(this.totalCost, this.steps, this.landed);
}

class Trainer {
  final eng.GameEngine baseEngine; // keep a template for baseline cfg
  final FeatureExtractor fe;
  final PolicyNetwork policy;
  final math.Random rnd;

  final double dt;     // seconds per step
  final double gamma;  // discount

  int _episodeIdx = 0;

  Trainer({
    required eng.GameEngine engine,
    required this.fe,
    required this.policy,
    this.dt = 1/60.0,
    this.gamma = 0.995,
    int seed = 99,
  })  : baseEngine = engine,
        rnd = math.Random(seed);

  EpisodeResult runEpisode({bool train = true, double lr = 3e-4}) {
    // ---- Build per-episode config (curriculum actually applied) ----
    final e = _episodeIdx;
    final easyPhase = e < 400;
    final cfg0 = baseEngine.cfg;

    final epCfg = eng.EngineConfig(
      worldW: cfg0.worldW,
      worldH: cfg0.worldH,
      t: cfg0.t,
      seed: rnd.nextInt(1 << 30),
      stepScale: cfg0.stepScale,
      padWidthFactor: easyPhase ? 1.6 : 1.0,
      landingSpeedMax: easyPhase ? 80.0 : cfg0.landingSpeedMax,
      landingAngleMaxRad: easyPhase ? 0.45 : cfg0.landingAngleMaxRad,
      // shaping weights unchanged; you can also anneal these if you like
      livingCost: cfg0.livingCost,
      effortCost: cfg0.effortCost,
      wDx: cfg0.wDx,
      wVyDown: cfg0.wVyDown,
      wVx: cfg0.wVx,
      wAngleDeg: cfg0.wAngleDeg,
    );

    // Fresh environment with those thresholds
    final env = eng.GameEngine(epCfg);

    final caches = <_Forward>[];
    final actions = <List<int>>[];
    final rewards = <double>[]; // reward = -cost (+ bonuses)
    double totalCost = 0.0;

    // ε-greedy exploration that anneals
    const epsStart = 0.30, epsMin = 0.05, tau = 300.0;
    final eps = math.max(epsMin, epsStart * math.exp(-_episodeIdx / tau));

    while (true) {
      final x = fe.extract(env);
      var (th, lf, rt, probs, cache) = policy.act(x, rnd);

      if (rnd.nextDouble() < eps) {
        th = rnd.nextBool();
        final dir = rnd.nextInt(3); // 0: none, 1: left, 2: right
        lf = dir == 1;
        rt = dir == 2;
        if (lf && rt) rt = false;
      }

      final info = env.step(dt, eng.ControlInput(thrust: th, left: lf, right: rt));

      caches.add(cache);
      actions.add([th ? 1 : 0, lf ? 1 : 0, rt ? 1 : 0]);

      final costStep = info.costDelta; // >= 0
      totalCost += costStep;

      // entropy bonus (average over heads)
      double H = 0.0;
      for (final p in probs) {
        final pc = p.clamp(1e-6, 1 - 1e-6);
        H += -(pc * safeLog(pc) + (1 - pc) * safeLog(1 - pc));
      }
      H /= 3.0;

      // Small success bonus only on terminal landing step (helps discovery).
      final successBonus = (info.terminal && info.onPad && env.status == eng.GameStatus.landed) ? 50.0 : 0.0;

      final reward = -costStep + 0.01 * H + successBonus;
      rewards.add(reward);

      if (info.terminal || caches.length > 4000) break;
    }

    // discounted returns
    final G = List<double>.filled(rewards.length, 0.0);
    double running = 0.0;
    for (int t = rewards.length - 1; t >= 0; t--) {
      running = rewards[t] + gamma * running;
      G[t] = running;
    }

    if (train && caches.isNotEmpty) {
      policy.updateFromEpisode(
        caches: caches,
        actions: actions,
        returns_: G,
        lr: lr,
        entropyBeta: 0.01,
      );
    }

    _episodeIdx++;
    final landed = env.status == eng.GameStatus.landed;
    return EpisodeResult(totalCost, rewards.length, landed);
  }
}
