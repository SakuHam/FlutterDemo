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
    final vx = (L.vel.x / 200.0).clamp(-1.2, 1.2);
    final vy = (L.vel.y / 200.0).clamp(-1.2, 1.2);

    // Use sin/cos of angle instead of raw angle
    final sinA = math.sin(L.angle);
    final cosA = math.cos(L.angle);

    final fuel = clamp01(L.fuel / cfg.t.maxFuel);
    final padCenter = clamp01(T.padCenter / cfg.worldW);
    final dxCenter = ((L.pos.x - T.padCenter) / cfg.worldW).clamp(-1.0, 1.0);

    final groundY = T.heightAt(L.pos.x);
    final dGround = ((groundY - L.pos.y) / cfg.worldH).clamp(-1.0, 1.0);
    final gyL = T.heightAt(math.max(0, L.pos.x - 20));
    final gyR = T.heightAt(math.min(cfg.worldW, L.pos.x + 20));
    final slope = (((gyR - gyL) / 40.0) / 0.5).clamp(-1.5, 1.5);

    final count = groundSamples;
    final half = (count - 1) ~/ 2;
    final List<double> samples = [];
    for (int i = -half; i <= half; i++) {
      final sx = (L.pos.x + i * stridePx).clamp(0.0, cfg.worldW);
      final sy = T.heightAt(sx);
      final rel = ((sy - L.pos.y) / cfg.worldH).clamp(-1.0, 1.0);
      samples.add(rel);
    }

    // Order: [px,py,vx,vy,sinA,cosA,fuel,padCenter,dxCenter,dGround,slope,samples...]
    return [px, py, vx, vy, sinA, cosA, fuel, padCenter, dxCenter, dGround, slope, ...samples];
  }

  int get inputSize => 11 + groundSamples; // angle -> (sin,cos)
}

/// -------- Tiny math helpers --------
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
List<double> vecAdd(List<double> a, List<double> b) {
  final out = List<double>.filled(a.length, 0);
  for (int i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}
List<List<double>> zeros(int m, int n) => List.generate(m, (_) => List<double>.filled(n, 0));
List<List<double>> outer(List<double> a, List<double> b) {
  final out = zeros(a.length, b.length);
  for (int i = 0; i < a.length; i++) {
    final ai = a[i];
    for (int j = 0; j < b.length; j++) out[i][j] = ai * b[j];
  }
  return out;
}
void addInPlaceVec(List<double> a, List<double> b) { for (int i = 0; i < a.length; i++) a[i] += b[i]; }
void addInPlaceMat(List<List<double>> A, List<List<double>> B) {
  for (int i = 0; i < A.length; i++) {
    final Ai = A[i], Bi = B[i];
    for (int j = 0; j < Ai.length; j++) Ai[j] += Bi[j];
  }
}

double relu(double x) => x > 0 ? 0.0 + x : 0.0;
List<double> reluVec(List<double> v) => v.map(relu).toList();
double safeLog(double p) => math.log(p.clamp(1e-8, 1 - 1e-8));

List<double> softmax(List<double> z) {
  final maxZ = z.reduce(math.max);
  double sum = 0.0;
  final exps = List<double>.filled(z.length, 0.0);
  for (int i = 0; i < z.length; i++) { final e = math.exp(z[i] - maxZ); exps[i] = e; sum += e; }
  for (int i = 0; i < z.length; i++) exps[i] /= sum;
  return exps;
}

/// --------- Policy Network: shared trunk + two categorical heads + value --------
class PolicyNetwork {
  final int inputSize;
  final int h1;
  final int h2;

  late List<List<double>> W1, W2;
  late List<double> b1, b2;

  // Heads
  late List<List<double>> W_thr; // 3 x h2  (throttle: off/low/high)
  late List<double> b_thr;
  late List<List<double>> W_turn; // 3 x h2  (turn: none/left/right)
  late List<double> b_turn;
  late List<double> W_val; // 1 x h2 (value)
  late double b_val;

  final math.Random rnd;

  PolicyNetwork({
    required this.inputSize,
    this.h1 = 128,
    this.h2 = 128,
    int seed = 1234,
  }) : rnd = math.Random(seed) {
    W1 = _init(h1, inputSize); b1 = List<double>.filled(h1, 0);
    W2 = _init(h2, h1);        b2 = List<double>.filled(h2, 0);

    W_thr = _init(3, h2); b_thr = List<double>.filled(3, 0);
    W_turn = _init(3, h2); b_turn = List<double>.filled(3, 0);
    W_val = List<double>.generate(h2, (_) => (rnd.nextDouble() * 2 - 1) * math.sqrt(2.0 / (h2 + 1)));
    b_val = 0.0;

    // Slight prior toward using thrust
    b_thr[0] = 0.0; // off
    b_thr[1] = 0.4; // low
    b_thr[2] = 0.2; // high
  }

  List<List<double>> _init(int rows, int cols) {
    final limit = math.sqrt(6.0 / (rows + cols));
    return List.generate(rows, (_) => List<double>.generate(cols, (_) => (rnd.nextDouble() * 2 - 1) * limit));
  }
}

class _Forward {
  final List<double> x;
  final List<double> z1, h1;
  final List<double> z2, h2;
  final List<double> thrLogits, thrP;   // 3
  final List<double> turnLogits, turnP; // 3
  final double value;
  _Forward(this.x, this.z1, this.h1, this.z2, this.h2, this.thrLogits, this.thrP, this.turnLogits, this.turnP, this.value);
}

extension PolicyOps on PolicyNetwork {
  _Forward _forward(List<double> x) {
    final z1 = vecAdd(matVec(W1, x), b1);
    final h1v = reluVec(z1);
    final z2 = vecAdd(matVec(W2, h1v), b2);
    final h2v = reluVec(z2);

    // --- Throttle head (3-way softmax) ---
    final thrLogits = matVec(W_thr, h2v);
    for (int i = 0; i < thrLogits.length; i++) thrLogits[i] += b_thr[i];
    final thrP = softmax(thrLogits);

    // --- Turn head (3-way softmax) ---
    final turnLogits = matVec(W_turn, h2v);
    for (int i = 0; i < turnLogits.length; i++) turnLogits[i] += b_turn[i];
    final turnP = softmax(turnLogits);

    // --- Value head ---
    double val = 0.0;
    for (int i = 0; i < h2v.length; i++) val += W_val[i] * h2v[i];
    val += b_val;

    return _Forward(x, z1, h1v, z2, h2v, thrLogits, thrP, turnLogits, turnP, val);
  }

  (int thrClass, int turnClass, double power, List<double> thrP, List<double> turnP, _Forward cache)
  act(List<double> x, math.Random rnd) {
    final f = _forward(x);

    int sample(List<double> p) {
      final r = rnd.nextDouble();
      double c = 0.0;
      for (int i = 0; i < p.length; i++) { c += p[i]; if (r <= c) return i; }
      return p.length - 1;
    }

    final thrC = sample(f.thrP);
    final turnC = sample(f.turnP);
    final double power = (thrC == 0) ? 0.0 : (thrC == 1 ? 0.7 : 1.0);
    return (thrC, turnC, power, f.thrP, f.turnP, f);
  }
}

/// --------- PPO Trainer ----------
class EpisodeResult {
  final double totalCost; // 0 is best
  final int steps;
  final bool landed;
  EpisodeResult(this.totalCost, this.steps, this.landed);
}

class PPOTrainer {
  final eng.GameEngine baseEngine; // template cfg
  final FeatureExtractor fe;
  final PolicyNetwork policy;
  final math.Random rnd;
  final double dt;
  final double gamma;

  // PPO hyperparams
  final double clipEps;
  final double entropyBeta;
  final double valueCoef;
  final double l2;
  final double lr;
  final int rolloutSteps;   // collect this many env steps per update
  final int ppoEpochs;      // how many epochs per batch
  final int miniBatch;      // minibatch size
  final int maxStepsPerEp;  // cap episode length to reduce variance

  PPOTrainer({
    required eng.GameEngine engine,
    required this.fe,
    required this.policy,
    this.dt = 1 / 60.0,
    this.gamma = 0.995,
    this.clipEps = 0.2,
    this.entropyBeta = 0.001,
    this.valueCoef = 0.5,
    this.l2 = 1e-6,
    this.lr = 1e-4,
    this.rolloutSteps = 4096,
    this.ppoEpochs = 4,
    this.miniBatch = 512,
    this.maxStepsPerEp = 900,
    int seed = 99,
  })  : baseEngine = engine,
        rnd = math.Random(seed);

  // Storage for a rollout
  final List<List<double>> _states = [];
  final List<int> _actThr = [];
  final List<int> _actTurn = [];
  final List<double> _pow = [];
  final List<double> _logpOld = [];
  final List<double> _values = [];
  final List<double> _rewards = [];
  final List<bool> _dones = [];

  void _clearBuffer() {
    _states.clear();
    _actThr.clear();
    _actTurn.clear();
    _pow.clear();
    _logpOld.clear();
    _values.clear();
    _rewards.clear();
    _dones.clear();
  }

  double _logProbCat(List<double> p, int a) => safeLog(p[a]);

  EpisodeResult trainStep(int episodeIdx, {bool logOneEp = false}) {
    _clearBuffer();

    // Build curriculum cfg per episode start
    final bool easy = episodeIdx < 400;
    final c0 = baseEngine.cfg;
    final env = eng.GameEngine(eng.EngineConfig(
      worldW: c0.worldW, worldH: c0.worldH, t: c0.t,
      seed: rnd.nextInt(1<<30), stepScale: c0.stepScale,
      padWidthFactor: easy ? 1.6 : 1.0,
      landingSpeedMax: easy ? 80.0 : c0.landingSpeedMax,
      landingAngleMaxRad: easy ? 0.45 : c0.landingAngleMaxRad,
      livingCost: c0.livingCost, effortCost: c0.effortCost,
      wDx: c0.wDx, wDy: c0.wDy, wVyDown: c0.wVyDown, wVx: c0.wVx, wAngleDeg: c0.wAngleDeg,
    ));

    // Rollout across possibly multiple episodes until rolloutSteps gathered
    double lastEpCost = 0.0;
    int lastEpSteps = 0;
    bool lastEpLanded = false;
    int collected = 0;

    while (collected < rolloutSteps) {
      // Start a fresh episode in env
      env.reset(seed: rnd.nextInt(1<<30));
      double epCost = 0.0;
      int steps = 0;
      bool landed = false;

      while (true) {
        final s = fe.extract(env);
        final action = policy.act(s, rnd);
        final int thrC = action.$1;
        final int turnC = action.$2;
        final double power = action.$3;
        final thrP = action.$4;
        final turnP = action.$5;
        final cache = action.$6; // contains value via forward pass
        final info = env.step(
          dt,
          eng.ControlInput(thrust: power > 0.0, left: turnC == 1, right: turnC == 2),
        );

        // Store transition
        _states.add(s);
        _actThr.add(thrC);
        _actTurn.add(turnC);
        _pow.add(power);
        final double logp = _logProbCat(thrP, thrC) + _logProbCat(turnP, turnC);
        _logpOld.add(logp);
        _values.add(cache.value);
        _rewards.add(-info.costDelta); // reward = -cost (no success bonus)
        _dones.add(info.terminal);

        epCost += info.costDelta;
        steps++;
        collected++;

        if (info.terminal || steps >= maxStepsPerEp || collected >= rolloutSteps) {
          landed = info.terminal && info.onPad && env.status == eng.GameStatus.landed;
          break;
        }
      }

      lastEpCost = epCost;
      lastEpSteps = steps;
      lastEpLanded = landed;
    }

    // Compute advantages & returns (GAE)
    final int N = _rewards.length;
    final adv = List<double>.filled(N, 0.0);
    final ret = List<double>.filled(N, 0.0);

    double gae = 0.0;
    const double lam = 0.95;
    for (int t = N - 1; t >= 0; t--) {
      final double v = _values[t];
      final double vNext = (t == N - 1 || _dones[t]) ? 0.0 : _values[t + 1];
      final double delta = _rewards[t] + gamma * vNext - v;
      gae = delta + gamma * lam * (t == N - 1 || _dones[t] ? 0.0 : gae);
      adv[t] = gae;
      ret[t] = v + adv[t];
    }
    // Normalize advantages
    final double meanA = adv.reduce((a, b) => a + b) / adv.length;
    double varA = 0.0; for (final a in adv) { final d = a - meanA; varA += d * d; }
    final double stdA = math.sqrt(varA / adv.length + 1e-8);
    for (int i = 0; i < adv.length; i++) adv[i] = (adv[i] - meanA) / stdA;

    // PPO update
    _ppoUpdate(states: _states, thrA: _actThr, turnA: _actTurn, logpOld: _logpOld, ret: ret, adv: adv);

    return EpisodeResult(lastEpCost, lastEpSteps, lastEpLanded);
  }

  void _ppoUpdate({
    required List<List<double>> states,
    required List<int> thrA,
    required List<int> turnA,
    required List<double> logpOld,
    required List<double> ret,
    required List<double> adv,
  }) {
    final idx = List<int>.generate(states.length, (i) => i);

    for (int epoch = 0; epoch < ppoEpochs; epoch++) {
      // shuffle indices
      for (int i = idx.length - 1; i > 0; i--) {
        final j = rnd.nextInt(i + 1);
        final tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
      }

      for (int start = 0; start < idx.length; start += miniBatch) {
        final end = math.min(start + miniBatch, idx.length);
        // grads accumulators
        final dW1 = zeros(policy.W1.length, policy.W1[0].length);
        final dW2 = zeros(policy.W2.length, policy.W2[0].length);
        final db1 = List<double>.filled(policy.b1.length, 0);
        final db2 = List<double>.filled(policy.b2.length, 0);

        final dW_thr = zeros(policy.W_thr.length, policy.W_thr[0].length);
        final db_thr = List<double>.filled(policy.b_thr.length, 0);
        final dW_turn = zeros(policy.W_turn.length, policy.W_turn[0].length);
        final db_turn = List<double>.filled(policy.b_turn.length, 0);

        final dW_val = List<double>.filled(policy.W_val.length, 0);
        double db_val = 0.0;

        for (int k = start; k < end; k++) {
          final i = idx[k];
          final s = states[i];
          final f = policy._forward(s);

// --- log prob new (sum of both heads) ---
          final double logpNew = safeLog(f.thrP[thrA[i]]) + safeLog(f.turnP[turnA[i]]);
          final double ratio = math.exp((logpNew - logpOld[i]).clamp(-20.0, 20.0)); // r_t
          final double advI = adv[i];

// PPO clipping rule for gradient:
// If adv > 0: use unclipped grad only when r < 1+eps (else 0).
// If adv < 0: use unclipped grad only when r > 1-eps (else 0).
          bool useGrad;
          if (advI >= 0) {
            useGrad = ratio < (1.0 + clipEps);
          } else {
            useGrad = ratio > (1.0 - clipEps);
          }

          // grad of log p(a|s) wrt logits for softmax is (p - onehot(a))
          final dz_thr = List<double>.from(f.thrP);
          dz_thr[thrA[i]] -= 1.0;
          final dz_turn = List<double>.from(f.turnP);
          dz_turn[turnA[i]] -= 1.0;

          // scale by -adv * r when allowed, else zero (since clipped branch is constant)
          final double scalePG = useGrad ? (-advI * ratio) : 0.0;
          for (int t = 0; t < dz_thr.length; t++) dz_thr[t] *= scalePG;
          for (int t = 0; t < dz_turn.length; t++) dz_turn[t] *= scalePG;

          // --- Cheap entropy regularization (fast) ---
          if (entropyBeta > 0.0) {
            const double u = 1.0 / 3.0; // uniform target
            for (int t = 0; t < f.thrP.length; t++) dz_thr[t] -= entropyBeta * (f.thrP[t] - u);
            for (int t = 0; t < f.turnP.length; t++) dz_turn[t] -= entropyBeta * (f.turnP[t] - u);
          }

          // --- Cheap entropy regularization: push probs toward uniform ---
          // Gradient of KL(p || uniform) wrt logits is proportional to (p - u).
          // We *subtract* beta*(p - u) to encourage higher entropy.
          if (entropyBeta > 0.0) {
            const double u = 1.0 / 3.0;
            for (int t = 0; t < f.thrP.length; t++) {
              dz_thr[t] -= entropyBeta * (f.thrP[t] - u);
            }
            for (int t = 0; t < f.turnP.length; t++) {
              dz_turn[t] -= entropyBeta * (f.turnP[t] - u);
            }
          }
          // value loss grad: 0.5*(V - ret)^2
          final double dv = (f.value - ret[i]) * valueCoef;

          // Backprop heads -> trunk
          addInPlaceMat(dW_thr, outer(dz_thr, f.h2)); addInPlaceVec(db_thr, dz_thr);
          addInPlaceMat(dW_turn, outer(dz_turn, f.h2)); addInPlaceVec(db_turn, dz_turn);

          final dh2 = List<double>.filled(f.h2.length, 0.0);
          for (int a = 0; a < policy.W_thr.length; a++) {
            for (int b = 0; b < policy.W_thr[0].length; b++) dh2[b] += policy.W_thr[a][b] * dz_thr[a];
          }
          for (int a = 0; a < policy.W_turn.length; a++) {
            for (int b = 0; b < policy.W_turn[0].length; b++) dh2[b] += policy.W_turn[a][b] * dz_turn[a];
          }
          for (int j = 0; j < policy.W_val.length; j++) {
            dW_val[j] += dv * f.h2[j];
            dh2[j] += policy.W_val[j] * dv;
          }
          db_val += dv;

          final dz2 = List<double>.generate(f.z2.length, (j) => f.z2[j] > 0 ? dh2[j] : 0.0);
          addInPlaceMat(dW2, outer(dz2, f.h1)); addInPlaceVec(db2, dz2);

          final dh1 = List<double>.filled(f.h1.length, 0.0);
          for (int a = 0; a < policy.W2.length; a++) {
            for (int b = 0; b < policy.W2[0].length; b++) dh1[b] += policy.W2[a][b] * dz2[a];
          }
          final dz1 = List<double>.generate(f.z1.length, (j) => f.z1[j] > 0 ? dh1[j] : 0.0);
          addInPlaceMat(dW1, outer(dz1, f.x)); addInPlaceVec(db1, dz1);
        }

        // L2 regularization (add to grads)
        void addL2ToMatGrad(List<List<double>> dW, List<List<double>> W, double coef) {
          for (int i = 0; i < dW.length; i++) {
            for (int j = 0; j < dW[0].length; j++) dW[i][j] += coef * W[i][j];
          }
        }
        void addL2ToVecGrad(List<double> dW, List<double> W, double coef) {
          for (int j = 0; j < dW.length; j++) dW[j] += coef * W[j];
        }
        addL2ToMatGrad(dW1, policy.W1, l2);
        addL2ToMatGrad(dW2, policy.W2, l2);
        addL2ToMatGrad(dW_thr, policy.W_thr, l2);
        addL2ToMatGrad(dW_turn, policy.W_turn, l2);
        addL2ToVecGrad(dW_val, policy.W_val, l2);

        // Gradient clip by global norm
        double sqSumMat(List<List<double>> M) { double s = 0.0; for (final r in M) { for (final v in r) s += v*v; } return s; }
        double sqSumVec(List<double> v) { double s = 0.0; for (final x in v) s += x*x; return s; }
        const double clip = 1.0;
        double gn2 = 0.0;
        gn2 += sqSumMat(dW1) + sqSumMat(dW2) + sqSumMat(dW_thr) + sqSumMat(dW_turn);
        gn2 += sqSumVec(db1) + sqSumVec(db2) + sqSumVec(db_thr) + sqSumVec(db_turn);
        gn2 += sqSumVec(dW_val) + (db_val * db_val);
        final double gnorm = math.sqrt(gn2 + 1e-12);
        final double scale = gnorm > clip ? clip / gnorm : 1.0;
        if (scale < 1.0) {
          void scaleMat(List<List<double>> M) { for (final r in M) { for (int j = 0; j < r.length; j++) r[j] *= scale; } }
          void scaleVec(List<double> v) { for (int j = 0; j < v.length; j++) v[j] *= scale; }
          scaleMat(dW1); scaleMat(dW2); scaleMat(dW_thr); scaleMat(dW_turn);
          scaleVec(db1); scaleVec(db2); scaleVec(db_thr); scaleVec(db_turn);
          scaleVec(dW_val); db_val *= scale;
        }

        // SGD step
        void sgd(List<List<double>> W, List<List<double>> dW) {
          for (int i = 0; i < W.length; i++) {
            for (int j = 0; j < W[0].length; j++) W[i][j] -= lr * dW[i][j];
          }
        }
        void sgdB(List<double> b, List<double> db) { for (int i = 0; i < b.length; i++) b[i] -= lr * db[i]; }

        sgd(policy.W_thr, dW_thr); sgdB(policy.b_thr, db_thr);
        sgd(policy.W_turn, dW_turn); sgdB(policy.b_turn, db_turn);
        for (int j = 0; j < policy.W_val.length; j++) policy.W_val[j] -= lr * dW_val[j];
        policy.b_val -= lr * db_val;

        sgd(policy.W2, dW2); sgdB(policy.b2, db2);
        sgd(policy.W1, dW1); sgdB(policy.b1, db1);
      }
    }
  }
}
