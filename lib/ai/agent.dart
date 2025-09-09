// lib/ai/agent.dart
import 'dart:math' as math;
import '../engine/game_engine.dart' as eng;

/// -------- Feature Extraction (state -> input vector) --------
class FeatureExtractor {
  final int groundSamples; // e.g., 5 samples around x with stridePx
  final double stridePx;   // e.g., 40 px
  FeatureExtractor({this.groundSamples = 3, this.stridePx = 48});

  List<double> extract(eng.GameEngine e) {
    final L = e.lander;
    final T = e.terrain;
    final cfg = e.cfg;

    final px = L.pos.x / cfg.worldW;              // 0..1
    final py = L.pos.y / cfg.worldH;              // 0..1
    final vx = (L.vel.x / 200.0).clamp(-2.0, 2.0);
    final vy = (L.vel.y / 200.0).clamp(-2.0, 2.0);
    final sinA = math.sin(L.angle);
    final cosA = math.cos(L.angle);
    final fuel = (L.fuel / cfg.t.maxFuel).clamp(0.0, 1.0);

    final padCenter = T.padCenter / cfg.worldW;   // 0..1
    final dxCenter = ((L.pos.x - T.padCenter) / cfg.worldW).clamp(-1.0, 1.0);

    // Distance to ground & slope
    final groundY = T.heightAt(L.pos.x);
    final dGround = ((groundY - L.pos.y) / cfg.worldH).clamp(-1.0, 1.0);
    final gyL = T.heightAt(math.max(0, L.pos.x - 20));
    final gyR = T.heightAt(math.min(cfg.worldW, L.pos.x + 20));
    final slope = (((gyR - gyL) / 40.0) / 0.5).clamp(-2.0, 2.0);

    // Local ground samples (relative height around lander)
    final int n = groundSamples;
    final half = (n - 1) ~/ 2;
    final List<double> samples = [];
    for (int i = -half; i <= half; i++) {
      final sx = (L.pos.x + i * stridePx).clamp(0.0, cfg.worldW);
      final sy = T.heightAt(sx);
      final rel = ((sy - L.pos.y) / cfg.worldH).clamp(-1.0, 1.0);
      samples.add(rel);
    }

    return [
      px, py, vx, vy, sinA, cosA, fuel,
      padCenter, dxCenter, dGround, slope,
      ...samples,
    ];
  }

  int get inputSize => 11 + groundSamples; // updated for sin/cos
}

/// --------- Tiny Math helpers ----------
List<double> vecAdd(List<double> a, List<double> b) {
  final n = a.length; final out = List<double>.filled(n, 0);
  for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
  return out;
}
List<double> matVec(List<List<double>> W, List<double> x) {
  final m = W.length, n = W[0].length;
  final out = List<double>.filled(m, 0);
  for (int i = 0; i < m; i++) {
    double s = 0.0;
    final Wi = W[i];
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
double relu(double x) => x > 0 ? x : 0.0;
double dRelu(double x) => x > 0 ? 1.0 : 0.0;
double sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));
List<double> reluVec(List<double> v) => v.map(relu).toList();

List<double> softmax(List<double> z) {
  final m = z.reduce((a,b)=>a>b?a:b);
  double sum = 0.0;
  final e = List<double>.filled(z.length, 0);
  for (int i=0;i<z.length;i++) { e[i] = math.exp(z[i]-m); sum += e[i]; }
  for (int i=0;i<z.length;i++) e[i] /= (sum + 1e-12);
  return e;
}

/// --------- Policy Network (2 hidden layers, thrust Bernoulli + turn 3-class) ----------
class PolicyNetwork {
  final int inputSize;
  final int h1;
  final int h2;

  // shared trunk
  late List<List<double>> W1, W2;
  late List<double> b1, b2;

  // heads
  late List<List<double>> W_thr;   // [1 x h2]
  late List<double> b_thr;         // [1]
  late List<List<double>> W_turn;  // [3 x h2]
  late List<double> b_turn;        // [3]

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

class _Forward {
  final List<double> x;
  final List<double> z1, h1;
  final List<double> z2, h2;
  final List<double> thrLogits, thrP;   // 1-d
  final List<double> turnLogits, turnP; // 3-d
  _Forward(this.x, this.z1, this.h1, this.z2, this.h2, this.thrLogits, this.thrP, this.turnLogits, this.turnP);
}

extension PolicyOps on PolicyNetwork {
  _Forward _forward(List<double> x, {double angle = 0.0, double dx = 0.0}) {
    // trunk
    final z1 = vecAdd(matVec(W1, x), b1);
    final h1v = reluVec(z1);
    final z2 = vecAdd(matVec(W2, h1v), b2);
    final h2v = reluVec(z2);

    // thrust logits (Bernoulli)
    final thrLogits = List<double>.from(matVec(W_thr, h2v));
    thrLogits[0] += b_thr[0];
    final thrP = [sigmoid(thrLogits[0])];

    // turn logits (3-class)
    final turnLogits = vecAdd(matVec(W_turn, h2v), b_turn);

    // Priors to encourage sensible exploration (optional but helpful)
    // 1) steer toward pad horizontally based on dx sign
    final kDx = 0.4;
    turnLogits[1] += (dx > 0 ? kDx : -kDx) * dx.abs(); // left if pad is left
    turnLogits[2] += (dx < 0 ? kDx : -kDx) * dx.abs(); // right if pad is right
    // 2) attitude prior: if tilted right (angle>0), nudge LEFT; vice versa
    final sign = angle >= 0 ? 1.0 : -1.0;
    final angN = (angle.abs() / (math.pi / 2)).clamp(0.0, 1.0);
    const kAng = 0.6;
    turnLogits[1] += (sign > 0 ? kAng : -kAng) * angN; // left if tilted right
    turnLogits[2] += (sign < 0 ? kAng : -kAng) * angN; // right if tilted left

    final turnP = softmax(turnLogits);
    return _Forward(x, z1, h1v, z2, h2v, thrLogits, thrP, turnLogits, turnP);
  }

  (bool thrust, bool left, bool right, List<double> probsThr, List<double> probsTurn, _Forward cache)
  act(List<double> x, math.Random rnd, {double angle = 0.0, double dx = 0.0}) {
    final f = _forward(x, angle: angle, dx: dx);
    final thr = rnd.nextDouble() < f.thrP[0];
    int cls = 0;
    final r = rnd.nextDouble();
    double c = 0.0;
    for (int i = 0; i < 3; i++) { c += f.turnP[i]; if (r <= c) { cls = i; break; } }
    final left = cls == 1;
    final right = cls == 2;
    return (thr, left, right, f.thrP, f.turnP, f);
  }

  void updateFromEpisode({
    required List<_Forward> caches,
    required List<List<int>> actions,   // [thr(0/1), turnCls(0/1/2)]
    required List<double> returns_,
    double lr = 3e-4,
    double l2 = 1e-6,
  }) {
    // Normalize advantages
    final adv = List<double>.from(returns_);
    final mean = adv.reduce((a,b)=>a+b)/adv.length;
    double variance = 0.0; for (final v in adv) { variance += (v-mean)*(v-mean); }
    variance /= adv.length;
    final std = math.sqrt(variance + 1e-8);
    for (int i=0; i<adv.length; i++) adv[i] = (adv[i]-mean)/std;

    // grads
    final dW1 = zeros(W1.length, W1[0].length);
    final dW2 = zeros(W2.length, W2[0].length);
    final dW_thr = zeros(W_thr.length, W_thr[0].length);
    final dW_turn = zeros(W_turn.length, W_turn[0].length);
    final db1 = List<double>.filled(b1.length, 0);
    final db2 = List<double>.filled(b2.length, 0);
    final db_thr = List<double>.filled(b_thr.length, 0);
    final db_turn = List<double>.filled(b_turn.length, 0);

    for (int t = 0; t < caches.length; t++) {
      final f = caches[t];
      final a_thr = actions[t][0];    // 0/1
      final a_cls = actions[t][1];    // 0/1/2
      final A = adv[t];

      // thrust: Bernoulli -> grad loglik at logits: (p - a)
      final dlog_thr = (f.thrP[0] - a_thr) * A; // scalar
      addInPlaceVec(db_thr, [dlog_thr]);
      addInPlaceMat(dW_thr, outer([dlog_thr], f.h2));

      // turn: 3-class softmax CE
      final dlog_turn = List<double>.generate(3, (k) => ((f.turnP[k] - (k == a_cls ? 1.0 : 0.0)) * A));
      addInPlaceVec(db_turn, dlog_turn);
      addInPlaceMat(dW_turn, outer(dlog_turn, f.h2));

      // backprop to h2
      final dh2 = List<double>.filled(f.h2.length, 0.0);
      // from thrust head
      for (int j = 0; j < W_thr[0].length; j++) {
        dh2[j] += W_thr[0][j] * dlog_thr;
      }
      // from turn head
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < W_turn[0].length; j++) {
          dh2[j] += W_turn[i][j] * dlog_turn[i];
        }
      }
      final dz2 = List<double>.generate(f.z2.length, (i) => dh2[i] * dRelu(f.z2[i]));
      addInPlaceVec(db2, dz2);
      addInPlaceMat(dW2, outer(dz2, f.h1));

      // backprop to h1
      final dh1 = List<double>.filled(f.h1.length, 0.0);
      for (int i = 0; i < W2.length; i++) {
        for (int j = 0; j < W2[0].length; j++) {
          dh1[j] += W2[i][j] * dz2[i];
        }
      }
      final dz1 = List<double>.generate(f.z1.length, (i) => dh1[i] * dRelu(f.z1[i]));
      addInPlaceVec(db1, dz1);
      addInPlaceMat(dW1, outer(dz1, f.x));
    }

    // L2
    void reg(List<List<double>> W) {
      for (final row in W) {
        for (int j = 0; j < row.length; j++) row[j] += l2 * row[j];
      }
    }
    reg(dW1); reg(dW2); reg(dW_thr); reg(dW_turn);

    // SGD
    void sgd(List<List<double>> W, List<List<double>> dW, double lr) {
      for (int i = 0; i < W.length; i++) {
        for (int j = 0; j < W[0].length; j++) {
          W[i][j] -= lr * dW[i][j];
        }
      }
    }
    void sgdB(List<double> b, List<double> db, double lr) {
      for (int i = 0; i < b.length; i++) b[i] -= lr * db[i];
    }

    sgd(W_thr, dW_thr, lr); sgdB(b_thr, db_thr, lr);
    sgd(W_turn, dW_turn, lr); sgdB(b_turn, db_turn, lr);
    sgd(W2, dW2, lr); sgdB(b2, db2, lr);
    sgd(W1, dW1, lr); sgdB(b1, db1, lr);
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

  Trainer({
    required this.env,
    required this.fe,
    required this.policy,
    this.dt = 1/60.0,
    this.gamma = 0.99,
    int seed = 7,
  }) : rnd = math.Random(seed);

  EpisodeResult runEpisode({bool train = true, double lr = 3e-4}) {
    // For overfit/debug we keep terrain & spawn locked via cfg; still randomize seed if you like
    env.reset(seed: 1337);

    final caches = <_Forward>[];
    final actions = <List<int>>[];  // [thr(0/1), turnCls]
    final costs = <double>[];

    int safety = 0;
    while (true) {
      final x = fe.extract(env);
      // Pass angle and dx for priors consistency
      final angle = env.lander.angle;
      final dx = (env.lander.pos.x - env.terrain.padCenter) / env.cfg.worldW;

      final (th, lf, rt, probsThr, probsTurn, cache) = policy.act(x, rnd, angle: angle, dx: dx);
      final turnCls = lf ? 1 : (rt ? 2 : 0);

      final info = env.step(dt, eng.ControlInput(thrust: th, left: lf, right: rt));

      caches.add(cache);
      actions.add([th ? 1 : 0, turnCls]);
      costs.add(info.scoreDelta); // cost >= 0

      safety++;
      if (info.terminal || safety > 4000) break;
    }

    // Compute discounted returns (we minimize cost -> use negative returns)
    final G = List<double>.filled(costs.length, 0.0);
    double running = 0.0;
    for (int t = costs.length - 1; t >= 0; t--) {
      running = -costs[t] + gamma * running;
      G[t] = running;
    }

    if (train && caches.isNotEmpty) {
      policy.updateFromEpisode(
        caches: caches,
        actions: actions,
        returns_: G,
        lr: lr,
      );
    }

    final landed = env.status == eng.GameStatus.landed;
    final total = costs.fold(0.0, (a,b)=>a+b);
    return EpisodeResult(total, costs.length, landed);
  }
}
