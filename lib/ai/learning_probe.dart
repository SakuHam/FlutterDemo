// lib/ai/learning_probe.dart
import 'dart:math' as math;

import '../engine/game_engine.dart' as eng;
import '../engine/types.dart' as et;
import 'agent.dart' show FeatureExtractor, PolicyNetwork, Intent, predictiveIntentLabelAdaptive;

/// Run:
///   dart run lib/ai/learning_probe.dart
///
/// Goal: prove the model+features can learn when labels are clean.
/// Switch `labelMode` to "synthetic" (noiseless) or "predictive" (heuristic).

// ---------- Adam state (typed) ----------
class AdamStateM {
  List<List<double>> m;
  List<List<double>> v;
  AdamStateM(this.m, this.v);
}
class AdamStateV {
  List<double> m;
  List<double> v;
  AdamStateV(this.m, this.v);
}

void main(List<String> args) {
  runCapacityProbe(
    samples: 3000,
    rounds: 10,
    batch: 128,
    lr: 1e-3,          // Adam lr
    l2: 1e-6,
    seed: 1234,
    labelMode: LabelMode.synthetic, // start with noiseless labels
  );
}

enum LabelMode { synthetic, predictive }

void runCapacityProbe({
  int samples = 3000,
  int rounds = 10,
  int batch = 128,
  double lr = 1e-3,
  double l2 = 1e-6,
  int seed = 1234,
  LabelMode labelMode = LabelMode.synthetic,
}) {
  final cfg = _makeConfig(seed: seed);
  final env = eng.GameEngine(cfg);

  final fe = FeatureExtractor(groundSamples: 3, stridePx: 48);
  final policy = PolicyNetwork(inputSize: fe.inputSize, h1: 64, h2: 64, seed: seed);

  env.reset(seed: 123456);

  final rng = math.Random(seed ^ 0xC0FFEE);

  // ---------- synth dataset ----------
  final xs = <List<double>>[];
  final ys = <int>[];

  void synth(Intent it) {
    final padCx = env.terrain.padCenter.toDouble();
    final padHalfW =
    (((env.terrain.padX2 - env.terrain.padX1).abs()) * 0.5).clamp(12.0, env.cfg.worldW.toDouble());

    double x = padCx, h = 200, vx = 0, vy = 20;
    switch (it) {
      case Intent.goLeft:
        x = padCx + (0.92 + 0.07 * rng.nextDouble()) * padHalfW;
        h = 160.0 + 120.0 * rng.nextDouble();
        vx = 55.0 + 45.0 * rng.nextDouble();
        vy = 30.0 + 40.0 * rng.nextDouble();
        break;
      case Intent.goRight:
        x = padCx - (0.92 + 0.07 * rng.nextDouble()) * padHalfW;
        h = 160.0 + 120.0 * rng.nextDouble();
        vx = -(55.0 + 45.0 * rng.nextDouble());
        vy = 30.0 + 40.0 * rng.nextDouble();
        break;
      case Intent.descendSlow:
        x = padCx + (rng.nextDouble() * 0.06 - 0.03) * padHalfW;
        h = 0.65 * env.cfg.worldH + 0.15 * env.cfg.worldH * rng.nextDouble();
        vx = (rng.nextDouble() * 30.0) - 15.0;
        vy = 26.0 + 12.0 * rng.nextDouble();
        break;
      case Intent.brakeUp:
        x = padCx + (rng.nextDouble() * 0.06 - 0.03) * padHalfW;
        h = 40.0 + 40.0 * rng.nextDouble();
        vx = (rng.nextDouble() * 18.0) - 9.0;
        vy = 140.0 + 30.0 * rng.nextDouble();
        break;
      case Intent.hoverCenter:
        x = padCx + (rng.nextDouble() * 0.04 - 0.02) * padHalfW;
        h = 0.22 * env.cfg.worldH + 0.02 * env.cfg.worldH * rng.nextDouble();
        vx = (rng.nextDouble() * 14.0) - 7.0;
        vy = 4.0 + 6.0 * rng.nextDouble();
        break;
    }

    final gy = env.terrain.heightAt(x);
    env.lander.pos.x = x.clamp(10.0, env.cfg.worldW - 10.0);
    env.lander.pos.y = (gy - h).clamp(0.0, env.cfg.worldH - 10.0);
    env.lander.vel.x = vx;
    env.lander.vel.y = vy;
    env.lander.angle = 0.0;
    env.lander.fuel = env.cfg.t.maxFuel;

    final xvec = fe.extract(env);
    final int yIdx = (labelMode == LabelMode.synthetic)
        ? it.index
        : predictiveIntentLabelAdaptive(env, baseTauSec: 1.0, minTauSec: 0.45, maxTauSec: 1.35);

    xs.add(xvec);
    ys.add(yIdx);
  }

  final perClass = (samples / Intent.values.length).ceil();
  for (final it in Intent.values) {
    for (int i = 0; i < perClass; i++) synth(it);
  }

  // ---------- standardize features (mean/std over dataset) ----------
  final d = xs[0].length;
  final mean = List<double>.filled(d, 0.0);
  for (final x in xs) {
    for (int j = 0; j < d; j++) mean[j] += x[j];
  }
  for (int j = 0; j < d; j++) mean[j] /= xs.length;

  final varsum = List<double>.filled(d, 0.0);
  for (final x in xs) {
    for (int j = 0; j < d; j++) {
      final dx = x[j] - mean[j];
      varsum[j] += dx * dx;
    }
  }
  final std = List<double>.generate(d, (j) {
    final v = varsum[j] / math.max(1, xs.length - 1);
    final s = math.sqrt(v + 1e-9);
    return s < 1e-6 ? 1.0 : s; // avoid div by ~0
  });
  List<double> _norm(List<double> x) =>
      List<double>.generate(d, (j) => (x[j] - mean[j]) / std[j]);
  for (int i = 0; i < xs.length; i++) xs[i] = _norm(xs[i]);

  // ---------- helpers ----------
  List<double> _lrelu(List<double> v, {double a = 0.05}) =>
      v.map((x) => x >= 0 ? x : a * x).toList();
  double _dlrelu(double x, {double a = 0.05}) => x >= 0 ? 1.0 : a;

  List<double> _matVec(List<List<double>> W, List<double> x) {
    final m = W.length, n = W[0].length;
    final out = List<double>.filled(m, 0.0);
    for (int i = 0; i < m; i++) {
      double s = 0.0;
      final Wi = W[i];
      for (int j = 0; j < n; j++) s += Wi[j] * x[j];
      out[i] = s;
    }
    return out;
  }

  List<double> _add(List<double> a, List<double> b) {
    final out = List<double>.filled(a.length, 0.0);
    for (int i = 0; i < a.length; i++) out[i] = a[i] + b[i];
    return out;
  }

  List<double> _softmax(List<double> z) {
    double m = z[0];
    for (int i = 1; i < z.length; i++) if (z[i] > m) m = z[i];
    final exps = List<double>.generate(z.length, (i) => math.exp(z[i] - m));
    final s = exps.reduce((a, b) => a + b);
    return exps.map((e) => e / s).toList();
  }

  List<List<double>> _zeros(int m, int n) =>
      List.generate(m, (_) => List<double>.filled(n, 0.0));
  void _addInPlaceM(List<List<double>> A, List<List<double>> B) {
    for (int i = 0; i < A.length; i++) {
      final Ai = A[i], Bi = B[i];
      for (int j = 0; j < Ai.length; j++) Ai[j] += Bi[j];
    }
  }
  void _addInPlaceV(List<double> a, List<double> b) {
    for (int i = 0; i < a.length; i++) a[i] += b[i];
  }

  // ---------- metrics ----------
  List<double> intentProbs(PolicyNetwork p, List<double> x) {
    final z1 = _add(_matVec(p.W1, x), p.b1);
    final h1 = _lrelu(z1);
    final z2 = _add(_matVec(p.W2, h1), p.b2);
    final h2 = _lrelu(z2);
    final zI = _add(_matVec(p.W_intent, h2), p.b_intent);
    return _softmax(zI);
  }

  double accNow() {
    int ok = 0;
    for (int i = 0; i < xs.length; i++) {
      final p = intentProbs(policy, xs[i]);
      int best = 0;
      double bv = p[0];
      for (int k = 1; k < p.length; k++) {
        if (p[k] > bv) {
          bv = p[k];
          best = k;
        }
      }
      if (best == ys[i]) ok++;
    }
    return ok / xs.length;
  }

  double meanPCorrect() {
    double s = 0;
    for (int i = 0; i < xs.length; i++) {
      final p = intentProbs(policy, xs[i]);
      s += p[ys[i]];
    }
    return s / xs.length;
  }


  AdamStateM _adamForM(List<List<double>> W) =>
  AdamStateM(_zeros(W.length, W[0].length), _zeros(W.length, W[0].length));
  AdamStateV _adamForV(List<double> b) =>
  AdamStateV(List<double>.filled(b.length, 0.0), List<double>.filled(b.length, 0.0));

  final optM = <String, AdamStateM>{
  'W1': _adamForM(policy.W1),
  'W2': _adamForM(policy.W2),
  'W_int': _adamForM(policy.W_intent),
  };
  final optV = <String, AdamStateV>{
  'b1': _adamForV(policy.b1),
  'b2': _adamForV(policy.b2),
  'b_int': _adamForV(policy.b_intent),
  };

  void adamStepM(List<List<double>> W, List<List<double>> g, AdamStateM st,
  {required double lr, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8, required int t}) {
  final m = st.m;
  final v = st.v;
  for (int i = 0; i < W.length; i++) {
  for (int j = 0; j < W[0].length; j++) {
  final gi = g[i][j];
  m[i][j] = b1 * m[i][j] + (1 - b1) * gi;
  v[i][j] = b2 * v[i][j] + (1 - b2) * gi * gi;
  final mhat = m[i][j] / (1 - math.pow(b1, t));
  final vhat = v[i][j] / (1 - math.pow(b2, t));
  W[i][j] -= lr * (mhat as double) / (math.sqrt(vhat as double) + eps);
  }
  }
  }

  void adamStepV(List<double> b, List<double> g, AdamStateV st,
  {required double lr, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8, required int t}) {
  final m = st.m;
  final v = st.v;
  for (int i = 0; i < b.length; i++) {
  final gi = g[i];
  m[i] = b1 * m[i] + (1 - b1) * gi;
  v[i] = b2 * v[i] + (1 - b2) * gi * gi;
  final mhat = m[i] / (1 - math.pow(b1, t));
  final vhat = v[i] / (1 - math.pow(b2, t));
  b[i] -= lr * (mhat as double) / (math.sqrt(vhat as double) + eps);
  }
  }

  // ---------- training ----------
  print('Dataset: ${xs.length}  | labelMode=${labelMode.name}');
  print('Initial acc=${(accNow() * 100).toStringAsFixed(1)}%  '
  'meanP(correct)=${meanPCorrect().toStringAsFixed(3)}');

  final K = 5; // intents
  final order = List<int>.generate(xs.length, (i) => i);
  final rnd = math.Random(seed ^ 0xA11CE);
  int tstep = 0;

  for (int r = 0; r < rounds; r++) {
  // shuffle each epoch
  for (int i = order.length - 1; i > 0; i--) {
  final j = rnd.nextInt(i + 1);
  final tmp = order[i];
  order[i] = order[j];
  order[j] = tmp;
  }
  int pos = 0;

  final iters = (xs.length / batch).ceil();
  for (int it = 0; it < iters; it++) {
  final dW1 = _zeros(policy.W1.length, policy.W1[0].length);
  final dW2 = _zeros(policy.W2.length, policy.W2[0].length);
  final db1 = List<double>.filled(policy.b1.length, 0.0);
  final db2 = List<double>.filled(policy.b2.length, 0.0);

  final dW_int = _zeros(policy.W_intent.length, policy.W_intent[0].length);
  final db_int = List<double>.filled(policy.b_intent.length, 0.0);

  final bsz = math.min(batch, xs.length - pos);
  for (int j = 0; j < bsz; j++) {
  final i = order[pos + j];
  final x = xs[i];
  final y = ys[i];

  // forward
  final z1 = _add(_matVec(policy.W1, x), policy.b1);
  final h1 = _lrelu(z1);
  final z2 = _add(_matVec(policy.W2, h1), policy.b2);
  final h2 = _lrelu(z2);
  final zI = _add(_matVec(policy.W_intent, h2), policy.b_intent);
  final pI = _softmax(zI);

  // CE grad: (p - y)
  final dzI = List<double>.generate(K, (k) => pI[k] - (k == y ? 1.0 : 0.0));

  // intent head grads
  for (int k = 0; k < K; k++) {
  for (int j2 = 0; j2 < h2.length; j2++) dW_int[k][j2] += dzI[k] * h2[j2];
  db_int[k] += dzI[k];
  }

  // backprop into trunk
  final dh2 = List<double>.filled(h2.length, 0.0);
  for (int k = 0; k < K; k++) {
  final row = policy.W_intent[k];
  for (int j2 = 0; j2 < h2.length; j2++) dh2[j2] += row[j2] * dzI[k];
  }
  final dz2 = List<double>.generate(z2.length, (q) => dh2[q] * _dlrelu(z2[q]));
  for (int r2 = 0; r2 < policy.W2.length; r2++) {
  for (int c2 = 0; c2 < policy.W2[0].length; c2++) dW2[r2][c2] += dz2[r2] * h1[c2];
  db2[r2] += dz2[r2];
  }

  final dh1 = List<double>.filled(h1.length, 0.0);
  for (int r2 = 0; r2 < policy.W2.length; r2++) {
  for (int c2 = 0; c2 < policy.W2[0].length; c2++) dh1[c2] += policy.W2[r2][c2] * dz2[r2];
  }
  final dz1 = List<double>.generate(z1.length, (q) => dh1[q] * _dlrelu(z1[q]));
  for (int r1 = 0; r1 < policy.W1.length; r1++) {
  for (int c1 = 0; c1 < policy.W1[0].length; c1++) dW1[r1][c1] += dz1[r1] * x[c1];
  db1[r1] += dz1[r1];
  }
  }

  // mean over batch + L2
  final inv = 1.0 / math.max(1, bsz);
  void scaleM(List<List<double>> G, List<List<double>> W) {
  for (int i = 0; i < G.length; i++) {
  for (int j = 0; j < G[0].length; j++) {
  G[i][j] = G[i][j] * inv + l2 * W[i][j];
  }
  }
  }
  void scaleV(List<double> g) { for (int i = 0; i < g.length; i++) g[i] *= inv; }
  scaleM(dW1, policy.W1); scaleM(dW2, policy.W2); scaleM(dW_int, policy.W_intent);
  scaleV(db1); scaleV(db2); scaleV(db_int);

  // Adam step (typed states)
  tstep++;
  adamStepM(policy.W1, dW1, optM['W1']!, lr: lr, t: tstep);
  adamStepV(policy.b1, db1, optV['b1']!, lr: lr, t: tstep);
  adamStepM(policy.W2, dW2, optM['W2']!, lr: lr, t: tstep);
  adamStepV(policy.b2, db2, optV['b2']!, lr: lr, t: tstep);
  adamStepM(policy.W_intent, dW_int, optM['W_int']!, lr: lr, t: tstep);
  adamStepV(policy.b_intent, db_int, optV['b_int']!, lr: lr, t: tstep);

  pos += bsz;
  }

  print('Round ${r + 1}/$rounds  acc=${(accNow() * 100).toStringAsFixed(1)}%  '
  'meanP(correct)=${meanPCorrect().toStringAsFixed(3)}');
  }

  print('Final acc=${(accNow() * 100).toStringAsFixed(1)}%  '
  'meanP(correct)=${meanPCorrect().toStringAsFixed(3)}');
}

et.EngineConfig _makeConfig({int seed = 42}) {
  final t = et.Tunables(
    gravity: 0.18,
    thrustAccel: 0.42,
    rotSpeed: 1.6,
    maxFuel: 1000.0,
  );
  return et.EngineConfig(
    worldW: 800,
    worldH: 600,
    t: t,
    seed: seed,
    stepScale: 60.0,
    lockTerrain: true,
    terrainSeed: 1234567,
    lockSpawn: true,
    randomSpawnX: false,
    hardWalls: true,
  );
}
