// lib/ai/train_agent.dart
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import '../engine/types.dart' as et;
import '../engine/game_engine.dart' as eng;

import 'agent.dart';

/* ------------------------------- tiny arg parser ------------------------------- */

class _Args {
  final Map<String, String?> _kv = {};
  final Set<String> _flags = {};
  _Args(List<String> argv) {
    for (final a in argv) {
      if (!a.startsWith('--')) continue;
      final s = a.substring(2);
      final i = s.indexOf('=');
      if (i >= 0) {
        _kv[s.substring(0, i)] = s.substring(i + 1);
      } else {
        _flags.add(s);
      }
    }
  }
  String? getStr(String k, {String? def}) => _kv[k] ?? def;
  int getInt(String k, {int def = 0}) => int.tryParse(_kv[k] ?? '') ?? def;
  double getDouble(String k, {double def = 0.0}) => double.tryParse(_kv[k] ?? '') ?? def;
  bool getFlag(String k, {bool def = false}) => _flags.contains(k) ? true : def;
}

/* ------------------------------- feature signature ------------------------------ */

String _feSignature({
  required int groundSamples,
  required double stridePx,
  required double worldW,
  required double worldH,
}) {
  return 'gs=$groundSamples;stride=${stridePx.toStringAsFixed(2)};W=${worldW.toInt()};H=${worldH.toInt()}';
}

/* ------------------------------- norm update helper ----------------------------- */

void _normUpdate(RunningNorm? norm, List<double> x) {
  if (norm == null) return;
  try {
    // Prefer observe(x) if available
    (norm as dynamic).observe(x);
  } catch (_) {
    norm.normalize(x, update: true);
  }
}

/* ------------------------------- policy IO (json) ------------------------------ */

Map<String, dynamic> _weightsToJson({
  required PolicyNetwork p,
  required FeatureExtractor fe,
  required eng.GameEngine env,
  RunningNorm? norm,
}) {
  List<List<double>> to3(List<List<double>> W) =>
      W.map((r) => r.map((v) => v.toDouble()).toList()).toList();

  final sig = _feSignature(
    groundSamples: fe.groundSamples,
    stridePx: fe.stridePx,
    worldW: env.cfg.worldW,
    worldH: env.cfg.worldH,
  );

  final m = <String, dynamic>{
    'h1': p.h1,
    'h2': p.h2,
    'W1': to3(p.W1), 'b1': p.b1,
    'W2': to3(p.W2), 'b2': p.b2,
    'W_thr': to3(p.W_thr), 'b_thr': p.b_thr,
    'W_turn': to3(p.W_turn), 'b_turn': p.b_turn,
    'W_intent': to3(p.W_intent), 'b_intent': p.b_intent,
    'W_val': to3(p.W_val), 'b_val': p.b_val,
    'arch': {'input': p.inputSize, 'h1': p.h1, 'h2': p.h2, 'kIntents': PolicyNetwork.kIntents},
    'feature_extractor': {'groundSamples': fe.groundSamples, 'stridePx': fe.stridePx},
    'env_hint': {'worldW': env.cfg.worldW, 'worldH': env.cfg.worldH},
    'signature': sig,
  };

  // Save norm in BOTH formats (nested + top-level) for compatibility
  if (norm != null && norm.inited && norm.dim == p.inputSize) {
    m['norm'] = {
      'dim': norm.dim,
      'momentum': norm.momentum,
      'mean': norm.mean,
      'var': norm.var_,
      'signature': sig,
    };
    m['norm_mean'] = norm.mean;
    m['norm_var'] = norm.var_;
    m['norm_momentum'] = norm.momentum;
    m['norm_signature'] = sig;
  }
  return m;
}

void savePolicy(String path, PolicyNetwork p, FeatureExtractor fe, eng.GameEngine env, {RunningNorm? norm}) {
  final f = File(path);
  final jsonMap = _weightsToJson(p: p, fe: fe, env: env, norm: norm);
  f.writeAsStringSync(const JsonEncoder.withIndent('  ').convert(jsonMap));
  print('Saved policy → $path');
}

void _from3(List<List<double>> dst, List src) {
  for (int i = 0; i < dst.length; i++) {
    final ri = dst[i];
    final si = (src[i] as List);
    for (int j = 0; j < ri.length; j++) {
      ri[j] = (si[j] as num).toDouble();
    }
  }
}

bool tryLoadPolicy(
    String path,
    PolicyNetwork p, {
      RunningNorm? norm,
      FeatureExtractor? fe,
      eng.GameEngine? env,
      bool ignoreLoadedNorm = false,
    }) {
  final f = File(path);
  if (!f.existsSync()) return false;

  dynamic raw;
  try {
    raw = json.decode(f.readAsStringSync());
  } catch (e) {
    print('Failed to parse $path: $e');
    return false;
  }
  if (raw is! Map<String, dynamic>) {
    print('Unexpected JSON root in $path (expected object). Skipping load.');
    return false;
  }
  final m = raw;

  // Helper: accept either a raw List<List<num>> OR an object with {data: List<List<num>>}
  List _mat(dynamic v) {
    if (v is List) return v;
    if (v is Map && v['data'] is List) return v['data'] as List;
    throw StateError('matrix field is not a List or {data: List}');
  }

  bool _tryFillMat(List<List<double>> dst, String key) {
    final v = m[key];
    if (v == null) return false;
    try { _from3(dst, _mat(v)); return true; } catch (_) { return false; }
  }
  bool _tryFillVec(List<double> dst, String key) {
    final v = m[key];
    if (v == null) return false;
    try {
      final L = (v as List).map((e) => (e as num).toDouble()).toList();
      final n = math.min(dst.length, L.length);
      for (int i = 0; i < n; i++) dst[i] = L[i];
      return true;
    } catch (_) { return false; }
  }

  int ok = 0, total = 0;

  total += 2;
  if (_tryFillMat(p.W1, 'W1')) ok++;
  if (_tryFillVec(p.b1, 'b1')) ok++;
  total += 2;
  if (_tryFillMat(p.W2, 'W2')) ok++;
  if (_tryFillVec(p.b2, 'b2')) ok++;

  total += 2;
  if (_tryFillMat(p.W_thr, 'W_thr')) ok++;
  if (_tryFillVec(p.b_thr, 'b_thr')) ok++;

  total += 2;
  if (_tryFillMat(p.W_turn, 'W_turn')) ok++;
  if (_tryFillVec(p.b_turn, 'b_turn')) ok++;

  total += 2;
  if (_tryFillMat(p.W_intent, 'W_intent')) ok++;
  if (_tryFillVec(p.b_intent, 'b_intent')) ok++;

  total += 2;
  if (_tryFillMat(p.W_val, 'W_val')) ok++;
  if (_tryFillVec(p.b_val, 'b_val')) ok++;

  // Guarded norm load (prefer nested, but accept top-level too)
  if (!ignoreLoadedNorm && norm != null && fe != null && env != null) {
    final sigNow = _feSignature(
      groundSamples: fe.groundSamples,
      stridePx: fe.stridePx,
      worldW: env.cfg.worldW,
      worldH: env.cfg.worldH,
    );
    final sigTop = m['signature'] as String?;

    bool loadedNorm = false;

    final mNorm = (m['norm'] as Map?)?.cast<String, dynamic>();
    if (mNorm != null) {
      final dim = (mNorm['dim'] as num?)?.toInt() ?? -1;
      final sigFile = (mNorm['signature'] as String?) ?? '';
      if (dim == norm.dim && sigFile == sigNow && sigTop == sigNow) {
        try {
          norm.mean = (mNorm['mean'] as List).map((e) => (e as num).toDouble()).toList();
          norm.var_ = (mNorm['var'] as List).map((e) => (e as num).toDouble()).toList();
          if (mNorm['momentum'] is num) norm.momentum = (mNorm['momentum'] as num).toDouble();
          norm.inited = true;
          print('Loaded feature norm (nested) (dim=${norm.dim}) from $path (signature match).');
          loadedNorm = true;
        } catch (_) {
          print('Feature norm present but malformed (nested) → ignoring.');
        }
      }
    }

    if (!loadedNorm) {
      final nm = m['norm_mean'];
      final nv = m['norm_var'];
      final nsig = m['norm_signature'];
      if (nm is List && nv is List && nsig == sigNow && sigTop == sigNow) {
        final n = math.min(norm.dim, math.min(nm.length, nv.length));
        try {
          norm.mean = List<double>.generate(n, (i) => (nm[i] as num).toDouble())
            ..length = norm.dim;
          norm.var_ = List<double>.generate(n, (i) => (nv[i] as num).toDouble())
            ..length = norm.dim;
          if (m['norm_momentum'] is num) norm.momentum = (m['norm_momentum'] as num).toDouble();
          norm.inited = true;
          print('Loaded feature norm (top-level) (dim=${norm.dim}) from $path (signature match).');
          loadedNorm = true;
        } catch (_) {
          print('Feature norm present but malformed (top-level) → ignoring.');
        }
      }
    }

    if (!loadedNorm) {
      print('Saved norm signature/dim mismatch or missing → ignoring loaded norm.');
    }
  }

  final tag = (ok == total) ? '' : '  (PARTIAL — IGNORING)';
  print('Loaded policy ← $path ($ok/$total tensors filled)$tag');
  return ok > 0;
}

/* --------------------------------- env config --------------------------------- */

et.EngineConfig makeConfig({
  int seed = 42,
  bool lockTerrain = false,
  bool lockSpawn = false,
  bool randomSpawnX = true,
  double worldW = 800,
  double worldH = 600,
  double? maxFuel,
}) {
  final t = et.Tunables(
    gravity: 0.18,
    thrustAccel: 0.42,
    rotSpeed: 1.6,
    maxFuel: maxFuel ?? 1000.0,
  );
  return et.EngineConfig(
    worldW: worldW,
    worldH: worldH,
    t: t,
    seed: seed,
    stepScale: 60.0,
    lockTerrain: lockTerrain,
    terrainSeed: 1234567,
    lockSpawn: lockSpawn,
    randomSpawnX: randomSpawnX,
    hardWalls: true,
  );
}

/* ------------------------------ determinism probe ------------------------------ */

typedef _RolloutRes = ({int steps, double cost});

_RolloutRes _probeDeterminism(eng.GameEngine env, {int maxSteps = 200}) {
  var cost = 0.0;
  int t = 0;
  while (t < maxSteps) {
    final info = env.step(1 / 60.0, const et.ControlInput(thrust: false, left: false, right: false));
    cost += info.costDelta;
    if (info.terminal) break;
    t++;
  }
  return (steps: t, cost: cost);
}

/* ----------------------------------- eval ------------------------------------- */

class EvalStats {
  double meanCost = 0;
  double medianCost = 0;
  double landPct = 0;
  double crashPct = 0;
  double meanSteps = 0;
  double meanAbsDx = 0;
}

EvalStats evaluate({
  required eng.GameEngine env,
  required Trainer trainer,
  int episodes = 50,
  int seed = 123,
}) {
  final rnd = math.Random(seed);
  final costs = <double>[];
  int landed = 0, crashed = 0, stepsSum = 0;
  double absDxSum = 0.0;

  for (int i = 0; i < episodes; i++) {
    env.reset(seed: rnd.nextInt(1 << 30));
    final res = trainer.runEpisode(
      train: false,
      greedy: true,
      scoreIsReward: false,
    );
    costs.add(res.totalCost);
    stepsSum += res.steps;
    if (env.status == et.GameStatus.landed) landed++; else crashed++;

    final padCx = env.terrain.padCenter;
    absDxSum += (env.lander.pos.x - padCx).abs();
  }

  costs.sort();
  final st = EvalStats();
  st.meanCost = costs.isEmpty ? 0 : costs.reduce((a,b)=>a+b) / costs.length;
  st.medianCost = costs.isEmpty ? 0 : costs[costs.length ~/ 2];
  st.landPct = 100.0 * landed / episodes;
  st.crashPct = 100.0 * crashed / episodes;
  st.meanSteps = stepsSum / episodes;
  st.meanAbsDx = absDxSum / episodes;
  return st;
}

/* ----------------------------- norm reset & warmup ----------------------------- */

void _resetFeatureNorm(RunningNorm? norm) {
  if (norm == null) return;
  norm.inited = false;
  norm.mean = List.filled(norm.dim, 0.0);
  norm.var_ = List.filled(norm.dim, 1.0);
}

void _warmFeatureNorm({
  required RunningNorm? norm,
  required Trainer trainer,
  required FeatureExtractor fe,
  required eng.GameEngine env,
  int perClass = 600,
  int seed = 4242,
}) {
  if (norm == null) return;
  final r = math.Random(seed);
  env.reset(seed: 1234567);
  int accepted = 0, target = perClass * PolicyNetwork.kIntents;

  while (accepted < target) {
    final want = r.nextInt(PolicyNetwork.kIntents);

    final padCx = env.terrain.padCenter.toDouble();
    final padHalfW = (((env.terrain.padX2 - env.terrain.padX1).abs()) * 0.5)
        .clamp(12.0, env.cfg.worldW.toDouble());
    double x=padCx, h=180, vx=0, vy=20;

    switch (want) {
      case 1: // goLeft → need dx < -0.08*W (spawn to LEFT of pad)
        x  = (padCx - (0.22 + 0.18*r.nextDouble()) * env.cfg.worldW)
            .clamp(10.0, env.cfg.worldW - 10.0);
        h  = 120 + 120*r.nextDouble();
        vx = (r.nextDouble()*14.0) - 7.0;
        vy = 28.0 + 10.0*r.nextDouble();
        break;
      case 2: // goRight → need dx > +0.08*W (spawn to RIGHT of pad)
        x  = (padCx + (0.22 + 0.18*r.nextDouble()) * env.cfg.worldW)
            .clamp(10.0, env.cfg.worldW - 10.0);
        h  = 120 + 120*r.nextDouble();
        vx = (r.nextDouble()*14.0) - 7.0;
        vy = 28.0 + 10.0*r.nextDouble();
        break;
      case 3: // descendSlow
        x  = padCx + (r.nextDouble()*0.05 - 0.025) * padHalfW;
        h  = 0.55*env.cfg.worldH + 0.20*env.cfg.worldH*r.nextDouble();
        vx = (r.nextDouble()*24.0) - 12.0;
        vy = 24.0 + 12.0*r.nextDouble();
        break;
      case 4: // brakeUp
        x  = padCx + (r.nextDouble()*0.05 - 0.025) * padHalfW;
        h  = 40.0 + 50.0*r.nextDouble();
        vx = (r.nextDouble()*14.0) - 7.0;
        vy = 120.0 + 50.0*r.nextDouble();
        break;
      default: // hover
        x  = padCx + (r.nextDouble()*0.03 - 0.015) * padHalfW;
        h  = 0.20*env.cfg.worldH + 0.15*env.cfg.worldH*r.nextDouble();
        vx = (r.nextDouble()*10.0) - 5.0;
        vy = (r.nextDouble()*10.0) - 5.0;
        break;
    }

    final gy = env.terrain.heightAt(x);
    env.lander
      ..pos.x = x.clamp(10.0, env.cfg.worldW-10.0)
      ..pos.y = (gy - h).clamp(0.0, env.cfg.worldH-10.0)
      ..vel.x = vx
      ..vel.y = vy
      ..angle = 0.0
      ..fuel = env.cfg.t.maxFuel;

    if (predictiveIntentLabelAdaptive(env) != want) continue;

    final feat = fe.extract(env);
    _normUpdate(trainer.norm, feat);
    accepted++;
  }
  print('Feature norm warmed with $accepted synthetic samples.');
}

/* ---------------------------- intent pretrain (local) -------------------------- */

class _PretrainIntentStats {
  final double acc; final int n;
  _PretrainIntentStats(this.acc, this.n);
}

_PretrainIntentStats _pretrainIntentLocal({
  required Trainer trainer,
  required FeatureExtractor fe,
  required eng.GameEngine env,
  required PolicyNetwork policy,
  int samples = 6000,
  int epochs = 2,
  double lr = 5e-4,
  double alignWeight = 2.0,
  int seed = 1337,
}) {
  final rng = math.Random(seed);
  final K = PolicyNetwork.kIntents;
  final targetPerClass = (samples / K).ceil();

  // balanced buffers
  final xsByK = List.generate(K, (_) => <List<double>>[]);
  final ysByK = List.generate(K, (_) => <int>[]);

  void _setState({required double x, required double height, required double vx, required double vy}) {
    final gy = env.terrain.heightAt(x);
    final y = (gy - height).clamp(0.0, env.cfg.worldH - 10.0);
    env.lander.pos.x = x.clamp(10.0, env.cfg.worldW - 10.0);
    env.lander.pos.y = y;
    env.lander.vel.x = vx;
    env.lander.vel.y = vy;
    env.lander.angle = 0.0;
    env.lander.fuel  = env.cfg.t.maxFuel;
  }

  env.reset(seed: 123456);

  bool _synthOneFor(int intentIdx, math.Random r) {
    final padCx = env.terrain.padCenter.toDouble();
    final padHalfW = (((env.terrain.padX2 - env.terrain.padX1).abs()) * 0.5)
        .clamp(12.0, env.cfg.worldW.toDouble());
    double x=padCx, h=180, vx=0, vy=20;

    switch (intentIdx) {
      case 1: // goLeft → dx < -0.08*W
        x  = (padCx - (0.22 + 0.18*r.nextDouble()) * env.cfg.worldW)
            .clamp(10.0, env.cfg.worldW - 10.0);
        h  = 120 + 120*r.nextDouble();
        vx = (r.nextDouble()*14.0) - 7.0;
        vy = 28.0 + 10.0*r.nextDouble();
        break;
      case 2: // goRight → dx > +0.08*W
        x  = (padCx + (0.22 + 0.18*r.nextDouble()) * env.cfg.worldW)
            .clamp(10.0, env.cfg.worldW - 10.0);
        h  = 120 + 120*r.nextDouble();
        vx = (r.nextDouble()*14.0) - 7.0;
        vy = 28.0 + 10.0*r.nextDouble();
        break;
      case 3: // descendSlow
        x  = padCx + (r.nextDouble()*0.05 - 0.025) * padHalfW;
        h  = 0.55*env.cfg.worldH + 0.20*env.cfg.worldH*r.nextDouble();
        vx = (r.nextDouble()*24.0) - 12.0;
        vy = 24.0 + 12.0*r.nextDouble();
        break;
      case 4: // brakeUp
        x  = padCx + (r.nextDouble()*0.05 - 0.025) * padHalfW;
        h  = 40.0 + 50.0*r.nextDouble();
        vx = (r.nextDouble()*14.0) - 7.0;
        vy = 120.0 + 50.0*r.nextDouble();
        break;
      default: // hover
        x  = padCx + (r.nextDouble()*0.03 - 0.015) * padHalfW;
        h  = 0.20*env.cfg.worldH + 0.15*env.cfg.worldH*r.nextDouble();
        vx = (r.nextDouble()*10.0) - 5.0;
        vy = (r.nextDouble()*10.0) - 5.0;
        break;
    }

    _setState(x: x, height: h, vx: vx, vy: vy);

    final y = predictiveIntentLabelAdaptive(env, baseTauSec: 1.0, minTauSec: 0.45, maxTauSec: 1.35);
    if (y != intentIdx) return false;

    final feat = fe.extract(env);
    xsByK[intentIdx].add(feat);
    ysByK[intentIdx].add(intentIdx);
    return true;
  }

  const maxAttemptsPerClass = 20000;
  for (int k = 0; k < K; k++) {
    int attempts = 0;
    while (xsByK[k].length < targetPerClass && attempts < maxAttemptsPerClass) {
      _synthOneFor(k, rng);
      attempts++;
    }
    if (xsByK[k].isEmpty) {
      final fallback = (k == 0 ? 3 : 0);
      while (xsByK[k].length < math.max(16, targetPerClass ~/ 4) &&
          xsByK[fallback].isNotEmpty) {
        xsByK[k].add(List<double>.from(xsByK[fallback][rng.nextInt(xsByK[fallback].length)]));
        ysByK[k].add(k);
      }
    }
  }

  // Oversample partially-filled classes to hit targetPerClass
  for (int k = 0; k < K; k++) {
    if (xsByK[k].isEmpty) continue;
    final need = targetPerClass - xsByK[k].length;
    for (int i = 0; i < need; i++) {
      final src = xsByK[k][rng.nextInt(xsByK[k].length)];
      xsByK[k].add(List<double>.from(src));
      ysByK[k].add(k);
    }
  }

  final xs = <List<double>>[]; final ys = <int>[];
  for (int k = 0; k < K; k++) { xs.addAll(xsByK[k]); ys.addAll(ysByK[k]); }
  final N = xs.length;
  final perm = List<int>.generate(N, (i) => i)..shuffle(rng);

  // Warm/freeze norm on this data
  for (final x in xs) {
    _normUpdate(trainer.norm, x);
  }

  // Simple CE training on intent head (uses PolicyNetwork.updateFromEpisode intent path)
  const B = 64;
  for (int ep = 0; ep < epochs; ep++) {
    perm.shuffle(rng);
    for (int off = 0; off < N; off += B) {
      final end = math.min(off + B, N);
      final decisionCaches = <ForwardCache>[];
      final intentChoices  = <int>[];
      final decisionReturns= <double>[];
      final alignLabels    = <int>[];

      for (int i = off; i < end; i++) {
        final idx = perm[i];
        final xN = trainer.norm?.normalize(xs[idx], update: false) ?? xs[idx];
        final (_pred, _p, cache) = policy.actIntentGreedy(xN);
        decisionCaches.add(cache);
        intentChoices.add(ys[idx]);
        decisionReturns.add(cache.v);   // A=0 trick (baseline-free)
        alignLabels.add(ys[idx]);
      }

      policy.updateFromEpisode(
        decisionCaches: decisionCaches,
        intentChoices: intentChoices,
        decisionReturns: decisionReturns,
        alignLabels: alignLabels,
        alignWeight: alignWeight,
        lr: lr,
        entropyBeta: 0.0,
        valueBeta: 0.0,
        huberDelta: 1.0,
        intentMode: true,
      );
    }
  }

  // Evaluate + confusion matrix
  final xsShuf = List<List<double>>.generate(N, (i) => xs[perm[i]]);
  final ysShuf = List<int>.generate(N, (i) => ys[perm[i]]);

  final conf = List.generate(K, (_) => List<int>.filled(K, 0));
  final counts = List<int>.filled(K, 0);
  int correct = 0;
  for (int i = 0; i < N; i++) {
    final xN = trainer.norm?.normalize(xsShuf[i], update: false) ?? xsShuf[i];
    final (pred, _p, _cache) = policy.actIntentGreedy(xN);
    final y = ysShuf[i];
    counts[y] += 1;
    conf[y][pred] += 1;
    if (pred == y) correct++;
  }
  final acc = N == 0 ? 0.0 : correct / N;

  print('Pretrain confusion (rows=label, cols=pred):');
  for (int r = 0; r < K; r++) {
    final row = List.generate(K, (c) => conf[r][c].toString().padLeft(4)).join(' ');
    final rowAcc = counts[r] == 0 ? 0.0 : conf[r][r] / counts[r];
    print('${kIntentNames[r].padRight(12)} | $row   (acc=${(rowAcc*100).toStringAsFixed(1)}%  n=${counts[r]})');
  }

  return _PretrainIntentStats(acc, N);
}

/* ------------------------------------ main ------------------------------------ */

void main(List<String> argv) {
  final args = _Args(argv);

  final seed = args.getInt('seed', def: 7);

  final pretrainN = args.getInt('pretrain_intent', def: 6000);
  final pretrainEpochs = args.getInt('pretrain_epochs', def: 2);
  final pretrainAlign = args.getDouble('pretrain_align', def: 2.0);
  final pretrainLr = args.getDouble('pretrain_lr', def: 3e-4);
  final pretrainAll = args.getFlag('pretrain_all', def: false);
  final pretrainActionsN = args.getInt('pretrain_actions_n', def: 0);
  final resetActionHeads = args.getFlag('reset_action_heads', def: false);
  final onlyPretrain = args.getFlag('only_pretrain', def: false);
  final ignoreLoadedNorm = args.getFlag('ignore_loaded_norm', def: false);

  final iters = args.getInt('train_iters', def: args.getInt('iters', def: 200));
  final batch = args.getInt('batch', def: 32);
  final lr = args.getDouble('lr', def: 3e-4);
  final valueBeta = args.getDouble('value_beta', def: args.getDouble('valueBeta', def: 0.5));
  final huberDelta = args.getDouble('huber_delta', def: 1.0);

  final planHold = args.getInt('plan_hold', def: 1);
  final tempIntent = args.getDouble('intent_temp', def: 1.0);
  final intentEntropy = args.getDouble('intent_entropy', def: args.getDouble('intentEntropyBeta', def: 0.0));
  final useLearned = args.getFlag('use_learned_controller', def: false);
  final blendPolicy = args.getDouble('blend_policy', def: 1.0);
  final actionAlignWeight = args.getDouble('action_align', def: 0.0);

  final lockTerrain = args.getFlag('lock_terrain', def: false);
  final lockSpawn = args.getFlag('lock_spawn', def: false);
  final randomSpawnX = !args.getFlag('fixed_spawn_x', def: false);
  final maxFuel = args.getDouble('max_fuel', def: 1000.0);

  final determinism = args.getFlag('determinism_probe', def: true);

  final cfg = makeConfig(
    seed: seed,
    lockTerrain: lockTerrain,
    lockSpawn: lockSpawn,
    randomSpawnX: randomSpawnX,
    maxFuel: maxFuel,
  );
  final env = eng.GameEngine(cfg);

  final fe = FeatureExtractor(groundSamples: 3, stridePx: 48);
  final policy = PolicyNetwork(inputSize: fe.inputSize, h1: 64, h2: 64, seed: seed);
  print('Loaded init policy. h1=${policy.h1} h2=${policy.h2} | FE(gs=${fe.groundSamples} stride=${fe.stridePx})');

  final trainer = Trainer(
    env: env,
    fe: fe,
    policy: policy,
    dt: 1/60.0,
    gamma: 0.99,
    seed: seed,
    twoStage: true,
    planHold: planHold,
    tempIntent: tempIntent,
    intentEntropyBeta: intentEntropy,
    useLearnedController: useLearned,
    blendPolicy: blendPolicy.clamp(0.0, 1.0),
    intentAlignWeight: pretrainAll ? pretrainAlign : 0.25,
    actionAlignWeight: actionAlignWeight,
    normalizeFeatures: true,
  );

  // Try to load existing weights (+norm w/ signature guard)
  tryLoadPolicy(
    'policy_pretrained.json',
    policy,
    norm: trainer.norm,
    fe: fe,
    env: env,
    ignoreLoadedNorm: ignoreLoadedNorm,
  );

  if (determinism) {
    env.reset(seed: 1234);
    final a = _probeDeterminism(env, maxSteps: 165);
    env.reset(seed: 1234);
    final b = _probeDeterminism(env, maxSteps: 165);
    final ok = (a.steps == b.steps) && ((a.cost - b.cost).abs() < 1e-6);
    print('Determinism probe: steps ${a.steps} vs ${b.steps} | cost ${a.cost.toStringAsFixed(6)} vs ${b.cost.toStringAsFixed(6)} => ${ok ? "OK" : "MISMATCH"}');
  }

  // Optionally reset action heads
  if (resetActionHeads) {
    List<List<double>> _randInit(int rows, int cols, int s) {
      final r = math.Random(s);
      final limit = math.sqrt(6.0 / (rows + cols));
      return List.generate(rows, (_) => List<double>.generate(cols, (_) => (r.nextDouble()*2-1)*limit));
    }
    policy.W_thr = _randInit(1, policy.h2, seed ^ 0xA1);
    policy.b_thr = List<double>.filled(1, 0.0);
    policy.W_turn = _randInit(3, policy.h2, seed ^ 0xB2);
    policy.b_turn = List<double>.filled(3, 0.0);
    print('Action heads reset (W_thr, b_thr, W_turn, b_turn).');
  }

  // ===== PRETRAIN =====
  if (pretrainAll || onlyPretrain || pretrainN > 0 || pretrainActionsN > 0) {
    if (ignoreLoadedNorm || !((trainer.norm?.inited) ?? false)) {
      _resetFeatureNorm(trainer.norm);
      _warmFeatureNorm(norm: trainer.norm, trainer: trainer, fe: fe, env: env, perClass: 700, seed: seed ^ 0xACE);
    }

    // Intent pretrain (local, CE)
    if (pretrainN > 0) {
      print('Pretraining intent on $pretrainN snapshots (epochs=$pretrainEpochs, align=$pretrainAlign, lr=$pretrainLr) ...');
      final st = _pretrainIntentLocal(
        trainer: trainer,
        fe: fe,
        env: env,
        policy: policy,
        samples: pretrainN,
        epochs: pretrainEpochs,
        lr: pretrainLr,
        alignWeight: pretrainAlign,
        seed: seed ^ 0xBEEF,
      );
      print('Intent pretrain → acc=${(st.acc*100).toStringAsFixed(1)}% n=${st.n}');
      savePolicy('policy_pretrained.json', policy, fe, env, norm: trainer.norm);
    }

    // Action heads pretrain – not implemented in this agent build
    if (pretrainActionsN > 0) {
      print('Note: --pretrain_actions_n=$pretrainActionsN requested, '
          'but this agent build has no Trainer.pretrainActions(). Skipping action pretrain.');
    }

    if (onlyPretrain) {
      print('Only-pretrain mode: saved → policy_pretrained.json. Exiting.');
      return;
    }
  }

  // quick eval baseline
      {
    final ev = evaluate(env: env, trainer: trainer, episodes: 20, seed: seed ^ 0x999);
    print('Eval(real) → meanCost=${ev.meanCost.toStringAsFixed(3)} | median=${ev.medianCost.toStringAsFixed(3)} | land%=${ev.landPct.toStringAsFixed(1)} | crash%=${ev.crashPct.toStringAsFixed(1)} | steps=${ev.meanSteps.toStringAsFixed(1)} | mean|dx|=${ev.meanAbsDx.toStringAsFixed(1)}');
  }

  // ===== MAIN TRAIN LOOP =====
  final rnd = math.Random(seed ^ 0xDEADBEEF);

  for (int it = 0; it < iters; it++) {
    double lastCost = 0.0;
    int lastSteps = 0;
    bool lastLanded = false;

    for (int b = 0; b < batch; b++) {
      env.reset(seed: rnd.nextInt(1 << 30));
      final res = trainer.runEpisode(
        train: true,
        greedy: false,
        scoreIsReward: false,
        lr: lr,
        valueBeta: valueBeta,
        huberDelta: huberDelta,
      );
      lastCost = res.totalCost;
      lastSteps = res.steps;
      lastLanded = res.landed;
    }

    print('Iter ${it + 1} | batch=$batch | last-ep steps: $lastSteps | cost: ${lastCost.toStringAsFixed(3)} | landed: ${lastLanded ? "Y" : "N"}');

    if ((it + 1) % 5 == 0) {
      final ev = evaluate(env: env, trainer: trainer, episodes: 40, seed: seed ^ (0x1111 * (it + 1)));
      print('Eval(real) → meanCost=${ev.meanCost.toStringAsFixed(3)} | median=${ev.medianCost.toStringAsFixed(3)} | land%=${ev.landPct.toStringAsFixed(1)} | crash%=${ev.crashPct.toStringAsFixed(1)} | steps=${ev.meanSteps.toStringAsFixed(1)} | mean|dx|=${ev.meanAbsDx.toStringAsFixed(1)}');
      savePolicy('policy_iter_${it + 1}.json', policy, fe, env, norm: trainer.norm);
    }
  }

  savePolicy('policy_final.json', policy, fe, env, norm: trainer.norm);
  print('Training done. Saved → policy_final.json');
}

/* ----------------------------------- usage ------------------------------------

Pretrain only (intent; reset action heads; ignore old norm):
  dart run lib/ai/train_agent.dart \
    --pretrain_intent=10000 --pretrain_epochs=3 --pretrain_align=3.0 --pretrain_lr=0.0005 \
    --reset_action_heads --ignore_loaded_norm --only_pretrain

Full train (after pretrain):
  dart run lib/ai/train_agent.dart \
    --train_iters=200 --batch=32 --lr=0.0003 \
    --plan_hold=1 --use_learned_controller --blend_policy=0.75 \
    --value_beta=0.7

-------------------------------------------------------------------------------- */
