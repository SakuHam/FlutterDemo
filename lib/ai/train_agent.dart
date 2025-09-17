// lib/ai/train_agent.dart
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import '../engine/types.dart' as et;
import '../engine/game_engine.dart' as eng;

import 'agent.dart'; // uses nn_helper.dart internally

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

/* ------------------------------- matrix helpers -------------------------------- */

List<List<double>> _deepCopyMat(List<List<double>> W) =>
    List.generate(W.length, (i) => List<double>.from(W[i]));

void _assignMat(List<List<double>> dst, List src) {
  for (int i = 0; i < dst.length; i++) {
    final ri = dst[i];
    final si = (src[i] as List);
    for (int j = 0; j < ri.length; j++) {
      ri[j] = (si[j] as num).toDouble();
    }
  }
}

List<List<double>> _xavier(int out, int inp, int seed) {
  final r = math.Random(seed);
  final limit = math.sqrt(6.0 / (out + inp));
  return List.generate(out, (_) => List<double>.generate(inp, (_) => (r.nextDouble() * 2 - 1) * limit));
}

/* ------------------------------- policy IO (json) ------------------------------ */

Map<String, dynamic> _weightsToJson({
  required PolicyNetwork p,
  required FeatureExtractor fe,
  required eng.GameEngine env,
  RunningNorm? norm,
}) {
  String sig = _feSignature(
    groundSamples: fe.groundSamples,
    stridePx: fe.stridePx,
    worldW: env.cfg.worldW,
    worldH: env.cfg.worldH,
  );

  // trunk layers -> [{W,b}, ...]
  final trunkJson = <Map<String, dynamic>>[];
  for (final layer in p.trunk.layers) {
    trunkJson.add({
      'W': _deepCopyMat(layer.W),
      'b': List<double>.from(layer.b),
    });
  }

  Map<String, dynamic> headJson(layer) => {
    'W': _deepCopyMat(layer.W),
    'b': List<double>.from(layer.b),
  };

  final m = <String, dynamic>{
    'arch': {
      'input': p.inputSize,
      'hidden': p.hidden,
      'kIntents': PolicyNetwork.kIntents,
    },
    'trunk': trunkJson,
    'heads': {
      'intent': headJson(p.heads.intent),
      'turn': headJson(p.heads.turn),
      'thr': headJson(p.heads.thr),
      'val': headJson(p.heads.val),
    },
    'feature_extractor': {'groundSamples': fe.groundSamples, 'stridePx': fe.stridePx},
    'env_hint': {'worldW': env.cfg.worldW, 'worldH': env.cfg.worldH},
    'signature': sig,
    'format': 'v2', // mark new format
  };

  if (norm != null && norm.inited && norm.dim == p.inputSize) {
    m['norm'] = {
      'dim': norm.dim,
      'momentum': norm.momentum,
      'mean': norm.mean,
      'var': norm.var_,
      'signature': sig,
    };
    // legacy mirror (optional)
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

  int ok = 0, total = 0;

  // Prefer new v2 format
  final arch = (m['arch'] as Map?)?.cast<String, dynamic>();
  final trunkJ = (m['trunk'] as List?)?.cast<dynamic>();
  final headsJ = (m['heads'] as Map?)?.cast<String, dynamic>();

  if (arch != null && trunkJ != null && headsJ != null) {
    final hiddenFile = (arch['hidden'] as List?)?.map((e) => (e as num).toInt()).toList() ?? <int>[];
    final hiddenNow = p.hidden;

    if (hiddenFile.length != hiddenNow.length ||
        hiddenFile.asMap().entries.any((e) => e.value != hiddenNow[e.key])) {
      print(
          'Warning: Saved hidden sizes ${hiddenFile} don\'t match current ${hiddenNow}. Attempting best-effort load (shape-mismatch tensors skipped).');
    }

    // Trunk layers
    for (int li = 0; li < p.trunk.layers.length; li++) {
      if (li >= trunkJ.length) break;
      final layerObj = (trunkJ[li] as Map).cast<String, dynamic>();
      final Wj = layerObj['W'];
      final bj = layerObj['b'];
      final L = p.trunk.layers[li];

      // Shape guard
      if (Wj is List && Wj.isNotEmpty && (Wj[0] as List).length == L.W[0].length && Wj.length == L.W.length) {
        _assignMat(L.W, Wj);
        ok++;
      }
      if (bj is List && bj.length == L.b.length) {
        for (int i = 0; i < L.b.length; i++) L.b[i] = (bj[i] as num).toDouble();
        ok++;
      }
      total += 2;
    }

    // Heads
    bool _loadHead(Map<String, dynamic>? hj, List<List<double>> W, List<double> b) {
      if (hj == null) return false;
      bool any = false;
      final Wj = hj['W'];
      final bj = hj['b'];
      if (Wj is List && Wj.isNotEmpty && (Wj[0] as List).length == W[0].length && Wj.length == W.length) {
        _assignMat(W, Wj);
        ok++;
        any = true;
      }
      if (bj is List && bj.length == b.length) {
        for (int i = 0; i < b.length; i++) b[i] = (bj[i] as num).toDouble();
        ok++;
        any = true;
      }
      total += 2;
      return any;
    }

    _loadHead((headsJ['intent'] as Map?)?.cast<String, dynamic>(), p.heads.intent.W, p.heads.intent.b);
    _loadHead((headsJ['turn'] as Map?)?.cast<String, dynamic>(), p.heads.turn.W, p.heads.turn.b);
    _loadHead((headsJ['thr'] as Map?)?.cast<String, dynamic>(), p.heads.thr.W, p.heads.thr.b);
    _loadHead((headsJ['val'] as Map?)?.cast<String, dynamic>(), p.heads.val.W, p.heads.val.b);
  } else {
    // Legacy fields not supported for deep MLP (W1/W2...)—skip with message
    print('Note: $path doesn\'t look like v2 format (trunk/heads). Skipping weight load.');
  }

  // Norm (guarded by signature)
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

  final tag = (ok == total) ? '' : '  (PARTIAL)';
  print('Loaded policy ← $path ($ok/$total tensors filled)$tag');
  return ok > 0;
}

/* --------------------------------- env config --------------------------------- */

et.EngineConfig makeConfig({
  int seed = 42,
  bool lockTerrain = true,
  bool lockSpawn = true,
  bool randomSpawnX = false,
  double worldW = 800,
  double worldH = 600,
  double? maxFuel,
  bool crashOnTilt = false,
}) {
  final t = et.Tunables(
    gravity: 0.18,
    thrustAccel: 0.42,
    rotSpeed: 1.6,
    maxFuel: maxFuel ?? 1000.0,
    crashOnTilt: crashOnTilt,
    landingMaxVx: 28.0,
    landingMaxVy: 38.0,
    landingMaxOmega: 3.5,
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
      case 1: // goLeft
        x  = (padCx - (0.22 + 0.18*r.nextDouble()) * env.cfg.worldW)
            .clamp(10.0, env.cfg.worldW - 10.0);
        h  = 120 + 120*r.nextDouble();
        vx = (r.nextDouble()*14.0) - 7.0;
        vy = 28.0 + 10.0*r.nextDouble();
        break;
      case 2: // goRight
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
      case 1: // goLeft
        x  = (padCx - (0.22 + 0.18*r.nextDouble()) * env.cfg.worldW)
            .clamp(10.0, env.cfg.worldW - 10.0);
        h  = 120 + 120*r.nextDouble();
        vx = (r.nextDouble()*14.0) - 7.0;
        vy = 28.0 + 10.0*r.nextDouble();
        break;
      case 2: // goRight
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
        decisionReturns.add(cache.v);
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

_PretrainIntentStats _mineAndFixDescendSlow({
  required Trainer trainer,
  required FeatureExtractor fe,
  required eng.GameEngine env,
  required PolicyNetwork policy,
  int N = 6000,
  int epochs = 2,
  double lr = 5e-4,
  double weight = 3.0,
  int seed = 42424,
}) {
  final r = math.Random(seed);
  env.reset(seed: 777);
  final xs = <List<double>>[];
  final ys = <int>[];

  while (xs.length < N) {
    final padCx = env.terrain.padCenter.toDouble();

    env.lander.pos.x = (padCx + (r.nextDouble() * 0.10 - 0.05) * env.cfg.worldW)
        .clamp(10.0, env.cfg.worldW - 10.0);
    final gy = env.terrain.heightAt(env.lander.pos.x);
    final height = 160 + r.nextDouble() * 240;
    env.lander.pos.y = (gy - height).clamp(0.0, env.cfg.worldH - 10.0);
    env.lander.vel.x = (r.nextDouble() * 18.0) - 9.0;
    env.lander.vel.y = 22.0 + r.nextDouble() * 20.0;
    env.lander.angle = 0.0;
    env.lander.fuel  = env.cfg.t.maxFuel;

    final yTeacher = predictiveIntentLabelAdaptive(env, baseTauSec: 1.0, minTauSec: 0.45, maxTauSec: 1.35);
    if (yTeacher != 3) continue;

    final x = fe.extract(env);

    final xN = trainer.norm?.normalize(x, update: false) ?? x;
    final (pred, _p, _c) = policy.actIntentGreedy(xN);

    final keep = (pred != 3) || (r.nextDouble() < 0.25);
    if (!keep) continue;

    xs.add(x);
    ys.add(3);
  }

  for (final x in xs) {
    try { (trainer.norm as dynamic).observe(x); } catch (_) { trainer.norm?.normalize(x, update: true); }
  }

  final idxs = List<int>.generate(xs.length, (i) => i)..shuffle(r);
  const B = 64;
  for (int ep = 0; ep < epochs; ep++) {
    idxs.shuffle(r);
    for (int off = 0; off < xs.length; off += B) {
      final end = math.min(off + B, xs.length);
      final caches = <ForwardCache>[];
      final labels = <int>[];
      final returns = <double>[];
      for (int i = off; i < end; i++) {
        final xi = trainer.norm?.normalize(xs[idxs[i]], update: false) ?? xs[idxs[i]];
        final (_pred, _p, cache) = policy.actIntentGreedy(xi);
        caches.add(cache);
        labels.add(3);
        returns.add(cache.v);
      }
      policy.updateFromEpisode(
        decisionCaches: caches,
        intentChoices: labels,
        decisionReturns: returns,
        alignLabels: labels,
        alignWeight: weight,
        lr: lr,
        entropyBeta: 0.0,
        valueBeta: 0.0,
        huberDelta: 1.0,
        intentMode: true,
      );
    }
  }

  int correct = 0;
  for (int i = 0; i < xs.length; i++) {
    final xi = trainer.norm?.normalize(xs[i], update: false) ?? xs[i];
    final (pred, _p, _c) = policy.actIntentGreedy(xi);
    if (pred == 3) correct++;
  }
  return _PretrainIntentStats(correct / xs.length, xs.length);
}

/* ------------------------------------ main ------------------------------------ */

List<int> _parseHiddenList(String? s, {List<int> fallback = const [64, 64]}) {
  if (s == null || s.trim().isEmpty) return List<int>.from(fallback);
  final parts = s.split(',').map((t) => t.trim()).where((t) => t.isNotEmpty).toList();
  final out = <int>[];
  for (final p in parts) {
    final v = int.tryParse(p);
    if (v != null && v > 0) out.add(v);
  }
  return out.isEmpty ? List<int>.from(fallback) : out;
}

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

  // hidden layers (comma-separated), e.g. --hidden=96,96,64
  final hidden = _parseHiddenList(args.getStr('hidden'), fallback: const [64, 64]);

  double bestMeanCost = double.infinity;
  int bestCostIter = -1;

  final cfg = makeConfig(
    seed: seed,
    lockTerrain: lockTerrain,
    lockSpawn: lockSpawn,
    randomSpawnX: randomSpawnX,
    maxFuel: maxFuel,
  );
  final env = eng.GameEngine(cfg);

  final fe = FeatureExtractor(groundSamples: 3, stridePx: 48);
  final policy = PolicyNetwork(inputSize: fe.inputSize, hidden: hidden, seed: seed);
  print('Loaded init policy. hidden=${policy.hidden} | FE(gs=${fe.groundSamples} stride=${fe.stridePx})');

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
  /*
  tryLoadPolicy(
    'policy_pretrained.json',
    policy,
    norm: trainer.norm,
    fe: fe,
    env: env,
    ignoreLoadedNorm: ignoreLoadedNorm,
  );
  */

  if (determinism) {
    env.reset(seed: 1234);
    final a = _probeDeterminism(env, maxSteps: 165);
    env.reset(seed: 1234);
    final b = _probeDeterminism(env, maxSteps: 165);
    final ok = (a.steps == b.steps) && ((a.cost - b.cost).abs() < 1e-6);
    print('Determinism probe: steps ${a.steps} vs ${b.steps} | cost ${a.cost.toStringAsFixed(6)} vs ${b.cost.toStringAsFixed(6)} => ${ok ? "OK" : "MISMATCH"}');
  }

  // Optionally reset action heads (turn/thr)
  if (resetActionHeads) {
    final hiddenDim = policy.trunk.layers.isEmpty
        ? policy.inputSize
        : policy.trunk.layers.last.b.length;
    policy.heads.thr.W = _xavier(1, hiddenDim, seed ^ 0xA1);
    policy.heads.thr.b = List<double>.filled(1, 0.0);
    policy.heads.turn.W = _xavier(3, hiddenDim, seed ^ 0xB2);
    policy.heads.turn.b = List<double>.filled(3, 0.0);
    print('Action heads reset (thr & turn).');
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

      final stDescFix = _pretrainDescendSlowTargeted(
        trainer: trainer,
        fe: fe,
        env: env,
        policy: policy,
        perBand: 2500,
        epochs: 2,
        lr: pretrainLr,
        weight: 3.5,
        seed: seed ^ 0xFACE,
      );
      print('DescendSlow targeted fix → acc=${(stDescFix.acc*100).toStringAsFixed(1)}% n=${stDescFix.n}');

      calibrateIntentBiasToTeacher(
        trainer: trainer, fe: fe, env: env, policy: policy,
        N: 6000, iters: 50, lr: 0.5, seed: seed ^ 0xCA1,
      );

      /*
      final mined = _mineAndFixDescendSlow(
        trainer: trainer, fe: fe, env: env, policy: policy,
        N: 6000, epochs: 2, lr: pretrainLr, weight: 3.0, seed: seed ^ 0xC0DE,
      );
      print('DescendSlow hard-neg fix → acc=${(mined.acc*100).toStringAsFixed(1)}% n=${mined.n}');
      */

      savePolicy('policy_pretrained.json', policy, fe, env, norm: trainer.norm);
    }

    if (pretrainActionsN > 0) {
      print('Note: --pretrain_actions_n=$pretrainActionsN requested, but this agent build has no separate action-pretrain. Skipping.');
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
      if (ev.meanCost < bestMeanCost) {
        bestMeanCost = ev.meanCost;
        bestCostIter = it + 1;
        savePolicy('policy_best_cost.json', policy, fe, env, norm: trainer.norm);
        print('★ New BEST by cost at iter ${it + 1}: meanCost=${ev.meanCost.toStringAsFixed(3)} → saved policy_best_cost.json');
      }
      savePolicy('policy_iter_${it + 1}.json', policy, fe, env, norm: trainer.norm);
    }
  }

  savePolicy('policy_final.json', policy, fe, env, norm: trainer.norm);
  print('Training done. Saved → policy_final.json');
}

void calibrateIntentBiasToTeacher({
  required Trainer trainer,
  required FeatureExtractor fe,
  required eng.GameEngine env,
  required PolicyNetwork policy,
  int N = 6000,
  int iters = 40,
  double lr = 0.35,
  int seed = 777123,
}) {
  final r = math.Random(seed);
  env.reset(seed: 4242);

  final K = PolicyNetwork.kIntents;
  final xsRaw = <List<double>>[];
  final tCounts = List<int>.filled(K, 0);

  // 1) Collect (raw) random snapshots + teacher labels
  for (int i = 0; i < N; i++) {
    final padCx = env.terrain.padCenter.toDouble();
    env.lander.pos.x = (padCx + (r.nextDouble()*400 - 200)).clamp(10.0, env.cfg.worldW - 10.0);
    final gy = env.terrain.heightAt(env.lander.pos.x);
    env.lander.pos.y = (gy - (60 + 300*r.nextDouble())).clamp(0.0, env.cfg.worldH - 10.0);
    env.lander.vel.x = r.nextDouble()*180 - 90;
    env.lander.vel.y = r.nextDouble()*140 + 10;
    env.lander.angle = 0.0;
    env.lander.fuel  = env.cfg.t.maxFuel;

    final y = predictiveIntentLabelAdaptive(env, baseTauSec: 1.0, minTauSec: 0.45, maxTauSec: 1.35);
    tCounts[y] += 1;

    xsRaw.add(fe.extract(env));
  }

  // 2) Build a temporary local norm on xsRaw (don’t touch trainer.norm)
  final tmpNorm = RunningNorm(fe.inputSize, momentum: 0.995);
  for (final x in xsRaw) { tmpNorm.normalize(x, update: true); }
  final xs = xsRaw.map((x) => tmpNorm.normalize(x, update: false)).toList();

  // 3) Teacher marginal
  final eps = 1e-6;
  final tMarg = List<double>.generate(K, (k) => (tCounts[k] + eps) / (N + K*eps));

  // 4) Fit intent biases only: b_k += lr * (log(t_k) - log(pMean_k))
  for (int it = 0; it < iters; it++) {
    final pSum = List<double>.filled(K, 0.0);
    for (final x in xs) {
      final (_pred, p, _cache) = policy.actIntentGreedy(x);
      for (int k = 0; k < K; k++) pSum[k] += p[k];
    }
    final pMean = pSum.map((s) => s / N).toList();
    for (int k = 0; k < K; k++) {
      final g = math.log(tMarg[k]) - math.log(pMean[k] + eps);
      policy.heads.intent.b[k] += lr * g; // << updated to new location
    }
  }

  print('Calibrated intent biases to teacher marginals on $N snapshots (local norm).');
}

_PretrainIntentStats _pretrainDescendSlowTargeted({
  required Trainer trainer,
  required FeatureExtractor fe,
  required eng.GameEngine env,
  required PolicyNetwork policy,
  int perBand = 2500,
  int epochs = 2,
  double lr = 5e-4,
  double weight = 3.5,
  int seed = 60606,
}) {
  final r = math.Random(seed);
  final xs = <List<double>>[];
  final ys = <int>[];

  env.reset(seed: 909090);

  int minedBand1 = 0, minedBand2 = 0;

  bool _tryPush() {
    final yTeach = predictiveIntentLabelAdaptive(env, baseTauSec: 1.0, minTauSec: 0.45, maxTauSec: 1.35);
    if (yTeach != 3) return false;

    final xRaw = fe.extract(env);
    final xN = trainer.norm?.normalize(xRaw, update: false) ?? xRaw;
    final (pred, _p, _c) = policy.actIntentGreedy(xN);

    if (pred == 3) return false;            // not hard
    if (pred != 0 && pred != 4) return false; // only hover/brakeUp confusions

    xs.add(xRaw);
    ys.add(3);
    return true;
  }

  // Band 1
  while (minedBand1 < perBand) {
    final padCx = env.terrain.padCenter.toDouble();
    final W = env.cfg.worldW;
    final H = env.cfg.worldH;

    final sign = (r.nextBool() ? 1.0 : -1.0);
    final dx = sign * (0.08 * W * r.nextDouble());
    final px = (padCx + dx).clamp(10.0, W - 10.0);
    final gy = env.terrain.heightAt(px);

    final height = 130.0 + r.nextDouble() * 90.0;
    final vy = 20.0 + r.nextDouble() * 12.0;
    final vx = (r.nextDouble() * 16.0) - 8.0;

    env.lander
      ..pos.x = px
      ..pos.y = (gy - height).clamp(0.0, H - 10.0)
      ..vel.x = vx
      ..vel.y = vy
      ..angle = 0.0
      ..fuel = env.cfg.t.maxFuel;

    if (_tryPush()) minedBand1++;
  }

  // Band 2
  while (minedBand2 < perBand) {
    final padCx = env.terrain.padCenter.toDouble();
    final W = env.cfg.worldW;
    final H = env.cfg.worldH;

    final sign = (r.nextBool() ? 1.0 : -1.0);
    final dx = sign * (0.08 * W * r.nextDouble());
    final px = (padCx + dx).clamp(10.0, W - 10.0);
    final gy = env.terrain.heightAt(px);

    final height = 200.0 + r.nextDouble() * 120.0;
    final vy = 36.0 + r.nextDouble() * 24.0;
    final vx = (r.nextDouble() * 16.0) - 8.0;

    env.lander
      ..pos.x = px
      ..pos.y = (gy - height).clamp(0.0, H - 10.0)
      ..vel.x = vx
      ..vel.y = vy
      ..angle = 0.0
      ..fuel = env.cfg.t.maxFuel;

    if (_tryPush()) minedBand2++;
  }

  for (final x in xs) {
    try { (trainer.norm as dynamic).observe(x); } catch (_) { trainer.norm?.normalize(x, update: true); }
  }

  final N = xs.length;
  final idx = List<int>.generate(N, (i) => i)..shuffle(r);
  const B = 64;
  for (int ep = 0; ep < epochs; ep++) {
    idx.shuffle(r);
    for (int off = 0; off < N; off += B) {
      final end = math.min(off + B, N);
      final caches = <ForwardCache>[];
      final labels = <int>[];
      final returns = <double>[];

      for (int i = off; i < end; i++) {
        final xi = trainer.norm?.normalize(xs[idx[i]], update: false) ?? xs[idx[i]];
        final (_pred, _p, cache) = policy.actIntentGreedy(xi);
        caches.add(cache);
        labels.add(3);
        returns.add(cache.v);
      }

      policy.updateFromEpisode(
        decisionCaches: caches,
        intentChoices: labels,
        decisionReturns: returns,
        alignLabels: labels,
        alignWeight: weight,
        lr: lr,
        entropyBeta: 0.0,
        valueBeta: 0.0,
        huberDelta: 1.0,
        intentMode: true,
      );
    }
  }

  int correct = 0;
  for (int i = 0; i < N; i++) {
    final xi = trainer.norm?.normalize(xs[i], update: false) ?? xs[i];
    final (pred, _p, _c) = policy.actIntentGreedy(xi);
    if (pred == 3) correct++;
  }

  return _PretrainIntentStats(correct / N, N);
}

/* ----------------------------------- usage ------------------------------------

Pretrain only (intent; reset action heads; ignore old norm):
  dart run lib/ai/train_agent.dart \
    --hidden=64,64 \
    --pretrain_intent=10000 --pretrain_epochs=3 --pretrain_align=3.0 --pretrain_lr=0.0005 \
    --reset_action_heads --ignore_loaded_norm --only_pretrain

Full train (after pretrain):
  dart run lib/ai/train_agent.dart \
    --hidden=96,96,64 \
    --train_iters=200 --batch=32 --lr=0.0003 \
    --plan_hold=1 --use_learned_controller --blend_policy=0.75 \
    --value_beta=0.7

-------------------------------------------------------------------------------- */
