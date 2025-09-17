// lib/ai/train_agent.dart
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import '../engine/types.dart' as et;
import '../engine/game_engine.dart' as eng;
import '../engine/raycast.dart' as rc; // RayConfig, RayHitKind

import 'agent.dart'; // FeatureExtractor (rays), PolicyNetwork, Trainer, etc.

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
  required int inputSize,
  required int rayCount,
  required bool kindsOneHot,
  required double worldW,
  required double worldH,
}) {
  return 'kind=rays;in=$inputSize;rays=$rayCount;1hot=$kindsOneHot;W=${worldW.toInt()};H=${worldH.toInt()}';
}

/* ------------------------------- norm update helper ----------------------------- */

void _normUpdate(RunningNorm? norm, List<double> x) {
  if (norm == null) return;
  try {
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

/* ------------------------------ rays-only teacher ------------------------------ */

/// Find the nearest **pad** ray and return its **ship-frame** angle and normalized distance.
/// Returns (present, angle, distNorm).
(bool present, double angle, double distNorm) _nearestPadRayPolar(eng.GameEngine env) {
  final rays = env.rays;
  if (rays.isEmpty) return (false, 0.0, 1.0);

  final L = env.lander;
  final cfg = env.cfg;
  final maxD = math.sqrt(cfg.worldW * cfg.worldW + cfg.worldH * cfg.worldH);

  bool any = false;
  double bestD = 1e30, bestAngleShip = 0.0;

  for (final h in rays) {
    if (h.kind != rc.RayHitKind.pad) continue;

    final dx = h.p.x - L.pos.x;
    final dy = h.p.y - L.pos.y;
    final d  = math.sqrt(dx*dx + dy*dy);
    if (d < bestD) {
      bestD = d;
      final worldAng = math.atan2(dy, dx);
      // ship-frame: subtract yaw, wrap to [-pi, pi]
      double a = worldAng - L.angle;
      while (a <= -math.pi) a += 2 * math.pi;
      while (a >   math.pi) a -= 2 * math.pi;
      bestAngleShip = a;
      any = true;
    }
  }

  if (!any) return (false, 0.0, 1.0);
  return (true, bestAngleShip, (bestD / maxD).clamp(0.0, 1.0));
}

/// Intent teacher that uses ONLY **pad rays** + kinematics (no pad center / terrain samples).
int _teacherIntentFromRays(eng.GameEngine env) {
  final L = env.lander;
  final (_hasPad, a, dN) = _nearestPadRayPolar(env);

  // Emergency brake if screaming downwards regardless of pad rays.
  if (L.vel.y > 55.0) return 4; // brakeUp

  if (!_hasPad) {
    // No pad visible: bias to hover/slow descent based on vertical speed.
    if (L.vel.y > 18.0) return 4;     // brakeUp if too fast
    if (L.vel.y > 8.0)  return 3;     // descendSlow
    return 0;                          // hover
  }

  // Angle gates in ship-frame:
  const angDead = 12 * math.pi / 180;   // ~12°
  const farGate = 0.30;                 // >30% of screen diag = "far"

  if (a > angDead)  return 1; // goLeft  (pad is to left of nose)
  if (a < -angDead) return 2; // goRight (pad is to right of nose)

  // Angle aligned: choose descend/hover based on distance & vertical speed
  if (dN > farGate) {
    // far: close laterally first by gentle descend
    return 3; // descendSlow
  } else {
    // near: hold hover unless falling too fast
    if (L.vel.y > 16.0) return 3; // descendSlow to manage rate
    return 0;                     // hover
  }
}

/* ------------------------------- policy IO (json) ------------------------------ */

Map<String, dynamic> _weightsToJson({
  required PolicyNetwork p,
  required int rayCount,
  required bool kindsOneHot,
  required eng.GameEngine env,
  RunningNorm? norm,
}) {
  final sig = _feSignature(
    inputSize: p.inputSize,
    rayCount: rayCount,
    kindsOneHot: kindsOneHot,
    worldW: env.cfg.worldW,
    worldH: env.cfg.worldH,
  );

  final trunkJson = <Map<String, dynamic>>[];
  for (final layer in p.trunk.layers) {
    trunkJson.add({'W': _deepCopyMat(layer.W), 'b': List<double>.from(layer.b)});
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
    'feature_extractor': {
      'kind': 'rays',
      'rayCount': rayCount,
      'kindsOneHot': kindsOneHot,
    },
    'env_hint': {'worldW': env.cfg.worldW, 'worldH': env.cfg.worldH},
    'signature': sig,
    'format': 'v2rays',
  };

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

void savePolicy(
    String path,
    PolicyNetwork p, {
      required int rayCount,
      required bool kindsOneHot,
      required eng.GameEngine env,
      RunningNorm? norm,
    }) {
  final f = File(path);
  final jsonMap = _weightsToJson(
    p: p,
    rayCount: rayCount,
    kindsOneHot: kindsOneHot,
    env: env,
    norm: norm,
  );
  f.writeAsStringSync(const JsonEncoder.withIndent('  ').convert(jsonMap));
  print('Saved policy → $path');
}

bool tryLoadPolicy(
    String path,
    PolicyNetwork p, {
      RunningNorm? norm,
      required int rayCount,
      required bool kindsOneHot,
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
    print('Unexpected JSON root (expected object). Skipping load.');
    return false;
  }
  final m = raw;

  int ok = 0, total = 0;

  final arch = (m['arch'] as Map?)?.cast<String, dynamic>();
  final trunkJ = (m['trunk'] as List?)?.cast<dynamic>();
  final headsJ = (m['heads'] as Map?)?.cast<String, dynamic>();

  if (arch != null && trunkJ != null && headsJ != null) {
    // Trunk
    for (int li = 0; li < p.trunk.layers.length; li++) {
      if (li >= trunkJ.length) break;
      final layerObj = (trunkJ[li] as Map).cast<String, dynamic>();
      final Wj = layerObj['W'];
      final bj = layerObj['b'];
      final L = p.trunk.layers[li];

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
    print('Note: $path does not look like v2 format (trunk/heads). Skipping weight load.');
  }

  // Norm (guarded by signature)
  if (!ignoreLoadedNorm && norm != null && env != null) {
    final sigNow = _feSignature(
      inputSize: p.inputSize,
      rayCount: rayCount,
      kindsOneHot: kindsOneHot,
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
        try {
          final n = math.min(norm.dim, math.min(nm.length, nv.length));
          norm.mean = List<double>.generate(n, (i) => (nm[i] as num).toDouble())..length = norm.dim;
          norm.var_ = List<double>.generate(n, (i) => (nv[i] as num).toDouble())..length = norm.dim;
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
    if (env.status == et.GameStatus.landed) {
      landed++;
    } else {
      crashed++;
    }

    final padCx = env.terrain.padCenter;
    absDxSum += (env.lander.pos.x - padCx).abs();
  }

  costs.sort();
  final st = EvalStats();
  st.meanCost = costs.isEmpty ? 0 : costs.reduce((a, b) => a + b) / costs.length;
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
  required FeatureExtractorRays fe,
  required eng.GameEngine env,
  int perClass = 600,
  int seed = 4242,
}) {
  if (norm == null) return;
  final r = math.Random(seed);
  env.reset(seed: 1234567);
  int accepted = 0, target = perClass * PolicyNetwork.kIntents;

  while (accepted < target) {
    // randomize a state (don’t rely on pad center in the features)
    final W = env.cfg.worldW;
    final H = env.cfg.worldH;
    env.lander
      ..pos.x = (r.nextDouble() * (W - 20.0) + 10.0)
      ..pos.y = (r.nextDouble() * (H * 0.75)).clamp(0.0, H - 10.0)
      ..vel.x = r.nextDouble() * 160.0 - 80.0
      ..vel.y = r.nextDouble() * 120.0 + 10.0
      ..angle = 0.0
      ..fuel  = env.cfg.t.maxFuel;

    final y = _teacherIntentFromRays(env);
    // keep class balance by sampling “want” at random
    if (r.nextInt(PolicyNetwork.kIntents) != y) continue;

    final feat = fe.extract(env);
    _normUpdate(trainer.norm, feat);
    accepted++;
  }
  print('Feature norm warmed with $accepted synthetic samples (rays-teacher).');
}

/* ---------------------------- intent pretrain (local) -------------------------- */

class _PretrainIntentStats {
  final double acc; final int n;
  _PretrainIntentStats(this.acc, this.n);
}

_PretrainIntentStats _pretrainIntentLocal({
  required Trainer trainer,
  required FeatureExtractorRays fe,
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

  final xsByK = List.generate(K, (_) => <List<double>>[]);
  final ysByK = List.generate(K, (_) => <int>[]);

  env.reset(seed: 123456);

  bool _synthOneFor(int intentIdx, math.Random r) {
    // randomize around screen (no pad/terrain dependence in features)
    final W = env.cfg.worldW;
    final H = env.cfg.worldH;

    env.lander
      ..pos.x = (r.nextDouble() * (W - 20.0) + 10.0)
      ..pos.y = (r.nextDouble() * (H * 0.80)).clamp(0.0, H - 10.0)
      ..vel.x = r.nextDouble() * 160.0 - 80.0
      ..vel.y = 10.0 + r.nextDouble() * 140.0
      ..angle = 0.0
      ..fuel  = env.cfg.t.maxFuel;

    final y = _teacherIntentFromRays(env);
    if (y != intentIdx) return false;

    final feat = fe.extract(env);
    xsByK[intentIdx].add(feat);
    ysByK[intentIdx].add(intentIdx);
    return true;
  }

  const maxAttemptsPerClass = 30000;
  for (int k = 0; k < K; k++) {
    int attempts = 0;
    while (xsByK[k].length < targetPerClass && attempts < maxAttemptsPerClass) {
      _synthOneFor(k, rng);
      attempts++;
    }
    if (xsByK[k].isEmpty) {
      // back-off: copy from a neighbor class
      final fallback = (k == 0 ? 3 : 0);
      while (xsByK[k].length < math.max(16, targetPerClass ~/ 4) &&
          xsByK[fallback].isNotEmpty) {
        xsByK[k].add(List<double>.from(xsByK[fallback][rng.nextInt(xsByK[fallback].length)]));
        ysByK[k].add(k);
      }
    }
  }

  // Oversample partially-filled classes
  for (int k = 0; k < K; k++) {
    if (xsByK[k].isEmpty) continue;
    final need = targetPerClass - xsByK[k].length;
    for (int i = 0; i < need; i++) {
      final src = xsByK[k][rng.nextInt(xsByK[k].length)];
      xsByK[k].add(List<double>.from(src));
      ysByK[k].add(k);
    }
  }

  final xs = <List<double>>[];
  final ys = <int>[];
  for (int k = 0; k < K; k++) {
    xs.addAll(xsByK[k]);
    ys.addAll(ysByK[k]);
  }
  final N = xs.length;
  final perm = List<int>.generate(N, (i) => i)..shuffle(rng);

  // Warm/freeze norm on this data
  for (final x in xs) {
    _normUpdate(trainer.norm, x);
  }

  // Simple CE training on intent head
  const B = 64;
  for (int ep = 0; ep < epochs; ep++) {
    perm.shuffle(rng);
    for (int off = 0; off < N; off += B) {
      final end = math.min(off + B, N);
      final decisionCaches = <ForwardCache>[];
      final intentChoices = <int>[];
      final decisionReturns = <double>[];
      final alignLabels = <int>[];

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
    print('${kIntentNames[r].padRight(12)} | $row   (acc=${(rowAcc * 100).toStringAsFixed(1)}%  n=${counts[r]})');
  }

  return _PretrainIntentStats(acc, N);
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

  // Ensure rays active
  env.rayCfg = const rc.RayConfig(
    rayCount: 180,
    includeFloor: false,
    forwardAligned: true,
  );

  // Build FE and probe actual feature length
  final fe = FeatureExtractorRays(rayCount:180, forwardAligned:true); // rays-based FE in agent.dart
  env.reset(seed: seed ^ 0xC0FFEE);
  env.step(1 / 60.0, const et.ControlInput(thrust: false, left: false, right: false));

  final rayCount = env.rayCfg.rayCount;
  final x0 = fe.extract(env);
  final inDim = x0.length;

  final kindsOneHot = (inDim == 6 + rayCount * 4);
  final expected726 = 6 + rayCount * 4;
  final expectedScalar = 6 + rayCount;

  if (fe.inputSize != inDim) {
    print('Note: FE.inputSize=${fe.inputSize} vs actual=$inDim (expected $expected726 or $expectedScalar); continuing.');
  }

  final policy = PolicyNetwork(inputSize: inDim, hidden: hidden, seed: seed);
  print('Loaded init policy. hidden=${policy.hidden} | FE(kind=rays, in=$inDim, rays=$rayCount, oneHot=$kindsOneHot)');

  final useNorm = (fe.inputSize == inDim);

  final trainer = Trainer(
    env: env,
    fe: fe,
    policy: policy,
    dt: 1 / 60.0,
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
    normalizeFeatures: useNorm,
  );

  if (!useNorm) {
    print('Feature norm disabled (trainer.norm dim=${fe.inputSize} != feature len=$inDim).');
  }

  // Optional: load existing weights
  /*
  tryLoadPolicy(
    'policy_pretrained.json',
    policy,
    norm: useNorm ? trainer.norm : null,
    rayCount: rayCount,
    kindsOneHot: kindsOneHot,
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

  if (resetActionHeads) {
    final hiddenDim = policy.trunk.layers.isEmpty ? policy.inputSize : policy.trunk.layers.last.b.length;
    policy.heads.thr.W = _xavier(1, hiddenDim, seed ^ 0xA1);
    policy.heads.thr.b = List<double>.filled(1, 0.0);
    policy.heads.turn.W = _xavier(3, hiddenDim, seed ^ 0xB2);
    policy.heads.turn.b = List<double>.filled(3, 0.0);
    print('Action heads reset (thr & turn).');
  }

  // ===== PRETRAIN =====
  if (pretrainAll || onlyPretrain || pretrainN > 0) {
    if (useNorm && (ignoreLoadedNorm || !((trainer.norm?.inited) ?? false))) {
      _resetFeatureNorm(trainer.norm);
      _warmFeatureNorm(
        norm: trainer.norm,
        trainer: trainer,
        fe: fe,
        env: env,
        perClass: 700,
        seed: seed ^ 0xACE,
      );
    }

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
      print('Intent pretrain → acc=${(st.acc * 100).toStringAsFixed(1)}% n=${st.n}');

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
      print('DescendSlow targeted fix → acc=${(stDescFix.acc * 100).toStringAsFixed(1)}% n=${stDescFix.n}');

      calibrateIntentBiasToTeacher(
        trainer: trainer,
        fe: fe,
        env: env,
        policy: policy,
        N: 6000,
        iters: 50,
        lr: 0.5,
        seed: seed ^ 0xCA1,
      );

      savePolicy(
        'policy_pretrained.json',
        policy,
        rayCount: rayCount,
        kindsOneHot: kindsOneHot,
        env: env,
        norm: useNorm ? trainer.norm : null,
      );
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
        savePolicy(
          'policy_best_cost.json',
          policy,
          rayCount: rayCount,
          kindsOneHot: kindsOneHot,
          env: env,
          norm: useNorm ? trainer.norm : null,
        );
        print('★ New BEST by cost at iter ${it + 1}: meanCost=${ev.meanCost.toStringAsFixed(3)} → saved policy_best_cost.json');
      }
      savePolicy(
        'policy_iter_${it + 1}.json',
        policy,
        rayCount: rayCount,
        kindsOneHot: kindsOneHot,
        env: env,
        norm: useNorm ? trainer.norm : null,
      );
    }
  }

  savePolicy(
    'policy_final.json',
    policy,
    rayCount: rayCount,
    kindsOneHot: kindsOneHot,
    env: env,
    norm: useNorm ? trainer.norm : null,
  );
  print('Training done. Saved → policy_final.json');
}

/* ------------------------ bias calibration & targeted fix ---------------------- */

void calibrateIntentBiasToTeacher({
  required Trainer trainer,
  required FeatureExtractorRays fe,
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

  for (int i = 0; i < N; i++) {
    final W = env.cfg.worldW;
    final H = env.cfg.worldH;
    env.lander
      ..pos.x = (r.nextDouble() * (W - 20.0) + 10.0)
      ..pos.y = (r.nextDouble() * (H * 0.80)).clamp(0.0, H - 10.0)
      ..vel.x = r.nextDouble() * 180 - 90
      ..vel.y = r.nextDouble() * 140 + 10
      ..angle = 0.0
      ..fuel  = env.cfg.t.maxFuel;

    final y = _teacherIntentFromRays(env);
    tCounts[y] += 1;

    xsRaw.add(fe.extract(env));
  }

  final tmpNorm = RunningNorm(policy.inputSize, momentum: 0.995);
  for (final x in xsRaw) {
    tmpNorm.normalize(x, update: true);
  }
  final xs = xsRaw.map((x) => tmpNorm.normalize(x, update: false)).toList();

  final eps = 1e-6;
  final tMarg = List<double>.generate(K, (k) => (tCounts[k] + eps) / (N + K * eps));

  for (int it = 0; it < iters; it++) {
    final pSum = List<double>.filled(K, 0.0);
    for (final x in xs) {
      final (_pred, p, _cache) = policy.actIntentGreedy(x);
      for (int k = 0; k < K; k++) pSum[k] += p[k];
    }
    final pMean = pSum.map((s) => s / N).toList();
    for (int k = 0; k < K; k++) {
      final g = math.log(tMarg[k]) - math.log(pMean[k] + eps);
      policy.heads.intent.b[k] += lr * g;
    }
  }

  print('Calibrated intent biases to teacher marginals on $N snapshots (rays-teacher).');
}

_PretrainIntentStats _pretrainDescendSlowTargeted({
  required Trainer trainer,
  required FeatureExtractorRays fe,
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

  env.reset(seed: 909090);

  bool _tryPush() {
    final yTeach = _teacherIntentFromRays(env);
    if (yTeach != 3) return false; // only mine true descendSlow

    final xRaw = fe.extract(env);
    final xN = trainer.norm?.normalize(xRaw, update: false) ?? xRaw;
    final (pred, _p, _c) = policy.actIntentGreedy(xN);

    if (pred == 3) return false;            // not hard
    // keep classic confusions: hover/brakeUp
    if (pred != 0 && pred != 4) return false;

    xs.add(xRaw);
    return true;
  }

  int minedBand1 = 0, minedBand2 = 0;

  // Band 1 (near-ish to anywhere on screen)
  while (minedBand1 < perBand) {
    final W = env.cfg.worldW;
    final H = env.cfg.worldH;

    env.lander
      ..pos.x = (r.nextDouble() * (W - 20.0) + 10.0)
      ..pos.y = (H * 0.25 + r.nextDouble() * H * 0.25).clamp(0.0, H - 10.0)
      ..vel.x = r.nextDouble() * 16.0 - 8.0
      ..vel.y = 20.0 + r.nextDouble() * 12.0
      ..angle = 0.0
      ..fuel  = env.cfg.t.maxFuel;

    if (_tryPush()) minedBand1++;
  }

  // Band 2 (higher & faster)
  while (minedBand2 < perBand) {
    final W = env.cfg.worldW;
    final H = env.cfg.worldH;

    env.lander
      ..pos.x = (r.nextDouble() * (W - 20.0) + 10.0)
      ..pos.y = (H * 0.45 + r.nextDouble() * H * 0.35).clamp(0.0, H - 10.0)
      ..vel.x = r.nextDouble() * 16.0 - 8.0
      ..vel.y = 36.0 + r.nextDouble() * 24.0
      ..angle = 0.0
      ..fuel  = env.cfg.t.maxFuel;

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
        labels.add(3);            // descendSlow
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

Train (no intent pretrain, norm auto-disabled if sizes mismatch):
  dart run lib/ai/train_agent.dart \
    --hidden=64,64 \
    --train_iters=400 --batch=1 --lr=0.0003 --plan_hold=1 --blend_policy=1.0

Intent pretrain:
  dart run lib/ai/train_agent.dart \
    --hidden=64,64 \
    --pretrain_intent=10000 --pretrain_epochs=3 --pretrain_align=3.0 --pretrain_lr=0.0005 \
    --reset_action_heads --only_pretrain

Full train after pretrain:
  dart run lib/ai/train_agent.dart \
    --hidden=96,96,64 \
    --train_iters=200 --batch=32 --lr=0.0003 \
    --plan_hold=1 --blend_policy=0.75 --value_beta=0.7
-------------------------------------------------------------------------------- */
