// lib/ai/train_agent.dart
import 'dart:convert';
import 'dart:io';
import 'dart:isolate';
import 'dart:math' as math;

import '../engine/types.dart' as et;
import '../engine/game_engine.dart' as eng;
import '../engine/raycast.dart'; // RayConfig

import 'agent.dart' as ai; // FeatureExtractorRays, PolicyNetwork, Trainer, RunningNorm, kIntentNames, predictiveIntentLabelAdaptive
import 'agent.dart';
import 'curriculum/final_approach.dart';
import 'nn_helper.dart' as nn;
import 'potential_field.dart'; // buildPotentialField, PotentialField

// MODULAR CURRICULA
import 'curriculum/core.dart';
import 'curriculum/speed_min.dart';
import 'curriculum/hard_approach.dart';

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
  bool getFlag(String k, {bool def = false}) {
    if (_flags.contains(k)) return true;            // --flag
    final v = _kv[k];                                // --flag=true / --flag=false
    if (v == null) return def;
    final s = v.toLowerCase();
    return s == '1' || s == 'true' || s == 'yes' || s == 'on';
  }

  Map<String, String?> get kv => _kv;
  Set<String> get flags => _flags;
}

int _iclamp(int v, int lo, int hi) => v < lo ? lo : (v > hi ? hi : v);

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

/* ------------------------------- matrix helpers -------------------------------- */

List<List<double>> _deepCopyMat(List<List<double>> W) =>
    List.generate(W.length, (i) => List<double>.from(W[i]));

List<List<double>> _xavier(int out, int inp, int seed) {
  final r = math.Random(seed);
  final limit = math.sqrt(6.0 / (out + inp));
  return List.generate(out, (_) => List<double>.generate(inp, (_) => (r.nextDouble() * 2 - 1) * limit));
}

/* ------------------------------- policy IO (json) ------------------------------ */

Map<String, dynamic> _policyToJson({
  required PolicyNetwork p,
  required int rayCount,
  required bool kindsOneHot,
  required eng.GameEngine env,
  RunningNorm? norm,
  // NEW: persist runtime hints so runtime_policy can pick sane defaults
  bool fixPolarityWithPadRays = true,
  double runtimeIntentTemp = 1.0,
}) {
  final sig = _feSignature(
    inputSize: p.inputSize,
    rayCount: rayCount,
    kindsOneHot: kindsOneHot,
    worldW: env.cfg.worldW,
    worldH: env.cfg.worldH,
  );

  Map<String, dynamic> headJson(layer) => {
    'W': _deepCopyMat(layer.W),
    'b': List<double>.from(layer.b),
  };

  final trunkJson = <Map<String, dynamic>>[];
  for (final layer in p.trunk.layers) {
    trunkJson.add({'W': _deepCopyMat(layer.W), 'b': List<double>.from(layer.b)});
  }

  final t = env.cfg.t;
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

    // Persist physics knobs so runtime can mirror them
    'physics': {
      'gravity': t.gravity,
      'thrustAccel': t.thrustAccel,
      'rotSpeed': t.rotSpeed,
      'maxFuel': t.maxFuel,

      'rcsEnabled': t.rcsEnabled,
      'rcsAccel': t.rcsAccel,
      'rcsBodyFrame': t.rcsBodyFrame,

      'downThrEnabled': t.downThrEnabled,
      'downThrAccel': t.downThrAccel,
      'downThrBurn': t.downThrBurn,
    },

    // NEW: give runtime defaults that match what we want at inference
    'mode_hints': {
      'fixPolarityWithPadRays': fixPolarityWithPadRays,
      'intentTemp': runtimeIntentTemp,
    },

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
    // legacy mirrors
    m['norm_mean'] = norm.mean;
    m['norm_var'] = norm.var_;
    m['norm_momentum'] = norm.momentum;
    m['norm_signature'] = sig;
  }

  return m;
}

void _savePolicy({
  required String path,
  required PolicyNetwork p,
  required int rayCount,
  required bool kindsOneHot,
  required eng.GameEngine env,
  RunningNorm? norm,
}) {
  final f = File(path);
  final jsonMap = _policyToJson(
    p: p,
    rayCount: rayCount,
    kindsOneHot: kindsOneHot,
    env: env,
    norm: norm,
    fixPolarityWithPadRays: true,
    runtimeIntentTemp: 1.0,
  );
  f.writeAsStringSync(const JsonEncoder.withIndent('  ').convert(jsonMap));
  print('Saved policy → $path');
}

/* ------------------------------ policy LOADER (v2) ----------------------------- */

void _loadPolicyIntoNetwork({
  required String path,
  required PolicyNetwork target,
  required eng.GameEngine env,
  RunningNorm? norm,
}) {
  final raw = File(path).readAsStringSync();
  final j = json.decode(raw) as Map<String, dynamic>;

  // Expect v2 format
  final arch = (j['arch'] as Map?)?.cast<String, dynamic>();
  final trunkJ = (j['trunk'] as List?)?.cast<dynamic>();
  final headsJ = (j['heads'] as Map?)?.cast<String, dynamic>();
  if (arch == null || trunkJ == null || headsJ == null) {
    throw StateError('Loaded policy is not in v2 format (missing arch/trunk/heads).');
  }

  List<List<double>> _as2d(dynamic v) =>
      (v as List).map<List<double>>((r) => (r as List).map<double>((x) => (x as num).toDouble()).toList()).toList();
  List<double> _as1d(dynamic v) => (v as List).map<double>((x) => (x as num).toDouble()).toList();

  final inDim = (arch['input'] as num).toInt();
  if (inDim != target.inputSize) {
    throw StateError('Loaded policy input dim $inDim != runtime input ${target.inputSize}');
  }

  // Check hidden sizes match
  final hiddenLoad = ((arch['hidden'] as List?) ?? const []).map((e) => (e as num).toInt()).toList();
  if (hiddenLoad.length != target.hidden.length ||
      !List.generate(hiddenLoad.length, (i) => hiddenLoad[i] == target.hidden[i]).every((x) => x)) {
    throw StateError('Loaded hidden sizes $hiddenLoad != runtime ${target.hidden}');
  }

  // Fill trunk
  if (trunkJ.length != target.trunk.layers.length) {
    throw StateError('Loaded trunk layers=${trunkJ.length} != runtime ${target.trunk.layers.length}');
  }
  for (int li = 0; li < trunkJ.length; li++) {
    final obj = (trunkJ[li] as Map).cast<String, dynamic>();
    final W = _as2d(obj['W']);
    final B = _as1d(obj['b']);
    final L = target.trunk.layers[li];
    if (W.length != L.W.length || W[0].length != L.W[0].length || B.length != L.b.length) {
      throw StateError('Trunk layer $li shape mismatch.');
    }
    for (int i = 0; i < L.W.length; i++) {
      for (int j2 = 0; j2 < L.W[0].length; j2++) L.W[i][j2] = W[i][j2];
    }
    for (int i = 0; i < L.b.length; i++) L.b[i] = B[i];
  }

  // Heads
  void _loadHead(String name, var head) {
    final hj = (headsJ[name] as Map).cast<String, dynamic>();
    final W = _as2d(hj['W']);
    final B = _as1d(hj['b']);
    if (W.length != head.W.length || W[0].length != head.W[0].length || B.length != head.b.length) {
      throw StateError('Head "$name" shape mismatch.');
    }
    for (int i = 0; i < head.W.length; i++) {
      for (int j2 = 0; j2 < head.W[0].length; j2++) head.W[i][j2] = W[i][j2];
    }
    for (int i = 0; i < head.b.length; i++) head.b[i] = B[i];
  }

  _loadHead('intent', target.heads.intent);
  _loadHead('turn', target.heads.turn);
  _loadHead('thr', target.heads.thr);
  _loadHead('val', target.heads.val);

  // Norm (optional)
  final nm = (j['norm'] as Map?)?.cast<String, dynamic>();
  if (nm != null && norm != null) {
    final dim = (nm['dim'] as num?)?.toInt() ?? -1;
    if (dim == norm.dim) {
      final mean = (nm['mean'] as List).map((e) => (e as num).toDouble()).toList();
      final var_ = (nm['var'] as List).map((e) => (e as num).toDouble()).toList();
      if (mean.length == norm.dim && var_.length == norm.dim) {
        norm.mean = mean;
        norm.var_ = var_;
        norm.inited = true;
      }
    }
  }

  print('Loaded policy from $path (hidden=$hiddenLoad, inDim=$inDim).');
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

  // physics flags/values
  double gravity = 0.18,
  double thrustAccel = 0.42,
  double rotSpeed = 1.6,

  bool rcsEnabled = false,
  double rcsAccel = 0.12,
  bool rcsBodyFrame = true,

  bool downThrEnabled = false,
  double downThrAccel = 0.30,
  double downThrBurn = 10.0,
}) {
  final t = et.Tunables(
    gravity: gravity,
    thrustAccel: thrustAccel,
    rotSpeed: rotSpeed,
    maxFuel: maxFuel ?? 1000.0,
    crashOnTilt: crashOnTilt,
    landingMaxVx: 28.0,
    landingMaxVy: 38.0,
    landingMaxOmega: 3.5,

    // RCS
    rcsEnabled: rcsEnabled,
    rcsAccel: rcsAccel,
    rcsBodyFrame: rcsBodyFrame,

    // Downward thruster
    downThrEnabled: downThrEnabled,
    downThrAccel: downThrAccel,
    downThrBurn: downThrBurn,
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

/* ----------------------------- determinism probe ------------------------------ */

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

class _EvalChunkResult {
  final List<double> costs;
  final int landed;
  final int crashed;
  final int stepsSum;
  final double absDxSum;
  _EvalChunkResult(this.costs, this.landed, this.crashed, this.stepsSum, this.absDxSum);
}

({double lo, double hi}) _wilson95(int success, int n) {
  if (n <= 0) return (lo: 0, hi: 0);
  const z = 1.96;
  final p = success / n;
  final denom = 1 + z * z / n;
  final center = (p + z * z / (2 * n)) / denom;
  final half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom;
  return (lo: 100.0 * (center - half), hi: 100.0 * (center + half));
}

_EvalChunkResult _evalChunk({
  required int episodes,
  required int seed,
  required int attemptsPerTerrain,
  required PolicyNetwork policyClone,
  required List<int> hidden,
  required et.EngineConfig cfg,
  // match training/runtime exactly
  required int planHold,
  required double blendPolicy,
  required double tempIntent,
  required double intentEntropy,
  required bool evalDebug,
  required int evalDebugFailN,
}) {
  final env = eng.GameEngine(cfg);
  env.rayCfg = const RayConfig(rayCount: 180, includeFloor: false, forwardAligned: false);
  final fe = FeatureExtractorRays(rayCount: env.rayCfg.rayCount);

  final trainer = Trainer(
    env: env,
    fe: fe,
    policy: policyClone,
    dt: 1 / 60.0,
    gamma: 0.99,
    seed: seed,
    twoStage: true,
    planHold: planHold,
    tempIntent: tempIntent,
    intentEntropyBeta: intentEntropy,
    useLearnedController: false,
    blendPolicy: blendPolicy,
    intentAlignWeight: 0.0,
    intentPgWeight: 0.0,
    actionAlignWeight: 0.0,
    normalizeFeatures: true,
    gateScoreMin: -1e9,
    gateOnlyLanded: false,
    gateVerbose: false,
    externalRewardHook: null,
  );

  final rnd = math.Random(seed);
  final costs = <double>[];
  int landed = 0, crashed = 0, stepsSum = 0;
  double absDxSum = 0.0;

  int terrAttempts = 0;
  int currentTerrainSeed = rnd.nextInt(1 << 30);

  int printedFails = 0;

  for (int i = 0; i < episodes; i++) {
    if (terrAttempts == 0) {
      currentTerrainSeed = rnd.nextInt(1 << 30);
    }
    env.reset(seed: currentTerrainSeed);
    final res = trainer.runEpisode(train: false, greedy: false, scoreIsReward: false);
    terrAttempts = (terrAttempts + 1) % attemptsPerTerrain;

    costs.add(res.totalCost);
    stepsSum += res.steps;

    final ok = (env.status == et.GameStatus.landed);
    if (ok) {
      landed++;
    } else {
      crashed++;
      if (evalDebug && printedFails < evalDebugFailN) {
        final L = env.lander;
        final T = env.terrain;
        final gx = T.heightAt(L.pos.x.toDouble());
        final h  = (gx - L.pos.y).toDouble();
        print('[EVAL DBG FAIL] ep=$i terrSeed=$currentTerrainSeed '
            'status=${env.status} x=${L.pos.x.toStringAsFixed(1)} y=${L.pos.y.toStringAsFixed(1)} '
            'h=${h.toStringAsFixed(1)} vx=${L.vel.x.toStringAsFixed(1)} vy=${L.vel.y.toStringAsFixed(1)} '
            'fuel=${L.fuel.toStringAsFixed(1)} '
            'padCx=${T.padCenter.toStringAsFixed(1)} | cost=${res.totalCost.toStringAsFixed(3)} steps=${res.steps}');
        printedFails++;
      }
    }
    final padCx = env.terrain.padCenter;
    absDxSum += (env.lander.pos.x - padCx).abs();
  }

  return _EvalChunkResult(costs, landed, crashed, stepsSum, absDxSum);
}

Future<EvalStats> evaluateParallel({
  required et.EngineConfig cfg,
  required PolicyNetwork policy,
  required List<int> hidden,
  required int episodes,
  required int attemptsPerTerrain,
  required int seed,
  required int workers,
  // match training/runtime
  required int planHold,
  required double blendPolicy,
  required double tempIntent,
  required double intentEntropy,
  // debug
  required bool evalDebug,
  required int evalDebugFailN,
}) async {
  final per = episodes ~/ workers;
  final extra = episodes % workers;

  // clone params once for read-only use
  PolicyNetwork _clonePolicy(PolicyNetwork p) {
    final cp = PolicyNetwork(inputSize: p.inputSize, hidden: List<int>.from(hidden), seed: seed ^ 0xA11CE);
    // trunk
    for (int li = 0; li < cp.trunk.layers.length; li++) {
      final Ld = cp.trunk.layers[li];
      final Ls = p.trunk.layers[li];
      for (int i = 0; i < Ld.W.length; i++) {
        for (int j = 0; j < Ld.W[0].length; j++) {
          Ld.W[i][j] = Ls.W[i][j];
        }
      }
      for (int i = 0; i < Ld.b.length; i++) Ld.b[i] = Ls.b[i];
    }
    // heads
    List<List<double>> _copyW(List<List<double>> W) =>
        List.generate(W.length, (i) => List<double>.from(W[i]));
    cp.heads.intent.W = _copyW(p.heads.intent.W); cp.heads.intent.b = List<double>.from(p.heads.intent.b);
    cp.heads.turn.W   = _copyW(p.heads.turn.W);   cp.heads.turn.b   = List<double>.from(p.heads.turn.b);
    cp.heads.thr.W    = _copyW(p.heads.thr.W);    cp.heads.thr.b    = List<double>.from(p.heads.thr.b);
    cp.heads.val.W    = _copyW(p.heads.val.W);    cp.heads.val.b    = List<double>.from(p.heads.val.b);
    return cp;
  }

  final futures = <Future<_EvalChunkResult>>[];
  for (int w = 0; w < workers; w++) {
    final nThis = per + (w < extra ? 1 : 0);
    if (nThis == 0) continue;

    final pClone = _clonePolicy(policy);
    final seedW = seed ^ (0xBEEF << (w & 15));

    futures.add(Isolate.run(() {
      try {
        return _evalChunk(
          episodes: nThis,
          seed: seedW,
          attemptsPerTerrain: attemptsPerTerrain,
          policyClone: pClone,
          hidden: hidden,
          cfg: cfg,
          planHold: planHold,
          blendPolicy: blendPolicy,
          tempIntent: tempIntent,
          intentEntropy: intentEntropy,
          evalDebug: evalDebug,
          evalDebugFailN: evalDebugFailN,
        );
      } catch (e, st) {
        stderr.writeln('[EVAL WORKER ERROR] $e\n$st');
        return _EvalChunkResult(const [], 0, 0, 0, 0.0);
      }
    }));
  }

  final chunks = await Future.wait(futures);

  // reduce
  final costs = <double>[];
  int landed = 0, crashed = 0, stepsSum = 0;
  double absDxSum = 0.0;

  for (final c in chunks) {
    costs.addAll(c.costs);
    landed += c.landed;
    crashed += c.crashed;
    stepsSum += c.stepsSum;
    absDxSum += c.absDxSum;
  }

  costs.sort();
  final st = EvalStats();
  st.meanCost = costs.isEmpty ? 0 : costs.reduce((a, b) => a + b) / costs.length;
  st.medianCost = costs.isEmpty ? 0 : costs[costs.length ~/ 2];
  final total = landed + crashed;
  st.landPct = total == 0 ? 0 : 100.0 * landed / total;
  st.crashPct = total == 0 ? 0 : 100.0 * crashed / total;
  st.meanSteps = total == 0 ? 0 : stepsSum / total;
  st.meanAbsDx = total == 0 ? 0 : absDxSum / total;

  print('Eval: N=$episodes | workers=$workers | '
      '| land%=${st.landPct.toStringAsFixed(1)} '
      '| meanCost=${st.meanCost.toStringAsFixed(3)} '
      '| median=${st.medianCost.toStringAsFixed(3)} | steps=${st.meanSteps.toStringAsFixed(1)} '
      '| mean|dx|=${st.meanAbsDx.toStringAsFixed(1)}');

  return st;
}

EvalStats evaluateSequential({
  required eng.GameEngine env,
  required Trainer trainer,
  int episodes = 40,
  int seed = 123,
  int attemptsPerTerrain = 1,
  bool evalDebug = false,
  int evalDebugFailN = 3,
}) {
  final rnd = math.Random(seed);
  final costs = <double>[];
  int landed = 0, crashed = 0, stepsSum = 0;
  double absDxSum = 0.0;

  int terrAttempts = 0;
  int currentTerrainSeed = rnd.nextInt(1 << 30);

  int printedFails = 0;

  for (int i = 0; i < episodes; i++) {
    if (terrAttempts == 0) currentTerrainSeed = rnd.nextInt(1 << 30);
    env.reset(seed: currentTerrainSeed);

    final res = trainer.runEpisode(train: false, greedy: false, scoreIsReward: false);

    terrAttempts = (terrAttempts + 1) % attemptsPerTerrain;

    costs.add(res.totalCost);
    stepsSum += res.steps;
    if (env.status == et.GameStatus.landed) {
      landed++;
    } else {
      crashed++;
      if (evalDebug && printedFails < evalDebugFailN) {
        final L = env.lander;
        final T = env.terrain;
        final gx = T.heightAt(L.pos.x.toDouble());
        final h  = (gx - L.pos.y).toDouble();
        print('[EVAL DBG FAIL] ep=$i terrSeed=$currentTerrainSeed '
            'status=${env.status} x=${L.pos.x.toStringAsFixed(1)} y=${L.pos.y.toStringAsFixed(1)} '
            'h=${h.toStringAsFixed(1)} vx=${L.vel.x.toStringAsFixed(1)} vy=${L.vel.y.toStringAsFixed(1)} '
            'fuel=${L.fuel.toStringAsFixed(1)} '
            'padCx=${T.padCenter.toStringAsFixed(1)} | cost=${res.totalCost.toStringAsFixed(3)} steps=${res.steps}');
        printedFails++;
      }
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

  print('Eval: N=$episodes | workers=1 '
      '| land%=${st.landPct.toStringAsFixed(1)} '
      '| meanCost=${st.meanCost.toStringAsFixed(3)} '
      '| median=${st.medianCost.toStringAsFixed(3)} | steps=${st.meanSteps.toStringAsFixed(1)} '
      '| mean|dx|=${st.meanAbsDx.toStringAsFixed(1)}');

  return st;
}

/* ----------------------------- PF velocity+accel reward ------------------------ */

class PFShapingCfg {
  // --- Velocity shaping ---
  final double wAlign;       // reward per unit cos(v, flow)
  final double wVelDelta;    // penalty for ||v - v_pf|| / vmax
  final double vMinClose;
  final double vMaxFar;
  final double alpha;
  final double vmax;
  final double padTightFrac;
  final double hTight;
  final double latBoost;
  final double velPenaltyBoost;
  final double alignBoost;
  final double vMinTouchdown;
  final double feasiness;
  final double xBias;

  // --- Acceleration (Δv) matching to feasible PF Δv ---
  final double wAccAlign;    // reward for cos(dv_actual, dv_pf)
  final double wAccErr;      // penalty for ||dv_actual - dv_pf|| / dv_pf_cap
  final double accEma;       // EMA smoothing for dv_actual (helps noisy policies)

  // Debug
  final bool debug;

  // NEW: extra shaping weights (vy cap penalty & pad-toward reward)
  final double wVyCap;   // penalty weight for exceeding adaptive vy cap
  final double wPadTow;  // reward weight for lateral velocity toward pad

  const PFShapingCfg({
    // velocity
    this.wAlign = 1.0,
    this.wVelDelta = 0.6,
    this.vMinClose = 8.0,
    this.vMaxFar = 90.0,
    this.alpha = 1.2,
    this.vmax = 140.0,
    this.padTightFrac = 0.10,
    this.hTight = 140.0,
    this.latBoost = 4.0,
    this.velPenaltyBoost = 3.0,
    this.alignBoost = 1.5,
    this.vMinTouchdown = 2.0,
    this.feasiness = 0.75,
    this.xBias = 3.0,
    // acceleration
    this.wAccAlign = 2.0,
    this.wAccErr = 1.0,
    this.accEma = 0.2,
    // debug
    this.debug = false,
    // NEW extras
    this.wVyCap = 0.02,
    this.wPadTow = 0.12,
  });
}

ai.ExternalRewardHook makePFRewardHook({
  required eng.GameEngine env,
  PFShapingCfg cfg = const PFShapingCfg(),
}) {
  final pf = buildPotentialField(env, nx: 160, ny: 120, iters: 1200, omega: 1.7, tol: 1e-4);

  // Effective max linear accel per real second
  final double aMax = env.cfg.t.thrustAccel * 0.05 * env.cfg.stepScale; // px/s^2

  // --- Keep previous velocity to measure actual Δv ---
  double? prevVx, prevVy;
  double smDvX = 0.0, smDvY = 0.0; // EMA-smoothed dv (for stability)

  // Debug accumulators (persist across steps within an episode)
  int dbgN = 0;
  double sumAbsDVpfX = 0, sumAbsDVpfY = 0;
  double sumWLat = 0, sumWVer = 0;
  double sumAlignAbs = 0, sumVelPen = 0;

  return ({required eng.GameEngine env, required double dt, required int tStep}) {
    final x = env.lander.pos.x.toDouble();
    final y = env.lander.pos.y.toDouble();
    final vx = env.lander.vel.x.toDouble();
    final vy = env.lander.vel.y.toDouble();

    // Proximity to pad
    final padCx = env.terrain.padCenter.toDouble();
    final dx = x - padCx;
    final dxAbs = dx.abs();
    final W = env.cfg.worldW.toDouble();
    final gy = env.terrain.heightAt(x);
    final h  = (gy - y).toDouble().clamp(0.0, 1000.0);

    final tightX = cfg.padTightFrac * W;
    final px_ = math.exp(- (dxAbs*dxAbs) / (tightX*tightX + 1e-6));
    final ph_ = math.exp(- (h*h)       / (cfg.hTight*cfg.hTight + 1e-6));
    final prox = (px_ * ph_).clamp(0.0, 1.0);

    // Alignment to PF flow (directional)
    var flow = pf.sampleFlow(x, y);
    final vmag = math.sqrt(vx*vx + vy*vy);
    double align = 0.0;
    if (vmag > 1e-6) {
      align = (vx / vmag) * flow.nx + (vy / vmag) * flow.ny;
    }

    // Kill-down near pad
    if (prox > 0.65 && flow.fy < 0.0) {
      flow = (fx: flow.fx, fy: 0.0, nx: flow.nx, ny: 0.0, mag: flow.mag);
    }

    // Base PF target velocity (before feasibility clamp)
    var sugg = pf.suggestVelocity(
      x, y,
      vMinClose: cfg.vMinClose,
      vMaxFar: cfg.vMaxFar,
      alpha: cfg.alpha,
      clampSpeed: 9999.0,
    );

    // Flare toward touchdown
    final flareLat = (1.0 - 0.70 * prox);
    final flareVer = (1.0 - 0.45 * prox);
    final magNow = math.sqrt(sugg.vx*sugg.vx + sugg.vy*sugg.vy) + 1e-9;
    sugg = (vx: sugg.vx * flareLat, vy: sugg.vy * flareVer);

    final magTarget = ((1.0 - prox) * magNow + prox * cfg.vMinTouchdown).clamp(0.0, magNow);
    final magNew = math.sqrt(sugg.vx*sugg.vx + sugg.vy*sugg.vy) + 1e-9;
    final kMag = (magTarget / magNew).clamp(0.0, 1.0);
    sugg = (vx: sugg.vx * kMag, vy: sugg.vy * kMag);

    // --- Feasibility clamp (defines the *desired Δv* we want to match) ---
    final dv_pf_x_raw = sugg.vx - vx;
    final dv_pf_y_raw = sugg.vy - vy;
    final dv_pf_mag_raw = math.sqrt(dv_pf_x_raw*dv_pf_x_raw + dv_pf_y_raw*dv_pf_y_raw);

    final dv_pf_cap = (aMax * dt * cfg.feasiness).clamp(0.0, 1e9);
    double dv_pf_x = dv_pf_x_raw, dv_pf_y = dv_pf_y_raw;
    if (dv_pf_mag_raw > dv_pf_cap && dv_pf_mag_raw > 1e-9) {
      final s = dv_pf_cap / dv_pf_mag_raw;
      dv_pf_x *= s;
      dv_pf_y *= s;
      // And update "feasible" target velocity accordingly:
      sugg = (vx: vx + dv_pf_x, vy: vy + dv_pf_y);
    }

    // --- Border avoidance (X walls) ---
    final wallTau = 5.0;
    final wallMarginFrac = 0.22;
    final wallBlendMax = 0.80;
    final wallVInward = 0.90;
    final wallVelPenalty = 3.0;
    final Wworld = env.cfg.worldW.toDouble();
    final distL = x;
    final distR = (Wworld - x);
    final baseBand = wallMarginFrac * Wworld;

    final vxTowardL = (-vx).clamp(0.0, double.infinity);
    final vxTowardR = ( vx).clamp(0.0, double.infinity);

    final warnL = baseBand + wallTau * vxTowardL;
    final warnR = baseBand + wallTau * vxTowardR;

    double proxL = 1.0 - (distL / (warnL + 1e-6)).clamp(0.0, 1.0);
    double proxR = 1.0 - (distR / (warnR + 1e-6)).clamp(0.0, 1.0);

    final borderProx = math.max(proxL, proxR);
    double inwardX = 0.0;
    if (proxL >= proxR && proxL > 0.0) inwardX =  1.0;
    if (proxR >  proxL && proxR > 0.0) inwardX = -1.0;

    final gamma = 1.5;
    final blendIn = wallBlendMax * math.pow(borderProx, gamma);

    final vInward = (wallVInward + 0.8 * (vxTowardL + vxTowardR)).clamp(40.0, 200.0);
    final suggWallVx = inwardX * vInward;

    sugg = (
    vx: (1.0 - blendIn) * sugg.vx + blendIn * suggWallVx,
    vy: sugg.vy * (1.0 - 0.35 * borderProx)
    );

    // --- Ceiling avoidance (speed-aware) ---
    final double Hworld = env.cfg.worldH.toDouble();
    final double distTop = y;
    final double baseBandY = 0.35 * Hworld;
    final double tauY = 5.0;
    final double vyTowardTop = (-vy).clamp(0.0, double.infinity);
    final double warnTop = baseBandY + tauY * vyTowardTop;
    double proxTop = 1.0 - (distTop / (warnTop + 1e-6)).clamp(0.0, 1.0);
    if (vy > 0) proxTop *= 0.6;
    final double gammaY = 1.5;
    final double blendTop = 0.75 * math.pow(proxTop, gammaY);
    final double vDownward = (70.0 + 1.0 * vyTowardTop).clamp(40.0, 220.0);

    sugg = (
    vx: sugg.vx * (1.0 - 0.15 * proxTop),
    vy: (1.0 - blendTop) * sugg.vy + blendTop * vDownward
    );

    final double wallBoostY = 1.0 + 2.5 * proxTop;
    final wallBoost = 1.0 + wallVelPenalty * borderProx;

    // --- Velocity error term (with X bias near pad) ---
    final prox2 = prox * prox;
    final wLat = (cfg.xBias) * (1.0 + cfg.latBoost * prox2);
    final wVer = 1.0 * (1.0 + 0.7 * cfg.latBoost * prox2);

    final dvx_vel = (vx - sugg.vx) * wLat;
    final dvy_vel = (vy - sugg.vy) * wVer;
    final vErr = math.sqrt(dvx_vel*dvx_vel + dvy_vel*dvy_vel) / cfg.vmax;

    final wVelEff = cfg.wVelDelta
        * (1.0 + 0.6 * cfg.velPenaltyBoost * prox2)
        * wallBoost
        * wallBoostY;
    final wAlignEff = cfg.wAlign * (1.0 + cfg.alignBoost * prox);

    // Touchdown bonus (gated)
    final touchTarget = cfg.vMinTouchdown.clamp(0.5, 15.0);
    const double touchWeight = 3.0;
    final bool inTouchBand = (h < 90.0) && (dxAbs < 0.12 * W);
    final double vmagNow2 = math.sqrt(vx*vx + vy*vy) + 1e-9;
    final double touchBonus = inTouchBand
        ? touchWeight * (touchTarget - vmagNow2) / (touchTarget + 1e-6)
        : 0.0;

    // --- Acceleration (Δv) matching ---
    double rAcc = 0.0;
    if (prevVx != null && prevVy != null && dt > 0) {
      // Actual dv in this step
      double dvx_act = (vx - prevVx!);
      double dvy_act = (vy - prevVy!);

      // Smooth it (EMA) to reduce jitter
      smDvX = cfg.accEma * dvx_act + (1.0 - cfg.accEma) * smDvX;
      smDvY = cfg.accEma * dvy_act + (1.0 - cfg.accEma) * smDvY;

      final dv_pf_mag = math.sqrt(dv_pf_x*dv_pf_x + dv_pf_y*dv_pf_y) + 1e-12;
      final dv_act_mag = math.sqrt(smDvX*smDvX + smDvY*smDvY) + 1e-12;

      // Cosine alignment of Δv vectors
      final cosAcc = ((smDvX * dv_pf_x) + (smDvY * dv_pf_y)) / (dv_pf_mag * dv_act_mag);
      final accAlign = cosAcc.clamp(-1.0, 1.0);

      // Magnitude mismatch normalized to feasible cap
      final errX = (smDvX - dv_pf_x);
      final errY = (smDvY - dv_pf_y);
      final accErr = (math.sqrt(errX*errX + errY*errY) / (dv_pf_cap + 1e-9)).clamp(0.0, 5.0);

      rAcc = cfg.wAccAlign * accAlign - cfg.wAccErr * accErr;
    }

    // Update previous v for next step
    prevVx = vx; prevVy = vy;

    // Optional PF debug (prints once per ~240 frames)
    if (cfg.debug) {
      sumAbsDVpfX += dv_pf_x.abs();
      sumAbsDVpfY += dv_pf_y.abs();
      sumWLat += wLat;
      sumWVer += wVer;
      sumVelPen += (wVelEff * vErr);
      sumAlignAbs += wAlignEff * align.abs();

      dbgN++;
      if ((dbgN % 240) == 0) {
        final mX = (sumAbsDVpfX / dbgN).toStringAsFixed(2);
        final mY = (sumAbsDVpfY / dbgN).toStringAsFixed(2);
        final mWL = (sumWLat / dbgN).toStringAsFixed(2);
        final mWV = (sumWVer / dbgN).toStringAsFixed(2);
        final mVel = (sumVelPen / dbgN).toStringAsFixed(3);
        final mAli = (sumAlignAbs / dbgN).toStringAsFixed(3);
        print('[PFDBG] frames=$dbgN | mean|dv_pf_x|=$mX mean|dv_pf_y|=$mY '
            '| mean wLat=$mWL wVer=$mWV | velPen=$mVel align=$mAli');
      }
    }

    // ===== NEW: adaptive vertical speed cap penalty (stronger near ground) =====
    double vyCapDown = 10.0 + 1.0 * math.sqrt(h);
    vyCapDown = vyCapDown.clamp(10.0, 45.0);
    if (h < 120) vyCapDown = math.min(vyCapDown, 18.0);
    if (h < 60)  vyCapDown = math.min(vyCapDown, 10.0);
    final vyExcess = math.max(0.0, vy - vyCapDown); // vy positive = downward in engine
    final rVyCap = -cfg.wVyCap * vyExcess; // penalize overspeed

    // ===== NEW: lateral velocity toward pad center (polarity shaping) =====
    // positive when moving toward pad; scaled by how far we are from pad
    final double toward = (-dx.sign) * vx; // >0 when vx points toward pad center
    final double farScale = (dxAbs / (0.25 * W)).clamp(0.0, 1.0); // emphasize when far
    // normalize by ~120 px/s typical horizontal scale to keep small magnitude
    final rPadTow = cfg.wPadTow * farScale * (toward / 120.0).clamp(-1.0, 1.0);

    // Total reward
    final r = wAlignEff * align + touchBonus - wVelEff * vErr + rAcc + rVyCap + rPadTow;
    return r;
  };
}

/* ----------------------------------- main ------------------------------------ */

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

void main(List<String> argv) async {
  final args = _Args(argv);

  // ---- Curriculum registry ----
  final registry = CurriculumRegistry()
    ..register('speedmin', () => SpeedMinCurriculum())
    ..register('hardapp', () => HardApproach())
    ..register('final', () => FinalApproach());

  // ---- Common CLI ----
  final seed = args.getInt('seed', def: 7);

  // Optional load
  final loadPath = args.getStr('load_policy');

  // Consolidation (anti-forgetting)
  final anchorOnLoad = args.getFlag('anchor_on_load', def: false);
  final anchorAfterCurr = args.getFlag('anchor_after_curriculum', def: true);
  final lambdaTrunk = args.getDouble('consolidate_trunk', def: 1e-3);
  final lambdaHeads = args.getDouble('consolidate_heads', def: 5e-4);

  // NEW: choose curricula via --curricula="speedmin,hardapp"
  String curriculaSpec = args.getStr('curricula', def: '') ?? '';

  // Back-compat shim for older flags:
  // --curriculum (bool), --lowalt_iters (maps to "speedmin"), --hardapp_iters (maps to "hardapp")
  final legacyCurr = args.getFlag('curriculum', def: false);
  final lowaltIters = args.getInt('lowalt_iters', def: 0);   // treat as speedmin
  final hardappIters = args.getInt('hardapp_iters', def: 0); // treat as hardapp

  if (curriculaSpec.isEmpty && (legacyCurr || lowaltIters > 0 || hardappIters > 0)) {
    final keys = <String>[];
    if (lowaltIters > 0) keys.add('speedmin');
    if (hardappIters > 0) keys.add('hardapp');
    if (keys.isEmpty) keys.add('speedmin'); // default if just --curriculum was passed
    curriculaSpec = keys.join(',');
  }

  final curricula = registry.fromConfig(curriculaSpec);

  // Global default iterations
  int curIters = args.getInt('curriculum_iters', def: 0);

  final warmAfterCurr = args.getFlag('warm_norm_after_curriculum', def: true);

  // Stage 2 (PF)
  final iters = args.getInt('train_iters', def: args.getInt('iters', def: 200));
  final batch = args.getInt('batch', def: 1);
  final lr = args.getDouble('lr', def: 3e-4);
  final valueBeta = args.getDouble('value_beta', def: 0.5);
  final huberDelta = args.getDouble('huber_delta', def: 1.0);

  final planHold = args.getInt('plan_hold', def: 1);
  final tempIntent = args.getDouble('intent_temp', def: 1.0);
  final intentEntropy = args.getDouble('intent_entropy', def: 0.0);
  final useLearned = args.getFlag('use_learned_controller', def: false);
  final blendPolicy = args.getDouble('blend_policy', def: 1.0);
  final intentAlignWeight = args.getDouble('intent_align', def: 0.25);
  final intentPgWeight = args.getDouble('intent_pg', def: 0.6);
  final actionAlignWeight = args.getDouble('action_align', def: 0.0);

  final lockTerrain = args.getFlag('lock_terrain', def: false);
  final lockSpawn = args.getFlag('lock_spawn', def: false);
  final randomSpawnX = !args.getFlag('fixed_spawn_x', def: false);
  final maxFuel = args.getDouble('max_fuel', def: 1000.0);
  final crashOnTilt = args.getFlag('crash_on_tilt', def: false);

  // physics flags for CLI
  final gravity = args.getDouble('gravity', def: 0.18);
  final thrustAccel = args.getDouble('thrust_accel', def: 0.42);
  final rotSpeed = args.getDouble('rot_speed', def: 1.6);

  final rcsEnabled = args.getFlag('rcs_enabled', def: false);
  final rcsAccel = args.getDouble('rcs_accel', def: 0.12);
  final rcsBodyFrame = !args.getFlag('rcs_world_frame', def: false); // default body frame

  final downThrEnabled = args.getFlag('down_thr_enabled', def: false);
  final downThrAccel = args.getDouble('down_thr_accel', def: 0.30);
  final downThrBurn = args.getDouble('down_thr_burn', def: 10.0);

  // PF reward CLI knobs
  final pfAlign = args.getDouble('pf_align', def: 1.0);
  final pfVelDelta = args.getDouble('pf_vel_delta', def: 0.6);
  final pfVminClose = args.getDouble('pf_vmin_close', def: 8.0);
  final pfVmaxFar = args.getDouble('pf_vmax_far', def: 90.0);
  final pfAlpha = args.getDouble('pf_alpha', def: 1.2);
  final pfVmax = args.getDouble('pf_vmax', def: 140.0);
  final pfXBias = args.getDouble('pf_x_bias', def: 3.0);
  final pfAccAlign = args.getDouble('pf_acc_align', def: 2.0);
  final pfAccErr   = args.getDouble('pf_acc_err',   def: 1.0);
  final pfAccEma   = args.getDouble('pf_acc_ema',   def: 0.2);
  final pfDebug    = args.getFlag('pf_debug',       def: false);

  // NEW: extra shaping CLI (vy cap penalty & pad-toward reward)
  final pfVyCapW = args.getDouble('pf_vy_cap_w', def: 0.02);
  final pfPadTowW = args.getDouble('pf_pad_tow_w', def: 0.12);

  // attempts per terrain + eval cadence/size + parallel + debug
  final attemptsPerTerrain = _iclamp(args.getInt('attempts_per_terrain', def: 1), 1, 1000000);
  final evalEvery = _iclamp(args.getInt('eval_every', def: 10), 1, 1000000);
  final evalEpisodes = _iclamp(args.getInt('eval_episodes', def: 80), 1, 1000000);
  final evalParallel = args.getFlag('eval_parallel', def: false);
  final evalWorkers = _iclamp(args.getInt('eval_workers', def: Platform.numberOfProcessors), 1, 512);
  final evalDebug = args.getFlag('eval_debug', def: false);
  final evalDebugFailN = _iclamp(args.getInt('eval_debug_fail', def: 3), 0, 1000);

  final determinism = args.getFlag('determinism_probe', def: true);
  final hidden = _parseHiddenList(args.getStr('hidden'), fallback: const [64, 64]);

  // Trainer-internal gating (PF mean reward gating)
  final gateScoreMin = args.getDouble('gate_min', def: -1e9);
  final gateOnlyLanded = args.getFlag('gate_landed', def: false);
  final gateVerbose = args.getFlag('gate_verbose', def: true);

  double bestMeanCost = double.infinity;

  // ----- Build env -----
  final cfg = makeConfig(
    seed: seed,
    lockTerrain: lockTerrain,
    lockSpawn: lockSpawn,
    randomSpawnX: randomSpawnX,
    maxFuel: maxFuel,
    crashOnTilt: crashOnTilt,

    gravity: gravity,
    thrustAccel: thrustAccel,
    rotSpeed: rotSpeed,

    rcsEnabled: rcsEnabled,
    rcsAccel: rcsAccel,
    rcsBodyFrame: rcsBodyFrame,

    downThrEnabled: downThrEnabled,
    downThrAccel: downThrAccel,
    downThrBurn: downThrBurn,
  );
  final env = eng.GameEngine(cfg);

  // Rays active (forward-aligned)
  env.rayCfg = const RayConfig(rayCount: 180, includeFloor: false, forwardAligned: false);

  // FE probe
  final fe = FeatureExtractorRays(rayCount: env.rayCfg.rayCount);
  env.reset(seed: seed ^ 0xC0FFEE);
  env.step(1 / 60.0, const et.ControlInput(thrust: false, left: false, right: false));
  final inDim = fe.extract(env).length;
  final kindsOneHot = (inDim == 6 + env.rayCfg.rayCount * 4);

  // ----- Policy -----
  final policy = PolicyNetwork(inputSize: inDim, hidden: hidden, seed: seed);
  print('Init policy. hidden=${policy.hidden} | FE(kind=rays, in=$inDim, rays=${env.rayCfg.rayCount}, oneHot=$kindsOneHot)');

  // ===== Optional: LOAD existing policy weights =====
  if (loadPath != null && loadPath.trim().isNotEmpty) {
    _loadPolicyIntoNetwork(
      path: loadPath,
      target: policy,
      env: env,
      norm: null,
    );
    if (anchorOnLoad) {
      policy.captureConsolidationAnchor();
      policy.consolidateEnabled = true;
      policy.consolidateTrunk = lambdaTrunk;
      policy.consolidateHeads = lambdaHeads;
      print('Consolidation anchor captured from loaded policy. '
          'λ_trunk=${policy.consolidateTrunk} λ_heads=${policy.consolidateHeads}');
    }
  }

  // ===== Trainer (Stage 2 runtime container) =====
  ai.ExternalRewardHook? pfHook; // captured by trainer closure

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
    intentAlignWeight: intentAlignWeight,
    intentPgWeight: intentPgWeight,
    actionAlignWeight: actionAlignWeight,
    normalizeFeatures: true,
    gateScoreMin: gateScoreMin,
    gateOnlyLanded: gateOnlyLanded,
    gateVerbose: gateVerbose,
    externalRewardHook: (({required eng.GameEngine env, required double dt, required int tStep}) {
      return pfHook != null ? pfHook!(env: env, dt: dt, tStep: tStep) : 0.0;
    }),
  );

  // Determinism probe
  if (determinism) {
    env.reset(seed: 1234);
    final a = _probeDeterminism(env, maxSteps: 165);
    env.reset(seed: 1234);
    final b = _probeDeterminism(env, maxSteps: 165);
    final ok = (a.steps == b.steps) && ((a.cost - b.cost).abs() < 1e-6);
    print('Determinism probe: steps ${a.steps} vs ${b.steps} | cost ${a.cost.toStringAsFixed(6)} vs ${b.cost.toStringAsFixed(6)} => ${ok ? "OK" : "MISMATCH"}');
  }

  // ===== MODULAR CURRICULA (Stage 1/micro-stages) =====
  if (curricula.isNotEmpty && (curIters > 0 || lowaltIters > 0 || hardappIters > 0)) {
    print('=== Curricula: ${curricula.map((c) => c.key).join(", ")} ===');
    for (final cur in curricula) {
      final itersThis =
      cur.key == 'hardapp'   && hardappIters > 0 ? hardappIters :
      cur.key == 'speedmin'  && lowaltIters  > 0 ? lowaltIters  :
      curIters; // fallback to global

      if (itersThis <= 0) continue;

      cur.configure(args.kv, args.flags);
      await cur.run(
        iters: itersThis,
        env: env,
        fe: fe,
        policy: policy,
        norm: trainer.norm,
        planHold: planHold,
        tempIntent: tempIntent,
        gamma: 0.99,
        lr: lr,
        intentAlignWeight: intentAlignWeight,
        intentPgWeight: intentPgWeight,
        actionAlignWeight: actionAlignWeight,
        gateVerbose: gateVerbose,
        seed: seed,
      );
    }

    _savePolicy(
      path: 'policy_curriculum.json',
      p: policy,
      rayCount: env.rayCfg.rayCount,
      kindsOneHot: kindsOneHot,
      env: env,
      norm: trainer.norm,
    );
    print('★ Curricula complete → saved policy_curriculum.json');

    if (anchorAfterCurr) {
      policy.captureConsolidationAnchor();
      policy.consolidateEnabled = true;
      policy.consolidateTrunk = lambdaTrunk;
      policy.consolidateHeads = lambdaHeads;
      print('Consolidation anchor captured after curricula. '
          'λ_trunk=${policy.consolidateTrunk} λ_heads=${policy.consolidateHeads}');
    }

    if (warmAfterCurr) {
      // Optionally re-warm feature norm here if you have a helper.
    }
  }

  // ===== Baseline eval =====
      {
    if (evalParallel) {
      await evaluateParallel(
        cfg: cfg,
        policy: policy,
        hidden: hidden,
        episodes: evalEpisodes,
        attemptsPerTerrain: attemptsPerTerrain,
        seed: seed ^ 0x999,
        workers: evalWorkers,
        planHold: planHold,
        blendPolicy: blendPolicy,
        tempIntent: tempIntent,
        intentEntropy: intentEntropy,
        evalDebug: evalDebug,
        evalDebugFailN: evalDebugFailN,
      );
    } else {
      evaluateSequential(
        env: env,
        trainer: trainer,
        episodes: evalEpisodes,
        seed: seed ^ 0x999,
        attemptsPerTerrain: attemptsPerTerrain,
        evalDebug: evalDebug,
        evalDebugFailN: evalDebugFailN,
      );
    }
  }

  // ===== MAIN TRAIN LOOP (Stage 2: PF reward) =====
  final rnd = math.Random(seed ^ 0xDEADBEEF);
  final rayCount = env.rayCfg.rayCount;

  int terrAttempts = 0;
  int currentTerrainSeed = rnd.nextInt(1 << 30);

  final pfCfg = PFShapingCfg(
    wAlign: pfAlign,
    wVelDelta: pfVelDelta,
    vMinClose: pfVminClose,
    vMaxFar: pfVmaxFar,
    alpha: pfAlpha,
    vmax: pfVmax,
    xBias: pfXBias,
    wAccAlign: pfAccAlign,
    wAccErr: pfAccErr,
    accEma: pfAccEma,
    debug: pfDebug,
    wVyCap: pfVyCapW,   // NEW
    wPadTow: pfPadTowW, // NEW
  );

  for (int it = 0; it < iters; it++) {
    double lastCost = 0.0;
    int lastSteps = 0;
    bool lastLanded = false;
    double lastSegMean = 0.0;

    for (int b = 0; b < batch; b++) {
      if (terrAttempts == 0) {
        currentTerrainSeed = rnd.nextInt(1 << 30);
      }
      env.reset(seed: currentTerrainSeed);
      // Rebuild PF per episode so it matches terrain/pad
      pfHook = makePFRewardHook(env: env, cfg: pfCfg);

      final res = trainer.runEpisode(
        train: true,
        greedy: false,
        scoreIsReward: false,
        lr: lr,
        valueBeta: valueBeta,
        huberDelta: huberDelta,
      );

      terrAttempts = (terrAttempts + 1) % attemptsPerTerrain;

      lastCost = res.totalCost;
      lastSteps = res.steps;
      lastLanded = res.landed;
      lastSegMean = res.segMean;
    }

    if (gateVerbose && ((it + 1) % 10 == 0)) {
      final tag = lastLanded ? 'L' : 'NL';
      print('[TRAIN] iter=${it + 1} | segMean=${lastSegMean.toStringAsFixed(3)} | steps=$lastSteps | landed=$tag');
    }

    // periodic eval + save
    if ((it + 1) % evalEvery == 0) {
      final EvalStats ev;
      if (evalParallel) {
        ev = await evaluateParallel(
          cfg: cfg,
          policy: policy,
          hidden: hidden,
          episodes: evalEpisodes,
          attemptsPerTerrain: attemptsPerTerrain,
          seed: seed ^ (0x1111 * (it + 1)),
          workers: evalWorkers,
          planHold: planHold,
          blendPolicy: blendPolicy,
          tempIntent: tempIntent,
          intentEntropy: intentEntropy,
          evalDebug: evalDebug,
          evalDebugFailN: evalDebugFailN,
        );
      } else {
        ev = evaluateSequential(
          env: env,
          trainer: trainer,
          episodes: evalEpisodes,
          seed: seed ^ (0x1111 * (it + 1)),
          attemptsPerTerrain: attemptsPerTerrain,
          evalDebug: evalDebug,
          evalDebugFailN: evalDebugFailN,
        );
      }

      if (ev.meanCost < bestMeanCost) {
        bestMeanCost = ev.meanCost;
        _savePolicy(
          path: 'policy_best_cost.json',
          p: policy,
          rayCount: rayCount,
          kindsOneHot: kindsOneHot,
          env: env,
          norm: trainer.norm,
        );
        print('★ New BEST by cost at iter ${it + 1}: meanCost=${ev.meanCost.toStringAsFixed(3)} → saved policy_best_cost.json');
      }

      _savePolicy(
        path: 'policy_iter_${it + 1}.json',
        p: policy,
        rayCount: rayCount,
        kindsOneHot: kindsOneHot,
        env: env,
        norm: trainer.norm,
      );
    }
  }

  _savePolicy(
    path: 'policy_final.json',
    p: policy,
    rayCount: rayCount,
    kindsOneHot: kindsOneHot,
    env: env,
    norm: trainer.norm,
  );
  print('Training done. Saved → policy_final.json');

  // Helpful hint
  if (curricula.isEmpty) {
    print('Tip: you can run modular curricula with --curricula="speedmin,hardapp" '
        'or legacy flags --curriculum --lowalt_iters=... --hardapp_iters=...');
  } else {
    print('Curricula used: ${curricula.map((c) => c.key).join(", ")}');
  }
}
