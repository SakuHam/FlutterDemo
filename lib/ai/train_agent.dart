// lib/ai/train_agent.dart
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter_application_1/ai/curriculum/pad_align.dart';
import 'package:flutter_application_1/ai/pf_reward.dart';

import '../engine/types.dart' as et;
import '../engine/game_engine.dart' as eng;
import '../engine/raycast.dart'; // RayConfig

import 'agent.dart' as ai; // FeatureExtractorRays, PolicyNetwork, Trainer, RunningNorm, kIntentNames
import 'agent.dart';
import 'curriculum/final_approach.dart';
import 'curriculum/final_dagger.dart';
import 'curriculum/final_simple.dart';
import 'curriculum/pad_align_progressive.dart' as padprog show PadAlignProgressiveCurriculum;
import 'nn_helper.dart' as nn;
import 'potential_field.dart'; // buildPotentialField, PotentialField

// MODULAR CURRICULA
import 'curriculum/core.dart';
import 'curriculum/speed_min.dart';
import 'curriculum/hard_approach.dart';

// NEW: eval separated
import 'eval.dart' as eval;

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
      'dur': {
        'W': [ List<double>.from(p.durHead.W[0]) ], // 1 x H
        'b': [ p.durHead.b[0] ],
      },
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

    'ray_config': {
      'rayCount': rayCount,
      'includeFloor': env.rayCfg.includeFloor,
      'forwardAligned': env.rayCfg.forwardAligned,
    },
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
  final durJ = headsJ['dur'];
  if (durJ != null) {
    List<List<double>> _as2d(dynamic v) => (v as List)
        .map<List<double>>((r) => (r as List).map<double>((x) => (x as num).toDouble()).toList())
        .toList();
    List<double> _as1d(dynamic v) => (v as List).map<double>((x) => (x as num).toDouble()).toList();

    final Wd = _as2d((durJ as Map)['W']);
    final Bd = _as1d(durJ['b']);
    if (Wd.length == 1 && Wd[0].length == target.durHead.W[0].length && Bd.length == 1) {
      for (int j = 0; j < target.durHead.W[0].length; j++) {
        target.durHead.W[0][j] = Wd[0][j];
      }
      target.durHead.b[0] = Bd[0];
    } else {
      // shape mismatch → ignore quietly or throw, your call
    }
  }

  print('Loaded policy from $path (hidden=$hiddenLoad, inDim=$inDim).');
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

/* ----------------------------- PF velocity+accel reward ------------------------ */
// (unchanged from your reference) — keep your makePFRewardHook and PFShapingCfg as-is here
// ... [KEEP THE ENTIRE PF REWARD SECTION FROM YOUR CURRENT FILE] ...

// === Paste your PFShapingCfg and makePFRewardHook definitions here unchanged ===

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
    ..register('padalign', () => PadAlignCurriculum())
    ..register('padalign_progressive', () => padprog.PadAlignProgressiveCurriculum())
    ..register('hardapp', () => HardApproach())
    ..register('final_simple', () => FinalSimple())
    ..register('final_dagger', () => FinalDagger())
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
  final legacyCurr = args.getFlag('curriculum', def: false);
  final lowaltIters = args.getInt('lowalt_iters', def: 0);   // treat as speedmin
  final hardappIters = args.getInt('hardapp_iters', def: 0); // treat as hardapp

  if (curriculaSpec.isEmpty && (legacyCurr || lowaltIters > 0 || hardappIters > 0)) {
    final keys = <String>[];
    if (lowaltIters > 0) keys.add('speedmin');
    if (hardappIters > 0) keys.add('hardapp');
    if (keys.isEmpty) keys.add('speedmin');
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
  final rcsBodyFrame = !args.getFlag('rcs_world_frame', def: false);

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

  // NEW: extra shaping CLI
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

  // ---------------------- NEW: probabilistic gating CLI ----------------------
  final gateProbEnabled      = args.getFlag('gate_prob', def: true);
  final gateProbK            = args.getDouble('gate_prob_k', def: 8.0);
  final gateProbMin          = args.getDouble('gate_prob_min', def: 0.05);
  final gateProbMax          = args.getDouble('gate_prob_max', def: 0.95);
  final gateProbLandedBoost  = args.getDouble('gate_prob_landed_boost', def: 0.15);
  final gateProbNearPadBoost = args.getDouble('gate_prob_nearpad_boost', def: 0.10);

  // NEW: gentle floor (avoid p=0 lockout) + optional deadzone
  final gateProbDeadzoneZ    = args.getDouble('gate_prob_deadzone_z', def: -1e9); // disabled
  final gateProbFloor        = args.getDouble('gate_prob_floor', def: 0.02);      // small min acceptance

  // ---------------------- Hebbian CLI knobs (new) ----------------------
  final hebbOn       = args.getFlag('hebbian', def: false);
  final hebbEta      = args.getDouble('hebb_eta', def: 3e-4);
  final hebbClip     = args.getDouble('hebb_clip', def: 0.02);
  final hebbRowCap   = args.getDouble('hebb_row_cap', def: 2.5);
  final hebbOja      = args.getFlag('hebb_oja', def: true);
  final hebbTrunk    = args.getFlag('hebb_trunk', def: true);
  final hebbHeadInt  = args.getFlag('hebb_head_intent', def: false); // keep false by default
  final hebbHeadTurn = args.getFlag('hebb_head_turn', def: true);
  final hebbHeadThr  = args.getFlag('hebb_head_thr', def: true);
  final hebbHeadVal  = args.getFlag('hebb_head_val', def: false);

  final hebbModGain   = args.getDouble('hebb_mod_gain', def: 0.6);
  final hebbModClip   = args.getDouble('hebb_mod_clip', def: 1.5);
  final minIntEntropy = args.getDouble('min_intent_entropy', def: 0.5);
  final maxSameRun    = args.getInt('max_same_intent_run', def: 48);

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
  env.rayCfg = const RayConfig(rayCount: 180, includeFloor: false, forwardAligned: true);

  // FE probe
  final fe = FeatureExtractorRays(rayCount: env.rayCfg.rayCount);
  env.reset(seed: seed ^ 0xC0FFEE);
  env.step(1 / 60.0, const et.ControlInput(thrust: false, left: false, right: false));
  final inDim = fe.extract(lander: env.lander, terrain: env.terrain, worldW: env.cfg.worldW, worldH: env.cfg.worldH, rays: env.rays).length;
  final kindsOneHot = (inDim == 5 + env.rayCfg.rayCount * 4);

  // ----- Policy -----
  final policy = PolicyNetwork(inputSize: inDim, hidden: hidden, seed: seed);
  print('Init policy. hidden=${policy.hidden} | FE(kind=rays, in=$inDim, rays=${env.rayCfg.rayCount}, oneHot=$kindsOneHot)');

  // Enable training for curricula stage (typical: trunk+intent+action)
  policy.setTrunkTrainable(true);
  policy.setHeadsTrainable(intent: true, action: true, value: false);
  print('[STAGE] Curricula: trainable { trunk=ON, intent=ON, action=ON, value=OFF }');

  final loadedNorm = ai.RunningNorm(inDim, momentum: 0.995);

  // ===== Optional: LOAD existing policy weights =====
  if (loadPath != null && loadPath.trim().isNotEmpty) {
    _loadPolicyIntoNetwork(
      path: loadPath,
      target: policy,
      env: env,
      norm: loadedNorm,
    );
    if (anchorOnLoad) {
      policy.captureConsolidationAnchor();
      policy.consolidateEnabled = true;
      policy.consolidateTrunk = lambdaTrunk;
      policy.consolidateHeads = lambdaHeads;
      print('Consolidation anchor captured from loaded policy. '
          'λ_trunk=${policy.consolidateTrunk} λ_heads=${policy.consolidateHeads}');
    }
    // If loading a curriculum-complete model and going straight to PF, you may prefer:
    // policy.setTrunkTrainable(false);
    // policy.setHeadsTrainable(intent: false, action: true, value: false);
    // print('[STAGE] PF (from loaded): trainable { trunk=OFF, intent=OFF, action=ON, value=OFF }');
  }

  // ===== Trainer =====
  ai.ExternalRewardHook? pfHook;

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

    // Gate core
    gateScoreMin: gateScoreMin,
    gateOnlyLanded: gateOnlyLanded,
    gateVerbose: gateVerbose,

    // Probabilistic gating wiring (with safe defaults)
    gateProbEnabled: gateProbEnabled,
    gateProbK: gateProbK,
    gateProbMin: gateProbMin,
    gateProbMax: gateProbMax,
    gateProbLandedBoost: gateProbLandedBoost,
    gateProbNearPadBoost: gateProbNearPadBoost,

    // NEW: deadzone & floor
    gateProbDeadzoneZ: gateProbDeadzoneZ,
    gateProbFloor: gateProbFloor,

    // external dense reward hook (PF)
    externalRewardHook: (({required eng.GameEngine env, required double dt, required int tStep}) {
      return pfHook != null ? pfHook!(env: env, dt: dt, tStep: tStep) : 0.0;
    }),

    // Hebbian + guards
    hebbian: HebbianConfig(
      enabled: hebbOn,
      useOja: hebbOja,
      eta: hebbEta,
      clip: hebbClip,
      rowL2Cap: hebbRowCap,
      trunk: hebbTrunk,
      headIntent: hebbHeadInt,
      headTurn: hebbHeadTurn,
      headThr: hebbHeadThr,
      headVal: hebbHeadVal,
    ),
    hebbModGain: hebbModGain,
    hebbModAbsClip: hebbModClip,
    minIntentEntropy: minIntEntropy,
    maxSameIntentRun: maxSameRun,
  );

  if (loadedNorm.inited) {
    trainer.norm?.copyFrom(loadedNorm);
    print('Restored feature normalization from policy JSON.');
  }

  // Determinism probe
  if (determinism) {
    env.reset(seed: 1234);
    final a = _probeDeterminism(env, maxSteps: 165);
    env.reset(seed: 1234);
    final b = _probeDeterminism(env, maxSteps: 165);
    final ok = (a.steps == b.steps) && ((a.cost - b.cost).abs() < 1e-6);
    print('Determinism probe: steps ${a.steps} vs ${b.steps} | cost ${a.cost.toStringAsFixed(6)} vs ${b.cost.toStringAsFixed(6)} => ${ok ? "OK" : "MISMATCH"}');
  }

  // ===== MODULAR CURRICULA =====
  if (curricula.isNotEmpty && (curIters > 0 || lowaltIters > 0 || hardappIters > 0)) {
    print('=== Curricula: ${curricula.map((c) => c.key).join(", ")} ===');
    for (final cur in curricula) {
      final itersThis =
      cur.key == 'hardapp'   && hardappIters > 0 ? hardappIters :
      cur.key == 'speedmin'  && lowaltIters  > 0 ? lowaltIters  :
      cur.key == 'padalign'  && lowaltIters  > 0 ? lowaltIters  :
      cur.key == 'padalign_progressive'  && lowaltIters  > 0 ? lowaltIters  :
      cur.key == 'final'  && lowaltIters  > 0 ? lowaltIters  :
      cur.key == 'final_simple'  && lowaltIters  > 0 ? lowaltIters  :
      cur.key == 'final_dagger'  && lowaltIters  > 0 ? lowaltIters  :
      curIters;

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

    // Freeze curriculum features/intent; train action in PF
    policy.setTrunkTrainable(false);
    policy.setHeadsTrainable(intent: false, action: true, value: false);
    print('[STAGE] PF: trainable { trunk=OFF, intent=OFF, action=ON, value=OFF }');
    if (policy.consolidateEnabled) {
      print('[STAGE] PF: consolidation L2-SP active (λ_trunk=${policy.consolidateTrunk}, λ_heads=${policy.consolidateHeads})');
    }
  }

  // ===== Baseline eval =====
      {
    if (evalParallel) {
      await eval.evaluateParallel(
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
      eval.evaluateSequential(
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

  // ===== MAIN TRAIN LOOP (PF) =====
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
    wVyCap: pfVyCapW,
    wPadTow: pfPadTowW,
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
      final eval.EvalStats ev;
      if (evalParallel) {
        ev = await eval.evaluateParallel(
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
          evalDebug: true,
          evalDebugFailN: 3,
        );
      } else {
        ev = eval.evaluateSequential(
          env: env,
          trainer: trainer,
          episodes: evalEpisodes,
          seed: seed ^ (0x1111 * (it + 1)),
          attemptsPerTerrain: attemptsPerTerrain,
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

  if (curricula.isEmpty) {
    print('Tip: you can run modular curricula with --curricula="speedmin,hardapp" '
        'or legacy flags --curriculum --lowalt_iters=... --hardapp_iters=...');
  } else {
    print('Curricula used: ${curricula.map((c) => c.key).join(", ")}');
  }
}
