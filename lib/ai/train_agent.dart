// lib/ai/train_agent.dart
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter_application_1/ai/curriculum/pad_align.dart';
import 'package:flutter_application_1/ai/curriculum/planner_bc.dart';
import 'package:flutter_application_1/ai/pf_reward.dart';

import '../engine/types.dart' as et;
import '../engine/game_engine.dart' as eng;
import '../engine/raycast.dart'; // RayConfig

import 'agent.dart' as ai; // FeatureExtractorRays, PolicyNetwork, Trainer, RunningNorm
import 'curriculum/final_approach.dart';
import 'curriculum/final_dagger.dart';
import 'curriculum/final_simple.dart';
import 'curriculum/pad_align_progressive.dart' as padprog show PadAlignProgressiveCurriculum;
import 'nn_helper.dart' as nn;
import 'policy_io.dart'; // PolicyBundle I/O
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
    if (_flags.contains(k)) return true;
    final v = _kv[k];
    if (v == null) return def;
    final s = v.toLowerCase();
    return s == '1' || s == 'true' || s == 'yes' || s == 'on';
  }

  Map<String, String?> get kv => _kv;
  Set<String> get flags => _flags;
}

int _iclamp(int v, int lo, int hi) => v < lo ? lo : (v > hi ? hi : v);

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

Future<void> main(List<String> argv) async {
  final args = _Args(argv);

  // ---- Curriculum registry ----
  final registry = CurriculumRegistry()
    ..register('speedmin', () => SpeedMinCurriculum())
    ..register('padalign', () => PadAlignCurriculum())
    ..register('padalign_progressive', () => padprog.PadAlignProgressiveCurriculum())
    ..register('hardapp', () => HardApproach())
    ..register('final_simple', () => FinalSimple())
    ..register('final_dagger', () => FinalDagger())
    ..register('planner_bc', () => PlannerBC())
    ..register('final', () => FinalApproach());

  // ---- Common CLI ----
  final seed = args.getInt('seed', def: 7);

  // Loader behavior flags
  final forceScratch = args.getFlag('force_scratch', def: false);
  final inheritArch  = args.getFlag('inherit_arch',  def: true);
  final strictLoad   = args.getFlag('strict_load',   def: false);

  // Optional load (explicit), else default to curriculum snapshot if present
  String? loadPath = args.getStr('load_policy');
  if (!forceScratch &&
      (loadPath == null || loadPath.trim().isEmpty) &&
      File('policy_curriculum.json').existsSync()) {
    loadPath = 'policy_curriculum.json';
    print('No --load_policy provided; found and will try to load "$loadPath".');
  }

  // Curricula selection
  String curriculaSpec = args.getStr('curricula', def: '') ?? '';
  final legacyCurr = args.getFlag('curriculum', def: false);
  final lowaltIters = args.getInt('lowalt_iters', def: 0);
  final hardappIters = args.getInt('hardapp_iters', def: 0);
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

  // Curriculum-chunked eval cadence (0 = disabled, run once per curriculum)
  final curEvalEvery = _iclamp(args.getInt('cur_eval_every', def: 0), 0, 1 << 30);
  final curEvalEpisodes = _iclamp(args.getInt('cur_eval_episodes', def: 120), 1, 1000000);
  final curEvalParallel = args.getFlag('cur_eval_parallel', def: false);
  final curEvalWorkers = _iclamp(args.getInt('cur_eval_workers', def: math.max(1, Platform.numberOfProcessors ~/ 2)), 1, 512);

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

  // Environment flags
  final lockTerrain = args.getFlag('lock_terrain', def: false);
  final lockSpawn = args.getFlag('lock_spawn', def: false);
  final randomSpawnX = !args.getFlag('fixed_spawn_x', def: false);
  final maxFuel = args.getDouble('max_fuel', def: 1000.0);
  final crashOnTilt = args.getFlag('crash_on_tilt', def: false);

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

  // extra shaping
  final pfVyCapW = args.getDouble('pf_vy_cap_w', def: 0.02);
  final pfPadTowW = args.getDouble('pf_pad_tow_w', def: 0.12);

  // attempts per terrain + eval cadence
  final attemptsPerTerrain = _iclamp(args.getInt('attempts_per_terrain', def: 1), 1, 1000000);
  final evalEvery = _iclamp(args.getInt('eval_every', def: 10), 1, 1000000);
  final evalEpisodes = _iclamp(args.getInt('eval_episodes', def: 80), 1, 1000000);
  final evalParallel = args.getFlag('eval_parallel', def: false);
  final evalWorkers = _iclamp(args.getInt('eval_workers', def: Platform.numberOfProcessors), 1, 512);
  final evalDebug = args.getFlag('eval_debug', def: false);
  final evalDebugFailN = _iclamp(args.getInt('eval_debug_fail', def: 3), 0, 1000);

  final determinism = args.getFlag('determinism_probe', def: true);

  // Hidden sizes: CLI > bundle (if inherit_arch) > default
  final cliHiddenRaw = args.getStr('hidden');
  final cliProvidedHidden = (cliHiddenRaw != null && cliHiddenRaw.trim().isNotEmpty);
  final hiddenDefault = _parseHiddenList(cliHiddenRaw, fallback: const [64, 64]);

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
  final fe = ai.FeatureExtractorRays(rayCount: env.rayCfg.rayCount);
  env.reset(seed: seed ^ 0xC0FFEE);
  env.step(1 / 60.0, const et.ControlInput(thrust: false, left: false, right: false));
  final inDim = fe.extract(
    lander: env.lander,
    terrain: env.terrain,
    worldW: env.cfg.worldW,
    worldH: env.cfg.worldH,
    rays: env.rays,
  ).length;
  final kindsOneHot = (inDim == 5 + env.rayCfg.rayCount * 4);

  // ===== Try to parse bundle early (for arch adoption) =====
  PolicyBundle? loadedBundle;
  if (!forceScratch &&
      loadPath != null &&
      loadPath.trim().isNotEmpty &&
      File(loadPath).existsSync()) {
    try {
      loadedBundle = PolicyBundle.loadFromPath(loadPath);
    } catch (e) {
      final msg = 'WARN: Failed to parse "$loadPath": $e';
      if (strictLoad) {
        print('$msg\n(strict_load=true → exiting)');
        rethrow;
      } else {
        print('$msg\nProceeding from scratch.');
        loadedBundle = null;
      }
    }
  } else if (loadPath != null && loadPath.trim().isNotEmpty) {
    print('WARNING: --load_policy="$loadPath" not found on disk. Proceeding from scratch.');
  }

  // Decide hidden sizes now
  List<int> chosenHidden = List<int>.from(hiddenDefault);
  if (!cliProvidedHidden && inheritArch && loadedBundle != null && loadedBundle.hidden.isNotEmpty) {
    chosenHidden = List<int>.from(loadedBundle.hidden);
    print('Adopting hidden sizes from bundle: $chosenHidden (override with --hidden=...).');
  }

  // ----- Policy -----
  final policy = ai.PolicyNetwork(inputSize: inDim, hidden: chosenHidden, seed: seed);
  print('Init policy. hidden=${policy.hidden} | FE(kind=rays, in=$inDim, rays=${env.rayCfg.rayCount}, oneHot=$kindsOneHot)');

  // Enable training for curricula stage (typical: trunk+intent+action)
  policy.setTrunkTrainable(true);
  policy.setHeadsTrainable(intent: true, action: true, value: false);
  print('[STAGE] Curricula: trainable { trunk=ON, intent=ON, action=ON, value=OFF }');

  // ===== If bundle parsed, try to copy weights with shape checks =====
  if (loadedBundle != null) {
    try {
      if (loadedBundle.inputDim != inDim) {
        throw StateError('Input dim mismatch: loaded=${loadedBundle.inputDim} runtime=$inDim '
            '(check rayCount/kindsOneHot/world size).');
      }
      if (loadedBundle.hidden.length != policy.hidden.length ||
          !List.generate(loadedBundle.hidden.length, (i) => loadedBundle?.hidden[i] == policy.hidden[i])
              .every((x) => x)) {
        throw StateError('Hidden sizes mismatch. Loaded=${loadedBundle.hidden} Runtime=${policy.hidden}');
      }

      loadBundleIntoNetwork(bundle: loadedBundle, target: policy, env: env);
      print('Loaded weights from "$loadPath".');
    } catch (e) {
      final msg = 'WARN: Failed to load weights from "$loadPath": $e';
      if (strictLoad) {
        print('$msg\n(strict_load=true → exiting)');
        rethrow;
      } else {
        print('$msg\nContinuing from randomly initialized weights.');
      }
    }
  }

  // ===== Trainer (norm will be restored from bundle if present) =====
  ai.ExternalRewardHook? pfHook;
  final trainer = ai.Trainer(
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

    // simple gating flags available in current Trainer
    gateScoreMin: -double.infinity,
    gateOnlyLanded: false,
    gateVerbose: true,

    // external dense reward hook (PF)
    externalRewardHook: (({required eng.GameEngine env, required double dt, required int tStep}) {
      return pfHook != null ? pfHook!(env: env, dt: dt, tStep: tStep) : 0.0;
    }),
  );

  // Try to restore feature norm from bundle if available
  bool normReady = false;
  if (loadedBundle != null && trainer.norm != null) {
    try {
      normReady = restoreNormFromBundle(bundle: loadedBundle, runtimeNorm: trainer.norm!);
    } catch (e) {
      print('WARN: failed to restore feature norm from bundle: $e (continuing).');
      normReady = false;
    }
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

  // ===== MODULAR CURRICULA (with optional chunked eval) =====
  int globalCurrSteps = 0;
  if (curricula.isNotEmpty && (curIters > 0 || lowaltIters > 0 || hardappIters > 0)) {
    print('=== Curricula: ${curricula.map((c) => c.key).join(", ")} ===');
    for (final cur in curricula) {
      final totalForCur =
      cur.key == 'hardapp' && hardappIters > 0 ? hardappIters :
      cur.key == 'speedmin' && lowaltIters  > 0 ? lowaltIters  :
      cur.key == 'padalign' && lowaltIters  > 0 ? lowaltIters  :
      cur.key == 'padalign_progressive' && lowaltIters > 0 ? lowaltIters :
      cur.key == 'final' && lowaltIters > 0 ? lowaltIters :
      cur.key == 'final_simple' && lowaltIters > 0 ? lowaltIters :
      cur.key == 'final_dagger' && lowaltIters > 0 ? lowaltIters :
      cur.key == 'planner_bc' && lowaltIters > 0 ? lowaltIters :
      curIters;

      if (totalForCur <= 0) continue;

      cur.configure(args.kv, args.flags);

      if (curEvalEvery > 0 && curEvalEvery < totalForCur) {
        int done = 0;
        int chunkIdx = 0;
        while (done < totalForCur) {
          final itThis = math.min(curEvalEvery, totalForCur - done);
          chunkIdx++;
          print('[CUR] ${cur.key} • chunk=$chunkIdx iters=$itThis (accum ${done + itThis}/$totalForCur)');

          await cur.run(
            iters: itThis,
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
            gateVerbose: true,
            seed: seed ^ (0xC11C + chunkIdx),
          );

          done += itThis;
          globalCurrSteps += itThis;

          // Save a curriculum step snapshot
          final snapPath = 'policy_curriculum_step_$globalCurrSteps.json';
          savePolicyBundle(
            path: snapPath,
            p: policy,
            env: env,
            norm: trainer.norm,
          );

          // Eval pulse after each chunk
          if (curEvalParallel) {
            await eval.evaluateParallel(
              cfg: cfg,
              policy: policy,
              hidden: policy.hidden,
              episodes: curEvalEpisodes,
              attemptsPerTerrain: attemptsPerTerrain,
              seed: seed ^ (0xE001 ^ globalCurrSteps),
              workers: curEvalWorkers,
              planHold: planHold,
              blendPolicy: blendPolicy,
              tempIntent: tempIntent,
              intentEntropy: 0.0,
              evalDebug: false,
              evalDebugFailN: 3,
            );
          } else {
            eval.evaluateSequential(
              env: env,
              trainer: trainer..useLearnedController = true,
              episodes: curEvalEpisodes,
              seed: seed ^ (0xE001 ^ globalCurrSteps),
              attemptsPerTerrain: attemptsPerTerrain,
              evalDebug: false,
              evalDebugFailN: 3,
            );
          }
        }
      } else {
        // Single-shot
        print('[CUR] ${cur.key} • iters=$totalForCur');
        await cur.run(
          iters: totalForCur,
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
          gateVerbose: true,
          seed: seed,
        );
        globalCurrSteps += totalForCur;

        // Optional eval at the end of this curriculum
        if (curEvalEvery > 0) {
          if (curEvalParallel) {
            await eval.evaluateParallel(
              cfg: cfg,
              policy: policy,
              hidden: policy.hidden,
              episodes: curEvalEpisodes,
              attemptsPerTerrain: attemptsPerTerrain,
              seed: seed ^ (0xE001 ^ globalCurrSteps),
              workers: curEvalWorkers,
              planHold: planHold,
              blendPolicy: blendPolicy,
              tempIntent: tempIntent,
              intentEntropy: 0.0,
              evalDebug: false,
              evalDebugFailN: 3,
            );
          } else {
            eval.evaluateSequential(
              env: env,
              trainer: trainer..useLearnedController = true,
              episodes: curEvalEpisodes,
              seed: seed ^ (0xE001 ^ globalCurrSteps),
              attemptsPerTerrain: attemptsPerTerrain,
              evalDebug: false,
              evalDebugFailN: 3,
            );
          }
        }
      }
    }

    // Save bundle after curricula
    savePolicyBundle(
      path: 'policy_curriculum.json',
      p: policy,
      env: env,
      norm: trainer.norm,
    );
    print('★ Curricula complete → saved policy_curriculum.json');

    // If we did not restore norm and it's still uninitialized, warm it up briefly
    if (!normReady && trainer.norm != null && !trainer.norm!.inited) {
      final episodes = 8;
      print('Warming feature normalization over $episodes…');
      for (int i = 0; i < episodes; i++) {
        env.reset(seed: (seed ^ (0xA11CE + i)));
        trainer.runEpisode(
          train: false,
          greedy: true,
          scoreIsReward: false,
          lr: lr,
          valueBeta: valueBeta,
          huberDelta: huberDelta,
        );
      }
      print('Norm warm complete.');
      normReady = true;
    }

    // Freeze features/intent; train action in PF
    policy.setTrunkTrainable(false);
    policy.setHeadsTrainable(intent: false, action: true, value: false);
    print('[STAGE] PF: trainable { trunk=OFF, intent=OFF, action=ON, value=OFF }');
  } else {
    // No curricula → optional quick norm warm
    if (!normReady && trainer.norm != null && !trainer.norm!.inited) {
      final episodes = 8;
      print('Warming feature normalization over $episodes (free-flight)…');
      for (int i = 0; i < episodes; i++) {
        env.reset(seed: (seed ^ (0xB11CE + i)));
        trainer.runEpisode(
          train: false,
          greedy: true,
          scoreIsReward: false,
          lr: lr,
          valueBeta: valueBeta,
          huberDelta: huberDelta,
        );
      }
      print('Norm warm complete.');
      normReady = true;
    }

    // Freeze features/intent; train action in PF
    policy.setTrunkTrainable(false);
    policy.setHeadsTrainable(intent: false, action: true, value: false);
    print('[STAGE] PF: trainable { trunk=OFF, intent=OFF, action=ON, value=OFF }');
  }

  // ===== Baseline eval =====
      {
    if (evalParallel) {
      await eval.evaluateParallel(
        cfg: cfg,
        policy: policy,
        hidden: policy.hidden,
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

  double bestMeanCost = double.infinity;

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

    if (((it + 1) % 10) == 0) {
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
          hidden: policy.hidden,
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
        savePolicyBundle(
          path: 'policy_best_cost.json',
          p: policy,
          env: env,
          norm: trainer.norm,
        );
        print('★ New BEST by cost at iter ${it + 1}: meanCost=${ev.meanCost.toStringAsFixed(3)} → saved policy_best_cost.json');
      }

      savePolicyBundle(
        path: 'policy_iter_${it + 1}.json',
        p: policy,
        env: env,
        norm: trainer.norm,
      );
    }
  }

  savePolicyBundle(
    path: 'policy_final.json',
    p: policy,
    env: env,
    norm: trainer.norm,
  );
  print('Training done. Saved → policy_final.json');

  if (curricula.isEmpty) {
    print('Tip: you can run modular curricula with --curricula="speedmin,hardapp" '
        'or legacy flags --curriculum --lowalt_iters=... --hardapp_iters=... '
        '(padalign_progressive, planner_bc and final_dagger are also available).');
  } else {
    print('Curricula used: ${curricula.map((c) => c.key).join(", ")}');
  }
}
