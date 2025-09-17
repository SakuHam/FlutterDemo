// lib/ai/train_agent.dart
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import '../engine/types.dart' as et;
import '../engine/game_engine.dart' as eng;
import '../engine/raycast.dart'; // RayConfig

import 'agent.dart' as ai; // FeatureExtractorRays, PolicyNetwork, Trainer, RunningNorm, kIntentNames, predictiveIntentLabelAdaptive
import 'agent.dart';       // bring symbols into scope (PolicyNetwork etc.)
import 'potential_field.dart'; // <-- PF: buildPotentialField, PotentialField

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
    // legacy mirror (optional)
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
  );
  f.writeAsStringSync(const JsonEncoder.withIndent('  ').convert(jsonMap));
  print('Saved policy → $path');
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

EvalStats evaluate({
  required eng.GameEngine env,
  required Trainer trainer,
  int episodes = 40,
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

/* ----------------------------- norm warmup (optional) -------------------------- */

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
    final want = r.nextInt(PolicyNetwork.kIntents);

    final padCx = env.terrain.padCenter.toDouble();
    final padHalfW =
    (((env.terrain.padX2 - env.terrain.padX1).abs()) * 0.5).clamp(12.0, env.cfg.worldW.toDouble());
    double x = padCx, h = 180, vx = 0, vy = 20;

    switch (want) {
      case 1: // goLeft
        x = (padCx - (0.22 + 0.18 * r.nextDouble()) * env.cfg.worldW).clamp(10.0, env.cfg.worldW - 10.0);
        h = 120 + 120 * r.nextDouble();
        vx = (r.nextDouble() * 14.0) - 7.0;
        vy = 28.0 + 10.0 * r.nextDouble();
        break;
      case 2: // goRight
        x = (padCx + (0.22 + 0.18 * r.nextDouble()) * env.cfg.worldW).clamp(10.0, env.cfg.worldW - 10.0);
        h = 120 + 120 * r.nextDouble();
        vx = (r.nextDouble() * 14.0) - 7.0;
        vy = 28.0 + 10.0 * r.nextDouble();
        break;
      case 3: // descendSlow
        x = padCx + (r.nextDouble() * 0.05 - 0.025) * padHalfW;
        h = 0.55 * env.cfg.worldH + 0.20 * env.cfg.worldH * r.nextDouble();
        vx = (r.nextDouble() * 24.0) - 12.0;
        vy = 24.0 + 12.0 * r.nextDouble();
        break;
      case 4: // brakeUp
        x = padCx + (r.nextDouble() * 0.05 - 0.025) * padHalfW;
        h = 40.0 + 50.0 * r.nextDouble();
        vx = (r.nextDouble() * 14.0) - 7.0;
        vy = 120.0 + 50.0 * r.nextDouble();
        break;
      case 5: // brakeLeft  (moving right too fast near pad)
        x = (padCx + (r.nextDouble() * 0.02 - 0.01) * padHalfW).clamp(10.0, padHalfW - 10.0);
        h = 120 + 100 * r.nextDouble();
        vx = 28.0 + 24.0 * r.nextDouble(); // +vx
        vy = 18.0 + 18.0 * r.nextDouble();
        break;
      case 6: // brakeRight (moving left too fast near pad)
        x = (padCx + (r.nextDouble() * 0.02 - 0.01) * padHalfW).clamp(10.0, padHalfW - 10.0);
        h = 120 + 100 * r.nextDouble();
        vx = -(28.0 + 24.0 * r.nextDouble()); // -vx
        vy = 18.0 + 18.0 * r.nextDouble();
        break;
      default: // hover
        x = padCx + (r.nextDouble() * 0.03 - 0.015) * padHalfW;
        h = 0.20 * env.cfg.worldH + 0.15 * env.cfg.worldH * r.nextDouble();
        vx = (r.nextDouble() * 10.0) - 5.0;
        vy = (r.nextDouble() * 10.0) - 5.0;
        break;
    }

    final gy = env.terrain.heightAt(x);
    env.lander
      ..pos.x = x.clamp(10.0, env.cfg.worldW - 10.0)
      ..pos.y = (gy - h).clamp(0.0, env.cfg.worldH - 10.0)
      ..vel.x = vx
      ..vel.y = vy
      ..angle = 0.0
      ..fuel = env.cfg.t.maxFuel;

    if (predictiveIntentLabelAdaptive(env) != want) continue;

    final feat = fe.extract(env);
    try {
      (trainer.norm as dynamic).observe(feat);
    } catch (_) {
      trainer.norm?.normalize(feat, update: true);
    }
    accepted++;
  }
  print('Feature norm warmed with $accepted synthetic samples.');
}

/* ----------------------------- PF shaping helpers ------------------------------ */

class PFShapingCfg {
  final double wDeltaPhi;   // reward per unit potential drop
  final double wAlign;      // reward per unit cos(v, flow)
  final double wVelDelta;   // penalty per unit ||v - v_pf|| / vmax
  final double clampSpeed;  // suggestVelocity clamp
  final double vmax;        // normalization for velocity error
  const PFShapingCfg({
    this.wDeltaPhi = 4.0,
    this.wAlign = 1.5,
    this.wVelDelta = 0.4,
    this.clampSpeed = 90.0,
    this.vmax = 140.0,
  });
}

/// Build a per-step reward hook for current episode/terrain.
/// Requires Trainer to call `externalRewardHook(env:..., dt:..., tStep:...)` each step.
ai.ExternalRewardHook makePFRewardHook({
  required eng.GameEngine env,
  PFShapingCfg cfg = const PFShapingCfg(),
}) {
  // Build PF once for this episode's terrain
  final pf = buildPotentialField(env, nx: 160, ny: 120, iters: 1200, omega: 1.7, tol: 1e-4);

  double prevPhi = pf.samplePhi(env.lander.pos.x, env.lander.pos.y);

  return ({required eng.GameEngine env, required double dt, required int tStep}) {
    final x = env.lander.pos.x;
    final y = env.lander.pos.y;

    // 1) Δφ: positive when moving downhill (toward pad)
    final phi = pf.samplePhi(x, y);
    final dPhi = (prevPhi - phi);
    prevPhi = phi;

    // 2) Alignment between actual velocity and −∇φ
    final vx = env.lander.vel.x;
    final vy = env.lander.vel.y;
    final flow = pf.sampleFlow(x, y); // returns unit (nx,ny)
    final vmag = math.sqrt(vx * vx + vy * vy);
    double align = 0.0;
    if (vmag > 1e-6) {
      align = (vx / vmag) * flow.nx + (vy / vmag) * flow.ny; // [-1,1]
    }

    // 3) Velocity delta to PF "prediction" vector (suggested velocity)
    final sugg = pf.suggestVelocity(x, y, clampSpeed: cfg.clampSpeed);
    final dvx = vx - sugg.vx;
    final dvy = vy - sugg.vy;
    final vErr = math.sqrt(dvx * dvx + dvy * dvy) / cfg.vmax;

    final r = cfg.wDeltaPhi * dPhi + cfg.wAlign * align - cfg.wVelDelta * vErr;

    // Optional temporal decay:
    // return r * (1.0 - 0.0008 * tStep).clamp(0.5, 1.0);

    return r;
  };
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

  final determinism = args.getFlag('determinism_probe', def: true);
  final hidden = _parseHiddenList(args.getStr('hidden'), fallback: const [64, 64]);

  // Trainer-internal gating (score is “higher is better”)
  final gateScoreMin = args.getDouble('gate_min', def: -1e9); // e.g., try 4.0, 6.0, etc
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
  );
  final env = eng.GameEngine(cfg);

  // Ensure rays active (forward-aligned)
  env.rayCfg = const RayConfig(
    rayCount: 180,
    includeFloor: false,
    forwardAligned: true,
  );

  // FE probe
  final fe = FeatureExtractorRays(rayCount: env.rayCfg.rayCount);
  env.reset(seed: seed ^ 0xC0FFEE);
  env.step(1 / 60.0, const et.ControlInput(thrust: false, left: false, right: false));
  final inDim = fe.extract(env).length;
  final kindsOneHot = (inDim == 6 + env.rayCfg.rayCount * 4);

  // ----- Policy -----
  final policy = PolicyNetwork(inputSize: inDim, hidden: hidden, seed: seed);
  print('Loaded init policy. hidden=${policy.hidden} | FE(kind=rays, in=$inDim, rays=${env.rayCfg.rayCount}, oneHot=$kindsOneHot)');

  // ===== PF shaping hook (mutable; rebuilt per episode) =====
  ai.ExternalRewardHook? pfHook = makePFRewardHook(env: env);

  // ----- Trainer -----
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
    // gating inside trainer (prints [TRAIN] lines)
    gateScoreMin: gateScoreMin,
    gateOnlyLanded: gateOnlyLanded,
    gateVerbose: gateVerbose,

    // NEW: add dense PF reward per step
    externalRewardHook: (({required eng.GameEngine env, required double dt, required int tStep}) {
      return pfHook != null ? pfHook!(env: env, dt: dt, tStep: tStep) : 0.0;
    }),
  );

  // Determinism probe (physics)
  if (determinism) {
    env.reset(seed: 1234);
    final a = _probeDeterminism(env, maxSteps: 165);
    env.reset(seed: 1234);
    final b = _probeDeterminism(env, maxSteps: 165);
    final ok = (a.steps == b.steps) && ((a.cost - b.cost).abs() < 1e-6);
    print('Determinism probe: steps ${a.steps} vs ${b.steps} | cost ${a.cost.toStringAsFixed(6)} vs ${b.cost.toStringAsFixed(6)} => ${ok ? "OK" : "MISMATCH"}');
  }

  // Optional: warm the feature norm a bit so early updates aren’t wild
  _warmFeatureNorm(
    norm: trainer.norm,
    trainer: trainer,
    fe: fe,
    env: env,
    perClass: 500,
    seed: seed ^ 0xACE,
  );

  // Quick baseline eval
      {
    final ev = evaluate(env: env, trainer: trainer, episodes: 20, seed: seed ^ 0x999);
    print(
        'Eval(real) → meanCost=${ev.meanCost.toStringAsFixed(3)} | median=${ev.medianCost.toStringAsFixed(3)} | land%=${ev.landPct.toStringAsFixed(1)} | crash%=${ev.crashPct.toStringAsFixed(1)} | steps=${ev.meanSteps.toStringAsFixed(1)} | mean|dx|=${ev.meanAbsDx.toStringAsFixed(1)}');
  }

  // ===== MAIN TRAIN LOOP =====
  final rnd = math.Random(seed ^ 0xDEADBEEF);
  final rayCount = env.rayCfg.rayCount;

  for (int it = 0; it < iters; it++) {
    double lastCost = 0.0;
    int lastSteps = 0;
    bool lastLanded = false;

    for (int b = 0; b < batch; b++) {
      env.reset(seed: rnd.nextInt(1 << 30));

      // Rebuild PF-based reward hook for this terrain/episode
      pfHook = makePFRewardHook(env: env);

      final res = trainer.runEpisode(
        train: true, // Trainer handles gating internally & prints [TRAIN] lines
        greedy: false,
        scoreIsReward: false,
        lr: lr,
        valueBeta: valueBeta,
        huberDelta: huberDelta,
      );

      // For iteration summary
      lastCost = res.totalCost;
      lastSteps = res.steps;
      lastLanded = res.landed;
    }

    print('Iter ${it + 1} | batch=$batch | last-ep steps: $lastSteps | cost: ${lastCost.toStringAsFixed(3)} | landed: ${lastLanded ? "Y" : "N"}');

    // periodic eval + save
    if ((it + 1) % 5 == 0) {
      final ev = evaluate(env: env, trainer: trainer, episodes: 40, seed: seed ^ (0x1111 * (it + 1)));
      print(
          'Eval(real) → meanCost=${ev.meanCost.toStringAsFixed(3)} | median=${ev.medianCost.toStringAsFixed(3)} | land%=${ev.landPct.toStringAsFixed(1)} | crash%=${ev.crashPct.toStringAsFixed(1)} | steps=${ev.meanSteps.toStringAsFixed(1)} | mean|dx|=${ev.meanAbsDx.toStringAsFixed(1)}');

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
}

/* ----------------------------------- usage ------------------------------------

Train with PF shaping + internal gating on segment score (higher is better). Example:

  dart run lib/ai/train_agent.dart \
    --hidden=96,96,64 \
    --train_iters=400 --batch=1 --lr=0.0003 --plan_hold=1 \
    --blend_policy=1.0 --intent_align=0.25 --intent_pg=0.6 \
    --gate_min=4.0 --gate_landed \
    --determinism_probe

Notes:
- Requires Trainer to support `externalRewardHook` (small change in agent.dart).
- PF shaping terms:
    r = wDeltaPhi * Δφ  +  wAlign * cos(v, −∇φ)  −  wVelDelta * ||v − v_pf||/vmax
  Tune weights in PFShapingCfg if needed.
-------------------------------------------------------------------------------- */
