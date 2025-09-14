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
      if (a.startsWith('--')) {
        final s = a.substring(2);
        final i = s.indexOf('=');
        if (i >= 0) {
          _kv[s.substring(0, i)] = s.substring(i + 1);
        } else {
          _flags.add(s);
        }
      }
    }
  }
  String? getStr(String k, {String? def}) => _kv[k] ?? (_flags.contains(k) ? 'true' : def);

  int getInt(String k, {int def = 0}) {
    final v = getStr(k);
    return v == null ? def : (int.tryParse(v) ?? def);
  }

  double getDouble(String k, {double def = 0.0}) {
    final v = getStr(k);
    return v == null ? def : (double.tryParse(v) ?? def);
  }

  bool getBool(String k, {bool def = false}) {
    if (_flags.contains(k)) return true;
    final v = _kv[k];
    if (v == null) return def;
    final s = v.toLowerCase().trim();
    return s == '1' || s == 'true' || s == 'yes' || s == 'y';
  }
}

/* ------------------------------- policy IO (json) ------------------------------ */

Map<String, dynamic> _weightsToJson(PolicyNetwork p) {
  List<List<double>> to3(List<List<double>> W) =>
      W.map((r) => r.map((v) => v.toDouble()).toList()).toList();

  return {
    'h1': p.h1,
    'h2': p.h2,
    // Trunk
    'W1': to3(p.W1), 'b1': p.b1,
    'W2': to3(p.W2), 'b2': p.b2,
    // Legacy action heads (optional)
    'W_thr': to3(p.W_thr), 'b_thr': p.b_thr,
    'W_turn': to3(p.W_turn), 'b_turn': p.b_turn,
    // Planner heads
    'W_intent': to3(p.W_intent), 'b_intent': p.b_intent,
    'W_thrplan': to3(p.W_thrplan), 'b_thrplan': p.b_thrplan,
    // Critic
    'W_val': to3(p.W_val), 'b_val': p.b_val,
  };
}

void savePolicy(String path, PolicyNetwork p) {
  final f = File(path);
  f.writeAsStringSync(const JsonEncoder.withIndent('  ').convert(_weightsToJson(p)));
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

bool tryLoadPolicy(String path, PolicyNetwork p) {
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
  final m = raw as Map<String, dynamic>;

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

  // Trunk
  total += 2;
  if (_tryFillMat(p.W1, 'W1')) ok++;
  if (_tryFillVec(p.b1, 'b1')) ok++;
  total += 2;
  if (_tryFillMat(p.W2, 'W2')) ok++;
  if (_tryFillVec(p.b2, 'b2')) ok++;

  // Legacy action heads (optional)
  total += 2;
  if (_tryFillMat(p.W_thr, 'W_thr')) ok++;
  if (_tryFillVec(p.b_thr, 'b_thr')) ok++;

  total += 2;
  if (_tryFillMat(p.W_turn, 'W_turn')) ok++;
  if (_tryFillVec(p.b_turn, 'b_turn')) ok++;

  // Planner heads
  total += 2;
  if (_tryFillMat(p.W_intent, 'W_intent')) ok++;
  if (_tryFillVec(p.b_intent, 'b_intent')) ok++;

  total += 2;
  if (_tryFillMat(p.W_thrplan, 'W_thrplan')) ok++;
  if (_tryFillVec(p.b_thrplan, 'b_thrplan')) ok++;

  // Critic
  total += 2;
  if (_tryFillMat(p.W_val, 'W_val')) ok++;
  if (_tryFillVec(p.b_val, 'b_val')) ok++;

  print('Loaded policy ← $path ($ok/$total tensors filled)');
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

/* ------------------------------------ main ------------------------------------ */

void main(List<String> argv) {
  final args = _Args(argv);

  final seed = args.getInt('seed', def: 7);

  // ---- RL / planner args (with aliases) ----
  final iters = args.getInt('train_iters', def: args.getInt('iters', def: 200));
  final batch = args.getInt('batch', def: 32);
  final lr = args.getDouble('lr', def: 3e-4);
  final valueBeta = args.getDouble('value_beta', def: args.getDouble('valueBeta', def: 0.5));
  final huberDelta = args.getDouble('huber_delta', def: 1.0);

  final planHold = args.getInt('plan_hold', def: 1);
  final tempIntent = args.getDouble('intent_temp', def: 1.0);
  final intentEntropy = args.getDouble('intent_entropy', def: args.getDouble('intentEntropyBeta', def: 0.0));
  final useLearned = args.getBool('use_learned_controller', def: args.getBool('use_learned', def: false));
  final blendPolicy = args.getDouble('blend_policy', def: 1.0);

  // Env args
  final lockTerrain = args.getBool('lock_terrain', def: false);
  final lockSpawn = args.getBool('lock_spawn', def: false);
  final randomSpawnX = !args.getBool('fixed_spawn_x', def: false);
  final maxFuel = args.getDouble('max_fuel', def: 1000.0);

  // Pretrain control (skipped by default since Trainer.pretrainIntentOnSnapshots may be absent)
  final doPretrain = args.getBool('do_pretrain', def: false);
  final determinism = args.getBool('determinism_probe', def: true);

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

  tryLoadPolicy('policy_pretrained.json', policy);

  if (determinism) {
    env.reset(seed: 1234);
    final a = _probeDeterminism(env, maxSteps: 165);
    env.reset(seed: 1234);
    final b = _probeDeterminism(env, maxSteps: 165);
    final ok = (a.steps == b.steps) && ((a.cost - b.cost).abs() < 1e-6);
    print('Determinism probe: steps ${a.steps} vs ${b.steps} | cost ${a.cost.toStringAsFixed(6)} vs ${b.cost.toStringAsFixed(6)} => ${ok ? "OK" : "MISMATCH"}');
  }

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
  );

  // ---------- Optional pretrain (off by default) ----------
  if (doPretrain) {
    print('Pretrain requested, but this build skips pretrain unless Trainer exposes it. Proceeding without pretrain.');
    // If you bring back Trainer.pretrainIntentOnSnapshots, you can wire it here.
    // final stats = trainer.pretrainIntentOnSnapshots(...);
    // savePolicy('policy_pretrained.json', policy);
  }

  {
    final ev = evaluate(env: env, trainer: trainer, episodes: 20, seed: seed ^ 0x999);
    print('Eval(real) → meanCost=${ev.meanCost.toStringAsFixed(3)} | median=${ev.medianCost.toStringAsFixed(3)} | land%=${ev.landPct.toStringAsFixed(1)} | crash%=${ev.crashPct.toStringAsFixed(1)} | steps=${ev.meanSteps.toStringAsFixed(1)} | mean|dx|=${ev.meanAbsDx.toStringAsFixed(1)}');
  }

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
      savePolicy('policy_iter_${it + 1}.json', policy);
    }
  }

  savePolicy('policy_final.json', policy);
  print('Training done. Saved → policy_final.json');
}

/* ----------------------------------- usage ------------------------------------

Example (your command with aliases supported):
  dart run lib/ai/train_agent.dart \
    --batch=1 --iters=400 --lr=2e-4 \
    --intentEntropyBeta=0.0002 --valueBeta=0.7 \
    --use_learned_controller=true --blend_policy=0.5

Optional pretrain (if you re-add it to Trainer):
  dart run lib/ai/train_agent.dart --do_pretrain
-------------------------------------------------------------------------------- */
