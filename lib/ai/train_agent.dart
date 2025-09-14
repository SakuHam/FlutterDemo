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
  String? getStr(String k, {String? def}) => _kv[k] ?? def;
  int getInt(String k, {int def = 0}) =>
      int.tryParse(_kv[k] ?? '') ?? def;
  double getDouble(String k, {double def = 0.0}) =>
      double.tryParse(_kv[k] ?? '') ?? def;
  bool getFlag(String k, {bool def = false}) =>
      _flags.contains(k) ? true : def;
}

/* ------------------------------- policy IO (json) ------------------------------ */

Map<String, dynamic> _weightsToJson(PolicyNetwork p) {
  List<List<double>> to3(List<List<double>> W) =>
      W.map((r) => r.map((v) => v.toDouble()).toList()).toList();

  return {
    'h1': p.h1,
    'h2': p.h2,
    'W1': to3(p.W1), 'b1': p.b1,
    'W2': to3(p.W2), 'b2': p.b2,
    'W_thr': to3(p.W_thr), 'b_thr': p.b_thr,
    'W_turn': to3(p.W_turn), 'b_turn': p.b_turn,
    'W_intent': to3(p.W_intent), 'b_intent': p.b_intent,
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

  // Trunk
  total += 2;
  if (_tryFillMat(p.W1, 'W1')) ok++;
  if (_tryFillVec(p.b1, 'b1')) ok++;
  total += 2;
  if (_tryFillMat(p.W2, 'W2')) ok++;
  if (_tryFillVec(p.b2, 'b2')) ok++;

  // Action heads (optional in old files)
  total += 2;
  if (_tryFillMat(p.W_thr, 'W_thr')) ok++;
  if (_tryFillVec(p.b_thr, 'b_thr')) ok++;

  total += 2;
  if (_tryFillMat(p.W_turn, 'W_turn')) ok++;
  if (_tryFillVec(p.b_turn, 'b_turn')) ok++;

  // Intent head
  total += 2;
  if (_tryFillMat(p.W_intent, 'W_intent')) ok++;
  if (_tryFillVec(p.b_intent, 'b_intent')) ok++;

  // Value head
  total += 2;
  if (_tryFillMat(p.W_val, 'W_val')) ok++;
  if (_tryFillVec(p.b_val, 'b_val')) ok++;

  final part = (ok == total) ? '' : '  (PARTIAL — IGNORING)';
  print('Loaded policy ← $path ($ok/$total tensors filled)$part');
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
  final pretrainN = args.getInt('pretrain_intent', def: 6000);
  final pretrainEpochs = args.getInt('pretrain_epochs', def: 2);
  final pretrainAlign = args.getDouble('pretrain_align', def: 2.0);
  final pretrainLr = args.getDouble('pretrain_lr', def: 3e-4);
  final onlyPretrain = args.getFlag('only_pretrain', def: false);

  final iters = args.getInt('train_iters', def: 200);
  final batch = args.getInt('batch', def: 32);
  final lr = args.getDouble('lr', def: 3e-4);
  final valueBeta = args.getDouble('value_beta', def: 0.5);
  final huberDelta = args.getDouble('huber_delta', def: 1.0);

  final planHold = args.getInt('plan_hold', def: 1);
  final tempIntent = args.getDouble('intent_temp', def: 1.0);
  final intentEntropy = args.getDouble('intent_entropy', def: 0.0);
  final useLearned = args.getFlag('use_learned_controller', def: false);
  final blendPolicy = args.getDouble('blend_policy', def: 1.0);

  // New knobs to steer early supervision from CLI
  final intentAlignWeight = args.getDouble('intent_align', def: 0.25);
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
    // NEW
    intentAlignWeight: intentAlignWeight,
    actionAlignWeight: actionAlignWeight,
    normalizeFeatures: true, // turn on online feature normalization
  );

  print('Pretraining intent on $pretrainN snapshots (epochs=$pretrainEpochs, align=$pretrainAlign, lr=$pretrainLr) ...');
  final stats = trainer.pretrainIntentOnSnapshots(
    samples: pretrainN,
    epochs: pretrainEpochs,
    alignWeight: pretrainAlign,
    lr: pretrainLr,
    seed: seed ^ 0xABCD1234,
  );

  final acc = (stats['acc'] ?? 0.0) * 100.0;
  final n = (stats['n'] ?? 0.0).toInt();
  print('Pretrain done → acc=${acc.toStringAsFixed(1)}% over n=$n samples');

  savePolicy('policy_pretrained.json', policy);

  if (onlyPretrain) {
    print('Only-pretrain mode: saved → policy_pretrained.json. Exiting.');
    return;
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

Pretrain only:
  dart run lib/ai/train_agent.dart \
    --pretrain_intent=8000 --pretrain_epochs=2 --pretrain_align=2.0 \
    --pretrain_lr=0.0005 --only_pretrain

Full train (phase 1: learn intents, heuristic controller):
  dart run lib/ai/train_agent.dart \
    --pretrain_intent=8000 --pretrain_epochs=2 --pretrain_align=2.0 --pretrain_lr=0.0005 \
    --plan_hold=2 --use_learned_controller=false \
    --train_iters=200 --batch=16 --lr=0.0003 \
    --intent_align=0.25 --action_align=0.0 --intent_entropy=0.0001

Phase 2 (distill controller once intents are stable):
  dart run lib/ai/train_agent.dart \
    --train_iters=150 --batch=16 --lr=0.0003 \
    --use_learned_controller=true --blend_policy=0.5 \
    --intent_align=0.10 --action_align=0.6 --intent_entropy=0.00005
-------------------------------------------------------------------------------- */
