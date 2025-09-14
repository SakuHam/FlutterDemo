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

bool tryLoadPolicyStrict(String path, PolicyNetwork p) {
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

  final strict = (ok == total);
  print('Loaded policy ← $path ($ok/$total tensors filled)  ${strict ? "OK" : "(PARTIAL — IGNORING)"}');
  return strict;
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

/* ---------------------------- local pretrain routine --------------------------- */

/// Simple supervised pretrain on synthetic snapshots with predictive labels.
/// Returns accuracy over a fresh probe set.
Map<String, double> pretrainIntentOnSnapshots({
  required eng.GameEngine env,
  required FeatureExtractor fe,
  required PolicyNetwork policy,
  int samples = 6000,
  int epochs = 2,
  double lr = 3e-4,
  double alignWeight = 2.0,
  int seed = 1337,
}) {
  final rng = math.Random(seed);

  // Lock terrain for stable labels
  env.reset(seed: 123456);
  fe.reset();

  // Helpers to set a synthetic state
  void _setState({
    required double x,
    required double height,
    required double vx,
    required double vy,
  }) {
    final gy = env.terrain.heightAt(x);
    final y = (gy - height).clamp(0.0, env.cfg.worldH - 10.0);
    env.lander.pos.x = x.clamp(10.0, env.cfg.worldW - 10.0);
    env.lander.pos.y = y;
    env.lander.vel.x = vx;
    env.lander.vel.y = vy;
    env.lander.angle = 0.0;
    env.lander.fuel  = env.cfg.t.maxFuel;
    // reset FE velocity history so accel feature doesn't spike due to teleports
    fe.reset();
  }

  // Synthesize examples for each intent
  void _synthForIntent(Intent it) {
    final padCx = env.terrain.padCenter.toDouble();
    final padHalfW = (((env.terrain.padX2 - env.terrain.padX1).abs()) * 0.5)
        .clamp(12.0, env.cfg.worldW.toDouble());

    double x=padCx, h=200, vx=0, vy=20;
    switch (it) {
      case Intent.goLeft:
        x  = padCx + (0.90 + 0.08 * rng.nextDouble()) * padHalfW;
        h  = 160.0 + 120.0 * rng.nextDouble();
        vx = 60.0 + 40.0 * rng.nextDouble();
        vy = 30.0 + 40.0 * rng.nextDouble();
        break;
      case Intent.goRight:
        x  = padCx - (0.90 + 0.08 * rng.nextDouble()) * padHalfW;
        h  = 160.0 + 120.0 * rng.nextDouble();
        vx = -(60.0 + 40.0 * rng.nextDouble());
        vy = 30.0 + 40.0 * rng.nextDouble();
        break;
      case Intent.descendSlow:
        x  = padCx + (rng.nextDouble()*0.06 - 0.03) * padHalfW;
        h  = 0.65 * env.cfg.worldH + 0.15 * env.cfg.worldH * rng.nextDouble();
        vx = (rng.nextDouble()*30.0) - 15.0;
        vy = 28.0 + 12.0 * rng.nextDouble();
        break;
      case Intent.brakeUp:
        x  = padCx + (rng.nextDouble()*0.06 - 0.03) * padHalfW;
        h  = 40.0 + 40.0 * rng.nextDouble();
        vx = (rng.nextDouble()*18.0) - 9.0;
        vy = 140.0 + 30.0 * rng.nextDouble();
        break;
      case Intent.hoverCenter:
        x  = padCx + (rng.nextDouble()*0.04 - 0.02) * padHalfW;
        h  = 0.22 * env.cfg.worldH + 0.02 * env.cfg.worldH * rng.nextDouble();
        vx = (rng.nextDouble()*14.0) - 7.0;
        vy = 4.0 + 6.0 * rng.nextDouble();
        break;
    }

    _setState(x: x, height: h, vx: vx, vy: vy);
  }

  int _labelNow() => predictiveIntentLabelAdaptive(
    env, baseTauSec: 1.0, minTauSec: 0.45, maxTauSec: 1.35,
  );

  // Mini-batch buffers
  const int B = 32;
  final decisionCaches = [];
  final decisionRewards = <double>[]; // zeros during pretrain; we use align loss
  final intentChoices  = <int>[];
  final alignLabels    = <int>[];

  void _accOne(dynamic cache, int yIdx) {
    decisionCaches.add(cache);
    intentChoices.add(yIdx);
    alignLabels.add(yIdx);
    decisionRewards.add(0.0); // no RL signal in pretrain
  }

  void _flush() {
    if (decisionCaches.isEmpty) return;
    policy.updateFromEpisode(
      decisionCaches: decisionCaches as dynamic,
      intentChoices: intentChoices,
      decisionRewards: decisionRewards,
      gamma: 0.99,                // unused because rewards are zeros
      gaeLambda: 0.95,
      bootstrap: false,
      alignLabels: alignLabels,
      alignWeight: alignWeight,   // supervised CE on intent head
      lr: lr,
      entropyBeta: 0.0,
      valueBeta: 0.0,             // no critic in pretrain
      huberDelta: 1.0,
    );
    decisionCaches.clear();
    decisionRewards.clear();
    intentChoices.clear();
    alignLabels.clear();
  }

  // small held-out probe
  double _probeAcc(int N) {
    int ok=0;
    for (int i=0;i<N;i++) {
      final it = Intent.values[rng.nextInt(Intent.values.length)];
      _synthForIntent(it);
      final y = _labelNow();
      final pred = policy.actIntentGreedy(fe.extract(env)).$1;
      if (pred == y) ok++;
    }
    return N == 0 ? 0.0 : ok / N;
  }

  // Training
  final intents = Intent.values;
  final perClass = (samples / intents.length).ceil();

  for (int e = 0; e < epochs; e++) {
    final baseOrder = intents.toList()..shuffle(rng);

    for (int i = 0; i < perClass; i++) {
      final order = baseOrder.toList()..shuffle(rng);
      for (final it in order) {
        _synthForIntent(it);
        final x = fe.extract(env);
        final yIdx = _labelNow();
        final res = policy.actIntentGreedy(x);
        _accOne(res.$3, yIdx);

        if (decisionCaches.length >= B) _flush();
      }
    }
    _flush();
    final acc = _probeAcc(400);
    print('pretrain epoch ${e+1}/$epochs  probe acc=${(acc*100).toStringAsFixed(1)}%');
  }

  final acc = _probeAcc(1200);
  return {'acc': acc, 'n': 1200.0};
}

/* ------------------------------------ main ------------------------------------ */

void main(List<String> argv) {
  final args = _Args(argv);

  // General / training knobs
  final seed = args.getInt('seed', def: 7);

  // Pretrain knobs (+ aliases)
  final pretrainN = args.getInt('pretrain_intent', def: 6000);
  final pretrainEpochs = args.getInt('pretrain_epochs', def: 2);
  final pretrainAlign = args.getDouble('pretrain_align', def: 2.0);
  final pretrainLr = args.getDouble('pretrain_lr', def: 3e-4);
  final onlyPretrain = args.getFlag('only_pretrain', def: args.getFlag('only_pretrain=true', def: false));

  // RL knobs (+ aliases for your previous runs)
  final iters = args.getInt('train_iters', def: args.getInt('iters', def: 200));
  final batch = args.getInt('batch', def: 32);
  final lr = args.getDouble('lr', def: 3e-4);
  final valueBeta = args.getDouble('value_beta', def: args.getDouble('valueBeta', def: 0.5));
  final huberDelta = args.getDouble('huber_delta', def: 1.0);

  final planHold = args.getInt('plan_hold', def: 1);
  final tempIntent = args.getDouble('intent_temp', def: 1.0);
  final intentEntropy =
  args.getDouble('intent_entropy', def: args.getDouble('intentEntropyBeta', def: 0.0));
  final useLearned = args.getFlag('use_learned_controller', def: args.getFlag('use_learned_controller=true', def: false));
  final blendPolicy = args.getDouble('blend_policy', def: 1.0);

  // Env toggles
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

  final fe = FeatureExtractor(groundSamples: 3, stridePx: 48, dt: 1/60.0);
  final policy = PolicyNetwork(inputSize: fe.inputSize, h1: 64, h2: 64, seed: seed);
  print('Loaded init policy. h1=${policy.h1} h2=${policy.h2} | FE(gs=${fe.groundSamples} stride=${fe.stridePx})');

  // Load if exact match; ignore partials
  final loaded = tryLoadPolicyStrict('policy_pretrained.json', policy);

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
    segHardMax: 150,
    segMin: 50,
  );

  // Always (re)pretrain unless user really wants to skip — ensures 14/14 tensors match the new heads.
  print('Pretraining intent on $pretrainN snapshots (epochs=$pretrainEpochs, align=$pretrainAlign, lr=$pretrainLr) ...');
  final stats = pretrainIntentOnSnapshots(
    env: env,
    fe: fe,
    policy: policy,
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
    --pretrain_intent=10000 --pretrain_epochs=2 --pretrain_align=2.0 \
    --pretrain_lr=0.0005 --only_pretrain

Full train (heuristic turns; segmented GAE inside agent):
  dart run lib/ai/train_agent.dart \
    --pretrain_intent=10000 --pretrain_epochs=2 --pretrain_align=2.0 --pretrain_lr=0.0005 \
    --plan_hold=2 --use_learned_controller=false --blend_policy=0.0 \
    --train_iters=300 --batch=24 --lr=0.0003 --value_beta=0.5
-------------------------------------------------------------------------------- */
