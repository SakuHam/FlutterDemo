// lib/ai/train_agent.dart
import 'dart:convert';
import 'dart:io';

import '../engine/game_engine.dart' as eng;
import '../engine/types.dart' as et;

import 'agent.dart';

/// ------------------------ tiny flag/arg parser (no package:args) ------------------------
class Flags {
  final Map<String, String> _kv = {};
  final Set<String> _bools = {};
  Flags(List<String> args) {
    for (final a in args) {
      if (!a.startsWith('--')) continue;
      final s = a.substring(2);
      final eq = s.indexOf('=');
      if (eq >= 0) {
        _kv[s.substring(0, eq)] = s.substring(eq + 1);
      } else {
        _bools.add(s);
      }
    }
  }
  String getS(String k, String d) => _kv.containsKey(k) ? _kv[k]! : d;
  int getI(String k, int d) => int.tryParse(getS(k, d.toString())) ?? d;
  double getD(String k, double d) => double.tryParse(getS(k, d.toString())) ?? d;
  bool getB(String k, bool d) =>
      _bools.contains(k) ? true : (getS(k, d ? 'true' : 'false') == 'true');
}

/// ----------------------------- JSON I/O helpers -----------------------------
Map<String, dynamic> _matToJson(List<List<double>> M) => {
  'rows': M.length,
  'cols': M.isEmpty ? 0 : M[0].length,
  'data': M.expand((r) => r).toList(),
};

List<List<double>> _matFromJson(Map<String, dynamic> j) {
  final rows = j['rows'] as int;
  final cols = j['cols'] as int;
  final flat =
  (j['data'] as List).cast<num>().map((e) => e.toDouble()).toList();
  final out = <List<double>>[];
  for (int i = 0; i < rows; i++) {
    out.add(List<double>.from(flat.sublist(i * cols, (i + 1) * cols)));
  }
  return out;
}

Map<String, dynamic> policyToJson(PolicyNetwork p) => {
  'inputSize': p.inputSize,
  'h1': p.h1,
  'h2': p.h2,
  'W1': _matToJson(p.W1),
  'W2': _matToJson(p.W2),
  'b1': p.b1,
  'b2': p.b2,
  'W_thr': _matToJson(p.W_thr),
  'b_thr': p.b_thr,
  'W_turn': _matToJson(p.W_turn),
  'b_turn': p.b_turn,
  'W_intent': _matToJson(p.W_intent),
  'b_intent': p.b_intent,
  'W_val': _matToJson(p.W_val),
  'b_val': p.b_val,
};

void policyFromJson(PolicyNetwork p, Map<String, dynamic> j) {
  p.W1 = _matFromJson(j['W1']);
  p.b1 = (j['b1'] as List).cast<num>().map((e) => e.toDouble()).toList();
  p.W2 = _matFromJson(j['W2']);
  p.b2 = (j['b2'] as List).cast<num>().map((e) => e.toDouble()).toList();
  p.W_thr = _matFromJson(j['W_thr']);
  p.b_thr = (j['b_thr'] as List).cast<num>().map((e) => e.toDouble()).toList();
  p.W_turn = _matFromJson(j['W_turn']);
  p.b_turn =
      (j['b_turn'] as List).cast<num>().map((e) => e.toDouble()).toList();
  p.W_intent = _matFromJson(j['W_intent']);
  p.b_intent =
      (j['b_intent'] as List).cast<num>().map((e) => e.toDouble()).toList();
  p.W_val = _matFromJson(j['W_val']);
  p.b_val = (j['b_val'] as List).cast<num>().map((e) => e.toDouble()).toList();
}

void savePolicy(PolicyNetwork p, String path) {
  final j = policyToJson(p);
  File(path)
      .writeAsStringSync(const JsonEncoder.withIndent('  ').convert(j));
  print('Saved policy → $path');
}

bool loadPolicyIfExists(PolicyNetwork p, String path) {
  final f = File(path);
  if (!f.existsSync()) return false;
  final j = jsonDecode(f.readAsStringSync()) as Map<String, dynamic>;
  policyFromJson(p, j);
  print('Loaded policy from $path');
  return true;
}

/// ----------------------------- Determinism probe -----------------------------
class _ProbeStats {
  final int steps;
  final double cost;
  _ProbeStats(this.steps, this.cost);
}

_ProbeStats _roll(eng.GameEngine env, {int maxSteps = 1000, int seed = 1}) {
  env.reset(seed: seed);
  double cost = 0.0;
  int t = 0;
  while (true) {
    final info = env.step(
        1 / 60.0, const et.ControlInput(thrust: false, left: false, right: false));
    cost += info.costDelta;
    if (info.terminal || t >= maxSteps) break;
    t++;
  }
  return _ProbeStats(t, cost);
}

void determinismProbe(eng.GameEngine env) {
  final a = _roll(env, maxSteps: 1000, seed: 42);
  final b = _roll(env, maxSteps: 1000, seed: 42);
  final ok =
      (a.steps == b.steps) && ((a.cost - b.cost).abs() < 1e-6);
  print(
      'Determinism probe: steps ${a.steps} vs ${b.steps} | cost ${a.cost} vs ${b.cost} => ${ok ? "OK" : "MISMATCH"}');
}

/// ----------------------------------- Main ------------------------------------
void main(List<String> args) async {
  final f = Flags(args);

  // IO
  final initPath = f.getS('init_policy', '');
  final savePath = f.getS('save', 'policy.json');
  final savePretrainedPath =
  f.getS('save_pretrained', 'policy_pretrained.json');

  // Feature extractor
  final gs = f.getI('fe_gs', 3);
  final stride = f.getD('fe_stride', 48.0);

  // Net sizes
  final h1 = f.getI('h1', 64);
  final h2 = f.getI('h2', 64);

  // Seeds
  final seed = f.getI('seed', 7);

  // World & tunables (match your EngineConfig ctor)
  final worldW = f.getD('world_w', 800.0);
  final worldH = f.getD('world_h', 480.0);
  final maxFuel = f.getD('max_fuel', 1000.0);
  final gravity = f.getD('gravity', 0.18);
  final thrustAccel = f.getD('thrust_accel', 0.42);
  final rotSpeed = f.getD('rot_speed', 1.6);

  // Training loop
  final episodes = f.getI('episodes', 400);
  final greedyEval = f.getB('greedy_eval', false);
  final scoreIsReward = f.getB('score_is_reward', false);

  // Single-stage exploration (if two_stage=false)
  final singleTempThr = f.getD('temp_thr', 1.0);
  final singleTempTurn = f.getD('temp_turn', 1.0);
  final singleEps = f.getD('epsilon', 0.0);
  final singleEntropy = f.getD('entropy_beta', 0.0);

  // Two-stage planner
  final twoStage = f.getB('two_stage', true);
  final planHold = f.getI('plan_hold', 1);
  final tempIntent = f.getD('temp_intent', 1.0);
  final intentEntropyBeta = f.getD('intent_entropy_beta', 0.0);
  final valueBeta = f.getD('value_beta', 0.5);
  final huberDelta = f.getD('huber_delta', 1.0);
  final lr = f.getD('lr', 3e-4);

  // Learned controller
  final useLearnedController = f.getB('use_learned_controller', false);
  final blendPolicy = f.getD('blend_policy', 1.0);

  // Pretrain intent
  final preN = f.getI('pretrain_intent', 0);
  final preEpochs = f.getI('pretrain_epochs', 2);
  final preAlign = f.getD('pretrain_align', 2.0);
  final preLr = f.getD('pretrain_lr', 3e-4);
  final onlyPretrain = f.getB('only_pretrain', false);

  // Optional safety knob echo (agent has fixed 50.0 margin for now)
  final brakeMargin = f.getD('brake_margin', 50.0);
  print('Note: brake_margin=$brakeMargin (agent.dart currently hardcodes its margin).');

  // --- Build EngineConfig & env (your ctor requires worldW, worldH, t)
  final tun = et.Tunables(
    gravity: gravity,
    thrustAccel: thrustAccel,
    rotSpeed: rotSpeed,
    maxFuel: maxFuel,
  );

  final cfg = et.EngineConfig(
    worldW: worldW,
    worldH: worldH,
    t: tun,
    seed: seed,
    // The rest use your EngineConfig defaults. Override here if desired:
    // lockTerrain: true, terrainSeed: 12345, randomSpawnX: true, etc.
  );

  final env = eng.GameEngine(cfg);

  final fe = FeatureExtractor(groundSamples: gs, stridePx: stride);
  final policy =
  PolicyNetwork(inputSize: fe.inputSize, h1: h1, h2: h2, seed: seed);

  // Load initial policy if provided
  if (initPath.isNotEmpty && File(initPath).existsSync()) {
    loadPolicyIfExists(policy, initPath);
  } else {
    print('Loaded init policy. h1=$h1 h2=$h2 | FE(gs=$gs stride=$stride)');
  }

  // Determinism check
  determinismProbe(env);

  // --- Trainer
  final trainer = Trainer(
    env: env,
    fe: fe,
    policy: policy,
    dt: 1 / 60.0,
    gamma: 0.99,
    seed: seed,
    // single-stage
    tempThr: singleTempThr,
    tempTurn: singleTempTurn,
    epsilon: singleEps,
    entropyBeta: singleEntropy,
    // two-stage
    twoStage: twoStage,
    planHold: planHold,
    tempIntent: tempIntent,
    intentEntropyBeta: intentEntropyBeta,
    // learned controller
    useLearnedController: useLearnedController,
    blendPolicy: blendPolicy,
  );

  // --- Pretrain intent head (optional)
  if (preN > 0) {
    print(
        'Pretraining intent on $preN snapshots (epochs=$preEpochs, align=$preAlign, lr=$preLr) ...');
    final stats = trainer.pretrainIntentOnSnapshots(
      samples: preN,
      epochs: preEpochs,
      lr: preLr,
      alignWeight: preAlign,
      seed: seed ^ 0xA5A5A5A5,
    );
    final accPct = (stats['acc'] ?? 0.0) * 100.0;
    print(
        'Pretrain eval → acc=${accPct.toStringAsFixed(1)}% over n=${stats['n']?.toInt()} samples');
    savePolicy(policy, savePretrainedPath);

    if (onlyPretrain) {
      print('Only-pretrain mode: saved → $savePretrainedPath. Exiting.');
      return;
    }
  }

  // --- Main training loop
  double bestScore = double.infinity;
  String bestPath = savePath;

  for (int ep = 0; ep < episodes; ep++) {
    env.reset(seed: seed + ep * 1337);

    final res = trainer.runEpisode(
      train: true,
      lr: lr,
      greedy: false,
      scoreIsReward: scoreIsReward,
      valueBeta: valueBeta,
      huberDelta: huberDelta,
    );

    final landed = res.landed ? '✓' : '×';
    final cost = res.totalCost.toStringAsFixed(3);

    if (twoStage) {
      final switches = res.intentSwitches;
      final countsStr = res.intentCounts.isEmpty
          ? ''
          : ' | intents=' +
          List.generate(
              PolicyNetwork.kIntents, (i) => res.intentCounts[i])
              .join(',');
      print(
          'EP ${ep.toString().padLeft(4)} | steps=${res.steps.toString().padLeft(4)} | cost=$cost | landed=$landed | switches=$switches$countsStr');
    } else {
      print(
          'EP ${ep.toString().padLeft(4)} | steps=${res.steps.toString().padLeft(4)} | cost=$cost | landed=$landed '
              '| turn=${res.turnSteps} L=${res.leftSteps} R=${res.rightSteps} thr=${res.thrustSteps} p_thr=${res.avgThrProb.toStringAsFixed(2)}');
    }

    if (res.totalCost < bestScore) {
      bestScore = res.totalCost;
      bestPath = savePath;
      savePolicy(policy, bestPath);
    }

    if (ep % 50 == 0 && ep > 0) {
      final cp = savePath.replaceFirst('.json', '_ep$ep.json');
      savePolicy(policy, cp);
    }

    if (greedyEval && (ep % 20 == 0)) {
      env.reset(seed: seed + 777 + ep);
      final evalRes = trainer.runEpisode(
        train: false,
        lr: lr,
        greedy: true,
        scoreIsReward: scoreIsReward,
        valueBeta: valueBeta,
        huberDelta: huberDelta,
      );
      final eCost = evalRes.totalCost.toStringAsFixed(3);
      final eLanded = evalRes.landed ? '✓' : '×';
      print('  [GreedyEval] steps=${evalRes.steps} cost=$eCost landed=$eLanded');
    }
  }

  print(
      'Training complete. Best cost=${bestScore.toStringAsFixed(3)} → saved $bestPath');
}
