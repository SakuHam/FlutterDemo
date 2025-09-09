// lib/ai/train_agent.dart
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'agent.dart'
    show
    FeatureExtractor,
    PolicyNetwork,
    Trainer,
    relu,
    reluVec,
    sigmoid,
    softmax;

import '../engine/game_engine.dart' as eng;

/// ---------- Tiny IO helpers ----------
Future<Map<String, dynamic>> _readJson(String path) async =>
    json.decode(await File(path).readAsString()) as Map<String, dynamic>;

Future<void> _writePrettyJson(String path, Map<String, dynamic> js) async {
  final out = const JsonEncoder.withIndent('  ').convert(js);
  await File(path).writeAsString(out);
  stdout.writeln('Saved policy → $path');
}

List<List<double>> _mat(dynamic m) => (m as List)
    .map<List<double>>((r) => (r as List).map<double>((x) => (x as num).toDouble()).toList())
    .toList();
List<double> _vec(dynamic v) => (v as List).map<double>((x) => (x as num).toDouble()).toList();

(PolicyNetwork policy, int gs, double stride) _loadPolicyInto(
    PolicyNetwork policy,
    Map<String, dynamic> js,
    ) {
  final inputSizeJ = js['inputSize'] as int? ?? policy.inputSize;
  final h1J = js['h1'] as int? ?? policy.h1;
  final h2J = js['h2'] as int? ?? policy.h2;

  if (policy.inputSize != inputSizeJ || policy.h1 != h1J || policy.h2 != h2J) {
    policy = PolicyNetwork(inputSize: inputSizeJ, h1: h1J, h2: h2J, seed: 1234);
  }

  policy.W1 = _mat(js['W1']); policy.b1 = _vec(js['b1']);
  policy.W2 = _mat(js['W2']); policy.b2 = _vec(js['b2']);

  if (js.containsKey('W_thr') && js.containsKey('W_turn')) {
    policy.W_thr = _mat(js['W_thr']); policy.b_thr = _vec(js['b_thr']);
    policy.W_turn = _mat(js['W_turn']); policy.b_turn = _vec(js['b_turn']);
  } else {
    throw ArgumentError('Policy JSON is missing W_thr/W_turn heads.');
  }

  final fe = (js['fe'] as Map<String, dynamic>?) ?? const {};
  final gs = (fe['groundSamples'] as int?) ?? (inputSizeJ - 10); // fallback if missing
  final stride = (fe['stridePx'] as num?)?.toDouble() ?? 48.0;

  return (policy, gs, stride);
}

Future<void> _savePolicy(
    String path,
    PolicyNetwork p,
    int inputSize,
    int gs,
    double stride,
    ) async {
  final js = {
    'inputSize': inputSize,
    'h1': p.h1,
    'h2': p.h2,
    'W1': p.W1,
    'b1': p.b1,
    'W2': p.W2,
    'b2': p.b2,
    'W_thr': p.W_thr,
    'b_thr': p.b_thr,
    'W_turn': p.W_turn,
    'b_turn': p.b_turn,
    'fe': {'groundSamples': gs, 'stridePx': stride},
  };
  await _writePrettyJson(path, js);
}

void main(List<String> args) async {
  // CLI
  String? initPath;
  int iters = 20000;
  int saveEvery = 50;
  int batch = 16;              // K episodes per outer iteration
  double baseLr = 3e-4;        // per-outer LR, per-ep LR = baseLr / batch
  double tempThr = 0.9;        // exploration temps
  double tempTurn = 1.2;
  double epsilon = 0.05;       // ε-greedy
  double entropyBeta = 0.003;  // entropy bonus

  for (final a in args) {
    if (a.startsWith('--init=')) initPath = a.substring(7).trim();
    if (a.startsWith('--iters=')) iters = int.parse(a.substring(8));
    if (a.startsWith('--save_every=')) saveEvery = int.parse(a.substring(13));
    if (a.startsWith('--batch=')) batch = int.parse(a.substring(8));
    if (a.startsWith('--lr=')) baseLr = double.parse(a.substring(5));
    if (a.startsWith('--temp_thr=')) tempThr = double.parse(a.substring(11));
    if (a.startsWith('--temp_turn=')) tempTurn = double.parse(a.substring(12));
    if (a.startsWith('--eps=')) epsilon = double.parse(a.substring(6));
    if (a.startsWith('--entropy=')) entropyBeta = double.parse(a.substring(10));
  }

  // Engine config (Flutter-free)
  final cfg = eng.EngineConfig(
    t: eng.Tunables(
      gravity: 0.18,
      thrustAccel: 0.42,
      rotSpeed: 1.6,
      maxFuel: 100.0,
    ),
    worldW: 360,
    worldH: 640,
    lockTerrain: false,
    terrainSeed: 12345,
    lockSpawn: false,
    randomSpawnX: true,
    spawnXMin: 0.20,
    spawnXMax: 0.80,
    landingSpeedMax: 12.0,                         // stricter landing
    landingAngleMaxRad: 8 * math.pi / 180.0,
    wDx: 200.0,
    wDy: 180.0,
    wVx: 90.0,
    wVyDown: 200.0,                                // higher with stricter landings
    wAngleDeg: 80.0,
  );

  final env = eng.GameEngine(cfg);

  // Feature Extractor defaults
  int feGS = 3;
  double feStride = 48.0;

  // Policy
  var policy = PolicyNetwork(
    inputSize: FeatureExtractor(groundSamples: feGS, stridePx: feStride).inputSize,
    h1: 64,
    h2: 64,
    seed: 1234,
  );

  // Load init (BC or previous)
  Map<String, dynamic>? initJs;
  if (initPath != null && await File(initPath).exists()) {
    initJs = await _readJson(initPath!);
  } else {
    if (await File('policy_bc_init.json').exists()) {
      initJs = await _readJson('policy_bc_init.json');
    } else if (await File('policy.json').exists()) {
      initJs = await _readJson('policy.json');
    }
  }

  if (initJs != null) {
    final loaded = _loadPolicyInto(policy, initJs);
    policy = loaded.$1;
    feGS = loaded.$2;
    feStride = loaded.$3;

    // If FE size still mismatches, FORCE-align FE to the policy:
    final expectedGS = policy.inputSize - 10;
    if (feGS != expectedGS) {
      stderr.writeln(
        'WARNING: FE(gs=$feGS) -> aligning to policy.inputSize=${policy.inputSize} (gs=$expectedGS).',
      );
      feGS = expectedGS;
    }

    // final consistency check
    final feCheck = FeatureExtractor(groundSamples: feGS, stridePx: feStride);
    if (feCheck.inputSize != policy.inputSize) {
      stderr.writeln(
        'ERROR: FeatureExtractor.inputSize=${feCheck.inputSize} but policy.inputSize=${policy.inputSize}. '
            'Adjust code/JSON so they match.',
      );
      // hard stop to avoid runtime RangeError
      exit(1);
    }

    stdout.writeln('Loaded init policy. h1=${policy.h1} h2=${policy.h2} | FE(gs=$feGS stride=$feStride)');
  } else {
    stdout.writeln('No init policy found; training from scratch.');
  }

  // Trainer
  final rnd = math.Random(7);
  final trainer = Trainer(
    env: env,
    fe: FeatureExtractor(groundSamples: feGS, stridePx: feStride),
    policy: policy,
    dt: 1 / 60.0,
    gamma: 0.99,
    seed: 13,
    tempThr: tempThr,
    tempTurn: tempTurn,
    epsilon: epsilon,
    entropyBeta: entropyBeta,
  );

  // Training loop (batched)
  double bestCost = double.infinity;
  double bestLandRate = 0.0;
  int landedTotal = 0;
  final landWindow = <bool>[];
  const landWindowSize = 200;

  for (int it = 1; it <= iters; it++) {
    final epLr = baseLr / batch;
    double lastCost = 0.0;
    int lastSteps = 0;
    bool lastLanded = false;

    for (int k = 0; k < batch; k++) {
      env.reset(seed: rnd.nextInt(1 << 30));
      final ep = trainer.runEpisode(train: true, lr: epLr);

      lastCost = ep.totalCost;
      lastSteps = ep.steps;
      lastLanded = ep.landed;

      if (ep.landed) landedTotal++;
      landWindow.add(ep.landed);
      if (landWindow.length > landWindowSize) landWindow.removeAt(0);

      if (ep.totalCost < bestCost) {
        bestCost = ep.totalCost;
        await _savePolicy('policy_best_cost.json', policy, trainer.fe.inputSize, feGS, feStride);
      }
    }

    if (it % 5 == 0) {
      final landRate = landWindow.isEmpty
          ? 0.0
          : (landWindow.where((x) => x).length / landWindow.length * 100.0);

      if (landRate > bestLandRate) {
        bestLandRate = landRate;
        await _savePolicy('policy_best_land.json', policy, trainer.fe.inputSize, feGS, feStride);
      }

      stdout.writeln(
        'Iter $it | batch=$batch | last-ep steps: $lastSteps | cost: ${lastCost.toStringAsFixed(3)} '
            '| landed: ${lastLanded ? 'Y' : 'N'} | best(cost): ${bestCost.toStringAsFixed(3)} '
            '| land%(${landWindowSize})=${landRate.toStringAsFixed(1)} | best(land%)=${bestLandRate.toStringAsFixed(1)}',
      );
    }

    if (it % saveEvery == 0) {
      await _savePolicy('policy.json', policy, trainer.fe.inputSize, feGS, feStride);
    }
  }

  await _savePolicy('policy.json', policy, trainer.fe.inputSize, feGS, feStride);
  stdout.writeln('Done. Best cost: ${bestCost.toStringAsFixed(3)}');
}
