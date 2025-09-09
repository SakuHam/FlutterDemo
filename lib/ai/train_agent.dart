// lib/ai/train_agent.dart
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'agent.dart'
    show
    FeatureExtractor,
    PolicyNetwork,
    Trainer,
    // matrix helpers only if you need them elsewhere (not required here)
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

/// Convert dynamic JSON -> List<List<double>>
List<List<double>> _mat(dynamic m) => (m as List)
    .map<List<double>>(
      (r) => (r as List).map<double>((x) => (x as num).toDouble()).toList(),
)
    .toList();

/// Convert dynamic JSON -> List<double>
List<double> _vec(dynamic v) =>
    (v as List).map<double>((x) => (x as num).toDouble()).toList();

/// Load weights into an existing or (if needed) re-created network.
/// Returns (policy, feGroundSamples, feStridePx).
(PolicyNetwork policy, int gs, double stride) _loadPolicyInto(
    PolicyNetwork policy,
    Map<String, dynamic> js,
    ) {
  final inputSizeJ = js['inputSize'] as int? ?? policy.inputSize;
  final h1J = js['h1'] as int? ?? policy.h1;
  final h2J = js['h2'] as int? ?? policy.h2;

  // Rebuild the net if shapes differ
  if (policy.inputSize != inputSizeJ || policy.h1 != h1J || policy.h2 != h2J) {
    policy = PolicyNetwork(inputSize: inputSizeJ, h1: h1J, h2: h2J, seed: 1234);
  }

  // Mandatory weights (new head layout)
  policy.W1 = _mat(js['W1']);
  policy.b1 = _vec(js['b1']);
  policy.W2 = _mat(js['W2']);
  policy.b2 = _vec(js['b2']);

  if (js.containsKey('W_thr') && js.containsKey('W_turn')) {
    policy.W_thr = _mat(js['W_thr']);
    policy.b_thr = _vec(js['b_thr']);
    policy.W_turn = _mat(js['W_turn']);
    policy.b_turn = _vec(js['b_turn']);
  } else {
    throw ArgumentError(
      'Policy JSON is missing W_thr/W_turn heads. Make sure you exported the newer format.',
    );
  }

  // Feature-extractor settings baked in the JSON (fallbacks if absent)
  final fe = (js['fe'] as Map<String, dynamic>?) ?? const {};
  final gs = (fe['groundSamples'] as int?) ?? 3;
  final stride = (fe['stridePx'] as num?)?.toDouble() ?? 48.0;

  return (policy, gs, stride);
}

/// Save policy + the FE settings you actually trained with.
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
  // ---- CLI flags ----
  // Usage examples:
  //   dart run lib/ai/train_agent.dart
  //   dart run lib/ai/train_agent.dart --init=policy_bc_init.json
  //   dart run lib/ai/train_agent.dart --iters=2000 --save_every=100
  String? initPath;
  int iters = 20000;
  int saveEvery = 50;

  for (final a in args) {
    if (a.startsWith('--init=')) initPath = a.substring('--init='.length).trim();
    if (a.startsWith('--iters=')) iters = int.parse(a.substring('--iters='.length));
    if (a.startsWith('--save_every=')) saveEvery = int.parse(a.substring('--save_every='.length));
  }

  // ---- Engine config (keep it Flutter-free) ----
  final cfg = eng.EngineConfig(
    t: eng.Tunables(
      gravity: 0.18,
      thrustAccel: 0.42,
      rotSpeed: 1.6,
      maxFuel: 100.0,
    ),
    worldW: 360,
    worldH: 640,
    lockTerrain: false,       // random terrain by default (set true to overfit)
    terrainSeed: 12345,       // used only when lockTerrain=true
    lockSpawn: true,          // fixed Y/vel/angle, X still random if randomSpawnX=true
    randomSpawnX: true,
    spawnXMin: 0.20,
    spawnXMax: 0.80,
    // landing tolerances same as your latest engine
    landingSpeedMax: 40.0,
    landingAngleMaxRad: 0.25,
    // cost weights (tune as you like)
    wDx: 200.0,
    wDy: 180.0,
    wVx: 90.0,
    wVyDown: 160.0,
    wAngleDeg: 80.0,
//    wFuel: 0.05,
//    wOutOfBounds: 600.0,
//    wOffPad: 250.0,
//    wTiltWrongWay: 45.0,
  );

  final env = eng.GameEngine(cfg);

  // ---- Feature Extractor (defaults; may be overwritten by init policy) ----
  int feGS = 3;
  double feStride = 48.0;
  final fe = FeatureExtractor(groundSamples: feGS, stridePx: feStride);

  // ---- Policy ----
  var policy = PolicyNetwork(inputSize: fe.inputSize, h1: 64, h2: 64, seed: 1234);

  // ---- Try to load an init policy ----
  Map<String, dynamic>? initJs;
  if (initPath != null && await File(initPath).exists()) {
    initJs = await _readJson(initPath!);
  } else {
    // Auto fallback: try policy_bc_init.json then policy.json if no --init provided
    if (await File('policy_bc_init.json').exists()) {
      initJs = await _readJson('policy_bc_init.json');
    } else if (await File('policy.json').exists()) {
      initJs = await _readJson('policy.json');
    }
  }

  if (initJs != null) {
    // Load into policy (may rebuild shapes)
    final loaded = _loadPolicyInto(policy, initJs);
    policy = loaded.$1;
    feGS = loaded.$2;
    feStride = loaded.$3;

    // Rebuild FE to match the loaded policy’s baked settings
    final fe2 = FeatureExtractor(groundSamples: feGS, stridePx: feStride);
    if (fe2.inputSize != policy.inputSize) {
      // If input size still mismatches, the JSON and your FeatureExtractor code disagree.
      stderr.writeln(
          'WARNING: loaded policy.inputSize=${policy.inputSize} but FE.inputSize=${fe2.inputSize}. '
              'Adjust FeatureExtractor or the policy JSON fe settings so they match.');
    }
    stdout.writeln(
        'Loaded init policy. h1=${policy.h1} h2=${policy.h2} | FE(gs=$feGS stride=$feStride)');
  } else {
    stdout.writeln('No init policy found; training from scratch.');
  }

  // ---- Trainer ----
  final rnd = math.Random(7);
  final trainer = Trainer(
    env: env,
    fe: FeatureExtractor(groundSamples: feGS, stridePx: feStride),
    policy: policy,
    dt: 1 / 60.0,
    gamma: 0.99,
    seed: 13,
  );

  // ---- Training loop ----
  double bestCost = double.infinity;
  int landedCount = 0;

  for (int it = 1; it <= iters; it++) {
    // IMPORTANT: reseed env each episode so spawn X randomizes even if terrain is locked
    env.reset(seed: rnd.nextInt(1 << 30));

    final ep = trainer.runEpisode(train: true, lr: 3e-4);

    if (ep.landed) landedCount++;
    final epCost = ep.totalCost;
    if (epCost < bestCost) {
      bestCost = epCost;
      await _savePolicy('policy_best_cost.json', policy, trainer.fe.inputSize, feGS, feStride);
    }

    if (it % 5 == 0) {
      final landRate = (landedCount / it * 100).toStringAsFixed(1);
      stdout.writeln(
          'Iter $it | last-ep steps: ${ep.steps} | cost: ${epCost.toStringAsFixed(3)} '
              '| landed: ${ep.landed ? 'Y' : 'N'} | best(cost): ${bestCost.toStringAsFixed(3)} | land%: $landRate');
    }

    if (it % saveEvery == 0) {
      await _savePolicy('policy.json', policy, trainer.fe.inputSize, feGS, feStride);
    }
  }

  // Final save
  await _savePolicy('policy.json', policy, trainer.fe.inputSize, feGS, feStride);
  stdout.writeln('Done. Best cost: ${bestCost.toStringAsFixed(3)}');
}
