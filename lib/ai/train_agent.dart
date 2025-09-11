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

// Use eng for GameEngine (physics stepper)
import '../engine/game_engine.dart' as eng;
// Use et for types/configs (Tunables, EngineConfig, etc.)
import '../engine/types.dart' as et;

/// ---------- Tiny IO helpers ----------
Future<Map<String, dynamic>> _readJson(String path) async =>
    json.decode(await File(path).readAsString()) as Map<String, dynamic>;

Future<void> _writePrettyJson(String path, Map<String, dynamic> js) async {
  final out = const JsonEncoder.withIndent('  ').convert(js);
  await File(path).writeAsString(out);
  stdout.writeln('Saved policy → $path');
}

List<List<double>> _mat(dynamic m) => (m as List)
    .map<List<double>>(
      (r) => (r as List).map<double>((x) => (x as num).toDouble()).toList(),
)
    .toList();

List<double> _vec(dynamic v) =>
    (v as List).map<double>((x) => (x as num).toDouble()).toList();

/// Load a policy JSON into (possibly re-instantiated) PolicyNetwork.
/// Returns (policy, groundSamples, stridePx).
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

  // trunk
  policy
    ..W1 = _mat(js['W1'])
    ..b1 = _vec(js['b1'])
    ..W2 = _mat(js['W2'])
    ..b2 = _vec(js['b2']);

  // ---- Heads: load what exists (support legacy & two-stage) ----
  final hasThrTurn = js.containsKey('W_thr') &&
      js.containsKey('b_thr') &&
      js.containsKey('W_turn') &&
      js.containsKey('b_turn');

  if (hasThrTurn) {
    policy
      ..W_thr = _mat(js['W_thr'])
      ..b_thr = _vec(js['b_thr'])
      ..W_turn = _mat(js['W_turn'])
      ..b_turn = _vec(js['b_turn']);
  }

  final hasIntent = js.containsKey('W_intent') && js.containsKey('b_intent');
  if (hasIntent) {
    policy
      ..W_intent = _mat(js['W_intent'])
      ..b_intent = _vec(js['b_intent']);
  }

  final hasValue = js.containsKey('W_val') && js.containsKey('b_val');
  if (hasValue) {
    policy
      ..W_val = _mat(js['W_val'])
      ..b_val = _vec(js['b_val']);
  }

  // If neither legacy nor intent heads exist, bail with a clear message.
  if (!hasThrTurn && !hasIntent) {
    throw ArgumentError(
      'Policy JSON has no action heads: expected either '
          'single-stage (W_thr/W_turn) or two-stage (W_intent).',
    );
  }

  final fe = (js['fe'] as Map<String, dynamic>?) ?? const {};
  // Training FE = 10 + groundSamples (px,py,vx,vy,ang,fuel,padCenter,dxCenter,dGround,slope)
  final gs = (fe['groundSamples'] as int?) ?? (inputSizeJ - 10);
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
  final js = <String, dynamic>{
    'inputSize': inputSize,
    'h1': p.h1,
    'h2': p.h2,
    'W1': p.W1,
    'b1': p.b1,
    'W2': p.W2,
    'b2': p.b2,
    // Save both legacy and two-stage heads if present (they always exist in PolicyNetwork)
    'W_thr': p.W_thr,
    'b_thr': p.b_thr,
    'W_turn': p.W_turn,
    'b_turn': p.b_turn,
    'W_intent': p.W_intent,
    'b_intent': p.b_intent,
    'W_val': p.W_val,
    'b_val': p.b_val,
    'fe': {'groundSamples': gs, 'stridePx': stride},
  };
  await _writePrettyJson(path, js);
}

void main(List<String> args) async {
  // ---------------- CLI ----------------
  String? initPath;
  int iters = 20000;
  int saveEvery = 50;
  int batch = 16; // episodes per outer iteration
  double baseLr = 3e-4; // per-episode LR (before /batch below)

  double tempThr = 0.9; // exploration (train-time, single-stage only)
  double tempTurn = 1.2;
  double epsilon = 0.05;
  double entropyBeta = 0.003;

  // Deterministic overfit/eval controls
  bool overfit = false; // train & eval on a single fixed seed
  bool noExplore = false; // zero exploration for training (optional)
  int fixedSeed = 424242;
  int evalSeeds = 5; // deterministic eval across N seeds
  bool logEvalEverySave = false;

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

    if (a == '--overfit=true') overfit = true;
    if (a == '--no_explore=true') noExplore = true;
    if (a.startsWith('--fixed_seed=')) fixedSeed = int.parse(a.substring(13));
    if (a.startsWith('--eval_seeds=')) evalSeeds = int.parse(a.substring(13));
    if (a == '--log_eval_on_save=true') logEvalEverySave = true;
  }

  // --------------- Engine config (Flutter-free) ---------------
  final cfg = et.EngineConfig(
    t: et.Tunables(
      gravity: 0.18,
      thrustAccel: 0.42,
      rotSpeed: 1.6,
      maxFuel: 100.0,
    ),
    worldW: 360,
    worldH: 640,
    lockTerrain: true,
    terrainSeed: 12345,
    lockSpawn: true,
    randomSpawnX: false,
    spawnXMin: 0.20,
    spawnXMax: 0.80,
    landingSpeedMax: 12.0, // stricter
    landingAngleMaxRad: 8 * math.pi / 180.0,
    wDx: 200.0,
    wDy: 180.0,
    wVx: 90.0,
    wVyDown: 200.0, // higher with stricter landings
    wAngleDeg: 80.0,
  );

  final env = eng.GameEngine(cfg);

  // -------- Feature Extractor defaults --------
  int feGS = 3;
  double feStride = 48.0;

  // --------------- Policy ---------------
  var policy = PolicyNetwork(
    inputSize: FeatureExtractor(groundSamples: feGS, stridePx: feStride).inputSize,
    h1: 64,
    h2: 64,
    seed: 1234,
  );

  // Bias throttle slightly off at the start (legacy head)
  policy.b_thr[0] = -0.2;

  // --------------- Load init (BC or previous) ---------------
  Map<String, dynamic>? initJs;
  if (initPath != null && await File(initPath!).exists()) {
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
    stdout.writeln(
        'Loaded init policy. h1=${policy.h1} h2=${policy.h2} | FE(gs=$feGS stride=$feStride)');
  } else {
    stdout.writeln('No init policy found; training from scratch.');
  }

  // -------- Strict FE–policy size validation (NO auto alignment) --------
  final feCheck = FeatureExtractor(groundSamples: feGS, stridePx: feStride);
  if (feCheck.inputSize != policy.inputSize) {
    stderr.writeln(
      'ERROR: FeatureExtractor.inputSize=${feCheck.inputSize} '
          '!= policy.inputSize=${policy.inputSize}. '
          'Fix your JSON/FE config to match exactly.',
    );
    exit(1);
  }

  // --------------- Trainers ---------------
  final rnd = math.Random(7);

  // Train-time exploration (optionally disabled)
  final double trainEps = noExplore ? 0.0 : epsilon;
  final double trainTempThr = noExplore ? 0.2 : tempThr; // (legacy head)
  final double trainTempTurn = noExplore ? 0.7 : tempTurn;
  final double trainEntropy = noExplore ? 0.01 : entropyBeta;

  final trainer = Trainer(
    env: env,
    fe: FeatureExtractor(groundSamples: feGS, stridePx: feStride),
    policy: policy,
    dt: 1 / 60.0,
    gamma: 0.99,
    seed: 13,
    // single-stage exploration (unused when twoStage=true)
    tempThr: trainTempThr,
    tempTurn: trainTempTurn,
    epsilon: trainEps,
    entropyBeta: trainEntropy,
    // two-stage on:
    twoStage: true,
    planHold: 12, // re-plan intent every 12 frames (~0.2s @ 60Hz)
    tempIntent: 0.8, // softer sampling for planner
    intentEntropyBeta: 0.01, // small entropy on intents
  );

  // Deterministic evaluation trainer (no exploration)
  final evalTrainer = Trainer(
    env: env,
    fe: FeatureExtractor(groundSamples: feGS, stridePx: feStride),
    policy: policy,
    dt: 1 / 60.0,
    gamma: 0.99,
    seed: 13,
    tempThr: 1e-6,
    tempTurn: 1e-6,
    epsilon: 0.0,
    entropyBeta: 0.0,
    twoStage: true, // keep the same architecture
    planHold: 12,
    tempIntent: 1e-6, // greedy planner
    intentEntropyBeta: 0.0,
  );

  {
    final s = fixedSeed;
    env.reset(seed: s);
    final ep1 = evalTrainer.runEpisode(train: false, lr: 0.0, greedy: true);
    env.reset(seed: s);
    final ep2 = evalTrainer.runEpisode(train: false, lr: 0.0, greedy: true);
    final same =
        (ep1.steps == ep2.steps) && ((ep1.totalCost - ep2.totalCost).abs() < 1e-9);
    stdout.writeln(
        'Determinism probe: steps ${ep1.steps} vs ${ep2.steps} | '
            'cost ${ep1.totalCost.toStringAsFixed(6)} vs ${ep2.totalCost.toStringAsFixed(6)} '
            '=> ${same ? "OK" : "NONDETERMINISTIC!"}');
  }

  int nextEpisodeSeed() {
    if (overfit) return fixedSeed;
    return rnd.nextInt(1 << 30);
  }

  // --------------- Deterministic evaluation helper ---------------
  Future<
      ({
      double avgCost,
      int avgSteps,
      double landPct,
      double turnPct,
      double thrustPct
      })> evalDeterministic({int nSeeds = 5}) async {
    final seeds = List.generate(
      nSeeds,
          (i) => overfit ? fixedSeed : (1000 + i), // fixed small set
    );

    double sumCost = 0.0;
    int sumSteps = 0;
    int lands = 0;
    int sumTurnSteps = 0;
    int sumThrustSteps = 0;

    for (final s in seeds) {
      env.reset(seed: s);
      final ep = evalTrainer.runEpisode(train: false, lr: 0.0, greedy: true);
      sumCost += ep.totalCost;
      sumSteps += ep.steps;
      sumTurnSteps += ep.turnSteps;
      sumThrustSteps += ep.thrustSteps;
      if (ep.landed) lands++;
    }

    final avgCost = sumCost / nSeeds;
    final avgSteps = (sumSteps / nSeeds).round();
    final landPct = 100.0 * lands / nSeeds;
    final turnPct = sumSteps == 0 ? 0.0 : 100.0 * (sumTurnSteps / sumSteps);
    final thrustPct = sumSteps == 0 ? 0.0 : 100.0 * (sumThrustSteps / sumSteps);
    return (
    avgCost: avgCost,
    avgSteps: avgSteps,
    landPct: landPct,
    turnPct: turnPct,
    thrustPct: thrustPct
    );
  }

  // --------------- Training loop (batched) ---------------
  double bestEvalCost = double.infinity;
  double bestEvalLandPct = 0.0;

  int landedTotal = 0;
  final landWindow = <bool>[];
  const landWindowSize = 200;

  for (int it = 1; it <= iters; it++) {
    // Per-episode LR: split across episodes in batch
    final epLr = baseLr / batch;

    double lastCost = 0.0;
    int lastSteps = 0;
    bool lastLanded = false;
    int lastTurnSteps = 0;
    int lastThrustSteps = 0;

    for (int k = 0; k < batch; k++) {
      env.reset(seed: nextEpisodeSeed());
      final ep = trainer.runEpisode(
        train: true,
        lr: epLr,
        greedy: false, // IMPORTANT: always sample during training
      );

      lastCost = ep.totalCost;
      lastSteps = ep.steps;
      lastLanded = ep.landed;
      lastTurnSteps = ep.turnSteps;
      lastThrustSteps = ep.thrustSteps;

      if (ep.landed) landedTotal++;
      landWindow.add(ep.landed);
      if (landWindow.length > landWindowSize) landWindow.removeAt(0);
    }

    if (it % 5 == 0) {
      final landRate = landWindow.isEmpty
          ? 0.0
          : (landWindow.where((x) => x).length / landWindow.length * 100.0);

      final turnPct = lastSteps == 0 ? 0.0 : 100.0 * (lastTurnSteps / lastSteps);
      final thrustPct = lastSteps == 0 ? 0.0 : 100.0 * (lastThrustSteps / lastSteps);

      stdout.writeln(
        'Iter $it | batch=$batch | last-ep steps: $lastSteps | cost: ${lastCost.toStringAsFixed(3)} '
            '| landed: ${lastLanded ? 'Y' : 'N'} | turn%: ${turnPct.toStringAsFixed(1)} '
            '| thrust%: ${thrustPct.toStringAsFixed(1)} | land%($landWindowSize)=${landRate.toStringAsFixed(1)}',
      );

      // Deterministic eval for logging and model selection
      final ev = await evalDeterministic(nSeeds: evalSeeds);
      stdout.writeln(
        'Eval → avgCost=${ev.avgCost.toStringAsFixed(3)} | steps=${ev.avgSteps} '
            '| land%=${ev.landPct.toStringAsFixed(1)} | turn%=${ev.turnPct.toStringAsFixed(1)} '
            '| thrust%=${ev.thrustPct.toStringAsFixed(1)}',
      );

      // Model selection: prefer lower cost, break ties by land%
      bool isBest = false;
      if (ev.avgCost < bestEvalCost - 1e-9) {
        isBest = true;
      } else if ((ev.avgCost - bestEvalCost).abs() <= 1e-9 &&
          ev.landPct > bestEvalLandPct) {
        isBest = true;
      }

      if (isBest) {
        bestEvalCost = ev.avgCost;
        bestEvalLandPct = ev.landPct;
        await _savePolicy(
            'policy_best_eval.json', policy, trainer.fe.inputSize, feGS, feStride);
      }
    }

    if (it % saveEvery == 0) {
      await _savePolicy('policy.json', policy, trainer.fe.inputSize, feGS, feStride);
      if (logEvalEverySave) {
        final ev = await evalDeterministic(nSeeds: evalSeeds);
        stdout.writeln(
          '[On Save] Eval → avgCost=${ev.avgCost.toStringAsFixed(3)} | steps=${ev.avgSteps} '
              '| land%=${ev.landPct.toStringAsFixed(1)} | turn%=${ev.thrustPct.toStringAsFixed(1)}',
        );
      }
    }
  }

  // Final save + deterministic eval
  await _savePolicy('policy.json', policy, trainer.fe.inputSize, feGS, feStride);
  final ev = await evalDeterministic(nSeeds: evalSeeds);
  stdout.writeln(
    'Done. BestEvalCost=${bestEvalCost.toStringAsFixed(3)} bestEvalLand%=${bestEvalLandPct.toStringAsFixed(1)} '
        '| FinalEval avgCost=${ev.avgCost.toStringAsFixed(3)} steps=${ev.avgSteps} '
        'land%=${ev.landPct.toStringAsFixed(1)} turn%=${ev.turnPct.toStringAsFixed(1)} '
        'thrust%=${ev.thrustPct.toStringAsFixed(1)}',
  );
}
