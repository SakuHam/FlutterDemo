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

/* ------------------------------ Tiny IO helpers ------------------------------ */

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

  // heads (support both single-stage and two-stage)
  if (js.containsKey('W_thr')) policy.W_thr = _mat(js['W_thr']);
  if (js.containsKey('b_thr')) policy.b_thr = _vec(js['b_thr']);
  if (js.containsKey('W_turn')) policy.W_turn = _mat(js['W_turn']);
  if (js.containsKey('b_turn')) policy.b_turn = _vec(js['b_turn']);

  if (js.containsKey('W_intent')) policy.W_intent = _mat(js['W_intent']);
  if (js.containsKey('b_intent')) policy.b_intent = _vec(js['b_intent']);

  if (js.containsKey('W_val')) policy.W_val = _mat(js['W_val']);
  if (js.containsKey('b_val')) policy.b_val = _vec(js['b_val']);

  // FE metadata
  final fe = (js['fe'] as Map<String, dynamic>?) ?? const {};
  final gs = (fe['groundSamples'] as int?) ?? (inputSizeJ - 10);
  final stride = (fe['stridePx'] as num?)?.toDouble() ?? 48.0;

  // sanity: ensure at least one head exists
  final hasAnyHead = (js.containsKey('W_intent') && js.containsKey('b_intent')) ||
      (js.containsKey('W_thr') && js.containsKey('b_thr') && js.containsKey('W_turn') && js.containsKey('b_turn'));
  if (!hasAnyHead) {
    throw ArgumentError('Policy JSON has no heads (intent or thr/turn).');
  }

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
    // save all heads
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

double _median(List<double> xs) {
  if (xs.isEmpty) return 0.0;
  final a = List<double>.from(xs)..sort();
  final n = a.length;
  return n.isOdd ? a[n ~/ 2] : 0.5 * (a[n ~/ 2 - 1] + a[n ~/ 2]);
}

/* ----------------------------------- Main ----------------------------------- */

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
  int evalSeeds = 5;
  bool logEvalEverySave = false;

  // Intent pretrain flags
  int pretrainIntent = 0;     // number of snapshot samples (0 = disabled)
  int pretrainEpochs = 1;
  double pretrainAlignW = 1.0;
  double pretrainLr = 5e-4;
  bool onlyPretrain = false;

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

    if (a.startsWith('--pretrain_intent=')) pretrainIntent = int.parse(a.substring(18));
    if (a.startsWith('--pretrain_epochs=')) pretrainEpochs = int.parse(a.substring(18));
    if (a.startsWith('--pretrain_align=')) pretrainAlignW = double.parse(a.substring(17));
    if (a.startsWith('--pretrain_lr=')) pretrainLr = double.parse(a.substring(14));
    if (a == '--only_pretrain=true') onlyPretrain = true;
  }

  // --------------- Engine configs ---------------
  final uiLinearScale = 3.0;   // matches UI: 0.05 * 60
  final uiRotScale    = 0.5;   // UI halves rotation rate

  // Locked evaluation config (phone-like deterministic target)
  final cfg = et.EngineConfig(
    t: et.Tunables(
      gravity:     0.18 * uiLinearScale,  // 0.54
      thrustAccel: 0.42 * uiLinearScale,  // 1.26
      rotSpeed:    1.6  * uiRotScale,     // 0.8
      maxFuel:     100.0,
    ),
    worldW: 360,
    worldH: 640,
    hardWalls: true,
    lockTerrain: true,
    terrainSeed: 12345,
    lockSpawn: true,
    randomSpawnX: false,
    spawnXMin: 0.20,
    spawnXMax: 0.80,
    landingSpeedMax: 12.0,
    landingAngleMaxRad: 8 * math.pi / 180.0,
    wDx: 200.0,
    wDy: 180.0,
    wVx: 90.0,
    wVyDown: 200.0,
    wAngleDeg: 80.0,
  );

  // Training config (adds diversity to make intent learn pad direction)
  final cfgTrain = et.EngineConfig(
    t: cfg.t,
    worldW: cfg.worldW,
    worldH: cfg.worldH,
    hardWalls: cfg.hardWalls,
    lockTerrain: true,   // you can set false later for more diversity
    terrainSeed: 12345,
    lockSpawn: false,    // ← unlock spawn
    randomSpawnX: true,  // ← randomize X
    spawnXMin: 0.15,
    spawnXMax: 0.85,
    landingSpeedMax: cfg.landingSpeedMax,
    landingAngleMaxRad: cfg.landingAngleMaxRad,

    // Slight curriculum: emphasize lateral correction first
    wDx: 260.0,
    wDy: 160.0,
    wVx: 110.0,
    wVyDown: 180.0,
    wAngleDeg: cfg.wAngleDeg,
  );

  // --------------- Envs ---------------
  final env = eng.GameEngine(cfgTrain);         // training env (diverse)
  final evalEnvSingle = eng.GameEngine(cfg);    // single eval env for probe

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
      'Loaded init policy. h1=${policy.h1} h2=${policy.h2} | FE(gs=$feGS stride=$feStride)',
    );
  } else {
    stdout.writeln('No init policy found; training from scratch.');
  }

  // -------- Strict FE–policy size validation --------
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
    // two-stage — more agile planner
    twoStage: true,
    planHold: 6,             // was 12
    tempIntent: 1.1,         // was 0.8
    intentEntropyBeta: 0.03, // was 0.01
  );

//  trainer.debugMicroOverfitIntent(
//      perClass: 10, steps: 300, lr: 0.01, alignWeight: 5.0);
//  return; // early exit so you only run the probe

  // Greedy evaluation trainer on a locked env
  final evalTrainer = Trainer(
    env: evalEnvSingle,
    fe: FeatureExtractor(groundSamples: feGS, stridePx: feStride),
    policy: policy,
    dt: 1 / 60.0,
    gamma: 0.99,
    seed: 13,
    tempThr: 1e-6,
    tempTurn: 1e-6,
    epsilon: 0.0,
    entropyBeta: 0.0,
    twoStage: true,
    planHold: 6,
    tempIntent: 1e-6,
    intentEntropyBeta: 0.0,
  );

  // === Pretrain intent (optional) ===
  if (pretrainIntent > 0) {
    stdout.writeln(
        'Pretraining intent on $pretrainIntent snapshots (epochs=$pretrainEpochs, '
            'align=$pretrainAlignW, lr=$pretrainLr) ...');
    final stats = trainer.pretrainIntentOnSnapshots(
      samples: pretrainIntent,
      epochs: pretrainEpochs,
      lr: pretrainLr,
      alignWeight: pretrainAlignW,
    );
    stdout.writeln(
      'Pretrain done → acc=${(stats["acc"]! * 100).toStringAsFixed(1)}% '
          'over n=${stats["n"]!.toInt()} samples',
    );

    await _savePolicy(
        'policy_pretrained.json', policy, trainer.fe.inputSize, feGS, feStride);

    if (onlyPretrain) {
      stdout.writeln('Only-pretrain mode: saved → policy_pretrained.json. Exiting.');
      return;
    }
  }

  // -------- Quick determinism probe (locked env, greedy) --------
      {
    final s = fixedSeed;
    evalEnvSingle.reset(seed: s);
    final ep1 = evalTrainer.runEpisode(train: false, lr: 0.0, greedy: true);
    evalEnvSingle.reset(seed: s);
    final ep2 = evalTrainer.runEpisode(train: false, lr: 0.0, greedy: true);
    final same =
        (ep1.steps == ep2.steps) && ((ep1.totalCost - ep2.totalCost).abs() < 1e-9);
    stdout.writeln(
      'Determinism probe: steps ${ep1.steps} vs ${ep2.steps} | '
          'cost ${ep1.totalCost.toStringAsFixed(6)} vs ${ep2.totalCost.toStringAsFixed(6)} '
          '=> ${same ? "OK" : "NONDETERMINISTIC!"}',
    );
  }

  int nextEpisodeSeed() => overfit ? fixedSeed : rnd.nextInt(1 << 30);

  /// Realistic evaluation across randomized scenarios (stable seed set).
  Future<({
  double meanCost,
  double medianCost,
  double landPct,
  double crashPct,
  double meanSteps,
  double meanDxAbs
  })> evalRealistic({int episodes = 32}) async {
    final seeds = List<int>.generate(episodes, (i) => 1_000_000 + i);

    final costs = <double>[];
    final steps = <int>[];
    int landed = 0;
    int crashed = 0;
    double sumDxAbs = 0.0;

    for (final s in seeds) {
      final evalEnv = eng.GameEngine(cfg);
      final etr = Trainer(
        env: evalEnv,
        fe: FeatureExtractor(groundSamples: feGS, stridePx: feStride),
        policy: policy,
        dt: 1 / 60.0,
        gamma: 0.99,
        seed: 13,
        tempThr: 1e-6,
        tempTurn: 1e-6,
        epsilon: 0.0,
        entropyBeta: 0.0,
        twoStage: true,
        planHold: 6,
        tempIntent: 1e-6,
        intentEntropyBeta: 0.0,
      );

      evalEnv.reset(seed: s);

      final ep = etr.runEpisode(train: false, lr: 0.0, greedy: true);

      final padCx = evalEnv.terrain.padCenter;
      final dxAbs = (evalEnv.lander.pos.x - padCx).abs();
      sumDxAbs += dxAbs;

      costs.add(ep.totalCost);
      steps.add(ep.steps);

      if (ep.landed) landed++;
      if (!ep.landed && evalEnv.status == et.GameStatus.crashed) crashed++;
    }

    double mean(List<double> a) =>
        a.isEmpty ? 0.0 : a.reduce((x, y) => x + y) / a.length;

    return (
    meanCost: mean(costs),
    medianCost: _median(costs),
    landPct: seeds.isEmpty ? 0.0 : (100.0 * landed / seeds.length),
    crashPct: seeds.isEmpty ? 0.0 : (100.0 * crashed / seeds.length),
    meanSteps: steps.isEmpty ? 0.0 : steps.reduce((a, b) => a + b) / steps.length,
    meanDxAbs: seeds.isEmpty ? 0.0 : (sumDxAbs / seeds.length),
    );
  }

  // Best realistic evaluation so far
  var bestStats = (
  meanCost: double.infinity,
  medianCost: double.infinity,
  landPct: 0.0,
  crashPct: 100.0,
  meanSteps: double.infinity,
  meanDxAbs: double.infinity
  );

  // --------------- Training loop (batched) ---------------
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
        'Iter $it | batch=$batch | last-ep steps: $lastSteps | '
            'cost: ${lastCost.toStringAsFixed(3)} | '
            'landed: ${lastLanded ? 'Y' : 'N'} | '
            'turn%: ${turnPct.toStringAsFixed(1)} | '
            'thrust%: ${thrustPct.toStringAsFixed(1)} | '
            'land%($landWindowSize)=${landRate.toStringAsFixed(1)}',
      );

      // Realistic evaluation (episodes scaled from evalSeeds)
      final ev = await evalRealistic(episodes: math.max(16, evalSeeds * 6));
      stdout.writeln(
        'Eval(real) → '
            'meanCost=${ev.meanCost.toStringAsFixed(3)} | '
            'median=${ev.medianCost.toStringAsFixed(3)} | '
            'land%=${ev.landPct.toStringAsFixed(1)} | '
            'crash%=${ev.crashPct.toStringAsFixed(1)} | '
            'steps=${ev.meanSteps.toStringAsFixed(1)} | '
            'mean|dx|=${ev.meanDxAbs.toStringAsFixed(1)}',
      );

      // Selection rule with tol
      const tol = 1e-6;
      bool better = false;
      if (ev.meanCost + tol < bestStats.meanCost) {
        better = true;
      } else if ((ev.meanCost - bestStats.meanCost).abs() <= tol) {
        if (ev.landPct > bestStats.landPct + tol) {
          better = true;
        } else if ((ev.landPct - bestStats.landPct).abs() <= tol) {
          if (ev.medianCost + tol < bestStats.medianCost) {
            better = true;
          } else if ((ev.medianCost - bestStats.medianCost).abs() <= tol) {
            if (ev.meanDxAbs + tol < bestStats.meanDxAbs) better = true;
          }
        }
      }

      if (better) {
        bestStats = ev;
        await _savePolicy(
            'policy_best_eval.json', policy, trainer.fe.inputSize, feGS, feStride);
        stdout.writeln('↑ New best (realistic) — saved policy_best_eval.json');
      }
    }

    if (it % saveEvery == 0) {
      await _savePolicy('policy.json', policy, trainer.fe.inputSize, feGS, feStride);
      if (logEvalEverySave) {
        final ev = await evalRealistic(episodes: math.max(16, evalSeeds * 6));
        stdout.writeln(
          '[On Save] Eval(real) → '
              'meanCost=${ev.meanCost.toStringAsFixed(3)} | '
              'median=${ev.medianCost.toStringAsFixed(3)} | '
              'land%=${ev.landPct.toStringAsFixed(1)} | '
              'crash%=${ev.crashPct.toStringAsFixed(1)} | '
              'steps=${ev.meanSteps.toStringAsFixed(1)} | '
              'mean|dx|=${ev.meanDxAbs.toStringAsFixed(1)}',
        );
      }
    }
  }

  // Final save + realistic eval
  await _savePolicy('policy.json', policy, trainer.fe.inputSize, feGS, feStride);
  final ev = await evalRealistic(episodes: math.max(32, evalSeeds * 6));
  stdout.writeln(
    'Done. Best(real) meanCost=${bestStats.meanCost.toStringAsFixed(3)} '
        'land%=${bestStats.landPct.toStringAsFixed(1)} | '
        'FinalEval(real) meanCost=${ev.meanCost.toStringAsFixed(3)} '
        'median=${ev.medianCost.toStringAsFixed(3)} '
        'land%=${ev.landPct.toStringAsFixed(1)} crash%=${ev.crashPct.toStringAsFixed(1)} '
        'steps=${ev.meanSteps.toStringAsFixed(1)} mean|dx|=${ev.meanDxAbs.toStringAsFixed(1)}',
  );
}
