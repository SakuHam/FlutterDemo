// lib/ai/train_agent.dart
import 'dart:io';
import 'dart:convert';

import '../engine/game_engine.dart' as eng;
import 'agent.dart';

/// Save the current policy weights to a JSON file compatible with runtime_policy.dart
Future<void> savePolicyJson(PolicyNetwork p, String outPath) async {
  final map = <String, dynamic>{
    'W1': p.W1,
    'b1': p.b1,
    'W2': p.W2,
    'b2': p.b2,
    'W_thr': p.W_thr,
    'b_thr': p.b_thr,
    'W_turn': p.W_turn,
    'b_turn': p.b_turn,
    'W_val': p.W_val,
    'b_val': p.b_val,
  };
  final jsonStr = const JsonEncoder.withIndent('  ').convert(map);

  final file = File(outPath);
  await file.parent.create(recursive: true);
  await file.writeAsString(jsonStr);
  stdout.writeln('Saved policy to: $outPath');
}

void main(List<String> args) async {
  // Output path: default build/policy.json, or pass a custom path as arg[0]
  final String outPath = args.isNotEmpty ? args[0] : 'build/policy.json';

  // --- Engine config (same as before; tweak if you changed it) ---
  final t = eng.Tunables(
    gravity: 0.18,
    thrustAccel: 0.42,
    rotSpeed: 1.6,
    maxFuel: 100.0,
  );

  final cfg = eng.EngineConfig(
    worldW: 360,
    worldH: 640,
    t: t,
    seed: 42,
    stepScale: 60.0,
    // start a bit lenient; PPOTrainer tightens via its simple curriculum
    padWidthFactor: 1.6,
    landingSpeedMax: 80.0,
    landingAngleMaxRad: 0.45,
    livingCost: 0.002,
    effortCost: 0.0001,
    wDx: 0.0025,
    wDy: 0.0020,
    wVyDown: 0.025,
    wVx: 0.0015,
    wAngleDeg: 0.025,
  );

  final engine = eng.GameEngine(cfg);

  // Keep these aligned with your runtime policy (runtime_policy.dart)
  final fe = FeatureExtractor(groundSamples: 3, stridePx: 48);
  final policy = PolicyNetwork(inputSize: fe.inputSize, h1: 64, h2: 64, seed: 123);

  final trainer = PPOTrainer(
    engine: engine,
    fe: fe,
    policy: policy,
    dt: 1 / 60.0,
    gamma: 0.995,
    clipEps: 0.2,
    entropyBeta: 0.0005, // small, cheap entropy (pairs with cheap proxy in agent.dart)
    valueCoef: 0.5,
    l2: 1e-6,
    lr: 1e-4,
    rolloutSteps: 2048,
    ppoEpochs: 2,
    miniBatch: 256,
    maxStepsPerEp: 900,
    seed: 99,
  );

  const totalIters = 200; // each iter = one PPO update over rolloutSteps
  double bestCost = double.infinity;
  int landedCount = 0;
  int episodesSeen = 0;

  for (int it = 1; it <= totalIters; it++) {
    final r = trainer.trainStep(episodesSeen, logOneEp: it % 5 == 0);
    episodesSeen++; // approx—each rollout ends with one episode summary

    if (r.landed) landedCount++;
    if (r.totalCost < bestCost) bestCost = r.totalCost;

    if (it % 5 == 0) {
      final landRate = (landedCount / episodesSeen * 100).toStringAsFixed(1);
      stdout.writeln(
        'Iter $it | last-ep steps: ${r.steps} | cost: ${r.totalCost.toStringAsFixed(3)} '
            '| landed: ${r.landed ? "Y" : "N"} | best(cost): ${bestCost.toStringAsFixed(3)} | land%: $landRate',
      );
    }

    // Save a checkpoint every 10 iterations
    if (it % 10 == 0) {
      await savePolicyJson(policy, outPath);
    }
  }

  // Final save at the end
  await savePolicyJson(policy, outPath);

  // Quick evaluation (optional)
  stdout.writeln('--- Evaluation (10 rollouts) ---');
  int evalLanded = 0;
  double evalCostSum = 0;
  const evalRuns = 10;
  for (int i = 0; i < evalRuns; i++) {
    final r = trainer.trainStep(episodesSeen + i, logOneEp: false);
    evalCostSum += r.totalCost;
    if (r.landed) evalLanded++;
  }
  stdout.writeln('Avg cost: ${(evalCostSum / evalRuns).toStringAsFixed(3)}, landed: $evalLanded/$evalRuns');
  stdout.writeln('Final policy saved to: $outPath');
}
