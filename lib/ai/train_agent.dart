// lib/ai/train_agent.dart
import 'dart:io';
import '../engine/game_engine.dart' as eng;
import 'agent.dart';

void main(List<String> args) async {
  final t = eng.Tunables(
    gravity: 0.18,
    thrustAccel: 0.42,
    rotSpeed: 1.6,
    maxFuel: 100.0,
  );

  // Start with a lenient config; Trainer will ramp down to normal.
  final cfg = eng.EngineConfig(
    worldW: 360,
    worldH: 640,
    t: t,
    seed: 42,
    stepScale: 60.0,
    padWidthFactor: 1.6,      // wider pad in early curriculum
    landingSpeedMax: 80.0,    // gentler requirement initially
    landingAngleMaxRad: 0.45, // gentler requirement initially
    livingCost: 0.002,
    effortCost: 0.001,
    wDx: 0.0025,
    wVyDown: 0.005,
    wVx: 0.0015,
    wAngleDeg: 0.03,
  );

  final engine = eng.GameEngine(cfg);
  final fe = FeatureExtractor(groundSamples: 5, stridePx: 40);
  final policy = PolicyNetwork(inputSize: fe.inputSize, h1: 128, h2: 128, outputs: 3, seed: 123);

  final trainer = Trainer(
    engine: engine,
    fe: fe,
    policy: policy,
    dt: 1/60.0,
    gamma: 0.995,
    seed: 99,
  );

  const episodes = 1000;
  double bestCost = double.infinity;
  int landedCount = 0;

  for (int ep = 1; ep <= episodes; ep++) {
    final r = trainer.runEpisode(train: true, lr: 3e-4);

    if (r.landed) landedCount++;
    if (r.totalCost < bestCost) bestCost = r.totalCost;

    if (ep % 10 == 0) {
      final rate = (landedCount / ep * 100).toStringAsFixed(1);
      stdout.writeln(
          'Ep $ep | steps: ${r.steps} | cost: ${r.totalCost.toStringAsFixed(1)} '
              '| landed: ${r.landed ? "Y" : "N"} | best(cost): ${bestCost.toStringAsFixed(1)} | land%: $rate');
    }
  }

  stdout.writeln('--- Evaluation ---');
  const evalRuns = 10;
  int evalLanded = 0;
  double evalCostSum = 0;
  for (int i = 0; i < evalRuns; i++) {
    final r = trainer.runEpisode(train: false);
    evalCostSum += r.totalCost;
    if (r.landed) evalLanded++;
  }
  stdout.writeln('Avg cost: ${(evalCostSum / evalRuns).toStringAsFixed(1)}, landed: $evalLanded/$evalRuns');
}
