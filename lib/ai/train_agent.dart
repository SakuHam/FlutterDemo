// lib/ai/train_agent.dart
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import '../engine/game_engine.dart' as eng;
import 'agent.dart';

void main() async {
  final cfg = eng.EngineConfig(
    worldW: 360,
    worldH: 640,
    t: eng.Tunables(
      gravity: 0.18,
      thrustAccel: 0.45, // small boost to help braking
      rotSpeed: 1.9,     // a bit easier to control at first
      maxFuel: 140.0,
    ),
    stepScale: 60.0,
    // Easier landing to start (tighten later)
    padWidthFactor: 1.5,
    landingSpeedMax: 95.0,
    landingAngleMaxRad: 0.55,
    // Border shaping (not used with hardWalls=true, but kept for later)
    borderMargin: 0.0,
    borderPenaltyPerSec: 0.0,
    wrapPenalty: 0.0,
    // ===== Overfit settings =====
    lockTerrain: true,
    terrainSeed: 123, // fixed terrain
    lockSpawn: true,
    spawnX: 0.25,
    spawnY: 120.0,
    spawnVx: 0.0,
    spawnVy: 0.0,
    spawnAngle: 0.0,
    hardWalls: true,   // no wrap while overfitting
    // Angle shaping stronger near ground (already set in defaults)
  );

  final env = eng.GameEngine(cfg);
  final fe = FeatureExtractor(groundSamples: 3, stridePx: 48);
  final policy = PolicyNetwork(inputSize: fe.inputSize, h1: 64, h2: 64, seed: 1234);
  final trainer = Trainer(env: env, fe: fe, policy: policy, dt: 1/60.0, gamma: 0.995, seed: 7);

  final rnd = math.Random(7);
  int landedCount = 0;
  double bestCost = double.infinity;

  const iters = 300;
  for (int i = 1; i <= iters; i++) {
    // Optional tiny anneal toward target physics later
    if (i == 150) {
      env.cfg = env.cfg.copyWith(
        t: env.cfg.t.copyWith(rotSpeed: 1.6, maxFuel: 120.0),
        padWidthFactor: 1.2,
        landingSpeedMax: 85.0,
        landingAngleMaxRad: 0.50,
      );
    }

    final ep = trainer.runEpisode(train: true, lr: 3e-4);
    if (ep.landed) landedCount++;
    if (ep.totalCost < bestCost) bestCost = ep.totalCost;

    if (i % 5 == 0) {
      final landRate = (landedCount / i * 100.0);
      stdout.writeln(
          'Iter $i | steps: ${ep.steps} | cost: ${ep.totalCost.toStringAsFixed(3)} | '
              'landed: ${ep.landed ? 'Y' : 'N'} | best(cost): ${bestCost.toStringAsFixed(3)} | '
              'land%: ${landRate.toStringAsFixed(1)}'
      );
    }
  }

  // ----- Save weights to JSON (runtime_policy.dart loader compatible) -----
  final js = {
    'inputSize': policy.inputSize,
    'h1': policy.h1,
    'h2': policy.h2,
    'W1': policy.W1,
    'b1': policy.b1,
    'W2': policy.W2,
    'b2': policy.b2,
    'W_thr': policy.W_thr,
    'b_thr': policy.b_thr,
    'W_turn': policy.W_turn,
    'b_turn': policy.b_turn,
    'fe': {
      'groundSamples': fe.groundSamples,
      'stridePx': fe.stridePx,
    }
  };
  final out = const JsonEncoder.withIndent('  ').convert(js);
  final file = File('policy.json');
  await file.writeAsString(out);
  stdout.writeln('Saved policy to ${file.path} (copy to assets/ai/policy.json for runtime).');
}
