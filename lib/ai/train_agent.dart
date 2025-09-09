// lib/ai/train_agent.dart
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import '../engine/game_engine.dart' as eng;
import 'agent.dart';

Future<void> savePolicy(String path, PolicyNetwork policy, FeatureExtractor fe) async {
  final js = {
    'inputSize': policy.inputSize,
    'h1': policy.h1,
    'h2': policy.h2,
    'W1': policy.W1, 'b1': policy.b1,
    'W2': policy.W2, 'b2': policy.b2,
    'W_thr': policy.W_thr, 'b_thr': policy.b_thr,
    'W_turn': policy.W_turn, 'b_turn': policy.b_turn,
    'fe': {'groundSamples': fe.groundSamples, 'stridePx': fe.stridePx},
  };
  final out = const JsonEncoder.withIndent('  ').convert(js);
  await File(path).writeAsString(out);
  stdout.writeln('Saved policy → $path');
}

void main() async {
  // ===== Overfit-friendly config (tighten later) =====
  final cfg = eng.EngineConfig(
    worldW: 360,
    worldH: 640,
    t: eng.Tunables(
      gravity: 0.16,
      thrustAccel: 0.60,
      rotSpeed: 2.4,
      maxFuel: 160.0,
    ),
    stepScale: 60.0,

    padWidthFactor: 2.0,
    landingSpeedMax: 110.0,
    landingAngleMaxRad: 0.60,

    livingCost: 0.0,
    effortCost: 0.0,
    wDx: 0.15,
    wDy: 0.0015,
    wVyDown: 0.09,
    wVx: 0.008,
    wAngleDeg: 0.08,
    angleNearGroundBoost: 4.0,
    wAngleRate: 0.01,

    hardWalls: true,
    borderMargin: 0.0,
    borderPenaltyPerSec: 0.0,
    wrapPenalty: 0.0,

    ceilingMargin: 40.0,
    ceilingPenaltyPerSec: 4.0,
    ceilingHitPenalty: 8.0,

    lockTerrain: true,
    terrainSeed: 123,
    lockSpawn: false,
    spawnX: 0.25,
    spawnY: 120.0,
    spawnVx: 0.0,
    spawnVy: 0.0,
    spawnAngle: 0.0,
    randomSpawnX: true,
    spawnXMin: 0.20,
    spawnXMax: 0.80,
  );

  final env = eng.GameEngine(cfg);
  final fe = FeatureExtractor(groundSamples: 3, stridePx: 48);
  final policy = PolicyNetwork(inputSize: fe.inputSize, h1: 64, h2: 64, seed: 1234);
  final trainer = Trainer(env: env, fe: fe, policy: policy, dt: 1/60.0, gamma: 0.995, seed: 7);

  // Training loop with temperature + checkpoints
  final rnd = math.Random(7);
  int landedCount = 0;
  double bestCost = double.infinity;

  const iters = 400;
  for (int i = 1; i <= iters; i++) {
    // Simple temperature schedule: hotter early, cool later
    final frac = (i / iters).clamp(0.0, 1.0);
    final tThr = 1.3 - 0.8 * frac;   // 1.3 → 0.5
    final tTurn = 1.7 - 1.0 * frac;  // 1.7 → 0.7

    // Optional anneal toward harder physics after some iters
    if (i == 200) {
      env.cfg = env.cfg.copyWith(
        t: env.cfg.t.copyWith(rotSpeed: 1.9, thrustAccel: 0.50, maxFuel: 140.0),
        padWidthFactor: 1.5,
        landingSpeedMax: 95.0,
        landingAngleMaxRad: 0.55,
      );
    }

    final ep = trainer.runEpisode(
      train: true,
      lr: 3e-4,
      tThr: tThr,
      tTurn: tTurn,
      addLandingBonus: true,
      landingBonus: 30.0,
    );
    if (ep.landed) landedCount++;
    if (ep.totalCost < bestCost) {
      bestCost = ep.totalCost;
      await savePolicy('policy_best_cost.json', policy, fe);
      // also overwrite the main file so the app can pick it up immediately if you want
      await savePolicy('policy.json', policy, fe);
    }

    if (i % 5 == 0) {
      final landRate = (landedCount / i * 100.0);
      stdout.writeln(
          'Iter $i | steps: ${ep.steps} | cost: ${ep.totalCost.toStringAsFixed(3)} | '
              'landed: ${ep.landed ? 'Y' : 'N'} | best(cost): ${bestCost.toStringAsFixed(3)} | '
              'land%: ${landRate.toStringAsFixed(1)}  | Tthr=${tThr.toStringAsFixed(2)} Tturn=${tTurn.toStringAsFixed(2)}'
      );
    }

    // Save rolling "last" occasionally
    if (i % 25 == 0) {
      await savePolicy('policy_last.json', policy, fe);
    }
  }

  await savePolicy('policy_final.json', policy, fe);
  // Optional: keep `policy.json` as the final artifact
  await savePolicy('policy.json', policy, fe);
}
