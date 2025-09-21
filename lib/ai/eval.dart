// lib/ai/eval.dart
import 'dart:isolate';
import 'dart:math' as math;

import '../engine/types.dart' as et;
import '../engine/game_engine.dart' as eng;
import '../engine/raycast.dart'; // RayConfig

import 'agent.dart'; // FeatureExtractorRays, PolicyNetwork, Trainer

/* ------------------------------- public types -------------------------------- */

class EvalStats {
  double meanCost = 0;
  double medianCost = 0;
  double landPct = 0;
  double crashPct = 0;
  double meanSteps = 0;
  double meanAbsDx = 0;
}

/* ------------------------------- internals ----------------------------------- */

class _EvalChunkResult {
  final List<double> costs;
  final int landed;
  final int crashed;
  final int stepsSum;
  final double absDxSum;
  _EvalChunkResult(this.costs, this.landed, this.crashed, this.stepsSum, this.absDxSum);
}

({double lo, double hi}) _wilson95(int success, int n) {
  if (n <= 0) return (lo: 0, hi: 0);
  const z = 1.96;
  final p = success / n;
  final denom = 1 + z * z / n;
  final center = (p + z * z / (2 * n)) / denom;
  final half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom;
  return (lo: 100.0 * (center - half), hi: 100.0 * (center + half));
}

_EvalChunkResult _evalChunk({
  required int episodes,
  required int seed,
  required int attemptsPerTerrain,
  required PolicyNetwork policyClone,
  required List<int> hidden,
  required et.EngineConfig cfg,
  // match training/runtime exactly
  required int planHold,
  required double blendPolicy,
  required double tempIntent,
  required double intentEntropy,
  required bool evalDebug,
  required int evalDebugFailN,
}) {
  final env = eng.GameEngine(cfg);
  env.rayCfg = const RayConfig(rayCount: 180, includeFloor: false, forwardAligned: false);
  final fe = FeatureExtractorRays(rayCount: env.rayCfg.rayCount);

  final trainer = Trainer(
    env: env,
    fe: fe,
    policy: policyClone,
    dt: 1 / 60.0,
    gamma: 0.99,
    seed: seed,
    twoStage: true,
    planHold: planHold,
    tempIntent: tempIntent,
    intentEntropyBeta: intentEntropy,
    useLearnedController: false,
    blendPolicy: blendPolicy,
    intentAlignWeight: 0.0,
    intentPgWeight: 0.0,
    actionAlignWeight: 0.0,
    normalizeFeatures: true,
    gateScoreMin: -1e9,
    gateOnlyLanded: false,
    gateVerbose: false,
    externalRewardHook: null,
  );

  final rnd = math.Random(seed);
  final costs = <double>[];
  int landed = 0, crashed = 0, stepsSum = 0;
  double absDxSum = 0.0;

  int terrAttempts = 0;
  int currentTerrainSeed = rnd.nextInt(1 << 30);

  int printedFails = 0;

  for (int i = 0; i < episodes; i++) {
    if (terrAttempts == 0) {
      currentTerrainSeed = rnd.nextInt(1 << 30);
    }
    env.reset(seed: currentTerrainSeed);
    final res = trainer.runEpisode(train: false, greedy: false, scoreIsReward: false);
    terrAttempts = (terrAttempts + 1) % attemptsPerTerrain;

    costs.add(res.totalCost);
    stepsSum += res.steps;

    final ok = (env.status == et.GameStatus.landed);
    if (ok) {
      landed++;
    } else {
      crashed++;
      if (evalDebug && printedFails < evalDebugFailN) {
        final L = env.lander;
        final T = env.terrain;
        final gx = T.heightAt(L.pos.x.toDouble());
        final h  = (gx - L.pos.y).toDouble();
        // Keep this concise: enough to debug without spamming.
        // ignore: avoid_print
        print('[EVAL DBG FAIL] ep=$i terrSeed=$currentTerrainSeed '
            'status=${env.status} x=${L.pos.x.toStringAsFixed(1)} y=${L.pos.y.toStringAsFixed(1)} '
            'h=${h.toStringAsFixed(1)} vx=${L.vel.x.toStringAsFixed(1)} vy=${L.vel.y.toStringAsFixed(1)} '
            'fuel=${L.fuel.toStringAsFixed(1)} padCx=${T.padCenter.toStringAsFixed(1)} '
            '| cost=${res.totalCost.toStringAsFixed(3)} steps=${res.steps}');
        printedFails++;
      }
    }
    final padCx = env.terrain.padCenter;
    absDxSum += (env.lander.pos.x - padCx).abs();
  }

  return _EvalChunkResult(costs, landed, crashed, stepsSum, absDxSum);
}

/* --------------------------------- API --------------------------------------- */

Future<EvalStats> evaluateParallel({
  required et.EngineConfig cfg,
  required PolicyNetwork policy,
  required List<int> hidden,
  required int episodes,
  required int attemptsPerTerrain,
  required int seed,
  required int workers,
  // match training/runtime
  required int planHold,
  required double blendPolicy,
  required double tempIntent,
  required double intentEntropy,
  // debug
  required bool evalDebug,
  required int evalDebugFailN,
}) async {
  final per = episodes ~/ workers;
  final extra = episodes % workers;

  // clone params once for read-only use
  PolicyNetwork _clonePolicy(PolicyNetwork p) {
    final cp = PolicyNetwork(inputSize: p.inputSize, hidden: List<int>.from(hidden), seed: seed ^ 0xA11CE);
    // trunk
    for (int li = 0; li < cp.trunk.layers.length; li++) {
      final Ld = cp.trunk.layers[li];
      final Ls = p.trunk.layers[li];
      for (int i = 0; i < Ld.W.length; i++) {
        for (int j = 0; j < Ld.W[0].length; j++) {
          Ld.W[i][j] = Ls.W[i][j];
        }
      }
      for (int i = 0; i < Ld.b.length; i++) Ld.b[i] = Ls.b[i];
    }
    // heads
    List<List<double>> _copyW(List<List<double>> W) =>
        List.generate(W.length, (i) => List<double>.from(W[i]));
    cp.heads.intent.W = _copyW(p.heads.intent.W); cp.heads.intent.b = List<double>.from(p.heads.intent.b);
    cp.heads.turn.W   = _copyW(p.heads.turn.W);   cp.heads.turn.b   = List<double>.from(p.heads.turn.b);
    cp.heads.thr.W    = _copyW(p.heads.thr.W);    cp.heads.thr.b    = List<double>.from(p.heads.thr.b);
    cp.heads.val.W    = _copyW(p.heads.val.W);    cp.heads.val.b    = List<double>.from(p.heads.val.b);
    return cp;
  }

  final futures = <Future<_EvalChunkResult>>[];
  for (int w = 0; w < workers; w++) {
    final nThis = per + (w < extra ? 1 : 0);
    if (nThis == 0) continue;

    final pClone = _clonePolicy(policy);
    final seedW = seed ^ (0xBEEF << (w & 15));

    futures.add(Isolate.run(() {
      try {
        return _evalChunk(
          episodes: nThis,
          seed: seedW,
          attemptsPerTerrain: attemptsPerTerrain,
          policyClone: pClone,
          hidden: hidden,
          cfg: cfg,
          planHold: planHold,
          blendPolicy: blendPolicy,
          tempIntent: tempIntent,
          intentEntropy: intentEntropy,
          evalDebug: evalDebug,
          evalDebugFailN: evalDebugFailN,
        );
      } catch (e, st) {
        // ignore: avoid_print
        print('[EVAL WORKER ERROR] $e\n$st');
        return _EvalChunkResult(const [], 0, 0, 0, 0.0);
      }
    }));
  }

  final chunks = await Future.wait(futures);

  // reduce
  final costs = <double>[];
  int landed = 0, crashed = 0, stepsSum = 0;
  double absDxSum = 0.0;

  for (final c in chunks) {
    costs.addAll(c.costs);
    landed += c.landed;
    crashed += c.crashed;
    stepsSum += c.stepsSum;
    absDxSum += c.absDxSum;
  }

  costs.sort();
  final st = EvalStats();
  st.meanCost = costs.isEmpty ? 0 : costs.reduce((a, b) => a + b) / costs.length;
  st.medianCost = costs.isEmpty ? 0 : costs[costs.length ~/ 2];
  final total = landed + crashed;
  st.landPct = total == 0 ? 0 : 100.0 * landed / total;
  st.crashPct = total == 0 ? 0 : 100.0 * crashed / total;
  st.meanSteps = total == 0 ? 0 : stepsSum / total;
  st.meanAbsDx = total == 0 ? 0 : absDxSum / total;

  // ignore: avoid_print
  print('Eval: N=$episodes | workers=$workers | '
      '| land%=${st.landPct.toStringAsFixed(1)} '
      '| meanCost=${st.meanCost.toStringAsFixed(3)} '
      '| median=${st.medianCost.toStringAsFixed(3)} | steps=${st.meanSteps.toStringAsFixed(1)} '
      '| mean|dx|=${st.meanAbsDx.toStringAsFixed(1)}');

  return st;
}

EvalStats evaluateSequential({
  required eng.GameEngine env,
  required Trainer trainer,
  int episodes = 40,
  int seed = 123,
  int attemptsPerTerrain = 1,
  bool evalDebug = false,
  int evalDebugFailN = 3,
}) {
  final rnd = math.Random(seed);
  final costs = <double>[];
  int landed = 0, crashed = 0, stepsSum = 0;
  double absDxSum = 0.0;

  int terrAttempts = 0;
  int currentTerrainSeed = rnd.nextInt(1 << 30);

  int printedFails = 0;

  for (int i = 0; i < episodes; i++) {
    if (terrAttempts == 0) currentTerrainSeed = rnd.nextInt(1 << 30);
    env.reset(seed: currentTerrainSeed);

    final res = trainer.runEpisode(train: false, greedy: false, scoreIsReward: false);

    terrAttempts = (terrAttempts + 1) % attemptsPerTerrain;

    costs.add(res.totalCost);
    stepsSum += res.steps;
    if (env.status == et.GameStatus.landed) {
      landed++;
    } else {
      crashed++;
      if (evalDebug && printedFails < evalDebugFailN) {
        final L = env.lander;
        final T = env.terrain;
        final gx = T.heightAt(L.pos.x.toDouble());
        final h  = (gx - L.pos.y).toDouble();
        // ignore: avoid_print
        print('[EVAL DBG FAIL] ep=$i terrSeed=$currentTerrainSeed '
            'status=${env.status} x=${L.pos.x.toStringAsFixed(1)} y=${L.pos.y.toStringAsFixed(1)} '
            'h=${h.toStringAsFixed(1)} vx=${L.vel.x.toStringAsFixed(1)} vy=${L.vel.y.toStringAsFixed(1)} '
            'fuel=${L.fuel.toStringAsFixed(1)} '
            'padCx=${T.padCenter.toStringAsFixed(1)} | cost=${res.totalCost.toStringAsFixed(3)} steps=${res.steps}');
        printedFails++;
      }
    }
    final padCx = env.terrain.padCenter;
    absDxSum += (env.lander.pos.x - padCx).abs();
  }

  costs.sort();
  final st = EvalStats();
  st.meanCost = costs.isEmpty ? 0 : costs.reduce((a, b) => a + b) / costs.length;
  st.medianCost = costs.isEmpty ? 0 : costs[costs.length ~/ 2];
  st.landPct = 100.0 * landed / episodes;
  st.crashPct = 100.0 * crashed / episodes;
  st.meanSteps = stepsSum / episodes;
  st.meanAbsDx = absDxSum / episodes;

  // ignore: avoid_print
  print('Eval: N=$episodes | workers=1 '
      '| land%=${st.landPct.toStringAsFixed(1)} '
      '| meanCost=${st.meanCost.toStringAsFixed(3)} '
      '| median=${st.medianCost.toStringAsFixed(3)} | steps=${st.meanSteps.toStringAsFixed(1)} '
      '| mean|dx|=${st.meanAbsDx.toStringAsFixed(1)}');

  return st;
}
