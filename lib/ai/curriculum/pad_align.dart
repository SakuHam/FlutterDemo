// lib/ai/curriculum/pad_align.dart
//
// Micro-stage: “pad-align (balanced)”
// Goal: learn to choose goLeft/goRight intents with very high accuracy.
// Key tactics:
//  - Perfectly BALANCED spawns left/right of pad (no class bias).
//  - Enforce MIN offset from pad center (no ambiguous/hover states).
//  - Supervise ONLY on L/R frames (dense, clean signal).
//  - Many tiny episodes per iteration.
//  - Fitness print shows overall & per-class accuracy + bias + Δ|dx|/step.
//
// CLI knobs (examples):
//   --curricula=padalign
//   --padalign_batch=8 --padalign_steps_min=6 --padalign_steps_max=12
//   --padalign_band_frac=0.15 --padalign_min_offset_px=20
//   --padalign_print_every=25
//
// This curriculum updates ONLY the intent head (intentMode=true).

import 'dart:math' as math;

import '../../engine/types.dart' as et;
import '../../engine/game_engine.dart' as eng;
import '../agent.dart';
import 'core.dart';

class PadAlignCfg {
  final int batch;            // episodes per iteration (↑ for more labels)
  final int stepsMin;         // min steps per episode
  final int stepsMax;         // max steps per episode
  final double bandFrac;      // horizontal band as fraction of world width
  final double minOffsetPx;   // enforce >= this |dx| at spawn (no-center)
  final bool balancedSides;   // alternate L/R spawns exactly
  final bool randomVY;        // add small downward vy
  final double vyMin, vyMax;  // vy range if randomVY
  final int printEvery;       // fitness print cadence
  final bool verbose;

  const PadAlignCfg({
    this.batch = 8,
    this.stepsMin = 6,
    this.stepsMax = 12,
    this.bandFrac = 0.15,
    this.minOffsetPx = 20.0,
    this.balancedSides = true,
    this.randomVY = true,
    this.vyMin = 8.0,
    this.vyMax = 18.0,
    this.printEvery = 25,
    this.verbose = true,
  });

  PadAlignCfg copyWith({
    int? batch,
    int? stepsMin,
    int? stepsMax,
    double? bandFrac,
    double? minOffsetPx,
    bool? balancedSides,
    bool? randomVY,
    double? vyMin,
    double? vyMax,
    int? printEvery,
    bool? verbose,
  }) {
    return PadAlignCfg(
      batch: batch ?? this.batch,
      stepsMin: stepsMin ?? this.stepsMin,
      stepsMax: stepsMax ?? this.stepsMax,
      bandFrac: bandFrac ?? this.bandFrac,
      minOffsetPx: minOffsetPx ?? this.minOffsetPx,
      balancedSides: balancedSides ?? this.balancedSides,
      randomVY: randomVY ?? this.randomVY,
      vyMin: vyMin ?? this.vyMin,
      vyMax: vyMax ?? this.vyMax,
      printEvery: printEvery ?? this.printEvery,
      verbose: verbose ?? this.verbose,
    );
  }
}

class PadAlignCurriculum extends Curriculum {
  @override
  String get key => 'padalign';

  PadAlignCfg cfg = const PadAlignCfg();

  // Rolling fitness
  final List<double> _accWindow = <double>[];
  final int _accWinMax = 40;

  int _lrTotal = 0, _lrCorrect = 0;
  int _lSeen = 0, _lCorrect = 0, _rSeen = 0, _rCorrect = 0;
  int _goLeftPicks = 0, _goRightPicks = 0;
  double _sumDxDelta = 0.0; int _dxDeltaN = 0;

  @override
  Curriculum configure(Map<String, String?> kv, Set<String> flags) {
    final cli = CliView(kv, flags);
    cfg = cfg.copyWith(
      batch: cli.getInt('padalign_batch', def: 8),
      stepsMin: cli.getInt('padalign_steps_min', def: 6),
      stepsMax: cli.getInt('padalign_steps_max', def: 12),
      bandFrac: cli.getDouble('padalign_band_frac', def: 0.15),
      minOffsetPx: cli.getDouble('padalign_min_offset_px', def: 20.0),
      balancedSides: cli.getFlag('padalign_balanced', def: true),
      randomVY: cli.getFlag('padalign_random_vy', def: true),
      vyMin: cli.getDouble('padalign_vy_min', def: 8.0),
      vyMax: cli.getDouble('padalign_vy_max', def: 18.0),
      printEvery: cli.getInt('padalign_print_every', def: 25),
      verbose: cli.getFlag('padalign_verbose', def: true),
    );
    _resetFitness();
    return this;
  }

  void _resetFitness() {
    _accWindow.clear();
    _lrTotal = 0; _lrCorrect = 0;
    _lSeen = 0; _lCorrect = 0; _rSeen = 0; _rCorrect = 0;
    _goLeftPicks = 0; _goRightPicks = 0;
    _sumDxDelta = 0.0; _dxDeltaN = 0;
  }

  // Label: which lateral intent is correct? (no deadband here)
  int _lrLabel(eng.GameEngine env) {
    final padCx = env.terrain.padCenter.toDouble();
    final dx = padCx - env.lander.pos.x.toDouble();
    return dx >= 0 ? Intent.goRight.index : Intent.goLeft.index;
  }

  String _sparkline(List<double> vals, {int width = 12}) {
    if (vals.isEmpty) return '';
    const blocks = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    List<double> take;
    if (vals.length <= width) {
      take = List<double>.from(vals);
    } else {
      take = List<double>.filled(width, 0.0);
      for (int i = 0; i < width; i++) {
        final a = (i * vals.length) ~/ width;
        final b = ((i + 1) * vals.length) ~/ width;
        double s = 0.0; int n = 0;
        for (int j = a; j < b; j++) { s += vals[j]; n++; }
        take[i] = n > 0 ? s / n : 0.0;
      }
    }
    final buf = StringBuffer('⎡');
    for (final v in take) {
      final cl = v.clamp(0.0, 1.0);
      final idx = (cl * (blocks.length - 1)).round();
      buf.write(blocks[idx]);
    }
    buf.write('⎤');
    return buf.toString();
  }

  void _recordEpisodeFitness({
    required double correctFrac,
    required int lSeen, required int lOk,
    required int rSeen, required int rOk,
    required int leftPicks, required int rightPicks,
    required double meanDxDeltaPerStep,
  }) {
    _accWindow.add(correctFrac);
    if (_accWindow.length > _accWinMax) _accWindow.removeAt(0);

    final seen = lSeen + rSeen;
    _lrCorrect += (correctFrac * seen).round();
    _lrTotal   += seen;
    _lSeen     += lSeen; _lCorrect += lOk;
    _rSeen     += rSeen; _rCorrect += rOk;
    _goLeftPicks  += leftPicks;
    _goRightPicks += rightPicks;

    _sumDxDelta += meanDxDeltaPerStep;
    _dxDeltaN   += 1;
  }

  void _printFitnessLine(int iter) {
    final acc  = _lrTotal > 0 ? (_lrCorrect / _lrTotal) : 0.0;
    final lAcc = _lSeen   > 0 ? (_lCorrect / _lSeen)    : 0.0;
    final rAcc = _rSeen   > 0 ? (_rCorrect / _rSeen)    : 0.0;
    final picks = _goLeftPicks + _goRightPicks;
    final biasL = picks > 0 ? (_goLeftPicks / picks) : 0.5;
    final dxStep = _dxDeltaN > 0 ? (_sumDxDelta / _dxDeltaN) : 0.0;

    print('[PADALIGN/FIT] it=$iter'
        '  acc=${(100*acc).toStringAsFixed(1)}%'
        '  L_ok=${(100*lAcc).toStringAsFixed(0)}%'
        '  R_ok=${(100*rAcc).toStringAsFixed(0)}%'
        '  bias(L)=${(100*biasL).toStringAsFixed(0)}%  ${_sparkline(_accWindow)}'
        '  Δ|dx|/step=${dxStep.toStringAsFixed(1)}');
  }

  // Spawn exactly left or right of pad with min offset
  void _spawnBalanced({
    required eng.GameEngine env,
    required math.Random rnd,
    required bool spawnLeft,
    required double bandFrac,
    required double minOffsetPx,
    required bool randomVY,
    required double vyMin,
    required double vyMax,
  }) {
    final padCx = env.terrain.padCenter.toDouble();
    final W = env.cfg.worldW.toDouble();

    final maxSpan = (bandFrac * W).clamp(1.0, W * 0.45);
    final offset = minOffsetPx + rnd.nextDouble() * math.max(1.0, maxSpan - minOffsetPx);
    final x = (spawnLeft ? (padCx - offset) : (padCx + offset)).clamp(10.0, W - 10.0);

    final gy = env.terrain.heightAt(x);
    final h = 80.0 + rnd.nextDouble() * 60.0;
    final vx0 = (rnd.nextDouble() * 16.0) - 8.0;
    final vy0 = randomVY ? (vyMin + rnd.nextDouble() * (vyMax - vyMin)) : 0.0;

    env.lander
      ..pos.x = x
      ..pos.y = (gy - h).clamp(0.0, env.cfg.worldH - 10.0)
      ..vel.x = vx0
      ..vel.y = vy0
      ..angle = 0.0
      ..fuel = env.cfg.t.maxFuel;
  }

  EpisodeResult _runEpisode({
    required eng.GameEngine env,
    required FeatureExtractorRays fe,
    required PolicyNetwork policy,
    required RunningNorm? norm,
    required math.Random rnd,
    required int stepsTarget,
    required bool spawnLeft,     // enforce balance outside
    required double bandFrac,
    required double minOffsetPx,
    bool dry = true,
  }) {
    env.reset(seed: rnd.nextInt(1 << 30));
    _spawnBalanced(
      env: env, rnd: rnd, spawnLeft: spawnLeft,
      bandFrac: bandFrac, minOffsetPx: minOffsetPx,
      randomVY: cfg.randomVY, vyMin: cfg.vyMin, vyMax: cfg.vyMax,
    );

    // Supervision containers (intent only)
    final decisionCaches  = <ForwardCache>[];
    final intentChoices   = <int>[];
    final decisionReturns = <double>[];
    final alignLabels     = <int>[];

    // Fitness tallies
    int epLSeen = 0, epRSeen = 0, epLCorrect = 0, epRCorrect = 0;
    int epLeftPicks = 0, epRightPicks = 0;
    double epDxDeltaSum = 0.0; int epDxDeltaN = 0;

    int steps = 0;
    double totalCost = 0.0;
    bool landed = false;

    while (steps < stepsTarget) {
      // Extract features
      var xfeat = fe.extract(
        lander: env.lander,
        terrain: env.terrain,
        worldW: env.cfg.worldW,
        worldH: env.cfg.worldH,
        rays: env.rays,
      );
      if (norm != null) {
        norm.observe(xfeat);
        xfeat = norm.normalize(xfeat, update: false);
      }

      // Policy forward for cache & fitness
      final (greedyIdx, _probs, cache) = policy.actIntentGreedy(xfeat);

      // L/R label (always defined because we enforce minOffsetPx)
      final label = _lrLabel(env); // goLeft or goRight
      final chosenIdx = label;     // teacher forces lateral intent

      // Fitness tallies
      if (Intent.values[greedyIdx] == Intent.goLeft ||
          Intent.values[greedyIdx] == Intent.goRight) {
        final correct = (greedyIdx == label);
        if (label == Intent.goLeft.index) { epLSeen++; if (correct) epLCorrect++; }
        else                               { epRSeen++; if (correct) epRCorrect++; }
        if (greedyIdx == Intent.goLeft.index)  epLeftPicks++;
        if (greedyIdx == Intent.goRight.index) epRightPicks++;
      }

      // Supervision (intent only)
      decisionCaches.add(cache);
      intentChoices.add(chosenIdx);
      alignLabels.add(label);
      decisionReturns.add(0.0); // pure supervised

      // Δ|dx|/step metric
      final padCx0 = env.terrain.padCenter.toDouble();
      final dx0 = (env.lander.pos.x.toDouble() - padCx0).abs();

      // Teacher controller for chosen intent
      final u = controllerForIntent(Intent.values[chosenIdx], env);
      final info = env.step(1 / 60.0, u);
      totalCost += info.costDelta;

      final padCx1 = env.terrain.padCenter.toDouble();
      final dx1 = (env.lander.pos.x.toDouble() - padCx1).abs();
      epDxDeltaSum += (dx1 - dx0);
      epDxDeltaN   += 1;

      steps++;
      if (info.terminal) { landed = (env.status == et.GameStatus.landed); break; }
    }

    // Supervised update (intent head only)
    if (!dry && decisionCaches.isNotEmpty) {
      policy.updateFromEpisode(
        decisionCaches: decisionCaches,
        intentChoices: intentChoices,
        decisionReturns: decisionReturns,
        alignLabels: alignLabels,
        alignWeight: 1.0,
        intentPgWeight: 0.0,
        lr: 1e-3,
        entropyBeta: 0.0,
        valueBeta: 0.0,
        huberDelta: 1.0,
        intentMode: true,
        actionCaches: const [],
        actionTurnTargets: const [],
        actionThrustTargets: const [],
        actionAlignWeight: 0.0,
      );
    }

    // Episode fitness record
    final epSeen = epLSeen + epRSeen;
    final epAcc = epSeen > 0 ? (epLCorrect + epRCorrect) / epSeen : 0.0;
    final meanDxDeltaPerStep = epDxDeltaN > 0 ? (epDxDeltaSum / epDxDeltaN) : 0.0;

    _recordEpisodeFitness(
      correctFrac: epAcc,
      lSeen: epLSeen, lOk: epLCorrect,
      rSeen: epRSeen, rOk: epRCorrect,
      leftPicks: epLeftPicks, rightPicks: epRightPicks,
      meanDxDeltaPerStep: meanDxDeltaPerStep,
    );

    return EpisodeResult(
      steps: steps,
      totalCost: totalCost,
      landed: landed,
      segMean: 0.0,
    );
  }

  @override
  Future<void> run({
    required int iters,
    required eng.GameEngine env,
    required FeatureExtractorRays fe,
    required PolicyNetwork policy,
    required RunningNorm? norm,
    required int planHold,
    required double tempIntent,
    required double gamma,
    required double lr,
    required double intentAlignWeight,
    required double intentPgWeight,
    required double actionAlignWeight,
    required bool gateVerbose,
    required int seed,
  }) async {
    final rnd = math.Random(seed ^ 0x51A1);
    if (gateVerbose) {
      print('[CUR/padalign] start iters=$iters batch=${cfg.batch} '
          'steps=${cfg.stepsMin}..${cfg.stepsMax} bandFrac=${cfg.bandFrac} '
          'minOff=${cfg.minOffsetPx} balanced=${cfg.balancedSides ? "Y" : "N"}');
    }

    // === Baseline: no updates ===
    _resetFitness();
    for (int b = 0; b < cfg.batch; b++) {
      final stepsTarget = (cfg.stepsMin == cfg.stepsMax)
          ? cfg.stepsMin
          : (cfg.stepsMin + rnd.nextInt(math.max(1, cfg.stepsMax - cfg.stepsMin + 1)));
      final spawnLeft = cfg.balancedSides ? (b % 2 == 0) : rnd.nextBool();
      _runEpisode(
        env: env, fe: fe, policy: policy, norm: norm, rnd: rnd,
        stepsTarget: stepsTarget, spawnLeft: spawnLeft,
        bandFrac: cfg.bandFrac, minOffsetPx: cfg.minOffsetPx,
        dry: true, // <-- probe
      );
    }
    _printFitnessLine(0); // prints as it=0 before any updates

    for (int it = 0; it < iters; it++) {
      for (int b = 0; b < cfg.batch; b++) {
        final stepsTarget = (cfg.stepsMin == cfg.stepsMax)
            ? cfg.stepsMin
            : (cfg.stepsMin + rnd.nextInt(math.max(1, cfg.stepsMax - cfg.stepsMin + 1)));

        // enforce perfect balance if enabled
        final spawnLeft = cfg.balancedSides
            ? (((it * cfg.batch) + b) % 2 == 0)
            : (rnd.nextBool());

        _runEpisode(
          env: env,
          fe: fe,
          policy: policy,
          norm: norm,
          rnd: rnd,
          stepsTarget: stepsTarget,
          spawnLeft: spawnLeft,
          bandFrac: cfg.bandFrac,
          minOffsetPx: cfg.minOffsetPx,
        );
      }

      if (((it + 1) % cfg.printEvery) == 0) {
        _printFitnessLine(it + 1);
      }
    }
  }
}
