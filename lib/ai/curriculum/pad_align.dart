// lib/ai/curriculum/pad_align.dart
//
// Micro-stage: “pad-align”
// - Very short episodes (few steps), random spawn near pad center
// - Teacher FORCES lateral intent (goLeft/goRight) based on pad side
// - NN is trained with supervised intent alignment (no PG needed here)
// - We still log what the *policy would have picked* to compute fitness
// - Prints fitness summary with a rolling sparkline and |dx| trend

import 'dart:math' as math;

import '../../engine/types.dart' as et;
import '../../engine/game_engine.dart' as eng;
import '../agent.dart';
import 'core.dart'; // Curriculum, CliView

class PadAlignCfg {
  final int batch;            // episodes per iteration
  final int stepsMin;         // min episode steps (tiny)
  final int stepsMax;         // max episode steps
  final double bandFrac;      // horizontal spawn band around pad center (fraction of world width)
  final double tempIntent;    // temperature for policy *evaluation* (fitness probe)
  final bool verbose;
  final int printEvery;       // print fitness every N iters
  final bool randomVY;        // small downward vy to make it non-trivial
  final double vyMin, vyMax;  // start vy range if randomVY
  final bool allowHoverWhenCentered; // treat near-zero dx as hover (else ignore)

  const PadAlignCfg({
    this.batch = 1,
    this.stepsMin = 8,
    this.stepsMax = 24,
    this.bandFrac = 0.18,
    this.tempIntent = 1.0,
    this.verbose = true,
    this.printEvery = 25,
    this.randomVY = true,
    this.vyMin = 8.0,
    this.vyMax = 18.0,
    this.allowHoverWhenCentered = true,
  });

  PadAlignCfg copyWith({
    int? batch,
    int? stepsMin,
    int? stepsMax,
    double? bandFrac,
    double? tempIntent,
    bool? verbose,
    int? printEvery,
    bool? randomVY,
    double? vyMin,
    double? vyMax,
    bool? allowHoverWhenCentered,
  }) {
    return PadAlignCfg(
      batch: batch ?? this.batch,
      stepsMin: stepsMin ?? this.stepsMin,
      stepsMax: stepsMax ?? this.stepsMax,
      bandFrac: bandFrac ?? this.bandFrac,
      tempIntent: tempIntent ?? this.tempIntent,
      verbose: verbose ?? this.verbose,
      printEvery: printEvery ?? this.printEvery,
      randomVY: randomVY ?? this.randomVY,
      vyMin: vyMin ?? this.vyMin,
      vyMax: vyMax ?? this.vyMax,
      allowHoverWhenCentered: allowHoverWhenCentered ?? this.allowHoverWhenCentered,
    );
  }
}

class PadAlignCurriculum extends Curriculum {
  @override
  String get key => 'padalign';

  PadAlignCfg cfg = const PadAlignCfg();

  // ---------- Fitness tracking state (rolling) ----------
  final List<double> _accWindow = <double>[]; // episode accuracies 0..1
  final int _accWinMax = 40;

  int _lrTotal = 0;     // total LR decisions considered
  int _lrCorrect = 0;   // total correct LR decisions
  int _lSeen = 0, _lCorrect = 0;
  int _rSeen = 0, _rCorrect = 0;
  int _goLeftPicks = 0, _goRightPicks = 0;

  // Horizontal distance trend (per-episode mean of per-step Δ|dx|)
  double _sumDxDelta = 0.0;
  int _dxDeltaN = 0;

  @override
  Curriculum configure(Map<String, String?> kv, Set<String> flags) {
    final cli = CliView(kv, flags);
    cfg = cfg.copyWith(
      batch:    cli.getInt('padalign_batch', def: 1),
      stepsMin: cli.getInt('padalign_steps_min', def: 8),
      stepsMax: cli.getInt('padalign_steps_max', def: 24),
      bandFrac: cli.getDouble('padalign_band_frac', def: 0.18),
      tempIntent: cli.getDouble('padalign_temp', def: 1.0),
      verbose:  cli.getFlag('padalign_verbose', def: true),
      printEvery: cli.getInt('padalign_print_every', def: 25),
      randomVY: cli.getFlag('padalign_random_vy', def: true),
      vyMin:    cli.getDouble('padalign_vy_min', def: 8.0),
      vyMax:    cli.getDouble('padalign_vy_max', def: 18.0),
      allowHoverWhenCentered: cli.getFlag('padalign_allow_hover', def: true),
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

  // ------------- Helpers -------------
  bool _isLateralIntent(int idx) {
    final i = Intent.values[idx];
    return i == Intent.goLeft || i == Intent.goRight;
  }

  int _lrLabelForEnv(eng.GameEngine env, {double deadbandPx = 2.0}) {
    final padCx = env.terrain.padCenter.toDouble();
    final dx = padCx - env.lander.pos.x.toDouble();
    if (dx > deadbandPx) return Intent.goRight.index;
    if (dx < -deadbandPx) return Intent.goLeft.index;
    return -1; // inside deadband; either hover or ignore
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
    required double correctFrac, // 0..1
    required int lSeen, required int lOk,
    required int rSeen, required int rOk,
    required int leftPicks, required int rightPicks,
    required double meanDxDeltaPerStep, // negative is good
  }) {
    _accWindow.add(correctFrac);
    if (_accWindow.length > _accWinMax) _accWindow.removeAt(0);

    final seen = (lSeen + rSeen);
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
    double acc  = _lrTotal > 0 ? (_lrCorrect / _lrTotal) : 0.0;
    double lAcc = _lSeen   > 0 ? (_lCorrect / _lSeen)    : 0.0;
    double rAcc = _rSeen   > 0 ? (_rCorrect / _rSeen)    : 0.0;
    final picks = _goLeftPicks + _goRightPicks;
    final biasL = picks > 0 ? (_goLeftPicks / picks) : 0.5;
    final bar   = _sparkline(_accWindow, width: 12);
    final dxStep = _dxDeltaN > 0 ? (_sumDxDelta / _dxDeltaN) : 0.0;

    print('[PADALIGN/FIT] it=$iter'
        '  acc=${(100*acc).toStringAsFixed(1)}%'
        '  L_ok=${(100*lAcc).toStringAsFixed(0)}%'
        '  R_ok=${(100*rAcc).toStringAsFixed(0)}%'
        '  bias(L)=${(100*biasL).toStringAsFixed(0)}%  $bar'
        '  Δ|dx|/step=${dxStep.toStringAsFixed(dxStep.abs()<10?1:0)}');
  }

  // ------------- Episode runner -------------
  EpisodeResult _runEpisode({
    required eng.GameEngine env,
    required FeatureExtractorRays fe,
    required PolicyNetwork policy,
    required RunningNorm? norm,
    required math.Random rnd,
    required int stepsTarget,
    required double bandFrac,
    required double tempForProbe,
  }) {
    // Spawn near pad center
    final padCx = env.terrain.padCenter.toDouble();
    final W = env.cfg.worldW.toDouble();
    final x = (padCx + (rnd.nextDouble() * 2 - 1) * (bandFrac * W)).clamp(10.0, W - 10.0);
    final gy = env.terrain.heightAt(x);
    final h = 80.0 + rnd.nextDouble() * 60.0; // modest height
    final vx0 = (rnd.nextDouble() * 16.0) - 8.0;
    final vy0 = cfg.randomVY ? (cfg.vyMin + rnd.nextDouble() * (cfg.vyMax - cfg.vyMin))
        : 0.0;

    env.reset(seed: rnd.nextInt(1 << 30));
    env.lander
      ..pos.x = x
      ..pos.y = (gy - h).clamp(0.0, env.cfg.worldH - 10.0)
      ..vel.x = vx0
      ..vel.y = vy0
      ..angle = 0.0
      ..fuel = env.cfg.t.maxFuel;

    // Supervision containers (intent head only)
    final decisionCaches  = <ForwardCache>[];
    final intentChoices   = <int>[];      // we use teacher label as the chosen intent
    final decisionReturns = <double>[];   // keep 0; we do pure supervised align here
    final alignLabels     = <int>[];      // teacher labels

    // Fitness tallies (per episode)
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

      // Policy forward (for cache + fitness probe)
      final (greedyIdx, probs, cache) = policy.actIntentGreedy(xfeat);
      // probe with temperature (optional; still take greedyIdx for correctness test)
      // If you prefer, compute argmax(softmax(logits/T)) here.

      // Teacher label: which side is the pad?
      final label = _lrLabelForEnv(env, deadbandPx: 2.0);
      int chosenIdx;
      if (label < 0) {
        // centered horizontally: either ignore or hover
        if (cfg.allowHoverWhenCentered) {
          chosenIdx = Intent.hover.index;
        } else {
          // ignore this step for training — but still step the env with a safe control
          chosenIdx = Intent.descendSlow.index;
        }
      } else {
        // FORCE lateral teacher
        chosenIdx = label;
      }

      // Fitness: how often would the policy pick the correct lateral intent?
      if (label >= 0 && _isLateralIntent(greedyIdx)) {
        final correct = (greedyIdx == label);
        if (label == Intent.goLeft.index) { epLSeen++; if (correct) epLCorrect++; }
        if (label == Intent.goRight.index){ epRSeen++; if (correct) epRCorrect++; }
        if (greedyIdx == Intent.goLeft.index)  epLeftPicks++;
        if (greedyIdx == Intent.goRight.index) epRightPicks++;
      }

      // Record supervision ONLY when we have a lateral label (or we allow hover)
      if (label >= 0 || cfg.allowHoverWhenCentered) {
        decisionCaches.add(cache);
        intentChoices.add(chosenIdx); // teacher chosen intent
        alignLabels.add(label >= 0 ? label : Intent.hover.index);
        decisionReturns.add(0.0); // pure supervised alignment; no advantage
      }

      // Horizontal distance trend
      final padCx0 = env.terrain.padCenter.toDouble();
      final dx0 = (env.lander.pos.x.toDouble() - padCx0).abs();

      // Map chosen intent to low-level teacher controller
      final intent = Intent.values[chosenIdx];
      final u = controllerForIntent(intent, env);

      // Step env
      final info = env.step(1 / 60.0, u);
      totalCost += info.costDelta;

      final padCx1 = env.terrain.padCenter.toDouble();
      final dx1 = (env.lander.pos.x.toDouble() - padCx1).abs();
      epDxDeltaSum += (dx1 - dx0); // negative is good (|dx| decreased)
      epDxDeltaN   += 1;

      steps++;
      if (info.terminal) { landed = (env.status == et.GameStatus.landed); break; }
    }

    // Do a small supervised update from this micro-episode
    if (decisionCaches.isNotEmpty) {
      // Intents only; set action head weights to zero from the caller (train loop)
      policy.updateFromEpisode(
        decisionCaches: decisionCaches,
        intentChoices: intentChoices,
        decisionReturns: decisionReturns,
        alignLabels: alignLabels,
        alignWeight: 1.0,         // strong supervised signal
        intentPgWeight: 0.0,      // no policy gradient needed here
        lr: 1e-3,                 // actual lr is passed by Trainer; this is ignored by Policy impl
        entropyBeta: 0.0,
        valueBeta: 0.0,
        huberDelta: 1.0,
        intentMode: true,
        // no action head supervision in this micro-stage
        actionCaches: const [],
        actionTurnTargets: const [],
        actionThrustTargets: const [],
        actionAlignWeight: 0.0,
      );
    }

    // Episode fitness summary
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

  // ------------- Main run loop -------------
  @override
  Future<void> run({
    required int iters,
    required eng.GameEngine env,
    required FeatureExtractorRays fe,
    required PolicyNetwork policy,
    required RunningNorm? norm,
    required int planHold,              // unused here
    required double tempIntent,         // we use cfg.tempIntent for probe; trainer passes something
    required double gamma,              // unused (no returns)
    required double lr,                 // Trainer's LR; used inside policy.updateFromEpisode
    required double intentAlignWeight,  // we still accept these, but the episode sets intentMode
    required double intentPgWeight,
    required double actionAlignWeight,
    required bool gateVerbose,
    required int seed,
  }) async {
    final rnd = math.Random(seed ^ 0x51A1);
    if (gateVerbose) {
      print('[CUR/padalign] start iters=$iters batch=${cfg.batch} '
          'steps=${cfg.stepsMin}..${cfg.stepsMax} bandFrac=${cfg.bandFrac}');
    }

    for (int it = 0; it < iters; it++) {
      for (int b = 0; b < cfg.batch; b++) {
        final stepsTarget = (cfg.stepsMin == cfg.stepsMax)
            ? cfg.stepsMin
            : (cfg.stepsMin + rnd.nextInt(math.max(1, cfg.stepsMax - cfg.stepsMin + 1)));

        _runEpisode(
          env: env,
          fe: fe,
          policy: policy,
          norm: norm,
          rnd: rnd,
          stepsTarget: stepsTarget,
          bandFrac: cfg.bandFrac,
          tempForProbe: cfg.tempIntent,
        );
      }

      if (((it + 1) % cfg.printEvery) == 0) {
        _printFitnessLine(it + 1);
      }
    }
  }
}
