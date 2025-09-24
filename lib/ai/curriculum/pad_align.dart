// lib/ai/curriculum/pad_align.dart
//
// Micro-curriculum: "PadAlign"
// Trains the intent head to pick goLeft/goRight purely from pad offset sign,
// over very short rollouts with random terrain + spawn near the pad.
// Prints a fitness line with an accuracy sparkline whose blocks explicitly
// map 0–100% accuracy to height (with optional gamma compression).

import 'dart:math' as math;

import '../../engine/types.dart' as et;
import '../../engine/game_engine.dart' as eng;
import '../../engine/raycast.dart' as rc;

import '../agent.dart'; // FeatureExtractorRays, PolicyNetwork, RunningNorm, kIntentNames
import 'core.dart';

class PadAlignCfg {
  final int batch;            // episodes per iter
  final int iters;            // total iterations
  final int minSteps;         // min steps per micro episode
  final int maxSteps;         // max steps per micro episode
  final double bandFrac;      // horizontal spawn band around pad center (fraction of W)
  final double minOffsetPx;   // ensure at least this |dx| so label is unambiguous
  final bool balanceLR;       // enforce ~50/50 left/right spawns
  final bool verboseEveryFit; // print fitness every N iters

  // Sparkline/fitness print config
  final int sparkWidth;       // number of blocks in the sparkline
  final int fitEvery;         // print every N iterations
  final int probeWindow;      // moving window size for fitness stats
  final double sparkGamma;    // gamma (<1 expands mid-range differences)

  const PadAlignCfg({
    this.batch = 32,
    this.iters = 1200,
    this.minSteps = 6,
    this.maxSteps = 12,
    this.bandFrac = 0.15,
    this.minOffsetPx = 20.0,
    this.balanceLR = true,
    this.verboseEveryFit = true,
    this.sparkWidth = 12,
    this.fitEvery = 25,
    this.probeWindow = 400,   // collects recent decisions for fitness
    this.sparkGamma = 0.65,   // 0.5..0.8 is good; <1 highlights 50–85% region
  });

  PadAlignCfg copyWith({
    int? batch,
    int? iters,
    int? minSteps,
    int? maxSteps,
    double? bandFrac,
    double? minOffsetPx,
    bool? balanceLR,
    bool? verboseEveryFit,
    int? sparkWidth,
    int? fitEvery,
    int? probeWindow,
    double? sparkGamma,
  }) => PadAlignCfg(
    batch: batch ?? this.batch,
    iters: iters ?? this.iters,
    minSteps: minSteps ?? this.minSteps,
    maxSteps: maxSteps ?? this.maxSteps,
    bandFrac: bandFrac ?? this.bandFrac,
    minOffsetPx: minOffsetPx ?? this.minOffsetPx,
    balanceLR: balanceLR ?? this.balanceLR,
    verboseEveryFit: verboseEveryFit ?? this.verboseEveryFit,
    sparkWidth: sparkWidth ?? this.sparkWidth,
    fitEvery: fitEvery ?? this.fitEvery,
    probeWindow: probeWindow ?? this.probeWindow,
    sparkGamma: sparkGamma ?? this.sparkGamma,
  );
}

class PadAlignCurriculum extends Curriculum {
  @override
  String get key => 'padalign';

  PadAlignCfg cfg = const PadAlignCfg();

  @override
  Curriculum configure(Map<String, String?> kv, Set<String> flags) {
    final cli = CliView(kv, flags);
    cfg = cfg.copyWith(
      batch: cli.getInt('padalign_batch', def: 32),
      iters: cli.getInt('padalign_iters', def: 1400),
      minSteps: cli.getInt('padalign_min_steps', def: 6),
      maxSteps: cli.getInt('padalign_max_steps', def: 12),
      bandFrac: cli.getDouble('padalign_band_frac', def: 0.15),
      minOffsetPx: cli.getDouble('padalign_min_off', def: 20.0),
      balanceLR: cli.getFlag('padalign_balanced', def: true),
      sparkWidth: cli.getInt('padalign_spark_w', def: 12),
      fitEvery: cli.getInt('padalign_fit_every', def: 25),
      probeWindow: cli.getInt('padalign_probe_win', def: 400),
      sparkGamma: cli.getDouble('padalign_gamma', def: 0.65),
    );
    return this;
  }

  // ---------- Sparkline (explicit 0–100% accuracy mapping + gamma) ----------
  static const _bars = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']; // 9 levels

  String _sparklineAcc(List<double> accHistory,
      {int width = 12, double gamma = 0.65}) {
    if (accHistory.isEmpty || width <= 0) return '⎡${' ' * math.max(1, width)}⎤';

    // Take the *most recent* history and bin to fixed width.
    final n = accHistory.length;
    final start = 0;
    final end = n;
    final perBin = (end - start) / width;

    final buf = StringBuffer('⎡');
    for (int i = 0; i < width; i++) {
      final lo = (start + i * perBin).floor();
      final hi = (start + (i + 1) * perBin).floor();
      final l = lo.clamp(0, n - 1);
      final h = math.max(l + 1, hi.clamp(1, n));

      double sum = 0.0;
      for (int k = l; k < h; k++) sum += accHistory[k];
      final mean = sum / (h - l);         // mean accuracy in [0,1]

      // Gamma map accuracy → emphasize mid-range (explicit 0..100% mapping)
      final y = math.pow(mean.clamp(0.0, 1.0), gamma).toDouble(); // still 0..1
      final level = (y * 8.0).clamp(0.0, 8.0).round();            // 0..8
      buf.write(_bars[level]);
    }
    buf.write('⎤');
    return buf.toString();
  }

  // ---------- Spawn helper (tiny, local, near-pad) ----------
  void _spawnNearPad(eng.GameEngine env, math.Random r) {
    final padCx = env.terrain.padCenter.toDouble();
    final W = env.cfg.worldW.toDouble();
    final band = (cfg.bandFrac * W).clamp(1.0, W * 0.49);

    // If balancing is enabled, force left/right alternation (roughly).
    double dx;
    if (cfg.balanceLR) {
      final wantLeft = r.nextBool();
      if (wantLeft) {
        dx = - (cfg.minOffsetPx + r.nextDouble() * (band - cfg.minOffsetPx));
      } else {
        dx =   (cfg.minOffsetPx + r.nextDouble() * (band - cfg.minOffsetPx));
      }
    } else {
      // Free sample but ensure |dx| >= minOffsetPx
      final raw = (r.nextDouble() * 2 - 1) * band;
      if (raw.abs() < cfg.minOffsetPx) {
        dx = (raw.isNegative ? -1.0 : 1.0) * cfg.minOffsetPx;
      } else {
        dx = raw;
      }
    }

    final x = (padCx + dx).clamp(10.0, W - 10.0);
    final gy = env.terrain.heightAt(x);

    // Small, controlled randomization on height/speeds; no rotation.
    final h = 140.0 + 80.0 * r.nextDouble();
    final vy = 8.0 + 6.0 * r.nextDouble();
    final vx = (r.nextDouble() * 6.0 - 3.0);

    env.lander
      ..pos.x = x
      ..pos.y = (gy - h).clamp(0.0, env.cfg.worldH - 10.0)
      ..vel.x = vx
      ..vel.y = vy
      ..angle = 0.0
      ..fuel = env.cfg.t.maxFuel;
  }

  // Label from dx sign
  int _labelFor(eng.GameEngine env) {
    final padCx = env.terrain.padCenter.toDouble();
    final dx = padCx - env.lander.pos.x.toDouble();
    return dx >= 0 ? intentToIndex(Intent.goRight) : intentToIndex(Intent.goLeft);
  }

  // Hard supervision loss for intent head (cross-entropy on intent logits)
  void _superviseIntent({
    required PolicyNetwork policy,
    required List<ForwardCache> caches,
    required List<int> choices,
    required List<int> labels,
    required double lr,
    double weight = 1.0,
  }) {
    // We store the forward caches; call policy.updateFromEpisode with intentMode=true
    final tn = caches.length;
    final returns = List<double>.filled(tn, 0.0); // not used for pure SL
    policy.updateFromEpisode(
      decisionCaches: caches,
      intentChoices: choices,          // argmax placeholders; not used for SL alignment
      decisionReturns: returns,
      alignLabels: labels,             // <-- the supervised labels (L/R)
      alignWeight: weight,             // scale of supervised cross-entropy
      intentPgWeight: 0.0,             // no PG here
      lr: lr,
      entropyBeta: 0.0,
      valueBeta: 0.0,
      huberDelta: 1.0,
      intentMode: true,                // train intent head
      // no action-head supervision in this micro-curriculum
      actionCaches: const [],
      actionTurnTargets: const [],
      actionThrustTargets: const [],
      actionAlignWeight: 0.0,
    );
  }

  // --- Fitness probe history ---
  final _accHistory = <double>[]; // per-episode accuracy (0..1)
  final _dxDeltaPerStep = <double>[]; // mean |dx| change per step (progress sign)
  int _nL = 0, _nR = 0, _okL = 0, _okR = 0;

  void _pushAcc(double v) {
    _accHistory.add(v.clamp(0.0, 1.0));
    if (_accHistory.length > cfg.probeWindow) {
      _accHistory.removeAt(0);
    }
  }

  void _pushDxDelta(double d) {
    _dxDeltaPerStep.add(d);
    if (_dxDeltaPerStep.length > cfg.probeWindow) {
      _dxDeltaPerStep.removeAt(0);
    }
  }

  String _fmtPct(double x) => '${(100.0 * x).toStringAsFixed(1)}%';

  String _fitnessLine({
    required int iter,
  }) {
    double meanAcc = 0.0;
    for (final a in _accHistory) meanAcc += a;
    final acc = _accHistory.isEmpty ? 0.0 : (meanAcc / _accHistory.length);

    final lTot = _nL == 0 ? 1 : _nL;
    final rTot = _nR == 0 ? 1 : _nR;
    final lOk = _okL / lTot.toDouble();
    final rOk = _okR / rTot.toDouble();

    double dxm = 0.0;
    for (final d in _dxDeltaPerStep) dxm += d;
    dxm = _dxDeltaPerStep.isEmpty ? 0.0 : (dxm / _dxDeltaPerStep.length);

    final biasL = _nL + _nR == 0 ? 0.0 : (_nL / (_nL + _nR).toDouble());

    final spark = _sparklineAcc(_accHistory, width: cfg.sparkWidth, gamma: cfg.sparkGamma);
    return '[PADALIGN/FIT] it=$iter  acc=${_fmtPct(acc)}  '
        'L_ok=${_fmtPct(lOk)}  R_ok=${_fmtPct(rOk)}  '
        'bias(L)=${_fmtPct(biasL)}  $spark  Δ|dx|/step=${dxm.toStringAsFixed(1)}';
  }

  @override
  Future<void> run({
    required int iters,
    required eng.GameEngine env,
    required FeatureExtractorRays fe,
    required PolicyNetwork policy,
    required RunningNorm? norm,
    required int planHold,              // unused here (we always force a short window)
    required double tempIntent,         // unused in hard SL
    required double gamma,              // RL gamma, unused
    required double lr,
    required double intentAlignWeight,
    required double intentPgWeight,
    required double actionAlignWeight,
    required bool gateVerbose,
    required int seed,
  }) async {
    final rnd = math.Random(seed ^ 0xBEE5);

    if (gateVerbose) {
      final minS = cfg.minSteps;
      final maxS = cfg.maxSteps;
      final band = cfg.bandFrac;
      final minOff = cfg.minOffsetPx;
      final bal = cfg.balanceLR ? 'Y' : 'N';
      print('[CUR/padalign] start iters=${cfg.iters} batch=${cfg.batch} '
          'steps=$minS..$maxS bandFrac=$band minOff=$minOff balanced=$bal');
      // Initial “dry run” fitness read:
      print(_fitnessLine(iter: 0));
    }

    final caches = <ForwardCache>[];
    final choices = <int>[];
    final labels = <int>[];

    for (int it = 1; it <= cfg.iters; it++) {
      caches.clear(); choices.clear(); labels.clear();

      for (int b = 0; b < cfg.batch; b++) {
        // Reset terrain & rays each episode for variety
        env.reset(seed: rnd.nextInt(1 << 30));
        _spawnNearPad(env, rnd);

        // Build tiny rollout (we only care about first intent decision)
        final stepsThis = cfg.minSteps + rnd.nextInt(math.max(1, cfg.maxSteps - cfg.minSteps + 1));
        final trueLabel = _labelFor(env); // target: goRight if pad is right, else goLeft
        if (trueLabel == intentToIndex(Intent.goLeft)) _nL++; else _nR++;

        // Extract features at t=0 (or at a random early step to add slight variation)
        // Step a few frames without supervision to jitter rays/FE a bit
        final jitter = rnd.nextInt(math.max(1, stepsThis ~/ 2));
        for (int j = 0; j < jitter; j++) {
          env.step(1 / 60.0, const et.ControlInput(thrust: false, left: false, right: false));
        }

        var x = fe.extract(
          lander: env.lander,
          terrain: env.terrain,
          worldW: env.cfg.worldW,
          worldH: env.cfg.worldH,
          rays: env.rays,
        );
        if (norm != null) {
          norm.observe(x);
          x = norm.normalize(x, update: false);
        }

        // Forward (intent logits)
        final (idxGreedy, _p, cache) = policy.actIntentGreedy(x);

        // Tally accuracy
        final ok = (idxGreedy == trueLabel);
        _pushAcc(ok ? 1.0 : 0.0);
        if (trueLabel == intentToIndex(Intent.goLeft)) {
          if (ok) _okL++;
        } else {
          if (ok) _okR++;
        }

        caches.add(cache);
        choices.add(idxGreedy);
        labels.add(trueLabel);

        // Optional: quick progress metric (how |dx| changes if we briefly follow intent)
        final padCx = env.terrain.padCenter.toDouble();
        final dx0 = (env.lander.pos.x.toDouble() - padCx).abs();
        // Step a couple frames with the teacher *for that intent* to see dx trend
        final wantLeft = (trueLabel == intentToIndex(Intent.goLeft));
        final u = wantLeft
            ? const et.ControlInput(thrust: false, left: true, right: false)
            : const et.ControlInput(thrust: false, left: false, right: true);
        int s = 0;
        while (s < 3) {
          env.step(1 / 60.0, u);
          s++;
        }
        final dx1 = (env.lander.pos.x.toDouble() - padCx).abs();
        _pushDxDelta((dx0 - dx1) / math.max(1, s));

        // End this micro-episode right away (we don’t need more frames)
      }

      // One supervised update per iteration
      _superviseIntent(
        policy: policy,
        caches: caches,
        choices: choices,
        labels: labels,
        lr: lr,
        weight: math.max(0.1, intentAlignWeight), // ensure it has a non-trivial effect
      );

      // Periodic fitness print
      if (gateVerbose && (it % cfg.fitEvery == 0 || it == cfg.iters)) {
        print(_fitnessLine(iter: it));
      }
    }
  }
}
