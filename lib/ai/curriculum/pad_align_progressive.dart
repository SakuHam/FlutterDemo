// lib/ai/curriculum/pad_align_progressive.dart
//
// Progressive pad-align curriculum ("pad_align_progressive") with
// automatic LR polarity detection/flip to robustly move toward the pad.
//
// Stages 0..3: same idea as before (short → longer drills).
// Probe uses absolute dx shrink per second:
//   progress/sec = (|dx_before| - |dx_after|) / (steps * dt)
// Positive means we moved closer to pad; negative means we drifted away.

import 'dart:math' as math;

import '../../engine/types.dart' as et;
import '../../engine/game_engine.dart' as eng;

import '../agent.dart'; // FeatureExtractorRays, PolicyNetwork, RunningNorm, Intent helpers
import 'core.dart';

/* =============================== Config =================================== */

class PadAlignProgressiveCfg {
  // Outer loop
  final int iters;
  final int batch;

  // Printing & stats
  final int fitEvery;
  final int probeWindow;   // rolling window for stats
  final int sparkWidth;
  final double sparkGamma;
  final bool verbose;

  // Promotion gates (rolling-window targets)
  final double promoAcc;   // mean accuracy to promote stage
  final double promoDx;    // target progress px/s (>= this) to promote
  final int maxStage;      // clamp max stage [0..3]

  // Stage 0..3 base parameters
  final int s0MinSteps, s0MaxSteps;
  final double s0BandFrac;
  final double s0MinOffPx;

  final int s1MinSteps, s1MaxSteps;
  final double s1BandFrac;
  final double s1MinOffPx;

  final int s2MinSteps, s2MaxSteps;
  final double s2BandFrac;
  final double s2MinOffPx;

  final int s3MinSteps, s3MaxSteps;
  final double s3BandFrac;
  final double s3MinOffPx;

  // Behavior toggles
  final bool balanceLR;           // ~50/50 left/right spawns
  final bool allowActionAlign;    // optional: supervised hints at stage 3

  // Probe execution
  final int probeSteps;           // how many steps to apply the probed intent
  final double probeDt;           // per-step dt during probe (usually 1/60)

  const PadAlignProgressiveCfg({
    this.iters = 1600,
    this.batch = 32,
    this.fitEvery = 25,
    this.probeWindow = 500,
    this.sparkWidth = 12,
    this.sparkGamma = 0.65,
    this.verbose = true,
    this.promoAcc = 0.84,
    this.promoDx = 2.0, // px/s; conservative default; promote when >= this
    this.maxStage = 3,

    this.s0MinSteps = 6,  this.s0MaxSteps = 12,  this.s0BandFrac = 0.15, this.s0MinOffPx = 20.0,
    this.s1MinSteps = 10, this.s1MaxSteps = 18,  this.s1BandFrac = 0.22, this.s1MinOffPx = 14.0,
    this.s2MinSteps = 16, this.s2MaxSteps = 26,  this.s2BandFrac = 0.28, this.s2MinOffPx = 10.0,
    this.s3MinSteps = 26, this.s3MaxSteps = 46,  this.s3BandFrac = 0.30, this.s3MinOffPx = 8.0,

    this.balanceLR = true,
    this.allowActionAlign = false,

    this.probeSteps = 6,
    this.probeDt = 1.0 / 60.0,
  });

  PadAlignProgressiveCfg copyWith({
    int? iters, int? batch,
    int? fitEvery, int? probeWindow, int? sparkWidth,
    double? sparkGamma, bool? verbose,
    double? promoAcc, double? promoDx, int? maxStage,
    int? s0MinSteps, int? s0MaxSteps, double? s0BandFrac, double? s0MinOffPx,
    int? s1MinSteps, int? s1MaxSteps, double? s1BandFrac, double? s1MinOffPx,
    int? s2MinSteps, int? s2MaxSteps, double? s2BandFrac, double? s2MinOffPx,
    int? s3MinSteps, int? s3MaxSteps, double? s3BandFrac, double? s3MinOffPx,
    bool? balanceLR, bool? allowActionAlign,
    int? probeSteps, double? probeDt,
  }) => PadAlignProgressiveCfg(
    iters: iters ?? this.iters,
    batch: batch ?? this.batch,
    fitEvery: fitEvery ?? this.fitEvery,
    probeWindow: probeWindow ?? this.probeWindow,
    sparkWidth: sparkWidth ?? this.sparkWidth,
    sparkGamma: sparkGamma ?? this.sparkGamma,
    verbose: verbose ?? this.verbose,
    promoAcc: promoAcc ?? this.promoAcc,
    promoDx: promoDx ?? this.promoDx,
    maxStage: maxStage ?? this.maxStage,
    s0MinSteps: s0MinSteps ?? this.s0MinSteps,
    s0MaxSteps: s0MaxSteps ?? this.s0MaxSteps,
    s0BandFrac: s0BandFrac ?? this.s0BandFrac,
    s0MinOffPx: s0MinOffPx ?? this.s0MinOffPx,
    s1MinSteps: s1MinSteps ?? this.s1MinSteps,
    s1MaxSteps: s1MaxSteps ?? this.s1MaxSteps,
    s1BandFrac: s1BandFrac ?? this.s1BandFrac,
    s1MinOffPx: s1MinOffPx ?? this.s1MinOffPx,
    s2MinSteps: s2MinSteps ?? this.s2MinSteps,
    s2MaxSteps: s2MaxSteps ?? this.s2MaxSteps,
    s2BandFrac: s2BandFrac ?? this.s2BandFrac,
    s2MinOffPx: s2MinOffPx ?? this.s2MinOffPx,
    s3MinSteps: s3MinSteps ?? this.s3MinSteps,
    s3MaxSteps: s3MaxSteps ?? this.s3MaxSteps,
    s3BandFrac: s3BandFrac ?? this.s3BandFrac,
    s3MinOffPx: s3MinOffPx ?? this.s3MinOffPx,
    balanceLR: balanceLR ?? this.balanceLR,
    allowActionAlign: allowActionAlign ?? this.allowActionAlign,
    probeSteps: probeSteps ?? this.probeSteps,
    probeDt: probeDt ?? this.probeDt,
  );
}

/* =============================== Curriculum =============================== */

class PadAlignProgressiveCurriculum extends Curriculum {
  @override
  String get key => 'padalign_progressive';

  PadAlignProgressiveCfg cfg = const PadAlignProgressiveCfg();

  // ---- NEW: LR polarity auto-detection state ----
  bool _invertLR = false;       // if true, swap goLeft <-> goRight
  int _polVotes = 0;
  int _polAgree = 0;

  @override
  Curriculum configure(Map<String, String?> kv, Set<String> flags) {
    final cli = CliView(kv, flags);
    cfg = cfg.copyWith(
      iters: cli.getInt('padprog_iters', def: 1600),
      batch: cli.getInt('padprog_batch', def: 32),
      fitEvery: cli.getInt('padprog_fit_every', def: 25),
      probeWindow: cli.getInt('padprog_probe_win', def: 500),
      sparkWidth: cli.getInt('padprog_spark_w', def: 12),
      sparkGamma: cli.getDouble('padprog_gamma', def: 0.65),
      verbose: cli.getFlag('padprog_verbose', def: true),
      promoAcc: cli.getDouble('padprog_promo_acc', def: 0.84),
      promoDx: cli.getDouble('padprog_promo_dx', def: 2.0),
      maxStage: cli.getInt('padprog_max_stage', def: 3),
      allowActionAlign: cli.getFlag('padprog_action_align', def: false),
      probeSteps: cli.getInt('padprog_probe_steps', def: 6),
      probeDt: cli.getDouble('padprog_probe_dt', def: 1.0 / 60.0),
    );
    return this;
  }

  /* --------------------------- UI helpers / stats -------------------------- */

  static const _bars = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

  String _sparkline(List<double> accHistory, {int width = 12, double gamma = 0.65}) {
    if (accHistory.isEmpty || width <= 0) return '⎡${' ' * math.max(1, width)}⎤';
    final n = accHistory.length;
    final perBin = n / width;
    final buf = StringBuffer('⎡');
    for (int i = 0; i < width; i++) {
      final lo = (i * perBin).floor().clamp(0, n - 1);
      final hi = ((i + 1) * perBin).floor().clamp(1, n);
      double s = 0.0;
      for (int k = lo; k < math.max(lo + 1, hi); k++) s += accHistory[k];
      final m = s / (math.max(lo + 1, hi) - lo);
      final y = math.pow(m.clamp(0.0, 1.0), gamma).toDouble();
      buf.write(_bars[(y * 8.0).clamp(0.0, 8.0).round()]);
    }
    buf.write('⎤');
    return buf.toString();
  }

  final _accHistory = <double>[];       // per-episode 0/1 accuracy
  final _dxPerSec = <double>[];         // progress (|dx| shrink)/sec

  void _pushAcc(double v, int win) { _accHistory.add(v.clamp(0.0, 1.0)); if (_accHistory.length > win) _accHistory.removeAt(0); }
  void _pushDx(double v, int win) { _dxPerSec.add(v); if (_dxPerSec.length > win) _dxPerSec.removeAt(0); }

  String _fmtPct(double x) => '${(100.0 * x).toStringAsFixed(1)}%';

  String _fitnessLine({
    required int iter,
    required int stage,
    required int nL, required int nR, required int okL, required int okR,
    required double gateDx,
  }) {
    double acc = 0.0; for (final a in _accHistory) acc += a;
    acc = _accHistory.isEmpty ? 0.0 : acc / _accHistory.length;

    double dxm = 0.0; for (final d in _dxPerSec) dxm += d;
    dxm = _dxPerSec.isEmpty ? 0.0 : dxm / _dxPerSec.length;

    final lOk = nL == 0 ? 0.0 : okL / nL.toDouble();
    final rOk = nR == 0 ? 0.0 : okR / nR.toDouble();
    final biasL = (nL + nR) == 0 ? 0.0 : nL / (nL + nR).toDouble();

    final spark = _sparkline(_accHistory, width: cfg.sparkWidth, gamma: cfg.sparkGamma);
    return '[PADPROG/FIT] it=$iter st=$stage acc=${_fmtPct(acc)} '
        'L_ok=${_fmtPct(lOk)} R_ok=${_fmtPct(rOk)} bias(L)=${_fmtPct(biasL)} '
        '$spark  Δ|dx|/sec=${dxm.toStringAsFixed(2)}  gateDx=${gateDx.toStringAsFixed(2)}';
  }

  /* ------------------------------ Spawning -------------------------------- */

  void _spawnNearPad({
    required int stage,
    required eng.GameEngine env,
    required math.Random r,
    required bool balanceLR,
    required double bandFrac,
    required double minOffPx,
    double hLo = 120.0,
    double hHi = 220.0,
    double vyLo = 6.0,
    double vyHi = 10.0,
    double vxAbs = 3.0,
  }) {
    final padCx = env.terrain.padCenter.toDouble();
    final W = env.cfg.worldW.toDouble();
    final band = (bandFrac * W).clamp(1.0, W * 0.49);

    double dx;
    if (balanceLR) {
      final left = r.nextBool();
      final span = band - minOffPx;
      final mag = minOffPx + r.nextDouble() * math.max(1e-6, span);
      dx = left ? -mag : mag; // dx = (x - padCx); we add to padCx below
    } else {
      final raw = (r.nextDouble() * 2 - 1) * band;
      dx = raw.abs() < minOffPx ? (raw.isNegative ? -minOffPx : minOffPx) : raw;
    }

    final x = (padCx + dx).clamp(10.0, W - 10.0);
    final gy = env.terrain.heightAt(x);

    // Stage-based altitude & velocities
    double h = hLo + (hHi - hLo) * r.nextDouble();
    double vy = vyLo + (vyHi - vyLo) * r.nextDouble();
    double vx = (r.nextDouble() * 2 - 1) * vxAbs;

    if (stage >= 2) {
      h += 60.0 + 80.0 * r.nextDouble();
      vy += 2.0 * r.nextDouble();
      vx *= 1.2;
    }
    if (stage >= 3) {
      h += 60.0;
      vy += 1.5;
      vx *= 1.2;
    }

    env.lander
      ..pos.x = x
      ..pos.y = (gy - h).clamp(0.0, env.cfg.worldH - 10.0)
      ..vel.x = vx
      ..vel.y = vy
      ..angle = 0.0
      ..fuel = env.cfg.t.maxFuel;
  }

  // Ground-truth label from sign(dx) at a given moment, respecting _invertLR.
  int _labelFor(eng.GameEngine env) {
    final padCx = env.terrain.padCenter.toDouble();
    final dx = padCx - env.lander.pos.x.toDouble(); // + if pad is to the RIGHT
    final rightLabel = intentToIndex(Intent.goRight);
    final leftLabel  = intentToIndex(Intent.goLeft);
    final lbl = (dx >= 0) ? rightLabel : leftLabel;
    return _invertLR ? ((lbl == rightLabel) ? leftLabel : rightLabel) : lbl;
  }

  // Simple teacher for stage 3 (optional supervised hints)
  (int turn, bool thrust) _teacherS3(eng.GameEngine env) {
    final padCx = env.terrain.padCenter.toDouble();
    final x = env.lander.pos.x.toDouble();
    final vx = env.lander.vel.x.toDouble();
    final vy = env.lander.vel.y.toDouble();
    final dx = padCx - x;

    int turn;
    final wantRight = (dx > 0);
    if (dx.abs() < 4.0 && vx.abs() < 1.0) {
      turn = 0;
    } else {
      turn = wantRight ? 1 : -1;
    }

    final gy = env.terrain.heightAt(x);
    final h = (gy - env.lander.pos.y).toDouble().clamp(0.0, 1e9);
    final needFlare = (h < 60.0 && vy < 10.0) || (vy > 9.0);
    final bool thrust = needFlare;
    return (turn, thrust);
  }

  /* ------------------------- Polarity auto-detect -------------------------- */

  void _polarityVote({
    required double dxBefore,
    required double dxAfter,
    required int intentIdx,
  }) {
    // Only vote when we *intended* to move toward the pad in label space.
    final padIsRight = dxBefore >= 0.0;
    final triedLeft  = (intentIdx == intentToIndex(Intent.goLeft));
    final triedRight = (intentIdx == intentToIndex(Intent.goRight));
    final attemptedToward =
        (padIsRight && triedRight) || (!padIsRight && triedLeft);
    if (!attemptedToward) return;

    final improved = (dxAfter.abs() < dxBefore.abs()); // moved closer?
    _polVotes++;
    final expectImprove = !_invertLR; // if polarity is correct, improvement should be true
    final agrees = (improved == expectImprove);
    if (agrees) _polAgree++;

    if (_polVotes >= 60) {
      final agreeFrac = _polAgree / _polVotes;
      if (agreeFrac < 0.20) {
        _invertLR = !_invertLR;
        print('[PADPROG/POLARITY] Flipping LR mapping. invertLR=${_invertLR ? "ON" : "OFF"} '
            '(agree=${(100*agreeFrac).toStringAsFixed(1)}% over $_polVotes votes)');
      } else {
        print('[PADPROG/POLARITY] Keeping mapping. invertLR=${_invertLR ? "ON" : "OFF"} '
            '(agree=${(100*agreeFrac).toStringAsFixed(1)}% over $_polVotes votes)');
      }
      _polVotes = 0; _polAgree = 0;
    }
  }

  /* --------------------- Small supervised update helpers ------------------- */

  void _superviseIntent({
    required PolicyNetwork policy,
    required List<ForwardCache> caches,
    required List<int> choices,
    required List<int> labels,
    required double lr,
    double weight = 1.0,
  }) {
    final tn = caches.length;
    final returns = List<double>.filled(tn, 0.0);
    policy.updateFromEpisode(
      decisionCaches: caches,
      intentChoices: choices,
      decisionReturns: returns,
      alignLabels: labels,
      alignWeight: math.max(0.1, weight),
      intentPgWeight: 0.0,
      lr: lr,
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

  void _superviseActionsStage3({
    required PolicyNetwork policy,
    required List<ForwardCache> actionCaches,
    required List<int> turnTargets,
    required List<bool> thrustTargets,
    required double lr,
    required double actionAlignWeight,
  }) {
    if (actionCaches.isEmpty) return;
    final returns = List<double>.filled(actionCaches.length, 0.0);
    policy.updateFromEpisode(
      decisionCaches: const [],
      intentChoices: const [],
      decisionReturns: returns,  // not used here
      alignLabels: const [],
      alignWeight: 0.0,
      intentPgWeight: 0.0,
      lr: lr,
      entropyBeta: 0.0,
      valueBeta: 0.0,
      huberDelta: 1.0,
      intentMode: false,
      actionCaches: actionCaches,
      actionTurnTargets: turnTargets,
      actionThrustTargets: thrustTargets,
      actionAlignWeight: actionAlignWeight,
    );
  }

  /* --------------------------------- Run ---------------------------------- */

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
    final rnd = math.Random(seed ^ 0xA11E);
    final wasTrain = policy.trunk.trainMode;
    policy.trunk.trainMode = false; // freeze trunk during micro-drills

    int stage = 0.clamp(0, cfg.maxStage);
    if (cfg.verbose || gateVerbose) {
      print('[CUR/pad_align_progressive] start iters=${cfg.iters} '
          'batch=${cfg.batch} maxStage=${cfg.maxStage} promoAcc=${cfg.promoAcc} promoDx=${cfg.promoDx} '
          'actionAlign=${cfg.allowActionAlign ? "N" : "N"}');
      // initial line
      print(_fitnessLine(iter: 0, stage: stage, nL: 0, nR: 0, okL: 0, okR: 0, gateDx: cfg.promoDx));
    }

    var nL = 0, nR = 0, okL = 0, okR = 0;

    int _minSteps(int st) => switch (st) { 0 => cfg.s0MinSteps, 1 => cfg.s1MinSteps, 2 => cfg.s2MinSteps, _ => cfg.s3MinSteps };
    int _maxSteps(int st) => switch (st) { 0 => cfg.s0MaxSteps, 1 => cfg.s1MaxSteps, 2 => cfg.s2MaxSteps, _ => cfg.s3MaxSteps };
    double _band(int st)   => switch (st) { 0 => cfg.s0BandFrac, 1 => cfg.s1BandFrac, 2 => cfg.s2BandFrac, _ => cfg.s3BandFrac };
    double _minOff(int st) => switch (st) { 0 => cfg.s0MinOffPx, 1 => cfg.s1MinOffPx, 2 => cfg.s2MinOffPx, _ => cfg.s3MinOffPx };

    final intentCaches = <ForwardCache>[];
    final intentChoices = <int>[];
    final intentLabels = <int>[];

    final actionCaches = <ForwardCache>[];
    final turnTargets = <int>[];
    final thrustTargets = <bool>[];

    final dtProbe = cfg.probeDt;

    for (int it = 1; it <= cfg.iters; it++) {
      intentCaches.clear(); intentChoices.clear(); intentLabels.clear();
      actionCaches.clear(); turnTargets.clear(); thrustTargets.clear();

      for (int b = 0; b < cfg.batch; b++) {
        env.reset(seed: rnd.nextInt(1 << 30));

        _spawnNearPad(
          stage: stage,
          env: env,
          r: rnd,
          balanceLR: cfg.balanceLR,
          bandFrac: _band(stage),
          minOffPx: _minOff(stage),
          hLo: 120.0, hHi: 220.0,
          vyLo: 6.0, vyHi: 10.0,
          vxAbs: 3.0,
        );

        // Let it sit a few frames to settle rays
        final settle = 2 + rnd.nextInt(3);
        for (int j = 0; j < settle; j++) {
          env.step(1 / 60.0, const et.ControlInput(thrust: false, left: false, right: false));
        }

        // Features + (optional) normalization
        var x = fe.extract(
          lander: env.lander,
          terrain: env.terrain,
          worldW: env.cfg.worldW,
          worldH: env.cfg.worldH,
          rays: env.rays,
        );
        if (norm != null) { norm.observe(x); x = norm.normalize(x, update: false); }

        // Intent forward (greedy) + supervised label (respecting _invertLR)
        final (idxGreedy, _p, cacheIntent) = policy.actIntentGreedy(x);

        int label = _labelFor(env);

        // ---- Polarity-respecting execution intent for the PROBE ----
        int execIdx = idxGreedy;
        if (_invertLR) {
          if (execIdx == intentToIndex(Intent.goLeft))  execIdx = intentToIndex(Intent.goRight);
          else if (execIdx == intentToIndex(Intent.goRight)) execIdx = intentToIndex(Intent.goLeft);
        }

        // accuracy bookkeeping
        final ok = (idxGreedy == label);
        _pushAcc(ok ? 1.0 : 0.0, cfg.probeWindow);
        if (label == intentToIndex(Intent.goLeft)) { nL++; if (ok) okL++; }
        else { nR++; if (ok) okR++; }

        intentCaches.add(cacheIntent);
        intentChoices.add(idxGreedy);
        intentLabels.add(label);

        // -------- Progress probe: measure |dx| shrink/sec while applying execIdx --------
        final padCx = env.terrain.padCenter.toDouble();
        final dx0 = padCx - env.lander.pos.x.toDouble();

        // Build a very light control matching execIdx (no thrust during probe).
        final execIntent = Intent.values[execIdx];
        final u = et.ControlInput(
          thrust: false,
          left:  execIntent == Intent.goLeft,
          right: execIntent == Intent.goRight,
        );

        final stepsProbe = math.max(1, cfg.probeSteps);
        for (int k = 0; k < stepsProbe; k++) {
          env.step(dtProbe, u);
        }

        final dx1 = padCx - env.lander.pos.x.toDouble();
        final shrink = (dx0.abs() - dx1.abs()); // + if closer
        final progPerSec = shrink / (stepsProbe * dtProbe); // px/s
        _pushDx(progPerSec.clamp(-100.0, 100.0), cfg.probeWindow);

        // Vote for polarity (only when “exec toward pad” should help)
        _polarityVote(dxBefore: dx0, dxAfter: dx1, intentIdx: execIdx);

        // If stage 3 + allow hints: add a few supervised action frames
        if (stage >= 3 && cfg.allowActionAlign) {
          final extra = 6 + rnd.nextInt(8);
          for (int k = 0; k < extra; k++) {
            final (turnT, thrB) = _teacherS3(env);

            var x2 = fe.extract(
              lander: env.lander,
              terrain: env.terrain,
              worldW: env.cfg.worldW,
              worldH: env.cfg.worldH,
              rays: env.rays,
            );
            if (norm != null) { norm.observe(x2); x2 = norm.normalize(x2, update: false); }

            // Reuse intent forward cache for SL on action heads; we only need the last hidden.
            final (_, __, cacheAction) = policy.actIntentGreedy(x2);
            actionCaches.add(cacheAction);
            // turnT: -1 (left), 0 (neutral), +1 (right) → map to {0,1,2}
            final turnIdx = (turnT < 0) ? 0 : (turnT > 0 ? 2 : 1);
            turnTargets.add(turnIdx);
            thrustTargets.add(thrB);

            // step with teacher to vary state
            env.step(1 / 60.0, et.ControlInput(
              thrust: thrB,
              left: turnT < 0,
              right: turnT > 0,
            ));
          }
        }
      } // batch

      // Intent head supervised update
      _superviseIntent(
        policy: policy,
        caches: intentCaches,
        choices: intentChoices,
        labels: intentLabels,
        lr: lr,
        weight: math.max(0.1, intentAlignWeight),
      );

      // Optional action head alignment at Stage 3
      if (stage >= 3 && cfg.allowActionAlign && actionAlignWeight > 0.0) {
        _superviseActionsStage3(
          policy: policy,
          actionCaches: actionCaches,
          turnTargets: turnTargets,
          thrustTargets: thrustTargets,
          lr: lr,
          actionAlignWeight: actionAlignWeight,
        );
      }

      // Fitness print + promotion
      if (cfg.verbose || gateVerbose) {
        if (it % cfg.fitEvery == 0 || it == cfg.iters) {
          // Average progress/sec in window
          double dxm = 0.0; for (final d in _dxPerSec) dxm += d;
          final progAvg = _dxPerSec.isEmpty ? 0.0 : (dxm / _dxPerSec.length);

          final gateDx = cfg.promoDx; // fixed target; you can make it adaptive if you prefer
          print(_fitnessLine(
            iter: it, stage: stage, nL: nL, nR: nR, okL: okL, okR: okR, gateDx: gateDx,
          ));

          if (stage < cfg.maxStage) {
            double acc = 0.0; for (final a in _accHistory) acc += a;
            final accAvg = _accHistory.isEmpty ? 0.0 : (acc / _accHistory.length);
            if (accAvg >= cfg.promoAcc && progAvg >= gateDx) {
              stage = (stage + 1).clamp(0, cfg.maxStage);
              print('[PADPROG] ↑ promoted to stage=$stage '
                  '(acc=${_fmtPct(accAvg)} Δ|dx|/sec=${progAvg.toStringAsFixed(2)})');
              _accHistory.clear();
              _dxPerSec.clear();
              nL = nR = okL = okR = 0;
            }
          }
        }
      }
    } // iters

    policy.trunk.trainMode = wasTrain;
  }
}
