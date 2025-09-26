// lib/ai/curriculum/pad_align_progressive.dart
//
// Progressive pad-align curriculum ("pad_align_progressive") with a reliable
// micro-probe that forces translation toward the pad during measurement.
// The probe saves/restores the lander state so it does not affect training.
//
// Stages:
//   0: tiny near-pad drills
//   1: longer near-pad
//   2: higher spawn + free drift
//   3: mini-landing flavor + (optional) action head teacher
//
// Progress metric:
//   We measure Δ|dx|/sec by running a short burst of "toward-pad" control
//   with thrust. This avoids the "tilt only → no lateral movement yet" issue
//   that made the old vx-proxy unreliable.
//
// CLI knobs added:
//   --padprog_micro_k=36            // total probe frames (warmup+measured)
//   --padprog_micro_warm_frac=0.25  // fraction of K spent as warmup

import 'dart:math' as math;

import '../../engine/types.dart' as et;
import '../../engine/game_engine.dart' as eng;

import '../agent.dart'; // FeatureExtractorRays, PolicyNetwork, RunningNorm, controllerForIntent, Intent, intentToIndex, indexToIntent
import 'core.dart';

class PadAlignProgressiveCfg {
  // Outer loop
  final int iters;
  final int batch;

  // Printing & stats
  final int fitEvery;
  final int probeWindow;
  final int sparkWidth;
  final double sparkGamma;
  final bool verbose;

  // Promotion gates (rolling-window targets)
  final double promoAcc;   // mean accuracy to promote stage
  final double promoDx;    // mean progress metric (px/sec), >= this
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
  final bool allowActionAlign;    // optional: provide turn/thrust teacher in S3

  // Micro-probe knobs
  final int microK;               // total frames for probe (warm+meas)
  final double microWarmFrac;     // fraction of K spent warming (no measurement)

  const PadAlignProgressiveCfg({
    this.iters = 1600,
    this.batch = 32,
    this.fitEvery = 25,
    this.probeWindow = 500,
    this.sparkWidth = 12,
    this.sparkGamma = 0.65,
    this.verbose = true,
    this.promoAcc = 0.84,
    this.promoDx = 8.0, // px/sec target AFTER using reliable probe
    this.maxStage = 3,
    this.s0MinSteps = 6,  this.s0MaxSteps = 12,  this.s0BandFrac = 0.15, this.s0MinOffPx = 20.0,
    this.s1MinSteps = 10, this.s1MaxSteps = 18,  this.s1BandFrac = 0.22, this.s1MinOffPx = 14.0,
    this.s2MinSteps = 16, this.s2MaxSteps = 26,  this.s2BandFrac = 0.28, this.s2MinOffPx = 10.0,
    this.s3MinSteps = 26, this.s3MaxSteps = 46,  this.s3BandFrac = 0.30, this.s3MinOffPx = 8.0,
    this.balanceLR = true,
    this.allowActionAlign = false,

    this.microK = 36,             // 0.6 sec at 60 Hz
    this.microWarmFrac = 0.25,    // first 25% warmup; last 75% measured
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
    int? microK, double? microWarmFrac,
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
    microK: microK ?? this.microK,
    microWarmFrac: microWarmFrac ?? this.microWarmFrac,
  );
}

class PadAlignProgressiveCurriculum extends Curriculum {
  @override
  String get key => 'padalign_progressive';

  PadAlignProgressiveCfg cfg = const PadAlignProgressiveCfg();

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
      promoDx: cli.getDouble('padprog_promo_dx', def: 8.0), // px/sec gate
      maxStage: cli.getInt('padprog_max_stage', def: 3),
      allowActionAlign: cli.getFlag('padprog_action_align', def: false),

      // NEW probe knobs
      microK: cli.getInt('padprog_micro_k', def: 36),
      microWarmFrac: cli.getDouble('padprog_micro_warm_frac', def: 0.25),
    );
    return this;
  }

  // ---------- Sparkline ----------
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

  // ---------- Rolling stats ----------
  final _accHistory = <double>[];
  final _dxPerSecHistory = <double>[];

  void _pushAcc(double v, int win) {
    _accHistory.add(v.clamp(0.0, 1.0));
    if (_accHistory.length > win) _accHistory.removeAt(0);
  }
  void _pushDxPerSec(double v, int win) {
    _dxPerSecHistory.add(v);
    if (_dxPerSecHistory.length > win) _dxPerSecHistory.removeAt(0);
  }

  String _fmtPct(double x) => '${(100.0 * x).toStringAsFixed(1)}%';

  String _fitnessLine({
    required int iter,
    required int stage,
    required int nL, required int nR, required int okL, required int okR,
  }) {
    double acc = 0.0; for (final a in _accHistory) acc += a;
    acc = _accHistory.isEmpty ? 0.0 : acc / _accHistory.length;

    double dxs = 0.0; for (final d in _dxPerSecHistory) dxs += d;
    dxs = _dxPerSecHistory.isEmpty ? 0.0 : dxs / _dxPerSecHistory.length;

    final lOk = nL == 0 ? 0.0 : okL / nL.toDouble();
    final rOk = nR == 0 ? 0.0 : okR / nR.toDouble();
    final biasL = (nL + nR) == 0 ? 0.0 : nL / (nL + nR).toDouble();

    final spark = _sparkline(_accHistory, width: cfg.sparkWidth, gamma: cfg.sparkGamma);
    return '[PADPROG/FIT] it=$iter st=$stage acc=${_fmtPct(acc)} '
        'L_ok=${_fmtPct(lOk)} R_ok=${_fmtPct(rOk)} bias(L)=${_fmtPct(biasL)} '
        '$spark  Δ|dx|/sec=${dxs.toStringAsFixed(2)}  gateDx=${cfg.promoDx.toStringAsFixed(2)}';
  }

  // ---------- Spawning (stage-aware) ----------
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
      dx = left ? -mag : mag;
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

  // Ground-truth label from sign(dx) at a given moment.
  int _labelFor(eng.GameEngine env) {
    final padCx = env.terrain.padCenter.toDouble();
    final dx = padCx - env.lander.pos.x.toDouble();
    return dx >= 0 ? intentToIndex(Intent.goRight) : intentToIndex(Intent.goLeft);
  }

  // ----- Reliable micro-probe to measure Δ|dx|/sec -----
  // Saves lander state, runs a toward-pad control with thrust for K frames,
  // measures |dx| reduction during the measured window, restores state.
  double _measureProgressDxPerSec({
    required eng.GameEngine env,
    required int K,
    required double warmFrac,
  }) {
    final L = env.lander;
    // Snapshot lander (minimal fields we mutate/need)
    final snap = (
    x: L.pos.x.toDouble(),
    y: L.pos.y.toDouble(),
    vx: L.vel.x.toDouble(),
    vy: L.vel.y.toDouble(),
    ang: L.angle.toDouble(),
    fuel: L.fuel.toDouble(),
    );

    double towardSign() {
      final padCx = env.terrain.padCenter.toDouble();
      final dx = padCx - env.lander.pos.x.toDouble();
      return dx >= 0 ? 1.0 : -1.0; // +1 → need goRight, -1 → goLeft
    }

    et.ControlInput _towardPad() {
      final sign = towardSign();
      final intent = (sign >= 0) ? Intent.goRight : Intent.goLeft;
      final u0 = controllerForIntent(intent, env);
      // Force thrust during probe so tilt → vx happens within short K
      return et.ControlInput(
        thrust: true,
        left: u0.left,
        right: u0.right,
        sideLeft: u0.sideLeft,
        sideRight: u0.sideRight,
        downThrust: u0.downThrust,
      );
    }

    final int warm = (K * warmFrac).clamp(0, K - 1).toInt();
    final int meas = (K - warm).clamp(1, K);

    // Warmup (unmeasured)
    for (int i = 0; i < warm; i++) {
      env.step(1 / 60.0, _towardPad());
    }

    // Baseline AFTER warmup
    final padCx = env.terrain.padCenter.toDouble();
    final dx0Abs = (padCx - env.lander.pos.x.toDouble()).abs();

    // Measured segment
    for (int i = 0; i < meas; i++) {
      env.step(1 / 60.0, _towardPad());
    }

    final dx1Abs = (padCx - env.lander.pos.x.toDouble()).abs();
    final dPerSec = (dx0Abs - dx1Abs) / (meas / 60.0);

    // Restore snapshot
    L.pos.x = snap.x;
    L.pos.y = snap.y;
    L.vel.x = snap.vx;
    L.vel.y = snap.vy;
    L.angle = snap.ang;
    L.fuel  = snap.fuel;

    return dPerSec; // positive if we reduced |dx|
  }

  // Optional simple teacher for Stage 3 mini-landing drills (turn/thrust).
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
      decisionReturns: returns,
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
    policy.trunk.trainMode = false; // freeze trunk for these micro-drills

    int stage = 0.clamp(0, cfg.maxStage);
    if (cfg.verbose || gateVerbose) {
      print('[CUR/pad_align_progressive] start iters=${cfg.iters} '
          'batch=${cfg.batch} maxStage=${cfg.maxStage} promoAcc=${cfg.promoAcc} promoDx=${cfg.promoDx} '
          'microK=${cfg.microK} warm=${(cfg.microWarmFrac*100).toStringAsFixed(0)}% '
          'actionAlign=${cfg.allowActionAlign ? "Y" : "N"}');
      print(_fitnessLine(iter: 0, stage: stage, nL: 0, nR: 0, okL: 0, okR: 0));
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

        final stepsThis = _minSteps(stage) + rnd.nextInt(math.max(1, _maxSteps(stage) - _minSteps(stage) + 1));

        // Jitter (no thrusters) to vary rays a bit
        final jitter = stage == 0 ? rnd.nextInt(math.max(1, stepsThis ~/ 2))
            : rnd.nextInt(math.max(1, stepsThis ~/ 3));
        for (int j = 0; j < jitter; j++) {
          env.step(1 / 60.0, const et.ControlInput(thrust: false, left: false, right: false));
        }

        // Short free-drift before labeling (harder from S1+)
        final preFree = stage >= 1 ? rnd.nextInt(3) : 0;
        for (int j = 0; j < preFree; j++) {
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

        // Intent forward + supervised label
        final trueLabel = _labelFor(env);
        final (idxGreedy, _p, cacheIntent) = policy.actIntentGreedy(x);

        final ok = (idxGreedy == trueLabel);
        _pushAcc(ok ? 1.0 : 0.0, cfg.probeWindow);
        if (trueLabel == intentToIndex(Intent.goLeft)) { nL++; if (ok) okL++; }
        else { nR++; if (ok) okR++; }

        intentCaches.add(cacheIntent);
        intentChoices.add(idxGreedy);
        intentLabels.add(trueLabel);

        // --- Reliable micro-probe for progress (does not alter episode state)
        final dxPerSec = _measureProgressDxPerSec(
          env: env,
          K: cfg.microK,
          warmFrac: cfg.microWarmFrac,
        );
        _pushDxPerSec(dxPerSec, cfg.probeWindow);

        // Stage 3: optional supervised action hints (mini-landing flavor)
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

            final (_, __, cacheAction) = policy.actIntentGreedy(x2);
            actionCaches.add(cacheAction);
            turnTargets.add(turnT);
            thrustTargets.add(thrB);

            final u = et.ControlInput(
              thrust: thrB,
              left: turnT < 0,
              right: turnT > 0,
            );
            env.step(1 / 60.0, u);
          }
        }
        // Micro-episode ends here.
      }

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
          print(_fitnessLine(iter: it, stage: stage, nL: nL, nR: nR, okL: okL, okR: okR));

          if (stage < cfg.maxStage) {
            double acc = 0.0; for (final a in _accHistory) acc += a;
            acc = _accHistory.isEmpty ? 0.0 : acc / _accHistory.length;

            double dxs = 0.0; for (final d in _dxPerSecHistory) dxs += d;
            dxs = _dxPerSecHistory.isEmpty ? 0.0 : dxs / _dxPerSecHistory.length;

            if (acc >= cfg.promoAcc && dxs >= cfg.promoDx) {
              stage = (stage + 1).clamp(0, cfg.maxStage);
              print('[PADPROG] ↑ promoted to stage=$stage (acc=${_fmtPct(acc)} Δ|dx|/sec=${dxs.toStringAsFixed(2)})');
              _accHistory.clear();
              _dxPerSecHistory.clear();
              nL = nR = okL = okR = 0;
            }
          }
        }
      }
    }

    policy.trunk.trainMode = wasTrain;
  }
}
