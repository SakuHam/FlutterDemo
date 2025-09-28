// lib/ai/curriculum/final_dagger.dart
// Final landing stage (DAgger-lite): imitate a strong teacher near touchdown,
// log land% and a scalar fitness, and keep the loop simple/stable.

import 'dart:math' as math;

import '../../engine/types.dart' as et;
import '../../engine/game_engine.dart' as eng;
import '../agent.dart';
import '../nn_helper.dart' as nn;
import '../teacher.dart' hide predictiveIntentLabelAdaptive;
import 'core.dart';

/// Good-landing predicate used for logging/fitness.
bool landWin(
    eng.GameEngine env, {
      double dxFrac = 0.06,
      double vyMax = 4.0,
      double vxMax = 20.0,
      double angMaxDeg = 12.0,
    }) {
  if (env.status != et.GameStatus.landed) return false;

  final L = env.lander;
  final padCx = env.terrain.padCenter.toDouble();
  final W = env.cfg.worldW.toDouble();

  final dxAbs = (L.pos.x.toDouble() - padCx).abs();
  final vyAbs = L.vel.y.abs().toDouble();
  final vxAbs = L.vel.x.abs().toDouble();
  final angDeg = (L.angle.abs().toDouble() * 180.0 / math.pi);

  final centered = dxAbs <= dxFrac * W;
  final softVert = vyAbs <= vyMax;
  final softLat  = vxAbs <= vxMax;
  final upright  = angDeg <= angMaxDeg;

  return centered && softVert && softLat && upright;
}

/// 0..1 landing quality (used for informational logging).
double landingQuality(
    eng.GameEngine env, {
      double dxFrac = 0.06,
      double vyMax = 4.0,
    }) {
  final L = env.lander;
  final padCx = env.terrain.padCenter.toDouble();
  final W = env.cfg.worldW.toDouble();

  final dxAbs = (L.pos.x.toDouble() - padCx).abs();
  final vyAbs = L.vel.y.abs().toDouble();

  final dxNorm = (dxAbs / (dxFrac * W)).clamp(0.0, 2.0);
  final vyNorm = (vyAbs / vyMax).clamp(0.0, 2.0);

  final sDx = (1.0 - 0.5 * dxNorm).clamp(0.0, 1.0);
  final sVy = (1.0 - 0.5 * vyNorm).clamp(0.0, 1.0);
  return math.sqrt(sDx * sVy);
}

class FinalDaggerCfg {
  // Episode layout
  final int batch;
  final int iters;
  final int maxSteps;

  // Spawn ranges (relative to pad)
  final double hMin;
  final double hMax;
  final double xNearFrac;
  final double xFarFrac;
  final double farProb;
  final double vyMin;
  final double vyMax;
  final double vxNearMax;
  final double vxFarMax;

  // Acceptance / logging thresholds
  final double acceptDxFrac;
  final double acceptVyMax;

  // Controller mix
  final bool useLearnedTurn;     // use model for left/right, imitate thrust
  final double thrustBlend;      // prob-space blend M vs teacher for thrust

  // Duration holding
  final double durMin;
  final double durMax;
  final double durBlend;         // 1=model, 0=teacher (simple dyn rule)

  // Supervision strength
  final double actionAlignWeight;
  final double lr;

  // Logging
  final bool verbose;
  final int logEvery;            // in episodes

  const FinalDaggerCfg({
    this.batch = 8,
    this.iters = 600,
    this.maxSteps = 900,

    this.hMin = 70.0,
    this.hMax = 140.0,
    this.xNearFrac = 0.12,
    this.xFarFrac = 0.40,
    this.farProb = 0.35,
    this.vyMin = 6.0,
    this.vyMax = 16.0,
    this.vxNearMax = 4.0,
    this.vxFarMax = 12.0,

    this.acceptDxFrac = 0.06,
    this.acceptVyMax = 4.0,

    this.useLearnedTurn = true,
    this.thrustBlend = 0.6,

    this.durMin = 1.0,
    this.durMax = 18.0,
    this.durBlend = 0.6,

    this.actionAlignWeight = 1.0,
    this.lr = 3e-4,

    this.verbose = true,
    this.logEvery = 25,
  });

  FinalDaggerCfg copyWith({
    int? batch,
    int? iters,
    int? maxSteps,
    double? hMin,
    double? hMax,
    double? xNearFrac,
    double? xFarFrac,
    double? farProb,
    double? vyMin,
    double? vyMax,
    double? vxNearMax,
    double? vxFarMax,
    double? acceptDxFrac,
    double? acceptVyMax,
    bool? useLearnedTurn,
    double? thrustBlend,
    double? durMin,
    double? durMax,
    double? durBlend,
    double? actionAlignWeight,
    double? lr,
    bool? verbose,
    int? logEvery,
  }) => FinalDaggerCfg(
    batch: batch ?? this.batch,
    iters: iters ?? this.iters,
    maxSteps: maxSteps ?? this.maxSteps,
    hMin: hMin ?? this.hMin,
    hMax: hMax ?? this.hMax,
    xNearFrac: xNearFrac ?? this.xNearFrac,
    xFarFrac: xFarFrac ?? this.xFarFrac,
    farProb: farProb ?? this.farProb,
    vyMin: vyMin ?? this.vyMin,
    vyMax: vyMax ?? this.vyMax,
    vxNearMax: vxNearMax ?? this.vxNearMax,
    vxFarMax: vxFarMax ?? this.vxFarMax,
    acceptDxFrac: acceptDxFrac ?? this.acceptDxFrac,
    acceptVyMax: acceptVyMax ?? this.acceptVyMax,
    useLearnedTurn: useLearnedTurn ?? this.useLearnedTurn,
    thrustBlend: thrustBlend ?? this.thrustBlend,
    durMin: durMin ?? this.durMin,
    durMax: durMax ?? this.durMax,
    durBlend: durBlend ?? this.durBlend,
    actionAlignWeight: actionAlignWeight ?? this.actionAlignWeight,
    lr: lr ?? this.lr,
    verbose: verbose ?? this.verbose,
    logEvery: logEvery ?? this.logEvery,
  );
}

class FinalDagger extends Curriculum {
  @override
  String get key => 'final_dagger';

  FinalDaggerCfg cfg = const FinalDaggerCfg();

  @override
  Curriculum configure(Map<String, String?> kv, Set<String> flags) {
    final c = CliView(kv, flags);
    cfg = cfg.copyWith(
      batch:           c.getInt('finals_batch',        def: cfg.batch),
      iters:           c.getInt('finals_iters',        def: cfg.iters),
      maxSteps:        c.getInt('finals_max_steps',    def: cfg.maxSteps),
      hMin:            c.getDouble('finals_hmin',      def: cfg.hMin),
      hMax:            c.getDouble('finals_hmax',      def: cfg.hMax),
      xNearFrac:       c.getDouble('finals_x_near',    def: cfg.xNearFrac),
      xFarFrac:        c.getDouble('finals_x_far',     def: cfg.xFarFrac),
      farProb:         c.getDouble('finals_far_prob',  def: cfg.farProb),
      vyMin:           c.getDouble('finals_vy_min',    def: cfg.vyMin),
      vyMax:           c.getDouble('finals_vy_max',    def: cfg.vyMax),
      vxNearMax:       c.getDouble('finals_vx_near',   def: cfg.vxNearMax),
      vxFarMax:        c.getDouble('finals_vx_far',    def: cfg.vxFarMax),
      acceptDxFrac:    c.getDouble('finals_accept_dx_frac', def: cfg.acceptDxFrac),
      acceptVyMax:     c.getDouble('finals_accept_vy', def: cfg.acceptVyMax),
      useLearnedTurn:  c.getFlag('finals_use_learned_turn', def: cfg.useLearnedTurn),
      thrustBlend:     c.getDouble('finals_thrust_blend',   def: cfg.thrustBlend),
      durMin:          c.getDouble('finals_dur_min',   def: cfg.durMin),
      durMax:          c.getDouble('finals_dur_max',   def: cfg.durMax),
      durBlend:        c.getDouble('finals_dur_blend', def: cfg.durBlend),
      actionAlignWeight: c.getDouble('finals_action_w', def: cfg.actionAlignWeight),
      lr:              c.getDouble('finals_lr',        def: cfg.lr),
      verbose:         c.getFlag('finals_verbose',     def: cfg.verbose),
      logEvery:        c.getInt('finals_log_every',    def: cfg.logEvery),
    );
    return this;
  }

  // Spawn near or far relative to pad center (y positive is down).
  void _initStart(eng.GameEngine env, math.Random r) {
    final far = (r.nextDouble() < cfg.farProb);
    final padCx = env.terrain.padCenter.toDouble();
    final W = env.cfg.worldW.toDouble();
    final frac = far ? cfg.xFarFrac : cfg.xNearFrac;

    final xOff = (r.nextDouble() * 2 - 1) * (frac * W);
    final x = (padCx + xOff).clamp(10.0, W - 10.0);

    final h = cfg.hMin + (cfg.hMax - cfg.hMin) * r.nextDouble();
    final vy = cfg.vyMin + (cfg.vyMax - cfg.vyMin) * r.nextDouble();
    final vxMax = far ? cfg.vxFarMax : cfg.vxNearMax;
    final vx = (r.nextDouble() * 2 - 1) * vxMax;
    final gy = env.terrain.heightAt(x);

    env.lander
      ..pos.x = x
      ..pos.y = (gy - h).clamp(0.0, env.cfg.worldH - 10.0)
      ..vel.x = vx
      ..vel.y = vy
      ..angle = 0.0
      ..fuel = env.cfg.t.maxFuel;
  }

  // Simple teacher: pick intent adaptively, convert to control.
  et.ControlInput _teacherFor(env) {
    final y = predictiveIntentLabelAdaptive(env);
    final intent = indexToIntent(y);
    return controllerForIntent(intent, env);
  }

  // Per-episode DAgger roll + imitation update
  EpisodeResult _runEpisode({
    required eng.GameEngine env,
    required FeatureExtractorRays fe,
    required PolicyNetwork policy,
    required RunningNorm? norm,
    required math.Random rnd,
    required double dt,
    required bool useLearnedTurn,
    required double thrustBlend,
    required double durMin,
    required double durMax,
    required double durBlend,
    required double actionAlignWeight,
    required double lr,
    required int maxSteps,
  }) {
    policy.trunk.trainMode = true;

    final actionCaches        = <ForwardCache>[];
    final actionTurnTargets   = <int>[];
    final actionThrustTargets = <bool>[];

    env.reset(seed: rnd.nextInt(1 << 30));
    _initStart(env, rnd);

    int steps = 0;
    bool landed = false;
    double totalCost = 0.0;

    int framesLeft = 0; // tiny hold from duration head vs teacher
    while (true) {
      if (framesLeft <= 0) {
        // choose a short hold based on model (duration head) mixed with a tiny teacher rule:
        var x = fe.extract(
          lander: env.lander, terrain: env.terrain,
          worldW: env.cfg.worldW.toDouble(), worldH: env.cfg.worldH.toDouble(),
          rays: env.rays,
        );
        if (norm != null) {
          norm.observe(x);
          x = norm.normalize(x, update: false);
        }
        final c = policy.forwardFull(x);
        final predHold = c.durFrames.clamp(durMin, durMax);

        // teacher suggests 1–2 frames if stable & centered, else 1
        final padCx = env.terrain.padCenter.toDouble();
        final W = env.cfg.worldW.toDouble();
        final dxAbs = (env.lander.pos.x.toDouble() - padCx).abs();
        final vxAbs = env.lander.vel.x.toDouble().abs();
        int teachHold = 1;
        if (dxAbs < 0.05 * W && vxAbs < 18.0) teachHold = 2;

        final useHold = durBlend * predHold + (1.0 - durBlend) * teachHold;
        framesLeft = useHold.round().clamp(1, 12);
      }

      // teacher action for labels
      final uT = _teacherFor(env);

      // model features for action heads
      var xAct = fe.extract(
        lander: env.lander, terrain: env.terrain,
        worldW: env.cfg.worldW.toDouble(), worldH: env.cfg.worldH.toDouble(),
        rays: env.rays,
      );
      if (norm != null) xAct = norm.normalize(xAct, update: false);

      final (thM, lM, rM, probs, cAct) = policy.actGreedy(xAct);

      // Exec mix: learned turn (opt-in) + blended thrust
      final execLeft  = useLearnedTurn ? lM : uT.left;
      final execRight = useLearnedTurn ? rM : uT.right;

      // thrust in probability space with simple PWM-like rounding
      final pM = probs[0].clamp(0.0, 1.0);
      final pT = uT.thrust ? 1.0 : 0.0;
      final pX = (thrustBlend * pM) + ((1.0 - thrustBlend) * pT);
      final execThrust = (pX >= 0.5);

      // Apply (keep side/down from teacher)
      final info = env.step(dt, et.ControlInput(
        thrust: execThrust,
        left: execLeft,
        right: execRight,
        sideLeft: uT.sideLeft,
        sideRight: uT.sideRight,
        downThrust: uT.downThrust,
      ));
      totalCost += info.costDelta;
      steps++;
      framesLeft--;

      // Imitation labels
      actionCaches.add(cAct);
      actionTurnTargets.add(uT.left ? 0 : (uT.right ? 2 : 1));
      actionThrustTargets.add(uT.thrust);

      if (info.terminal || steps >= maxSteps) {
        landed = env.status == et.GameStatus.landed;
        break;
      }
    }

    // Pure action-head imitation update (DAgger-lite)
    if (actionCaches.isNotEmpty && actionAlignWeight > 0) {
      policy.updateFromEpisode(
        decisionCaches: const [], intentChoices: const [], decisionReturns: const [],
        alignLabels: const [], alignWeight: 0.0, intentPgWeight: 0.0,
        lr: lr, entropyBeta: 0.0, valueBeta: 0.0, huberDelta: 1.0, intentMode: true,
        actionCaches: actionCaches,
        actionTurnTargets: actionTurnTargets,
        actionThrustTargets: actionThrustTargets,
        actionAlignWeight: actionAlignWeight,
      );
    }

    return EpisodeResult(
      steps: steps,
      totalCost: totalCost,
      landed: landed,
      segMean: landingQuality(env, dxFrac: cfg.acceptDxFrac, vyMax: cfg.acceptVyMax),
    );
  }

  @override
  Future<void> run({
    required int iters,                 // ignored; we use cfg.iters
    required eng.GameEngine env,
    required FeatureExtractorRays fe,
    required PolicyNetwork policy,
    required RunningNorm? norm,
    required int planHold,              // ignored in this simpler stage
    required double tempIntent,         // ignored
    required double gamma,              // ignored
    required double lr,
    required double intentAlignWeight,  // ignored
    required double intentPgWeight,     // ignored
    required double actionAlignWeight,
    required bool gateVerbose,          // used as verbose
    required int seed,
  }) async {
    final r = math.Random(seed ^ 0xF1A1);
    final emaA = 0.96;
    double emaLand = 0.0;
    double emaQ    = 0.0;

    int epCount = 0;
    int winsInWindow = 0;
    int predsTurnCorrect = 0, predsTurnTotal = 0;
    int predsThrCorrect  = 0, predsThrTotal  = 0;

    // announce config
    if (cfg.verbose || gateVerbose) {
      print('[FINALS/DA] start iters=${cfg.iters} batch=${cfg.batch} '
          'h=[${cfg.hMin}-${cfg.hMax}] vy=[${cfg.vyMin}-${cfg.vyMax}] '
          'vxNearMax=${cfg.vxNearMax} vxFarMax=${cfg.vxFarMax}');
    }

    for (int it = 0; it < cfg.iters; it++) {
      for (int b = 0; b < cfg.batch; b++) {
        // Run one episode, but also compute per-episode predictions accuracy
        // by doing a light second pass on-the-fly inside _runEpisode’s main loop.
        // (We’ll do a tiny inline duplication to gather stats without changing training.)

        // Shadow env to collect accuracy: easiest is to sample predictions each step
        // right before step() in _runEpisode, but we already do it there; so we mirror here:
        env.reset(seed: r.nextInt(1 << 30));
        _initStart(env, r);

        int steps = 0;
        bool epLanded = false;
        double lastQ = 0.0;

        int framesLeft = 0;
        while (true) {
          if (framesLeft <= 0) {
            var x = fe.extract(
              lander: env.lander, terrain: env.terrain,
              worldW: env.cfg.worldW.toDouble(), worldH: env.cfg.worldH.toDouble(),
              rays: env.rays,
            );
            if (norm != null) {
              norm.observe(x);
              x = norm.normalize(x, update: false);
            }
            final c = policy.forwardFull(x);
            final predHold = c.durFrames.clamp(cfg.durMin, cfg.durMax);

            // tiny teacher rule for hold
            final padCx = env.terrain.padCenter.toDouble();
            final W = env.cfg.worldW.toDouble();
            final dxAbs = (env.lander.pos.x.toDouble() - padCx).abs();
            final vxAbs = env.lander.vel.x.toDouble().abs();
            int teachHold = 1;
            if (dxAbs < 0.05 * W && vxAbs < 18.0) teachHold = 2;

            final useHold = cfg.durBlend * predHold + (1.0 - cfg.durBlend) * teachHold;
            framesLeft = useHold.round().clamp(1, 12);
          }

          // labels
          final uT = _teacherFor(env);

          // model prediction
          var xAct = fe.extract(
            lander: env.lander, terrain: env.terrain,
            worldW: env.cfg.worldW.toDouble(), worldH: env.cfg.worldH.toDouble(),
            rays: env.rays,
          );
          if (norm != null) xAct = norm.normalize(xAct, update: false);
          final (thM, lM, rM, probs, cAct) = policy.actGreedy(xAct);

          // accuracy counters
          final turnIdxPred = lM ? 0 : (rM ? 2 : 1);
          final turnIdxTeach = uT.left ? 0 : (uT.right ? 2 : 1);
          if (turnIdxPred == turnIdxTeach) predsTurnCorrect++;
          predsTurnTotal++;

          final thrTeach = uT.thrust;
          if ((thM && thrTeach) || (!thM && !thrTeach)) predsThrCorrect++;
          predsThrTotal++;

          // exec as training does
          final pM = probs[0].clamp(0.0, 1.0);
          final pT = thrTeach ? 1.0 : 0.0;
          final pX = (cfg.thrustBlend * pM) + ((1.0 - cfg.thrustBlend) * pT);
          final execThrust = (pX >= 0.5);

          final execLeft  = cfg.useLearnedTurn ? lM : uT.left;
          final execRight = cfg.useLearnedTurn ? rM : uT.right;

          final info = env.step(1 / 60.0, et.ControlInput(
            thrust: execThrust,
            left: execLeft,
            right: execRight,
            sideLeft: uT.sideLeft,
            sideRight: uT.sideRight,
            downThrust: uT.downThrust,
          ));
          steps++;
          framesLeft--;

          if (info.terminal || steps >= cfg.maxSteps) {
            epLanded = env.status == et.GameStatus.landed;
            lastQ = landingQuality(env, dxFrac: cfg.acceptDxFrac, vyMax: cfg.acceptVyMax);
            break;
          }
        }

        // stats
        epCount++;
        final win = landWin(env, dxFrac: cfg.acceptDxFrac, vyMax: cfg.acceptVyMax);
        if (win) winsInWindow++;
        emaLand = emaA * emaLand + (1 - emaA) * (win ? 1.0 : 0.0);
        emaQ    = emaA * emaQ    + (1 - emaA) * lastQ;

        // Now do the actual training pass (shares RNG/state but resets env anyway)
        _runEpisode(
          env: env, fe: fe, policy: policy, norm: norm, rnd: r,
          dt: 1/60.0,
          useLearnedTurn: cfg.useLearnedTurn,
          thrustBlend: cfg.thrustBlend,
          durMin: cfg.durMin, durMax: cfg.durMax, durBlend: cfg.durBlend,
          actionAlignWeight: actionAlignWeight,
          lr: lr,
          maxSteps: cfg.maxSteps,
        );

        // logging every cfg.logEvery episodes
        if ((epCount % cfg.logEvery) == 0 && (cfg.verbose || gateVerbose)) {
          final winRate = (winsInWindow / cfg.logEvery);
          final accTurn = predsTurnTotal > 0 ? 100.0 * predsTurnCorrect / predsTurnTotal : 0.0;
          final accThr  = predsThrTotal  > 0 ? 100.0 * predsThrCorrect  / predsThrTotal  : 0.0;

          final fitness = 0.65 * emaLand + 0.35 * emaQ;
          print('[FINALS/DA] ep=$epCount  land%: win=${(100*winRate).toStringAsFixed(1)} '
              'ema=${(100*emaLand).toStringAsFixed(1)}  '
              'q: win=${lastQ.toStringAsFixed(2)} ema=${emaQ.toStringAsFixed(2)}  '
              'acc(turn)=${accTurn.toStringAsFixed(1)}  acc(thrust)=${accThr.toStringAsFixed(1)} fitness=${fitness.toStringAsFixed(3)}');

          // reset window counters
          winsInWindow = 0;
          predsTurnCorrect = 0; predsTurnTotal = 0;
          predsThrCorrect  = 0; predsThrTotal  = 0;
        }
      }
    }

    if (cfg.verbose || gateVerbose) {
      print('[FINALS/DA] done iters=${cfg.iters} batch=${cfg.batch}');
    }
  }
}
