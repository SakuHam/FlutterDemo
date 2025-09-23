// lib/ai/curriculum/hard_approach.dart
// Micro-stage: “hard approach” with adaptive lateral spawn difficulty + escalating retries.
// - Retries reuse the SAME terrain + spawn
// - Intent selection is NEVER greedy (temperature + entropy floor)
// - Exploration escalates across retries (temp, ε-action noise, lateral bias flips, assist)
// - Cohorts (terrain+spawn) count ONCE toward nearFrac adaptation (retries are "(nc)")

import 'dart:math' as math;

import '../../engine/types.dart' as et;
import '../../engine/game_engine.dart' as eng;
import '../agent.dart';
import '../nn_helper.dart' as nn;
import 'core.dart';

/// ===== Utility records (top-level; Dart doesn't support nested classes) =====
class SpawnSetup {
  final double x;
  final double h;
  final double vy;
  final double vx;
  const SpawnSetup({required this.x, required this.h, required this.vy, required this.vx});
}

class EpisodeWithSpawn {
  final EpisodeResult res;
  final SpawnSetup spawn;
  const EpisodeWithSpawn(this.res, this.spawn);
}

/// =============================== Config =====================================
class HardApproachCfg {
  final int batch;
  final int minSteps;      // do at least this many steps before allowing early termination
  final int warmFrames;    // frames where we don't allow main thrust (to establish descent)
  final double vyMin;      // minimum downward start speed
  final double vyMax;      // maximum downward start speed
  final double hMin;       // minimum spawn height
  final double hMax;       // maximum spawn height

  // Spawn band around pad center (fraction of W)
  final double nearPadFrac;        // starting band (tight)
  final bool verbose;

  // === Adaptive difficulty knobs ===
  final bool adaptDifficulty;      // enable auto-widening of spawn band
  final double nearFracMax;        // cap for widening
  final double nearFracStepUp;     // step to widen on good performance
  final double nearFracStepDown;   // step to tighten (if you want to punish regressions)
  final int    adaptWindow;        // evaluate every N episodes
  final double promoteAt;          // widen if land% >= this
  final double demoteAt;           // tighten if land% < this (set <0 to disable)

  // === Retry escalation knobs (make controls try harder on repeat) ===
  final int    retryMax;           // how many times to retry same terrain+spawn
  final double retryTempStep;      // add to intent temperature each retry (multiplier on base)
  final double retryActionEpsBase; // ε for random action flips (thrust/turn) at k=0
  final double retryActionEpsStep; // ε increment per retry
  final double retryLatBias;       // logit bonus toward pad-aligned lateral intent (±)
  final double retryPlanHoldBump;  // add % to planHold after 20th retry, ramps up slowly
  final bool   retryTeacherAssist; // progressively stronger flare/brake assist

  const HardApproachCfg({
    this.batch = 1,
    this.minSteps = 32,
    this.warmFrames = 12,
    this.vyMin = 28.0,
    this.vyMax = 36.0,
    this.hMin = 120.0,
    this.hMax = 320.0,
    this.nearPadFrac = 0.04,       // start very close to pad center
    this.verbose = true,

    this.adaptDifficulty = true,
    this.nearFracMax = 0.22,
    this.nearFracStepUp = 0.02,
    this.nearFracStepDown = 0.01,
    this.adaptWindow = 40,
    this.promoteAt = 0.65,
    this.demoteAt = 0.35,

    this.retryMax = 100,
    this.retryTempStep = 0.25,
    this.retryActionEpsBase = 0.02,
    this.retryActionEpsStep = 0.02,
    this.retryLatBias = 0.20,
    this.retryPlanHoldBump = 0.4,
    this.retryTeacherAssist = true,
  });

  HardApproachCfg copyWith({
    int? batch,
    int? minSteps,
    int? warmFrames,
    double? vyMin,
    double? vyMax,
    double? hMin,
    double? hMax,
    double? nearPadFrac,
    bool? verbose,

    bool? adaptDifficulty,
    double? nearFracMax,
    double? nearFracStepUp,
    double? nearFracStepDown,
    int? adaptWindow,
    double? promoteAt,
    double? demoteAt,

    int? retryMax,
    double? retryTempStep,
    double? retryActionEpsBase,
    double? retryActionEpsStep,
    double? retryLatBias,
    double? retryPlanHoldBump,
    bool? retryTeacherAssist,
  }) => HardApproachCfg(
    batch:            batch ?? this.batch,
    minSteps:         minSteps ?? this.minSteps,
    warmFrames:       warmFrames ?? this.warmFrames,
    vyMin:            vyMin ?? this.vyMin,
    vyMax:            vyMax ?? this.vyMax,
    hMin:             hMin ?? this.hMin,
    hMax:             hMax ?? this.hMax,
    nearPadFrac:      nearPadFrac ?? this.nearPadFrac,
    verbose:          verbose ?? this.verbose,

    adaptDifficulty:  adaptDifficulty ?? this.adaptDifficulty,
    nearFracMax:      nearFracMax ?? this.nearFracMax,
    nearFracStepUp:   nearFracStepUp ?? this.nearFracStepUp,
    nearFracStepDown: nearFracStepDown ?? this.nearFracStepDown,
    adaptWindow:      adaptWindow ?? this.adaptWindow,
    promoteAt:        promoteAt ?? this.promoteAt,
    demoteAt:         demoteAt ?? this.demoteAt,

    retryMax:         retryMax ?? this.retryMax,
    retryTempStep:    retryTempStep ?? this.retryTempStep,
    retryActionEpsBase: retryActionEpsBase ?? this.retryActionEpsBase,
    retryActionEpsStep: retryActionEpsStep ?? this.retryActionEpsStep,
    retryLatBias:     retryLatBias ?? this.retryLatBias,
    retryPlanHoldBump: retryPlanHoldBump ?? this.retryPlanHoldBump,
    retryTeacherAssist: retryTeacherAssist ?? this.retryTeacherAssist,
  );
}

/// ============================== Curriculum ==================================
class HardApproach extends Curriculum {
  @override
  String get key => 'hardapp';

  HardApproachCfg cfg = const HardApproachCfg();

  // Current adaptive spawn band (starts at cfg.nearPadFrac; widens over time)
  double _curNearFrac = 0.04;

  // Rolling landing stats for adaptation
  int _winCount = 0;
  int _winLanded = 0;

  @override
  Curriculum configure(Map<String, String?> kv, Set<String> flags) {
    final cli = CliView(kv, flags);
    cfg = cfg.copyWith(
      batch:      cli.getInt('hardapp_batch',       def: 1),
      minSteps:   cli.getInt('hardapp_min_steps',   def: 32),
      warmFrames: cli.getInt('hardapp_warm_frames', def: 12),
      vyMin:      cli.getDouble('hardapp_vy_min',   def: 28.0),
      vyMax:      cli.getDouble('hardapp_vy_max',   def: 36.0),
      hMin:       cli.getDouble('hardapp_hmin',     def: 120.0),
      hMax:       cli.getDouble('hardapp_hmax',     def: 320.0),
      nearPadFrac:cli.getDouble('hardapp_near_frac',def: 0.04),
      verbose:    cli.getFlag('hardapp_verbose',    def: true),

      adaptDifficulty:  cli.getFlag('hardapp_adapt',        def: true),
      nearFracMax:      cli.getDouble('hardapp_near_max',   def: 0.22),
      nearFracStepUp:   cli.getDouble('hardapp_near_step',  def: 0.02),
      nearFracStepDown: cli.getDouble('hardapp_near_down',  def: 0.01),
      adaptWindow:      cli.getInt('hardapp_adapt_win',     def: 40),
      promoteAt:        cli.getDouble('hardapp_promote_at', def: 0.65),
      demoteAt:         cli.getDouble('hardapp_demote_at',  def: 0.35),

      retryMax:         cli.getInt('hardapp_retry_max',     def: 100),
      retryTempStep:    cli.getDouble('hardapp_retry_tstep',def: 0.25),
      retryActionEpsBase: cli.getDouble('hardapp_retry_eps',def: 0.02),
      retryActionEpsStep: cli.getDouble('hardapp_retry_eps_step', def: 0.02),
      retryLatBias:     cli.getDouble('hardapp_retry_latbias', def: 0.20),
      retryPlanHoldBump:cli.getDouble('hardapp_retry_planbump', def: 0.4),
      retryTeacherAssist: cli.getFlag('hardapp_retry_assist', def: true),
    );
    _curNearFrac = cfg.nearPadFrac.clamp(0.0, cfg.nearFracMax);
    _winCount = 0; _winLanded = 0;
    return this;
  }

  // -------------------- Spawn helpers --------------------
  SpawnSetup _applyStart(
      eng.GameEngine env,
      math.Random r,
      HardApproachCfg c,
      double curNearFrac, {
        SpawnSetup? fixed,
      }) {
    final padCx = env.terrain.padCenter.toDouble();
    final W = env.cfg.worldW.toDouble();

    final double x = fixed?.x ??
        (padCx + (r.nextDouble() * 2 - 1) * (curNearFrac * W)).clamp(10.0, W - 10.0);
    final double h  = fixed?.h  ?? (c.hMin + (c.hMax - c.hMin) * r.nextDouble());
    final double vy = fixed?.vy ?? (c.vyMin + (c.vyMax - c.vyMin) * r.nextDouble());
    final double vx = fixed?.vx ?? ((r.nextDouble() * 16.0) - 8.0);

    final gy = env.terrain.heightAt(x);

    env.lander
      ..pos.x = x
      ..pos.y = (gy - h).clamp(0.0, env.cfg.worldH - 10.0)
      ..vel.x = vx
      ..vel.y = vy   // NOTE: positive is downward in this engine
      ..angle = 0.0
      ..fuel = env.cfg.t.maxFuel;

    return SpawnSetup(x: x, h: h, vy: vy, vx: vx);
  }

  static et.ControlInput _teacherWarm(Intent intent, eng.GameEngine env, HardApproachCfg cfg) {
    final base = controllerForIntent(intent, env);
    bool thrust = false; // suppress main thrust at warm-in

    // Optional: encourage down-thrust if rising/too slow (mirrors descendSlow)
    final t = (env.cfg.t as dynamic);
    final downEnabled = (t.downThrEnabled ?? false) == true;

    final L = env.lander;
    final gy = env.terrain.heightAt(L.pos.x);
    final h  = (gy - L.pos.y).toDouble().clamp(0.0, 1e9);
    final vCap = (0.10 * h + 8.0).clamp(8.0, 26.0);
    final needDown = downEnabled && (L.vel.y < 0.6 * vCap);

    return et.ControlInput(
      thrust: thrust,
      left: base.left,
      right: base.right,
      sideLeft: base.sideLeft,
      sideRight: base.sideRight,
      downThrust: needDown,
    );
  }

  static double _vyPredictNoThrust(eng.GameEngine env, {double tauReact = 0.35}) {
    final vy = env.lander.vel.y.toDouble();
    final g  = env.cfg.t.gravity;
    return vy + g * tauReact;
  }

  // -------- Retry escalation helpers --------
  double _retryTempFor(int k, double baseTemp) {
    // Start at least 0.8, ramp more aggressively; cap to 4.0 to avoid chaos
    final t = math.max(0.8, baseTemp) * (1.0 + cfg.retryTempStep * (k + 1));
    return t.clamp(0.8, 4.0);
  }

  double _retryEpsFor(int k) {
    // Start at 0.05 and grow to 0.35
    final e = math.max(0.05, cfg.retryActionEpsBase) + cfg.retryActionEpsStep * (k + 1);
    return e.clamp(0.05, 0.35);
  }

  double _retryEtaFor(int k) {
    // Entropy floor: mixture with uniform (keeps exploration alive)
    return (0.05 + 0.01 * k).clamp(0.05, 0.35);
  }

  double _retryPlanHoldScaleFor(int k) {
    if (k < 20) return 1.0;
    final frac = ((k - 20) / 60.0).clamp(0.0, 1.0);
    return 1.0 + cfg.retryPlanHoldBump * frac;
  }

  void _applyLateralBiasToLogits(
      List<double> logits,
      eng.GameEngine env, {
        required double biasMag,
        required int goLeftIdx,
        required int goRightIdx,
        required bool alternateSign,
      }) {
    if (biasMag <= 1e-9) return;
    final L = env.lander;
    final padX = env.terrain.padCenter.toDouble();
    final dx = (padX - L.pos.x.toDouble());
    double s = (dx >= 0) ? 1.0 : -1.0;
    if (alternateSign) s = -s; // flip on odd retries to escape symmetry traps
    logits[goRightIdx] += s > 0 ? biasMag : -biasMag * 0.25;
    logits[goLeftIdx]  += s < 0 ? biasMag : -biasMag * 0.25;
  }

  // === Sample from logits with temperature + entropy floor (never greedy) ===
  int _sampleWithEntropyFloor({
    required List<double> logits,   // unnormalized
    required double temp,           // temperature
    required double eta,            // mix weight with uniform (0..0.5 typically)
    required math.Random rnd,
  }) {
    final K = logits.length;
    final z = List<double>.from(logits);
    final T = temp.clamp(1e-6, 10.0);
    for (int i = 0; i < K; i++) z[i] /= T;
    final sm = nn.Ops.softmax(z);               // softmax(z/T)
    final mix = 1.0 - eta;
    final p = List<double>.filled(K, 0.0);
    for (int i = 0; i < K; i++) p[i] = mix * sm[i] + eta * (1.0 / K);

    final u = rnd.nextDouble();
    double acc = 0.0;
    for (int i = 0; i < K; i++) { acc += p[i]; if (u <= acc) return i; }
    return K - 1;
  }

  EpisodeWithSpawn _runEpisode({
    required eng.GameEngine env,
    required FeatureExtractorRays fe,
    required PolicyNetwork policy,
    required RunningNorm? norm,
    required math.Random rnd,
    required int planHold,
    required double tempIntent,
    required double gamma,
    required double lr,
    required double intentAlignWeight,
    required double intentPgWeight,
    required double actionAlignWeight,
    required double curNearFrac,
    int? fixedSeed,
    SpawnSetup? fixedSpawn,

    // Escalation knobs for this attempt
    double? tempOverride,
    double actionEps = 0.0,
    double latBias = 0.0,
    bool altBiasSign = false,
    int assistLevel = 0,
    double planHoldScale = 1.0,

    // Retry index for logs/schedules: 0 = first attempt, 1..N = retries
    int retryIndex = 0,

    // Logging controls
    String logPrefix = '',
    bool countedForAdapt = true,
  }) {
    policy.trunk.trainMode = true;

    final decisionRewards = <double>[];
    final decisionCaches  = <ForwardCache>[];
    final intentChoices   = <int>[];
    final decisionReturns = <double>[];
    final alignLabels     = <int>[];

    final actionCaches        = <ForwardCache>[];
    final actionTurnTargets   = <int>[];
    final actionThrustTargets = <bool>[];

    // Terrain + spawn (fixed if provided)
    env.reset(seed: fixedSeed ?? rnd.nextInt(1 << 30));
    final usedSpawn = _applyStart(env, rnd, cfg, curNearFrac, fixed: fixedSpawn);

    int framesLeft = 0;
    int currentIntentIdx = intentToIndex(Intent.descendSlow);
    double accReward = 0.0;

    int steps = 0;
    const int maxSteps = 900;   // hard guard to prevent infinite loops
    bool landed = false;
    double totalCost = 0.0;

    double _vCapLoose(double h) {
      // stronger braking on higher assist levels
      final base = 0.12, add = (assistLevel == 0) ? 0.0 : (assistLevel == 1 ? 0.03 : 0.06);
      final off  = (assistLevel == 0) ? 10.0 : (assistLevel == 1 ? 12.0 : 14.0);
      return ((base + add) * h + off).clamp(10.0, 36.0);
    }

    while (true) {
      // Refresh intent decision when plan window expires
      if (framesLeft <= 0) {
        var x = fe.extract(
          lander: env.lander,
          terrain: env.terrain,
          worldW: env.cfg.worldW,
          worldH: env.cfg.worldH,
          rays: env.rays,
        );
        final yTeacher = predictiveIntentLabelAdaptive(env);

        if (norm != null) {
          norm.observe(x);
          x = norm.normalize(x, update: false);
        }

        // --- Intent policy forward with optional lateral bias & temperature ---
        final (_greedyIdx, p, cache) = policy.actIntentGreedy(x);

        // Use probs to get "logits"; if raw logits available, use those instead.
        final logits = p.map((pp) => math.log(pp.clamp(1e-12, 1.0))).toList();
        _applyLateralBiasToLogits(
          logits, env,
          biasMag: latBias,
          goLeftIdx: intentToIndex(Intent.goLeft),
          goRightIdx: intentToIndex(Intent.goRight),
          alternateSign: altBiasSign,
        );
        final Tuse = (tempOverride ?? tempIntent);
        final eta  = _retryEtaFor(retryIndex);
        final pick = _sampleWithEntropyFloor(logits: logits, temp: Tuse, eta: eta, rnd: rnd);
        currentIntentIdx = pick;

        decisionCaches.add(cache);
        intentChoices.add(pick);
        alignLabels.add(yTeacher);
        decisionRewards.add(accReward);
        accReward = 0.0;

        // compute decision-window advantages
        final tn = decisionRewards.length;
        final tmp = List<double>.filled(tn, 0.0);
        double G = 0.0;
        for (int i = tn - 1; i >= 0; i--) { G = decisionRewards[i] + gamma * G; tmp[i] = G; }
        double mean = 0.0; for (final v in tmp) mean += v; mean /= math.max(1, tn);
        double var0 = 0.0; for (final v in tmp) { final d = v - mean; var0 += d * d; }
        var0 = (var0 / math.max(1, tn)).clamp(1e-9, double.infinity);
        final std = math.sqrt(var0);
        decisionReturns
          ..clear()
          ..addAll(tmp.map((v) => (v - mean) / std));

        framesLeft = math.max(1, (planHold * planHoldScale).round());
      }

      final intent = indexToIntent(currentIntentIdx);
      final useWarm = (steps < cfg.warmFrames);
      final uTeacher = useWarm ? _teacherWarm(intent, env, cfg) : controllerForIntent(intent, env);

      // --- Optional ε-noise on actions to "try more" ---
      et.ControlInput uTry = uTeacher;
      if (!useWarm && actionEps > 1e-9) {
        // random thrust toggle with small probability
        if (rnd.nextDouble() < actionEps) {
          uTry = et.ControlInput(
            thrust: !uTry.thrust,
            left: uTry.left, right: uTry.right,
            sideLeft: uTry.sideLeft, sideRight: uTry.sideRight,
            downThrust: uTry.downThrust,
          );
        }
        // small chance to flip a lateral choice or engage side thrusters
        if (rnd.nextDouble() < actionEps) {
          final flipLR = rnd.nextBool();
          final left  = flipLR ? !uTry.left  : uTry.left;
          final right = flipLR ? !uTry.right : uTry.right;
          final sl = uTry.sideLeft  || (!uTry.sideRight && rnd.nextDouble() < 0.33);
          final sr = uTry.sideRight || (!uTry.sideLeft  && rnd.nextDouble() < 0.33);
          uTry = et.ControlInput(
            thrust: uTry.thrust, left: left, right: right,
            sideLeft: sl, sideRight: sr, downThrust: uTry.downThrust,
          );
        }
      }

      // Step the env (with stronger flare assist if configured)
      et.StepInfo info;
      if (!useWarm && intent == Intent.descendSlow) {
        final L = env.lander;
        final gy = env.terrain.heightAt(L.pos.x);
        final h  = (gy - L.pos.y).toDouble().clamp(0.0, 1e9);
        final vCap = _vCapLoose(h);
        final tau = (assistLevel >= 2) ? 0.75 : 0.6;
        final vyNext = _vyPredictNoThrust(env, tauReact: tau);
        final needUp = (L.vel.y > vCap) || (vyNext > 0.9 * vCap);
        final u2 = et.ControlInput(
          thrust: uTry.thrust || needUp,
          left: uTry.left, right: uTry.right,
          sideLeft: uTry.sideLeft, sideRight: uTry.sideRight,
          downThrust: uTry.downThrust,
        );
        info = env.step(1 / 60.0, u2);
      } else {
        info = env.step(1 / 60.0, uTry);
      }
      totalCost += info.costDelta;

      // action head supervision
      var xAct = fe.extract(
        lander: env.lander,
        terrain: env.terrain,
        worldW: env.cfg.worldW,
        worldH: env.cfg.worldH,
        rays: env.rays,
      );
      if (norm != null) xAct = norm.normalize(xAct, update: false);
      final (_, __, ___, ____, cAct) = policy.actGreedy(xAct);
      actionCaches.add(cAct);
      actionTurnTargets.add(uTeacher.left ? 0 : (uTeacher.right ? 2 : 1));
      actionThrustTargets.add(uTeacher.thrust);

      // dense reward: simple speed-min
      final vx = env.lander.vel.x.toDouble();
      final vy = env.lander.vel.y.toDouble();
      final v = math.sqrt(vx * vx + vy * vy);
      accReward += -0.01 * v;

      // progression + decision window countdown
      steps++;
      framesLeft--;

      // Enforce a minimum number of steps before allowing termination
      if (steps < cfg.minSteps && env.status != et.GameStatus.playing) {
        // Restart same micro-episode with the exact same terrain + spawn.
        env.reset(seed: fixedSeed ?? rnd.nextInt(1 << 30));
        _applyStart(env, rnd, cfg, curNearFrac, fixed: usedSpawn);
        framesLeft = 0;
        continue;
      }

      if (env.status != et.GameStatus.playing || steps >= maxSteps) {
        landed = env.status == et.GameStatus.landed;
        if (decisionRewards.isNotEmpty && accReward.abs() > 0) {
          decisionRewards[decisionRewards.length - 1] += accReward;
          accReward = 0.0;
        }
        break;
      }
    }

    final bool accept = landed;
    if (accept) {
      // Single update from this episode
      policy.updateFromEpisode(
        decisionCaches: decisionCaches,
        intentChoices: intentChoices,
        decisionReturns: decisionReturns,
        alignLabels: alignLabels,
        alignWeight: intentAlignWeight,
        intentPgWeight: intentPgWeight,
        lr: lr,
        entropyBeta: 0.0,
        valueBeta: 0.0,
        huberDelta: 1.0,
        intentMode: true,
        actionCaches: actionCaches,
        actionTurnTargets: actionTurnTargets,
        actionThrustTargets: actionThrustTargets,
        actionAlignWeight: actionAlignWeight,
      );
    }

    if (cfg.verbose) {
      final L = env.lander;
      final gx = env.terrain.heightAt(L.pos.x.toDouble());
      final h  = (gx - L.pos.y).toDouble();
      final nc = countedForAdapt ? '' : ' (nc)'; // <-- not counted
      final Tuse = (tempOverride ?? tempIntent);
      final eta  = _retryEtaFor(retryIndex);
      print('${logPrefix}[HARDAPP$nc] steps=$steps landed=${landed ? "Y" : "N"} '
          'vy=${L.vel.y.toStringAsFixed(1)} h=${h.toStringAsFixed(1)} '
          'nearFrac=${curNearFrac.toStringAsFixed(3)} '
          'spawn=(x=${usedSpawn.x.toStringAsFixed(1)}, h=${usedSpawn.h.toStringAsFixed(1)}, '
          'vy=${usedSpawn.vy.toStringAsFixed(1)}, vx=${usedSpawn.vx.toStringAsFixed(1)})'
          '${nc.isNotEmpty ? " [temp=${Tuse.toStringAsFixed(2)} eta=${eta.toStringAsFixed(2)} "
          "eps=${actionEps.toStringAsFixed(2)} assist=$assistLevel]" : ""}');
    }

    return EpisodeWithSpawn(
      EpisodeResult(steps: steps, totalCost: totalCost, landed: landed, segMean: 0.0),
      usedSpawn,
    );
  }

  // ---- Difficulty adaptation (counted ONCE per cohort) ----
  double _emaLand = 0.0;
  int _noChangeWindows = 0;

  void _maybeAdaptDifficulty({required bool landed}) {
    if (!cfg.adaptDifficulty) return;

    _winCount++;
    if (landed) _winLanded++;

    if (_winCount >= cfg.adaptWindow) {
      final rate = _winLanded / _winCount.toDouble();

      // EMA smoothing
      final alpha = 0.25;
      _emaLand = (_emaLand == 0.0) ? rate : (1 - alpha) * _emaLand + alpha * rate;

      double next = _curNearFrac;
      final old = _curNearFrac;

      if (_emaLand >= cfg.promoteAt) {
        final margin = (_emaLand - cfg.promoteAt).clamp(0.0, 0.5);
        final k = 1.0 + 2.0 * margin;
        next = (next + k * cfg.nearFracStepUp).clamp(0.0, cfg.nearFracMax);
        _noChangeWindows = 0;
      } else if (cfg.demoteAt >= 0.0 && _emaLand < cfg.demoteAt) {
        next = (next - cfg.nearFracStepDown).clamp(0.0, cfg.nearFracMax);
        _noChangeWindows = 0;
      } else {
        _noChangeWindows++;
        if (_noChangeWindows >= 4 && next < cfg.nearFracMax) {
          next = (next + 0.5 * cfg.nearFracStepUp).clamp(0.0, cfg.nearFracMax);
          _noChangeWindows = 0;
        }
      }

      if (cfg.verbose) {
        print('[HARDAPP/ADAPT] win=$_winCount landed=$_winLanded '
            'rate=${rate.toStringAsFixed(2)} ema=${_emaLand.toStringAsFixed(2)} '
            'nearFrac ${_curNearFrac.toStringAsFixed(3)} -> ${next.toStringAsFixed(3)}');
      }

      if ((next - old).abs() > 1e-6) _curNearFrac = next;

      _winCount = 0;
      _winLanded = 0;
    }
  }

  // =============================== Run loop ==================================
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
    final rnd = math.Random(seed ^ 0xA11A);
    final verboseEvery = 25;

    if (gateVerbose) {
      print('[CUR/hardapp] start iters=$iters batch=${cfg.batch} '
          'nearStart=${_curNearFrac.toStringAsFixed(3)} max=${cfg.nearFracMax}');
    }

    for (int it = 0; it < iters; it++) {
      for (int b = 0; b < cfg.batch; b++) {
        // Fixed terrain+spawn cohort
        final terrainSeed = rnd.nextInt(1 << 30);

        // ----- First attempt (COUNTED) -----
        EpisodeWithSpawn ep = _runEpisode(
          env: env, fe: fe, policy: policy, norm: norm, rnd: rnd,
          planHold: planHold, tempIntent: tempIntent, gamma: gamma, lr: lr,
          intentAlignWeight: intentAlignWeight, intentPgWeight: intentPgWeight,
          actionAlignWeight: 0.25, curNearFrac: _curNearFrac,
          fixedSeed: terrainSeed, fixedSpawn: null,
          tempOverride: tempIntent, actionEps: _retryEpsFor(0),
          latBias: cfg.retryLatBias, altBiasSign: false,
          assistLevel: cfg.retryTeacherAssist ? 0 : 0,
          planHoldScale: _retryPlanHoldScaleFor(0),
          retryIndex: 0,
          logPrefix: '', countedForAdapt: true,
        );

        bool landedAny = ep.res.landed;
        final fixedSpawn = ep.spawn;

        // Capture signature after the attempt (for stuck detection)
        int? lastSteps = ep.res.steps;
        bool? lastLanded = ep.res.landed;
        double? lastVy, lastH;
        {
          final L = env.lander;
          final gy = env.terrain.heightAt(L.pos.x.toDouble());
          final h = (gy - L.pos.y).toDouble();
          lastVy = L.vel.y.toDouble();
          lastH  = h;
        }
        int stuckCount = 0;

        // ----- Retries (NOT COUNTED) -----
        if (!ep.res.landed) {
          if (cfg.verbose) {
            print('  [HARDAPP/RETRY] no landing → escalating controls up to ${cfg.retryMax}x (nc)');
          }
          for (int k = 1; k <= cfg.retryMax; k++) {
            // Default escalation schedules
            bool altSign = (k % 2 == 1);
            int assist = cfg.retryTeacherAssist ? (k < 10 ? 0 : (k < 35 ? 1 : 2)) : 0;
            double tempK = _retryTempFor(k, tempIntent);
            double epsK  = _retryEpsFor(k);
            double planScaleK = _retryPlanHoldScaleFor(k);

            // Run retry
            ep = _runEpisode(
              env: env, fe: fe, policy: policy, norm: norm, rnd: rnd,
              planHold: planHold, tempIntent: tempIntent, gamma: gamma, lr: lr,
              intentAlignWeight: intentAlignWeight, intentPgWeight: intentPgWeight,
              actionAlignWeight: 0.25, curNearFrac: _curNearFrac,
              fixedSeed: terrainSeed, fixedSpawn: fixedSpawn,
              tempOverride: tempK,
              actionEps: epsK,
              latBias: cfg.retryLatBias, altBiasSign: altSign,
              assistLevel: assist, planHoldScale: planScaleK,
              retryIndex: k,
              logPrefix: '  ', countedForAdapt: false, // <-- indented + not counted
            );

            landedAny = landedAny || ep.res.landed;

            // --- Stuck-buster: compare signatures; if identical, jack exploration for NEXT retry
            final L = env.lander; // state at end of retry
            final gy = env.terrain.heightAt(L.pos.x.toDouble());
            final h  = (gy - L.pos.y).toDouble();
            final sigSame = (lastSteps == ep.res.steps) && (lastLanded == ep.res.landed);
            final sameVy  = (lastVy != null && (L.vel.y.toDouble() - lastVy!).abs() < 1e-9);
            final sameH   = (lastH  != null && (h - lastH!).abs() < 1e-9);
            final isStuck = sigSame && sameVy && sameH;

            if (cfg.verbose) {
              final etaK = _retryEtaFor(k);
              print('  [HARDAPP/EXP (nc)] retry=$k temp=${tempK.toStringAsFixed(2)} '
                  'eta=${etaK.toStringAsFixed(2)} eps=${epsK.toStringAsFixed(2)} '
                  'assist=$assist altBias=${altSign ? "flip" : "norm"} '
                  '${isStuck ? "→ STUCK" : ""}');
            }

            if (isStuck) {
              stuckCount++;
              // For the NEXT retry, push to max exploration immediately
              altSign = !altSign;
              assist = 2;
              tempK = 3.5;
              epsK = 0.35;
              planScaleK = 1.0 + cfg.retryPlanHoldBump;
            } else {
              stuckCount = 0;
            }

            // Update "last" signature
            lastSteps  = ep.res.steps;
            lastLanded = ep.res.landed;
            lastVy     = L.vel.y.toDouble();
            lastH      = h;

            if (ep.res.landed) {
              if (cfg.verbose) {
                print('  [HARDAPP/RETRY] landed on retry $k (nc)');
              }
              break;
            }
          }
        }

        // ✅ Count ONCE per cohort using aggregate success
        _maybeAdaptDifficulty(landed: landedAny);
      }

      if (gateVerbose && ((it + 1) % verboseEvery == 0)) {
        print('[CUR/hardapp] iter=${it + 1}/$iters '
            'nearFrac=${_curNearFrac.toStringAsFixed(3)}');
      }
    }
    if (gateVerbose) print('[CUR/hardapp] done iters=$iters');
  }
}
