// lib/ai/curriculum/final_approach.dart
// Final approach micro-stage: pad-centric touchdown with quality gating.
// Spawns both near the pad and (sometimes) far to practice late recentering.

import 'dart:math' as math;

import '../../engine/types.dart' as et;
import '../../engine/game_engine.dart' as eng;
import '../agent.dart';
import '../nn_helper.dart' as nn;
import 'core.dart';

/* -------------------------------------------------------------------------- */
/*                               Stats helpers                                */
/* -------------------------------------------------------------------------- */

class _FinalOutcome {
  final bool landed;
  final double q;        // 0..1 quality
  final bool acceptG;    // ACCEPT[G]
  final bool acceptO;    // ACCEPT[O]
  _FinalOutcome(this.landed, this.q, this.acceptG, this.acceptO);
}

class _FinalStats {
  final int window;
  int _idx = 0;
  final List<bool> _ring;    // landed?
  final List<double> _qRing; // quality in [0,1]
  int landedTotal = 0;
  int episodes = 0;
  int acceptG = 0;
  int acceptO = 0;
  int reject = 0;

  double emaLand = 0.0;  // EMA of land% (0..1)
  double emaQ = 0.0;     // EMA of quality
  final double _emaAlpha;  // tweakable

  _FinalStats({this.window = 200, double emaAlpha = 0.06})
      : _ring = List<bool>.filled(window, false),
        _qRing = List<double>.filled(window, 0.0),
        _emaAlpha = emaAlpha;

  void add(_FinalOutcome o) {
    // window maintenance
    final old = _ring[_idx];
    if (old) landedTotal--;
    _ring[_idx] = o.landed;
    _qRing[_idx] = o.q;
    if (o.landed) landedTotal++;

    _idx = (_idx + 1) % window;
    episodes++;

    if (o.acceptG) acceptG++; else if (o.acceptO) acceptO++; else reject++;

    // EMAs (treat landed as 0/1)
    final lp = o.landed ? 1.0 : 0.0;
    emaLand = (1 - _emaAlpha) * emaLand + _emaAlpha * lp;
    emaQ    = (1 - _emaAlpha) * emaQ    + _emaAlpha * o.q;
  }

  double landPctWindow() {
    final denom = episodes < window ? episodes : window;
    if (denom <= 0) return 0.0;
    return landedTotal / denom;
  }

  double meanQWindow() {
    final denom = episodes < window ? episodes : window;
    if (denom <= 0) return 0.0;
    double s = 0.0;
    for (int i = 0; i < denom; i++) {
      final idx = (_idx - 1 - i);
      final j = (idx >= 0) ? idx : (idx % window + window) % window;
      s += _qRing[j];
    }
    return s / denom;
  }
}

/* -------------------------------------------------------------------------- */
/*                              Config & curriculum                            */
/* -------------------------------------------------------------------------- */

class FinalApproachCfg {
  // Episode layout
  final int batch;
  final int warmFrames;          // initial conservative frames (kept for API)
  final int minSteps;            // enforce minimum episode length
  final int maxSteps;            // hard cap to prevent infinite loops

  // Spawn ranges
  final double hMin;
  final double hMax;
  final double xNearFrac;        // near-pad lateral offset (fraction of W)
  final double xFarFrac;         // far lateral offset (fraction of W)
  final double farProb;          // probability to use far start (kept for CLI)
  final double vyMin;            // y-down: positive is descending
  final double vyMax;
  final double vxNearMax;        // lateral speed cap near-start
  final double vxFarMax;         // lateral speed cap far-start

  // Touchdown target
  final double vyTouchdown;      // desired slow vertical at touchdown (y-down, +)

  // Acceptance (quality gate)
  final double acceptDxFrac;     // within this fraction of W from pad center
  final double acceptVyMax;      // max |vy| at touchdown to accept
  final bool   gateRejectSkipsUpdate; // kept for CLI compatibility

  // Logging
  final bool verbose;

  const FinalApproachCfg({
    // episode
    this.batch = 1,
    this.warmFrames = 10,
    this.minSteps = 24,
    this.maxSteps = 900,

    // spawn (y-down world)
    this.hMin = 80.0,            // start flare a bit earlier
    this.hMax = 160.0,
    this.xNearFrac = 0.12,
    this.xFarFrac = 0.40,
    this.farProb = 0.35,
    this.vyMin = 8.0,
    this.vyMax = 22.0,
    this.vxNearMax = 5.0,
    this.vxFarMax = 20.0,

    // target
    this.vyTouchdown = 3.5,      // align with acceptance / teacher

    // acceptance (slightly looser; prevents discarding “barely-ok” touchdowns)
    this.acceptDxFrac = 0.06,
    this.acceptVyMax = 4.0,      // was 3.5
    this.gateRejectSkipsUpdate = true,

    // logs
    this.verbose = true,
  });

  FinalApproachCfg copyWith({
    int? batch,
    int? warmFrames,
    int? minSteps,
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
    double? vyTouchdown,
    double? acceptDxFrac,
    double? acceptVyMax,
    bool? gateRejectSkipsUpdate,
    bool? verbose,
  }) => FinalApproachCfg(
    batch:                 batch ?? this.batch,
    warmFrames:            warmFrames ?? this.warmFrames,
    minSteps:              minSteps ?? this.minSteps,
    maxSteps:              maxSteps ?? this.maxSteps,
    hMin:                  hMin ?? this.hMin,
    hMax:                  hMax ?? this.hMax,
    xNearFrac:             xNearFrac ?? this.xNearFrac,
    xFarFrac:              xFarFrac ?? this.xFarFrac,
    farProb:               farProb ?? this.farProb,
    vyMin:                 vyMin ?? this.vyMin,
    vyMax:                 vyMax ?? this.vyMax,
    vxNearMax:             vxNearMax ?? this.vxNearMax,
    vxFarMax:              vxFarMax ?? this.vxFarMax,
    vyTouchdown:           vyTouchdown ?? this.vyTouchdown,
    acceptDxFrac:          acceptDxFrac ?? this.acceptDxFrac,
    acceptVyMax:           acceptVyMax ?? this.acceptVyMax,
    gateRejectSkipsUpdate: gateRejectSkipsUpdate ?? this.gateRejectSkipsUpdate,
    verbose:               verbose ?? this.verbose,
  );
}

class FinalApproach extends Curriculum {
  @override
  String get key => 'final';

  FinalApproachCfg cfg = const FinalApproachCfg();

  // Deterministic spawn cycler to reduce variance:
  // 0..7 cover: near/far × left/right × vyLow/vyHigh
  int _spawnTick = 0;

  @override
  Curriculum configure(Map<String, String?> kv, Set<String> flags) {
    final c = CliView(kv, flags);
    cfg = cfg.copyWith(
      batch:                 c.getInt('final_batch',        def: cfg.batch),
      warmFrames:            c.getInt('final_warm_frames',  def: cfg.warmFrames),
      minSteps:              c.getInt('final_min_steps',    def: cfg.minSteps),
      maxSteps:              c.getInt('final_max_steps',    def: cfg.maxSteps),
      hMin:                  c.getDouble('final_hmin',      def: cfg.hMin),
      hMax:                  c.getDouble('final_hmax',      def: cfg.hMax),
      xNearFrac:             c.getDouble('final_x_near',    def: cfg.xNearFrac),
      xFarFrac:              c.getDouble('final_x_far',     def: cfg.xFarFrac),
      farProb:               c.getDouble('final_far_prob',  def: cfg.farProb), // kept for CLI, not used in cycler
      vyMin:                 c.getDouble('final_vy_min',    def: cfg.vyMin),
      vyMax:                 c.getDouble('final_vy_max',    def: cfg.vyMax),
      vxNearMax:             c.getDouble('final_vx_near',   def: cfg.vxNearMax),
      vxFarMax:              c.getDouble('final_vx_far',    def: cfg.vxFarMax),
      vyTouchdown:           c.getDouble('final_vy_touch',  def: cfg.vyTouchdown),
      acceptDxFrac:          c.getDouble('final_accept_dx_frac', def: cfg.acceptDxFrac),
      acceptVyMax:           c.getDouble('final_accept_vy', def: cfg.acceptVyMax),
      gateRejectSkipsUpdate: c.getFlag('final_gate_skip_update', def: cfg.gateRejectSkipsUpdate),
      verbose:               c.getFlag('final_verbose',     def: cfg.verbose),
    );
    return this;
  }

  /* ------------------------------------------------------------------------ */
  /*                                 Spawning                                 */
  /* ------------------------------------------------------------------------ */

  // Stratified spawn near/far × left/right × vyLow/vyHigh (y-down world).
  void _initStart(eng.GameEngine env, math.Random r) {
    final mode = _spawnTick++ & 7; // 0..7 loop
    final isFar = (mode & 1) == 1;               // toggle far/near
    final rightSide = (mode & 2) == 2;           // toggle left/right
    final vyHigh = (mode & 4) == 4;              // toggle vy low/high

    final padCx = env.terrain.padCenter.toDouble();
    final W = env.cfg.worldW.toDouble();
    final frac = isFar ? cfg.xFarFrac : cfg.xNearFrac;

    // Place to the chosen side with small jitter
    final side = rightSide ? 1.0 : -1.0;
    final xBase = padCx + side * frac * W;
    final x = (xBase + (r.nextDouble() - 0.5) * 0.12 * W).clamp(10.0, W - 10.0);

    // Height bucket with jitter
    final hBucket = vyHigh ? 0.75 : 0.35; // relative position in [hMin, hMax]
    final hJitter = (r.nextDouble() - 0.5) * 0.15;
    final h = cfg.hMin + (cfg.hMax - cfg.hMin) * (hBucket + hJitter).clamp(0.0, 1.0);

    // vy bucket (y-down: + is descending)
    final vyMid = vyHigh ? (0.75 * (cfg.vyMin + cfg.vyMax)) : (0.45 * (cfg.vyMin + cfg.vyMax) / 2.0 + cfg.vyMin);
    final vy = (vyMid + (r.nextDouble() - 0.5) * 6.0).clamp(cfg.vyMin, cfg.vyMax);

    // vx cap per far/near, pick sign toward pad center
    final vxCap = isFar ? cfg.vxFarMax : cfg.vxNearMax;
    final toCenterSign = rightSide ? -1.0 : 1.0;
    final vx = toCenterSign * (0.6 * vxCap + (r.nextDouble() * 0.4 * vxCap)); // moving roughly toward center

    final gy = env.terrain.heightAt(x);

    env.lander
      ..pos.x = x
      ..pos.y = (gy - h).clamp(0.0, env.cfg.worldH - 10.0) // above ground (y-down)
      ..vel.x = vx
      ..vel.y = vy
      ..angle = 0.0
      ..fuel = env.cfg.t.maxFuel;
  }

  /* ------------------------------------------------------------------------ */
  /*                            Teacher & targets                              */
  /* ------------------------------------------------------------------------ */

  // Height-aware vertical target (y-down).
  double _vyTarget(double h) {
    // Earlier + softer flare:
    // >120: approach; 80..120: start slowing; 40..80: flare entry; 20..40: settle; <=20: touchdown speed.
    if (h > 120) return 36.0;
    if (h > 80)  return 28.0 + 0.20 * (h - 80.0);  // 28..36
    if (h > 40)  return 16.0 + 0.30 * (h - 40.0);  // 16..28
    if (h > 20)  return  8.0 + 0.20 * (h - 20.0);  //  8..12
    return cfg.vyTouchdown;                        // ≈ 3.5
  }

  double _latGain(double h) {
    // Stronger far from ground; gentle as we near pad to avoid overshoot.
    if (h > 120) return 0.10;
    if (h > 80)  return 0.08;
    if (h > 40)  return 0.065;
    return 0.055;
  }

  // PD-like lateral + soft vertical. Stronger lateral drive when far from pad.
  et.ControlInput _teacher(eng.GameEngine env) {
    final L = env.lander;
    final padCx = env.terrain.padCenter.toDouble();
    final W = env.cfg.worldW.toDouble();
    final gy = env.terrain.heightAt(L.pos.x);
    final h  = (gy - L.pos.y).toDouble().clamp(0.0, 1e9);

    final dx = (L.pos.x - padCx).toDouble();
    final vx = L.vel.x.toDouble();
    final vy = L.vel.y.toDouble();

    // Lateral target speed proportional to dx, with height-aware max.
    final kx = _latGain(h);
    final vLatMaxBase = (32.0 + 0.30 * h).clamp(22.0, 100.0); // tightens near ground
    // Farther than near band → allow more, but still clamp
    final farScale = (dx.abs() / (cfg.xFarFrac * W)).clamp(0.0, 1.5);
    final vLatMax = (vLatMaxBase * (1.0 + 0.7 * farScale)).clamp(22.0, 140.0);

    final vTargetX = (-kx * dx).clamp(-vLatMax, vLatMax);

    // Error and turn decision; deadzone tightens near ground.
    final errX = (vx - vTargetX);
    final dz = (h > 60.0) ? 2.0 : 1.0; // smaller deadzone when low
    final turnLeft  = errX >  dz;
    final turnRight = errX < -dz;

    // Vertical: explicit vyTarget with hysteresis to avoid chatter.
    final vyT = _vyTarget(h);
    // If well-centered, be a tad stricter
    final centerTight = (dx.abs() < 0.10 * W) ? 0.8 : 0.0;
    final upMargin = 0.8 + 0.02 * h;    // more slack high up, tighter low
    final wantUp = (vy > (vyT - centerTight) + upMargin);

    // Optional tiny cone bias (very soft) when low & still off-center.
    bool sideLeft = false, sideRight = false;
    if (h <= 28.0) {
      final cone = ((h - 10.0) / 18.0).clamp(0.0, 1.0); // fades to 0 by h=10
      if (dx.abs() > 0.06 * W) {
        if (dx > 0) sideLeft = (cone > 0.0);  // nudge toward center
        if (dx < 0) sideRight = (cone > 0.0);
      }
    }

    return et.ControlInput(
      thrust: wantUp,          // main engine up reduces y (y-down)
      left:   turnLeft,
      right:  turnRight,
      sideLeft: sideLeft,      // soft RCS nudge near ground
      sideRight: sideRight,
      downThrust: false,
    );
  }

  /* ------------------------------------------------------------------------ */
  /*                               Train episode                               */
  /* ------------------------------------------------------------------------ */

  _FinalOutcome _runEpisode({
    required eng.GameEngine env,
    required FeatureExtractorRays fe,
    required PolicyNetwork policy,
    required RunningNorm? norm,
    required math.Random rnd,
    required int planHold,
    required double tempIntent, // will be overridden to 0.0 in run()
    required double gamma,
    required double lr,
    required double intentAlignWeight,
    required double intentPgWeight,
    required double actionAlignWeight,
    required bool gateVerbose,
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

    env.reset(seed: rnd.nextInt(1 << 30));
    _initStart(env, rnd);

    int framesLeft = 0;
    int currentIntentIdx = intentToIndex(Intent.descendSlow);
    double accReward = 0.0;

    int steps = 0;
    bool landed = false;
    double totalCost = 0.0;

    // shaping weights (height-aware use below)
    const wDx = 0.0030;      // lateral distance penalty
    const wVy = 0.0100;      // deviation from vyTarget
    const wCenter = 0.0015;  // soft center reward
    const wTouch = 2.2;      // touchdown bonus

    double q = 0.0;          // quality computed at the end
    bool acceptedGreat = false, acceptedOkay = false;

    while (true) {
      if (framesLeft <= 0) {
        var x = fe.extract(
            lander: env.lander, terrain: env.terrain,
            worldW: env.cfg.worldW, worldH: env.cfg.worldH, rays: env.rays);
        final yTeacher = predictiveIntentLabelAdaptive(env);
        if (norm != null) {
          norm.observe(x);
          x = norm.normalize(x, update: false);
        }
        final (idxGreedy, p, cache) = policy.actIntentGreedy(x);

        // FINAL: no exploration — this is a reflex, learn it deterministically
        final pick = idxGreedy;

        currentIntentIdx = pick;
        decisionCaches.add(cache);
        intentChoices.add(pick);
        alignLabels.add(yTeacher);
        decisionRewards.add(accReward);
        accReward = 0.0;

        // standardized returns for PG
        final T = decisionRewards.length;
        final tmp = List<double>.filled(T, 0.0);
        double G = 0.0;
        for (int i = T - 1; i >= 0; i--) { G = decisionRewards[i] + gamma * G; tmp[i] = G; }
        double mean = 0.0; for (final v in tmp) mean += v; mean /= math.max(1, T);
        double var0 = 0.0; for (final v in tmp) { final d = v - mean; var0 += d * d; }
        var0 = (var0 / math.max(1, T)).clamp(1e-9, double.infinity);
        final std = math.sqrt(var0);
        decisionReturns..clear()..addAll(tmp.map((v) => (v - mean) / std));

        framesLeft = planHold;
      }

      // Teacher
      final u = _teacher(env);
      final info = env.step(1 / 60.0, u);
      totalCost += info.costDelta;

      // Action supervision
      var xAct = fe.extract(
          lander: env.lander, terrain: env.terrain,
          worldW: env.cfg.worldW, worldH: env.cfg.worldH, rays: env.rays);
      if (norm != null) xAct = norm.normalize(xAct, update: false);
      final (_, __, ___, ____, cAct) = policy.actGreedy(xAct);
      actionCaches.add(cAct);
      actionTurnTargets.add(u.left ? 0 : (u.right ? 2 : 1));
      actionThrustTargets.add(u.thrust);

      // Dense shaping
      final L = env.lander;
      final padCx = env.terrain.padCenter.toDouble();
      final W = env.cfg.worldW.toDouble();
      final gy = env.terrain.heightAt(L.pos.x);
      final h  = (gy - L.pos.y).toDouble().clamp(0.0, 1e9);

      final dxAbs = (L.pos.x - padCx).abs();
      final vy = L.vel.y.toDouble();
      final vyT = _vyTarget(h);

      double r = 0.0;
      r -= wDx * dxAbs;
      // penalize deviation from vy target (not raw |vy|)
      // slightly reweight near ground
      final vyW = (h > 40.0) ? wVy : (wVy * 1.25);
      r -= vyW * (vy - vyT).abs();

      // subtle centering lift that grows near ground
      final centerGain = (1.0 - (h / (cfg.hMax + 1e-6))).clamp(0.0, 1.0);
      r += wCenter * centerGain * ((0.20 * W - dxAbs).clamp(0.0, 1e9) / (0.20 * W + 1e-6));

      // touchdown bonus if centered & slow
      final centered = dxAbs < cfg.acceptDxFrac * W;
      final slow     = vy.abs() < (cfg.acceptVyMax + 0.5); // small cushion vs acceptVyMax
      if (centered && slow) r += wTouch;

      accReward += r;

      steps++;
      framesLeft--;

      // Restart if terminal too early (keep minSteps)
      if (steps < cfg.minSteps && env.status != et.GameStatus.playing) {
        env.reset(seed: rnd.nextInt(1 << 30));
        _initStart(env, rnd);
        framesLeft = 0;
        continue;
      }

      if (env.status != et.GameStatus.playing || steps >= cfg.maxSteps) {
        landed = env.status == et.GameStatus.landed;
        if (decisionRewards.isNotEmpty && accReward.abs() > 0) {
          decisionRewards[decisionRewards.length - 1] += accReward;
        }
        break;
      }
    }

    // === Quality gate (tiered) ===
    double _qualityScore() {
      final L = env.lander;
      final padCx = env.terrain.padCenter.toDouble();
      final W = env.cfg.worldW.toDouble();
      final dxAbs = (L.pos.x - padCx).abs();
      final vyAbs = L.vel.y.abs().toDouble();

      final dxNorm = (dxAbs / (cfg.acceptDxFrac * W)).clamp(0.0, 2.0); // 1.0 at threshold
      final vyNorm = (vyAbs / (cfg.acceptVyMax)).clamp(0.0, 2.0);

      final sDx = (1.0 - 0.5 * dxNorm).clamp(0.0, 1.0);
      final sVy = (1.0 - 0.5 * vyNorm).clamp(0.0, 1.0);
      // emphasize “both good”
      return math.sqrt(sDx * sVy);
    }

    bool landedOk = false;
    {
      final L = env.lander;
      final padCx = env.terrain.padCenter.toDouble();
      final W = env.cfg.worldW.toDouble();
      final dxAbs = (L.pos.x - padCx).abs();
      final vyAbs = L.vel.y.abs().toDouble();

      landedOk = landed && (dxAbs <= cfg.acceptDxFrac * W) && (vyAbs <= 1.3 * cfg.acceptVyMax);
      q = _qualityScore();

      // landed → quality tiers
      final acceptedGreat = landed && q >= 0.70;
      final acceptedOkay  = landed && q >= 0.40 && q < 0.70;

      if (cfg.verbose) {
        final tag = acceptedGreat ? 'ACCEPT[G]'
            : acceptedOkay  ? 'ACCEPT[O]'
            : landedOk      ? 'ACCEPT[loose]'
            : 'REJECT';
//        print('[FINAL] $tag q=${q.toStringAsFixed(2)} '
//            'steps=$steps landed=${landed ? "Y" : "N"}');
      }

// === Quality gate (tiered) ===
      q = _qualityScore();          // as before
      final bool softPositive = q >= 0.35; // NEW: allow non-landed "almost there"

// === Apply update with tiered weights ===
      if (acceptedGreat) {
        policy.updateFromEpisode(
          decisionCaches: decisionCaches,
          intentChoices: intentChoices,
          decisionReturns: decisionReturns,
          alignLabels: alignLabels,
          alignWeight: 2.0, //intentAlignWeight,
          intentPgWeight: 2.0, //intentPgWeight,
          lr: lr,
          entropyBeta: 0.0, valueBeta: 0.0, huberDelta: 1.0,
          intentMode: true,
          actionCaches: actionCaches,
          actionTurnTargets: actionTurnTargets,
          actionThrustTargets: actionThrustTargets,
          actionAlignWeight: actionAlignWeight,
        );
      } else if (acceptedOkay || landedOk || softPositive) { // ← include soft-positive non-landed
        policy.updateFromEpisode(
          decisionCaches: decisionCaches,
          intentChoices: intentChoices,
          decisionReturns: decisionReturns,
          alignLabels: alignLabels,
          alignWeight: 0.35 * intentAlignWeight,     // a bit stronger than before
          intentPgWeight: 0.40 * intentPgWeight,
          lr: 0.7 * lr,
          entropyBeta: 0.0, valueBeta: 0.0, huberDelta: 1.0,
          intentMode: true,
          actionCaches: actionCaches,
          actionTurnTargets: actionTurnTargets,
          actionThrustTargets: actionThrustTargets,
          actionAlignWeight: math.max(0.35, 1.2 * actionAlignWeight),
        );
      } else {
        // FAILS: tiny PG + strong imitation so we still learn something from them
        policy.updateFromEpisode(
          decisionCaches: decisionCaches,             // ← use caches (not empty)
          intentChoices: intentChoices,
          decisionReturns: decisionReturns,
          alignLabels: alignLabels,
          alignWeight: 0.10 * intentAlignWeight,      // NEW: small label nudging
          intentPgWeight: 0.08 * intentPgWeight,      // NEW: tiny PG on fails
          lr: 0.6 * lr,
          entropyBeta: 0.0, valueBeta: 0.0, huberDelta: 1.0,
          intentMode: true,
          actionCaches: actionCaches,
          actionTurnTargets: actionTurnTargets,
          actionThrustTargets: actionThrustTargets,
          actionAlignWeight: math.max(0.45, 1.5 * actionAlignWeight),
        );
      }

      return _FinalOutcome(landed, q, acceptedGreat, acceptedOkay);
    }
  }

  /* ------------------------------------------------------------------------ */
  /*                                   Run                                    */
  /* ------------------------------------------------------------------------ */

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
    final rnd = math.Random(seed ^ 0xF1A1);
    final verboseEvery = 25;
    final summaryEvery = 50;
    final stats = _FinalStats(window: 240, emaAlpha: 0.06); // smoother EMA

    if (gateVerbose) {
      print('[CUR/final] start iters=$iters batch=${cfg.batch} farProb=${cfg.farProb} '
          'accept_dx_frac=${cfg.acceptDxFrac} accept_vy<=${cfg.acceptVyMax}');
    }

    for (int it = 0; it < iters; it++) {
      for (int b = 0; b < cfg.batch; b++) {
        final outcome = _runEpisode(
          env: env,
          fe: fe,
          policy: policy,
          norm: norm,
          rnd: rnd,
          planHold: planHold,
          tempIntent: 0.0,  // FINAL: force greedy intents (no exploration)
          gamma: gamma,
          lr: lr,
          intentAlignWeight: intentAlignWeight,
          intentPgWeight: intentPgWeight,
          actionAlignWeight: actionAlignWeight,
          gateVerbose: gateVerbose,
        );
        stats.add(outcome);
      }

      if (gateVerbose && ((it + 1) % verboseEvery == 0)) {
//        print('[CUR/final] iter=${it + 1}/$iters');
      }

      if (gateVerbose && ((it + 1) % summaryEvery == 0)) {
        final lpW = (100.0 * stats.landPctWindow()).toStringAsFixed(1);
        final lpE = (100.0 * stats.emaLand).toStringAsFixed(1);
        final mqW = stats.meanQWindow().toStringAsFixed(2);
        final mqE = stats.emaQ.toStringAsFixed(2);
        final ep  = stats.episodes;
        final g = stats.acceptG, o = stats.acceptO, r = stats.reject;
        final fitness = (0.7 * stats.emaLand + 0.3 * stats.emaQ).toStringAsFixed(3);
        print('[FINAL/SUM] ep=$ep  land%: win=$lpW  ema=$lpE  '
            'q: win=$mqW ema=$mqE  accepts: G=$g O=$o  reject=$r  fit=$fitness');
      }
    }

    if (gateVerbose) print('[CUR/final] done iters=$iters');
  }
}
