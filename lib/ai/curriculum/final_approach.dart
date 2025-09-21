// lib/ai/curriculum/final_approach.dart
// Final approach micro-stage: pad-centric touchdown with quality gating.
// Spawns both near the pad and (sometimes) far to practice late recentering.

import 'dart:math' as math;

import '../../engine/types.dart' as et;
import '../../engine/game_engine.dart' as eng;
import '../agent.dart';
import '../nn_helper.dart' as nn;
import 'core.dart';

class FinalApproachCfg {
  // Episode layout
  final int batch;
  final int warmFrames;          // initial conservative frames
  final int minSteps;            // enforce minimum episode length
  final int maxSteps;            // hard cap to prevent infinite loops

  // Spawn ranges
  final double hMin;
  final double hMax;
  final double xNearFrac;        // near-pad lateral offset (fraction of W)
  final double xFarFrac;         // far lateral offset (fraction of W)
  final double farProb;          // probability to use far start
  final double vyMin;
  final double vyMax;
  final double vxNearMax;        // lateral speed cap near-start
  final double vxFarMax;         // lateral speed cap far-start

  // Touchdown target
  final double vyTouchdown;      // desire slow vertical at touchdown

  // Acceptance (quality gate)
  final double acceptDxFrac;     // must be within this fraction of W from pad center
  final double acceptVyMax;      // max vertical speed at touchdown to accept
  final bool   gateRejectSkipsUpdate; // if true, skip update entirely on reject

  // Logging
  final bool verbose;

  const FinalApproachCfg({
    // episode
    this.batch = 1,
    this.warmFrames = 10,
    this.minSteps = 24,
    this.maxSteps = 900,

    // spawn
    this.hMin = 70.0,
    this.hMax = 160.0,
    this.xNearFrac = 0.12,
    this.xFarFrac = 0.40,
    this.farProb = 0.35,
    this.vyMin = 8.0,
    this.vyMax = 22.0,
    this.vxNearMax = 5.0,
    this.vxFarMax = 20.0,

    // target
    this.vyTouchdown = 3.0,

    // acceptance
    this.acceptDxFrac = 0.06,
    this.acceptVyMax = 3.5,
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
    batch: batch ?? this.batch,
    warmFrames: warmFrames ?? this.warmFrames,
    minSteps: minSteps ?? this.minSteps,
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
    vyTouchdown: vyTouchdown ?? this.vyTouchdown,
    acceptDxFrac: acceptDxFrac ?? this.acceptDxFrac,
    acceptVyMax: acceptVyMax ?? this.acceptVyMax,
    gateRejectSkipsUpdate: gateRejectSkipsUpdate ?? this.gateRejectSkipsUpdate,
    verbose: verbose ?? this.verbose,
  );
}

class FinalApproach extends Curriculum {
  @override
  String get key => 'final';

  FinalApproachCfg cfg = const FinalApproachCfg();

  @override
  Curriculum configure(Map<String, String?> kv, Set<String> flags) {
    final c = CliView(kv, flags);
    cfg = cfg.copyWith(
      batch:                 c.getInt('final_batch',        def: 1),
      warmFrames:            c.getInt('final_warm_frames',  def: 10),
      minSteps:              c.getInt('final_min_steps',    def: 24),
      maxSteps:              c.getInt('final_max_steps',    def: 900),
      hMin:                  c.getDouble('final_hmin',      def: 70.0),
      hMax:                  c.getDouble('final_hmax',      def: 160.0),
      xNearFrac:             c.getDouble('final_x_near',    def: 0.12),
      xFarFrac:              c.getDouble('final_x_far',     def: 0.40),
      farProb:               c.getDouble('final_far_prob',  def: 0.35),
      vyMin:                 c.getDouble('final_vy_min',    def: 8.0),
      vyMax:                 c.getDouble('final_vy_max',    def: 22.0),
      vxNearMax:             c.getDouble('final_vx_near',   def: 5.0),
      vxFarMax:              c.getDouble('final_vx_far',    def: 20.0),
      vyTouchdown:           c.getDouble('final_vy_touch',  def: 3.0),
      acceptDxFrac:          c.getDouble('final_accept_dx_frac', def: 0.06),
      acceptVyMax:           c.getDouble('final_accept_vy', def: 3.5),
      gateRejectSkipsUpdate: c.getFlag('final_gate_skip_update', def: true),
      verbose:               c.getFlag('final_verbose',     def: true),
    );
    return this;
  }

  // Spawn near or far relative to pad center.
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

    // Lateral target speed scales with |dx| & height; stronger when far.
    final farScale = (dx.abs() / (cfg.xFarFrac * W)).clamp(0.0, 1.5);
    final vLatMaxBase = (30.0 + 0.35 * h).clamp(20.0, 110.0);
    final vLatMax = (vLatMaxBase * (1.0 + 0.8 * farScale)).clamp(20.0, 160.0);

    final vTargetX = (-dx).clamp(-vLatMax, vLatMax);

    final errX = (vx - vTargetX);
    // deadzone +/-2, otherwise turn toward reducing errX
    final turnLeft  = errX >  2.0;
    final turnRight = errX < -2.0;

    // Vertical: cap vy near target touchdown; the closer/centered we are, the stricter
    final centerTight = (dx.abs() < 0.10 * W) ? 1.0 : 0.0;
    final vCap = (0.09 * h + 8.0 - 2.0 * centerTight).clamp(6.0, 24.0);
    final wantUp = vy > math.max(vCap, cfg.vyTouchdown);

    return et.ControlInput(
      thrust: wantUp,
      left: turnLeft,
      right: turnRight,
      sideLeft: false,
      sideRight: false,
      downThrust: false,
    );
  }

  EpisodeResult _runEpisode({
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

    // shaping weights
    const wDx = 0.0030;  // lateral distance penalty
    const wVy = 0.0030;  // vertical speed penalty
    const wCenter = 0.0015; // soft reward for being centered (decays with height)
    const wTouch = 2.2;  // touchdown bonus when centered & slow

    while (true) {
      if (framesLeft <= 0) {
        var x = fe.extract(env);
        final yTeacher = predictiveIntentLabelAdaptive(env);
        if (norm != null) {
          norm.observe(x);
          x = norm.normalize(x, update: false);
        }
        final (idxGreedy, p, cache) = policy.actIntentGreedy(x);

        // Soft sample to keep exploration alive
        int pick;
        if (tempIntent <= 1e-6) {
          pick = idxGreedy;
        } else {
          final z = p.map((pp) => math.log(pp.clamp(1e-12, 1.0))).toList();
          for (int i = 0; i < z.length; i++) z[i] /= tempIntent;
          final sm = nn.Ops.softmax(z);
          final u = rnd.nextDouble();
          double acc = 0.0; pick = sm.length - 1;
          for (int i = 0; i < sm.length; i++) { acc += sm[i]; if (u <= acc) { pick = i; break; } }
        }
        currentIntentIdx = pick;

        decisionCaches.add(cache);
        intentChoices.add(pick);
        alignLabels.add(yTeacher);
        decisionRewards.add(accReward);
        accReward = 0.0;

        // advantage over the current decision windows
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

      // Teacher (same both phases, conservative enough)
      final u = _teacher(env);
      final info = env.step(1 / 60.0, u);
      totalCost += info.costDelta;

      // Action supervision
      var xAct = fe.extract(env);
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
      final vyAbs = L.vel.y.abs().toDouble();

      double r = 0.0;
      r -= wDx * dxAbs;
      r -= wVy * vyAbs;

      // subtle centering lift that grows near ground
      final centerGain = (1.0 - (h / (cfg.hMax + 1e-6))).clamp(0.0, 1.0);
      r += wCenter * centerGain * ( (0.20 * W - dxAbs).clamp(0.0, 1e9) / (0.20 * W + 1e-6) );

      // touchdown bonus if centered & slow
      final centered = dxAbs < cfg.acceptDxFrac * W;
      final slow     = vyAbs < cfg.vyTouchdown + 0.8;
      if (centered && slow) r += wTouch;

      accReward += r;

      steps++;
      framesLeft--;

      // If terminal too early, restart episode once we hit minSteps
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

    // === Quality gate: accept ONLY centered + soft touchdowns ===
    // === Quality gate (tiered) ===
    // Score in [0,1]: 1 when centered+slow; decays with |dx| and |vy|.
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

    bool acceptedGreat = false, acceptedOkay = false, landedOk = false;
    double q = 0.0;
    {
      final L = env.lander;
      final padCx = env.terrain.padCenter.toDouble();
      final W = env.cfg.worldW.toDouble();
      final dxAbs = (L.pos.x - padCx).abs();
      final vyAbs = L.vel.y.abs().toDouble();

      landedOk = landed && (dxAbs <= cfg.acceptDxFrac * W) && (vyAbs <= 1.3 * cfg.acceptVyMax);
      q = _qualityScore();

      acceptedGreat = landed && q >= 0.70;
      acceptedOkay  = landed && q >= 0.40 && q < 0.70;

      if (cfg.verbose) {
        final tag = acceptedGreat ? 'ACCEPT[G]'
            : acceptedOkay  ? 'ACCEPT[O]'
            : landedOk      ? 'ACCEPT[loose]'
            : 'REJECT';
        print('[FINAL] $tag q=${q.toStringAsFixed(2)} '
            'steps=$steps landed=${landed ? "Y" : "N"}');
      }
    }

    // === Apply update with tiered weights ===
    if (acceptedGreat) {
      // full PG + action supervision
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
    } else if (acceptedOkay || landedOk || !cfg.gateRejectSkipsUpdate) {
      // softened PG + stronger imitation to encourage centering
      policy.updateFromEpisode(
        decisionCaches: decisionCaches,
        intentChoices: intentChoices,
        decisionReturns: decisionReturns,
        alignLabels: alignLabels,
        alignWeight: 0.3 * intentAlignWeight,
        intentPgWeight: 0.35 * intentPgWeight,
        lr: 0.6 * lr,
        entropyBeta: 0.0,
        valueBeta: 0.0,
        huberDelta: 1.0,
        intentMode: true,
        actionCaches: actionCaches,
        actionTurnTargets: actionTurnTargets,
        actionThrustTargets: actionThrustTargets,
        actionAlignWeight: math.max(0.25, 1.2 * actionAlignWeight),
      );
    } else {
      // hard reject → imitation-only (no PG)
      policy.updateFromEpisode(
        decisionCaches: const [],
        intentChoices: const [],
        decisionReturns: const [],
        alignLabels: const [],
        alignWeight: 0.0,
        intentPgWeight: 0.0,
        lr: lr,
        entropyBeta: 0.0,
        valueBeta: 0.0,
        huberDelta: 1.0,
        intentMode: true,
        actionCaches: actionCaches,
        actionTurnTargets: actionTurnTargets,
        actionThrustTargets: actionThrustTargets,
        actionAlignWeight: math.max(0.35, 1.5 * actionAlignWeight),
      );
    }

    return EpisodeResult(steps: steps, totalCost: totalCost, landed: landed, segMean: 0.0);
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
    final rnd = math.Random(seed ^ 0xF1A1);
    final verboseEvery = 25;

    if (gateVerbose) {
      print('[CUR/final] start iters=$iters batch=${cfg.batch} farProb=${cfg.farProb} '
          'accept_dx_frac=${cfg.acceptDxFrac} accept_vy<=${cfg.acceptVyMax}');
    }

    for (int it = 0; it < iters; it++) {
      for (int b = 0; b < cfg.batch; b++) {
        _runEpisode(
          env: env,
          fe: fe,
          policy: policy,
          norm: norm,
          rnd: rnd,
          planHold: planHold,
          tempIntent: tempIntent,
          gamma: gamma,
          lr: lr,
          intentAlignWeight: intentAlignWeight,
          intentPgWeight: intentPgWeight,
          actionAlignWeight: actionAlignWeight,
          gateVerbose: gateVerbose,
        );
      }
      if (gateVerbose && ((it + 1) % verboseEvery == 0)) {
        print('[CUR/final] iter=${it + 1}/$iters');
      }
    }

    if (gateVerbose) print('[CUR/final] done iters=$iters');
  }
}
