// lib/ai/curriculum/hard_approach.dart
// Micro-stage: “hard approach” extracted from your monolith (same logic, polished)

import 'dart:math' as math;

import '../../engine/types.dart' as et;
import '../../engine/game_engine.dart' as eng;
import '../agent.dart';
import '../nn_helper.dart' as nn;
import 'core.dart';

class HardApproachCfg {
  final int batch;
  final int minSteps;      // do at least this many steps before allowing early termination
  final int warmFrames;    // frames where we don't allow main thrust (to establish descent)
  final double vyMin;      // minimum downward start speed
  final double vyMax;      // maximum downward start speed
  final double hMin;       // minimum spawn height
  final double hMax;       // maximum spawn height
  final double nearPadFrac;// spawn near pad horizontally (fraction of W)
  final bool verbose;

  const HardApproachCfg({
    this.batch = 1,
    this.minSteps = 32,
    this.warmFrames = 12,
    this.vyMin = 28.0,
    this.vyMax = 36.0,
    this.hMin = 120.0,
    this.hMax = 320.0,
    this.nearPadFrac = 0.08,
    this.verbose = true,
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
  }) => HardApproachCfg(
    batch: batch ?? this.batch,
    minSteps: minSteps ?? this.minSteps,
    warmFrames: warmFrames ?? this.warmFrames,
    vyMin: vyMin ?? this.vyMin,
    vyMax: vyMax ?? this.vyMax,
    hMin: hMin ?? this.hMin,
    hMax: hMax ?? this.hMax,
    nearPadFrac: nearPadFrac ?? this.nearPadFrac,
    verbose: verbose ?? this.verbose,
  );
}

class HardApproach extends Curriculum {
  @override
  String get key => 'hardapp';

  HardApproachCfg cfg = const HardApproachCfg();

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
      nearPadFrac:cli.getDouble('hardapp_near_frac',def: 0.08),
      verbose:    cli.getFlag('hardapp_verbose',    def: true),
    );
    return this;
  }

  static void _initStart(eng.GameEngine env, math.Random r, HardApproachCfg c) {
    final padCx = env.terrain.padCenter.toDouble();
    final W = env.cfg.worldW.toDouble();
    final x = (padCx + (r.nextDouble() * 2 - 1) * (c.nearPadFrac * W)).clamp(10.0, W - 10.0);
    final h = (c.hMin + (c.hMax - c.hMin) * r.nextDouble());
    double vy = c.vyMin + (c.vyMax - c.vyMin) * r.nextDouble();
    final vx = (r.nextDouble() * 16.0) - 8.0;
    final gy = env.terrain.heightAt(x);

    env.lander
      ..pos.x = x
      ..pos.y = (gy - h).clamp(0.0, env.cfg.worldH - 10.0)
      ..vel.x = vx
      ..vel.y = vy   // NOTE: positive is downward in this engine
      ..angle = 0.0
      ..fuel = env.cfg.t.maxFuel;
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
  }) {
    policy.trunk.trainMode = true;

    // Per-episode containers (intent + action heads)
    final decisionRewards = <double>[];
    final decisionCaches  = <ForwardCache>[];
    final intentChoices   = <int>[];
    final decisionReturns = <double>[];
    final alignLabels     = <int>[];

    final actionCaches        = <ForwardCache>[];
    final actionTurnTargets   = <int>[];
    final actionThrustTargets = <bool>[];

    env.reset(seed: rnd.nextInt(1 << 30));
    _initStart(env, rnd, cfg);

    int framesLeft = 0;
    int currentIntentIdx = intentToIndex(Intent.descendSlow);
    double accReward = 0.0;

    int steps = 0;
    const int maxSteps = 900;   // hard guard to prevent infinite loops
    bool landed = false;
    double totalCost = 0.0;

    double _vCapLoose(double h) => (0.12 * h + 10.0).clamp(10.0, 30.0);

    while (true) {
      // Refresh intent decision when plan window expires
      if (framesLeft <= 0) {
        var x = fe.extract(env);
        final yTeacher = predictiveIntentLabelAdaptive(env);
        if (norm != null) {
          norm.observe(x);
          x = norm.normalize(x, update: false);
        }
        final (idxGreedy, p, cache) = policy.actIntentGreedy(x);

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

        // compute decision-window advantages
        final T = decisionRewards.length;
        final tmp = List<double>.filled(T, 0.0);
        double G = 0.0;
        for (int i = T - 1; i >= 0; i--) { G = decisionRewards[i] + gamma * G; tmp[i] = G; }
        double mean = 0.0; for (final v in tmp) mean += v; mean /= math.max(1, T);
        double var0 = 0.0; for (final v in tmp) { final d = v - mean; var0 += d * d; }
        var0 = (var0 / math.max(1, T)).clamp(1e-9, double.infinity);
        final std = math.sqrt(var0);
        decisionReturns
          ..clear()
          ..addAll(tmp.map((v) => (v - mean) / std));

        framesLeft = planHold;
      }

      final intent = indexToIntent(currentIntentIdx);
      final useWarm = (steps < cfg.warmFrames);
      final uTeacher = useWarm ? _teacherWarm(intent, env, cfg) : controllerForIntent(intent, env);

      // Step the env (optionally override thrust for loose vertical cap)
      et.StepInfo info;
      if (!useWarm && intent == Intent.descendSlow) {
        final L = env.lander;
        final gy = env.terrain.heightAt(L.pos.x);
        final h  = (gy - L.pos.y).toDouble().clamp(0.0, 1e9);
        final vCap = _vCapLoose(h);
        final vyNext = _vyPredictNoThrust(env, tauReact: 0.6);
        final needUp = (L.vel.y > vCap) || (vyNext > 0.9 * vCap);
        final u2 = et.ControlInput(
          thrust: needUp,
          left: uTeacher.left, right: uTeacher.right,
          sideLeft: uTeacher.sideLeft, sideRight: uTeacher.sideRight,
          downThrust: uTeacher.downThrust,
        );
        info = env.step(1 / 60.0, u2);
      } else {
        info = env.step(1 / 60.0, uTeacher);
      }
      totalCost += info.costDelta;

      // action head supervision
      var xAct = fe.extract(env);
      if (norm != null) xAct = norm.normalize(xAct, update: false);
      final (_, __, ___, ____, cAct) = policy.actGreedy(xAct);
      actionCaches.add(cAct);
      actionTurnTargets.add(uTeacher.left ? 0 : (uTeacher.right ? 2 : 1));
      actionThrustTargets.add(uTeacher.thrust);

      // dense reward: simple speed-min is OK here
      final vx = env.lander.vel.x.toDouble();
      final vy = env.lander.vel.y.toDouble();
      final v = math.sqrt(vx * vx + vy * vy);
      accReward += -0.01 * v;

      // ✅ real progression + decision window countdown
      steps++;
      framesLeft--;

      // Enforce a minimum number of steps before allowing termination
      if (steps < cfg.minSteps && env.status != et.GameStatus.playing) {
        // If we crashed/landed too early, restart this micro-episode in-place.
        env.reset(seed: rnd.nextInt(1 << 30));
        _initStart(env, rnd, cfg);
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

    if (cfg.verbose) {
      final L = env.lander;
      final gx = env.terrain.heightAt(L.pos.x.toDouble());
      final h  = (gx - L.pos.y).toDouble();
      print('[HARDAPP] steps=$steps landed=${landed ? "Y" : "N"} vy=${L.vel.y.toStringAsFixed(1)} h=${h.toStringAsFixed(1)}');
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
    final rnd = math.Random(seed ^ 0xA11A);
    final verboseEvery = 25;

    if (gateVerbose) print('[CUR/hardapp] start iters=$iters batch=${cfg.batch}');
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
          actionAlignWeight: 0.25, // supervise action a bit stronger here
        );
      }
      if (gateVerbose && ((it + 1) % verboseEvery == 0)) {
        print('[CUR/hardapp] iter=${it + 1}/$iters');
      }
    }
    if (gateVerbose) print('[CUR/hardapp] done iters=$iters');
  }
}
