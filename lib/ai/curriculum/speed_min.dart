// Stage 1: speed-min on hard-set approach vectors (your previous curriculum condensed)

import 'dart:math' as math;

import '../../engine/types.dart' as et;
import '../../engine/game_engine.dart' as eng;
import '../agent.dart';
import 'core.dart';

class SpeedMinCurriculum extends Curriculum {
  @override
  String get key => 'speedmin';

  int _batch = 1;
  int _minSteps = 24;
  bool _verbose = true;

  @override
  Curriculum configure(Map<String, String?> kv, Set<String> flags) {
    final cli = CliView(kv, flags);
    _batch = cli.getInt('curriculum_batch', def: 1);
    _minSteps = cli.getInt('cur_min_steps', def: 24);
    _verbose = cli.getFlag('cur_verbose', def: true);
    return this;
  }

  // Simple “speed penalty” dense shaping
  static double _speedPenalty(eng.GameEngine env) {
    final vx = env.lander.vel.x.toDouble();
    final vy = env.lander.vel.y.toDouble();
    final v = math.sqrt(vx * vx + vy * vy);
    return -0.01 * v;
  }

  // Your previous initializer — spawns somewhere reasonable and points down.
  static void _initStart(eng.GameEngine env, math.Random r) {
    final padCx = env.terrain.padCenter.toDouble();
    final W = env.cfg.worldW.toDouble();

    final x = (padCx + (r.nextDouble() * 2 - 1) * (0.20 * W)).clamp(10.0, W - 10.0);
    final h = (100.0 + 260.0 * r.nextDouble());
    final vx = (r.nextDouble() * 24.0) - 12.0;
    final vy = 24.0 + 22.0 * r.nextDouble(); // downward positive

    final gy = env.terrain.heightAt(x);
    env.lander
      ..pos.x = x
      ..pos.y = (gy - h).clamp(0.0, env.cfg.worldH - 10.0)
      ..vel.x = vx
      ..vel.y = vy
      ..angle = 0.0
      ..fuel = env.cfg.t.maxFuel;
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
    final rnd = math.Random(seed ^ 0xABCD01);
    policy.trunk.trainMode = true;

    for (int it = 0; it < iters; it++) {
      for (int b = 0; b < _batch; b++) {
        env.reset(seed: rnd.nextInt(1 << 30));
        _initStart(env, rnd);

        int steps = 0;
        double totalReward = 0.0;

        // minimal teacher: choose descendSlow or brakeUp depending on vcap
        for (int it = 0; it < iters; it++) {
          // intent step every planHold frames
          final xVec = (norm != null) ? norm.normalize(fe.extract(lander: env.lander, terrain: env.terrain, worldW: env.cfg.worldW, worldH: env.cfg.worldH, rays: env.rays), update: true) : fe.extract(lander: env.lander, terrain: env.terrain, worldW: env.cfg.worldW, worldH: env.cfg.worldH, rays: env.rays);
          final (idxGreedy, p, cacheIntent) = policy.actIntentGreedy(xVec);

          final intent = indexToIntent(idxGreedy);
          var u = controllerForIntent(intent, env);

          final info = env.step(1 / 60.0, u);
          totalReward += _speedPenalty(env);

          // supervise action head weakly (optional)
          final xAct = (norm != null) ? norm.normalize(fe.extract(lander: env.lander, terrain: env.terrain, worldW: env.cfg.worldW, worldH: env.cfg.worldH, rays: env.rays), update: false) : fe.extract(lander: env.lander, terrain: env.terrain, worldW: env.cfg.worldW, worldH: env.cfg.worldH, rays: env.rays);
          final (_, __, ___, ____ , cacheAct) = policy.actGreedy(xAct);
          policy.updateFromEpisode(
            decisionCaches: [cacheIntent],
            intentChoices: [idxGreedy],
            decisionReturns: [totalReward],
            alignLabels: [predictiveIntentLabelAdaptive(env)],
            alignWeight: intentAlignWeight,
            intentPgWeight: intentPgWeight,
            lr: lr,
            entropyBeta: 0.0,
            valueBeta: 0.0,
            huberDelta: 1.0,
            intentMode: true,
            actionCaches: [cacheAct],
            actionTurnTargets: [u.left ? 0 : (u.right ? 2 : 1)],
            actionThrustTargets: [u.thrust],
            actionAlignWeight: actionAlignWeight,
          );

          steps++;
          if ((env.status != et.GameStatus.playing && steps >= _minSteps) || steps > 600) break;
          // short rollouts — curriculum flavor

        }
      }
      if (_verbose && ((it + 1) % 100 == 0)) {
        print('[CUR/speedmin] iter=${it + 1}/$iters');
      }
    }
  }
}
