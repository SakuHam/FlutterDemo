import 'dart:math' as math;

import '../../engine/types.dart' as et;
import '../../engine/game_engine.dart' as eng;
import '../../engine/raycast.dart';

import '../agent.dart'; // FeatureExtractorRays, PolicyNetwork, RunningNorm, ForwardCache, intentToIndex, Intent, Trainer
import '../potential_field.dart'; // buildPotentialField, PotentialField
import '../eval.dart' as eval;
import '../policy_io.dart' show savePolicyBundle; // <-- SAVE

import 'core.dart';

class PlannerBCCfg {
  final int iters;          // outer iters
  final int batch;          // episodes per iter
  final double lr;
  final double actionAlignW;
  final double intentAlignW;
  final bool superviseIntent;
  final int rayCount;
  final bool forwardAligned;
  final int attemptsPerTerrain;

  // planner shaping-approx params (mirror your runtime)
  final double vMinClose;
  final double vMaxFar;
  final double alpha;
  final double clampSpeed;

  final double tempIntent;  // just for feature cache forward
  final int planHold;       // keep trunk usage consistent

  // In-curriculum evaluation
  final int evalEvery;      // 0 = disable eval inside PBC
  final int evalEpisodes;   // episodes per eval pulse
  final int evalSeed;       // RNG seed for eval

  // Periodic save
  final int saveEvery;      // 0 = disable periodic saves
  final String savePath;    // path to write bundle

  const PlannerBCCfg({
    this.iters = 2000,
    this.batch = 8,
    this.lr = 3e-4,
    this.actionAlignW = 1.0,
    this.intentAlignW = 0.0,
    this.superviseIntent = false,
    this.rayCount = 180,
    this.forwardAligned = true,
    this.attemptsPerTerrain = 1,
    this.vMinClose = 8.0,
    this.vMaxFar = 90.0,
    this.alpha = 1.2,
    this.clampSpeed = 9999.0,
    this.tempIntent = 1.25,
    this.planHold = 2,

    this.evalEvery = 10,
    this.evalEpisodes = 120,
    this.evalSeed = 1337,

    this.saveEvery = 10,
    this.savePath = 'policy_planner_bc.json',
  });

  PlannerBCCfg copyWith({
    int? iters,
    int? batch,
    double? lr,
    double? actionAlignW,
    double? intentAlignW,
    bool? superviseIntent,
    int? rayCount,
    bool? forwardAligned,
    int? attemptsPerTerrain,
    double? vMinClose,
    double? vMaxFar,
    double? alpha,
    double? clampSpeed,
    double? tempIntent,
    int? planHold,
    int? evalEvery,
    int? evalEpisodes,
    int? evalSeed,
    int? saveEvery,
    String? savePath,
  }) => PlannerBCCfg(
    iters: iters ?? this.iters,
    batch: batch ?? this.batch,
    lr: lr ?? this.lr,
    actionAlignW: actionAlignW ?? this.actionAlignW,
    intentAlignW: intentAlignW ?? this.intentAlignW,
    superviseIntent: superviseIntent ?? this.superviseIntent,
    rayCount: rayCount ?? this.rayCount,
    forwardAligned: forwardAligned ?? this.forwardAligned,
    attemptsPerTerrain: attemptsPerTerrain ?? this.attemptsPerTerrain,
    vMinClose: vMinClose ?? this.vMinClose,
    vMaxFar: vMaxFar ?? this.vMaxFar,
    alpha: alpha ?? this.alpha,
    clampSpeed: clampSpeed ?? this.clampSpeed,
    tempIntent: tempIntent ?? this.tempIntent,
    planHold: planHold ?? this.planHold,
    evalEvery: evalEvery ?? this.evalEvery,
    evalEpisodes: evalEpisodes ?? this.evalEpisodes,
    evalSeed: evalSeed ?? this.evalSeed,
    saveEvery: saveEvery ?? this.saveEvery,
    savePath: savePath ?? this.savePath,
  );
}

class PlannerBC extends Curriculum {
  @override
  String get key => 'planner_bc';

  PlannerBCCfg cfg = const PlannerBCCfg();

  @override
  Curriculum configure(Map<String, String?> kv, Set<String> flags) {
    final cli = CliView(kv, flags);
    cfg = cfg.copyWith(
      iters: cli.getInt('pbc_iters', def: 2000),
      batch: cli.getInt('pbc_batch', def: 8),
      lr: cli.getDouble('pbc_lr', def: 3e-4),
      actionAlignW: cli.getDouble('pbc_action_w', def: 1.0),
      intentAlignW: cli.getDouble('pbc_intent_w', def: 0.0),
      superviseIntent: cli.getFlag('pbc_supervise_intent', def: false),
      rayCount: cli.getInt('pbc_rays', def: 180),
      forwardAligned: cli.getFlag('pbc_forward', def: true),
      attemptsPerTerrain: cli.getInt('pbc_attempts', def: 1),
      vMinClose: cli.getDouble('pbc_vmin', def: 8.0),
      vMaxFar: cli.getDouble('pbc_vmax', def: 90.0),
      alpha: cli.getDouble('pbc_alpha', def: 1.2),
      clampSpeed: cli.getDouble('pbc_vclamp', def: 9999.0),
      tempIntent: cli.getDouble('pbc_int_temp', def: 1.25),
      planHold: cli.getInt('pbc_plan_hold', def: 2),

      // Eval controls
      evalEvery: cli.getInt('pbc_eval_every', def: 10),
      evalEpisodes: cli.getInt('pbc_eval_eps', def: 120),
      evalSeed: cli.getInt('pbc_eval_seed', def: 1337),

      // Save controls
      saveEvery: cli.getInt('pbc_save_every', def: 10),
      savePath: cli.getStr('pbc_save_path', def: 'policy_planner_bc.json'),
    );
    return this;
  }

  // Map desired velocity into simple turn/thrust targets
  (int turnIdx, bool thrust) _discretizeControl({
    required eng.GameEngine env,
    required double svx,
    required double svy,
  }) {
    final vx = env.lander.vel.x.toDouble();
    final vy = env.lander.vel.y.toDouble();
    final dvx = (svx - vx);
    final dvy = (svy - vy);

    int turnIdx;
    const dead = 0.6;
    if (dvx > dead) turnIdx = 2;
    else if (dvx < -dead) turnIdx = 0;
    else turnIdx = 1;

    final needUp = dvy < -1.0 || vy > 6.0;
    final thrust = needUp;

    return (turnIdx, thrust);
  }

  int _intentLabel(eng.GameEngine env) {
    final padCx = env.terrain.padCenter.toDouble();
    final dx = padCx - env.lander.pos.x.toDouble();
    final rightLabel = intentToIndex(Intent.goRight);
    final leftLabel  = intentToIndex(Intent.goLeft);
    return (dx >= 0) ? rightLabel : leftLabel;
  }

  void _evalPulse({
    required eng.GameEngine trainEnv,
    required PolicyNetwork policy,
    required int episodes,
    required int attemptsPerTerrain,
    required int seed,
    required int planHold,
    required double tempIntent,
  }) {
    final e = eng.GameEngine(trainEnv.cfg);
    e.rayCfg = RayConfig(
      rayCount: trainEnv.rayCfg.rayCount,
      includeFloor: false,
      forwardAligned: trainEnv.rayCfg.forwardAligned,
    );

    final trainer = Trainer(
      env: e,
      fe: FeatureExtractorRays(rayCount: e.rayCfg.rayCount),
      policy: policy,
      dt: 1 / 60.0,
      gamma: 0.99,
      seed: seed,
      twoStage: true,
      planHold: planHold,
      tempIntent: tempIntent,
      intentEntropyBeta: 0.0,
      useLearnedController: true,
      blendPolicy: 1.0,
      intentAlignWeight: 0.0,
      intentPgWeight: 0.0,
      actionAlignWeight: 0.0,
      normalizeFeatures: true,
      gateScoreMin: -1e9,
      gateOnlyLanded: false,
      gateVerbose: false,
      externalRewardHook: null,
    );

    eval.evaluateSequential(
      env: e,
      trainer: trainer,
      episodes: episodes,
      seed: seed,
      attemptsPerTerrain: attemptsPerTerrain,
      evalDebug: false,
      evalDebugFailN: 3,
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
    final wasTrain = policy.trunk.trainMode;
    policy.trunk.trainMode = true;

    env.rayCfg = RayConfig(
      rayCount: cfg.rayCount,
      includeFloor: false,
      forwardAligned: cfg.forwardAligned,
    );

    final rnd = math.Random(seed ^ 0xDADA);
    int terrAttempts = 0;
    int currentTerrainSeed = rnd.nextInt(1 << 30);

    for (int it = 0; it < cfg.iters; it++) {
      final actionCaches = <ForwardCache>[];
      final turnTargets  = <int>[];
      final thrustTargets= <bool>[];

      final intentCaches = <ForwardCache>[];
      final intentChoices= <int>[];
      final intentLabels = <int>[];

      for (int b = 0; b < cfg.batch; b++) {
        if (terrAttempts == 0) currentTerrainSeed = rnd.nextInt(1 << 30);
        env.reset(seed: currentTerrainSeed);

        final pf = buildPotentialField(
          env,
          nx: 160, ny: 120,
          iters: 300000,
          omega: 1.7,
          tol: 1e-4,
        );

        final maxSteps = 240;
        for (int t = 0; t < maxSteps; t++) {
          if (t < 2) {
            env.step(1 / 60.0, const et.ControlInput(thrust: false, left: false, right: false));
          }

          var x = fe.extract(
            lander: env.lander,
            terrain: env.terrain,
            worldW: env.cfg.worldW,
            worldH: env.cfg.worldH,
            rays: env.rays,
          );
          if (norm != null) { norm.observe(x); x = norm.normalize(x, update: false); }

          final (_idx, _p, cache) = policy.actIntentGreedy(x);

          final L = env.lander;
          final xw = L.pos.x.toDouble();
          final yw = L.pos.y.toDouble();

          final base = pf.suggestVelocity(
            xw, yw,
            vMinClose: cfg.vMinClose,
            vMaxFar: cfg.vMaxFar,
            alpha: cfg.alpha,
            clampSpeed: cfg.clampSpeed,
          );
          double svx = base.vx;
          double svy = base.vy;

          final padCx = env.terrain.padCenter.toDouble();
          final dxAbs = (xw - padCx).abs();
          final gy = env.terrain.heightAt(xw);
          final h = (gy - yw).toDouble().clamp(0.0, 1000.0);
          final W = env.cfg.worldW.toDouble();
          final tightX = 0.10 * W;
          final ph = math.exp(- (h * h) / (140.0 * 140.0 + 1e-6));
          final px = math.exp(- (dxAbs * dxAbs) / (tightX * tightX + 1e-6));
          final prox = (px * ph).clamp(0.0, 1.0);
          final vMinTouchdown = 2.0;
          final flareLat = (1.0 - 0.90 * prox);
          final flareVer = (1.0 - 0.70 * prox);
          svx *= flareLat; svy *= flareVer;

          final (turnIdx, thrB) = _discretizeControl(env: env, svx: svx, svy: svy);

          actionCaches.add(cache);
          turnTargets.add(turnIdx);
          thrustTargets.add(thrB);

          if (cfg.superviseIntent && cfg.intentAlignW > 0.0) {
            intentCaches.add(cache);
            intentChoices.add(_idx);
            intentLabels.add(_intentLabel(env));
          }

          env.step(1 / 60.0, et.ControlInput(
            thrust: thrB,
            left:  turnIdx == 0,
            right: turnIdx == 2,
          ));
          if (env.status != et.GameStatus.playing) break;
        }

        terrAttempts = (terrAttempts + 1) % cfg.attemptsPerTerrain;
      } // batch

      // Supervised updates
      if (actionCaches.isNotEmpty && cfg.actionAlignW > 0) {
        final returnsDummy = List<double>.filled(actionCaches.length, 0.0);
        policy.updateFromEpisode(
          decisionCaches: const [],
          intentChoices: const [],
          decisionReturns: returnsDummy,
          alignLabels: const [],
          alignWeight: 0.0,
          intentPgWeight: 0.0,
          lr: cfg.lr,
          entropyBeta: 0.0,
          valueBeta: 0.0,
          huberDelta: 1.0,
          intentMode: false,
          actionCaches: actionCaches,
          actionTurnTargets: turnTargets,
          actionThrustTargets: thrustTargets,
          actionAlignWeight: cfg.actionAlignW,
        );
      }

      if (cfg.superviseIntent && intentCaches.isNotEmpty && cfg.intentAlignW > 0) {
        final returnsDummy = List<double>.filled(intentCaches.length, 0.0);
        policy.updateFromEpisode(
          decisionCaches: intentCaches,
          intentChoices: intentChoices,
          decisionReturns: returnsDummy,
          alignLabels: intentLabels,
          alignWeight: cfg.intentAlignW,
          intentPgWeight: 0.0,
          lr: cfg.lr,
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

      if ((it + 1) % 10 == 0) {
        // ignore: avoid_print
        print('[PBC] iter=${it + 1} | actionSL=${actionCaches.length}'
            '${cfg.superviseIntent ? " + intentSL=${intentCaches.length}" : ""}');
      }

      // Eval every cfg.evalEvery iters
      if (cfg.evalEvery > 0 && ((it + 1) % cfg.evalEvery == 0)) {
        // ignore: avoid_print
        print('[PBC/EVAL] it=${it + 1} • episodes=${cfg.evalEpisodes}');
        _evalPulse(
          trainEnv: env,
          policy: policy,
          episodes: cfg.evalEpisodes,
          attemptsPerTerrain: cfg.attemptsPerTerrain,
          seed: cfg.evalSeed ^ (0xE001 ^ (it + 1)),
          planHold: cfg.planHold,
          tempIntent: cfg.tempIntent,
        );
      }

      // Save every cfg.saveEvery iters
      if (cfg.saveEvery > 0 && ((it + 1) % cfg.saveEvery == 0)) {
        try {
          savePolicyBundle(
            path: cfg.savePath,
            p: policy,
            env: env,
            norm: norm,
          );
          // ignore: avoid_print
          print('★ PBC save at iter ${it + 1} → ${cfg.savePath}');
        } catch (e) {
          // ignore: avoid_print
          print('[PBC] save failed at iter ${it + 1}: $e');
        }
      }
    }

    policy.trunk.trainMode = wasTrain;
  }
}
