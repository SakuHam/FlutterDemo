// lib/ai/curriculum/final_simple.dart
// FINAL (simple): pure behavioral cloning of a deterministic teacher.
// No PG, no gates — supervised action heads until the touchdown reflex is learned.

import 'dart:math' as math;

import '../../engine/types.dart' as et;
import '../../engine/game_engine.dart' as eng;
import '../agent.dart';      // PolicyNetwork, FeatureExtractorRays, RunningNorm, ForwardCache
import 'core.dart';          // Curriculum

class FinalSimpleCfg {
  final int batch;
  final int iters;
  final int minSteps;
  final int maxSteps;
  final double hMin, hMax;           // spawn altitude (y-down; above ground)
  final double xNearFrac, xFarFrac;  // lateral spawn bands as frac of W
  final double vyMin, vyMax;         // y-down: +vy descends
  final double vxNearMax, vxFarMax;
  final double vyTouch;              // desired vy near touchdown (y-down, +)
  final bool verbose;

  const FinalSimpleCfg({
    this.batch = 8,
    this.iters = 600,
    this.minSteps = 24,
    this.maxSteps = 360,
    this.hMin = 70.0,
    this.hMax = 140.0,
    this.xNearFrac = 0.12,
    this.xFarFrac = 0.40,
    this.vyMin = 6.0,
    this.vyMax = 16.0,
    this.vxNearMax = 4.0,
    this.vxFarMax = 12.0,
    this.vyTouch = 3.8,
    this.verbose = true,
  });

  FinalSimpleCfg copyWith({
    int? batch, int? iters, int? minSteps, int? maxSteps,
    double? hMin, double? hMax, double? xNearFrac, double? xFarFrac,
    double? vyMin, double? vyMax, double? vxNearMax, double? vxFarMax,
    double? vyTouch, bool? verbose,
  }) => FinalSimpleCfg(
    batch:     batch     ?? this.batch,
    iters:     iters     ?? this.iters,
    minSteps:  minSteps  ?? this.minSteps,
    maxSteps:  maxSteps  ?? this.maxSteps,
    hMin:      hMin      ?? this.hMin,
    hMax:      hMax      ?? this.hMax,
    xNearFrac: xNearFrac ?? this.xNearFrac,
    xFarFrac:  xFarFrac  ?? this.xFarFrac,
    vyMin:     vyMin     ?? this.vyMin,
    vyMax:     vyMax     ?? this.vyMax,
    vxNearMax: vxNearMax ?? this.vxNearMax,
    vxFarMax:  vxFarMax  ?? this.vxFarMax,
    vyTouch:   vyTouch   ?? this.vyTouch,
    verbose:   verbose   ?? this.verbose,
  );
}

class FinalSimple extends Curriculum {
  @override
  String get key => 'final_simple';

  FinalSimpleCfg cfg = const FinalSimpleCfg();
  int _spawnTick = 0; // 0..7 stratified: near/far × left/right × vyLow/vyHigh

  @override
  Curriculum configure(Map<String, String?> kv, Set<String> flags) {
    final c = CliView(kv, flags);
    cfg = cfg.copyWith(
      batch:     c.getInt('finals_batch',      def: cfg.batch),
      iters:     c.getInt('finals_iters',      def: cfg.iters),
      minSteps:  c.getInt('finals_min',        def: cfg.minSteps),
      maxSteps:  c.getInt('finals_max',        def: cfg.maxSteps),
      hMin:      c.getDouble('finals_hmin',    def: cfg.hMin),
      hMax:      c.getDouble('finals_hmax',    def: cfg.hMax),
      xNearFrac: c.getDouble('finals_x_near',  def: cfg.xNearFrac),
      xFarFrac:  c.getDouble('finals_x_far',   def: cfg.xFarFrac),
      vyMin:     c.getDouble('finals_vy_min',  def: cfg.vyMin),
      vyMax:     c.getDouble('finals_vy_max',  def: cfg.vyMax),
      vxNearMax: c.getDouble('finals_vx_near', def: cfg.vxNearMax),
      vxFarMax:  c.getDouble('finals_vx_far',  def: cfg.vxFarMax),
      vyTouch:   c.getDouble('finals_vy_touch',def: cfg.vyTouch),
      verbose:   c.getFlag('finals_verbose',   def: cfg.verbose),
    );
    return this;
  }

  /* ------------------------------------------------------------------------ */
  /*                                 Spawning                                 */
  /* ------------------------------------------------------------------------ */

  // Stratified spawn near/far × left/right × vyLow/vyHigh (y-down world).
  void _spawn(eng.GameEngine env, math.Random r) {
    final mode = _spawnTick++ & 7;              // 0..7 loop
    final isFar = (mode & 1) == 1;
    final rightSide = (mode & 2) == 2;
    final vyHigh = (mode & 4) == 4;

    final W = env.cfg.worldW.toDouble();
    final padCx = env.terrain.padCenter.toDouble();
    final frac = isFar ? cfg.xFarFrac : cfg.xNearFrac;
    final side = rightSide ? 1.0 : -1.0;

    final xBase = padCx + side * frac * W;
    final x = (xBase + (r.nextDouble() - 0.5) * 0.10 * W).clamp(10.0, W - 10.0);

    final hBucket = vyHigh ? 0.70 : 0.40;
    final h = cfg.hMin + (cfg.hMax - cfg.hMin) *
        (hBucket + (r.nextDouble() - 0.5) * 0.12).clamp(0.0, 1.0);

    // y-down: + vy is descending
    final vyMid = vyHigh
        ? (0.8 * (cfg.vyMin + cfg.vyMax) * 0.5 + 0.5 * cfg.vyMax)
        : (0.6 * (cfg.vyMin + cfg.vyMax) * 0.5 + 0.5 * cfg.vyMin);
    final vy = (vyMid + (r.nextDouble() - 0.5) * 4.0).clamp(cfg.vyMin, cfg.vyMax);

    final vxCap = isFar ? cfg.vxFarMax : cfg.vxNearMax;
    final toCenter = rightSide ? -1.0 : 1.0;
    final vx = toCenter * (0.6 * vxCap + r.nextDouble() * 0.4 * vxCap);

    final gy = env.terrain.heightAt(x);

    env.lander
      ..pos.x = x
      ..pos.y = (gy - h).clamp(0.0, env.cfg.worldH - 10.0) // y-down: above ground
      ..vel.x = vx
      ..vel.y = vy
      ..angle = 0.0
      ..fuel = env.cfg.t.maxFuel;

    env.status = et.GameStatus.playing;
  }

  /* ------------------------------------------------------------------------ */
  /*                            Teacher & targets                              */
  /* ------------------------------------------------------------------------ */

  // Height-aware vertical target (y-down).
  double _vyTarget(double h) {
    if (h > 120) return 34.0;
    if (h > 80)  return 26.0 + 0.20 * (h - 80.0);
    if (h > 40)  return 14.0 + 0.30 * (h - 40.0);
    if (h > 20)  return  8.0 + 0.20 * (h - 20.0);
    return cfg.vyTouch; // ~3.8
  }

  double _latGain(double h) {
    if (h > 120) return 0.10;
    if (h > 80)  return 0.08;
    if (h > 40)  return 0.070;
    return 0.065;
  }

  // Deterministic teacher (y-down world).
  et.ControlInput _teacher(eng.GameEngine env) {
    final L = env.lander;
    final padCx = env.terrain.padCenter.toDouble();
    final W = env.cfg.worldW.toDouble();

    final gy = env.terrain.heightAt(L.pos.x);
    final h  = (gy - L.pos.y).toDouble().clamp(0.0, 1e9);

    final dx = (L.pos.x - padCx).toDouble();
    final vx = L.vel.x.toDouble();
    final vy = L.vel.y.toDouble();

    // Lateral target
    final kx = _latGain(h);
    final vLatMaxBase = (30.0 + 0.28 * h).clamp(22.0, 96.0);
    final farScale = (dx.abs() / (cfg.xFarFrac * W)).clamp(0.0, 1.5);
    final vLatMax = (vLatMaxBase * (1.0 + 0.6 * farScale)).clamp(22.0, 128.0);
    final vTx = (-kx * dx).clamp(-vLatMax, vLatMax);

    final errX = vx - vTx;
    final dz = (h > 60.0) ? 2.0 : 1.0;
    final left  = errX >  dz;
    final right = errX < -dz;

    // Vertical target with earlier brake-up low
    final vyT = _vyTarget(h);
    final centerTight = (dx.abs() < 0.10 * W) ? 0.7 : 0.0;
    final upMargin = (h > 40.0) ? (0.7 + 0.015 * h) : (0.55 + 0.010 * h);
    final thrust = (vy > (vyT - centerTight) + upMargin);

    // Tiny cone bias near ground
    bool sideLeft = false, sideRight = false;
    if (h <= 28.0 && dx.abs() > 0.06 * W) {
      final cone = ((h - 10.0) / 18.0).clamp(0.0, 1.0);
      if (dx > 0) sideLeft = (cone > 0.0);
      if (dx < 0) sideRight = (cone > 0.0);
    }

    return et.ControlInput(
      thrust: thrust,
      left: left, right: right,
      sideLeft: sideLeft, sideRight: sideRight,
      downThrust: false,
    );
  }

  /* ------------------------------------------------------------------------ */
  /*                             Supervised episode                            */
  /* ------------------------------------------------------------------------ */

  _BCOutcome _runEpisode({
    required eng.GameEngine env,
    required FeatureExtractorRays fe,
    required PolicyNetwork policy,
    required RunningNorm? norm,
    required math.Random rnd,
    void Function(int turnIdx, bool thrustOn)? teacherHook, // for logging teacher distribution
  }) {
    policy.trunk.trainMode = true;

    final actionCaches = <ForwardCache>[];
    final turnTargets  = <int>[];   // 0=left, 1=none, 2=right (teacher label order)
    final thrustTargets= <bool>[];

    int steps = 0;
    bool landed = false;

    env.reset(seed: rnd.nextInt(1 << 30));
    _spawn(env, rnd);

    while (true) {
      // Teacher label
      final u = _teacher(env);

      // Extract features
      var x = fe.extract(
          lander: env.lander, terrain: env.terrain,
          worldW: env.cfg.worldW, worldH: env.cfg.worldH, rays: env.rays
      );
      if (norm != null) { norm.observe(x); x = norm.normalize(x, update: false); }

      // Forward pass (we only need cache for supervised update)
      final (_, __, ___, ____, cacheAct) = policy.actGreedy(x);

      // Targets from teacher
      final turnTgt   = u.left ? 0 : (u.right ? 2 : 1);
      final thrustTgt = u.thrust;

      // Collect caches/targets
      actionCaches.add(cacheAct);
      turnTargets.add(turnTgt);
      thrustTargets.add(thrustTgt);

      // Optional: log teacher distribution
      if (teacherHook != null) teacherHook(turnTgt, thrustTgt);

      // Step env with teacher control
      final _ = env.step(1 / 60.0, u);

      steps++;
      if (env.status != et.GameStatus.playing || steps >= cfg.maxSteps) break;

      // If terminal too early, restart to ensure minimum frames for supervision
      if (steps < cfg.minSteps && env.status != et.GameStatus.playing) {
        env.reset(seed: rnd.nextInt(1 << 30));
        _spawn(env, rnd);
      }
    }

    landed = env.status == et.GameStatus.landed;

    // Supervised update (no PG, no intent head in this phase)
    policy.updateFromEpisode(
      decisionCaches: const [], intentChoices: const [],
      decisionReturns: const [], alignLabels: const [],
      alignWeight: 0.0, intentPgWeight: 0.0,
      lr: 1.0, entropyBeta: 0.0, valueBeta: 0.0,
      huberDelta: 1.0, intentMode: false,        // <— disable intent head update here
      actionCaches: actionCaches,
      actionTurnTargets: turnTargets,
      actionThrustTargets: thrustTargets,
      actionAlignWeight: 1.0,                    // strong imitation
    );

    return _BCOutcome(landed: landed);
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
    required int planHold,              // ignored in BC
    required double tempIntent,         // ignored in BC
    required double gamma,              // ignored in BC
    required double lr,
    required double intentAlignWeight,  // ignored in BC
    required double intentPgWeight,     // ignored in BC
    required double actionAlignWeight,  // scales CE loss (we keep 1.0 internally)
    required bool gateVerbose,
    required int seed,
  }) async {
    final rnd = math.Random(seed ^ 0xBEEFCAFE);
    final stats = _Roll(window: 200);

    if (cfg.verbose) {
      print('[FINALS/BC] start iters=${cfg.iters} batch=${cfg.batch} '
          'h=[${cfg.hMin}-${cfg.hMax}] vy=[${cfg.vyMin}-${cfg.vyMax}] '
          'vxNearMax=${cfg.vxNearMax} vxFarMax=${cfg.vxFarMax}');
    }

    for (int it = 0; it < (cfg.iters); it++) {
      for (int b = 0; b < cfg.batch; b++) {
        final out = _runEpisode(
          env: env, fe: fe, policy: policy, norm: norm, rnd: rnd,
          teacherHook: (turnIdx, thrustOn) => stats.addTeacherSample(turnIdx, thrustOn),
        );
        stats.add(out.landed);
      }
      if (cfg.verbose && ((it + 1) % 25 == 0)) {
        final lpW  = (100 * stats.landWin()).toStringAsFixed(1);
        final lpE  = (100 * stats.landEma).toStringAsFixed(1);
        final d = stats.teacherDist();
        final thrPct = (100 * d.thrustOn).toStringAsFixed(1);
        print('[FINALS/BC] it=${it + 1}/${cfg.iters}  land%: win=$lpW ema=$lpE  '
            'teacher(turn) L:${d.left} N:${d.none} R:${d.right}  '
            'teacher(thrust_on)=${thrPct}%');
      }
    }
  }
}

class _BCOutcome {
  final bool landed;
  _BCOutcome({required this.landed});
}

class _Roll {
  final int window;
  final List<bool> land;

  // Teacher label distribution sampling
  int tLeft = 0, tNone = 0, tRight = 0, tThrustOn = 0, tTotal = 0;

  int i = 0, n = 0;
  double landEma = 0.0;
  final double a = 0.08;

  _Roll({this.window = 200})
      : land = List<bool>.filled(window, false);

  void add(bool l) {
    land[i] = l;
    i = (i + 1) % window; n++;
    landEma = (1 - a) * landEma + a * (l ? 1.0 : 0.0);
  }

  void addTeacherSample(int turnIdx, bool thrustOn) {
    if (turnIdx == 0) tLeft++;
    else if (turnIdx == 1) tNone++;
    else if (turnIdx == 2) tRight++;
    tThrustOn += thrustOn ? 1 : 0;
    tTotal++;
  }

  ({int left, int none, int right, double thrustOn}) teacherDist() {
    final tot = tTotal == 0 ? 1 : tTotal;
    return (left: tLeft, none: tNone, right: tRight, thrustOn: tThrustOn / tot);
  }

  double landWin() {
    final m = n < window ? n : window;
    if (m == 0) return 0.0;
    int c = 0;
    for (int k = 0; k < m; k++) if (land[k]) c++;
    return c / m;
  }
}
