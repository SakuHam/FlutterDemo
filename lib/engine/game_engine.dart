// lib/engine/game_engine.dart
import 'dart:math' as math;

import 'types.dart';
import 'scoring.dart';

// Helpers
import 'hull_contact.dart'; // ContactResolver, ContactKind
import 'raycast.dart';      // RayCaster, RayConfig, RayHit

/// The Game Engine (physics + shaping cost) with adaptive sub-stepping to prevent tunneling.
class GameEngine {
  EngineConfig cfg;
  late Terrain terrain;
  late LanderState lander;
  GameStatus status = GameStatus.playing;

  math.Random _rnd;
  Terrain? _frozenTerrain;
  double _lastAngle = 0.0;

  // Heuristic assists
  static const double _autoLevelBelow = 140.0;   // px
  static const double _autoLevelGain  = 3.0;     // 1/sec
  static const double _flareBelow     = 40.0;    // px
  static const double _flareGain      = 8.0;     // 1/sec
  static const double _rotFriction    = 0.6;     // rad/sec toward 0

  // Substep limits (anti-tunneling)
  static const double _maxVertexStepPx = 3.0;      // max vertex travel per substep
  static const double _maxAngleStepRad = 7.0 * math.pi / 180.0; // max rotation per substep
  static const int    _maxSubsteps     = 12;       // absolute cap for perf

  // Sensors
  RayCaster _sensors = RayCaster(const RayConfig());
  List<RayHit> _rays = const [];
  List<RayHit> get rays => _rays;
  RayConfig get rayCfg => _sensors.cfg;
  set rayCfg(RayConfig v) => _sensors = RayCaster(v);

  GameEngine(this.cfg) : _rnd = math.Random(cfg.seed) {
    reset(seed: cfg.seed);
  }

  // ---------------- Lifecycle ----------------
  void reset({
    int? seed,
    double? padWidthFactor,
    double? landingSpeedMax,
    double? landingAngleMaxRad,
  }) {
    if (seed != null) _rnd = math.Random(seed);

    // Optional curriculum overrides
    if (padWidthFactor != null || landingSpeedMax != null || landingAngleMaxRad != null) {
      cfg = cfg.copyWith(
        padWidthFactor: padWidthFactor ?? cfg.padWidthFactor,
        landingSpeedMax: landingSpeedMax ?? cfg.landingSpeedMax,
        landingAngleMaxRad: landingAngleMaxRad ?? cfg.landingAngleMaxRad,
      );
    }

    // Terrain
    if (cfg.lockTerrain) {
      _frozenTerrain ??= Terrain.generate(
        cfg.worldW, cfg.worldH, cfg.terrainSeed, cfg.padWidthFactor,
      );
      terrain = _frozenTerrain!;
    } else {
      terrain = Terrain.generate(
        cfg.worldW, cfg.worldH, _rnd.nextInt(1 << 30), cfg.padWidthFactor,
      );
    }

    // Spawn X (random or fixed fraction)
    final double fracX = cfg.randomSpawnX
        ? (() {
      final a = cfg.spawnXMin.clamp(0.0, 1.0);
      final b = cfg.spawnXMax.clamp(0.0, 1.0);
      final lo = math.min(a, b), hi = math.max(a, b);
      return lo + _rnd.nextDouble() * (hi - lo);
    })()
        : cfg.spawnX;

    // Lander init (lockSpawn keeps Y/vel/angle)
    if (cfg.lockSpawn) {
      lander = LanderState(
        pos: Vector2(cfg.worldW * fracX, cfg.spawnY),
        vel: Vector2(cfg.spawnVx, cfg.spawnVy),
        angle: cfg.spawnAngle,
        fuel: cfg.t.maxFuel,
      );
    } else {
      lander = LanderState(
        pos: Vector2(cfg.worldW * fracX, cfg.spawnY),
        vel: Vector2(0, 0),
        angle: 0.0,
        fuel: cfg.t.maxFuel,
      );
    }

    _lastAngle = lander.angle;
    status = GameStatus.playing;

    // Prime sensors
    _rays = _sensors.castAll(
      terrain: terrain,
      engineCfg: cfg,
      origin: lander.pos,
      angle: lander.angle,
    );
  }

  /// Physics step + contact resolution + shaping cost. `dt` in seconds.
  /// Uses adaptive sub-stepping to avoid tunneling through thin terrain edges.
  StepInfo step(double dt, ControlInput u) {
    if (status != GameStatus.playing) {
      return StepInfo(
        costDelta: 0.0, terminal: true, landed: status == GameStatus.landed, onPad: false,
      );
    }

    final t = cfg.t;

    // ----- Controls -> target angular change estimate (for substep sizing) -----
    // Approximate maximum angular rate this frame (used only for substep count).
    double estOmega = 0.0;
    {
      double angle = lander.angle;
      final rot = t.rotSpeed * 0.5;
      if (u.left ^ u.right) estOmega = rot; // turning
      // add a bit for auto-level assist pull
      final gyNow = terrain.heightAt(lander.pos.x);
      final hNow  = (gyNow - lander.pos.y).toDouble();
      if (hNow < _autoLevelBelow) estOmega += _autoLevelGain * (dt.clamp(0.0, 0.05));
      if (hNow < _flareBelow)     estOmega += _flareGain * (dt.clamp(0.0, 0.05));
    }

    // Estimate max vertex travel this frame (rough upper bound): ship center speed +  angular envelope.
    final s = cfg.stepScale;
    final vx = lander.vel.x * dt * s;
    final vy = lander.vel.y * dt * s;
    final centerTravel = math.sqrt(vx*vx + vy*vy);
    const shipRadius = 24.0; // conservative bound from center to any vertex
    final rotTravel  = estOmega * shipRadius * dt;
    final maxTravel  = centerTravel + rotTravel;

    int nPos = (maxTravel / _maxVertexStepPx).ceil();
    int nAng = (estOmega * dt / _maxAngleStepRad).ceil();
    int substeps = math.max(1, math.min(_maxSubsteps, math.max(nPos, nAng)));

    double hRemaining = dt;
    double stepCostAcc = 0.0;
    bool terminal = false;
    bool landedOk = false;
    bool onPad = false;

    // Run substeps
    for (int k = 0; k < substeps; k++) {
      final dtk = hRemaining / (substeps - k);
      hRemaining -= dtk;

      // ----- Controls -> rotation (+ damping) for this substep -----
      double angle = lander.angle;
      final rot = t.rotSpeed * 0.5;
      if (u.left && !u.right) angle -= rot * dtk;
      if (u.right && !u.left) angle += rot * dtk;
      if (!(u.left ^ u.right)) {
        final pull = (_rotFriction * dtk);
        if (pull > 0.0) angle -= angle * pull.clamp(0.0, 0.5);
      }

      // ----- Auto-level & flare assists (using current altitude) -----
      final gyNow = terrain.heightAt(lander.pos.x);
      final hNow  = (gyNow - lander.pos.y).toDouble();
      if (hNow < _autoLevelBelow) {
        final kauto = _autoLevelGain * dtk;
        angle -= angle * kauto.clamp(0.0, 0.5);
      }
      if (hNow < _flareBelow) {
        final kflare = _flareGain * dtk;
        angle -= angle * kflare.clamp(0.0, 0.85);
      }

      // ----- Acceleration & fuel burn -----
      Vector2 accel = Vector2(0, t.gravity * 0.05);
      double fuel = lander.fuel;
      final bool thrusting = u.thrust && fuel > 0.0;
      final double power = thrusting ? 1.0 : 0.0;

      if (thrusting) {
        accel.x += math.sin(angle) * (t.thrustAccel * 0.05);
        accel.y += -math.cos(angle) * (t.thrustAccel * 0.05);
        fuel = (fuel - 20.0 * dtk).clamp(0.0, t.maxFuel);
      }

      // ----- Integrate (semi-implicit Euler) -----
      final vel = Vector2(
        lander.vel.x + accel.x * dtk * s,
        lander.vel.y + accel.y * dtk * s,
      );
      Vector2 pos = Vector2(
        lander.pos.x + vel.x * dtk * s,
        lander.pos.y + vel.y * dtk * s,
      );

      // Late flare snap for this microstep
      final gyPost = terrain.heightAt(pos.x);
      final hPost  = (gyPost - pos.y).toDouble();
      if (hPost < _flareBelow * 0.6) {
        final kf = (_flareGain * 1.35) * dtk;
        angle -= angle * kf.clamp(0.0, 0.95);
      }

      // ----- Contact resolution on this substep -----
      final resolver = ContactResolver(cfg: cfg, terrain: terrain);
      final contact = resolver.resolve(pos, vel, angle);

      bool _gentle(Vector2 v, double ang) {
        final okSpeed = v.length <= cfg.landingSpeedMax;
        final okAngle = !cfg.t.crashOnTilt || ang.abs() <= cfg.landingAngleMaxRad;
        return okSpeed && okAngle;
      }

      if (contact.kind == ContactKind.pad) {
        if (_gentle(contact.vel, angle)) {
          lander = LanderState(pos: contact.pos, vel: Vector2(0, 0), angle: angle, fuel: fuel);
          _lastAngle = angle;
          status = GameStatus.landed;
          terminal = true;
          landedOk = true;
          onPad = true;
          break;
        } else {
          lander = LanderState(pos: contact.pos, vel: contact.vel, angle: angle, fuel: fuel);
          _lastAngle = angle;
          status = GameStatus.crashed;
          terminal = true;
          landedOk = false;
          onPad = true;
          break;
        }
      }

      if (contact.kind == ContactKind.terrain ||
          contact.kind == ContactKind.wall    ||
          contact.kind == ContactKind.ceiling) {
        lander = LanderState(pos: contact.pos, vel: contact.vel, angle: angle, fuel: fuel);
        _lastAngle = angle;
        status = GameStatus.crashed;
        terminal = true;
        landedOk = false;
        onPad = false;
        break;
      }

      // ----- Effort + shaping cost for this substep (no collision authority here) -----
      double stepCost = 0.0;
      if (power > 0) stepCost += cfg.effortCost * power * dtk;

      final score = Scoring.apply(
        cfg: cfg,
        terrain: terrain,
        pos: pos,
        vel: vel,
        angle: angle,
        prevAngle: _lastAngle,
        dt: dtk,
        u: u,
        intentIdx: u.intentIdx,
      );
      stepCost += score.cost; // cost only

      // Commit microstep physics
      lander = LanderState(pos: pos, vel: vel, angle: angle, fuel: fuel);
      _lastAngle = angle;

      stepCostAcc += stepCost;
      // Continue to next microstep
    }

    // Sensors update (once per frame)
    _rays = _sensors.castAll(
      terrain: terrain,
      engineCfg: cfg,
      origin: lander.pos,
      angle: lander.angle,
    );

    // If we ended early due to contact, return terminal StepInfo now
    if (terminal) {
      return StepInfo(
        costDelta: stepCostAcc,
        terminal: true,
        landed: landedOk,
        onPad: onPad,
      );
    }

    // Otherwise still playing
    return StepInfo(
      costDelta: stepCostAcc,
      terminal: false,
      landed: false,
      onPad: false,
    );
  }

  // ---- NN convenience passthroughs ----
  List<double> rayChannels3({bool useProximity = false}) =>
      _sensors.channels3(hits: _rays, engineCfg: cfg, useProximity: useProximity);

  ({double minD, double bearing, double visible}) padSummary() =>
      _sensors.padSummary(hits: _rays, engineCfg: cfg);
}
