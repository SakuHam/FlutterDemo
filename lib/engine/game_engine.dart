// lib/engine/game_engine.dart
import 'dart:math' as math;

import 'types.dart';
import 'scoring.dart';

// Helpers
import 'hull_contact.dart';
import 'raycast.dart';

/// The Game Engine (physics + Scoring) — now thin and readable.
class GameEngine {
  EngineConfig cfg;
  late Terrain terrain;
  late LanderState lander;
  GameStatus status = GameStatus.playing;

  math.Random _rnd;
  Terrain? _frozenTerrain;
  double _lastAngle = 0.0;

  // Heuristic assists (unchanged)
  static const double _autoLevelBelow = 140.0;   // px
  static const double _autoLevelGain = 3.0;      // 1/sec
  static const double _flareBelow = 40.0;        // px
  static const double _flareGain = 8.0;          // 1/sec
  static const double _rotFriction = 0.6;        // rad/sec toward 0

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

    if (padWidthFactor != null || landingSpeedMax != null || landingAngleMaxRad != null) {
      cfg = cfg.copyWith(
        padWidthFactor: padWidthFactor ?? cfg.padWidthFactor,
        landingSpeedMax: landingSpeedMax ?? cfg.landingSpeedMax,
        landingAngleMaxRad: landingAngleMaxRad ?? cfg.landingAngleMaxRad,
      );
    }

    // Terrain
    if (cfg.lockTerrain) {
      _frozenTerrain ??= Terrain.generate(cfg.worldW, cfg.worldH, cfg.terrainSeed, cfg.padWidthFactor);
      terrain = _frozenTerrain!;
    } else {
      terrain = Terrain.generate(cfg.worldW, cfg.worldH, _rnd.nextInt(1 << 30), cfg.padWidthFactor);
    }

    // Spawn X
    final fracX = cfg.randomSpawnX
        ? (() {
      final a = cfg.spawnXMin.clamp(0.0, 1.0);
      final b = cfg.spawnXMax.clamp(0.0, 1.0);
      final lo = math.min(a, b), hi = math.max(a, b);
      return lo + _rnd.nextDouble() * (hi - lo);
    })()
        : cfg.spawnX;

    // Lander init
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

  /// Physics + (closest-point) contact + scoring. dt is seconds.
  StepInfo step(double dt, ControlInput u) {
    if (status != GameStatus.playing) {
      return StepInfo(costDelta: 0.0, terminal: true, landed: status == GameStatus.landed, onPad: false);
    }

    final s = cfg.stepScale;
    final t = cfg.t;

    // ----- Rotation from controls + damping -----
    double angle = lander.angle;
    final rot = t.rotSpeed * 0.5;
    if (u.left && !u.right) angle -= rot * dt;
    if (u.right && !u.left) angle += rot * dt;
    if (!(u.left ^ u.right)) {
      final pull = (_rotFriction * dt);
      if (pull > 0.0) angle -= angle * pull.clamp(0.0, 0.5);
    }

    // ----- Auto-level + flare assist (altitude-aware) -----
    final gyNow = terrain.heightAt(lander.pos.x);
    final hNow = (gyNow - lander.pos.y).toDouble();
    if (hNow < _autoLevelBelow) {
      final k = _autoLevelGain * dt;
      angle -= angle * k.clamp(0.0, 0.5);
    }
    if (hNow < _flareBelow) {
      final k = _flareGain * dt;
      angle -= angle * k.clamp(0.0, 0.85);
    }

    // ----- Accel & fuel -----
    Vector2 accel = Vector2(0, t.gravity * 0.05);
    double fuel = lander.fuel;
    final thrusting = u.thrust && fuel > 0.0;
    final double power = thrusting ? 1.0 : 0.0;

    if (thrusting) {
      accel.x += math.sin(angle) * (t.thrustAccel * 0.05);
      accel.y += -math.cos(angle) * (t.thrustAccel * 0.05);
      fuel = (fuel - 20.0 * dt).clamp(0.0, t.maxFuel);
    }

    // ----- Integrate (semi-implicit Euler) -----
    final vel = Vector2(lander.vel.x + accel.x * dt * s,
        lander.vel.y + accel.y * dt * s);
    Vector2 pos = Vector2(lander.pos.x + vel.x * dt * s,
        lander.pos.y + vel.y * dt * s);

    // Late flare snap
    final gyPost = terrain.heightAt(pos.x);
    final hPost = (gyPost - pos.y).toDouble();
    if (hPost < _flareBelow * 0.6) {
      final k = (_flareGain * 1.35) * dt;
      angle -= angle * k.clamp(0.0, 0.95);
    }

    // ----- Contact resolution (closest-point) -----
    final resolver = ContactResolver(cfg: cfg, terrain: terrain);
    final contact = resolver.resolve(pos, vel, angle);

    bool gentle(Vector2 v, double ang) {
      final okSpeed = v.length <= cfg.landingSpeedMax;
      final okAngle = !cfg.t.crashOnTilt || ang.abs() <= cfg.landingAngleMaxRad;
      return okSpeed && okAngle;
    }

    if (contact.kind == ContactKind.pad) {
      if (gentle(contact.vel, angle)) {
        lander = LanderState(pos: contact.pos, vel: Vector2(0, 0), angle: angle, fuel: fuel);
        _lastAngle = angle;
        status = GameStatus.landed;
        _rays = _sensors.castAll(terrain: terrain, engineCfg: cfg, origin: lander.pos, angle: lander.angle);
        return StepInfo(costDelta: 0.0, terminal: true, landed: true, onPad: true);
      } else {
        lander = LanderState(pos: contact.pos, vel: contact.vel, angle: angle, fuel: fuel);
        _lastAngle = angle;
        status = GameStatus.crashed;
        _rays = _sensors.castAll(terrain: terrain, engineCfg: cfg, origin: lander.pos, angle: lander.angle);
        return StepInfo(costDelta: 0.0, terminal: true, landed: false, onPad: true);
      }
    }

    if (contact.kind == ContactKind.terrain ||
        contact.kind == ContactKind.wall ||
        contact.kind == ContactKind.ceiling) {
      lander = LanderState(pos: contact.pos, vel: contact.vel, angle: angle, fuel: fuel);
      _lastAngle = angle;
      status = GameStatus.crashed; // change here if you want sliding
      _rays = _sensors.castAll(terrain: terrain, engineCfg: cfg, origin: lander.pos, angle: lander.angle);
      return StepInfo(costDelta: 0.0, terminal: true, landed: false, onPad: false);
    }

    // ----- Effort cost + Scoring -----
    double stepCost = 0.0;
    if (power > 0) stepCost += cfg.effortCost * power * dt;

    final out = Scoring.apply(
      cfg: cfg,
      terrain: terrain,
      pos: pos,
      vel: vel,
      angle: angle,
      prevAngle: _lastAngle,
      dt: dt,
      u: u,
      intentIdx: u.intentIdx,
    );

    stepCost += out.cost;

    // Commit
    lander = LanderState(pos: out.pos, vel: out.vel, angle: angle, fuel: fuel);
    _lastAngle = angle;
    if (out.terminal) {
      status = out.landed ? GameStatus.landed : GameStatus.crashed;
    }

    // Sensors update
    _rays = _sensors.castAll(
      terrain: terrain,
      engineCfg: cfg,
      origin: lander.pos,
      angle: lander.angle,
    );

    return StepInfo(
      costDelta: stepCost,
      terminal: out.terminal,
      landed: out.landed,
      onPad: out.onPad,
    );
  }

  // ---- NN convenience passthroughs ----
  List<double> rayChannels3({bool useProximity = false}) =>
      _sensors.channels3(hits: _rays, engineCfg: cfg, useProximity: useProximity);

  ({double minD, double bearing, double visible}) padSummary() =>
      _sensors.padSummary(hits: _rays, engineCfg: cfg);
}
