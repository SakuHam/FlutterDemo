import 'dart:math' as math;
import 'types.dart';
import 'scoring.dart';

/// =======================
/// The Game Engine (physics + calls into Scoring)
/// =======================
class GameEngine {
  EngineConfig cfg;
  late Terrain terrain;
  late LanderState lander;
  GameStatus status = GameStatus.playing;

  math.Random _rnd;
  Terrain? _frozenTerrain;
  double _lastAngle = 0.0;

  GameEngine(this.cfg) : _rnd = math.Random(cfg.seed) {
    reset(seed: cfg.seed);
  }

  void reset({
    int? seed,
    double? padWidthFactor,
    double? landingSpeedMax,
    double? landingAngleMaxRad,
  }) {
    // Only reseed if a seed is explicitly provided
    if (seed != null) {
      _rnd = math.Random(seed);
    }

    // Apply optional curriculum overrides
    if (padWidthFactor != null || landingSpeedMax != null || landingAngleMaxRad != null) {
      cfg = cfg.copyWith(
        padWidthFactor: padWidthFactor ?? cfg.padWidthFactor,
        landingSpeedMax: landingSpeedMax ?? cfg.landingSpeedMax,
        landingAngleMaxRad: landingAngleMaxRad ?? cfg.landingAngleMaxRad,
      );
    }

    // Terrain: frozen or fresh
    if (cfg.lockTerrain) {
      _frozenTerrain ??= Terrain.generate(cfg.worldW, cfg.worldH, cfg.terrainSeed, cfg.padWidthFactor);
      terrain = _frozenTerrain!;
    } else {
      terrain = Terrain.generate(cfg.worldW, cfg.worldH, _rnd.nextInt(1 << 30), cfg.padWidthFactor);
    }

    // ---- Spawn X (always honor randomSpawnX when enabled) ----
    double fracX;
    if (cfg.randomSpawnX) {
      final a = cfg.spawnXMin.clamp(0.0, 1.0);
      final b = cfg.spawnXMax.clamp(0.0, 1.0);
      final lo = math.min(a, b), hi = math.max(a, b);
      fracX = lo + _rnd.nextDouble() * (hi - lo);
    } else {
      fracX = cfg.spawnX; // fixed fraction (0..1)
    }

    // Build lander state; lockSpawn controls Y/vel/angle/fuel only
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
  }

  /// Physics + scoring. dt is in seconds.
  StepInfo step(double dt, ControlInput u) {
    if (status != GameStatus.playing) {
      return StepInfo(costDelta: 0.0, terminal: true, landed: status == GameStatus.landed, onPad: false);
    }

    final s = cfg.stepScale; // arcade tuning
    final t = cfg.t;

    // ----- Controls -> rotation -----
    double angle = lander.angle;
    final rot = t.rotSpeed * 0.5;
    if (u.left && !u.right) angle -= rot * dt;
    if (u.right && !u.left) angle += rot * dt;

    // ----- Acceleration & fuel -----
    Vector2 accel = Vector2(0, t.gravity * 0.05);
    double fuel = lander.fuel;
    final bool thrusting = u.thrust && fuel > 0.0;
    final double power = thrusting ? 1.0 : 0.0;

    if (thrusting) {
      accel.x += math.sin(angle) * (t.thrustAccel * 0.05);
      accel.y += -math.cos(angle) * (t.thrustAccel * 0.05);
      // fuel burn in units/sec (same rate as UI)
      fuel = (fuel - 20.0 * dt).clamp(0.0, t.maxFuel);
    }

    // ----- Integrate (semi-implicit Euler) -----
    final vel = Vector2(
      lander.vel.x + accel.x * dt * s,
      lander.vel.y + accel.y * dt * s,
    );
    Vector2 pos = Vector2(
      lander.pos.x + vel.x * dt * s,
      lander.pos.y + vel.y * dt * s,
    );

    // ----- Effort cost (optional: physics knows the power) -----
    double stepCost = 0.0;
    if (power > 0) stepCost += cfg.effortCost * power * dt;

    // ----- Scoring/penalties/collisions (centralized) -----
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

    // Combine effort + scoring cost
    stepCost += out.cost;

    // Commit state
    lander = LanderState(pos: out.pos, vel: out.vel, angle: angle, fuel: fuel);
    _lastAngle = angle;
    if (out.terminal) status = out.landed ? GameStatus.landed : GameStatus.crashed;

    return StepInfo(
      costDelta: stepCost,
      terminal: out.terminal,
      landed: out.landed,
      onPad: out.onPad,
    );
  }
}
