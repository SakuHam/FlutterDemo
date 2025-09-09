// lib/engine/game_engine.dart
import 'dart:math' as math;

/// =======================
/// Config & basic structs
/// =======================

class Tunables {
  double gravity;      // base gravity strength
  double thrustAccel;  // engine acceleration (px/s^2 @ power=1)
  double rotSpeed;     // radians per second
  double maxFuel;      // fuel units

  Tunables({
    this.gravity = 0.18,
    this.thrustAccel = 0.42,
    this.rotSpeed = 1.6,
    this.maxFuel = 100.0,
  });

  Tunables copyWith({
    double? gravity,
    double? thrustAccel,
    double? rotSpeed,
    double? maxFuel,
  }) {
    return Tunables(
      gravity: gravity ?? this.gravity,
      thrustAccel: thrustAccel ?? this.thrustAccel,
      rotSpeed: rotSpeed ?? this.rotSpeed,
      maxFuel: maxFuel ?? this.maxFuel,
    );
  }
}

class EngineConfig {
  final double worldW;
  final double worldH;
  final Tunables t;

  final int seed;
  final double stepScale; // multiplies dt for arcade-y tuning (60 ~ 60 FPS)

  // Landing tolerance (training-time)
  final double padWidthFactor;      // scales generated pad width
  final double landingSpeedMax;     // px/s
  final double landingAngleMaxRad;  // radians

  // Shaping costs (all >= 0; total cost is sum)
  final double livingCost;   // cost/sec just for time
  final double effortCost;   // cost/sec while burning (power-weighted)
  final double wDx;          // cost weight for |dx| / worldW
  final double wDy;          // cost weight for |dy| / worldH
  final double wVyDown;      // cost weight for downward vy+ (normalized)
  final double wVx;          // cost weight for |vx| (normalized)
  final double wAngleDeg;    // cost weight for |angle| in degrees / 180
  final double angleNearGroundBoost; // multiplies angle cost near ground
  final double wAngleRate;           // penalize |d(angle)/dt| to smooth

  // Border punishment (compatible with wrap)
  final double borderMargin;          // px near left/right edges
  final double borderPenaltyPerSec;   // cost/sec at wall (ramps to 0 at margin)
  final double wrapPenalty;           // one-off cost when actually wrapping

  // ===== Overfit/Debug controls =====
  final bool lockTerrain;     // reuse identical terrain every reset
  final int terrainSeed;      // seed for terrain when locked
  final bool lockSpawn;       // fixed initial state every episode
  final double spawnX;        // fraction of worldW (0..1)
  final double spawnY;        // px
  final double spawnVx;       // px/s
  final double spawnVy;       // px/s
  final double spawnAngle;    // radians
  final bool hardWalls;       // if true, clamp X to [0, W] (no wrap)

  // EngineConfig additions
  final double ceilingMargin;          // px from the top where penalties start
  final double ceilingPenaltyPerSec;   // cost/sec when fully at the top
  final double ceilingHitPenalty;      // one-shot when clamping at the top

  final bool randomSpawnX;   // if true, pick X uniformly in [spawnXMin, spawnXMax]
  final double spawnXMin;    // fraction of worldW (0..1)
  final double spawnXMax;    // fraction of worldW (0..1)

  final double wThrustAssist;   // bonus when thrusting while descending (px/s scaled)
  final double wTurnAssist;     // bonus when turning to reduce |angle| (rad scaled)
  final double wPadAlignAssist; // bonus when turning toward pad (dx scaled)
  final double wIdlePenalty;    // penalty for doing nothing while descending fast

  EngineConfig({
    required this.worldW,
    required this.worldH,
    required this.t,
    this.seed = 42,
    this.stepScale = 60.0,
    this.padWidthFactor = 1.0,
    this.landingSpeedMax = 80.0,
    this.landingAngleMaxRad = 0.45,
    this.livingCost = 0.002,
    this.effortCost = 0.0001,
    this.wDx = 0.0025,
    this.wDy = 0.0020,
    this.wVyDown = 0.025,
    this.wVx = 0.0015,
    this.wAngleDeg = 0.035,
    this.angleNearGroundBoost = 2.0,
    this.wAngleRate = 0.01,
    this.borderMargin = 30.0,
    this.borderPenaltyPerSec = 2.0,
    this.wrapPenalty = 20.0,
    this.lockTerrain = false,
    this.terrainSeed = 12345,
    this.lockSpawn = false,
    this.spawnX = 0.25,
    this.spawnY = 120.0,
    this.spawnVx = 0.0,
    this.spawnVy = 0.0,
    this.spawnAngle = 0.0,
    this.hardWalls = true,
    this.ceilingMargin = 40.0,
    this.ceilingPenaltyPerSec = 3.0,
    this.ceilingHitPenalty = 10.0,
    this.randomSpawnX = true,
    this.spawnXMin = 0.15,
    this.spawnXMax = 0.85,
    this.wThrustAssist = 0.25,
    this.wTurnAssist = 0.20,
    this.wPadAlignAssist = 0.10,
    this.wIdlePenalty = 0.10,
  });

  EngineConfig copyWith({
    double? worldW,
    double? worldH,
    Tunables? t,
    int? seed,
    double? stepScale,
    double? padWidthFactor,
    double? landingSpeedMax,
    double? landingAngleMaxRad,
    double? livingCost,
    double? effortCost,
    double? wDx,
    double? wDy,
    double? wVyDown,
    double? wVx,
    double? wAngleDeg,
    double? angleNearGroundBoost,
    double? wAngleRate,
    double? borderMargin,
    double? borderPenaltyPerSec,
    double? wrapPenalty,
    bool? lockTerrain,
    int? terrainSeed,
    bool? lockSpawn,
    double? spawnX,
    double? spawnY,
    double? spawnVx,
    double? spawnVy,
    double? spawnAngle,
    bool? hardWalls,
    double? ceilingMargin,
    double? ceilingPenaltyPerSec,
    double? ceilingHitPenalty,
    bool? randomSpawnX,
    double? spawnXMin,
    double? spawnXMax,
  }) {
    return EngineConfig(
      worldW: worldW ?? this.worldW,
      worldH: worldH ?? this.worldH,
      t: t ?? this.t,
      seed: seed ?? this.seed,
      stepScale: stepScale ?? this.stepScale,
      padWidthFactor: padWidthFactor ?? this.padWidthFactor,
      landingSpeedMax: landingSpeedMax ?? this.landingSpeedMax,
      landingAngleMaxRad: landingAngleMaxRad ?? this.landingAngleMaxRad,
      livingCost: livingCost ?? this.livingCost,
      effortCost: effortCost ?? this.effortCost,
      wDx: wDx ?? this.wDx,
      wDy: wDy ?? this.wDy,
      wVyDown: wVyDown ?? this.wVyDown,
      wVx: wVx ?? this.wVx,
      wAngleDeg: wAngleDeg ?? this.wAngleDeg,
      angleNearGroundBoost: angleNearGroundBoost ?? this.angleNearGroundBoost,
      wAngleRate: wAngleRate ?? this.wAngleRate,
      borderMargin: borderMargin ?? this.borderMargin,
      borderPenaltyPerSec: borderPenaltyPerSec ?? this.borderPenaltyPerSec,
      wrapPenalty: wrapPenalty ?? this.wrapPenalty,
      lockTerrain: lockTerrain ?? this.lockTerrain,
      terrainSeed: terrainSeed ?? this.terrainSeed,
      lockSpawn: lockSpawn ?? this.lockSpawn,
      spawnX: spawnX ?? this.spawnX,
      spawnY: spawnY ?? this.spawnY,
      spawnVx: spawnVx ?? this.spawnVx,
      spawnVy: spawnVy ?? this.spawnVy,
      spawnAngle: spawnAngle ?? this.spawnAngle,
      hardWalls: hardWalls ?? this.hardWalls,
      ceilingMargin: ceilingMargin ?? this.ceilingMargin,
      ceilingPenaltyPerSec: ceilingPenaltyPerSec ?? this.ceilingPenaltyPerSec,
      ceilingHitPenalty: ceilingHitPenalty ?? this.ceilingHitPenalty,
      randomSpawnX: randomSpawnX ?? this.randomSpawnX,
      spawnXMin: spawnXMin ?? this.spawnXMin,
      spawnXMax: spawnXMax ?? this.spawnXMax,
    );
  }
}

enum GameStatus { playing, landed, crashed }

class ControlInput {
  final bool thrust;
  final bool left;
  final bool right;

  const ControlInput({required this.thrust, required this.left, required this.right});
}

class Vector2 {
  double x, y;
  Vector2(this.x, this.y);

  Vector2 operator +(Vector2 o) => Vector2(x + o.x, y + o.y);
  Vector2 operator -(Vector2 o) => Vector2(x - o.x, y - o.y);
  Vector2 operator *(double s) => Vector2(x * s, y * s);
  double get length => math.sqrt(x * x + y * y);
}

class LanderState {
  Vector2 pos;   // center of mass (px)
  Vector2 vel;   // px/s
  double angle;  // radians (0 = up)
  double fuel;   // 0..maxFuel

  LanderState({
    required this.pos,
    required this.vel,
    required this.angle,
    required this.fuel,
  });

  LanderState copy() => LanderState(
    pos: Vector2(pos.x, pos.y),
    vel: Vector2(vel.x, vel.y),
    angle: angle,
    fuel: fuel,
  );
}

class Terrain {
  final List<Vector2> ridge; // piecewise linear ridge
  final double padX1;
  final double padX2;
  final double padY;

  Terrain({
    required this.ridge,
    required this.padX1,
    required this.padX2,
    required this.padY,
  });

  double get padCenter => (padX1 + padX2) * 0.5;

  double heightAt(double x) {
    final pts = ridge;
    for (int i = 0; i < pts.length - 1; i++) {
      final a = pts[i];
      final b = pts[i + 1];
      if ((x >= a.x && x <= b.x) || (x >= b.x && x <= a.x)) {
        final t = (x - a.x) / (b.x - a.x);
        return a.y + (b.y - a.y) * t;
      }
    }
    return pts.last.y;
  }

  bool isOnPad(double x) => x >= padX1 && x <= padX2;

  static Terrain generate(double w, double h, int seed, double padWidthFactor) {
    final rnd = math.Random(seed);
    final List<Vector2> pts = [];
    const int segments = 24;
    for (int i = 0; i <= segments; i++) {
      final x = w * i / segments;
      final base = h * 0.78;
      final noise = (math.sin(i * 0.8) + rnd.nextDouble() * 0.5) * 24.0;
      pts.add(Vector2(x, base + noise));
    }

    // Landing pad
    final rawPadWidth = w * 0.16 * padWidthFactor;
    final padWidth = rawPadWidth.clamp(36.0, w * 0.6);
    final padCenterX = w * (0.35 + rnd.nextDouble() * 0.3);
    final padX1 = math.max(10.0, math.min(padCenterX - padWidth / 2, w - padWidth - 10.0));
    final padX2 = padX1 + padWidth;
    final padY = h * 0.76;

    for (int i = 0; i < pts.length; i++) {
      if (pts[i].x >= padX1 && pts[i].x <= padX2) {
        pts[i] = Vector2(pts[i].x, padY);
      }
    }

    // Valley look
    pts[0] = Vector2(0, h * 0.92);
    pts[pts.length - 1] = Vector2(w, h * 0.92);

    return Terrain(ridge: pts, padX1: padX1, padX2: padX2, padY: padY);
  }
}

class StepInfo {
  final double costDelta;   // >= 0 ; lower is better
  final double scoreDelta;  // alias for compatibility
  final bool terminal;
  final bool landed;

  // diagnostics
  final bool onPad;

  StepInfo({
    required this.costDelta,
    required this.terminal,
    required this.landed,
    required this.onPad,
  }) : scoreDelta = costDelta;
}

/// =======================
/// The Game Engine
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

  /// Physics + shaping costs. dt is in seconds.
  StepInfo step(double dt, ControlInput u) {
    if (status != GameStatus.playing) {
      return StepInfo(costDelta: 0.0, terminal: true, landed: status == GameStatus.landed, onPad: false);
    }

    final s = cfg.stepScale; // arcade tuning
    final t = cfg.t;

    // Controls -> rotation
    double angle = lander.angle;
    final rot = t.rotSpeed * 0.5;
    if (u.left && !u.right) angle -= rot * dt;
    if (u.right && !u.left) angle += rot * dt;

    // Acceleration
    Vector2 accel = Vector2(0, t.gravity * 0.05);
    double fuel = lander.fuel;
    bool thrusting = u.thrust && fuel > 0.0;
    double power = thrusting ? 1.0 : 0.0;

    if (thrusting) {
      accel.x += math.sin(angle) * (t.thrustAccel * 0.05);
      accel.y += -math.cos(angle) * (t.thrustAccel * 0.05);
      // fuel burn in units/sec (same rate as UI)
      fuel = (fuel - 20.0 * dt).clamp(0.0, t.maxFuel);
    }

    // Integrate (semi-implicit Euler)
    final vel = Vector2(
      lander.vel.x + accel.x * dt * s,
      lander.vel.y + accel.y * dt * s,
    );
    Vector2 pos = Vector2(
      lander.pos.x + vel.x * dt * s,
      lander.pos.y + vel.y * dt * s,
    );

    // =========================
    // Border penalty + wrap or walls
    // =========================
    double cost = 0.0;

    // ----- TOP CEILING (y decreases upward; top is y=0) -----
    if (pos.y < 0) {
      pos.y = 0;
      if (vel.y < 0) vel.y = 0;           // kill upward velocity
      cost += cfg.ceilingHitPenalty;      // one-shot bump
    }

    // Continuous penalty near the top (ramps 0..1 inside margin)
    if (pos.y < cfg.ceilingMargin) {
      final frac = (1.0 - (pos.y / cfg.ceilingMargin)).clamp(0.0, 1.0);
      cost += frac * cfg.ceilingPenaltyPerSec * dt;
    }

    // Continuous edge penalty (pre-wrap position)
    {
      final x = pos.x;
      final dLeft = x;
      final dRight = cfg.worldW - x;
      final dEdge = dLeft < dRight ? dLeft : dRight;

      if (!cfg.hardWalls && dEdge < cfg.borderMargin) {
        final frac = (1.0 - (dEdge / cfg.borderMargin)).clamp(0.0, 1.0);
        cost += frac * cfg.borderPenaltyPerSec * dt;
      }
    }

    if (cfg.hardWalls) {
      bool hit = false;
      if (pos.x < 0) { pos.x = 0; hit = true; }
      if (pos.x > cfg.worldW) { pos.x = cfg.worldW; hit = true; }
      if (hit) { cost += cfg.wrapPenalty; /* small nudge */ }
    } else {
      bool crossed = false;
      if (pos.x < 0) { pos = Vector2(cfg.worldW + pos.x, pos.y); crossed = true; }
      else if (pos.x > cfg.worldW) { pos = Vector2(pos.x - cfg.worldW, pos.y); crossed = true; }
      if (crossed) cost += cfg.wrapPenalty;
    }

    // Clamp Y to world (don’t let it go above top too much)
    if (pos.y < 0) pos.y = 0;

    // =========================
    // Base shaping (dense)
    // =========================
    // Distances
    final groundY = terrain.heightAt(pos.x);
    final padCenterX = terrain.padCenter;

    // Living time
    cost += cfg.livingCost * dt;

    // Effort (proportional to power)
    if (power > 0) cost += cfg.effortCost * power * dt;

    // Horizontal distance to pad center (normalized)
    final dxN = (pos.x - padCenterX).abs() / cfg.worldW;
    cost += cfg.wDx * dxN;

    // Vertical distance to ground (normalized to worldH)
    final dyN = ((groundY - pos.y).abs()) / cfg.worldH;
    cost += cfg.wDy * dyN;

    // Velocity shaping (normalize vx, vy roughly by 200 px/s)
    final vxN = (vel.x.abs() / 200.0).clamp(0.0, 2.0);
    final vyDownN = (vel.y > 0 ? (vel.y / 200.0) : 0.0).clamp(0.0, 2.0);
    cost += cfg.wVx * vxN + cfg.wVyDown * vyDownN;

    // Angle shaping (degrees normalized by 180), stronger near ground
    final hN = (dyN).clamp(0.0, 1.0);
    final angleDeg = angle.abs() * 180.0 / math.pi;
    final angleCost = cfg.wAngleDeg * (angleDeg / 180.0) * (1.0 + cfg.angleNearGroundBoost * (1.0 - hN));
    cost += angleCost;

    // Angular rate (smoothness)
    final dAng = ((angle - _lastAngle).abs()) / (dt + 1e-8);
    cost += cfg.wAngleRate * (dAng / math.pi); // normalize by π rad/s

    // =========================
    // Collision / Terminal
    // =========================
    bool terminal = false;
    bool landed = false;
    final bool onPadNow = terrain.isOnPad(pos.x);

    final gentleSpeed = vel.length;
    final gentleAngle = angle.abs();

    final collided = (pos.y >= groundY - 2.0);

    if (collided) {
      terminal = true;
      // Landing check with current curriculum tolerances
      final okSpeed = gentleSpeed <= cfg.landingSpeedMax;
      final okAngle = gentleAngle <= cfg.landingAngleMaxRad;

      if (onPadNow && okSpeed && okAngle) {
        landed = true;
        // Small landing penalty term (kept >=0) so "perfect" is still ~0
        final landPenalty = 0.2 * (gentleSpeed / (cfg.landingSpeedMax + 1e-6)) +
            0.2 * (gentleAngle / (cfg.landingAngleMaxRad + 1e-6));
        cost += landPenalty.clamp(0.0, 1.0);

        // Snap to pad
        pos = Vector2(pos.x, terrain.padY - 18.0); // halfHeight ~ 18px (UI parity)
      } else {
        // Crash penalty: bigger than any wrap/edge nuisances
        cost += 120.0;
      }
    }

// --- Existing penalties (position/velocity/angle/fuel/etc) ---
// double costDelta = ... your current shaping ...

// ===== Action-aware shaping (bonuses & a small idle penalty) =====
    const double vyCap = 120.0;                  // scale for descent speed
    const double angCap = math.pi / 4;           // 45° scale for angle correction
    final double dxDead = 0.03 * cfg.worldW;     // deadzone for pad alignment
    const double angDead = 5 * math.pi / 180.0;  // 5° deadzone for angle correction

    // (A) Thrust-assist bonus when descending
    if (u.thrust && vel.y > 0) {
      final vyN = (vel.y / vyCap).clamp(0.0, 1.0); // 0..1 for 0..vyCap
      cost -= cfg.wThrustAssist * vyN;
    }

    // (B) Turn-assist bonus when turning toward upright (reduce |angle|)
    if (angle > angDead && u.left) {
      final aN = ((angle - angDead) / angCap).clamp(0.0, 1.0);
      cost -= cfg.wTurnAssist * aN;
    } else if (angle < -angDead && u.right) {
      final aN = (((-angle) - angDead) / angCap).clamp(0.0, 1.0);
      cost -= cfg.wTurnAssist * aN;
    }

    // (C) Pad-align bonus when turning toward the pad horizontally
    final dx = (pos.x - terrain.padCenter);
    if (dx.abs() > dxDead) {
      if (dx > 0 && u.left) {
        // pad is to the left → turning left helps
        final dN = ((dx.abs() - dxDead) / (0.5 * cfg.worldW)).clamp(0.0, 1.0);
        cost -= cfg.wPadAlignAssist * dN;
      } else if (dx < 0 && u.right) {
        // pad is to the right → turning right helps
        final dN = ((dx.abs() - dxDead) / (0.5 * cfg.worldW)).clamp(0.0, 1.0);
        cost -= cfg.wPadAlignAssist * dN;
      }
    }

    // (D) Idle penalty when falling fast & not acting
    if (!u.thrust && !u.left && !u.right && vel.y > 80) {
      final vN = ((vel.y - 80) / (vyCap - 80)).clamp(0.0, 1.0);
      cost += cfg.wIdlePenalty * vN;
    }

    if (cost < 0) cost = 0.0;

    // Commit state
    lander = LanderState(pos: pos, vel: vel, angle: angle, fuel: fuel);
    _lastAngle = angle;
    if (terminal) status = landed ? GameStatus.landed : GameStatus.crashed;

    return StepInfo(
      costDelta: cost,
      terminal: terminal,
      landed: landed,
      onPad: onPadNow,
    );
  }
}
