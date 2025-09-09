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

  // Border punishment while keeping wrap
  final double borderMargin;          // px near left/right edges
  final double borderPenaltyPerSec;   // cost/sec when at the wall (ramps to 0 at margin)
  final double wrapPenalty;           // one-off cost when actually wrapping

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
    this.wAngleDeg = 0.025,
    this.borderMargin = 30.0,
    this.borderPenaltyPerSec = 2.0,
    this.wrapPenalty = 20.0,
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
    double? borderMargin,
    double? borderPenaltyPerSec,
    double? wrapPenalty,
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
      borderMargin: borderMargin ?? this.borderMargin,
      borderPenaltyPerSec: borderPenaltyPerSec ?? this.borderPenaltyPerSec,
      wrapPenalty: wrapPenalty ?? this.wrapPenalty,
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

  // NEW: whether current x is over the pad (for UI/diagnostics)
  final bool onPad;

  StepInfo({
    required this.costDelta,
    required this.terminal,
    required this.landed,
    required this.onPad,
  }) : scoreDelta = costDelta; // alias
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

  GameEngine(this.cfg) : _rnd = math.Random(cfg.seed) {
    reset(seed: cfg.seed);
  }

  void reset({
    int? seed,
    double? padWidthFactor,
    double? landingSpeedMax,
    double? landingAngleMaxRad,
  }) {
    if (seed != null) {
      _rnd = math.Random(seed); // ✅ re-seed the RNG properly (Random has no setSeed)
    }
    // Update optional curriculum overrides
    if (padWidthFactor != null || landingSpeedMax != null || landingAngleMaxRad != null) {
      cfg = cfg.copyWith(
        padWidthFactor: padWidthFactor ?? cfg.padWidthFactor,
        landingSpeedMax: landingSpeedMax ?? cfg.landingSpeedMax,
        landingAngleMaxRad: landingAngleMaxRad ?? cfg.landingAngleMaxRad,
      );
    }

    terrain = Terrain.generate(cfg.worldW, cfg.worldH, _rnd.nextInt(1 << 30), cfg.padWidthFactor);

    // Spawn roughly left/top
    lander = LanderState(
      pos: Vector2(cfg.worldW * 0.2, 120),
      vel: Vector2(0, 0),
      angle: 0.0,
      fuel: cfg.t.maxFuel,
    );
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
    // Border penalty + wrap
    // =========================
    double cost = 0.0;

    // A) Continuous edge penalty (pre-wrap horizontal position)
        {
      final x = pos.x;
      final dLeft = x;
      final dRight = cfg.worldW - x;
      final dEdge = dLeft < dRight ? dLeft : dRight;

      if (dEdge < cfg.borderMargin) {
        final frac = (1.0 - (dEdge / cfg.borderMargin)).clamp(0.0, 1.0);
        cost += frac * cfg.borderPenaltyPerSec * dt;
      }
    }

    // B) Apply wrap with one-shot penalty
    bool crossed = false;
    if (pos.x < 0) {
      pos = Vector2(cfg.worldW + pos.x, pos.y);
      crossed = true;
    } else if (pos.x > cfg.worldW) {
      pos = Vector2(pos.x - cfg.worldW, pos.y);
      crossed = true;
    }
    if (crossed) {
      cost += cfg.wrapPenalty;
    }

    // Clamp Y to world (don’t let it go above top too much)
    if (pos.y < 0) pos.y = 0;

    // =========================
    // Base shaping (dense)
    // =========================
    // Living time
    cost += cfg.livingCost * dt;

    // Effort (proportional to power)
    if (power > 0) cost += cfg.effortCost * power * dt;

    // Horizontal distance to pad center (normalized)
    final padCenterX = terrain.padCenter;
    final dxN = (pos.x - padCenterX).abs() / cfg.worldW;
    cost += cfg.wDx * dxN;

    // Vertical distance to ground (normalized to worldH)
    final groundY = terrain.heightAt(pos.x);
    final dyN = ((groundY - pos.y).abs()) / cfg.worldH;
    cost += cfg.wDy * dyN;

    // Velocity shaping (normalize vx, vy roughly by 200 px/s)
    final vxN = (vel.x.abs() / 200.0).clamp(0.0, 2.0);
    final vyDownN = (vel.y > 0 ? (vel.y / 200.0) : 0.0).clamp(0.0, 2.0);
    cost += cfg.wVx * vxN + cfg.wVyDown * vyDownN;

    // Angle shaping (degrees normalized by 180)
    final angleDeg = angle.abs() * 180.0 / math.pi;
    cost += cfg.wAngleDeg * (angleDeg / 180.0);

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

    // Commit state
    lander = LanderState(pos: pos, vel: vel, angle: angle, fuel: fuel);
    if (terminal) status = landed ? GameStatus.landed : GameStatus.crashed;

    return StepInfo(
      costDelta: cost,
      terminal: terminal,
      landed: landed,
      onPad: onPadNow,
    );
  }
}
