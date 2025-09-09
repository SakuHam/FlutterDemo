// lib/engine/game_engine.dart
import 'dart:math' as math;

/// --------- Math + data utilities (no Flutter types) ----------
class Vec2 {
  double x, y;
  Vec2(this.x, this.y);
  Vec2.zero() : x = 0, y = 0;
  Vec2 operator +(Vec2 o) => Vec2(x + o.x, y + o.y);
  Vec2 operator -(Vec2 o) => Vec2(x - o.x, y - o.y);
  Vec2 operator *(double s) => Vec2(x * s, y * s);
  double get length => math.sqrt(x * x + y * y);
}

enum GameStatus { playing, landed, crashed }

class Tunables {
  double gravity;
  double thrustAccel;
  double rotSpeed;
  double maxFuel;
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
  }) => Tunables(
    gravity: gravity ?? this.gravity,
    thrustAccel: thrustAccel ?? this.thrustAccel,
    rotSpeed: rotSpeed ?? this.rotSpeed,
    maxFuel: maxFuel ?? this.maxFuel,
  );
}

class EngineConfig {
  final double worldW;
  final double worldH;
  final Tunables t;
  final int seed;
  final double stepScale;

  // Curriculum
  final double padWidthFactor;
  final double landingSpeedMax;
  final double landingAngleMaxRad;

  // Per-step shaping
  final double livingCost;
  final double effortCost;   // ↓ lower so thrust isn’t discouraged
  final double wDx;          // |dx-to-pad|
  final double wDy;          // NEW: |dy to padY|
  final double wVyDown;      // ↓/↑ tuning falling speed
  final double wVx;          // |vx|^2
  final double wAngleDeg;    // |angle(deg)|

  const EngineConfig({
    required this.worldW,
    required this.worldH,
    required this.t,
    this.seed = 42,
    this.stepScale = 60.0,
    this.padWidthFactor = 1.0,
    this.landingSpeedMax = 40.0,
    this.landingAngleMaxRad = 0.25,

    this.livingCost = 0.002,
    this.effortCost = 0.0003,   // was 0.001
    this.wDx = 0.0025,
    this.wDy = 0.0020,          // NEW
    this.wVyDown = 0.020,       // was 0.005 -> stronger
    this.wVx = 0.0015,
    this.wAngleDeg = 0.03,
  });
}

class LanderGeom {
  static const double halfWidth = 14;
  static const double halfHeight = 18;
}

class LanderState {
  final Vec2 pos;     // center
  final Vec2 vel;
  final double angle; // radians, 0 = up
  final double fuel;

  const LanderState({
    required this.pos,
    required this.vel,
    required this.angle,
    required this.fuel,
  });

  LanderState copyWith({
    Vec2? pos,
    Vec2? vel,
    double? angle,
    double? fuel,
  }) => LanderState(
    pos: pos ?? this.pos,
    vel: vel ?? this.vel,
    angle: angle ?? this.angle,
    fuel: fuel ?? this.fuel,
  );

  ({Vec2 left, Vec2 right}) footPoints() {
    final c = math.cos(angle);
    final s = math.sin(angle);
    final bottomCenter = Vec2(pos.x, pos.y + LanderGeom.halfHeight);
    Vec2 rot(Vec2 v) => Vec2(c * v.x - s * v.y, s * v.x + c * v.y);
    final leftLocal = Vec2(-LanderGeom.halfWidth, 0);
    final rightLocal = Vec2(LanderGeom.halfWidth, 0);
    return (left: bottomCenter + rot(leftLocal), right: bottomCenter + rot(rightLocal));
  }
}

class Terrain {
  final List<Vec2> ridge; // polyline ridge
  final double padX1;
  final double padX2;
  final double padY;

  Terrain(this.ridge, this.padX1, this.padX2, this.padY);

  static Terrain generate(double width, double height,
      {int seed = 42, double padWidthFactor = 1.0}) {
    final rnd = math.Random(seed);
    final List<Vec2> pts = [];
    const int segments = 24;
    for (int i = 0; i <= segments; i++) {
      final x = width * i / segments;
      final base = height * 0.78;
      final noise = (math.sin(i * 0.8) + rnd.nextDouble() * 0.5) * 24.0;
      pts.add(Vec2(x, base + noise));
    }

    final padWidth = width * 0.16 * padWidthFactor;
    final padCenterX = width * (0.35 + rnd.nextDouble() * 0.3);
    final padX1 = math.max(10.0, math.min(padCenterX - padWidth / 2, width - padWidth - 10.0));
    final padX2 = padX1 + padWidth;
    final padY = height * 0.76;

    for (int i = 0; i < pts.length; i++) {
      if (pts[i].x >= padX1 && pts[i].x <= padX2) {
        pts[i] = Vec2(pts[i].x, padY);
      }
    }
    // ends lower
    pts[0] = Vec2(0, height * 0.92);
    pts[pts.length - 1] = Vec2(width, height * 0.92);

    return Terrain(pts, padX1, padX2, padY);
  }

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
    return ridge.last.y;
  }

  bool isOnPad(double x) => x >= padX1 && x <= padX2;
  double get padCenter => (padX1 + padX2) * 0.5;
}

class ControlInput {
  final bool thrust, left, right;
  const ControlInput({this.thrust = false, this.left = false, this.right = false});
}

class StepInfo {
  final GameStatus status;
  final bool terminal;
  final double speed;
  final double angleAbs;
  final bool onPad;
  final double costDelta;  // >= 0
  const StepInfo({
    required this.status,
    required this.terminal,
    required this.speed,
    required this.angleAbs,
    required this.onPad,
    required this.costDelta,
  });
}

class GameEngine {
  final EngineConfig cfg;
  late Terrain terrain;
  late LanderState lander;
  GameStatus status = GameStatus.playing;
  double cost = 0.0;

  final math.Random _rnd;

  GameEngine(this.cfg) : _rnd = math.Random(cfg.seed) {
    reset(seed: cfg.seed);
  }

  void reset({int? seed}) {
    final s = seed ?? _rnd.nextInt(1 << 31);
    terrain = Terrain.generate(
      cfg.worldW,
      cfg.worldH,
      seed: s,
      padWidthFactor: cfg.padWidthFactor,
    );
    status = GameStatus.playing;
    cost = 0.0;
    lander = LanderState(
      pos: Vec2(cfg.worldW * 0.2, 120),
      vel: Vec2.zero(),
      angle: 0.0,
      fuel: cfg.t.maxFuel,
    );
  }

  StepInfo step(double dtSeconds, ControlInput input) {
    if (status != GameStatus.playing) {
      return StepInfo(status: status, terminal: true, speed: 0, angleAbs: 0, onPad: false, costDelta: 0);
    }

    final dt = dtSeconds.clamp(0.0, 1 / 20.0);
    final t = cfg.t;

    // Rotation
    double angle = lander.angle;
    final rot = t.rotSpeed * 0.5;
    if (input.left && !input.right) angle -= rot * dt;
    if (input.right && !input.left) angle += rot * dt;

    // Accel
    Vec2 accel = Vec2(0, t.gravity * 0.05);
    double fuel = lander.fuel;
    final thrustingNow = input.thrust && fuel > 0;
    if (thrustingNow) {
      final ax = math.sin(angle);
      final ay = -math.cos(angle);
      accel = accel + Vec2(ax, ay) * (t.thrustAccel * 0.05);
      fuel = (fuel - 20 * dt).clamp(0, t.maxFuel);
    }

    // Integrate
    final vel = lander.vel + accel * (dt * cfg.stepScale);
    Vec2 pos = lander.pos + vel * (dt * cfg.stepScale);

    // Wrap X
    final w = cfg.worldW;
    if (pos.x < 0) pos = Vec2(w + pos.x, pos.y);
    if (pos.x > w) pos = Vec2(pos.x - w, pos.y);

    // -------- Shaping cost each step (dense signal) --------
    double stepCost = 0.0;
    stepCost += cfg.livingCost;
    if (thrustingNow) stepCost += cfg.effortCost;

    final dx = (pos.x - terrain.padCenter).abs();
    final dy = (pos.y - terrain.padY).abs(); // NEW: vertical distance to pad height
    final vx = vel.x;
    final vy = vel.y;
    final angleDeg = (angle.abs() * 180 / math.pi);

    // Penalize distance to target, falling too fast, horizontal speed, tilt
    stepCost += cfg.wDx * dx;
    stepCost += cfg.wDy * dy;               // NEW
    if (vy > 0) stepCost += cfg.wVyDown * (vy * vy);
    stepCost += cfg.wVx * (vx * vx);
    stepCost += cfg.wAngleDeg * angleDeg;

    // -------- Collision detection --------
    final feet = _footPointsAt(pos, angle);
    final groundYLeft   = terrain.heightAt(feet.$1.x);
    final groundYRight  = terrain.heightAt(feet.$2.x);
    final groundYCenter = terrain.heightAt(pos.x);
    final collided = (feet.$1.y >= groundYLeft) ||
        (feet.$2.y >= groundYRight) ||
        (pos.y >= groundYCenter - 2);

    final speedNow = vel.length;
    final angleAbs = angle.abs();

    if (collided) {
      final onPad  = terrain.isOnPad(pos.x);
      final gentle = speedNow < cfg.landingSpeedMax && angleAbs < cfg.landingAngleMaxRad;

      if (onPad && gentle) {
        status = GameStatus.landed;
        pos = Vec2(pos.x, terrain.padY - LanderGeom.halfHeight);

        // Small terminal landing cost (prefer slow, straight, centered, fuel saved)
        final fuelN = (fuel / cfg.t.maxFuel).clamp(0.0, 1.0);
        final centerOffset = (pos.x - terrain.padCenter).abs();
        final landingCost =
            0.01 * speedNow * speedNow
                + 0.05 * angleDeg
                + 0.002 * centerOffset
                + 0.40 * (1.0 - fuelN);

        stepCost += landingCost.clamp(0.0, 1e6);
        cost += stepCost;

        lander = lander.copyWith(pos: pos, vel: Vec2.zero(), angle: angle, fuel: fuel);
        return StepInfo(status: status, terminal: true, speed: speedNow, angleAbs: angleAbs, onPad: true, costDelta: stepCost);
      } else {
        status = GameStatus.crashed;

        // Crash cost: big but not astronomical; worse with speed+tilt
        final crashCost = 120.0 + 0.03 * speedNow * speedNow + 0.6 * angleDeg;
        stepCost += crashCost.clamp(0.0, 1e9);
        cost += stepCost;

        lander = lander.copyWith(pos: pos, vel: Vec2.zero(), angle: angle, fuel: fuel);
        return StepInfo(status: status, terminal: true, speed: speedNow, angleAbs: angleAbs, onPad: onPad, costDelta: stepCost);
      }
    }

    // Commit
    cost += stepCost;
    lander = lander.copyWith(pos: pos, vel: vel, angle: angle, fuel: fuel);

    return StepInfo(
      status: status,
      terminal: false,
      speed: speedNow,
      angleAbs: angleAbs,
      onPad: false,
      costDelta: stepCost,
    );
  }

  List<double> observation() {
    final px = lander.pos.x / cfg.worldW;
    final py = lander.pos.y / cfg.worldH;
    final vx = lander.vel.x / 200.0;
    final vy = lander.vel.y / 200.0;
    final ang = lander.angle / math.pi;
    final fuelN = lander.fuel / cfg.t.maxFuel;
    final padCenterN = terrain.padCenter / cfg.worldW;
    final dxToCenter = (lander.pos.x - terrain.padCenter) / cfg.worldW;
    return [px, py, vx, vy, ang, fuelN, padCenterN, dxToCenter];
  }

  (Vec2, Vec2) _footPointsAt(Vec2 pos, double ang) {
    final c = math.cos(ang);
    final s = math.sin(ang);
    final bottomCenter = Vec2(pos.x, pos.y + LanderGeom.halfHeight);
    Vec2 rot(Vec2 v) => Vec2(c * v.x - s * v.y, s * v.x + c * v.y);
    final leftLocal  = Vec2(-LanderGeom.halfWidth, 0);
    final rightLocal = Vec2(LanderGeom.halfWidth, 0);
    return (bottomCenter + rot(leftLocal), bottomCenter + rot(rightLocal));
  }
}
