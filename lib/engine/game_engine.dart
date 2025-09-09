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
  double distanceTo(Vec2 o) => (this - o).length;
}

enum GameStatus { playing, landed, crashed }

class Tunables {
  double gravity;      // base gravity strength
  double thrustAccel;  // engine acceleration
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
  /// Time scale multiplier (kept to match your UI feel)
  final double stepScale;
  const EngineConfig({
    required this.worldW,
    required this.worldH,
    required this.t,
    this.seed = 42,
    this.stepScale = 60.0,
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

  static Terrain generate(double width, double height, {int seed = 42}) {
    final rnd = math.Random(seed);
    final List<Vec2> pts = [];
    const int segments = 24;
    for (int i = 0; i <= segments; i++) {
      final x = width * i / segments;
      final base = height * 0.78;
      final noise = (math.sin(i * 0.8) + rnd.nextDouble() * 0.5) * 24.0;
      pts.add(Vec2(x, base + noise));
    }

    final padWidth = width * 0.16;
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
  final double speed;      // absolute speed at collision or current frame
  final double angleAbs;   // |angle| at collision or current frame
  final bool onPad;        // at collision
  final double scoreDelta; // change in score on this step
  const StepInfo({
    required this.status,
    required this.terminal,
    required this.speed,
    required this.angleAbs,
    required this.onPad,
    required this.scoreDelta,
  });
}

class GameEngine {
  final EngineConfig cfg;
  late Terrain terrain;
  late LanderState lander;
  GameStatus status = GameStatus.playing;
  double score = 0.0;
  final math.Random _rnd;

  GameEngine(this.cfg) : _rnd = math.Random(cfg.seed) {
    reset(seed: cfg.seed);
  }

  void reset({int? seed}) {
    final s = seed ?? _rnd.nextInt(1 << 31);
    terrain = Terrain.generate(cfg.worldW, cfg.worldH, seed: s);
    status = GameStatus.playing;
    score  = 0.0;
    lander = LanderState(
      pos: Vec2(cfg.worldW * 0.2, 120),
      vel: Vec2.zero(),
      angle: 0.0,
      fuel: cfg.t.maxFuel,
    );
  }

  /// One physics + game-logic tick.
  /// Returns info you can use both for UI and RL (reward shaping).
  StepInfo step(double dtSeconds, ControlInput input) {
    if (status != GameStatus.playing) {
      return StepInfo(status: status, terminal: true, speed: 0, angleAbs: 0, onPad: false, scoreDelta: 0);
    }

    // Clamp dt (safety)
    final dt = dtSeconds.clamp(0.0, 1 / 20.0);
    final t = cfg.t;

    // --- Controls & rotation (scale 0.5 like in your UI) ---
    double angle = lander.angle;
    final rot = t.rotSpeed * 0.5;
    if (input.left && !input.right) angle -= rot * dt;
    if (input.right && !input.left) angle += rot * dt;

    // --- Acceleration (0.05 scalers) ---
    Vec2 accel = Vec2(0, t.gravity * 0.05);
    double fuel = lander.fuel;
    final thrustingNow = input.thrust && fuel > 0;
    if (thrustingNow) {
      final ax = math.sin(angle);
      final ay = -math.cos(angle);
      accel = accel + Vec2(ax, ay) * (t.thrustAccel * 0.05);
      fuel = (fuel - 20 * dt).clamp(0, t.maxFuel);
    }

    // --- Integrate (semi-implicit Euler; keep your 60x time scale) ---
    final vel = lander.vel + accel * (dt * cfg.stepScale);
    Vec2 pos = lander.pos + vel * (dt * cfg.stepScale);

    // --- Wrap horizontally ---
    final w = cfg.worldW;
    if (pos.x < 0) pos = Vec2(w + pos.x, pos.y);
    if (pos.x > w) pos = Vec2(pos.x - w, pos.y);

    // --- Collision ---
    final feet = _footPointsAt(pos, angle);
    final groundYLeft   = terrain.heightAt(feet.$1.x);
    final groundYRight  = terrain.heightAt(feet.$2.x);
    final groundYCenter = terrain.heightAt(pos.x);
    final collided = (feet.$1.y >= groundYLeft) ||
        (feet.$2.y >= groundYRight) ||
        (pos.y >= groundYCenter - 2);

    double stepReward = 0.0;
    final speedNow = vel.length;
    final angleAbs = angle.abs();

    // small living bonus to encourage control (optional)
    stepReward += 0.1; // per frame survival

    if (collided) {
      final onPad  = terrain.isOnPad(pos.x);
      final gentle = speedNow < 40 && angleAbs < 0.25;

      if (onPad && gentle) {
        status = GameStatus.landed;
        pos = Vec2(pos.x, terrain.padY - LanderGeom.halfHeight);

        // --------- Scoring on successful landing ----------
        // Base + fuel bonus - penalties for speed/angle/offset from pad center
        final angleDeg = angleAbs * 180 / math.pi;
        final centerOffset = (pos.x - terrain.padCenter).abs();
        final landingScore =
            1000.0
                + fuel * 2.0
                - (speedNow * 10.0)
                - (angleDeg * 5.0)
                - (centerOffset * 0.8);

        stepReward += landingScore.clamp(-5000.0, 2000.0);
        score += stepReward;

        lander = lander.copyWith(pos: pos, vel: Vec2.zero(), angle: angle, fuel: fuel);
        return StepInfo(
          status: status,
          terminal: true,
          speed: speedNow,
          angleAbs: angleAbs,
          onPad: true,
          scoreDelta: stepReward,
        );
      } else {
        status = GameStatus.crashed;

        // --------- Scoring on crash ----------
        // Harsh penalty; faster + crooked hits are worse.
        final angleDeg = angleAbs * 180 / math.pi;
        final crashScore =
            -200.0
                - (speedNow * 8.0)
                - (angleDeg * 4.0);

        stepReward += crashScore.clamp(-5000.0, 0.0);
        score += stepReward;

        lander = lander.copyWith(pos: pos, vel: Vec2.zero(), angle: angle, fuel: fuel);
        return StepInfo(
          status: status,
          terminal: true,
          speed: speedNow,
          angleAbs: angleAbs,
          onPad: onPad,
          scoreDelta: stepReward,
        );
      }
    }

    // No collision: commit state and accumulate living reward
    score += stepReward;
    lander = lander.copyWith(pos: pos, vel: vel, angle: angle, fuel: fuel);

    return StepInfo(
      status: status,
      terminal: false,
      speed: speedNow,
      angleAbs: angleAbs,
      onPad: false,
      scoreDelta: stepReward,
    );
  }

  // For RL: normalized observation vector in a compact order.
  // You can adapt ranges as you like.
  List<double> observation() {
    final px = lander.pos.x / cfg.worldW;        // 0..1
    final py = lander.pos.y / cfg.worldH;        // 0..1+
    final vx = lander.vel.x / 200.0;             // roughly -1..1
    final vy = lander.vel.y / 200.0;             // roughly -1..1
    final ang = lander.angle / math.pi;          // -1..1 (approximately)
    final fuelN = lander.fuel / cfg.t.maxFuel;   // 0..1
    final padCenterN = terrain.padCenter / cfg.worldW;
    final dxToCenter = (lander.pos.x - terrain.padCenter) / cfg.worldW; // -1..1
    return [px, py, vx, vy, ang, fuelN, padCenterN, dxToCenter];
  }

  // Helpers
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
