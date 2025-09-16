import 'dart:math' as math;
import 'types.dart';
import 'scoring.dart';

enum RayHitKind { terrain, wall, pad }

class RayHit {
  final Vector2 p;   // hit position (world px)
  final double t;    // param length along ray (px since dir is unit)
  final RayHitKind kind;
  const RayHit(this.p, this.t, this.kind);
}

class RayConfig {
  final int rayCount;
  final bool includeFloor;      // include y = worldH boundary
  final bool forwardAligned;    // bin 0 = ship forward (-Y)
  final int raySegments;        // how many samples for terrain polyline
  const RayConfig({
    this.rayCount = 180,
    this.includeFloor = false,
    this.forwardAligned = true,
    this.raySegments = 48,      // 24..96 is plenty for our terrain
  });

  RayConfig copyWith({int? rayCount, bool? includeFloor, bool? forwardAligned, int? raySegments}) => RayConfig(
    rayCount: rayCount ?? this.rayCount,
    includeFloor: includeFloor ?? this.includeFloor,
    forwardAligned: forwardAligned ?? this.forwardAligned,
    raySegments: raySegments ?? this.raySegments,
  );
}

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

  // --- Autolevel heuristics ---
  static const double _autoLevelBelow = 140.0;   // px
  static const double _autoLevelGain = 3.0;      // 1/sec
  static const double _flareBelow = 40.0;        // px
  static const double _flareGain = 8.0;          // 1/sec
  static const double _rotFriction = 0.6;        // rad/sec toward 0

  // --- Raycasting state ---
  RayConfig rayCfg = const RayConfig();
  List<RayHit> _rays = const [];
  List<RayHit> get rays => _rays;

  GameEngine(this.cfg) : _rnd = math.Random(cfg.seed) {
    reset(seed: cfg.seed);
  }

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

    if (cfg.lockTerrain) {
      _frozenTerrain ??= Terrain.generate(cfg.worldW, cfg.worldH, cfg.terrainSeed, cfg.padWidthFactor);
      terrain = _frozenTerrain!;
    } else {
      terrain = Terrain.generate(cfg.worldW, cfg.worldH, _rnd.nextInt(1 << 30), cfg.padWidthFactor);
    }

    // Spawn X
    double fracX;
    if (cfg.randomSpawnX) {
      final a = cfg.spawnXMin.clamp(0.0, 1.0);
      final b = cfg.spawnXMax.clamp(0.0, 1.0);
      final lo = math.min(a, b), hi = math.max(a, b);
      fracX = lo + _rnd.nextDouble() * (hi - lo);
    } else {
      fracX = cfg.spawnX;
    }

    // Lander state
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

    // Prime rays
    _rays = _castAll(from: lander.pos, angle: lander.angle);
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

    if (!(u.left ^ u.right)) {
      final pull = (_rotFriction * dt);
      if (pull > 0.0) angle -= angle * pull.clamp(0.0, 0.5);
    }

    // ----- Altitude-aware autolevel & flare assist -----
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

    // ----- Acceleration & fuel -----
    Vector2 accel = Vector2(0, t.gravity * 0.05);
    double fuel = lander.fuel;
    final bool thrusting = u.thrust && fuel > 0.0;
    final double power = thrusting ? 1.0 : 0.0;

    if (thrusting) {
      accel.x += math.sin(angle) * (t.thrustAccel * 0.05);
      accel.y += -math.cos(angle) * (t.thrustAccel * 0.05);
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

    // Re-evaluate height after integration; flare snap near ground
    final gyPost = terrain.heightAt(pos.x);
    final hPost = (gyPost - pos.y).toDouble();
    if (hPost < _flareBelow * 0.6) {
      final k = (_flareGain * 1.35) * dt;
      angle -= angle * k.clamp(0.0, 0.95);
    }

    // ----- Effort cost -----
    double stepCost = 0.0;
    if (power > 0) stepCost += cfg.effortCost * power * dt;

    // ----- Scoring/penalties/collisions -----
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

    // Commit state
    lander = LanderState(pos: out.pos, vel: out.vel, angle: angle, fuel: fuel);
    _lastAngle = angle;
    if (out.terminal) status = out.landed ? GameStatus.landed : GameStatus.crashed;

    // ----- Recompute engine-side rays for this new state -----
    _rays = _castAll(from: lander.pos, angle: lander.angle);

    return StepInfo(
      costDelta: stepCost,
      terminal: out.terminal,
      landed: out.landed,
      onPad: out.onPad,
    );
  }

  /// =======================
  /// Engine-side Raycasting
  /// =======================

  // Build a sampled polyline from heightAt(x) without touching Terrain internals.
  List<Vector2> _terrainPolyline() {
    final int segs = math.max(4, rayCfg.raySegments);
    final double W = cfg.worldW.toDouble();
    final List<Vector2> pts = List<Vector2>.generate(segs + 1, (i) {
      final x = (W * i) / segs;
      final y = terrain.heightAt(x);
      return Vector2(x, y);
    }, growable: false);
    return pts;
  }

  List<RayHit> _castAll({required Vector2 from, required double angle}) {
    final int n = rayCfg.rayCount;
    if (n <= 0) return const [];
    final List<RayHit> hits = List.filled(n, RayHit(Vector2(0, 0), double.infinity, RayHitKind.wall));

    final double base = rayCfg.forwardAligned ? (angle - math.pi / 2.0) : 0.0;
    final double twoPi = math.pi * 2.0;

    // Cache polyline once per step
    final List<Vector2> ridge = _terrainPolyline();

    for (int i = 0; i < n; i++) {
      final th = base + twoPi * (i / n);
      final dir = Vector2(math.cos(th), math.sin(th)); // unit
      hits[i] = _castOne(from, dir, ridge);
    }
    return hits;
  }

  RayHit _castOne(Vector2 o, Vector2 d, List<Vector2> ridge) {
    double bestT = double.infinity;
    RayHitKind bestKind = RayHitKind.wall;
    Vector2 bestP = o;

    // Terrain polyline (sampled)
    for (int i = 0; i < ridge.length - 1; i++) {
      final A = ridge[i];
      final B = ridge[i + 1];
      final sol = _raySegment(o, d, A, B);
      if (sol != null && sol.$1 < bestT) {
        bestT = sol.$1;
        bestP = Vector2(o.x + d.x * bestT, o.y + d.y * bestT);
        bestKind = _segmentIsPad(A, B) ? RayHitKind.pad : RayHitKind.terrain;
      }
    }

    // Arena bounds
    final double W = cfg.worldW.toDouble();
    final double H = cfg.worldH.toDouble();

    _hitLineSegment(o, d, Vector2(0, 0), Vector2(W, 0), RayHitKind.wall, (t, p, k) {
      if (t < bestT) { bestT = t; bestP = p; bestKind = k; }
    });
    _hitLineSegment(o, d, Vector2(0, 0), Vector2(0, H), RayHitKind.wall, (t, p, k) {
      if (t < bestT) { bestT = t; bestP = p; bestKind = k; }
    });
    _hitLineSegment(o, d, Vector2(W, 0), Vector2(W, H), RayHitKind.wall, (t, p, k) {
      if (t < bestT) { bestT = t; bestP = p; bestKind = k; }
    });
    if (rayCfg.includeFloor) {
      _hitLineSegment(o, d, Vector2(0, H), Vector2(W, H), RayHitKind.wall, (t, p, k) {
        if (t < bestT) { bestT = t; bestP = p; bestKind = k; }
      });
    }

    if (!bestT.isFinite) {
      // Fallback: project far and clamp to world
      final far = Vector2(o.x + d.x * 5000.0, o.y + d.y * 5000.0);
      final clamped = Vector2(far.x.clamp(0.0, W), far.y.clamp(0.0, H));
      final t = (clamped - o).length;
      return RayHit(clamped, t, RayHitKind.wall);
    }
    return RayHit(bestP, bestT, bestKind);
  }

  // Ray o + t*d vs segment AB. Returns (t,u) or null.
  (double, double)? _raySegment(Vector2 o, Vector2 d, Vector2 a, Vector2 b) {
    final vx = d.x, vy = d.y;
    final sx = b.x - a.x, sy = b.y - a.y;
    final det = (-sx * vy + vx * sy);
    if (det.abs() < 1e-9) return null; // parallel

    final oxax = o.x - a.x;
    final oway = o.y - a.y;
    final t = (-sy * oxax + sx * oway) / det;
    final u = (-vy * oxax + vx * oway) / det;

    if (t >= 0 && u >= 0 && u <= 1) return (t, u);
    return null;
  }

  void _hitLineSegment(
      Vector2 o, Vector2 d, Vector2 a, Vector2 b, RayHitKind kind,
      void Function(double t, Vector2 p, RayHitKind kind) accept,
      ) {
    final sol = _raySegment(o, d, a, b);
    if (sol != null) {
      final t = sol.$1;
      if (t >= 0) {
        final p = Vector2(o.x + d.x * t, o.y + d.y * t);
        accept(t, p, kind);
      }
    }
  }

  bool _segmentIsPad(Vector2 a, Vector2 b) {
    const eps = 1e-6;
    bool xIn(Vector2 p) => p.x >= terrain.padX1 - eps && p.x <= terrain.padX2 + eps;
    return xIn(a) && xIn(b) && (a.y - terrain.padY).abs() < 1e-6 && (b.y - terrain.padY).abs() < 1e-6;
  }

  /// NN-friendly channels (unchanged)
  List<double> rayChannels3({bool useProximity = false}) {
    final hits = _rays;
    if (hits.isEmpty) return const <double>[];

    final int n = hits.length;
    final double diag = math.sqrt(cfg.worldW * cfg.worldW + cfg.worldH * cfg.worldH);

    final pad = List<double>.filled(n, 1.0, growable: false);
    final ter = List<double>.filled(n, 1.0, growable: false);
    final wal = List<double>.filled(n, 1.0, growable: false);

    double enc(double d) {
      final dn = (d / diag).clamp(0.0, 1.0);
      if (!useProximity) return dn;
      return 1.0 - dn; // proximity
    }

    for (int i = 0; i < n; i++) {
      final h = hits[i];
      final v = enc(h.t);
      switch (h.kind) {
        case RayHitKind.pad:     pad[i] = v; break;
        case RayHitKind.terrain: ter[i] = v; break;
        case RayHitKind.wall:    wal[i] = v; break;
      }
    }

    return <double>[...pad, ...ter, ...wal];
  }

  ({double minD, double bearing, double visible}) padSummary() {
    if (_rays.isEmpty) return (minD: 1.0, bearing: 0.0, visible: 0.0);
    final int n = _rays.length;
    final double diag = math.sqrt(cfg.worldW * cfg.worldW + cfg.worldH * cfg.worldH);

    double minD = 1.0;
    int minIdx = -1;
    for (int i = 0; i < n; i++) {
      if (_rays[i].kind == RayHitKind.pad) {
        final dn = (_rays[i].t / diag).clamp(0.0, 1.0);
        if (dn < minD) { minD = dn; minIdx = i; }
      }
    }
    final visible = minD < 1.0 ? 1.0 : 0.0;
    double bearing = 0.0;
    if (minIdx >= 0) {
      bearing = (minIdx / (n / 2.0)) - 1.0; // map [0..n) -> [-1,1]
      if (bearing < -1) bearing += 2;
      if (bearing > 1)  bearing -= 2;
    }
    return (minD: minD, bearing: bearing, visible: visible);
  }
}
