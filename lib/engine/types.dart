// lib/engine/types.dart
import 'dart:math' as math;

/// =======================
/// Config & basic structs
/// =======================

class Tunables {
  double gravity;      // base gravity strength
  double thrustAccel;  // engine acceleration (px/s^2 @ power=1)
  double rotSpeed;     // radians per second
  double maxFuel;      // fuel units

  // Touchdown tolerances / assists
  final bool crashOnTilt;           // if false, angle is ignored for touchdown safety
  final double landingMaxVx;        // px/s horizontal speed allowed at contact
  final double landingMaxVy;        // px/s downward speed allowed at contact
  final double landingMaxOmega;     // rad/s allowed at contact (optional)

  Tunables({
    this.gravity = 0.18,
    this.thrustAccel = 0.42,
    this.rotSpeed = 1.6,
    this.maxFuel = 1000.0,
    this.crashOnTilt = true,        // old behavior = true
    this.landingMaxVx = 28.0,
    this.landingMaxVy = 38.0,
    this.landingMaxOmega = 2.0,
  });

  Tunables copyWith({
    double? gravity,
    double? thrustAccel,
    double? rotSpeed,
    double? maxFuel,
    bool? crashOnTilt,
    double? landingMaxVx,
    double? landingMaxVy,
    double? landingMaxOmega,
  }) {
    return Tunables(
      gravity: gravity ?? this.gravity,
      thrustAccel: thrustAccel ?? this.thrustAccel,
      rotSpeed: rotSpeed ?? this.rotSpeed,
      maxFuel: maxFuel ?? this.maxFuel,
      crashOnTilt: crashOnTilt ?? this.crashOnTilt,
      landingMaxVx: landingMaxVx ?? this.landingMaxVx,
      landingMaxVy: landingMaxVy ?? this.landingMaxVy,
      landingMaxOmega: landingMaxOmega ?? this.landingMaxOmega,
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

  // Heuristic assists (kept; optional)
  final double wThrustAssist;
  final double wTurnAssist;
  final double wPadAlignAssist;
  final double wIdlePenalty;

  // ===== Anti-hover / fuel discouragement knobs =====
  final double fuelPenaltyPerSec;   // cost/sec while thrust is ON
  final double upwardPenaltyPerVel; // cost per (px/s) upward velocity (vy < 0)
  final double loiterPenaltyPerSec; // cost/sec when too high & not descending fast enough
  final double loiterMinVyDown;     // desired minimum downward speed (px/s, vy+)
  final double loiterAltFrac;       // start enforcing loitering above this fraction of worldH

  // ===== Pad-alignment shaping =====
  final double padTurnDeadzoneFrac;     // dx/worldW tolerance before we care
  final double padFarFrac;              // "far" when |dx| > padFarFrac * padHalfWidth
  final double padDirTurnPenaltyPerSec; // cost/sec if turning opposite of pad direction
  final double padIdleTurnPenaltyPerSec;// cost/sec if not turning when far from pad
  final double padVelAwayPenaltyPerVel; // cost per (px/s) of velocity away from pad
  final double padAngleAwayPenaltyPerRad; // cost per rad when leaning away

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

    // Anti-hover defaults
    this.fuelPenaltyPerSec = 6.0,
    this.upwardPenaltyPerVel = 0.05,
    this.loiterPenaltyPerSec = 20.0,
    this.loiterMinVyDown = 45.0,
    this.loiterAltFrac = 0.35,

    // Pad-alignment defaults
    this.padTurnDeadzoneFrac = 0.03,
    this.padFarFrac = 1.0,
    this.padDirTurnPenaltyPerSec = 6.0,
    this.padIdleTurnPenaltyPerSec = 2.0,
    this.padVelAwayPenaltyPerVel = 0.02,
    this.padAngleAwayPenaltyPerRad = 0.02,
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
    double? wThrustAssist,
    double? wTurnAssist,
    double? wPadAlignAssist,
    double? wIdlePenalty,
    double? fuelPenaltyPerSec,
    double? upwardPenaltyPerVel,
    double? loiterPenaltyPerSec,
    double? loiterMinVyDown,
    double? loiterAltFrac,
    double? padTurnDeadzoneFrac,
    double? padFarFrac,
    double? padDirTurnPenaltyPerSec,
    double? padIdleTurnPenaltyPerSec,
    double? padVelAwayPenaltyPerVel,
    double? padAngleAwayPenaltyPerRad,
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
      wThrustAssist: wThrustAssist ?? this.wThrustAssist,
      wTurnAssist: wTurnAssist ?? this.wTurnAssist,
      wPadAlignAssist: wPadAlignAssist ?? this.wPadAlignAssist,
      wIdlePenalty: wIdlePenalty ?? this.wIdlePenalty,
      fuelPenaltyPerSec: fuelPenaltyPerSec ?? this.fuelPenaltyPerSec,
      upwardPenaltyPerVel: upwardPenaltyPerVel ?? this.upwardPenaltyPerVel,
      loiterPenaltyPerSec: loiterPenaltyPerSec ?? this.loiterPenaltyPerSec,
      loiterMinVyDown: loiterMinVyDown ?? this.loiterMinVyDown,
      loiterAltFrac: loiterAltFrac ?? this.loiterAltFrac,
      padTurnDeadzoneFrac: padTurnDeadzoneFrac ?? this.padTurnDeadzoneFrac,
      padFarFrac: padFarFrac ?? this.padFarFrac,
      padDirTurnPenaltyPerSec: padDirTurnPenaltyPerSec ?? this.padDirTurnPenaltyPerSec,
      padIdleTurnPenaltyPerSec: padIdleTurnPenaltyPerSec ?? this.padIdleTurnPenaltyPerSec,
      padVelAwayPenaltyPerVel: padVelAwayPenaltyPerVel ?? this.padVelAwayPenaltyPerVel,
      padAngleAwayPenaltyPerRad: padAngleAwayPenaltyPerRad ?? this.padAngleAwayPenaltyPerRad,
    );
  }
}

enum GameStatus { playing, landed, crashed }

class ControlInput {
  final bool thrust;
  final bool left;
  final bool right;
  final int? intentIdx;

  const ControlInput({
    required this.thrust,
    required this.left,
    required this.right,
    this.intentIdx,
  });
}

class Vector2 {
  double x, y;
  Vector2(this.x, this.y);

  Vector2 operator +(Vector2 o) => Vector2(x + o.x, y + o.y);
  Vector2 operator -(Vector2 o) => Vector2(x - o.x, y - o.y);
  Vector2 operator *(double s) => Vector2(x * s, y * s);
  double get length => math.sqrt(x * x + y * y);
}

/// Lander state
class LanderState {
  final int seed = 12345;
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

/// =======================
/// Polygon terrain support
/// =======================

/// Tag polygon edges so sensors and contact can differentiate pad vs terrain.
enum PolyEdgeKind { terrain, pad }

class PolyEdge {
  final Vector2 a;
  final Vector2 b;
  final PolyEdgeKind kind;
  const PolyEdge(this.a, this.b, this.kind);
}

/// A polygon with an outer ring and optional hole rings (for caverns).
/// Rings should be simple (non-self-intersecting). We normalize winding:
/// outer CCW, holes CW (convention; only consistency matters).
class PolyShape {
  final List<Vector2> outer;         // CCW
  final List<List<Vector2>> holes;   // CW
  final List<PolyEdge> edges;        // derived (outer + holes)

  PolyShape._(this.outer, this.holes, this.edges);

  factory PolyShape.fromRings({
    required List<Vector2> outer,
    List<List<Vector2>> holes = const [],
    double? padX1,
    double? padX2,
    double? padY,
  }) {
    final out = _ensureWinding(outer, ccw: true);
    final hol = [for (final h in holes) _ensureWinding(h, ccw: false)];

    final edges = <PolyEdge>[];
    void addRing(List<Vector2> r) {
      final n = r.length;
      for (int i = 0; i < n; i++) {
        final a = r[i];
        final b = r[(i + 1) % n];
        final tagAsPad = (padX1 != null && padX2 != null && padY != null) &&
            _segmentIsPad(a, b, padX1, padX2, padY);
        edges.add(PolyEdge(a, b, tagAsPad ? PolyEdgeKind.pad : PolyEdgeKind.terrain));
      }
    }

    addRing(out);
    for (final h in hol) addRing(h);

    return PolyShape._(out, hol, edges);
  }

  /// Vertical query: smallest y >= 0 where x hits the polygon boundary.
  /// Returns +∞ if no hit in [0, worldH].
  double verticalHitY({required double x, required double worldH}) {
    double bestY = double.infinity;
    for (final e in edges) {
      final y = _xRayHitY(e.a, e.b, x);
      if (y != null && y >= 0.0 && y < bestY) bestY = y;
    }
    if (!bestY.isFinite) return double.infinity;
    return math.min(bestY, worldH);
  }

  /// Raycast against all edges; returns nearest hit (t, p, kind) or null.
  (double t, Vector2 p, PolyEdgeKind kind)? raycast({
    required Vector2 origin,
    required Vector2 dir,
  }) {
    double bestT = double.infinity;
    Vector2? bestP;
    PolyEdgeKind bestK = PolyEdgeKind.terrain;

    for (final e in edges) {
      final sol = _raySegment(origin, dir, e.a, e.b);
      if (sol != null && sol.$1 >= 0.0 && sol.$1 < bestT) {
        bestT = sol.$1;
        bestP = Vector2(origin.x + dir.x * bestT, origin.y + dir.y * bestT);
        bestK = e.kind;
      }
    }
    if (!bestT.isFinite || bestP == null) return null;
    return (bestT, bestP, bestK);
  }

  // ---- helpers ----

  static List<Vector2> _ensureWinding(List<Vector2> ring, {required bool ccw}) {
    double area = 0.0;
    for (int i = 0; i < ring.length; i++) {
      final a = ring[i];
      final b = ring[(i + 1) % ring.length];
      area += (a.x * b.y - b.x * a.y);
    }
    final isCCW = area > 0;
    if (isCCW == ccw) return ring;
    return List<Vector2>.from(ring.reversed);
  }

  static bool _segmentIsPad(Vector2 a, Vector2 b, double x1, double x2, double y) {
    const eps = 1e-3;
    final minX = math.min(a.x, b.x), maxX = math.max(a.x, b.x);
    final flatY = (a.y - y).abs() < eps && (b.y - y).abs() < eps;
    final overlap = !(maxX < x1 || minX > x2);
    return flatY && overlap;
  }

  // Intersection of vertical line x = X with segment AB → Y or null.
  static double? _xRayHitY(Vector2 a, Vector2 b, double X) {
    if ((X < math.min(a.x, b.x)) || (X > math.max(a.x, b.x))) return null;
    final dx = (b.x - a.x);
    if (dx.abs() < 1e-9) return math.min(a.y, b.y); // vertical seg → top endpoint
    final t = (X - a.x) / dx; // 0..1 along AB by x
    if (t < 0.0 || t > 1.0) return null;
    return a.y + (b.y - a.y) * t;
  }

  // Parametric ray/segment intersection. Returns (t,u) or null.
  static (double, double)? _raySegment(Vector2 o, Vector2 d, Vector2 a, Vector2 b) {
    final vx = d.x, vy = d.y;
    final sx = b.x - a.x, sy = b.y - a.y;
    final det = (-sx * vy + vx * sy);
    if (det.abs() < 1e-9) return null; // parallel

    final oxax = o.x - a.x;
    final oway = o.y - a.y;
    final t = (-sy * oxax + sx * oway) / det;
    final u = (-vy * oxax + vx * oway) / det;
    if (u < 0.0 || u > 1.0) return null;
    return (t, u);
  }
}

/// =======================
/// Terrain (back-compat wrapper)
/// =======================
/// Terrain now owns a polygon (outer + holes) but keeps the legacy fields
/// so existing code continues to work.
class Terrain {
  final PolyShape poly;

  // Legacy fields (preserved)
  final List<Vector2> ridge; // kept for compatibility (approx “roof” polyline)
  final double padX1;
  final double padX2;
  final double padY;

  Terrain({
    required this.poly,
    required this.ridge,
    required this.padX1,
    required this.padX2,
    required this.padY,
  });

  double get padCenter => (padX1 + padX2) * 0.5;

  /// Back-compat: returns the first boundary y above (>=0) at x, clamped to worldH.
  /// For overhangs/caverns, this is the *nearest* boundary above the top of the screen,
  /// which is what you want for landing/autolevel checks.
  double heightAt(double x, {double worldH = 100000.0}) {
    return poly.verticalHitY(x: x, worldH: worldH);
  }

  bool isOnPad(double x) => x >= padX1 && x <= padX2;

  /// Generate a polygonal terrain with the landing pad spliced into the roof.
  static Terrain generate(double w, double h, int seed, double padWidthFactor) {
    final rnd = math.Random(seed);

    // 1) Build the *roof* polyline (left→right), independent from bottom closure.
    final roof = <Vector2>[];
    const int segments = 28;
    for (int i = 0; i <= segments; i++) {
      final x = w * i / segments;
      final base = h * 0.78;
      final noise = (math.sin(i * 0.8) + rnd.nextDouble() * 0.5) * 24.0;
      roof.add(Vector2(x, base + noise));
    }

    // 2) Choose pad span on roof.
    final rawPadWidth = w * 0.16 * padWidthFactor;
    final padWidth = rawPadWidth.clamp(36.0, w * 0.6);
    final padCenterX = w * (0.35 + rnd.nextDouble() * 0.3);
    final padX1 = math.max(10.0, math.min(padCenterX - padWidth / 2, w - padWidth - 10.0));
    final padX2 = padX1 + padWidth;

    // Interpolate roof Y at pad endpoints (and mid) and pick a stable padY on/under roof.
    double interpRoofY(double x) => _interpYOnPolyline(roof, x);
    final y1 = interpRoofY(padX1);
    final y2 = interpRoofY(padX2);
    final ym = interpRoofY((padX1 + padX2) * 0.5);
    final padY = math.min(y1, math.min(y2, ym)); // keep pad "embedded" in roof

    // 3) Splice a *flat* pad edge into the roof between [padX1, padX2].
    final roofWithPad = _insertPadIntoRoof(roof, padX1, padX2, padY);

    // 4) Build an *outer* polygon by closing roof to the bottom corners.
    final outer = <Vector2>[
      Vector2(0, h),       // bottom-left
      ...roofWithPad,      // roof (now includes flat pad edge)
      Vector2(w, h),       // bottom-right
    ];

    // 5) Optional cavern hole (unchanged behavior).
    final holes = <List<Vector2>>[];
    if (rnd.nextBool()) {
      final cx = w * (0.30 + rnd.nextDouble() * 0.4);
      final cy = h * 0.70;
      final rx = w * 0.10;
      final ry = h * 0.06;
      const k = 24;
      final ring = List<Vector2>.generate(k, (i) {
        final th = 2 * math.pi * i / k;
        return Vector2(cx + rx * math.cos(th), cy + ry * math.sin(th));
      });
      holes.add(ring);
    }

    // 6) Build polygon with pad edges *tagged* (the flat segment we spliced).
    final poly = PolyShape.fromRings(
      outer: outer,
      holes: holes,
      padX1: padX1,
      padX2: padX2,
      padY: padY,
    );

    // 7) Legacy ridge for old UIs (use roofWithPad so the painter can draw a flat pad).
    final ridge = roofWithPad;

    return Terrain(
      poly: poly,
      ridge: ridge,
      padX1: padX1,
      padX2: padX2,
      padY: padY,
    );
  }
}

/// Interpolate y on a monotone-by-x polyline (assumes points are ordered by x).
double _interpYOnPolyline(List<Vector2> line, double x) {
  for (int i = 0; i < line.length - 1; i++) {
    final a = line[i];
    final b = line[i + 1];
    if ((x >= a.x && x <= b.x) || (x >= b.x && x <= a.x)) {
      final t = (x - a.x) / (b.x - a.x);
      return a.y + (b.y - a.y) * t;
    }
  }
  // Out of range: clamp to nearest endpoint
  if (x < line.first.x) return line.first.y;
  return line.last.y;
}

/// Splice a flat segment [padX1..padX2] at y=padY into the roof polyline.
/// Removes original roof points inside the span and inserts intersection points
/// at the span ends + flat pad vertices.
List<Vector2> _insertPadIntoRoof(
    List<Vector2> roof,
    double padX1,
    double padX2,
    double padY,
    ) {
  assert(padX2 > padX1);
  final out = <Vector2>[];
  bool inside = false;

  double lerpAt(Vector2 a, Vector2 b, double x) {
    final t = (x - a.x) / (b.x - a.x);
    return a.y + (b.y - a.y) * t;
  }

  for (int i = 0; i < roof.length - 1; i++) {
    final a = roof[i];
    final b = roof[i + 1];

    // push first point once
    if (out.isEmpty) out.add(a);

    bool crosses(double x) =>
        (x >= math.min(a.x, b.x)) && (x <= math.max(a.x, b.x)) && (a.x != b.x);

    // Enter pad span
    if (!inside && crosses(padX1)) {
      final yEnter = lerpAt(a, b, padX1);
      out.add(Vector2(padX1, yEnter));      // intersection on roof
      out.add(Vector2(padX1, padY));        // drop/raise to flat pad
      inside = true;
    }

    // Exit pad span
    if (inside && crosses(padX2)) {
      final yExit = lerpAt(a, b, padX2);
      out.add(Vector2(padX2, padY));        // end of flat pad
      out.add(Vector2(padX2, yExit));       // connect back to roof
      inside = false;
      // After reconnecting, continue with normal path from B
      out.add(b);
      continue;
    }

    // While inside pad span, skip original roof points (we already inserted flat pad).
    if (!inside) {
      out.add(b);
    }
  }

  // Edge-case: if span extended beyond last segment (unlikely), close it.
  if (inside) {
    final last = roof.last;
    out.add(Vector2(padX2, padY));
    out.add(Vector2(padX2, last.y));
    out.add(last);
  }

  // Merge tiny duplicates that can be created by numeric ties
  const eps = 1e-6;
  final compact = <Vector2>[];
  for (final p in out) {
    if (compact.isEmpty) {
      compact.add(p);
    } else {
      final q = compact.last;
      if ((p.x - q.x).abs() > eps || (p.y - q.y).abs() > eps) {
        compact.add(p);
      }
    }
  }
  return compact;
}

/// Step outcome
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
