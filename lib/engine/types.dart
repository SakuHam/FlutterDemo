// lib/engine/types.dart
import 'dart:math' as math;

class Tunables {
  double gravity;
  double thrustAccel;
  double rotSpeed;
  double maxFuel;

  final bool crashOnTilt;
  final double landingMaxVx;
  final double landingMaxVy;
  final double landingMaxOmega;

  Tunables({
    this.gravity = 0.18,
    this.thrustAccel = 0.42,
    this.rotSpeed = 1.6,
    this.maxFuel = 1000.0,
    this.crashOnTilt = true,
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
  final double stepScale;

  final double padWidthFactor;
  final double landingSpeedMax;
  final double landingAngleMaxRad;

  final double livingCost;
  final double effortCost;
  final double wDx;
  final double wDy;
  final double wVyDown;
  final double wVx;
  final double wAngleDeg;
  final double angleNearGroundBoost;
  final double wAngleRate;

  final double borderMargin;
  final double borderPenaltyPerSec;
  final double wrapPenalty;

  final bool lockTerrain;
  final int terrainSeed;
  final bool lockSpawn;
  final double spawnX;
  final double spawnY;
  final double spawnVx;
  final double spawnVy;
  final double spawnAngle;
  final bool hardWalls;

  final double ceilingMargin;
  final double ceilingPenaltyPerSec;
  final double ceilingHitPenalty;

  final bool randomSpawnX;
  final double spawnXMin;
  final double spawnXMax;

  final double wThrustAssist;
  final double wTurnAssist;
  final double wPadAlignAssist;
  final double wIdlePenalty;

  final double fuelPenaltyPerSec;
  final double upwardPenaltyPerVel;
  final double loiterPenaltyPerSec;
  final double loiterMinVyDown;
  final double loiterAltFrac;

  final double padTurnDeadzoneFrac;
  final double padFarFrac;
  final double padDirTurnPenaltyPerSec;
  final double padIdleTurnPenaltyPerSec;
  final double padVelAwayPenaltyPerVel;
  final double padAngleAwayPenaltyPerRad;

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
    this.fuelPenaltyPerSec = 6.0,
    this.upwardPenaltyPerVel = 0.05,
    this.loiterPenaltyPerSec = 20.0,
    this.loiterMinVyDown = 45.0,
    this.loiterAltFrac = 0.35,
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

class LanderState {
  final int seed = 12345;
  Vector2 pos;
  Vector2 vel;
  double angle;
  double fuel;

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

enum PolyEdgeKind { terrain, pad }

class PolyEdge {
  final Vector2 a;
  final Vector2 b;
  final PolyEdgeKind kind;
  const PolyEdge(this.a, this.b, this.kind);
}

class PolyShape {
  final List<Vector2> outer;         // CCW
  final List<List<Vector2>> holes;   // CW
  final List<PolyEdge> edges;

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

  double verticalHitY({required double x, required double worldH}) {
    double bestY = double.infinity;
    for (final e in edges) {
      final y = _xRayHitY(e.a, e.b, x);
      if (y != null && y >= 0.0 && y < bestY) bestY = y;
    }
    if (!bestY.isFinite) return double.infinity;
    return math.min(bestY, worldH);
  }

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

  static double? _xRayHitY(Vector2 a, Vector2 b, double X) {
    if ((X < math.min(a.x, b.x)) || (X > math.max(a.x, b.x))) return null;
    final dx = (b.x - a.x);
    if (dx.abs() < 1e-9) return math.min(a.y, b.y);
    final t = (X - a.x) / dx;
    if (t < 0.0 || t > 1.0) return null;
    return a.y + (b.y - a.y) * t;
  }

  static (double, double)? _raySegment(Vector2 o, Vector2 d, Vector2 a, Vector2 b) {
    final vx = d.x, vy = d.y;
    final sx = b.x - a.x, sy = b.y - a.y;
    final det = (-sx * vy + vx * sy);
    if (det.abs() < 1e-9) return null;

    final oxax = o.x - a.x;
    final oway = o.y - a.y;
    final t = (-sy * oxax + sx * oway) / det;
    final u = (-vy * oxax + vx * oway) / det;
    if (u < 0.0 || u > 1.0) return null;
    return (t, u);
  }
}

class Terrain {
  final PolyShape poly;

  final List<Vector2> ridge;
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

  double heightAt(double x, {double worldH = 100000.0}) {
    return poly.verticalHitY(x: x, worldH: worldH);
  }

  bool isOnPad(double x) => x >= padX1 && x <= padX2;

  /// Higher-reaching terrain:
  /// - Shifted band upward (smaller Y = higher on screen)
  /// - Slightly larger amplitude + two extra harmonics for detail
  /// - Keeps valleys tamed so gameplay stays fair
  static Terrain generate(double w, double h, int seed, double padWidthFactor) {
    final rnd = math.Random(seed);

    final roof = <Vector2>[];
    const int segments = 36;      // more segments for smoother relief

    // Screen Y grows downward. Smaller fraction => visually higher.
    final baseFrac   = 0.56;      // was 0.66  (lift the whole terrain up)
    final noiseFrac  = 0.10;      // was 0.06  (allow taller peaks)
    final bandTop    = 0.34;      // was 0.54  (let peaks go higher)
    final bandBottom = 0.70;      // was 0.76  (keep valleys reasonable)

    final baseY   = h * baseFrac;
    final topY    = h * bandTop;
    final botY    = h * bandBottom;
    final noiseAmp = h * noiseFrac;

    // Simple fBm-ish mixture (sine layers + small jitter) for more shape variety
    final phaseA = rnd.nextDouble() * math.pi * 2.0;
    final phaseB = rnd.nextDouble() * math.pi * 2.0;
    final phaseC = rnd.nextDouble() * math.pi * 2.0;

    for (int i = 0; i <= segments; i++) {
      final x = w * i / segments;
      final t = i * 1.0;

      // Mix multiple frequencies; keep weights summing to ~1
      final n1 = math.sin(t * 0.55 + phaseA);      // broad undulation
      final n2 = math.sin(t * 0.18 + phaseB);      // long swell
      final n3 = math.sin(t * 1.10 + phaseC);      // small detail

      final jitter = (rnd.nextDouble() - 0.5) * 0.35;
      final noise = (0.55 * n1 + 0.30 * n2 + 0.15 * n3 + jitter) * noiseAmp;

      double y = baseY + noise;

      // Clamp inside the band (peaks high, valleys not too deep)
      if (y < topY) y = topY;
      if (y > botY) y = botY;

      roof.add(Vector2(x, y));
    }

    // Landing pad -------------------------------------------------------------
    final rawPadWidth = w * 0.16 * padWidthFactor;
    final padWidth = rawPadWidth.clamp(36.0, w * 0.6);
    final padCenterX = w * (0.35 + rnd.nextDouble() * 0.3);
    final padX1 = math.max(10.0, math.min(padCenterX - padWidth / 2, w - padWidth - 10.0));
    final padX2 = padX1 + padWidth;

    double interpRoofY(double x) => _interpYOnPolyline(roof, x);
    final y1 = interpRoofY(padX1);
    final y2 = interpRoofY(padX2);
    final ym = interpRoofY((padX1 + padX2) * 0.5);

    // Keep the pad just below the local min of the roof span for clearance
    final padYTarget = h * 0.58; // lift pad a bit, since terrain is higher overall
    final localMin = math.min(y1, math.min(y2, ym));
    final padY = math.min(padYTarget, localMin);

    final roofWithPad = _insertPadIntoRoof(roof, padX1, padX2, padY);

    // Close polygon; keep ends slightly deeper to form a gentle bowl at edges
    final endsFrac = 0.78; // was 0.80; tiny lift so edges aren’t too low now that band moved up
    final outer = <Vector2>[
      Vector2(0, h * endsFrac),
      ...roofWithPad,
      Vector2(w, h * endsFrac),
    ];

    final poly = PolyShape.fromRings(
      outer: outer,
      holes: const [],
      padX1: padX1,
      padX2: padX2,
      padY: padY,
    );

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

double _interpYOnPolyline(List<Vector2> line, double x) {
  for (int i = 0; i < line.length - 1; i++) {
    final a = line[i];
    final b = line[i + 1];
    if ((x >= a.x && x <= b.x) || (x >= b.x && x <= a.x)) {
      final t = (x - a.x) / (b.x - a.x);
      return a.y + (b.y - a.y) * t;
    }
  }
  if (x < line.first.x) return line.first.y;
  return line.last.y;
}

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

    if (out.isEmpty) out.add(a);

    bool crosses(double x) =>
        (x >= math.min(a.x, b.x)) && (x <= math.max(a.x, b.x)) && (a.x != b.x);

    if (!inside && crosses(padX1)) {
      final yEnter = lerpAt(a, b, padX1);
      out.add(Vector2(padX1, yEnter));
      out.add(Vector2(padX1, padY));
      inside = true;
    }

    if (inside && crosses(padX2)) {
      final yExit = lerpAt(a, b, padX2);
      out.add(Vector2(padX2, padY));
      out.add(Vector2(padX2, yExit));
      inside = false;
      out.add(b);
      continue;
    }

    if (!inside) out.add(b);
  }

  if (inside) {
    final last = roof.last;
    out.add(Vector2(padX2, padY));
    out.add(Vector2(padX2, last.y));
    out.add(last);
  }

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

class StepInfo {
  final double costDelta;
  final double scoreDelta;
  final bool terminal;
  final bool landed;
  final bool onPad;

  StepInfo({
    required this.costDelta,
    required this.terminal,
    required this.landed,
    required this.onPad,
  }) : scoreDelta = costDelta;
}
