// lib/engine/raycast.dart
import 'dart:math' as math;

import 'types.dart';

enum RayHitKind { terrain, wall, pad }

class RayHit {
  final Vector2 p;   // hit position (world px)
  final double t;    // param length along ray
  final RayHitKind kind;
  const RayHit(this.p, this.t, this.kind);
}

class RayConfig {
  final int rayCount;
  final bool includeFloor;    // include y = worldH
  final bool forwardAligned;  // bin 0 = ship forward (-Y)
  const RayConfig({
    this.rayCount = 180,
    this.includeFloor = false,
    this.forwardAligned = false,
  });

  RayConfig copyWith({int? rayCount, bool? includeFloor, bool? forwardAligned}) => RayConfig(
    rayCount: rayCount ?? this.rayCount,
    includeFloor: includeFloor ?? this.includeFloor,
    forwardAligned: forwardAligned ?? this.forwardAligned,
  );
}

/// Stateless raycaster (holds only config).
class RayCaster {
  RayConfig cfg;
  RayCaster(this.cfg);

  List<RayHit> castAll({
    required Terrain terrain,
    required EngineConfig engineCfg,
    required Vector2 origin,
    required double angle,
  }) {
    final n = cfg.rayCount;
    if (n <= 0) return const [];

    final hits = List<RayHit>.generate(
      n,
          (_) => RayHit(Vector2(0, 0), double.infinity, RayHitKind.wall),
      growable: false,
    );

    // Bin 0 points to ship forward (-Y) -> angle - pi/2 in screen coords.
    final base = cfg.forwardAligned ? (angle - math.pi / 2.0) : 0.0;
    final twoPi = math.pi * 2.0;

    for (int i = 0; i < n; i++) {
      final th = base + twoPi * (i / n);
      final dir = Vector2(math.cos(th), math.sin(th));
      hits[i] = _castOne(
        terrain: terrain,
        engineCfg: engineCfg,
        o: origin,
        d: dir,
      );
    }
    return hits;
  }

  RayHit _castOne({
    required Terrain terrain,
    required EngineConfig engineCfg,
    required Vector2 o,
    required Vector2 d,
  }) {
    double bestT = double.infinity;
    RayHitKind bestKind = RayHitKind.wall;
    Vector2 bestP = o;

    // ---- Polygon edges (outer + holes), pad-tagged ----
    for (final e in terrain.poly.edges) {
      final sol = _raySegment(o, d, e.a, e.b);
      if (sol != null && sol.$1 >= 0.0 && sol.$1 < bestT) {
        bestT = sol.$1;
        bestP = Vector2(o.x + d.x * bestT, o.y + d.y * bestT);
        bestKind = (e.kind == PolyEdgeKind.pad) ? RayHitKind.pad : RayHitKind.terrain;
      }
    }

    // ---- Arena bounds (ceiling/left/right[/floor]) ----
    final W = engineCfg.worldW.toDouble();
    final H = engineCfg.worldH.toDouble();

    void acc(double t, Vector2 p, RayHitKind k) {
      if (t < bestT) { bestT = t; bestP = p; bestKind = k; }
    }

    _hitLine(o, d, Vector2(0, 0), Vector2(W, 0), RayHitKind.wall, acc);       // ceiling
    _hitLine(o, d, Vector2(0, 0), Vector2(0, H), RayHitKind.wall, acc);       // left
    _hitLine(o, d, Vector2(W, 0), Vector2(W, H), RayHitKind.wall, acc);       // right
    if (cfg.includeFloor) {
      _hitLine(o, d, Vector2(0, H), Vector2(W, H), RayHitKind.wall, acc);     // floor
    }

    if (!bestT.isFinite) {
      // Fallback: clamp to world
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
    if (u < 0.0 || u > 1.0) return null;
    return (t, u);
  }

  void _hitLine(
      Vector2 o,
      Vector2 d,
      Vector2 a,
      Vector2 b,
      RayHitKind kind,
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

  /// === NN helpers ===

  /// Pack hits into 3 distance channels (pad/terrain/wall) in [0,1] (1=far). If
  /// [useProximity] is true, returns proximity (1=near).
  List<double> channels3({
    required List<RayHit> hits,
    required EngineConfig engineCfg,
    bool useProximity = false,
  }) {
    if (hits.isEmpty) return const <double>[];
    final n = hits.length;
    final diag = math.sqrt(engineCfg.worldW * engineCfg.worldW +
        engineCfg.worldH * engineCfg.worldH);

    final pad = List<double>.filled(n, 1.0, growable: false);
    final ter = List<double>.filled(n, 1.0, growable: false);
    final wal = List<double>.filled(n, 1.0, growable: false);

    double enc(double d) {
      final dn = (d / diag).clamp(0.0, 1.0);
      return useProximity ? (1.0 - dn) : dn;
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

  /// Compact pad summary (bearing ∈ [-1,1], 0 = forward).
  ({double minD, double bearing, double visible}) padSummary({
    required List<RayHit> hits,
    required EngineConfig engineCfg,
  }) {
    if (hits.isEmpty) return (minD: 1.0, bearing: 0.0, visible: 0.0);
    final n = hits.length;
    final diag = math.sqrt(engineCfg.worldW * engineCfg.worldW +
        engineCfg.worldH * engineCfg.worldH);

    double minD = 1.0;
    int minIdx = -1;
    for (int i = 0; i < n; i++) {
      if (hits[i].kind == RayHitKind.pad) {
        final dn = (hits[i].t / diag).clamp(0.0, 1.0);
        if (dn < minD) { minD = dn; minIdx = i; }
      }
    }

    final visible = minD < 1.0 ? 1.0 : 0.0;
    double bearing = 0.0;
    if (minIdx >= 0) {
      bearing = (minIdx / (n / 2.0)) - 1.0; // [0..n) -> [-1,1]
      if (bearing < -1) bearing += 2;
      if (bearing >  1) bearing -= 2;
    }
    return (minD: minD, bearing: bearing, visible: visible);
  }
}
