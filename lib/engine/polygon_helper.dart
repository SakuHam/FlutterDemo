// lib/engine/polygon_helper.dart
import 'dart:math' as math;
import 'types.dart';

/// Tag edges so sensors and contact can differentiate pad/terrain.
enum PolyEdgeKind { terrain, pad }

class PolyEdge {
  final Vector2 a;
  final Vector2 b;
  final PolyEdgeKind kind;
  const PolyEdge(this.a, this.b, this.kind);
}

/// A polygon with an outer ring and optional hole rings (for caverns).
/// Rings must be closed (first != last; we close internally).
class PolyShape {
  final List<Vector2> outer;         // recommended CCW
  final List<List<Vector2>> holes;   // recommended CW
  final List<PolyEdge> edges;        // derived (outer + holes)

  PolyShape._(this.outer, this.holes, this.edges);

  /// Build from rings. Will normalize winding (outer CCW, holes CW) and
  /// produce an edge list. You can pass [padX1,padX2,padY] to mark inline
  /// pad edges; leave null if you’ll tag pads yourself later.
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
    void addRing(List<Vector2> r, PolyEdgeKind defaultKind) {
      final n = r.length;
      for (int i = 0; i < n; i++) {
        final a = r[i];
        final b = r[(i + 1) % n];
        final kind =
        (padX1 != null && padX2 != null && padY != null && _segmentIsPad(a, b, padX1, padX2, padY))
            ? PolyEdgeKind.pad
            : defaultKind;
        edges.add(PolyEdge(a, b, kind));
      }
    }

    addRing(out, PolyEdgeKind.terrain);
    for (final h in hol) {
      addRing(h, PolyEdgeKind.terrain); // holes can hold pads too, will be tagged if flat & in range
    }

    return PolyShape._(out, hol, edges);
  }

  /// Vertical query: smallest y >= 0 where x hits the polygon boundary.
  /// This respects overhangs and caverns. Returns +∞ if no hit in [0, H].
  double verticalHitY({
    required double x,
    required double worldH,
  }) {
    double bestY = double.infinity;
    for (final e in edges) {
      final y = _xRayHitY(e.a, e.b, x);
      if (y != null && y >= 0.0 && y < bestY) {
        bestY = y;
      }
    }
    if (!bestY.isFinite) return double.infinity;
    return math.min(bestY, worldH);
  }

  /// Raycast all polygon edges; returns nearest hit if any.
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

  // ---------- helpers ----------

  static List<Vector2> _ensureWinding(List<Vector2> ring, {required bool ccw}) {
    // Compute signed area (positive for CCW in screen coords if y down?).
    // Here we use standard math (y up). Our screen y is down, but for edge
    // building only consistency matters.
    double area = 0.0;
    for (int i = 0; i < ring.length; i++) {
      final a = ring[i];
      final b = ring[(i + 1) % ring.length];
      area += (a.x * b.y - b.x * a.y);
    }
    final isCCW = area > 0;
    if (isCCW == ccw) return ring;
    final rev = List<Vector2>.from(ring.reversed);
    return rev;
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
    // Reject if entirely to left/right.
    if ((X < math.min(a.x, b.x)) || (X > math.max(a.x, b.x))) return null;

    final dx = (b.x - a.x);
    if (dx.abs() < 1e-9) {
      // Nearly vertical segment; take the higher (smaller y) endpoint as boundary.
      return math.min(a.y, b.y);
    }
    final t = (X - a.x) / dx; // 0..1 along AB in x
    if (t < 0.0 || t > 1.0) return null;
    final y = a.y + (b.y - a.y) * t;
    return y;
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
