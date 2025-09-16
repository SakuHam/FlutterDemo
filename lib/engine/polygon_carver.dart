// lib/engine/polygon_carver.dart
import 'dart:math' as math;
import 'types.dart';

/// Carves a circular brush (center cx,cy; radius r) out of the terrain's
/// *outer* polygon. It DOES NOT create holes; instead, it replaces the section
/// that would pass through the circle with a clockwise arc along the circle,
/// yielding a concavity in the outer ring.
///
/// Returns a NEW Terrain (poly + ridge) and keeps pad tagging by reusing
/// padX1/padX2/padY when rebuilding the polygon.
class TerrainCarver {
  /// Main entry. You may call this multiple times to dig multiple spots.
  static Terrain carveCircle({
    required Terrain terrain,
    required double cx,
    required double cy,
    required double r,
    int arcSegments = 18, // resolution of the inserted arc
  }) {
    final outer = terrain.poly.outer;
    if (outer.length < 3) return terrain;

    final carved = _carveOuter(outer, cx, cy, r, arcSegments);

    // Recreate polygon + pad tagging; keep holes empty by design.
    final poly = PolyShape.fromRings(
      outer: carved,
      holes: const [],
      padX1: terrain.padX1,
      padX2: terrain.padX2,
      padY: terrain.padY,
    );

    // For legacy ridge drawing, we can approximate a "roof" again by
    // taking the part of outer that is not the bottom corners. If your
    // painter only uses poly, you can just return an empty ridge.
    // Here we keep it simple: reuse the whole outer (it draws fine).
    final ridge = carved;

    return Terrain(
      poly: poly,
      ridge: ridge,
      padX1: terrain.padX1,
      padX2: terrain.padX2,
      padY: terrain.padY,
    );
  }

  // ---------- Implementation ----------

  static List<Vector2> _carveOuter(
      List<Vector2> ring,
      double cx,
      double cy,
      double r,
      int arcSegments,
      ) {
    final C = Vector2(cx, cy);
    final rr = r * r;

    bool outside(Vector2 p) => _dist2(p, C) > rr + 1e-9;
    double angOf(Vector2 p) => math.atan2(p.y - cy, p.x - cx);

    // Find intersections for each edge and build the new path on the fly.
    final out = <Vector2>[];
    if (ring.isEmpty) return out;

    // Work with closed indexing
    Vector2 A = ring[0];
    bool aOut = outside(A);
    out.add(A); // start with first point (assume ring CCW)

    for (int i = 0; i < ring.length; i++) {
      final Vector2 B = ring[(i + 1) % ring.length];
      final bool bOut = outside(B);

      final xs = _segmentCircleIntersections(A, B, C, r);

      if (xs.isEmpty) {
        // No intersection: keep B only if we are outside OR (we're inside but we are skipping),
        // In difference operation, if both inside, we skip adding B (we are "cutting across").
        if (aOut && bOut) {
          out.add(B);
        } else if (aOut && !bOut) {
          // A outside, B inside, but no intersection numerically -> clamp B to circle boundary
          final pin = _projectToCircle(B, C, r);
          out.add(pin);
        } else if (!aOut && bOut) {
          // A inside, B outside, but no intersection numerically -> clamp A to circle then add B
          final pout = _projectToCircle(A, C, r);
          // replace last point with projected exit (to avoid tiny spike)
          out[out.length - 1] = pout;
          out.add(B);
        } else {
          // both inside: skip B (we are digging)
        }
      } else if (xs.length == 1) {
        // Tangent or grazing.
        final t = xs[0].$1;
        final P = _lerp(A, B, t);
        if (aOut && bOut) {
          // just add the tangent point & continue
          out.add(P);
          out.add(B);
        } else if (aOut && !bOut) {
          // entering circle at P -> add P then skip until exit on later edges
          out.add(P);
          // do not add B (inside)
        } else if (!aOut && bOut) {
          // exiting circle at P -> replace last (which is inside) with P, then add B
          out[out.length - 1] = P;
          out.add(B);
        } else {
          // both inside and tangent: ignore
        }
      } else {
        // Two intersections: entering at the first (smaller t), exit at the second (larger t)
        final tEnter = math.min(xs[0].$1, xs[1].$1);
        final tExit  = math.max(xs[0].$1, xs[1].$1);
        final Penter = _lerp(A, B, tEnter);
        final Pexit  = _lerp(A, B, tExit);

        if (aOut) {
          // coming from outside -> add entry point
          out.add(Penter);
        } else {
          // We were inside -> replace last with Penter (clean cut)
          out[out.length - 1] = Penter;
        }

        // Insert clockwise arc from exit angle to entry angle to "go around" the carved hole.
        // For an outer CCW ring, the correct difference detour is CW on the circle.
        final thEnter = angOf(Penter);
        final thExit  = angOf(Pexit);
        _appendArcCW(out, cx, cy, r, thExit, thEnter, arcSegments);

        // After the arc, continue from exit point to B (if B is outside)
        if (bOut) {
          out.add(Pexit);
          out.add(B);
        } else {
          // B is inside; keep Pexit as the last and skip B
          out.add(Pexit);
        }
      }

      A = B;
      aOut = bOut;
    }

    // Remove near-duplicates and tiny zigs
    return _dedup(out);
  }

  static double _dist2(Vector2 a, Vector2 b) {
    final dx = a.x - b.x, dy = a.y - b.y;
    return dx * dx + dy * dy;
  }

  static Vector2 _lerp(Vector2 a, Vector2 b, double t) =>
      Vector2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);

  static Vector2 _projectToCircle(Vector2 p, Vector2 c, double r) {
    final dx = p.x - c.x, dy = p.y - c.y;
    final L = math.sqrt(dx * dx + dy * dy);
    if (L < 1e-9) return Vector2(c.x + r, c.y);
    final s = r / L;
    return Vector2(c.x + dx * s, c.y + dy * s);
  }

  /// Segment-circle intersection in param t on AB (0..1). Returns 0, 1, or 2 hits.
  static List<(double, Vector2)> _segmentCircleIntersections(
      Vector2 A,
      Vector2 B,
      Vector2 C,
      double r,
      ) {
    final d = Vector2(B.x - A.x, B.y - A.y);
    final f = Vector2(A.x - C.x, A.y - C.y);

    final a = d.x * d.x + d.y * d.y;
    final b = 2.0 * (f.x * d.x + f.y * d.y);
    final c = (f.x * f.x + f.y * f.y) - r * r;

    final disc = b * b - 4 * a * c;
    if (disc < 0) return const [];
    final sqrtDisc = math.sqrt(disc);

    double t1 = (-b - sqrtDisc) / (2 * a);
    double t2 = (-b + sqrtDisc) / (2 * a);

    final out = <(double, Vector2)>[];
    if (t1 >= 0.0 && t1 <= 1.0) out.add((t1, _lerp(A, B, t1)));
    if (t2 >= 0.0 && t2 <= 1.0 && (t2 - t1).abs() > 1e-9) {
      out.add((t2, _lerp(A, B, t2)));
    }
    // sort by t
    out.sort((a1, a2) => a1.$1.compareTo(a2.$1));
    return out;
  }

  /// Append clockwise arc points from angle thStart to thEnd (CW).
  static void _appendArcCW(
      List<Vector2> out,
      double cx,
      double cy,
      double r,
      double thStart,
      double thEnd,
      int segs,
      ) {
    // Normalize angles to [-pi, pi)
    double norm(double a) {
      while (a <= -math.pi) a += 2 * math.pi;
      while (a > math.pi) a -= 2 * math.pi;
      return a;
    }

    double s = norm(thStart), e = norm(thEnd);
    // We want CW: decreasing angle. If e > s, go the "other way" by subtracting 2π.
    if (e > s) e -= 2 * math.pi;

    final steps = math.max(2, segs);
    for (int i = 1; i <= steps; i++) {
      final t = i / steps;
      final th = s + (e - s) * t; // moves downward in angle (CW)
      out.add(Vector2(cx + r * math.cos(th), cy + r * math.sin(th)));
    }
  }

  static List<Vector2> _dedup(List<Vector2> pts) {
    if (pts.isEmpty) return pts;
    const eps = 1e-6;
    final out = <Vector2>[pts.first];
    for (int i = 1; i < pts.length; i++) {
      final a = out.last, b = pts[i];
      if ((a.x - b.x).abs() > eps || (a.y - b.y).abs() > eps) out.add(b);
    }
    // optional: also drop tiny spikes (collinear triplets)
    final cleaned = <Vector2>[];
    for (int i = 0; i < out.length; i++) {
      final p0 = out[(i - 1 + out.length) % out.length];
      final p1 = out[i];
      final p2 = out[(i + 1) % out.length];
      final cross = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
      if (cross.abs() < 1e-6 && (p1.x - p0.x).abs() < 1e-5 && (p1.y - p0.y).abs() < 1e-5) {
        // super tiny segment; skip p1
      } else {
        cleaned.add(p1);
      }
    }
    return cleaned;
  }
}
