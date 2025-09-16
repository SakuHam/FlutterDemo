// lib/engine/polygon_carver.dart
import 'dart:math' as math;
import 'types.dart';

/// Index-preserving circle carver that *removes material* from the single OUTER ring.
/// No holes are created; if the brush circle has no intersections with the outline,
/// the operation is a no-op by design (since holes are disallowed).
class TerrainCarver {
  static Terrain carveCircle({
    required Terrain terrain,
    required double cx,
    required double cy,
    required double r,
    double maxArcErrorPx = 1.0,   // arc tessellation error (smaller = more points)
    double projectEpsPx = 0.75,   // snap near-boundary points to circle
    double simplifyEpsPx = 0.6,   // dedup & tiny-collinear epsilon
    double rdpTolerancePx = 0.9,  // Douglas–Peucker tolerance
  }) {
    final outer = terrain.poly.outer;
    if (outer.length < 3) return terrain;

    final carved = _carveIndexed(
      outer,
      Vector2(cx, cy),
      r,
      maxArcErrorPx,
      projectEpsPx,
    );

    if (carved.length < 3) {
      // Fully eaten or numerically unstable -> keep original to remain healthy
      return terrain;
    }

    // Light clean-up and enforce CCW
    final simp1 = _dedup(carved, simplifyEpsPx);
    final simp2 = _removeTinyCollinears(simp1, simplifyEpsPx);
    final simp3 = _rdp(simp2, rdpTolerancePx);
    final healthy = _ensureCCW(simp3);

    final poly = PolyShape.fromRings(
      outer: healthy,
      holes: const [],
      padX1: terrain.padX1,
      padX2: terrain.padX2,
      padY: terrain.padY,
    );

    return Terrain(
      poly: poly,
      ridge: healthy,
      padX1: terrain.padX1,
      padX2: terrain.padX2,
      padY: terrain.padY,
    );
  }

  // ---------- Core: index-preserving splicer ----------
  static List<Vector2> _carveIndexed(
      List<Vector2> ringIn,
      Vector2 C,
      double r,
      double maxArcErr,
      double projectEps,
      ) {
    const eps = 1e-9;

    // Helpers
    double dist(Vector2 p) {
      final dx = p.x - C.x, dy = p.y - C.y;
      return math.sqrt(dx * dx + dy * dy);
    }
    bool outside(Vector2 p) => dist(p) > r + 1e-9;
    Vector2 snapToCircle(Vector2 p) {
      final dx = p.x - C.x, dy = p.y - C.y;
      final L = math.sqrt(dx * dx + dy * dy);
      if (L < 1e-12) return Vector2(C.x + r, C.y);
      final s = r / L;
      return Vector2(C.x + dx * s, C.y + dy * s);
    }
    double ang(Vector2 p) => math.atan2(p.y - C.y, p.x - C.x);
    bool samePt(Vector2 a, Vector2 b) =>
        (a.x - b.x).abs() <= eps && (a.y - b.y).abs() <= eps;

    // ---- Rotate ring so we start at the first OUTSIDE vertex (stable state) ----
    final ring = List<Vector2>.from(ringIn);
    int start = -1;
    for (int i = 0; i < ring.length; i++) {
      if (outside(ring[i])) { start = i; break; }
    }
    if (start == -1) {
      // Entire ring is inside the brush (would create a hole). Do nothing.
      return List<Vector2>.from(ringIn);
    }
    if (start != 0) {
      final rotated = <Vector2>[];
      for (int i = 0; i < ring.length; i++) {
        rotated.add(ring[(start + i) % ring.length]);
      }
      ring
        ..clear()
        ..addAll(rotated);
    }

    // Snap near-boundary endpoints
    for (int i = 0; i < ring.length; i++) {
      final p = ring[i];
      if (!outside(p) && (r - dist(p)).abs() <= projectEps) {
        ring[i] = snapToCircle(p);
      }
    }

    final out = <Vector2>[];

    // Seed with first vertex (guaranteed outside after rotation)
    Vector2 A = ring[0];
    bool aOut = outside(A);
    out.add(A);

    // --- Walk edges (A->B) in order, including wrap at the end ---
    for (int i = 0; i < ring.length; i++) {
      final Vector2 B = ring[(i + 1) % ring.length];
      final bool bOut = outside(B);

      // Find intersections, sort by t
      final hits = List<(double, Vector2)>.from(
        _segmentCircleIntersections(A, B, C, r),
      );
      if (hits.length > 1) {
        hits.sort((l, r2) => l.$1.compareTo(r2.$1));
      }

      // Tangent case: one hit and both endpoints outside -> don't flip state
      if (hits.length == 1 && aOut && bOut) {
        if (out.isEmpty || !samePt(out.last, B)) out.add(B);
        A = B; aOut = bOut;
        continue;
      }

      // Emit piecewise, flipping state on proper hits
      double tPrev = 0.0;
      bool stateOut = aOut;
      Vector2? enterP;

      void addPoint(Vector2 p) {
        if (out.isEmpty || !samePt(out.last, p)) out.add(p);
      }

      void emitUntil(double t1) {
        // endpoint point
        final Vector2 p1 = (t1 >= 1.0 - eps) ? B : _lerp(A, B, t1);
        if (stateOut) addPoint(p1);
      }

      for (int h = 0; h <= hits.length; h++) {
        final double t1 = (h < hits.length) ? hits[h].$1 : 1.0;
        final Vector2 pHit = (h < hits.length) ? hits[h].$2 : B;

        emitUntil(t1);

        if (h < hits.length) {
          if (stateOut) {
            // outside -> inside: remember enter point
            enterP = pHit;
          } else {
            // inside -> outside: add CW arc (exit -> enter)
            final exitP = pHit;
            if (enterP != null) {
              _appendArcCWAdaptive(out, C.x, C.y, r, ang(exitP), ang(enterP!), maxArcErr);
              addPoint(exitP);
              enterP = null;
            }
          }
          stateOut = !stateOut;
          // replace last with exact hit (avoid spike)
          if (out.isNotEmpty) out[out.length - 1] = pHit;
        }
        tPrev = t1;
      }

      A = B; aOut = bOut;
    }

    // Close ring cleanly (avoid duplicate last==first)
    if (out.length >= 2 && samePt(out.first, out.last)) {
      out.removeLast();
    }
    return out;
  }

  // ---------- Geometry bits ----------
  static Vector2 _lerp(Vector2 a, Vector2 b, double t) =>
      Vector2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);

  /// Edge-circle intersections on AB: returns (t, point) with 0<t<1.
  static List<(double, Vector2)> _segmentCircleIntersections(
      Vector2 A, Vector2 B, Vector2 C, double r,
      ) {
    final d = Vector2(B.x - A.x, B.y - A.y);
    final f = Vector2(A.x - C.x, A.y - C.y);

    final a = d.x * d.x + d.y * d.y;
    final b = 2.0 * (f.x * d.x + f.y * d.y);
    final c = (f.x * f.x + f.y * f.y) - r * r;

    final disc = b * b - 4 * a * c;
    if (disc <= 0) return const []; // <= 0: no real, or tangent w/ t outside (we ignore)

    final sDisc = math.sqrt(disc);
    final t1 = (-b - sDisc) / (2 * a);
    final t2 = (-b + sDisc) / (2 * a);

    final out = <(double, Vector2)>[];
    if (t1 > 0.0 && t1 < 1.0) out.add((t1, _lerp(A, B, t1)));
    if (t2 > 0.0 && t2 < 1.0 && (t2 - t1).abs() > 1e-9) out.add((t2, _lerp(A, B, t2)));
    return out;
  }

  /// Append a CW arc from thStart -> thEnd approximated by minimal points
  /// s.t. chord error <= maxErr.
  static void _appendArcCWAdaptive(
      List<Vector2> out, double cx, double cy, double r,
      double thStart, double thEnd, double maxErr,
      ) {
    double norm(double a) {
      while (a <= -math.pi) a += 2 * math.pi;
      while (a >  math.pi) a -= 2 * math.pi;
      return a;
    }
    double s = norm(thStart), e = norm(thEnd);
    if (e > s) e -= 2 * math.pi; // enforce CW (decreasing angle)

    final sweep = (s - e).abs();
    final maxDelta = (maxErr <= 0 || maxErr >= r)
        ? (math.pi / 4)
        : 2 * math.acos(math.max(-1.0, math.min(1.0, 1 - maxErr / r)));
    final segs = math.max(2, (sweep / maxDelta).ceil());

    for (int i = 1; i <= segs; i++) {
      final t = i / segs;
      final th = s + (e - s) * t;
      final p = Vector2(cx + r * math.cos(th), cy + r * math.sin(th));
      if (out.isEmpty ||
          (out.last.x - p.x).abs() > 1e-9 ||
          (out.last.y - p.y).abs() > 1e-9) {
        out.add(p);
      }
    }
  }

  // ---------- Simplifiers & orientation ----------
  static List<Vector2> _dedup(List<Vector2> pts, double eps) {
    if (pts.isEmpty) return pts;
    final out = <Vector2>[pts.first];
    for (int i = 1; i < pts.length; i++) {
      final a = out.last, b = pts[i];
      if ((a.x - b.x).abs() > eps || (a.y - b.y).abs() > eps) out.add(b);
    }
    return out;
  }

  static List<Vector2> _removeTinyCollinears(List<Vector2> pts, double eps) {
    if (pts.length < 3) return pts;
    final out = <Vector2>[];
    for (int i = 0; i < pts.length; i++) {
      final p0 = pts[(i - 1 + pts.length) % pts.length];
      final p1 = pts[i];
      final p2 = pts[(i + 1) % pts.length];
      final cross = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
      // drop p1 if nearly collinear and very close to either neighbor
      if (cross.abs() < eps) {
        final d10 = (p1.x - p0.x).abs() + (p1.y - p0.y).abs();
        final d12 = (p1.x - p2.x).abs() + (p1.y - p2.y).abs();
        if (d10 < eps || d12 < eps) continue;
      }
      out.add(p1);
    }
    return out;
  }

  static List<Vector2> _rdp(List<Vector2> pts, double tol) {
    if (pts.length < 4) return pts;
    final open = List<Vector2>.from(pts)..add(pts.first);
    final keep = List<bool>.filled(open.length, false);
    _rdpRecurse(open, 0, open.length - 1, tol, keep);
    final out = <Vector2>[];
    for (int i = 0; i < open.length - 1; i++) {
      if (keep[i] || i == 0) out.add(open[i]);
    }
    return out;
  }

  static void _rdpRecurse(List<Vector2> pts, int i0, int i1, double tol, List<bool> keep) {
    if (i1 <= i0 + 1) { keep[i0] = true; keep[i1] = true; return; }
    final a = pts[i0], b = pts[i1];
    int idx = -1; double maxD = -1;
    for (int i = i0 + 1; i < i1; i++) {
      final d = _pointSegmentDistance(pts[i], a, b);
      if (d > maxD) { maxD = d; idx = i; }
    }
    if (maxD > tol) {
      _rdpRecurse(pts, i0, idx, tol, keep);
      _rdpRecurse(pts, idx, i1, tol, keep);
    } else {
      keep[i0] = true; keep[i1] = true;
    }
  }

  static double _pointSegmentDistance(Vector2 p, Vector2 a, Vector2 b) {
    final vx = b.x - a.x, vy = b.y - a.y;
    final wx = p.x - a.x, wy = p.y - a.y;
    final c1 = vx * wx + vy * wy;
    if (c1 <= 0) return math.sqrt(wx * wx + wy * wy);
    final c2 = vx * vx + vy * vy;
    if (c2 <= c1) {
      final dx = p.x - b.x, dy = p.y - b.y;
      return math.sqrt(dx * dx + dy * dy);
    }
    final t = c1 / c2;
    final px = a.x + t * vx, py = a.y + t * vy;
    final dx = p.x - px, dy = p.y - py;
    return math.sqrt(dx * dx + dy * dy);
  }

  static List<Vector2> _ensureCCW(List<Vector2> ring) {
    if (ring.length < 3) return ring;
    if (_signedArea(ring) < 0) {
      return List<Vector2>.from(ring.reversed);
    }
    return ring;
  }

  static double _signedArea(List<Vector2> ring) {
    double a = 0;
    for (int i = 0; i < ring.length; i++) {
      final p = ring[i];
      final q = ring[(i + 1) % ring.length];
      a += p.x * q.y - p.y * q.x;
    }
    return 0.5 * a;
  }
}
