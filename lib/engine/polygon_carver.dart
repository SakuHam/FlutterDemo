import 'dart:math' as math;
import 'types.dart';

/// Bomb-proof circle carver for a single OUTER ring (no holes).
/// Algorithm:
/// 1) Build an indexed vertex list [(idx, p, isHit?, inside?)] from the ring.
/// 2) For every original edge, compute circle intersections and INSERT them
///    after the edge start, keeping indices consistent.
/// 3) Rotate so we start from an OUTSIDE vertex/hit.
/// 4) Walk once:
///    - On outside→inside transition: mark ENTER hit.
///    - Skip all points strictly inside the circle until the next HIT (EXIT).
///    - Splice a CW arc along the circle from EXIT -> ENTER.
/// 5) Ensure CCW orientation and return a clean ring.
///
/// No RDP or aggressive simplifiers are needed; we only dedup tiny neighbors.
/// Arc tessellation uses a chord-error bound to avoid zig-zags (“residue”).
class TerrainCarver {
  static Terrain carveCircle({
    required Terrain terrain,
    required double cx,
    required double cy,
    required double r,
    double arcMaxErrPx = 1.0, // arc tessellation chord error
    double nearCircleSnapPx = 0.75, // snap near-boundary ring verts to the circle
    double dedupEps = 1e-6,   // neighbor dedup epsilon
  }) {
    final ring0 = terrain.poly.outer;
    if (ring0.length < 3) return terrain;

// Detect orientation of the input outer ring
    final bool outerIsCCW = _signedArea(ring0) >= 0;

    final C = Vector2(cx, cy);
    final ring = List<Vector2>.from(ring0);
    final n0 = ring.length;

    // ==== 1) Build indexed vertex list
    final verts = <_Vtx>[];
    for (int i = 0; i < n0; i++) {
      final p = ring[i];
      final d = _dist(p, C);
      bool inside = d < r - 1e-9;
      // Snap near-boundary original vertices to circle
      if (!inside && (d - r).abs() <= nearCircleSnapPx) {
        final s = r / (d <= 1e-12 ? r : d);
        final sp = Vector2(C.x + (p.x - C.x) * s, C.y + (p.y - C.y) * s);
        verts.add(_Vtx(i, sp, isHit: true, inside: false));
      } else {
        verts.add(_Vtx(i, p, isHit: false, inside: inside));
      }
    }

    // Helper to locate current position of original vertex index in verts
    int _findCurrentPos(int originalIdx) {
      for (int k = 0; k < verts.length; k++) {
        if (verts[k].origIdx == originalIdx && !verts[k].insertedAfterEdge)
          return k;
      }
      // For indices of intersection insertions we set insertedAfterEdge=true,
      // so originalIdx still identifies only the true pre-existing vertex.
      // If not found (shouldn’t happen), fallback to 0.
      return 0;
    }

    // ==== 2) For each edge, INSERT intersections in order of t
    for (int i = 0; i < n0; i++) {
      final aIdx = i;
      final bIdx = (i + 1) % n0;
      final A = ring[aIdx];
      final B = ring[bIdx];

      final hits = _segmentCircleIntersections(A, B, C, r);
      if (hits.isEmpty) continue;

      hits.sort((l, r2) => l.$1.compareTo(r2.$1));

      // Where to insert? Immediately AFTER the current A in the *current* verts list.
      int posA = _findCurrentPos(aIdx);
      int insertPos = posA + 1;

      for (final (t, phit) in hits) {
        final v = _Vtx(
          aIdx, // associate with edge start for ordering
          phit,
          isHit: true,
          inside: false,
          insertedAfterEdge: true,
          tOnEdge: t,
        );
        verts.insert(insertPos, v);
        insertPos++; // subsequent hits go after the previous insertion
      }
    }

    if (verts.where((v) => !v.inside).isEmpty) {
      // Whole ring is (numerically) inside the circle — we are not allowed to create holes
      return terrain;
    }

    // ==== 3) Rotate to start at an OUTSIDE vertex/hit
    int start = 0;
    for (int i = 0; i < verts.length; i++) {
      if (!verts[i].inside) { start = i; break; }
    }
    if (start != 0) {
      final ro = <_Vtx>[];
      for (int k = 0; k < verts.length; k++) {
        ro.add(verts[(start + k) % verts.length]);
      }
      verts
        ..clear()
        ..addAll(ro);
    }

    // ==== 4) Single pass: build the new ring
    final out = <Vector2>[];
    int i = 0;
    while (i < verts.length) {
      final cur = verts[i];
      final nxt = verts[(i + 1) % verts.length];

      // Always keep current if it's outside or a boundary hit
      if (!cur.inside || cur.isHit) {
        _pushDedup(out, cur.p, eps: dedupEps);
      }

      // Detect outside -> inside transition right after a HIT (ENTER).
      final entering = cur.isHit && nxt.inside;

      if (entering) {
        final enterHit = cur;

        // Advance until EXIT hit
        int j = (i + 1) % verts.length;
        _Vtx? exitHit;
        while (true) {
          final v = verts[j];
          if (v.isHit && !v.inside) {
            exitHit = v;
            break;
          }
          j = (j + 1) % verts.length;
          if (j == i) {
            // Safety: no exit found (shouldn’t happen with a simple circle)
            break;
          }
        }

        if (exitHit != null) {
          if (outerIsCCW) {
            _appendArcCCW(out, C, r, exitHit.p, enterHit.p, arcMaxErrPx);
          } else {
            _appendArcCW(out, C, r, exitHit.p, enterHit.p, arcMaxErrPx);
          }
          // Also place the EXIT point to reconnect with the outer contour
          _pushDedup(out, exitHit.p, eps: dedupEps);

          // Jump i to the exit hit, continue from there
          i = verts.indexOf(exitHit);
        } else {
          // Pathological: give up this run safely (keep current)
        }
      }

      i++;
    }

    // Clean close (avoid duplicate first==last)
    if (out.length >= 2 &&
        (out.first.x - out.last.x).abs() <= dedupEps &&
        (out.first.y - out.last.y).abs() <= dedupEps) {
      out.removeLast();
    }

    // Ensure CCW orientation
    final ccw = _signedArea(out) >= 0 ? out : List<Vector2>.from(out.reversed);

    // Rebuild terrain (pad tagging recalculated in PolyShape)
    final poly = PolyShape.fromRings(
      outer: ccw,
      holes: const [],
      padX1: terrain.padX1,
      padX2: terrain.padX2,
      padY: terrain.padY,
    );

    return Terrain(
      poly: poly,
      ridge: ccw,
      padX1: terrain.padX1,
      padX2: terrain.padX2,
      padY: terrain.padY,
    );
  }

  static double _twoPi = math.pi * 2.0;

  /// normalize to [0, 2π)
  static double _norm2pi(double a) {
    a = a % _twoPi;
    if (a < 0) a += _twoPi;
    return a;
  }

  /// angle of point relative to center
  static double _ang(Vector2 p, Vector2 C) => math.atan2(p.y - C.y, p.x - C.x);

  /// Append CW arc from exitP -> enterP with max chord error (shortest CW sweep).
  static void _appendArcCW(
      List<Vector2> out,
      Vector2 C,
      double r,
      Vector2 exitP,
      Vector2 enterP,
      double maxErr,
      ) {
    final s = _norm2pi(_ang(exitP, C));
    final e = _norm2pi(_ang(enterP, C));

    // CW sweep size in [0, 2π)
    final delta = _norm2pi(s - e);
    if (delta <= 1e-12) return; // no arc

    // chord-error -> angular step
    final maxDelta = (maxErr <= 0 || maxErr >= r)
        ? (math.pi / 4)
        : 2 * math.acos(math.max(-1.0, math.min(1.0, 1 - maxErr / r)));
    final segs = math.max(2, (delta / maxDelta).ceil());

    for (int i = 1; i <= segs; i++) {
      final t = i / segs;
      final th = s - t * delta;                  // CW = decreasing angle
      final p = Vector2(C.x + r * math.cos(th),  // sample on circle
          C.y + r * math.sin(th));
      _pushDedup(out, p, eps: 1e-9);
    }
  }

  /// Append CCW arc from exitP -> enterP with max chord error (shortest CCW sweep).
  static void _appendArcCCW(
      List<Vector2> out,
      Vector2 C,
      double r,
      Vector2 exitP,
      Vector2 enterP,
      double maxErr,
      ) {
    final s = _norm2pi(_ang(exitP, C));
    final e = _norm2pi(_ang(enterP, C));

    // CCW sweep size in [0, 2π)
    final delta = _norm2pi(e - s);
    if (delta <= 1e-12) return; // no arc

    final maxDelta = (maxErr <= 0 || maxErr >= r)
        ? (math.pi / 4)
        : 2 * math.acos(math.max(-1.0, math.min(1.0, 1 - maxErr / r)));
    final segs = math.max(2, (delta / maxDelta).ceil());

    for (int i = 1; i <= segs; i++) {
      final t = i / segs;
      final th = s + t * delta;                  // CCW = increasing angle
      final p = Vector2(C.x + r * math.cos(th),
          C.y + r * math.sin(th));
      _pushDedup(out, p, eps: 1e-9);
    }
  }

  // ----------- helpers -----------

  static double _dist(Vector2 a, Vector2 b) {
    final dx = a.x - b.x, dy = a.y - b.y;
    return math.sqrt(dx*dx + dy*dy);
  }

  static void _pushDedup(List<Vector2> out, Vector2 p, {double eps = 1e-9}) {
    if (out.isEmpty) { out.add(p); return; }
    final q = out.last;
    if ((p.x - q.x).abs() > eps || (p.y - q.y).abs() > eps) out.add(p);
  }

  static List<(double, Vector2)> _segmentCircleIntersections(
      Vector2 A, Vector2 B, Vector2 C, double r,
      ) {
    final d = Vector2(B.x - A.x, B.y - A.y);
    final f = Vector2(A.x - C.x, A.y - C.y);

    final a = d.x * d.x + d.y * d.y;
    final b = 2.0 * (f.x * d.x + f.y * d.y);
    final c = (f.x * f.x + f.y * f.y) - r * r;

    final disc = b * b - 4 * a * c;
    if (disc < 0) return const [];

    final sDisc = math.sqrt(disc);
    final t1 = (-b - sDisc) / (2 * a);
    final t2 = (-b + sDisc) / (2 * a);

    final out = <(double, Vector2)>[];
    if (t1 > 0.0 && t1 < 1.0) out.add((t1, _lerp(A, B, t1)));
    if (t2 > 0.0 && t2 < 1.0 && (t2 - t1).abs() > 1e-12) out.add((t2, _lerp(A, B, t2)));
    return out;
  }

  static Vector2 _lerp(Vector2 a, Vector2 b, double t) =>
      Vector2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);

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

/// Indexed vertex record
class _Vtx {
  final int origIdx;           // original vertex index (for ordering/association)
  final Vector2 p;
  final bool isHit;            // true if exactly on the circle (intersection or snapped)
  final bool inside;           // strictly inside the circle
  final bool insertedAfterEdge;// true for intersection insertions
  final double? tOnEdge;       // param along the source edge for sorting (0..1)

  _Vtx(
      this.origIdx,
      this.p, {
        required this.isHit,
        required this.inside,
        this.insertedAfterEdge = false,
        this.tOnEdge,
      });
}
