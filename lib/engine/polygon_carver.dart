// lib/engine/polygon_carver.dart
import 'dart:math' as math;
import 'types.dart';

/// Bomb-proof polygon carvers for a single OUTER ring (no holes).
/// - Circle: splices circle arcs between ENTER/EXIT hits.
/// - Rectangle (axis-aligned): splices the *actual rectangle boundary* (corners included).
/// Both refuse to create holes (if shape would engulf the whole polygon, no-op).
class TerrainCarver {
  // ==========================================================================
  // CIRCLE CARVER (your original algorithm; kept intact)
  // ==========================================================================
  static Terrain carveCircle({
    required Terrain terrain,
    required double cx,
    required double cy,
    required double r,
    double arcMaxErrPx = 1.0,      // arc tessellation chord error
    double nearCircleSnapPx = 0.75,// snap near-boundary verts to circle
    double dedupEps = 1e-6,        // neighbor dedup epsilon
  }) {
    final ring0 = terrain.poly.outer;
    if (ring0.length < 3) return terrain;

    final bool outerIsCCW = _signedArea(ring0) >= 0;
    final C = Vector2(cx, cy);

    // 1) Build indexed vertex list
    final verts = <_Vtx>[];
    for (int i = 0; i < ring0.length; i++) {
      final p = ring0[i];
      final d = _dist(p, C);
      bool inside = d < r - 1e-9;
      if (!inside && (d - r).abs() <= nearCircleSnapPx) {
        final s = r / (d <= 1e-12 ? r : d);
        final sp = Vector2(C.x + (p.x - C.x) * s, C.y + (p.y - C.y) * s);
        verts.add(_Vtx(i, sp, isHit: true, inside: false));
      } else {
        verts.add(_Vtx(i, p, isHit: false, inside: inside));
      }
    }

    int _findCurrentPos(int origIdx) {
      for (int k = 0; k < verts.length; k++) {
        if (verts[k].origIdx == origIdx && !verts[k].insertedAfterEdge) return k;
      }
      return 0;
    }

    // 2) Insert segment ⟂ circle intersections
    for (int i = 0; i < ring0.length; i++) {
      final aIdx = i;
      final bIdx = (i + 1) % ring0.length;
      final A = ring0[aIdx];
      final B = ring0[bIdx];

      final hits = _segmentCircleIntersections(A, B, C, r);
      if (hits.isEmpty) continue;

      hits.sort((l, r2) => l.$1.compareTo(r2.$1));
      int posA = _findCurrentPos(aIdx);
      int insertPos = posA + 1;

      for (final (t, phit) in hits) {
        verts.insert(insertPos, _Vtx(
          aIdx, phit,
          isHit: true,
          inside: false,
          insertedAfterEdge: true,
          tOnEdge: t,
        ));
        insertPos++;
      }
    }

    if (verts.where((v) => !v.inside).isEmpty) return terrain; // refuse to create holes

    // 3) Rotate to first outside point
    int start = 0;
    for (int i = 0; i < verts.length; i++) { if (!verts[i].inside) { start = i; break; } }
    if (start != 0) {
      final ro = <_Vtx>[];
      for (int k = 0; k < verts.length; k++) ro.add(verts[(start + k) % verts.length]);
      verts..clear()..addAll(ro);
    }

    // 4) Single pass: copy outside, splice arc on ENTER..EXIT
    final out = <Vector2>[];
    int i = 0;
    while (i < verts.length) {
      final cur = verts[i];
      final nxt = verts[(i + 1) % verts.length];

      if (!cur.inside || cur.isHit) _pushDedup(out, cur.p, eps: dedupEps);

      final entering = cur.isHit && nxt.inside;
      if (entering) {
        int j = (i + 1) % verts.length;
        _Vtx? exitHit;
        while (true) {
          final v = verts[j];
          if (v.isHit && !v.inside) { exitHit = v; break; }
          j = (j + 1) % verts.length;
          if (j == i) break;
        }

        if (exitHit != null) {
          if (outerIsCCW) {
            _appendArcCW(out, C, r, cur.p, exitHit.p, arcMaxErrPx);
          } else {
            _appendArcCCW(out, C, r, cur.p, exitHit.p, arcMaxErrPx);
          }
          i = verts.indexOf(exitHit);
        }
      }

      i++;
    }

    // Close & orient CCW
    if (out.length >= 2 &&
        (out.first.x - out.last.x).abs() <= dedupEps &&
        (out.first.y - out.last.y).abs() <= dedupEps) out.removeLast();
    final ccw = _signedArea(out) >= 0 ? out : List<Vector2>.from(out.reversed);

    final poly = PolyShape.fromRings(
      outer: ccw, holes: const [],
      padX1: terrain.padX1, padX2: terrain.padX2, padY: terrain.padY,
    );
    return Terrain(
      poly: poly,
      ridge: ccw,
      padX1: terrain.padX1, padX2: terrain.padX2, padY: terrain.padY,
    );
  }

  // ==========================================================================
  // RECTANGLE CARVER (axis-aligned, true rectangle – no circle approximation)
  // Boundary is treated as OUTSIDE; only strict interior is “inside”.
  // ==========================================================================
  static Terrain carveRectangle({
    required Terrain terrain,
    required double x0,
    required double y0,
    required double x1,
    required double y1,
    double nearEdgeSnapPx = 0.9, // slightly larger helps grazing hits
    double dedupEps = 1e-6,
  }) {
    final ring0 = terrain.poly.outer;
    if (ring0.length < 3) return terrain;

    final left   = math.min(x0, x1);
    final right  = math.max(x0, x1);
    final top    = math.min(y0, y1);
    final bottom = math.max(y0, y1);
    final rect = _Rect(left, top, right, bottom);
    if (rect.width < 1e-6 || rect.height < 1e-6) return terrain; // degenerate

    final bool outerIsCCW = _signedArea(ring0) >= 0;

    // 1) Build vertex list with strict-inside (boundary=outside) + snap near boundary as hits
    final verts = <_VtxRect>[];
    for (int i = 0; i < ring0.length; i++) {
      final p0 = ring0[i];
      var p = p0;
      bool isHit = false;

      if (rect.minEdgeDistance(p) <= nearEdgeSnapPx) {
        // Snap to closest side
        final dL = (p.x - rect.left).abs();
        final dR = (p.x - rect.right).abs();
        final dT = (p.y - rect.top).abs();
        final dB = (p.y - rect.bottom).abs();
        final m = math.min(math.min(dL, dR), math.min(dT, dB));
        if (m == dL) p = Vector2(rect.left,  p.y);
        else if (m == dR) p = Vector2(rect.right, p.y);
        else if (m == dT) p = Vector2(p.x, rect.top);
        else              p = Vector2(p.x, rect.bottom);
        isHit = true; // on boundary
      }

      final inside = rect.containsStrict(p);
      verts.add(_VtxRect(
        i, p,
        isHit: isHit,
        inside: inside,
      ));
    }

    int _findCurrentPos(int origIdx) {
      for (int k = 0; k < verts.length; k++) {
        if (verts[k].origIdx == origIdx && !verts[k].insertedAfterEdge) return k;
      }
      return 0;
    }

    // 2) Insert segment ⟂ rectangle intersections
    for (int i = 0; i < ring0.length; i++) {
      final aIdx = i;
      final bIdx = (i + 1) % ring0.length;
      final A = ring0[aIdx];
      final B = ring0[bIdx];

      final hits = _segmentRectIntersections(A, B, rect);
      if (hits.isEmpty) continue;

      hits.sort((a, b) => a.t.compareTo(b.t));
      int posA = _findCurrentPos(aIdx);
      int insertPos = posA + 1;

      for (final h in hits) {
        verts.insert(insertPos, _VtxRect(
          aIdx, h.p,
          isHit: true,
          inside: false, // boundary is outside
          insertedAfterEdge: true,
          tOnEdge: h.t,
          rectSide: h.side,
          rectU: h.u,
        ));
        insertPos++;
      }
    }

    if (verts.where((v) => !v.inside).isEmpty) return terrain; // refuse to create holes

    // 3) Rotate to first OUTSIDE vertex
    int start = 0;
    for (int i = 0; i < verts.length; i++) { if (!verts[i].inside) { start = i; break; } }
    if (start != 0) {
      final ro = <_VtxRect>[];
      for (int k = 0; k < verts.length; k++) ro.add(verts[(start + k) % verts.length]);
      verts..clear()..addAll(ro);
    }

    // 4) Single pass with robust ENTER (hit→inside) / EXIT (hit→outside)
    final out = <Vector2>[];
    int i = 0;
    int guard = 0;
    while (i < verts.length) {
      if (++guard > verts.length * 3) break; // hard guard

      final cur = verts[i];
      final nxt = verts[(i + 1) % verts.length];

      if (!cur.inside || cur.isHit) _pushDedup(out, cur.p, eps: dedupEps);

      final entering = cur.isHit && nxt.inside;
      if (entering) {
        int j = (i + 1) % verts.length;
        _VtxRect? exitHit;
        int seekGuard = 0;
        while (seekGuard++ < verts.length + 4) {
          final v = verts[j];
          if (v.isHit && !v.inside) { exitHit = v; break; } // boundary hit that is outside
          j = (j + 1) % verts.length;
        }

        if (exitHit != null && cur.rectSide != null && exitHit.rectSide != null) {
          final sA = cur.rectSide!, uA = cur.rectU ?? 0.0;
          final sB = exitHit.rectSide!, uB = exitHit.rectU ?? 0.0;

          if (outerIsCCW) {
            _appendRectBoundaryCW(out, rect, sA, uA, sB, uB, dedupEps);
          } else {
            _appendRectBoundaryCCW(out, rect, sA, uA, sB, uB, dedupEps);
          }

          i = verts.indexOf(exitHit); // continue from EXIT
        }
      }

      i++;
    }

    // Close & orient CCW
    if (out.length >= 2 &&
        (out.first.x - out.last.x).abs() <= dedupEps &&
        (out.first.y - out.last.y).abs() <= dedupEps) out.removeLast();
    final ccw = _signedArea(out) >= 0 ? out : List<Vector2>.from(out.reversed);

    final poly = PolyShape.fromRings(
      outer: ccw, holes: const [],
      padX1: terrain.padX1, padX2: terrain.padX2, padY: terrain.padY,
    );
    return Terrain(
      poly: poly,
      ridge: ccw,
      padX1: terrain.padX1, padX2: terrain.padX2, padY: terrain.padY,
    );
  }

  // ==========================================================================
  // PAD HELPERS
  // ==========================================================================
  static Terrain retagPadToNearestFlatEdge({
    required Terrain terrain,
    required double cx,
    required double y,
    double targetWidth = 80.0,
    double maxDX = 140.0,
    double maxDY = 28.0,
    double maxSlopeTan = 0.10,  // ~6°
  }) {
    final edges = terrain.poly.edges;
    int oldPadIdx = -1;
    for (int i = 0; i < edges.length; i++) {
      if (edges[i].kind == PolyEdgeKind.pad) {
        oldPadIdx = i;
        break;
      }
    }

    int bestIdx = -1;
    double bestScore = -1e9;

    for (int i = 0; i < edges.length; i++) {
      final e = edges[i];
      if (e.kind == PolyEdgeKind.pad) continue;

      final ax = e.a.x.toDouble(),
          ay = e.a.y.toDouble();
      final bx = e.b.x.toDouble(),
          by = e.b.y.toDouble();
      final dx = bx - ax,
          dy = by - ay;

      final slope = (dy.abs() / (dx.abs() + 1e-9));
      if (slope > maxSlopeTan) continue;

      final mx = 0.5 * (ax + bx),
          my = 0.5 * (ay + by);
      if ((mx - cx).abs() > maxDX) continue;
      if ((my - y).abs() > maxDY) continue;

      final xSpan = (bx - ax).abs();
      final widthFit = (xSpan >= 0.7 * targetWidth) ? 1.0 : (xSpan /
          (0.7 * targetWidth));
      final dist = math.sqrt((mx - cx) * (mx - cx) + (my - y) * (my - y));
      final score = 2.0 * widthFit - 0.02 * dist - 0.5 * slope;

      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }

    if (bestIdx < 0) return terrain;

// 3) retag: demote old pad, promote new one
    if (oldPadIdx >= 0) {
      edges[oldPadIdx].kind = PolyEdgeKind.terrain;
    }
    edges[bestIdx].kind = PolyEdgeKind.pad;

// 4) Sync terrain + poly pad metadata to the NEW edge
    try {
      final e = edges[bestIdx];
      final x1 = e.a.x.toDouble();
      final y1 = e.a.y.toDouble();
      final x2 = e.b.x.toDouble();
      final y2 = e.b.y.toDouble();

      final px1 = math.min(x1, x2);
      final px2 = math.max(x1, x2);
      final py = 0.5 * (y1 + y2);
      final pW = (x2 - x1).abs();

      // Terrain fields (best-effort; use copyWith if your Terrain is immutable)
      try {
        (terrain as dynamic).padX1 = px1;
      } catch (_) {}
      try {
        (terrain as dynamic).padX2 = px2;
      } catch (_) {}
      try {
        (terrain as dynamic).padY = py;
      } catch (_) {}

      // Convenient extras some systems use
      try {
        (terrain as dynamic).padCenter = 0.5 * (px1 + px2);
      } catch (_) {}
      try {
        (terrain as dynamic).padWidth = pW;
      } catch (_) {}
      try {
        (terrain as dynamic).padHalfWidth = 0.5 * pW;
      } catch (_) {}

      // If your PolyShape also caches pad extents, update those too:
      try {
        (terrain.poly as dynamic).padX1 = px1;
      } catch (_) {}
      try {
        (terrain.poly as dynamic).padX2 = px2;
      } catch (_) {}
      try {
        (terrain.poly as dynamic).padY = py;
      } catch (_) {}
    } catch (_) {
      /* ok */
    }

// Rebuild any caches
    try {
      (terrain as dynamic).rebuildDerived();
    } catch (_) {}
    try {
      (terrain.poly as dynamic).rebuildDerived();
    } catch (_) {}

    return terrain;
  }

  static Terrain digRectShelfAndRetagPad({
    required Terrain terrain,
    required double cx,
    required double y,             // local ground estimate (heightAt(cx))
    required double padWidth,
    double padThickness = 16.0,    // vertical thickness of the shelf
    double insetBelow = 10.0,      // place shelf a bit below ground
    double nearEdgeSnapPx = 0.9,
  }) {
    final halfW = padWidth * 0.5;
    final halfH = padThickness * 0.5;
    final yShelf = y - insetBelow;

    final x0 = cx - halfW;
    final x1 = cx + halfW;
    final y0 = yShelf - halfH;
    final y1 = yShelf + halfH;

    Terrain t = carveRectangle(
      terrain: terrain,
      x0: x0, y0: y0, x1: x1, y1: y1,
      nearEdgeSnapPx: nearEdgeSnapPx,
    );

    t = retagPadToNearestFlatEdge(
      terrain: t,
      cx: cx,
      y: yShelf,
      targetWidth: padWidth,
      maxDX: math.max(halfW * 1.2, 80.0),
      maxDY: math.max(halfH * 1.5, 24.0),
      maxSlopeTan: 0.12,
    );

    try { (t as dynamic).rebuildDerived(); } catch (_) {}
    return t;
  }

  // ==========================================================================
  // Shared helpers
  // ==========================================================================
  static double _twoPi = math.pi * 2.0;
  static double _norm2pi(double a) { a = a % _twoPi; if (a < 0) a += _twoPi; return a; }
  static double _ang(Vector2 p, Vector2 C) => math.atan2(p.y - C.y, p.x - C.x);

  static void _appendArcCW(
      List<Vector2> out, Vector2 C, double r,
      Vector2 enterP, Vector2 exitP, double maxErr,
      ) {
    final s = _norm2pi(_ang(enterP, C));
    final e = _norm2pi(_ang(exitP, C));
    final delta = _norm2pi(s - e); if (delta <= 1e-12) return;
    final maxDelta = (maxErr <= 0 || maxErr >= r)
        ? (math.pi / 4)
        : 2 * math.acos(math.max(-1.0, math.min(1.0, 1 - maxErr / r)));
    final segs = math.max(2, (delta / maxDelta).ceil());
    for (int i = 1; i <= segs; i++) {
      final th = s - (i / segs) * delta;
      _pushDedup(out, Vector2(C.x + r * math.cos(th), C.y + r * math.sin(th)), eps: 1e-9);
    }
  }

  static void _appendArcCCW(
      List<Vector2> out, Vector2 C, double r,
      Vector2 enterP, Vector2 exitP, double maxErr,
      ) {
    final s = _norm2pi(_ang(enterP, C));
    final e = _norm2pi(_ang(exitP, C));
    final delta = _norm2pi(e - s); if (delta <= 1e-12) return;
    final maxDelta = (maxErr <= 0 || maxErr >= r)
        ? (math.pi / 4)
        : 2 * math.acos(math.max(-1.0, math.min(1.0, 1 - maxErr / r)));
    final segs = math.max(2, (delta / maxDelta).ceil());
    for (int i = 1; i <= segs; i++) {
      final th = s + (i / segs) * delta;
      _pushDedup(out, Vector2(C.x + r * math.cos(th), C.y + r * math.sin(th)), eps: 1e-9);
    }
  }

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

// ============================================================================
// Rectangle helpers + robust segment↔rect intersection and boundary walkers
// ============================================================================

class _Rect {
  final double left, top, right, bottom;
  _Rect(this.left, this.top, this.right, this.bottom);
  double get width  => right - left;
  double get height => bottom - top;

  // STRICT interior: boundary is OUTSIDE
  bool containsStrict(Vector2 p, {double eps = 1e-9}) =>
      p.x > left  + eps && p.x < right - eps &&
          p.y > top   + eps && p.y < bottom - eps;

  double minEdgeDistance(Vector2 p) {
    final dxL = (p.x - left).abs();
    final dxR = (p.x - right).abs();
    final dyT = (p.y - top).abs();
    final dyB = (p.y - bottom).abs();
    return math.min(math.min(dxL, dxR), math.min(dyT, dyB));
  }
}

class _SegRectHit {
  _SegRectHit(this.t, this.p, this.side, this.u);
  final double t;
  final Vector2 p;
  final int side;   // 0=top,1=right,2=bottom,3=left (CW)
  final double u;   // 0..1 along that side (CW)
}

List<_SegRectHit> _segmentRectIntersections(Vector2 A, Vector2 B, _Rect r) {
  final dx = B.x - A.x, dy = B.y - A.y;
  final hits = <_SegRectHit>[];

  void addIf(double t, Vector2 p, int side, double u) {
    if (t > 1e-9 && t < 1.0 - 1e-9) hits.add(_SegRectHit(t, p, side, u.clamp(0.0, 1.0)));
  }

  // Top y=top (side 0): u = (x-left)/w
  if (dy.abs() > 1e-12) {
    final t = (r.top - A.y) / dy;
    final x = A.x + t * dx;
    if (x >= r.left - 1e-9 && x <= r.right + 1e-9) {
      addIf(t, Vector2(x, r.top), 0, (x - r.left) / (r.width <= 1e-12 ? 1.0 : r.width));
    }
  }

  // Bottom y=bottom (side 2): u = (right-x)/w
  if (dy.abs() > 1e-12) {
    final t = (r.bottom - A.y) / dy;
    final x = A.x + t * dx;
    if (x >= r.left - 1e-9 && x <= r.right + 1e-9) {
      addIf(t, Vector2(x, r.bottom), 2, (r.right - x) / (r.width <= 1e-12 ? 1.0 : r.width));
    }
  }

  // Right x=right (side 1): u = (y-top)/h
  if (dx.abs() > 1e-12) {
    final t = (r.right - A.x) / dx;
    final y = A.y + t * dy;
    if (y >= r.top - 1e-9 && y <= r.bottom + 1e-9) {
      addIf(t, Vector2(r.right, y), 1, (y - r.top) / (r.height <= 1e-12 ? 1.0 : r.height));
    }
  }

  // Left x=left (side 3): u = (bottom-y)/h
  if (dx.abs() > 1e-12) {
    final t = (r.left - A.x) / dx;
    final y = A.y + t * dy;
    if (y >= r.top - 1e-9 && y <= r.bottom + 1e-9) {
      addIf(t, Vector2(r.left, y), 3, (r.bottom - y) / (r.height <= 1e-12 ? 1.0 : r.height));
    }
  }

  hits.sort((a, b) => a.t.compareTo(b.t));
  final out = <_SegRectHit>[];
  for (final h in hits) {
    if (out.isEmpty ||
        (h.p.x - out.last.p.x).abs() > 1e-9 ||
        (h.p.y - out.last.p.y).abs() > 1e-9) {
      out.add(h);
    }
  }
  return out;
}

double _rectPerimParam(int side, double u) => (side & 3) + u.clamp(0.0, 1.0);

void _appendRectBoundaryCW(
    List<Vector2> out,
    _Rect r,
    int sideA, double uA,
    int sideB, double uB,
    double dedupEps,
    ) {
  double perA = _rectPerimParam(sideA, uA);
  double perB = _rectPerimParam(sideB, uB);
  if (perB < perA - 1e-12) perB += 4.0;

  int curSide = sideA & 3;
  double curU = uA.clamp(0.0, 1.0);

  Vector2 pt(int side, double u) {
    switch (side & 3) {
      case 0: return Vector2(r.left + u * r.width, r.top);
      case 1: return Vector2(r.right, r.top + u * r.height);
      case 2: return Vector2(r.right - u * r.width, r.bottom);
      default:return Vector2(r.left, r.bottom - u * r.height);
    }
  }

  _pushDedup(out, pt(curSide, curU), eps: dedupEps);

  int guard = 0;
  while (true) {
    if (++guard > 12) break;
    final curPer = _rectPerimParam(curSide, curU);
    final endThisSide = _rectPerimParam(curSide, 1.0);
    if (perB <= endThisSide + 1e-12) {
      final uEnd = (curU + (perB - curPer)).clamp(0.0, 1.0);
      _pushDedup(out, pt(curSide, uEnd), eps: dedupEps);
      break;
    }
    _pushDedup(out, pt(curSide, 1.0), eps: dedupEps); // corner
    curSide = (curSide + 1) & 3; curU = 0.0;
  }
}

void _appendRectBoundaryCCW(
    List<Vector2> out,
    _Rect r,
    int sideA, double uA,
    int sideB, double uB,
    double dedupEps,
    ) {
  double perA = _rectPerimParam(sideA, uA);
  double perB = _rectPerimParam(sideB, uB);
  if (perB > perA + 1e-12) perB -= 4.0;

  int curSide = sideA & 3;
  double curU = uA.clamp(0.0, 1.0);

  Vector2 pt(int side, double u) {
    switch (side & 3) {
      case 0: return Vector2(r.left + u * r.width, r.top);
      case 1: return Vector2(r.right, r.top + u * r.height);
      case 2: return Vector2(r.right - u * r.width, r.bottom);
      default:return Vector2(r.left, r.bottom - u * r.height);
    }
  }

  _pushDedup(out, pt(curSide, curU), eps: dedupEps);

  int guard = 0;
  while (true) {
    if (++guard > 12) break;
    final curPer = _rectPerimParam(curSide, curU);
    final startThisSide = _rectPerimParam(curSide, 0.0);
    if (perB >= startThisSide - 1e-12) {
      final uEnd = (curU - (startThisSide - perB)).clamp(0.0, 1.0);
      _pushDedup(out, pt(curSide, uEnd), eps: dedupEps);
      break;
    }
    _pushDedup(out, pt(curSide, 0.0), eps: dedupEps); // corner
    curSide = (curSide + 3) & 3; curU = 1.0;
  }
}

// Top-level helper so bottom-of-file rectangle walkers can call it.
void _pushDedup(List<Vector2> out, Vector2 p, {double eps = 1e-9}) {
  if (out.isEmpty) { out.add(p); return; }
  final q = out.last;
  if ((p.x - q.x).abs() > eps || (p.y - q.y).abs() > eps) out.add(p);
}

// ============================================================================
// Vertex records
// ============================================================================

class _Vtx {
  final int origIdx;
  final Vector2 p;
  final bool isHit;
  final bool inside;
  final bool insertedAfterEdge;
  final double? tOnEdge;

  _Vtx(
      this.origIdx,
      this.p, {
        required this.isHit,
        required this.inside,
        this.insertedAfterEdge = false,
        this.tOnEdge,
      });
}

class _VtxRect {
  final int origIdx;
  final Vector2 p;
  final bool isHit;
  final bool inside;
  final bool insertedAfterEdge;
  final double? tOnEdge;

  final int? rectSide; // 0..3
  final double? rectU; // 0..1 along that side (CW)

  _VtxRect(
      this.origIdx,
      this.p, {
        required this.isHit,
        required this.inside,
        this.insertedAfterEdge = false,
        this.tOnEdge,
        this.rectSide,
        this.rectU,
      });
}
