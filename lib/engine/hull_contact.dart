// lib/engine/hull_contact.dart
import 'dart:math' as math;

import 'types.dart';

/// Public contact kinds used by the engine.
enum ContactKind { none, terrain, pad, wall, ceiling }

/// Lander hull (triangle) geometry & utilities.
class Hull {
  // Match your UI painter triangle.
  static const double halfW = 14.0;
  static const double halfH = 18.0;

  static final Vector2 _vTop       = Vector2(0.0, -halfH);
  static final Vector2 _vLeftFoot  = Vector2(-halfW,  halfH);
  static final Vector2 _vRightFoot = Vector2( halfW,  halfH);

  static Vector2 _rot(Vector2 v, double c, double s) =>
      Vector2(c * v.x - s * v.y, s * v.x + c * v.y);

  static ({Vector2 top, Vector2 left, Vector2 right}) verts(Vector2 pos, double ang) {
    final c = math.cos(ang), s = math.sin(ang);
    final top   = pos + _rot(_vTop,       c, s);
    final left  = pos + _rot(_vLeftFoot,  c, s);
    final right = pos + _rot(_vRightFoot, c, s);
    return (top: top, left: left, right: right);
  }

  /// Returns the support (extreme) point in direction (dirX, dirY).
  static Vector2 supportPoint(Vector2 pos, double ang, double dirX, double dirY) {
    final v = verts(pos, ang);
    Vector2 best = v.top;
    double bestDot = best.x * dirX + best.y * dirY;

    void test(Vector2 p) {
      final d = p.x * dirX + p.y * dirY;
      if (d > bestDot) { bestDot = d; best = p; }
    }

    test(v.left);
    test(v.right);
    return best;
  }
}

/// Closest-point contact resolver against terrain + world bounds,
/// with an extra vertex-in-polygon guard to prevent corner tunneling.
class ContactResolver {
  final EngineConfig cfg;
  final Terrain terrain;

  const ContactResolver({required this.cfg, required this.terrain});

  /// Returns (kind, correctedPos, correctedVel).
  ({ContactKind kind, Vector2 pos, Vector2 vel})
  resolve(Vector2 pos, Vector2 vel, double angle) {
    final W = cfg.worldW.toDouble();
    final H = cfg.worldH.toDouble();
    const eps = 1e-6;

    // ===== Bounds first (ceil & walls) =====

    // Ceiling (top support)
    final top = Hull.supportPoint(pos, angle, 0.0, -1.0);
    if (top.y <= 0.0) {
      final dy = 0.0 - top.y;
      final newPos = Vector2(pos.x, pos.y + dy);                   // push down
      final newVel = Vector2(vel.x, math.max(0.0, vel.y));         // kill upward
      return (kind: ContactKind.ceiling, pos: newPos, vel: newVel);
    }

    // Left wall
    final leftPt = Hull.supportPoint(pos, angle, -1.0, 0.0);
    if (leftPt.x <= 0.0) {
      final dx = 0.0 - leftPt.x;
      final newPos = Vector2(pos.x + dx, pos.y);                   // push right
      final newVel = Vector2(math.max(0.0, vel.x), vel.y);         // kill leftward
      return (kind: ContactKind.wall, pos: newPos, vel: newVel);
    }

    // Right wall
    final rightPt = Hull.supportPoint(pos, angle, 1.0, 0.0);
    if (rightPt.x >= W) {
      final dx = W - rightPt.x;
      final newPos = Vector2(pos.x + dx, pos.y);                   // push left
      final newVel = Vector2(math.min(0.0, vel.x), vel.y);         // kill rightward
      return (kind: ContactKind.wall, pos: newPos, vel: newVel);
    }

    // ===== Vertex-in-polygon guard (prevents corner tunneling) =====
    // If any hull vertex is inside the solid polygon, push out along nearest edge normal.
    final outer = terrain.poly.outer;
    if (outer.length >= 3) {
      final vtx = Hull.verts(pos, angle);
      final List<Vector2> hullPts = [vtx.top, vtx.left, vtx.right];

      // Determine outer ring orientation to pick outward normal correctly.
      final bool outerIsCCW = _signedArea(outer) > 0.0;

      for (final v in hullPts) {
        if (_pointInOuterCCW(v, outer)) {
          final nearest = _nearestEdgeToPoint(v, outerIsCCW);
          // Small bias so we end up clearly outside
          const double bias = 0.5;
          final corr = Vector2(
            nearest.normal.x * (nearest.dist + bias),
            nearest.normal.y * (nearest.dist + bias),
          );
          final newPos = Vector2(pos.x + corr.x, pos.y + corr.y);

          // Kill inward velocity component (inelastic pushout)
          final vn = vel.x * nearest.normal.x + vel.y * nearest.normal.y;
          Vector2 newVel = vel;
          if (vn < 0) {
            newVel = Vector2(
              vel.x - vn * nearest.normal.x,
              vel.y - vn * nearest.normal.y,
            );
          }

          final kind = (nearest.kind == PolyEdgeKind.pad)
              ? ContactKind.pad
              : ContactKind.terrain;

          return (kind: kind, pos: newPos, vel: newVel);
        }
      }
    }

    // ===== Ground / Pad (bottom support, legacy height probe) =====
    // Kept for compatibility/feel; the vertex guard above already fixes corner misses.
    final bottom = Hull.supportPoint(pos, angle, 0.0, 1.0);
    final groundY = terrain.heightAt(bottom.x);
    if (bottom.y >= groundY - eps) {
      final dy = groundY - bottom.y;
      final newPos = Vector2(pos.x, pos.y + dy);                   // snap bottom onto ground
      final onPad = terrain.isOnPad(bottom.x) &&
          (groundY - terrain.padY).abs() < 1e-6;
      final kind = onPad ? ContactKind.pad : ContactKind.terrain;
      final newVel = Vector2(vel.x, math.min(0.0, vel.y));         // kill downward
      return (kind: kind, pos: newPos, vel: newVel);
    }

    return (kind: ContactKind.none, pos: pos, vel: vel);
  }

  // ---------- Geometry helpers ----------

  /// Ray-crossing test for a point against the OUTER ring (assumed CCW, but works regardless).
  /// Boundary points are treated as inside (<= eps).
  bool _pointInOuterCCW(Vector2 p, List<Vector2> ring) {
    bool inside = false;
    for (int i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      final a = ring[i], b = ring[j];
      final yi = a.y, yj = b.y;
      final xi = a.x, xj = b.x;
      final denom = (yj - yi);
      final xIntersect = (denom.abs() < 1e-12)
          ? xi
          : (xj - xi) * (p.y - yi) / denom + xi;
      final cond = ((yi > p.y) != (yj > p.y)) && (p.x < xIntersect);
      if (cond) inside = !inside;
    }
    if (!inside) {
      // Boundary as inside
      const eps = 1e-6;
      for (int i = 0; i < ring.length; i++) {
        final a = ring[i];
        final b = ring[(i + 1) % ring.length];
        if (_pointSegmentDistance(p, a, b) <= eps) return true;
      }
    }
    return inside;
  }

  double _pointSegmentDistance(Vector2 p, Vector2 a, Vector2 b) {
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

  /// Nearest edge (by perpendicular distance) to point p.
  /// Returns closest point on the edge, outward normal (unit), distance, and edge kind.
  ({
  Vector2 closest,
  Vector2 normal,
  double dist,
  PolyEdgeKind kind,
  }) _nearestEdgeToPoint(Vector2 p, bool outerIsCCW) {
    final edges = terrain.poly.edges;
    double bestD = double.infinity;
    Vector2 bestP = p;
    Vector2 bestN = Vector2(0, -1);
    PolyEdgeKind bestK = PolyEdgeKind.terrain;

    for (final e in edges) {
      final a = e.a, b = e.b;
      // project p onto segment ab
      final abx = b.x - a.x, aby = b.y - a.y;
      final apx = p.x - a.x, apy = p.y - a.y;
      final ab2 = abx * abx + aby * aby;
      final double t = (ab2 <= 1e-12) ? 0.0 : ((apx * abx + apy * aby) / ab2).clamp(0.0, 1.0);
      final q = Vector2(a.x + abx * t, a.y + aby * t);
      final dx = p.x - q.x, dy = p.y - q.y;
      final d = math.sqrt(dx * dx + dy * dy);
      if (d < bestD) {
        bestD = d;
        bestP = q;
        // For CCW outer ring, we treat the left normal (-dy, dx) as pointing OUT of solid.
        // For CW outer, right normal (dy, -dx) points outward.
        double nx, ny;
        if (outerIsCCW) {
          nx = -aby; ny = abx;
        } else {
          nx = aby;  ny = -abx;
        }
        final len = math.sqrt(nx * nx + ny * ny);
        if (len > 1e-12) { nx /= len; ny /= len; }
        bestN = Vector2(nx, ny);
        bestK = e.kind;
      }
    }
    return (closest: bestP, normal: bestN, dist: bestD, kind: bestK);
  }

  double _signedArea(List<Vector2> ring) {
    double a = 0;
    for (int i = 0; i < ring.length; i++) {
      final p = ring[i];
      final q = ring[(i + 1) % ring.length];
      a += p.x * q.y - p.y * q.x;
    }
    return 0.5 * a;
  }
}
