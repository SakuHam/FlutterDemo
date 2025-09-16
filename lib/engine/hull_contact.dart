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

  static ({Vector2 top, Vector2 left, Vector2 right})
  verts(Vector2 pos, double ang) {
    final c = math.cos(ang), s = math.sin(ang);
    final top = pos + _rot(_vTop, c, s);
    final left = pos + _rot(_vLeftFoot, c, s);
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

/// Closest-point contact resolver against terrain + world bounds.
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

    // Ground / Pad (bottom support)
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
}
