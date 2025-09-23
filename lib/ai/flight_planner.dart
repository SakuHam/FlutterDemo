// lib/ai/flight_planner.dart
import 'dart:math' as math;
import 'package:flutter/material.dart' show Offset;
import '../engine/types.dart' as et;
import '../engine/game_engine.dart' as eng;
import '../engine/raycast.dart';

/// One waypoint of the planned trajectory in world space.
class TrajectoryPoint {
  final double t;   // seconds from "now"
  final double x;
  final double y;
  final double vx;
  final double vy;
  final double kappa; // signed curvature estimate (1/r), for UI tinting/QA

  const TrajectoryPoint({
    required this.t,
    required this.x,
    required this.y,
    required this.vx,
    required this.vy,
    required this.kappa,
  });
}

class FlightPlan {
  final List<TrajectoryPoint> pts;
  final int version;
  const FlightPlan({required this.pts, required this.version});

  bool get isEmpty => pts.isEmpty;
  double get horizon => pts.isEmpty ? 0.0 : pts.last.t;
}

/// Lightweight potential-field based planner with capped accelerations.
/// It simulates a kinematic point mass with gravity and simple thrust caps.
class FlightPlanner {
  final double horizonSec;  // e.g. 2.0
  final int steps;          // e.g. 60 (dt ≈ 1/30 s)
  final double kPad;        // attraction toward pad (px/s^2 scale)
  final double kAlt;        // altitude shaping gain (px/s^2 scale)
  final double kDamp;       // velocity damping (1/s)
  final double kRepel;      // terrain/wall repulsion weight
  final double repelRange;  // px, rays closer than this push harder
  final double lateralAccelMax; // px/s^2
  final double upAccelMax;       // px/s^2 (net upward, after gravity compensation)
  final double downAccelMax;     // px/s^2

  int _version = 1;

  FlightPlanner({
    this.horizonSec = 2.0,
    this.steps = 60,
    this.kPad = 0.9,
    this.kAlt = 0.35,
    this.kDamp = 0.9,
    this.kRepel = 2500.0,
    this.repelRange = 160.0,
    this.lateralAccelMax = 90.0,
    this.upAccelMax = 110.0,
    this.downAccelMax = 65.0,
  });

  FlightPlan plan(eng.GameEngine env) {
    final L = env.lander;
    final T = env.terrain;
    final g  = env.cfg.t.gravity;       // px/s^2 downward (+vy direction)
    final dt = (horizonSec / math.max(1, steps));

    // Start from current state.
    double x = L.pos.x.toDouble();
    double y = L.pos.y.toDouble();
    double vx = L.vel.x.toDouble();
    double vy = L.vel.y.toDouble();

    // Attractor target: pad center at ground height.
    final padCx = T.padCenter.toDouble();
    final padCy = T.heightAt(padCx);

    // Helper: sample PF vector at (x,y).
    Offset _pf(double px, double py) {
      // --- Pad attraction (toward (padCx, padCy)) -------------
      double ax = padCx - px;
      double ay = padCy - py;
      final al = math.sqrt(ax*ax + ay*ay);
      if (al > 1e-6) { ax /= al; ay /= al; }

      // --- Altitude shaping: gently prefer being above terrain ---
      // target height h* grows with distance from pad (safer approach arcs)
      final groundY = T.heightAt(px);
      final dxPad = (px - padCx).abs();
      final hStar = (110.0 + 0.20 * dxPad).clamp(90.0, 260.0);
      final eH = (groundY + hStar) - py; // positive => we want to go up
      final altUp = eH;

      // --- Velocity damping (so curves settle) ------------------
      final dampX = -kDamp * vx;
      final dampY = -kDamp * vy;

      // --- Repulsion from nearby terrain/walls via rays ---------
      // Heuristic: use env.rays; if a ray hits terrain close, push away.
      double rx = 0.0, ry = 0.0;
      for (final r in env.rays) {
        if (r.kind == RayHitKind.terrain || r.kind == RayHitKind.wall) {
          final dx = r.p.x - px;
          final dy = r.p.y - py;
          final d = math.sqrt(dx*dx + dy*dy);
          if (d > 1e-6 && d < repelRange) {
            // repulsion points from hit toward us: normalize (-dx,-dy)
            final w = kRepel * (1.0 / (d * d)); // 1/d^2
            rx += -dx * (w / d);
            ry += -dy * (w / d);
          }
        }
      }

      // Combine
      double axTot = kPad * ax + dampX + rx;
      double ayTot = kPad * ay + kAlt * altUp + dampY + ry;

      // Net vertical accel must consider gravity and thrust caps.
      // We apply caps later, but bias ay upward to offset gravity when needed.
      return Offset(axTot, ayTot);
    }

    // Curvature estimator.
    double _curv(double vx, double vy, double ax, double ay) {
      final v2 = vx*vx + vy*vy;
      if (v2 < 1e-6) return 0.0;
      final num = (vx * ay - vy * ax).abs();
      return num / (v2 * math.sqrt(v2) + 1e-9); // ~|v×a|/|v|^3
    }

    final out = <TrajectoryPoint>[];
    double tAcc = 0.0;
    for (int i = 0; i < steps; i++) {
      final f = _pf(x, y);

      // Split accel into lateral & vertical components in world frame.
      double ax = f.dx;
      double ay = f.dy;

      // Cap lateral accel magnitude.
      final aLat = ax.abs();
      if (aLat > lateralAccelMax) ax *= (lateralAccelMax / aLat);

      // Vertical accel target must fight gravity; cap up/down separately.
      // The environment's vy increases downward, and g is positive downward.
      // We want "control ay" that, together with +g, stays within caps.
      // Let a_net = g + a_ctrl. So we cap a_ctrl.
      double aCtrl = ay;
      if (aCtrl > 0) {
        // trying to accelerate downward
        if (aCtrl > downAccelMax) aCtrl = downAccelMax;
      } else {
        // trying to accelerate upward
        final up = -aCtrl;
        if (up > upAccelMax) aCtrl = -upAccelMax;
      }

      // Integrate simple dynamics (semi-implicit Euler).
      // Net vertical acceleration is g + aCtrl.
      ax = ax;
      ay = aCtrl + g;

      vx += ax * dt;
      vy += ay * dt;
      x  += vx * dt;
      y  += vy * dt;

      // Keep inside world bounds a bit (soft clamp y).
      x = x.clamp(0.0, env.cfg.worldW.toDouble());
      y = y.clamp(0.0, env.cfg.worldH.toDouble());

      final kappa = _curv(vx, vy, ax, ay);
      out.add(TrajectoryPoint(
        t: tAcc,
        x: x, y: y, vx: vx, vy: vy, kappa: kappa,
      ));
      tAcc += dt;
    }

    return FlightPlan(pts: out, version: _version++);
  }
}
