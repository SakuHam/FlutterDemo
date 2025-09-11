import 'dart:math' as math;
import 'types.dart';

/// Result of scoring + boundary handling. May adjust pos/vel (e.g., ceiling clamp).
class ScoringOutcome {
  final Vector2 pos;
  final Vector2 vel;
  final double cost;
  final bool terminal;
  final bool landed;
  final bool onPad;

  ScoringOutcome({
    required this.pos,
    required this.vel,
    required this.cost,
    required this.terminal,
    required this.landed,
    required this.onPad,
  });
}

/// Minimal scoring: horizontal distance to pad center + ceiling/border + landing.
class Scoring {
  static ScoringOutcome apply({
    required EngineConfig cfg,
    required Terrain terrain,
    required Vector2 pos,
    required Vector2 vel,
    required double angle,      // kept only for landing check (angle tolerance)
    required double prevAngle,  // unused (kept for signature parity)
    required double dt,
    required ControlInput u,    // unused in this minimal scoring
  }) {
    double cost = 0.0;

    // ----- TOP CEILING handling -----
    if (pos.y < 0) {
      pos = Vector2(pos.x, 0);
      if (vel.y < 0) vel = Vector2(vel.x, 0);
      cost += cfg.ceilingHitPenalty; // one-shot bump
    }

    // Continuous penalty near the top (ramps 0..1 inside margin)
    if (pos.y < cfg.ceilingMargin) {
      final frac = (1.0 - (pos.y / cfg.ceilingMargin)).clamp(0.0, 1.0);
      cost += frac * cfg.ceilingPenaltyPerSec * dt;  // time integrated
    }

    // ----- Border proximity penalty (if wrapping enabled) -----
    // (Leave it off when hardWalls==true, only wrap/hit still applies.)
        {
      final x = pos.x;
      final dLeft = x;
      final dRight = cfg.worldW - x;
      final dEdge = dLeft < dRight ? dLeft : dRight;

      if (!cfg.hardWalls && dEdge < cfg.borderMargin) {
        final frac = (1.0 - (dEdge / cfg.borderMargin)).clamp(0.0, 1.0);
        cost += frac * cfg.borderPenaltyPerSec * dt; // time integrated
      }
    }

    // Wrap or hard walls (one-shot)
    if (cfg.hardWalls) {
      bool hit = false;
      double x = pos.x;
      if (x < 0) { x = 0; hit = true; }
      if (x > cfg.worldW) { x = cfg.worldW; hit = true; }
      if (hit) cost += cfg.wrapPenalty;
      pos = Vector2(x, pos.y);
    } else {
      bool crossed = false;
      double x = pos.x;
      if (x < 0) { x = cfg.worldW + x; crossed = true; }
      else if (x > cfg.worldW) { x = x - cfg.worldW; crossed = true; }
      if (crossed) cost += cfg.wrapPenalty;
      pos = Vector2(x, pos.y);
    }

    // Clamp Y to world top (already handled above)
    if (pos.y < 0) pos = Vector2(pos.x, 0);

    // =========================
    // The ONLY dense shaping we keep: horizontal distance to pad center
    // =========================
    final padCenterX = terrain.padCenter;
    final dxN = (pos.x - padCenterX).abs() / cfg.worldW; // 0..~0.5
    cost += (cfg.wDx * dxN) * dt;

    // =========================
    // Collision / Terminal (landing vs crash)
    // =========================
    bool terminal = false;
    bool landed = false;
    final bool onPadNow = terrain.isOnPad(pos.x);

    final groundY = terrain.heightAt(pos.x);
    final collided = (pos.y >= groundY - 2.0);

    if (collided) {
      terminal = true;

      // Use existing tolerances for a valid landing.
      final okSpeed = vel.length <= cfg.landingSpeedMax;
      final okAngle = angle.abs() <= cfg.landingAngleMaxRad;

      if (onPadNow && okSpeed && okAngle) {
        landed = true;
        // Very small landing penalty (kept >=0) so "perfect" stays near 0.
        final landPenalty = 0.2 * (vel.length / (cfg.landingSpeedMax + 1e-6)) +
            0.2 * (angle.abs() / (cfg.landingAngleMaxRad + 1e-6));
        cost += landPenalty.clamp(0.0, 1.0);

        // Snap to pad for stable terminal state
        pos = Vector2(pos.x, terrain.padY - 18.0); // halfHeight ~ 18px
      } else {
        // Crash penalty (one-shot): bigger than border/wrap nuisances.
        cost += 120.0;
      }
    }

    // No negative cost.
    if (cost < 0) cost = 0.0;

    return ScoringOutcome(
      pos: pos,
      vel: vel,
      cost: cost,
      terminal: terminal,
      landed: landed,
      onPad: onPadNow,
    );
  }
}
