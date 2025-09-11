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

/// Centralized scoring logic (time-integrated shaping + events).
/// Everything about penalties/bonuses/landing/crash lives here.
class Scoring {
  static ScoringOutcome apply({
    required EngineConfig cfg,
    required Terrain terrain,
    required Vector2 pos,
    required Vector2 vel,
    required double angle,
    required double prevAngle,
    required double dt,
    required ControlInput u,
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

    // Continuous edge penalty (pre-wrap position)
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

    // Wrap or hard walls
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

    // Clamp Y to world (don’t let it go above top too much)
    if (pos.y < 0) pos = Vector2(pos.x, 0);

    // =========================
    // Base shaping (dense) — TIME INTEGRATED
    // =========================
    final groundY = terrain.heightAt(pos.x);
    final padCenterX = terrain.padCenter;

    // Living time
    cost += cfg.livingCost * dt;

    // NOTE: effort cost must be handled by caller via "power" if desired.
    // Here we assume caller already added cfg.effortCost * power * dt to cost,
    // or prefers to keep effort inside physics. We won't add it here.

    // Horizontal distance to pad center (normalized)
    final dxN = (pos.x - padCenterX).abs() / cfg.worldW;
    cost += (cfg.wDx * dxN) * dt;

    // Vertical distance to ground (normalized to worldH)
    final dyN = ((groundY - pos.y).abs()) / cfg.worldH;
    cost += (cfg.wDy * dyN) * dt;

    // Velocity shaping (normalize vx, vy roughly by 200 px/s)
    final vxN = (vel.x.abs() / 200.0).clamp(0.0, 2.0);
    final vyDownN = (vel.y > 0 ? (vel.y / 200.0) : 0.0).clamp(0.0, 2.0);
    cost += (cfg.wVx * vxN + cfg.wVyDown * vyDownN) * dt;

    // Angle shaping (degrees normalized by 180), stronger near ground
    final hN = (dyN).clamp(0.0, 1.0);
    final angleDeg = angle.abs() * 180.0 / math.pi;
    final angleCost = cfg.wAngleDeg * (angleDeg / 180.0) * (1.0 + cfg.angleNearGroundBoost * (1.0 - hN));
    cost += angleCost * dt;

    // Angular rate (smoothness) — rate is per second, integrate over dt
    final dAng = ((angle - prevAngle).abs()) / (dt + 1e-8);
    cost += cfg.wAngleRate * (dAng / math.pi) * dt;

    // =========================
    // Collision / Terminal
    // =========================
    bool terminal = false;
    bool landed = false;
    final bool onPadNow = terrain.isOnPad(pos.x);

    final gentleSpeed = vel.length;
    final gentleAngle = angle.abs();

    final collided = (pos.y >= groundY - 2.0);

    if (collided) {
      terminal = true;
      // Landing check with current curriculum tolerances
      final okSpeed = gentleSpeed <= cfg.landingSpeedMax;
      final okAngle = gentleAngle <= cfg.landingAngleMaxRad;

      if (onPadNow && okSpeed && okAngle) {
        landed = true;
        // Small landing penalty term (kept >=0) so "perfect" is still ~0
        final landPenalty = 0.2 * (gentleSpeed / (cfg.landingSpeedMax + 1e-6)) +
            0.2 * (gentleAngle / (cfg.landingAngleMaxRad + 1e-6));
        cost += landPenalty.clamp(0.0, 1.0); // one-shot

        // Snap to pad
        pos = Vector2(pos.x, terrain.padY - 18.0); // halfHeight ~ 18px (UI parity)
      } else {
        // Crash penalty: bigger than any wrap/edge nuisances
        cost += 120.0; // one-shot
      }
    }

    // ===== Action-aware shaping (bonuses & a small idle penalty) — TIME INTEGRATED =====
    const double vyCap = 120.0;                  // scale for descent speed
    const double angCap = math.pi / 4;           // 45° scale for angle correction
    final double dxDead = 0.03 * cfg.worldW;     // deadzone for pad alignment
    const double angDead = 5 * math.pi / 180.0;  // 5° deadzone for angle correction

    // (A) Thrust-assist bonus when descending
    if (u.thrust && vel.y > 0) {
      final vyN = (vel.y / vyCap).clamp(0.0, 1.0); // 0..1 for 0..vyCap
      cost -= (cfg.wThrustAssist * vyN) * dt;
    }

    // (B) Turn-assist bonus when turning toward upright (reduce |angle|)
    if (angle > angDead && u.left) {
      final aN = ((angle - angDead) / angCap).clamp(0.0, 1.0);
      cost -= (cfg.wTurnAssist * aN) * dt;
    } else if (angle < -angDead && u.right) {
      final aN = (((-angle) - angDead) / angCap).clamp(0.0, 1.0);
      cost -= (cfg.wTurnAssist * aN) * dt;
    }

    // (C) Pad-align bonus when turning toward the pad horizontally
    final dx = (pos.x - terrain.padCenter);
    if (dx.abs() > dxDead) {
      if (dx > 0 && u.left) {
        // pad is to the left → turning left helps
        final dN = ((dx.abs() - dxDead) / (0.5 * cfg.worldW)).clamp(0.0, 1.0);
        cost -= (cfg.wPadAlignAssist * dN) * dt;
      } else if (dx < 0 && u.right) {
        // pad is to the right → turning right helps
        final dN = ((dx.abs() - dxDead) / (0.5 * cfg.worldW)).clamp(0.0, 1.0);
        cost -= (cfg.wPadAlignAssist * dN) * dt;
      }
    }

    // (D) Idle penalty when falling fast & not acting
    if (!u.thrust && !u.left && !u.right && vel.y > 80) {
      final vN = ((vel.y - 80) / (vyCap - 80)).clamp(0.0, 1.0);
      cost += (cfg.wIdlePenalty * vN) * dt;
    }

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
