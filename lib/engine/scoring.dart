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

/// Shaping that encourages: (1) getting to pad center, (2) purposeful descent,
/// (3) not hovering, (4) alignment of turn/velocity/attitude with pad direction.
/// Collision yields terminal (land/crash). All terms are ≥ 0.
class Scoring {
  static ScoringOutcome apply({
    required EngineConfig cfg,
    required Terrain terrain,
    required Vector2 pos,
    required Vector2 vel,
    required double angle,      // used for landing tolerance & attitude shaping
    required double prevAngle,  // unused here, kept for signature parity
    required double dt,
    required ControlInput u,    // actual control chosen this frame
    int? intentIdx,
  }) {
    double cost = 0.0;

    // ----- TOP CEILING handling -----
    if (pos.y < 0) {
      pos = Vector2(pos.x, 0);
      if (vel.y < 0) vel = Vector2(vel.x, 0);
      cost += cfg.ceilingHitPenalty;
    }
    if (pos.y < cfg.ceilingMargin) {
      final frac = (1.0 - (pos.y / cfg.ceilingMargin)).clamp(0.0, 1.0);
      cost += frac * cfg.ceilingPenaltyPerSec * dt;
    }

    // ----- Border proximity penalty (if wrapping disabled) -----
        {
      final x = pos.x;
      final dLeft = x;
      final dRight = cfg.worldW - x;
      final dEdge = dLeft < dRight ? dLeft : dRight;

      if (!cfg.hardWalls && dEdge < cfg.borderMargin) {
        final frac = (1.0 - (dEdge / cfg.borderMargin)).clamp(0.0, 1.0);
        cost += frac * cfg.borderPenaltyPerSec * dt;
      }
    }

    // Wrap or clamp X (one-shot)
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
    if (pos.y < 0) pos = Vector2(pos.x, 0);

    // ===== Pad-centering shaping (distance to pad center) =====
    final padCx = terrain.padCenter;
    final padHalfW = ((terrain.padX2 - terrain.padX1) * 0.5).clamp(1.0, cfg.worldW);
    final dx = pos.x - padCx;
    final dxN = dx.abs() / padHalfW;

    double centerTerm;
    if (dxN <= 1.0) {
      centerTerm = dxN * dxN;                 // inside pad: quadratic
    } else {
      centerTerm = 1.0 + (dxN - 1.0) * 1.5;   // outside pad: steeper linear
    }
    cost += cfg.wDx * centerTerm * dt;

    // =========================
    // Intent alignment shaping (two-stage only)
    // Encourage goRight when left of pad, goLeft when right of pad.
    // Penalize the opposite. Magnitude scales with |dx|.
    // =========================
    if ((intentIdx ?? u.intentIdx) != null) {
      final int k = (intentIdx ?? u.intentIdx)!; // 0..K-1 from your Intent enum
      // keep in sync with your enum order:
      // 0:hoverCenter, 1:goLeft, 2:goRight, 3:descendSlow, 4:brakeUp
      final bool goLeft  = (k == 1);
      final bool goRight = (k == 2);
      if (goLeft || goRight) {
        final bool correct = (goRight && dx < 0) || (goLeft && dx > 0);
        final double alignMag = (dx.abs() / cfg.worldW).clamp(0.0, 0.5); // 0..0.5
        // Use your existing weights: wPadAlignAssist as a bonus when correct.
        // Apply a slightly larger penalty when wrong to avoid dithering.
        if (correct) {
          cost -= cfg.wPadAlignAssist * alignMag * dt;
        } else {
          cost += (cfg.wPadAlignAssist * 1.5) * alignMag * dt;
        }
      }
    }

    // ===== Anti-hover shaping =====

    // (1) Fuel / thrust penalty (per second while burning)
    if (u.thrust) {
      cost += cfg.fuelPenaltyPerSec * dt;
    }

    // (2) Upward motion penalty (vy is +down; vy<0 => going up)
    if (vel.y < 0) {
      cost += cfg.upwardPenaltyPerVel * (-vel.y) * dt;
    }

    // (3) Loitering penalty: high altitude & not descending fast enough
    final groundY = terrain.heightAt(pos.x);
    final height = groundY - pos.y;       // px above ground
    if (height > cfg.loiterAltFrac * cfg.worldH) {
      if (vel.y < cfg.loiterMinVyDown) {
        final deficit = (cfg.loiterMinVyDown - vel.y); // positive if too slow downward
        final frac = (deficit / (cfg.loiterMinVyDown + 1e-6)).clamp(0.0, 1.0);
        cost += cfg.loiterPenaltyPerSec * frac * dt;
      }
    }

    // ===== Pad-alignment shaping (state + commanded action) =====
    final dxFracWorld = dx.abs() / cfg.worldW;
    final farFromPad = (dx.abs() > cfg.padFarFrac * padHalfW);
    if (dxFracWorld > cfg.padTurnDeadzoneFrac) {
      final desiredSign = dx.sign; // +1 => pad to the right, -1 => left
      final turningLeft  = u.left;
      final turningRight = u.right;

      // Penalize turning opposite of pad direction
      final turningOpposite = (desiredSign > 0 && turningLeft) || (desiredSign < 0 && turningRight);
      if (turningOpposite) {
        cost += cfg.padDirTurnPenaltyPerSec * dt;
      } else if (!turningLeft && !turningRight && farFromPad) {
        // Idle turn when far
        cost += cfg.padIdleTurnPenaltyPerSec * dt;
      }

      // Penalize horizontal velocity away from pad (dx * vx < 0)
      if ((dx * vel.x) < 0) {
        cost += cfg.padVelAwayPenaltyPerVel * vel.x.abs() * dt;
      }

      // Penalize tilt away from pad (angle sign should match dx sign)
      if ((dx * angle) < 0) {
        final angAway = angle.abs().clamp(0.0, math.pi / 2);
        cost += cfg.padAngleAwayPenaltyPerRad * angAway * dt;
      }
    }

    // ===== Collision / Terminal (landing vs crash) =====
    bool terminal = false;
    bool landed = false;
    final bool onPadNow = terrain.isOnPad(pos.x);

    final collided = (pos.y >= groundY - 2.0);
    if (collided) {
      terminal = true;

      final okSpeed = vel.length <= cfg.landingSpeedMax;
      final okAngle = angle.abs() <= cfg.landingAngleMaxRad;

      if (onPadNow && okSpeed && okAngle) {
        landed = true;

        // Landing shaping with center-at-touchdown
        final landPenaltySpeed = 0.2 * (vel.length / (cfg.landingSpeedMax + 1e-6));
        final landPenaltyAngle = 0.2 * (angle.abs() / (cfg.landingAngleMaxRad + 1e-6));
        double landCenterPenalty;
        if (dxN <= 1.0) landCenterPenalty = 0.6 * (dxN * dxN);
        else landCenterPenalty = 0.6 + 0.8 * (dxN - 1);

        cost += (landPenaltySpeed + landPenaltyAngle + landCenterPenalty).clamp(0.0, 2.0);

        // Snap y to pad for stable terminal state
        pos = Vector2(pos.x, terrain.padY - 18.0); // halfHeight ~ 18px
      } else {
        cost += 120.0; // crash penalty
      }
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
