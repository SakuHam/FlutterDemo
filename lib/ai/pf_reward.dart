// lib/ai/pf_reward.dart
import 'dart:math' as math;

import '../engine/game_engine.dart' as eng;
import 'potential_field.dart'; // buildPotentialField
import 'agent.dart' as ai;     // ExternalRewardHook typedef

/// Shaping/knobs for PF-based dense reward.
class PFShapingCfg {
  // --- Velocity shaping ---
  final double wAlign;       // reward per unit cos(v, flow)
  final double wVelDelta;    // penalty for ||v - v_pf|| / vmax
  final double vMinClose;
  final double vMaxFar;
  final double alpha;
  final double vmax;
  final double padTightFrac;
  final double hTight;
  final double latBoost;
  final double velPenaltyBoost;
  final double alignBoost;
  final double vMinTouchdown;
  final double feasiness;
  final double xBias;

  // --- Acceleration (Δv) matching to feasible PF Δv ---
  final double wAccAlign;    // reward for cos(dv_actual, dv_pf)
  final double wAccErr;      // penalty for ||dv_actual - dv_pf|| / dv_pf_cap
  final double accEma;       // EMA smoothing for dv_actual

  // Debug
  final bool debug;

  // Extras: vy cap penalty & pad-toward shaping
  final double wVyCap;   // penalty for exceeding adaptive vy cap (downward)
  final double wPadTow;  // reward for lateral velocity toward pad center

  const PFShapingCfg({
    // velocity
    this.wAlign = 1.0,
    this.wVelDelta = 0.6,
    this.vMinClose = 8.0,
    this.vMaxFar = 90.0,
    this.alpha = 1.2,
    this.vmax = 140.0,
    this.padTightFrac = 0.10,
    this.hTight = 140.0,
    this.latBoost = 4.0,
    this.velPenaltyBoost = 3.0,
    this.alignBoost = 1.5,
    this.vMinTouchdown = 2.0,
    this.feasiness = 0.75,
    this.xBias = 3.0,
    // acceleration
    this.wAccAlign = 2.0,
    this.wAccErr = 1.0,
    this.accEma = 0.2,
    // debug
    this.debug = false,
    // extras
    this.wVyCap = 0.02,
    this.wPadTow = 0.12,
  });
}

/// Build the dense PF-based reward hook used by Trainer.
ai.ExternalRewardHook makePFRewardHook({
  required eng.GameEngine env,
  PFShapingCfg cfg = const PFShapingCfg(),
}) {
  final pf = buildPotentialField(env, nx: 160, ny: 120, iters: 1200, omega: 1.7, tol: 1e-4);

  // Effective max linear accel per real second
  final double aMax = env.cfg.t.thrustAccel * 0.05 * env.cfg.stepScale; // px/s^2

  // Keep previous velocity to measure actual Δv
  double? prevVx, prevVy;
  double smDvX = 0.0, smDvY = 0.0; // EMA-smoothed dv

  // Optional debug accumulators
  int dbgN = 0;
  double sumAbsDVpfX = 0, sumAbsDVpfY = 0;
  double sumWLat = 0, sumWVer = 0;
  double sumAlignAbs = 0, sumVelPen = 0;

  return ({required eng.GameEngine env, required double dt, required int tStep}) {
    final x = env.lander.pos.x.toDouble();
    final y = env.lander.pos.y.toDouble();
    final vx = env.lander.vel.x.toDouble();
    final vy = env.lander.vel.y.toDouble();

    // Proximity to pad
    final padCx = env.terrain.padCenter.toDouble();
    final dx = x - padCx;
    final dxAbs = dx.abs();
    final W = env.cfg.worldW.toDouble();
    final gy = env.terrain.heightAt(x);
    final h  = (gy - y).toDouble().clamp(0.0, 1000.0);

    final tightX = cfg.padTightFrac * W;
    final px_ = math.exp(- (dxAbs * dxAbs) / (tightX * tightX + 1e-6));
    final ph_ = math.exp(- (h * h) / (cfg.hTight * cfg.hTight + 1e-6));
    final prox = (px_ * ph_).clamp(0.0, 1.0);

    // Alignment to PF flow
    var flow = pf.sampleFlow(x, y);
    final vmag = math.sqrt(vx*vx + vy*vy);
    double align = 0.0;
    if (vmag > 1e-6) {
      align = (vx / vmag) * flow.nx + (vy / vmag) * flow.ny;
    }

    // Kill downward component near pad
    if (prox > 0.65 && flow.fy < 0.0) {
      flow = (fx: flow.fx, fy: 0.0, nx: flow.nx, ny: 0.0, mag: flow.mag);
    }

    // Base PF target velocity (before feasibility clamp)
    var sugg = pf.suggestVelocity(
      x, y,
      vMinClose: cfg.vMinClose,
      vMaxFar: cfg.vMaxFar,
      alpha: cfg.alpha,
      clampSpeed: 9999.0,
    );

    // Flare toward touchdown
    final flareLat = (1.0 - 0.70 * prox);
    final flareVer = (1.0 - 0.45 * prox);
    final magNow = math.sqrt(sugg.vx*sugg.vx + sugg.vy*sugg.vy) + 1e-9;
    sugg = (vx: sugg.vx * flareLat, vy: sugg.vy * flareVer);
    final magTarget = ((1.0 - prox) * magNow + prox * cfg.vMinTouchdown).clamp(0.0, magNow);
    final magNew = math.sqrt(sugg.vx*sugg.vx + sugg.vy*sugg.vy) + 1e-9;
    final kMag = (magTarget / magNew).clamp(0.0, 1.0);
    sugg = (vx: sugg.vx * kMag, vy: sugg.vy * kMag);

    // Feasibility clamp (desired Δv)
    final dv_pf_x_raw = sugg.vx - vx;
    final dv_pf_y_raw = sugg.vy - vy;
    final dv_pf_mag_raw = math.sqrt(dv_pf_x_raw*dv_pf_x_raw + dv_pf_y_raw*dv_pf_y_raw);
    final dv_pf_cap = (aMax * dt * cfg.feasiness).clamp(0.0, 1e9);
    double dv_pf_x = dv_pf_x_raw, dv_pf_y = dv_pf_y_raw;
    if (dv_pf_mag_raw > dv_pf_cap && dv_pf_mag_raw > 1e-9) {
      final s = dv_pf_cap / dv_pf_mag_raw;
      dv_pf_x *= s;
      dv_pf_y *= s;
      sugg = (vx: vx + dv_pf_x, vy: vy + dv_pf_y);
    }

    // Border avoidance (X) & ceiling avoidance (Y)
    final wallTau = 5.0;
    final wallMarginFrac = 0.22;
    final wallBlendMax = 0.80;
    final wallVInward = 0.90;
    final wallVelPenalty = 3.0;
    final Wworld = env.cfg.worldW.toDouble();
    final distL = x;
    final distR = (Wworld - x);
    final baseBand = wallMarginFrac * Wworld;

    final vxTowardL = (-vx).clamp(0.0, double.infinity);
    final vxTowardR = ( vx).clamp(0.0, double.infinity);
    final warnL = baseBand + wallTau * vxTowardL;
    final warnR = baseBand + wallTau * vxTowardR;

    double proxL = 1.0 - (distL / (warnL + 1e-6)).clamp(0.0, 1.0);
    double proxR = 1.0 - (distR / (warnR + 1e-6)).clamp(0.0, 1.0);
    final borderProx = math.max(proxL, proxR);
    double inwardX = 0.0;
    if (proxL >= proxR && proxL > 0.0) inwardX =  1.0;
    if (proxR >  proxL && proxR > 0.0) inwardX = -1.0;
    final gamma = 1.5;
    final blendIn = wallBlendMax * math.pow(borderProx, gamma);
    final vInward = (wallVInward + 0.8 * (vxTowardL + vxTowardR)).clamp(40.0, 200.0);
    final suggWallVx = inwardX * vInward;
    sugg = (
    vx: (1.0 - blendIn) * sugg.vx + blendIn * suggWallVx,
    vy: sugg.vy * (1.0 - 0.35 * borderProx)
    );

    final double Hworld = env.cfg.worldH.toDouble();
    final double distTop = y;
    final double baseBandY = 0.35 * Hworld;
    final double tauY = 5.0;
    final double vyTowardTop = (-vy).clamp(0.0, double.infinity);
    final double warnTop = baseBandY + tauY * vyTowardTop;
    double proxTop = 1.0 - (distTop / (warnTop + 1e-6)).clamp(0.0, 1.0);
    if (vy > 0) proxTop *= 0.6;
    final double gammaY = 1.5;
    final double blendTop = 0.75 * math.pow(proxTop, gammaY);
    final double vDownward = (70.0 + 1.0 * vyTowardTop).clamp(40.0, 220.0);
    sugg = (
    vx: sugg.vx * (1.0 - 0.15 * proxTop),
    vy: (1.0 - blendTop) * sugg.vy + blendTop * vDownward
    );

    final double wallBoostY = 1.0 + 2.5 * proxTop;
    final wallBoost = 1.0 + wallVelPenalty * borderProx;

    // Velocity error with X bias near pad
    final prox2 = prox * prox;
    final wLat = (cfg.xBias) * (1.0 + cfg.latBoost * prox2);
    final wVer = 1.0 * (1.0 + 0.7 * cfg.latBoost * prox2);
    final dvx_vel = (vx - sugg.vx) * wLat;
    final dvy_vel = (vy - sugg.vy) * wVer;
    final vErr = math.sqrt(dvx_vel*dvx_vel + dvy_vel*dvy_vel) / cfg.vmax;

    final wVelEff = cfg.wVelDelta
        * (1.0 + 0.6 * cfg.velPenaltyBoost * prox2)
        * wallBoost
        * wallBoostY;
    final wAlignEff = cfg.wAlign * (1.0 + cfg.alignBoost * prox);

    // Touchdown bonus (gated)
    final touchTarget = cfg.vMinTouchdown.clamp(0.5, 15.0);
    const double touchWeight = 3.0;
    final bool inTouchBand = (h < 90.0) && (dxAbs < 0.12 * W);
    final double vmagNow2 = math.sqrt(vx*vx + vy*vy) + 1e-9;
    final double touchBonus = inTouchBand
        ? touchWeight * (touchTarget - vmagNow2) / (touchTarget + 1e-6)
        : 0.0;

    // Acceleration (Δv) matching
    double rAcc = 0.0;
    if (prevVx != null && prevVy != null && dt > 0) {
      double dvx_act = (vx - prevVx!);
      double dvy_act = (vy - prevVy!);
      smDvX = cfg.accEma * dvx_act + (1.0 - cfg.accEma) * smDvX;
      smDvY = cfg.accEma * dvy_act + (1.0 - cfg.accEma) * smDvY;

      final dv_pf_mag = math.sqrt(dv_pf_x*dv_pf_x + dv_pf_y*dv_pf_y) + 1e-12;
      final dv_act_mag = math.sqrt(smDvX*smDvX + smDvY*smDvY) + 1e-12;
      final cosAcc = ((smDvX * dv_pf_x) + (smDvY * dv_pf_y)) / (dv_pf_mag * dv_act_mag);
      final accAlign = cosAcc.clamp(-1.0, 1.0);
      final errX = (smDvX - dv_pf_x);
      final errY = (smDvY - dv_pf_y);
      final accErr = (math.sqrt(errX*errX + errY*errY) / (dv_pf_cap + 1e-9)).clamp(0.0, 5.0);

      rAcc = cfg.wAccAlign * accAlign - cfg.wAccErr * accErr;
    }

    // Update prev v
    prevVx = vx; prevVy = vy;

    if (cfg.debug) {
      sumAbsDVpfX += dv_pf_x.abs();
      sumAbsDVpfY += dv_pf_y.abs();
      sumWLat += wLat;
      sumWVer += wVer;
      sumVelPen += (wVelEff * vErr);
      sumAlignAbs += wAlignEff * align.abs();
      dbgN++;
      if ((dbgN % 240) == 0) {
        print('[PFDBG] frames=$dbgN | mean|dv_pf_x|=${(sumAbsDVpfX / dbgN).toStringAsFixed(2)} '
            'mean|dv_pf_y|=${(sumAbsDVpfY / dbgN).toStringAsFixed(2)} '
            '| mean wLat=${(sumWLat / dbgN).toStringAsFixed(2)} wVer=${(sumWVer / dbgN).toStringAsFixed(2)} '
            '| velPen=${(sumVelPen / dbgN).toStringAsFixed(3)} align=${(sumAlignAbs / dbgN).toStringAsFixed(3)}');
      }
    }

    // Extra: adaptive vertical speed cap penalty (downward is +vy)
    double vyCapDown = 10.0 + 1.0 * math.sqrt(h);
    vyCapDown = vyCapDown.clamp(10.0, 45.0);
    if (h < 120) vyCapDown = math.min(vyCapDown, 18.0);
    if (h < 60)  vyCapDown = math.min(vyCapDown, 10.0);
    final vyExcess = math.max(0.0, vy - vyCapDown);
    final rVyCap = -cfg.wVyCap * vyExcess;

    // Extra: lateral velocity toward pad center
    final double toward = (-dx.sign) * vx; // >0 when moving toward pad
    final double farScale = (dxAbs / (0.25 * W)).clamp(0.0, 1.0);
    final rPadTow = cfg.wPadTow * farScale * (toward / 120.0).clamp(-1.0, 1.0);

    // Total reward
    final r = wAlignEff * align + touchBonus - wVelEff * vErr + rAcc + rVyCap + rPadTow;
    return r;
  };
}
