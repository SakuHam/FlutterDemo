// lib/ai/teacher.dart
import 'dart:math' as math;

import '../engine/game_engine.dart' as eng;
import '../engine/raycast.dart'; // for env.rays kinds
import '../engine/types.dart' as et;

import 'agent.dart' show Intent, intentToIndex;

/// Better heuristic teacher used by curricula to label short-horizon intent.
/// Returns an index into Intent (goLeft/goRight/descendSlow/brakeUp/brakeLeft/brakeRight).
int predictiveIntentLabelAdaptive(
    eng.GameEngine env, {
      double baseTauSec = 1.0,
      double minTauSec = 0.45,
      double maxTauSec = 1.35,
    }) {
  final L = env.lander;
  final T = env.terrain;
  final px = L.pos.x.toDouble();
  final py = L.pos.y.toDouble();
  final vx = L.vel.x.toDouble();
  final vy = L.vel.y.toDouble();

  final padCx = T.padCenter.toDouble();
  final gy = T.heightAt(px);
  final h  = (gy - py).toDouble().clamp(0.0, 1e9);
  final W  = env.cfg.worldW.toDouble();

  // height-adaptive lookahead
  final hNorm = (h / 320.0).clamp(0.0, 1.6);
  final tau   = (baseTauSec * (0.7 + 0.5 * hNorm)).clamp(minTauSec, maxTauSec);

  // predict short-term future (no thrust)
  final g   = env.cfg.t.gravity;
  final xF  = px + vx * tau;
  final vyF = vy + g * tau;

  // pad vector (prefer rays if pad is seen)
  double pdx = padCx - px, pdy = 0.0;
  double sx = 0.0, sy = 0.0, wsum = 0.0;
  for (final r in env.rays) {
    if (r.kind != RayHitKind.pad) continue;
    final dx = r.p.x - px, dy = r.p.y - py;
    final d2 = dx*dx + dy*dy;
    if (d2 <= 1e-9) continue;
    final w = 1.0 / math.sqrt(d2 + 1e-6);
    sx += w * dx; sy += w * dy; wsum += w;
  }
  if (wsum > 0) {
    final inv = 1.0 / wsum; pdx = sx * inv; pdy = sy * inv;
  }

  int i(Intent v) => intentToIndex(v);

  // emergency vertical brake if low & falling too fast near pad center
  double vCapBrakeUp(double hh) => (0.07 * hh + 6.0).clamp(6.0, 16.0);
  final tooLow      = h < 140.0;
  final tooFastDown = vyF > math.max(40.0, vCapBrakeUp(h) + 10.0);
  final nearPadLat  = (px - padCx).abs() <= 0.18 * W;
  if (tooLow && tooFastDown && nearPadLat) return i(Intent.brakeUp);

  // center band: control locally
  final padEnter = 0.08 * W;
  final dxNow = px - padCx;
  if (dxNow.abs() <= padEnter) {
    if (vx >  25.0) return i(Intent.brakeRight);
    if (vx < -25.0) return i(Intent.brakeLeft);
    return i(Intent.descendSlow);
  }

  // will cross center within tau?
  final dxF = xF - padCx;
  final crossesCenter = (dxNow * dxF) < 0.0;
  if (crossesCenter) {
    if (vx >  12.0) return i(Intent.brakeRight);
    if (vx < -12.0) return i(Intent.brakeLeft);
    return i(Intent.descendSlow);
  }

  // drifting away from center line?
  final driftingAway = dxF.abs() > dxNow.abs() + 2.0;
  if (driftingAway) return dxNow > 0.0 ? i(Intent.goLeft) : i(Intent.goRight);

  // small lateral push when nearly stopped but still outside enter band
  if (!crossesCenter && vx.abs() < 6.0 && dxNow.abs() > padEnter) {
    return dxNow > 0.0 ? i(Intent.goLeft) : i(Intent.goRight);
  }

  // if pad seen by rays: align velocity with pad vector (cross/dot tests)
      {
    final vFx = vx, vFy = vyF;
    final vFmag = math.sqrt(vFx*vFx + vFy*vFy) + 1e-9;
    final padLen = math.sqrt(pdx*pdx + pdy*pdy);
    if (padLen > 1e-6) {
      double crossZ(ax, ay, bx, by) => ax*by - ay*bx;
      double dot(ax, ay, bx, by) => ax*bx + ay*by;
      final cp = crossZ(vFx, vFy, pdx, pdy);   // >0 means pad left of velocity
      final dp = dot   (vFx, vFy, pdx, pdy);
      final cpThresh = 0.015 * vFmag * (padLen > 1.0 ? padLen : 1.0);
      final dpBad    = -0.030 * vFmag * (padLen > 1.0 ? padLen : 1.0);
      final misaligned = (cp.abs() > cpThresh) || (dp < dpBad);
      if (misaligned) return cp > 0.0 ? i(Intent.goLeft) : i(Intent.goRight);
    } else {
      // no pad rays; guard vertical speed
      double vCapHover(double hh) => (0.06 * hh + 6.0).clamp(6.0, 18.0);
      if (vyF > 0.85 * vCapHover(h)) return i(Intent.brakeUp);
    }
  }

  // leaving center band outward? nudge back
  final willExitSoon = (dxF.abs() > padEnter) && (dxNow.abs() <= padEnter);
  final vxOutward    = (dxNow.sign == vx.sign) && vx.abs() > 18.0;
  if ((willExitSoon || vxOutward) && h > 90.0) {
    return (dxNow >= 0.0) ? i(Intent.goLeft) : i(Intent.goRight);
  }

  return i(Intent.descendSlow);
}
