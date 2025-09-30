// lib/ai/teachers.dart
import 'dart:math' as math;
import '../engine/game_engine.dart' as eng;
import '../engine/types.dart' as et;
import 'agent.dart';

// Tunable thresholds (safe defaults)
class IntentTeacherCfg {
  final double farDxFrac;      // far if |dx| > farDxFrac * worldW
  final double nearDxFrac;     // near pad band
  final double highH;          // high altitude (descend slowly)
  final double vCapHoverK;     // hover cap multiplier vs agent’s hover cap
  final double vxBrake;        // when |vx| > this, prefer brake intents
  final double vyBrakeK;       // when vy > k * vCapDesc, brakeUp
  final int    holdMin, holdMax; // duration bounds
  const IntentTeacherCfg({
    this.farDxFrac = 0.20,
    this.nearDxFrac = 0.08,
    this.highH = 140.0,
    this.vCapHoverK = 0.95,
    this.vxBrake = 12.0,
    this.vyBrakeK = 0.9,
    this.holdMin = 2,
    this.holdMax = 10,
  });
}

/// Returns (intentIdx, holdFrames)
(int, int) predictiveIntentLabelAdaptive(eng.GameEngine env, {IntentTeacherCfg cfg = const IntentTeacherCfg()}) {
  final L = env.lander;
  final T = env.terrain;
  final padCx = T.padCenter.toDouble();
  final dx = padCx - L.pos.x.toDouble();
  final gy = T.heightAt(L.pos.x.toDouble());
  final h  = (gy - L.pos.y).toDouble().clamp(0.0, 1e9);

  final vx = L.vel.x.toDouble();
  final vy = L.vel.y.toDouble();

  final worldW = env.cfg.worldW;
  final nearDx = cfg.nearDxFrac * worldW;
  final farDx  = cfg.farDxFrac * worldW;

  // Speed caps used by in-agent controllers
  double vCapHover(double hh)  => (0.06 * hh + 6.0).clamp(6.0, 18.0);
  double vCapDesc (double hh)  => (0.10 * hh + 8.0).clamp(8.0, 26.0);

  final vHover = vCapHover(h);
  final vDesc  = vCapDesc(h);

  // 1) vertical safety first: if we’re falling too fast → brakeUp
  if (vy > cfg.vyBrakeK * vDesc) {
    return (intentToIndex(Intent.brakeUp), _hold(h, cfg.holdMin, cfg.holdMax));
  }

  // 2) horizontal stabilization when too fast sideways
  if (vx.abs() > cfg.vxBrake) {
    if (vx > 0) return (intentToIndex(Intent.brakeRight), _hold(h, cfg.holdMin, cfg.holdMax));
    return (intentToIndex(Intent.brakeLeft), _hold(h, cfg.holdMin, cfg.holdMax));
  }

  // 3) far from pad → go toward it
  if (dx.abs() > farDx) {
    return (dx >= 0)
        ? (intentToIndex(Intent.goRight), _hold(h, cfg.holdMin, cfg.holdMax))
        : (intentToIndex(Intent.goLeft) , _hold(h, cfg.holdMin, cfg.holdMax));
  }

  // 4) near the pad band → controlled descend (slower if high)
  if (dx.abs() <= nearDx) {
    if (h > cfg.highH) {
      return (intentToIndex(Intent.descendSlow), _hold(h, cfg.holdMin + 1, cfg.holdMax + 2));
    }
    return (intentToIndex(Intent.descendSlow), _hold(h, cfg.holdMin, cfg.holdMax));
  }

  // 5) otherwise small nudges toward pad, conservative hover if too floaty
  final tooFloaty = (vy < 0 && vy.abs() < cfg.vCapHoverK * vHover);
  if (tooFloaty) {
    return (intentToIndex(Intent.hover), _hold(h, cfg.holdMin, cfg.holdMax));
  }
  return (dx >= 0)
      ? (intentToIndex(Intent.goRight), _hold(h, cfg.holdMin, cfg.holdMax))
      : (intentToIndex(Intent.goLeft) , _hold(h, cfg.holdMin, cfg.holdMax));
}

int _hold(double h, int lo, int hi) {
  final hh = h.clamp(0.0, 400.0);
  final frac = (hh / 400.0);
  final frames = (lo + frac * (hi - lo)).round();
  return frames.clamp(lo, hi);
}
