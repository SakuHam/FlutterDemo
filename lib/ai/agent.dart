// lib/ai/agent.dart
import 'dart:math' as math;
import '../engine/raycast.dart';
import 'nn_helper.dart' as nn;                 // MLPTrunk/PolicyHeads with silu+noise+dropout
import '../engine/types.dart' as et;
import '../engine/game_engine.dart' as eng;

/* -------------------------------------------------------------------------- */
/*                                INTENT SET                                  */
/* -------------------------------------------------------------------------- */

enum Intent { hover, goLeft, goRight, descendSlow, brakeUp, brakeLeft, brakeRight }

const List<String> kIntentNames = [
  'hover',       // 0
  'goLeft',      // 1
  'goRight',     // 2
  'descendSlow', // 3
  'brakeUp',     // 4
  'brakeLeft',   // 5
  'brakeRight',  // 6
];

int intentToIndex(Intent i) => i.index;
Intent indexToIntent(int k) => Intent.values[k.clamp(0, Intent.values.length - 1)];

double _clip(double x, double a, double b) => x < a ? a : (x > b ? b : x);

/* -------------------------------------------------------------------------- */
/*                             HEBBIAN CONFIG/UTIL                            */
/* -------------------------------------------------------------------------- */

class HebbianConfig {
  final bool enabled;
  final bool useOja;   // Oja-stabilized (recommended)
  final double eta;    // learning rate multiplier
  final double decay;  // used if !useOja
  final double clip;   // per-weight delta clip
  final double rowL2Cap; // cap per-row L2 norm after update (<=0 disables)

  // where to apply
  final bool trunk;
  final bool headIntent; // default false to avoid intent lock-in
  final bool headTurn;
  final bool headThr;
  final bool headVal; // usually false

  const HebbianConfig({
    this.enabled = false,
    this.useOja = true,
    this.eta = 3e-4,
    this.decay = 1e-4,
    this.clip = 0.02,
    this.rowL2Cap = 2.5,

    this.trunk = true,
    this.headIntent = false,
    this.headTurn = true,
    this.headThr = true,
    this.headVal = false,
  });
}

double _clipD(double x, double a) => x < -a ? -a : (x > a ? a : x);

double _l2(List<double> v) {
  double s = 0.0; for (final x in v) s += x * x; return math.sqrt(s);
}
void _capRowL2(List<double> row, double cap) {
  if (cap <= 0) return;
  final n = _l2(row);
  if (n > cap && n > 0) {
    final k = cap / n;
    for (int j = 0; j < row.length; j++) row[j] *= k;
  }
}

/* -------------------------------------------------------------------------- */
/*                               RUNNING  NORM                                */
/* -------------------------------------------------------------------------- */

class RunningNorm {
  int dim;
  List<double> mean;
  List<double> var_; // variance, not std
  double momentum;
  bool inited;
  final double eps;

  RunningNorm(this.dim, {this.momentum = 0.995, this.eps = 1e-6})
      : mean = List<double>.filled(dim, 0.0),
        var_ = List<double>.filled(dim, 1.0),
        inited = false;

  void reset({double initVar = 1.0}) {
    for (int i = 0; i < dim; i++) {
      mean[i] = 0.0;
      var_[i] = initVar;
    }
    inited = false;
  }

  void _resizeTo(int newDim) {
    dim = newDim;
    mean = List<double>.filled(dim, 0.0);
    var_  = List<double>.filled(dim, 1.0);
    inited = false;
  }

  void observe(List<double> x) {
    if (x.length != dim) _resizeTo(x.length);
    if (!inited) {
      for (int i = 0; i < dim; i++) {
        mean[i] = x[i];
        var_[i] = 1.0;
      }
      inited = true;
      return;
    }
    final a = 1.0 - momentum;
    for (int i = 0; i < dim; i++) {
      final mPrev = mean[i];
      final xi = x[i];
      final m = momentum * mPrev + a * xi;
      final dev = xi - m;
      final devPrev = xi - mPrev;
      final v = momentum * var_[i] + a * (dev * devPrev);
      mean[i] = m;
      var_[i] = v <= 0 ? eps : v;
    }
  }

  List<double> normalize(List<double> x, {bool update = false}) {
    if (x.length != dim) _resizeTo(x.length);
    if (update) observe(x);
    final out = List<double>.filled(x.length, 0.0);
    for (int i = 0; i < x.length; i++) {
      out[i] = (x[i] - mean[i]) / math.sqrt(var_[i] + eps);
    }
    return out;
  }
}

/* -------------------------------------------------------------------------- */
/*                             FEATURE EXTRACTORS                              */
/* -------------------------------------------------------------------------- */

class FeatureExtractor {
  final int groundSamples;
  final double stridePx;

  FeatureExtractor({this.groundSamples = 3, this.stridePx = 48});

  // [px, py, vx, vy, ang, fuel, padCx, dxToPad, hAboveGround, slope, groundSamples...]
  int get inputSize => 10 + groundSamples;

  List<double> extract(eng.GameEngine env) {
    final L = env.lander;
    final T = env.terrain;
    final px = L.pos.x.toDouble();
    final py = L.pos.y.toDouble();
    final vx = L.vel.x.toDouble();
    final vy = L.vel.y.toDouble();
    final ang = L.angle.toDouble();
    final fuel = L.fuel;

    final padCx = T.padCenter.toDouble();
    final dxToPad = (px - padCx);

    final gy = T.heightAt(px);
    final hAbove = (gy - py);

    final hL = T.heightAt(_clip(px - 8.0, 0.0, env.cfg.worldW));
    final hR = T.heightAt(_clip(px + 8.0, 0.0, env.cfg.worldW));
    final slope = (hR - hL) / 16.0;

    final feats = <double>[
      px / env.cfg.worldW,
      py / env.cfg.worldH,
      vx / 200.0,
      vy / 200.0,
      ang / math.pi,
      fuel / (env.cfg.t.maxFuel > 0 ? env.cfg.t.maxFuel : 1.0),
      padCx / env.cfg.worldW,
      dxToPad / (0.5 * env.cfg.worldW),
      hAbove / 300.0,
      slope / 2.0,
    ];

    for (int i = 0; i < groundSamples; i++) {
      final off = (i - (groundSamples ~/ 2)) * stridePx;
      final gx = _clip(px + off, 0.0, env.cfg.worldW);
      final gyS = T.heightAt(gx);
      feats.add((gyS - py) / 300.0);
    }
    return feats;
  }
}

/// Ray-based features: lander scalars + per-ray channels.
class FeatureExtractorRays {
  final int rayCount;
  final bool kindsOneHot;
  FeatureExtractorRays({required this.rayCount, this.kindsOneHot = true});

  int get inputSize => 5 + rayCount * (kindsOneHot ? 4 : 1);

// In runtime_policy.dart, inside _RuntimeFE_Rays.extract(...)

  @override
  List<double> extract({
    required et.LanderState lander,
    required et.Terrain terrain,
    required double worldW,
    required double worldH,
    required List<RayHit>? rays,
    double uiMaxFuel = 100.0,
  }) {
    if (rays == null) {
      throw StateError('Runtime FE (rays) requires a rays list. Pass engine.rays.');
    }

    // --- Position / angle / fuel (same as before) ---
    final px = (lander.pos.x / worldW);
    final py = (lander.pos.y / worldH);
    final ang = (lander.angle / math.pi).clamp(-2.0, 2.0);
    final fuel = (lander.fuel / (uiMaxFuel > 0 ? uiMaxFuel : 1.0)).clamp(0.0, 1.0);

    // --- height-normalized vertical speed ---
    final gy = terrain.heightAt(lander.pos.x);
    final h  = (gy - lander.pos.y).toDouble().clamp(0.0, 1e9);
    // smooth cap that grows with height (same family used in controllers)
    double vCap = (0.10 * h + 8.0).clamp(8.0, 26.0);         // px/s
    final hnVy  = (lander.vel.y.toDouble() / (vCap > 1e-6 ? vCap : 1.0)).clamp(-3.0, 3.0);

    // --- Polar velocity representation ---
    final vx = lander.vel.x.toDouble();
    final vy = lander.vel.y.toDouble();
    double speed = math.sqrt(vx * vx + vy * vy);
    // soft clip speed and normalize to ~[0,1]
    final sClip = 140.0;
    speed = (speed / sClip).clamp(0.0, 1.5);

    final heading = math.atan2(vy, vx); // [-pi, pi]
    final sinH = math.sin(heading);
    final cosH = math.cos(heading);

    // --- Pad vector + alignment (robust to visibility) ---
    final padCx = (terrain.padX1 + terrain.padX2) * 0.5;
    final padCy = terrain.heightAt(padCx);
    double pxToPad = padCx - lander.pos.x;
    double pyToPad = padCy - lander.pos.y;

    // Try rays-averaged pad vector if available (weighted by 1/d)
    double sx = 0.0, sy = 0.0, wsum = 0.0;
    for (final r in rays) {
      if (r.kind != RayHitKind.pad) continue;
      final dx = r.p.x - lander.pos.x;
      final dy = r.p.y - lander.pos.y;
      final d2 = dx*dx + dy*dy;
      if (d2 <= 1e-9) continue;
      final w = 1.0 / math.sqrt(d2 + 1e-6);
      sx += w * dx; sy += w * dy; wsum += w;
    }
    bool padVecValid = false;
    if (wsum > 0) {
      final inv = 1.0 / wsum;
      pxToPad = sx * inv; pyToPad = sy * inv;
      padVecValid = true;
    }

    // normalize pad dir
    double padLen = math.sqrt(pxToPad*pxToPad + pyToPad*pyToPad);
    double pnx = 0.0, pny = 0.0;
    if (padLen > 1e-6) { pnx = pxToPad / padLen; pny = pyToPad / padLen; }

    // velocity dir (if speed very small, keep 0 to avoid NaNs)
    double vnx = (speed > 1e-6) ? cosH : 0.0;
    double vny = (speed > 1e-6) ? sinH : 0.0;

    double _angWrap(double a) {
      // wrap to [-pi, pi]
      const twoPi = math.pi * 2.0;
      a = a % twoPi;
      if (a > math.pi) a -= twoPi;
      if (a < -math.pi) a += twoPi;
      return a;
    }

    // alignment features
    final cosDelta = vnx * pnx + vny * pny;            // ∈ [-1,1]
    final sinDelta = vnx * pny - vny * pnx;            // >0: pad is to the left of v̂
    final angDelta = _angWrap(math.atan2(pny, pnx) - math.atan2(vny, vnx)); // [-pi, pi]
    final angDeltaPi = (angDelta / math.pi).clamp(-1.0, 1.0);
    final padVis = padVecValid ? 1.0 : 0.0;            // visibility flag

    // --- Assemble base features (6 → now 7) ---

    // --- Assemble base features (6 → now 7) ---
    final out = <double>[
//      px,                      // 0..1
//      py,                      // 0..1
      speed,                   // ~0..1.5 clipped
      hnVy,
//      sinH,                    // -1..1
//      cosH,                    // -1..1
      ang,                     // -2..2
//      fuel,                    // 0..1
      // alignment block:
      angDeltaPi,
//      cosDelta,                // -1..1
//      sinDelta,                // -1..1
      padVis,                  // 0/1
    ];

    // --- Rays channels, same as before ---
    final maxD = math.sqrt(worldW * worldW + worldH * worldH);
    final int n = rayCount;
    for (int i = 0; i < n; i++) {
      RayHit? rh = (i < rays.length) ? rays[i] : null;
      double d;
      if (rh == null) {
        d = maxD;
      } else {
        final dx = rh.p.x - lander.pos.x;
        final dy = rh.p.y - lander.pos.y;
        d = math.sqrt(dx * dx + dy * dy);
      }
      final dN = (d / maxD).clamp(0.0, 1.0);

      if (!kindsOneHot) {
        out.add(dN);
      } else {
        double tTerr = 0, tPad = 0, tWall = 0;
        if (rh != null) {
          switch (rh.kind) {
            case RayHitKind.terrain: tTerr = 1; break;
            case RayHitKind.pad:     tPad  = 1; break;
            case RayHitKind.wall:    tWall = 1; break;
          }
        }
        out.addAll([dN, tTerr, tPad, tWall]);
      }
    }

    return out;
  }
}

/* -------------------------------------------------------------------------- */
/*                         PAD-RAY POLARITY (TRAINING)                         */
/* -------------------------------------------------------------------------- */

({double x, double y, bool valid}) _avgPadVectorFromRays({
  required List<RayHit> rays,
  required double px,
  required double py,
}) {
  double sx = 0.0, sy = 0.0, wsum = 0.0;
  for (final r in rays) {
    if (r.kind != RayHitKind.pad) continue;
    final dx = r.p.x - px;
    final dy = r.p.y - py;
    final d2 = dx * dx + dy * dy;
    if (d2 <= 1e-9) continue;
    final w = 1.0 / math.sqrt(d2 + 1e-6);
    sx += w * dx;
    sy += w * dy;
    wsum += w;
  }
  if (wsum <= 0) return (x: 0.0, y: 0.0, valid: false);
  final inv = 1.0 / wsum;
  return (x: sx * inv, y: sy * inv, valid: true);
}

/* -------------------------------------------------------------------------- */
/*                                CONTROLLERS                                  */
/* -------------------------------------------------------------------------- */

int predictiveIntentLabelAdaptive(
    eng.GameEngine env, {
      double baseTauSec = 1.0,
      double minTauSec = 0.45,
      double maxTauSec = 1.35,
    }) {
  final L = env.lander;
  final T = env.terrain;
  final cfg = env.cfg;

  // ---- State (world space) ----
  final px = L.pos.x.toDouble();
  final py = L.pos.y.toDouble();
  final vx = L.vel.x.toDouble();
  final vy = L.vel.y.toDouble();

  final padCx = T.padCenter.toDouble();
  final gy = T.heightAt(px);
  final h  = (gy - py).toDouble().clamp(0.0, 1e9);
  final W  = cfg.worldW.toDouble();

  // ---- Height-adaptive look-ahead (reaction horizon) ----
  final hNorm = (h / 320.0).clamp(0.0, 1.6);
  final tau   = (baseTauSec * (0.7 + 0.5 * hNorm)).clamp(minTauSec, maxTauSec);

  // ---- Predict short-term future (ignore lateral accel) ----
  final g   = cfg.t.gravity;
  final xF  = px + vx * tau;
  final vyF = vy + g * tau;

  // ---- Build a to-pad vector (prefer rays, fall back to center) ----
  // Try rays-averaged pad vector (weighted by 1/d) for robustness.
  double pdx = 0.0, pdy = 0.0; bool padVecValid = false;
  double sx = 0.0, sy = 0.0, wsum = 0.0;
  for (final r in env.rays) {
    if (r.kind != RayHitKind.pad) continue;
    final dx = r.p.x - px;
    final dy = r.p.y - py;
    final d2 = dx*dx + dy*dy;
    if (d2 <= 1e-9) continue;
    final w = 1.0 / math.sqrt(d2 + 1e-6);
    sx += w * dx; sy += w * dy; wsum += w;
  }
  if (wsum > 0.0) {
    final inv = 1.0 / wsum;
    pdx = sx * inv; pdy = sy * inv;
    padVecValid = true;
  } else {
    // Fallback: use horizontal to the pad center; keep vertical neutral to avoid bias.
    pdx = (padCx - px);
    pdy = 0.0;
    padVecValid = false;
  }

  // ---- Helper math ----
  double crossZ(double ax, double ay, double bx, double by) => ax*by - ay*bx; // sign = “B is to the left of A”
  double dot   (double ax, double ay, double bx, double by) => ax*bx + ay*by;

  // Normalize reference only for angle tests (keep raw for dot/cross thresholds):
  final padLen = math.sqrt(pdx*pdx + pdy*pdy);
  final pnx = padLen > 1e-6 ? (pdx / padLen) : 0.0;
  final pny = padLen > 1e-6 ? (pdy / padLen) : 0.0;

  // Future-velocity vector & speed
  final vFx = vx;
  final vFy = vyF;
  final vFmag = math.sqrt(vFx*vFx + vFy*vFy) + 1e-9;

  // ---- Quick vertical safety (emergency brake) ----
  double _vCapBrakeUp(double hh) => (0.07 * hh + 6.0).clamp(6.0, 16.0);
  final vCapStrong   = _vCapBrakeUp(h);
  final tooLow       = h < 140.0;
  final tooFastDown  = vyF > math.max(40.0, vCapStrong + 10.0);
  final nearPadLat   = (px - padCx).abs() <= 0.18 * W;
  if (tooLow && tooFastDown && nearPadLat) {
    return intentToIndex(Intent.brakeUp);
  }

  // ---- If centered laterally, manage vertical and lateral braking locally ----
  final padEnter = 0.08 * W;
  final dxNow    = px - padCx;
  if (dxNow.abs() <= padEnter) {
    if (vx >  25.0) return intentToIndex(Intent.brakeRight);
    if (vx < -25.0) return intentToIndex(Intent.brakeLeft);
    return intentToIndex(Intent.descendSlow);
  }

  // ---- Will we cross pad center soon? (x sign flip within τ) ----
  final dxF = xF - padCx;
  final crossesCenter = (dxNow * dxF) < 0.0;

  if (crossesCenter) {
    // Prioritize killing lateral speed so we don’t overshoot the pad.
    if (vx >  12.0) return intentToIndex(Intent.brakeRight);
    if (vx < -12.0) return intentToIndex(Intent.brakeLeft);
    return intentToIndex(Intent.descendSlow);
  }

  // ---- Drifting away from pad? (|dx| growing) -> steer toward pad ----
  final driftingAway = dxF.abs() > dxNow.abs() + 2.0; // small hysteresis
  if (driftingAway) {
    return (dxNow > 0.0) ? intentToIndex(Intent.goLeft)
        : intentToIndex(Intent.goRight);
  }

  // ---- If we have a usable pad vector, compare it with future velocity ----
  // Use signed cross to decide left/right; require meaningful misalignment.
  if (padVecValid) {
    final cp = crossZ(vFx, vFy, pdx, pdy);   // >0: pad is to the LEFT of vF
    final dp = dot   (vFx, vFy, pdx, pdy);

    // Scale thresholds with speed so decisions are stable across regimes.
    // “cpThresh” ~ how far off-axis we tolerate; “dpBad” ~ going mostly away.
    final cpThresh = 0.015 * vFmag * (padLen > 1.0 ? padLen : 1.0); // proportional to |vF| and distance
    final dpBad    = -0.030 * vFmag * (padLen > 1.0 ? padLen : 1.0);

    final misaligned = (cp.abs() > cpThresh) || (dp < dpBad);
    if (misaligned) {
      // If pad lies to the left of our future velocity, yaw/translate LEFT.
      return (cp > 0.0) ? intentToIndex(Intent.goLeft)
          : intentToIndex(Intent.goRight);
    }
  } else {
    // No reliable pad vector — bias to safety: slow descent if getting fast.
    double _vCapHover(double hh) => (0.06 * hh + 6.0).clamp(6.0, 18.0);
    final vCapHover = _vCapHover(h);
    final needUp    = (vy > vCapHover) || (vyF > 0.85 * vCapHover);
    if (needUp) return intentToIndex(Intent.brakeUp);
  }

  // ---- Boundary guard: if moving outward near lateral boundary, nudge inward ----
  final padExit     = 0.14 * W;
  final willExitSoon = (dxF.abs() > padEnter) && (dxNow.abs() <= padEnter);
  final vxIsOutward  = (dxNow.sign == vx.sign) && vx.abs() > 18.0;
  if ((willExitSoon || vxIsOutward) && h > 90.0) {
    return (dxNow >= 0.0) ? intentToIndex(Intent.goLeft)
        : intentToIndex(Intent.goRight);
  }

  // ---- Default: keep a controlled descent ----
  return intentToIndex(Intent.descendSlow);
}

bool _canStrafe(eng.GameEngine env, {double minH = 110.0, double maxVy = 35.0}) {
  final t = env.cfg.t;
  final hasRcs = (t as dynamic);
  final enabled = (hasRcs as dynamic).rcsEnabled ?? false;
  if (!enabled) return false;

  final L = env.lander;
  final gy = env.terrain.heightAt(L.pos.x);
  final h  = (gy - L.pos.y).toDouble();

  if (h <= minH) return false;
  if (L.vel.y.abs() >= maxVy) return false;

  // NOTE: no angle/tilt constraint here — strafing is allowed at any tilt.
  return true;
}

// --- helpers: smooth vertical caps + short-term lookahead --------------------
double _vCapHover(double h)  => (0.06 * h + 6.0).clamp(6.0, 18.0);
double _vCapDesc(double h)   => (0.10 * h + 8.0).clamp(8.0, 26.0);
double _vCapBrakeUp(double h)=> (0.07 * h + 6.0).clamp(6.0, 16.0);

/// Predict near-future vertical speed (px/s) after tauReact without thrust.
double _vyPredictNoThrust(eng.GameEngine env, {double tauReact = 0.35}) {
  final vy = env.lander.vel.y.toDouble();
  final g  = env.cfg.t.gravity;
  return vy + g * tauReact;
}

/// Should we pre-boost now to avoid exceeding cap soon?
bool _needPreBoost({
  required double vCap,
  required eng.GameEngine env,
  double warnFrac = 0.85,      // trigger before cap at 80% of vCap
  double tauReact = 1.0,       // seconds of look-ahead
  double extraPad = 2.5,       // px/s extra safety margin
}) {
  final vyNow   = env.lander.vel.y.toDouble();
  final vyNext  = _vyPredictNoThrust(env, tauReact: tauReact);
  final vWarn   = vCap * warnFrac;
  return (vyNow > vWarn - extraPad) || (vyNext > vWarn);
}

et.ControlInput controllerForIntent(Intent intent, eng.GameEngine env) {
  final L = env.lander;
  final T = env.terrain;
  final px = L.pos.x.toDouble();
  final gy = T.heightAt(px);
  final h  = (gy - L.pos.y).toDouble().clamp(0.0, 1e9);
  final vx = L.vel.x.toDouble();
  final vy = L.vel.y.toDouble();

  // Distance to pad center
  final padCx = T.padCenter.toDouble();
  final dx    = (px - padCx);

  // Height-adaptive vertical caps
  double vCapDesc   = (0.11 * h +  9.0).clamp( 9.0, 24.0);
  double vCapHover  = (0.07 * h +  7.0).clamp( 7.0, 18.0);
  double vCapBrake  = (0.07 * h +  6.0).clamp( 6.0, 16.0);

  // Lateral target profile: far → faster, near → slower
  // |vxt| in [vMin..vMax] scaled by |dx|/W
  final W = env.cfg.worldW.toDouble();
  final far = (dx.abs() / (0.45 * W)).clamp(0.0, 1.0); // 0 at center, 1 at ~0.45W
  final vMin = 12.0, vMax = 45.0;
  final vxt = (vMin + (vMax - vMin) * far) * (dx > 0 ? -1.0 : 1.0); // sign points TOWARD pad

  // Helper: vertical pre-boost check (predict a bit ahead)
  bool needUp(double cap, {double tau = 1.0, double warn = 0.80, double pad = 2.0}) {
    final g = env.cfg.t.gravity;
    final vyNext = vy + g * tau;
    final warnCap = cap * warn;
    return (vy > warnCap - pad) || (vyNext > warnCap);
  }

  // Prefer world-frame RCS if available; if not, tilt+main
  final tcfg = env.cfg.t as dynamic;
  final rcsEnabled   = (tcfg.rcsEnabled ?? false) == true;
  final rcsBodyFrame = (tcfg.rcsBodyFrame ?? true) == true;

  // Common altitude hold for all intents (slightly relaxed while translating)
  bool altHold({double relax = 1.0}) {
    final cap = math.min(vCapDesc * relax, (0.085 * h + 9.0) * relax);
    return needUp(cap, tau: 1.2, warn: 0.80, pad: 2.0);
  }

  // Build outputs
  bool thr=false, left=false, right=false, sL=false, sR=false, dT=false;

  switch (intent) {
    case Intent.brakeUp: {
      final need = vy > vCapBrake || needUp(vCapBrake, tau: 1.35, warn: 0.75, pad: 3.0);
      thr = need;
      break;
    }

    case Intent.descendSlow: {
      final need = vy > vCapDesc || needUp(vCapDesc, tau: 1.3, warn: 0.80, pad: 2.0);
      // Optionally push down if too slow and down-thruster exists
      final downEn = ((tcfg.downThrEnabled ?? false) == true);
      dT = downEn && (vy < 0.55 * vCapDesc);
      thr = need;
      break;
    }

    case Intent.brakeLeft: { // want vxt <= 0 (moving right → brake rightward)
      // Brake rule: if vx > -2, just align; otherwise push right to reduce |vx|
      final allowTranslate = (h > 90.0 && h < 360.0) && (vy < 42.0);
      if (rcsEnabled && !rcsBodyFrame && allowTranslate) {
        // world-frame strafing right to reduce leftward drift
        sL = true; // RCS pushing to the right (your engine: sideLeft => +X if rcsBodyFrame=false)
        thr = altHold(relax: 1.08);
      } else {
        // tilt left or right as needed; while braking, tilt opposite vx
        final wantTiltRight = (vx < -3.0);
        left  = false;
        right = wantTiltRight;
        thr = altHold(relax: wantTiltRight ? 1.10 : 1.0);
      }
      break;
    }

    case Intent.brakeRight: {
      final allowTranslate = (h > 90.0 && h < 360.0) && (vy < 42.0);
      if (rcsEnabled && !rcsBodyFrame && allowTranslate) {
        sR = true; // push left
        thr = altHold(relax: 1.08);
      } else {
        final wantTiltLeft = (vx > 3.0);
        left  = wantTiltLeft;
        right = false;
        thr = altHold(relax: wantTiltLeft ? 1.10 : 1.0);
      }
      break;
    }

    case Intent.goLeft: {
      final allowTranslate = (h > 90.0 && h < 360.0) && (vy < 42.0);
      // Track the target vxt (negative) — aggressively if far
      final err = vx - vxt;
      final strong = far > 0.4 || dx.abs() > 0.18 * W;
      if (rcsEnabled && !rcsBodyFrame && allowTranslate) {
        sR = true; // push left
        thr = altHold(relax: strong ? 1.12 : 1.06);
      } else {
        // no RCS: lean left and keep thrust on to keep altitude while sliding
        left = true;
        thr  = altHold(relax: strong ? 1.12 : 1.06);
      }
      break;
    }

    case Intent.goRight: {
      final allowTranslate = (h > 90.0 && h < 360.0) && (vy < 42.0);
      final err = vx - vxt;
      final strong = far > 0.4 || dx.abs() > 0.18 * W;
      if (rcsEnabled && !rcsBodyFrame && allowTranslate) {
        sL = true; // push right
        thr = altHold(relax: strong ? 1.12 : 1.06);
      } else {
        right = true;
        thr   = altHold(relax: strong ? 1.12 : 1.06);
      }
      break;
    }

    case Intent.hover:
    default: {
      thr = needUp(vCapHover, tau: 1.2, warn: 0.80, pad: 2.0);
      break;
    }
  }

  return et.ControlInput(
    thrust: thr, left: left, right: right,
    sideLeft: sL, sideRight: sR,
    downThrust: dT,
  );
}
/* -------------------------------------------------------------------------- */
/*                               POLICY NETWORK                                */
/* -------------------------------------------------------------------------- */

class ForwardCache {
  final List<double> x;
  final List<List<double>> acts; // a0..aL (hidden activations)
  final List<double> intentLogits;
  final List<double> intentProbs;
  final List<double> turnLogits; // 3 logits
  final double thrLogit;
  final double thrProb;
  final double v; // value head
  ForwardCache({
    required this.x,
    required this.acts,
    required this.intentLogits,
    required this.intentProbs,
    required this.turnLogits,
    required this.thrLogit,
    required this.thrProb,
    required this.v,
  });
}

class PolicyNetwork {
  static const int kIntents = 7;

  final int inputSize;
  final List<int> hidden;
  final nn.MLPTrunk trunk;
  final nn.PolicyHeads heads;

  // ---- Consolidation (L2-SP) anti-forgetting ----
  bool consolidateEnabled = false;
  double consolidateTrunk = 0.0;   // e.g. 1e-3
  double consolidateHeads = 0.0;   // e.g. 5e-4

  // Snapshots (previous-stage weights)
  List<List<List<double>>>? _snapWTrunk;
  List<List<double>>? _snapBTrunk;

  List<List<double>>? _snapWIntent;
  List<double>? _snapBIntent;

  List<List<double>>? _snapWTurn;
  List<double>? _snapBTurn;

  List<List<double>>? _snapWThr;
  List<double>? _snapBThr;

  PolicyNetwork({
    required this.inputSize,
    List<int> hidden = const [64, 64],
    int seed = 0,
  })  : hidden = List<int>.from(hidden),
        trunk  = nn.MLPTrunk(
          inputSize: inputSize,
          hiddenSizes: hidden,
          seed: seed ^ 0x7777,
          activation: nn.Activation.silu, // robust under noise
          trainNoiseStd: 0.02,            // small Gaussian noise on z (train only)
          dropoutProb: 0.10,              // 10% dropout on a (train only)
          trainMode: false,               // will be toggled per episode
        ),
        heads  = nn.PolicyHeads(
          hidden.isEmpty ? inputSize : hidden.last,
          intents: kIntents,
          seed: seed ^ 0x8888,
        ) {
    assert(heads.intent.b.length == kIntents, 'intent head outDim mismatch');
    assert(heads.intent.W.length == kIntents, 'intent head rows mismatch');
  }

  void captureConsolidationAnchor() {
    // trunk
    _snapWTrunk = [
      for (final L in trunk.layers)
        [ for (final row in L.W) List<double>.from(row) ]
    ];
    _snapBTrunk = [
      for (final L in trunk.layers) List<double>.from(L.b)
    ];
    // heads
    _snapWIntent = [ for (final row in heads.intent.W) List<double>.from(row) ];
    _snapBIntent = List<double>.from(heads.intent.b);

    _snapWTurn   = [ for (final row in heads.turn.W)   List<double>.from(row) ];
    _snapBTurn   = List<double>.from(heads.turn.b);

    _snapWThr    = [ for (final row in heads.thr.W)    List<double>.from(row) ];
    _snapBThr    = List<double>.from(heads.thr.b);

    consolidateEnabled = true;
  }

  ForwardCache _forwardFull(List<double> x) {
    final acts = trunk.forwardAll(x);
    final h = acts.last;
    final intentLogits = heads.intent.forward(h);
    final intentProbs  = nn.Ops.softmax(intentLogits);
    final turnLogits   = heads.turn.forward(h);
    final thrLogit     = heads.thr.forward(h)[0];
    final thrProb      = nn.Ops.sigmoid(thrLogit);
    final v            = heads.val.forward(h)[0];

    return ForwardCache(
      x: List<double>.from(x),
      acts: acts,
      intentLogits: intentLogits,
      intentProbs: intentProbs,
      turnLogits: turnLogits,
      thrLogit: thrLogit,
      thrProb: thrProb,
      v: v,
    );
  }

  /// Forward that also returns layer-wise inputs/outputs for Hebbian.
  /// Layer indexing:
  ///   0..trunk.layers.length-1  -> trunk layers (pre=input of layer, post=activation)
  ///   then heads in order: intent, turn, thr, val (post = raw logits/value)
  (List<double> intentLogits, List<double> turnLogits, double thrLogit, double valScalar,
  List<List<double>> preActs, List<List<double>> postActs)
  forwardWithActs(List<double> x) {
    final preActs = <List<double>>[];
    final postActs = <List<double>>[];

    // Trunk
    List<double> h = x;
    for (final layer in trunk.layers) {
      final pre = List<double>.from(h);
      preActs.add(pre);
      final z = List<double>.filled(layer.b.length, 0.0);
      for (int i = 0; i < layer.W.length; i++) {
        double s = layer.b[i];
        final Wi = layer.W[i];
        for (int j = 0; j < Wi.length; j++) s += Wi[j] * h[j];
        z[i] = s;
      }
      final a = List<double>.generate(z.length, (i) => nn.silu(z[i]));
      postActs.add(a);
      h = a;
    }

    // Heads: linear logits/value
    List<double> _head(List<List<double>> W, List<double> b, List<double> inp) {
      final z = List<double>.filled(b.length, 0.0);
      for (int i = 0; i < W.length; i++) {
        double s = b[i];
        final Wi = W[i];
        for (int j = 0; j < Wi.length; j++) s += Wi[j] * inp[j];
        z[i] = s;
      }
      return z;
    }

    final headPre = List<double>.from(h);

    preActs.add(headPre);
    final intentLogits = _head(heads.intent.W, heads.intent.b, h);
    postActs.add(List<double>.from(intentLogits));

    preActs.add(headPre);
    final turnLogits = _head(heads.turn.W, heads.turn.b, h);
    postActs.add(List<double>.from(turnLogits));

    preActs.add(headPre);
    final thrLogit = _head(heads.thr.W, heads.thr.b, h)[0];
    postActs.add([thrLogit]);

    preActs.add(headPre);
    final valScalar = _head(heads.val.W, heads.val.b, h)[0];
    postActs.add([valScalar]);

    return (intentLogits, turnLogits, thrLogit, valScalar, preActs, postActs);
  }

  /// Reward-modulated Hebbian tick.
  /// dW_ij = eta * mod * ( post_i * pre_j  - (useOja ? post_i^2 * W_ij : decay * W_ij) )
  void hebbianTick({
    required HebbianConfig cfg,
    required List<List<double>> preActs,
    required List<List<double>> postActs,
    required double mod,
  }) {
    if (!cfg.enabled) return;
    if (!mod.isFinite) return;

    int li = 0;

    // Trunk
    for (final layer in trunk.layers) {
      if (cfg.trunk) {
        final pre = preActs[li];
        final post = postActs[li];
        _hebbLayerUpdate(layer.W, cfg, pre, post, mod);
      }
      li++;
    }

    // Heads (order matches forwardWithActs appends)
    if (cfg.headIntent) {
      _hebbLayerUpdate(heads.intent.W, cfg, preActs[li], postActs[li], mod);
    }
    li++;

    if (cfg.headTurn) {
      _hebbLayerUpdate(heads.turn.W, cfg, preActs[li], postActs[li], mod);
    }
    li++;

    if (cfg.headThr) {
      _hebbLayerUpdate(heads.thr.W, cfg, preActs[li], postActs[li], mod);
    }
    li++;

    if (cfg.headVal) {
      _hebbLayerUpdate(heads.val.W, cfg, preActs[li], postActs[li], mod);
    }
  }

  void _hebbLayerUpdate(
      List<List<double>> W,
      HebbianConfig cfg,
      List<double> pre,
      List<double> post,
      double mod,
      ) {
    final m = mod;
    final eta = cfg.eta;
    final clip = cfg.clip;
    final decay = cfg.decay;

    for (int i = 0; i < W.length; i++) {
      final Wi = W[i];
      final pi = post[i];
      final stabFactor = cfg.useOja ? (pi * pi) : decay; // Oja vs decay
      for (int j = 0; j < Wi.length; j++) {
        final hebb = pi * pre[j];
        final stab = stabFactor * Wi[j];
        final dw = eta * m * (hebb - stab);
        Wi[j] += _clipD(dw, clip);
      }
      if (cfg.rowL2Cap > 0) _capRowL2(Wi, cfg.rowL2Cap);
    }
  }

  (int, List<double>, ForwardCache) actIntentGreedy(List<double> x) {
    final c = _forwardFull(x);
    int arg = 0;
    double best = c.intentProbs[0];
    for (int i = 1; i < c.intentProbs.length; i++) {
      if (c.intentProbs[i] > best) { best = c.intentProbs[i]; arg = i; }
    }
    return (arg, c.intentProbs, c);
  }

  (bool, bool, bool, List<double>, ForwardCache) actGreedy(List<double> x) {
    final c = _forwardFull(x);
    final thrust = c.thrProb >= 0.5;
    int tArg = 0;
    double best = c.turnLogits[0];
    for (int i = 1; i < 3; i++) {
      if (c.turnLogits[i] > best) { best = c.turnLogits[i]; tArg = i; }
    }
    final left = (tArg == 0);
    final right = (tArg == 2);
    final probs = <double>[ c.thrProb, ...nn.Ops.softmax(c.turnLogits) ];
    return (thrust, left, right, probs, c);
  }

  void updateFromEpisode({
    required List<ForwardCache> decisionCaches,     // for intent head
    required List<int> intentChoices,              // chosen intents
    required List<double> decisionReturns,         // advantages (signed)
    required List<int> alignLabels,                // teacher labels for intent
    required double alignWeight,                   // CE supervision
    required double intentPgWeight,                // PG on intent
    required double lr,
    required double entropyBeta,
    required double valueBeta,
    required double huberDelta,
    required bool intentMode,

    List<ForwardCache>? actionCaches,              // action supervision (turn/thr)
    List<int>? actionTurnTargets,
    List<bool>? actionThrustTargets,
    double actionAlignWeight = 0.0,
  }) {
    final int H = trunk.layers.isEmpty ? inputSize : trunk.layers.last.b.length;
    assert(heads.intent.b.length > 0 && heads.intent.W.isNotEmpty,
    'intent head not initialized');
    assert(heads.turn.b.length == 3 && heads.turn.W.length == 3,
    'turn head must have 3 logits');
    assert(heads.thr.b.length == 1 && heads.thr.W.length == 1,
    'thrust head must have 1 logit');
    assert(heads.intent.W[0].length == H &&
        heads.turn.W[0].length   == H &&
        heads.thr.W[0].length    == H,
    'head input dims must match trunk output');

    double _clipGrad(double g, [double c = 1.0]) {
      if (!g.isFinite) return 0.0;
      if (g > c) return c;
      if (g < -c) return -c;
      return g;
    }

    // Trunk accumulators
    List<List<List<double>>> gW_trunk = [
      for (final lin in trunk.layers)
        List.generate(lin.W.length, (_) => List<double>.filled(lin.W[0].length, 0.0))
    ];
    List<List<double>> gb_trunk = [
      for (final lin in trunk.layers) List<double>.filled(lin.b.length, 0.0)
    ];

    // Heads grads
    final gW_int = List.generate(heads.intent.W.length,
            (_) => List<double>.filled(heads.intent.W[0].length, 0.0));
    final gb_int = List<double>.filled(heads.intent.b.length, 0.0);

    final gW_turn = List.generate(heads.turn.W.length,
            (_) => List<double>.filled(heads.turn.W[0].length, 0.0));
    final gb_turn = List<double>.filled(heads.turn.b.length, 0.0);

    final gW_thr = List.generate(heads.thr.W.length,
            (_) => List<double>.filled(heads.thr.W[0].length, 0.0));
    final gb_thr = List<double>.filled(heads.thr.b.length, 0.0);

    // ----- Intent CE (optional) -----
    if (intentMode && alignWeight > 0 && decisionCaches.isNotEmpty) {
      final N = decisionCaches.length;
      for (int n = 0; n < N; n++) {
        final c = decisionCaches[n];
        final h = c.acts.last;
        final y = alignLabels[n].clamp(0, PolicyNetwork.kIntents - 1);
        final dLog = nn.Ops.crossEntropyGrad(c.intentProbs, y); // (p - y)
        // Entropy grad wrt logits: dH = -β * ( (1 + log p) - ⟨1 + log p⟩ ), but we can use a simpler:
        // ∂( -Σ p log p )/∂logit = β * (p * (log p + 1) - const).
        // A simple practical version is to add β * p to the gradient (pushes toward uniform).
        if (entropyBeta > 0) {
          for (int i = 0; i < dLog.length; i++) {
            dLog[i] += entropyBeta * c.intentProbs[i]; // small push to spread mass
          }
        }
        final dH = heads.intent.backward(x: h, dOut: dLog, gW: gW_int, gb: gb_int);
        trunk.backwardFromTopGrad(
          dTop: dH, acts: c.acts, gW: gW_trunk, gb: gb_trunk, x0: c.x,
        );
      }
      final scale = lr * alignWeight / N;
      for (int i = 0; i < heads.intent.b.length; i++) {
        heads.intent.b[i] -= _clipGrad(scale * gb_int[i]);
        for (int j = 0; j < heads.intent.W[0].length; j++) {
          heads.intent.W[i][j] -= _clipGrad(scale * gW_int[i][j]);
        }
      }
      // zero CE accumulators so PG doesn't double-apply trunk grads
      for (int i = 0; i < gb_int.length; i++) gb_int[i] = 0.0;
      for (int i = 0; i < gW_int.length; i++) {
        for (int j = 0; j < gW_int[0].length; j++) gW_int[i][j] = 0.0;
      }
      for (int li = 0; li < gb_trunk.length; li++) {
        for (int j = 0; j < gb_trunk[li].length; j++) gb_trunk[li][j] = 0.0;
        for (int r = 0; r < gW_trunk[li].length; r++) {
          for (int c2 = 0; c2 < gW_trunk[li][0].length; c2++) gW_trunk[li][r][c2] = 0.0;
        }
      }
    }

    // ----- Intent PG (REINFORCE with advantage) -----
    if (intentPgWeight > 0 && decisionCaches.isNotEmpty) {
      final N = decisionCaches.length;
      for (int n = 0; n < N; n++) {
        final c = decisionCaches[n];
        final h = c.acts.last;
        final chosen = intentChoices[n].clamp(0, PolicyNetwork.kIntents - 1);
        final adv = decisionReturns[n]; // signed advantage
        final dLog = List<double>.generate(
          c.intentProbs.length,
              (i) => (c.intentProbs[i] - (i == chosen ? 1.0 : 0.0)) * (-adv),
        );
        final dH = heads.intent.backward(x: h, dOut: dLog, gW: gW_int, gb: gb_int);
        trunk.backwardFromTopGrad(
          dTop: dH, acts: c.acts, gW: gW_trunk, gb: gb_trunk, x0: c.x,
        );
      }

      final scale = lr * intentPgWeight / decisionCaches.length;
      for (int i = 0; i < heads.intent.b.length; i++) {
        heads.intent.b[i] -= _clipGrad(scale * gb_int[i]);
        for (int j = 0; j < heads.intent.W[0].length; j++) {
          heads.intent.W[i][j] -= _clipGrad(scale * gW_int[i][j]);
        }
      }
    }

    // ----- Action supervision (optional) -----
    final hasAction = actionAlignWeight > 0.0 &&
        actionCaches != null &&
        actionTurnTargets != null &&
        actionThrustTargets != null &&
        actionCaches.isNotEmpty;

    if (hasAction) {
      final M = actionCaches!.length;
      double meanThrLogit = 0.0;
      double teacherThrRate = 0.0;

      for (int n = 0; n < M; n++) {
        final c = actionCaches![n];
        final h = c.acts.last;

        final turnProbs = nn.Ops.softmax(c.turnLogits);
        final yt = actionTurnTargets![n].clamp(0, 2);
        final dTurn = nn.Ops.crossEntropyGrad(turnProbs, yt, numClasses: 3);

        final yb = actionThrustTargets![n] ? 1.0 : 0.0;
        final dThr = nn.Ops.bceGradFromLogit(c.thrLogit, yb);
        teacherThrRate += yb;

        final dH_turn = heads.turn.backward(x: h, dOut: dTurn, gW: gW_turn, gb: gb_turn);
        final dH_thr  = heads.thr.backward (x: h, dOut: [dThr], gW: gW_thr , gb: gb_thr );

        final dH = List<double>.filled(h.length, 0.0);
        for (int i = 0; i < h.length; i++) dH[i] = dH_turn[i] + dH_thr[i];

        trunk.backwardFromTopGrad(
          dTop: dH, acts: c.acts, gW: gW_trunk, gb: gb_trunk, x0: c.x,
        );

        meanThrLogit += c.thrLogit;
      }

      final scale = lr * actionAlignWeight / actionCaches.length;
      for (int i = 0; i < heads.turn.b.length; i++) {
        heads.turn.b[i] -= _clipGrad(scale * gb_turn[i]);
        for (int j = 0; j < heads.turn.W[0].length; j++) {
          heads.turn.W[i][j] -= _clipGrad(scale * gW_turn[i][j]);
        }
      }
      heads.thr.b[0] -= _clipGrad(scale * gb_thr[0]);
      for (int j = 0; j < heads.thr.W[0].length; j++) {
        heads.thr.W[0][j] -= _clipGrad(scale * gW_thr[0][j]);
      }

      // quick thrust bias calibration
      meanThrLogit /= actionCaches.length;
      teacherThrRate /= actionCaches.length;
      final logitTarget = nn.Ops.logit(teacherThrRate);
      final calibStep = 0.25;
      heads.thr.b[0] += (logitTarget - meanThrLogit) * calibStep;
    }

    // ===================== L2-SP consolidation (optional) =====================
    if (consolidateEnabled) {
      // Trunk penalty: add dL/dW += λ (W - W0), dL/db += λ (b - b0)
      if (consolidateTrunk > 0.0 && _snapWTrunk != null && _snapBTrunk != null) {
        for (int li = 0; li < trunk.layers.length; li++) {
          final L = trunk.layers[li];
          final W0 = _snapWTrunk![li];
          final b0 = _snapBTrunk![li];
          for (int i = 0; i < L.W.length; i++) {
            for (int j = 0; j < L.W[0].length; j++) {
              gW_trunk[li][i][j] += consolidateTrunk * (L.W[i][j] - W0[i][j]);
            }
            gb_trunk[li][i] += consolidateTrunk * (L.b[i] - b0[i]);
          }
        }
      }
      // Heads penalty (usually smaller λ)
      if (consolidateHeads > 0.0) {
        if (_snapWIntent != null && _snapBIntent != null) {
          for (int i = 0; i < heads.intent.W.length; i++) {
            for (int j = 0; j < heads.intent.W[0].length; j++) {
              gW_int[i][j] += consolidateHeads * (heads.intent.W[i][j] - _snapWIntent![i][j]);
            }
            gb_int[i] += consolidateHeads * (heads.intent.b[i] - _snapBIntent![i]);
          }
        }
        if (_snapWTurn != null && _snapBTurn != null) {
          for (int i = 0; i < heads.turn.W.length; i++) {
            for (int j = 0; j < heads.turn.W[0].length; j++) {
              gW_turn[i][j] += consolidateHeads * (heads.turn.W[i][j] - _snapWTurn![i][j]);
            }
            gb_turn[i] += consolidateHeads * (heads.turn.b[i] - _snapBTurn![i]);
          }
        }
        if (_snapWThr != null && _snapBThr != null) {
          for (int j = 0; j < heads.thr.W[0].length; j++) {
            gW_thr[0][j] += consolidateHeads * (heads.thr.W[0][j] - _snapWThr![0][j]);
          }
          gb_thr[0] += consolidateHeads * (heads.thr.b[0] - _snapBThr![0]);
        }
      }
    }
    // =================== end consolidation (optional) ========================

    // ----- Apply trunk update (shared grads) -----
    final trunkScale = lr;
    for (int li = 0; li < trunk.layers.length; li++) {
      final L = trunk.layers[li];
      final gb = gb_trunk[li];
      final gW = gW_trunk[li];
      for (int i = 0; i < L.b.length; i++) L.b[i] -= _clipGrad(trunkScale * gb[i]);
      for (int i = 0; i < L.W.length; i++) {
        for (int j = 0; j < L.W[0].length; j++) {
          L.W[i][j] -= _clipGrad(trunkScale * gW[i][j]);
        }
      }
    }
  }
}

/* -------------------------------------------------------------------------- */
/*                                   TRAINER                                  */
/* -------------------------------------------------------------------------- */

class EpisodeResult {
  final int steps;
  final double totalCost;
  final bool landed;
  final double segMean; // here: mean PF reward (for logging/gating)
  EpisodeResult({
    required this.steps,
    required this.totalCost,
    required this.landed,
    required this.segMean,
  });
}

// external dense reward hook (potential-field reward; velocity-only)
typedef ExternalRewardHook = double Function({
required eng.GameEngine env,
required double dt,
required int tStep,
});

class _Ema {
  double m; final double a;
  _Ema({this.a = 0.98, double init = 0.0}) : m = init;
  double update(double x) { m = a * m + (1 - a) * x; return m; }
}

double _entropyFromLogits(List<double> logits) {
  // softmax -> probs in a numerically stable way
  final maxZ = logits.reduce((a,b) => a > b ? a : b);
  var sum = 0.0;
  final exps = List<double>.filled(logits.length, 0.0);
  for (int i = 0; i < logits.length; i++) { exps[i] = math.exp(logits[i] - maxZ); sum += exps[i]; }
  double H = 0.0;
  for (final e in exps) {
    final p = e / sum;
    if (p > 0) H += -p * math.log(p);
  }
  return H;
}

class Trainer {
  final eng.GameEngine env;
  final FeatureExtractorRays fe;
  final PolicyNetwork policy;
  final double dt;
  final double gamma;
  final int seed;
  final bool twoStage;
  final int planHold;
  final double tempIntent;
  final double intentEntropyBeta;
  final bool useLearnedController;
  final double blendPolicy; // probability-space blend for thrust
  final double intentAlignWeight;
  final double intentPgWeight;
  final double actionAlignWeight;
  final bool normalizeFeatures;

  // gating/logging
  final double gateScoreMin;
  final bool gateOnlyLanded;
  final bool gateVerbose;

  // probabilistic gate knobs (NEW)
  final bool gateProbEnabled;
  final double gateProbK;
  final double gateProbMin;
  final double gateProbMax;
  final double gateProbLandedBoost;
  final double gateProbNearPadBoost;

  // accept near-pad crashes (configurable)
  final bool gateAcceptNearPadCrashes;
  final double gatePadFrac;
  final double gatePadHeight;
  final double gateMaxImpactSpeed;

  // align teacher label polarity with pad rays
  final bool alignPolarityWithPadRays;

  final RunningNorm? norm;
  int _epCounter = 0;

  // PWM thrust state
  double _pwmA = 0.0;
  int _pwmCount = 0;
  int _pwmOn = 0;

  // external per-step reward hook (PF)
  final ExternalRewardHook? externalRewardHook;

  // Hebbian plasticity config + guards
  final HebbianConfig hebbian;
  final double hebbModGain;        // scales centered reward
  final double hebbModAbsClip;     // absolute clip for modulation
  final double minIntentEntropy;   // skip head Hebbian below this
  final int maxSameIntentRun;      // skip head Hebbian if repeating too long
  final _Ema _modEma = _Ema(a: 0.98);
  int _sameIntentRun = 0;
  int _lastIntent = -1;

  Trainer({
    required this.env,
    required this.fe,
    required this.policy,
    required this.dt,
    required this.gamma,
    required this.seed,
    required this.twoStage,
    required this.planHold,
    required this.tempIntent,
    required this.intentEntropyBeta,
    required this.useLearnedController,
    required this.blendPolicy,
    required this.intentAlignWeight,
    this.intentPgWeight = 0.6,
    required this.actionAlignWeight,
    required this.normalizeFeatures,

    this.gateScoreMin = -double.infinity,
    this.gateOnlyLanded = false,
    this.gateVerbose = true,

    // probabilistic gate defaults (enabled by default -> set false to get hard gate)
    this.gateProbEnabled = true,
    this.gateProbK = 8.0,
    this.gateProbMin = 0.05,
    this.gateProbMax = 0.95,
    this.gateProbLandedBoost = 0.15,
    this.gateProbNearPadBoost = 0.10,

    // near-pad crash acceptance
    this.gateAcceptNearPadCrashes = false,
    this.gatePadFrac = 0.12,
    this.gatePadHeight = 160.0,
    this.gateMaxImpactSpeed = 60.0,

    // training-time polarity alignment
    this.alignPolarityWithPadRays = true,

    this.externalRewardHook,

    // Hebbian + guards
    HebbianConfig? hebbian,
    this.hebbModGain = 0.6,
    this.hebbModAbsClip = 1.5,
    this.minIntentEntropy = 0.5, // nats
    this.maxSameIntentRun = 48,  // ~0.8s @60Hz
  }) : hebbian = hebbian ?? const HebbianConfig(),
        norm = RunningNorm(fe.inputSize, momentum: 0.995);

  int _sampleCategorical(List<double> probs, math.Random r, double temp) {
    if (temp <= 1e-6) {
      int arg = 0; double best = probs[0];
      for (int i = 1; i < probs.length; i++) if (probs[i] > best) { best = probs[i]; arg = i; }
      return arg;
    }
    final z = probs.map((p) => math.log(_clip(p, 1e-12, 1.0))).toList();
    for (int i = 0; i < z.length; i++) z[i] /= temp;
    final sm = nn.Ops.softmax(z);
    final u = r.nextDouble();
    double acc = 0.0;
    for (int i = 0; i < sm.length; i++) {
      acc += sm[i];
      if (u <= acc) return i;
    }
    return sm.length - 1;
  }

  EpisodeResult runEpisode({
    required bool train,
    required bool greedy,
    required bool scoreIsReward,   // kept for API compat; unused
    double lr = 3e-4,
    double valueBeta = 0.5,
    double huberDelta = 1.0,
  }) {
    // Toggle trunk noise/dropout depending on train/eval
    policy.trunk.trainMode = train;

    final r = math.Random(seed ^ (_epCounter++));
    final decisionRewards = <double>[];  // per-decision reward (PF only)

    final actionCaches = <ForwardCache>[];
    final actionTurnTargets = <int>[];
    final actionThrustTargets = <bool>[];

    final decisionCaches = <ForwardCache>[];
    final intentChoices = <int>[];
    final decisionReturns = <double>[];
    final alignLabels = <int>[];

    env.reset(seed: r.nextInt(1 << 30));
    double totalCost = 0.0;

    // PF logging
    double segSum = 0.0;
    int segCount = 0;

    _pwmA = 0.0; _pwmCount = 0; _pwmOn = 0;

    int framesLeft = 0;
    int currentIntentIdx = 0;

    int steps = 0;
    bool landed = false;

    // accumulate PF reward until next decision boundary
    double pfAcc = 0.0;

    while (true) {
      if (framesLeft <= 0) {
        var x = fe.extract(lander: env.lander, terrain: env.terrain, worldW: env.cfg.worldW, worldH: env.cfg.worldH, rays: env.rays);
        int yTeacher = predictiveIntentLabelAdaptive(
          env, baseTauSec: 1.0, minTauSec: 0.45, maxTauSec: 1.35,
        );

        // flip teacher left/right if pad-ray polarity disagrees
        if (alignPolarityWithPadRays) {
          final L = env.lander;
          final rays = env.rays;
          if (rays.isNotEmpty) {
            final av = _avgPadVectorFromRays(rays: rays, px: L.pos.x.toDouble(), py: L.pos.y.toDouble());
            if (av.valid) {
              final padIsLeft = av.x < 0.0;
              if (yTeacher == intentToIndex(Intent.goLeft) && !padIsLeft) {
                yTeacher = intentToIndex(Intent.goRight);
              } else if (yTeacher == intentToIndex(Intent.goRight) && padIsLeft) {
                yTeacher = intentToIndex(Intent.goLeft);
              }
            }
          }
        }

        if (normalizeFeatures) {
          norm?.observe(x);
          x = norm?.normalize(x, update: false) ?? x;
        }

        final (idxGreedy, p, cache) = policy.actIntentGreedy(x);
        // Optional tiny exploration (disabled by default):
        // final eps = 0.03;
        // final smoothed = List<double>.generate(p.length, (i) => (1 - eps) * p[i] + eps / p.length);
        // final idx = greedy ? idxGreedy : _sampleCategorical(smoothed, r, tempIntent);
        final idx = greedy ? idxGreedy : _sampleCategorical(p, r, tempIntent);
        currentIntentIdx = idx;

        // repetition tracking for lock guard
        if (_lastIntent == currentIntentIdx) {
          _sameIntentRun++;
        } else {
          _sameIntentRun = 0;
          _lastIntent = currentIntentIdx;
        }

        if (train) {
          // PF-only: push the accumulated PF reward for the last window
          decisionCaches.add(cache);
          intentChoices.add(idx);
          alignLabels.add(yTeacher);
          decisionRewards.add(pfAcc);
          pfAcc = 0.0;
        }

        // compute discounted returns (advantages)
        if (train && decisionCaches.isNotEmpty) {
          final T = decisionRewards.length;
          if (T != decisionCaches.length ||
              T != intentChoices.length ||
              T != alignLabels.length) {
            throw StateError('length mismatch in decision arrays');
          }

          final tmp = List<double>.filled(T, 0.0);
          double G = 0.0;
          for (int i = T - 1; i >= 0; i--) {
            G = decisionRewards[i] + gamma * G;
            tmp[i] = G;
          }

          double mean = 0.0;
          for (final v in tmp) mean += v;
          mean /= T;

          double var0 = 0.0;
          for (final v in tmp) { final d = v - mean; var0 += d * d; }
          var0 = (var0 / T).clamp(1e-9, double.infinity);
          final std = math.sqrt(var0);

          decisionReturns.clear();
          for (final v in tmp) decisionReturns.add((v - mean) / std);
        }

        // adaptive plan hold
        final padCx = env.terrain.padCenter.toDouble();
        final dxAbs = (env.lander.pos.x.toDouble() - padCx).abs();
        final vxAbs = env.lander.vel.x.toDouble().abs();
        final gy = env.terrain.heightAt(env.lander.pos.x.toDouble());
        final h = (gy - env.lander.pos.y).toDouble();
        final W = env.cfg.worldW.toDouble();

        int dynHold = 1;
        if (dxAbs > 0.12 * W || vxAbs > 60.0) dynHold = 1;
        if (dynHold == 1 && h > 320.0 && dxAbs < 0.04 * W && vxAbs < 25.0) dynHold = 2;
        framesLeft = dynHold;
      }

      final intent = indexToIntent(currentIntentIdx);
      final uTeacher = controllerForIntent(intent, env);

      var xAct = fe.extract(lander: env.lander, terrain: env.terrain, worldW: env.cfg.worldW, worldH: env.cfg.worldH, rays: env.rays);
      xAct = norm?.normalize(xAct, update: false) ?? xAct;

      // Hebbian-aware forward (for activations)
      final fw = policy.forwardWithActs(xAct);
      // Also keep greedy path for caches/UI:
      final (thBool, lf, rt, probs, cAct) = policy.actGreedy(xAct);

      final groundY = env.terrain.heightAt(env.lander.pos.x);
      final height  = (groundY - env.lander.pos.y).toDouble();

      final pThrModel   = probs[0].clamp(0.0, 1.0);
      final pThrTeacher = uTeacher.thrust ? 1.0 : 0.0;
      final pThrExec = blendPolicy * pThrModel + (1.0 - blendPolicy) * pThrTeacher;

      _pwmA = (_pwmA + pThrExec).clamp(0.0, 10.0);

      bool thrustPWM = false;
      while (_pwmA >= 1.0) {
        thrustPWM = true;
        _pwmA -= 1.0;
      }
      if (height < 90.0 && !thrustPWM && pThrExec > 0.65) {
        thrustPWM = true;
        _pwmA = (_pwmA - 0.65).clamp(0.0, 0.999);
      }

      final execThrust = thrustPWM;
      final execLeft   = useLearnedController ? lf : uTeacher.left;
      final execRight  = useLearnedController ? rt : uTeacher.right;

      // pass teacher’s strafing & down-thrust to the environment
      final execSideLeft  = uTeacher.sideLeft;
      final execSideRight = uTeacher.sideRight;
      final execDown      = uTeacher.downThrust;

      // ----- PF-only reward accumulation / modulation -----
      final r_pf = externalRewardHook?.call(env: env, dt: dt, tStep: steps) ?? 0.0;
      pfAcc += r_pf;
      segSum += r_pf; // for logging/gating
      segCount++;

      // --- Reward-modulated Hebbian tick (centered, clipped, guarded) ---
      if (train && hebbian.enabled) {
        // Center & scale the reward, then hard-clip
        final centered = r_pf - _modEma.update(r_pf);
        double mod = centered * hebbModGain;
        if (mod >  hebbModAbsClip) mod =  hebbModAbsClip;
        if (mod < -hebbModAbsClip) mod = -hebbModAbsClip;

        if (mod != 0.0) {
          final intentEntropy = _entropyFromLogits(fw.$1); // fw.$1 = intentLogits
          final skipHeadsForLock = (_sameIntentRun >= maxSameIntentRun) || (intentEntropy < minIntentEntropy);

          final cfg = skipHeadsForLock
              ? HebbianConfig(
            enabled: true,
            useOja: hebbian.useOja,
            eta: hebbian.eta,
            decay: hebbian.decay,
            clip: hebbian.clip,
            rowL2Cap: hebbian.rowL2Cap,
            trunk: hebbian.trunk,
            headIntent: false,
            headTurn: false,
            headThr: false,
            headVal: false,
          )
              : hebbian;

          policy.hebbianTick(
            cfg: cfg,
            preActs: fw.$5,   // pre
            postActs: fw.$6,  // post
            mod: mod,
          );
        }
      }

      actionCaches.add(cAct);
      actionTurnTargets.add(uTeacher.left ? 0 : (uTeacher.right ? 2 : 1));
      actionThrustTargets.add(uTeacher.thrust);

      _pwmCount++; if (execThrust) _pwmOn++;
      if ((_pwmCount % 240) == 0) {
        _pwmCount = 0; _pwmOn = 0;
      }

      final info = env.step(dt, et.ControlInput(
        thrust: execThrust,
        left: execLeft,
        right: execRight,
        sideLeft: execSideLeft,
        sideRight: execSideRight,
        downThrust: execDown,
        intentIdx: currentIntentIdx,
      ));
      totalCost += info.costDelta;
      steps++;
      framesLeft--;

      if (info.terminal) {
        landed = env.status == et.GameStatus.landed;

        // push any remaining PF reward into the final decision window
        if (train && decisionRewards.isNotEmpty && pfAcc.abs() > 0) {
          decisionRewards[decisionRewards.length - 1] += pfAcc;
          pfAcc = 0.0;
        }
        break;
      }
      if (steps > 5000) break;
    }

    final segMean = (segCount > 0) ? (segSum / segCount) : 0.0;

    // ---- GATED UPDATE + LOGGING ----
    bool nearPadCrashOK = false;
    if (!landed && env.status == et.GameStatus.crashed && gateAcceptNearPadCrashes) {
      final L = env.lander;
      final W = env.cfg.worldW.toDouble();
      final padCx = env.terrain.padCenter.toDouble();
      final dxAbs = (L.pos.x.toDouble() - padCx).abs();
      final gy = env.terrain.heightAt(L.pos.x.toDouble());
      final h  = (gy - L.pos.y).toDouble().clamp(0.0, double.infinity);
      final speed = math.sqrt(L.vel.x * L.vel.x + L.vel.y * L.vel.y).toDouble();
      nearPadCrashOK =
          (dxAbs <= gatePadFrac * W) &&
              (h    <= gatePadHeight) &&
              (speed <= gateMaxImpactSpeed);
    }

    // Probabilistic gate (with hard-landed constraint if requested)
    bool accept;
    double sampledP = 1.0; // for logging
    if (gateOnlyLanded && !landed && !nearPadCrashOK) {
      accept = false;
      sampledP = 0.0;
    } else if (!gateProbEnabled) {
      // Legacy hard gate
      accept = segMean >= gateScoreMin;
      sampledP = accept ? 1.0 : 0.0;
    } else {
      // p = min + (max-min) * sigmoid(k*(segMean - gateScoreMin))
      final z = gateProbK * (segMean - gateScoreMin);
      final s = 1.0 / (1.0 + math.exp(-z));
      double p = gateProbMin + (gateProbMax - gateProbMin) * s;
      if (landed)         p = (p + gateProbLandedBoost ).clamp(gateProbMin, gateProbMax);
      if (nearPadCrashOK) p = (p + gateProbNearPadBoost).clamp(gateProbMin, gateProbMax);
      sampledP = p;
      accept = (math.Random(seed ^ (_epCounter << 7) ^ steps).nextDouble() < p);
    }

    if (gateVerbose) {
      final tag = landed ? 'L' : (nearPadCrashOK ? 'NP' : 'NL');
      final decisionMsg = accept ? 'ACCEPT' : 'DROP';
      print('[GATE] segMean=${segMean.toStringAsFixed(3)} '
          'thr=${gateScoreMin.toStringAsFixed(3)} '
          'p=${sampledP.toStringAsFixed(3)} '
          'result=$decisionMsg [$tag] steps=$steps');
    }

    if (train) {
      if (accept && (decisionCaches.isNotEmpty || actionCaches.isNotEmpty)) {
        policy.updateFromEpisode(
          decisionCaches: decisionCaches,
          intentChoices: intentChoices,
          decisionReturns: decisionReturns,   // advantages from PF-only rewards
          alignLabels: alignLabels,
          alignWeight: intentAlignWeight,
          intentPgWeight: intentPgWeight,
          lr: lr,
          entropyBeta: 0.0,
          valueBeta: valueBeta,
          huberDelta: huberDelta,
          intentMode: true,
          actionCaches: actionCaches,
          actionTurnTargets: actionTurnTargets,
          actionThrustTargets: actionThrustTargets,
          actionAlignWeight: actionAlignWeight,
        );
      }
    }

    return EpisodeResult(steps: steps, totalCost: totalCost, landed: landed, segMean: segMean);
  }
}
