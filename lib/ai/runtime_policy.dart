// lib/ai/runtime_policy.dart
import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/services.dart' show rootBundle;

// UI types
import 'package:flutter/material.dart' show Offset;
import '../engine/game_engine.dart';
import '../engine/types.dart' show LanderState, Terrain, RayHitKind;
import '../engine/raycast.dart' show RayHit, RayHitKind; // for ray-based FE

// Intent bus (runtime)
import 'intent_bus.dart';
// Plan bus (runtime overlay)
import 'plan_bus.dart';
import 'potential_field.dart' as pf;

/* =============================================================================
   Physics bundle loaded from policy JSON (used by runtime controls)
   ========================================================================== */

class RuntimePhysics {
  final double gravity;
  final double thrustAccel;
  final double rotSpeed;

  // Side thrusters (RCS)
  final bool rcsEnabled;
  final double rcsAccel;
  final bool rcsBodyFrame;

  // Downward (belly) thruster
  final bool downThrEnabled;
  final double downThrAccel;
  final double downThrBurn;

  const RuntimePhysics({
    this.gravity = 0.18,
    this.thrustAccel = 0.42,
    this.rotSpeed = 1.6,
    this.rcsEnabled = false,
    this.rcsAccel = 0.12,
    this.rcsBodyFrame = true,
    this.downThrEnabled = false,
    this.downThrAccel = 0.30,
    this.downThrBurn = 10.0,
  });

  static RuntimePhysics fromJsonMap(Map<String, dynamic>? m) {
    if (m == null) return const RuntimePhysics();
    double _d(String k, double dflt) {
      final v = m[k];
      if (v is num) return v.toDouble();
      return dflt;
    }
    bool _b(String k, bool dflt) {
      final v = m[k];
      if (v is bool) return v;
      return dflt;
    }
    return RuntimePhysics(
      gravity        : _d('gravity', 0.18),
      thrustAccel    : _d('thrustAccel', 0.42),
      rotSpeed       : _d('rotSpeed', 1.6),
      rcsEnabled     : _b('rcsEnabled', false),
      rcsAccel       : _d('rcsAccel', 0.12),
      rcsBodyFrame   : _b('rcsBodyFrame', true),
      downThrEnabled : _b('downThrEnabled', false),
      downThrAccel   : _d('downThrAccel', 0.30),
      downThrBurn    : _d('downThrBurn', 10.0),
    );
  }

  Map<String, dynamic> toJson() => {
    'gravity'        : gravity,
    'thrustAccel'    : thrustAccel,
    'rotSpeed'       : rotSpeed,
    'rcsEnabled'     : rcsEnabled,
    'rcsAccel'       : rcsAccel,
    'rcsBodyFrame'   : rcsBodyFrame,
    'downThrEnabled' : downThrEnabled,
    'downThrAccel'   : downThrAccel,
    'downThrBurn'    : downThrBurn,
  };
}

/* =============================================================================
   Feature extractors
   ========================================================================== */

abstract class _RuntimeFE {
  int get inputSize;
  List<double> extract({
    required LanderState lander,
    required Terrain terrain,
    required double worldW,
    required double worldH,
    List<RayHit>? rays,
    double uiMaxFuel = 100.0,
  });
}

/// -------- Legacy terrain/pad/ground-samples (back-compat) --------
class _RuntimeFE_Legacy implements _RuntimeFE {
  final int groundSamples;
  final double stridePx;
  const _RuntimeFE_Legacy({this.groundSamples = 3, this.stridePx = 48});

  @override
  int get inputSize => 10 + groundSamples;

  @override
  List<double> extract({
    required LanderState lander,
    required Terrain terrain,
    required double worldW,
    required double worldH,
    List<RayHit>? rays, // unused in legacy mode
    double uiMaxFuel = 100.0,
  }) {
    final pxAbs = lander.pos.x;
    final pyAbs = lander.pos.y;

    final px = (pxAbs / worldW);
    final py = (pyAbs / worldH);
    final vx = (lander.vel.x / 200.0).clamp(-3.0, 3.0);
    final vy = (lander.vel.y / 200.0).clamp(-3.0, 3.0);
    final ang = (lander.angle / math.pi).clamp(-2.0, 2.0);
    final fuel = (lander.fuel / (uiMaxFuel > 0 ? uiMaxFuel : 1.0)).clamp(0.0, 1.0);

    final padCenterAbs = (terrain.padX1 + terrain.padX2) * 0.5;
    final padCenter = (padCenterAbs / worldW);
    final dxToPad = ((pxAbs - padCenterAbs) / (0.5 * worldW)).clamp(-1.5, 1.5);

    final gyCenter = terrain.heightAt(pxAbs);
    final hAbove = ((gyCenter - pyAbs) / 300.0).clamp(-2.0, 2.0);

    final gyL = terrain.heightAt((pxAbs - 8.0).clamp(0.0, worldW));
    final gyR = terrain.heightAt((pxAbs + 8.0).clamp(0.0, worldW));
    final slope = (((gyR - gyL) / 16.0) / 2.0).clamp(-2.0, 2.0);

    final samples = <double>[];
    final n = groundSamples;
    final centerIdx = n ~/ 2;
    for (int i = 0; i < n; i++) {
      final rel = (i - centerIdx).toDouble();
      final sx = (pxAbs + rel * stridePx).clamp(0.0, worldW);
      final sy = terrain.heightAt(sx);
      samples.add(((sy - pyAbs) / 300.0).clamp(-2.0, 2.0));
    }

    return [px, py, vx, vy, ang, fuel, padCenter, dxToPad, hAbove, slope, ...samples];
  }
}

/// -------- Ray-based FE (MATCHES training FeatureExtractorRays) --------
class _RuntimeFE_Rays implements _RuntimeFE {
  final int rayCount;
  final bool kindsOneHot;
  const _RuntimeFE_Rays({required this.rayCount, required this.kindsOneHot});

  @override
  int get inputSize => 5 + rayCount * (kindsOneHot ? 4 : 1);

  @override
  List<double> extract({
    required LanderState lander,
    required Terrain terrain,
    required double worldW,
    required double worldH,
    List<RayHit>? rays,
    double uiMaxFuel = 100.0,
  }) {
    if (rays == null) {
      throw StateError(
          'Runtime FE (rays) requires a rays list. Pass engine.rays into actWithIntent(...).');
    }

    double _angWrap(double a) {
      const twoPi = math.pi * 2.0;
      a = a % twoPi;
      if (a > math.pi) a -= twoPi;
      if (a < -math.pi) a += twoPi;
      return a;
    }

    // height-normalized vertical speed
    final gy = terrain.heightAt(lander.pos.x);
    final h  = (gy - lander.pos.y).toDouble().clamp(0.0, 1e9);
    double vCap = (0.10 * h + 8.0).clamp(8.0, 26.0);
    final hnVy  = (lander.vel.y.toDouble() / (vCap > 1e-6 ? vCap : 1.0)).clamp(-3.0, 3.0);

    // speed
    final vx = lander.vel.x.toDouble();
    final vy = lander.vel.y.toDouble();
    double speed = math.sqrt(vx * vx + vy * vy);
    const sClip = 140.0;
    speed = (speed / sClip).clamp(0.0, 1.5);

    // ship angle (normalized by pi)
    final ang = (lander.angle / math.pi).clamp(-2.0, 2.0);

    // pad vector (rays-avg if possible; else pad center)
    final padCx = (terrain.padX1 + terrain.padX2) * 0.5;
    final padCy = terrain.heightAt(padCx);
    double pxToPad = padCx - lander.pos.x;
    double pyToPad = padCy - lander.pos.y;

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

    // unit vectors
    final vLen = math.max(1e-9, math.sqrt(vx*vx + vy*vy));
    final vnx = vx / vLen, vny = vy / vLen;

    final pLen = math.max(1e-9, math.sqrt(pxToPad*pxToPad + pyToPad*pyToPad));
    final pnx = pxToPad / pLen, pny = pyToPad / pLen;

    // signed angle delta normalized by pi
    final angDelta = _angWrap(math.atan2(pny, pnx) - math.atan2(vny, vnx));
    final angDeltaPi = (angDelta / math.pi).clamp(-1.0, 1.0);
    final padVis = padVecValid ? 1.0 : 0.0;

    final out = <double>[
      speed, hnVy, ang, angDeltaPi, padVis,
    ];

    // rays block
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

({double x, double y, bool valid}) _avgPadVector({
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

/* =============================================================================
   Tiny math
   ========================================================================== */

List<double> _matVec(List<List<double>> W, List<double> x) {
  final m = W.length, n = x.length;
  final out = List<double>.filled(m, 0.0);
  for (int i = 0; i < m; i++) {
    double s = 0.0;
    final Wi = W[i];
    for (int j = 0; j < n; j++) s += Wi[j] * x[j];
    out[i] = s;
  }
  return out;
}

List<double> _addBias(List<double> v, List<double> b) {
  final out = List<double>.filled(v.length, 0.0);
  for (int i = 0; i < v.length; i++) out[i] = v[i] + b[i];
  return out;
}

double _tanhScalar(double x) {
  final ax = x.abs();
  if (ax > 20.0) return x.isNegative ? -1.0 : 1.0;
  final e = math.exp(-2.0 * ax);
  final t = (1.0 - e) / (1.0 + e);
  return x.isNegative ? -t : t;
}

List<double> _tanhVec(List<double> v) {
  final out = List<double>.filled(v.length, 0.0);
  for (int i = 0; i < v.length; i++) out[i] = _tanhScalar(v[i]);
  return out;
}

List<double> _softmax(List<double> z) {
  double m = z[0];
  for (int i = 1; i < z.length; i++) if (z[i] > m) m = z[i];
  double s = 0.0;
  final out = List<double>.filled(z.length, 0.0);
  for (int i = 0; i < z.length; i++) {
    final e = math.exp(z[i] - m);
    out[i] = e.isFinite ? e : 0.0;
    s += out[i];
  }
  final inv = (s > 0 && s.isFinite) ? (1.0 / s) : 0.0;
  for (int i = 0; i < z.length; i++) out[i] *= inv;
  return out;
}

int _argmax(List<double> a) {
  var idx = 0;
  var best = a[0];
  for (int i = 1; i < a.length; i++) {
    if (a[i] > best) {
      best = a[i];
      idx = i;
    }
  }
  return idx;
}

/* =============================================================================
   MLP & Norm
   ========================================================================== */

class _Linear {
  List<List<double>> W;
  List<double> b;
  _Linear(this.W, this.b);

  int get outDim => b.length;
  int get inDim => W.isEmpty ? 0 : W[0].length;

  List<double> forward(List<double> x) => _addBias(_matVec(W, x), b);
}

class _MLP {
  final List<_Linear> layers; // tanh activations between
  _MLP(this.layers);

  List<double> forward(List<double> x) {
    var h = x;
    for (int i = 0; i < layers.length; i++) {
      h = _tanhVec(layers[i].forward(h));
    }
    return h;
  }

  int get outDim => layers.isEmpty ? 0 : layers.last.outDim;
}

class _RuntimeNorm {
  bool inited;
  final int dim;
  List<double> mean;
  List<double> var_; // variance
  _RuntimeNorm(this.dim, {List<double>? mean, List<double>? var_})
      : inited = (mean != null && var_ != null),
        mean = mean ?? List<double>.filled(dim, 0.0),
        var_ = var_ ?? List<double>.filled(dim, 1.0);

  List<double> apply(List<double> x) {
    if (!inited || x.length != dim) return x;
    final out = List<double>.filled(dim, 0.0);
    for (int i = 0; i < dim; i++) {
      out[i] = (x[i] - mean[i]) / math.sqrt(var_[i] + 1e-6);
    }
    return out;
  }
}

/* =============================================================================
   Intents (must mirror training)
   ========================================================================== */

enum Intent { hover, goLeft, goRight, descendSlow, brakeUp, brakeLeft, brakeRight }
const List<String> kIntentNames = [
  'hover',
  'goLeft',
  'goRight',
  'descendSlow',
  'brakeUp',
  'brakeLeft',
  'brakeRight',
];

/* =============================================================================
   Low-level controllers
   ========================================================================== */

({bool thrust, bool left, bool right}) _controllerForIntentUI(
    Intent intent, {
      required LanderState lander,
      required Terrain terrain,
      required double worldW,
      required double worldH,
    }) {
  final padCx = (terrain.padX1 + terrain.padX2) * 0.5;
  final dx = lander.pos.x - padCx;
  final vx = lander.vel.x;
  final vy = lander.vel.y;
  final angle = lander.angle;

  bool left = false, right = false, thrust = false;

  final groundY = terrain.heightAt(lander.pos.x);
  final height = (groundY - lander.pos.y).clamp(0.0, 1e9);
  final ceilingDist = (lander.pos.y - 0.0).clamp(0.0, 1e9);

  const double vxGoalAbs = 80.0;
  const double kAngV = 0.015;
  const double kDxHover = 0.40;

  double maxTilt = 20 * math.pi / 180;
  if (ceilingDist < 140) {
    final a = (ceilingDist / 140.0).clamp(0.0, 1.0);
    final softMax = 12 * math.pi / 180;
    maxTilt = softMax + a * (maxTilt - softMax);
  }

  double vxDes = switch (intent) {
    Intent.goLeft => -vxGoalAbs,
    Intent.goRight => vxGoalAbs,
    Intent.brakeLeft => 0.0,
    Intent.brakeRight => 0.0,
    Intent.hover => -kDxHover * dx,
    _ => 0.0,
  };

  final vxErr = (vxDes - vx);
  final normErr = (vxErr.abs() / vxGoalAbs).clamp(0.0, 1.0);
  final levelGain = normErr * normErr;

  double targetAngle = (kAngV * vxErr * levelGain).clamp(-maxTilt, maxTilt);

  const angDead = 3 * math.pi / 180;
  if (angle > targetAngle + angDead) left = true;
  if (angle < targetAngle - angDead) right = true;

  double vyCapDown = 10.0 + 1.0 * math.sqrt(height.clamp(0.0, 9999.0));
  vyCapDown = vyCapDown.clamp(10.0, 45.0);

  double targetVy = vyCapDown;
  if (intent == Intent.descendSlow) targetVy = math.min(targetVy, 18.0);
  if (intent == Intent.brakeUp) targetVy = -15.0;

  if (height < 120) targetVy = math.min(targetVy, 18.0);
  if (height < 60) targetVy = math.min(targetVy, 10.0);

  final eVy = vy - targetVy;
  thrust = eVy > 0;

  const double vxErrTh = 20.0;
  final bool tiltAligned =
      (targetAngle > 6 * math.pi / 180 && angle > 3 * math.pi / 180) ||
          (targetAngle < -6 * math.pi / 180 && angle < -3 * math.pi / 180);
  final bool lateralIntent = intent == Intent.goLeft ||
      intent == Intent.goRight ||
      intent == Intent.brakeLeft ||
      intent == Intent.brakeRight;

  if (lateralIntent && tiltAligned && vxErr.abs() > vxErrTh && ceilingDist > 100 && vy > -6) {
    thrust = true;
  }

  if (ceilingDist < 40 && vy < 2) thrust = false;
  if (lander.pos.y < 4) thrust = false;

  return (thrust: thrust, left: left, right: right);
}

/// Extended controller that can also emit side/down thrusters based on physics.
({bool thrust, bool left, bool right, bool sideLeft, bool sideRight, bool downThrust})
_controllerForIntentExt(
    Intent intent, {
      required LanderState lander,
      required Terrain terrain,
      required double worldW,
      required double worldH,
      required RuntimePhysics phys,
    }) {
  final base = _controllerForIntentUI(
    intent,
    lander: lander,
    terrain: terrain,
    worldW: worldW,
    worldH: worldH,
  );

  bool sideLeft = false, sideRight = false, downThrust = false;

  // Simple, safe heuristics mirroring training-time teacher
  final padCx = (terrain.padX1 + terrain.padX2) * 0.5;
  final dx = (lander.pos.x - padCx).abs();
  final vx = lander.vel.x;
  final vy = lander.vel.y;
  final groundY = terrain.heightAt(lander.pos.x);
  final h = (groundY - lander.pos.y).toDouble();
  final W = worldW;

  // Use RCS for lateral intents when nearly level & safe
  bool canStrafe() {
    if (!phys.rcsEnabled) return false;
    final maxTilt = 0.10; // rad
    if (lander.angle.abs() > maxTilt) return false;
    if (!(h > 110 && h < 300)) return false;
    if (lander.vel.y.abs() >= 35.0) return false;
    return true;
  }

  switch (intent) {
    case Intent.goLeft:
      if (canStrafe()) sideRight = true; // push left
      break;
    case Intent.goRight:
      if (canStrafe()) sideLeft = true; // push right
      break;
    case Intent.brakeLeft:
      if (canStrafe() && vx < -4.0) sideLeft = true; // push rightwards (reduce |vx|)
      break;
    case Intent.brakeRight:
      if (canStrafe() && vx > 4.0) sideRight = true; // push leftwards
      break;
    case Intent.descendSlow:
    // In very low/zero gravity, use belly thruster to establish descent
      if (phys.downThrEnabled && phys.gravity.abs() < 1e-6) {
        final vCap = (0.10 * h + 8.0).clamp(8.0, 26.0);
        if (vy < 0.7 * vCap) downThrust = true; // start descending
      }
      break;
    default:
      break;
  }

  return (
  thrust: base.thrust,
  left: base.left,
  right: base.right,
  sideLeft: sideLeft,
  sideRight: sideRight,
  downThrust: downThrust
  );
}

/* =============================================================================
   Runtime policy
   ========================================================================== */

class RuntimeTwoStagePolicy {
  // Architecture (inferred)
  final int inputSize;
  final _MLP trunk; // arbitrary hidden layers (tanh)
  final _Linear headIntent; // (K, Hlast)
  // Optional loaded heads (not used by planner, kept for parity)
  final _Linear? headTurn; // (3, Hlast)
  final _Linear? headThr; // (1, Hlast)
  final _Linear? headVal; // (1, Hlast)

  // FE & planner config
  final _RuntimeFE fe;
  final int planHold; // base frames to hold an intent

  // Saved feature normalization (optional)
  final _RuntimeNorm? norm;
  final String? signature;

  // Loaded physics (NEW)
  final RuntimePhysics physics;

  pf.PotentialField? _pf; // ← current PF (optional)

  // Call this from GamePage whenever you (re)build the PF
  void setPotentialField(pf.PotentialField? field) {
    _pf = field;
  }

  // Planner state
  int _framesLeft = 0;
  int _currentIntentIdx = -1;
  List<double>? _lastReplanProbs;

  // Thrust latch (keeps engine ON for a few frames when climbing is required)
  int _thrustLatch = 0;
  int _thrustLatchBoost = 10; // tune 8..14 (≈0.13–0.23s @60Hz)

  // Diagnostics: last seen pad fraction
  double _padSeenFrac = 0.0;

  final bool fixPolarityWithPadRays; // swap goLeft/goRight if pad-avg disagrees
  final bool mirrorX; // hard flip left/right mapping (debug/safety)

  // ===== NEW: runtime knobs =====
  double intentTemp;           // 1.0 = unchanged; >1.0 flatter; <1.0 sharper
  bool stochasticPlanner;      // if true, sample intent ~ softmax(logits / T)
  final math.Random _rnd;

  RuntimeTwoStagePolicy._({
    required this.inputSize,
    required this.trunk,
    required this.headIntent,
    this.headTurn,
    this.headThr,
    this.headVal,
    required this.fe,
    required this.planHold,
    required this.norm,
    required this.signature,
    required this.physics, // NEW
    this.fixPolarityWithPadRays = false,
    this.mirrorX = false,
    this.intentTemp = 1.0,
    this.stochasticPlanner = true,
    math.Random? rnd,
  }) : _rnd = rnd ?? math.Random(0xC0FFEE);

  static RuntimeTwoStagePolicy fromJson(
      String jsonString, {
        _RuntimeFE? fe,
        int planHold = 1,
        // NEW knobs (optional)
        double intentTemp = 1.0,
        bool stochasticPlanner = true,
        math.Random? rnd,
      }) {
    final Map<String, dynamic> j = json.decode(jsonString);

    List<List<double>> _as2d(dynamic v) =>
        (v as List).map<List<double>>((r) => (r as List).map<double>((x) => (x as num).toDouble()).toList()).toList();
    List<double> _as1d(dynamic v) => (v as List).map<double>((x) => (x as num).toDouble()).toList();

    _RuntimeNorm? _readNorm(Map<String, dynamic> root, int expectDim, String? expectSig) {
      final nm = (root['norm'] as Map?)?.cast<String, dynamic>();
      if (nm != null) {
        final dim = (nm['dim'] as num?)?.toInt() ?? -1;
        final sig = nm['signature'] as String?;
        if (dim == expectDim && (expectSig == null || sig == expectSig)) {
          return _RuntimeNorm(dim, mean: _as1d(nm['mean']), var_: _as1d(nm['var']));
        }
      }
      final nmv = root['norm_mean'];
      final nvv = root['norm_var'];
      final nsig = root['norm_signature'];
      if (nmv is List && nvv is List) {
        if (expectSig == null || nsig == expectSig) {
          final mean = _as1d(nmv);
          final var_ = _as1d(nvv);
          if (mean.length == expectDim && var_.length == expectDim) {
            return _RuntimeNorm(expectDim, mean: mean, var_: var_);
          }
        }
      }
      return null;
    }

    // v2?
    final arch = (j['arch'] as Map?)?.cast<String, dynamic>();
    final trunkJ = (j['trunk'] as List?)?.cast<dynamic>();
    final headsJ = (j['heads'] as Map?)?.cast<String, dynamic>();
    final isV2 = (arch != null && trunkJ != null && headsJ != null);

    if (isV2) {
      final inputSize = (arch!['input'] as num).toInt();
      final sig = j['signature'] as String?;

      // Build trunk
      final layers = <_Linear>[];
      int expectIn = inputSize;
      for (int li = 0; li < trunkJ!.length; li++) {
        final layerObj = (trunkJ[li] as Map).cast<String, dynamic>();
        final W = _as2d(layerObj['W']);
        final b = _as1d(layerObj['b']);
        if (W.isEmpty || W[0].length != expectIn || W.length != b.length) {
          throw StateError(
              'Trunk layer $li shape mismatch: got ${W.length}x${W[0].length}, bias ${b.length}, expected in=$expectIn');
        }
        layers.add(_Linear(W, b));
        expectIn = b.length;
      }
      final trunk = _MLP(layers);
      final lastDim = trunk.outDim;

      _Linear _readHead(String key) {
        final hj = (headsJ![key] as Map).cast<String, dynamic>();
        final W = _as2d(hj['W']);
        final b = _as1d(hj['b']);
        if (W.isEmpty || W[0].length != lastDim || W.length != b.length) {
          throw StateError('Head "$key" shape mismatch.');
        }
        return _Linear(W, b);
      }

      final headIntent = _readHead('intent');
      _Linear? headTurn, headThr, headVal;
      if (headsJ.containsKey('turn')) headTurn = _readHead('turn');
      if (headsJ.containsKey('thr')) headThr = _readHead('thr');
      if (headsJ.containsKey('val')) headVal = _readHead('val');

      // FE choice
      _RuntimeFE fe0;
      if (fe != null) {
        fe0 = fe;
      } else {
        final fej = (j['feature_extractor'] as Map?)?.cast<String, dynamic>() ?? const {};
        final kind = (fej['kind'] as String?) ?? 'legacy';
        if (kind == 'rays') {
          final rayCount = ((fej['rayCount'] ?? 180) as num).toInt();
          final kindsOneHot = ((fej['kindsOneHot'] ?? true) as bool);
          fe0 = _RuntimeFE_Rays(rayCount: rayCount, kindsOneHot: kindsOneHot);
        } else {
          final gs = ((fej['groundSamples'] ?? 3) as num).toInt();
          final stride = ((fej['stridePx'] ?? 48) as num).toDouble();
          fe0 = _RuntimeFE_Legacy(groundSamples: gs, stridePx: stride);
        }
      }

      final norm = _readNorm(j, inputSize, sig);

      // NEW: read physics bundle (present if trained with the new saver)
      final physics = RuntimePhysics.fromJsonMap(
        (j['physics'] as Map?)?.cast<String, dynamic>(),
      );

      return RuntimeTwoStagePolicy._(
        inputSize: inputSize,
        trunk: trunk,
        headIntent: headIntent,
        headTurn: headTurn,
        headThr: headThr,
        headVal: headVal,
        fe: fe0,
        planHold: planHold,
        norm: norm,
        signature: sig,
        physics: physics,
        fixPolarityWithPadRays: false,
        mirrorX: false,
        intentTemp: intentTemp,
        stochasticPlanner: stochasticPlanner,
        rnd: rnd,
      );
    }

    // Legacy v1
    List<List<double>> _as2dReq(String k) => _as2d(j[k]);
    List<double> _as1dReq(String k) => _as1d(j[k]);

    final inputSize = (j['inputSize'] as num).toInt();
    final W1 = _as2dReq('W1');
    final b1 = _as1dReq('b1');
    final W2 = _as2dReq('W2');
    final b2 = _as1dReq('b2');
    final W_int = _as2dReq('W_intent');
    final b_int = _as1dReq('b_intent');

    _Linear? headTurn, headThr, headVal;
    if (j.containsKey('W_turn') && j.containsKey('b_turn')) {
      headTurn = _Linear(_as2d(j['W_turn']), _as1d(j['b_turn']));
    }
    if (j.containsKey('W_thr') && j.containsKey('b_thr')) {
      headThr = _Linear(_as2d(j['W_thr']), _as1d(j['b_thr']));
    }
    if (j.containsKey('W_val') && j.containsKey('b_val')) {
      headVal = _Linear(_as2d(j['W_val']), _as1d(j['b_val']));
    }

    final trunk = _MLP([_Linear(W1, b1), _Linear(W2, b2)]);
    final headIntent = _Linear(W_int, b_int);

    // Try to read legacy FE hints (optional)
    final feh = (j['fe'] as Map?)?.cast<String, dynamic>() ??
        (j['feature_extractor'] as Map?)?.cast<String, dynamic>() ??
        const {};
    final gs = ((feh['groundSamples'] ?? 3) as num).toInt();
    final stride = ((feh['stridePx'] ?? 48) as num).toDouble();
    final fe0 = _RuntimeFE_Legacy(groundSamples: gs, stridePx: stride);

    final sig = j['signature'] as String?;
    final norm = _readNorm(j, inputSize, sig);

    // Legacy models have no physics block → fallbacks
    final physics = const RuntimePhysics();

    return RuntimeTwoStagePolicy._(
      inputSize: inputSize,
      trunk: trunk,
      headIntent: headIntent,
      headTurn: headTurn,
      headThr: headThr,
      headVal: headVal,
      fe: fe0,
      planHold: planHold,
      norm: norm,
      signature: sig,
      physics: physics,
      intentTemp: intentTemp,
      stochasticPlanner: stochasticPlanner,
      rnd: rnd,
    );
  }

  static Future<RuntimeTwoStagePolicy> loadFromAsset(
      String assetPath, {
        int planHold = 12,
        // NEW knobs (optional)
        double intentTemp = 1.0,
        bool stochasticPlanner = false,
        math.Random? rnd,
      }) async {
    final js = await rootBundle.loadString(assetPath);
    return RuntimeTwoStagePolicy.fromJson(
      js,
      planHold: planHold,
      intentTemp: intentTemp,
      stochasticPlanner: stochasticPlanner,
      rnd: rnd,
    );
  }

  // Expose physics to the game layer (read-only)
  RuntimePhysics get phys => physics;

  void resetPlanner() {
    _framesLeft = 0;
    _currentIntentIdx = -1;
    _thrustLatch = 0;
    _padSeenFrac = 0.0;
  }

  // ===== NEW: runtime setters you can call from UI/loop =====
  void setIntentTemperature(double t) {
    intentTemp = t.clamp(1e-6, 1000.0);
  }

  void setStochasticPlanner(bool on) {
    stochasticPlanner = on;
  }

  // --- Dynamic plan-hold multiplier (1..4) based on calmness/centering
  int _dynamicPlanHoldMul({
    required LanderState lander,
    required Terrain terrain,
    required double worldW,
  }) {
    final padCx = (terrain.padX1 + terrain.padX2) * 0.5;
    final dxAbs = (lander.pos.x - padCx).abs();
    final vxAbs = lander.vel.x.abs();
    final gy = terrain.heightAt(lander.pos.x);
    final h  = (gy - lander.pos.y).toDouble().clamp(0.0, 1e9);
    final W  = worldW.toDouble();

    int hold = 1; // fast by default
    if (dxAbs > 0.12 * W || vxAbs > 60.0) hold = 1;
    if (h > 280.0 && dxAbs < 0.08 * W && vxAbs < 40.0) hold = 2;
    if (h > 340.0 && dxAbs < 0.06 * W && vxAbs < 30.0) hold = 3;
    if (h > 380.0 && dxAbs < 0.04 * W && vxAbs < 20.0) hold = 4;
    if (h < 160.0) hold = math.max(hold, 2); // near pad: modest dwell
    return hold.clamp(1, 8);
  }

  double _vCapBrakeUp(double h) => (0.07 * h + 6.0).clamp(6.0, 16.0);

  /// Back-compat: original API (no side/down thrusters).
  /// If your engine supports extra channels, call [actWithIntentExt] instead.
  (bool thrust, bool left, bool right, int intentIdx, List<double> probs) actWithIntent({
    required LanderState lander,
    required Terrain terrain,
    required double worldW,
    required double worldH,
    List<RayHit>? rays, // REQUIRED when FE is rays
    int step = 0,
    double uiMaxFuel = 100.0,
  }) {
    // Re-plan if needed
    if (_framesLeft <= 0) {
      var x = fe.extract(
        lander: lander,
        terrain: terrain,
        worldW: worldW,
        worldH: worldH,
        rays: rays,
        uiMaxFuel: uiMaxFuel,
      );
      if (norm != null) x = norm!.apply(x);

      final h = trunk.forward(x);
      final rawLogits = headIntent.forward(h);

      // ===== Temperature + (optional) sampling =====
      final z = List<double>.generate(rawLogits.length, (i) => rawLogits[i] / intentTemp);
      final probs = _softmax(z);

      int idx;
      if (stochasticPlanner) {
        final u = _rnd.nextDouble();
        double acc = 0.0;
        idx = probs.length - 1;
        for (int i = 0; i < probs.length; i++) {
          acc += probs[i];
          if (u <= acc) { idx = i; break; }
        }
      } else {
        idx = _argmax(probs);
      }

      _currentIntentIdx = idx;

      // Dynamic plan-hold multiplier
      final holdMul = _dynamicPlanHoldMul(lander: lander, terrain: terrain, worldW: worldW);
      _framesLeft = (planHold * holdMul).clamp(1, 120);

      _lastReplanProbs = probs;

      // Optional polarity fix using average pad vector
      if (fixPolarityWithPadRays && rays != null && rays.isNotEmpty) {
        final av = _avgPadVector(rays: rays, px: lander.pos.x, py: lander.pos.y);
        if (av.valid) {
          final padIsLeft = av.x < 0.0;
          final isLeft = _currentIntentIdx == Intent.goLeft.index;
          final isRight = _currentIntentIdx == Intent.goRight.index;
          if ((isLeft && !padIsLeft) || (isRight && padIsLeft)) {
            _currentIntentIdx = isLeft ? Intent.goRight.index : Intent.goLeft.index;
          }
        }
      }

      // Pad visibility diagnostics (for overlays/logs)
      if (rays != null && rays.isNotEmpty) {
        int padHits = 0;
        for (final r in rays) { if (r.kind == RayHitKind.pad) padHits++; }
        _padSeenFrac = padHits / rays.length;
      } else {
        _padSeenFrac = 0.0;
      }

      // === Publish a plan preview for the overlay (if PF available) ===
      try {
        final field = _pf;
        if (field != null) {
          final planPts = _buildPFPlanPolyline(
            lander: lander,
            terrain: terrain,
            worldW: worldW,
            worldH: worldH,
            field: field,
            g: physics.gravity,
          );
          final widths = _makeUncertaintyWidths(
            pts: planPts,
            terrain: terrain,
            lander: lander,
          );
          PlanBus.instance.push(points: planPts, widths: widths, source: 'pf');
        }
      } catch (_) {
        // ignore planning errors
      }

      IntentBus.instance.publishIntent(
        IntentEvent(
          intent: kIntentNames[_currentIntentIdx],
          probs: probs,
          step: step,
          meta: {
            'plan_hold': planHold,
            'hold_mul': holdMul,
            'T': intentTemp,
            'stochastic': stochasticPlanner,
            'padSeenFrac': _padSeenFrac,
          },
        ),
      );
    }

    final idxNow = _currentIntentIdx < 0 ? 0 : _currentIntentIdx;
    final intent = Intent.values[idxNow];

    var ctrl = _controllerForIntentUI(
      intent,
      lander: lander,
      terrain: terrain,
      worldW: worldW,
      worldH: worldH,
    );

    // ====== Thrust LATCH & emergency-up ======
    final groundY = terrain.heightAt(lander.pos.x.toDouble());
    final height  = (groundY - lander.pos.y).toDouble().clamp(0.0, 1e9);
    final vCapUp  = _vCapBrakeUp(height);
    final needEmergencyUp = (height < 220.0) && (lander.vel.y > 0.9 * vCapUp);

    if (intent == Intent.brakeUp || needEmergencyUp) {
      _thrustLatch = math.max(_thrustLatch, _thrustLatchBoost);
    }

    bool thrustOut = ctrl.thrust;
    if (_thrustLatch > 0) {
      thrustOut = true;
      _thrustLatch--;
    }

    // Optional global turn inversion (debug/safety)
    if (mirrorX) {
      final swapped = (thrust: thrustOut, left: ctrl.right, right: ctrl.left);
      IntentBus.instance.publishControl(ControlEvent(
        thrust: swapped.thrust,
        left: swapped.left,
        right: swapped.right,
        step: step,
        meta: {
          'intent': kIntentNames[idxNow],
          'mirrorX': true,
          'hold_left': _framesLeft,
          'thrust_latch': _thrustLatch,
          'padSeenFrac': _padSeenFrac,
        },
      ));
      _framesLeft -= 1;
      return (swapped.thrust, swapped.left, swapped.right, idxNow, _lastReplanProbs ?? const []);
    }

    IntentBus.instance.publishControl(
      ControlEvent(
        thrust: thrustOut,
        left: ctrl.left,
        right: ctrl.right,
        step: step,
        meta: {
          'intent': kIntentNames[idxNow],
          'hold_left': _framesLeft,
          'thrust_latch': _thrustLatch,
          'padSeenFrac': _padSeenFrac,
        },
      ),
    );

    _framesLeft -= 1;
    return (thrustOut, ctrl.left, ctrl.right, idxNow, _lastReplanProbs ?? const []);
  }

  /// NEW: Extended API that also returns side/down thrusters based on physics.
  /// Use this in the live engine so you can pass all controls to `GameEngine.step`.
  (bool thrust, bool left, bool right, bool sideLeft, bool sideRight, bool downThrust,
  int intentIdx, List<double> probs) actWithIntentExt({
    required LanderState lander,
    required Terrain terrain,
    required double worldW,
    required double worldH,
    List<RayHit>? rays, // REQUIRED when FE is rays
    int step = 0,
    double uiMaxFuel = 100.0,
  }) {
    // Reuse the planning path from the legacy method to keep behavior identical
    final legacy = actWithIntent(
      lander: lander,
      terrain: terrain,
      worldW: worldW,
      worldH: worldH,
      rays: rays,
      step: step,
      uiMaxFuel: uiMaxFuel,
    );
    final idxNow = legacy.$4;
    final intent = Intent.values[idxNow];

    final ext = _controllerForIntentExt(
      intent,
      lander: lander,
      terrain: terrain,
      worldW: worldW,
      worldH: worldH,
      phys: physics,
    );

    // Mirror left/right if requested
    bool left = legacy.$2, right = legacy.$3;
    bool sideLeft = ext.sideLeft, sideRight = ext.sideRight;
    if (mirrorX) {
      final tmp = left; left = right; right = tmp;
      final tmp2 = sideLeft; sideLeft = sideRight; sideRight = tmp2;
    }

    // Publish extended control for debugging UIs (optional)
    IntentBus.instance.publishControl(ControlEvent(
      thrust: legacy.$1,
      left: left,
      right: right,
      step: step,
      meta: {
        'intent': kIntentNames[idxNow],
        'sideLeft': sideLeft,
        'sideRight': sideRight,
        'downThrust': ext.downThrust,
        'hold_left': _framesLeft,
        'thrust_latch': _thrustLatch,
        'padSeenFrac': _padSeenFrac,
      },
    ));

    return (
    legacy.$1, // thrust (main, with latch)
    left,
    right,
    sideLeft,
    sideRight,
    ext.downThrust,
    idxNow,
    legacy.$5
    );
  }

  // Acceleration-limited plan that follows the PF vector field toward the pad.
  List<Offset> _buildPFPlanPolyline({
    required LanderState lander,
    required Terrain terrain,
    required double worldW,
    required double worldH,
    required pf.PotentialField field,
    required double g, // gravity (downward +)
  }) {
    double x  = lander.pos.x.toDouble();
    double y  = lander.pos.y.toDouble();
    double vx = lander.vel.x.toDouble();
    double vy = lander.vel.y.toDouble();

    final padCx = (terrain.padX1 + terrain.padX2) * 0.5;
    final padCy = terrain.heightAt(padCx);

    // rollout params
    const int    maxSteps = 220;  // long enough to reach the pad
    const double dt       = 0.035;
    const double aMax     = 120.0;   // px/s^2 accel budget
    const double vMinTD   =  2.0;    // min near touchdown
    const double vMaxPF   = 95.0;    // global cap
    const double wallK    = 0.16;    // side-wall inward bias
    const double wallFrac = 0.12;    // wall margin (fraction of world width)

    // flare shaping (prefer gentler vertical as we get close)
    double _flare(double dxAbs, double h) {
      final W = worldW.toDouble();
      final tightX = 0.10 * W;
      final px = math.exp(- (dxAbs*dxAbs) / (tightX*tightX + 1e-6));
      final ph = math.exp(- (h*h) / (140.0*140.0 + 1e-6));
      final prox = (px * ph).clamp(0.0, 1.0);
      return (1.0 - 0.80 * prox); // 1 far → 0.2 near
    }

    double _inwardVX(double xx) {
      final W = worldW.toDouble();
      final margin = wallFrac * W;
      final distL = xx.clamp(0.0, margin);
      final distR = (W - xx).clamp(0.0, margin);
      final nearL = 1.0 - (distL / margin);
      final nearR = 1.0 - (distR / margin);
      double inward = 0.0;
      if (nearL > 0.0 && nearL >= nearR) inward =  1.0;
      if (nearR > 0.0 && nearR >  nearL) inward = -1.0;
      return inward * 35.0 * math.max(nearL, nearR);
    }

    void _steerToward(double tvx, double tvy) {
      final dvx = tvx - vx, dvy = tvy - vy;
      final dv = math.sqrt(dvx*dvx + dvy*dvy);
      final dvCap = aMax * dt;
      if (dv > dvCap && dv > 1e-9) {
        final s = dvCap / dv;
        vx += dvx * s; vy += dvy * s;
      } else {
        vx = tvx; vy = tvy;
      }
    }

    final pts = <Offset>[Offset(x, y)];

    for (int i = 0; i < maxSteps; i++) {
      // ground/height
      final gY = terrain.heightAt(x);
      final h  = (gY - y).clamp(0.0, 1e9);
      final dxAbs = (x - padCx).abs();

      // 1) PF flow (direction + magnitude proxy)
      // Prefer sampleFlow; if unavailable, fall back to suggestVelocity
      double fx = 0.0, fy = 0.0, fmag = 0.0;
      try {
        final f = field.sampleFlow(x, y);  // has nx, ny, mag
        fx = f.nx; fy = f.ny; fmag = f.mag;
      } catch (_) {
        final s = field.suggestVelocity(
          x, y,
          vMinClose: 8.0,
          vMaxFar: vMaxPF,
          alpha: 1.2,
          clampSpeed: 9999.0,
        );
        final n = math.sqrt(s.vx*s.vx + s.vy*s.vy);
        if (n > 1e-6) { fx = s.vx / n; fy = s.vy / n; fmag = n; }
      }
      if (fmag < 1e-9) { fx = 0.0; fy = -1.0; }

      // 2) PF-ish speed schedule + flare
      final vFar = 10.0 + 85.0 * (1.0 - math.exp(-h / 240.0));
      final flare = _flare(dxAbs, h);
      final sv = (vFar * flare).clamp(vMinTD, vMaxPF);

      // desired velocity along PF
      double tvx = fx * sv;
      double tvy = fy * sv;

      // 3) wall nudge
      tvx = (1.0 - wallK) * tvx + wallK * _inwardVX(x);

      // 4) mild gravity compensation to keep graceful arcs
      final vyLook = vy + g * dt;
      tvy = (tvy - 0.35 * vyLook);

      // 5) steer with accel limit
      _steerToward(tvx, tvy);

      // physics preview with gravity
      vy += g * dt;
      x  += vx * dt;
      y  += vy * dt;

      // keep inside horizontal bounds
      x = x.clamp(0.0, worldW.toDouble());

      // skim above ground for visibility
      final gNow = terrain.heightAt(x);
      if (y >= gNow) {
        y = gNow - 0.001;
        if (vy > 0) vy = 0; // kill downward overshoot visually
      }

      pts.add(Offset(x, y));

      // stop condition: reached pad & slowed
      final nearX = (x - padCx).abs() <= 6.0;
      final nearY = (y - padCy).abs() <= 6.0;
      final slow  = math.sqrt(vx*vx + vy*vy) <= 6.0;
      if (nearX && nearY && slow) break;
    }

    // ensure a little length
    if (pts.length < 8) {
      final last = pts.last;
      while (pts.length < 8) pts.add(last);
    }
    return pts;
  }

  List<double> _makeUncertaintyWidths({
    required LanderState lander,
    required Terrain terrain,
    required List<Offset> pts,
  }) {
    if (pts.isEmpty) return const [];

    // speed & height right now (for base width)
    final vx = lander.vel.x.toDouble();
    final vy = lander.vel.y.toDouble();
    final sp = math.sqrt(vx * vx + vy * vy);

    final gy = terrain.heightAt(lander.pos.x.toDouble());
    final h0 = (gy - lander.pos.y).toDouble().clamp(0.0, 800.0);

    // base width depends a bit on speed & height
    final base = (6.0 + 0.05 * sp + 0.015 * h0).clamp(6.0, 38.0);

    final n = pts.length;
    final out = List<double>.filled(n, 0.0);

    for (int i = 0; i < n; i++) {
      final t = (n <= 1) ? 0.0 : (i / (n - 1)); // 0 at craft, 1 at pad/future
      // grow from skinny near the craft to wider toward the pad
      final grow = 0.45 + 0.75 * t;
      out[i] = base * grow;
    }

    if (n >= 2) {
      out[0] *= 0.35;        // tight at the craft
      out[n - 1] *= 1.10;    // a touch wider at the end
    }

    // quick smoothing
    for (int i = 1; i < n - 1; i++) {
      out[i] = (out[i - 1] + out[i] + out[i + 1]) / 3.0;
    }
    return out;
  }

  List<double> _makePlanWidths({
    required List<Offset> pts,
    required Terrain terrain,
    required pf.PotentialField field,
    required double worldW,
    required double worldH,
  }) {
    if (pts.isEmpty) return const [];
    final n = pts.length;
    final w = List<double>.filled(n, 0.0);

    double distToEnd = 0.0;
    // precompute cumulative backward distance
    final dist = List<double>.filled(n, 0.0);
    for (int i = n - 2; i >= 0; i--) {
      distToEnd += (pts[i + 1] - pts[i]).distance;
      dist[i] = distToEnd;
    }

    for (int i = 0; i < n; i++) {
      final p = pts[i];
      // PF magnitude as “confidence”; lower mag ⇒ flatter potential ⇒ more uncertainty
      double conf;
      try {
        final f = field.sampleFlow(p.dx, p.dy);
        conf = f.mag; // 0..?
      } catch (_) {
        conf = 1.0;
      }
      // normalize-ish
      conf = conf.clamp(0.0, 8.0) / 8.0; // ~[0..1]
      final invConf = 1.0 - conf;

      // base width
      double base = 6.0;
      // widen with distance remaining
      base += (dist[i] / 300.0).clamp(0.0, 14.0); // up to +14px far away
      // widen near walls
      final margin = 0.10 * worldW;
      final nearWall = (p.dx < margin) ? (1.0 - p.dx / margin)
          : (p.dx > worldW - margin) ? (1.0 - (worldW - p.dx) / margin)
          : 0.0;
      base += 10.0 * nearWall;
      // widen when PF is weak / ambiguous
      base += 14.0 * invConf;

      // taper at the very end (touchdown)
      final t = (i / (n - 1)).clamp(0.0, 1.0);
      final taper = 1.0 - math.pow(t, 2.0) as double; // bigger earlier, smaller near pad
      w[i] = (base * taper).clamp(3.0, 32.0);
    }
    return w;
  }
}
