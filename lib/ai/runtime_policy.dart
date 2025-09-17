// lib/ai/runtime_policy.dart
import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/services.dart' show rootBundle;

// UI types
import 'package:flutter/material.dart' show Offset;
import '../engine/types.dart' show LanderState, Terrain, RayHitKind;
import '../engine/raycast.dart' show RayHit, RayHitKind; // for ray-based FE

// Intent bus (runtime)
import 'intent_bus.dart';

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

/// -------- Ray-based (lander stats + ray channels) --------
class _RuntimeFE_Rays implements _RuntimeFE {
  final int rayCount;
  final bool kindsOneHot;
  const _RuntimeFE_Rays({required this.rayCount, required this.kindsOneHot});

  @override
  int get inputSize => 6 + rayCount * (kindsOneHot ? 4 : 1);

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
          'Runtime FE (rays) requires a rays list. Pass engine.rays into actWithIntent(..., rays: engine.rays).');
    }

    // Base 6
    final px = (lander.pos.x / worldW);
    final py = (lander.pos.y / worldH);
    final vx = (lander.vel.x / 200.0).clamp(-3.0, 3.0);
    final vy = (lander.vel.y / 200.0).clamp(-3.0, 3.0);
    final ang = (lander.angle / math.pi).clamp(-2.0, 2.0);
    final fuel = (lander.fuel / (uiMaxFuel > 0 ? uiMaxFuel : 1.0)).clamp(0.0, 1.0);

    final out = <double>[px, py, vx, vy, ang, fuel];

    // distance normalized by world diagonal
    final maxD = math.sqrt(worldW * worldW + worldH * worldH);

    // deterministic size
    final int n = rayCount;
    for (int i = 0; i < n; i++) {
      RayHit? rh;
      if (i < rays.length) {
        rh = rays[i];
      }
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
    final d2 = dx*dx + dy*dy;
    if (d2 <= 1e-9) continue;
    // Heavier weight when closer to the pad; clamp to avoid explosions
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
    if (a[i] > best) { best = a[i]; idx = i; }
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
   Low-level controller (heuristic) to execute the chosen intent
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
  final height  = (groundY - lander.pos.y).clamp(0.0, 1e9);
  final ceilingDist = (lander.pos.y - 0.0).clamp(0.0, 1e9);

  const double vxGoalAbs = 80.0;
  const double kAngV     = 0.015;
  const double kDxHover  = 0.40;

  double maxTilt = 20 * math.pi / 180;
  if (ceilingDist < 140) {
    final a = (ceilingDist / 140.0).clamp(0.0, 1.0);
    final softMax = 12 * math.pi / 180;
    maxTilt = softMax + a * (maxTilt - softMax);
  }

  double vxDes = switch (intent) {
    Intent.goLeft     => -vxGoalAbs,
    Intent.goRight    =>  vxGoalAbs,
    Intent.brakeLeft  =>  0.0,           // actively reduce |vx| when moving left
    Intent.brakeRight =>  0.0,           // actively reduce |vx| when moving right
    Intent.hover      => -kDxHover * dx, // center over pad
    _                 => 0.0,
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
  if (intent == Intent.brakeUp)     targetVy = -15.0;

  if (height < 120) targetVy = math.min(targetVy, 18.0);
  if (height <  60) targetVy = math.min(targetVy, 10.0);

  final eVy = vy - targetVy;
  thrust = eVy > 0;

  const double vxErrTh = 20.0;
  final bool tiltAligned =
      (targetAngle >  6 * math.pi / 180 && angle >  3 * math.pi / 180) ||
          (targetAngle < -6 * math.pi / 180 && angle < -3 * math.pi / 180);
  final bool lateralIntent = intent == Intent.goLeft
      || intent == Intent.goRight
      || intent == Intent.brakeLeft
      || intent == Intent.brakeRight;

  if (lateralIntent &&
      tiltAligned &&
      vxErr.abs() > vxErrTh &&
      ceilingDist > 100 &&
      vy > -6) {
    thrust = true;
  }

  if (ceilingDist < 40 && vy < 2) {
    thrust = false;
  }

  if (lander.pos.y < 4) thrust = false;

  return (thrust: thrust, left: left, right: right);
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
  final _Linear? headTurn;  // (3, Hlast)
  final _Linear? headThr;   // (1, Hlast)
  final _Linear? headVal;   // (1, Hlast)

  // FE & planner config
  final _RuntimeFE fe;
  final int planHold; // frames to hold an intent

  // Saved feature normalization (optional)
  final _RuntimeNorm? norm;
  final String? signature;

  // Planner state
  int _framesLeft = 0;
  int _currentIntentIdx = -1;
  List<double>? _lastReplanProbs;

  final bool fixPolarityWithPadRays; // swap goLeft/goRight if pad-avg disagrees
  final bool mirrorX;                // hard flip left/right mapping (debug/safety)

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
    this.fixPolarityWithPadRays = false,
    this.mirrorX = false,
  });

  static RuntimeTwoStagePolicy fromJson(
      String jsonString, {
        _RuntimeFE? fe,
        int planHold = 1,
      }) {
    final Map<String, dynamic> j = json.decode(jsonString);

    List<List<double>> _as2d(dynamic v) =>
        (v as List).map<List<double>>((r) => (r as List).map<double>((x)=> (x as num).toDouble()).toList()).toList();
    List<double> _as1d(dynamic v) =>
        (v as List).map<double>((x)=> (x as num).toDouble()).toList();

    _RuntimeNorm? _readNorm(Map<String, dynamic> root, int expectDim, String? expectSig) {
      final nm = (root['norm'] as Map?)?.cast<String, dynamic>();
      if (nm != null) {
        final dim = (nm['dim'] as num?)?.toInt() ?? -1;
        final sig = nm['signature'] as String?;
        if (dim == expectDim && (expectSig == null || sig == expectSig)) {
          return _RuntimeNorm(dim,
              mean: _as1d(nm['mean']),
              var_: _as1d(nm['var']));
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
          throw StateError('Trunk layer $li shape mismatch: got ${W.length}x${W[0].length}, bias ${b.length}, expected in=$expectIn');
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
      if (headsJ.containsKey('thr'))  headThr  = _readHead('thr');
      if (headsJ.containsKey('val'))  headVal  = _readHead('val');

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
        fixPolarityWithPadRays: true,
        mirrorX: false,
      );
    }

    // Legacy v1
    List<List<double>> _as2dReq(String k) => _as2d(j[k]);
    List<double> _as1dReq(String k) => _as1d(j[k]);

    final inputSize = (j['inputSize'] as num).toInt();
    final W1 = _as2dReq('W1'); final b1 = _as1dReq('b1');
    final W2 = _as2dReq('W2'); final b2 = _as1dReq('b2');
    final W_int = _as2dReq('W_intent'); final b_int = _as1dReq('b_intent');

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
    );
  }

  static Future<RuntimeTwoStagePolicy> loadFromAsset(
      String assetPath, {
        int planHold = 12,
      }) async {
    final js = await rootBundle.loadString(assetPath);
    return RuntimeTwoStagePolicy.fromJson(js, planHold: planHold);
  }

  void resetPlanner() { _framesLeft = 0; _currentIntentIdx = -1; }

  /// Main runtime step: choose/hold intent, emit control, publish bus events.
  ///
  /// If the loaded policy expects **ray features**, you **must** pass [rays].
  (bool thrust, bool left, bool right, int intentIdx, List<double> probs) actWithIntent({
    required LanderState lander,
    required Terrain terrain,
    required double worldW,
    required double worldH,
    List<RayHit>? rays,            // REQUIRED when FE is rays
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
      final logits = headIntent.forward(h);
      final probs = _softmax(logits);
      final idx = _argmax(probs);

      _currentIntentIdx = idx;
      _framesLeft = planHold;
      _lastReplanProbs = probs;

// Optional polarity fix using average of pad rays (more stable than nearest)
      if (fixPolarityWithPadRays && rays != null && rays.isNotEmpty) {
        final av = _avgPadVector(
          rays: rays,
          px: lander.pos.x,
          py: lander.pos.y,
        );
        if (av.valid) {
          // If pad is to the left (negative x from craft), we *expect* goLeft.
          final padIsLeft = av.x < 0.0;

          // Current NN choice:
          final isLeft  = _currentIntentIdx == Intent.goLeft.index;
          final isRight = _currentIntentIdx == Intent.goRight.index;

          // If NN picked left but pad vector says right (or vice versa), flip.
          if ((isLeft  && !padIsLeft) ||
              (isRight &&  padIsLeft)) {
            _currentIntentIdx =
            isLeft ? Intent.goRight.index : Intent.goLeft.index;

            // Optional: tiny debug pulse to confirm flips in logs
            IntentBus.instance.publishIntent(IntentEvent(
              intent: kIntentNames[_currentIntentIdx],
              probs: _lastReplanProbs ?? const [],
              step: step,
              meta: {'polarity_fix': true},
            ));
          }
        }
      }

      IntentBus.instance.publishIntent(
        IntentEvent(
          intent: kIntentNames[idx],
          probs: probs,
          step: step,
          meta: {'plan_hold': planHold},
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

    // Optional global turn inversion (debug/safety)
    if (mirrorX) {
      // Swap emitted turn commands
      final swapped = (thrust: ctrl.thrust, left: ctrl.right, right: ctrl.left);
      IntentBus.instance.publishControl(ControlEvent(
        thrust: swapped.thrust, left: swapped.left, right: swapped.right,
        step: step, meta: {'intent': kIntentNames[idxNow], 'mirrorX': true},
      ));
      _framesLeft -= 1;
      return (swapped.thrust, swapped.left, swapped.right, idxNow, _lastReplanProbs ?? const []);
    }

    IntentBus.instance.publishControl(
      ControlEvent(
        thrust: ctrl.thrust,
        left: ctrl.left,
        right: ctrl.right,
        step: step,
        meta: {'intent': kIntentNames[idxNow]},
      ),
    );

    _framesLeft -= 1;
    return (ctrl.thrust, ctrl.left, ctrl.right, idxNow, _lastReplanProbs ?? const []);
  }
}
