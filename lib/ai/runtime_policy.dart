// lib/ai/runtime_policy.dart
import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/services.dart' show rootBundle;

// UI types
import 'package:flutter/material.dart' show Offset;
import '../engine/types.dart' show LanderState, Terrain;

// Intent bus (runtime)
import 'intent_bus.dart';

/// ================= Runtime features (must mirror training) =================
class RuntimeFeatureExtractor {
  final int groundSamples;
  final double stridePx;
  const RuntimeFeatureExtractor({this.groundSamples = 3, this.stridePx = 48});

  /// Training layout (IMPORTANT, must match lib/ai/agent.dart FeatureExtractor):
  /// [ px/W, py/H, vx/200, vy/200, ang/pi, fuel/maxFuelUI,
  ///   padCx/W, dxPad/(0.5W), (gy - py)/300, slope/2, samples... ]
  ///
  /// NOTE: In UI we don't know Tunables.maxFuel; the training default is 1000,
  /// but the Flutter demo typically uses 100. Expose a param if you need it.
  List<double> extract({
    required LanderState lander,
    required Terrain terrain,
    required double worldW,
    required double worldH,
    double uiMaxFuel = 100.0, // set to your UI max fuel; training divides by maxFuel
  }) {
    final pxAbs = lander.pos.x;
    final pyAbs = lander.pos.y;

    final px = pxAbs / worldW;
    final py = pyAbs / worldH;
    final vx = (lander.vel.x / 200.0).clamp(-3.0, 3.0);
    final vy = (lander.vel.y / 200.0).clamp(-3.0, 3.0);
    final ang = (lander.angle / math.pi).clamp(-2.0, 2.0);
    final fuel = (lander.fuel / (uiMaxFuel > 0 ? uiMaxFuel : 1.0)).clamp(0.0, 1.0);

    final padCenterAbs = (terrain.padX1 + terrain.padX2) * 0.5;
    final padCenter = padCenterAbs / worldW;
    final dxToPad = ((pxAbs - padCenterAbs) / (0.5 * worldW)).clamp(-1.5, 1.5);

    final gyCenter = terrain.heightAt(pxAbs);
    final hAbove = ((gyCenter - pyAbs) / 300.0).clamp(-2.0, 2.0);

    // Slope uses +/- 8 px sample like training (agent.dart): slope = (hR - hL) / 16, then /2.0
    final gyL = terrain.heightAt((pxAbs - 8.0).clamp(0.0, worldW));
    final gyR = terrain.heightAt((pxAbs + 8.0).clamp(0.0, worldW));
    final slope = (((gyR - gyL) / 16.0) / 2.0).clamp(-2.0, 2.0);

    // local ground samples (same spacing used during training)
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

  int get inputSize => 10 + groundSamples;
}

/// ========================== Minimal math (pure Dart) ==========================
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

double _sigmoid(double x) {
  if (x >= 0) {
    final ex = math.exp(-x);
    return 1.0 / (1.0 + ex);
  } else {
    final ex = math.exp(x);
    return ex / (1.0 + ex);
  }
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

/// =================== Heads & MLP (arbitrary hidden layers) ====================
class _Linear {
  List<List<double>> W;
  List<double> b;
  _Linear(this.W, this.b);

  int get outDim => b.length;
  int get inDim => W.isEmpty ? 0 : W[0].length;

  List<double> forward(List<double> x) => _addBias(_matVec(W, x), b);
}

class _MLP {
  final List<_Linear> layers; // all hidden layers (tanh activations between)
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

/// ======================= Intents (mirror training enum) =======================
enum Intent { hover, goLeft, goRight, descendSlow, brakeUp }
const List<String> kIntentNames = ['hover','goLeft','goRight','descendSlow','brakeUp'];

/// ===== UI-side low-level controller (no GameEngine dependency!) =====
/// Converts an intent into (thrust,left,right) using only Lander/Terrain/world.
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
  final ceilingDist = (lander.pos.y - 0.0).clamp(0.0, 1e9); // UI ceiling at y=0

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
    Intent.goLeft   => -vxGoalAbs,
    Intent.goRight  =>  vxGoalAbs,
    Intent.hover    => -kDxHover * dx,
    _               => 0.0,
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
  final bool lateralIntent = intent == Intent.goLeft || intent == Intent.goRight;

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

  // UI hard ceiling at y=0
  if (lander.pos.y < 4) thrust = false;

  return (thrust: thrust, left: left, right: right);
}

/// =================== Two-stage runtime policy (planner) ===================
class RuntimeTwoStagePolicy {
  // Architecture (inferred)
  final int inputSize;
  final _MLP trunk; // arbitrary hidden layers (tanh)
  final _Linear headIntent; // (K, Hlast)
  // (Optionally loaded; unused by UI planner, kept for compatibility)
  final _Linear? headTurn;  // (3, Hlast)
  final _Linear? headThr;   // (1, Hlast)
  final _Linear? headVal;   // (1, Hlast)

  // FE & planner config
  final RuntimeFeatureExtractor fe;
  final int planHold; // frames to hold an intent

  // Saved feature normalization (optional)
  final _RuntimeNorm? norm;
  final String? signature;

  // Planner state
  int _framesLeft = 0;
  int _currentIntentIdx = -1;
  List<double>? _lastReplanProbs;

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
  });

  /// Create from JSON string. Supports:
  ///  - v2 format: { arch.hidden, trunk: [{W,b},...], heads: {intent,turn,thr,val}, signature, norm{...} }
  ///  - legacy v1: inputSize/h1/h2, W1/W2/b1/b2, W_intent/b_intent, ...
  static RuntimeTwoStagePolicy fromJson(
      String jsonString, {
        RuntimeFeatureExtractor? fe,
        int planHold = 1,
      }) {
    final Map<String, dynamic> j = json.decode(jsonString);

    List<List<double>> _as2d(dynamic v) =>
        (v as List).map<List<double>>((r) => (r as List).map<double>((x)=> (x as num).toDouble()).toList()).toList();
    List<double> _as1d(dynamic v) =>
        (v as List).map<double>((x)=> (x as num).toDouble()).toList();

    _RuntimeNorm? _readNorm(Map<String, dynamic> root, int expectDim, String? expectSig) {
      // prefer nested 'norm'
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
      // fallback to legacy top-level triplet (guard by signature if present)
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

    // v2 path?
    final arch = (j['arch'] as Map?)?.cast<String, dynamic>();
    final trunkJ = (j['trunk'] as List?)?.cast<dynamic>();
    final headsJ = (j['heads'] as Map?)?.cast<String, dynamic>();
    final v2 = (arch != null && trunkJ != null && headsJ != null);

    if (v2) {
      final inputSize = (arch!['input'] as num).toInt();
      final hidden = (arch['hidden'] as List).map((e) => (e as num).toInt()).toList();
      final sig = j['signature'] as String?;

      // build trunk
      final layers = <_Linear>[];
      int expectIn = inputSize;
      for (int li = 0; li < trunkJ!.length; li++) {
        final layerObj = (trunkJ[li] as Map).cast<String, dynamic>();
        final W = _as2d(layerObj['W']);
        final b = _as1d(layerObj['b']);
        // shape check (best effort)
        if (W.isEmpty || W[0].length != expectIn || W.length != b.length) {
          throw StateError('Trunk layer $li has mismatched shape (got ${W.length}x${W[0].length}, bias ${b.length}, expected in=$expectIn).');
        }
        layers.add(_Linear(W, b));
        expectIn = b.length;
      }
      final trunk = _MLP(layers);
      final lastDim = trunk.outDim;

      // heads
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

      final fe0 = fe ??
          RuntimeFeatureExtractor(
            groundSamples: ((j['feature_extractor']?['groundSamples'] ?? 3) as num).toInt(),
            stridePx: ((j['feature_extractor']?['stridePx'] ?? 48) as num).toDouble(),
          );

      final norm = _readNorm(j, fe0.inputSize, sig);

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

    // ----- Legacy v1 loader (W1/W2/... with two hidden tanh layers) -----
    final inputSize = (j['inputSize'] as num).toInt();
    final h1 = (j['h1'] as num).toInt();
    final h2 = (j['h2'] as num).toInt();

    final W1 = _as2d(j['W1']); final b1 = _as1d(j['b1']);
    final W2 = _as2d(j['W2']); final b2 = _as1d(j['b2']);

    final W_intent = _as2d(j['W_intent']); final b_intent = _as1d(j['b_intent']);

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

    // Build equivalent trunk (two layers, tanh)
    final trunk = _MLP([_Linear(W1, b1), _Linear(W2, b2)]);
    final headIntent = _Linear(W_intent, b_intent);

    final fe0 = fe ??
        RuntimeFeatureExtractor(
          groundSamples: ((j['fe']?['groundSamples'] ?? j['feature_extractor']?['groundSamples'] ?? 3) as num).toInt(),
          stridePx: ((j['fe']?['stridePx'] ?? j['feature_extractor']?['stridePx'] ?? 48) as num).toDouble(),
        );

    final sig = j['signature'] as String?;
    final norm = _readNorm(j, fe0.inputSize, sig);

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

  /// Reset internal planner so next call will re-plan immediately.
  void resetPlanner() { _framesLeft = 0; _currentIntentIdx = -1; }

  /// Main runtime step: choose/hold intent, emit control, publish bus events.
  ///
  /// Returns low-level control AND the chosen intent index + probs for zero-lag HUD.
  (bool thrust, bool left, bool right, int intentIdx, List<double> probs) actWithIntent({
    required LanderState lander,
    required Terrain terrain,
    required double worldW,
    required double worldH,
    int step = 0,
    double uiMaxFuel = 100.0, // match what you pass to FE at training-time if needed
  }) {
    // Re-plan if needed
    if (_framesLeft <= 0) {
      var x = fe.extract(
        lander: lander,
        terrain: terrain,
        worldW: worldW,
        worldH: worldH,
        uiMaxFuel: uiMaxFuel,
      );
      if (norm != null) x = norm!.apply(x);

      // trunk forward
      final h = trunk.forward(x);

      // intent head (greedy at runtime)
      final logits = headIntent.forward(h);
      final probs = _softmax(logits);
      final idx = _argmax(probs);

      _currentIntentIdx = idx;
      _framesLeft = planHold;
      _lastReplanProbs = probs;

      // Publish intent (UI/debug bus)
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
    final ctrl = _controllerForIntentUI(
      intent,
      lander: lander,
      terrain: terrain,
      worldW: worldW,
      worldH: worldH,
    );

    // Publish control (UI/debug bus)
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
