// lib/ai/runtime_policy.dart
import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/services.dart' show rootBundle;

// UI types
import 'package:flutter/material.dart' show Offset;
import '../game_page.dart' show Lander, Terrain;

// Intent bus (runtime)
import 'intent_bus.dart';

/// ================= Runtime features (must mirror training) =================
class RuntimeFeatureExtractor {
  final int groundSamples;
  final double stridePx;
  const RuntimeFeatureExtractor({this.groundSamples = 3, this.stridePx = 48});

  /// Training layout (IMPORTANT):
  /// [px, py, vx, vy, ang, fuel, padCenter, dxCenter, dGround, slope, samples...]
  List<double> extract({
    required Lander lander,
    required Terrain terrain,
    required double worldW,
    required double worldH,
  }) {
    final px = lander.position.dx / worldW;
    final py = lander.position.dy / worldH;
    final vx = (lander.velocity.dx / 200.0).clamp(-2.0, 2.0);
    final vy = (lander.velocity.dy / 200.0).clamp(-2.0, 2.0);
    final ang = (lander.angle / math.pi).clamp(-1.5, 1.5); // angle value, not sin/cos
    final fuel = (lander.fuel / 100.0).clamp(0.0, 1.0);    // Tunables.maxFuel=100 in UI

    final padCenter = ((terrain.padX1 + terrain.padX2) * 0.5) / worldW;
    final dxCenter =
    ((lander.position.dx - (terrain.padX1 + terrain.padX2) * 0.5) / worldW)
        .clamp(-1.0, 1.0);

    final gY = terrain.heightAt(lander.position.dx);
    final dGround = ((gY - lander.position.dy) / worldH).clamp(-1.0, 1.0);

    final gyL = terrain.heightAt((lander.position.dx - 20).clamp(0.0, worldW));
    final gyR = terrain.heightAt((lander.position.dx + 20).clamp(0.0, worldW));
    final slope = (((gyR - gyL) / 40.0) / 0.5).clamp(-2.0, 2.0);

    // local ground samples: exactly `groundSamples` with even/odd symmetry
    final n = groundSamples;
    final center = (n - 1) / 2.0;
    final samples = <double>[];
    for (int k = 0; k < n; k++) {
      final relIndex = k - center;
      final sx = (lander.position.dx + relIndex * stridePx).clamp(0.0, worldW);
      final sy = terrain.heightAt(sx);
      samples.add(((sy - lander.position.dy) / worldH).clamp(-1.0, 1.0));
    }

    return [px, py, vx, vy, ang, fuel, padCenter, dxCenter, dGround, slope, ...samples];
  }

  int get inputSize => 10 + groundSamples;
}

/// =============== Tiny math (pure Dart, no Flutter/engine deps) ===============
List<double> _matVec(List<List<double>> W, List<double> x) {
  final m = W.length, n = W[0].length;
  final out = List<double>.filled(m, 0.0);
  for (int i = 0; i < m; i++) {
    double s = 0.0; final Wi = W[i];
    for (int j = 0; j < n; j++) s += Wi[j] * x[j];
    out[i] = s;
  }
  return out;
}
List<double> _vecAdd(List<double> a, List<double> b) {
  final out = List<double>.filled(a.length, 0.0);
  for (int i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}
double _relu(double x) => x > 0 ? x : 0.0;
List<double> _reluVec(List<double> v) => v.map(_relu).toList();
double _sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));
List<double> _softmax(List<double> z) {
  final m = z.reduce((a,b)=>a>b?a:b);
  var s = 0.0;
  final e = List<double>.filled(z.length, 0.0);
  for (int i=0;i<z.length;i++) { e[i] = math.exp(z[i]-m); s += e[i]; }
  for (int i=0;i<z.length;i++) e[i] /= (s + 1e-12);
  return e;
}
int _argmax(List<double> a) {
  var bi = 0; var bv = a[0];
  for (int i=1;i<a.length;i++) if (a[i] > bv) { bv=a[i]; bi=i; }
  return bi;
}

/// ======================= Intents (mirror training enum) =======================
enum Intent { hoverCenter, goLeft, goRight, descendSlow, brakeUp }
const List<String> kIntentNames = ['hover','goLeft','goRight','descendSlow','brakeUp'];

/// ===== UI-side low-level controller (no GameEngine dependency!) =====
/// Converts an intent into (thrust,left,right) using only Lander/Terrain/world.
({bool thrust, bool left, bool right}) _controllerForIntentUI(
    Intent intent, {
      required Lander lander,
      required Terrain terrain,
      required double worldW,
      required double worldH,
    }) {
  final padCx = (terrain.padX1 + terrain.padX2) * 0.5;
  final dx = lander.position.dx - padCx;
  final vx = lander.velocity.dx;
  final vy = lander.velocity.dy;
  final angle = lander.angle;

  bool left = false, right = false, thrust = false;

  final groundY = terrain.heightAt(lander.position.dx);
  final height  = (groundY - lander.position.dy).clamp(0.0, 1e9);
  final ceilingDist = (lander.position.dy - 0.0).clamp(0.0, 1e9); // UI ceiling at y=0

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
    Intent.goLeft       => -vxGoalAbs,
    Intent.goRight      =>  vxGoalAbs,
    Intent.hoverCenter  => -kDxHover * dx,
    _                   => 0.0,
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
  if (lander.position.dy < 4) thrust = false;

  return (thrust: thrust, left: left, right: right);
}

/// =================== Two-stage runtime policy (planner) ===================
class RuntimeTwoStagePolicy {
  final int inputSize, h1, h2;

  // Trunk
  final List<List<double>> W1, W2;
  final List<double> b1, b2;

  // Legacy action heads (loaded for compatibility)
  final List<List<double>> W_thr, W_turn;
  final List<double> b_thr, b_turn;

  // Two-stage heads
  final List<List<double>> W_intent; // (K, h2)
  final List<double> b_intent;       // (K)

  // Critic
  final List<List<double>> W_val;    // (1, h2)
  final List<double> b_val;          // (1)

  // FE & planner config
  final RuntimeFeatureExtractor fe;
  final int planHold;                // frames to hold an intent

  // Planner state
  int _framesLeft = 0;
  int _currentIntentIdx = -1;
  List<double>? _lastReplanProbs;

  RuntimeTwoStagePolicy._({
    required this.inputSize,
    required this.h1,
    required this.h2,
    required this.W1,
    required this.b1,
    required this.W2,
    required this.b2,
    required this.W_thr,
    required this.b_thr,
    required this.W_turn,
    required this.b_turn,
    required this.W_intent,
    required this.b_intent,
    required this.W_val,
    required this.b_val,
    required this.fe,
    required this.planHold,
  });

  static RuntimeTwoStagePolicy fromJson(
      String jsonString, {
        RuntimeFeatureExtractor? fe,
        int planHold = 12,
      }) {
    final Map<String, dynamic> j = json.decode(jsonString);

    List<List<double>> _as2d(dynamic v) =>
        (v as List).map<List<double>>((r) => (r as List).map<double>((x)=> (x as num).toDouble()).toList()).toList();
    List<double> _as1d(dynamic v) =>
        (v as List).map<double>((x)=> (x as num).toDouble()).toList();

    final inputSize = j['inputSize'] as int;
    final h1 = j['h1'] as int;
    final h2 = j['h2'] as int;

    final W1 = _as2d(j['W1']); final b1 = _as1d(j['b1']);
    final W2 = _as2d(j['W2']); final b2 = _as1d(j['b2']);
    final W_thr = _as2d(j['W_thr']); final b_thr = _as1d(j['b_thr']);
    final W_turn = _as2d(j['W_turn']); final b_turn = _as1d(j['b_turn']);

    // Two-stage bits (must exist for planner runtime)
    final W_intent = _as2d(j['W_intent']);
    final b_intent = _as1d(j['b_intent']);
    final W_val = _as2d(j['W_val']);
    final b_val = _as1d(j['b_val']);

    final fe0 = fe ??
        RuntimeFeatureExtractor(
          groundSamples: (j['fe']?['groundSamples'] ?? 3) as int,
          stridePx: ((j['fe']?['stridePx'] ?? 48) as num).toDouble(),
        );

    // shape checks
    void _expect(bool c, String m) { if (!c) throw StateError(m); }
    _expect(W1.length == h1 && W1[0].length == inputSize, 'W1 shape');
    _expect(b1.length == h1, 'b1 len');
    _expect(W2.length == h2 && W2[0].length == h1, 'W2 shape');
    _expect(b2.length == h2, 'b2 len');
    _expect(W_thr.length == 1 && W_thr[0].length == h2, 'W_thr shape');
    _expect(b_thr.length == 1, 'b_thr len');
    _expect(W_turn.length == 3 && W_turn[0].length == h2, 'W_turn shape');
    _expect(b_turn.length == 3, 'b_turn len');
    _expect(W_intent[0].length == h2, 'W_intent shape');
    _expect(b_intent.length == W_intent.length, 'b_intent len');
    _expect(W_val.length == 1 && W_val[0].length == h2, 'W_val shape');
    _expect(b_val.length == 1, 'b_val len');
    _expect(fe0.inputSize == inputSize, 'feature size mismatch');

    return RuntimeTwoStagePolicy._(
      inputSize: inputSize, h1: h1, h2: h2,
      W1: W1, b1: b1, W2: W2, b2: b2,
      W_thr: W_thr, b_thr: b_thr, W_turn: W_turn, b_turn: b_turn,
      W_intent: W_intent, b_intent: b_intent,
      W_val: W_val, b_val: b_val,
      fe: fe0, planHold: planHold,
    );
  }

  static Future<RuntimeTwoStagePolicy> loadFromAsset(String assetPath, {int planHold = 12}) async {
    final js = await rootBundle.loadString(assetPath);
    return RuntimeTwoStagePolicy.fromJson(js, planHold: planHold);
  }

  /// Reset internal planner so next call will re-plan immediately.
  void resetPlanner() { _framesLeft = 0; _currentIntentIdx = -1; }

  /// Main runtime step: choose/hold intent, emit control, publish bus events.
  ///
  /// Returns low-level control AND the chosen intent index + probs for zero-lag HUD.
  (bool thrust, bool left, bool right, int intentIdx, List<double> probs) actWithIntent({
    required Lander lander,
    required Terrain terrain,
    required double worldW,
    required double worldH,
    int step = 0,
  }) {
    // Re-plan if needed
    if (_framesLeft <= 0) {
      final x = fe.extract(lander: lander, terrain: terrain, worldW: worldW, worldH: worldH);

      // trunk
      final z1 = _vecAdd(_matVec(W1, x), b1);
      final h1v = _reluVec(z1);
      final z2 = _vecAdd(_matVec(W2, h1v), b2);
      final h2v = _reluVec(z2);

      // intent head (greedy at runtime)
      final logits = _vecAdd(_matVec(W_intent, h2v), b_intent);
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
