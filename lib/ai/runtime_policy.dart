// lib/ai/runtime_policy.dart
import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/services.dart' show rootBundle;

// Game data types from your UI side
import 'package:flutter/material.dart' show Offset;
import 'package:flutter_application_1/engine/game_engine.dart';
import '../game_page.dart' show Lander, Terrain;

// Two-stage bits and bus (reuse training controller + intent names)
import 'agent.dart' show Intent, kIntentNames, controllerForIntent, intentToIndex, indexToIntent;
import 'intent_bus.dart';

/// === Runtime features (MIRRORS TRAINING: 10 + groundSamples) ===
/// Training FeatureExtractor used:
/// [px,py,vx,vy,ang,fuel, padCenter,dxCenter, dGround,slope, ...groundSamples]
class RuntimeFeatureExtractor {
  final int groundSamples;
  final double stridePx;
  const RuntimeFeatureExtractor({this.groundSamples = 3, this.stridePx = 48});

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
    final ang = (lander.angle / math.pi).clamp(-1.5, 1.5); // MATCH training
    // If you expose actual maxFuel, replace 100/140 accordingly; training used cfg.t.maxFuel.
    final fuel = (lander.fuel / 100.0).clamp(0.0, 1.0);

    final padCenter = ((terrain.padX1 + terrain.padX2) * 0.5) / worldW;
    final dxCenter = ((lander.position.dx - (terrain.padX1 + terrain.padX2) * 0.5) / worldW)
        .clamp(-1.0, 1.0);

    double groundY(double x) => terrain.heightAt(x);
    final gY = groundY(lander.position.dx);
    final dGround = ((gY - lander.position.dy) / worldH).clamp(-1.0, 1.0);

    final gyL = groundY((lander.position.dx - 20).clamp(0.0, worldW));
    final gyR = groundY((lander.position.dx + 20).clamp(0.0, worldW));
    final slope = (((gyR - gyL) / 40.0) / 0.5).clamp(-2.0, 2.0);

    // symmetric local samples (works for even/odd n)
    final n = groundSamples;
    final samples = <double>[];
    final center = (n - 1) / 2.0;
    for (int k = 0; k < n; k++) {
      final relIndex = k - center;
      final sx = (lander.position.dx + relIndex * stridePx).clamp(0.0, worldW);
      final sy = groundY(sx);
      final rel = ((sy - lander.position.dy) / worldH).clamp(-1.0, 1.0);
      samples.add(rel);
    }

    return [
      px, py, vx, vy, ang, fuel,
      padCenter, dxCenter,
      dGround, slope,
      ...samples,
    ];
  }

  int get inputSize => 10 + groundSamples;
}

/// === Tiny math ===
List<double> matVec(List<List<double>> W, List<double> x) {
  final m = W.length, n = W[0].length;
  final out = List<double>.filled(m, 0.0);
  for (int i = 0; i < m; i++) {
    double s = 0.0; final Wi = W[i];
    for (int j = 0; j < n; j++) s += Wi[j] * x[j];
    out[i] = s;
  }
  return out;
}
List<double> vecAdd(List<double> a, List<double> b) {
  final out = List<double>.filled(a.length, 0.0);
  for (int i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}
double relu(double x) => x > 0 ? x : 0.0;
List<double> reluVec(List<double> v) => v.map(relu).toList();
double sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));
List<double> softmax(List<double> z) {
  final m = z.reduce((a,b)=>a>b?a:b);
  double sum = 0.0;
  final e = List<double>.filled(z.length, 0.0);
  for (int i=0;i<z.length;i++) { e[i] = math.exp(z[i]-m); sum += e[i]; }
  for (int i=0;i<z.length;i++) e[i] /= (sum + 1e-12);
  return e;
}
int argmax(List<double> a) {
  var bi = 0; var bv = a[0];
  for (int i=1;i<a.length;i++) { if (a[i] > bv) { bv = a[i]; bi = i; } }
  return bi;
}

/// === Two-stage runtime policy (trunk -> intent(K) head; controller does actions) ===
class RuntimeTwoStagePolicy {
  final int inputSize, h1, h2;

  // Trunk
  late final List<List<double>> W1, W2;
  late final List<double> b1, b2;

  // Intent head (K x h2)
  late final List<List<double>> W_intent;
  late final List<double> b_intent;
  late final int numIntents;

  // (Optional) single-stage heads if you still want to expose them for debugging
  late final List<List<double>>? W_thr;
  late final List<double>? b_thr;
  late final List<List<double>>? W_turn;
  late final List<double>? b_turn;

  // FE + planner state
  final RuntimeFeatureExtractor fe;
  final int planHold;           // frames to hold selected intent
  int _framesLeft = 0;          // countdown
  int _currentIntentIdx = -1;   // last chosen intent

  RuntimeTwoStagePolicy._({
    required this.inputSize,
    required this.h1,
    required this.h2,
    required this.W1,
    required this.b1,
    required this.W2,
    required this.b2,
    required this.W_intent,
    required this.b_intent,
    required this.numIntents,
    required this.fe,
    required this.planHold,
    this.W_thr,
    this.b_thr,
    this.W_turn,
    this.b_turn,
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

    // Two-stage: require intent head
    if (!j.containsKey('W_intent') || !j.containsKey('b_intent')) {
      throw StateError('Policy JSON lacks W_intent/b_intent (two-stage).');
    }
    final W_intent = _as2d(j['W_intent']);
    final b_intent = _as1d(j['b_intent']);
    final numIntents = W_intent.length;

    // Optional single-stage heads (may or may not exist)
    List<List<double>>? W_thr, W_turn;
    List<double>? b_thr, b_turn;
    if (j.containsKey('W_thr') && j.containsKey('b_thr') &&
        j.containsKey('W_turn') && j.containsKey('b_turn')) {
      W_thr  = _as2d(j['W_thr']);  b_thr  = _as1d(j['b_thr']);
      W_turn = _as2d(j['W_turn']); b_turn = _as1d(j['b_turn']);
    }

    final fe0 = fe ?? RuntimeFeatureExtractor(
      groundSamples: (j['fe']?['groundSamples'] ?? 3) as int,
      stridePx: ((j['fe']?['stridePx'] ?? 48) as num).toDouble(),
    );

    // shape checks
    void _expect(bool c, String m) { if (!c) throw StateError(m); }
    _expect(W1.length == h1 && W1[0].length == inputSize, 'W1 shape');
    _expect(b1.length == h1, 'b1 len');
    _expect(W2.length == h2 && W2[0].length == h1, 'W2 shape');
    _expect(b2.length == h2, 'b2 len');
    _expect(W_intent[0].length == h2, 'W_intent cols must equal h2');
    _expect(b_intent.length == numIntents, 'b_intent len');
    _expect(fe0.inputSize == inputSize, 'feature size mismatch (runtime vs training)');

    return RuntimeTwoStagePolicy._(
      inputSize: inputSize, h1: h1, h2: h2,
      W1: W1, b1: b1, W2: W2, b2: b2,
      W_intent: W_intent, b_intent: b_intent, numIntents: numIntents,
      fe: fe0, planHold: planHold,
      W_thr: W_thr, b_thr: b_thr, W_turn: W_turn, b_turn: b_turn,
    );
  }

  static Future<RuntimeTwoStagePolicy> loadFromAsset(
      String assetPath, {
        RuntimeFeatureExtractor? fe,
        int planHold = 12,
      }) async {
    final js = await rootBundle.loadString(assetPath);
    return RuntimeTwoStagePolicy.fromJson(js, fe: fe, planHold: planHold);
  }

  /// Reset the planner (e.g., on episode reset or user toggling AI)
  void resetPlanner() {
    _framesLeft = 0;
    _currentIntentIdx = -1;
  }

  /// Forward trunk + intent head → pick an intent (greedy)
  int _selectIntentGreedy(List<double> x) {
    // trunk
    final z1 = vecAdd(matVec(W1, x), b1);
    final h1v = reluVec(z1);
    final z2 = vecAdd(matVec(W2, h1v), b2);
    final h2v = reluVec(z2);

    // intent head
    final logits = matVec(W_intent, h2v);
    for (int i = 0; i < logits.length; i++) logits[i] += b_intent[i];
    final p = softmax(logits);
    final idx = argmax(p);
    // publish probs for UI visibility
    IntentBus.instance.publishIntent(IntentEvent(
      intent: kIntentNames[idx],
      probs: p,
      step: 0,
      meta: {'source': 'runtime', 'mode': 'greedy'},
    ));
    return idx;
  }

  /// Act: maintain an intent for `planHold` frames, emit control via controller
  /// Returns (thrust,left,right) for this frame and publishes intents/controls.
  (bool thrust, bool left, bool right) actWithIntent({
    required Lander lander,
    required Terrain terrain,
    required double worldW,
    required double worldH,
    int step = 0,
  }) {
    // (Re)plan if needed
    if (_framesLeft <= 0) {
      final x = fe.extract(lander: lander, terrain: terrain, worldW: worldW, worldH: worldH);
      _currentIntentIdx = _selectIntentGreedy(x);
      _framesLeft = planHold;
    }

    final intent = indexToIntent(_currentIntentIdx);

    // Heartbeat publish so late subscribers see current intent
    IntentBus.instance.publishIntent(IntentEvent(
      intent: kIntentNames[_currentIntentIdx],
      probs: const [], // already published on (re)plan
      step: step,
      meta: {'hold': true, 'framesLeft': _framesLeft},
    ));

    // Low-level control from the same controller used in training
    final u = controllerForIntent(intent, _RuntimeEnvView(lander, terrain, worldW, worldH) as GameEngine);

    // Publish control too (optional, great for HUD)
    IntentBus.instance.publishControl(ControlEvent(
      thrust: u.thrust, left: u.left, right: u.right, step: step,
      meta: {'intent': kIntentNames[_currentIntentIdx]},
    ));

    _framesLeft -= 1;
    return (u.thrust, u.left, u.right);
  }
}

/// Minimal adapter so controllerForIntent() (expects training env) can read state.
/// We only provide what's needed: lander pos/vel/angle/fuel, terrain pad + heightAt.
class _RuntimeEnvView {
  final _RLander lander;
  final _RTerrain terrain;
  final _RCfg cfg;
  _RuntimeEnvView(Lander L, Terrain T, double worldW, double worldH)
      : lander = _RLander(L),
        terrain = _RTerrain(T),
        cfg = _RCfg(worldW: worldW, worldH: worldH);
}
class _RLander {
  final _V2 pos;
  final _V2 vel;
  final double angle;
  final double fuel;
  _RLander(Lander L)
      : pos = _V2(L.position.dx, L.position.dy),
        vel = _V2(L.velocity.dx, L.velocity.dy),
        angle = L.angle,
        fuel = L.fuel.toDouble();
}
class _V2 { final double x,y; const _V2(this.x,this.y); }
class _RTerrain {
  final double padCenter;
  final Terrain _t;
  _RTerrain(this._t)
      : padCenter = ((_t.padX1 + _t.padX2) * 0.5);
  double heightAt(double x) => _t.heightAt(x);
}
class _RCfg {
  final double worldW, worldH;
  // Provide a ceiling margin similar to engine cfg (use small default)
  final double ceilingMargin;
  _RCfg({required this.worldW, required this.worldH, this.ceilingMargin = 8.0});
}
