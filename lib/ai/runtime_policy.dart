// lib/ai/runtime_policy.dart
import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/services.dart' show rootBundle;

// Adjust these imports to match your app structure
import 'package:flutter/material.dart' show Offset;
import '../game_page.dart' show Lander, Terrain;

/// === Runtime features (MUST mirror training) ===
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
    final sinA = math.sin(lander.angle);
    final cosA = math.cos(lander.angle);
    // Use your real maxFuel if you expose it; 140.0 is fine if you trained with that.
    final fuel = (lander.fuel / 140.0).clamp(0.0, 1.0);

    final padCenter = ((terrain.padX1 + terrain.padX2) * 0.5) / worldW;
    final dxCenter = ((lander.position.dx - (terrain.padX1 + terrain.padX2) * 0.5) / worldW)
        .clamp(-1.0, 1.0);

    double groundY(double x) => terrain.heightAt(x);
    final gY = groundY(lander.position.dx);
    final dGround = ((gY - lander.position.dy) / worldH).clamp(-1.0, 1.0);

    final gyL = groundY((lander.position.dx - 20).clamp(0.0, worldW));
    final gyR = groundY((lander.position.dx + 20).clamp(0.0, worldW));
    final slope = (((gyR - gyL) / 40.0) / 0.5).clamp(-2.0, 2.0);

    // local samples
    final n = groundSamples;
    final half = (n - 1) ~/ 2;
    final samples = <double>[];
    for (int i = -half; i <= half; i++) {
      final sx = (lander.position.dx + i * stridePx).clamp(0.0, worldW);
      final sy = groundY(sx);
      final rel = ((sy - lander.position.dy) / worldH).clamp(-1.0, 1.0);
      samples.add(rel);
    }

    return [
      px, py, vx, vy, sinA, cosA, fuel,
      padCenter, dxCenter, dGround, slope,
      ...samples,
    ];
  }

  int get inputSize => 11 + groundSamples;
}

/// === Tiny math ===
List<double> matVec(List<List<double>> W, List<double> x) {
  final m = W.length, n = W[0].length;
  final out = List<double>.filled(m, 0.0);
  for (int i = 0; i < m; i++) {
    double s = 0.0;
    final Wi = W[i];
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

/// === Runtime policy (shared trunk -> thrust(1) + turn(3)) ===
class RuntimePolicy {
  final int inputSize, h1, h2;
  late final List<List<double>> W1, W2;
  late final List<double> b1, b2;
  late final List<List<double>> W_thr, W_turn;
  late final List<double> b_thr, b_turn;

  final RuntimeFeatureExtractor fe;

  RuntimePolicy._(
      this.inputSize, this.h1, this.h2,
      this.W1, this.b1, this.W2, this.b2,
      this.W_thr, this.b_thr, this.W_turn, this.b_turn,
      this.fe,
      );

  static RuntimePolicy fromJson(
      String jsonString, {
        RuntimeFeatureExtractor? fe,
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
    _expect(fe0.inputSize == inputSize, 'feature size mismatch');

    return RuntimePolicy._(
      inputSize, h1, h2,
      W1, b1, W2, b2,
      W_thr, b_thr, W_turn, b_turn,
      fe0,
    );
  }

  /// Greedy/deterministic action for gameplay
  (bool thrust, bool left, bool right) act({
    required Lander lander,
    required Terrain terrain,
    required double worldW,
    required double worldH,
  }) {
    final x = fe.extract(lander: lander, terrain: terrain, worldW: worldW, worldH: worldH);

    // trunk
    final z1 = vecAdd(matVec(W1, x), b1);
    final h1v = reluVec(z1);
    final z2 = vecAdd(matVec(W2, h1v), b2);
    final h2v = reluVec(z2);

    // heads
    final thrLogits = matVec(W_thr, h2v)..[0] += b_thr[0];
    final turnLogits = vecAdd(matVec(W_turn, h2v), b_turn);

    // ==== gentle priors (greatly reduced) ====
    final angle = lander.angle;
    final dx = (lander.position.dx - ((terrain.padX1 + terrain.padX2) * 0.5)) / worldW;

    // 1) steer toward pad ONLY if clearly to one side (deadzone)
    const double kDx = 0.12;         // was 0.4: too strong
    const double dxDead = 0.05;      // ~5% screen width deadzone
    if (dx.abs() > dxDead) {
      turnLogits[1] += (dx > 0 ? kDx : -kDx) * (dx.abs());
      turnLogits[2] += (dx < 0 ? kDx : -kDx) * (dx.abs());
    }

    // 2) attitude prior ONLY if angle is meaningfully tilted
    const double kAng = 0.22;        // was 0.6: too strong
    const double angDead = 5 * math.pi / 180; // 5 deg
    if (angle.abs() > angDead) {
      final sign = angle >= 0 ? 1.0 : -1.0; // angle>0 means tilted right -> need LEFT
      final angN = (angle.abs() / (math.pi / 2)).clamp(0.0, 1.0);
      turnLogits[1] += (sign > 0 ? kAng : -kAng) * angN;
      turnLogits[2] += (sign < 0 ? kAng : -kAng) * angN;
    }

    // 3) (optional) falling-fast nudge for thrust (helps avoid “never boost”)
    final y = lander.position.dy;
    final ground = terrain.heightAt(lander.position.dx);
    final alt = (ground - y);              // px above ground
    final vy = lander.velocity.dy;         // +down
    if (alt < worldH * 0.25 && vy > 50) {  // low and falling fast
      thrLogits[0] += 0.6;                 // nudge to burn
    }
    // NOTE: Removed the top-ceiling thrust suppression — engine already penalizes it.

    // decisions
    final thrP = sigmoid(thrLogits[0]);
    final turnP = softmax(turnLogits);
    final cls = argmax(turnP);

    // Slightly lower thrust threshold so it actually burns early on
    final thrust = thrP > 0.35;  // was 0.5
    final left = cls == 1;
    final right = cls == 2;
    return (thrust, left, right);
  }

  static Future<RuntimePolicy> loadFromAsset(String assetPath) async {
    final js = await rootBundle.loadString(assetPath);
    return RuntimePolicy.fromJson(js);
  }
}
