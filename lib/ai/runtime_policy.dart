// lib/ai/runtime_policy.dart
import 'dart:math' as math;
import 'dart:convert';

/// ===== Lightweight runtime policy for on-device inference (no training) =====
/// Architecture: Input -> ReLU(64) -> ReLU(64) -> two softmax heads:
///   - throttle: 3 classes (off/low/high) mapped to power {0.0, 0.8, 1.0}
///   - turn:     3 classes (none/left/right)
///
/// We load weights from a JSON asset (see format at bottom).

double _relu(double x) => x > 0 ? x : 0.0;

List<double> _matVec(List<List<double>> W, List<double> x) {
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

List<double> _vecAddInPlace(List<double> a, List<double> b) {
  for (int i = 0; i < a.length; i++) a[i] += b[i];
  return a;
}

List<double> _reluVec(List<double> v) => v.map(_relu).toList();

List<double> _softmax(List<double> z) {
  double m = z[0];
  for (int i = 1; i < z.length; i++) if (z[i] > m) m = z[i];
  double sum = 0.0;
  final exps = List<double>.filled(z.length, 0.0);
  for (int i = 0; i < z.length; i++) {
    final e = math.exp(z[i] - m);
    exps[i] = e; sum += e;
  }
  for (int i = 0; i < z.length; i++) exps[i] /= sum;
  return exps;
}

/// Extracts the same features your trainer used (sin/cos angle + local ground).
class RuntimeFeatureExtractor {
  final int groundSamples;
  final double stridePx;
  RuntimeFeatureExtractor({this.groundSamples = 3, this.stridePx = 48});

  /// Game types are your in-game classes (not the engine file):
  ///  - [lander] has: position Offset, velocity Offset, angle (rads), fuel
  ///  - [terrain] has: heightAt(x), padX1, padX2, padY
  ///  - [worldW], [worldH] from Layout constraints
  List<double> extract({
    required dynamic lander,   // expects fields: position, velocity, angle, fuel
    required dynamic terrain,  // expects methods/fields: heightAt(x), padX1,padX2,padY
    required double worldW,
    required double worldH,
  }) {
    double clamp01(double v) => v < 0 ? 0 : (v > 1 ? 1 : v);

    final pos = lander.position; // Offset
    final vel = lander.velocity; // Offset
    final px = clamp01(pos.dx / worldW);
    final py = clamp01(pos.dy / worldH);
    final vx = (vel.dx / 200.0).clamp(-1.2, 1.2);
    final vy = (vel.dy / 200.0).clamp(-1.2, 1.2);

    final sinA = math.sin(lander.angle);
    final cosA = math.cos(lander.angle);

    // If you have Tunables in UI, pass maxFuel via caller; here we assume lander.fuel already scaled 0..max
    final double maxFuel = 100.0;
    final fuel = clamp01(lander.fuel / maxFuel);

    final padCenter = clamp01(((terrain.padX1 + terrain.padX2) * 0.5) / worldW);
    final dxCenter = ((pos.dx - (terrain.padX1 + terrain.padX2) * 0.5) / worldW).clamp(-1.0, 1.0);

    final groundY = terrain.heightAt(pos.dx);
    final dGround = ((groundY - pos.dy) / worldH).clamp(-1.0, 1.0);

    final gyL = terrain.heightAt(math.max(0, pos.dx - 20));
    final gyR = terrain.heightAt(math.min(worldW, pos.dx + 20));
    final slope = (((gyR - gyL) / 40.0) / 0.5).clamp(-1.5, 1.5);

    final count = groundSamples;
    final half = (count - 1) ~/ 2;
    final samples = <double>[];
    for (int i = -half; i <= half; i++) {
      final sx = (pos.dx + i * stridePx).clamp(0.0, worldW);
      final sy = terrain.heightAt(sx);
      final rel = ((sy - pos.dy) / worldH).clamp(-1.0, 1.0);
      samples.add(rel);
    }

    return [px, py, vx, vy, sinA, cosA, fuel, padCenter, dxCenter, dGround, slope, ...samples];
  }

  int get inputSize => 11 + groundSamples;
}

/// Lightweight policy just for inference.
class RuntimePolicy {
  final List<List<double>> W1, W2, W_thr, W_turn;
  final List<double> b1, b2, b_thr, b_turn, W_val;
  final double b_val;

  final RuntimeFeatureExtractor fe;

  RuntimePolicy({
    required this.W1,
    required this.b1,
    required this.W2,
    required this.b2,
    required this.W_thr,
    required this.b_thr,
    required this.W_turn,
    required this.b_turn,
    required this.W_val,
    required this.b_val,
    required this.fe,
  });

  static RuntimePolicy fromJson(String jsonString, {RuntimeFeatureExtractor? fe}) {
    final m = json.decode(jsonString) as Map<String, dynamic>;
    List<List<double>> _mm(List list) =>
        (list as List).map<List<double>>((r) => (r as List).map<double>((v) => (v as num).toDouble()).toList()).toList();
    List<double> _vv(List list) => (list as List).map<double>((v) => (v as num).toDouble()).toList();

    return RuntimePolicy(
      W1: _mm(m['W1']),
      b1: _vv(m['b1']),
      W2: _mm(m['W2']),
      b2: _vv(m['b2']),
      W_thr: _mm(m['W_thr']),
      b_thr: _vv(m['b_thr']),
      W_turn: _mm(m['W_turn']),
      b_turn: _vv(m['b_turn']),
      W_val: _vv(m['W_val']),
      b_val: (m['b_val'] as num).toDouble(),
      fe: fe ?? RuntimeFeatureExtractor(),
    );
  }

  /// Returns (thrust, left, right)
  (bool thrust, bool left, bool right) act({
    required dynamic lander,
    required dynamic terrain,
    required double worldW,
    required double worldH,
  }) {
    final x = fe.extract(lander: lander, terrain: terrain, worldW: worldW, worldH: worldH);

    // Trunk
    final z1 = _vecAddInPlace(_matVec(W1, x), b1);
    final h1 = _reluVec(z1);
    final z2 = _vecAddInPlace(_matVec(W2, h1), b2);
    final h2 = _reluVec(z2);

    // Heads
    final thrLogits = _vecAddInPlace(_matVec(W_thr, h2), b_thr);
    final thrP = _softmax(thrLogits);
    final turnLogits = _vecAddInPlace(_matVec(W_turn, h2), b_turn);
    final turnP = _softmax(turnLogits);

    // Sample (simple argmax for runtime stability)
    int _argmax(List<double> p) {
      int im = 0; double bm = p[0];
      for (int i = 1; i < p.length; i++) { if (p[i] > bm) { bm = p[i]; im = i; } }
      return im;
    }

    final thrC = _argmax(thrP);   // 0 off, 1 low, 2 high
    final turnC = _argmax(turnP); // 0 none, 1 left, 2 right

    // Map to your game's boolean controls
    final power = (thrC == 0) ? 0.0 : (thrC == 1 ? 0.8 : 1.0);
    final thrust = power > 0.0;
    final left = turnC == 1;
    final right = turnC == 2;

    return (thrust, left, right);
  }
}
