// lib/ai/nn_helper.dart
import 'dart:math' as math;

/// Simple activation helpers (tanh for hidden layers).
class Act {
  static double tanh(double x) {
    final ax = x.abs();
    if (ax > 20.0) return x.isNegative ? -1.0 : 1.0;
    final e = math.exp(-2.0 * ax);
    final t = (1.0 - e) / (1.0 + e);
    return x.isNegative ? -t : t;
  }

  /// Derivative of tanh when you already have y = tanh(x).
  static double dtanhFromY(double y) => 1.0 - y * y;
}

/// Fully-connected linear layer: y = W*x + b
class Linear {
  List<List<double>> W; // [out][in]
  List<double> b;       // [out]

  Linear(int out, int inp, {int seed = 0})
      : W = _xavier(h: out, w: inp, seed: seed),
        b = List<double>.filled(out, 0.0);

  static List<List<double>> _xavier({required int h, required int w, int seed = 0}) {
    final r = math.Random(seed);
    final limit = math.sqrt(6.0 / (h + w));
    return List.generate(h, (_) => List<double>.generate(w, (_) => (r.nextDouble() * 2 - 1) * limit));
  }

  List<double> forward(List<double> x) {
    final out = List<double>.filled(W.length, 0.0);
    for (int i = 0; i < W.length; i++) {
      double s = b[i];
      final wi = W[i];
      for (int j = 0; j < x.length; j++) s += wi[j] * x[j];
      out[i] = s;
    }
    return out;
  }

  /// Accumulate grads (dL/dW, dL/db) and return dL/dx.
  List<double> backward({
    required List<double> x,
    required List<double> dOut, // dL/d(out)
    required List<List<double>> gW,
    required List<double> gb,
  }) {
    final inDim = x.length;
    final outDim = dOut.length;

    // grads
    for (int i = 0; i < outDim; i++) {
      gb[i] += dOut[i];
      final wi = gW[i];
      final d = dOut[i];
      for (int j = 0; j < inDim; j++) wi[j] += d * x[j];
    }

    // propagate to previous
    final dx = List<double>.filled(inDim, 0.0);
    for (int j = 0; j < inDim; j++) {
      double s = 0.0;
      for (int i = 0; i < outDim; i++) s += W[i][j] * dOut[i];
      dx[j] = s;
    }
    return dx;
  }
}

/// MLP trunk with arbitrary hidden sizes (tanh activations).
class MLPTrunk {
  final List<Linear> layers;
  MLPTrunk._(this.layers);

  factory MLPTrunk({required int inputSize, required List<int> hiddenSizes, int seed = 0}) {
    final layers = <Linear>[];
    var inp = inputSize;
    for (int i = 0; i < hiddenSizes.length; i++) {
      layers.add(Linear(hiddenSizes[i], inp, seed: seed ^ (0x1000 + i)));
      inp = hiddenSizes[i];
    }
    return MLPTrunk._(layers);
  }

  /// Returns all activations (a0=input, a1=tanh(z1), ..., aL).
  List<List<double>> forwardAll(List<double> x) {
    final acts = <List<double>>[List<double>.from(x)];
    var h = x;
    for (final lin in layers) {
      final z = lin.forward(h);
      for (int i = 0; i < z.length; i++) z[i] = Act.tanh(z[i]);
      acts.add(z);
      h = z;
    }
    return acts;
  }

  /// Backward from gradient at top activation (dL/d aL). Accumulates into gW/gb.
  void backwardFromTopGrad({
    required List<double> dTop,
    required List<List<double>> acts,
    required List<List<List<double>>> gW,
    required List<List<double>> gb,
  }) {
    var d = List<double>.from(dTop); // dL/d aL
    for (int li = layers.length - 1; li >= 0; li--) {
      final aPrev = acts[li];     // a_{l-1}
      final a     = acts[li + 1]; // a_l (tanh output)
      for (int k = 0; k < d.length; k++) d[k] *= Act.dtanhFromY(a[k]); // chain through tanh
      d = layers[li].backward(x: aPrev, dOut: d, gW: gW[li], gb: gb[li]);
    }
  }
}

/// Action/Value/Intent heads that sit on top of the trunk’s last hidden layer.
class PolicyHeads {
  final Linear intent; // 5-way logits
  final Linear turn;   // 3-way logits
  final Linear thr;    // 1 logit (sigmoid)
  final Linear val;    // 1 value

  PolicyHeads(int hidden, {int seed = 0})
      : intent = Linear(5, hidden, seed: seed ^ 0x3001),
        turn   = Linear(3, hidden, seed: seed ^ 0x3002),
        thr    = Linear(1, hidden, seed: seed ^ 0x3003),
        val    = Linear(1, hidden, seed: seed ^ 0x3004);
}

// Place below the imports and above/alongside Act/Linear/MLPTrunk.

class Ops {
  /// Stable softmax (max-shift); returns probabilities.
  static List<double> softmax(List<double> z) {
    if (z.isEmpty) return <double>[];
    double m = z[0];
    for (int i = 1; i < z.length; i++) if (z[i] > m) m = z[i];

    double s = 0.0;
    final out = List<double>.filled(z.length, 0.0);
    for (int i = 0; i < z.length; i++) {
      final e = math.exp(z[i] - m);
      out[i] = e.isFinite ? e : 0.0;
      s += out[i];
    }
    if (s <= 0 || !s.isFinite) return List<double>.filled(z.length, 0.0);
    final inv = 1.0 / s;
    for (int i = 0; i < z.length; i++) out[i] *= inv;
    return out;
  }

  /// Stable sigmoid.
  static double sigmoid(double x) {
    if (x >= 0) {
      final ex = math.exp(-x);
      return 1.0 / (1.0 + ex);
    } else {
      final ex = math.exp(x);
      return ex / (1.0 + ex);
    }
  }

  /// Logit(p) with clamping for numerical safety.
  static double logit(double p, {double eps = 1e-6}) {
    final q = p.clamp(eps, 1.0 - eps);
    return math.log(q / (1.0 - q));
  }

  /// Cross-entropy grad for a single target class: grad = (p - y_onehot).
  static List<double> crossEntropyGrad(List<double> probs, int target, {int? numClasses}) {
    final k = numClasses ?? probs.length;
    final g = List<double>.filled(k, 0.0);
    for (int i = 0; i < k; i++) g[i] = (i < probs.length ? probs[i] : 0.0);
    if (target >= 0 && target < k) g[target] -= 1.0;
    return g;
  }

  /// Cross-entropy loss given logits and target (uses softmax internally).
  static double crossEntropyLossFromLogits(List<double> logits, int target, {double eps = 1e-12}) {
    if (target < 0 || target >= logits.length) return 0.0;
    final p = softmax(logits);
    return -math.log((p[target]).clamp(eps, 1.0));
  }

  /// BCE gradient wrt logit: sigmoid(logit) - y  (y in {0,1}).
  static double bceGradFromLogit(double logit, double y) => sigmoid(logit) - y;
}
