import 'dart:math' as math;

/* --------------------------- tiny init / activations --------------------------- */

List<List<double>> _xavier(int outDim, int inDim, int seed) {
  final r = math.Random(seed);
  final lim = math.sqrt(6.0 / (outDim + inDim));
  return List.generate(
    outDim,
        (_) => List<double>.generate(inDim, (_) => (r.nextDouble() * 2 - 1) * lim),
  );
}

double _tanh(double x) {
  final ax = x.abs();
  if (ax > 20.0) return x.isNegative ? -1.0 : 1.0;
  final e = math.exp(-2.0 * ax);
  final t = (1.0 - e) / (1.0 + e);
  return x.isNegative ? -t : t;
}

/* ------------------------------------ Ops ------------------------------------ */

class Ops {
  static List<double> softmax(List<double> z) {
    double m = z[0];
    for (int i = 1; i < z.length; i++) if (z[i] > m) m = z[i];
    double s = 0.0;
    final out = List<double>.filled(z.length, 0.0);
    for (int i = 0; i < z.length; i++) {
      final e = math.exp(z[i] - m);
      out[i] = e.isFinite ? e : 0.0;
      s += out[i];
    }
    final inv = (s > 0) ? 1.0 / s : 0.0;
    for (int i = 0; i < z.length; i++) out[i] *= inv;
    return out;
  }

  static double sigmoid(double x) {
    if (x >= 0) {
      final e = math.exp(-x);
      return 1.0 / (1.0 + e);
    } else {
      final e = math.exp(x);
      return e / (1.0 + e);
    }
  }

  /// d L / d logits for CE with softmax: (p - y)
  static List<double> crossEntropyGrad(List<double> p, int y, {int? numClasses}) {
    final k = numClasses ?? p.length;
    final out = List<double>.filled(k, 0.0);
    final n = math.min(k, p.length);
    for (int i = 0; i < n; i++) out[i] = p[i];
    if (y >= 0 && y < n) out[y] -= 1.0;
    return out;
  }

  /// BCE grad wrt logit: σ(z) - y (equivalently: dL/dz)
  static double bceGradFromLogit(double logit, double y) {
    return sigmoid(logit) - y;
  }

  static double logit(double p) {
    final q = p.clamp(1e-9, 1 - 1e-9);
    return math.log(q / (1.0 - q));
  }
}

/* ----------------------------------- Linear ---------------------------------- */

class Linear {
  List<List<double>> W; // [out][in]
  List<double> b;       // [out]

  Linear(int inDim, int outDim, {int seed = 0})
      : W = _xavier(outDim, inDim, seed),
        b = List<double>.filled(outDim, 0.0);

  int get inDim => W.isEmpty ? 0 : W[0].length;
  int get outDim => b.length;

  List<double> forward(List<double> x) {
    final out = List<double>.filled(outDim, 0.0);
    final nOut = math.min(outDim, W.length);
    for (int i = 0; i < nOut; i++) {
      final Wi = W[i];
      final nIn = math.min(inDim, x.length);
      double s = (i < b.length) ? b[i] : 0.0;
      for (int j = 0; j < nIn; j++) s += Wi[j] * x[j];
      out[i] = s;
    }
    return out;
  }

  /// Backward wrt inputs, and **accumulate** grads into provided gW/gb buffers.
  /// - x: input vector used in forward
  /// - dOut: gradient wrt layer outputs (same length as outDim ideally)
  /// - gW/gb: external accumulators (shape must match W/b). Safe if over-sized.
  List<double> backward({
    required List<double> x,
    required List<double> dOut,
    required List<List<double>> gW,
    required List<double> gb,
  }) {
    final nOut = math.min(outDim, math.min(W.length, dOut.length));
    final nIn  = math.min(inDim, x.length);

    // dL/dx = W^T * dOut
    final dX = List<double>.filled(inDim, 0.0);
    for (int i = 0; i < nOut; i++) {
      final gi = dOut[i];
      final Wi = W[i];
      // accumulate grads for W and b
      if (i < gb.length) gb[i] += gi;
      if (i < gW.length) {
        final gWi = gW[i];
        for (int j = 0; j < nIn && j < gWi.length; j++) {
          gWi[j] += gi * x[j];
        }
      }
      // input grad
      for (int j = 0; j < nIn; j++) {
        dX[j] += Wi[j] * gi;
      }
    }
    return dX;
  }
}

/* ---------------------------------- MLP trunk -------------------------------- */

class MLPTrunk {
  final List<Linear> layers;

  MLPTrunk({required int inputSize, required List<int> hiddenSizes, int seed = 0})
      : layers = _build(inputSize, hiddenSizes, seed);

  static List<Linear> _build(int input, List<int> hidden, int seed) {
    final l = <Linear>[];
    var inDim = input;
    for (int i = 0; i < hidden.length; i++) {
      final h = hidden[i];
      l.add(Linear(inDim, h, seed: seed ^ (0xA5A5 + i)));
      inDim = h;
    }
    return l;
  }

  /// Returns tanh activations for each layer: [a1, a2, ... aL].
  List<List<double>> forwardAll(List<double> x) {
    var h = x;
    final acts = <List<double>>[];
    for (final lin in layers) {
      final z = lin.forward(h);
      final a = List<double>.generate(z.length, (i) => _tanh(z[i]));
      acts.add(a);
      h = a;
    }
    return acts;
  }

  /// Backprop through the trunk given dTop = dL/da_L and the cached activations.
  /// You MUST pass the original input x0 of the trunk.
  void backwardFromTopGrad({
    required List<double> dTop,
    required List<List<double>> acts,
    required List<List<List<double>>> gW, // same shape as layers[i].W
    required List<List<double>> gb,       // same shape as layers[i].b
    required List<double> x0,
  }) {
    assert(acts.length == layers.length);
    assert(gW.length == layers.length && gb.length == layers.length);

    var d = List<double>.from(dTop);

    for (int li = layers.length - 1; li >= 0; li--) {
      final lin = layers[li];
      final a   = acts[li];
      // dL/dz = dL/da * (1 - a^2)  (tanh’)
      final dz = List<double>.filled(a.length, 0.0);
      final n = math.min(a.length, d.length);
      for (int i = 0; i < n; i++) dz[i] = d[i] * (1.0 - a[i] * a[i]);

      // input to this layer is previous activation or x0 if first layer
      final inp = (li == 0) ? x0 : acts[li - 1];

      // accumulate grads + compute next d (wrt inputs)
      d = lin.backward(x: inp, dOut: dz, gW: gW[li], gb: gb[li]);
      // now `d` is dL/d(input_of_this_layer) — becomes dTop for the next layer down
    }
  }
}

/* ---------------------------------- HEADS ------------------------------------ */

class PolicyHeads {
  final Linear intent; // K logits
  final Linear turn;   // 3 logits
  final Linear thr;    // 1 logit
  final Linear val;    // 1 value

  PolicyHeads(int hiddenSize, {int intents = 5, int seed = 0})
      : intent = Linear(hiddenSize, intents, seed: seed ^ 0x11),
        turn   = Linear(hiddenSize, 3,      seed: seed ^ 0x22),
        thr    = Linear(hiddenSize, 1,      seed: seed ^ 0x33),
        val    = Linear(hiddenSize, 1,      seed: seed ^ 0x44);
}
