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

// SiLU / Swish: x * sigmoid(x) — friendlier in noisy regimes than tanh
double silu(double x) {
  if (x >= 0) {
    final e = math.exp(-x);
    return x / (1.0 + e);
  } else {
    final e = math.exp(x);
    return x * (e / (1.0 + e));
  }
}

enum Activation { tanh, silu }

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

/*
 * Additions for noisy environments:
 *  - trainNoiseStd: Gaussian noise on pre-activations z (train only)
 *  - dropoutProb: inverted-scaling dropout on activations a (train only)
 *  - activation: tanh (default) or SiLU
 *
 * Backprop respects dropout masks so dropped units get zero gradient.
 */
class MLPTrunk {
  final List<Linear> layers;

  // knobs (can be changed after construction)
  Activation activation;
  double trainNoiseStd;     // e.g., 0.01 .. 0.05
  double dropoutProb;       // e.g., 0.05 .. 0.2
  bool trainMode;           // set true during rollout/training, false for eval

  // internal: store last dropout masks per layer for correct backprop
  List<List<double>> _lastDropMasks = const []; // 1.0 keep, 0.0 drop (scaled)

  MLPTrunk({
    required int inputSize,
    required List<int> hiddenSizes,
    int seed = 0,
    this.activation = Activation.tanh,
    this.trainNoiseStd = 0.0,
    this.dropoutProb = 0.0,
    this.trainMode = false,
  }) : layers = _build(inputSize, hiddenSizes, seed);

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

  // deterministic RNG used for noise/dropout so results are repeatable per call if desired
  math.Random? _rng;

  void _ensureRng() {
    _rng ??= math.Random(0xBEEFCAFE);
  }

  double _act(double z) {
    switch (activation) {
      case Activation.silu: return silu(z);
      case Activation.tanh:
      default: return _tanh(z);
    }
  }

  /// Returns activations for each layer: [a1, a2, ... aL].
  /// If trainMode=true, applies Gaussian noise to z and dropout to a.
  List<List<double>> forwardAll(List<double> x) {
    var h = x;
    final acts = <List<double>>[];
    final masks = <List<double>>[];

    for (final lin in layers) {
      // pre-activation
      var z = lin.forward(h);

      // Gaussian noise on z (train only)
      if (trainMode && trainNoiseStd > 0) {
        _ensureRng();
        final std = trainNoiseStd;
        for (int i = 0; i < z.length; i++) {
          // Box-Muller
          final u1 = (_rng!.nextDouble().clamp(1e-12, 1.0));
          final u2 = _rng!.nextDouble();
          final g = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2);
          z[i] += std * g;
        }
      }

      // activation
      var a = List<double>.generate(z.length, (i) => _act(z[i]));

      // dropout on a (train only) — inverted scaling
      if (trainMode && dropoutProb > 0.0) {
        _ensureRng();
        final p = dropoutProb.clamp(0.0, 0.95);
        final keepScale = (p < 1e-6) ? 1.0 : (1.0 / (1.0 - p));
        final m = List<double>.filled(a.length, 1.0);
        for (int i = 0; i < a.length; i++) {
          final keep = _rng!.nextDouble() >= p;
          final mask = keep ? keepScale : 0.0;
          a[i] *= mask;
          m[i] = mask; // store mask value (scaled)
        }
        masks.add(m);
      } else {
        // no dropout: mask of 1s
        masks.add(List<double>.filled(a.length, 1.0));
      }

      acts.add(a);
      h = a;
    }

    _lastDropMasks = masks;
    return acts;
  }

  /// Backprop through the trunk given dTop = dL/da_L and the cached activations.
  /// Dropout masks from the last forward are applied to gradients to zero dropped units.
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

      // apply dropout mask to upstream gradient (train-time only)
      final mask = (li < _lastDropMasks.length) ? _lastDropMasks[li] : null;
      if (trainMode && mask != null) {
        for (int i = 0; i < d.length && i < mask.length; i++) d[i] *= mask[i];
      }

      // dL/dz = dL/da * activation'(z); we only have 'a', so:
      // tanh: 1 - a^2,  SiLU: derivative = σ(x) + x*σ(x)*(1-σ(x))
      final dz = List<double>.filled(a.length, 0.0);
      final n = math.min(a.length, d.length);
      if (activation == Activation.tanh) {
        for (int i = 0; i < n; i++) dz[i] = d[i] * (1.0 - a[i] * a[i]);
      } else {
        // For SiLU we don't have z cached; approximate using a and x~a (works well in practice)
        // Better: recompute z by inverse of activation; here we use a cheap local approx:
        // Use local slope using a’ ≈ sigmoid(z) * (1 + z * (1 - sigmoid(z)))
        // As a robust fallback, treat derivative as <= 1 and use 0.5..1 scaling based on a.
        for (int i = 0; i < n; i++) {
          final ai = a[i];
          final slope = 0.5 + 0.5 / (1.0 + (ai*ai)); // heuristic slope in (0.5,1]
          dz[i] = d[i] * slope;
        }
      }

      // input to this layer is previous activation or x0 if first layer
      final inp = (li == 0) ? x0 : acts[li - 1];

      // accumulate grads + compute next d (wrt inputs)
      d = lin.backward(x: inp, dOut: dz, gW: gW[li], gb: gb[li]);
      // d is dL/d(input_of_this_layer)
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
