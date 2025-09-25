import 'dart:math' as math;

/* =========================== tiny init / activations =========================== */

double _randn(math.Random r) {
  final u1 = r.nextDouble().clamp(1e-12, 1.0);
  final u2 = r.nextDouble();
  return math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2);
}

List<List<double>> _xavier(int outDim, int inDim, int seed) {
  final r = math.Random(seed);
  final lim = math.sqrt(6.0 / (outDim + inDim));
  return List.generate(
    outDim,
        (_) => List<double>.generate(inDim, (_) => (r.nextDouble() * 2 - 1) * lim),
  );
}

List<List<double>> _kaiming(int outDim, int inDim, int seed) {
  final r = math.Random(seed);
  final std = math.sqrt(2.0 / inDim);
  return List.generate(
    outDim,
        (_) => List<double>.generate(inDim, (_) => std * _randn(r)),
  );
}

double _tanh(double x) {
  final ax = x.abs();
  if (ax > 20.0) return x.isNegative ? -1.0 : 1.0;
  final e = math.exp(-2.0 * ax);
  final t = (1.0 - e) / (1.0 + e);
  return x.isNegative ? -t : t;
}

// SiLU / Swish: x * sigmoid(x)
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
enum InitKind { xavier, kaiming }

/* ------------------------------------ Ops ------------------------------------ */

class Ops {
  static int argmax(List<double> v) {
    if (v.isEmpty) return -1;
    int bestIdx = 0;
    double best = v[0];
    for (int i = 1; i < v.length; i++) {
      final x = v[i];
      if (x > best) {
        best = x;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  static int argmin(List<double> v) {
    if (v.isEmpty) return -1;
    int bestIdx = 0;
    double best = v[0];
    for (int i = 1; i < v.length; i++) {
      final x = v[i];
      if (x < best) {
        best = x;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  /// log1p(y) ≈ log(1+y), but stable for small |y|.
  static double log1p(double y) {
    // handle edge cases
    if (y <= -1.0) return double.nan; // log(≤0) undefined
    final ay = y.abs();
    if (ay > 1e-4) {
      return math.log(1.0 + y);
    }
    // series for small y: y - y^2/2 + y^3/3 - y^4/4 ...
    final y2 = y * y;
    final y3 = y2 * y;
    final y4 = y3 * y;
    return y - 0.5 * y2 + (1.0 / 3.0) * y3 - 0.25 * y4;
  }

  static double softplus(double x) {
    // stable: softplus(x) = log(1 + exp(x)) = max(0,x) + log1p(exp(-|x|))
    final ax = x.abs();
    return log1p(math.exp(-ax)) + (x > 0 ? x : 0.0);
  }

  // (optional) vector helper if you ever want it
  static List<double> softplusVec(List<double> xs) =>
      xs.map((v) => softplus(v)).toList();

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

  /// dL/dlogits for CE with softmax: (p - y_onehot)
  static List<double> crossEntropyGrad(List<double> p, int y, {int? numClasses}) {
    final k = numClasses ?? p.length;
    final out = List<double>.filled(k, 0.0);
    final n = math.min(k, p.length);
    for (int i = 0; i < n; i++) out[i] = p[i];
    if (y >= 0 && y < n) out[y] -= 1.0;
    return out;
  }

  /// Label-smoothed variant (ε distributes small prob mass to non-targets).
  static List<double> crossEntropyGradSmoothed(List<double> p, int y, double eps) {
    final k = p.length;
    final out = List<double>.filled(k, 0.0);
    final u = eps / k;
    for (int i = 0; i < k; i++) {
      final target = (i == y) ? (1.0 - eps + u) : u;
      out[i] = p[i] - target; // dL/dz = p - y_smooth
    }
    return out;
  }

  /// BCE grad wrt logit: σ(z) - y
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

  Linear(int inDim, int outDim, {int seed = 0, InitKind init = InitKind.xavier})
      : W = (init == InitKind.xavier)
      ? _xavier(outDim, inDim, seed)
      : _kaiming(outDim, inDim, seed),
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

/* ---------------------------------- RMSNorm ---------------------------------- */

class RMSNorm {
  final int dim;
  final double eps;
  final List<double> g; // gain

  RMSNorm(this.dim, {this.eps = 1e-6})
      : g = List<double>.filled(dim, 1.0);

  List<double> forward(List<double> x) {
    double ss = 0.0;
    for (final xi in x) ss += xi * xi;
    final inv = 1.0 / math.sqrt((ss / x.length) + eps);
    final out = List<double>.filled(x.length, 0.0);
    for (int i = 0; i < x.length && i < g.length; i++) {
      out[i] = x[i] * inv * g[i];
    }
    return out;
  }
}

/* ---------------------------------- MLP trunk -------------------------------- */

/*
 * Noise-robust trunk:
 *  - Kaiming init when activation == SiLU (Xavier for tanh).
 *  - Exact SiLU derivative (cache z).
 *  - Optional RMSNorm per layer (before activation).
 *  - Gaussian noise on pre-activations (train only).
 *  - Inverted-dropout on activations (train only).
 *  - Optional input noise + clipping at the first layer (train only).
 *
 * Backprop respects dropout masks so dropped units get zero gradient.
 */
class MLPTrunk {
  final List<Linear> layers;

  // knobs (can be changed after construction)
  Activation activation;
  double trainNoiseStd;     // e.g., 0.01 .. 0.05 (z-noise)
  double dropoutProb;       // e.g., 0.05 .. 0.2
  bool trainMode;           // set true during rollout/training, false for eval

  // normalization & input robustness
  bool useRmsNorm;          // apply RMSNorm before activation per layer
  double inputNoiseStd;     // e.g., 0.001 .. 0.02
  double inputClip;         // e.g., 5.0 (±clip). <=0 disables

  // internal caches
  List<List<double>> _lastDropMasks = const []; // 1.0 keep, 0.0 drop (scaled)
  List<List<double>> _lastZs = const [];        // pre-activations per layer
  late final List<RMSNorm?> _norms;

  MLPTrunk({
    required int inputSize,
    required List<int> hiddenSizes,
    int seed = 0,
    this.activation = Activation.tanh,
    this.trainNoiseStd = 0.0,
    this.dropoutProb = 0.0,
    this.trainMode = false,
    this.useRmsNorm = true,
    this.inputNoiseStd = 0.0,
    this.inputClip = 0.0,
  }) : layers = _build(inputSize, hiddenSizes, seed, activation),
        _norms = List<RMSNorm?>.filled(hiddenSizes.length, null) {
    if (useRmsNorm) {
      for (int i = 0; i < hiddenSizes.length; i++) {
        _norms[i] = RMSNorm(hiddenSizes[i]);
      }
    }
  }

  static List<Linear> _build(int input, List<int> hidden, int seed, Activation act) {
    final l = <Linear>[];
    var inDim = input;
    final init = (act == Activation.silu) ? InitKind.kaiming : InitKind.xavier;
    for (int i = 0; i < hidden.length; i++) {
      final h = hidden[i];
      l.add(Linear(inDim, h, seed: seed ^ (0xA5A5 + i), init: init));
      inDim = h;
    }
    return l;
  }

  // deterministic RNG used for noise/dropout (can be swapped for a trainer-provided RNG)
  math.Random? _rng;
  void _ensureRng() => _rng ??= math.Random(0xBEEFCAFE);

  double _act(double z) => (activation == Activation.silu) ? silu(z) : _tanh(z);

  // Optional input pre-processing for robustness
  List<double> _noisyClippedInput(List<double> x) {
    if (!trainMode && inputClip <= 0 && inputNoiseStd <= 0) return x;
    final out = List<double>.from(x);
    if (trainMode && inputNoiseStd > 0) {
      _ensureRng();
      for (int i = 0; i < out.length; i++) out[i] += inputNoiseStd * _randn(_rng!);
    }
    if (inputClip > 0) {
      final c = inputClip;
      for (int i = 0; i < out.length; i++) out[i] = out[i].clamp(-c, c);
    }
    return out;
  }

  /// Returns activations for each layer: [a1, a2, ... aL].
  /// If trainMode=true, applies Gaussian noise to z and dropout to a.
  List<List<double>> forwardAll(List<double> x) {
    var h = _noisyClippedInput(x);
    final acts = <List<double>>[];
    final zs   = <List<double>>[];
    final masks = <List<double>>[];

    for (int li = 0; li < layers.length; li++) {
      // pre-activation
      var z = layers[li].forward(h);

      // optionally normalize pre-acts
      if (useRmsNorm && _norms[li] != null) {
        z = _norms[li]!.forward(z);
      }

      // Gaussian noise on z (train only)
      if (trainMode && trainNoiseStd > 0) {
        _ensureRng();
        final std = trainNoiseStd;
        for (int i = 0; i < z.length; i++) z[i] += std * _randn(_rng!);
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
      zs.add(z);
      h = a;
    }

    _lastDropMasks = masks;
    _lastZs = zs;
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
      final z   = (li < _lastZs.length) ? _lastZs[li] : a; // fallback to a if z missing

      // apply dropout mask to upstream gradient (train-time only)
      final mask = (li < _lastDropMasks.length) ? _lastDropMasks[li] : null;
      if (trainMode && mask != null) {
        for (int i = 0; i < d.length && i < mask.length; i++) d[i] *= mask[i];
      }

      // dL/dz = dL/da * activation'(z)
      final dz = List<double>.filled(a.length, 0.0);
      final n = math.min(a.length, d.length);

      if (activation == Activation.tanh) {
        // tanh'(z) = 1 - a^2 (we have a)
        for (int i = 0; i < n; i++) dz[i] = d[i] * (1.0 - a[i] * a[i]);
      } else {
        // exact SiLU derivative using cached z
        for (int i = 0; i < n; i++) {
          final zi = (i < z.length) ? z[i] : 0.0;
          final s  = Ops.sigmoid(zi);
          final grad = s + zi * s * (1.0 - s);
          dz[i] = d[i] * grad;
        }
      }

      // input to this layer is previous activation or x0 if first layer
      final inp = (li == 0) ? x0 : acts[li - 1];

      // accumulate grads + compute next d (wrt inputs)
      d = lin.backward(x: inp, dOut: dz, gW: gW[li], gb: gb[li]);
      // d is now dL/d(input_of_this_layer)
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
      : intent = Linear(hiddenSize, intents, seed: seed ^ 0x11,
      init: InitKind.xavier), // heads often fine with Xavier
        turn   = Linear(hiddenSize, 3,      seed: seed ^ 0x22,
            init: InitKind.xavier),
        thr    = Linear(hiddenSize, 1,      seed: seed ^ 0x33,
            init: InitKind.xavier),
        val    = Linear(hiddenSize, 1,      seed: seed ^ 0x44,
            init: InitKind.xavier);
}

/* ----------------------------- Training utilities ---------------------------- */

/// Global L2 gradient clipping.
void clipGradsL2(List<List<List<double>>> gW, List<List<double>> gb, double maxNorm) {
  if (maxNorm <= 0) return;
  double ss = 0.0;
  for (final gw in gW) {
    for (final row in gw) {
      for (final v in row) ss += v * v;
    }
  }
  for (final gb1 in gb) {
    for (final v in gb1) ss += v * v;
  }
  final norm = math.sqrt(ss);
  if (norm > maxNorm && norm > 0) {
    final s = maxNorm / norm;
    for (final gw in gW) {
      for (final row in gw) {
        for (int j = 0; j < row.length; j++) row[j] *= s;
      }
    }
    for (final gb1 in gb) {
      for (int j = 0; j < gb1.length; j++) gb1[j] *= s;
    }
  }
}

/// Decoupled weight decay (AdamW-style): apply after the gradient step or
/// multiply weights by (1 - lr*wd) each update.
void applyWeightDecay(List<List<List<double>>> W, double lr, double wd) {
  if (wd <= 0) return;
  final decay = 1.0 - lr * wd;
  for (final wLayer in W) {
    for (final row in wLayer) {
      for (int j = 0; j < row.length; j++) row[j] *= decay;
    }
  }
}
