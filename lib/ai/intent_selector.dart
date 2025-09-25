import 'dart:math' as math;
import 'nn_helper.dart' as nn; // Ops.softmax, Ops.argmax

class IntentSelector {
  final int k;
  final int minHold;
  final int maxHold;
  final double temp;       // lower => more decisive
  final double gap;        // prob margin to allow switch
  final double switchCost; // subtract from non-current logits
  final double emaAlpha;   // EMA on probs
  final math.Random _rnd;

  int _cur = 0;
  int _hold = 0;
  List<double>? _emaP;

  IntentSelector({
    required this.k,
    this.minHold = 8,
    this.maxHold = 20,
    this.temp = 0.8,
    this.gap = 0.08,
    this.switchCost = 0.25,
    this.emaAlpha = 0.2,
    int seed = 0xA11CE,
  }) : _rnd = math.Random(seed);

  void reset({int? forceIdx}) {
    if (forceIdx != null) _cur = forceIdx.clamp(0, k - 1);
    _hold = 0;
    _emaP = null;
  }

  int selectFromLogits(List<double> logits) {
    if (_hold > 0) { _hold--; return _cur; }

    final z = List<double>.from(logits);
    for (int i = 0; i < k; i++) if (i != _cur) z[i] -= switchCost;

    final T = temp.clamp(1e-6, 10.0);
    for (int i = 0; i < k; i++) z[i] /= T;
    var p = nn.Ops.softmax(z);

    if (emaAlpha > 0) {
      _emaP ??= List<double>.from(p);
      for (int i = 0; i < k; i++) {
        _emaP![i] = (1 - emaAlpha) * _emaP![i] + emaAlpha * p[i];
      }
      p = _emaP!;
    }

    final newIdx = nn.Ops.argmax(p);
    final shouldSwitch = (newIdx != _cur) && (p[newIdx] > p[_cur] + gap);

    if (shouldSwitch) {
      _cur = newIdx;
      final span = (maxHold - minHold).clamp(0, 9999);
      _hold = minHold + (span > 0 ? _rnd.nextInt(span + 1) : 0);
    } else {
      _hold = 0;
    }
    return _cur;
  }
}
