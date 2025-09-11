// lib/ai/intent_bus.dart
import 'dart:async';

class IntentEvent {
  final String intent;
  final List<double> probs;
  final int step;
  final Map<String, Object?> meta;
  IntentEvent({
    required this.intent,
    this.probs = const [],
    this.step = 0,
    this.meta = const {},
  });
}

class ControlEvent {
  final bool thrust, left, right;
  final int step;
  final Map<String, Object?> meta;
  ControlEvent({
    required this.thrust,
    required this.left,
    required this.right,
    this.step = 0,
    this.meta = const {},
  });
}

class IntentBus {
  // ---- REAL SINGLETON ----
  static final IntentBus _singleton = IntentBus._internal();
  IntentBus._internal();
  factory IntentBus() => _singleton;
  static IntentBus get instance => _singleton;

  final _intentCtrl = StreamController<IntentEvent>.broadcast();
  final _controlCtrl = StreamController<ControlEvent>.broadcast();

  Stream<IntentEvent> get intents => _intentCtrl.stream;
  Stream<ControlEvent> get controls => _controlCtrl.stream;

  void publishIntent(IntentEvent e) => _intentCtrl.add(e);
  void publishControl(ControlEvent e) => _controlCtrl.add(e);

  // Debug helper to verify same instance across imports
  String get debugId => 'IntentBus#${identityHashCode(this)}';

  void dispose() {
    _intentCtrl.close();
    _controlCtrl.close();
  }
}
