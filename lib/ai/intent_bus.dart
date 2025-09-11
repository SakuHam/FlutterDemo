// lib/ai/intent_bus.dart
import 'dart:async';

class IntentEvent {
  final String intent;
  final List<double> probs;
  final int step;
  final Map<String, Object?> meta;
  IntentEvent({required this.intent, required this.probs, required this.step, this.meta = const {}});
}

class ControlEvent {
  final bool thrust, left, right;
  final int step;
  final Map<String, Object?> meta;
  ControlEvent({required this.thrust, required this.left, required this.right, required this.step, this.meta = const {}});
}

class IntentBus {
  IntentBus._();
  static final IntentBus instance = IntentBus._();

  final _intentCtl  = StreamController<IntentEvent>.broadcast();
  final _controlCtl = StreamController<ControlEvent>.broadcast();

  IntentEvent? _lastIntent;
  ControlEvent? _lastControl;

  // Normal streams
  Stream<IntentEvent> get intents => _intentCtl.stream;
  Stream<ControlEvent> get controls => _controlCtl.stream;

  // Streams that replay the latest event immediately to new subscribers
  Stream<IntentEvent> intentsWithReplay() async* {
    if (_lastIntent != null) yield _lastIntent!;
    yield* _intentCtl.stream;
  }
  Stream<ControlEvent> controlsWithReplay() async* {
    if (_lastControl != null) yield _lastControl!;
    yield* _controlCtl.stream;
  }

  void publishIntent(IntentEvent e) {
    _lastIntent = e;
    _intentCtl.add(e);
  }

  void publishControl(ControlEvent e) {
    _lastControl = e;
    _controlCtl.add(e);
  }

  void dispose() {
    _intentCtl.close();
    _controlCtl.close();
  }
}
