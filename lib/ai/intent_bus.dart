// lib/ai/intent_bus.dart
import 'dart:async';

class IntentEvent {
  final String intent;            // e.g. "goLeft"
  final List<double> probs;       // softmax over intents (optional)
  final int step;                 // env step when chosen (optional)
  final Map<String, Object?> meta; // anything else (episodeId, seed, etc.)

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
  IntentBus._internal();
  static final IntentBus instance = IntentBus._internal();

  final _intentCtrl = StreamController<IntentEvent>.broadcast();
  final _controlCtrl = StreamController<ControlEvent>.broadcast();

  Stream<IntentEvent> get intents => _intentCtrl.stream;
  Stream<ControlEvent> get controls => _controlCtrl.stream;

  void publishIntent(IntentEvent e) => _intentCtrl.add(e);
  void publishControl(ControlEvent e) => _controlCtrl.add(e);

  void dispose() {
    _intentCtrl.close();
    _controlCtrl.close();
  }
}

// (Optional) a short alias
IntentBus get intentBus => IntentBus.instance;
