// lib/ai/plan_bus.dart
import 'dart:async';
import 'package:flutter/material.dart' show Offset;

class PlanEvent {
  final List<Offset> points;
  final List<double>? widths;     // ← NEW (same length as points)
  final int version;
  final String source;
  PlanEvent({
    required this.points,
    this.widths,                  // ← NEW
    required this.version,
    this.source = 'policy',
  });
}

class PlanBus {
  PlanBus._();
  static final instance = PlanBus._();

  final _ctrl = StreamController<PlanEvent>.broadcast();
  Stream<PlanEvent> get stream => _ctrl.stream;

  int _version = 0;

  void push({required List<Offset> points, List<double>? widths, String source = 'policy'}) {
    _version++;
    _ctrl.add(PlanEvent(points: points, widths: widths, version: _version, source: source));
  }
}
