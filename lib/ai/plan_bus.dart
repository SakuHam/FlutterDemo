// lib/ai/plan_bus.dart
import 'dart:async';
import 'package:flutter/material.dart' show Offset;

/// A single plan update (polyline + monotonically increasing version).
class PlanEvent {
  final List<Offset> points;   // polyline in world coords
  final int version;           // increments each publish
  const PlanEvent({required this.points, required this.version});
}

/// Lightweight pub/sub bus for sharing plan polylines with the UI overlay.
/// Usage:
///   PlanBus.instance.publish(points);  // or .push(points)
///   PlanBus.instance.stream.listen((e) { ... e.points ... e.version ... });
class PlanBus {
  PlanBus._();
  static final PlanBus instance = PlanBus._();

  final _ctrl = StreamController<PlanEvent>.broadcast();
  int _version = 0;
  List<Offset> _last = const [];

  /// Publish a new plan polyline. (Alias: [push])
  void publish(List<Offset> points) {
    _version++;
    _last = List<Offset>.unmodifiable(points);
    _ctrl.add(PlanEvent(points: _last, version: _version));
  }

  /// Back-compat alias (some code refers to `.push(...)`).
  void push(List<Offset> points) => publish(points);

  /// Subscribe to plan updates.
  Stream<PlanEvent> get stream => _ctrl.stream;

  /// Synchronous snapshot of the most recent plan (if any).
  PlanEvent? get lastOrNull =>
      _version == 0 ? null : PlanEvent(points: _last, version: _version);

  void dispose() {
    _ctrl.close();
  }
}
