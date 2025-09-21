// Core interfaces + small utilities shared by curricula.

import 'dart:math' as math;

import '../../engine/types.dart' as et;
import '../../engine/game_engine.dart' as eng;
import '../../engine/raycast.dart'; // RayConfig
import '../agent.dart'; // FeatureExtractorRays, PolicyNetwork, Trainer, RunningNorm, EpisodeResult
import '../nn_helper.dart' as nn;

/// A curriculum/micro-stage is anything that can run N iterations of self-contained episodes
/// and optionally train the policy.
abstract class Curriculum {
  /// A short key used on CLI, e.g. "speedmin", "hardapp"
  String get key;

  /// Configure from CLI kv/flags (optional).
  /// Return `this` so calls can be chained.
  void configure(Map<String,String?> kv, Set<String> flags) {}

  /// Run `iters` iterations. Must NOT mutate global ray config or env cfg.
  Future<void> run({
    required int iters,
    required eng.GameEngine env,
    required FeatureExtractorRays fe,
    required PolicyNetwork policy,
    required RunningNorm? norm,
    required int planHold,
    required double tempIntent,
    required double gamma,
    required double lr,
    required double intentAlignWeight,
    required double intentPgWeight,
    required double actionAlignWeight,
    required bool gateVerbose,
    required int seed,
  });
}

/// Simple CLI adapter to re-use your tiny arg parser result.
class CliView {
  final Map<String, String?> kv;
  final Set<String> flags;
  const CliView(this.kv, this.flags);

  int getInt(String k, {int def = 0}) => int.tryParse(kv[k] ?? '') ?? def;
  double getDouble(String k, {double def = 0.0}) => double.tryParse(kv[k] ?? '') ?? def;
  bool getFlag(String k, {bool def = false}) {
    if (flags.contains(k)) return true;
    final v = kv[k];
    if (v == null) return def;
    final s = v.toLowerCase();
    return s == '1' || s == 'true' || s == 'yes' || s == 'on';
  }

  String? getStr(String k, {String? def}) => kv[k] ?? def;
}

/// Registry of curricula by key
class CurriculumRegistry {
  final Map<String, Curriculum Function()> _factories = {};
  void register(String key, Curriculum Function() mk) {
    _factories[key] = mk;
  }

  /// Parse comma/plus separated list: "speedmin,hardapp"
  List<Curriculum> fromConfig(String? s) {
    if (s == null || s.trim().isEmpty) return const [];
    final parts = s
        .split(RegExp(r'[,+]'))
        .map((t) => t.trim())
        .where((t) => t.isNotEmpty)
        .toList();
    final out = <Curriculum>[];
    for (final p in parts) {
      final f = _factories[p];
      if (f != null) out.add(f());
    }
    return out;
  }

  List<String> get knownKeys => _factories.keys.toList()..sort();
}
