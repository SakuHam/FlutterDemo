// tooling/learning_stability_check.dart
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter_application_1/ai/agent.dart'
    show FeatureExtractor, PolicyNetwork, Trainer;
import 'package:flutter_application_1/engine/game_engine.dart' as eng;

/// ---------- IO helpers ----------
Future<Map<String, dynamic>> _readJson(String path) async =>
    json.decode(await File(path).readAsString()) as Map<String, dynamic>;

List<List<double>> _mat(dynamic m) => (m as List)
    .map<List<double>>((r) => (r as List).map<double>((x) => (x as num).toDouble()).toList())
    .toList();
List<double> _vec(dynamic v) => (v as List).map<double>((x) => (x as num).toDouble()).toList();

PolicyNetwork _loadPolicy(String path, {int? enforceInputSize, int h1 = 64, int h2 = 64}) {
  final js = File(path).existsSync() ? File(path).readAsStringSync() : throw ArgumentError('No file: $path');
  final m = json.decode(js) as Map<String, dynamic>;

  final inputSize = (m['inputSize'] as int?) ?? enforceInputSize ?? (throw ArgumentError('inputSize missing'));
  final ph1 = (m['h1'] as int?) ?? h1;
  final ph2 = (m['h2'] as int?) ?? h2;

  final net = PolicyNetwork(inputSize: inputSize, h1: ph1, h2: ph2, seed: 1234);
  net.W1 = _mat(m['W1']); net.b1 = _vec(m['b1']);
  net.W2 = _mat(m['W2']); net.b2 = _vec(m['b2']);
  if (!(m.containsKey('W_thr') && m.containsKey('W_turn'))) {
    throw ArgumentError('Policy JSON missing W_thr/W_turn');
  }
  net.W_thr = _mat(m['W_thr']); net.b_thr = _vec(m['b_thr']);
  net.W_turn = _mat(m['W_turn']); net.b_turn = _vec(m['b_turn']);
  return net;
}

/// ---------- Simple stats ----------
class Stats {
  final double mean, std, min, p25, p50, p75, max;
  Stats(this.mean, this.std, this.min, this.p25, this.p50, this.p75, this.max);
  @override String toString() =>
      'mean=${mean.toStringAsFixed(3)} std=${std.toStringAsFixed(3)} '
          'min=${min.toStringAsFixed(3)} p25=${p25.toStringAsFixed(3)} '
          'p50=${p50.toStringAsFixed(3)} p75=${p75.toStringAsFixed(3)} max=${max.toStringAsFixed(3)}';
}
Stats _stats(List<double> v) {
  final n = v.length;
  final mean = v.reduce((a,b)=>a+b)/n;
  double var0 = 0.0; for (final x in v) { var0 += (x-mean)*(x-mean); }
  final std = math.sqrt(var0 / (n>1? n-1 : 1));
  final s = List<double>.from(v)..sort();
  double q(double p){
  final t = (p*(n-1)).clamp(0.0, (n-1).toDouble());
  final i = t.floor();
  final f = t - i;
  if (i+1<n) return s[i]*(1-f)+s[i+1]*f;
  return s[i];
  }
  return Stats(mean, std, s.first, q(0.25), q(0.5), q(0.75), s.last);
}

/// ---------- Eval over seeds (greedy) ----------
class EvalResult {
  final int seed;
  final double cost;
  final int steps;
  final bool landed;
  EvalResult(this.seed, this.cost, this.steps, this.landed);
}

Future<List<EvalResult>> _evaluatePolicyGreedy({
  required PolicyNetwork net,
  required FeatureExtractor fe,
  required eng.EngineConfig cfg,
  required List<int> seeds,
}) async {
  final env = eng.GameEngine(cfg);
  final eval = Trainer(
    env: env, fe: fe, policy: net,
    dt: 1/60.0, gamma: 0.99, seed: 13,
    tempThr: 1e-6, tempTurn: 1e-6, epsilon: 0.0, entropyBeta: 0.0,
  );
  final out = <EvalResult>[];
  for (final s in seeds) {
    env.reset(seed: s);
    final ep = eval.runEpisode(train: false, lr: 0.0, greedy: true);
    out.add(EvalResult(s, ep.totalCost, ep.steps, ep.landed));
  }
  return out;
}

/// ---------- Pretty print ----------
void _printTable(List<EvalResult> rows) {
  print('seed\tsteps\tcost\tlanded');
  for (final r in rows) {
    print('${r.seed}\t${r.steps}\t${r.cost.toStringAsFixed(3)}\t${r.landed ? "Y" : "N"}');
  }
}

/// ---------- Main ----------
void main(List<String> args) async {
  // CLI-ish defaults (tweak freely)
  final policyPath = (args.isNotEmpty ? args[0] : 'policy.json');
  final nSeeds = (args.length >= 2 ? int.parse(args[1]) : 25);
  final seedStart = (args.length >= 3 ? int.parse(args[2]) : 1000);
  final lockTerrain = true;        // keep terrain fixed to isolate spawn/trajectory variety
  final lockSpawn = true;          // spawn Y/v/angle fixed; X can still vary if randomSpawnX=true
  final randomSpawnX = false;      // start with fully fixed spawn; flip to true to add variety

  // Engine/FE must match training
  final cfg = eng.EngineConfig(
    worldW: 360, worldH: 640, t: eng.Tunables(
    gravity: 0.18, thrustAccel: 0.42, rotSpeed: 1.6, maxFuel: 100.0,
  ),
    seed: 42,
    stepScale: 60.0,
    lockTerrain: lockTerrain, terrainSeed: 12345,
    lockSpawn: lockSpawn,
    randomSpawnX: randomSpawnX,
    spawnX: 0.25, spawnY: 120.0, spawnVx: 0.0, spawnVy: 0.0, spawnAngle: 0.0,
    hardWalls: true,
    landingSpeedMax: 12.0,               // <- match your train_agent strictness
    landingAngleMaxRad: 8 * math.pi / 180.0,
    wDx: 200.0, wDy: 180.0, wVx: 90.0, wVyDown: 200.0, wAngleDeg: 80.0,
  );

  // If your policy.json has an `fe` block, use it; else default (3,48)
  int feGS = 3;
  double feStride = 48.0;
  try {
    final js = await _readJson(policyPath);
    final fe = (js['fe'] as Map<String, dynamic>?);
    if (fe != null) {
      feGS = (fe['groundSamples'] as int?) ?? feGS;
      feStride = (fe['stridePx'] as num?)?.toDouble() ?? feStride;
    }
  } catch (_) {}

  final fe = FeatureExtractor(groundSamples: feGS, stridePx: feStride);
  final net = _loadPolicy(policyPath, enforceInputSize: fe.inputSize);

  final seeds = List<int>.generate(nSeeds, (i) => seedStart + i);
  final rows = await _evaluatePolicyGreedy(
    net: net, fe: fe, cfg: cfg, seeds: seeds,
  );

  // Print raw table
  print('\n=== Greedy evaluation over ${rows.length} seeds ===');
  _printTable(rows);

  // Stats
  final costs = rows.map((r) => r.cost).toList();
  final steps = rows.map((r) => r.steps.toDouble()).toList();
  final lands = rows.where((r) => r.landed).length;

  final sc = _stats(costs);
  final ss = _stats(steps);

  final wobbleIdx = (sc.mean.abs() > 1e-9) ? (sc.std / sc.mean.abs()) : double.nan;

  print('\nCost stats:\n$sc');
  print('Steps stats:\n$ss');
  print('Landed: $lands / ${rows.length} = ${(100.0*lands/rows.length).toStringAsFixed(1)}%');
  print('Wobble index (std/|mean|): ${wobbleIdx.isNaN ? "NA" : wobbleIdx.toStringAsFixed(3)}');

  print('\nTip: Set randomSpawnX=true or lockTerrain=false to see how robustness changes.');
  print('     Use this tool before/after training to check if wobble is shrinking.\n');
}
