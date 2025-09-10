// tooling/determinism_probe.dart
import 'dart:math' as math;
import 'package:flutter_application_1/ai/agent.dart' as ai;
import 'package:flutter_application_1/engine/game_engine.dart' as eng;

/// Small epsilon for float comparisons
const double EPS = 1e-12;

bool _deq(double a, double b, [double eps = EPS]) => (a - b).abs() <= eps;

String _fmt(double x) => x.toStringAsFixed(9);

class _Mismatch implements Exception {
  final String where;
  final String details;
  _Mismatch(this.where, this.details);
  @override
  String toString() => 'Mismatch at $where: $details';
}

/// Compare two vectors component-wise.
void _assertVec(List<double> a, List<double> b, String where, [double eps = EPS]) {
  if (a.length != b.length) {
    throw _Mismatch(where, 'length ${a.length} != ${b.length}');
  }
  for (int i = 0; i < a.length; i++) {
    if (!_deq(a[i], b[i], eps)) {
      throw _Mismatch(where, 'idx=$i ${_fmt(a[i])} != ${_fmt(b[i])}');
    }
  }
}

void _assertEqual(double a, double b, String where, [double eps = EPS]) {
  if (!_deq(a, b, eps)) {
    throw _Mismatch(where, '${_fmt(a)} != ${_fmt(b)}');
  }
}

void _banner(String title) {
  print('\n=== $title ===');
}

void _pass(String msg) {
  print('PASS: $msg');
}

void _fail(String msg) {
  print('FAIL: $msg');
}

eng.EngineConfig _baseCfg({
  bool lockTerrain = true,
  bool lockSpawn = true,
  bool randomSpawnX = false,
  int terrainSeed = 12345,
  int seed = 424242,
}) {
  return eng.EngineConfig(
    worldW: 360,
    worldH: 640,
    t: eng.Tunables(
      gravity: 0.18,
      thrustAccel: 0.42,
      rotSpeed: 1.6,
      maxFuel: 100.0,
    ),
    seed: seed,
    stepScale: 60.0,
    lockTerrain: lockTerrain,
    terrainSeed: terrainSeed,
    lockSpawn: lockSpawn,
    randomSpawnX: randomSpawnX,
    spawnX: 0.25,
    spawnY: 120.0,
    spawnVx: 0.0,
    spawnVy: 0.0,
    spawnAngle: 0.0,
    hardWalls: true,
    landingSpeedMax: 80.0,
    landingAngleMaxRad: 0.45,
  );
}

/// Create a fixed control script for N steps.
List<eng.ControlInput> _fixedControls(int n) {
  // Simple deterministic pattern: thrust every 3rd step, alternate left/right.
  return List.generate(n, (i) {
    final thrust = (i % 3 == 0);
    final left = (i % 4 == 1);
    final right = (i % 4 == 3);
    return eng.ControlInput(thrust: thrust, left: left, right: right);
  });
}

/// Run N steps and return traces.
class Trace {
  final List<List<double>> pos;  // [x,y]
  final List<List<double>> vel;  // [vx,vy]
  final List<double> angle;
  final List<double> fuel;
  final List<double> cost;
  final List<bool> landed;
  final List<bool> terminal;
  Trace(this.pos, this.vel, this.angle, this.fuel, this.cost, this.landed, this.terminal);
}

Trace _rollout(eng.GameEngine env, List<eng.ControlInput> script, {double dt = 1/60}) {
  final pos = <List<double>>[];
  final vel = <List<double>>[];
  final ang = <double>[];
  final fuel = <double>[];
  final cost = <double>[];
  final landed = <bool>[];
  final term = <bool>[];

  for (final u in script) {
    final s = env.step(dt, u);
    pos.add([env.lander.pos.x, env.lander.pos.y]);
    vel.add([env.lander.vel.x, env.lander.vel.y]);
    ang.add(env.lander.angle);
    fuel.add(env.lander.fuel);
    cost.add(s.costDelta);
    landed.add(s.landed);
    term.add(s.terminal);
    if (s.terminal) break;
  }
  return Trace(pos, vel, ang, fuel, cost, landed, term);
}

void _compareTraces(Trace a, Trace b, String where) {
  final n = math.min(a.pos.length, b.pos.length);
  for (int i = 0; i < n; i++) {
    _assertVec(a.pos[i], b.pos[i], '$where.pos[$i]');
    _assertVec(a.vel[i], b.vel[i], '$where.vel[$i]');
    _assertEqual(a.angle[i], b.angle[i], '$where.angle[$i]');
    _assertEqual(a.fuel[i], b.fuel[i], '$where.fuel[$i]');
    _assertEqual(a.cost[i], b.cost[i], '$where.cost[$i]');
    if (a.landed[i] != b.landed[i]) {
      throw _Mismatch('$where.landed[$i]', '${a.landed[i]} != ${b.landed[i]}');
    }
    if (a.terminal[i] != b.terminal[i]) {
      throw _Mismatch('$where.terminal[$i]', '${a.terminal[i]} != ${b.terminal[i]}');
    }
  }
  if (a.pos.length != b.pos.length) {
    throw _Mismatch('$where.length', '${a.pos.length} != ${b.pos.length}');
  }
}

void testTerrainDeterminism() {
  _banner('Terrain.generate determinism');
  final t1 = eng.Terrain.generate(360, 640, 777, 1.0);
  final t2 = eng.Terrain.generate(360, 640, 777, 1.0);

  if (t1.ridge.length != t2.ridge.length) {
    _fail('ridge length ${t1.ridge.length} != ${t2.ridge.length}');
    return;
  }
  for (int i = 0; i < t1.ridge.length; i++) {
    final a = t1.ridge[i]; final b = t2.ridge[i];
    if (!_deq(a.x, b.x) || !_deq(a.y, b.y)) {
      _fail('ridge[$i] differs: (${_fmt(a.x)},${_fmt(a.y)}) vs (${_fmt(b.x)},${_fmt(b.y)})');
      return;
    }
  }
  if (!_deq(t1.padX1, t2.padX1) ||
      !_deq(t1.padX2, t2.padX2) ||
      !_deq(t1.padY,  t2.padY)) {
    _fail('pad differs');
    return;
  }
  // heightAt grid
  for (int i = 0; i <= 20; i++) {
    final x = 360.0 * i / 20.0;
    if (!_deq(t1.heightAt(x), t2.heightAt(x))) {
      _fail('heightAt($x) differs: ${_fmt(t1.heightAt(x))} vs ${_fmt(t2.heightAt(x))}');
      return;
    }
  }
  _pass('Terrain.generate & heightAt stable for same seed');
}

void testResetDeterminism() {
  _banner('GameEngine.reset determinism');
  // A) Locked terrain & spawn
      {
    final cfg = _baseCfg(lockTerrain: true, lockSpawn: true, randomSpawnX: false, seed: 111, terrainSeed: 999);
    final e1 = eng.GameEngine(cfg);
    final e2 = eng.GameEngine(cfg);
    // Both start at same state
    if (!_deq(e1.lander.pos.x, e2.lander.pos.x) ||
        !_deq(e1.lander.pos.y, e2.lander.pos.y) ||
        !_deq(e1.lander.angle, e2.lander.angle)) {
      _fail('lockTerrain+lockSpawn initial state differs');
      return;
    }
    // Reset with same seed
    e1.reset(seed: 222);
    e2.reset(seed: 222);
    if (!_deq(e1.lander.pos.x, e2.lander.pos.x) ||
        !_deq(e1.lander.pos.y, e2.lander.pos.y) ||
        !_deq(e1.lander.angle, e2.lander.angle)) {
      _fail('post-reset seed produced different state');
      return;
    }
  }

  // B) Unlocked terrain (should differ across runs for different seeds; match for same seed)
      {
    final cfg = _baseCfg(lockTerrain: false, lockSpawn: true, randomSpawnX: false);
    final e1 = eng.GameEngine(cfg);
    final e2 = eng.GameEngine(cfg);
    e1.reset(seed: 333);
    e2.reset(seed: 333);
    // Terrains should match for same seed path
    for (int i = 0; i < e1.terrain.ridge.length; i++) {
      final a = e1.terrain.ridge[i];
      final b = e2.terrain.ridge[i];
      if (!_deq(a.x, b.x) || !_deq(a.y, b.y)) {
        _fail('unlocked terrain not matching under same seed at idx=$i');
        return;
      }
    }
  }

  _pass('GameEngine.reset behaves deterministically under same seeds');
}

void testStepDeterminism() {
  _banner('GameEngine.step determinism (fixed script)');
  final cfg = _baseCfg(lockTerrain: true, lockSpawn: true, randomSpawnX: false, seed: 555, terrainSeed: 777);
  final e1 = eng.GameEngine(cfg);
  final e2 = eng.GameEngine(cfg);

  final script = _fixedControls(2000);
  final t1 = _rollout(e1, script);
  final t2 = _rollout(e2, script);

  try {
    _compareTraces(t1, t2, 'step');
    _pass('Identical traces for fixed control script');
  } on _Mismatch catch (m) {
    _fail(m.toString());
  }
}

void testFeatureExtractorDeterminism() {
  _banner('FeatureExtractor.extract determinism');
  final cfg = _baseCfg(lockTerrain: true, lockSpawn: true, randomSpawnX: false, seed: 123, terrainSeed: 321);
  final e1 = eng.GameEngine(cfg);
  final e2 = eng.GameEngine(cfg);

  final fe = ai.FeatureExtractor(groundSamples: 5, stridePx: 48);

  // Same initial state → same features
  final f1 = fe.extract(e1);
  final f2 = fe.extract(e2);
  try {
    _assertVec(f1, f2, 'fe.init');
    _pass('FE equal on init');
  } on _Mismatch catch (m) {
    _fail(m.toString());
    return;
  }

  // Advance both with same controls and re-check
  final script = _fixedControls(300);
  for (final u in script) {
    e1.step(1/60, u);
    e2.step(1/60, u);
    final a = fe.extract(e1);
    final b = fe.extract(e2);
    try {
      _assertVec(a, b, 'fe.step');
    } on _Mismatch catch (m) {
      _fail(m.toString());
      return;
    }
  }
  _pass('FE stable through trajectory');
}

void testPolicyDeterminism() {
  _banner('PolicyNetwork forward/greedy determinism');
  final net = ai.PolicyNetwork(inputSize: 10 + 5, h1: 64, h2: 64, seed: 2025);
  final rnd = math.Random(7); // only used for stochastic path (not here)

  // Fake input
  final x = List<double>.generate(15, (i) => math.sin(i * 0.37));

  // Two independent forwards must be identical
  final g1 = net.actGreedy(x);
  final g2 = net.actGreedy(x);

  final same =
      (g1.$1 == g2.$1) &&
          (g1.$2 == g2.$2) &&
          (g1.$3 == g2.$3) &&
          _deq(g1.$4[0], g2.$4[0]) &&
          _deq(g1.$4[1], g2.$4[1]) &&
          _deq(g1.$4[2], g2.$4[2]) &&
          _deq(g1.$4[3], g2.$4[3]);

  if (!same) {
    _fail('Policy greedy mismatch');
    return;
  }

  // Stochastic path: with fixed rng and temps=1, epsilon=0 must be reproducible sequence
  final s1 = net.act(x, rnd, tempThr: 1.0, tempTurn: 1.0, epsilon: 0.0);
  final s2 = net.act(x, rnd, tempThr: 1.0, tempTurn: 1.0, epsilon: 0.0);
  // same rnd instance advances; so s1 != s2 by design
  // Run again from fresh rngs to confirm repeatability:
  final rA = math.Random(42);
  final rB = math.Random(42);
  final sA = net.act(x, rA, tempThr: 1.0, tempTurn: 1.0, epsilon: 0.0);
  final sB = net.act(x, rB, tempThr: 1.0, tempTurn: 1.0, epsilon: 0.0);

  final stochSame = (sA.$1 == sB.$1) && (sA.$2 == sB.$2) && (sA.$3 == sB.$3);
  if (!stochSame) {
    _fail('Policy stochastic sampling not reproducible with same seed');
    return;
  }

  _pass('Policy forward/greedy deterministic; stochastic reproducible with seeded RNG');
}

void testTrainerEvalDeterminism() {
  _banner('Trainer.runEpisode determinism (eval greedy)');
  final cfg = _baseCfg(lockTerrain: true, lockSpawn: true, randomSpawnX: false, seed: 77, terrainSeed: 88);
  final env = eng.GameEngine(cfg);

  final fe = ai.FeatureExtractor(groundSamples: 3, stridePx: 48);
  final pol = ai.PolicyNetwork(inputSize: fe.inputSize, h1: 64, h2: 64, seed: 1234);

  final evalTrainer = ai.Trainer(
    env: env,
    fe: fe,
    policy: pol,
    dt: 1/60,
    gamma: 0.99,
    seed: 13,
    tempThr: 1e-6,
    tempTurn: 1e-6,
    epsilon: 0.0,
    entropyBeta: 0.0,
  );

  env.reset(seed: 999);
  final ep1 = evalTrainer.runEpisode(train: false, lr: 0.0, greedy: true);
  env.reset(seed: 999);
  final ep2 = evalTrainer.runEpisode(train: false, lr: 0.0, greedy: true);

  if (ep1.steps != ep2.steps || !_deq(ep1.totalCost, ep2.totalCost)) {
    _fail('Eval mismatch: steps ${ep1.steps} vs ${ep2.steps} | cost ${_fmt(ep1.totalCost)} vs ${_fmt(ep2.totalCost)}');
    return;
  }
  _pass('Trainer eval greedy stable on fixed seed');
}

void main() {
  try {
    testTerrainDeterminism();
    testResetDeterminism();
    testStepDeterminism();
    testFeatureExtractorDeterminism();
    testPolicyDeterminism();
    testTrainerEvalDeterminism();
  } catch (e, st) {
    _fail('Unhandled exception: $e');
    print(st);
  }

  print('\nDone.');
}
