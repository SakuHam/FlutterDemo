// lib/ai/debug_ai.dart
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import '../engine/game_engine.dart' as eng;   // GameEngine (physics)
import '../engine/types.dart' as et;          // ControlInput, etc.
import 'agent.dart'
    show
    FeatureExtractor,
    PolicyNetwork,
    PolicyOps,             // act/actGreedy/actIntent*
    controllerForIntent,   // low-level controller (engine-side)
    Intent,
    kIntentNames;

/// ----------------------- IO helpers -----------------------
Future<Map<String, dynamic>> _readJson(String path) async =>
    json.decode(await File(path).readAsString()) as Map<String, dynamic>;

List<List<double>> _mat(dynamic m) => (m as List)
    .map<List<double>>(
      (r) => (r as List).map<double>((x) => (x as num).toDouble()).toList(),
)
    .toList();

List<double> _vec(dynamic v) =>
    (v as List).map<double>((x) => (x as num).toDouble()).toList();

/// Loads a PolicyNetwork from JSON (supports both single-stage and two-stage heads).
/// Returns (policy, groundSamples, stridePx).
(PolicyNetwork policy, int gs, double stride) _loadPolicy(String path) {
  final js = File(path).existsSync()
      ? File(path).readAsStringSync()
      : (throw ArgumentError('Policy file not found: $path'));
  final j = json.decode(js) as Map<String, dynamic>;

  final inputSize = j['inputSize'] as int;
  final h1 = j['h1'] as int? ?? 64;
  final h2 = j['h2'] as int? ?? 64;

  var policy = PolicyNetwork(inputSize: inputSize, h1: h1, h2: h2, seed: 1234);

  // trunk
  policy
    ..W1 = _mat(j['W1'])
    ..b1 = _vec(j['b1'])
    ..W2 = _mat(j['W2'])
    ..b2 = _vec(j['b2']);

  // single-stage heads (if present)
  if (j.containsKey('W_thr')) {
    policy
      ..W_thr = _mat(j['W_thr'])
      ..b_thr = _vec(j['b_thr']);
  }
  if (j.containsKey('W_turn')) {
    policy
      ..W_turn = _mat(j['W_turn'])
      ..b_turn = _vec(j['b_turn']);
  }

  // two-stage heads (if present)
  if (j.containsKey('W_intent')) {
    policy
      ..W_intent = _mat(j['W_intent'])
      ..b_intent = _vec(j['b_intent']);
  }
  if (j.containsKey('W_val')) {
    policy
      ..W_val = _mat(j['W_val'])
      ..b_val = _vec(j['b_val']);
  }

  final fe = (j['fe'] as Map<String, dynamic>?) ?? const {};
  final gs = (fe['groundSamples'] as int?) ?? (inputSize - 10);
  final stride = (fe['stridePx'] as num?)?.toDouble() ?? 48.0;

  return (policy, gs, stride);
}

/// ----------------------- Sanity probes -----------------------

/// One-frame probe for the (single-stage) action heads at current state.
void aiSanityProbeSingle({
  required eng.GameEngine env,
  required PolicyNetwork policy,
  required FeatureExtractor fe,
  bool greedy = true,
  math.Random? rnd,
}) {
  final x = fe.extract(env);
  final res = greedy ? policy.actGreedy(x) : policy.act(x, rnd ?? math.Random());
  final thrust = res.$1, left = res.$2, right = res.$3;
  final probs = res.$4;

  final pThr = probs[0];
  final pNone = probs[1];
  final pLeft = probs[2];
  final pRight = probs[3];

  final padCx = env.terrain.padCenter;
  final dx = env.lander.pos.x - padCx;
  final ang = env.lander.angle;
  final vx = env.lander.vel.x;
  final vy = env.lander.vel.y;

  stdout.writeln(
    '[SingleStage] pThr=${pThr.toStringAsFixed(2)} | '
        'pTurn[n/l/r]=${pNone.toStringAsFixed(2)}/${pLeft.toStringAsFixed(2)}/${pRight.toStringAsFixed(2)} '
        '→ act: T=$thrust L=$left R=$right | '
        'dx=${dx.toStringAsFixed(1)} ang=${(ang*180/math.pi).toStringAsFixed(1)}° '
        'v=(${vx.toStringAsFixed(1)},${vy.toStringAsFixed(1)})',
  );
}

/// Assert that left decreases angle and right increases angle.
void assertTurnWiring(eng.GameEngine env, {double dt = 1 / 60.0}) {
  final a0 = env.lander.angle;
  env.step(dt, const et.ControlInput(thrust: false, left: true, right: false));
  final a1 = env.lander.angle;
  if (!(a1 < a0)) {
    throw StateError('Turn wiring: LEFT did not decrease angle (may be flipped).');
  }

  env.step(dt, const et.ControlInput(thrust: false, left: false, right: true));
  final a2 = env.lander.angle;
  if (!(a2 > a1)) {
    throw StateError('Turn wiring: RIGHT did not increase angle.');
  }
}

/// Run a short two-stage planner simulation (greedy intents) to see if intents map
/// to reasonable controls and produce motion.
void probeTwoStagePlanner({
  required eng.GameEngine env,
  required PolicyNetwork policy,
  required FeatureExtractor fe,
  int steps = 240,
  int planHold = 12,
  double dt = 1 / 60.0,
}) {
  if (policy.W_intent.isEmpty) {
    stdout.writeln('[TwoStage] No intent head found in policy (skipping planner probe).');
    return;
  }

  int framesLeft = 0;
  int currentIntentIdx = -1;

  int thrustSteps = 0, turnSteps = 0;
  final intentCount = List<int>.filled(policy.W_intent.length, 0);

// --- constants must mirror runtime/controller ---
  const double vxGoalAbs = 60.0;
  const double kAngV     = 0.012;
  const double kDxHover  = 0.40;
  const double maxTilt   = 15 * math.pi / 180;
  const double angDead   = 3 * math.pi / 180;

  int consecutiveOpposite = 0;
  const int warnAfter = 4; // require N opposite frames before warning

  for (int t = 0; t < steps; t++) {
    if (framesLeft <= 0) {
      final xPlan = fe.extract(env);
      final (idx, probs, _) = policy.actIntentGreedy(xPlan);
      currentIntentIdx = idx;
      framesLeft = planHold;
      intentCount[idx]++;
      final name = (idx >= 0 && idx < kIntentNames.length) ? kIntentNames[idx] : 'intent#$idx';
      stdout.writeln('[TwoStage] t=$t intent=$name probs=${probs.map((v)=>v.toStringAsFixed(2)).toList()}');
    }

    final intent = Intent.values[currentIntentIdx];
    final u = controllerForIntent(intent, env); // engine-side controller

    // ---- expected turn direction from controller math (not dx) ----
    final L  = env.lander;
    final padCx = env.terrain.padCenter;
    final dx = L.pos.x - padCx; // still useful for logging

    // desired horizontal velocity
    double vxDes = 0.0;
    switch (intent) {
      case Intent.goLeft:      vxDes = -vxGoalAbs; break;
      case Intent.goRight:     vxDes = vxGoalAbs; break;
      case Intent.hoverCenter: vxDes = -kDxHover * dx; break;
      case Intent.descendSlow:
      case Intent.brakeUp:     vxDes = 0.0; break;
    }

    // target angle and required turn sign
    final targetAngle = (kAngV * (vxDes - L.vel.x)).clamp(-maxTilt, maxTilt);
    final needTurn =
    targetAngle.abs() <= angDead ? 0 : (targetAngle > 0 ? 1 : -1); // +1=RIGHT, -1=LEFT
    final didTurn = u.right ? 1 : (u.left ? -1 : 0);

    // only flag persistent opposite commands
    if (needTurn != 0 && didTurn != 0 && (needTurn + didTurn == 0)) {
      consecutiveOpposite++;
      if (consecutiveOpposite >= warnAfter) {
        stdout.writeln(
            '  ⚠ Opposite turn for ${kIntentNames[currentIntentIdx]} '
                '(need ${needTurn>0?'RIGHT':'LEFT'}, did ${didTurn>0?'RIGHT':'LEFT'}) '
                'dx=${dx.toStringAsFixed(1)} vx=${L.vel.x.toStringAsFixed(1)} '
                'vxDes=${vxDes.toStringAsFixed(1)} targetAng=${(targetAngle*180/math.pi).toStringAsFixed(1)}°'
        );
        consecutiveOpposite = 0; // rate-limit warnings
      }
    } else {
      consecutiveOpposite = 0;
    }

    if (u.thrust) thrustSteps++;
    if (u.left || u.right) turnSteps++;

    env.step(dt, et.ControlInput(thrust: u.thrust, left: u.left, right: u.right));
    framesLeft--;
  }

  stdout.writeln('[TwoStage] summary: thrustSteps=$thrustSteps turnSteps=$turnSteps');
  stdout.writeln('[TwoStage] intent histogram: $intentCount');
  if (thrustSteps == 0 && turnSteps == 0) {
    stdout.writeln('  ⚠ Controls were never asserted — check head outputs or controller.');
  }
}

/// Quick single-stage roll to see if heads are “dead”.
void rollSingleStage({
  required eng.GameEngine env,
  required PolicyNetwork policy,
  required FeatureExtractor fe,
  int steps = 240,
  double dt = 1 / 60.0,
  math.Random? rnd,
}) {
  int thrustSteps = 0, leftSteps = 0, rightSteps = 0;
  for (int t = 0; t < steps; t++) {
    final x = fe.extract(env);
    final res = policy.act(x, rnd ?? math.Random(), tempThr: 0.9, tempTurn: 1.2, epsilon: 0.05);
    final th = res.$1, lf = res.$2, rt = res.$3;
    if (th) thrustSteps++;
    if (lf) leftSteps++;
    if (rt) rightSteps++;
    env.step(dt, et.ControlInput(thrust: th, left: lf, right: rt));
  }
  stdout.writeln('[SingleStage] summary: thrust=$thrustSteps left=$leftSteps right=$rightSteps');
  if (thrustSteps == 0 && leftSteps == 0 && rightSteps == 0) {
    stdout.writeln('  ⚠ Action heads appear inactive (all zeros).');
  }
}

/// ----------------------- Main (CLI) -----------------------
void main(List<String> args) async {
  // ---- CLI flags ----
  String policyPath = 'policy.json';
  int seed = 12345;
  int steps = 300;
  bool doSingleProbe = true;
  bool doSingleRoll = true;
  bool doTwoStageProbe = true;
  bool greedyProbe = true;

  for (final a in args) {
    if (a.startsWith('--policy=')) policyPath = a.substring(9);
    if (a.startsWith('--seed=')) seed = int.parse(a.substring(7));
    if (a.startsWith('--steps=')) steps = int.parse(a.substring(8));
    if (a == '--single_probe=false') doSingleProbe = false;
    if (a == '--single_roll=false') doSingleRoll = false;
    if (a == '--twostage=false') doTwoStageProbe = false;
    if (a == '--stochastic=true') greedyProbe = false;
  }

  // ---- Engine config (match trainer) ----
  final cfg = et.EngineConfig(
    t: et.Tunables(
      gravity: 0.18,
      thrustAccel: 0.42,
      rotSpeed: 1.6,
      maxFuel: 100.0,
    ),
    worldW: 360,
    worldH: 640,
    lockTerrain: true,
    terrainSeed: 12345,
    lockSpawn: true,
    randomSpawnX: false,
    spawnXMin: 0.20,
    spawnXMax: 0.80,
    landingSpeedMax: 12.0,
    landingAngleMaxRad: 8 * math.pi / 180.0,
    wDx: 200.0,
    wDy: 180.0,
    wVx: 90.0,
    wVyDown: 200.0,
    wAngleDeg: 80.0,
  );

  final env = eng.GameEngine(cfg)..reset(seed: seed);

  // ---- Load policy & FE ----
  final (policy, feGS, feStride) = _loadPolicy(policyPath);
  final fe = FeatureExtractor(groundSamples: feGS, stridePx: feStride);

  // FE–policy shape check
  if (fe.inputSize != policy.inputSize) {
    stderr.writeln('ERROR: FeatureExtractor.inputSize=${fe.inputSize} '
        '!= policy.inputSize=${policy.inputSize}. '
        'Fix your JSON/FE config to match exactly.');
    exit(1);
  }

  stdout.writeln('Loaded policy "$policyPath" | h1=${policy.h1} h2=${policy.h2} '
      '| FE(gs=$feGS stride=$feStride)');

  // ---- Wiring check (left/right) ----
  try {
    final envCopy = eng.GameEngine(cfg)..reset(seed: seed);
    assertTurnWiring(envCopy);
    stdout.writeln('Turn wiring OK (LEFT decreases angle, RIGHT increases).');
  } catch (e) {
    stderr.writeln('⚠ $e');
  }

  // ---- Single-stage one-frame probe ----
  if (doSingleProbe) {
    final envCopy = eng.GameEngine(cfg)..reset(seed: seed);
    aiSanityProbeSingle(
      env: envCopy,
      policy: policy,
      fe: fe,
      greedy: greedyProbe,
    );
  }

  // ---- Short single-stage roll (stochastic) ----
  if (doSingleRoll) {
    final envCopy = eng.GameEngine(cfg)..reset(seed: seed);
    rollSingleStage(env: envCopy, policy: policy, fe: fe, steps: steps);
  }

  // ---- Two-stage (planner) probe ----
  if (doTwoStageProbe) {
    final envCopy = eng.GameEngine(cfg)..reset(seed: seed);
    probeTwoStagePlanner(
      env: envCopy,
      policy: policy,
      fe: fe,
      steps: steps,
      planHold: 12,
    );
  }

  // ---- Extra heuristic: does chosen intent direction match dx sign? ----
  if (policy.W_intent.isNotEmpty) {
    final envCopy = eng.GameEngine(cfg)..reset(seed: seed);
    int mismatchCount = 0, checks = 0;

    for (int t = 0; t < steps; t++) {
      final x = fe.extract(envCopy);
      final (idx, probs, _) = policy.actIntentGreedy(x);
      final intentName = (idx >= 0 && idx < kIntentNames.length)
          ? kIntentNames[idx]
          : 'intent#$idx';

      final dx = envCopy.lander.pos.x - envCopy.terrain.padCenter;
      if (dx.abs() > 10) { // only check when clearly on one side
        if (dx > 0 && intentName == 'goRight') mismatchCount++;
        if (dx < 0 && intentName == 'goLeft') mismatchCount++;
        checks++;
      }

      final intent = Intent.values[idx];
      final u = controllerForIntent(intent, envCopy);
      envCopy.step(1/60.0, et.ControlInput(thrust: u.thrust, left: u.left, right: u.right));
    }

    if (checks > 0) {
      final pct = 100.0 * mismatchCount / checks;
      stdout.writeln('[IntentDirCheck] mismatches=$mismatchCount/$checks (${pct.toStringAsFixed(1)}%) '
          '→ high value suggests planner may be picking the wrong side intent.');
    }
  }

  stdout.writeln('Done.');
}
