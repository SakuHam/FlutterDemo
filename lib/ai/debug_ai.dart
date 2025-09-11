// lib/ai/debug_ai.dart
import 'dart:math' as math;

import '../engine/game_engine.dart' as eng;   // GameEngine (physics)
import '../engine/types.dart' as et;          // ControlInput, etc.
import 'agent.dart' show FeatureExtractor, PolicyNetwork, PolicyOps;

/// One-frame probe for the policy at the current env state.
/// - greedy=true: deterministic (argmax/threshold)
/// - greedy=false: stochastic sample
/// - doStep=true: advance the env one step with chosen actions
void aiSanityProbe({
  required eng.GameEngine env,
  required PolicyNetwork policy,
  required FeatureExtractor fe,
  bool greedy = true,
  bool doStep = false,
  double dt = 1 / 60.0,
  math.Random? rnd, // only used when greedy == false
}) {
  // 1) Build observation
  final x = fe.extract(env);

  // 2) Query policy
  final res = greedy
      ? policy.actGreedy(x)                        // (th, L, R, probs, cache)
      : policy.act(x, rnd ?? math.Random());       // same tuple

  final bool thrust = res.$1;
  final bool left   = res.$2;
  final bool right  = res.$3;
  final List<double> probs = res.$4;

  // 3) Unpack probabilities
  final double pThr   = probs[0]; // Bernoulli for thrust
  final double pNone  = probs[1]; // turn class 0: none
  final double pLeft  = probs[2]; // turn class 1: left
  final double pRight = probs[3]; // turn class 2: right

  // 4) Diagnostics
  final padCx = env.terrain.padCenter;
  final dx    = env.lander.pos.x - padCx;
  final ang   = env.lander.angle;
  final vx    = env.lander.vel.x;
  final vy    = env.lander.vel.y;

  print(
      '[AI] pThr=${pThr.toStringAsFixed(2)} | '
          'pTurn[n/l/r]=${pNone.toStringAsFixed(2)}/${pLeft.toStringAsFixed(2)}/${pRight.toStringAsFixed(2)} '
          '→ act: thr=$thrust L=$left R=$right | '
          'dx=${dx.toStringAsFixed(1)} '
          'ang=${(ang * 180 / math.pi).toStringAsFixed(1)}° '
          'v=(${vx.toStringAsFixed(1)},${vy.toStringAsFixed(1)})'
  );

  if (doStep) {
    // NOTE: ControlInput now comes from types.dart → et.ControlInput
    env.step(dt, et.ControlInput(thrust: thrust, left: left, right: right));
  }
}

/// Quick wiring asserts: left should DECREASE angle, right should INCREASE angle.
void assertTurnWiring(eng.GameEngine env, {double dt = 1/60.0}) {
  final a0 = env.lander.angle;
  env.step(dt, const et.ControlInput(thrust: false, left: true, right: false));
  final a1 = env.lander.angle;
  assert(a1 < a0, 'Left should decrease angle; wiring may be flipped/dropped.');

  env.step(dt, const et.ControlInput(thrust: false, left: false, right: true));
  final a2 = env.lander.angle;
  assert(a2 > a1, 'Right should increase angle.');
}
