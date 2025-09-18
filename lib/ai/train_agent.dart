// lib/ai/train_agent.dart
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import '../engine/types.dart' as et;
import '../engine/game_engine.dart' as eng;
import '../engine/raycast.dart'; // RayConfig

import 'agent.dart' as ai; // FeatureExtractorRays, PolicyNetwork, Trainer, RunningNorm, kIntentNames, predictiveIntentLabelAdaptive
import 'agent.dart';       // bring symbols into scope (PolicyNetwork etc.)
import 'potential_field.dart'; // buildPotentialField, PotentialField

/* ------------------------------- tiny arg parser ------------------------------- */

class _Args {
  final Map<String, String?> _kv = {};
  final Set<String> _flags = {};
  _Args(List<String> argv) {
    for (final a in argv) {
      if (!a.startsWith('--')) continue;
      final s = a.substring(2);
      final i = s.indexOf('=');
      if (i >= 0) {
        _kv[s.substring(0, i)] = s.substring(i + 1);
      } else {
        _flags.add(s);
      }
    }
  }
  String? getStr(String k, {String? def}) => _kv[k] ?? def;
  int getInt(String k, {int def = 0}) => int.tryParse(_kv[k] ?? '') ?? def;
  double getDouble(String k, {double def = 0.0}) => double.tryParse(_kv[k] ?? '') ?? def;
  bool getFlag(String k, {bool def = false}) => _flags.contains(k) ? true : def;
}

/* ------------------------------- feature signature ------------------------------ */

String _feSignature({
  required int inputSize,
  required int rayCount,
  required bool kindsOneHot,
  required double worldW,
  required double worldH,
}) {
  return 'kind=rays;in=$inputSize;rays=$rayCount;1hot=$kindsOneHot;W=${worldW.toInt()};H=${worldH.toInt()}';
}

/* ------------------------------- matrix helpers -------------------------------- */

List<List<double>> _deepCopyMat(List<List<double>> W) =>
    List.generate(W.length, (i) => List<double>.from(W[i]));

List<List<double>> _xavier(int out, int inp, int seed) {
  final r = math.Random(seed);
  final limit = math.sqrt(6.0 / (out + inp));
  return List.generate(out, (_) => List<double>.generate(inp, (_) => (r.nextDouble() * 2 - 1) * limit));
}

/* ------------------------------- policy IO (json) ------------------------------ */

Map<String, dynamic> _policyToJson({
  required PolicyNetwork p,
  required int rayCount,
  required bool kindsOneHot,
  required eng.GameEngine env,
  RunningNorm? norm,
}) {
  final sig = _feSignature(
    inputSize: p.inputSize,
    rayCount: rayCount,
    kindsOneHot: kindsOneHot,
    worldW: env.cfg.worldW,
    worldH: env.cfg.worldH,
  );

  final trunkJson = <Map<String, dynamic>>[];
  for (final layer in p.trunk.layers) {
    trunkJson.add({'W': _deepCopyMat(layer.W), 'b': List<double>.from(layer.b)});
  }

  Map<String, dynamic> headJson(layer) => {
    'W': _deepCopyMat(layer.W),
    'b': List<double>.from(layer.b),
  };

  final t = env.cfg.t;

  final m = <String, dynamic>{
    'arch': {
      'input': p.inputSize,
      'hidden': p.hidden,
      'kIntents': PolicyNetwork.kIntents,
    },
    'trunk': trunkJson,
    'heads': {
      'intent': headJson(p.heads.intent),
      'turn': headJson(p.heads.turn),
      'thr': headJson(p.heads.thr),
      'val': headJson(p.heads.val),
    },
    'feature_extractor': {
      'kind': 'rays',
      'rayCount': rayCount,
      'kindsOneHot': kindsOneHot,
    },
    'env_hint': {'worldW': env.cfg.worldW, 'worldH': env.cfg.worldH},

    // Persist physics knobs so runtime can mirror them
    'physics': {
      'gravity': t.gravity,
      'thrustAccel': t.thrustAccel,
      'rotSpeed': t.rotSpeed,
      'maxFuel': t.maxFuel,

      'rcsEnabled': t.rcsEnabled,
      'rcsAccel': t.rcsAccel,
      'rcsBodyFrame': t.rcsBodyFrame,

      'downThrEnabled': t.downThrEnabled,
      'downThrAccel': t.downThrAccel,
      'downThrBurn': t.downThrBurn,
    },

    'signature': sig,
    'format': 'v2rays',
  };

  if (norm != null && norm.inited && norm.dim == p.inputSize) {
    m['norm'] = {
      'dim': norm.dim,
      'momentum': norm.momentum,
      'mean': norm.mean,
      'var': norm.var_,
      'signature': sig,
    };
    // legacy mirrors
    m['norm_mean'] = norm.mean;
    m['norm_var'] = norm.var_;
    m['norm_momentum'] = norm.momentum;
    m['norm_signature'] = sig;
  }

  return m;
}

void _savePolicy({
  required String path,
  required PolicyNetwork p,
  required int rayCount,
  required bool kindsOneHot,
  required eng.GameEngine env,
  RunningNorm? norm,
}) {
  final f = File(path);
  final jsonMap = _policyToJson(
    p: p,
    rayCount: rayCount,
    kindsOneHot: kindsOneHot,
    env: env,
    norm: norm,
  );
  f.writeAsStringSync(const JsonEncoder.withIndent('  ').convert(jsonMap));
  print('Saved policy → $path');
}

/* --------------------------------- env config --------------------------------- */

et.EngineConfig makeConfig({
  int seed = 42,
  bool lockTerrain = true,
  bool lockSpawn = true,
  bool randomSpawnX = false,
  double worldW = 800,
  double worldH = 600,
  double? maxFuel,
  bool crashOnTilt = false,

  // physics flags/values
  double gravity = 0.18,
  double thrustAccel = 0.42,
  double rotSpeed = 1.6,

  bool rcsEnabled = false,
  double rcsAccel = 0.12,
  bool rcsBodyFrame = true,

  bool downThrEnabled = false,
  double downThrAccel = 0.30,
  double downThrBurn = 10.0,
}) {
  final t = et.Tunables(
    gravity: gravity,
    thrustAccel: thrustAccel,
    rotSpeed: rotSpeed,
    maxFuel: maxFuel ?? 1000.0,
    crashOnTilt: crashOnTilt,
    landingMaxVx: 28.0,
    landingMaxVy: 38.0,
    landingMaxOmega: 3.5,

    // RCS
    rcsEnabled: rcsEnabled,
    rcsAccel: rcsAccel,
    rcsBodyFrame: rcsBodyFrame,

    // Downward thruster
    downThrEnabled: downThrEnabled,
    downThrAccel: downThrAccel,
    downThrBurn: downThrBurn,
  );
  return et.EngineConfig(
    worldW: worldW,
    worldH: worldH,
    t: t,
    seed: seed,
    stepScale: 60.0,
    lockTerrain: lockTerrain,
    terrainSeed: 1234567,
    lockSpawn: lockSpawn,
    randomSpawnX: randomSpawnX,
    hardWalls: true,
  );
}

/* ----------------------------- determinism probe ------------------------------ */

typedef _RolloutRes = ({int steps, double cost});

_RolloutRes _probeDeterminism(eng.GameEngine env, {int maxSteps = 200}) {
  var cost = 0.0;
  int t = 0;
  while (t < maxSteps) {
    final info = env.step(1 / 60.0, const et.ControlInput(thrust: false, left: false, right: false));
    cost += info.costDelta;
    if (info.terminal) break;
    t++;
  }
  return (steps: t, cost: cost);
}

/* ----------------------------------- eval ------------------------------------- */

class EvalStats {
  double meanCost = 0;
  double medianCost = 0;
  double landPct = 0;
  double crashPct = 0;
  double meanSteps = 0;
  double meanAbsDx = 0;
}

EvalStats evaluate({
  required eng.GameEngine env,
  required Trainer trainer,
  int episodes = 40,
  int seed = 123,
  int attemptsPerTerrain = 1, // reuse terrain across N episodes
}) {
  final rnd = math.Random(seed);
  final costs = <double>[];
  int landed = 0, crashed = 0, stepsSum = 0;
  double absDxSum = 0.0;

  int terrAttempts = 0;
  int currentTerrainSeed = rnd.nextInt(1 << 30);

  for (int i = 0; i < episodes; i++) {
    if (terrAttempts == 0) {
      currentTerrainSeed = rnd.nextInt(1 << 30);
    }

    env.reset(seed: currentTerrainSeed);
    final res = trainer.runEpisode(
      train: false,
      greedy: true,
      scoreIsReward: false,
    );

    terrAttempts = (terrAttempts + 1) % attemptsPerTerrain;

    costs.add(res.totalCost);
    stepsSum += res.steps;
    if (env.status == et.GameStatus.landed) {
      landed++;
    } else {
      crashed++;
    }
    final padCx = env.terrain.padCenter;
    absDxSum += (env.lander.pos.x - padCx).abs();
  }

  costs.sort();
  final st = EvalStats();
  st.meanCost = costs.isEmpty ? 0 : costs.reduce((a, b) => a + b) / costs.length;
  st.medianCost = costs.isEmpty ? 0 : costs[costs.length ~/ 2];
  st.landPct = 100.0 * landed / episodes;
  st.crashPct = 100.0 * crashed / episodes;
  st.meanSteps = stepsSum / episodes;
  st.meanAbsDx = absDxSum / episodes;
  return st;
}

/* ----------------------------- norm warmup (optional) -------------------------- */

void _warmFeatureNorm({
  required RunningNorm? norm,
  required Trainer trainer,
  required FeatureExtractorRays fe,
  required eng.GameEngine env,
  int perClass = 600,
  int seed = 4242,
}) {
  if (norm == null) return;
  final r = math.Random(seed);
  env.reset(seed: 1234567);
  int accepted = 0, target = perClass * PolicyNetwork.kIntents;

  while (accepted < target) {
    final want = r.nextInt(PolicyNetwork.kIntents);

    final padCx = env.terrain.padCenter.toDouble();
    final padHalfW =
    (((env.terrain.padX2 - env.terrain.padX1).abs()) * 0.5).clamp(12.0, env.cfg.worldW.toDouble());
    double x = padCx, h = 180, vx = 0, vy = 20;

    switch (want) {
      case 1: // goLeft
        x = (padCx - (0.22 + 0.18 * r.nextDouble()) * env.cfg.worldW).clamp(10.0, env.cfg.worldW - 10.0);
        h = 120 + 120 * r.nextDouble();
        vx = (r.nextDouble() * 14.0) - 7.0;
        vy = 28.0 + 10.0 * r.nextDouble();
        break;
      case 2: // goRight
        x = (padCx + (0.22 + 0.18 * r.nextDouble()) * env.cfg.worldW).clamp(10.0, env.cfg.worldW - 10.0);
        h = 120 + 120 * r.nextDouble();
        vx = (r.nextDouble() * 14.0) - 7.0;
        vy = 28.0 + 10.0 * r.nextDouble();
        break;
      case 3: // descendSlow
        x = padCx + (r.nextDouble() * 0.05 - 0.025) * padHalfW;
        h = 0.55 * env.cfg.worldH + 0.20 * env.cfg.worldH * r.nextDouble();
        vx = (r.nextDouble() * 24.0) - 12.0;
        vy = 24.0 + 12.0 * r.nextDouble();
        break;
      case 4: // brakeUp
        x = padCx + (r.nextDouble() * 0.05 - 0.025) * padHalfW;
        h = 40.0 + 50.0 * r.nextDouble();
        vx = (r.nextDouble() * 14.0) - 7.0;
        vy = 120.0 + 50.0 * r.nextDouble();
        break;
      case 5: // brakeLeft
        x = (padCx + (r.nextDouble() * 0.02 - 0.01) * padHalfW).clamp(10.0, padHalfW - 10.0);
        h = 120 + 100 * r.nextDouble();
        vx = 28.0 + 24.0 * r.nextDouble(); // +vx
        vy = 18.0 + 18.0 * r.nextDouble();
        break;
      case 6: // brakeRight
        x = (padCx + (r.nextDouble() * 0.02 - 0.01) * padHalfW).clamp(10.0, padHalfW - 10.0);
        h = 120 + 100 * r.nextDouble();
        vx = -(28.0 + 24.0 * r.nextDouble()); // -vx
        vy = 18.0 + 18.0 * r.nextDouble();
        break;
      default: // hover
        x = padCx + (r.nextDouble() * 0.03 - 0.015) * padHalfW;
        h = 0.20 * env.cfg.worldH + 0.15 * env.cfg.worldH * r.nextDouble();
        vx = (r.nextDouble() * 10.0) - 5.0;
        vy = (r.nextDouble() * 10.0) - 5.0;
        break;
    }

    final gy = env.terrain.heightAt(x);
    env.lander
      ..pos.x = x.clamp(10.0, env.cfg.worldW - 10.0)
      ..pos.y = (gy - h).clamp(0.0, env.cfg.worldH - 10.0)
      ..vel.x = vx
      ..vel.y = vy
      ..angle = 0.0
      ..fuel = env.cfg.t.maxFuel;

    if (predictiveIntentLabelAdaptive(env) != want) continue;

    final feat = fe.extract(env);
    try {
      (trainer.norm as dynamic).observe(feat);
    } catch (_) {
      trainer.norm?.normalize(feat, update: true);
    }
    accepted++;
  }
  print('Feature norm warmed with $accepted synthetic samples.');
}

/* ----------------------------- PF velocity-only reward ------------------------- */

/* -------------------- PF velocity-only reward (final approach boost) -------------------- */

class PFShapingCfg {
  // base weights
  final double wAlign;       // reward per unit cos(v, flow)
  final double wVelDelta;    // base penalty for ||v - v_pf|| / vmax

  // distance-shaped target-speed parameters
  final double vMinClose;    // far-to-near baseline (will get tapered further near pad)
  final double vMaxFar;
  final double alpha;        // taper sharpness

  // scaling + norms
  final double vmax;         // normalization for velocity error

  // NEW: near-pad emphasis
  final double padTightFrac; // horizontal tight zone as fraction of worldW (e.g., 0.10)
  final double hTight;       // vertical tight zone in px (e.g., 140)
  final double latBoost;     // extra weight on lateral error near pad (e.g., 4.0)
  final double velPenaltyBoost; // multiplies overall v-error near pad (e.g., 3.0)
  final double alignBoost;      // multiplies align reward near pad (e.g., 1.5)
  final double vMinTouchdown;   // desired |v| right at pad (e.g., 1.5..3.0)

  // NEW: feasibility clamp (prevents unwinnable target)
  final double feasiness;    // fraction of a_max*dt allowed change in v_pf per frame (0..1)

  final double xBias;        // stronger X (lateral) weighting overall

  const PFShapingCfg({
    this.wAlign = 1.0,
    this.wVelDelta = 0.6,
    this.vMinClose = 8.0,
    this.vMaxFar = 90.0,
    this.alpha = 1.2,
    this.vmax = 140.0,

    this.padTightFrac = 0.10,
    this.hTight = 140.0,
    this.latBoost = 4.0,
    this.velPenaltyBoost = 3.0,
    this.alignBoost = 1.5,
    this.vMinTouchdown = 2.0,

    this.feasiness = 0.75,

    this.xBias = 3.0,
  });
}

/// Reward = wAlign_eff * cos(v, flow) - wVel_eff * ||W*(v - v_pf)|| / vmax
/// where W increases lateral weighting near the pad, and v_pf tapers to ~0 near pad.
ai.ExternalRewardHook makePFRewardHook({
  required eng.GameEngine env,
  PFShapingCfg cfg = const PFShapingCfg(),
}) {
  final pf = buildPotentialField(env, nx: 160, ny: 120, iters: 1200, omega: 1.7, tol: 1e-4);

  // Effective max linear accel per real second (rough estimate from engine):
  // In GameEngine: accel uses (t.thrustAccel * 0.05) then multiplied by (dt * stepScale).
  final double aMax = env.cfg.t.thrustAccel * 0.05 * env.cfg.stepScale; // px/s^2

  return ({required eng.GameEngine env, required double dt, required int tStep}) {
    final x = env.lander.pos.x.toDouble();
    final y = env.lander.pos.y.toDouble();
    final vx = env.lander.vel.x.toDouble();
    final vy = env.lander.vel.y.toDouble();

    // Proximity to pad
    final padCx = env.terrain.padCenter.toDouble();
    final dxAbs = (x - padCx).abs();
    final W = env.cfg.worldW.toDouble();

    final gy = env.terrain.heightAt(x);
    final h  = (gy - y).toDouble().clamp(0.0, 1000.0);

    final tightX = cfg.padTightFrac * W;
    final px_ = math.exp(- (dxAbs*dxAbs) / (tightX*tightX + 1e-6));
    final ph_ = math.exp(- (h*h)       / (cfg.hTight*cfg.hTight + 1e-6));
    final prox = (px_ * ph_).clamp(0.0, 1.0);

    // Alignment (unit flow) term
    final flow = pf.sampleFlow(x, y); // unit (nx, ny)
    final vmag = math.sqrt(vx*vx + vy*vy);
    double align = 0.0;
    if (vmag > 1e-6) {
      align = (vx / vmag) * flow.nx + (vy / vmag) * flow.ny; // [-1,1]
    }

    // Base PF suggestion (fast far, slow near)
    var sugg = pf.suggestVelocity(
      x, y,
      vMinClose: cfg.vMinClose,
      vMaxFar: cfg.vMaxFar,
      alpha: cfg.alpha,
      clampSpeed: 9999.0,
    );

    // Final-approach flare
    final flareLat = (1.0 - 0.90 * prox);
    final flareVer = (1.0 - 0.70 * prox);
    final magNow = math.sqrt(sugg.vx*sugg.vx + sugg.vy*sugg.vy) + 1e-9;
    sugg = (vx: sugg.vx * flareLat, vy: sugg.vy * flareVer);

    final magTarget = ((1.0 - prox) * magNow + prox * cfg.vMinTouchdown).clamp(0.0, magNow);
    final magNew = math.sqrt(sugg.vx*sugg.vx + sugg.vy*sugg.vy) + 1e-9;
    final kMag = (magTarget / magNew).clamp(0.0, 1.0);
    sugg = (vx: sugg.vx * kMag, vy: sugg.vy * kMag);

    // Feasibility clamp
    final dv_pf_x = sugg.vx - vx;
    final dv_pf_y = sugg.vy - vy;
    final dv_pf_mag = math.sqrt(dv_pf_x*dv_pf_x + dv_pf_y*dv_pf_y);
    final dv_pf_cap = (aMax * dt * cfg.feasiness).clamp(0.0, 1e9);
    if (dv_pf_mag > dv_pf_cap && dv_pf_mag > 1e-9) {
      final s = dv_pf_cap / dv_pf_mag;
      sugg = (vx: vx + dv_pf_x * s, vy: vy + dv_pf_y * s);
    }

    // Border avoidance: speed-aware & earlier
    final wallTau = 1.2;
    final wallMarginFrac = 0.22;
    final wallBlendMax = 0.80;
    final wallVInward = 0.90;
    final wallVelPenalty = 3.0;
    final Wworld = env.cfg.worldW.toDouble();
    final distL = x;
    final distR = (Wworld - x);
    final baseBand = wallMarginFrac * Wworld;

    final vxTowardL = (-vx).clamp(0.0, double.infinity);
    final vxTowardR = ( vx).clamp(0.0, double.infinity);

    final warnL = baseBand + wallTau * vxTowardL;
    final warnR = baseBand + wallTau * vxTowardR;

    double proxL = 1.0 - (distL / (warnL + 1e-6)).clamp(0.0, 1.0);
    double proxR = 1.0 - (distR / (warnR + 1e-6)).clamp(0.0, 1.0);

    final borderProx = math.max(proxL, proxR);
    double inwardX = 0.0;
    if (proxL >= proxR && proxL > 0.0) inwardX =  1.0;
    if (proxR >  proxL && proxR > 0.0) inwardX = -1.0;

    final gamma = 1.5;
    final blendIn = wallBlendMax * math.pow(borderProx, gamma);

    final vInward = (wallVInward + 0.8 * (vxTowardL + vxTowardR)).clamp(40.0, 200.0);
    final suggWallVx = inwardX * vInward;

    sugg = (
    vx: (1.0 - blendIn) * sugg.vx + blendIn * suggWallVx,
    vy: sugg.vy * (1.0 - 0.35 * borderProx)
    );

    final wallBoost = 1.0 + wallVelPenalty * borderProx;

    // Stronger X weighting; ramp faster near pad
    final prox2 = prox * prox;
    final wLat = (cfg.xBias) * (1.0 + cfg.latBoost * prox2);     // X
    final wVer = 1.0 * (1.0 + 0.7 * cfg.latBoost * prox2);       // Y

    final dvx = (vx - sugg.vx) * wLat;
    final dvy = (vy - sugg.vy) * wVer;
    final vErr = math.sqrt(dvx*dvx + dvy*dvy) / cfg.vmax;

    final wVelEff = cfg.wVelDelta * (1.0 + cfg.velPenaltyBoost * prox2) * wallBoost;
    final wAlignEff = cfg.wAlign * (1.0 + cfg.alignBoost * prox);

    // touchdown speed bonus
    final speed = vmag;
    final touchTarget = cfg.vMinTouchdown.clamp(0.5, 15.0);
    final touchWeight = 6.0;
    final touchBonus = touchWeight * prox2 * (touchTarget - speed) / (touchTarget + 1e-6);

    final r = wAlignEff * align + touchBonus - wVelEff * vErr;
    return r;
  };
}

/* ------------------------------------ main ------------------------------------ */

List<int> _parseHiddenList(String? s, {List<int> fallback = const [64, 64]}) {
  if (s == null || s.trim().isEmpty) return List<int>.from(fallback);
  final parts = s.split(',').map((t) => t.trim()).where((t) => t.isNotEmpty).toList();
  final out = <int>[];
  for (final p in parts) {
    final v = int.tryParse(p);
    if (v != null && v > 0) out.add(v);
  }
  return out.isEmpty ? List<int>.from(fallback) : out;
}

void main(List<String> argv) {
  final args = _Args(argv);

  final seed = args.getInt('seed', def: 7);

  final iters = args.getInt('train_iters', def: args.getInt('iters', def: 200));
  final batch = args.getInt('batch', def: 1);
  final lr = args.getDouble('lr', def: 3e-4);
  final valueBeta = args.getDouble('value_beta', def: 0.5);
  final huberDelta = args.getDouble('huber_delta', def: 1.0);

  final planHold = args.getInt('plan_hold', def: 1);
  final tempIntent = args.getDouble('intent_temp', def: 1.0);
  final intentEntropy = args.getDouble('intent_entropy', def: 0.0);
  final useLearned = args.getFlag('use_learned_controller', def: false);
  final blendPolicy = args.getDouble('blend_policy', def: 1.0);
  final intentAlignWeight = args.getDouble('intent_align', def: 0.25);
  final intentPgWeight = args.getDouble('intent_pg', def: 0.6);
  final actionAlignWeight = args.getDouble('action_align', def: 0.0);

  final lockTerrain = args.getFlag('lock_terrain', def: false);
  final lockSpawn = args.getFlag('lock_spawn', def: false);
  final randomSpawnX = !args.getFlag('fixed_spawn_x', def: false);
  final maxFuel = args.getDouble('max_fuel', def: 1000.0);
  final crashOnTilt = args.getFlag('crash_on_tilt', def: false);

  // physics flags for CLI
  final gravity = args.getDouble('gravity', def: 0.18);
  final thrustAccel = args.getDouble('thrust_accel', def: 0.42);
  final rotSpeed = args.getDouble('rot_speed', def: 1.6);

  final rcsEnabled = args.getFlag('rcs_enabled', def: false);
  final rcsAccel = args.getDouble('rcs_accel', def: 0.12);
  final rcsBodyFrame = !args.getFlag('rcs_world_frame', def: false); // default body frame

  final downThrEnabled = args.getFlag('down_thr_enabled', def: false);
  final downThrAccel = args.getDouble('down_thr_accel', def: 0.30);
  final downThrBurn = args.getDouble('down_thr_burn', def: 10.0);

  // PF velocity-only reward CLI knobs
  final pfAlign = args.getDouble('pf_align', def: 1.0);
  final pfVelDelta = args.getDouble('pf_vel_delta', def: 0.6);
  final pfVminClose = args.getDouble('pf_vmin_close', def: 8.0);
  final pfVmaxFar = args.getDouble('pf_vmax_far', def: 90.0);
  final pfAlpha = args.getDouble('pf_alpha', def: 1.2);
  final pfVmax = args.getDouble('pf_vmax', def: 140.0);
  final pfXBias = args.getDouble('pf_x_bias', def: 3.0);

  // NEW: attempts per terrain + eval cadence/size
  final attemptsPerTerrain = args.getInt('attempts_per_terrain', def: 1).clamp(1, 1000000);
  final evalEvery = args.getInt('eval_every', def: 10).clamp(1, 1000000);
  final evalEpisodes = args.getInt('eval_episodes', def: 80).clamp(1, 1000000);

  final determinism = args.getFlag('determinism_probe', def: true);
  final hidden = _parseHiddenList(args.getStr('hidden'), fallback: const [64, 64]);

  // Trainer-internal gating (now gating uses mean PF reward; higher is better)
  final gateScoreMin = args.getDouble('gate_min', def: -1e9);
  final gateOnlyLanded = args.getFlag('gate_landed', def: false);
  final gateVerbose = args.getFlag('gate_verbose', def: true);

  double bestMeanCost = double.infinity;

  // ----- Build env -----
  final cfg = makeConfig(
    seed: seed,
    lockTerrain: lockTerrain,
    lockSpawn: lockSpawn,
    randomSpawnX: randomSpawnX,
    maxFuel: maxFuel,
    crashOnTilt: crashOnTilt,

    gravity: gravity,
    thrustAccel: thrustAccel,
    rotSpeed: rotSpeed,

    rcsEnabled: rcsEnabled,
    rcsAccel: rcsAccel,
    rcsBodyFrame: rcsBodyFrame,

    downThrEnabled: downThrEnabled,
    downThrAccel: downThrAccel,
    downThrBurn: downThrBurn,
  );
  final env = eng.GameEngine(cfg);

  // Ensure rays active (forward-aligned)
  env.rayCfg = const RayConfig(
    rayCount: 180,
    includeFloor: false,
    forwardAligned: true,
  );

  // FE probe
  final fe = FeatureExtractorRays(rayCount: env.rayCfg.rayCount);
  env.reset(seed: seed ^ 0xC0FFEE);
  env.step(1 / 60.0, const et.ControlInput(thrust: false, left: false, right: false));
  final inDim = fe.extract(env).length;
  final kindsOneHot = (inDim == 6 + env.rayCfg.rayCount * 4);

  // ----- Policy -----
  final policy = PolicyNetwork(inputSize: inDim, hidden: hidden, seed: seed);
  print('Loaded init policy. hidden=${policy.hidden} | FE(kind=rays, in=$inDim, rays=${env.rayCfg.rayCount}, oneHot=$kindsOneHot)');

  // ===== PF velocity-only reward hook (rebuilt per episode) =====
  PFShapingCfg pfCfg = PFShapingCfg(
    wAlign: pfAlign,
    wVelDelta: pfVelDelta,
    vMinClose: pfVminClose,
    vMaxFar: pfVmaxFar,
    alpha: pfAlpha,
    vmax: pfVmax,
    xBias: pfXBias,
  );
  ai.ExternalRewardHook? pfHook = makePFRewardHook(env: env, cfg: pfCfg);

  // ----- Trainer -----
  final trainer = Trainer(
    env: env,
    fe: fe,
    policy: policy,
    dt: 1 / 60.0,
    gamma: 0.99,
    seed: seed,
    twoStage: true,
    planHold: planHold,
    tempIntent: tempIntent,
    intentEntropyBeta: intentEntropy,
    useLearnedController: useLearned,
    blendPolicy: blendPolicy.clamp(0.0, 1.0),
    intentAlignWeight: intentAlignWeight,
    intentPgWeight: intentPgWeight,
    actionAlignWeight: actionAlignWeight,
    normalizeFeatures: true,
    // gating inside trainer (prints [TRAIN] lines)
    gateScoreMin: gateScoreMin,
    gateOnlyLanded: gateOnlyLanded,
    gateVerbose: gateVerbose,

    // add dense PF reward per step (velocity-only)
    externalRewardHook: (({required eng.GameEngine env, required double dt, required int tStep}) {
      return pfHook != null ? pfHook!(env: env, dt: dt, tStep: tStep) : 0.0;
    }),
  );

  // Determinism probe (physics)
  if (determinism) {
    env.reset(seed: 1234);
    final a = _probeDeterminism(env, maxSteps: 165);
    env.reset(seed: 1234);
    final b = _probeDeterminism(env, maxSteps: 165);
    final ok = (a.steps == b.steps) && ((a.cost - b.cost).abs() < 1e-6);
    print('Determinism probe: steps ${a.steps} vs ${b.steps} | cost ${a.cost.toStringAsFixed(6)} vs ${b.cost.toStringAsFixed(6)} => ${ok ? "OK" : "MISMATCH"}');
  }

  // Optional: warm the feature norm a bit so early updates aren’t wild
  _warmFeatureNorm(
    norm: trainer.norm,
    trainer: trainer,
    fe: fe,
    env: env,
    perClass: 500,
    seed: seed ^ 0xACE,
  );

  // Quick baseline eval (use half of evalEpisodes, clamped to [10, evalEpisodes])
      {
    final baseEvalN = math.max(10, math.min(evalEpisodes, 40));
    final ev = evaluate(
      env: env,
      trainer: trainer,
      episodes: baseEvalN,
      seed: seed ^ 0x999,
      attemptsPerTerrain: attemptsPerTerrain,
    );
    print(
        'Eval(real) → meanCost=${ev.meanCost.toStringAsFixed(3)} | median=${ev.medianCost.toStringAsFixed(3)} | land%=${ev.landPct.toStringAsFixed(1)} | crash%=${ev.crashPct.toStringAsFixed(1)} | steps=${ev.meanSteps.toStringAsFixed(1)} | mean|dx|=${ev.meanAbsDx.toStringAsFixed(1)}'
    );
  }

  // ===== MAIN TRAIN LOOP =====
  final rnd = math.Random(seed ^ 0xDEADBEEF);
  final rayCount = env.rayCfg.rayCount;

  // Group multiple attempts per terrain
  int terrAttempts = 0;
  int currentTerrainSeed = rnd.nextInt(1 << 30);

  for (int it = 0; it < iters; it++) {
    double lastCost = 0.0;
    int lastSteps = 0;
    bool lastLanded = false;

    for (int b = 0; b < batch; b++) {
      // Pick a new terrain seed when starting a new group
      if (terrAttempts == 0) {
        currentTerrainSeed = rnd.nextInt(1 << 30);
      }

      // Reuse terrain within the group
      env.reset(seed: currentTerrainSeed);

      // Rebuild PF-based reward hook for this terrain/episode
      pfHook = makePFRewardHook(env: env, cfg: pfCfg);

      final res = trainer.runEpisode(
        train: true, // Trainer handles gating internally & prints [TRAIN] lines
        greedy: false,
        scoreIsReward: false,
        lr: lr,
        valueBeta: valueBeta,
        huberDelta: huberDelta,
      );

      // Advance attempt counter within the group
      terrAttempts = (terrAttempts + 1) % attemptsPerTerrain;

      // For iteration summary
      lastCost = res.totalCost;
      lastSteps = res.steps;
      lastLanded = res.landed;
    }

    print('Iter ${it + 1} | attempts/terrain=$attemptsPerTerrain | last-ep steps: $lastSteps | cost: ${lastCost.toStringAsFixed(3)} | landed: ${lastLanded ? "Y" : "N"}');

    // periodic eval + save (now driven by CLI)
    if ((it + 1) % evalEvery == 0) {
      final ev = evaluate(
        env: env,
        trainer: trainer,
        episodes: evalEpisodes,
        seed: seed ^ (0x1111 * (it + 1)),
        attemptsPerTerrain: attemptsPerTerrain,
      );
      print(
          'Eval(real) → meanCost=${ev.meanCost.toStringAsFixed(3)} | median=${ev.medianCost.toStringAsFixed(3)} | land%=${ev.landPct.toStringAsFixed(1)} | crash%=${ev.crashPct.toStringAsFixed(1)} | steps=${ev.meanSteps.toStringAsFixed(1)} | mean|dx|=${ev.meanAbsDx.toStringAsFixed(1)}'
      );

      if (ev.meanCost < bestMeanCost) {
        bestMeanCost = ev.meanCost;
        _savePolicy(
          path: 'policy_best_cost.json',
          p: policy,
          rayCount: rayCount,
          kindsOneHot: kindsOneHot,
          env: env,
          norm: trainer.norm,
        );
        print('★ New BEST by cost at iter ${it + 1}: meanCost=${ev.meanCost.toStringAsFixed(3)} → saved policy_best_cost.json');
      }

      _savePolicy(
        path: 'policy_iter_${it + 1}.json',
        p: policy,
        rayCount: rayCount,
        kindsOneHot: kindsOneHot,
        env: env,
        norm: trainer.norm,
      );
    }
  }

  _savePolicy(
    path: 'policy_final.json',
    p: policy,
    rayCount: rayCount,
    kindsOneHot: kindsOneHot,
    env: env,
    norm: trainer.norm,
  );
  print('Training done. Saved → policy_final.json');
}

/* ----------------------------------- usage ------------------------------------

Velocity-vector–only shaping examples:

1) Defaults:

  dart run lib/ai/train_agent.dart \
    --hidden=96,96,64 \
    --train_iters=400 --batch=1 --lr=0.0003 --plan_hold=1 \
    --blend_policy=1.0 --intent_align=0.25 --intent_pg=0.6 \
    --gate_min=0.0 --gate_landed

2) Stronger alignment, weaker velocity error:

  dart run lib/ai/train_agent.dart \
    --pf_align=1.5 --pf_vel_delta=0.4

3) Faster far, gentler near the pad:

  dart run lib/ai/train_agent.dart \
    --pf_vmin_close=6 --pf_vmax_far=110 --pf_alpha=1.4

Knobs added here:
- reuse terrain within groups:      --attempts_per_terrain=20
- eval cadence and size:            --eval_every=20 --eval_episodes=120

Notes:
- The trainer uses only this PF reward via `externalRewardHook`.
- `segMean` in logs reports mean PF reward (used for gating).
------------------------------------------------------------------------------ */
