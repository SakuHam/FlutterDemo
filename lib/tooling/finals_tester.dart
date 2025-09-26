// lib/tools/finals_tester.dart
//
// Final landing tester/diagnostic for the DAgger stage.
//
// What it does
// ------------
// - Spawns near/far final-approach states like final_dagger
// - Executes MIXED control: learned turn (opt-in), thrust blended in prob space
//   with the same late-latch/PWM rounding you use in training
// - Uses a tiny teacher hold (duration) blended with the model duration head
// - Records *why* episodes fail or succeed: off-pad, angle, |vx|, |vy|, or
//   not-landed status (e.g., hillside contact)
// - Splits stats by altitude bands and near-pad proximity
// - Reports model-vs-teacher action agreement (turn/thrust)
//
// How to run (example)
// --------------------
// import 'package:yourgame/engine/game_engine.dart' as eng;
// import 'package:yourgame/ai/agent.dart';
// import 'package:yourgame/tools/finals_tester.dart';
//
// final env = eng.GameEngine(cfg: ...);
// final fe  = FeatureExtractorRays(rayCount: 16, kindsOneHot: true);
// final policy = PolicyNetwork(inputSize: fe.inputSize, hidden: [64,64], seed: 123);
// final norm = RunningNorm(fe.inputSize);
//
// await FinalsTester(
//   env: env,
//   fe: fe,
//   policy: policy,
//   norm: norm,
//   seed: 42,
//   // mirror your final_dagger cfg here
// ).run(episodes: 2000);
//
// Look at the printed breakdown to see which predicate dominates failures.
// --------------------------------------------------------------------------

import 'dart:io' as io;
import 'dart:math' as math;

import '../engine/types.dart' as et;
import '../engine/game_engine.dart' as eng;
import '../ai/agent.dart';

class FinalsTesterCfg {
  // Episode layout
  final int maxSteps;

  // Spawn ranges (relative to pad)
  final double hMin;
  final double hMax;
  final double xNearFrac;
  final double xFarFrac;
  final double farProb;
  final double vyMin;
  final double vyMax;
  final double vxNearMax;
  final double vxFarMax;

  // Acceptance thresholds
  final double acceptDxFrac;
  final double acceptVyMax;
  final double acceptVxMax;
  final double acceptAngMaxDeg;

  // Mixed mode (mirror final_dagger)
  final bool useLearnedTurn;
  final double thrustBlend;   // prob-space blend M vs teacher
  final double durMin;
  final double durMax;
  final double durBlend;

  // Late-stage execution safety (optional stabilizers)
  final bool lowAltTiltFreeze;   // freeze left/right below hFreeze unless |vx|>vxLo
  final double hFreeze;
  final double vxFreezeLo;

  // Teacher thrust smoothing (reduce noisy labels)
  final bool smoothThrustLabels;

  const FinalsTesterCfg({
    this.maxSteps = 900,
    this.hMin = 70.0,
    this.hMax = 140.0,
    this.xNearFrac = 0.12,
    this.xFarFrac = 0.40,
    this.farProb = 0.35,
    this.vyMin = 6.0,
    this.vyMax = 16.0,
    this.vxNearMax = 4.0,
    this.vxFarMax = 12.0,

    this.acceptDxFrac = 0.06,
    this.acceptVyMax = 4.0,
    this.acceptVxMax = 26.0,
    this.acceptAngMaxDeg = 14.0,

    this.useLearnedTurn = true,
    this.thrustBlend = 0.6,
    this.durMin = 1.0,
    this.durMax = 18.0,
    this.durBlend = 0.6,

    this.lowAltTiltFreeze = true,
    this.hFreeze = 120.0,
    this.vxFreezeLo = 10.0,

    this.smoothThrustLabels = true,
  });
}

enum FailKind { none, notLanded, offPad, angle, vx, vy }

class _Counters {
  int episodes = 0;
  int landed = 0;

  // failure causes
  int failNotLanded = 0;
  int failOffPad = 0;
  int failAngle = 0;
  int failVx = 0;
  int failVy = 0;

  // teacher/model agreement
  int turnAgree = 0;
  int turnTotal = 0;
  int thrAgree = 0;
  int thrTotal = 0;

  // near-pad, low-alt emphasis
  int lowAltNearPadSamples = 0;
  int lowAltNearPadTurnAgree = 0;
  int lowAltNearPadThrAgree = 0;

  // simple hist buckets
  final List<double> touchdownDx = [];
  final List<double> touchdownVx = [];
  final List<double> touchdownVy = [];
  final List<double> touchdownAngDeg = [];
}

class FinalsTester {
  final eng.GameEngine env;
  final FeatureExtractorRays fe;
  final PolicyNetwork policy;
  final RunningNorm? norm;
  final FinalsTesterCfg cfg;
  final int seed;

  FinalsTester({
    required this.env,
    required this.fe,
    required this.policy,
    required this.norm,
    FinalsTesterCfg? cfg,
    this.seed = 12345,
  }) : cfg = cfg ?? const FinalsTesterCfg();

  // --- Utilities copied from your final stage acceptance ---
  bool _landWin(eng.GameEngine env) {
    if (env.status != et.GameStatus.landed) return false;
    final L = env.lander;
    final padCx = env.terrain.padCenter.toDouble();
    final W = env.cfg.worldW.toDouble();

    final dxAbs  = (L.pos.x.toDouble() - padCx).abs();
    final vyAbs  = L.vel.y.abs().toDouble();
    final vxAbs  = L.vel.x.abs().toDouble();
    final angDeg = (L.angle.abs().toDouble() * 180.0 / math.pi);

    final centered = dxAbs <= cfg.acceptDxFrac * W;
    final softVert = vyAbs <= cfg.acceptVyMax;
    final softLat  = vxAbs <= cfg.acceptVxMax;
    final upright  = angDeg <= cfg.acceptAngMaxDeg;
    return centered && softVert && softLat && upright;
  }

  // Spawn like final_dagger
  void _initStart(math.Random r) {
    final far = (r.nextDouble() < cfg.farProb);
    final padCx = env.terrain.padCenter.toDouble();
    final W = env.cfg.worldW.toDouble();
    final frac = far ? cfg.xFarFrac : cfg.xNearFrac;

    final xOff = (r.nextDouble() * 2 - 1) * (frac * W);
    final x = (padCx + xOff).clamp(10.0, W - 10.0);

    final h = cfg.hMin + (cfg.hMax - cfg.hMin) * r.nextDouble();
    final vy = cfg.vyMin + (cfg.vyMax - cfg.vyMin) * r.nextDouble();
    final vxMax = far ? cfg.vxFarMax : cfg.vxNearMax;
    final vx = (r.nextDouble() * 2 - 1) * vxMax;
    final gy = env.terrain.heightAt(x);

    env.lander
      ..pos.x = x
      ..pos.y = (gy - h).clamp(0.0, env.cfg.worldH - 10.0)
      ..vel.x = vx
      ..vel.y = vy
      ..angle = 0.0
      ..fuel = env.cfg.t.maxFuel;
  }

  // stronger teacher label smoothing for thrust to reduce label noise
  bool _smoothThrustTeach(bool thrustTeach) {
    if (!cfg.smoothThrustLabels) return thrustTeach;

    final gy = env.terrain.heightAt(env.lander.pos.x.toDouble());
    final h  = (gy - env.lander.pos.y).toDouble().clamp(0.0, 1e9);
    final vy = env.lander.vel.y.toDouble();
    final vCapDesc = _vCapDesc(h);
    final vyPred   = _vyPredictNoThrust(env, tauReact: 0.8);

    bool t = thrustTeach;

    if (!t) {
      final soonTooFast = (vy > 0.92 * vCapDesc) || (vyPred > 0.98 * vCapDesc);
      if (soonTooFast) t = true;
    }
    if (t && h > 200.0) {
      final safelyUnder = vy < 0.70 * vCapDesc && vyPred < 0.80 * vCapDesc;
      if (safelyUnder) t = false;
    }
    return t;
  }

  // failure diagnosis for touchdown frame
  FailKind _diagnoseFailure() {
    if (env.status != et.GameStatus.landed) return FailKind.notLanded;

    final L = env.lander;
    final padCx = env.terrain.padCenter.toDouble();
    final W = env.cfg.worldW.toDouble();

    final dxAbs  = (L.pos.x.toDouble() - padCx).abs();
    final vyAbs  = L.vel.y.abs().toDouble();
    final vxAbs  = L.vel.x.abs().toDouble();
    final angDeg = (L.angle.abs().toDouble() * 180.0 / math.pi);

    if (dxAbs > cfg.acceptDxFrac * W) return FailKind.offPad;
    if (angDeg > cfg.acceptAngMaxDeg) return FailKind.angle;
    if (vxAbs > cfg.acceptVxMax)      return FailKind.vx;
    if (vyAbs > cfg.acceptVyMax)      return FailKind.vy;
    return FailKind.none;
  }

  void _recordTouchdown(_Counters C) {
    final L = env.lander;
    final padCx = env.terrain.padCenter.toDouble();
    final W = env.cfg.worldW.toDouble();

    C.touchdownDx.add((L.pos.x.toDouble() - padCx).abs() / (cfg.acceptDxFrac * W)); // normalized
    C.touchdownVx.add(L.vel.x.abs().toDouble());
    C.touchdownVy.add(L.vel.y.abs().toDouble());
    C.touchdownAngDeg.add((L.angle.abs().toDouble() * 180.0 / math.pi));
  }

  // pretty-print quick percent
  String _pct(int num, int den) =>
      den == 0 ? '0.0%' : '${(100.0 * num / den).toStringAsFixed(1)}%';

  // simple summary of hist-like arrays
  String _minMedMax(List<double> xs) {
    if (xs.isEmpty) return 'n/a';
    final ys = List<double>.from(xs)..sort();
    double p(double q) {
      final i = (q * (ys.length - 1)).clamp(0, ys.length - 1).toInt();
      return ys[i];
    }
    final mn = ys.first, md = p(0.5), mx = ys.last;
    return 'min=${mn.toStringAsFixed(2)} med=${md.toStringAsFixed(2)} max=${mx.toStringAsFixed(2)}';
  }

  Future<void> run({required int episodes}) async {
    final r = math.Random(seed ^ 0xF1A1);
    final C = _Counters();

    // Latch/PWM state (mirror your trainer)
    double pwmA = 0.0;
    int pwmCount = 0;
    int pwmOn = 0;
    int thrustLatch = 0;
    const thrustLatchBoost = 8;

    for (int ep = 0; ep < episodes; ep++) {
      env.reset(seed: r.nextInt(1 << 30));
      _initStart(r);

      int framesLeft = 0;
      int steps = 0;
      bool done = false;

      while (!done && steps < cfg.maxSteps) {
        if (framesLeft <= 0) {
          // duration head mix (model + tiny teacher)
          List<double>? x = fe.extract(
            lander: env.lander, terrain: env.terrain,
            worldW: env.cfg.worldW.toDouble(), worldH: env.cfg.worldH.toDouble(),
            rays: env.rays,
          );
          if (norm != null) {
            norm?.observe(x);
            x = norm?.normalize(x, update: false);
          }
          final c = policy.forwardFull(x!);
          final predHold = c.durFrames.clamp(cfg.durMin, cfg.durMax);

          // tiny teacher hold: if centered & slow laterally, allow 2 frames; else 1
          final padCx = env.terrain.padCenter.toDouble();
          final W = env.cfg.worldW.toDouble();
          final dxAbs = (env.lander.pos.x.toDouble() - padCx).abs();
          final vxAbs = env.lander.vel.x.toDouble().abs();
          int teachHold = 1;
          if (dxAbs < 0.05 * W && vxAbs < 18.0) teachHold = 2;

          final useHold = cfg.durBlend * predHold + (1.0 - cfg.durBlend) * teachHold;
          framesLeft = useHold.round().clamp(1, 12);
        }

        // teacher for labels
        final uTeach0 = controllerForIntent(
          indexToIntent(predictiveIntentLabelAdaptive(env)),
          env,
        );

        // optional thrust label smoothing
        bool thrustTeach = cfg.smoothThrustLabels
            ? _smoothThrustTeach(uTeach0.thrust)
            : uTeach0.thrust;

        // model action
        var xAct = fe.extract(
          lander: env.lander, terrain: env.terrain,
          worldW: env.cfg.worldW.toDouble(), worldH: env.cfg.worldH.toDouble(),
          rays: env.rays,
        );
        if (norm != null) xAct = (norm?.normalize(xAct, update: false))!;
        final (thM, lM, rM, probs, _) = policy.actGreedy(xAct);

        // agreement tallies
        final turnTeachIdx = uTeach0.left ? 0 : (uTeach0.right ? 2 : 1);
        final turnPredIdx  = lM ? 0 : (rM ? 2 : 1);
        if (turnTeachIdx == turnPredIdx) C.turnAgree++;
        C.turnTotal++;

        if ((thM && thrustTeach) || (!thM && !thrustTeach)) C.thrAgree++;
        C.thrTotal++;

        // near-pad/low-alt agreement
        final padCx = env.terrain.padCenter.toDouble();
        final W = env.cfg.worldW.toDouble();
        final dxAbs = (env.lander.pos.x.toDouble() - padCx).abs();
        final gy = env.terrain.heightAt(env.lander.pos.x.toDouble());
        final h = (gy - env.lander.pos.y).toDouble();
        final isNearPadLowAlt = (dxAbs < 0.12 * W) && (h < 180.0);
        if (isNearPadLowAlt) {
          C.lowAltNearPadSamples++;
          if (turnTeachIdx == turnPredIdx) C.lowAltNearPadTurnAgree++;
          if ((thM && thrustTeach) || (!thM && !thrustTeach)) C.lowAltNearPadThrAgree++;
        }

        // mixed execution (mirror your trainer)
        final height  = h;
        final pM = probs[0].clamp(0.0, 1.0);
        final pT = thrustTeach ? 1.0 : 0.0;
        double pExec = cfg.thrustBlend * pM + (1.0 - cfg.thrustBlend) * pT;

        // late emergency up
        final vCapUp = _vCapBrakeUp(height);
        final needEmergencyUp = (height < 220.0) && (env.lander.vel.y > 0.9 * vCapUp);

        if (pT >= 0.5 || needEmergencyUp) {
          thrustLatch = math.max(thrustLatch, thrustLatchBoost);
        }

        // bias toward more certain flare when low&centered
        if (height < 180.0 && pT >= 0.5) {
          pExec = math.max(pExec, 0.75);
        }

        pwmA += pExec;
        bool thrustPWM = false;

        if (thrustLatch > 0) {
          thrustPWM = true;
          thrustLatch--;
        } else if (pwmA >= 1.0) {
          thrustPWM = true;
          pwmA -= 1.0;
        }

        if (!thrustPWM && height < 90.0 && pExec > 0.55) {
          thrustPWM = true;
          pwmA = math.max(0.0, pwmA - 0.5);
        }

        bool leftExec  = cfg.useLearnedTurn ? lM : uTeach0.left;
        bool rightExec = cfg.useLearnedTurn ? rM : uTeach0.right;

        // optional last-meters tilt freeze
        if (cfg.lowAltTiltFreeze && height < cfg.hFreeze) {
          final vxAbs = env.lander.vel.x.abs().toDouble();
          if (vxAbs <= cfg.vxFreezeLo) { leftExec = false; rightExec = false; }
        }

        final info = env.step(1/60.0, et.ControlInput(
          thrust: thrustPWM,
          left: leftExec,
          right: rightExec,
          sideLeft: uTeach0.sideLeft,
          sideRight: uTeach0.sideRight,
          downThrust: uTeach0.downThrust,
        ));
        steps++;
        framesLeft--;

        if (info.terminal || steps >= cfg.maxSteps) {
          done = true;
          break;
        }

        // keep PWM stats bounded (not used in report, but mirrors trainer safety)
        pwmCount++; if (thrustPWM) pwmOn++;
        if ((pwmCount % 240) == 0) {
          pwmCount = 0; pwmOn = 0;
        }
      } // while

      C.episodes++;
      final good = _landWin(env);
      if (good) C.landed++;

      _recordTouchdown(C);

      final fail = _diagnoseFailure();
      switch (fail) {
        case FailKind.none: break;
        case FailKind.notLanded: C.failNotLanded++; break;
        case FailKind.offPad:    C.failOffPad++;    break;
        case FailKind.angle:     C.failAngle++;     break;
        case FailKind.vx:        C.failVx++;        break;
        case FailKind.vy:        C.failVy++;        break;
      }
    } // episodes

    // ---------- Report ----------
    final landPct = _pct(C.landed, C.episodes);
    print('\n=== Finals Tester Report ===');
    print('Episodes: ${C.episodes}');
    print('Land win% (strict): $landPct');
    print('Failure breakdown (shares of all episodes):');
    print('  notLanded: ${_pct(C.failNotLanded, C.episodes)}'
        '  offPad: ${_pct(C.failOffPad, C.episodes)}'
        '  angle: ${_pct(C.failAngle, C.episodes)}'
        '  |vx|: ${_pct(C.failVx, C.episodes)}'
        '  |vy|: ${_pct(C.failVy, C.episodes)}');

    print('\nTeacher↔Model agreement (all steps):');
    print('  turn acc:  ${_pct(C.turnAgree, C.turnTotal)}   '
        '(${C.turnAgree}/${C.turnTotal})');
    print('  thrust acc:${_pct(C.thrAgree, C.thrTotal)}   '
        '(${C.thrAgree}/${C.thrTotal})');

    print('\nNear-pad & low-alt (dx < 0.12W & h < 180):');
    print('  samples: ${C.lowAltNearPadSamples}');
    print('  turn acc:  ${_pct(C.lowAltNearPadTurnAgree, C.lowAltNearPadSamples)}');
    print('  thrust acc:${_pct(C.lowAltNearPadThrAgree,  C.lowAltNearPadSamples)}');

    print('\nTouchdown stats (final frame):');
    print('  dx / (dxLimit): ${_minMedMax(C.touchdownDx)}  [>1.0 means off-pad by win criteria]');
    print('  |vx| (px/s):    ${_minMedMax(C.touchdownVx)}');
    print('  |vy| (px/s):    ${_minMedMax(C.touchdownVy)}');
    print('  |angle| (deg):  ${_minMedMax(C.touchdownAngDeg)}');

    print('\nTester cfg (key bits): '
        'accept dxFrac=${cfg.acceptDxFrac} vy<=${cfg.acceptVyMax} '
        'vx<=${cfg.acceptVxMax} ang<=${cfg.acceptAngMaxDeg}°, '
        'useLearnedTurn=${cfg.useLearnedTurn} thrustBlend=${cfg.thrustBlend}, '
        'durBlend=${cfg.durBlend}, lowAltTiltFreeze=${cfg.lowAltTiltFreeze}');
    print('==============================\n');
  }
}

// ============================ CLI entrypoint ============================

/*
eng.GameEngine makeEngine() {
  // TODO: adapt this to your engine’s constructor.
  // If your engine needs a config, build/provide it here.
  // Example placeholders:
  // final cfg = eng.EngineConfig.defaults();
  // return eng.GameEngine(cfg: cfg);
  //
  // If your GameEngine has a no-arg constructor, this may work:
  return eng.GameEngine();
}

 */

Map<String, String> _parseArgs(List<String> argv) {
  final map = <String, String>{};
  for (final a in argv) {
    if (a.startsWith('--')) {
      final eq = a.indexOf('=');
      if (eq > 0) {
        map[a.substring(2, eq)] = a.substring(eq + 1);
      } else {
        map[a.substring(2)] = 'true';
      }
    }
  }
  return map;
}

double _getD(Map<String,String> m, String k, double d) {
  final v = m[k];
  if (v == null) return d;
  return double.tryParse(v) ?? d;
}
int _getI(Map<String,String> m, String k, int d) {
  final v = m[k];
  if (v == null) return d;
  return int.tryParse(v) ?? d;
}
bool _getB(Map<String,String> m, String k, bool d) {
  final v = m[k];
  if (v == null) return d;
  final s = v.toLowerCase();
  if (s == '1' || s == 'true' || s == 'yes' || s == 'on') return true;
  if (s == '0' || s == 'false' || s == 'no' || s == 'off') return false;
  return d;
}

void _printHelp() {
  io.stdout.writeln('''
Finals tester (DAgger landing) CLI

Usage:
  dart run lib/tools/finals_tester.dart [--key=value]...

Common flags:
  --episodes=2000
  --seed=42
  --maxSteps=900

Spawn:
  --hMin=70  --hMax=140
  --xNearFrac=0.12  --xFarFrac=0.40  --farProb=0.35
  --vyMin=6  --vyMax=16
  --vxNearMax=4  --vxFarMax=12

Accept (win criteria):
  --acceptDxFrac=0.06
  --acceptVyMax=4
  --acceptVxMax=26
  --acceptAngMaxDeg=14

Mixed execution:
  --useLearnedTurn           (flag, default true)
  --thrustBlend=0.6          (0..1; 1=model only)
  --durMin=1  --durMax=18  --durBlend=0.6

Safety:
  --lowAltTiltFreeze         (flag, default true)
  --hFreeze=120  --vxFreezeLo=10

Label smoothing:
  --smoothThrustLabels       (flag, default true)

Examples:
  dart run lib/tools/finals_tester.dart --episodes=3000 --thrustBlend=0.55
  dart run lib/tools/finals_tester.dart --episodes=2000 --useLearnedTurn --lowAltTiltFreeze
''');
}

Future<void> main(List<String> argv) async {
  final a = _parseArgs(argv);
  if (_getB(a, 'help', false) || _getB(a, 'h', false)) {
    _printHelp();
    return;
  }

  final episodes = _getI(a, 'episodes', 2000);
  final seed     = _getI(a, 'seed', 42);

  // Build engine
  final cfg0 = et.EngineConfig(
    worldW: 640,
    worldH: 320,
    t: et.Tunables(),
  );
  final env = eng.GameEngine(cfg0);

  // Build FE + policy (match inputs to your runtime FE)
  final fe = FeatureExtractorRays(rayCount: 16, kindsOneHot: true);
  final policy = PolicyNetwork(inputSize: fe.inputSize, hidden: const [64, 64], seed: seed ^ 0xA51CE);
  final norm = RunningNorm(fe.inputSize);

  // OPTIONAL: if you have saved weights, load them here.
  // For example (pseudo):
  // await policy.loadFromJsonFile('weights.json');
  // Or copy action heads from a teacher snapshot:
  // policy.copyActionHeadsFrom(teacherPolicy);

  // Gather CLI overrides into tester cfg
  final cfg = FinalsTesterCfg(
    maxSteps:        _getI(a, 'maxSteps',        900),
    hMin:            _getD(a, 'hMin',            70.0),
    hMax:            _getD(a, 'hMax',            140.0),
    xNearFrac:       _getD(a, 'xNearFrac',       0.12),
    xFarFrac:        _getD(a, 'xFarFrac',        0.40),
    farProb:         _getD(a, 'farProb',         0.35),
    vyMin:           _getD(a, 'vyMin',           6.0),
    vyMax:           _getD(a, 'vyMax',           16.0),
    vxNearMax:       _getD(a, 'vxNearMax',       4.0),
    vxFarMax:        _getD(a, 'vxFarMax',        12.0),

    acceptDxFrac:    _getD(a, 'acceptDxFrac',    0.06),
    acceptVyMax:     _getD(a, 'acceptVyMax',     4.0),
    acceptVxMax:     _getD(a, 'acceptVxMax',     26.0),
    acceptAngMaxDeg: _getD(a, 'acceptAngMaxDeg', 14.0),

    useLearnedTurn:  _getB(a, 'useLearnedTurn',  true),
    thrustBlend:     _getD(a, 'thrustBlend',     0.6),
    durMin:          _getD(a, 'durMin',          1.0),
    durMax:          _getD(a, 'durMax',          18.0),
    durBlend:        _getD(a, 'durBlend',        0.6),

    lowAltTiltFreeze:_getB(a, 'lowAltTiltFreeze',true),
    hFreeze:         _getD(a, 'hFreeze',         120.0),
    vxFreezeLo:      _getD(a, 'vxFreezeLo',      10.0),

    smoothThrustLabels: _getB(a, 'smoothThrustLabels', true),
  );

  io.stdout.writeln('[FINALS/TEST] episodes=$episodes seed=$seed '
      'useLearnedTurn=${cfg.useLearnedTurn} thrustBlend=${cfg.thrustBlend} '
      'durBlend=${cfg.durBlend} lowAltTiltFreeze=${cfg.lowAltTiltFreeze}');

  final tester = FinalsTester(
    env: env,
    fe: fe,
    policy: policy,
    norm: norm,
    cfg: cfg,
    seed: seed,
  );

  await tester.run(episodes: episodes);
}

// ----------------------- small helpers from agent -----------------------
double _vCapDesc(double h)   => (0.10 * h + 8.0).clamp(8.0, 26.0);
double _vCapBrakeUp(double h)=> (0.07 * h + 6.0).clamp(6.0, 16.0);

double _vyPredictNoThrust(eng.GameEngine env, {double tauReact = 0.35}) {
  final vy = env.lander.vel.y.toDouble();
  final g  = env.cfg.t.gravity;
  return vy + g * tauReact;
}
