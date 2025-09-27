// lib/ai/live_trainer_isolate.dart
import 'dart:isolate';
import 'dart:math' as math;

// Engine
import '../engine/types.dart' as et;
import '../engine/game_engine.dart' as eng;
import '../engine/raycast.dart';

// Policy I/O (bundle load/save helpers & types)
import 'policy_io.dart' show
PolicyBundle,
savePolicyBundle,
loadBundleIntoNetwork,
restoreNormFromBundle;

// Policy + trainer (FeatureExtractorRays, PolicyNetwork, Trainer, RunningNorm)
import 'agent.dart' as ai;

// Curriculum
import 'curriculum/pad_align_progressive.dart' as padprog show PadAlignProgressiveCurriculum;
import 'curriculum/core.dart';

// Eval
import 'eval.dart' as eval;

/* ---------------------- messages from UI isolate ---------------------- */

class LiveTrainStart {
  final String outPath;
  final String? warmStart;    // e.g. '/path/to/policy_curriculum.json'
  final int iters;            // iterations per cycle (total)
  final int seed;
  final List<int>? hidden;    // optional NN hidden sizes
  LiveTrainStart({
    required this.outPath,
    this.warmStart,
    required this.iters,
    required this.seed,
    this.hidden,
  });
}

class LiveTrainStop {
  const LiveTrainStop();
}

/* ------------------------------ env cfg ------------------------------- */

et.EngineConfig _makeConfig({
  int seed = 7,
  double worldW = 800,
  double worldH = 600,
}) {
  final t = et.Tunables(
    gravity: 0.18,
    thrustAccel: 0.42,
    rotSpeed: 1.6,
    maxFuel: 1000.0,
    crashOnTilt: false,
    landingMaxVx: 28.0,
    landingMaxVy: 38.0,
    landingMaxOmega: 3.5,

    // keep simple at live-train: no RCS/DownThr
    rcsEnabled: false,
    rcsAccel: 0.12,
    rcsBodyFrame: true,
    downThrEnabled: false,
    downThrAccel: 0.30,
    downThrBurn: 10.0,
  );
  return et.EngineConfig(
    worldW: worldW,
    worldH: worldH,
    t: t,
    seed: seed,
    stepScale: 60.0,
    lockTerrain: false,
    terrainSeed: 1234567,
    lockSpawn: false,
    randomSpawnX: true,
    hardWalls: true,
  );
}

/* ----------------------------- utilities ------------------------------ */

int _iclamp(int v, int lo, int hi) => v < lo ? lo : (v > hi ? hi : v);
void _send(SendPort host, Map<String, Object?> msg) => host.send(msg);

/* ----------------------------- live worker ---------------------------- */

void liveTrainerMain(SendPort hostPort) async {
  final recv = ReceivePort();
  hostPort.send(recv.sendPort);

  bool running = false;

  // Local state
  late eng.GameEngine env;
  ai.FeatureExtractorRays? fe;
  ai.PolicyNetwork? policy;
  ai.Trainer? trainer;

  // Best checkpoint tracking
  double bestMeanCost = double.infinity;
  double bestLandPct = 0.0;

  await for (final msg in recv) {
    if (msg is LiveTrainStop || (msg is Map && msg['cmd'] == 'stop')) {
      running = false;
      _send(hostPort, {'type': 'status', 'phase': 'stopped', 'iters': 0});
      continue;
    }

    if (msg is LiveTrainStart || (msg is Map && msg['cmd'] == 'start')) {
      // Parse Start
      final conf = (msg is LiveTrainStart)
          ? msg
          : LiveTrainStart(
        outPath: (msg['outPath'] as String?) ?? 'policy_live.json',
        warmStart: msg['warmStart'] as String?,
        iters: (msg['iters'] as int?) ?? 3000,
        seed: (msg['seed'] as int?) ?? 7,
        hidden: (msg['hidden'] as List?)?.cast<int>(),
      );

      if (running) continue;
      running = true;

      // Immediately tell UI we’re starting (so HUD leaves Idle)
      _send(hostPort, {'type': 'status', 'phase': 'starting', 'iters': 0});

      try {
        // Build env
        env = eng.GameEngine(_makeConfig(seed: conf.seed));
        env.rayCfg = const RayConfig(rayCount: 180, includeFloor: false, forwardAligned: true);

        // FE + inDim
        fe = ai.FeatureExtractorRays(rayCount: env.rayCfg.rayCount);
        env.reset(seed: conf.seed ^ 0xC0FFEE);
        env.step(1 / 60.0, const et.ControlInput(thrust: false, left: false, right: false));
        final inDim = fe!.extract(
          lander: env.lander,
          terrain: env.terrain,
          worldW: env.cfg.worldW,
          worldH: env.cfg.worldH,
          rays: env.rays,
        ).length;

        final hidden = conf.hidden ?? const [64, 64];
        policy = ai.PolicyNetwork(inputSize: inDim, hidden: hidden, seed: conf.seed);
        policy!.setTrunkTrainable(true);
        policy!.setHeadsTrainable(intent: true, action: true, value: false);

        trainer = ai.Trainer(
          env: env,
          fe: fe!,
          policy: policy!,
          dt: 1 / 60.0,
          gamma: 0.99,
          seed: conf.seed,
          twoStage: true,
          planHold: 2,

          // === exploration knobs: slightly hotter intent + a touch of entropy
          tempIntent: 1.25,
          intentEntropyBeta: 0.02,

          useLearnedController: false,
          blendPolicy: 1.0,
          intentAlignWeight: 0.25,
          intentPgWeight: 0.60,
          actionAlignWeight: 0.0,
          normalizeFeatures: true,

          // gating: relaxed to keep samples flowing
          gateScoreMin: -1e9,
          gateOnlyLanded: false,
          gateVerbose: false,

          gateProbEnabled: true,
          gateProbK: 8.0,
          gateProbMin: 0.05,
          gateProbMax: 0.95,
          gateProbLandedBoost: 0.15,
          gateProbNearPadBoost: 0.10,
          gateProbDeadzoneZ: -1e9,
          gateProbFloor: 0.02,

          externalRewardHook: null, // curriculum supplies its own reward/targets
        );

        // Optional warm start
        PolicyBundle? loaded;
        if ((conf.warmStart ?? '').trim().isNotEmpty) {
          try {
            loaded = PolicyBundle.loadFromPath(conf.warmStart!.trim());
            loadBundleIntoNetwork(bundle: loaded, target: policy!, env: env);
            if (trainer!.norm != null) {
              final ok = restoreNormFromBundle(bundle: loaded, runtimeNorm: trainer!.norm!);
              if (!ok) {
                _send(hostPort, {
                  'type': 'warn',
                  'message': 'Warm start norm missing or mismatched; will warm locally.'
                });
              }
            }
            _send(hostPort, {'type': 'status', 'phase': 'warmstart_loaded', 'iters': 0});
          } catch (e) {
            _send(hostPort, {'type': 'warn', 'message': 'Warm start failed: $e'});
          }
        }

        // ===== Norm warm-up (critical for early progress) =====
        if (trainer!.norm != null && !trainer!.norm!.inited) {
          _send(hostPort, {'type': 'status', 'phase': 'norm_warm', 'iters': 0});
          for (int i = 0; i < 8; i++) {
            env.reset(seed: conf.seed ^ (0xA11CE + i));
            trainer!.runEpisode(
              train: false,
              greedy: true,
              scoreIsReward: false,
              lr: 3e-4,
              valueBeta: 0.5,
              huberDelta: 1.0,
            );
          }
          _send(hostPort, {'type': 'status', 'phase': 'norm_ready', 'iters': 0});
        }

        // === Live training ===
        final cur = padprog.PadAlignProgressiveCurriculum();

        // Wire curriculum heartbeat → UI (optional, but helps HUD tick)
        // Will emit: {'type':'progress','phase':'curriculum', ...}
        // (The UI treats non-'evaluating' progress as training.)
        // NOTE: This property exists in our fork; harmless if no-op in yours.
        // ignore: invalid_use_of_visible_for_testing_member
        // ignore: invalid_use_of_protected_member
        // If your class doesn't expose a sink, comment this line.
        // (We keep it guarded to avoid breaking older code.)
        try {
          // If your class has a 'progressSink' field as suggested:
          // (m) => _send(hostPort, m);
          // We use dynamic to avoid hard dependency.
          // ignore: avoid_dynamic_calls
          // ignore: cast_nullable_to_non_nullable
          (cur as dynamic).progressSink = (Map<String, Object?> m) => _send(hostPort, m);
        } catch (_) {
          // no-op if curriculum has no sink
        }

        final rng = math.Random(conf.seed);

        // Split the requested iters into stable chunks — observe progress every chunk
        final totalIters = _iclamp(conf.iters, 200, 200000);
        final chunk = math.max(300, totalIters ~/ 10); // ~10 pulses per cycle
        int done = 0;

        // tell UI we’re running (it=0)
        _send(hostPort, {
          'type': 'status',
          'phase': 'running',
          'iters': 0,
          'total': totalIters,
        });

        // track “stuck” condition
        int stuckCount = 0;
        double lastPulseLand = 0.0;

        bestMeanCost = double.infinity;
        bestLandPct = 0.0;

        while (running && done < totalIters) {
          final itThis = math.min(chunk, totalIters - done);

          // per-chunk running heartbeat
          _send(hostPort, {
            'type': 'status',
            'phase': 'running',
            'iters': done,
            'total': totalIters,
          });

          // Configure & run curriculum chunk
          cur.configure(const {}, const {});
          await cur.run(
            iters: itThis,
            env: env,
            fe: fe!,
            policy: policy!,
            norm: trainer!.norm,
            planHold: 2,
            tempIntent: 1.25,   // keep slightly exploratory
            gamma: 0.99,
            lr: 3e-4,
            intentAlignWeight: 0.25,
            intentPgWeight: 0.60,
            actionAlignWeight: 0.0,
            gateVerbose: false,
            seed: conf.seed ^ rng.nextInt(1 << 29),
          );

          done += itThis;

          // Eval pulse
          final ev = eval.evaluateSequential(
            env: env,
            trainer: trainer!,
            episodes: 300,                  // smooth noise
            seed: conf.seed ^ done ^ 0xBEEF,
            attemptsPerTerrain: 1,
            evalDebug: false,
            evalDebugFailN: 0,
          );

          _send(hostPort, {
            'type': 'progress',
            'phase': 'evaluating',
            'it': done,
            'total': totalIters,
            'landPct': ev.landPct,
            'meanCost': ev.meanCost,
            'meanSteps': ev.meanSteps,
          });

          // Save only on improvement (lower cost or higher land%)
          bool improved = false;
          if (ev.meanCost + 1e-6 < bestMeanCost) {
            bestMeanCost = ev.meanCost;
            improved = true;
          }
          if (ev.landPct > bestLandPct + 0.25) { // require small delta to avoid churn
            bestLandPct = ev.landPct;
            improved = true;
          }
          if (improved) {
            try {
              savePolicyBundle(
                path: conf.outPath,
                p: policy!,
                env: env,
                norm: trainer!.norm,
              );
              _send(hostPort, {
                'type': 'saved',
                'path': conf.outPath,
                'landPct': ev.landPct,
                'meanCost': ev.meanCost,
              });
            } catch (e) {
              _send(hostPort, {'type': 'error', 'message': 'Save failed: $e'});
            }
          }

          // “Stuck” heuristic: if land% hasn’t improved across pulses
          if (ev.landPct <= lastPulseLand + 0.1) {
            stuckCount++;
          } else {
            stuckCount = 0;
          }
          lastPulseLand = ev.landPct;

          if (stuckCount >= 3) {
            // Nudge exploration a bit & re-warm norm lightly
            _send(hostPort, {'type': 'hint', 'message': 'Progress stalled — nudging exploration.'});
            trainer!.tempIntent = (trainer!.tempIntent * 1.05).clamp(1.0, 1.6);
            trainer!.intentEntropyBeta = (trainer!.intentEntropyBeta * 1.25 + 0.005).clamp(0.0, 0.05);

            if (trainer!.norm != null) {
              for (int i = 0; i < 3; i++) {
                env.reset(seed: conf.seed ^ (0xCAFE + i + done));
                trainer!.runEpisode(
                  train: false,
                  greedy: true,
                  scoreIsReward: false,
                  lr: 3e-4,
                  valueBeta: 0.5,
                  huberDelta: 1.0,
                );
              }
            }
            stuckCount = 0; // reset and try again
          }
        }

        // Final status
        _send(hostPort, {'type': 'status', 'phase': 'stopped', 'iters': done, 'total': totalIters});
      } catch (e) {
        _send(hostPort, {'type': 'error', 'message': 'Live trainer crashed: $e'});
        running = false;
        _send(hostPort, {'type': 'status', 'phase': 'stopped', 'iters': 0});
      }
    }
  }
}
