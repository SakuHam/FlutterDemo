// lib/ui/game_page.dart
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart' show Ticker;
import 'package:path_provider/path_provider.dart';
import 'package:flutter/services.dart' show rootBundle;

// 🚦 intent bus (singleton)
import '../ai/intent_bus.dart'; // <-- adjust path if needed

/// ======= Demo DTOs (same as in pretrain) =======

class DemoStep {
  final List<double> x; // feature vector
  final int thr;        // 0/1
  final int turn;       // 0:none,1:left,2:right
  DemoStep(this.x, this.thr, this.turn);

  Map<String, dynamic> toJson() => {'x': x, 'thr': thr, 'turn': turn};
}

class DemoEpisode {
  final List<DemoStep> steps;
  final bool landed;
  DemoEpisode(this.steps, {this.landed = false});

  Map<String, dynamic> toJson() =>
      {'landed': landed, 'steps': steps.map((s) => s.toJson()).toList()};
}

class DemoSet {
  final int inputSize;
  final List<DemoEpisode> episodes;
  DemoSet({required this.inputSize, required this.episodes});

  Map<String, dynamic> toJson() =>
      {'inputSize': inputSize, 'episodes': episodes.map((e) => e.toJson()).toList()};

  String toPrettyJson() => const JsonEncoder.withIndent('  ').convert(toJson());
}

/// ======= Game types =======

class GamePage extends StatefulWidget {
  const GamePage({super.key});
  @override
  State<GamePage> createState() => _GamePageState();
}

enum GameStatus { playing, landed, crashed }

class Tunables {
  double gravity; // base gravity strength
  double thrustAccel; // engine acceleration
  double rotSpeed; // radians per second
  double maxFuel; // fuel units
  Tunables({
    this.gravity = 0.18,
    this.thrustAccel = 0.42,
    this.rotSpeed = 1.6,
    this.maxFuel = 100.0,
  });

  Tunables copyWith({
    double? gravity,
    double? thrustAccel,
    double? rotSpeed,
    double? maxFuel,
  }) =>
      Tunables(
        gravity: gravity ?? this.gravity,
        thrustAccel: thrustAccel ?? this.thrustAccel,
        rotSpeed: rotSpeed ?? this.rotSpeed,
        maxFuel: maxFuel ?? this.maxFuel,
      );
}

class Particle {
  Offset pos;
  Offset vel;
  double life; // 0..1
  Particle({required this.pos, required this.vel, required this.life});
}

class _GamePageState extends State<GamePage> with SingleTickerProviderStateMixin {
  late final Ticker _ticker;

  static const double kLandingSpeedMax = 5.0;                 // was 40.0
  static const double kLandingAngleMaxRad = 8 * math.pi / 180; // was ~14.3° (0.25 rad)

  // Tunables
  Tunables t = Tunables();

  // World & entities
  Size? _worldSize; // set in LayoutBuilder
  Terrain? _terrain; // generated once per size
  Lander lander = const Lander(
    position: Offset(160, 120),
    velocity: Offset.zero,
    angle: 0,
    fuel: 100,
  );
  final List<Particle> _particles = [];

  // Input (manual)
  bool thrust = false;
  bool left = false;
  bool right = false;

  // AI
  bool aiPlay = false;
  SimplePolicy? _policy; // loaded from policy.json
  String _policyInfo = 'No policy loaded';

  // Demo recording
  bool recordDemos = false;
  final List<DemoEpisode> _demoEpisodes = [];
  final List<DemoStep> _currentSteps = [];
  int _feGroundSamples = 3; // will be overwritten by policy.fe if present
  double _feStridePx = 48;

  // Game state
  GameStatus status = GameStatus.playing;

  // Randomness for UI terrain/spawn
  final math.Random _uiRnd = math.Random();
  int _terrainSeed = DateTime.now().microsecondsSinceEpoch;
  bool _useFixedSeed = false; // set true to freeze terrain for debugging

  // Timing
  late DateTime _last;

  // ------------ Intent Bus (toast) ------------
  StreamSubscription<IntentEvent>? _intentSub;
  String? _toastText;
  bool _toastVisible = false;
  Timer? _toastTimer;

  // UI-synthesized intent cadence
  int _uiPlanHold = 12;     // frames to hold a UI intent
  int _uiFramesLeft = 0;
  String? _uiCurrentIntent; // last computed UI intent

  @override
  void initState() {
    super.initState();
    _ticker = createTicker(_onTick)..start();
    _last = DateTime.now();

    // Subscribe to the global singleton bus
    debugPrint('[BUS/UI] initState');
    _intentSub = IntentBus.instance.intents.listen((e) {
      // show toast on ANY event (even if AI is off) so UI is never silent
      _showIntentToast(e.intent);
      debugPrint('[BUS/UI] intent=${e.intent} probs=${e.probs}');
    });

    WidgetsBinding.instance.addPostFrameCallback((_) => _reset());
  }

  @override
  void dispose() {
    _ticker.dispose();
    _intentSub?.cancel();
    _toastTimer?.cancel();
    super.dispose();
  }

  void _showIntentToast(String text) {
    setState(() {
      _toastText = text;
      _toastVisible = true;
    });
    _toastTimer?.cancel();
    _toastTimer = Timer(const Duration(milliseconds: 900), () {
      if (!mounted) return;
      setState(() => _toastVisible = false);
    });
  }

  void setPreset(String name) {
    setState(() {
      if (name == 'Easy') {
        t = t.copyWith(gravity: 0.12, thrustAccel: 0.65, rotSpeed: 2.2, maxFuel: 140);
      } else if (name == 'Classic') {
        t = t.copyWith(gravity: 0.18, thrustAccel: 0.42, rotSpeed: 1.6, maxFuel: 100);
      } else if (name == 'Hard') {
        t = t.copyWith(gravity: 0.22, thrustAccel: 0.38, rotSpeed: 1.3, maxFuel: 80);
      }
      _reset();
    });
  }

  void _reset() {
    final size = _worldSize ?? const Size(360, 640);

    // Close any ongoing recording episode
    if (recordDemos && _currentSteps.isNotEmpty) {
      _demoEpisodes.add(DemoEpisode(List.of(_currentSteps), landed: status == GameStatus.landed));
      _currentSteps.clear();
    }

    // Randomize terrain (unless fixed)
    if (!_useFixedSeed) {
      _terrainSeed = DateTime.now().microsecondsSinceEpoch ^ _uiRnd.nextInt(1 << 30);
      _terrain = Terrain.generate(size, seed: _terrainSeed);
    } else {
      _terrain ??= Terrain.generate(size, seed: 12345);
    }

    setState(() {
      status = GameStatus.playing;
      _particles.clear();

      // Randomize spawn X in [0.2, 0.8] of screen width
      final fracX = 0.20 + _uiRnd.nextDouble() * 0.60;
      lander = Lander(
        position: Offset(size.width * fracX, 120),
        velocity: Offset.zero,
        angle: 0,
        fuel: t.maxFuel,
      );

      // Clear manual controls
      thrust = left = right = false;
      // Leave aiPlay as-is (user toggle)

      // reset UI intent cadence
      _uiFramesLeft = 0;
      _uiCurrentIntent = null;
    });

    debugPrint('[GamePage.reset] W=${size.width.toStringAsFixed(1)} '
        'H=${size.height.toStringAsFixed(1)} seed=$_terrainSeed '
        'spawnX=${lander.position.dx.toStringAsFixed(1)}');
  }

  Future<void> _saveDemos() async {
    if (recordDemos && _currentSteps.isNotEmpty) {
      _demoEpisodes.add(DemoEpisode(List.of(_currentSteps), landed: false));
      _currentSteps.clear();
    }
    final inputSize = 11 + _feGroundSamples; // MUST match trainer/runtime
    final ds = DemoSet(inputSize: inputSize, episodes: _demoEpisodes);

    try {
      final dir = await getApplicationDocumentsDirectory();
      final file = File('${dir.path}/demos.json');
      await file.writeAsString(ds.toPrettyJson());
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Saved ${_demoEpisodes.length} demos → ${file.path}')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Save failed: $e')));
    }
  }

  Future<void> _loadPolicy() async {
    try {
      String jsonText;

      // 1) Try Documents/policy.json (so you can drop in new weights from the trainer)
      try {
        final dir = await getApplicationDocumentsDirectory();
        final file = File('${dir.path}/policy.json');
        if (await file.exists()) {
          jsonText = await file.readAsString();
        } else {
          throw const FileSystemException('policy.json not in Documents');
        }
      } catch (_) {
        // 2) Fallback: bundled asset
        jsonText = await rootBundle.loadString('assets/ai/policy.json');
      }

      final js = json.decode(jsonText) as Map<String, dynamic>;
      final pol = SimplePolicy.fromJson(js);

      setState(() {
        _policy = pol;
        _policyInfo =
        'Loaded policy: h1=${pol.h1}, h2=${pol.h2}, fe(gs=${pol.feGroundSamples}, stride=${pol.feStridePx})';
        _feGroundSamples = pol.feGroundSamples;
        _feStridePx = pol.feStridePx;
      });

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(_policyInfo)));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Load failed: $e')));
    }
  }

  // ---------- UI-synthesized intent (so toast shows even when idle) ----------
  String _chooseUiIntent({
    required double dxToPad,   // normalized ~[-1,+1]
    required double vy,        // px/s
    required double height,    // px above ground
    required double angle,     // rad
  }) {
    // Soft, readable rules; tweak to your taste
    if (vy < -25) return 'brakeUp';            // moving up fast -> stop it
    if (vy > 90)  return 'descendSlow';        // falling too fast -> slow down

    const dxDead = 0.05; // 5% screen width
    if (dxToPad > dxDead)  return 'goLeft';
    if (dxToPad < -dxDead) return 'goRight';

    return 'hover'; // default when idle-ish
  }

  void _onTick(Duration _) {
    final now = DateTime.now();
    double dt = now.difference(_last).inMicroseconds / 1e6;
    dt = dt.clamp(0, 1 / 20); // max 50 ms step
    _last = now;

    if (!mounted || status != GameStatus.playing || _terrain == null) return;

    setState(() {
      // ---------- Always compute & publish a UI intent every few frames ----------
      final wW = _worldSize?.width ?? 360.0;
      final dxN = (lander.position.dx - _terrain!.padCenter) / (wW == 0 ? 1.0 : wW);
      final groundY = _terrain!.heightAt(lander.position.dx);
      final height = (groundY - lander.position.dy);
      if (_uiFramesLeft <= 0) {
        final intent = _chooseUiIntent(
          dxToPad: dxN,
          vy: lander.velocity.dy,
          height: height,
          angle: lander.angle,
        );
        if (intent != _uiCurrentIntent) {
          _uiCurrentIntent = intent;
          // publish so toast shows even when AI is off
          IntentBus.instance.publishIntent(IntentEvent(
            intent: intent,
            meta: {'source': 'ui'},
          ));
        } else {
          // still publish periodically so there's always activity
          IntentBus.instance.publishIntent(IntentEvent(
            intent: intent,
            meta: {'source': 'ui/keepalive'},
          ));
        }
        _uiFramesLeft = _uiPlanHold;
      } else {
        _uiFramesLeft -= 1;
      }

      // --- If AI is active, compute controls here (override manual) ---
      if (aiPlay && _policy != null && _worldSize != null) {
        final feats = _extractFeatures(
          position: lander.position,
          velocity: lander.velocity,
          angle: lander.angle,
          fuel: lander.fuel,
          terrain: _terrain!,
          worldW: _worldSize!.width,
          worldH: _worldSize!.height,
          groundSamples: _feGroundSamples,
          stridePx: _feStridePx,
        );
        final act = _policy!.actGreedy(
          feats,
          angle: lander.angle,
          dxToPad: dxN,
          thrThresh: 0.35, // works well for greedy
        );
        thrust = act.thrust;
        left = act.left;
        right = act.right;
      }

      // --- Controls & rotation (apply scale 0.5) ---
      double angle = lander.angle;
      final rot = t.rotSpeed * 0.5;
      if (left && !right) angle -= rot * dt;
      if (right && !left) angle += rot * dt;

      // --- Acceleration (apply 0.05 scalers to gravity and thrust) ---
      Offset accel = Offset(0, t.gravity * 0.05);
      double fuel = lander.fuel;
      final thrustingNow = thrust && fuel > 0;
      if (thrustingNow) {
        accel += Offset(math.sin(angle), -math.cos(angle)) * (t.thrustAccel * 0.05);
        fuel = (fuel - 20 * dt).clamp(0, t.maxFuel);
      }

      // --- Integrate (semi-implicit Euler) ---
      Offset vel = lander.velocity + accel * dt * 60; // tuned time scale
      Offset pos = lander.position + vel * dt * 60;

      // --- Hard ceiling (prevent boosting out) ---
      if (pos.dy < 0) {
        pos = Offset(pos.dx, 0);
        if (vel.dy < 0) vel = Offset(vel.dx, 0);
      }

      // --- Hard walls (no wrap) ---
      final w = _worldSize?.width ?? 360.0;
      if (pos.dx < 0) {
        pos = Offset(0, pos.dy);
        vel = Offset(0, vel.dy);
      }
      if (pos.dx > w) {
        pos = Offset(w, pos.dy);
        vel = Offset(0, vel.dy);
      }

      // --- Particle exhaust ---
      if (thrustingNow) {
        final c = math.cos(angle);
        final s = math.sin(angle);
        final axis = Offset(-s, c); // down in ship frame
        final flameBase = pos + axis * Lander.halfHeight;
        final rnd = math.Random();
        for (int i = 0; i < 6; i++) {
          final perp = Offset(-c, -s);
          final spread = (rnd.nextDouble() - 0.5) * 0.35;
          final dir = (axis + perp * spread);
          final speed = 60 + rnd.nextDouble() * 60;
          _particles.add(Particle(pos: flameBase, vel: dir * speed, life: 1.0));
        }
      }

      // --- Update particles ---
      for (int i = _particles.length - 1; i >= 0; i--) {
        final p = _particles[i];
        p.life -= dt * 1.8;
        p.vel += Offset(0, t.gravity * 0.05 * 0.2); // tiny gravity on smoke
        p.pos += p.vel * dt;
        if (p.life <= 0) _particles.removeAt(i);
      }

      // --- Collision with terrain ---
      final feet = lander.footPoints(pos, angle);
      final groundYLeft = _terrain!.heightAt(feet.left.dx);
      final groundYRight = _terrain!.heightAt(feet.right.dx);
      final groundYCenter = _terrain!.heightAt(pos.dx);

      final collided = feet.left.dy >= groundYLeft ||
          feet.right.dy >= groundYRight ||
          pos.dy >= groundYCenter - 2;

      if (collided) {
        final onPad = _terrain!.isOnPad(pos.dx);
        final speed = vel.distance;
        final gentle = speed <= kLandingSpeedMax && angle.abs() <= kLandingAngleMaxRad;
        if (onPad && gentle) {
          status = GameStatus.landed;
          pos = Offset(pos.dx, _terrain!.padY - Lander.halfHeight);
          vel = Offset.zero;
        } else {
          status = GameStatus.crashed;
        }

        // Close demo episode on terminal
        if (recordDemos && _currentSteps.isNotEmpty) {
          _demoEpisodes.add(DemoEpisode(List.of(_currentSteps), landed: status == GameStatus.landed));
          _currentSteps.clear();
        }
      }

      // --- RECORDING: store a demo sample for human play only ---
      if (recordDemos && !aiPlay && _terrain != null && _worldSize != null) {
        final features = _extractFeatures(
          position: pos,
          velocity: vel,
          angle: angle,
          fuel: fuel,
          terrain: _terrain!,
          worldW: _worldSize!.width,
          worldH: _worldSize!.height,
          groundSamples: _feGroundSamples,
          stridePx: _feStridePx,
        );
        final thrI = thrust ? 1 : 0;
        final turnI = left ? 1 : (right ? 2 : 0);
        _currentSteps.add(DemoStep(features, thrI, turnI));
      }

      // --- Commit state ---
      lander = lander.copyWith(position: pos, velocity: vel, angle: angle, fuel: fuel);
    });
  }

  /// Matches training/runtime feature layout:
  /// [px, py, vx, vy, sinA, cosA, fuel, padCenter, dxCenter, dGround, slope, samples...]
  List<double> _extractFeatures({
    required Offset position,
    required Offset velocity,
    required double angle,
    required double fuel,
    required Terrain terrain,
    required double worldW,
    required double worldH,
    required int groundSamples,
    required double stridePx,
  }) {
    final px = position.dx / worldW;
    final py = position.dy / worldH;
    final vx = (velocity.dx / 200.0).clamp(-2.0, 2.0);
    final vy = (velocity.dy / 200.0).clamp(-2.0, 2.0);
    final sinA = math.sin(angle);
    final cosA = math.cos(angle);
    final fuelN = (fuel / t.maxFuel).clamp(0.0, 1.0);

    final padCenter = ((_terrain!.padX1 + _terrain!.padX2) * 0.5) / worldW;
    final dxCenter =
    ((position.dx - (_terrain!.padX1 + _terrain!.padX2) * 0.5) / worldW).clamp(-1.0, 1.0);

    final gY = terrain.heightAt(position.dx);
    final dGround = ((gY - position.dy) / worldH).clamp(-1.0, 1.0);
    final gyL = terrain.heightAt((position.dx - 20).clamp(0.0, worldW));
    final gyR = terrain.heightAt((position.dx + 20).clamp(0.0, worldW));
    final slope = (((gyR - gyL) / 40.0) / 0.5).clamp(-2.0, 2.0);

    final n = groundSamples;
    final half = (n - 1) ~/ 2;
    final samples = <double>[];
    for (int i = -half; i <= half; i++) {
      final sx = (position.dx + i * stridePx).clamp(0.0, worldW);
      final sy = terrain.heightAt(sx);
      final rel = ((sy - position.dy) / worldH).clamp(-1.0, 1.0);
      samples.add(rel);
    }

    return [
      px, py, vx, vy, sinA, cosA, fuelN,
      padCenter, dxCenter, dGround, slope,
      ...samples,
    ];
  }

  Widget _menuButton() {
    return ElevatedButton.icon(
      style: ElevatedButton.styleFrom(
        backgroundColor: Colors.black.withOpacity(0.55),
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
      onPressed: () async {
        final leave = await showDialog<bool>(
          context: context,
          builder: (ctx) => AlertDialog(
            title: const Text('Return to Menu?'),
            content: const Text('Current run will be lost.'),
            actions: [
              TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
              FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Leave')),
            ],
          ),
        );
        if (leave == true) Navigator.pop(context);
      },
      icon: const Icon(Icons.home),
      label: const Text('Menu'),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: LayoutBuilder(
        builder: (context, constraints) {
          _worldSize = Size(constraints.maxWidth, constraints.maxHeight);
          _terrain ??= Terrain.generate(
            _worldSize!,
            seed: _useFixedSeed ? 12345 : _terrainSeed,
          );

          return Stack(
            children: [
              // Game canvas
              Positioned.fill(
                child: CustomPaint(
                  painter: GamePainter(
                    lander: lander,
                    terrain: _terrain!,
                    thrusting: thrust && lander.fuel > 0 && status == GameStatus.playing,
                    status: status,
                    particles: _particles,
                  ),
                ),
              ),

              // HUD
              SafeArea(
                child: Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Padding(
                        padding: const EdgeInsets.all(12.0),
                        child: _menuButton(),
                      ),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          _hudBox(title: 'Fuel', value: lander.fuel.toStringAsFixed(0)),
                          _hudBox(
                            title: 'Vx/Vy',
                            value:
                            '${lander.velocity.dx.toStringAsFixed(1)} / ${lander.velocity.dy.toStringAsFixed(1)}',
                          ),
                          _hudBox(
                            title: 'Angle',
                            value: '${(lander.angle * 180 / math.pi).toStringAsFixed(0)}°',
                          ),
                          PopupMenuButton<String>(
                            icon: const Icon(Icons.tune),
                            onSelected: setPreset,
                            itemBuilder: (context) => const [
                              PopupMenuItem(value: 'Easy', child: Text('Easy')),
                              PopupMenuItem(value: 'Classic', child: Text('Classic')),
                              PopupMenuItem(value: 'Hard', child: Text('Hard')),
                            ],
                          ),
                        ],
                      ),

                      const SizedBox(height: 8),

                      // AI & Recording controls
                      Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        crossAxisAlignment: WrapCrossAlignment.center,
                        children: [
                          ElevatedButton.icon(
                            onPressed: _loadPolicy,
                            icon: const Icon(Icons.download),
                            label: const Text('Load policy'),
                          ),
                          FilterChip(
                            label: Text('AI Play${_policy == null ? ' (load first)' : ''}'),
                            selected: aiPlay,
                            onSelected: (v) {
                              if (_policy == null) {
                                ScaffoldMessenger.of(context).showSnackBar(
                                  const SnackBar(content: Text('Load policy.json first')),
                                );
                                return;
                              }
                              setState(() {
                                aiPlay = v;
                                if (aiPlay) {
                                  // When AI starts, stop recording (we only want human demos)
                                  recordDemos = false;
                                  // clear manual inputs to avoid sticky states
                                  thrust = left = right = false;
                                }
                              });
                            },
                          ),
                          const SizedBox(width: 8),
                          FilterChip(
                            label: const Text('Record'),
                            selected: recordDemos,
                            onSelected: (v) {
                              if (aiPlay && v) {
                                ScaffoldMessenger.of(context).showSnackBar(
                                  const SnackBar(content: Text('Disable AI Play to record demos')),
                                );
                                return;
                              }
                              setState(() => recordDemos = v);
                            },
                          ),
                          ElevatedButton.icon(
                            onPressed: _saveDemos,
                            icon: const Icon(Icons.save),
                            label: const Text('Save demos'),
                          ),
                          const SizedBox(width: 12),
                          FilterChip(
                            label: const Text('Fixed terrain'),
                            selected: _useFixedSeed,
                            onSelected: (v) {
                              setState(() {
                                _useFixedSeed = v;
                                _terrain = null; // regenerate next build
                              });
                              _reset();
                            },
                          ),
                          ElevatedButton.icon(
                            onPressed: _reset,
                            icon: const Icon(Icons.refresh),
                            label: const Text('Reset'),
                          ),
                        ],
                      ),

                      Padding(
                        padding: const EdgeInsets.only(top: 6),
                        child: Text(
                          _policyInfo,
                          style: const TextStyle(fontSize: 12, color: Colors.white70),
                        ),
                      ),

                      if (status != GameStatus.playing)
                        Padding(
                          padding: const EdgeInsets.only(top: 8.0),
                          child: Center(
                            child: ElevatedButton.icon(
                              onPressed: _reset,
                              icon: const Icon(Icons.replay),
                              label: const Text('Play again'),
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              ),

              // Controls (manual only – still visible when AI plays, but toggling has no effect)
              _buildControls(),

              // Status banner
              if (status == GameStatus.landed || status == GameStatus.crashed)
                Center(
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                    decoration: BoxDecoration(
                      color: Colors.black.withOpacity(0.6),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.white24),
                    ),
                    child: Text(
                      status == GameStatus.landed ? 'Touchdown! 🟢' : 'Crashed 💥',
                      style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                    ),
                  ),
                ),

              // -------- Bottom toast for intent --------
              Positioned(
                left: 0,
                right: 0,
                bottom: 18,
                child: IgnorePointer(
                  child: AnimatedOpacity(
                    duration: const Duration(milliseconds: 200),
                    opacity: _toastVisible ? 1.0 : 0.0,
                    child: Center(
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.65),
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(color: Colors.white24),
                        ),
                        child: Text(
                          _toastText ?? '',
                          style: const TextStyle(
                            fontWeight: FontWeight.w600,
                            letterSpacing: 0.4,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _buildControls() {
    return IgnorePointer(
      ignoring: status != GameStatus.playing,
      child: Align(
        alignment: Alignment.bottomCenter,
        child: Padding(
          padding: const EdgeInsets.only(bottom: 24.0),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _holdButton(icon: Icons.rotate_left, onChanged: (v) => setState(() => left = v)),
              _holdButton(
                  icon: Icons.local_fire_department,
                  onChanged: (v) => setState(() => thrust = v),
                  big: true),
              _holdButton(icon: Icons.rotate_right, onChanged: (v) => setState(() => right = v)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _holdButton({
    required IconData icon,
    required ValueChanged<bool> onChanged,
    bool big = false,
  }) {
    final size = big ? 96.0 : 72.0;
    return Listener(
      onPointerDown: (_) => onChanged(true),
      onPointerUp: (_) => onChanged(false),
      onPointerCancel: (_) => onChanged(false),
      child: Container(
        width: size,
        height: size,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: big ? Colors.orange.withOpacity(0.15) : Colors.white10,
          border: Border.all(color: Colors.white30),
        ),
        child: Icon(icon, size: big ? 42 : 32),
      ),
    );
  }

  Widget _hudBox({required String title, required String value}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.45),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.white24),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: const TextStyle(fontSize: 12, color: Colors.white70)),
          Text(value, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }
}

class Lander {
  final Offset position; // center of mass
  final Offset velocity;
  final double angle; // radians, 0 = up
  final double fuel;

  static const double halfWidth = 14; // px
  static const double halfHeight = 18; // px

  const Lander({
    required this.position,
    required this.velocity,
    required this.angle,
    required this.fuel,
  });

  Lander copyWith({Offset? position, Offset? velocity, double? angle, double? fuel}) => Lander(
    position: position ?? this.position,
    velocity: velocity ?? this.velocity,
    angle: angle ?? this.angle,
    fuel: fuel ?? this.fuel,
  );

  ({Offset left, Offset right}) footPoints(Offset pos, double ang) {
    final c = math.cos(ang);
    final s = math.sin(ang);
    final bottomCenter = pos + const Offset(0, halfHeight);
    const leftLocal = Offset(-halfWidth, 0);
    const rightLocal = Offset(halfWidth, 0);
    Offset rot(Offset v) => Offset(c * v.dx - s * v.dy, s * v.dx + c * v.dy);
    return (left: bottomCenter + rot(leftLocal), right: bottomCenter + rot(rightLocal));
  }
}

class Terrain {
  final List<Offset> points; // polyline ridge
  final double padX1;
  final double padX2;
  final double padY;

  Terrain({required this.points, required this.padX1, required this.padX2, required this.padY});

  static Terrain generate(Size size, {int? seed}) {
    final rnd = math.Random(seed ?? DateTime.now().microsecondsSinceEpoch);
    final width = size.width;
    final height = size.height;

    // Ridge
    final List<Offset> pts = [];
    const int segments = 24;
    for (int i = 0; i <= segments; i++) {
      final x = width * i / segments;
      final base = height * 0.78;
      final noise = (math.sin(i * 0.8) + rnd.nextDouble() * 0.5) * 24.0;
      pts.add(Offset(x, base + noise));
    }

    // Landing pad
    final padWidth = width * 0.16;
    final padCenterX = width * (0.35 + rnd.nextDouble() * 0.3);
    final padX1 = (padCenterX - padWidth / 2).clamp(10.0, width - padWidth - 10.0);
    final padX2 = padX1 + padWidth;
    final padY = height * 0.76;

    for (int i = 0; i < pts.length; i++) {
      if (pts[i].dx >= padX1 && pts[i].dx <= padX2) {
        pts[i] = Offset(pts[i].dx, padY);
      }
    }

    // Ends lower for valley look
    pts[0] = Offset(0, height * 0.92);
    pts[pts.length - 1] = Offset(width, height * 0.92);

    return Terrain(points: pts, padX1: padX1, padX2: padX2, padY: padY);
  }

  double get padCenter => (padX1 + padX2) * 0.5;

  double heightAt(double x) {
    final pts = points;
    for (int i = 0; i < pts.length - 1; i++) {
      final a = pts[i];
      final b = pts[i + 1];
      if ((x >= a.dx && x <= b.dx) || (x >= b.dx && x <= a.dx)) {
        final t = (x - a.dx) / (b.dx - a.dx);
        return a.dy + (b.dy - a.dy) * t;
      }
    }
    return points.last.dy;
  }

  bool isOnPad(double x) => x >= padX1 && x <= padX2;
}

class GamePainter extends CustomPainter {
  final Lander lander;
  final Terrain terrain;
  final bool thrusting;
  final GameStatus status;
  final List<Particle> particles;

  GamePainter({
    required this.lander,
    required this.terrain,
    required this.thrusting,
    required this.status,
    required this.particles,
  });

  @override
  void paint(Canvas canvas, Size size) {
    _paintStars(canvas, size);
    _paintTerrain(canvas, size);
    _paintPad(canvas);
    _paintParticles(canvas);
    _paintLander(canvas);
  }

  void _paintStars(Canvas canvas, Size size) {
    final paint = Paint()..color = Colors.white.withOpacity(0.8);
    final rnd = math.Random(1);
    for (int i = 0; i < 120; i++) {
      final dx = rnd.nextDouble() * size.width;
      final dy = rnd.nextDouble() * size.height * 0.6;
      canvas.drawCircle(Offset(dx, dy), rnd.nextDouble() * 1.2, paint);
    }
  }

  void _paintTerrain(Canvas canvas, Size size) {
    final path = Path()..moveTo(terrain.points.first.dx, terrain.points.first.dy);
    for (int i = 1; i < terrain.points.length; i++) {
      path.lineTo(terrain.points[i].dx, terrain.points[i].dy);
    }
    path.lineTo(size.width, size.height);
    path.lineTo(0, size.height);
    path.close();

    final ground = Paint()
      ..shader = const LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [Color(0xFF1C2330), Color(0xFF0E141B)],
      ).createShader(Offset.zero & size);
    canvas.drawPath(path, ground);

    final outline = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..color = Colors.white10;
    final ridge = Path()..moveTo(terrain.points.first.dx, terrain.points.first.dy);
    for (int i = 1; i < terrain.points.length; i++) {
      ridge.lineTo(terrain.points[i].dx, terrain.points[i].dy);
    }
    canvas.drawPath(ridge, outline);
  }

  void _paintPad(Canvas canvas) {
    final padRect = Rect.fromLTWH(
      terrain.padX1,
      terrain.padY - 4,
      (terrain.padX2 - terrain.padX1),
      8,
    );
    final paint = Paint()..color = status == GameStatus.crashed ? Colors.red : Colors.greenAccent;
    canvas.drawRRect(RRect.fromRectAndRadius(padRect, const Radius.circular(4)), paint);
  }

  void _paintParticles(Canvas canvas) {
    for (final p in particles) {
      final alpha = (p.life.clamp(0.0, 1.0) * 200).toInt();
      final paint = Paint()..color = Colors.orange.withAlpha(alpha);
      canvas.drawCircle(p.pos, 2.0 + (1 - p.life) * 1.5, paint);
    }
  }

  void _paintLander(Canvas canvas) {
    final c = math.cos(lander.angle);
    final s = math.sin(lander.angle);

    final body = [
      const Offset(0, -Lander.halfHeight),
      const Offset(-Lander.halfWidth, Lander.halfHeight),
      const Offset(Lander.halfWidth, Lander.halfHeight),
    ];
    Offset rot(Offset v) => Offset(c * v.dx - s * v.dy, s * v.dx + c * v.dy);

    final path = Path();
    final p0 = lander.position + rot(body[0]);
    final p1 = lander.position + rot(body[1]);
    final p2 = lander.position + rot(body[2]);
    path.moveTo(p0.dx, p0.dy);
    path.lineTo(p1.dx, p1.dy);
    path.lineTo(p2.dx, p2.dy);
    path.close();

    if (thrusting && lander.fuel > 0) {
      final flameBase = lander.position + rot(const Offset(0, Lander.halfHeight));
      final flameTip = lander.position + rot(const Offset(0, Lander.halfHeight + 22));
      final flamePaint = Paint()
        ..shader = const RadialGradient(colors: [Colors.yellow, Colors.deepOrange]).createShader(
          Rect.fromCircle(center: flameBase, radius: 24),
        );
      final flamePath = Path()
        ..moveTo(flameBase.dx - 6, flameBase.dy)
        ..quadraticBezierTo(flameBase.dx, flameTip.dy, flameBase.dx + 6, flameBase.dy)
        ..close();
      canvas.drawPath(flamePath, flamePaint);
    }

    final shipPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = Colors.white.withOpacity(0.9);
    canvas.drawPath(path, shipPaint);

    final cockpit = Paint()..color = Colors.lightBlueAccent.withOpacity(0.9);
    canvas.drawCircle(lander.position + rot(const Offset(0, -Lander.halfHeight + 6)), 4, cockpit);

    final legs = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..color = Colors.white70;
    canvas.drawLine(
      lander.position + rot(const Offset(-Lander.halfWidth * 0.6, Lander.halfHeight * 0.6)),
      lander.position + rot(const Offset(-Lander.halfWidth, Lander.halfHeight)),
      legs,
    );
    canvas.drawLine(
      lander.position + rot(const Offset(Lander.halfWidth * 0.6, Lander.halfHeight * 0.6)),
      lander.position + rot(const Offset(Lander.halfWidth, Lander.halfHeight)),
      legs,
    );
  }

  @override
  bool shouldRepaint(covariant GamePainter old) {
    return old.lander != lander ||
        old.terrain != terrain ||
        old.thrusting != thrusting ||
        old.status != status ||
        old.particles != particles;
  }
}

/// ===== Simple runtime policy (loads policy.json and acts greedily) =====

class SimplePolicy {
  final int inputSize;
  final int h1;
  final int h2;
  final List<List<double>> W1, W2, W_thr, W_turn;
  final List<double> b1, b2, b_thr, b_turn;
  final int feGroundSamples;
  final double feStridePx;

  SimplePolicy({
    required this.inputSize,
    required this.h1,
    required this.h2,
    required this.W1,
    required this.b1,
    required this.W2,
    required this.b2,
    required this.W_thr,
    required this.b_thr,
    required this.W_turn,
    required this.b_turn,
    required this.feGroundSamples,
    required this.feStridePx,
  });

  static SimplePolicy fromJson(Map<String, dynamic> j) {
    List<List<double>> _mat(dynamic m) =>
        (m as List).map<List<double>>((r) => (r as List).map((x) => (x as num).toDouble()).toList()).toList();
    List<double> _vec(dynamic v) => (v as List).map((x) => (x as num).toDouble()).toList();

    final fe = (j['fe'] as Map<String, dynamic>?);
    final gs = fe?['groundSamples'] as int? ?? 3;
    final sp = (fe?['stridePx'] as num?)?.toDouble() ?? 48.0;

    return SimplePolicy(
      inputSize: j['inputSize'] as int,
      h1: j['h1'] as int,
      h2: j['h2'] as int,
      W1: _mat(j['W1']),
      b1: _vec(j['b1']),
      W2: _mat(j['W2']),
      b2: _vec(j['b2']),
      W_thr: _mat(j['W_thr']),
      b_thr: _vec(j['b_thr']),
      W_turn: _mat(j['W_turn']),
      b_turn: _vec(j['b_turn']),
      feGroundSamples: gs,
      feStridePx: sp,
    );
  }

  // small helpers
  static List<double> _matVec(List<List<double>> W, List<double> x) {
    final out = List<double>.filled(W.length, 0.0);
    for (int i = 0; i < W.length; i++) {
      double s = 0.0;
      final Wi = W[i];
      for (int j = 0; j < Wi.length; j++) s += Wi[j] * x[j];
      out[i] = s;
    }
    return out;
  }

  static List<double> _reluVec(List<double> v) =>
      v.map((e) => e > 0 ? e : 0.0).toList();

  static double _sigmoid(double z) => 1.0 / (1.0 + math.exp(-z));

  static int _argmax(List<double> a) {
    var bi = 0; var bv = a[0];
    for (var i=1;i<a.length;i++) { if (a[i] > bv) { bv=a[i]; bi=i; } }
    return bi;
  }

  ({bool thrust, bool left, bool right}) actGreedy(
      List<double> x, {
        required double angle,
        required double dxToPad, // -1..+1 approx
        double thrThresh = 0.35,
      }) {
    // forward
    final z1 = _matVec(W1, x);
    for (int i=0;i<z1.length;i++) z1[i] += b1[i];
    final h1v = _reluVec(z1);
    final z2 = _matVec(W2, h1v);
    for (int i=0;i<z2.length;i++) z2[i] += b2[i];
    final h2v = _reluVec(z2);

    // thrust (Bernoulli)
    final thrLogit = _matVec(W_thr, h2v)[0] + b_thr[0];
    final thrP = _sigmoid(thrLogit);
    final thr = thrP > thrThresh;

    // turn (3-class)
    final turnLogits = _matVec(W_turn, h2v);
    for (int i=0;i<3;i++) turnLogits[i] += b_turn[i];

    // same gentle priors we used during training sampling (tiny nudges)
    const kDx = 0.2; const dxDead = 0.05;
    if (dxToPad.abs() > dxDead) {
      turnLogits[1] += (dxToPad > 0 ? kDx : -kDx) * dxToPad.abs(); // left if pad is left
      turnLogits[2] += (dxToPad < 0 ? kDx : -kDx) * dxToPad.abs(); // right if pad is right
    }
    const kAng = 0.3; const angDead = 5 * math.pi / 180;
    if (angle.abs() > angDead) {
      final sign = angle >= 0 ? 1.0 : -1.0;
      final angN = (angle.abs() / (math.pi / 2)).clamp(0.0, 1.0);
      turnLogits[1] += (sign > 0 ? kAng : -kAng) * angN; // counter-rotate
      turnLogits[2] += (sign < 0 ? kAng : -kAng) * angN;
    }

    final cls = _argmax(turnLogits);
    final left = cls == 1;
    final right = cls == 2;
    return (thrust: thr, left: left, right: right);
  }
}
