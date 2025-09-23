// lib/game_page.dart
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart' show Ticker;
import 'dart:async';

// Engine
import 'engine/types.dart' as et;
import 'engine/types.dart';
import 'engine/game_engine.dart';
import 'engine/raycast.dart';
import 'engine/polygon_carver.dart';

// Runtime policy (AI)
import 'ai/runtime_policy.dart';

// Potential field
import 'ai/potential_field.dart';

// NEW: Plan bus (planner → UI)
import 'ai/plan_bus.dart';

/// Simple UI particle for exhaust/smoke
class Particle {
  Offset pos;
  Offset vel;
  double life; // 0..1
  Particle({required this.pos, required this.vel, required this.life});
}

enum DebugVisMode { rays, potential, velocity, none }

class GamePage extends StatefulWidget {
  const GamePage({super.key});
  @override
  State<GamePage> createState() => _GamePageState();
}

class _GamePageState extends State<GamePage> with SingleTickerProviderStateMixin {
  late final Ticker _ticker;
  Duration _lastElapsed = Duration.zero;

  Size? _worldSize;
  GameEngine? _engine;

  // Controls (manual or AI writes here)
  bool _thrust = false;
  bool _left = false;
  bool _right = false;

  // NEW: extra channels (driven by AI; UI still 3 buttons)
  bool _sideLeft = false;
  bool _sideRight = false;
  bool _downThrust = false;

  // UI particles
  final List<Particle> _particles = [];

  // Visualization (chip cycles these)
  DebugVisMode _visMode = DebugVisMode.rays;
  PotentialField? _pf; // cached; rebuild when terrain changes

  // AI play toggle
  bool _aiPlay = false;

  // Carver (brush)
  double _brushRadius = 28.0;
  bool _carveMode = true;
  bool _terrainDirty = false;

  // ===== AI runtime policy =====
  RuntimeTwoStagePolicy? _policy;   // loaded async from assets
  bool get _aiReady => _policy != null;

  // HUD: last AI intent/probs (optional)
  int? _lastIntentIdx;
  List<double> _lastIntentProbs = const [];

  // === NEW: per-frame vectors to draw at the lander ===
  Offset? _vecPF;      // raw PF suggestion
  Offset? _vecPolicy;  // policy's preferred velocity (shaped)

  // ===== NEW: ray alignment toggle (FWD vs WORLD) =====
  bool _forwardAligned = true;

  // ===== NEW: Plan overlay state (from PlanBus) =====
  List<Offset> _planPts = const [];
  int _planVersion = 0;
  bool _showPlan = true;
  StreamSubscription<PlanEvent>? _planSub;

  @override
  void initState() {
    super.initState();
    _ticker = createTicker(_onTick)..start();
    _loadPolicy(); // fire-and-forget asset load

    // Listen to planner path updates
    _planSub = PlanBus.instance.stream.listen((e) {
      if (!_showPlan) return; // avoid churn when hidden
      setState(() {
        _planPts = e.points;
        _planVersion = e.version;
      });
    });
  }

  Future<void> _loadPolicy() async {
    try {
      final p = await RuntimeTwoStagePolicy.loadFromAsset(
        'assets/ai/policy.json',
        planHold: 2,
      );
      p.setStochasticPlanner(true);
      p.setIntentTemperature(1.8);

      if (!mounted) return;

      _applyPolicyPhysicsToEngine(p);

      setState(() {
        _policy = p;
      });
      _toast('AI model loaded');
    } catch (e) {
      debugPrint('Failed to load AI policy: $e');
      _toast('AI model failed to load');
    }
  }

  // Apply the physics bundle from the policy JSON to the engine tunables (if supported).
  void _applyPolicyPhysicsToEngine(RuntimeTwoStagePolicy p) {
    final e = _engine;
    if (e == null) return;
    final phys = p.phys; // RuntimePhysics

    try {
      e.cfg = e.cfg.copyWith(
        t: e.cfg.t.copyWith(
          gravity: phys.gravity,
          thrustAccel: phys.thrustAccel,
          rotSpeed: phys.rotSpeed,
          rcsEnabled: phys.rcsEnabled,
          rcsAccel: phys.rcsAccel,
          rcsBodyFrame: phys.rcsBodyFrame,
          downThrEnabled: phys.downThrEnabled,
          downThrAccel: phys.downThrAccel,
          downThrBurn: phys.downThrBurn,
        ),
      );
    } catch (_) {
      debugPrint('Note: engine Tunables missing new physics fields; skipping runtime sync.');
    }
  }

  @override
  void dispose() {
    _planSub?.cancel();
    _ticker.dispose();
    super.dispose();
  }

  void _ensureEngine(Size size) {
    if (_worldSize == null ||
        (_worldSize!.width != size.width || _worldSize!.height != size.height) ||
        _engine == null) {
      _worldSize = size;

      final cfg = EngineConfig(
        worldW: size.width,
        worldH: size.height,
        t: Tunables(),
      );

      _engine = GameEngine(cfg);
      _engine!.rayCfg = RayConfig(
        rayCount: 180,
        includeFloor: false,
        forwardAligned: _forwardAligned, // ← ship-forward aligned by default
      );

      final p = _policy;
      if (p != null) _applyPolicyPhysicsToEngine(p);

      _rebuildPF(); // build once on init/resize
      _policy?.setPotentialField(_pf);
      setState(() {});
    }
  }

  void _syncRayAlignmentToEngine() {
    final e = _engine;
    if (e == null) return;
    e.rayCfg = e.rayCfg.copyWith(forwardAligned: _forwardAligned);
  }

  void _rebuildPF() {
    final e = _engine;
    if (e == null) return;
    _pf = buildPotentialField(
      e,
      nx: 160,
      ny: 120,
      iters: 1200,
      omega: 1.7,
      tol: 1e-4,
    );
  }

  void _reset() {
    _engine?.reset();
    _policy?.resetPlanner();
    _particles.clear();
    _thrust = _left = _right = false;
    _sideLeft = _sideRight = _downThrust = false;
    _lastIntentIdx = null;
    _lastIntentProbs = const [];
    _vecPF = _vecPolicy = null;
    _planPts = const [];
    _planVersion = 0;
    _rebuildPF();
    _policy?.setPotentialField(_pf);
    setState(() {});
  }

  void _toggleAI() {
    if (!_aiReady) {
      _toast('AI not loaded yet');
      return;
    }
    setState(() {
      _aiPlay = !_aiPlay;
      if (_aiPlay) {
        _policy?.resetPlanner();
        _thrust = _left = _right = false;
        _sideLeft = _sideRight = _downThrust = false;
      }
    });
    _toast(_aiPlay ? 'AI: ON' : 'AI: OFF');
  }

  void _togglePlan() {
    setState(() => _showPlan = !_showPlan);
    if (!_showPlan) {
      setState(() { _planPts = const []; });
    }
    _toast(_showPlan ? 'Plan: ON' : 'Plan: OFF');
  }

  void _cycleVis() {
    setState(() {
      final v = DebugVisMode.values;
      _visMode = v[(_visMode.index + 1) % v.length];
    });
    _toast('View: ${_visModeLabel()}');
  }

  String _visModeLabel() {
    switch (_visMode) {
      case DebugVisMode.rays: return 'Rays';
      case DebugVisMode.potential: return 'Potential';
      case DebugVisMode.velocity: return 'Velocity';
      case DebugVisMode.none: return 'None';
    }
  }

  void _toast(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).hideCurrentSnackBar();
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(msg),
        behavior: SnackBarBehavior.floating,
        margin: const EdgeInsets.only(left: 12, right: 12, bottom: 18),
        duration: const Duration(milliseconds: 900),
      ),
    );
  }

  // --- Compute policy & PF velocity targets (mirrors training reward shaping) ---
  void _computeBestVelVectors(GameEngine e, PotentialField pf) {
    final L = e.lander;
    final x = L.pos.x.toDouble();
    final y = L.pos.y.toDouble();
    final vx = L.vel.x.toDouble();
    final vy = L.vel.y.toDouble();
    final W = e.cfg.worldW.toDouble();

    // PF base suggestion
    final base = pf.suggestVelocity(
      x, y,
      vMinClose: 8.0,
      vMaxFar: 90.0,
      alpha: 1.2,
      clampSpeed: 9999.0,
    );
    _vecPF = Offset(base.vx, base.vy);

    // --- Proximity to pad (for flare) ---
    final padCx = e.terrain.padCenter.toDouble();
    final dxAbs = (x - padCx).abs();
    final gy = e.terrain.heightAt(x);
    final h = (gy - y).toDouble().clamp(0.0, 1000.0);

    final tightX = 0.10 * W;
    final ph = math.exp(- (h * h) / (140.0 * 140.0 + 1e-6));
    final px = math.exp(- (dxAbs * dxAbs) / (tightX * tightX + 1e-6));
    final prox = (px * ph).clamp(0.0, 1.0);

    // --- Final-approach flare (prefer killing lateral more) ---
    final vMinTouchdown = 2.0;
    final flareLat = (1.0 - 0.90 * prox);
    final flareVer = (1.0 - 0.70 * prox);

    double svx = base.vx * flareLat;
    double svy = base.vy * flareVer;

    final magNow = (math.sqrt(svx * svx + svy * svy) + 1e-9);
    final magTarget = ((1.0 - prox) * magNow + prox * vMinTouchdown).clamp(0.0, magNow);
    final kMag = (magTarget / magNow).clamp(0.0, 1.0);
    svx *= kMag; svy *= kMag;

    // --- Feasibility clamp ---
    final aMax = e.cfg.t.thrustAccel * 0.05 * e.cfg.stepScale;
    final dt = 1 / e.cfg.stepScale;
    final feasiness = 0.75;
    final dvxNeed = svx - vx;
    final dvyNeed = svy - vy;
    final dvMag = math.sqrt(dvxNeed * dvxNeed + dvyNeed * dvyNeed);
    final dvCap = (aMax * dt * feasiness).clamp(0.0, 1e9);
    if (dvMag > dvCap && dvMag > 1e-9) {
      final s = dvCap / dvMag;
      svx = vx + dvxNeed * s;
      svy = vy + dvyNeed * s;
    }

    // --- Wall avoidance (velocity-only blend inward) ---
    final margin = 0.12 * W;
    final distL = (x).clamp(0.0, margin);
    final distR = (W - x).clamp(0.0, margin);
    final nearL = 1.0 - (distL / margin);
    final nearR = 1.0 - (distR / margin);
    final borderProx = math.max(nearL, nearR);

    double inwardX = 0.0;
    if (nearL > 0.0 && nearL >= nearR) inwardX = 1.0;
    if (nearR > 0.0 && nearR >  nearL) inwardX = -1.0;

    final vInward = 40.0;
    final blendIn = 0.60 * borderProx;
    final vxWall = inwardX * vInward;

    svx = (1.0 - blendIn) * svx + blendIn * vxWall;

    _vecPolicy = Offset(svx, svy);
  }

  // ------------ TICK ------------
  void _onTick(Duration elapsed) {
    if (!mounted) return;

    double dt = (elapsed - _lastElapsed).inMicroseconds / 1e6;
    _lastElapsed = elapsed;
    if (dt <= 0) dt = 1 / 60.0;
    dt = dt.clamp(0, 1 / 20.0);

    final engine = _engine;
    if (engine == null) return;
    if (engine.status != GameStatus.playing) return;

    // Compute per-frame velocity vectors for debug view
    final pf = _pf;
    if (pf != null && _visMode == DebugVisMode.velocity) {
      _computeBestVelVectors(engine, pf);
    } else {
      _vecPF = _vecPolicy = null;
    }

    // If AI is on, compute controls (override manual)
    if (_aiPlay && _aiReady) {
      final lander = engine.lander;     // et.LanderState
      final terr = engine.terrain;      // et.Terrain

      final (th, lf, rt, sL, sR, dT, idx, probs) = _policy!.actWithIntentExt(
        lander: lander,
        terrain: terr,
        worldW: engine.cfg.worldW,
        worldH: engine.cfg.worldH,
        rays: engine.rays,
        step: 0,
        uiMaxFuel: engine.cfg.t.maxFuel,
      );

      _thrust = th;
      _left = lf;
      _right = rt;
      _sideLeft = sL;
      _sideRight = sR;
      _downThrust = dT;

      _lastIntentIdx = idx;
      _lastIntentProbs = probs;
    }

    final info = engine.step(
      dt,
      ControlInput(
        thrust: _thrust,
        left: _left,
        right: _right,
        sideLeft: _sideLeft,
        sideRight: _sideRight,
        downThrust: _downThrust,
      ),
    );

    // Exhaust particles
    final lander = engine.lander;

    _maybeEmitMainFlame(lander);
    _maybeEmitSideRCS(lander);
    _maybeEmitDownwardFlame(lander);

    for (int i = _particles.length - 1; i >= 0; i--) {
      final p = _particles[i];
      p.life -= dt * 1.8;
      p.vel += const Offset(0, 0.18 * 0.05 * 0.2);
      p.pos += p.vel * dt;
      if (p.life <= 0) _particles.removeAt(i);
    }

    if (info.terminal) {
      // engine already set status
    }

    setState(() {});
  }

  void _maybeEmitMainFlame(LanderState lander) {
    final engine = _engine!;
    if (_thrust && lander.fuel > 0 && engine.status == GameStatus.playing) {
      final c = math.cos(lander.angle);
      final s = math.sin(lander.angle);
      const halfH = 18.0;
      final axis = Offset(-s, c); // down in ship frame
      final flameBase = Offset(lander.pos.x, lander.pos.y) + axis * halfH;
      final rnd = math.Random();
      for (int i = 0; i < 6; i++) {
        final perp = Offset(-c, -s);
        final spread = (rnd.nextDouble() - 0.5) * 0.35;
        final dir = (axis + perp * spread);
        final speed = 60 + rnd.nextDouble() * 60;
        _particles.add(Particle(pos: flameBase, vel: dir * speed, life: 1.0));
      }
    }
  }

  void _maybeEmitSideRCS(LanderState lander) {
    if ((!_sideLeft && !_sideRight) || lander.fuel <= 0) return;

    final c = math.cos(lander.angle);
    final s = math.sin(lander.angle);
    const halfH = 12.0;
    const halfW = 14.0;

    final upBody = Offset(s, -c);     // ship “up”
    final rightBody = Offset(c, s);   // ship “right”
    final leftPort = Offset(lander.pos.x, lander.pos.y) + upBody * 0 + rightBody * (-halfW) + upBody * halfH;
    final rightPort = Offset(lander.pos.x, lander.pos.y) + rightBody * (halfW) + upBody * halfH;

    final rnd = math.Random();
    if (_sideLeft) {
      final dir = -rightBody;
      for (int i = 0; i < 2; i++) {
        final jitter = Offset((rnd.nextDouble() - 0.5) * 0.4, (rnd.nextDouble() - 0.5) * 0.2);
        _particles.add(Particle(pos: leftPort, vel: (dir + jitter) * (70 + rnd.nextDouble() * 40), life: 0.8));
      }
    }
    if (_sideRight) {
      final dir = rightBody;
      for (int i = 0; i < 2; i++) {
        final jitter = Offset((rnd.nextDouble() - 0.5) * 0.4, (rnd.nextDouble() - 0.5) * 0.2);
        _particles.add(Particle(pos: rightPort, vel: (dir + jitter) * (70 + rnd.nextDouble() * 40), life: 0.8));
      }
    }
  }

  void _maybeEmitDownwardFlame(LanderState lander) {
    if (!_downThrust || lander.fuel <= 0) return;

    final c = math.cos(lander.angle);
    final s = math.sin(lander.angle);
    const halfH = 6.0;

    final port = Offset(lander.pos.x, lander.pos.y) + Offset(0, halfH);
    final rnd = math.Random();
    for (int i = 0; i < 3; i++) {
      final spread = (rnd.nextDouble() - 0.5) * 0.4;
      final dir = Offset(spread, 1.0); // straight down in screen space
      final speed = 60 + rnd.nextDouble() * 50;
      _particles.add(Particle(pos: port, vel: dir * speed, life: 0.9));
    }
  }

  // ------------ Carving ------------
  void _carveAt(Offset p) {
    if (!_carveMode) return;
    final e = _engine;
    if (e == null) return;
    final carved = TerrainCarver.carveCircle(
      terrain: e.terrain,
      cx: p.dx,
      cy: p.dy,
      r: _brushRadius,
    );
    e.terrain = carved;
    _terrainDirty = true;    // mark dirty
    setState(() {});
  }

  Future<void> _showBrushDialog() async {
    double temp = _brushRadius;
    await showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Brush size'),
        content: StatefulBuilder(
          builder: (_, setLocal) => SizedBox(
            width: 280,
            child: Slider(
              value: temp,
              min: 12,
              max: 80,
              divisions: 17,
              label: temp.toStringAsFixed(0),
              onChanged: (v) => setLocal(() => temp = v),
            ),
          ),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          FilledButton(
            onPressed: () {
              setState(() => _brushRadius = temp);
              Navigator.pop(ctx);
            },
            child: const Text('Apply'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: LayoutBuilder(
        builder: (context, constraints) {
          _ensureEngine(Size(constraints.maxWidth, constraints.maxHeight));
          final engine = _engine!;
          final status = engine.status;

          return Stack(
            children: [
              // Game canvas + tap/drag carving
              Positioned.fill(
                child:GestureDetector(
                  behavior: HitTestBehavior.opaque,
                  onTapDown: (d) => _carveAt(d.localPosition),
                  onTapUp: (_) {
                    if (_terrainDirty) { _rebuildPF(); _policy?.setPotentialField(_pf); _terrainDirty = false; }
                  },
                  onPanStart: (d) => _carveAt(d.localPosition),
                  onPanUpdate: (d) => _carveAt(d.localPosition),
                  onPanEnd: (_) {
                    if (_terrainDirty) { _rebuildPF(); _policy?.setPotentialField(_pf); _terrainDirty = false; }
                  },
                  onPanCancel: () {
                    if (_terrainDirty) { _rebuildPF(); _policy?.setPotentialField(_pf); _terrainDirty = false; }
                  },
                  child: CustomPaint(
                    painter: GamePainter(
                      lander: engine.lander,
                      terrain: engine.terrain,
                      thrusting: _thrust && engine.lander.fuel > 0 && status == GameStatus.playing,
                      status: status,
                      particles: _particles,
                      // Only feed rays to painter when mode == rays
                      rays: _visMode == DebugVisMode.rays ? engine.rays : const [],
                      // Potential field & vis mode
                      pf: _pf,
                      visMode: _visMode,
                      // NEW: pass vectors to draw
                      vecPF: _vecPF,
                      vecPolicy: _vecPolicy,
                      // NEW: plan path to draw
                      planPoints: _showPlan ? _planPts : const [],
                      planVersion: _planVersion,
                      // NEW: pass a hint for painter (to draw forward axis only when useful)
                      showForwardAxis: _visMode == DebugVisMode.rays,
                    ),
                  ),
                ),
              ),

              // HUD
              SafeArea(
                child: Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      // Top row
                      Row(
                        children: [
                          ElevatedButton.icon(
                            onPressed: () => Navigator.of(context).maybePop(),
                            icon: const Icon(Icons.home),
                            label: const Text('Menu'),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.black.withOpacity(0.55),
                              foregroundColor: Colors.white,
                              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                            ),
                          ),
                          const Spacer(),
                          const SizedBox(width: 8),
                          _aiToggleIcon(),
                          const SizedBox(width: 8),
                          _planToggleIcon(), // NEW: plan overlay toggle
                          const SizedBox(width: 8),

                          Tooltip(
                            message: 'Tap to cycle view • Long-press to rebuild field',
                            child: FilterChip(
                              label: Text(_visModeLabel()),
                              selected: _visMode != DebugVisMode.none,
                              onSelected: (_) => _cycleVis(),
                            ),
                          ),
                          const SizedBox(width: 8),

                          /*
                          // Optional: forward/world ray alignment toggle
                          Tooltip(
                            message: 'Ray alignment: ${_forwardAligned ? "Forward (ship frame)" : "World"}',
                            child: FilterChip(
                              label: Text(_forwardAligned ? 'Rays: FWD' : 'Rays: WORLD'),
                              selected: _forwardAligned,
                              onSelected: (v) {
                                setState(() {
                                  _forwardAligned = v;
                                  _syncRayAlignmentToEngine();
                                });
                              },
                            ),
                          ),
                          const SizedBox(width: 8),
                          */

                          ElevatedButton.icon(
                            onPressed: _reset,
                            icon: const Icon(Icons.refresh),
                            label: const Text(''),
                          ),
                        ],
                      ),
                      const SizedBox(height: 10),

                      Row(
                        children: [
                          _hudBox(title: 'Fuel', value: engine.lander.fuel.toStringAsFixed(0)),
                          if (_aiReady)
                            Tooltip(
                              message: _lastIntentIdx == null
                                  ? 'AI idle'
                                  : 'AI intent: ${kIntentNames[_lastIntentIdx!.clamp(0, kIntentNames.length - 1)]}',
                              child: Container(
                                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                                decoration: BoxDecoration(
                                  color: Colors.black.withOpacity(0.45),
                                  borderRadius: BorderRadius.circular(10),
                                  border: Border.all(color: Colors.white24),
                                ),
                                child: Row(
                                  children: [
                                    const Icon(Icons.bolt, size: 16, color: Colors.white70),
                                    const SizedBox(width: 6),
                                    Text(
                                      _lastIntentIdx == null
                                          ? 'AI ready'
                                          : kIntentNames[_lastIntentIdx!.clamp(0, kIntentNames.length - 1)],
                                      style: const TextStyle(color: Colors.white),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          const Spacer(),

                          // Brush button
                          Tooltip(
                            message: _carveMode
                                ? 'Brush ON — tap to disable; long-press to resize'
                                : 'Brush OFF — tap to enable; long-press to resize',
                            child: GestureDetector(
                              onTap: () => setState(() => _carveMode = !_carveMode),
                              onLongPress: _showBrushDialog,
                              child: Container(
                                width: 36,
                                height: 36,
                                decoration: BoxDecoration(
                                  color: _carveMode ? Colors.green.withOpacity(0.20) : Colors.white10,
                                  borderRadius: BorderRadius.circular(10),
                                  border: Border.all(
                                    color: _carveMode ? Colors.greenAccent : Colors.white30,
                                  ),
                                ),
                                alignment: Alignment.center,
                                child: const Icon(Icons.brush, size: 18, color: Colors.white),
                              ),
                            ),
                          ),
                        ],
                      ),

                      const SizedBox(height: 8),
                      if (status != GameStatus.playing)
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
                    ],
                  ),
                ),
              ),

              // Touch controls (ignored if terminal or AI on)
              _buildControls(),
            ],
          );
        },
      ),
    );
  }

  Widget _aiToggleIcon() {
    final active = _aiPlay;
    return Tooltip(
      message: _aiReady
          ? (active ? 'AI Play: ON' : 'AI Play: OFF')
          : 'AI loading…',
      child: InkResponse(
        onTap: _toggleAI,
        radius: 24,
        child: Container(
          padding: const EdgeInsets.all(6),
          decoration: BoxDecoration(
            color: active ? Colors.green.withOpacity(0.20) : Colors.white10,
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: active ? Colors.greenAccent : Colors.white30),
          ),
          child: Icon(
            Icons.smart_toy_outlined,
            color: active ? Colors.greenAccent : (_aiReady ? Colors.white70 : Colors.orangeAccent),
            size: 22,
          ),
        ),
      ),
    );
  }

  Widget _planToggleIcon() {
    final active = _showPlan;
    return Tooltip(
      message: active ? 'Flight plan: ON' : 'Flight plan: OFF',
      child: InkResponse(
        onTap: _togglePlan,
        radius: 24,
        child: Container(
          padding: const EdgeInsets.all(6),
          decoration: BoxDecoration(
            color: active ? Colors.cyanAccent.withOpacity(0.18) : Colors.white10,
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: active ? Colors.cyanAccent : Colors.white30),
          ),
          child: Icon(
            active ? Icons.timeline : Icons.timeline_outlined,
            color: active ? Colors.cyanAccent : Colors.white70,
            size: 22,
          ),
        ),
      ),
    );
  }

  Widget _buildControls() {
    final disabled = _engine?.status != GameStatus.playing || _aiPlay;
    return IgnorePointer(
      ignoring: disabled,
      child: Align(
        alignment: Alignment.bottomCenter,
        child: Padding(
          padding: const EdgeInsets.only(bottom: 24.0),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _holdButton(icon: Icons.rotate_left, onChanged: (v) => setState(() => _left = v)),
              _holdButton(
                icon: Icons.local_fire_department,
                onChanged: (v) => setState(() => _thrust = v),
                big: true,
              ),
              _holdButton(icon: Icons.rotate_right, onChanged: (v) => setState(() => _right = v)),
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
        child: Icon(icon, size: big ? 42 : 32, color: Colors.white),
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

/// Painter that draws polygon terrain (light gray), pad-highlighted edges,
/// rays/potential/velocity overlays, particles, the lander triangle, and the planned path.
class GamePainter extends CustomPainter {
  final et.LanderState lander;
  final et.Terrain terrain;
  final bool thrusting;
  final GameStatus status;
  final List<Particle> particles;
  final List<RayHit> rays;

  // Potential field visualization
  final PotentialField? pf;
  final DebugVisMode visMode;

  // NEW: vectors to overlay at the lander
  final Offset? vecPF;
  final Offset? vecPolicy;

  // NEW: planned path (world-space) and version
  final List<Offset> planPoints;
  final int planVersion;

  // NEW: painter hint to draw forward axis when looking at rays
  final bool showForwardAxis;

  GamePainter({
    required this.lander,
    required this.terrain,
    required this.thrusting,
    required this.status,
    required this.particles,
    required this.rays,
    required this.pf,
    required this.visMode,
    required this.vecPF,
    required this.vecPolicy,
    required this.planPoints,
    required this.planVersion,
    required this.showForwardAxis,
  });

  @override
  void paint(Canvas canvas, Size size) {
    _paintStars(canvas, size);
    _paintTerrainPoly(canvas, size);
    _paintEdgesOverlay(canvas);

    // Potential/Velocity overlays
    if (visMode == DebugVisMode.potential && pf != null) {
      _paintPotentialHeat(canvas, pf!, heatDownsample: 3, alpha: 130);
    } else if (visMode == DebugVisMode.velocity && pf != null) {
      _paintPotentialVectors(canvas, pf!, stride: 8);
    } else if (visMode == DebugVisMode.rays) {
      _paintRays(canvas);
    }

    // Planned path (always on when toggled)
    if (planPoints.length >= 2) {
      _paintPlanPath(canvas, planPoints);
    }

    _paintParticles(canvas);
    _paintLander(canvas);

    if (visMode == DebugVisMode.velocity) {
      _paintBestVelTriplet(canvas);
    }
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

  void _paintTerrainPoly(Canvas canvas, Size size) {
    final poly = terrain.poly;
    final path = Path()..fillType = PathFillType.evenOdd;

    if (poly.outer.isNotEmpty) {
      final a0 = poly.outer.first;
      path.moveTo(a0.x, a0.y);
      for (int i = 1; i < poly.outer.length; i++) {
        final p = poly.outer[i];
        path.lineTo(p.x, p.y);
      }
      path.close();
    }
    for (final hole in poly.holes) {
      if (hole.isEmpty) continue;
      final h0 = hole.first;
      path.moveTo(h0.x, h0.y);
      for (int i = 1; i < hole.length; i++) {
        final p = hole[i];
        path.lineTo(p.x, p.y);
      }
      path.close();
    }

    final ground = Paint()..color = const Color(0xFFD8D8D8); // light gray
    canvas.drawPath(path, ground);
  }

  void _paintEdgesOverlay(Canvas canvas) {
    final edges = terrain.poly.edges;

    final Paint ridgePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..color = Colors.black26;

    final Paint padPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = (status == GameStatus.crashed ? Colors.red : Colors.greenAccent);

    for (final e in edges) {
      final p1 = Offset(e.a.x, e.a.y);
      final p2 = Offset(e.b.x, e.b.y);
      if (e.kind == PolyEdgeKind.pad) {
        canvas.drawLine(p1, p2, padPaint);
      } else {
        canvas.drawLine(p1, p2, ridgePaint);
      }
    }
  }

  void _paintRays(Canvas canvas) {
    if (rays.isEmpty) return;
    final origin = Offset(lander.pos.x, lander.pos.y);

    final wallPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1
      ..color = Colors.blueAccent.withOpacity(0.55);

    final terrPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2
      ..color = Colors.red.withOpacity(0.8);

    final padPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.6
      ..color = Colors.greenAccent.withOpacity(0.95);

    for (final h in rays) {
      final end = Offset(h.p.x, h.p.y);
      switch (h.kind) {
        case RayHitKind.wall:
          canvas.drawLine(origin, end, wallPaint);
          break;
        case RayHitKind.terrain:
          canvas.drawLine(origin, end, terrPaint);
          break;
        case RayHitKind.pad:
          canvas.drawLine(origin, end, padPaint);
          break;
      }
      final dotPaint = Paint()
        ..color = (h.kind == RayHitKind.pad)
            ? Colors.greenAccent
            : (h.kind == RayHitKind.wall ? Colors.blueAccent : Colors.red);
      canvas.drawCircle(end, 1.7, dotPaint);
    }
  }

  void _paintPlanPath(Canvas canvas, List<Offset> pts) {
    // Chaikin smoothing (2 iterations) for a clean curve
    List<Offset> smooth = _chaikin(pts, 2);
    if (smooth.length < 2) return;

    final path = Path()..moveTo(smooth.first.dx, smooth.first.dy);
    for (int i = 1; i < smooth.length; i++) {
      path.lineTo(smooth[i].dx, smooth[i].dy);
    }

    final glow = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 6.0
      ..color = const Color(0xFF6BE7FF).withOpacity(0.18)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 4);
    final stroke = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = const Color(0xFF6BE7FF).withOpacity(0.85)
      ..strokeJoin = StrokeJoin.round
      ..strokeCap = StrokeCap.round
      ..isAntiAlias = true;

    canvas.drawPath(path, glow);
    canvas.drawPath(path, stroke);

    // dotted cadence markers (every ~0.2 s if ~60 points over 2 s)
    final dot = Paint()..style = PaintingStyle.fill..color = const Color(0xFF6BE7FF).withOpacity(0.7);
    for (int i = 0; i < smooth.length; i += 6) {
      canvas.drawCircle(smooth[i], 1.7, dot);
    }
  }

  List<Offset> _chaikin(List<Offset> pts, int iters) {
    if (pts.length < 3 || iters <= 0) return pts;
    var out = pts;
    for (int k = 0; k < iters; k++) {
      if (out.length < 3) break;
      final next = <Offset>[];
      next.add(out.first);
      for (int i = 0; i < out.length - 1; i++) {
        final p = out[i];
        final q = out[i + 1];
        final a = Offset(0.75 * p.dx + 0.25 * q.dx, 0.75 * p.dy + 0.25 * q.dy);
        final b = Offset(0.25 * p.dx + 0.75 * q.dx, 0.25 * p.dy + 0.75 * q.dy);
        next..add(a)..add(b);
      }
      next.add(out.last);
      out = next;
    }
    return out;
  }

  void _paintPotentialHeat(Canvas canvas, PotentialField pf, {int heatDownsample = 3, int alpha = 160}) {
    final paint = Paint()..style = PaintingStyle.fill;
    final nx = pf.gridNx, ny = pf.gridNy;
    final dx = pf.gridDx, dy = pf.gridDy;

    double vmin = 1e9, vmax = -1e9;
    for (int j = 0; j < ny; j += heatDownsample) {
      for (int i = 0; i < nx; i += heatDownsample) {
        final v = pf.phiAtIndex(i, j);
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
      }
    }
    final span = (vmax - vmin).abs() < 1e-9 ? 1.0 : (vmax - vmin);

    for (int j = 0; j < ny - 1; j += heatDownsample) {
      for (int i = 0; i < nx - 1; i += heatDownsample) {
        final v = pf.phiAtIndex(i, j);
        final t = ((v - vmin) / span).clamp(0.0, 1.0);
        paint.color = _lerpTurbo(t).withAlpha(alpha);
        final rect = Rect.fromLTWH(i * dx, j * dy, dx * heatDownsample, dy * heatDownsample);
        canvas.drawRect(rect, paint);
      }
    }

    final obst = Paint()..style = PaintingStyle.stroke..strokeWidth = 1.0..color = const Color(0xAA000000);
    final padP = Paint()..style = PaintingStyle.stroke..strokeWidth = 1.0..color = const Color(0xAA00FFFF);
    for (int j = 0; j < ny; j += heatDownsample) {
      for (int i = 0; i < nx; i += heatDownsample) {
        final m = pf.maskAtIndex(i, j);
        if (m == 2) {
          canvas.drawRect(Rect.fromLTWH(i * dx, j * dy, dx * heatDownsample, dy * heatDownsample), obst);
        } else if (m == 1) {
          canvas.drawRect(Rect.fromLTWH(i * dx, j * dy, dx * heatDownsample, dy * heatDownsample), padP);
        }
      }
    }
  }

  void _paintPotentialVectors(
      Canvas canvas,
      PotentialField pf, {
        int stride = 8,
        double arrowScale = 32.0,
      }) {
    final nx = pf.gridNx, ny = pf.gridNy;
    final dx = pf.gridDx, dy = pf.gridDy;

    double maxMag = 1e-9;
    for (int j = 1; j < ny - 1; j += stride) {
      for (int i = 1; i < nx - 1; i += stride) {
        if (pf.maskAtIndex(i, j) == 2) continue; // obstacle
        final flow = pf.sampleFlow(i * dx, j * dy);
        if (flow.mag > maxMag) maxMag = flow.mag;
      }
    }

    for (int j = 1; j < ny - 1; j += stride) {
      for (int i = 1; i < nx - 1; i += stride) {
        if (pf.maskAtIndex(i, j) == 2) continue;
        final x = i * dx;
        final y = j * dy;
        final flow = pf.sampleFlow(x, y);

        final n = (flow.mag / maxMag).clamp(0.0, 1.0);
        final color = HSVColor.fromAHSV(
          1.0,
          200.0 - 160.0 * n,
          0.90,
          0.35 + 0.65 * n,
        ).toColor();

        final lenScale = arrowScale * 1.0;

        final sx = x, sy = y;
        final ex = x + flow.nx * lenScale;
        final ey = y + flow.ny * lenScale;

        final line = Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.0
          ..color = color;

        final head = Paint()
          ..style = PaintingStyle.fill
          ..color = color;

        canvas.drawLine(Offset(sx, sy), Offset(ex, ey), line);
        _arrowHead(canvas, head, Offset(ex, ey), math.atan2(ey - sy, ex - sx), 7.0, 5.0);
      }
    }
  }

  // === NEW: draw actual v, PF suggestion, and policy target at the ship ===
  void _paintBestVelTriplet(Canvas canvas) {
    final pos = Offset(lander.pos.x, lander.pos.y);

    // Arrow scale just for on-screen visibility (px/s → px of arrow)
    const double k = 3.0;

    // Actual velocity (white)
    final vNow = Offset(lander.vel.x, lander.vel.y);
    if (vNow.distance > 1e-3) {
      _drawArrow(canvas, pos, pos + vNow * k * 10.0,
          line: Paint()..color = Colors.white..strokeWidth = 2.0,
          head: Paint()..color = Colors.white);
    }

    // PF suggestion (cyan)
    if (vecPF != null && vecPF!.distance > 1e-3) {
      _drawArrow(canvas, pos, pos + vecPF! * k,
          line: Paint()..color = Colors.cyanAccent..strokeWidth = 2.0,
          head: Paint()..color = Colors.cyanAccent);
    }

    // Policy-preferred (magenta)
    if (vecPolicy != null && vecPolicy!.distance > 1e-3) {
      _drawArrow(canvas, pos, pos + vecPolicy! * k * 10.0,
          line: Paint()..color = Colors.pinkAccent..strokeWidth = 2.4,
          head: Paint()..color = Colors.pinkAccent);
    }

    // Tiny legend
    final tp = TextPainter(
      text: const TextSpan(
        text: 'v (white), PF (cyan), policy (magenta)',
        style: TextStyle(color: Colors.white70, fontSize: 11),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas, pos + const Offset(12, -26));
  }

  void _drawArrow(Canvas canvas, Offset a, Offset b, {required Paint line, required Paint head}) {
    canvas.drawLine(a, b, line);
    final ang = math.atan2(b.dy - a.dy, b.dx - a.dx);
    _arrowHead(canvas, head, b, ang, 8.0, 5.0);
  }

  void _arrowHead(Canvas canvas, Paint paint, Offset tip, double ang, double len, double w) {
    final left = Offset(
      tip.dx - len * math.cos(ang) + w * math.sin(ang),
      tip.dy - len * math.sin(ang) - w * math.cos(ang),
    );
    final right = Offset(
      tip.dx - len * math.cos(ang) - w * math.sin(ang),
      tip.dy - len * math.sin(ang) + w * math.cos(ang),
    );
    final path = Path()
      ..moveTo(tip.dx, tip.dy)
      ..lineTo(left.dx, left.dy)
      ..lineTo(right.dx, right.dy)
      ..close();
    canvas.drawPath(path, paint);
  }

  Color _lerpTurbo(double t) {
    t = t.clamp(0.0, 1.0);
    final r = (34.61 + t * (1172.33 + t * (-10793.56 + t * (33300.12 + t * (-38394.49 + t * 14825.05)))))/255.0;
    final g = (23.31 + t * (557.33 + t * (1225.33 + t * (-3574.96 + t * (4479.07 + t * -1930.66)))))/255.0;
    final b = (27.2 + t * (3211.1 + t * (-15327.97 + t * (27814.0 + t * (-22569.18 + t * 6838.66)))))/255.0;
    int c(double v) => (v.clamp(0.0, 1.0) * 255).round();
    return Color.fromARGB(255, c(r), c(g), c(b));
  }

  void _paintParticles(Canvas canvas) {
    for (final p in particles) {
      final alpha = (p.life.clamp(0.0, 1.0) * 200).toInt();
      final paint = Paint()..color = Colors.orange.withAlpha(alpha);
      canvas.drawCircle(p.pos, 2.0 + (1 - p.life) * 1.5, paint);
    }
  }

  void _paintLander(Canvas canvas) {
    const halfW = 14.0;
    const halfH = 18.0;

    final c = math.cos(lander.angle);
    final s = math.sin(lander.angle);

    Offset rot(Offset v) => Offset(c * v.dx - s * v.dy, s * v.dx + c * v.dy);
    final pos = Offset(lander.pos.x, lander.pos.y);

    final body = [
      const Offset(0, -halfH),
      const Offset(-halfW, halfH),
      const Offset(halfW, halfH),
    ];

    final p0 = pos + rot(body[0]);
    final p1 = pos + rot(body[1]);
    final p2 = pos + rot(body[2]);

    final path = Path()
      ..moveTo(p0.dx, p0.dy)
      ..lineTo(p1.dx, p1.dy)
      ..lineTo(p2.dx, p2.dy)
      ..close();

    if (thrusting && lander.fuel > 0) {
      final flameBase = pos + rot(const Offset(0, halfH));
      final flamePaint = Paint()
        ..shader = const RadialGradient(colors: [Colors.yellow, Colors.deepOrange]).createShader(
          Rect.fromCircle(center: flameBase, radius: 24),
        );
      final flamePath = Path()
        ..moveTo(flameBase.dx - 6, flameBase.dy)
        ..quadraticBezierTo(flameBase.dx, flameBase.dy + 22, flameBase.dx + 6, flameBase.dy)
        ..close();
      canvas.drawPath(flamePath, flamePaint);
    }

    final shipPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = Colors.white.withOpacity(0.9);
    canvas.drawPath(path, shipPaint);

    final cockpit = Paint()..color = Colors.lightBlueAccent.withOpacity(0.9);
    canvas.drawCircle(pos + rot(const Offset(0, -halfH + 6)), 4, cockpit);

    final legs = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..color = Colors.white70;
    canvas.drawLine(
      pos + rot(Offset(-halfW * 0.6, halfH * 0.6)),
      pos + rot(const Offset(-halfW, halfH)),
      legs,
    );
    canvas.drawLine(
      pos + rot(Offset(halfW * 0.6, halfH * 0.6)),
      pos + rot(const Offset(halfW, halfH)),
      legs,
    );

    // NEW: draw a short forward-axis line (from the nose) to visualize ship heading
    if (showForwardAxis) {
      final nose = pos + rot(const Offset(0, -halfH));
      final forward = Offset(-math.sin(lander.angle), math.cos(lander.angle));
      final axisPaint = Paint()
        ..color = Colors.amberAccent
        ..strokeWidth = 2;
      canvas.drawLine(nose, nose + forward * 26.0, axisPaint);
    }
  }

  @override
  bool shouldRepaint(covariant GamePainter old) {
    return old.lander != lander ||
        old.terrain != terrain ||
        old.thrusting != thrusting ||
        old.status != status ||
        old.particles != particles ||
        old.rays != rays ||
        old.pf != pf ||
        old.visMode != visMode ||
        old.vecPF != vecPF ||
        old.vecPolicy != vecPolicy ||
        old.planVersion != planVersion ||
        old.planPoints != planPoints ||
        old.showForwardAxis != showForwardAxis;
  }
}
