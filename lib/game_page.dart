// lib/game_page.dart
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart' show Ticker;

// Engine
import 'engine/types.dart' as et;
import 'engine/types.dart';
import 'engine/game_engine.dart';
import 'engine/raycast.dart';
import 'engine/polygon_carver.dart';

// Runtime policy (AI)
import 'ai/runtime_policy.dart';

/// Simple UI particle for exhaust/smoke
class Particle {
  Offset pos;
  Offset vel;
  double life; // 0..1
  Particle({required this.pos, required this.vel, required this.life});
}

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

  // UI particles
  final List<Particle> _particles = [];

  // Toggles
  bool _showRays = true;
  bool _aiPlay = false;

  // Carver (brush)
  double _brushRadius = 28.0;
  bool _carveMode = true;

  // ===== AI runtime policy =====
  RuntimeTwoStagePolicy? _policy;   // loaded async from assets
  bool get _aiReady => _policy != null;

  // HUD: last AI intent/probs (optional)
  int? _lastIntentIdx;
  List<double> _lastIntentProbs = const [];

  @override
  void initState() {
    super.initState();
    _ticker = createTicker(_onTick)..start();
    _loadPolicy(); // fire-and-forget asset load
  }

  Future<void> _loadPolicy() async {
    try {
      // Adjust asset path if you save with a different name
      final p = await RuntimeTwoStagePolicy.loadFromAsset(
        'assets/ai/policy.json',
        planHold: 2, // re-plan every 2 frames for smoothness; tweak as you like
      );
      if (!mounted) return;
      setState(() {
        _policy = p;
      });
      _toast('AI model loaded');
    } catch (e) {
      // Non-fatal: user can still play with manual controls
      debugPrint('Failed to load AI policy: $e');
      _toast('AI model failed to load');
    }
  }

  @override
  void dispose() {
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
      _engine!.rayCfg = const RayConfig(
        rayCount: 180,
        includeFloor: false,
        forwardAligned: true,
      );
      setState(() {});
    }
  }

  void _reset() {
    _engine?.reset();
    _policy?.resetPlanner(); // also reset the planner
    _particles.clear();
    _thrust = _left = _right = false;
    _lastIntentIdx = null;
    _lastIntentProbs = const [];
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
      }
    });
    _toast(_aiPlay ? 'AI: ON' : 'AI: OFF');
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

    // If AI is on, compute controls (override manual)
    if (_aiPlay && _aiReady) {
      final lander = engine.lander;     // et.LanderState
      final terr = engine.terrain;      // et.Terrain

      final (th, lf, rt, idx, probs) = _policy!.actWithIntent(
        lander: lander,
        terrain: terr,
        worldW: engine.cfg.worldW,
        worldH: engine.cfg.worldH,
        step: 0,
        uiMaxFuel: engine.cfg.t.maxFuel,
      );

      _thrust = th;
      _left = lf;
      _right = rt;

      _lastIntentIdx = idx;
      _lastIntentProbs = probs;
    }

    final info = engine.step(
      dt,
      ControlInput(thrust: _thrust, left: _left, right: _right),
    );

    // Exhaust particles
    final lander = engine.lander;
    if (_thrust && lander.fuel > 0) {
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
    // Update particles
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
                child: GestureDetector(
                  behavior: HitTestBehavior.opaque,
                  onTapDown: (d) => _carveAt(d.localPosition),
                  onPanStart: (d) => _carveAt(d.localPosition),
                  onPanUpdate: (d) => _carveAt(d.localPosition),
                  child: CustomPaint(
                    painter: GamePainter(
                      lander: engine.lander,
                      terrain: engine.terrain,
                      thrusting: _thrust && engine.lander.fuel > 0 && status == GameStatus.playing,
                      status: status,
                      particles: _particles,
                      rays: _showRays ? engine.rays : const [],
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
                      // Top row compact for phones
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
                          const SizedBox(width: 8),
                          _aiToggleIcon(),
                          const SizedBox(width: 8),
                          FilterChip(
                            label: const Text('Show rays'),
                            selected: _showRays,
                            onSelected: (v) => setState(() => _showRays = v),
                          ),
                          const SizedBox(width: 8),
                          ElevatedButton.icon(
                            onPressed: _reset,
                            icon: const Icon(Icons.refresh),
                            label: const Text('Reset'),
                          ),
                        ],
                      ),
                      const SizedBox(height: 10),

                      // Fuel row + tiny icon-only Brush button (right side)
                      Row(
                        children: [
                          _hudBox(title: 'Fuel', value: engine.lander.fuel.toStringAsFixed(0)),
                          const SizedBox(width: 10),
                          _hudBox(
                            title: 'Vx/Vy',
                            value:
                            '${engine.lander.vel.x.toStringAsFixed(1)} / ${engine.lander.vel.y.toStringAsFixed(1)}',
                          ),
                          const SizedBox(width: 10),
                          _hudBox(
                            title: 'Angle',
                            value: '${(engine.lander.angle * 180 / math.pi).toStringAsFixed(0)}°',
                          ),
                          const Spacer(),

                          // --- Icon-only Brush button ---
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
                                  color: _carveMode
                                      ? Colors.green.withOpacity(0.20)
                                      : Colors.white10,
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
/// 360° rays, particles, and the lander triangle.
class GamePainter extends CustomPainter {
  final et.LanderState lander;
  final et.Terrain terrain;
  final bool thrusting;
  final GameStatus status;
  final List<Particle> particles;
  final List<RayHit> rays;

  GamePainter({
    required this.lander,
    required this.terrain,
    required this.thrusting,
    required this.status,
    required this.particles,
    required this.rays,
  });

  @override
  void paint(Canvas canvas, Size size) {
    _paintStars(canvas, size);
    _paintTerrainPoly(canvas, size);
    _paintEdgesOverlay(canvas);
    _paintRays(canvas);
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

    // Terrain rays now RED for visibility
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
  }

  @override
  bool shouldRepaint(covariant GamePainter old) {
    return old.lander != lander ||
        old.terrain != terrain ||
        old.thrusting != thrusting ||
        old.status != status ||
        old.particles != particles ||
        old.rays != rays;
  }
}
