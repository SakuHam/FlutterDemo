// lib/game_page.dart
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart' show Ticker;

// Engine
import 'engine/types.dart';
import 'engine/game_engine.dart';
import 'engine/raycast.dart';

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

  @override
  void initState() {
    super.initState();
    _ticker = createTicker(_onTick)..start();
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

      // Defaults via constructor
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
    _particles.clear();
    _thrust = _left = _right = false;
    setState(() {});
  }

  void _toggleAI() {
    setState(() {
      _aiPlay = !_aiPlay;
      if (_aiPlay) {
        // clear manual inputs when AI takes over
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

  void _onTick(Duration elapsed) {
    if (!mounted) return;

    double dt = (elapsed - _lastElapsed).inMicroseconds / 1e6;
    _lastElapsed = elapsed;
    if (dt <= 0) dt = 1 / 60.0;
    dt = dt.clamp(0, 1 / 20.0);

    final engine = _engine;
    if (engine == null) return;
    if (engine.status != GameStatus.playing) return;

    // If AI is on, compute controls here (override manual).
    if (_aiPlay) {
      final u = _aiHeuristic(engine);
      _thrust = u.thrust;
      _left = u.left;
      _right = u.right;
    }

    // Step engine with current controls
    final info = engine.step(
      dt,
      ControlInput(thrust: _thrust, left: _left, right: _right),
    );

    // Simple exhaust particles
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
      p.vel += const Offset(0, 0.18 * 0.05 * 0.2); // tiny gravity on smoke
      p.pos += p.vel * dt;
      if (p.life <= 0) _particles.removeAt(i);
    }

    if (info.terminal) {
      // no-op; engine already set status
    }

    setState(() {});
  }

  /// Lightweight heuristic autopilot:
  /// - Align toward pad center with a PD rule.
  /// - Manage descent: keep vy within target band depending on altitude.
  ControlInput _aiHeuristic(GameEngine engine) {
    final s = engine.lander;
    final terrain = engine.terrain;

    final padX = terrain.padCenter;
    final dx = (padX - s.pos.x);            // + -> pad to the right
    final vx = s.vel.x;
    final vy = s.vel.y;                      // + down
    final angle = s.angle;

    // Altitude from polygon roof above lander.x
    final groundY = terrain.heightAt(s.pos.x, worldH: engine.cfg.worldH);
    final alt = groundY.isFinite ? (groundY - s.pos.y) : 9999.0;

    // PD target angle: lean into the pad direction and opposing current vx.
    // Clamp to safe tilt range.
    final kx = 0.0009;    // position gain
    final kv = 0.020;     // velocity gain
    double targetAng = (kx * dx + kv * vx).clamp(-0.6, 0.6);

    // Near the ground, flare toward level
    if (alt < 80) targetAng *= (alt / 80).clamp(0.0, 1.0);

    // Turn decision with deadband
    const dead = 0.03;
    bool left = false, right = false;
    if (angle < targetAng - dead) {
      right = true; // rotate CW (angle increases)
    } else if (angle > targetAng + dead) {
      left = true;  // rotate CCW (angle decreases)
    }

    // Descent profile (target vy+). Faster up high, gentle near ground.
    double targetVy = 60.0;      // px/s down allowed far away
    if (alt < 200) targetVy = 45.0;
    if (alt < 120) targetVy = 32.0;
    if (alt < 60)  targetVy = 22.0;
    if (alt < 30)  targetVy = 14.0;

    // If falling faster than targetVy, burn.
    bool thrust = (vy > targetVy);

    // Also burn if we are tilted heavily and close to ground, to regain control
    if (alt < 50 && angle.abs() > 0.35) thrust = true;

    // Small hover assist when almost over pad
    final closeToCenter = dx.abs() < (engine.cfg.worldW * 0.03);
    if (closeToCenter && alt < 26 && vy > 8.0) thrust = true;

    // Out of fuel => nothing we can do
    if (s.fuel <= 0.0) thrust = false;

    return ControlInput(thrust: thrust, left: left, right: right);
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
              // Game canvas
              Positioned.fill(
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

              // HUD + AI toggle
              SafeArea(
                child: Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
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
                          // AI toggle button (nice chip-like icon)
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
                      Row(
                        children: [
                          _hudBox(title: 'Fuel', value: engine.lander.fuel.toStringAsFixed(0)),
                          const SizedBox(width: 10),
                          _hudBox(
                            title: 'Vx/Vy',
                            value: '${engine.lander.vel.x.toStringAsFixed(1)} / ${engine.lander.vel.y.toStringAsFixed(1)}',
                          ),
                          const SizedBox(width: 10),
                          _hudBox(
                            title: 'Angle',
                            value: '${(engine.lander.angle * 180 / math.pi).toStringAsFixed(0)}°',
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
      message: active ? 'AI Play: ON' : 'AI Play: OFF',
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
            color: active ? Colors.greenAccent : Colors.white70,
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

/// Painter that draws polygon terrain (with holes), edges (pad highlighted),
/// 360° rays from engine, particles, and the lander triangle.
class GamePainter extends CustomPainter {
  final LanderState lander;     // engine type
  final Terrain terrain;        // engine type (has .poly)
  final bool thrusting;
  final GameStatus status;      // engine enum
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

  // ---------- Background ----------
  void _paintStars(Canvas canvas, Size size) {
    final paint = Paint()..color = Colors.white.withOpacity(0.8);
    final rnd = math.Random(1);
    for (int i = 0; i < 120; i++) {
      final dx = rnd.nextDouble() * size.width;
      final dy = rnd.nextDouble() * size.height * 0.6;
      canvas.drawCircle(Offset(dx, dy), rnd.nextDouble() * 1.2, paint);
    }
  }

  // ---------- Terrain (polygon fill with holes) ----------
  void _paintTerrainPoly(Canvas canvas, Size size) {
    final poly = terrain.poly;

    final path = Path()..fillType = PathFillType.evenOdd;

    // Outer ring
    if (poly.outer.isNotEmpty) {
      final a0 = poly.outer.first;
      path.moveTo(a0.x, a0.y);
      for (int i = 1; i < poly.outer.length; i++) {
        final p = poly.outer[i];
        path.lineTo(p.x, p.y);
      }
      path.close();
    }

    // Holes
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

    final ground = Paint()
      ..shader = const LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [Color(0xFF1C2330), Color(0xFF0E141B)],
      ).createShader(Offset.zero & size);

    canvas.drawPath(path, ground);
  }

  // ---------- Edge overlay (pad edges highlighted) ----------
  void _paintEdgesOverlay(Canvas canvas) {
    final edges = terrain.poly.edges;

    final Paint ridgePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..color = Colors.white10;

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

  // ---------- Rays (engine-provided) ----------
  void _paintRays(Canvas canvas) {
    if (rays.isEmpty) return;
    final origin = Offset(lander.pos.x, lander.pos.y);

    final wallPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1
      ..color = Colors.blueAccent.withOpacity(0.45);

    final terrPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1
      ..color = Colors.white.withOpacity(0.35);

    final padPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5
      ..color = Colors.greenAccent.withOpacity(0.9);

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
      final dotPaint = Paint()..color = (h.kind == RayHitKind.pad)
          ? Colors.greenAccent
          : (h.kind == RayHitKind.wall ? Colors.blueAccent : Colors.white70);
      canvas.drawCircle(end, 1.5, dotPaint);
    }
  }

  // ---------- Particles ----------
  void _paintParticles(Canvas canvas) {
    for (final p in particles) {
      final alpha = (p.life.clamp(0.0, 1.0) * 200).toInt();
      final paint = Paint()..color = Colors.orange.withAlpha(alpha);
      canvas.drawCircle(p.pos, 2.0 + (1 - p.life) * 1.5, paint);
    }
  }

  // ---------- Lander ----------
  void _paintLander(Canvas canvas) {
    // Match engine hull (14 x 18 triangle)
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
      final flameTip = pos + rot(const Offset(0, halfH + 22));
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
