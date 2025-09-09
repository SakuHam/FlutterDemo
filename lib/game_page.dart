// lib/game_page.dart
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart' show Ticker;

import 'engine/game_engine.dart' as eng; // >>>

class GamePage extends StatefulWidget {
  const GamePage({super.key});
  @override
  State<GamePage> createState() => _GamePageState();
}

enum GameStatus { playing, landed, crashed }

class Tunables {
  double gravity;      // base gravity strength
  double thrustAccel;  // engine acceleration
  double rotSpeed;     // radians per second
  double maxFuel;      // fuel units
  Tunables({
    this.gravity = 0.18,
    this.thrustAccel = 0.42,
    this.rotSpeed = 1.6,
    this.maxFuel = 100.0,
  });
}

class Particle {
  Offset pos;
  Offset vel;
  double life; // 0..1
  Particle({required this.pos, required this.vel, required this.life});
}

class _GamePageState extends State<GamePage> with SingleTickerProviderStateMixin {
  late final Ticker _ticker;

  // HUD/controls tunables (mirrors engine Tunables)
  final Tunables t = Tunables();

  // World & engine
  Size? _worldSize;
  late eng.GameEngine _engine;               // >>>
  late eng.EngineConfig _cfg;                // >>>

  // Render/adapter state
  GameStatus status = GameStatus.playing;
  final List<Particle> _particles = [];

  // Input
  bool thrust = false;
  bool left = false;
  bool right = false;

  // Timing
  late DateTime _last;

  @override
  void initState() {
    super.initState();
    _ticker = createTicker(_onTick)..start();
    _last = DateTime.now();
  }

  @override
  void dispose() {
    _ticker.dispose();
    super.dispose();
  }

  void _ensureEngine(Size size, {int seed = 42}) {
    // Build engine config based on current tunables
    _cfg = eng.EngineConfig(
      worldW: size.width,
      worldH: size.height,
      t: eng.Tunables(
        gravity: t.gravity,
        thrustAccel: t.thrustAccel,
        rotSpeed: t.rotSpeed,
        maxFuel: t.maxFuel,
      ),
      seed: seed,
    );
    _engine = eng.GameEngine(_cfg);
  }

  void setPreset(String name) {
    setState(() {
      if (name == 'Easy') {
        t.gravity = 0.12; t.thrustAccel = 0.65; t.rotSpeed = 2.2; t.maxFuel = 140;
      } else if (name == 'Classic') {
        t.gravity = 0.18; t.thrustAccel = 0.42; t.rotSpeed = 1.6; t.maxFuel = 100;
      } else if (name == 'Hard') {
        t.gravity = 0.22; t.thrustAccel = 0.38; t.rotSpeed = 1.3; t.maxFuel = 80;
      }
      // Recreate engine with new tunables but same world size
      final sz = _worldSize ?? const Size(360, 640);
      _ensureEngine(sz, seed: 42);
      _particles.clear();
      status = GameStatus.playing;
    });
  }

  void _reset() {
    setState(() {
      _engine.reset(seed: 42);
      _particles.clear();
      status = GameStatus.playing;
    });
  }

  void _onTick(Duration _) {
    final now = DateTime.now();
    double dt = now.difference(_last).inMicroseconds / 1e6;
    dt = dt.clamp(0, 1 / 20); // max 50 ms step
    _last = now;

    if (!mounted || _worldSize == null) return;
    if (status != GameStatus.playing) return;

    setState(() {
      final info = _engine.step(dt, eng.ControlInput(thrust: thrust, left: left, right: right));

      // Particles (UI-only)
      final lander = _engine.lander;
      if (thrust && lander.fuel > 0) {
        final c = math.cos(lander.angle);
        final s = math.sin(lander.angle);
        final axis = Offset(-s, c);
        final flameBase = Offset(lander.pos.x, lander.pos.y) + axis * Lander.halfHeight;
        final rnd = math.Random();
        for (int i = 0; i < 6; i++) {
          final perp = Offset(-c, -s);
          final spread = (rnd.nextDouble() - 0.5) * 0.35;
          final dir = (axis + perp * spread);
          final speed = 60 + rnd.nextDouble() * 60;
          _particles.add(Particle(pos: flameBase, vel: dir * speed, life: 1.0));
        }
      }
      for (int i = _particles.length - 1; i >= 0; i--) {
        final p = _particles[i];
        p.life -= dt * 1.8;
        p.vel += const Offset(0, 1.0) * (t.gravity * 0.05 * 0.2); // tiny gravity
        p.pos += p.vel * dt;
        if (p.life <= 0) _particles.removeAt(i);
      }

      if (info.terminal) {
        status = (info.status == eng.GameStatus.landed) ? GameStatus.landed : GameStatus.crashed;
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, constraints) {
      final size = Size(constraints.maxWidth, constraints.maxHeight);
      if (_worldSize == null) {
        _worldSize = size;
        _ensureEngine(size, seed: 42);
      } else {
        _worldSize = size;
      }

      // Adapter lander/terrain to painter
      final lander = _engine.lander;
      final terrain = _engine.terrain;

      return Stack(
        children: [
          // Game canvas
          Positioned.fill(
            child: CustomPaint(
              painter: GamePainter(
                lander: Lander(
                  position: Offset(lander.pos.x, lander.pos.y),
                  velocity: Offset(lander.vel.x, lander.vel.y),
                  angle: lander.angle,
                  fuel: lander.fuel,
                ),
                terrain: Terrain(
                  points: terrain.ridge
                      .map((v) => Offset(v.x, v.y))
                      .toList(growable: false),
                  padX1: terrain.padX1,
                  padX2: terrain.padX2,
                  padY: terrain.padY,
                ),
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
                      _hudBox(title: 'Fuel', value: _engine.lander.fuel.toStringAsFixed(0)),
                      _hudBox(
                        title: 'Vx/Vy',
                        value:
                        '${_engine.lander.vel.x.toStringAsFixed(1)} / ${_engine.lander.vel.y.toStringAsFixed(1)}',
                      ),
                      _hudBox(
                        title: 'Angle',
                        value: '${(_engine.lander.angle * 180 / math.pi).toStringAsFixed(0)}°',
                      ),
                      _hudBox(
                        title: 'Score', // >>> NEW
                        value: _engine.score.toStringAsFixed(0),
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
                  if (status != GameStatus.playing)
                    Padding(
                      padding: const EdgeInsets.only(top: 8.0),
                      child: Center(
                        child: ElevatedButton.icon(
                          onPressed: _reset,
                          icon: const Icon(Icons.refresh),
                          label: const Text('Reset'),
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),

          // Controls
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
        ],
      );
    });
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
              _holdButton(icon: Icons.local_fire_department, onChanged: (v) => setState(() => thrust = v), big: true),
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

/// ====== Painter layer (unchanged from yours, uses adapter structs) ======
class Lander {
  final Offset position; // center
  final Offset velocity;
  final double angle; // radians
  final double fuel;

  static const double halfWidth = 14;
  static const double halfHeight = 18;

  const Lander({
    required this.position,
    required this.velocity,
    required this.angle,
    required this.fuel,
  });

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
  final List<Offset> points;
  final double padX1;
  final double padX2;
  final double padY;
  Terrain({required this.points, required this.padX1, required this.padX2, required this.padY});

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
      final flameTip  = lander.position + rot(const Offset(0, Lander.halfHeight + 22));
      final flamePaint = Paint()
        ..shader = const RadialGradient(colors: [Colors.yellow, Colors.deepOrange]).createShader(
          Rect.fromCircle(center: flameBase, radius: 24),
        );
      final flamePath = Path()
        ..moveTo(flameBase.dx - 6, flameBase.dy)
        ..quadraticBezierTo(flameTip.dx, flameTip.dy, flameBase.dx + 6, flameBase.dy)
        ..close();
      canvas.drawPath(flamePath, flamePaint);
    }

    final shipPaint = Paint()..color = Colors.white.withOpacity(0.9);
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
