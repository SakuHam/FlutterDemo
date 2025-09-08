import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart' show Ticker;
import 'package:flutter/rendering.dart' show debugPaintBaselinesEnabled;

void main() {
  // Make sure debug baselines are off if Inspector toggled them.
  assert(() {
    debugPaintBaselinesEnabled = false;
    return true;
  }());
  runApp(const MoonLanderApp());
}

class MoonLanderApp extends StatelessWidget {
  const MoonLanderApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Moon Lander',
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0B0F14),
        textTheme: const TextTheme(bodyMedium: TextStyle(fontFamily: 'monospace')),
      ),
      home: const MainPage(),
    );
  }
}

class MainPage extends StatefulWidget {
  const MainPage({super.key});
  @override
  State<MainPage> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> {
  // Layout
  static const double cellSize = 96; // square cell size
  static const double gap = 16;      // spacing between cells
  final GlobalKey _stackKey = GlobalKey();

  // Icon model
  late List<_AppIcon> icons;

  @override
  void initState() {
    super.initState();
    icons = [
      _AppIcon(
        id: 'game',
        label: 'Moon Lander',
        icon: Icons.rocket_launch,
        row: 0,
        col: 0,
        onTap: () {
          Navigator.push(context, MaterialPageRoute(builder: (_) => const GamePage()));
        },
      ),
      _AppIcon(
        id: 'music',
        label: 'Music',
        icon: Icons.music_note,
        row: 0,
        col: 1,
        onTap: null, // no action yet
      ),
      _AppIcon(
        id: 'settings',
        label: 'Settings',
        icon: Icons.settings,
        row: 0,
        col: 2,
        onTap: null, // no action yet
      ),
    ];
  }

  // Convert (row,col) to pixel offset inside the stack (top-left of the icon)
  Offset _cellToOffset(int row, int col) {
    final double x = col * (cellSize + gap);
    final double y = row * (cellSize + gap);
    return Offset(x, y);
  }

  // Given a local (inside Stack) offset, return nearest (row,col)
  ({int row, int col}) _offsetToCell(Offset local, int cols, int rows) {
    double cx = local.dx / (cellSize + gap);
    double cy = local.dy / (cellSize + gap);
    int col = cx.round().clamp(0, cols - 1);
    int row = cy.round().clamp(0, rows - 1);
    return (row: row, col: col);
  }

  // Get number of columns that fit in current width
  int _colsForWidth(double width) {
    if (width <= 0) return 1;
    final usable = width; // padding handled by SafeArea + outer padding
    final withCell = cellSize + gap;
    return math.max(1, ( (usable + gap) / withCell ).floor());
  }

  // Find icon occupying a cell (if any)
  int? _indexAt(int row, int col) {
    for (int i = 0; i < icons.length; i++) {
      if (icons[i].row == row && icons[i].col == col) return i;
    }
    return null;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) {
            final padding = const EdgeInsets.symmetric(horizontal: 20, vertical: 24);
            final double w = constraints.maxWidth - padding.horizontal;
            final double h = constraints.maxHeight - padding.vertical;
            final cols = _colsForWidth(w);
            // Enough rows to place all icons (grow as needed)
            final rows = math.max( (icons.map((e) => e.row).fold<int>(0, math.max)) + 1,
                ((icons.length + cols - 1) ~/ cols) );

            // Size of the stack area so we can position freely
            final stackWidth  = math.max(w, cols * (cellSize + gap) - gap);
            final stackHeight = math.max(h, rows * (cellSize + gap) - gap);

            return Padding(
              padding: padding,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Apps',
                    style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 12),
                  Expanded(
                    child: Container(
                      decoration: BoxDecoration(
                        color: const Color(0xFF0E141B),
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(color: Colors.white12),
                      ),
                      child: Stack(
                        key: _stackKey,
                        children: [
                          // Optional faint grid
                          Positioned.fill(
                            child: CustomPaint(
                              painter: _GridPainter(
                                cell: cellSize,
                                gap: gap,
                                cols: cols,
                                rows: rows,
                              ),
                            ),
                          ),
                          // Draggable icons
                          ...icons.map((app) {
                            final pos = _cellToOffset(app.row, app.col);
                            return Positioned(
                              left: pos.dx,
                              top: pos.dy,
                              child: _DraggableIcon(
                                size: cellSize,
                                icon: app.icon,
                                label: app.label,
                                onTap: app.onTap,
                                onDragEnd: (globalEnd) {
                                  // Convert global to local within the stack
                                  final box = _stackKey.currentContext!.findRenderObject() as RenderBox;
                                  final localEnd = box.globalToLocal(globalEnd);
                                  // Clamp within the content area
                                  final clamped = Offset(
                                    localEnd.dx.clamp(0.0, stackWidth - cellSize),
                                    localEnd.dy.clamp(0.0, stackHeight - cellSize),
                                  );
                                  final cell = _offsetToCell(clamped, cols, rows);

                                  setState(() {
                                    // If another icon is there, swap
                                    final other = _indexAt(cell.row, cell.col);
                                    if (other != null) {
                                      final tmpRow = icons[other].row;
                                      final tmpCol = icons[other].col;
                                      icons[other].row = app.row;
                                      icons[other].col = app.col;
                                      app
                                        ..row = tmpRow
                                        ..col = tmpCol;
                                    } else {
                                      app
                                        ..row = cell.row
                                        ..col = cell.col;
                                    }
                                  });
                                },
                              ),
                            );
                          }).toList(),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }
}

// --- Models & widgets ---

class _AppIcon {
  final String id;
  final String label;
  final IconData icon;
  final VoidCallback? onTap;
  int row;
  int col;

  _AppIcon({
    required this.id,
    required this.label,
    required this.icon,
    required this.row,
    required this.col,
    required this.onTap,
  });
}

class _DraggableIcon extends StatelessWidget {
  final double size;
  final IconData icon;
  final String label;
  final VoidCallback? onTap;
  final ValueChanged<Offset> onDragEnd; // global offset at end

  const _DraggableIcon({
    required this.size,
    required this.icon,
    required this.label,
    required this.onTap,
    required this.onDragEnd,
  });

  @override
  Widget build(BuildContext context) {
    final tile = _IconTile(size: size, icon: icon, label: label, onTap: onTap);
    return LongPressDraggable(
      feedback: Opacity(
        opacity: 0.9,
        child: Material(
          color: Colors.transparent,
          child: _IconTile(size: size, icon: icon, label: label, onTap: null),
        ),
      ),
      childWhenDragging: Opacity(opacity: 0.15, child: IgnorePointer(child: tile)),
      onDragEnd: (details) => onDragEnd(details.offset),
      child: tile,
    );
  }
}

class _IconTile extends StatelessWidget {
  final double size;
  final IconData icon;
  final String label;
  final VoidCallback? onTap;

  const _IconTile({
    required this.size,
    required this.icon,
    required this.label,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: size,
      height: size,
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: onTap,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: size * 0.58,
              height: size * 0.58,
              decoration: BoxDecoration(
                color: Colors.white10,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: Colors.white24),
              ),
              child: Icon(icon, size: size * 0.40),
            ),
            const SizedBox(height: 8),
            Text(
              label,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(fontSize: 13),
            ),
          ],
        ),
      ),
    );
  }
}

class _GridPainter extends CustomPainter {
  final double cell;
  final double gap;
  final int cols;
  final int rows;

  _GridPainter({
    required this.cell,
    required this.gap,
    required this.cols,
    required this.rows,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white12
      ..style = PaintingStyle.stroke;

    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        final left = c * (cell + gap);
        final top  = r * (cell + gap);
        final rect = Rect.fromLTWH(left, top, cell, cell);
        canvas.drawRRect(RRect.fromRectAndRadius(rect, const Radius.circular(16)), paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant _GridPainter oldDelegate) {
    return oldDelegate.cell != cell ||
        oldDelegate.gap != gap ||
        oldDelegate.cols != cols ||
        oldDelegate.rows != rows;
  }
}

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

  // Tunables (we apply scaling factors in the physics step)
  final Tunables t = Tunables();

  // World & entities
  Size? _worldSize;            // set in LayoutBuilder
  Terrain? _terrain;           // generated once per size
  Lander lander = const Lander(
    position: Offset(160, 120),
    velocity: Offset.zero,
    angle: 0,
    fuel: 100,
  );
  final List<Particle> _particles = [];

  // Input
  bool thrust = false;
  bool left = false;
  bool right = false;

  // Game state
  GameStatus status = GameStatus.playing;

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

  void setPreset(String name) {
    setState(() {
      if (name == 'Easy') {
        t.gravity = 0.12; t.thrustAccel = 0.65; t.rotSpeed = 2.2; t.maxFuel = 140;
      } else if (name == 'Classic') {
        t.gravity = 0.18; t.thrustAccel = 0.42; t.rotSpeed = 1.6; t.maxFuel = 100;
      } else if (name == 'Hard') {
        t.gravity = 0.22; t.thrustAccel = 0.38; t.rotSpeed = 1.3; t.maxFuel = 80;
      }
      _reset();
    });
  }

  void _reset() {
    final size = _worldSize ?? const Size(360, 640);
    setState(() {
      status = GameStatus.playing;
      _particles.clear();
      lander = Lander(
        position: Offset(size.width * 0.2, 120),
        velocity: Offset.zero,
        angle: 0,
        fuel: t.maxFuel,
      );
    });
  }

  void _onTick(Duration _) {
    final now = DateTime.now();
    double dt = now.difference(_last).inMicroseconds / 1e6;
    dt = dt.clamp(0, 1 / 20); // max 50 ms step
    _last = now;

    if (!mounted || status != GameStatus.playing || _terrain == null) return;

    setState(() {
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

      // --- Wrap horizontally using world width ---
      final w = _worldSize?.width ?? 360.0;
      if (pos.dx < 0) pos = Offset(w + pos.dx, pos.dy);
      if (pos.dx > w) pos = Offset(pos.dx - w, pos.dy);

      // --- Particle exhaust (fix axis & base so it's centered) ---
      if (thrustingNow) {
        final c = math.cos(angle);
        final s = math.sin(angle);
        // Engine axis: down in ship frame (0,+1) rotated → (-sinθ, +cosθ)
        final axis = Offset(-s, c);
        // Base point directly under the ship center
        final flameBase = pos + axis * Lander.halfHeight;
        final rnd = math.Random();
        for (int i = 0; i < 6; i++) {
          // Spread: a little along a perpendicular to axis
          final perp = Offset(-c, -s); // perpendicular
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
        final gentle = speed < 40 && angle.abs() < 0.25;
        if (onPad && gentle) {
          status = GameStatus.landed;
          pos = Offset(pos.dx, _terrain!.padY - Lander.halfHeight);
          vel = Offset.zero;
        } else {
          status = GameStatus.crashed;
        }
      }

      lander = lander.copyWith(position: pos, velocity: vel, angle: angle, fuel: fuel);
    });
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        _worldSize = Size(constraints.maxWidth, constraints.maxHeight);
        _terrain ??= Terrain.generate(_worldSize!);
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

            // HUD (top)
            SafeArea(
              child: Padding(
                padding: const EdgeInsets.all(12.0),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        _hudBox(title: 'Fuel', value: lander.fuel.toStringAsFixed(0)),
                        _hudBox(
                          title: 'Vx/Vy',
                          value: '${lander.velocity.dx.toStringAsFixed(1)} / ${lander.velocity.dy.toStringAsFixed(1)}',
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
      },
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

  static Terrain generate(Size size) {
    final rnd = math.Random(42);
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
