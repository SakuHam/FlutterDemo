// lib/ui/game_page.dart
import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart' show Ticker;

// ===== Adjust these paths to your project structure =====
import '../engine/game_engine.dart' as eng;
import '../engine/types.dart' as eng;

// Optional — if you’re using the policy & intent bus
import 'package:flutter_application_1/ai/intent_bus.dart';
import 'package:flutter_application_1/ai/runtime_policy.dart';

/// ---------------------------------------------------------------------------
/// Shims so RuntimeTwoStagePolicy can keep using (Lander, Terrain) signatures.
/// They are just read-only adapters around engine state.
/// ---------------------------------------------------------------------------
class Lander {
  final Offset position;
  final Offset velocity;
  final double angle;
  final double fuel;

  Lander.fromEngine(eng.LanderState s)
      : position = Offset(s.pos.x, s.pos.y),
        velocity = Offset(s.vel.x, s.vel.y),
        angle = s.angle,
        fuel = s.fuel;
}

class Terrain {
  final eng.Terrain _t;
  Terrain(this._t);

  double heightAt(double x) => _t.heightAt(x);
  double get padX1 => _t.padX1;
  double get padX2 => _t.padX2;
  double get padY  => _t.padY;
}

/// ---------------------------------------------------------------------------
/// GamePage — UI is now a thin shell over the engine. No ray math here.
/// ---------------------------------------------------------------------------
class GamePage extends StatefulWidget {
  const GamePage({super.key, this.config});

  /// Optionally pass a ready EngineConfig. If null, we create one from the
  /// screen size via the EngineConfig constructor defaults.
  final eng.EngineConfig? config;

  @override
  State<GamePage> createState() => _GamePageState();
}

class _GamePageState extends State<GamePage> with SingleTickerProviderStateMixin {
  late final Ticker _ticker;

  // Engine + config
  eng.GameEngine? _engine;
  eng.EngineConfig? _cfg;

  // Manual input (ignored when AI is on)
  bool thrust = false;
  bool left = false;
  bool right = false;

  // AI (optional)
  bool aiPlay = false;
  RuntimeTwoStagePolicy? _policy;
  String _policyInfo = 'No policy loaded';
  String _intentLabel = '—';
  List<double>? _intentProbs;

  // Intent bus (optional)
  StreamSubscription<IntentEvent>? _intentSub;
  StreamSubscription<ControlEvent>? _controlSub;

  // Ray UI toggles (we just mirror these into engine.rayCfg)
  bool _showRays = true;
  int _rayCountUi = 180;
  bool _rayIncludeFloor = false;

  // Timing
  Duration _lastElapsed = Duration.zero;
  int _frame = 0;

  @override
  void initState() {
    super.initState();
    _ticker = createTicker(_onTick)..start();

    // Optional debug bus hooks
    _intentSub = IntentBus.instance.intentsWithReplay().listen((e) {
      debugPrint('[BUS/UI] intent=${e.intent} probs=${e.probs}');
    });
    _controlSub = IntentBus.instance.controlsWithReplay().listen((c) {
      debugPrint('[BUS/UI] control: T=${c.thrust} L=${c.left} R=${c.right} meta=${c.meta}');
    });

    // Optional: load policy from assets
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      try {
        final pol = await RuntimeTwoStagePolicy.loadFromAsset(
          'assets/ai/policy.json',
          planHold: 1,
        );
        setState(() {
          _policy = pol;
          _policyInfo =
          'Loaded policy: h1=${pol.h1}, h2=${pol.h2}, fe(gs=${pol.fe.groundSamples}, stride=${pol.fe.stridePx}), planHold=12';
        });
      } catch (e) {
        debugPrint('Policy load failed (ok if not using AI): $e');
      }
    });
  }

  @override
  void dispose() {
    _ticker.dispose();
    _intentSub?.cancel();
    _controlSub?.cancel();
    super.dispose();
  }

  // Build a default EngineConfig using its constructor defaults
  eng.EngineConfig _makeEngineConfig(Size world) {
    return eng.EngineConfig(
      worldW: world.width,
      worldH: world.height,
      t: eng.Tunables(
        gravity: 0.18,
        thrustAccel: 0.42,
        rotSpeed: 1.6,
        maxFuel: 1000.0,
        crashOnTilt: true,
        landingMaxVx: 28.0,
        landingMaxVy: 38.0,
        landingMaxOmega: 2.0,
      ),
      // Everything else keeps EngineConfig’s default values.
    );
  }

  void _ensureEngine(Size world) {
    if (_engine != null) return;
    _cfg = widget.config ?? _makeEngineConfig(world);
    _engine = eng.GameEngine(_cfg!);
    // Mirror ray UI -> engine
    _engine!.rayCfg = _engine!.rayCfg.copyWith(
      rayCount: _rayCountUi,
      includeFloor: _rayIncludeFloor,
      forwardAligned: true,
    );
  }

  void _toggleAI() {
    setState(() => aiPlay = !aiPlay);
    _toast(aiPlay ? 'AI: ON' : 'AI: OFF');
  }

  void _reset() {
    _engine?.reset();
    setState(() {
      _intentLabel = '—';
      _intentProbs = null;
      _frame = 0;
    });
  }

  void _onTick(Duration elapsed) {
    final engine = _engine;
    if (engine == null) return;

    double dt = (elapsed - _lastElapsed).inMicroseconds / 1e6;
    _lastElapsed = elapsed;
    if (dt <= 0) dt = 1 / 60.0;
    dt = dt.clamp(0, 1 / 20.0);

    if (engine.status != eng.GameStatus.playing) return;

    // Controls: AI overrides manual
    bool T = thrust, L = left, R = right;
    if (aiPlay && _policy != null) {
      final (thrB, leftB, rightB, idx, probs) = _policy!.actWithIntent(
        lander: Lander.fromEngine(engine.lander),
        terrain: Terrain(engine.terrain),
        worldW: engine.cfg.worldW.toDouble(),
        worldH: engine.cfg.worldH.toDouble(),
        step: _frame,
      );
      T = thrB; L = leftB; R = rightB;
      _intentLabel = 'Intent #$idx';
      _intentProbs = probs;
    }

    engine.step(dt, eng.ControlInput(thrust: T, left: L, right: R, intentIdx: null));
    _frame++;
    setState(() {}); // just repaint; no heavy work here
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: LayoutBuilder(builder: (ctx, c) {
        final size = Size(c.maxWidth, c.maxHeight);
        _ensureEngine(size);
        final engine = _engine!;
        final lander = engine.lander;
        final terrain = engine.terrain;
        final status = engine.status;

        // Keep engine ray settings in sync with UI chips
        if (engine.rayCfg.rayCount != _rayCountUi ||
            engine.rayCfg.includeFloor != _rayIncludeFloor) {
          engine.rayCfg = engine.rayCfg.copyWith(
            rayCount: _rayCountUi,
            includeFloor: _rayIncludeFloor,
          );
        }

        return Stack(
          children: [
            // Canvas
            Positioned.fill(
              child: CustomPaint(
                painter: _GamePainter(
                  world: size,
                  lander: lander,
                  terrain: terrain,
                  status: status,
                  rays: _showRays ? engine.rays : const [],
                ),
              ),
            ),

            // HUD
            SafeArea(
              child: Padding(
                padding: const EdgeInsets.all(12.0),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Row(
                      children: [
                        _menuButton(),
                        const SizedBox(width: 8),
                        ElevatedButton.icon(
                          onPressed: _reset,
                          icon: const Icon(Icons.refresh),
                          label: const Text('Reset'),
                        ),
                        const Spacer(),
                        _aiToggleIcon(),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        _hudBox('Fuel', engine.lander.fuel.toStringAsFixed(0)),
                        _intentChip(_intentLabel, _intentProbs),
                      ],
                    ),
                    const SizedBox(height: 8),
                    // Ray controls
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      crossAxisAlignment: WrapCrossAlignment.center,
                      children: [
                        FilterChip(
                          label: const Text('Rays'),
                          selected: _showRays,
                          onSelected: (v) => setState(() => _showRays = v),
                        ),
                        FilterChip(
                          label: const Text('Floor as wall'),
                          selected: _rayIncludeFloor,
                          onSelected: (v) => setState(() => _rayIncludeFloor = v),
                        ),
                        const Text('Count:', style: TextStyle(fontSize: 12, color: Colors.white70)),
                        ChoiceChip(
                          label: const Text('90'),
                          selected: _rayCountUi == 90,
                          onSelected: (_) => setState(() => _rayCountUi = 90),
                        ),
                        ChoiceChip(
                          label: const Text('180'),
                          selected: _rayCountUi == 180,
                          onSelected: (_) => setState(() => _rayCountUi = 180),
                        ),
                        ChoiceChip(
                          label: const Text('360'),
                          selected: _rayCountUi == 360,
                          onSelected: (_) => setState(() => _rayCountUi = 360),
                        ),
                      ],
                    ),
                    if (_policy != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 6),
                        child: Text(_policyInfo,
                            style: const TextStyle(fontSize: 12, color: Colors.white70)),
                      ),

                    if (status != eng.GameStatus.playing)
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

            // Manual controls
            _buildControls(),

            // Status banner
            if (status == eng.GameStatus.landed || status == eng.GameStatus.crashed)
              Center(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.6),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.white24),
                  ),
                  child: Text(
                    status == eng.GameStatus.landed ? 'Touchdown! 🟢' : 'Crashed 💥',
                    style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                  ),
                ),
              ),
          ],
        );
      }),
    );
  }

  // ===== UI helpers =====

  Widget _aiToggleIcon() {
    final active = aiPlay;
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

  Widget _buildControls() {
    return Align(
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
    );
  }

  Widget _hudBox(String title, String value) {
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

  Widget _intentChip(String label, List<double>? probs) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white12,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.white24),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Intent', style: TextStyle(fontSize: 12, color: Colors.white70)),
          Text(label, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
          if (probs != null && probs.isNotEmpty)
            Text(
              'p:${probs.map((p) => p.toStringAsFixed(2)).join("/")}',
              style: const TextStyle(fontSize: 10, color: Colors.white54),
            ),
        ],
      ),
    );
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
}

/// ---------------------------------------------------------------------------
/// Painter — reads engine state and paints; NO physics, NO ray casting here.
/// ---------------------------------------------------------------------------
class _GamePainter extends CustomPainter {
  final Size world;
  final eng.LanderState lander;
  final eng.Terrain terrain;
  final eng.GameStatus status;
  final List<eng.RayHit> rays;

  _GamePainter({
    required this.world,
    required this.lander,
    required this.terrain,
    required this.status,
    required this.rays,
  });

  @override
  void paint(Canvas canvas, Size size) {
    _paintStars(canvas, size);
    _paintTerrain(canvas, size);
    _paintPad(canvas);
    _paintRays(canvas);
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
    final pts = terrain.ridge;
    if (pts.isEmpty) return;

    final path = Path()..moveTo(pts.first.x, pts.first.y);
    for (int i = 1; i < pts.length; i++) {
      path.lineTo(pts[i].x, pts[i].y);
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
    final ridge = Path()..moveTo(pts.first.x, pts.first.y);
    for (int i = 1; i < pts.length; i++) {
      ridge.lineTo(pts[i].x, pts[i].y);
    }
    canvas.drawPath(ridge, outline);

    // Visualize world bounds (for context of wall hits)
    final boundPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1
      ..color = Colors.white12;
    canvas.drawLine(const Offset(0, 0), Offset(size.width, 0), boundPaint); // ceiling
    canvas.drawLine(const Offset(0, 0), Offset(0, size.height), boundPaint); // left
    canvas.drawLine(Offset(size.width, 0), Offset(size.width, size.height), boundPaint); // right
  }

  void _paintPad(Canvas canvas) {
    final padRect = Rect.fromLTWH(
      terrain.padX1,
      terrain.padY - 4,
      (terrain.padX2 - terrain.padX1),
      8,
    );
    final paint = Paint()..color = status == eng.GameStatus.crashed ? Colors.red : Colors.greenAccent;
    canvas.drawRRect(RRect.fromRectAndRadius(padRect, const Radius.circular(4)), paint);
  }

  void _paintRays(Canvas canvas) {
    if (rays.isEmpty) return;

    final pTerrain = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1
      ..color = Colors.cyanAccent.withOpacity(0.65);
    final pWall = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1
      ..color = Colors.pinkAccent.withOpacity(0.55);
    final pPad = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..color = Colors.limeAccent.withOpacity(0.9);

    final origin = Offset(lander.pos.x, lander.pos.y);

    for (final h in rays) {
      final hit = Offset(h.p.x, h.p.y);
      final paint = switch (h.kind) {
        eng.RayHitKind.pad => pPad,
        eng.RayHitKind.terrain => pTerrain,
        eng.RayHitKind.wall => pWall,
      };
      canvas.drawLine(origin, hit, paint);
      final dot = Paint()..color = paint.color.withOpacity(0.95);
      canvas.drawCircle(hit, 1.8, dot);
    }
  }

  void _paintLander(Canvas canvas) {
    final c = math.cos(lander.angle);
    final s = math.sin(lander.angle);

    const halfWidth = 14.0;
    const halfHeight = 18.0;

    final body = [
      const Offset(0, -halfHeight),
      const Offset(-halfWidth, halfHeight),
      const Offset(halfWidth, halfHeight),
    ];
    Offset rot(Offset v) => Offset(c * v.dx - s * v.dy, s * v.dx + c * v.dy);

    final pos = Offset(lander.pos.x, lander.pos.y);
    final path = Path();
    final p0 = pos + rot(body[0]);
    final p1 = pos + rot(body[1]);
    final p2 = pos + rot(body[2]);
    path.moveTo(p0.dx, p0.dy);
    path.lineTo(p1.dx, p1.dy);
    path.lineTo(p2.dx, p2.dy);
    path.close();

    final shipPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = Colors.white.withOpacity(0.9);
    canvas.drawPath(path, shipPaint);

    final cockpit = Paint()..color = Colors.lightBlueAccent.withOpacity(0.9);
    canvas.drawCircle(pos + rot(const Offset(0, -halfHeight + 6)), 4, cockpit);

    final legs = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..color = Colors.white70;
    canvas.drawLine(
      pos + rot(const Offset(-halfWidth * 0.6, halfHeight * 0.6)),
      pos + rot(const Offset(-halfWidth, halfHeight)),
      legs,
    );
    canvas.drawLine(
      pos + rot(const Offset(halfWidth * 0.6, halfHeight * 0.6)),
      pos + rot(const Offset(halfWidth, halfHeight)),
      legs,
    );
  }

  @override
  bool shouldRepaint(covariant _GamePainter old) {
    return old.world != world ||
        old.lander != lander ||
        old.terrain != terrain ||
        old.status != status ||
        old.rays != rays;
  }
}
