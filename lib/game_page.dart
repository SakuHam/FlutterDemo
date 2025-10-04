// lib/game_page.dart
import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart' show Ticker;
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';

// Engine
import 'ai/cavern_detector.dart';
import 'engine/types.dart' as et;
import 'engine/game_engine.dart';
import 'engine/raycast.dart';
import 'engine/polygon_carver.dart';

// Runtime policy (AI)
import 'ai/runtime_policy.dart' hide kIntentNames;

// Potential field + plan bus
import 'ai/potential_field.dart';
import 'ai/plan_bus.dart';

// Live trainer isolate
import 'ai/live_trainer_isolate.dart';

// Intent names (for HUD tooltip)
import 'ai/agent.dart' show kIntentNames;

/// Simple UI particle for exhaust/smoke
class Particle {
  Offset pos;
  Offset vel;
  double life; // 0..1
  Particle({required this.pos, required this.vel, required this.life});
}

// Added: aiVision mode
enum DebugVisMode { rays, potential, velocity, aiVision, none }

/// Progress phases and simple model for the live trainer HUD
enum ProgressPhase { idle, starting, training, evaluating, stopped }

class _ProgressModel {
  ProgressPhase phase = ProgressPhase.idle;
  int it = 0;
  int total = 0;
  double landPct = 0;
  double meanCost = 0;
  double meanSteps = 0;

  void reset() {
    phase = ProgressPhase.idle;
    it = total = 0;
    landPct = meanCost = meanSteps = 0;
  }
}

/// Snapshot of a raycast frame for fading trails
class RayFrame {
  final Offset origin; // lander position at capture time
  final double angle;  // lander angle at capture time (for AI Vision)
  final List<RayHit> hits; // absolute world-space hit points
  RayFrame({required this.origin, required this.angle, required this.hits});
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
  RuntimeTwoStagePolicy? _policy;   // loaded async from assets or live file
  bool get _aiReady => _policy != null;

  // HUD: last AI intent/probs (optional)
  int? _lastIntentIdx;
  List<double> _lastIntentProbs = const [];

  // === NEW: per-frame vectors to draw at the lander ===
  Offset? _vecPF;      // raw PF suggestion
  Offset? _vecPolicy;  // policy's preferred velocity (shaped)

  // ===== NEW: ray alignment toggle (FWD vs WORLD) =====
  bool _forwardAligned = true;

  // ===== Plan bus (PF path preview + widths) =====
  StreamSubscription<PlanEvent>? _planSub;
  List<Offset> _planPts = const [];
  List<double>? _planWidths;
  int _planVersion = 0;

  // ===== Live training isolate & progress =====
  Isolate? _trainerIso;
  SendPort? _trainerSend;
  ReceivePort? _trainerRecv;
  final _progress = _ProgressModel();

  // Writable path for live policy + seeding flag
  String? _livePath;
  bool _liveSeeded = false;

  // Live trainer config (simple knob for iterations)
  int _liveIters = 120000; // adjust as desired

  // ===== Cavern detection (for HUD + painter) =====
  final CavernDetector _detector = CavernDetector(
    jumpThresh: 220.0,            // baseline; we adapt from this at runtime
    voidBoost: 1.20,
    minVoidSpanRad: math.pi / 48, // ~3.75°
    minSpanSamples: 3,
  );
  int _cavernCount = 0;
  int _rayCount = 0;       // rays in AI Vision (all directions)
  double _maxEdgeDrDth = 0.0;
  List<CavernHypothesis> _caverns = const []; // computed once per frame

  // ===== NEW: ray history (fading trail) =====
  static const int _rayHistLen = 10;
  final List<RayFrame?> _rayHistory = List<RayFrame?>.filled(_rayHistLen, null, growable: false);
  int _rayHistHead = 0; // index to overwrite next

  void _pushRayHistorySnapshot(GameEngine e) {
    // Only snapshot when the ray-based views are active
    if (!(_visMode == DebugVisMode.rays || _visMode == DebugVisMode.aiVision)) return;
    final hits = e.rays;
    if (hits.isEmpty) return;
    final frame = RayFrame(
      origin: Offset(e.lander.pos.x, e.lander.pos.y),
      angle: e.lander.angle,
      hits: List<RayHit>.from(hits, growable: false),
    );
    _rayHistory[_rayHistHead] = frame;
    _rayHistHead = (_rayHistHead + 1) % _rayHistLen;
  }

  List<RayFrame> _orderedRayHistoryNewestFirst() {
    // Return newest->oldest, skipping nulls
    final out = <RayFrame>[];
    for (int k = 1; k <= _rayHistLen; k++) {
      final idx = (_rayHistHead - k) % _rayHistLen;
      final i = idx < 0 ? idx + _rayHistLen : idx;
      final f = _rayHistory[i];
      if (f != null) out.add(f);
    }
    return out;
  }

  @override
  void initState() {
    super.initState();
    _ticker = createTicker(_onTick)..start();
    _loadPolicy();           // load baseline runtime policy (from assets)
    _initLivePathAndSeed();  // prepare a writable live policy file

    // Subscribe to plan bus for PF/fallback plans
    _planSub = PlanBus.instance.stream.listen((e) {
      setState(() {
        _planPts = e.points;
        _planWidths = e.widths;
        _planVersion = e.version;
      });
    });
  }

  Future<void> _initLivePathAndSeed() async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final path = '${dir.path}/policy_live.json';
      _livePath = path;

      final f = File(path);
      if (!await f.exists()) {
        try {
          final txt = await rootBundle.loadString('assets/ai/policy.json');
          await f.writeAsString(txt);
          _liveSeeded = true;
          debugPrint('Seeded live policy at $path');
        } catch (e) {
          debugPrint('Failed to seed live policy: $e');
        }
      } else {
        _liveSeeded = true;
      }
      if (mounted) setState(() {});
    } catch (e) {
      debugPrint('initLivePath failed: $e');
    }
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

      p.usePadAlignPlanner();
      _applyPolicyPhysicsToEngine(p);

      setState(() => _policy = p);
      _toast('AI model loaded');
    } catch (e) {
      debugPrint('Failed to load AI policy: $e');
      _toast('AI model failed to load');
    }
  }

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
    _ticker.dispose();
    _planSub?.cancel();
    _stopLiveTraining();
    super.dispose();
  }

  void _ensureEngine(Size size) {
    if (_worldSize == null ||
        (_worldSize!.width != size.width || _worldSize!.height != size.height) ||
        _engine == null) {
      _worldSize = size;

      final cfg = et.EngineConfig(
        worldW: size.width,
        worldH: size.height,
        t: et.Tunables(),
      );

      _engine = GameEngine(cfg);
      _engine!.rayCfg = RayConfig(
        rayCount: 180,
        includeFloor: false,
        forwardAligned: _forwardAligned, // ship-forward aligned
      );

      final p = _policy;
      if (p != null) _applyPolicyPhysicsToEngine(p);

      _rebuildPF();
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
    _policy?.setPotentialField(_pf);
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
    _planWidths = null;

    // Clear ray history too
    for (var i = 0; i < _rayHistory.length; i++) {
      _rayHistory[i] = null;
    }
    _rayHistHead = 0;

    _rebuildPF();
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
      case DebugVisMode.aiVision: return 'AI Vision';
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

  void _computeBestVelVectors(GameEngine e, PotentialField pf) {
    final L = e.lander;
    final x = L.pos.x.toDouble();
    final y = L.pos.y.toDouble();
    final vx = L.vel.x.toDouble();
    final vy = L.vel.y.toDouble();
    final W = e.cfg.worldW.toDouble();

    final base = pf.suggestVelocity(
      x, y,
      vMinClose: 8.0,
      vMaxFar: 90.0,
      alpha: 1.2,
      clampSpeed: 9999.0,
    );
    _vecPF = Offset(base.vx, base.vy);

    final padCx = e.terrain.padCenter.toDouble();
    final dxAbs = (x - padCx).abs();
    final gy = e.terrain.heightAt(x);
    final h = (gy - y).toDouble().clamp(0.0, 1000.0);

    final tightX = 0.10 * W;
    final ph = math.exp(- (h * h) / (140.0 * 140.0 + 1e-6));
    final px = math.exp(- (dxAbs * dxAbs) / (tightX * tightX + 1e-6));
    final prox = (px * ph).clamp(0.0, 1.0);

    final vMinTouchdown = 2.0;
    final flareLat = (1.0 - 0.90 * prox);
    final flareVer = (1.0 - 0.70 * prox);

    double svx = base.vx * flareLat;
    double svy = base.vy * flareVer;

    final magNow = (math.sqrt(svx * svx + svy * svy) + 1e-9);
    final magTarget = ((1.0 - prox) * magNow + prox * vMinTouchdown).clamp(0.0, magNow);
    final kMag = (magTarget / magNow).clamp(0.0, 1.0);
    svx *= kMag; svy *= kMag;

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
    if (engine.status != et.GameStatus.playing) return;

    final pf = _pf;
    if (pf != null && _visMode == DebugVisMode.velocity) {
      _computeBestVelVectors(engine, pf);
    } else {
      _vecPF = _vecPolicy = null;
    }

    if (_aiPlay && _aiReady) {
      final lander = engine.lander;
      final terr = engine.terrain;

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
      et.ControlInput(
        thrust: _thrust,
        left: _left,
        right: _right,
        sideLeft: _sideLeft,
        sideRight: _sideRight,
        downThrust: _downThrust,
      ),
    );

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

    // ---- HUD & cavern detection (AI Vision only) ----
    if (_visMode == DebugVisMode.aiVision) {
      final rays = engine.rays;
      final L = engine.lander;

      // Body-frame transform for crude edge metric (use ALL rays; draw both hemispheres)
      final cs = math.cos(-L.angle), sn = math.sin(-L.angle);
      final bodyPts = <Offset>[];
      for (final h in rays) {
        final dx = h.p.x - L.pos.x, dy = h.p.y - L.pos.y;
        final lx = cs * dx - sn * dy, ly = sn * dx + cs * dy;
        bodyPts.add(Offset(lx, ly));
      }
      _rayCount = bodyPts.length;

      // crude |dr/dθ| peak over sorted angles (all rays)
      double maxEdge = 0.0;
      if (bodyPts.length >= 5) {
        bodyPts.sort((a, b) =>
            math.atan2(a.dx, -a.dy).compareTo(math.atan2(b.dx, -b.dy)));
        for (int i = 1; i < bodyPts.length - 1; i++) {
          final r0 = bodyPts[i - 1].distance;
          final r2 = bodyPts[i + 1].distance;
          final th0 = math.atan2(bodyPts[i - 1].dx, -bodyPts[i - 1].dy);
          final th2 = math.atan2(bodyPts[i + 1].dx, -bodyPts[i + 1].dy);
          final dth = (th2 - th0).abs();
          if (dth < 1e-4) continue;
          final dr = (r2 - r0).abs() / dth;
          if (dr > maxEdge) maxEdge = dr;
        }
      }
      _maxEdgeDrDth = maxEdge;

      // --- Build forward/back subsets (for robust detection) ---
      /*
      final fwd = <RayHit>[];
      final back = <RayHit>[];
      for (final h in rays) {
        final dx = h.p.x - L.pos.x, dy = h.p.y - L.pos.y;
        final lx = cs * dx - sn * dy, ly = sn * dx + cs * dy;
        if (ly < 0) {
          fwd.add(h);
        } else {
          back.add(h);
        }
      }

       */

      // --- Adaptive jump threshold based on observed contrast ---
      final base = _detector;
      final effJump = (_maxEdgeDrDth.isFinite && _maxEdgeDrDth > 0)
          ? math.min(base.jumpThresh, math.max(60.0, 0.20 * _maxEdgeDrDth)) // was 0.35 & 100
          : base.jumpThresh;

      // Use a detector instance with the adapted jump (no other changes)
      final adaptive = CavernDetector(
        jumpThresh: effJump,
        voidBoost: base.voidBoost,
        minVoidSpanRad: base.minVoidSpanRad,
        minSpanSamples: base.minSpanSamples,
      );

      List<CavernHypothesis> hyps = adaptive.detect(rays: rays, lander: L);

      _caverns = hyps;
      _cavernCount = hyps.length;
    } else {
      _caverns = const [];
      _cavernCount = 0;
      _rayCount = 0;
      _maxEdgeDrDth = 0.0;
    }

    // ---- Capture ray history AFTER rays are updated this step ----
    _pushRayHistorySnapshot(engine);

    if (info.terminal) {
      // engine already set status
    }

    setState(() {});
  }

  void _maybeEmitMainFlame(et.LanderState lander) {
    final engine = _engine!;
    if (_thrust && lander.fuel > 0 && engine.status == et.GameStatus.playing) {
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

  void _maybeEmitSideRCS(et.LanderState lander) {
    if ((!_sideLeft && !_sideRight) || lander.fuel <= 0) return;

    final c = math.cos(lander.angle);
    final s = math.sin(lander.angle);
    const halfH = 12.0;
    const halfW = 14.0;

    final upBody = Offset(s, -c);
    final rightBody = Offset(c, s);
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

  void _maybeEmitDownwardFlame(et.LanderState lander) {
    if (!_downThrust || lander.fuel <= 0) return;

    const halfH = 6.0;

    final port = Offset(lander.pos.x, lander.pos.y) + const Offset(0, halfH);
    final rnd = math.Random();
    for (int i = 0; i < 3; i++) {
      final spread = (rnd.nextDouble() - 0.5) * 0.4;
      final dir = Offset(spread, 1.0);
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
    _terrainDirty = true;
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

  // ---------- Live training isolate control ----------
  Future<void> _startLiveTraining() async {
    if (_trainerIso != null) return;
    if (_livePath == null) {
      await _initLivePathAndSeed();
      if (_livePath == null) {
        _toast('No writable path for live policy');
        return;
      }
    }

    _progress.phase = ProgressPhase.starting;
    setState(() {});

    _trainerRecv = ReceivePort();
    final errPort  = ReceivePort();
    final exitPort = ReceivePort();

    _trainerRecv!.listen((msg) async {
      if (msg is SendPort) {
        _trainerSend = msg;
        _trainerSend!.send({
          'cmd': 'start',
          'outPath': _livePath,
          'warmStart': null,
          'iters': _liveIters,
          'seed': 7,
        });
        return;
      }

      if (msg is Map) {
        final type = msg['type'] as String? ?? '';
        switch (type) {
          case 'status': {
            final ph = (msg['phase'] as String?) ?? 'idle';
            _progress.phase = ph == 'stopped'
                ? ProgressPhase.stopped
                : ph.contains('warm') || ph == 'starting'
                ? ProgressPhase.starting
                : ProgressPhase.training;
            _progress.it = (msg['iters'] is num) ? (msg['iters'] as num).toInt() : _progress.it;
            if (mounted) setState(() {});
            break;
          }
          case 'progress': {
            final p = (msg['phase'] as String?) ?? '';
            final isEval = p.contains('evaluating');
            final isCurric = p.contains('curriculum');

            _progress.phase = isEval ? ProgressPhase.evaluating : ProgressPhase.training;

            _progress.it = (msg['it'] ?? 0) as int;
            _progress.total = (msg['total'] ?? 0) as int;

            if (isEval) {
              _progress.landPct   = ((msg['landPct']   ?? 0.0) as num).toDouble();
              _progress.meanCost  = ((msg['meanCost']  ?? 0.0) as num).toDouble();
              _progress.meanSteps = ((msg['meanSteps'] ?? 0.0) as num).toDouble();
            } else if (isCurric) {
              final acc = ((msg['accWindow'] ?? 0.0) as num).toDouble();
              final dx  = ((msg['dxPerSec']  ?? 0.0) as num).toDouble();
              _progress.landPct = (acc * 100.0).clamp(0.0, 100.0);
              _progress.meanCost = -dx;
              _progress.meanSteps = dx.abs();
            } else {
              _progress.landPct   = ((msg['landPct']   ?? _progress.landPct) as num).toDouble();
              _progress.meanCost  = ((msg['meanCost']  ?? _progress.meanCost) as num).toDouble();
              _progress.meanSteps = ((msg['meanSteps'] ?? _progress.meanSteps) as num).toDouble();
            }
            setState(() {});
            break;
          }
          case 'saved': {
            final path = (msg['path'] as String?) ?? _livePath!;
            await _hotReloadPolicy(path);
            _toast('Live policy updated');
            break;
          }
          case 'warn':  debugPrint('trainer warn: ${msg['message']}'); break;
          case 'hint':  debugPrint('trainer hint: ${msg['message']}'); break;
          case 'error': debugPrint('trainer error: ${msg['message']}'); break;
        }
      }
    });

    errPort.listen((e) => debugPrint('trainer isolate error: $e'));
    exitPort.listen((_) => debugPrint('trainer isolate exit'));

    _trainerIso = await Isolate.spawn(
      liveTrainerMain,
      _trainerRecv!.sendPort,
      onError: errPort.sendPort,
      onExit:  exitPort.sendPort,
      errorsAreFatal: false,
    );
  }

  Future<void> _stopLiveTraining() async {
    final sp = _trainerSend;
    if (sp != null) sp.send(const LiveTrainStop());
    _trainerRecv?.close();
    _trainerIso?.kill(priority: Isolate.immediate);
    _trainerIso = null;
    _trainerSend = null;
    _trainerRecv = null;
    _progress.reset();
    setState(() {});
  }

  void _requestTrainerSave() {
    final sp = _trainerSend;
    if (sp == null) return;
    sp.send({'cmd': 'save'});
  }

  Future<void> _hotReloadPolicy(String path) async {
    try {
      final txt = await File(path).readAsString();
      final newP = RuntimeTwoStagePolicy.fromJson(txt, planHold: 2);

      newP.setStochasticPlanner(true);
      newP.setIntentTemperature(1.8);
      newP.usePadAlignPlanner();
      _applyPolicyPhysicsToEngine(newP);

      setState(() {
        _policy = newP;
        _policy!.resetPlanner();
      });
      _toast('Live policy reloaded');
    } catch (e) {
      debugPrint('Hot reload failed: $e');
      _toast('Hot reload failed');
    }
  }

  bool get _canReload => _livePath != null && File(_livePath!).existsSync();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: LayoutBuilder(
        builder: (context, constraints) {
          _ensureEngine(Size(constraints.maxWidth, constraints.maxHeight));
          final engine = _engine!;
          final status = engine.status;

          final rayHistory = _orderedRayHistoryNewestFirst();

          return Stack(
            children: [
              Positioned.fill(
                child: GestureDetector(
                  behavior: HitTestBehavior.opaque,
                  onTapDown: (d) => _carveAt(d.localPosition),
                  onTapUp: (_) {
                    if (_terrainDirty) { _rebuildPF(); _terrainDirty = false; }
                  },
                  onPanStart: (d) => _carveAt(d.localPosition),
                  onPanUpdate: (d) => _carveAt(d.localPosition),
                  onPanEnd: (_) {
                    if (_terrainDirty) { _rebuildPF(); _terrainDirty = false; }
                  },
                  onPanCancel: () {
                    if (_terrainDirty) { _rebuildPF(); _terrainDirty = false; }
                  },
                  child: CustomPaint(
                    painter: GamePainter(
                      lander: engine.lander,
                      terrain: engine.terrain,
                      thrusting: _thrust && engine.lander.fuel > 0 && status == et.GameStatus.playing,
                      status: status,
                      particles: _particles,
                      rays: (_visMode == DebugVisMode.rays || _visMode == DebugVisMode.aiVision)
                          ? engine.rays
                          : const [],
                      pf: _pf,
                      visMode: _visMode,
                      vecPF: _vecPF,
                      vecPolicy: _vecPolicy,
                      showForwardAxis: _visMode == DebugVisMode.rays || _visMode == DebugVisMode.aiVision,
                      planPts: _planPts,
                      planWidths: _planWidths,
                      caverns: _caverns, // NEW: pass computed caverns to painter
                      rayHistory: rayHistory, // NEW: fading trail data
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
                          Tooltip(
                            message: 'Tap to cycle view • Long-press to rebuild field',
                            child: FilterChip(
                              label: Text(_visModeLabel()),
                              selected: _visMode != DebugVisMode.none,
                              onSelected: (_) => _cycleVis(),
                            ),
                          ),
                          const SizedBox(width: 8),
                          ElevatedButton.icon(
                            onPressed: _reset,
                            icon: const Icon(Icons.refresh),
                            label: const Text(''),
                          ),
                        ],
                      ),

                      const SizedBox(height: 10),

                      // Stats row
                      Row(
                        children: [
                          _hudBox(title: 'Fuel', value: engine.lander.fuel.toStringAsFixed(0)),
                          const SizedBox(width: 8),
                          if (_visMode == DebugVisMode.aiVision)
                            _hudBox(title: 'Caverns', value: '$_cavernCount'),
                          if (_visMode == DebugVisMode.aiVision) ...[
                            const SizedBox(width: 8),
                            _hudBox(title: 'Rays', value: '$_rayCount'),
                            const SizedBox(width: 8),
                            _hudBox(title: 'Max|dr/dθ|', value: _maxEdgeDrDth.toStringAsFixed(0)),
                          ],
                          if (_aiReady)
                            Expanded(
                              child: Align(
                                alignment: Alignment.centerRight,
                                child: Tooltip(
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
                                      mainAxisSize: MainAxisSize.min,
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
                              ),
                            ),
                          const SizedBox(width: 8),
                          Tooltip(
                            message: 'Live training iterations',
                            child: SizedBox(
                              width: 110,
                              child: Row(
                                children: [
                                  const Text('iters:', style: TextStyle(color: Colors.white70)),
                                  const SizedBox(width: 6),
                                  Expanded(
                                    child: TextFormField(
                                      initialValue: _liveIters.toString(),
                                      style: const TextStyle(color: Colors.white),
                                      decoration: const InputDecoration(
                                        isDense: true,
                                        contentPadding: EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                                        border: OutlineInputBorder(),
                                      ),
                                      keyboardType: TextInputType.number,
                                      onFieldSubmitted: (v) {
                                        final n = int.tryParse(v);
                                        if (n != null && n > 0) setState(() => _liveIters = n);
                                      },
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ],
                      ),

                      const SizedBox(height: 8),

                      // Live training HUD panel
                      _ProgressPanel(model: _progress),

                      const SizedBox(height: 8),

                      // Trainer buttons
                      Row(
                        children: [
                          ElevatedButton(
                            onPressed: (_trainerIso == null) ? _startLiveTraining : _stopLiveTraining,
                            child: Text(_trainerIso == null ? 'Train Live' : 'Stop Train'),
                          ),
                          const SizedBox(width: 8),
                          ElevatedButton(
                            onPressed: (_trainerIso != null) ? _requestTrainerSave : null,
                            child: const Text('Save'),
                          ),
                          const SizedBox(width: 8),
                          ElevatedButton(
                            onPressed: _canReload ? () => _hotReloadPolicy(_livePath!) : null,
                            child: const Text('Reload'),
                          ),
                        ],
                      ),

                      const SizedBox(height: 8),

                      if (status != et.GameStatus.playing)
                        Center(
                          child: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                            decoration: BoxDecoration(
                              color: Colors.black.withOpacity(0.6),
                              borderRadius: BorderRadius.circular(12),
                              border: Border.all(color: Colors.white24),
                            ),
                            child: Text(
                              status == et.GameStatus.landed ? 'Touchdown! 🟢' : 'Crashed 💥',
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
    final disabled = _engine?.status != et.GameStatus.playing || _aiPlay;
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
      onPointerUp:   (_) => onChanged(false),
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

/// Compact panel to visualize live training progress.
class _ProgressPanel extends StatelessWidget {
  final _ProgressModel model;
  const _ProgressPanel({required this.model});

  String _phaseLabel(ProgressPhase p) {
    switch (p) {
      case ProgressPhase.idle: return 'Idle';
      case ProgressPhase.starting: return 'Starting';
      case ProgressPhase.training: return 'Training';
      case ProgressPhase.evaluating: return 'Evaluating';
      case ProgressPhase.stopped: return 'Stopped';
    }
  }

  @override
  Widget build(BuildContext context) {
    final pct = (model.total > 0) ? (model.it / model.total).clamp(0.0, 1.0) : 0.0;
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.45),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white24),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.school, size: 16, color: Colors.white70),
              const SizedBox(width: 6),
              Text('Live Training • ${_phaseLabel(model.phase)}',
                  style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
              const Spacer(),
              Text(model.total > 0 ? '${model.it}/${model.total}' : '',
                  style: const TextStyle(color: Colors.white70)),
            ],
          ),
          const SizedBox(height: 8),
          _bar(context, pct, labelLeft: 'Iter', labelRight: '${(pct * 100).toStringAsFixed(0)}%'),
          const SizedBox(height: 8),
          _bar(context, (model.landPct / 100).clamp(0.0, 1.0),
              labelLeft: 'Land%', labelRight: '${model.landPct.toStringAsFixed(1)}%'),
          const SizedBox(height: 8),
          Row(
            children: [
              _stat('Mean Cost', model.meanCost.toStringAsFixed(2)),
              const SizedBox(width: 12),
              _stat('Steps', model.meanSteps.toStringAsFixed(1)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _bar(BuildContext context, double t, {required String labelLeft, required String labelRight}) {
    return Column(
      children: [
        Row(
          children: [
            Text(labelLeft, style: const TextStyle(color: Colors.white70, fontSize: 12)),
            const Spacer(),
            Text(labelRight, style: const TextStyle(color: Colors.white70, fontSize: 12)),
          ],
        ),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: LinearProgressIndicator(
            value: t,
            minHeight: 8,
            backgroundColor: Colors.white12,
            valueColor: const AlwaysStoppedAnimation<Color>(Colors.lightBlueAccent),
          ),
        ),
      ],
    );
  }

  Widget _stat(String k, String v) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white10,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white24),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text('$k: ', style: const TextStyle(color: Colors.white70, fontSize: 12)),
          Text(v, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }
}

// Helper for AI Vision drawing (body-frame hit)
class _BFHit {
  _BFHit(this.lx, this.ly, this.kind, this.isForward);
  final double lx, ly;
  final RayHitKind kind;
  final bool isForward;
}

class GamePainter extends CustomPainter {
  final et.LanderState lander;
  final et.Terrain terrain;
  final bool thrusting;
  final et.GameStatus status;
  final List<Particle> particles;
  final List<RayHit> rays;

  // Potential field visualization
  final PotentialField? pf;
  final DebugVisMode visMode;

  // NEW: vectors to overlay at the lander
  final Offset? vecPF;
  final Offset? vecPolicy;

  // NEW: painter hint to draw forward axis when looking at rays/vision
  final bool showForwardAxis;

  // Plan overlay (centerline + band)
  final List<Offset> planPts;
  final List<double>? planWidths;

  // Caverns to draw (computed in state)
  final List<CavernHypothesis> caverns;

  // NEW: ray history (newest->oldest, up to 10)
  final List<RayFrame> rayHistory;

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
    required this.showForwardAxis,
    required this.planPts,
    required this.planWidths,
    required this.caverns,
    required this.rayHistory,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (visMode != DebugVisMode.aiVision) {
      _paintStars(canvas, size);
    }

    // Hide ground polygon in AI Vision mode
    if (visMode != DebugVisMode.aiVision) {
      _paintTerrainPoly(canvas, size);
      _paintEdgesOverlay(canvas);
    }

    // Overlays
    if (visMode == DebugVisMode.potential && pf != null) {
      _paintPotentialHeat(canvas, pf!, heatDownsample: 3, alpha: 130);
    } else if (visMode == DebugVisMode.velocity && pf != null) {
      _paintPotentialVectors(canvas, pf!, stride: 8);
    } else if (visMode == DebugVisMode.rays) {
      _paintRaysWithHistory(canvas);
    } else if (visMode == DebugVisMode.aiVision) {
      _paintAIVisionWithHistory(canvas, size);
    }

    // Plan band + centerline
    _paintPlan(canvas);

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

    final ground = Paint()..color = const Color(0xFFD8D8D8);
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
      ..color = (status == et.GameStatus.crashed ? Colors.red : Colors.greenAccent);

    for (final e in edges) {
      final p1 = Offset(e.a.x, e.a.y);
      final p2 = Offset(e.b.x, e.b.y);
      if (e.kind == et.PolyEdgeKind.pad) {
        canvas.drawLine(p1, p2, padPaint);
      } else {
        canvas.drawLine(p1, p2, ridgePaint);
      }
    }
  }

  // ---------- Rays (world frame) with fading history ----------
  void _paintRaysWithHistory(Canvas canvas) {
    // Draw history first (oldest faintest), then current rays on top
    if (rayHistory.isNotEmpty) {
      // Iterate from oldest to newest so later ones overlay earlier
      for (int idx = rayHistory.length - 1; idx >= 0; idx--) {
        final rf = rayHistory[idx];
        final t = (idx + 1) / (rayHistory.length + 1); // 0..1
        final alphaScale = 0.08 + 0.42 * (1.0 - t);    // older ~0.08, newer ~0.50
        _paintRays(canvas,
          origin: rf.origin,
          hits: rf.hits,
          opacityScale: alphaScale,
          hitDotRadius: 1.2,
          lineWidthPad: 1.2,
          lineWidthTerr: 0.9,
          lineWidthWall: 0.8,
        );
      }
    }

    // Now draw the current frame bold
    _paintRays(canvas,
      origin: Offset(lander.pos.x, lander.pos.y),
      hits: rays,
      opacityScale: 1.0,
      hitDotRadius: 1.7,
      lineWidthPad: 1.6,
      lineWidthTerr: 1.2,
      lineWidthWall: 1.0,
    );
  }

  void _paintRays(
      Canvas canvas, {
        required Offset origin,
        required List<RayHit> hits,
        required double opacityScale,
        required double hitDotRadius,
        required double lineWidthPad,
        required double lineWidthTerr,
        required double lineWidthWall,
      }) {
    if (hits.isEmpty) return;

    final wallPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = lineWidthWall
      ..color = Colors.blueAccent.withOpacity(0.55 * opacityScale);

    final terrPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = lineWidthTerr
      ..color = Colors.red.withOpacity(0.80 * opacityScale);

    final padPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = lineWidthPad
      ..color = Colors.greenAccent.withOpacity(0.95 * opacityScale);

    for (final h in hits) {
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
            ? Colors.greenAccent.withOpacity(0.95 * opacityScale)
            : (h.kind == RayHitKind.wall
            ? Colors.blueAccent.withOpacity(0.85 * opacityScale)
            : Colors.red.withOpacity(0.85 * opacityScale));
      canvas.drawCircle(end, hitDotRadius, dotPaint);
    }
  }

  List<_BFHit> _subsetHemisphere(List<_BFHit> all, {required bool forward}) =>
      all.where((h) => h.isForward == forward).toList(growable: false);

// ---------- AI Vision (ship-local) with world-aligned history ----------
  void _paintAIVisionWithHistory(Canvas canvas, Size size) {
    // Draw the current panel as before (uses current origin+angle)
    _paintAIVision(canvas, size);

    if (rayHistory.isEmpty) return;

    // Draw history points reprojected from WORLD -> CURRENT ship-local
    // (so they are world-aligned, not fwd-aligned per-capture).
    for (int idx = rayHistory.length - 1; idx >= 0; idx--) {
      final rf = rayHistory[idx];
      final alpha = (0.08 + 0.42 * (1.0 - (idx + 1) / (rayHistory.length + 1))).clamp(0.05, 0.55);
      _paintAIVisionGhostWorldAligned(canvas, rf, alpha: alpha);
    }
  }

// Ghost overlay for a historical frame, but WORLD-aligned:
// project historical world hits into the CURRENT ship-local panel.
  void _paintAIVisionGhostWorldAligned(Canvas canvas, RayFrame rf, {required double alpha}) {
    if (rf.hits.isEmpty) return;

    // Panel is already translated+rotated in _paintAIVision, so here we
    // use the SAME transform: translate to CURRENT origin and rotate by -CURRENT angle.
    // Easiest way: just draw in that space directly by converting world -> current-local.
    final currPos = Offset(lander.pos.x, lander.pos.y);
    final c = math.cos(-lander.angle), s = math.sin(-lander.angle);

    // Faint guides and colored dots
    final guide = Paint()..color = Colors.white.withOpacity(0.25 * alpha);
    final terrPt = Paint()..color = const Color(0xFFFF5050).withOpacity(0.70 * alpha);
    final padPt  = Paint()..color = const Color(0xFF52E57D).withOpacity(0.80 * alpha);
    final wallPt = Paint()..color = const Color(0xFF58A8FF).withOpacity(0.75 * alpha);

    // We must mirror the panel transform here:
    canvas.save();
    canvas.translate(currPos.dx, currPos.dy);
    canvas.rotate(-lander.angle);

    for (final h in rf.hits) {
      final dx = h.p.x - currPos.dx;
      final dy = h.p.y - currPos.dy;
      final lx = c * dx - s * dy;
      final ly = s * dx + c * dy;
      final pos = Offset(lx, ly);
      canvas.drawCircle(pos, 1.5, guide);
      final paint = switch (h.kind) {
        RayHitKind.terrain => terrPt,
        RayHitKind.pad     => padPt,
        RayHitKind.wall    => wallPt,
      };
      canvas.drawCircle(pos, 2.8, paint);
    }

    canvas.restore();
  }

  // The original AI Vision panel for the CURRENT frame
  void _paintAIVision(Canvas canvas, Size size) {
    if (rays.isEmpty) return;

    final center = Offset(lander.pos.x, lander.pos.y);
    canvas.save();
    canvas.translate(center.dx, center.dy);
    canvas.rotate(-lander.angle);

    // Dark backdrop for readability
    canvas.drawRect(
      const Rect.fromLTWH(-650, -650, 1300, 700),
      Paint()..color = const Color(0xEE0C0C0C),
    );

    // Subtle local grid (helps scale perception)
    _drawLocalGrid(canvas);

    // FOV wedge (just a hint; not a mask)
    final fov = _estimateFovRadians();
    final wedge = Path()
      ..moveTo(0, 0)
      ..lineTo(500 * math.sin(-fov / 2), -500 * math.cos(-fov / 2))
      ..lineTo(500 * math.sin(fov / 2), -500 * math.cos(fov / 2))
      ..close();
    canvas.drawPath(
      wedge,
      Paint()
        ..style = PaintingStyle.fill
        ..color = const Color(0x2239C5FF),
    );

    // Build 360° body-frame hits (draw ALL points)
    final c = math.cos(-lander.angle), s = math.sin(-lander.angle);
    final hitsBF = <_BFHit>[];
    for (final h in rays) {
      final dx = h.p.x - lander.pos.x;
      final dy = h.p.y - lander.pos.y;
      final lx = c * dx - s * dy;
      final ly = s * dx + c * dy;
      final isFwd = (ly < 0); // forward points have -Y in this rotated canvas
      hitsBF.add(_BFHit(lx, ly, h.kind, isFwd));
    }

    // --- Draw cavern hypotheses (computed in state) ---
    for (final h in caverns) {
      final radius = h.depth * 0.30;
      
      final center = Offset(h.centroidLocal.x, h.centroidLocal.y);

      final base = Color.lerp(const Color(0xFF00E5FF), const Color(0xFFFF4081), (1.0 - h.score))!;
      final fill = Paint()
        ..style = PaintingStyle.fill
        ..color = base.withOpacity(0.18 + 0.35 * h.score); // better = more visible
      final stroke = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0
        ..color = base.withOpacity(0.55 + 0.35 * h.score);

      canvas.drawCircle(center, radius, fill);
      canvas.drawCircle(center, radius, stroke);
      canvas.drawCircle(center, 3.5, Paint()..color = stroke.color); // centroid dot
    }

    // Draw all points: forward bright, back dim; plus small grey guide dot
    final guide = Paint()..color = const Color(0x44FFFFFF);

    final terrPtF = Paint()..color = const Color(0xFFFF5050);
    final terrPtB = Paint()..color = const Color(0x66FF5050);
    final padPtF  = Paint()..color = const Color(0xFF52E57D);
    final padPtB  = Paint()..color = const Color(0x6652E57D);
    final wallPtF = Paint()..color = const Color(0xFF58A8FF);
    final wallPtB = Paint()..color = const Color(0x6658A8FF);

    for (final p in hitsBF) {
      final pos = Offset(p.lx, p.ly);
      canvas.drawCircle(pos, 2.0, guide);
      final paint = switch ((p.kind, p.isForward)) {
        (RayHitKind.terrain, true) => terrPtF,
        (RayHitKind.terrain, false)=> terrPtB,
        (RayHitKind.pad, true)     => padPtF,
        (RayHitKind.pad, false)    => padPtB,
        (RayHitKind.wall, true)    => wallPtF,
        (RayHitKind.wall, false)   => wallPtB,
      };
      canvas.drawCircle(pos, 4.0, paint);
    }

    // Draw a short forward axis line
    final axis = Paint()..color = Colors.amberAccent..strokeWidth = 2;
    canvas.drawLine(const Offset(0, 0), const Offset(0, -36), axis);

    canvas.restore();
  }

  // Ghost overlay for a historical AI Vision frame
  void _paintAIVisionGhost(Canvas canvas, RayFrame rf, {required double alpha}) {
    if (rf.hits.isEmpty) return;

    canvas.save();
    canvas.translate(rf.origin.dx, rf.origin.dy);
    canvas.rotate(-rf.angle);

    // No backdrop/grid for ghosts, just faint points/guide
    final guide = Paint()..color = Colors.white.withOpacity(0.25 * alpha);

    final terrPt = Paint()..color = const Color(0xFFFF5050).withOpacity(0.70 * alpha);
    final padPt  = Paint()..color = const Color(0xFF52E57D).withOpacity(0.80 * alpha);
    final wallPt = Paint()..color = const Color(0xFF58A8FF).withOpacity(0.75 * alpha);

    final c = math.cos(-rf.angle), s = math.sin(-rf.angle);

    for (final h in rf.hits) {
      final dx = h.p.x - rf.origin.dx;
      final dy = h.p.y - rf.origin.dy;
      final lx = c * dx - s * dy;
      final ly = s * dx + c * dy;
      final pos = Offset(lx, ly);
      canvas.drawCircle(pos, 1.5, guide);
      final paint = switch (h.kind) {
        RayHitKind.terrain => terrPt,
        RayHitKind.pad     => padPt,
        RayHitKind.wall    => wallPt,
      };
      canvas.drawCircle(pos, 2.8, paint);
    }

    canvas.restore();
  }

  double _estimateFovRadians() {
    return math.pi; // 180°
  }

  void _drawLocalGrid(Canvas canvas) {
    final grid = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1
      ..color = const Color(0x33FFFFFF);
    const step = 50.0;
    const maxR = 600.0;

    // Vertical lines
    for (double x = -maxR; x <= maxR; x += step) {
      canvas.drawLine(Offset(x, -maxR), Offset(x, 0), grid);
    }
    // Horizontal lines (forward only)
    for (double y = -maxR; y <= 0; y += step) {
      canvas.drawLine(Offset(-maxR, y), Offset(maxR, y), grid);
    }

    // Range rings
    final ringPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1
      ..color = const Color(0x22FFFFFF);
    for (double r = 100; r <= maxR; r += 100) {
      canvas.drawCircle(const Offset(0, 0), r, ringPaint);
    }
  }

  Offset _rot(Offset v, double a) {
    final c = math.cos(a), s = math.sin(a);
    return Offset(c * v.dx - s * v.dy, s * v.dx + c * v.dy);
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
        if (pf.maskAtIndex(i, j) == 2) continue;
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

  void _paintBestVelTriplet(Canvas canvas) {
    final pos = Offset(lander.pos.x, lander.pos.y);
    const double k = 3.0;

    final vNow = Offset(lander.vel.x, lander.vel.y);
    if (vNow.distance > 1e-3) {
      _drawArrow(canvas, pos, pos + vNow * k * 10.0,
          line: Paint()..color = Colors.white..strokeWidth = 2.0,
          head: Paint()..color = Colors.white);
    }

    if (vecPF != null && vecPF!.distance > 1e-3) {
      _drawArrow(canvas, pos, pos + vecPF! * k,
          line: Paint()..color = Colors.cyanAccent..strokeWidth = 2.0,
          head: Paint()..color = Colors.cyanAccent);
    }

    if (vecPolicy != null && vecPolicy!.distance > 1e-3) {
      _drawArrow(canvas, pos, pos + vecPolicy! * k * 10.0,
          line: Paint()..color = Colors.pinkAccent..strokeWidth = 2.4,
          head: Paint()..color = Colors.pinkAccent);
    }

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

  void _paintPlan(Canvas canvas) {
    if (planPts.length < 2) return;

    final widths = planWidths;
    final hasBand = widths != null && widths.length >= planPts.length;

    if (hasBand) {
      final left = <Offset>[];
      final right = <Offset>[];

      for (int i = 0; i < planPts.length; i++) {
        final prev = (i == 0) ? planPts[i] : planPts[i - 1];
        final next = (i == planPts.length - 1) ? planPts[i] : planPts[i + 1];
        final tvec = next - prev;
        final len = tvec.distance;

        final nx = (len > 1e-6) ? (-tvec.dy / len) : 0.0;
        final ny = (len > 1e-6) ? ( tvec.dx / len) : 0.0;

        final w = widths![i].isFinite ? widths[i] : 0.0;
        final p = planPts[i];
        left.add(Offset(p.dx + nx * w, p.dy + ny * w));
        right.add(Offset(p.dx - nx * w, p.dy - ny * w));
      }

      final ribbon = Path()
        ..moveTo(left.first.dx, left.first.dy);
      for (int i = 1; i < left.length; i++) {
        ribbon.lineTo(left[i].dx, left[i].dy);
      }
      for (int i = right.length - 1; i >= 0; i--) {
        ribbon.lineTo(right[i].dx, right[i].dy);
      }
      ribbon.close();

      final band = Paint()
        ..style = PaintingStyle.fill
        ..isAntiAlias = true
        ..color = const Color(0x6639C5FF);
      canvas.drawPath(ribbon, band);

      final edge = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1
        ..isAntiAlias = true
        ..color = const Color(0x8839C5FF);
      canvas.drawPath(ribbon, edge);
    }

    final p = Path()..moveTo(planPts.first.dx, planPts.first.dy);
    for (int i = 1; i < planPts.length; i++) {
      p.lineTo(planPts[i].dx, planPts[i].dy);
    }
    canvas.drawPath(
      p,
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2
        ..isAntiAlias = true
        ..color = const Color(0xFFFFFFFF),
    );
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
        old.showForwardAxis != showForwardAxis ||
        !listEquals(old.planPts, planPts) ||
        !_listEqD(old.planWidths, planWidths) ||
        !listEquals(old.caverns, caverns) ||
        !_rayHistEq(old.rayHistory, rayHistory);
  }

  bool _rayHistEq(List<RayFrame> a, List<RayFrame> b) {
    if (identical(a, b)) return true;
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      final A = a[i], B = b[i];
      if (A.origin != B.origin) return false;
      if ((A.angle - B.angle).abs() > 1e-9) return false;
      if (A.hits.length != B.hits.length) return false;
      // Shallow compare endpoints; RayHit lacks == so compare coordinates/kind
      for (int k = 0; k < A.hits.length; k++) {
        final h1 = A.hits[k], h2 = B.hits[k];
        if (h1.kind != h2.kind) return false;
        if ((h1.p.x - h2.p.x).abs() > 1e-6 || (h1.p.y - h2.p.y).abs() > 1e-6) return false;
      }
    }
    return true;
  }

  bool _listEqD(List<double>? a, List<double>? b) {
    if (identical(a, b)) return true;
    if (a == null || b == null) return a == b;
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if ((a[i] - b[i]).abs() > 1e-6) return false;
    }
    return true;
  }
}
