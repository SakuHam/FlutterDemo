// lib/engine/game_engine.dart
import 'dart:math' as math;

import 'types.dart';
import 'scoring.dart';

// Helpers
import 'hull_contact.dart'; // ContactResolver, ContactKind
import 'raycast.dart';      // RayCaster, RayConfig, RayHit

/// The Game Engine (physics + shaping cost) with adaptive sub-stepping to prevent tunneling.
class GameEngine {
  EngineConfig cfg;
  late Terrain terrain;
  late LanderState lander;
  GameStatus status = GameStatus.playing;

  math.Random _rnd;
  Terrain? _frozenTerrain;
  double _lastAngle = 0.0;

  // Heuristic assists
  static const double _autoLevelBelow = 140.0;   // px
  static const double _autoLevelGain  = 3.0;     // 1/sec
  static const double _flareBelow     = 40.0;    // px
  static const double _flareGain      = 8.0;     // 1/sec
  static const double _rotFriction    = 0.6;     // rad/sec toward 0

  // Substep limits (anti-tunneling)
  static const double _maxVertexStepPx = 3.0;      // max vertex travel per substep
  static const double _maxAngleStepRad = 7.0 * math.pi / 180.0; // max rotation per substep
  static const int    _maxSubsteps     = 12;       // absolute cap for perf

  // Sensors
  RayCaster _sensors = RayCaster(const RayConfig());
  List<RayHit> _rays = const [];
  List<RayHit> get rays => _rays;
  RayConfig get rayCfg => _sensors.cfg;
  set rayCfg(RayConfig v) => _sensors = RayCaster(v);

  GameEngine(this.cfg) : _rnd = math.Random(cfg.seed) {
    reset(seed: cfg.seed);
  }

  // ---------------- Lifecycle ----------------
  void reset({
    int? seed,
    double? padWidthFactor,
    double? landingSpeedMax,
    double? landingAngleMaxRad,
  }) {
    if (seed != null) _rnd = math.Random(seed);

    // Optional curriculum overrides
    if (padWidthFactor != null || landingSpeedMax != null || landingAngleMaxRad != null) {
      cfg = cfg.copyWith(
        padWidthFactor: padWidthFactor ?? cfg.padWidthFactor,
        landingSpeedMax: landingSpeedMax ?? cfg.landingSpeedMax,
        landingAngleMaxRad: landingAngleMaxRad ?? cfg.landingAngleMaxRad,
      );
    }

    // Terrain
    if (cfg.lockTerrain) {
      _frozenTerrain ??= Terrain.generate(
        cfg.worldW, cfg.worldH, cfg.terrainSeed, cfg.padWidthFactor,
      );
      terrain = _frozenTerrain!;
    } else {
      terrain = Terrain.generate(
        cfg.worldW, cfg.worldH, _rnd.nextInt(1 << 30), cfg.padWidthFactor,
      );
    }

    // ---- Spawn X (avoid spawning over pad) ----
    // Find pad horizontal bounds from terrain polygon edges
    double padMinX = double.infinity, padMaxX = -double.infinity;
    for (final e in terrain.poly.edges) {
      if (e.kind == PolyEdgeKind.pad) {
        padMinX = math.min(padMinX, math.min(e.a.x, e.b.x));
        padMaxX = math.max(padMaxX, math.max(e.a.x, e.b.x));
      }
    }
    final bool hasPadBounds = padMinX.isFinite && padMaxX.isFinite;

    // Safety margin so “above the pad” includes its immediate neighborhood
    const double _spawnExclMarginPx = 8.0;
    final double exclMin = hasPadBounds
        ? (padMinX - _spawnExclMarginPx).clamp(0.0, cfg.worldW.toDouble())
        : -1.0;
    final double exclMax = hasPadBounds
        ? (padMaxX + _spawnExclMarginPx).clamp(0.0, cfg.worldW.toDouble())
        : -2.0;

    double _sampleFracX() {
      final a = cfg.spawnXMin.clamp(0.0, 1.0);
      final b = cfg.spawnXMax.clamp(0.0, 1.0);
      final lo = math.min(a, b), hi = math.max(a, b);
      return lo + _rnd.nextDouble() * (hi - lo);
    }

    double _pickSpawnX() {
      // If we don't know pad bounds, fall back to the legacy behavior.
      if (!hasPadBounds) {
        final fracLegacy = cfg.randomSpawnX ? _sampleFracX() : cfg.spawnX;
        return cfg.worldW * fracLegacy;
      }

      // Try a few times to get an X outside the pad exclusion interval.
      const int maxTries = 16;
      for (int i = 0; i < maxTries; i++) {
        final frac = cfg.randomSpawnX ? _sampleFracX() : cfg.spawnX;
        final x = cfg.worldW * frac;
        if (x < exclMin || x > exclMax) return x;
      }

      // Fallback: snap just outside the exclusion band, keeping within world
      final frac = cfg.randomSpawnX ? _sampleFracX() : cfg.spawnX;
      final x0 = cfg.worldW * frac;
      final mid = (exclMin + exclMax) * 0.5;
      return (x0 < mid)
          ? (exclMin - 1.0).clamp(0.0, cfg.worldW.toDouble())
          : (exclMax + 1.0).clamp(0.0, cfg.worldW.toDouble());
    }

    final double spawnX = _pickSpawnX();

    // Lander init (lockSpawn keeps Y/vel/angle)
    if (cfg.lockSpawn) {
      lander = LanderState(
        pos: Vector2(spawnX, cfg.spawnY),
        vel: Vector2(cfg.spawnVx, cfg.spawnVy),
        angle: cfg.spawnAngle,
        fuel: cfg.t.maxFuel,
      );
    } else {
      lander = LanderState(
        pos: Vector2(spawnX, cfg.spawnY),
        vel: Vector2(0, 0),
        angle: 0.0,
        fuel: cfg.t.maxFuel,
      );
    }

    _lastAngle = lander.angle;
    status = GameStatus.playing;

    // Prime sensors
    _rays = _sensors.castAll(
      terrain: terrain,
      engineCfg: cfg,
      origin: lander.pos,
      angle: lander.angle,
    );
  }

  /// Physics step + contact resolution + shaping cost. `dt` in seconds.
  /// Uses adaptive sub-stepping to avoid tunneling through thin terrain edges.
  StepInfo step(double dt, ControlInput u) {
    if (status != GameStatus.playing) {
      return StepInfo(
        costDelta: 0.0, terminal: true, landed: status == GameStatus.landed, onPad: false,
      );
    }

    final t = cfg.t;

    // ----- Controls -> target angular change estimate (for substep sizing) -----
    double estOmega = 0.0;
    {
      final rot = t.rotSpeed * 0.5;
      if (u.left ^ u.right) estOmega = rot; // turning
      // add a bit for auto-level assist pull
      final gyNow = terrain.heightAt(lander.pos.x);
      final hNow  = (gyNow - lander.pos.y).toDouble();
      if (hNow < _autoLevelBelow) estOmega += _autoLevelGain * (dt.clamp(0.0, 0.05));
      if (hNow < _flareBelow)     estOmega += _flareGain * (dt.clamp(0.0, 0.05));
    }

    // Estimate max vertex travel this frame
    final s = cfg.stepScale;
    final vx = lander.vel.x * dt * s;
    final vy = lander.vel.y * dt * s;
    final centerTravel = math.sqrt(vx*vx + vy*vy);
    const shipRadius = 24.0;
    final rotTravel  = estOmega * shipRadius * dt;
    final maxTravel  = centerTravel + rotTravel;

    int nPos = (maxTravel / _maxVertexStepPx).ceil();
    int nAng = (estOmega * dt / _maxAngleStepRad).ceil();
    int substeps = math.max(1, math.min(_maxSubsteps, math.max(nPos, nAng)));

    double hRemaining = dt;
    double stepCostAcc = 0.0;
    bool terminal = false;
    bool landedOk = false;
    bool onPad = false;

    // Run substeps
    for (int k = 0; k < substeps; k++) {
      final dtk = hRemaining / (substeps - k);
      hRemaining -= dtk;

      // ----- Controls -> rotation (+ damping) for this substep -----
      double angle = lander.angle;
      final rot = t.rotSpeed * 0.5;
      if (u.left && !u.right) angle -= rot * dtk;
      if (u.right && !u.left) angle += rot * dtk;
      if (!(u.left ^ u.right)) {
        final pull = (_rotFriction * dtk);
        if (pull > 0.0) angle -= angle * pull.clamp(0.0, 0.5);
      }

      // ----- Auto-level & flare assists -----
      final gyNow = terrain.heightAt(lander.pos.x);
      final hNow  = (gyNow - lander.pos.y).toDouble();
      if (hNow < _autoLevelBelow) {
        final kauto = _autoLevelGain * dtk;
        angle -= angle * kauto.clamp(0.0, 0.5);
      }
      if (hNow < _flareBelow) {
        final kflare = _flareGain * dtk;
        angle -= angle * kflare.clamp(0.0, 0.85);
      }

      // ----- Acceleration & fuel burn -----
      // Keep your global accel scale.
      final double thrustA = t.thrustAccel * 0.05; // EQUALIZED THRUST POWER
      Vector2 accel = Vector2(0, t.gravity * 0.05);
      double fuel = lander.fuel;

      // Main engine
      final bool mainOn = u.thrust && fuel > 0.0;
      final double power = mainOn ? 1.0 : 0.0;
      if (mainOn) {
        accel.x += math.sin(angle) * thrustA;
        accel.y += -math.cos(angle) * thrustA;
        fuel = (fuel - 20.0 * dtk).clamp(0.0, t.maxFuel);
      }

      // Side thrusters (equal power to main, per-thruster fuel)
      if (t.rcsEnabled && fuel > 0.0) {
        final bool l = u.sideLeft;
        final bool r = u.sideRight;
        if (l || r) {
          double ax = 0.0, ay = 0.0;
          if (t.rcsBodyFrame) {
            // body-frame: +X is ship right
            final axLocal = (l ? 1.0 : 0.0) * thrustA + (r ? -1.0 : 0.0) * thrustA;
            final c = math.cos(angle), s2 = math.sin(angle);
            ax =  c * axLocal;
            ay =  s2 * axLocal;
          } else {
            // world-frame strafing
            ax = ((l ? 1.0 : 0.0) + (r ? -1.0 : 0.0)) * thrustA;
            ay = 0.0;
          }
          accel.x += ax;
          accel.y += ay;

          // Burn per active side thruster (so both pressed = 2x burn)
          final int activeSides = (l ? 1 : 0) + (r ? 1 : 0);
          fuel = (fuel - (20.0 * activeSides) * dtk).clamp(0.0, t.maxFuel);
        }
      }

      // "Top" thruster (pushes downward) — equal power & equal burn to main
      if (t.downThrEnabled && u.downThrust && fuel > 0.0) {
        // opposite vector to main engine
        accel.x += -math.sin(angle) * thrustA;
        accel.y +=  math.cos(angle) * thrustA;
        fuel = (fuel - 20.0 * dtk).clamp(0.0, t.maxFuel);
      }

      // ----- Integrate (semi-implicit Euler) -----
      final vel = Vector2(
        lander.vel.x + accel.x * dtk * s,
        lander.vel.y + accel.y * dtk * s,
      );
      Vector2 pos = Vector2(
        lander.pos.x + vel.x * dtk * s,
        lander.pos.y + vel.y * dtk * s,
      );

      // Late flare snap for this microstep
      final gyPost = terrain.heightAt(pos.x);
      final hPost  = (gyPost - pos.y).toDouble();
      if (hPost < _flareBelow * 0.6) {
        final kf = (_flareGain * 1.35) * dtk;
        angle -= angle * kf.clamp(0.0, 0.95);
      }

      // ----- Contact resolution on this substep -----
      final resolver = ContactResolver(cfg: cfg, terrain: terrain);
      final contact = resolver.resolve(pos, vel, angle);

      bool _gentle(Vector2 v, double ang) {
        final okSpeed = v.length <= cfg.landingSpeedMax;
        final okAngle = !cfg.t.crashOnTilt || ang.abs() <= cfg.landingAngleMaxRad;
        return okSpeed && okAngle;
      }

      if (contact.kind == ContactKind.pad) {
        if (_gentle(contact.vel, angle)) {
          lander = LanderState(pos: contact.pos, vel: Vector2(0, 0), angle: angle, fuel: fuel);
          _lastAngle = angle;
          status = GameStatus.landed;
          terminal = true;
          landedOk = true;
          onPad = true;
          break;
        } else {
          lander = LanderState(pos: contact.pos, vel: contact.vel, angle: angle, fuel: fuel);
          _lastAngle = angle;
          status = GameStatus.crashed;
          terminal = true;
          landedOk = false;
          onPad = true;
          break;
        }
      }

      if (contact.kind == ContactKind.terrain ||
          contact.kind == ContactKind.wall    ||
          contact.kind == ContactKind.ceiling) {
        lander = LanderState(pos: contact.pos, vel: contact.vel, angle: angle, fuel: fuel);
        _lastAngle = angle;
        status = GameStatus.crashed;
        terminal = true;
        landedOk = false;
        onPad = false;
        break;
      }

      // ----- Effort + shaping cost for this substep -----
      double stepCost = 0.0;
      if (power > 0) stepCost += cfg.effortCost * power * dtk;

      final score = Scoring.apply(
        cfg: cfg,
        terrain: terrain,
        pos: pos,
        vel: vel,
        angle: angle,
        prevAngle: _lastAngle,
        dt: dtk,
        u: u,
        intentIdx: u.intentIdx,
      );
      stepCost += score.cost;

      // Commit microstep physics
      lander = LanderState(pos: pos, vel: vel, angle: angle, fuel: fuel);
      _lastAngle = angle;

      stepCostAcc += stepCost;
    }

    // Sensors update (once per frame)
    _rays = _sensors.castAll(
      terrain: terrain,
      engineCfg: cfg,
      origin: lander.pos,
      angle: lander.angle,
    );

    if (terminal) {
      return StepInfo(
        costDelta: stepCostAcc,
        terminal: true,
        landed: landedOk,
        onPad: onPad,
      );
    }

    return StepInfo(
      costDelta: stepCostAcc,
      terminal: false,
      landed: false,
      onPad: false,
    );
  }

  // ---- NN convenience passthroughs ----
  List<double> rayChannels3({bool useProximity = false}) =>
      _sensors.channels3(hits: _rays, engineCfg: cfg, useProximity: useProximity);

  ({double minD, double bearing, double visible}) padSummary() =>
      _sensors.padSummary(hits: _rays, engineCfg: cfg);
}
