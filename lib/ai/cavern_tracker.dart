// lib/ai/cavern_tracker.dart
import 'dart:math' as math;
import 'dart:ui' show Offset;

import '../engine/raycast.dart';
import '../engine/types.dart' as et;
import 'cavern_detector.dart'; // for CavernHypothesis

/// Persistent world-aligned track for a detected cavern.
class CavernTrack {
  CavernTrack({
    required this.worldPos,
    required this.radius,
    required this.score,
  });

  Offset worldPos;        // world coords (px)
  double radius;          // px (display/probe radius)
  double score;           // keep best score seen
  int seenFrames = 0;     // frames matched this session
  int staleFrames = 0;    // frames since last match
  double exploreCredit = 0;
  bool researched = false;
}

/// Tracks caverns across frames, marks them researched when enough **ray endpoints**
/// land near the cavern center, and provides a painter-friendly (ship-local) view.
class CavernTracker {
  CavernTracker({
    this.matchRadius = 42.0,
    this.staleFramesToDrop = 180,        // ~3s at 60fps
    this.minEndpointsPerFrame = 1,       // endpoints near center required in a single frame
    this.endpointProximityFrac = 0.55,   // endpoint must be within radius * frac of center
    this.exploreCreditPerFrame = 3.0,
    this.exploreCreditNeeded = 10.0, //45.0,
    this.autoResearchOnContactFrac = 0.7, // if ship enters radius*frac => instant research
  });

  /// How close (px) a new detection must be to fuse with an existing track.
  final double matchRadius;

  /// Remove tracks not seen for this many frames.
  final int staleFramesToDrop;

  /// Count a frame as "probing" if at least this many endpoints are near center.
  final int minEndpointsPerFrame;

  /// Endpoint must land within (radius * endpointProximityFrac) of the center.
  final double endpointProximityFrac;

  /// Credit added per frame if probed sufficiently.
  final double exploreCreditPerFrame;

  /// Total credit threshold to mark a track as researched.
  final double exploreCreditNeeded;

  /// If ship center gets within radius*autoResearchOnContactFrac, mark researched.
  final double autoResearchOnContactFrac;

  final List<CavernTrack> _tracks = [];

  /// Clear all tracks (e.g., on game reset / new terrain).
  void reset() => _tracks.clear();

  /// Number of active tracks (incl. researched).
  int get trackCount => _tracks.length;

  /// Update tracker with this frame’s detections and rays.
  void update({
    required et.LanderState lander,
    required List<RayHit> rays,
    required List<CavernHypothesis> newHyps,
  }) {
    // 1) Fuse new detections into world-aligned tracks.
    for (final h in newHyps) {
      final radius = _heuristicRadius(h);
      final worldPos = _bodyToWorld(lander, Offset(h.centroidLocal.x, h.centroidLocal.y));
      final idx = _findMatch(worldPos);
      if (idx < 0) {
        _tracks.add(CavernTrack(worldPos: worldPos, radius: radius, score: h.score));
      } else {
        final t = _tracks[idx];
        t.worldPos = Offset.lerp(t.worldPos, worldPos, 0.35)!;
        t.radius   = (0.7 * t.radius + 0.3 * radius).clamp(16.0, 500.0);
        t.score    = math.max(t.score, h.score);
        t.seenFrames += 1;
        t.staleFrames = 0;
      }
    }

    // 2) Age non-updated tracks.
    for (final t in _tracks) {
      if (t.staleFrames == 0 && t.seenFrames > 0) continue; // updated this frame
      t.staleFrames++;
    }
    _tracks.removeWhere((t) => t.staleFrames > staleFramesToDrop);

    // 3) Exploration via **endpoint proximity**.
    final ship = Offset(lander.pos.x, lander.pos.y);
    for (final t in _tracks) {
      if (t.researched) continue;

      // Auto research if ship enters the area (optional but handy).
      if ((t.worldPos - ship).distance <= t.radius * autoResearchOnContactFrac) {
        t.exploreCredit = exploreCreditNeeded;
        t.researched = true;
        continue;
      }

      // Count how many **terrain** ray endpoints landed close enough to the center.
      final closeRadius = t.radius * endpointProximityFrac;
      int endpointsNear = 0;
      for (final h in rays) {
        if (h.kind != RayHitKind.terrain) continue; // only terrain endpoints
        final end = Offset(h.p.x, h.p.y);
        if ((end - t.worldPos).distance <= closeRadius) endpointsNear++;
      }

      if (endpointsNear >= minEndpointsPerFrame) {
        t.exploreCredit += exploreCreditPerFrame;
      }

      if (t.exploreCredit >= exploreCreditNeeded) {
        t.researched = true;
      }
    }
  }

  /// Painter-facing view: return **unresearched** caverns in ship-local coords.
  List<CavernHypothesis> visibleForPainter(et.LanderState lander) {
    final out = <CavernHypothesis>[];
    for (final t in _tracks) {
      if (t.researched) continue;
      final local = _worldToBody(lander, t.worldPos);
      final depth = local.distance;
      final widthAng = (depth > 1e-3) ? (t.radius / depth).clamp(0.02, 1.2) : 0.2;
      final th = math.atan2(local.dx, -local.dy);
      out.add(CavernHypothesis(
        thetaStart: th - widthAng * 0.5,
        thetaEnd:   th + widthAng * 0.5,
        widthAng:   widthAng,
        depth:      depth,
        score:      t.score,
        centroidLocal: et.Vector2(local.dx, local.dy),
      ));
    }
    return out;
  }

  // ---------- Internals ----------

  int _findMatch(Offset p) {
    for (int i = 0; i < _tracks.length; i++) {
      if ((_tracks[i].worldPos - p).distance <= matchRadius) return i;
    }
    return -1;
  }

  double _heuristicRadius(CavernHypothesis h) {
    final rDepth = h.depth * 0.30;
    final rAng   = (h.widthAng * 0.5) * (h.depth);
    return (0.5 * rDepth + 0.5 * rAng).clamp(20.0, 320.0);
  }

  Offset _bodyToWorld(et.LanderState L, Offset local) {
    final c = math.cos(L.angle), s = math.sin(L.angle);
    final wx = L.pos.x + c * local.dx - s * local.dy;
    final wy = L.pos.y + s * local.dx + c * local.dy;
    return Offset(wx, wy);
  }

  Offset _worldToBody(et.LanderState L, Offset world) {
    final dx = world.dx - L.pos.x, dy = world.dy - L.pos.y;
    final c = math.cos(-L.angle), s = math.sin(-L.angle);
    return Offset(c * dx - s * dy, s * dx + c * dy);
  }
}
