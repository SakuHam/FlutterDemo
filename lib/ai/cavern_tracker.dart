// lib/ai/cavern_tracker.dart
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Offset, Size, Canvas, Paint, Color, Path, Rect, PaintingStyle;

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

/// Tracks caverns across frames, keeps a world-aligned "seen" grid,
/// marks caverns researched using endpoint proximity, and exposes
/// a painter-friendly view for unresearched caverns.
class CavernTracker {
  CavernTracker({
    // Cavern tracking knobs
    this.matchRadius = 42.0,
    this.staleFramesToDrop = 180,        // ~3s at 60fps
    this.minEndpointsPerFrame = 3,       // endpoints near center required in a frame
    this.endpointProximityFrac = 0.55,   // endpoint must be within radius * frac
    this.exploreCreditPerFrame = 3.0,
    this.exploreCreditNeeded = 45.0,
    this.autoResearchOnContactFrac = 0.7,

    // Visibility grid knobs
    double? worldW,
    double? worldH,
    double cellSize = 4.0,
  }) : _cellSize = cellSize {
    if (worldW != null && worldH != null) {
      configureWorld(worldW, worldH, cellSize: cellSize);
    }
  }

  // ===== Cavern persistence =====
  final double matchRadius;
  final int staleFramesToDrop;
  final int minEndpointsPerFrame;
  final double endpointProximityFrac;
  final double exploreCreditPerFrame;
  final double exploreCreditNeeded;
  final double autoResearchOnContactFrac;

  final List<CavernTrack> _tracks = [];

  // ===== Visibility grid (world-aligned, coarse) =====
  double _worldW = 0, _worldH = 0;
  double _cellSize;
  int _nx = 0, _ny = 0;           // grid dims
  Uint8List? _seen;               // 0 = unseen, 1 = seen

  // ---------- Public API ----------

  /// Configure/resize the world-aligned seen grid (call when world size changes).
  void configureWorld(double worldW, double worldH, {double? cellSize}) {
    _worldW = worldW;
    _worldH = worldH;
    if (cellSize != null) _cellSize = cellSize.clamp(1.0, 64.0);
    _nx = (_worldW / _cellSize).ceil().clamp(1, 100000);
    _ny = (_worldH / _cellSize).ceil().clamp(1, 100000);
    _seen = Uint8List(_nx * _ny);
  }

  /// Clear everything (tracks + visibility grid).
  void reset() {
    _tracks.clear();
    if (_seen != null) {
      _seen!.fillRange(0, _seen!.length, 0);
    }
  }

  int get trackCount => _tracks.length;
  double get cellSize => _cellSize;
  (int nx, int ny) get gridSize => (_nx, _ny);

  /// Fraction of world that has been seen (0..1).
  double seenFraction() {
    final s = _seen;
    if (s == null || s.isEmpty) return 0.0;
    int acc = 0;
    for (int i = 0; i < s.length; i++) acc += s[i];
    return acc / s.length;
  }

  /// Update the tracker with this frame’s caverns + rays.
  void update({
    required et.LanderState lander,
    required List<RayHit> rays,
    required List<CavernHypothesis> newHyps,
  }) {
    // 1) Fuse detections into tracks (world-aligned)
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

    // 2) Age non-updated tracks
    for (final t in _tracks) {
      if (t.staleFrames == 0 && t.seenFrames > 0) continue; // updated this frame
      t.staleFrames++;
    }
    _tracks.removeWhere((t) => t.staleFrames > staleFramesToDrop);

    // 3) Visibility grid: stamp rays as seen
    if (_seen != null && _nx > 0 && _ny > 0) {
      final ship = Offset(lander.pos.x, lander.pos.y);
      for (final h in rays) {
        // We stamp *endpoints of any kind* to show explored LOS; if you want
        // terrain-only coverage, gate with (h.kind == RayHitKind.terrain)
        final end = Offset(h.p.x, h.p.y);
        _stampSupercoverLine(ship, end);
      }
    }

    // 4) Research caverns via endpoint proximity
    final ship = Offset(lander.pos.x, lander.pos.y);
    for (final t in _tracks) {
      if (t.researched) continue;

      // Auto research if ship enters the area
      if ((t.worldPos - ship).distance <= t.radius * autoResearchOnContactFrac) {
        t.exploreCredit = exploreCreditNeeded;
        t.researched = true;
        continue;
      }

      final closeRadius = t.radius * endpointProximityFrac;
      int endpointsNear = 0;
      for (final h in rays) {
        if (h.kind != RayHitKind.terrain) continue; // endpoints on rock only
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

  /// Painter-facing view: return **unresearched** caverns (ship-local).
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

  // ----- Visibility grid drawing helpers -----

  /// Paint the "seen" mask directly. Call early in your GamePainter.paint,
  /// before terrain/rays, so it sits behind overlays.
  void paintSeenMask(Canvas canvas, {Color? seenColor}) {
    final s = _seen;
    if (s == null || s.isEmpty) return;
    final paint = Paint()
      ..style = PaintingStyle.fill
      ..color = (seenColor ?? const Color(0x2222FF99)); // light, translucent

    // Merge horizontal runs per row into fewer rects to keep draw calls low.
    for (int j = 0; j < _ny; j++) {
      int i = 0;
      while (i < _nx) {
        // skip unseen
        while (i < _nx && s[_idx(i, j)] == 0) i++;
        if (i >= _nx) break;
        // collect run of seen cells
        int i0 = i;
        while (i < _nx && s[_idx(i, j)] == 1) i++;
        final i1 = i; // exclusive

        final x = i0 * _cellSize;
        final y = j * _cellSize;
        final w = (i1 - i0) * _cellSize;
        final h = _cellSize;
        canvas.drawRect(Rect.fromLTWH(x, y, w, h), paint);
      }
    }
  }

  /// Optional: return merged rects for the seen mask if you prefer to draw yourself.
  List<Rect> seenRects() {
    final out = <Rect>[];
    final s = _seen;
    if (s == null || s.isEmpty) return out;
    for (int j = 0; j < _ny; j++) {
      int i = 0;
      while (i < _nx) {
        while (i < _nx && s[_idx(i, j)] == 0) i++;
        if (i >= _nx) break;
        int i0 = i;
        while (i < _nx && s[_idx(i, j)] == 1) i++;
        final i1 = i;
        out.add(Rect.fromLTWH(i0 * _cellSize, j * _cellSize,
            (i1 - i0) * _cellSize, _cellSize));
      }
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

  // ---- Visibility grid rasterization (supercover line) ----
  int _idx(int ix, int iy) => iy * _nx + ix;

  void _setSeenCell(int ix, int iy) {
    if (ix < 0 || iy < 0 || ix >= _nx || iy >= _ny) return;
    _seen![_idx(ix, iy)] = 1;
  }

  (int, int) _worldToCell(Offset p) {
    final ix = (p.dx / _cellSize).floor();
    final iy = (p.dy / _cellSize).floor();
    return (ix, iy);
  }

  void _stampSupercoverLine(Offset a, Offset b) {
    if (_seen == null) return;

    // Cohen–Sutherland clip to grid bounds (simple, conservative)
    final ax = a.dx.clamp(0.0, _worldW - 1e-6);
    final ay = a.dy.clamp(0.0, _worldH - 1e-6);
    final bx = b.dx.clamp(0.0, _worldW - 1e-6);
    final by = b.dy.clamp(0.0, _worldH - 1e-6);

    var (x0, y0) = _worldToCell(Offset(ax, ay));
    var (x1, y1) = _worldToCell(Offset(bx, by));

    int dx = x1 - x0;
    int dy = y1 - y0;
    int sx = (dx >= 0) ? 1 : -1;
    int sy = (dy >= 0) ? 1 : -1;
    dx = dx.abs();
    dy = dy.abs();

    _setSeenCell(x0, y0);

    // Supercover Bresenham: touches every cell the ideal line passes through.
    if (dx >= dy) {
      int err = dx;
      int y = y0;
      int x = x0;
      for (int i = 0; i < dx; i++) {
        x += sx;
        err += 2 * dy;
        if (err > 2 * dx) {
          y += sy;
          err -= 2 * dx;
          _setSeenCell(x, y);
        }
        _setSeenCell(x, y);
      }
    } else {
      int err = dy;
      int x = x0;
      int y = y0;
      for (int i = 0; i < dy; i++) {
        y += sy;
        err += 2 * dx;
        if (err > 2 * dy) {
          x += sx;
          err -= 2 * dy;
          _setSeenCell(x, y);
        }
        _setSeenCell(x, y);
      }
    }
  }
}
