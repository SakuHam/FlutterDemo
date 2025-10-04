// lib/ai/cavern_detector.dart
import 'dart:math' as math;

import '../engine/types.dart' as et;
import '../engine/raycast.dart';

/// Result for one detected cavern-like opening (ship-local coordinates).
class CavernHypothesis {
  final double thetaStart;         // body-frame angle at left edge (rad)
  final double thetaEnd;           // body-frame angle at right edge (rad)
  final double widthAng;           // = thetaEnd - thetaStart (rad)
  final double depth;              // representative deep range (px)
  final double score;              // 0..1 (here: depth-normalized)
  final et.Vector2 centroidLocal;  // centroid in body frame (x right, y up; y negative forward)

  CavernHypothesis({
    required this.thetaStart,
    required this.thetaEnd,
    required this.widthAng,
    required this.depth,
    required this.score,
    required this.centroidLocal,
  });
}

/// Simplified detector:
/// - Take all **terrain** hits, sorted by body-frame angle.
/// - For every **adjacent pair** (i, i+1), create a cavern **between** them.
/// - The only ranking is the **range `t`** (depth): score = depth / maxDepth.
/// - No thresholds, no jump heuristics. False positives are acceptable.
///
/// Notes:
/// - We ignore `wall` (sky/bounds) and `pad` so you don't get "under-pad" ghosts.
/// - Works on whichever hemisphere you pass in (forward/back). If you want both,
///   call detect() on each subset and concatenate in the caller.
class CavernDetector {
  // Kept only for API compatibility; unused in this simplified version.
  final double jumpThresh;
  final double voidBoost;
  final double minVoidSpanRad;
  final int minSpanSamples;

  const CavernDetector({
    this.jumpThresh = 0.0,
    this.voidBoost = 1.0,
    this.minVoidSpanRad = 0.0,
    this.minSpanSamples = 0,
  });

  List<CavernHypothesis> detect({
    required List<RayHit> rays,
    required et.LanderState lander,
    double? jumpOverride,          // unused
    double? deepQuantileOverride,  // unused
  }) {
    if (rays.isEmpty) return const [];

    // ---- Collect TERRAIN-only hits in body frame (no pad, no wall) ----
    final List<_Samp> pts = [];
    final c = math.cos(-lander.angle), s = math.sin(-lander.angle);
    for (final h in rays) {
      final dx = h.p.x - lander.pos.x;
      final dy = h.p.y - lander.pos.y;
      final lx = c * dx - s * dy;
      final ly = s * dx + c * dy;
      final th = math.atan2(lx, -ly); // 0 = forward (up), + right; y-up in body frame
      pts.add(_Samp(th: th, r: h.t, lx: lx, ly: ly));
    }
    if (pts.length < 2) return const [];

    // Sort by angle and unwrap so adjacency is meaningful.
    pts.sort((a, b) => a.th.compareTo(b.th));
    final kept = _unwrapAngles(pts);
    final n = kept.length;

    // Find max depth for normalization (avoid div by zero).
    double rMax = 0.0;
    for (final p in kept) if (p.r > rMax) rMax = p.r;
    if (rMax <= 1e-9) rMax = 1.0;

    // ---- Create a "between" hypothesis for every adjacent pair ----
    final out = <CavernHypothesis>[];
    for (int i = 0; i < n - 1; i++) {
      final a = kept[i];
      final b = kept[i + 1];

      final th0 = a.th;
      final th1 = b.th;
      final dth = th1 - th0;
      if (!dth.isFinite || dth.abs() < 1e-9) continue;

      // Representative depth: choose max of the pair (or use avg if you prefer).
      final depth = math.max(a.r, b.r);

      // Mid-angle & mid-range for centroid position (you can bias toward far if desired).
      final thMid = 0.5 * (th0 + th1);
      final rMid  = 0.5 * (a.r + b.r);

      // Convert back to ship-local x/y (x right, y up; negative y is forward).
      final cx = rMid * math.sin(thMid);
      final cy = -rMid * math.cos(thMid);

      // Depth-only score in [0,1].
      final score = (depth / rMax).clamp(0.0, 1.0);

      final deltar = (a.r - b.r).abs();
      if (deltar > 50.0) {
        out.add(CavernHypothesis(
          thetaStart: th0,
          thetaEnd: th1,
          widthAng: dth,
          depth: depth,
          score: score,
          centroidLocal: et.Vector2(cx, cy),
        ));
      }
    }

    return out;
  }
}

// ===== Helpers =====

// Unwrap so angular list is continuous (avoid seam split).
List<_Samp> _unwrapAngles(List<_Samp> sorted) {
  if (sorted.isEmpty) return sorted;
  final out = <_Samp>[];
  double prev = sorted.first.th;
  out.add(sorted.first);
  for (int i = 1; i < sorted.length; i++) {
    double th = sorted[i].th;
    while (th - prev > math.pi) th -= 2 * math.pi;
    while (prev - th > math.pi) th += 2 * math.pi;
    out.add(_Samp(th: th, r: sorted[i].r, lx: sorted[i].lx, ly: sorted[i].ly));
    prev = th;
  }
  return out;
}

// ------------------------------------------------------------
// Internals

class _Samp {
  final double th;  // body-frame angle
  final double r;   // range t
  final double lx, ly;
  _Samp({required this.th, required this.r, required this.lx, required this.ly});
}
