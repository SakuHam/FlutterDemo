// lib/ai/cavern_detector.dart
import 'dart:math' as math;

import '../engine/types.dart' as et;
import '../engine/raycast.dart';

/// Result for one detected cavern-like opening in the fan.
class CavernHypothesis {
  final double thetaStart;         // body-frame angle at left edge (rad)
  final double thetaEnd;           // body-frame angle at right edge (rad)
  final double widthAng;           // = thetaEnd - thetaStart (rad)
  final double depth;              // representative deep range inside span (px)
  final double score;              // 0..1 heuristic quality
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

/// Heuristic detector for cavern-like “voids” in a polar range fan.
/// Requires RayHit.t to be the range (in px/units) along the ray.
class CavernDetector {
  /// Baseline threshold on |dr/dθ| (upper bound; an adaptive value is used per frame).
  final double jumpThresh;

  /// Score multiplier for long/deep spans.
  final double voidBoost;

  /// Minimum angular width (radians) for a candidate (absolute safeguard).
  final double minVoidSpanRad;

  /// Minimum raw sample count inside a span.
  final int minSpanSamples;

  const CavernDetector({
    this.jumpThresh = 150.0,
    this.voidBoost = 1.2,
    this.minVoidSpanRad = math.pi / 64, // ≈2.8°
    this.minSpanSamples = 2,
  });

  List<CavernHypothesis> detect({
    required List<RayHit> rays,
    required et.LanderState lander,
    double? jumpOverride,          // optional seed for edge threshold
    double? deepQuantileOverride,  // 0..1; default is 0.65 (more permissive)
  }) {
    if (rays.isEmpty) return const [];

    // --- Build body-frame samples (NO hemisphere filter here) ---
    // Ignore walls (world bounds/sky) to avoid false "deep" ranges.
    final List<_Samp> pts = [];
    final c = math.cos(-lander.angle), s = math.sin(-lander.angle);
    for (final h in rays) {
      if (h.kind == RayHitKind.wall) continue;
      final dx = h.p.x - lander.pos.x;
      final dy = h.p.y - lander.pos.y;
      final lx = c * dx - s * dy;
      final ly = s * dx + c * dy;
      final th = math.atan2(lx, -ly); // 0 = forward (up), + right
      pts.add(_Samp(th: th, r: h.t, kind: h.kind, lx: lx, ly: ly));
    }
    if (pts.length < 5) return const [];
    pts.sort((a, b) => a.th.compareTo(b.th));

    // Unwrap list so spans can cross the seam.
    final kept = _unwrapForwardFan(pts);
    final n = kept.length;
    if (n < 5) return const [];

    // --- Smooth ranges (median-of-3) ---
    final rSm = List<double>.filled(n, 0);
    for (int i = 0; i < n; i++) {
      if (i == 0 || i == n - 1) {
        rSm[i] = kept[i].r;
      } else {
        rSm[i] = _median3(kept[i - 1].r, kept[i].r, kept[i + 1].r);
      }
    }

    // --- Edge strength |dr/dθ| ---
    final dAbs = List<double>.filled(n, 0);
    for (int i = 1; i < n - 1; i++) {
      final r0 = rSm[i - 1], r2 = rSm[i + 1];
      final th0 = kept[i - 1].th, th2 = kept[i + 1].th;
      final dth = (th2 - th0).abs();
      if (dth < 1e-9) continue;
      dAbs[i] = (r2 - r0).abs() / dth;
    }

    // --- Deep threshold: percentile of smoothed ranges (permissive) ---
    final q = (deepQuantileOverride ?? 0.65).clamp(0.0, 1.0);
    final deepThresh = _percentile(rSm, q);

    final spans = <_Span>[];
    int i = 0;
    while (i < n) {
      // Find deep segment (allow one shallow gap).
      while (i < n && rSm[i] < deepThresh) i++;
      if (i >= n) break;
      int j = i;
      int shallowAllowance = 1;
      while (j < n) {
        if (rSm[j] >= deepThresh) {
          j++;
        } else if (shallowAllowance > 0) {
          shallowAllowance--;
          j++;
        } else {
          break;
        }
      }

      // --- Softer edge clipping: dual threshold + bounded scan + fallback ---
      final dAbsNZ = dAbs.where((v) => v.isFinite && v > 0).toList();
      final dPerc = dAbsNZ.isEmpty ? 0.0 : _percentile(dAbsNZ, 0.70);
      final baseJump = (jumpOverride ?? jumpThresh);

      final jumpHi = math.max(12.0, math.min(baseJump, dPerc) * 0.50);
      final jumpLo = jumpHi * 0.50;

      const int kEdgeBack = 6;
      const int kEdgeFwd  = 6;

      int left = i;
      bool hitLeftHi = false;
      for (int k = i; k >= math.max(1, i - kEdgeBack); k--) {
        if (dAbs[k] >= jumpHi) { left = k; hitLeftHi = true; break; }
        if (dAbs[k] >= jumpLo) { left = k; } // keep LO if no HI
      }

      int right = j - 1;
      bool hitRightHi = false;
      for (int k = j - 1; k <= math.min(n - 2, (j - 1) + kEdgeFwd); k++) {
        if (dAbs[k] >= jumpHi) { right = k; hitRightHi = true; break; }
        if (dAbs[k] >= jumpLo) { right = k; }
      }

      if (!hitLeftHi && left == i)   left  = i;       // don't shrink if no spike
      if (!hitRightHi && right == j) right = j - 1;
      if (right <= left) { left = i; right = j - 1; }

      // --- Span acceptance (adaptive) ---
      final th0 = kept[left].th;
      final th1 = kept[right].th;
      final width = th1 - th0;
      final sampCount = (right - left + 1);
      if (sampCount < 1) { i = j + 1; continue; }

      double _avgDTheta(List<_Samp> a) {
        if (a.length < 2) return math.pi / 180;
        double sum = 0;
        for (int t = 1; t < a.length; t++) sum += (a[t].th - a[t - 1].th).abs();
        return sum / (a.length - 1);
      }
      final avgDth = _avgDTheta(kept);

      const int kSteps = 3; // require ~3 samples across
      final double minAngBySteps = kSteps * avgDth;
      final double minAngAbsolute = minVoidSpanRad;
      final bool okAngular = width >= math.min(minAngAbsolute, minAngBySteps);
      final bool okSamples = sampCount >= math.max(minSpanSamples, kSteps);

      // Physical size gate to keep far-but-obvious holes.
      const double minArcPx = 20.0;
      final double depthP75 = _p75(rSm, left, right);
      final bool okPhysical = (width * depthP75) >= minArcPx;

      if ((okAngular && okSamples) || okPhysical) {
        // Centroid (body frame).
        double sx = 0, sy = 0;
        for (int k = left; k <= right; k++) { sx += kept[k].lx; sy += kept[k].ly; }
        final cx = sx / sampCount, cy = sy / sampCount;

        // Score: width & depth normalized to this frame.
        final rMax = rSm.reduce(math.max);
        final rMed = _median(rSm);
        final widthNorm = (width / math.pi).clamp(0.0, 1.0);
        final depthNorm = ((depthP75 - rMed) / (1e-9 + (rMax - rMed))).clamp(0.0, 1.0);
        double score = (0.55 * widthNorm + 0.45 * depthNorm) * voidBoost;
        score = score.clamp(0.0, 1.0);

        spans.add(_Span(
          thetaStart: th0,
          thetaEnd: th1,
          width: width,
          depth: depthP75,
          score: score,
          cx: cx,
          cy: cy,
        ));
      }

      i = j + 1;
    }

    // Merge overlap/touching (including wrap-around end<->start).
    final merged = _mergeSpansWithWrap(spans);

    return merged.map((s) => CavernHypothesis(
      thetaStart: s.thetaStart,
      thetaEnd: s.thetaEnd,
      widthAng: s.width,
      depth: s.depth,
      score: s.score,
      centroidLocal: et.Vector2(s.cx, s.cy),
    )).toList();
  }
}

// ===== Helpers =====

// Unwrap angles so list is continuous (avoids seam split).
List<_Samp> _unwrapForwardFan(List<_Samp> sorted) {
  if (sorted.isEmpty) return sorted;
  final out = <_Samp>[];
  double prev = sorted.first.th;
  out.add(sorted.first);
  for (int i = 1; i < sorted.length; i++) {
    double th = sorted[i].th;
    while (th - prev > math.pi) th -= 2 * math.pi;
    while (prev - th > math.pi) th += 2 * math.pi;
    out.add(_Samp(
      th: th, r: sorted[i].r, kind: sorted[i].kind,
      lx: sorted[i].lx, ly: sorted[i].ly,
    ));
    prev = th;
  }
  return out;
}

// Merge spans including wrap seam if ends are within ~1°.
List<_Span> _mergeSpansWithWrap(List<_Span> spans) {
  if (spans.isEmpty) return const [];
  spans.sort((a, b) => a.thetaStart.compareTo(b.thetaStart));
  final merged = <_Span>[];
  _Span cur = spans.first;
  for (int i = 1; i < spans.length; i++) {
    final s = spans[i];
    final touching = (s.thetaStart - cur.thetaEnd) <= (math.pi / 180); // 1°
    if (touching || s.thetaStart <= cur.thetaEnd) {
      cur.thetaEnd = math.max(cur.thetaEnd, s.thetaEnd);
      cur.width = cur.thetaEnd - cur.thetaStart;
      cur.depth = math.max(cur.depth, s.depth);
      cur.score = math.max(cur.score, s.score);
      final w0 = (cur.width).abs(), w1 = (s.width).abs(), ws = (w0 + w1).clamp(1e-6, 1e9);
      cur.cx = (cur.cx * w0 + s.cx * w1) / ws;
      cur.cy = (cur.cy * w0 + s.cy * w1) / ws;
    } else {
      merged.add(cur);
      cur = s;
    }
  }
  merged.add(cur);

  // Try wrap-merge last<->first
  if (merged.length >= 2) {
    final first = merged.first;
    final last = merged.last;
    if ((first.thetaStart - last.thetaEnd).abs() <= (math.pi / 180)) {
      last.thetaEnd = first.thetaEnd;
      last.width = last.thetaEnd - last.thetaStart;
      last.depth = math.max(last.depth, first.depth);
      last.score = math.max(last.score, first.score);
      final w0 = (last.width).abs(), w1 = (first.width).abs(), ws = (w0 + w1).clamp(1e-6, 1e9);
      last.cx = (last.cx * w0 + first.cx * w1) / ws;
      last.cy = (last.cy * w0 + first.cy * w1) / ws;
      merged.removeAt(0);
    }
  }
  return merged;
}

// Percentile (0..1).
double _percentile(List<double> data, double q) {
  if (data.isEmpty) return 0.0;
  final a = List<double>.from(data)..sort();
  final idx = (q.clamp(0.0, 1.0) * (a.length - 1)).toDouble();
  final lo = idx.floor();
  final hi = idx.ceil();
  if (lo == hi) return a[lo];
  final t = idx - lo;
  return a[lo] * (1 - t) + a[hi] * t;
}

// p75 on rSm[left..right].
double _p75(List<double> rSm, int left, int right) {
  if (right < left) return 0.0;
  final sub = rSm.sublist(left, right + 1)..sort();
  final idx = (sub.length * 3) ~/ 4;
  return sub[idx];
}

// ------------------------------------------------------------
// Internals
class _Samp {
  final double th;
  final double r;
  final RayHitKind kind;
  final double lx, ly; // body-frame coords
  _Samp({required this.th, required this.r, required this.kind, required this.lx, required this.ly});
}

class _Span {
  double thetaStart, thetaEnd, width, depth, score, cx, cy;
  _Span({
    required this.thetaStart,
    required this.thetaEnd,
    required this.width,
    required this.depth,
    required this.score,
    required this.cx,
    required this.cy,
  });
}

// (Unused helper kept for reference.)
List<_Span> _mergeSpans(List<_Span> spans) {
  if (spans.isEmpty) return const [];
  spans.sort((a, b) => a.thetaStart.compareTo(b.thetaStart));
  final out = <_Span>[spans.first];
  for (int i = 1; i < spans.length; i++) {
    final prev = out.last;
    final cur = spans[i];
    final touching = (cur.thetaStart - prev.thetaEnd) <= (math.pi / 180);
    if (touching || cur.thetaStart <= prev.thetaEnd) {
      prev.thetaEnd = math.max(prev.thetaEnd, cur.thetaEnd);
      prev.width = prev.thetaEnd - prev.thetaStart;
      prev.depth = math.max(prev.depth, cur.depth);
      prev.score = math.max(prev.score, cur.score);
      final w0 = prev.width.abs(), w1 = cur.width.abs(), wSum = (w0 + w1).clamp(1e-6, 1e9);
      prev.cx = (prev.cx * w0 + cur.cx * w1) / wSum;
      prev.cy = (prev.cy * w0 + cur.cy * w1) / wSum;
    } else {
      out.add(cur);
    }
  }
  return out;
}

double _median(List<double> v) {
  if (v.isEmpty) return 0;
  final a = List<double>.from(v)..sort();
  final m = a.length >> 1;
  return a.length.isOdd ? a[m] : 0.5 * (a[m - 1] + a[m]);
}

double _median3(double a, double b, double c) {
  if (a > b) { final t = a; a = b; b = t; }
  if (b > c) { final t = b; b = c; c = t; }
  if (a > b) { final t = a; a = b; b = t; }
  return b;
}
