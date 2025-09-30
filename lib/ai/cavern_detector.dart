// lib/ai/cavern_detector.dart
import 'dart:math' as math;

import '../engine/types.dart' as et;
import '../engine/raycast.dart';

/// Result for one detected cavern-like opening in the forward fan.
class CavernHypothesis {
  final double thetaStart;      // body-frame angle at left edge (rad)
  final double thetaEnd;        // body-frame angle at right edge (rad)
  final double widthAng;        // = thetaEnd - thetaStart (rad)
  final double depth;           // representative deep range inside span (px)
  final double score;           // 0..1 heuristic quality
  final et.Vector2 centroidLocal;   // centroid in body frame (x right, y up; y negative forward)

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
/// Key: relies on RayHit.t to be the param/range along the ray.
class CavernDetector {
  final double jumpThresh;       // threshold on |dr/dθ| for edge spikes
  final double voidBoost;        // multiplies score for long/deep spans
  final double minVoidSpanRad;   // minimum angular width of a cavern
  final int    minSpanSamples;   // minimum sample count inside the span

  const CavernDetector({
    this.jumpThresh = 150.0,
    this.voidBoost = 1.2,
    this.minVoidSpanRad = math.pi / 48, // ~3.75°
    this.minSpanSamples = 3,
  });

  List<CavernHypothesis> detect({
    required List<RayHit> rays,
    required et.LanderState lander,
    double? jumpOverride,          // NEW: adapt |dr/dθ| threshold
    double? deepQuantileOverride,  // NEW: 0..1 (e.g. 0.65 = more permissive)
  }) {
    if (rays.isEmpty) return const [];

    // --- Build (theta, range, kind) list in body frame and sort by theta ---
    final List<_Samp> pts = [];
    final c = math.cos(-lander.angle), s = math.sin(-lander.angle);

    for (final h in rays) {
      final dx = h.p.x - lander.pos.x;
      final dy = h.p.y - lander.pos.y;
      final lx = c * dx - s * dy;
      final ly = s * dx + c * dy;
      // Body θ: 0 points forward (up), positive to the right.
      final th = math.atan2(lx, -ly);
      pts.add(_Samp(th: th, r: h.t, kind: h.kind, lx: lx, ly: ly));
    }
    pts.sort((a, b) => a.th.compareTo(b.th));

    // If you only want forward hemisphere, filter here (ly < 0 keeps forward).
    final kept = pts.where((p) => p.ly < 0).toList();
    if (kept.length < 5) return const [];

    // --- Smooth ranges slightly (median-of-3) to reduce jitter ---
    final n = kept.length;
    final rSm = List<double>.filled(n, 0);
    for (int i = 0; i < n; i++) {
      if (i == 0 || i == n - 1) {
        rSm[i] = kept[i].r;
      } else {
        final a = kept[i - 1].r, b = kept[i].r, c3 = kept[i + 1].r;
        rSm[i] = _median3(a, b, c3);
      }
    }

    // --- Compute central-difference |dr/dθ| (edge strength) ---
    final dAbs = List<double>.filled(n, 0);
    double maxAbs = 1e-9;
    for (int i = 1; i < n - 1; i++) {
      final r0 = rSm[i - 1], r2 = rSm[i + 1];
      final th0 = kept[i - 1].th, th2 = kept[i + 1].th;
      final dth = (th2 - th0).abs();
      if (dth < 1e-9) continue;
      final v = (r2 - r0).abs() / dth;
      dAbs[i] = v;
      if (v > maxAbs) maxAbs = v;
    }

    // --- Deep threshold: dynamic (median + k*MAD) OR fraction of max ---
    final med = _median(rSm);
    final mad = _median(rSm.map((v) => (v - med).abs()).toList());
    var dyn = med + 2.5 * mad;       // adaptive “definitely deeper than typical”
    final frac = 0.60 * (rSm.reduce(math.max)); // also consider very deep absolute
    if (deepQuantileOverride != null) {
      final qq = deepQuantileOverride.clamp(0.0, 1.0);
      final sorted = List<double>.from(rSm)..sort();
      final idx = (qq * (sorted.length - 1)).round();
      dyn = math.min(dyn, sorted[idx]);  // pick the lower of (dyn, quantile)
    }
    final deepThresh = math.max(dyn, frac);

    // --- Find spans where r >= deepThresh, merge across single shallow gaps,
    //     and clip/expand to nearby |dr/dθ| spikes to respect edges. ---
    final spans = <_Span>[];
    int i = 0;
    while (i < n) {
      // skip until deep
      while (i < n && rSm[i] < deepThresh) i++;
      if (i >= n) break;
      int j = i;
      int shallowAllowance = 1; // allow one shallow sample inside the span
      while (j < n) {
        if (rSm[j] >= deepThresh) {
          j++;
        } else if (shallowAllowance > 0) {
          shallowAllowance--;
          j++; // tolerate one
        } else {
          break;
        }
      }

      // clip to nearby jump edges if any (use jumpThresh, but if maxAbs is small,
      // accept a lower fraction to still allow detection in synthetic data)
      final double jump = (jumpOverride != null)
          ? jumpOverride
          : (maxAbs.isFinite && maxAbs > 0)
          ? math.min(jumpThresh, math.max(0.25 * maxAbs, jumpThresh))
          : jumpThresh;

      int left = i;
      for (int k = i; k > 1; k--) {
        if (dAbs[k] >= jump) { left = k; break; }
      }
      int right = j - 1;
      for (int k = j - 1; k < n - 1; k++) {
        if (dAbs[k] >= jump) { right = k; break; }
      }

      final th0 = kept[left].th;
      final th1 = kept[right].th;
      final width = th1 - th0;
      final sampCount = (right - left + 1);

      if (width >= minVoidSpanRad && sampCount >= minSpanSamples) {
        // representative depth (p75 inside the span)
        final sub = rSm.sublist(left, right + 1)..sort();
        final depth = sub[(sub.length * 3) ~/ 4];

        // centroid (body frame)
        double sx = 0, sy = 0;
        int cc = 0;
        for (int k = left; k <= right; k++) {
          sx += kept[k].lx;
          sy += kept[k].ly;
          cc++;
        }
        final cx = sx / cc, cy = sy / cc;

        // score: normalized width * normalized depth, boosted a little
        final widthNorm = (width / (math.pi)).clamp(0.0, 1.0); // vs 180°
        final depthNorm = ((depth - med) / (1e-9 + (rSm.reduce(math.max) - med))).clamp(0.0, 1.0);
        double score = (0.55 * widthNorm + 0.45 * depthNorm);
        score = (score * voidBoost).clamp(0.0, 1.0);

        spans.add(_Span(
          thetaStart: th0,
          thetaEnd: th1,
          width: width,
          depth: depth,
          score: score,
          cx: cx,
          cy: cy,
        ));
      }

      i = j + 1;
    }

    // Merge overlapping/adjacent spans (keep best depth/score)
    final merged = _mergeSpans(spans);

    // Map to public type
    return merged
        .map((s) => CavernHypothesis(
      thetaStart: s.thetaStart,
      thetaEnd: s.thetaEnd,
      widthAng: s.width,
      depth: s.depth,
      score: s.score,
      centroidLocal: et.Vector2(s.cx, s.cy),
    ))
        .toList();
  }
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

List<_Span> _mergeSpans(List<_Span> spans) {
  if (spans.isEmpty) return const [];
  spans.sort((a, b) => a.thetaStart.compareTo(b.thetaStart));
  final out = <_Span>[spans.first];
  for (int i = 1; i < spans.length; i++) {
    final prev = out.last;
    final cur = spans[i];
    // consider touching if gap < 1°
    final touching = (cur.thetaStart - prev.thetaEnd) <= (math.pi / 180);
    if (touching || cur.thetaStart <= prev.thetaEnd) {
      // merge
      prev.thetaEnd = math.max(prev.thetaEnd, cur.thetaEnd);
      prev.width = prev.thetaEnd - prev.thetaStart;
      prev.depth = math.max(prev.depth, cur.depth);
      prev.score = math.max(prev.score, cur.score);
      // average centroid (weighted by width)
      final w0 = prev.width.abs();
      final w1 = cur.width.abs();
      final wSum = (w0 + w1).clamp(1e-6, 1e9);
      prev.cx = (prev.cx * w0 + cur.cx * w1) / wSum;
      prev.cy = (prev.cy * w0 + cur.cy * w1) / wSum;
    } else {
      out.add(cur);
    }
  }
  return out;
}
