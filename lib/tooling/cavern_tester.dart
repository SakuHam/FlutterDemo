// lib/tooling/cavern_tester.dart
import 'dart:math' as math;

import '../engine/types.dart' as et;
import '../engine/raycast.dart';
import '../ai/cavern_detector.dart';

void main(List<String> args) {
  // ---- tiny manual flag parser (no package:args) ----
  final flags = _parseFlags(args);
  double jumpThresh = _asDouble(flags['jump'], 220.0);
  double minSpanDeg = _asDouble(flags['minspan'], 3.75);
  int minSamples = _asInt(flags['minsamples'], 3);
  double voidBoost = _asDouble(flags['void'], 1.20);
  final dumpCsv = flags.containsKey('dump');

  final minSpanRad = minSpanDeg * math.pi / 180.0;

  // ---- Lander pose (world) ----
  final lander = _makeLander(x: 400, y: 400, angle: 0.0);

  // ---- Synthetic ray fan (body -90°..+90°) ----
  const int rayCount = 180;
  const double thMin = -math.pi / 2;
  const double thMax =  math.pi / 2;
  const double nearCeiling = 300.0;
  const double deepInVoid  = 900.0;
  const double gapStartDeg = -20.0;
  const double gapEndDeg   =  20.0;
  const double wallNear    = 260.0;

  final rays = _buildSyntheticRays(
    lander: lander,
    rayCount: rayCount,
    thMin: thMin,
    thMax: thMax,
    nearCeiling: nearCeiling,
    deepInVoid: deepInVoid,
    gapStartDeg: gapStartDeg,
    gapEndDeg: gapEndDeg,
    wallNear: wallNear,
  );

  // ---- Quick stats over the fan ----
  final stats = _fanStats(rays: rays, lander: lander);
  print('=== Synthetic case ===');
  print('Rays: $rayCount');
  print('Forward kept: ${stats.forwardCount}/${rays.length} '
      '(${(100.0*stats.forwardCount/rays.length).toStringAsFixed(1)}%)');
  print('Range[min..max]: ${stats.rMin.toStringAsFixed(0)}..${stats.rMax.toStringAsFixed(0)}');
  print('Max|dr/dθ|: ${stats.maxAbsDrOverDth.toStringAsFixed(1)}');

  // Show first 25 rays for eyeballing
  print('\nθ(deg)    r(px)   kind');
  for (int i = 0; i < math.min(25, rays.length); i++) {
    final th = _thetaBody(lander, rays[i].p);
    final r  = rays[i].t; // IMPORTANT: t is range along ray
    final kind = rays[i].kind.name;
    print('${(th*180/math.pi).toStringAsFixed(1)}  '
        '${r.toStringAsFixed(0).padLeft(6)}  $kind');
  }

  if (dumpCsv) {
    print('\n#CSV theta_deg,range,kind');
    for (final h in rays) {
      final th = _thetaBody(lander, h.p) * 180.0 / math.pi;
      final r  = h.t;
      print('${th.toStringAsFixed(3)},${r.toStringAsFixed(3)},${h.kind.name}');
    }
    print('#ENDCSV\n');
  }

  // ---- Run your detector ----
  final detector = CavernDetector(
    jumpThresh: jumpThresh,
    voidBoost: voidBoost,
    minVoidSpanRad: minSpanRad,
    minSpanSamples: minSamples,
  );

  final hyps = detector.detect(rays: rays, lander: lander);

  print('CAVERNS (CavernDetector): ${hyps.length}${hyps.isEmpty ? " (none detected)" : ""}');
  for (int i = 0; i < hyps.length; i++) {
    final h = hyps[i];
    final widthDeg = h.widthAng * 180.0 / math.pi;
    print('  #$i: θ=[${(h.thetaStart*180/math.pi).toStringAsFixed(1)}, '
        '${(h.thetaEnd*180/math.pi).toStringAsFixed(1)}]  '
        'width=${widthDeg.toStringAsFixed(1)}°  depth=${h.depth.toStringAsFixed(0)}  '
        'score=${(100*h.score).round()}  center=(${h.centroidLocal.x.toStringAsFixed(1)},'
        '${h.centroidLocal.y.toStringAsFixed(1)})');
  }

  // ---- Run a simple fallback heuristic to sanity-check the scene ----
  final simple = _simpleCavernScan(
    rays: rays,
    lander: lander,
    deepThreshold: (nearCeiling + deepInVoid) / 2,   // halfway between near & void
    minSpanRad: minSpanRad,
  );

  if (simple.isEmpty) {
    print('\n[SimpleScan] no cavern spans found (this would be surprising for the given scene).');
  } else {
    print('\n[SimpleScan] cavern-like spans: ${simple.length}');
    for (final s in simple) {
      print('  θ=[${(s[0]*180/math.pi).toStringAsFixed(1)}, '
          '${(s[1]*180/math.pi).toStringAsFixed(1)}]  '
          'width=${((s[1]-s[0])*180/math.pi).toStringAsFixed(1)}°');
    }
  }

  if (hyps.isEmpty && simple.isNotEmpty) {
    print('\nHint: The simple scan believes there is a deep void span, '
        'but CavernDetector returned none.\n'
        '- Lower --jump (e.g. --jump=120)\n'
        '- Lower --minspan a bit (e.g. --minspan=2.0)\n'
        '- Verify your CavernDetector filters (forward-only? kind filtering?)\n'
        '- Ensure rays are sorted by θ internally (the detector might expect that).\n');
  } else if (hyps.isEmpty) {
    print('\nHint: No caverns found. Try lowering --jump or increasing the deep range.\n'
        'Current jumpThresh: ${jumpThresh.toStringAsFixed(1)}  '
        'Observed Max|dr/dθ|: ${stats.maxAbsDrOverDth.toStringAsFixed(1)}');
  }
}

// ---------------------------------------------------------------------------
// lightweight flag parsing: supports --k=v and bare flags like --dump
Map<String, String?> _parseFlags(List<String> args) {
  final Map<String, String?> out = {};
  for (final a in args) {
    if (!a.startsWith('--')) continue;
    final s = a.substring(2);
    final eq = s.indexOf('=');
    if (eq == -1) {
      out[s] = 'true';
    } else {
      final k = s.substring(0, eq);
      final v = s.substring(eq + 1);
      out[k] = v;
    }
  }
  return out;
}

double _asDouble(String? s, double dflt) {
  final v = double.tryParse((s ?? '').trim());
  return v ?? dflt;
}

int _asInt(String? s, int dflt) {
  final v = int.tryParse((s ?? '').trim());
  return v ?? dflt;
}

// ---------------------------------------------------------------------------
// Data generation
List<RayHit> _buildSyntheticRays({
  required et.LanderState lander,
  required int rayCount,
  required double thMin,
  required double thMax,
  required double nearCeiling,
  required double deepInVoid,
  required double gapStartDeg,
  required double gapEndDeg,
  required double wallNear,
}) {
  final double thStep = (thMax - thMin) / (rayCount - 1);
  final gapStart = gapStartDeg * math.pi / 180.0;
  final gapEnd   = gapEndDeg   * math.pi / 180.0;

  final rays = <RayHit>[];

  final cW = math.cos(lander.angle);
  final sW = math.sin(lander.angle);

  for (int i = 0; i < rayCount; i++) {
    final th = thMin + i * thStep;       // body-frame angle
    final deg = th * 180.0 / math.pi;

    double r;
    if (deg.abs() > 75.0) {
      r = wallNear;                       // near side wall -> helps edges
    } else if (th >= gapStart && th <= gapEnd) {
      r = deepInVoid;                     // the void / cavern gap
    } else {
      r = nearCeiling;                    // normal shallow hits
    }

    // body (forward = -Y), then world
    final lx = r * math.sin(th);
    final ly = -r * math.cos(th);

    final wx = cW * lx - sW * ly;
    final wy = sW * lx + cW * ly;

    final x = lander.pos.x + wx;
    final y = lander.pos.y + wy;

    final kind = (deg.abs() > 75.0) ? RayHitKind.wall : RayHitKind.terrain;

    // IMPORTANT: use RayHit constructor that matches your engine:
    // from your snippet it looks like: RayHit(Vector2 p, double t, RayHitKind kind)
    rays.add(RayHit(et.Vector2(x, y), r, kind));
  }

  return rays;
}

// ---------------------------------------------------------------------------
// Stats & helpers

class _FanStats {
  final int forwardCount;
  final double rMin, rMax;
  final double maxAbsDrOverDth;
  _FanStats(this.forwardCount, this.rMin, this.rMax, this.maxAbsDrOverDth);
}

_FanStats _fanStats({required List<RayHit> rays, required et.LanderState lander}) {
  final c = math.cos(-lander.angle), s = math.sin(-lander.angle);
  int fwd = 0;
  final ranges = <double>[];
  final thetas = <double>[];
  double rMin = double.infinity, rMax = -double.infinity;

  for (final h in rays) {
    final dx = h.p.x - lander.pos.x;
    final dy = h.p.y - lander.pos.y;
    final lx = c * dx - s * dy;
    final ly = s * dx + c * dy;
    final r = h.t;          // range/param along ray
    if (ly < 0) fwd++;
    rMin = math.min(rMin, r);
    rMax = math.max(rMax, r);
    ranges.add(r);
    thetas.add(math.atan2(lx, -ly)); // body θ
  }

  double maxAbs = 0.0;
  for (int i = 1; i < ranges.length - 1; i++) {
    final r0 = ranges[i - 1];
    final r2 = ranges[i + 1];
    final th0 = thetas[i - 1];
    final th2 = thetas[i + 1];
    final dth = (th2 - th0).abs();
    if (dth < 1e-9) continue;
    final drdth = (r2 - r0).abs() / dth;
    if (drdth > maxAbs) maxAbs = drdth;
  }

  return _FanStats(fwd, rMin, rMax, maxAbs);
}

double _thetaBody(et.LanderState lander, et.Vector2 p) {
  final dx = p.x - lander.pos.x;
  final dy = p.y - lander.pos.y;
  final c = math.cos(-lander.angle), s = math.sin(-lander.angle);
  final lx = c * dx - s * dy;
  final ly = s * dx + c * dy;
  return math.atan2(lx, -ly); // body heading = 0 points "up"
}

// ---------------------------------------------------------------------------
// Simple fallback scan (for sanity)

List<List<double>> _simpleCavernScan({
  required List<RayHit> rays,
  required et.LanderState lander,
  required double deepThreshold,
  required double minSpanRad,
}) {
  // Sort by body θ
  final items = <_Item>[];
  for (final h in rays) {
    final th = _thetaBody(lander, h.p);
    items.add(_Item(th, h.t));
  }
  items.sort((a, b) => a.th.compareTo(b.th));

  final spans = <List<double>>[];
  int i = 0;
  while (i < items.length) {
    // Start a deep span where r >= deepThreshold
    if (items[i].r >= deepThreshold) {
      final thStart = items[i].th;
      int j = i + 1;
      while (j < items.length && items[j].r >= deepThreshold) {
        j++;
      }
      final thEnd = items[j - 1].th;
      if ((thEnd - thStart) >= minSpanRad) {
        spans.add([thStart, thEnd]);
      }
      i = j;
    } else {
      i++;
    }
  }
  return spans;
}

class _Item {
  final double th, r;
  _Item(this.th, this.r);
}

// ---------------------------------------------------------------------------
// Minimal LanderState construct (adapt to your actual API if needed)
et.LanderState _makeLander({
  required double x,
  required double y,
  required double angle,
}) {
  return et.LanderState(
    pos: et.Vector2(x, y),
    vel: et.Vector2(0, 0),
    angle: angle,
    fuel: 9999,
  );
}
