// lib/dev/carver_bench.dart
import 'dart:io';
import 'dart:math' as math;

import '../engine/types.dart';
import '../engine/polygon_carver.dart';

// Residue descriptor
class _KinkInfo {
  final int index;
  final double angleDeg;
  final double dev;
  _KinkInfo(this.index, this.angleDeg, this.dev);
}

/// One failure record
class BenchFailure {
  final int terrainSeed;
  final int strokeIndex;
  final String reason;
  final Terrain terrainSnapshot;
  BenchFailure({
    required this.terrainSeed,
    required this.strokeIndex,
    required this.reason,
    required this.terrainSnapshot,
  });

  @override
  String toString() =>
      'Failure(seed=$terrainSeed, stroke=$strokeIndex): $reason  '
          '(verts=${terrainSnapshot.poly.outer.length})';
}

/// Aggregate result
class BenchResult {
  final int terrainsTested;
  final int strokesTried;
  final List<BenchFailure> failures;
  BenchResult({
    required this.terrainsTested,
    required this.strokesTried,
    required this.failures,
  });

  bool get ok => failures.isEmpty;

  @override
  String toString() {
    final sb = StringBuffer();
    sb.writeln('Carver Bench: terrains=$terrainsTested, strokes=$strokesTried');
    if (ok) {
      sb.writeln('✅ No failures detected.');
    } else {
      sb.writeln('❌ ${failures.length} failures:');
      for (final f in failures.take(10)) {
        sb.writeln('  - $f');
      }
      if (failures.length > 10) {
        sb.writeln('  ... and ${failures.length - 10} more');
      }
    }
    return sb.toString();
  }
}

/// Run a randomized stress bench of the polygon carver.
/// You can call this from anywhere (e.g. a debug button) and read the returned result.
///
/// Example:
///   final res = await CarverBench.run(worldW: 360, worldH: 640);
///   debugPrint(res.toString());
class CarverBench {
  static Future<BenchResult> run({
    required double worldW,
    required double worldH,
    int terrainCount = 8,       // how many terrains (different seeds)
    int strokesPerTerrain = 60, // how many random brush strokes per terrain
    double minBrush = 14.0,
    double maxBrush = 42.0,
    int baseSeed = 12345,
    double padWidthFactor = 1.0,
    // validators thresholds
    double dupEps = 1e-3,
    double minEdgeLen = 0.75,
  }) async {
    final rnd = math.Random(baseSeed);
    final failures = <BenchFailure>[];

    for (int t = 0; t < terrainCount; t++) {
      final seed = rnd.nextInt(1 << 30);
      Terrain terrain = Terrain.generate(worldW, worldH, seed, padWidthFactor);

      for (int s = 0; s < strokesPerTerrain; s++) {
        // Random brush center within world; bias towards “ground area”
        final cx = rnd.nextDouble() * worldW;
        // try to pick a y that often intersects ground (lower half)
        final cy = (worldH * (0.45 + 0.5 * rnd.nextDouble())).clamp(0.0, worldH);
        final r = minBrush + rnd.nextDouble() * (maxBrush - minBrush);

        try {
          final next = TerrainCarver.carveCircle(
            terrain: terrain,
            cx: cx,
            cy: cy,
            r: r,
            maxArcErrorPx: 1.25,
            projectEpsPx: 0.9,
            simplifyEpsPx: 0.65,
            rdpTolerancePx: 0.9,
          );

          // Validate polygon health after carve
          final reason = _validateTerrain(next,
              dupEps: dupEps, minEdgeLen: minEdgeLen);
          if (reason != null) {
            failures.add(BenchFailure(
              terrainSeed: seed,
              strokeIndex: s,
              reason: reason,
              terrainSnapshot: next,
            ));
            // stop this terrain early; move to next seed
            break;
          }

          terrain = next;
        } catch (e, st) {
          failures.add(BenchFailure(
            terrainSeed: seed,
            strokeIndex: s,
            reason: 'Exception: $e\n$st',
            terrainSnapshot: terrain,
          ));
          break;
        }
      }
    }

    return BenchResult(
      terrainsTested: terrainCount,
      strokesTried: terrainCount * strokesPerTerrain,
      failures: failures,
    );
  }

  /// Validate polygon sanity. Returns null if OK, otherwise a short reason.
  static String? _validateTerrain(Terrain t,
      {required double dupEps, required double minEdgeLen}) {
    final ring = t.poly.outer;
    if (ring.length < 3) return 'degenerate: <3 vertices';

    if (_samePt(ring.first, ring.last, dupEps)) {
      return 'ring stores duplicate closing vertex';
    }

    final area2 = _signedArea2(ring);
    if (area2 <= 0) return 'non-CCW orientation';

    // duplicates / zero-length edges
    for (int i = 0; i < ring.length; i++) {
      final a = ring[i];
      final b = ring[(i + 1) % ring.length];
      if (_samePt(a, b, dupEps)) return 'duplicate / zero-length edge at $i';
    }

    // short edges
    for (int i = 0; i < ring.length; i++) {
      final a = ring[i];
      final b = ring[(i + 1) % ring.length];
      final dx = b.x - a.x, dy = b.y - a.y;
      final L = math.sqrt(dx * dx + dy * dy);
      if (L < minEdgeLen) return 'short edge ($L) at $i';
    }

    // self-intersections
    if (_hasSelfIntersections(ring, tol: dupEps)) {
      return 'self-intersection detected';
    }

    // --- NEW: micro-residue / kink detection ---
    final kink = _findMicroResidue(ring,
        angleDegThresh: 2.0,   // tiny corner
        devEps: dupEps * 6.0,  // near-collinear deviation
        edgeMax: 3.0);         // only care if neighboring edges are tiny-ish
    if (kink != null) {
      return 'micro-residue kink at ${kink.index} '
          '(angleDeg=${kink.angleDeg.toStringAsFixed(2)}, '
          'dev=${kink.dev.toStringAsFixed(3)})';
    }

    return null;
  }

  static _KinkInfo? _findMicroResidue(
      List<Vector2> ring, {
        required double angleDegThresh,
        required double devEps,
        required double edgeMax,
      }) {
    // angle at B of triangle A-B-C; small angle + small deviation = residue
    double angleAt(Vector2 a, Vector2 b, Vector2 c) {
      double dot = (a.x - b.x) * (c.x - b.x) + (a.y - b.y) * (c.y - b.y);
      double la = math.sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
      double lc = math.sqrt((c.x - b.x)*(c.x - b.x) + (c.y - b.y)*(c.y - b.y));
      if (la < 1e-12 || lc < 1e-12) return 0.0;
      double cos = (dot / (la * lc)).clamp(-1.0, 1.0);
      return math.acos(cos) * 180.0 / math.pi;
    }

    double segLen(Vector2 u, Vector2 v) {
      final dx = v.x - u.x, dy = v.y - u.y;
      return math.sqrt(dx*dx + dy*dy);
    }

    double pointLineDev(Vector2 p, Vector2 a, Vector2 b) {
      // distance from p to line ab
      final vx = b.x - a.x, vy = b.y - a.y;
      final wx = p.x - a.x, wy = p.y - a.y;
      final area2 = (vx * wy - vy * wx).abs();
      final L = math.sqrt(vx*vx + vy*vy);
      if (L < 1e-12) return segLen(p, a);
      return area2 / L;
    }

    for (int i = 0; i < ring.length; i++) {
      final a = ring[(i - 1 + ring.length) % ring.length];
      final b = ring[i];
      final c = ring[(i + 1) % ring.length];
      final ang = angleAt(a, b, c);
      final dev = pointLineDev(b, a, c);
      final la = segLen(a, b), lc = segLen(b, c);

      if (ang <= angleDegThresh && dev <= devEps && la <= edgeMax && lc <= edgeMax) {
        return _KinkInfo(i, ang, dev);
      }
    }
    return null;
  }

  static bool _samePt(Vector2 a, Vector2 b, double eps) =>
      (a.x - b.x).abs() <= eps && (a.y - b.y).abs() <= eps;

  static double _signedArea2(List<Vector2> ring) {
    double a2 = 0;
    for (int i = 0; i < ring.length; i++) {
      final p = ring[i];
      final q = ring[(i + 1) % ring.length];
      a2 += p.x * q.y - p.y * q.x;
    }
    return a2; // 2*area (sign is orientation)
  }

  static bool _hasSelfIntersections(List<Vector2> ring, {double tol = 1e-9}) {
    // segment i: (pi -> pi1)
    for (int i = 0; i < ring.length; i++) {
      final pi = ring[i];
      final pi1 = ring[(i + 1) % ring.length];

      for (int j = i + 1; j < ring.length; j++) {
        // skip neighbors (share a vertex) and the (first,last) wrap
        final isNeighbor = (j == i) ||
            (j == (i + 1) % ring.length) ||
            ((i == 0) && (j == ring.length - 1));
        if (isNeighbor) continue;

        final pj = ring[j];
        final pj1 = ring[(j + 1) % ring.length];

        if (_segmentsIntersect(pi, pi1, pj, pj1, tol)) {
          return true;
        }
      }
    }
    return false;
  }

  static bool _segmentsIntersect(
      Vector2 a, Vector2 b, Vector2 c, Vector2 d, double tol) {
    // Proper intersection with robust orientation tests;
    final o1 = _orient(a, b, c);
    final o2 = _orient(a, b, d);
    final o3 = _orient(c, d, a);
    final o4 = _orient(c, d, b);

    // general case
    if (o1 * o2 < 0 && o3 * o4 < 0) return true;

    // collinear overlaps (treat as intersection if they overlap interiorly)
    if (o1.abs() <= tol && _onSeg(a, b, c, tol)) return true;
    if (o2.abs() <= tol && _onSeg(a, b, d, tol)) return true;
    if (o3.abs() <= tol && _onSeg(c, d, a, tol)) return true;
    if (o4.abs() <= tol && _onSeg(c, d, b, tol)) return true;

    return false;
  }

  static double _orient(Vector2 a, Vector2 b, Vector2 c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  }

  static bool _onSeg(Vector2 a, Vector2 b, Vector2 p, double tol) {
    final minx = math.min(a.x, b.x) - tol, maxx = math.max(a.x, b.x) + tol;
    final miny = math.min(a.y, b.y) - tol, maxy = math.max(a.y, b.y) + tol;
    if (p.x < minx || p.x > maxx || p.y < miny || p.y > maxy) return false;
    // collinearity assumed by caller; bounding box check is enough here
    return true;
  }

  /// Convenience: run and log to console.
  static Future<void> runAndLog({
    required double worldW,
    required double worldH,
    int terrainCount = 8,
    int strokesPerTerrain = 60,
  }) async {
    final res = await run(
      worldW: worldW,
      worldH: worldH,
      terrainCount: terrainCount,
      strokesPerTerrain: strokesPerTerrain,
    );
    print(res.toString());
  }
}

void main(List<String> args) async {
  final opts = _parseArgs(args);

  if (opts['help'] == true) {
    _printUsage();
    exit(0);
  }

  final double worldW = (opts['w'] ?? 360).toDouble();
  final double worldH = (opts['h'] ?? 640).toDouble();
  final int terrains = opts['terrains'] ?? 8;
  final int strokes = opts['strokes'] ?? 60;
  final int baseSeed = opts['seed'] ?? 12345;
  final double minBrush = (opts['minBrush'] ?? 14.0).toDouble();
  final double maxBrush = (opts['maxBrush'] ?? 42.0).toDouble();
  final double padWidthFactor = (opts['padWidthFactor'] ?? 1.0).toDouble();

  // Run bench
  final res = await CarverBench.run(
    worldW: worldW,
    worldH: worldH,
    terrainCount: terrains,
    strokesPerTerrain: strokes,
    baseSeed: baseSeed,
    minBrush: minBrush,
    maxBrush: maxBrush,
    padWidthFactor: padWidthFactor,
  );

  // Print human-readable summary
  print(res.toString());

  // If there are failures, print one reproducible hint
  if (!res.ok && res.failures.isNotEmpty) {
    final f = res.failures.first;
    print('\nRepro tip: use terrain seed ${f.terrainSeed} and inspect after stroke ${f.strokeIndex}.');
  }

  // Return non-zero exit code on failure (useful in CI)
  exit(res.ok ? 0 : 1);
}

Map<String, dynamic> _parseArgs(List<String> args) {
  final map = <String, dynamic>{};
  final it = args.iterator;

  bool take(String k) {
    return k.startsWith('--');
  }

  while (it.moveNext()) {
    final a = it.current;
    if (a == '--help' || a == '-h') {
      map['help'] = true;
      continue;
    }
    if (!a.startsWith('--')) {
      stderr.writeln('Unexpected arg: $a');
      map['help'] = true;
      continue;
    }
    final eq = a.indexOf('=');
    String key, val;
    if (eq >= 0) {
      key = a.substring(2, eq);
      val = a.substring(eq + 1);
    } else {
      key = a.substring(2);
      if (!it.moveNext()) {
        stderr.writeln('Missing value for --$key');
        map['help'] = true;
        break;
      }
      val = it.current;
    }

    dynamic parsed = val;
    // try int
    final asInt = int.tryParse(val);
    if (asInt != null) {
      parsed = asInt;
    } else {
      // try double
      final asDouble = double.tryParse(val);
      if (asDouble != null) parsed = asDouble;
    }
    map[key] = parsed;
  }
  return map;
}

void _printUsage() {
  print('''
Carver Bench CLI

Usage:
  dart run bin/carver_bench.dart [options]

Options:
  --w <double>                World width (default 360)
  --h <double>                World height (default 640)
  --terrains <int>            Number of terrains/seeds to test (default 8)
  --strokes <int>             Random strokes per terrain (default 60)
  --seed <int>                Base RNG seed (default 12345)
  --minBrush <double>         Min brush radius (default 14)
  --maxBrush <double>         Max brush radius (default 42)
  --padWidthFactor <double>   Terrain pad width factor (default 1.0)
  -h, --help                  Show this help

Exit code:
  0  if all tests passed
  1  if any failures were detected
''');
}
