// lib/ai/potential_field.dart
import 'dart:math' as math;
import '../engine/game_engine.dart' as eng;
import '../engine/types.dart' as et;

/// Potential field: pad is sink (phi ≈ 0), far field is 1.0.
/// Obstacles (terrain/walls) enforce a no-flux boundary by mirroring the cell value.
/// Solve: ∇²φ = 0 on free space, Dirichlet on pad, Neumann (no-flux) at obstacles.
class PotentialField {
  final int nx, ny;
  final double dx, dy;
  final double worldW, worldH;

  final int maxIters;
  final double tol;
  final double omega;

  final List<double> _phi;   // nx*ny
  final List<int> _mask;     // 0=free, 1=pad Dirichlet, 2=obstacle, 3=outer Dirichlet
  double _padPhi = 0.0;
  double _farPhi = 1.0;

  // NEW: for speed shaping (fast far away, slow near sink)
  final double padCx;
  final double padY;
  final double worldDiag;

  PotentialField({
    required this.nx,
    required this.ny,
    required this.worldW,
    required this.worldH,
    this.maxIters = 2000,
    this.tol = 1e-4,
    this.omega = 1.7,

    // new
    required this.padCx,
    required this.padY,
    required this.worldDiag,
  })  : dx = worldW / (nx - 1),
        dy = worldH / (ny - 1),
        _phi = List<double>.filled(nx * ny, 1.0),
        _mask = List<int>.filled(nx * ny, 0);

  int _idx(int i, int j) => j * nx + i;

  // ---- Public helpers for debug rendering / sampling ----
  int get gridNx => nx;
  int get gridNy => ny;
  double get gridDx => dx;
  double get gridDy => dy;
  double get width => worldW;
  double get height => worldH;

  double phiAtIndex(int i, int j) {
    final ii = i.clamp(0, nx - 1);
    final jj = j.clamp(0, ny - 1);
    return _phi[_idx(ii, jj)];
  }

  int maskAtIndex(int i, int j) {
    final ii = i.clamp(0, nx - 1);
    final jj = j.clamp(0, ny - 1);
    return _mask[_idx(ii, jj)];
  }

  /// --- Geometry helpers ---
  static bool _pointInRing(List<et.Vector2> ring, double x, double y) {
    // Ray crossing test
    bool inside = false;
    for (int a = 0, b = ring.length - 1; a < ring.length; b = a++) {
      final ax = ring[a].x.toDouble(), ay = ring[a].y.toDouble();
      final bx = ring[b].x.toDouble(), by = ring[b].y.toDouble();
      final cond = ((ay > y) != (by > y)) &&
          (x < (bx - ax) * (y - ay) / ((by - ay) == 0.0 ? 1e-9 : (by - ay)) + ax);
      if (cond) inside = !inside;
    }
    return inside;
  }

  static bool _pointInPolyWithHoles({
    required List<et.Vector2> outer,
    required List<List<et.Vector2>> holes,
    required double x,
    required double y,
  }) {
    if (outer.isEmpty) return false;
    if (!_pointInRing(outer, x, y)) return false;
    for (final h in holes) {
      if (h.isNotEmpty && _pointInRing(h, x, y)) return false; // inside a hole → not in solid
    }
    return true; // inside outer and not in any hole → solid
  }

  /// Build masks from env with polygon-aware rasterization:
  /// - pad line → Dirichlet (phi=0) snapped to nearest grid row
  /// - obstacle where point (cell center) ∈ outer AND ∉ any hole
  /// - outer boundary → far-field Dirichlet (phi=1)
  /// - hard walls -> mark as obstacle (no-flux)
  void rasterizeFromEnv(eng.GameEngine env, {double padInflateX = 0.0}) {
    // Clear
    for (int k = 0; k < _mask.length; k++) {
      _mask[k] = 0;
      _phi[k] = _farPhi;
    }

    // Outer boundary as Dirichlet (far field)
    for (int i = 0; i < nx; i++) {
      _mask[_idx(i, 0)] = 3;
      _mask[_idx(i, ny - 1)] = 3;
    }
    for (int j = 0; j < ny; j++) {
      _mask[_idx(0, j)] = 3;
      _mask[_idx(nx - 1, j)] = 3;
    }

    final outer = env.terrain.poly.outer;
    final holes = env.terrain.poly.holes;

    // --- Terrain as obstacle: polygon fill (outer minus holes) ---
    // Use cell centers to classify.
    for (int j = 0; j < ny; j++) {
      final y = j * dy;
      for (int i = 0; i < nx; i++) {
        final x = i * dx;
        if (_pointInPolyWithHoles(outer: outer, holes: holes, x: x, y: y)) {
          final k = _idx(i, j);
          _mask[k] = 2; // obstacle
        }
      }
    }

    // --- Pad as Dirichlet (phi=0) exactly on the nearest grid row to padY ---
    final padX1 = env.terrain.padX1.toDouble() - padInflateX;
    final padX2 = env.terrain.padX2.toDouble() + padInflateX;
    final padY  = env.terrain.padY.toDouble();

    final jPad = (padY / dy).round().clamp(0, ny - 1);
    // also include a 1-row fallback if coarse grid (band of up to 2 rows)
    final jPad2 = ((padY + 0.45 * dy) / dy).round().clamp(0, ny - 1);
    final jMin = math.min(jPad, jPad2);
    final jMax = math.max(jPad, jPad2);

    for (int j = jMin; j <= jMax; j++) {
      for (int i = 0; i < nx; i++) {
        final x = i * dx;
        if (x >= padX1 && x <= padX2) {
          final k = _idx(i, j);
          _mask[k] = 1;       // Dirichlet (pad overrides obstacle)
          _phi[k] = _padPhi;  // 0.0
        }
      }
    }

    // Hard walls as obstacle
    if (env.cfg.hardWalls) {
      for (int j = 0; j < ny; j++) {
        _mask[_idx(0, j)] = 2;
        _mask[_idx(nx - 1, j)] = 2;
      }
    }
  }

  /// SOR relaxation with mirrored neighbors for no-flux obstacles
  void solve() {
    final invDx2 = 1.0 / (dx * dx);
    final invDy2 = 1.0 / (dy * dy);
    final denom = 2.0 * (invDx2 + invDy2);

    double maxDelta;
    for (int it = 0; it < maxIters; it++) {
      maxDelta = 0.0;
      for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {
          final k = _idx(i, j);
          final tag = _mask[k];
          if (tag == 1) { _phi[k] = _padPhi; continue; } // pad
          if (tag == 3) { _phi[k] = _farPhi; continue; } // outer
          if (tag == 2) { continue; }                    // obstacle: keep value

          double phiL = _phi[_idx(i - 1, j)];
          double phiR = _phi[_idx(i + 1, j)];
          double phiD = _phi[_idx(i, j - 1)];
          double phiU = _phi[_idx(i, j + 1)];

          if (_mask[_idx(i - 1, j)] == 2) phiL = _phi[k];
          if (_mask[_idx(i + 1, j)] == 2) phiR = _phi[k];
          if (_mask[_idx(i, j - 1)] == 2) phiD = _phi[k];
          if (_mask[_idx(i, j + 1)] == 2) phiU = _phi[k];

          final rhs = (phiL + phiR) * invDx2 + (phiD + phiU) * invDy2;
          final gs = rhs / denom;
          final newVal = (1.0 - omega) * _phi[k] + omega * gs;

          final d = (newVal - _phi[k]).abs();
          if (d > maxDelta) maxDelta = d;
          _phi[k] = newVal;
        }
      }
      if (maxDelta < tol) break;
    }
  }

  // --- Sampling utilities ---
  double samplePhi(double x, double y) {
    final gx = (x / dx).clamp(0.0, nx - 1.0);
    final gy = (y / dy).clamp(0.0, ny - 1.0);
    final i0 = gx.floor();
    final j0 = gy.floor();
    final i1 = math.min(i0 + 1, nx - 1);
    final j1 = math.min(j0 + 1, ny - 1);
    final tx = gx - i0;
    final ty = gy - j0;

    double v(int i, int j) => _phi[_idx(i, j)];

    final v00 = v(i0, j0);
    final v10 = v(i1, j0);
    final v01 = v(i0, j1);
    final v11 = v(i1, j1);

    final a = v00 * (1 - tx) + v10 * tx;
    final b = v01 * (1 - tx) + v11 * tx;
    return a * (1 - ty) + b * ty;
  }

  ({double fx, double fy, double nx, double ny, double mag}) sampleFlow(double x, double y) {
    final gx = (x / dx).clamp(1.0, nx - 2.0);
    final gy = (y / dy).clamp(1.0, ny - 2.0);
    final i = gx.round();
    final j = gy.round();

    double safePhi(int ii, int jj, int ci, int cj) {
      final m = _mask[_idx(ii, jj)];
      if (m == 2) return _phi[_idx(ci, cj)]; // mirror for obstacles
      return _phi[_idx(ii, jj)];
    }

    final pL = safePhi(i - 1, j, i, j);
    final pR = safePhi(i + 1, j, i, j);
    final pD = safePhi(i, j - 1, i, j);
    final pU = safePhi(i, j + 1, i, j);

    final dphidx = (pR - pL) / (2 * dx);
    final dphidy = (pU - pD) / (2 * dy);

    double fx = -dphidx;
    double fy = -dphidy;
    final mag = math.sqrt(fx * fx + fy * fy) + 1e-9;
    return (fx: fx, fy: fy, nx: fx / mag, ny: fy / mag, mag: mag);
  }

  // ---- NEW: distance-shaped target speed (fast far, slow near) ----

  /// Euclidean distance to sink (pad center).
  double distanceToSink(double x, double y) {
    final dx_ = x - padCx;
    final dy_ = y - padY;
    return math.sqrt(dx_ * dx_ + dy_ * dy_);
  }

  /// Speed profile HIGH when far and LOW near the sink.
  /// p = normalized distance in [0,1]; alpha>1 tapers more near sink.
  double _shapedSpeed({
    required double dist,
    double vMinClose = 8.0,
    double vMaxFar = 90.0,
    double alpha = 1.2,
  }) {
    final p = (dist / worldDiag).clamp(0.0, 1.0);
    final w = math.pow(p, alpha).toDouble(); // ~0 near sink, ~1 far
    return vMinClose + (vMaxFar - vMinClose) * w;
  }

  /// Suggested velocity along -∇φ with distance/altitude-shaped magnitude.
  ///
  /// - vMinClose / vMaxFar / alpha: distance taper
  /// - heightSlowdownH / heightSlowdownMin: extra softening as we approach padY
  /// - clampSpeed: final protection cap (kept for back-compat with old callsites)
  ({double vx, double vy}) suggestVelocity(
      double x,
      double y, {
        double vMinClose = 8.0,
        double vMaxFar = 90.0,
        double alpha = 1.2,
        double heightSlowdownH = 120.0,
        double heightSlowdownMin = 0.45,
        double clampSpeed = 9999.0, // very high by default; legacy callers can still cap
      }) {
    final flow = sampleFlow(x, y); // unit downhill
    final dist = distanceToSink(x, y);

    // base speed from distance
    double speed = _shapedSpeed(
      dist: dist,
      vMinClose: vMinClose,
      vMaxFar: vMaxFar,
      alpha: alpha,
    );

    // Optional extra taper based on absolute vertical offset to pad line
    final dyFromPad = (y - padY).abs();
    if (dyFromPad < heightSlowdownH) {
      final a = (dyFromPad / heightSlowdownH).clamp(0.0, 1.0);
      final k = heightSlowdownMin + (1.0 - heightSlowdownMin) * a; // [min,1]
      speed *= k;
    }

    // Final cap (legacy safety)
    if (clampSpeed.isFinite) speed = math.min(speed, clampSpeed);

    return (vx: flow.nx * speed, vy: flow.ny * speed);
  }

  List<double> debugPhiRow(int j) {
    final out = <double>[];
    for (int i = 0; i < nx; i++) out.add(_phi[_idx(i, j)]);
    return out;
  }
}

/// Convenience builder that creates & solves a field for the current env.
PotentialField buildPotentialField(
    eng.GameEngine env, {
      int nx = 160,
      int ny = 120,
      int iters = 1200,
      double omega = 1.7,
      double tol = 1e-4,
    }) {
  final pf = PotentialField(
    nx: nx,
    ny: ny,
    worldW: env.cfg.worldW.toDouble(),
    worldH: env.cfg.worldH.toDouble(),
    maxIters: iters,
    tol: tol,
    omega: omega,

    // NEW: provide sink + scale for speed shaping
    padCx: env.terrain.padCenter.toDouble(),
    padY: env.terrain.padY.toDouble(),
    worldDiag: math.sqrt(
      env.cfg.worldW * env.cfg.worldW + env.cfg.worldH * env.cfg.worldH,
    ),
  );
  pf.rasterizeFromEnv(env, padInflateX: 0.0);
  pf.solve();
  return pf;
}
