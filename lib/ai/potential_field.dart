// lib/ai/potential_field.dart
import 'dart:math' as math;
import '../engine/game_engine.dart' as eng;
import '../engine/types.dart' as et;

/// Potential field: pad is sink (phi ≈ 0), far field is 1.0.
/// Obstacles (terrain/walls) enforce a no-flux boundary by mirroring the cell value.
///
/// Solve: ∇²φ = 0 on free space, Dirichlet on pad, Neumann (no-flux) at obstacles.
/// We iterate Gauss–Seidel with SOR; result is a smooth “downhill to pad” landscape.
/// Use sampleFlow(x,y) to get -∇φ (target velocity direction).
class PotentialField {
  final int nx, ny;          // grid resolution
  final double dx, dy;       // world meters/pixels per cell
  final double worldW, worldH;

  // Solver params
  final int maxIters;
  final double tol;
  final double omega;        // SOR relaxation (1.0 = Gauss–Seidel)

  // Storage
  final List<double> _phi;   // size nx*ny
  final List<int> _mask;     // 0=free, 1=padDirichlet, 2=obstacle(no-flux), 3=outerDirichlet
  double _padPhi = 0.0;
  double _farPhi = 1.0;

  PotentialField({
    required this.nx,
    required this.ny,
    required this.worldW,
    required this.worldH,
    this.maxIters = 2000,
    this.tol = 1e-4,
    this.omega = 1.7,
  })  : dx = worldW / (nx - 1),
        dy = worldH / (ny - 1),
        _phi = List<double>.filled(nx * ny, 1.0),      // start from far field
        _mask = List<int>.filled(nx * ny, 0);

  int _idx(int i, int j) => j * nx + i;

  // ---- Public helpers for debug rendering / sampling ----
  int get gridNx => nx;
  int get gridNy => ny;
  double get gridDx => dx;
  double get gridDy => dy;
  double get width => worldW;
  double get height => worldH;

  /// Safely sample φ at grid index (clamped).
  double phiAtIndex(int i, int j) {
    final ii = i.clamp(0, nx - 1);
    final jj = j.clamp(0, ny - 1);
    return _phi[_idx(ii, jj)];
  }

  /// Mask at grid index (0=free, 1=pad Dirichlet, 2=obstacle, 3=outer Dirichlet).
  int maskAtIndex(int i, int j) {
    final ii = i.clamp(0, nx - 1);
    final jj = j.clamp(0, ny - 1);
    return _mask[_idx(ii, jj)];
  }

  /// Build masks from env:
  /// - pad area → Dirichlet (phi = _padPhi)
  /// - terrain body → obstacle (no-flux)
  /// - hard walls → obstacle
  /// - outer boundary → Dirichlet (phi = _farPhi)
  void rasterizeFromEnv(eng.GameEngine env, {double padInflate = 8.0}) {
    // Clear masks/initialize
    for (int k = 0; k < _mask.length; k++) {
      _mask[k] = 0;
      _phi[k] = _farPhi;
    }

    // Outer boundary as far-field Dirichlet
    for (int i = 0; i < nx; i++) {
      _mask[_idx(i, 0)] = 3;
      _mask[_idx(i, ny - 1)] = 3;
    }
    for (int j = 0; j < ny; j++) {
      _mask[_idx(0, j)] = 3;
      _mask[_idx(nx - 1, j)] = 3;
    }

// --- Pad area as Dirichlet (phi = 0) on the grid row closest to padY ---
    final padX1 = env.terrain.padX1.toDouble();
    final padX2 = env.terrain.padX2.toDouble();
    final padY  = env.terrain.padY.toDouble();

// nearest grid row to the physical pad line
    final jPad = (padY / dy).round().clamp(0, ny - 1);

// ensure we hit the pad even if grid is coarse: use a tiny vertical band
    final jPad2 = ((padY + 0.45 * dy) / dy).round().clamp(0, ny - 1);
    final jPadMin = math.min(jPad, jPad2);
    final jPadMax = math.max(jPad, jPad2);

    for (int j = jPadMin; j <= jPadMax; j++) {
      final y = j * dy;
      // Only accept rows that are not below the terrain at pad (rare, but safe)
      for (int i = 0; i < nx; i++) {
        final x = i * dx;
        if (x >= padX1 && x <= padX2) {
          final k = _idx(i, j);
          _mask[k] = 1;       // Dirichlet (pad)
          _phi[k]  = _padPhi; // 0.0
        }
      }
    }

    // --- Terrain as obstacle (no-flux) ---
    // For each column, mark cells with y >= groundHeight(x) as obstacle.
    // (Pad cells above override to Dirichlet already.)
    for (int i = 0; i < nx; i++) {
      final x = i * dx;
      final gy = env.terrain.heightAt(x);
      final jStart = (gy / dy).floor().clamp(0, ny - 1);
      for (int j = jStart; j < ny; j++) {
        final k = _idx(i, j);
        if (_mask[k] == 1) continue; // keep pad
        _mask[k] = 2;                // obstacle
      }
    }

    // Hard walls as obstacle (in addition to outer Dirichlet markers)
    if (env.cfg.hardWalls) {
      for (int j = 0; j < ny; j++) {
        _mask[_idx(0, j)] = 2;
        _mask[_idx(nx - 1, j)] = 2;
      }
    }
  }

  /// Run SOR relaxation. No-flux at obstacles is imposed by mirroring neighbor values:
  /// if a neighbor is obstacle, we reuse current cell value (zero normal gradient).
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
          if (tag == 1) { // Dirichlet pad
            _phi[k] = _padPhi;
            continue;
          }
          if (tag == 3) { // outer Dirichlet
            _phi[k] = _farPhi;
            continue;
          }
          if (tag == 2) { // obstacle: keep value (acts like Neumann)
            continue;
          }

          double phiL = _phi[_idx(i - 1, j)];
          double phiR = _phi[_idx(i + 1, j)];
          double phiD = _phi[_idx(i, j - 1)];
          double phiU = _phi[_idx(i, j + 1)];

          // If any neighbor is obstacle, treat as mirrored value (no-flux)
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

  /// Bilinear sample φ at world (x,y).
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

  /// Central-difference gradient of φ, then return flow = -∇φ at (x,y) in world units.
  /// Also returns a normalized direction (unitFlow) for convenience.
  ({double fx, double fy, double nx, double ny, double mag}) sampleFlow(double x, double y) {
    // Convert to grid coords
    final gx = (x / dx).clamp(1.0, nx - 2.0);
    final gy = (y / dy).clamp(1.0, ny - 2.0);
    final i = gx.round();
    final j = gy.round();

    double phi(int ii, int jj) => _phi[_idx(ii, jj)];

    // handle obstacles with mirrored values
    double safePhi(int ii, int jj, int ci, int cj) {
      final m = _mask[_idx(ii, jj)];
      if (m == 2) return _phi[_idx(ci, cj)]; // mirror
      return _phi[_idx(ii, jj)];
    }

    final pL = safePhi(i - 1, j, i, j);
    final pR = safePhi(i + 1, j, i, j);
    final pD = safePhi(i, j - 1, i, j);
    final pU = safePhi(i, j + 1, i, j);

    final dphidx = (pR - pL) / (2 * dx);
    final dphidy = (pU - pD) / (2 * dy);

    // Flow is downhill of phi
    double fx = -dphidx;
    double fy = -dphidy;
    final mag = math.sqrt(fx * fx + fy * fy) + 1e-9;
    final nfx = fx / mag;
    final nfy = fy / mag;
    return (fx: fx, fy: fy, nx: nfx, ny: nfy, mag: mag);
  }

  /// Suggest target velocity at (x,y) with a speed schedule:
  /// - farther/higher → faster; near pad → slow.
  /// clampSpeed is maximum allowed magnitude.
  ({double vx, double vy}) suggestVelocity(double x, double y, {double clampSpeed = 90.0}) {
    final flow = sampleFlow(x, y);
    // Simple speed based on potential (higher phi → faster)
    final phi = samplePhi(x, y);       // 0 near pad, ~1 far away
    final speed = (12.0 + 80.0 * phi).clamp(10.0, clampSpeed);
    return (vx: flow.nx * speed, vy: flow.ny * speed);
  }

  /// Debug helper: get φ row as list
  List<double> debugPhiRow(int j) {
    final out = <double>[];
    for (int i = 0; i < nx; i++) out.add(_phi[_idx(i, j)]);
    return out;
  }
}

/// Convenience builder that creates & solves a field for the current env.
/// Call whenever terrain changes (or once per episode if terrain is fixed).
PotentialField buildPotentialField(eng.GameEngine env, {
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
  );
  pf.rasterizeFromEnv(env, padInflate: 10.0);
  pf.solve();
  return pf;
}
