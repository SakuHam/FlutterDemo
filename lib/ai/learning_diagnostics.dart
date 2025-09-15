// lib/ai/learning_diagnostics.dart
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import '../engine/types.dart' as et;
import '../engine/game_engine.dart' as eng;
import 'agent.dart' as ai; // FeatureExtractor, PolicyNetwork, RunningNorm, intents, helpers

/* ------------------------------- tiny helpers -------------------------------- */

double _clip(double x, double a, double b) => x < a ? a : (x > b ? b : x);

/* ----------------------------- policy IO (+norm) ----------------------------- */

bool _normLooksDegenerate(ai.RunningNorm norm) {
  int tinyStd = 0;
  for (int i = 0; i < norm.dim; i++) {
    final v = norm.var_[i];
    if (!v.isFinite || v <= 0) return true;
    final std = math.sqrt(v);
    if (std < 0.15) tinyStd++;
    if (!norm.mean[i].isFinite || norm.mean[i].abs() > 5.0) return true;
  }
  // >25% of dims with tiny std? call it degenerate
  return tinyStd > (norm.dim ~/ 4);
}

void warmLocalNormIfMissing({
  required eng.GameEngine env,
  required ai.FeatureExtractor fe,
  required ai.RunningNorm norm,
  int nSamples = 3500,
  int seed = 424242,
}) {
  final r = math.Random(seed);
  env.reset(seed: 777);
  int accepted = 0;
  // reset local norm
  norm.inited = false;
  norm.mean = List.filled(norm.dim, 0.0);
  norm.var_  = List.filled(norm.dim, 1.0);

  while (accepted < nSamples) {
    final padCx = env.terrain.padCenter.toDouble();
    env.lander.pos.x = (padCx + (r.nextDouble()*400 - 200)).clamp(10.0, env.cfg.worldW - 10.0);
    final gy = env.terrain.heightAt(env.lander.pos.x);
    env.lander.pos.y = (gy - (60 + 300*r.nextDouble())).clamp(0.0, env.cfg.worldH - 10.0);
    env.lander.vel.x = r.nextDouble()*180 - 90;
    env.lander.vel.y = r.nextDouble()*140 + 10;
    env.lander.angle = 0.0;
    env.lander.fuel  = env.cfg.t.maxFuel;

    final x = fe.extract(env);
    norm.normalize(x, update: true); // observe
    accepted++;
  }
  print('Local norm warmed on-the-fly ($accepted samples).');
}

void _printNormSanity(ai.RunningNorm norm) {
  double maxAbsMean = 0, maxStdErr = 0;
  for (int i = 0; i < norm.dim; i++) {
    final std = math.sqrt(math.max(norm.var_[i], 0.0));
    maxAbsMean = math.max(maxAbsMean, norm.mean[i].abs());
    maxStdErr = math.max(maxStdErr, (std - 1.0).abs());
  }
  print('--- Norm sanity ---');
  print('max |mean| over dims = ${maxAbsMean.toStringAsFixed(3)}   max |std-1| = ${maxStdErr.toStringAsFixed(3)}');
}

bool tryLoadPolicyWithNorm(String path, ai.PolicyNetwork p, ai.RunningNorm norm) {
  final f = File(path);
  if (!f.existsSync()) {
    print('No policy file at $path');
    return false;
  }
  dynamic raw;
  try {
    raw = json.decode(f.readAsStringSync());
  } catch (e) {
    print('Failed to parse $path: $e');
    return false;
  }
  if (raw is! Map<String, dynamic>) {
    print('Unexpected JSON root in $path (expected object).');
    return false;
  }
  final m = raw as Map<String, dynamic>;

  List _mat(dynamic v) {
    if (v is List) return v;
    if (v is Map && v['data'] is List) return v['data'] as List;
    throw StateError('matrix field is not a List or {data: List}');
  }

  void _from3(List<List<double>> dst, List src) {
    for (int i = 0; i < dst.length; i++) {
      final ri = dst[i];
      final si = (src[i] as List);
      for (int j = 0; j < ri.length; j++) {
        ri[j] = (si[j] as num).toDouble();
      }
    }
  }

  bool _tryFillMat(List<List<double>> dst, String key) {
    final v = m[key];
    if (v == null) return false;
    try {
      _from3(dst, _mat(v));
      return true;
    } catch (_) {
      return false;
    }
  }

  bool _tryFillVec(List<double> dst, String key) {
    final v = m[key];
    if (v == null) return false;
    try {
      final L = (v as List).map((e) => (e as num).toDouble()).toList();
      final n = math.min(dst.length, L.length);
      for (int i = 0; i < n; i++) dst[i] = L[i];
      return true;
    } catch (_) {
      return false;
    }
  }

  int ok = 0, total = 0;
  total += 2;
  if (_tryFillMat(p.W1, 'W1')) ok++;
  if (_tryFillVec(p.b1, 'b1')) ok++;
  total += 2;
  if (_tryFillMat(p.W2, 'W2')) ok++;
  if (_tryFillVec(p.b2, 'b2')) ok++;
  total += 2;
  if (_tryFillMat(p.W_thr, 'W_thr')) ok++;
  if (_tryFillVec(p.b_thr, 'b_thr')) ok++;
  total += 2;
  if (_tryFillMat(p.W_turn, 'W_turn')) ok++;
  if (_tryFillVec(p.b_turn, 'b_turn')) ok++;
  total += 2;
  if (_tryFillMat(p.W_intent, 'W_intent')) ok++;
  if (_tryFillVec(p.b_intent, 'b_intent')) ok++;
  total += 2;
  if (_tryFillMat(p.W_val, 'W_val')) ok++;
  if (_tryFillVec(p.b_val, 'b_val')) ok++;

  // Normalization (if present). Accept either variance or std.
  final nm = m['norm_mean'];
  final nv = m['norm_var'];
  final ns = m['norm_std'];
  if (nm is List && (nv is List || ns is List)) {
    final srcVar = (nv ?? (ns as List).map((e) {
      final s = (e as num).toDouble();
      return s * s;
    }).toList()) as List;
    final n = math.min(norm.dim, math.min(nm.length, srcVar.length));
    for (int i = 0; i < n; i++) {
      norm.mean[i] = (nm[i] as num).toDouble();
      norm.var_[i] = (srcVar[i] as num).toDouble();
    }
    norm.inited = true;
    if (m['norm_momentum'] is num) {
      norm.momentum = (m['norm_momentum'] as num).toDouble();
    }
    print('Loaded feature norm (dim=$n) from $path');
  }

  print('Loaded $path ($ok/$total tensors filled)');
  return ok > 0;
}

/* --------------------------------- env config -------------------------------- */

/// Keep these identical to training/runtime to avoid eval drift.
/// If thrust “feels” too weak/strong in probes, verify these match your engine.
et.EngineConfig makeConfig({
  int seed = 42,
  bool lockTerrain = false,
  bool lockSpawn = false,
  bool randomSpawnX = true,
  double worldW = 800,
  double worldH = 600,
  double? maxFuel,
  // If you change these in runtime/training, change them here too:
  double gravity = 0.18,
  double thrustAccel = 0.42,
  double rotSpeed = 1.6,
  double stepScale = 60.0,
  bool hardWalls = true,
}) {
  final t = et.Tunables(
    gravity: gravity,
    thrustAccel: thrustAccel,
    rotSpeed: rotSpeed,
    maxFuel: maxFuel ?? 1000.0,
  );
  return et.EngineConfig(
    worldW: worldW,
    worldH: worldH,
    t: t,
    seed: seed,
    stepScale: stepScale,
    lockTerrain: lockTerrain,
    terrainSeed: 1234567,
    lockSpawn: lockSpawn,
    randomSpawnX: randomSpawnX,
    hardWalls: hardWalls,
  );
}

/* ------------------------------- norm utilities ------------------------------ */

/* -------------------------------- probe #1 ----------------------------------- */
/* Intent agreement on randomized snapshots (teacher = predictiveIntentLabelAdaptive)
   Uses EXACT SAME feature normalization as the policy (loaded or warmed).
*/

class _Conf {
  final int K;
  final List<List<int>> mat;
  final List<int> rowsN;
  _Conf(this.K)
      : mat = List.generate(K, (_) => List.filled(K, 0)),
        rowsN = List.filled(K, 0);
  void add(int y, int pred) {
    if (y < 0 || y >= K) return;
    if (pred < 0 || pred >= K) return;
    mat[y][pred] += 1;
    rowsN[y] += 1;
  }
}

void probeIntentAgreement({
  required eng.GameEngine env,
  required ai.FeatureExtractor fe,
  required ai.PolicyNetwork policy,
  required ai.RunningNorm norm,
  int N = 800,
  int seed = 123,
}) {
  final rnd = math.Random(seed);
  final K = ai.PolicyNetwork.kIntents;
  final conf = _Conf(K);
  int correct = 0;

  // keep terrain fixed so labels are stable-ish
  env.reset(seed: 777);

  for (int i = 0; i < N; i++) {
    // Randomize a plausible state quickly
    final padCx = env.terrain.padCenter.toDouble();
    env.lander.pos.x =
        _clip(padCx + (rnd.nextDouble() * 400 - 200), 10.0, env.cfg.worldW - 10.0);
    final gy = env.terrain.heightAt(env.lander.pos.x);
    env.lander.pos.y = _clip(gy - (60 + 300 * rnd.nextDouble()), 0.0, env.cfg.worldH - 10.0);
    env.lander.vel.x = rnd.nextDouble() * 180 - 90;
    env.lander.vel.y = rnd.nextDouble() * 140 + 10; // downward-ish
    env.lander.angle = 0.0;
    env.lander.fuel = env.cfg.t.maxFuel;

    final y = ai.predictiveIntentLabelAdaptive(env,
        baseTauSec: 1.0, minTauSec: 0.45, maxTauSec: 1.35);

    var x = fe.extract(env);
    x = norm.normalize(x, update: false);

    final (pred, _p, _cache) = policy.actIntentGreedy(x);

    if (pred == y) correct++;
    conf.add(y, pred);
  }

  final acc = N == 0 ? 0.0 : correct / N;

  print('--- Probe #1: Intent agreement on snapshots ---');
  print('N=$N  acc=${(acc * 100).toStringAsFixed(1)}%  (rows=teacher, cols=policy)');
  for (int r = 0; r < K; r++) {
    final row = List.generate(K, (c) => conf.mat[r][c].toString().padLeft(5)).join();
    final rowAcc = conf.rowsN[r] == 0 ? 0.0 : conf.mat[r][r] / conf.rowsN[r];
    print('${ai.kIntentNames[r].padRight(10)} |$row   (acc=${(rowAcc * 100).toStringAsFixed(1)}%  n=${conf.rowsN[r]})');
  }
  print('');
}

/* -------------------------------- probe #2 ----------------------------------- */
/* Teacher vs Student at decision states
   - Sample intents with policy intent head (greedy).
   - Build heuristic control for that intent (teacher).
   - Compare to policy action heads (student) from same state (greedy).
*/

void probeTeacherVsStudent({
  required eng.GameEngine env,
  required ai.FeatureExtractor fe,
  required ai.PolicyNetwork policy,
  required ai.RunningNorm norm,
  int maxDecisions = 4000,
  int planHold = 1,
  int seed = 999,
}) {
  final rnd = math.Random(seed);
  int agreeTurn = 0, agreeThrust = 0, agreeBoth = 0, total = 0;
  double thrProbSum = 0.0;
  final intentHist = List<int>.filled(ai.PolicyNetwork.kIntents, 0);

  env.reset(seed: rnd.nextInt(1 << 30));

  int framesLeft = 0;
  int currentIntentIdx = 0;
  int decisions = 0;

  while (decisions < maxDecisions) {
    if (framesLeft <= 0) {
      // decision state features
      var xPlan = fe.extract(env);
      xPlan = norm.normalize(xPlan, update: false);

      final (idx, probs, _cache) = policy.actIntentGreedy(xPlan);

      // simple safety: if fast & close to ground, force brakeUp
      final L = env.lander;
      final groundY = env.terrain.heightAt(L.pos.x);
      final height = groundY - L.pos.y;
      if (height < 80 && L.vel.y > 80) {
        currentIntentIdx = ai.intentToIndex(ai.Intent.brakeUp);
      } else {
        currentIntentIdx = idx;
      }

      // guard against NaNs
      final pSum = probs.fold<double>(0.0, (a, b) => a + (b.isFinite ? b : 0.0));
      if (!pSum.isFinite || pSum <= 0) {
        // fall back: keep intent histogram sane
      }

      intentHist[currentIntentIdx] += 1;
      framesLeft = planHold;
      decisions++;
    }

    // Teacher control (heuristic controller for chosen intent)
    final intent = ai.indexToIntent(currentIntentIdx);
    final uTeacher = ai.controllerForIntent(intent, env);

    // Student control (policy action heads)
    var xAct = fe.extract(env);
    xAct = norm.normalize(xAct, update: false);
    final (th, lf, rt, probs, _c) = policy.actGreedy(xAct);
    final uStudent = et.ControlInput(thrust: th, left: lf, right: rt);

    // Compare
    final tTurnTeacher = (uTeacher.left ? -1 : 0) + (uTeacher.right ? 1 : 0);
    final tTurnStudent = (uStudent.left ? -1 : 0) + (uStudent.right ? 1 : 0);
    if (tTurnTeacher == tTurnStudent) agreeTurn++;
    if (uTeacher.thrust == uStudent.thrust) agreeThrust++;
    if (tTurnTeacher == tTurnStudent && uTeacher.thrust == uStudent.thrust) agreeBoth++;
    total++;

    // Be conservative about which index in probs is thrust; if missing, infer from action.
    double pThr;
    if (probs.isEmpty) {
      pThr = th ? 1.0 : 0.0;
    } else {
      // Most implementations expose p(thrust) in probs[0]. Adjust here if yours differs.
      pThr = probs[0].isFinite ? probs[0] : (th ? 1.0 : 0.0);
    }
    thrProbSum += pThr;

    // Step env with teacher control (evaluation-style)
    final info = env.step(
      1 / 60.0,
      et.ControlInput(
        thrust: uTeacher.thrust,
        left: uTeacher.left,
        right: uTeacher.right,
        intentIdx: currentIntentIdx,
      ),
    );
    framesLeft -= 1;
    if (info.terminal) {
      env.reset(seed: rnd.nextInt(1 << 30));
      framesLeft = 0;
    }
  }

  print('--- Probe #2: Teacher vs Student at decision states (heuristic vs action heads) ---');
  final ap = total == 0 ? 0.0 : thrProbSum / total;
  print('decisions=$total  agreeTurn=${(100.0 * agreeTurn / total).toStringAsFixed(1)}%'
      '  agreeThrust=${(100.0 * agreeThrust / total).toStringAsFixed(1)}%'
      '  agreeBoth=${(100.0 * agreeBoth / total).toStringAsFixed(1)}%'
      '  avgThrProb=${ap.toStringAsFixed(3)}');
  print('intent histogram: $intentHist');
  print('');
}

/* ------------------------------- probe #3 (lite) ------------------------------ */
/* No-update trainability proxy:
   - Collect decision states and teacher labels (predictiveIntentLabelAdaptive).
   - Report cross-entropy and mean teacher-prob under policy.
   This doesn’t backprop (no private caches), but will reveal if predictions are
   badly calibrated against teacher.
*/

class _DecSample {
  final List<double> x; // normalized features
  final int y; // teacher label
  _DecSample(this.x, this.y);
}

double _ceOneHot(List<double> p, int y) {
  final py = _clip(p[y], 1e-8, 1.0);
  return -math.log(py);
}

void probeCECalibration({
  required eng.GameEngine env,
  required ai.FeatureExtractor fe,
  required ai.PolicyNetwork policy,
  required ai.RunningNorm norm,
  int maxDecisions = 1600,
  int planHold = 1,
  int seed = 2025,
}) {
  final rnd = math.Random(seed);
  env.reset(seed: rnd.nextInt(1 << 30));

  final samples = <_DecSample>[];
  int framesLeft = 0;
  int currentIntentIdx = 0;

  while (samples.length < maxDecisions) {
    if (framesLeft <= 0) {
      var xPlan = fe.extract(env);
      final y = ai.predictiveIntentLabelAdaptive(
        env,
        baseTauSec: 1.0,
        minTauSec: 0.45,
        maxTauSec: 1.35,
      );
      xPlan = norm.normalize(xPlan, update: false);
      samples.add(_DecSample(xPlan, y));

      // advance using policy-picked intent to gather diverse states
      final (idx, _p, _c) = policy.actIntentGreedy(xPlan);
      currentIntentIdx = idx;
      framesLeft = planHold;
    }

    final intent = ai.indexToIntent(currentIntentIdx);
    final u = ai.controllerForIntent(intent, env);
    final info = env.step(
      1 / 60.0,
      et.ControlInput(
        thrust: u.thrust,
        left: u.left,
        right: u.right,
        intentIdx: currentIntentIdx,
      ),
    );
    framesLeft -= 1;
    if (info.terminal) {
      env.reset(seed: rnd.nextInt(1 << 30));
      framesLeft = 0;
    }
  }

  double ce = 0.0, meanP = 0.0;
  for (final s in samples) {
    final (_pred, p, _c) = policy.actIntentGreedy(s.x);
    ce += _ceOneHot(p, s.y);
    meanP += p[s.y];
  }
  ce /= samples.length;
  meanP /= samples.length;

  print('--- Probe #3: CE calibration (teacher vs policy, no update) ---');
  print(
      'Batch size: ${samples.length}  mean CE=${ce.toStringAsFixed(4)}  meanP(teacher)=${meanP.toStringAsFixed(3)}');
  print('');
}

/* ------------------------------- probe #4 ----------------------------------- */
/* DescendSlow failure analysis:
   - Random snapshots → keep only teacher=descendSlow.
   - Slice by bins of |dx|, height, vy (matching teacher thresholds).
   - Show accuracy within each bin and which wrong class dominates.
*/

class _BinStats {
  int n = 0, correct = 0;
  final wrong = List<int>.filled(ai.PolicyNetwork.kIntents, 0);
  double sumPDesc = 0.0;
}

void probeDescendSlowAnalysis({
  required eng.GameEngine env,
  required ai.FeatureExtractor fe,
  required ai.PolicyNetwork policy,
  required ai.RunningNorm norm,
  int N = 4000,
  int seed = 42424,
}) {
  final r = math.Random(seed);
  env.reset(seed: 777);

  // bins
  final dxBins = <String, bool Function(double)>{
    '<0.05W'   : (u) => u < 0.05,
    '0.05–0.10W': (u) => u >= 0.05 && u < 0.10,
    '0.10–0.15W': (u) => u >= 0.10 && u < 0.15,
  };
  final hBins = <String, bool Function(double)>{
    '120–180' : (h) => h >= 120 && h < 180,
    '180–260' : (h) => h >= 180 && h < 260,
    '260+'    : (h) => h >= 260,
  };
  final vyBins = <String, bool Function(double)>{
    '20–28' : (v) => v >= 20 && v < 28,
    '28–36' : (v) => v >= 28 && v < 36,
    '36+'   : (v) => v >= 36,
  };

  // 3D map: dxBin → hBin → vyBin → stats
  final stats = <String, Map<String, Map<String, _BinStats>>>{};
  for (final dk in dxBins.keys) {
    stats[dk] = {};
    for (final hk in hBins.keys) {
      stats[dk]![hk] = {};
      for (final vk in vyBins.keys) {
        stats[dk]![hk]![vk] = _BinStats();
      }
    }
  }

  int kept = 0, drawn = 0;

  while (drawn < N) {
    drawn++;
    final padCx = env.terrain.padCenter.toDouble();

    // Randomize plausible snapshot
    env.lander.pos.x = _clip(padCx + (r.nextDouble()*400 - 200), 10.0, env.cfg.worldW - 10.0);
    final gy = env.terrain.heightAt(env.lander.pos.x);
    env.lander.pos.y = _clip(gy - (60 + 300*r.nextDouble()), 0.0, env.cfg.worldH - 10.0);
    env.lander.vel.x = r.nextDouble()*180 - 90;
    env.lander.vel.y = r.nextDouble()*140 + 10; // downward-ish
    env.lander.angle = 0.0;
    env.lander.fuel  = env.cfg.t.maxFuel;

    final yTeach = ai.predictiveIntentLabelAdaptive(env,
        baseTauSec: 1.0, minTauSec: 0.45, maxTauSec: 1.35);
    if (yTeach != ai.intentToIndex(ai.Intent.descendSlow)) continue; // keep only teacher=descendSlow

    // compute features to slice
    final px = env.lander.pos.x.toDouble();
    final py = env.lander.pos.y.toDouble();
    final dxAbs = (px - padCx).abs();
    final dxU = dxAbs / env.cfg.worldW; // fraction of W
    final height = env.terrain.heightAt(px) - py;
    final vy = env.lander.vel.y.toDouble();

    String? dKey, hKey, vKey;
    dxBins.forEach((k, f) { if (dKey == null && f(dxU)) dKey = k; });
    hBins.forEach((k, f) { if (hKey == null && f(height)) hKey = k; });
    vyBins.forEach((k, f) { if (vKey == null && f(vy)) vKey = k; });
    if (dKey == null || hKey == null || vKey == null) continue;

    // model prediction
    var x = fe.extract(env);
    x = norm.normalize(x, update: false);
    final (pred, p, _c) = policy.actIntentGreedy(x);

    final st = stats[dKey]![hKey]![vKey]!;
    st.n += 1;
    if (pred == yTeach) st.correct += 1; else st.wrong[pred] += 1;
    st.sumPDesc += p[ai.intentToIndex(ai.Intent.descendSlow)];
    kept++;
  }

  print('--- Probe #4: DescendSlow failure analysis (teacher=descendSlow only) ---');
  print('Kept samples: $kept');

  String _topWrong(List<int> wrong) {
    int best = -1, idx = -1;
    for (int i = 0; i < wrong.length; i++) {
      if (wrong[i] > best) { best = wrong[i]; idx = i; }
    }
    if (best <= 0) return '-';
    return '${ai.kIntentNames[idx]}($best)';
  }

  for (final dk in dxBins.keys) {
    print('dx bin = $dk');
    for (final hk in hBins.keys) {
      final row = <String>[];
      for (final vk in vyBins.keys) {
        final st = stats[dk]![hk]![vk]!;
        if (st.n == 0) { row.add('[n=0]'); continue; }
        final acc = 100.0 * st.correct / st.n;
        final pDesc = st.sumPDesc / st.n;
        final topW = _topWrong(st.wrong);
        row.add('[n=${st.n} acc=${acc.toStringAsFixed(1)}% pDesc=${pDesc.toStringAsFixed(2)} wrong=$topW]');
      }
      print('  h=$hk  |  ${row.join('  ')}');
    }
  }
  print('');
}

/* ------------------------------- probe #5 ----------------------------------- */
/* Intent head stats:
   - Average probabilities per intent (global bias).
   - Margin histogram: p(teacher) - max(other): how decisive the head is.
*/

void probeIntentHeadStats({
  required eng.GameEngine env,
  required ai.FeatureExtractor fe,
  required ai.PolicyNetwork policy,
  required ai.RunningNorm norm,
  int N = 5000,
  int seed = 10101,
}) {
  final r = math.Random(seed);
  env.reset(seed: 123);

  final K = ai.PolicyNetwork.kIntents;
  final probSum = List<double>.filled(K, 0.0);
  final countTeach = List<int>.filled(K, 0);
  final margins = <double>[];

  for (int i = 0; i < N; i++) {
    final padCx = env.terrain.padCenter.toDouble();
    env.lander.pos.x = _clip(padCx + (r.nextDouble()*400 - 200), 10.0, env.cfg.worldW - 10.0);
    final gy = env.terrain.heightAt(env.lander.pos.x);
    env.lander.pos.y = _clip(gy - (60 + 300*r.nextDouble()), 0.0, env.cfg.worldH - 10.0);
    env.lander.vel.x = r.nextDouble()*180 - 90;
    env.lander.vel.y = r.nextDouble()*140 + 10;
    env.lander.angle = 0.0;
    env.lander.fuel  = env.cfg.t.maxFuel;

    final y = ai.predictiveIntentLabelAdaptive(env,
        baseTauSec: 1.0, minTauSec: 0.45, maxTauSec: 1.35);
    countTeach[y]++;

    var x = fe.extract(env);
    x = norm.normalize(x, update: false);
    final (_pred, p, _c) = policy.actIntentGreedy(x);
    for (int k = 0; k < K; k++) probSum[k] += p[k];

    // margin wrt teacher
    double bestOther = -1e9;
    for (int k = 0; k < K; k++) if (k != y) bestOther = math.max(bestOther, p[k]);
    final m = p[y] - bestOther;
    margins.add(m);
  }

  final probAvg = probSum.map((s) => s / N).toList();
  double pct(double lo) => 100.0 * margins.where((m) => m < lo).length / margins.length;

  print('--- Probe #5: Intent head calibration & margins ---');
  print('Avg probs per intent:');
  for (int k = 0; k < K; k++) {
    print('  ${ai.kIntentNames[k].padRight(12)} : ${probAvg[k].toStringAsFixed(3)}   (teacher freq=${(100.0*countTeach[k]/N).toStringAsFixed(1)}%)');
  }
  margins.sort();
  print('Margin pctiles (p(teacher)-max(other)):');
  print('  <0.00: ${pct(0.00).toStringAsFixed(1)}%   <0.05: ${pct(0.05).toStringAsFixed(1)}%   <0.10: ${pct(0.10).toStringAsFixed(1)}%   <0.20: ${pct(0.20).toStringAsFixed(1)}%');
  print('');
}

/* ------------------------------------ main ----------------------------------- */

void main(List<String> argv) {
  // Make sure these match your training and runtime (see makeConfig defaults).
  final cfg = makeConfig(
    seed: 42,
    maxFuel: 1000.0,
    lockTerrain: true,
    randomSpawnX: true,
    // If you adjusted these in runtime/training, mirror them here:
    // gravity: 0.18,
    // thrustAccel: 0.42,
    // rotSpeed: 1.6,
    // stepScale: 60.0,
  );
  final env = eng.GameEngine(cfg);

  final fe = ai.FeatureExtractor(groundSamples: 3, stridePx: 48);
  final policy = ai.PolicyNetwork(inputSize: fe.inputSize, h1: 64, h2: 64, seed: 42);
  final norm = ai.RunningNorm(fe.inputSize, momentum: 0.995);

  final ok = tryLoadPolicyWithNorm('policy_pretrained.json', policy, norm);
  if (ok) {
    print('Loaded policy_pretrained.json');
  } else {
    print('No valid policy or norm found at policy_pretrained.json');
  }

  if (!norm.inited || _normLooksDegenerate(norm)) {
    print('⚠️  Loaded norm seems degenerate → warming LOCAL norm (3500 samples).');
    warmLocalNormIfMissing(env: env, fe: fe, norm: norm, nSamples: 3500, seed: 424242);
  }
  _printNormSanity(norm);

  // Probes
  probeIntentAgreement(env: env, fe: fe, policy: policy, norm: norm, N: 800);
  probeTeacherVsStudent(env: env, fe: fe, policy: policy, norm: norm, maxDecisions: 4000, planHold: 1);
  probeCECalibration(env: env, fe: fe, policy: policy, norm: norm, maxDecisions: 1600, planHold: 1);

  probeDescendSlowAnalysis(env: env, fe: fe, policy: policy, norm: norm, N: 4000);
  probeIntentHeadStats(env: env, fe: fe, policy: policy, norm: norm, N: 5000);
}
