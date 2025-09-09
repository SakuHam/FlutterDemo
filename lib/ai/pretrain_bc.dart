// lib/ai/pretrain_bc.dart
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'agent.dart' show
PolicyNetwork, relu, reluVec, sigmoid, softmax,
matVec, vecAdd, zeros, outer, addInPlaceVec, addInPlaceMat;

//
// Demo DTOs (must match your game-side recorder)
// If you already have lib/ai/demos.dart, you can import that instead.
//

class DemoStep {
  final List<double> x; // feature vector
  final int thr;        // 0/1
  final int turn;       // 0:none,1:left,2:right
  DemoStep(this.x, this.thr, this.turn);

  factory DemoStep.fromJson(Map<String, dynamic> j) => DemoStep(
    (j['x'] as List).map((e) => (e as num).toDouble()).toList(),
    j['thr'] as int,
    j['turn'] as int,
  );
}

class DemoEpisode {
  final List<DemoStep> steps;
  final bool landed;
  DemoEpisode(this.steps, {this.landed = false});

  factory DemoEpisode.fromJson(Map<String, dynamic> j) => DemoEpisode(
    ((j['steps'] as List).cast<Map<String, dynamic>>())
        .map(DemoStep.fromJson).toList(),
    landed: j['landed'] as bool? ?? false,
  );
}

class DemoSet {
  final int inputSize;
  final List<DemoEpisode> episodes;
  DemoSet({required this.inputSize, required this.episodes});

  factory DemoSet.fromJson(Map<String, dynamic> j) => DemoSet(
    inputSize: j['inputSize'] as int,
    episodes: ((j['episodes'] as List).cast<Map<String, dynamic>>())
        .map(DemoEpisode.fromJson).toList(),
  );
}

void _shuffle<T>(List<T> a, math.Random rnd) {
  for (int i = a.length - 1; i > 0; i--) {
    final j = rnd.nextInt(i + 1);
    final t = a[i]; a[i] = a[j]; a[j] = t;
  }
}

class BCBatch {
  final List<List<double>> X;
  final List<int> thr;   // 0/1
  final List<int> turn;  // 0/1/2
  BCBatch(this.X, this.thr, this.turn);
}

Iterable<BCBatch> makeBatches(List<DemoEpisode> eps, int batchSize) sync* {
  final X = <List<double>>[];
  final Tthr = <int>[];
  final Tturn = <int>[];
  for (final ep in eps) {
    for (final s in ep.steps) {
      X.add(s.x);
      Tthr.add(s.thr);
      Tturn.add(s.turn);
    }
  }

  final idx = List<int>.generate(X.length, (i) => i);
  final rnd = math.Random(42);
  _shuffle(idx, rnd);

  for (int i = 0; i < idx.length; i += batchSize) {
    final jEnd = math.min(i + batchSize, idx.length);
    final bx = <List<double>>[];
    final bthr = <int>[];
    final bturn = <int>[];
    for (int k = i; k < jEnd; k++) {
      final id = idx[k];
      bx.add(X[id]);
      bthr.add(Tthr[id]);
      bturn.add(Tturn[id]);
    }
    yield BCBatch(bx, bthr, bturn);
  }
}

// >>> FIXED: return Future<void> so we can `await` it
Future<void> savePolicy(String path, PolicyNetwork policy, int inputSize, int h1, int h2) async {
  final js = {
    'inputSize': inputSize,
    'h1': h1,
    'h2': h2,
    'W1': policy.W1, 'b1': policy.b1,
    'W2': policy.W2, 'b2': policy.b2,
    'W_thr': policy.W_thr, 'b_thr': policy.b_thr,
    'W_turn': policy.W_turn, 'b_turn': policy.b_turn,
    'fe': {'groundSamples': 3, 'stridePx': 48}, // adjust if you recorded differently
  };
  final out = const JsonEncoder.withIndent('  ').convert(js);
  await File(path).writeAsString(out);
  stdout.writeln('Saved policy → $path');
}

void main(List<String> args) async {
  if (args.isEmpty) {
    stdout.writeln('Usage: dart run lib/ai/pretrain_bc.dart <path_to_demos.json>');
    exit(1);
  }

  final demoPath = args[0];
  final txt = await File(demoPath).readAsString();
  final demos = DemoSet.fromJson(json.decode(txt) as Map<String, dynamic>);

  final inputSize = demos.inputSize;
  const h1 = 64, h2 = 64;
  final policy = PolicyNetwork(inputSize: inputSize, h1: h1, h2: h2, seed: 1234);

  // Hyperparameters
  const epochs = 6;
  const batchSize = 512;
  const lr = 1e-3;
  const l2 = 1e-6;

  double runningThrCE = 0.0, runningTurnCE = 0.0; int nThr = 0, nTurn = 0;

  for (int epoch = 1; epoch <= epochs; epoch++) {
    runningThrCE = 0.0; runningTurnCE = 0.0; nThr = 0; nTurn = 0;

    for (final batch in makeBatches(demos.episodes, batchSize)) {
      // accumulators
      final dW1 = zeros(policy.W1.length, policy.W1[0].length);
      final dW2 = zeros(policy.W2.length, policy.W2[0].length);
      final dW_thr = zeros(policy.W_thr.length, policy.W_thr[0].length);
      final dW_turn = zeros(policy.W_turn.length, policy.W_turn[0].length);
      final db1 = List<double>.filled(policy.b1.length, 0);
      final db2 = List<double>.filled(policy.b2.length, 0);
      final db_thr = List<double>.filled(policy.b_thr.length, 0);
      final db_turn = List<double>.filled(policy.b_turn.length, 0);

      for (int i = 0; i < batch.X.length; i++) {
        final x = batch.X[i];

        // forward (pure; no priors for BC)
        final z1 = vecAdd(matVec(policy.W1, x), policy.b1);
        final h1v = reluVec(z1);
        final z2 = vecAdd(matVec(policy.W2, h1v), policy.b2);
        final h2v = reluVec(z2);

        final thrLogit = matVec(policy.W_thr, h2v)[0] + policy.b_thr[0];
        final thrP = sigmoid(thrLogit);
        final tThr = batch.thr[i];
        // BCE loss
        runningThrCE += -(tThr == 1 ? math.log(thrP + 1e-12) : math.log(1 - thrP + 1e-12));
        nThr++;

        final turnLogits = vecAdd(matVec(policy.W_turn, h2v), policy.b_turn);
        final turnP = softmax(turnLogits);
        final tTurn = batch.turn[i];
        runningTurnCE += -math.log(turnP[tTurn] + 1e-12);
        nTurn++;

        // grads
        final dthr = (thrP - tThr); // dCE/dlogit
        addInPlaceVec(db_thr, [dthr]);
        addInPlaceMat(dW_thr, outer([dthr], h2v));

        final dturn = List<double>.generate(3, (k) => (turnP[k] - (k == tTurn ? 1.0 : 0.0)));
        addInPlaceVec(db_turn, dturn);
        addInPlaceMat(dW_turn, outer(dturn, h2v));

        // backprop to h2
        final dh2 = List<double>.filled(h2v.length, 0.0);
        for (int j = 0; j < policy.W_thr[0].length; j++) {
          dh2[j] += policy.W_thr[0][j] * dthr;
        }
        for (int c = 0; c < 3; c++) {
          for (int j = 0; j < policy.W_turn[0].length; j++) {
            dh2[j] += policy.W_turn[c][j] * dturn[c];
          }
        }
        final dz2 = List<double>.generate(z2.length, (j) => dh2[j] * (z2[j] > 0 ? 1.0 : 0.0));
        addInPlaceVec(db2, dz2);
        addInPlaceMat(dW2, outer(dz2, h1v));

        // backprop to h1
        final dh1 = List<double>.filled(h1v.length, 0.0);
        for (int r = 0; r < policy.W2.length; r++) {
          for (int c = 0; c < policy.W2[0].length; c++) {
            dh1[c] += policy.W2[r][c] * dz2[r];
          }
        }
        final dz1 = List<double>.generate(z1.length, (j) => dh1[j] * (z1[j] > 0 ? 1.0 : 0.0));
        addInPlaceVec(db1, dz1);
        addInPlaceMat(dW1, outer(dz1, x));
      }

      // L2 shrinkage on gradient (weight decay-like)
      void reg(List<List<double>> W) {
        for (final row in W) {
          for (int j = 0; j < row.length; j++) row[j] += l2 * row[j];
        }
      }
      reg(dW1); reg(dW2); reg(dW_thr); reg(dW_turn);

      // SGD step
      void sgd(List<List<double>> W, List<List<double>> dW) {
        for (int i = 0; i < W.length; i++) {
          for (int j = 0; j < W[0].length; j++) {
            W[i][j] -= lr * dW[i][j];
          }
        }
      }
      void sgdB(List<double> b, List<double> db) {
        for (int i = 0; i < b.length; i++) b[i] -= lr * db[i];
      }

      sgd(policy.W_thr, dW_thr); sgdB(policy.b_thr, db_thr);
      sgd(policy.W_turn, dW_turn); sgdB(policy.b_turn, db_turn);
      sgd(policy.W2, dW2); sgdB(policy.b2, db2);
      sgd(policy.W1, dW1); sgdB(policy.b1, db1);
    }

    stdout.writeln(
        'Epoch $epoch | thrCE=${(runningThrCE / (nThr > 0 ? nThr : 1)).toStringAsFixed(4)} '
            '| turnCE=${(runningTurnCE / (nTurn > 0 ? nTurn : 1)).toStringAsFixed(4)}'
    );
  }

  await savePolicy('policy_bc_init.json', policy, inputSize, h1, h2);
  await savePolicy('policy.json', policy, inputSize, h1, h2); // ready for the app to load
}
