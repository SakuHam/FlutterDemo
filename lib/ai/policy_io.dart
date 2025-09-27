// lib/ai/policy_io.dart
import 'dart:convert';
import 'dart:io';

import '../engine/game_engine.dart' as eng;
import '../engine/raycast.dart'; // RayConfig
import 'agent.dart' as ai; // PolicyNetwork, MLP, RunningNorm, FeatureExtractorRays

/* =============================================================================
   Helpers shared by save/load
   ============================================================================= */

List<List<double>> _deepCopyMat(List<List<double>> W) =>
    List.generate(W.length, (i) => List<double>.from(W[i]));

String _feSignature({
  required int inputSize,
  required int rayCount,
  required bool kindsOneHot,
  required double worldW,
  required double worldH,
}) {
  return 'kind=rays;in=$inputSize;rays=$rayCount;1hot=$kindsOneHot;W=${worldW.toInt()};H=${worldH.toInt()}';
}

/* =============================================================================
   Persisted normalization block (optional inside bundle)
   ============================================================================= */

class NormBlock {
  final int dim;
  final List<double> mean;
  final List<double> var_;
  final String? signature;
  const NormBlock({
    required this.dim,
    required this.mean,
    required this.var_,
    this.signature,
  });
}

/* =============================================================================
   Strict policy bundle (schema matches runtime_policy.dart v2)
   ============================================================================= */

class PolicyBundle {
  final int inputDim;
  final List<int> hidden;
  final Map<String, dynamic> trunk; // {layers: [{W: [][] , b: []}, ...]}
  final Map<String, dynamic> heads; // intent/turn/thr/val; optional dur
  final Map<String, dynamic> featureExtractor; // kind/rayCount/kindsOneHot
  final Map<String, dynamic> physics; // gravity/thrustAccel/rotSpeed/...
  final String signature;
  final Map<String, dynamic>? rayConfig; // rayCount/includeFloor/forwardAligned
  final NormBlock? norm;

  const PolicyBundle({
    required this.inputDim,
    required this.hidden,
    required this.trunk,
    required this.heads,
    required this.featureExtractor,
    required this.physics,
    required this.signature,
    required this.rayConfig,
    required this.norm,
  });

  static PolicyBundle loadFromPath(String path) {
    final raw = File(path).readAsStringSync();
    final j = json.decode(raw) as Map<String, dynamic>;

    final arch = (j['arch'] as Map?)?.cast<String, dynamic>();
    final trunk = (j['trunk'] as List?)?.cast<dynamic>();
    final heads = (j['heads'] as Map?)?.cast<String, dynamic>();
    if (arch == null || trunk == null || heads == null) {
      throw StateError('Bundle is not v2 (missing arch/trunk/heads).');
    }

    List<List<double>> _as2d(dynamic v) =>
        (v as List).map<List<double>>((r) => (r as List).map<double>((x) => (x as num).toDouble()).toList()).toList();
    List<double> _as1d(dynamic v) => (v as List).map<double>((x) => (x as num).toDouble()).toList();

    // norm block (new) or legacy mirrors
    NormBlock? nb;
    final nm = (j['norm'] as Map?)?.cast<String, dynamic>();
    if (nm != null) {
      nb = NormBlock(
        dim: (nm['dim'] as num).toInt(),
        mean: _as1d(nm['mean']),
        var_: _as1d(nm['var']),
        signature: nm['signature'] as String?,
      );
    } else if (j.containsKey('norm_mean') && j.containsKey('norm_var')) {
      nb = NormBlock(
        dim: (arch['input'] as num).toInt(),
        mean: _as1d(j['norm_mean']),
        var_: _as1d(j['norm_var']),
        signature: j['norm_signature'] as String?,
      );
    }

    return PolicyBundle(
      inputDim: (arch['input'] as num).toInt(),
      hidden  : ((arch['hidden'] as List?) ?? const []).map((e) => (e as num).toInt()).toList(),
      trunk   : {'layers': trunk.map((e) => (e as Map).cast<String, dynamic>()).toList()},
      heads   : heads,
      featureExtractor: (j['feature_extractor'] as Map?)?.cast<String, dynamic>() ?? const {},
      physics : (j['physics'] as Map?)?.cast<String, dynamic>() ?? const {},
      signature: (j['signature'] as String?) ?? '',
      rayConfig: (j['ray_config'] as Map?)?.cast<String, dynamic>(),
      norm: nb,
    );
  }

  void saveToPath(String path) {
    final f = File(path);
    f.writeAsStringSync(const JsonEncoder.withIndent('  ').convert(toJson()));
  }

  Map<String, dynamic> toJson() => {
    'arch': {
      'input': inputDim,
      'hidden': hidden,
      'kIntents': ai.PolicyNetwork.kIntents,
    },
    'trunk': (trunk['layers'] as List),
    'heads': heads,
    'feature_extractor': featureExtractor,
    'physics': physics,
    'signature': signature,
    if (rayConfig != null) 'ray_config': rayConfig,
    if (norm != null) 'norm': {
      'dim': norm!.dim,
      'momentum': 0.0, // informational
      'mean': norm!.mean,
      'var':  norm!.var_,
      'signature': norm!.signature ?? signature,
    },
    // legacy mirrors (optional)
    if (norm != null) 'norm_mean': norm!.mean,
    if (norm != null) 'norm_var' : norm!.var_,
    if (norm != null) 'norm_signature': norm!.signature ?? signature,
  };

  void copyInto(ai.PolicyNetwork target) {
    // trunk layers
    final layers = (trunk['layers'] as List).cast<Map<String, dynamic>>();
    if (layers.length != target.trunk.layers.length) {
      throw StateError('Trunk layers mismatch: load=${layers.length} runtime=${target.trunk.layers.length}');
    }
    for (int li = 0; li < layers.length; li++) {
      final src = layers[li];
      final W = (src['W'] as List)
          .map<List<double>>((r) => (r as List).map<double>((x) => (x as num).toDouble()).toList())
          .toList();
      final b = (src['b'] as List).map<double>((x) => (x as num).toDouble()).toList();
      final dst = target.trunk.layers[li];
      if (W.length != dst.W.length || W[0].length != dst.W[0].length || b.length != dst.b.length) {
        throw StateError('Trunk layer $li shape mismatch.');
      }
      for (int i = 0; i < dst.W.length; i++) {
        for (int j = 0; j < dst.W[0].length; j++) dst.W[i][j] = W[i][j];
      }
      for (int i = 0; i < dst.b.length; i++) dst.b[i] = b[i];
    }

    // heads (use dynamic to tolerate _Lin/Linear/etc.)
    void _loadHead(String name, dynamic dst) {
      final hj = (heads[name] as Map).cast<String, dynamic>();
      final W = (hj['W'] as List)
          .map<List<double>>((r) => (r as List).map<double>((x) => (x as num).toDouble()).toList())
          .toList();
      final b = (hj['b'] as List).map<double>((x) => (x as num).toDouble()).toList();

      // shape checks
      if (W.length != (dst.W as List).length ||
          W[0].length != (dst.W as List).first.length ||
          b.length != (dst.b as List).length) {
        throw StateError('Head "$name" shape mismatch.');
      }
      for (int i = 0; i < (dst.W as List).length; i++) {
        for (int j = 0; j < (dst.W as List).first.length; j++) {
          dst.W[i][j] = W[i][j];
        }
      }
      for (int i = 0; i < (dst.b as List).length; i++) {
        dst.b[i] = b[i];
      }
    }

    _loadHead('intent', target.heads.intent);
    _loadHead('turn',   target.heads.turn);
    _loadHead('thr',    target.heads.thr);
    _loadHead('val',    target.heads.val);

    // optional duration head (1 x H)
    if (heads.containsKey('dur')) {
      final dur = (heads['dur'] as Map).cast<String, dynamic>();
      final Wd = (dur['W'] as List).cast<List>().first.map<double>((x) => (x as num).toDouble()).toList();
      final Bd = (dur['b'] as List).map<double>((x) => (x as num).toDouble()).toList();
      final dst = target.durHead;
      if (Wd.length == dst.W[0].length && Bd.length == 1) {
        for (int j = 0; j < Wd.length; j++) dst.W[0][j] = Wd[j];
        dst.b[0] = Bd[0];
      } else {
        throw StateError('Duration head shape mismatch.');
      }
    }
  }
}

/* =============================================================================
   High-level save/load helpers (strict, schema-checked)
   ============================================================================= */

void savePolicyBundle({
  required String path,
  required ai.PolicyNetwork p,
  required eng.GameEngine env,
  required ai.RunningNorm? norm,
}) {
  // Probe FE to derive kindsOneHot
  final fe = ai.FeatureExtractorRays(rayCount: env.rayCfg.rayCount);
  final probeVec = fe.extract(
    lander: env.lander,
    terrain: env.terrain,
    worldW: env.cfg.worldW,
    worldH: env.cfg.worldH,
    rays: env.rays,
  );
  final kindsOneHot = (probeVec.length == 5 + env.rayCfg.rayCount * 4);

  final sig = _feSignature(
    inputSize: p.inputSize,
    rayCount: env.rayCfg.rayCount,
    kindsOneHot: kindsOneHot,
    worldW: env.cfg.worldW,
    worldH: env.cfg.worldH,
  );

  Map<String, dynamic> headJson(dynamic l) => {
    'W': _deepCopyMat(List<List<double>>.from(l.W)),
    'b': List<double>.from(List<double>.from(l.b)),
  };

  final trunkLayers = <Map<String, dynamic>>[];
  for (final layer in p.trunk.layers) {
    trunkLayers.add({'W': _deepCopyMat(layer.W), 'b': List<double>.from(layer.b)});
  }

  final physics = {
    'gravity': env.cfg.t.gravity,
    'thrustAccel': env.cfg.t.thrustAccel,
    'rotSpeed': env.cfg.t.rotSpeed,
    'maxFuel': env.cfg.t.maxFuel,
    'rcsEnabled': env.cfg.t.rcsEnabled,
    'rcsAccel': env.cfg.t.rcsAccel,
    'rcsBodyFrame': env.cfg.t.rcsBodyFrame,
    'downThrEnabled': env.cfg.t.downThrEnabled,
    'downThrAccel': env.cfg.t.downThrAccel,
    'downThrBurn': env.cfg.t.downThrBurn,
  };

  final bundleJson = <String, dynamic>{
    'arch': {
      'input': p.inputSize,
      'hidden': p.hidden,
      'kIntents': ai.PolicyNetwork.kIntents,
    },
    'trunk': trunkLayers,
    'heads': {
      'intent': headJson(p.heads.intent),
      'turn'  : headJson(p.heads.turn),
      'thr'   : headJson(p.heads.thr),
      'val'   : headJson(p.heads.val),
      'dur'   : {
        'W': [ List<double>.from(p.durHead.W[0]) ],
        'b': [ p.durHead.b[0] ],
      },
    },
    'feature_extractor': {
      'kind': 'rays',
      'rayCount': env.rayCfg.rayCount,
      'kindsOneHot': kindsOneHot,
    },
    'env_hint': {'worldW': env.cfg.worldW, 'worldH': env.cfg.worldH},
    'physics': physics,
    'signature': sig,
    'format': 'v2rays',
    'ray_config': {
      'rayCount': env.rayCfg.rayCount,
      'includeFloor': env.rayCfg.includeFloor,
      'forwardAligned': env.rayCfg.forwardAligned,
    },
  };

  if (norm != null && norm.inited && norm.dim == p.inputSize) {
    bundleJson['norm'] = {
      'dim': norm.dim,
      'momentum': norm.momentum,
      'mean': norm.mean,
      'var': norm.var_,
      'signature': sig,
    };
    // legacy mirrors
    bundleJson['norm_mean'] = norm.mean;
    bundleJson['norm_var'] = norm.var_;
    bundleJson['norm_momentum'] = norm.momentum;
    bundleJson['norm_signature'] = sig;
  }

  final f = File(path);
  f.writeAsStringSync(const JsonEncoder.withIndent('  ').convert(bundleJson));
  print('Saved policy → $path');
}

void loadBundleIntoNetwork({
  required PolicyBundle bundle,
  required ai.PolicyNetwork target,
  required eng.GameEngine env,
}) {
  // Probe FE input size using current env+rayCfg to ensure alignment
  final fe = ai.FeatureExtractorRays(rayCount: env.rayCfg.rayCount);
  final inProbe = fe
      .extract(
    lander: env.lander,
    terrain: env.terrain,
    worldW: env.cfg.worldW,
    worldH: env.cfg.worldH,
    rays: env.rays,
  )
      .length;

  if (bundle.inputDim != inProbe) {
    throw StateError('Loaded inputDim=${bundle.inputDim} != FE probe=$inProbe. '
        'Check rayCount/kindsOneHot/env dims.');
  }
  if (bundle.hidden.length != target.hidden.length ||
      !List.generate(bundle.hidden.length, (i) => bundle.hidden[i] == target.hidden[i]).every((x) => x)) {
    throw StateError('Hidden sizes mismatch. Loaded=${bundle.hidden} Runtime=${target.hidden}');
  }

  // Optional ray_config sanity
  final rc = bundle.rayConfig;
  if (rc != null) {
    final rcCount = (rc['rayCount'] as num?)?.toInt();
    if (rcCount != null && rcCount != env.rayCfg.rayCount) {
      throw StateError('rayCount mismatch. Loaded=$rcCount Runtime=${env.rayCfg.rayCount}');
    }
  }

  bundle.copyInto(target);
  print('Loaded weights (input=${bundle.inputDim}, hidden=${bundle.hidden}).');
}

bool restoreNormFromBundle({
  required PolicyBundle bundle,
  required ai.RunningNorm runtimeNorm,
}) {
  final nb = bundle.norm;
  if (nb == null) return false;
  if (nb.dim != runtimeNorm.dim) {
    print('Warning: bundle norm dim=${nb.dim} != runtime norm dim=${runtimeNorm.dim} → ignoring.');
    return false;
  }

  // Construct a temp RunningNorm and copy into runtime via existing API.
  final tmp = ai.RunningNorm(nb.dim, momentum: runtimeNorm.momentum);
  for (int i = 0; i < nb.dim; i++) {
    tmp.mean[i] = nb.mean[i];
    tmp.var_[i] = nb.var_[i];
  }
  tmp.inited = true;

  // Use the method your codebase already has.
  runtimeNorm.copyFrom(tmp);

  print('Restored feature normalization from bundle.');
  return true;
}
