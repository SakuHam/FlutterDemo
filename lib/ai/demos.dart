// lib/ai/demos.dart
import 'dart:convert';

class DemoStep {
  final List<double> x; // feature vector
  final int thr;        // 0/1
  final int turn;       // 0:none,1:left,2:right
  DemoStep(this.x, this.thr, this.turn);

  Map<String, dynamic> toJson() => {
    'x': x,
    'thr': thr,
    'turn': turn,
  };
  factory DemoStep.fromJson(Map<String, dynamic> j) =>
      DemoStep((j['x'] as List).map((e) => (e as num).toDouble()).toList(),
          j['thr'] as int, j['turn'] as int);
}

class DemoEpisode {
  final List<DemoStep> steps;
  final bool landed;
  DemoEpisode(this.steps, {this.landed = false});

  Map<String, dynamic> toJson() => {
    'landed': landed,
    'steps': steps.map((s) => s.toJson()).toList(),
  };
  factory DemoEpisode.fromJson(Map<String, dynamic> j) =>
      DemoEpisode(((j['steps'] as List).cast<Map<String,dynamic>>())
          .map(DemoStep.fromJson).toList(), landed: j['landed'] as bool? ?? false);
}

class DemoSet {
  final int inputSize;
  final List<DemoEpisode> episodes;
  DemoSet({required this.inputSize, required this.episodes});

  Map<String, dynamic> toJson() => {
    'inputSize': inputSize,
    'episodes': episodes.map((e) => e.toJson()).toList(),
  };
  factory DemoSet.fromJson(Map<String, dynamic> j) =>
      DemoSet(
        inputSize: j['inputSize'] as int,
        episodes: ((j['episodes'] as List).cast<Map<String,dynamic>>())
            .map(DemoEpisode.fromJson).toList(),
      );

  String toPrettyJson() => const JsonEncoder.withIndent('  ').convert(toJson());
  static DemoSet fromJsonString(String s) => DemoSet.fromJson(json.decode(s));
}
