// lib/ai/cli.dart
class Args {
  final Map<String, String?> _kv = {};
  final Set<String> _flags = {};

  Args(List<String> argv) {
    for (final a in argv) {
      if (!a.startsWith('--')) continue;
      final s = a.substring(2);
      final i = s.indexOf('=');
      if (i >= 0) {
        _kv[s.substring(0, i)] = s.substring(i + 1);
      } else {
        _flags.add(s);
      }
    }
  }

  String? getStr(String k, {String? def}) => _kv[k] ?? def;
  int getInt(String k, {int def = 0}) => int.tryParse(_kv[k] ?? '') ?? def;
  double getDouble(String k, {double def = 0.0}) => double.tryParse(_kv[k] ?? '') ?? def;

  bool getFlag(String k, {bool def = false}) {
    if (_flags.contains(k)) return true;            // --flag
    final v = _kv[k];                                // --flag=true / --flag=false
    if (v == null) return def;
    final s = v.toLowerCase();
    return s == '1' || s == 'true' || s == 'yes' || s == 'on';
  }
}
