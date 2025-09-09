// music.dart
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';

class MusicPage extends StatefulWidget {
  const MusicPage({super.key});
  @override
  State<MusicPage> createState() => _MusicPageState();
}

class _MusicPageState extends State<MusicPage> {
  late Future<List<File>> _mp3sFuture;
  bool _deepScan = false; // optional: scan whole /storage/emulated/0 if no results

  @override
  void initState() {
    super.initState();
    _mp3sFuture = _loadMp3s();
  }

  Future<void> _refresh() async {
    setState(() => _mp3sFuture = _loadMp3s());
  }

  Future<PermissionStatus> _ensureAudioPermission() async {
    // Android 13+: READ_MEDIA_AUDIO; Older: READ_EXTERNAL_STORAGE
    final audio = await Permission.audio.request();
    if (audio.isGranted) return PermissionStatus.granted;
    final storage = await Permission.storage.request();
    if (storage.isGranted) return PermissionStatus.granted;
    if (audio.isPermanentlyDenied || storage.isPermanentlyDenied) {
      return PermissionStatus.permanentlyDenied;
    }
    return PermissionStatus.denied;
  }

  List<Directory> _preferredAndroidDirs(List<Directory> platformMusicDirs) {
    // Build a prioritized list of candidate dirs (no duplicates)
    final seen = <String>{};
    final add = (Directory d) {
      final path = d.path;
      if (path.isEmpty) return null;
      if (seen.add(path)) return d;
      return null;
    };

    final dirs = <Directory>[];

    // 1) Platform Music directories
    for (final d in platformMusicDirs) {
      final added = add(d);
      if (added != null) dirs.add(added);
    }

    // 2) Common fallbacks
    final fallbacks = <String>[
      '/storage/emulated/0/Music',
      '/storage/emulated/0/Download',
      '/sdcard/Music',
      '/sdcard/Download',
    ];
    for (final path in fallbacks) {
      final d = Directory(path);
      if (d.existsSync()) {
        final added = add(d);
        if (added != null) dirs.add(added);
      }
    }

    // 3) Optional deep scan root (only if explicitly enabled)
    if (_deepScan) {
      final root = Directory('/storage/emulated/0');
      if (root.existsSync()) {
        final added = add(root);
        if (added != null) dirs.add(added);
      }
    }

    return dirs;
  }

  bool _isHiddenOrSandbox(String path) {
    // Skip app sandboxes & hidden dirs for speed/noise
    if (path.contains('/Android/')) return true;
    final segs = path.split('/');
    return segs.any((s) => s.startsWith('.'));
  }

  bool _isMp3(String path) => p.extension(path).toLowerCase() == '.mp3';

  Future<List<File>> _loadMp3s() async {
    if (!Platform.isAndroid) return [];

    final perm = await _ensureAudioPermission();
    if (perm != PermissionStatus.granted) {
      throw 'Permission to read audio was denied.';
    }

    // 1) Ask platform for Music directories
    List<Directory> musicDirs = [];
    try {
      final dirs = await getExternalStorageDirectories(type: StorageDirectory.music);
      if (dirs != null) {
        musicDirs = dirs.whereType<Directory>().toList();
      }
    } catch (_) {
      // ignore
    }

    // 2) Build candidate list
    final candidates = _preferredAndroidDirs(musicDirs);

    // 3) Scan + dedupe by canonical path
    final seenFiles = <String>{};
    final found = <File>[];

    for (final dir in candidates) {
      if (_isHiddenOrSandbox(dir.path)) continue;

      try {
        // Shallow-to-deep: recursive=true; followLinks=false to avoid loops
        await for (final entity in dir.list(recursive: true, followLinks: false)) {
          if (entity is! File) continue;

          final raw = entity.path;
          if (!_isMp3(raw)) continue;

          try {
            final real = await File(raw).resolveSymbolicLinks();
            if (seenFiles.add(real)) found.add(File(real));
          } catch (_) {
            // If cannot resolve, dedupe on raw path
            if (seenFiles.add(raw)) found.add(entity);
          }
        }
      } catch (_) {
        // ignore per-dir IO/permission errors
      }
    }

    // If nothing found AND deep scan was off, try once with deep scan on
    if (found.isEmpty && !_deepScan) {
      _deepScan = true;
      return _loadMp3s();
    }

    // Sort nicely: by folder then filename
    found.sort((a, b) {
      final ap = p.dirname(a.path).toLowerCase();
      final bp = p.dirname(b.path).toLowerCase();
      final c = ap.compareTo(bp);
      if (c != 0) return c;
      return p.basename(a.path).toLowerCase().compareTo(p.basename(b.path).toLowerCase());
    });

    return found;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Music'),
        actions: [
          PopupMenuButton<String>(
            onSelected: (v) {
              if (v == 'toggle_deep') {
                setState(() => _deepScan = !_deepScan);
                _refresh();
              }
            },
            itemBuilder: (context) => [
              CheckedPopupMenuItem(
                value: 'toggle_deep',
                checked: _deepScan,
                child: const Text('Deep scan /storage'),
              ),
            ],
          )
        ],
      ),
      body: FutureBuilder<List<File>>(
        future: _mp3sFuture,
        builder: (context, snap) {
          if (snap.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snap.hasError) {
            return _ErrorView(
              message: '${snap.error}',
              onOpenSettings: () => openAppSettings(),
              onRetry: _refresh,
            );
          }
          final files = snap.data ?? const [];
          if (files.isEmpty) {
            return _EmptyView(onRetry: _refresh, deepScan: _deepScan, onToggleDeep: () {
              setState(() => _deepScan = true);
              _refresh();
            });
          }

          return RefreshIndicator(
            onRefresh: _refresh,
            child: ListView.separated(
              physics: const AlwaysScrollableScrollPhysics(),
              itemCount: files.length,
              separatorBuilder: (_, __) => const Divider(height: 1, thickness: 0.5),
              itemBuilder: (context, i) {
                final f = files[i];
                final name = p.basename(f.path);
                final folder = p.dirname(f.path);
                return ListTile(
                  leading: const Icon(Icons.music_note),
                  title: Text(name, overflow: TextOverflow.ellipsis),
                  subtitle: Text(folder, maxLines: 1, overflow: TextOverflow.ellipsis),
                  onTap: () {
                    // Hook a player later (just_audio/audioplayers).
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text('Selected: $name')),
                    );
                  },
                );
              },
            ),
          );
        },
      ),
    );
  }
}

class _EmptyView extends StatelessWidget {
  const _EmptyView({required this.onRetry, required this.deepScan, required this.onToggleDeep});
  final Future<void> Function() onRetry;
  final bool deepScan;
  final VoidCallback onToggleDeep;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.queue_music, size: 56),
            const SizedBox(height: 12),
            Text(
              deepScan
                  ? 'No MP3 files found.\nPull to refresh, or add songs to Music/Download.'
                  : 'No MP3 files found.\nTry enabling Deep scan or add songs to Music/Download.',
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 12),
            Wrap(spacing: 8, children: [
              OutlinedButton(onPressed: onRetry, child: const Text('Scan again')),
              if (!deepScan) FilledButton(onPressed: onToggleDeep, child: const Text('Enable Deep scan')),
            ]),
          ],
        ),
      ),
    );
  }
}

class _ErrorView extends StatelessWidget {
  const _ErrorView({
    required this.message,
    required this.onOpenSettings,
    required this.onRetry,
  });

  final String message;
  final VoidCallback onOpenSettings;
  final Future<void> Function() onRetry;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.lock, size: 56),
            const SizedBox(height: 12),
            Text(message, textAlign: TextAlign.center),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              children: [
                OutlinedButton(onPressed: onRetry, child: const Text('Retry')),
                FilledButton(onPressed: onOpenSettings, child: const Text('Open Settings')),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
