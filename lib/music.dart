// music.dart
import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:audio_session/audio_session.dart';
import 'package:just_audio/just_audio.dart';
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
  bool _deepScan = false;

  // --- player state ---
  final AudioPlayer _player = AudioPlayer();
  List<File> _playlist = const [];
  int _currentIndex = -1; // index into _playlist

  @override
  void initState() {
    super.initState();
    _setupAudioSession();
    _mp3sFuture = _loadMp3s();
    // keep UI in sync with player’s currentIndex
    _player.currentIndexStream.listen((i) {
      if (i == null) return;
      setState(() => _currentIndex = i);
    });
  }

  Future<void> _setupAudioSession() async {
    final session = await AudioSession.instance;
    await session.configure(const AudioSessionConfiguration.music());
  }

  @override
  void dispose() {
    _player.dispose();
    super.dispose();
  }

  // ---------- permissions & scanning ----------
  Future<PermissionStatus> _ensureAudioPermission() async {
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
    final seen = <String>{};
    Directory? Function(Directory) add = (Directory d) {
      final path = d.path;
      if (path.isEmpty) return null;
      if (seen.add(path)) return d;
      return null;
    };

    final dirs = <Directory>[];

    for (final d in platformMusicDirs) {
      final added = add(d);
      if (added != null) dirs.add(added);
    }

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

    List<Directory> musicDirs = [];
    try {
      final dirs = await getExternalStorageDirectories(type: StorageDirectory.music);
      if (dirs != null) musicDirs = dirs.whereType<Directory>().toList();
    } catch (_) {}

    final candidates = _preferredAndroidDirs(musicDirs);

    final seenFiles = <String>{};
    final found = <File>[];

    for (final dir in candidates) {
      if (_isHiddenOrSandbox(dir.path)) continue;

      try {
        await for (final entity in dir.list(recursive: true, followLinks: false)) {
          if (entity is! File) continue;

          final raw = entity.path;
          if (!_isMp3(raw)) continue;

          try {
            final real = await File(raw).resolveSymbolicLinks();
            if (seenFiles.add(real)) found.add(File(real));
          } catch (_) {
            if (seenFiles.add(raw)) found.add(entity);
          }
        }
      } catch (_) {}
    }

    if (found.isEmpty && !_deepScan) {
      _deepScan = true;
      return _loadMp3s();
    }

    // Sort by folder then filename
    found.sort((a, b) {
      final ap = p.dirname(a.path).toLowerCase();
      final bp = p.dirname(b.path).toLowerCase();
      final c = ap.compareTo(bp);
      if (c != 0) return c;
      return p.basename(a.path).toLowerCase().compareTo(p.basename(b.path).toLowerCase());
    });

    return found;
  }

  Future<void> _refresh() async {
    setState(() => _mp3sFuture = _loadMp3s());
  }

  // ---------- playback helpers ----------
  Future<void> _playList(List<File> files, int startIndex) async {
    if (files.isEmpty || startIndex < 0 || startIndex >= files.length) return;

    _playlist = files;
    final sources = files
        .map((f) => AudioSource.uri(Uri.file(f.path)))
        .toList(growable: false);

    await _player.setAudioSource(ConcatenatingAudioSource(children: sources),
        initialIndex: startIndex);
    await _player.play();
    setState(() => _currentIndex = startIndex);
  }

  String _currentTitle() {
    if (_currentIndex < 0 || _currentIndex >= _playlist.length) return '';
    return p.basename(_playlist[_currentIndex].path);
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
      body: Column(
        children: [
          Expanded(
            child: FutureBuilder<List<File>>(
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
                  return _EmptyView(
                    onRetry: _refresh,
                    deepScan: _deepScan,
                    onToggleDeep: () {
                      setState(() => _deepScan = true);
                      _refresh();
                    },
                  );
                }

                return RefreshIndicator(
                  onRefresh: _refresh,
                  child: ListView.separated(
                    physics: const AlwaysScrollableScrollPhysics(),
                    itemCount: files.length,
                    separatorBuilder: (_, __) =>
                    const Divider(height: 1, thickness: 0.5),
                    itemBuilder: (context, i) {
                      final f = files[i];
                      final name = p.basename(f.path);
                      final folder = p.dirname(f.path);
                      final isPlayingThis =
                          _currentIndex == i && _player.playing;

                      return ListTile(
                        leading: Icon(
                          isPlayingThis ? Icons.equalizer : Icons.music_note,
                        ),
                        title: Text(name, overflow: TextOverflow.ellipsis),
                        subtitle: Text(folder,
                            maxLines: 1, overflow: TextOverflow.ellipsis),
                        onTap: () => _playList(files, i),
                      );
                    },
                  ),
                );
              },
            ),
          ),
          _PlayerBar(
            player: _player,
            titleBuilder: _currentTitle,
            hasQueue: () => _playlist.length > 1,
          ),
        ],
      ),
    );
  }
}

// ---------------- mini player widget ----------------

class _PlayerBar extends StatefulWidget {
  const _PlayerBar({
    required this.player,
    required this.titleBuilder,
    required this.hasQueue,
  });

  final AudioPlayer player;
  final String Function() titleBuilder;
  final bool Function() hasQueue;

  @override
  State<_PlayerBar> createState() => _PlayerBarState();
}

class _PlayerBarState extends State<_PlayerBar> {
  double? _dragValue;

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<PlayerState>(
      stream: widget.player.playerStateStream,
      builder: (context, stateSnap) {
        final playerState = stateSnap.data;
        final playing = playerState?.playing ?? false;

        return Material(
          color: const Color(0xFF121820),
          elevation: 12,
          child: SafeArea(
            top: false,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(12, 8, 12, 8),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // --- title & controls row
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          widget.titleBuilder(),
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      IconButton(
                        tooltip: 'Previous',
                        onPressed: widget.hasQueue()
                            ? () => widget.player.seekToPrevious()
                            : null,
                        icon: const Icon(Icons.skip_previous),
                      ),
                      IconButton(
                        tooltip: playing ? 'Pause' : 'Play',
                        onPressed: () async {
                          if (playing) {
                            await widget.player.pause();
                          } else {
                            await widget.player.play();
                          }
                        },
                        iconSize: 32,
                        icon: Icon(playing ? Icons.pause : Icons.play_arrow),
                      ),
                      IconButton(
                        tooltip: 'Next',
                        onPressed: widget.hasQueue()
                            ? () => widget.player.seekToNext()
                            : null,
                        icon: const Icon(Icons.skip_next),
                      ),
                      IconButton(
                        tooltip: 'Stop',
                        onPressed: () => widget.player.stop(),
                        icon: const Icon(Icons.stop),
                      ),
                    ],
                  ),
// --- position slider
                  StreamBuilder<Duration>(
                    stream: widget.player.positionStream,
                    builder: (context, posSnap) {
                      final position = posSnap.data ?? Duration.zero;
                      final duration = widget.player.duration ?? Duration.zero;

                      // Always use doubles for Slider
                      final double sliderMax = duration.inMilliseconds.toDouble().isFinite
                          ? duration.inMilliseconds.toDouble()
                          : 0.0;

                      double sliderValue = (_dragValue ?? position.inMilliseconds.toDouble());
                      if (!sliderValue.isFinite) sliderValue = 0.0;
                      sliderValue = sliderValue.clamp(0.0, sliderMax).toDouble();

                      // Avoid showing a dead slider when no track loaded
                      if (duration == Duration.zero) {
                        return const SizedBox.shrink();
                      }

                      String _fmt(Duration d) {
                        final h = d.inHours;
                        final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
                        final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
                        return h > 0 ? '$h:$m:$s' : '$m:$s';
                      }

                      return Column(
                        children: [
                          Slider(
                            min: 0.0,
                            max: sliderMax,                  // double
                            value: sliderValue,              // double
                            onChangeStart: (_) => setState(() => _dragValue = sliderValue),
                            onChanged: (v) => setState(() => _dragValue = v),
                            onChangeEnd: (v) async {
                              setState(() => _dragValue = null);
                              await widget.player.seek(Duration(milliseconds: v.round()));
                            },
                          ),
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Text(_fmt(position)),
                              Text(_fmt(duration)),
                            ],
                          ),
                        ],
                      );
                    },
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

// ---------------- simple empty/error views ----------------

class _EmptyView extends StatelessWidget {
  const _EmptyView({
    required this.onRetry,
    required this.deepScan,
    required this.onToggleDeep,
  });
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
              if (!deepScan)
                FilledButton(onPressed: onToggleDeep, child: const Text('Enable Deep scan')),
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
