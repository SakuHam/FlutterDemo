import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart' show RenderBox;

import 'game.dart'; // <-- brings in GamePage

void main() {
  runApp(const MoonLanderApp());
}

class MoonLanderApp extends StatelessWidget {
  const MoonLanderApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Moon Lander',
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0B0F14),
        textTheme: const TextTheme(bodyMedium: TextStyle(fontFamily: 'monospace')),
      ),
      home: const MainPage(),
    );
  }
}

class MainPage extends StatefulWidget {
  const MainPage({super.key});
  @override
  State<MainPage> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> {
  // Layout
  static const double cellSize = 96;
  static const double gap = 16;
  final GlobalKey _stackKey = GlobalKey();

  // Icon model
  late List<_AppIcon> icons;

  @override
  void initState() {
    super.initState();
    icons = [
      _AppIcon(
        id: 'game',
        label: 'Moon Lander',
        icon: Icons.rocket_launch,
        row: 0,
        col: 0,
        onTap: () {
          Navigator.push(context, MaterialPageRoute(builder: (_) => const GamePage()));
        },
      ),
      _AppIcon(
        id: 'music',
        label: 'Music',
        icon: Icons.music_note,
        row: 0,
        col: 1,
        onTap: null, // placeholder
      ),
      _AppIcon(
        id: 'settings',
        label: 'Settings',
        icon: Icons.settings,
        row: 0,
        col: 2,
        onTap: null, // placeholder
      ),
    ];
  }

  Offset _cellToOffset(int row, int col) {
    final double x = col * (cellSize + gap);
    final double y = row * (cellSize + gap);
    return Offset(x, y);
  }

  ({int row, int col}) _offsetToCell(Offset local, int cols, int rows) {
    double cx = local.dx / (cellSize + gap);
    double cy = local.dy / (cellSize + gap);
    int col = cx.round().clamp(0, cols - 1);
    int row = cy.round().clamp(0, rows - 1);
    return (row: row, col: col);
  }

  int _colsForWidth(double width) {
    if (width <= 0) return 1;
    final withCell = cellSize + gap;
    return math.max(1, (((width + gap) / withCell)).floor());
  }

  int? _indexAt(int row, int col) {
    for (int i = 0; i < icons.length; i++) {
      if (icons[i].row == row && icons[i].col == col) return i;
    }
    return null;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) {
            final padding = const EdgeInsets.symmetric(horizontal: 20, vertical: 24);
            final double w = constraints.maxWidth - padding.horizontal;
            final double h = constraints.maxHeight - padding.vertical;
            final cols = _colsForWidth(w);
            final rows = math.max(
              (icons.map((e) => e.row).fold<int>(0, math.max)) + 1,
              ((icons.length + cols - 1) ~/ cols),
            );

            final stackWidth  = math.max(w, cols * (cellSize + gap) - gap);
            final stackHeight = math.max(h, rows * (cellSize + gap) - gap);

            return Padding(
              padding: padding,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Apps', style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 12),
                  Expanded(
                    child: Container(
                      decoration: BoxDecoration(
                        color: const Color(0xFF0E141B),
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(color: Colors.white12),
                      ),
                      child: Stack(
                        key: _stackKey,
                        children: [
                          Positioned.fill(
                            child: CustomPaint(
                              painter: _GridPainter(
                                cell: cellSize, gap: gap, cols: cols, rows: rows,
                              ),
                            ),
                          ),
                          ...icons.map((app) {
                            final pos = _cellToOffset(app.row, app.col);
                            return Positioned(
                              left: pos.dx,
                              top: pos.dy,
                              child: _DraggableIcon(
                                size: cellSize,
                                icon: app.icon,
                                label: app.label,
                                onTap: app.onTap,
                                onDragEnd: (globalEnd) {
                                  final box = _stackKey.currentContext!.findRenderObject() as RenderBox;
                                  final localEnd = box.globalToLocal(globalEnd);
                                  final clamped = Offset(
                                    localEnd.dx.clamp(0.0, stackWidth - cellSize),
                                    localEnd.dy.clamp(0.0, stackHeight - cellSize),
                                  );
                                  final cell = _offsetToCell(clamped, cols, rows);

                                  setState(() {
                                    final other = _indexAt(cell.row, cell.col);
                                    if (other != null) {
                                      final tmpRow = icons[other].row;
                                      final tmpCol = icons[other].col;
                                      icons[other].row = app.row;
                                      icons[other].col = app.col;
                                      app..row = tmpRow..col = tmpCol;
                                    } else {
                                      app..row = cell.row..col = cell.col;
                                    }
                                  });
                                },
                              ),
                            );
                          }).toList(),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }
}

// --- Models & widgets ---

class _AppIcon {
  final String id;
  final String label;
  final IconData icon;
  final VoidCallback? onTap;
  int row;
  int col;

  _AppIcon({
    required this.id,
    required this.label,
    required this.icon,
    required this.row,
    required this.col,
    required this.onTap,
  });
}

class _DraggableIcon extends StatelessWidget {
  final double size;
  final IconData icon;
  final String label;
  final VoidCallback? onTap;
  final ValueChanged<Offset> onDragEnd; // global offset

  const _DraggableIcon({
    required this.size,
    required this.icon,
    required this.label,
    required this.onTap,
    required this.onDragEnd,
  });

  @override
  Widget build(BuildContext context) {
    final tile = _IconTile(size: size, icon: icon, label: label, onTap: onTap);
    return LongPressDraggable(
      feedback: Opacity(
        opacity: 0.9,
        child: Material(
          color: Colors.transparent,
          child: _IconTile(size: size, icon: icon, label: label, onTap: null),
        ),
      ),
      childWhenDragging: Opacity(opacity: 0.15, child: IgnorePointer(child: tile)),
      onDragEnd: (details) => onDragEnd(details.offset),
      child: tile,
    );
  }
}

class _IconTile extends StatelessWidget {
  final double size;
  final IconData icon;
  final String label;
  final VoidCallback? onTap;

  const _IconTile({
    required this.size,
    required this.icon,
    required this.label,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: size,
      height: size,
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: onTap,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: size * 0.58,
              height: size * 0.58,
              decoration: BoxDecoration(
                color: Colors.white10,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: Colors.white24),
              ),
              child: Icon(icon, size: size * 0.40),
            ),
            const SizedBox(height: 8),
            Text(label, overflow: TextOverflow.ellipsis, style: const TextStyle(fontSize: 13)),
          ],
        ),
      ),
    );
  }
}

class _GridPainter extends CustomPainter {
  final double cell;
  final double gap;
  final int cols;
  final int rows;

  _GridPainter({required this.cell, required this.gap, required this.cols, required this.rows});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white12
      ..style = PaintingStyle.stroke;
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        final left = c * (cell + gap);
        final top  = r * (cell + gap);
        final rect = Rect.fromLTWH(left, top, cell, cell);
        canvas.drawRRect(RRect.fromRectAndRadius(rect, const Radius.circular(16)), paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant _GridPainter old) {
    return old.cell != cell || old.gap != gap || old.cols != cols || old.rows != rows;
  }
}
