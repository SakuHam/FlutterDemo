// app_grid.dart
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart' show RenderBox;

/// Public model so it can be reused across pages.
class AppIcon {
  final String id;
  final String label;
  final IconData icon;
  final VoidCallback? onTap;
  int row;
  int col;

  AppIcon({
    required this.id,
    required this.label,
    required this.icon,
    required this.row,
    required this.col,
    required this.onTap,
  });
}

/// Reusable draggable snap-to-grid for launcher-style icon layouts.
/// - Paints a rounded-corner grid
/// - Drags long-press + snaps to nearest cell
/// - Swaps with occupant if dropping on taken cell
class AppIconGrid extends StatefulWidget {
  const AppIconGrid({
    super.key,
    required this.icons,
    this.cellSize = 96,
    this.gap = 16,
    this.header,
    this.decoration,
    this.padding = const EdgeInsets.all(0),
    this.onChanged,
  });

  /// Mutable list (rows/cols are updated in-place on drag).
  final List<AppIcon> icons;

  /// Tile size and gap between tiles.
  final double cellSize;
  final double gap;

  /// Optional header widget shown above the grid (e.g., a section title).
  final Widget? header;

  /// Optional decoration behind the grid.
  final BoxDecoration? decoration;

  /// Outer padding around the grid area.
  final EdgeInsets padding;

  /// Called after any drag completes and positions may have changed.
  final ValueChanged<List<AppIcon>>? onChanged;

  @override
  State<AppIconGrid> createState() => _AppIconGridState();
}

class _AppIconGridState extends State<AppIconGrid> {
  final GlobalKey _stackKey = GlobalKey();

  int _colsForWidth(double width) {
    if (width <= 0) return 1;
    final withCell = widget.cellSize + widget.gap;
    return math.max(1, (((width + widget.gap) / withCell)).floor());
  }

  Offset _cellToOffset(int row, int col) {
    final double x = col * (widget.cellSize + widget.gap);
    final double y = row * (widget.cellSize + widget.gap);
    return Offset(x, y);
  }

  ({int row, int col}) _offsetToCell(Offset local, int cols, int rows) {
    double cx = local.dx / (widget.cellSize + widget.gap);
    double cy = local.dy / (widget.cellSize + widget.gap);
    int col = cx.round().clamp(0, cols - 1);
    int row = cy.round().clamp(0, rows - 1);
    return (row: row, col: col);
  }

  int? _indexAt(int row, int col) {
    for (int i = 0; i < widget.icons.length; i++) {
      if (widget.icons[i].row == row && widget.icons[i].col == col) return i;
    }
    return null;
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final padding = widget.padding;
        final double w = constraints.maxWidth - padding.horizontal;
        final double h = constraints.maxHeight - padding.vertical;
        final cols = _colsForWidth(w);

        final rows = math.max(
          (widget.icons.map((e) => e.row).fold<int>(0, math.max)) + 1,
          ((widget.icons.length + cols - 1) ~/ cols),
        );

        final stackWidth  = math.max(w, cols * (widget.cellSize + widget.gap) - widget.gap);
        final stackHeight = math.max(h, rows * (widget.cellSize + widget.gap) - widget.gap);

        final gridContent = Container(
          decoration: widget.decoration ??
              BoxDecoration(
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
                    cell: widget.cellSize,
                    gap: widget.gap,
                    cols: cols,
                    rows: rows,
                  ),
                ),
              ),
              ...widget.icons.map((app) {
                final pos = _cellToOffset(app.row, app.col);
                return Positioned(
                  left: pos.dx,
                  top: pos.dy,
                  child: _DraggableIcon(
                    size: widget.cellSize,
                    icon: app.icon,
                    label: app.label,
                    onTap: app.onTap,
                    onDragEnd: (globalEnd) {
                      final box = _stackKey.currentContext!.findRenderObject() as RenderBox;
                      final localEnd = box.globalToLocal(globalEnd);
                      final clamped = Offset(
                        localEnd.dx.clamp(0.0, stackWidth - widget.cellSize),
                        localEnd.dy.clamp(0.0, stackHeight - widget.cellSize),
                      );
                      final cell = _offsetToCell(clamped, cols, rows);

                      setState(() {
                        final other = _indexAt(cell.row, cell.col);
                        if (other != null) {
                          final tmpRow = widget.icons[other].row;
                          final tmpCol = widget.icons[other].col;
                          widget.icons[other].row = app.row;
                          widget.icons[other].col = app.col;
                          app..row = tmpRow..col = tmpCol;
                        } else {
                          app..row = cell.row..col = cell.col;
                        }
                      });

                      widget.onChanged?.call(widget.icons);
                    },
                  ),
                );
              }).toList(),
            ],
          ),
        );

        return Padding(
          padding: padding,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (widget.header != null) ...[
                widget.header!,
                const SizedBox(height: 12),
              ],
              Expanded(child: gridContent),
            ],
          ),
        );
      },
    );
  }
}

/// --- Internal widgets/painter used by the grid ---

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
