"""The zone editor: a map of a track's course with its drift zones, to move, add, delete and check
against a recorded run. Everything it does to the zones is in zones.py; this file draws them and
turns mouse moves into those calls.

Opened from setup's last page (Zones...) or with `gradient-drift-setup zones <server or track>`.
"""
import os
import sys

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QKeySequence, QPainter, QPainterPath, QPalette, QPen, QPolygonF, QShortcut
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QFileDialog, QGraphicsEllipseItem, QGraphicsItem,
                               QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget)

from . import zones as Z

YELLOW = QColor('#f5c400')
ASPHALT = QColor('#3b3b3d')
GROUND = QColor('#6f8f55')
RUN = QColor('#2a78d6')
PICK_M = 12.0           # how near an edge (m) a press has to be to start drawing a zone there


def _pt(x, z):
    return QPointF(float(x), float(-z))     # the map from above: x across, z up the screen


def _xz(p):
    return p.x(), -p.y()


def _poly(xz):
    return QPolygonF([_pt(x, z) for x, z in xz])


def _pen(colour, width, style=Qt.SolidLine):
    pen = QPen(colour, width, style)
    pen.setCosmetic(True)           # the same width in pixels however far one zooms
    pen.setJoinStyle(Qt.MiterJoin)
    return pen


class MapView(QGraphicsView):
    """The map. Scroll to zoom, drag the empty map to move it; the editor gets presses on zones,
    handles and edges through its callbacks."""

    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(GROUND)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.panning = None

    def wheelEvent(self, event):
        f = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(f, f)

    def fit(self):
        self.fitInView(self.scene().itemsBoundingRect().adjusted(-20, -20, 20, 20), Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        scene = self.mapToScene(event.position().toPoint())
        if event.button() == Qt.LeftButton and self.editor.press(scene, self.items(event.position().toPoint())):
            return
        self.panning = event.position()
        self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.panning is not None:
            d = event.position() - self.panning
            self.panning = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        self.editor.drag(self.mapToScene(event.position().toPoint()))

    def mouseReleaseEvent(self, event):
        if self.panning is not None:
            self.panning = None
            self.unsetCursor()
            return
        self.editor.release(self.mapToScene(event.position().toPoint()))


class ZoneEditor(QDialog):
    def __init__(self, tracks, track, parent=None):
        super().__init__(parent)
        self.tracks, self.track = tracks, track
        self.course, zones, self.edited = Z.load(tracks, track)
        self.zones = Z.ordered(self.course, zones)
        self.history = []           # earlier zone lists, for Undo
        self.dirty = False
        self.selected = None
        self.runs, self.run = [], None
        self.scores = None
        self.dragging = None        # ('first' | 'last') while a handle is dragged
        self.drawing = None         # (side, first point) while a new zone is drawn
        self.zone_items, self.handles = [], []
        self.setWindowTitle(f'Zones: {track}')
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        self.resize(1200, 760)

        self.scene = QGraphicsScene(self)
        self.view = MapView(self)
        self.view.setScene(self.scene)
        self.draw_course()
        self.run_item = self.scene.addPath(QPainterPath(), _pen(RUN, 1.5))
        self.run_item.setZValue(5)
        self.preview = self.scene.addPolygon(QPolygonF(), _pen(YELLOW, 2, Qt.DashLine))
        self.preview.setZValue(6)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self.select)
        self.draw_button = QPushButton('Draw a zone')
        self.draw_button.setCheckable(True)
        self.draw_button.setToolTip('Then press near an edge of the road and drag along it, in either direction.')
        self.delete_button = QPushButton('Delete')
        self.delete_button.clicked.connect(self.delete)
        self.flip_button = QPushButton('Other edge')
        self.flip_button.setToolTip('Move the zone across to the other edge of the road, over the same stretch.')
        self.flip_button.clicked.connect(self.flip)
        self.kind = QComboBox()
        self.kind.addItems([k.capitalize() for k in Z.KINDS])
        self.kind.setToolTip('Outside zone: the rear is to be in it for its whole length. Touch and go: the rear '
                             'only has to reach it.')
        self.kind.activated.connect(self.set_kind)
        self.auto_button = QPushButton('Place automatically')
        self.auto_button.setToolTip("Put back the zones setup places from the course's shape (replaces these).")
        self.auto_button.clicked.connect(self.place_auto)
        self.undo_button = QPushButton('Undo')
        self.undo_button.clicked.connect(self.undo)
        QShortcut(QKeySequence.Undo, self, self.undo)
        QShortcut(QKeySequence.Delete, self, self.delete)
        self.lap_button = QPushButton('Load a recording...')
        self.lap_button.setToolTip("AI Car Monitor's recording (a run per driver), the plugin's telemetry or the "
                                   "game's telemetry recorder: its rear-axle path is drawn, and each zone scored.")
        self.lap_button.clicked.connect(self.load_lap)
        self.driver = QComboBox()
        self.driver.activated.connect(self.pick_run)
        self.driver.hide()
        self.line_score = QLabel()
        self.line_score.setWordWrap(True)
        self.warnings = QLabel()
        self.warnings.setWordWrap(True)
        self.save_button = QPushButton('Save')
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self.save)
        close = QPushButton('Close')
        close.clicked.connect(self.close)
        fit = QPushButton('Whole track')
        fit.clicked.connect(self.view.fit)

        side = QWidget()
        lay = QVBoxLayout(side)
        lay.addWidget(self.status)
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        for b in (self.draw_button, self.delete_button, self.flip_button):
            row.addWidget(b)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel('Kind'))
        row.addWidget(self.kind, 1)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.auto_button)
        row.addWidget(self.undo_button)
        row.addWidget(fit)
        lay.addLayout(row)
        lay.addWidget(self.warnings)
        lay.addSpacing(8)
        lay.addWidget(self.lap_button)
        lay.addWidget(self.driver)
        lay.addWidget(self.line_score)
        help_text = QLabel("Click a zone to pick it, then drag its round handles along the edge to move its ends. "
                           "Scroll to zoom; drag the empty map to move it. The plugin paints the zones on the "
                           "road for players once setup has copied them to the server. For the bots, zones "
                           "change what the trainer pays for: they drive them only after training on this track.")
        help_text.setWordWrap(True)
        help_text.setForegroundRole(QPalette.PlaceholderText)
        lay.addWidget(help_text)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.save_button)
        row.addWidget(close)
        lay.addLayout(row)
        split = QSplitter()
        split.addWidget(self.view)
        split.addWidget(side)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([860, 340])
        outer = QVBoxLayout(self)
        outer.addWidget(split)
        self.refresh()

    # --- drawing ----------------------------------------------------------------------------------

    def draw_course(self):
        c = self.course
        road = self.scene.addPolygon(_poly(np.r_[c.pos, c.neg[::-1]]), QPen(Qt.NoPen), QBrush(ASPHALT))
        road.setZValue(0)
        # the start/finish closes the loop: the road between the last point and the first
        self.scene.addPolygon(_poly([c.pos[-1], c.pos[0], c.neg[0], c.neg[-1]]), QPen(Qt.NoPen),
                              QBrush(ASPHALT)).setZValue(0)
        for e in (c.pos, c.neg):
            path = QPainterPath(_pt(*e[0]))
            for x, z in e[1:]:
                path.lineTo(_pt(x, z))
            path.closeSubpath()
            self.scene.addPath(path, _pen(QColor('white'), 1.2)).setZValue(1)
        start = self.scene.addLine(*(_pt(*c.pos[0]).toTuple() + _pt(*c.neg[0]).toTuple()),
                                   _pen(QColor('white'), 2, Qt.DashLine))
        start.setZValue(1)
        step = max(1, c.n // 14)
        for i in range(0, c.n, step):           # arrows down the middle: the direction of travel
            j = (i + 2) % c.n
            a, b = c.mid[i], c.mid[j]
            d = (b - a) / max(np.linalg.norm(b - a), 1e-6)
            n = np.array([-d[1], d[0]])
            tip = a + 4.0 * d
            arrow = self.scene.addPolygon(_poly([tip, a - 1.5 * n, a + 1.5 * n]), QPen(Qt.NoPen),
                                          QBrush(QColor('#9a9a9a')))
            arrow.setZValue(1)
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-200, -200, 200, 200))

    def draw_zones(self):
        for item in self.zone_items + self.handles:
            self.scene.removeItem(item)
        self.zone_items, self.handles = [], []
        for i, z in enumerate(self.zones):
            outer, inner = self.course.zone_outline(z)
            chosen = i == self.selected
            colour = QColor(YELLOW)
            if z['kind'] != Z.KINDS[0]:
                colour.setAlpha(170)
            brush = QBrush(colour, Qt.BDiagPattern if z['kind'] == Z.KINDS[0] else Qt.Dense7Pattern)
            item = self.scene.addPolygon(_poly(np.r_[outer, inner[::-1]]), _pen(colour, 4 if chosen else 2.5), brush)
            item.setZValue(3 if chosen else 2)
            item.setData(0, i)
            self.zone_items.append(item)
            stroke = QPainterPath(_pt(*outer[0]))      # the outer edge thick at any zoom: zones show on the whole map
            for x, zz in outer[1:]:
                stroke.lineTo(_pt(x, zz))
            line = self.scene.addPath(stroke, _pen(colour, 7 if chosen else 5))
            line.setZValue(3 if chosen else 2)
            line.setData(0, i)
            self.zone_items.append(line)
            label = QGraphicsSimpleTextItem(str(i + 1))
            label.setBrush(QBrush(QColor('white') if not chosen else YELLOW))
            font = label.font()
            font.setBold(True)
            font.setPointSize(11)
            label.setFont(font)
            mid = len(outer) // 2
            spot = outer[mid] + 2.5 * (outer[mid] - inner[mid]) / max(np.linalg.norm(outer[mid] - inner[mid]), 1e-6)
            label.setPos(_pt(*spot))
            label.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            label.setZValue(4)
            label.setData(0, i)
            self.scene.addItem(label)
            self.zone_items.append(label)
            if chosen:
                for end, xz in (('first', outer[0]), ('last', outer[-1])):
                    h = QGraphicsEllipseItem(QRectF(-7, -7, 14, 14))
                    h.setPos(_pt(*xz))
                    h.setFlag(QGraphicsItem.ItemIgnoresTransformations)
                    h.setBrush(QBrush(QColor('white')))
                    h.setPen(QPen(QColor('#222222'), 2))
                    h.setZValue(7)
                    h.setData(1, end)
                    h.setCursor(Qt.SizeAllCursor)
                    h.setToolTip('Drag along the edge to move this end' + (' (the start)' if end == 'first' else ' (the end)'))
                    self.scene.addItem(h)
                    self.handles.append(h)

    def draw_run(self):
        path = QPainterPath()
        if self.run is not None:
            r = self.run
            for k in range(len(r.rear_x)):
                p = _pt(r.rear_x[k], r.rear_z[k])
                if r.breaks[k]:
                    path.moveTo(p)
                else:
                    path.lineTo(p)
        self.run_item.setPath(path)

    def refresh(self, keep_view=False):
        c = self.course
        if self.selected is not None and self.selected >= len(self.zones):
            self.selected = None
        self.scores = Z.score(c, self.zones, self.run) if self.run is not None else None
        self.list.blockSignals(True)
        self.list.clear()
        for i, z in enumerate(self.zones):
            text = (f"{i + 1}.  {z['kind'].capitalize()}, {Z.side_name(z['side'])} edge\n"
                    f"     {c.along[z['first']]:.0f} - {c.along[z['last']]:.0f} m from the start, {z['length']:.0f} m long")
            if self.scores is not None:
                fill, depth, past = self.scores[i]
                text += ('\n     not reached by this run' if np.isnan(fill) else
                         f'\n     filled {fill:.0%}, rear {depth:.1f} m inside the edge, past it {past:.0%}')
            self.list.addItem(QListWidgetItem(text))
        if self.selected is not None:
            self.list.setCurrentRow(self.selected)
        self.list.blockSignals(False)
        state = ('edited by hand' if self.edited else 'placed automatically by setup') if self.zones or self.edited \
            else 'none yet'
        self.status.setText(f"<b>{self.track}</b>: {len(self.zones)} zone(s), {c.lap_m:.0f} m lap, zones "
                            f"{Z.DEPTH_M:g} m deep<br>File: {state}" + (' <b>(not saved)</b>' if self.dirty else ''))
        has = self.selected is not None
        for w in (self.delete_button, self.flip_button, self.kind):
            w.setEnabled(has)
        if has:
            self.kind.setCurrentIndex(Z.KINDS.index(self.zones[self.selected]['kind']))
        self.undo_button.setEnabled(bool(self.history))
        self.save_button.setEnabled(self.dirty)
        problems = Z.problems(c, self.zones)
        self.warnings.setText(('<span style="color:#d0342c">' + '<br>'.join(problems) + '</span>') if problems else '')
        if self.scores is not None:
            score = Z.line_score(self.scores)
            self.line_score.setText(f"{self.run.name}: line score {score:.1f} / 40 (Formula Drift style: mean fill of "
                                    "the zones reached x 40)" if not np.isnan(score) else
                                    f"{self.run.name}: reaches none of the zones")
        else:
            self.line_score.setText('')
        self.draw_zones()
        if not keep_view and not getattr(self, '_fitted', False):
            self._fitted = True
            self.view.fit()

    # --- changes ----------------------------------------------------------------------------------

    def change(self, zones, select=None):
        self.history.append((list(self.zones), self.selected))
        self.zones = zones
        self.selected = select
        self.dirty = True
        self.refresh(keep_view=True)

    def select(self, row):
        self.selected = row if 0 <= row < len(self.zones) else None
        self.refresh(keep_view=True)

    def delete(self):
        if self.selected is None:
            return
        self.change([z for i, z in enumerate(self.zones) if i != self.selected])

    def flip(self):
        if self.selected is None:
            return
        z = self.zones[self.selected]
        zones = list(self.zones)
        zones[self.selected] = Z.move(self.course, z, side=-z['side'])
        self.change(zones, self.selected)

    def set_kind(self, index):
        if self.selected is None or self.zones[self.selected]['kind'] == Z.KINDS[index]:
            return
        zones = list(self.zones)
        zones[self.selected] = Z.move(self.course, zones[self.selected], kind=Z.KINDS[index])
        self.change(zones, self.selected)

    def place_auto(self):
        if self.zones and QMessageBox.question(
                self, 'Place automatically', 'Replace these zones with the ones setup places from the course\'s '
                'shape? (Undo brings these back until you close the editor.)') != QMessageBox.Yes:
            return
        self.change(Z.ordered(self.course, Z.auto(self.course)))

    def undo(self):
        if not self.history:
            return
        self.zones, self.selected = self.history.pop()
        self.dirty = True
        self.refresh(keep_view=True)

    def put(self, zone):
        """Adds or replaces a zone and keeps the list in lap order, the zone still picked."""
        zones = Z.ordered(self.course, list(self.zones) + [zone])
        return zones, next(i for i, z in enumerate(zones) if z is zone)

    # --- the mouse ---------------------------------------------------------------------------------

    def press(self, scene, items):
        """A left press on the map; True when it was for a zone (else the map pans)."""
        x, z = _xz(scene)
        if self.draw_button.isChecked():
            side, point, dist = self.course.nearest_edge(x, z)
            if dist > PICK_M:
                return False
            self.drawing = (side, point)
            return True
        for item in items:
            if item.data(1) in ('first', 'last'):
                self.dragging = item.data(1)
                self.history.append((list(self.zones), self.selected))
                return True
        for item in items:
            if item.data(0) is not None:
                self.select(int(item.data(0)))
                return True
        return False

    def drag(self, scene):
        x, z = _xz(scene)
        if self.drawing is not None:
            side, first = self.drawing
            last = self.course.nearest_on_edge(side, x, z)
            if first != last:
                outer, inner = self.course.zone_outline(Z.new_zone(self.course, side, first, last))
                self.preview.setPolygon(_poly(np.r_[outer, inner[::-1]]))
        elif self.dragging is not None and self.selected is not None:
            zone = self.zones[self.selected]
            point = self.course.nearest_on_edge(zone['side'], x, z)
            if point == zone[self.dragging]:
                return
            moved = Z.move(self.course, zone, **{self.dragging: point})
            # moving one end past the other would flip the zone round the lap: keep it the way it was
            if moved['first'] != (zone['first'] if self.dragging == 'last' else point):
                return
            self.zones = list(self.zones)
            self.zones[self.selected] = moved
            self.dirty = True
            self.refresh(keep_view=True)

    def release(self, scene):
        if self.drawing is not None:
            side, first = self.drawing
            self.drawing = None
            self.preview.setPolygon(QPolygonF())
            last = self.course.nearest_on_edge(side, *_xz(scene))
            if (last - first) % self.course.n not in (0, self.course.n - 1, 1):
                zones, index = self.put(Z.new_zone(self.course, side, first, last))
                self.change(zones, index)
            self.draw_button.setChecked(False)
        elif self.dragging is not None:
            self.dragging = None
            if self.selected is not None:
                zones, index = self.put(self.zones.pop(self.selected))
                self.zones, self.selected = zones, index
            self.refresh(keep_view=True)

    # --- recordings, saving ------------------------------------------------------------------------

    def load_lap(self):
        path, _ = QFileDialog.getOpenFileName(self, 'A recording to check the zones against', '',
                                              'Recordings (*.csv *.csv.gz);;All files (*)')
        if not path:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            runs = Z.load_runs(path)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, 'Not read', f"Setup could not read that recording:\n\n{type(e).__name__}: {e}")
            return
        QApplication.restoreOverrideCursor()
        runs = [r for r in runs if len(r.x) >= 20]
        if not runs:
            QMessageBox.information(self, 'Nothing to show', 'That recording has no driving above '
                                    f'{Z.MOVING_KMH:g} km/h.')
            return
        self.runs = runs
        self.driver.clear()
        self.driver.addItems([f"{r.name} ({len(r.x)} samples)" for r in runs])
        self.driver.setVisible(len(runs) > 1)
        self.pick_run(0)

    def pick_run(self, index):
        self.run = self.runs[index] if 0 <= index < len(self.runs) else None
        self.draw_run()
        self.refresh(keep_view=True)

    def save(self):
        problems = Z.problems(self.course, self.zones)
        if problems and QMessageBox.question(self, 'Save anyway?', '\n'.join(problems) + '\n\nSave these zones anyway?'
                                             ) != QMessageBox.Yes:
            return False
        try:
            path, backup, edited = Z.save(self.tracks, self.track, self.course, self.zones)
        except OSError as e:
            QMessageBox.warning(self, 'Not saved', f'The zones could not be written:\n\n{e}')
            return False
        self.zones = Z.ordered(self.course, self.zones)
        self.dirty, self.edited = False, edited
        self.refresh(keep_view=True)
        QMessageBox.information(self, 'Saved', f'Saved to {path}' + (f'\n\nThe file it replaced is kept as {backup}'
                                                                     if backup else '') +
                                ("\n\nSetup keeps these zones when it makes the track's data again." if edited else
                                 "\n\nThese are the automatic zones, so setup places them afresh when it makes the "
                                 "track's data again.") +
                                "\n\nRun setup's changes again to copy them to the server, where the plugin paints them "
                                'for players. For the bots, zones change what the trainer pays for: they drive them '
                                'after training on this track.')
        return True

    def closeEvent(self, event):
        if self.dirty:
            answer = QMessageBox.question(self, 'Unsaved zones', 'Save the changes to the zones?',
                                          QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if answer == QMessageBox.Cancel or (answer == QMessageBox.Save and not self.save()):
                event.ignore()
                return
        event.accept()

    def reject(self):           # Esc asks too
        self.close()


def open_editor(tracks, track, parent=None):
    """Shows the editor for a track (modal over parent). False when the track has no course data."""
    if not Z.has_course(tracks, track):
        return False
    ZoneEditor(tracks, track, parent).exec()
    return True


def run(tracks, track):
    app = QApplication.instance() or QApplication(sys.argv)
    if not open_editor(tracks, track):
        print(f"no course data for {track} in {tracks}: setup makes it with the track's data")
        return 1
    del app
    return 0


if __name__ == '__main__':      # python -m gradientdrift_setup.zones_gui <track files folder> <track>
    sys.exit(run(os.path.abspath(sys.argv[1]), sys.argv[2]))
