"""The Qt face of setup: the walk-through, one page per step. Everything it shows comes from the
core (find, server, game, plan); this file only lays it out."""
import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPalette
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QScrollArea,
                               QSizePolicy, QSpinBox, QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
                               QWizard, QWizardPage)

from .apply import apply_plan, last_run, undo
from .find import (default_plugin_source, default_track_data, find_games, find_servers, in_game_content, is_game,
                   is_server, track_data_root)
from .game import car_facts, track_facts
from .identify import identify
from .plan import bot_slots, plan_bot
from .server import read_server

OK, WARN, BAD = '✔', '⚠', '✖'


def heading(text):
    label = QLabel(text)
    label.setWordWrap(True)
    return label


def hint(text):
    """A line of help under a field or at the foot of a page, in the window's quieter text colour."""
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.RichText)
    label.setForegroundRole(QPalette.PlaceholderText)
    return label


class PathPicker(QWidget):
    """A list of what was found, and a Browse button for anything else."""

    def __init__(self, found, check, browse_title, label=None, parent=None, normalize=None, refusal=None,
                 none_found='Nothing found automatically - use Browse.'):
        super().__init__(parent)
        self.check, self.browse_title, self.label = check, browse_title, label or (lambda p: p)
        self.normalize = normalize or (lambda p: p)
        self.refusal = refusal or (lambda p: 'That folder is not what this step needs.')
        self.list = QListWidget()
        for path in found:
            self.add(path)
        if self.list.count():
            self.list.setCurrentRow(0)
        browse = QPushButton('Browse...')
        browse.clicked.connect(self.browse)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.list)
        row = QHBoxLayout()
        self.message = QLabel('' if found else none_found)
        self.message.setWordWrap(True)
        row.addWidget(self.message, 1)
        row.addWidget(browse)
        lay.addLayout(row)

    def add(self, path):
        self.message.setText('') if hasattr(self, 'message') else None
        item = QListWidgetItem(self.label(path))
        item.setData(Qt.UserRole, path)
        item.setToolTip(path)
        self.list.addItem(item)

    def paths(self):
        return [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())]

    def browse(self):
        path = QFileDialog.getExistingDirectory(self, self.browse_title)
        if not path:
            return
        path = os.path.normpath(path)
        if not self.check(path):
            self.message.setText(f'{BAD} {self.refusal(path)}')
            return
        path = self.normalize(path)
        self.message.setText('')
        if path not in self.paths():
            self.add(path)
        self.list.setCurrentRow(self.paths().index(path))

    def value(self):
        item = self.list.currentItem()
        return item.data(Qt.UserRole) if item else ''


def server_label(path):
    s = read_server(path)
    if s.problems:
        return f'{os.path.basename(path)}   -   {s.problems[0]}'
    kunos = identify(path).kind == 'kunos'
    return (f'{s.name or os.path.basename(path)}   -   {s.track}' + (f' ({s.layout})' if s.layout else '')
            + ("   -   the game's own server, not AssettoServer" if kunos else ''))


class GamePage(QWizardPage):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.setTitle('Where is Assetto Corsa installed?')
        self.setSubTitle('Setup reads the track and the cars from the game. It changes nothing in it.')
        self.picker = PathPicker(state.get('games') or find_games(), is_game, 'The Assetto Corsa folder (the one with content\\)')
        self.picker.list.currentItemChanged.connect(lambda *_: self.completeChanged.emit())
        lay = QVBoxLayout(self)
        lay.addWidget(self.picker)
        lay.addWidget(hint('This is the game you play, not the server. With Steam it is usually '
                           '<code>steamapps\\common\\assettocorsa</code> in your Steam library. Setup builds the '
                           "plugin's track data and each bot car's numbers from what is in its <code>content</code> "
                           'folder, so the server\'s track and cars must be installed here too.'))

    def isComplete(self):
        return bool(self.picker.value())

    def validatePage(self):
        self.state['game'] = self.picker.value()
        return True


class ServerPage(QWizardPage):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.setTitle('Which server gets the drift bot?')
        self.setSubTitle('Pick your AssettoServer folder (the one with cfg\\server_cfg.ini). For a server on a '
                         'hosting plan, download its cfg folder into a folder here and pick that; the last page '
                         'then exports the files to upload.')
        self.picker = PathPicker([], is_server, 'The AssettoServer folder', label=server_label)
        self.picker.list.currentItemChanged.connect(self.show_server)
        self.picker.list.currentItemChanged.connect(lambda *_: self.completeChanged.emit())
        self.info = QLabel()
        self.info.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.addWidget(self.picker, 1)
        lay.addWidget(self.info)
        lay.addWidget(hint('Setup lists the AssettoServer folders it finds in or beside the game; Browse for one '
                           'elsewhere. Start the server once before setup if you have not yet, so AssettoServer '
                           'writes <code>cfg\\extra_cfg.yml</code>. Nothing is changed until the last page.'))

    def initializePage(self):
        if not self.picker.list.count():
            for path in find_servers([self.state['game']] + self.state.get('server_hints', [])):
                self.picker.add(path)
            if self.picker.list.count():
                self.picker.list.setCurrentRow(0)
                self.picker.message.setText('')
        self.show_server()

    def show_server(self, *_):
        path = self.picker.value()
        if not path:
            self.info.setText('')
            return
        s = read_server(path)
        if s.problems:
            self.info.setText(f'{BAD} ' + '; '.join(s.problems))
            return
        layout = f' ({s.layout})' if s.layout else ''
        prog = identify(path)
        if prog.kind == 'kunos':
            program = f'{BAD} {prog.note}'
        elif prog.kind == 'assettoserver':
            program = f'{OK} AssettoServer {prog.version or "(version not found)"}'
        else:
            program = f'{WARN} {prog.note}'
        self.info.setText(f'<b>{s.name or os.path.basename(path)}</b><br>{program}<br>Track: {s.track}{layout}<br>'
                          f'{len(s.entries)} car slots, {len(s.bots)} AI; AI is {"on" if s.enable_ai else "off"}; '
                          f'plugins: {", ".join(s.plugins) or "none"}')

    def isComplete(self):
        return bool(self.picker.value())

    def validatePage(self):
        self.state['server'] = read_server(self.picker.value())
        return not self.state['server'].problems and identify(self.picker.value()).kind != 'kunos'


class TrackPage(QWizardPage):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.setTitle('Checking the track')
        self.setSubTitle("Is the server's track in the game install, with what the bots need? Nothing to choose here.")
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setTextFormat(Qt.RichText)
        lay = QVBoxLayout(self)
        lay.addWidget(self.text)
        lay.addStretch(1)

    def initializePage(self):
        s, game = self.state['server'], self.state['game']
        t = track_facts(game, s.track, s.layout)
        layout = f' ({s.layout})' if s.layout else ''
        rows = [f'<b>{s.track}{layout}</b>']
        if not t.exists:
            rows.append(f'{BAD} The game install has no content\\tracks\\{s.track}.')
        else:
            rows.append(f'{OK} Model files: {", ".join(t.model_files) or "none listed"}')
            if t.ai_points >= 2:
                rows.append(f'{OK} AI line: {t.ai_points} points')
            else:
                rows.append(f'{BAD} AI line: {t.ai_note}. AssettoServer needs one for any AI car; '
                            'recording one by driving a lap is coming in a later version.')
        rows.append("<br>The plugin also needs data about this track (its walls, road and course). If the server "
                    "does not have it yet, setup makes it from the game install when you make the changes on the "
                    "last page; that takes a few minutes.")
        self.text.setText('<br>'.join(rows))


class CarPage(QWizardPage):
    SAME = '(as it is)'

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.skins = {}
        self.setTitle("Pick the bots' cars")
        self.setSubTitle("Any car the server already offers, slot by slot. A bot already in the entry list is put "
                         "in the car you pick for its slot.")
        self.all_car = QComboBox()
        self.all_car.setToolTip('Puts every bot in this car (and its skin); change single slots in the list below.')
        self.all_car.activated.connect(self.set_all)
        self.name = QLineEdit('DriftBot')
        self.name.setToolTip('The name of each bot setup adds (numbered when it adds more than one).')
        self.count = QSpinBox()
        self.count.setRange(1, 12)
        self.count.setToolTip('Bots in all, existing ones included (setup does not take bots away). '
                              'The plugin spreads them round the lap.')
        self.count.valueChanged.connect(self.fill_rows)
        self.slots = QTableWidget(0, 3)
        self.slots.setHorizontalHeaderLabels(['Slot', 'Car', 'Skin'])
        self.slots.verticalHeader().setVisible(False)
        head = self.slots.horizontalHeader()
        head.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        head.setSectionResizeMode(1, QHeaderView.Stretch)
        head.setSectionResizeMode(2, QHeaderView.Stretch)
        self.train = QSpinBox()
        self.train.setRange(0, 6)
        self.train.setSpecialValueText('as it is')
        self.train.setToolTip('Bots to a tandem train: the first car of each train drives, the others follow it '
                              'door to door. 1 = every bot drives on its own.')
        self.pace = QSpinBox()
        self.pace.setRange(0, 105)
        self.pace.setSuffix(' %')
        self.pace.setSpecialValueText('as it is')
        self.pace.setToolTip("How fast the bots drive, as a share of the flat-out speed round the lap "
                             "(a policy trained with paces). Lower is calmer.")
        self.note = QLabel()
        self.note.setWordWrap(True)
        form = QFormLayout(self)
        form.addRow('How many bots', self.count)
        form.addRow('Every bot in', self.all_car)
        form.addRow(self.slots)
        form.addRow("New bots' name", self.name)
        form.addRow('Bots to a train', self.train)
        form.addRow('Pace', self.pace)
        form.addRow(self.note)
        form.addRow(hint('Pick how many bots you want and their car; the rest can stay as it is. '
                         'Each bot takes a car slot on the server, so players keep the other slots. '
                         'Hover over a field for more about it.'))

    def initializePage(self):
        s = self.state['server']
        config = s.plugin_config or {}
        self.train.setValue(int(config.get('TrainSize', 0) or 0))
        self.pace.setValue(int(float(config.get('BotPacePercent', 0) or 0)))
        self.all_car.clear()
        self.all_car.addItems([self.SAME] + s.cars)
        self.slots.setRowCount(0)
        self.count.blockSignals(True)
        self.count.setMinimum(max(1, len(s.bots)))
        self.count.setValue(max(self.count.minimum(), self.state.get('bots') or 1))
        self.count.blockSignals(False)
        self.fill_rows()

    def car_skins(self, model):
        if model not in self.skins:
            c = car_facts(self.state['game'], model) if model else None
            used = [e.skin for e in self.state['server'].entries if e.model == model and e.skin]
            self.skins[model] = (list(dict.fromkeys(used + (c.skins if c else []))),
                                 c is not None and not c.exists)
        return self.skins[model]

    def fill_rows(self, *_):
        """One row per bot: the existing AI entries as they are, then the new ones in the last row's car."""
        s = self.state['server']
        keep = self.slots.rowCount()
        defaults = bot_slots(s, None, self.count.value())
        self.slots.setRowCount(len(defaults))
        for row in range(keep, len(defaults)):
            index, model, look = defaults[row]
            if row and index is None:
                model = self.slots.cellWidget(row - 1, 1).currentText()
                look = ''
            label = QTableWidgetItem(f'CAR_{index}' if index is not None else 'new')
            label.setFlags(label.flags() & ~Qt.ItemIsEditable)
            self.slots.setItem(row, 0, label)
            car = QComboBox()
            car.addItems(s.cars + ([model] if model and model not in s.cars else []))
            car.setCurrentText(model)
            skin = QComboBox()
            self.slots.setCellWidget(row, 1, car)
            self.slots.setCellWidget(row, 2, skin)
            car.currentTextChanged.connect(lambda m, r=row: self.fill_skins(r))
            self.fill_skins(row, look)
        self.slots.setMinimumHeight(min(8, len(defaults) + 1) * 26)
        self.check_cars()

    def fill_skins(self, row, look=None):
        model = self.slots.cellWidget(row, 1).currentText()
        skins, _ = self.car_skins(model)
        if look is None:          # the car changed: a bot keeps its skin only in its own car
            e = next((e for e in self.state['server'].bots if self.slots.item(row, 0).text() == f'CAR_{e.index}'), None)
            look = e.skin if e and e.model == model else ''
        box = self.slots.cellWidget(row, 2)
        box.clear()
        box.addItems(([look] if look and look not in skins else []) + skins)
        if look:
            box.setCurrentText(look)
        self.check_cars()

    def set_all(self, *_):
        model = self.all_car.currentText()
        if model == self.SAME:
            return
        for row in range(self.slots.rowCount()):
            self.slots.cellWidget(row, 1).setCurrentText(model)

    def check_cars(self):
        models = list(dict.fromkeys(self.slots.cellWidget(r, 1).currentText() for r in range(self.slots.rowCount())
                                    if self.slots.cellWidget(r, 1)))
        missing = [m for m in models if self.car_skins(m)[1]]
        if missing:
            self.note.setText(f'{BAD} The game install has no ' + ', '.join(f'content\\cars\\{m}' for m in missing) + '.')
        else:
            self.note.setText(f'{OK} ' + ('All found' if len(models) > 1 else 'Found') + ' in the game install.')

    def validatePage(self):
        slots = [(self.slots.cellWidget(r, 1).currentText(), self.slots.cellWidget(r, 2).currentText())
                 for r in range(self.slots.rowCount())]
        self.state.update(slots=slots, car=slots[0][0], skin=slots[0][1],
                          bot_name=self.name.text().strip() or 'DriftBot', bots=len(slots),
                          train_size=self.train.value() or None, pace_percent=self.pace.value() or None)
        return True


class ChangesPage(QWizardPage):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.setTitle("Here's what setup would change")
        self.setSubTitle('Click a change to see the lines before and after. Setup keeps every file it changes in '
                         'gradient-drift-setup-backups in the server folder, and Undo puts them back.')
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        mono = QFont('Consolas')
        mono.setStyleHint(QFont.Monospace)
        self.mono = mono
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.issues = QLabel()
        self.issues.setWordWrap(True)
        self.issues.setTextFormat(Qt.RichText)
        self.plugin = PathPicker([p for p in [default_plugin_source()] if p],
                                 lambda p: os.path.exists(os.path.join(p, 'GradientDriftPlugin.dll')),
                                 "The plugin's build folder (with GradientDriftPlugin.dll)",
                                 none_found="Not found: unzip the plugin download into setup's folder, or Browse.")
        self.track_data = PathPicker([p for p in [default_track_data()] if p],
                                     lambda p: os.path.isdir(p) and not in_game_content(p),
                                     "A folder of track data you already have (a folder per track, e.g. drift_rl\\tracks)",
                                     normalize=lambda p: track_data_root(p) or p,
                                     refusal=lambda p: "That is a track's folder in the game or server content. "
                                     "Leave Track files empty and setup makes the data from it.",
                                     none_found="Usually left empty: setup makes the track's data from the game. Browse only to use track "
                                     "data you already have.")
        for picker in (self.plugin, self.track_data):
            picker.list.setMaximumHeight(48)
            picker.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            picker.list.currentItemChanged.connect(lambda *_: self.initializePage())
        files = QFormLayout()
        files.addRow('Plugin build', self.plugin)
        files.addRow('Track files', self.track_data)
        self.apply_button = QPushButton('Make these changes')
        self.apply_button.clicked.connect(self.apply)
        self.undo_button = QPushButton('Undo the last changes')
        self.undo_button.clicked.connect(self.undo)
        self.export_button = QPushButton('Export for a hosted server...')
        self.export_button.setToolTip('Make the changes into a separate folder instead, with the files to upload '
                                      'to your host and HOW_TO_INSTALL.txt. The server folder is not changed.')
        self.export_button.clicked.connect(self.export)
        buttons = QHBoxLayout()
        buttons.addWidget(self.export_button)
        buttons.addStretch(1)
        buttons.addWidget(self.undo_button)
        buttons.addWidget(self.apply_button)
        lay = QVBoxLayout(self)
        lay.addWidget(self.summary)
        lay.addWidget(self.tree, 1)
        issues = QScrollArea()          # a long list of problems scrolls instead of squeezing the fields below
        issues.setWidget(self.issues)
        issues.setWidgetResizable(True)
        issues.setFrameShape(QScrollArea.NoFrame)
        lay.addWidget(issues, 1)
        lay.addLayout(files)
        lay.addWidget(hint('<b>To finish:</b> stop the server, click <b>Make these changes</b>, then start the '
                           'server again. Its log should say <code>GradientDrift: policy loaded from</code>; the '
                           'bots join when a player is on. A server on a hosting plan: '
                           '<b>Export for a hosted server</b> instead.'))
        lay.addLayout(buttons)

    def apply(self):
        plan = self.state['plan']
        answer = QMessageBox.question(self, 'Make the changes?',
                                      f'Setup will make {len(plan.changes)} change(s) to\n{plan.server.folder}\n\n'
                                      'The originals are kept, and Undo puts them back. Stop the server first if it '
                                      'is running; start it again afterwards.')
        if answer != QMessageBox.Yes:
            return
        done = []
        QApplication.setOverrideCursor(Qt.WaitCursor)     # making a track's data takes a few minutes
        try:
            backup = apply_plan(plan, log=done.append)
        except Exception as e:
            QMessageBox.critical(self, 'Nothing was changed', f'{type(e).__name__}: {e}')
            return
        finally:
            QApplication.restoreOverrideCursor()
        QMessageBox.information(self, 'Done', '\n'.join(done) + f'\n\nRestart the server for the changes to take effect.')
        self.state['server'] = read_server(plan.server.folder)
        self.initializePage()

    def export(self):
        from .export import INSTRUCTIONS, UPLOAD, export_server
        st = self.state
        out = QFileDialog.getExistingDirectory(self, 'An empty folder for the files to upload')
        if not out:
            return
        done = []
        QApplication.setOverrideCursor(Qt.WaitCursor)     # making a track's data takes a few minutes
        try:
            plan, files = export_server(st['server'].folder, os.path.normpath(out), st['game'], log=done.append,
                                        slots=st.get('slots'), bots=st['bots'], bot_name=st['bot_name'],
                                        plugin_source=self.plugin.value() or None,
                                        track_data=self.track_data.value() or None,
                                        train_size=st.get('train_size'), pace_percent=st.get('pace_percent'))
        except Exception as e:
            QMessageBox.critical(self, 'Nothing was exported', f'{type(e).__name__}: {e}')
            return
        finally:
            QApplication.restoreOverrideCursor()
        if files is None:
            QMessageBox.warning(self, 'Nothing was exported', '\n'.join(plan.blockers) or 'The server needs no changes.')
            return
        QMessageBox.information(self, 'Exported', f'{len(files)} file(s) to upload are in\n{os.path.join(out, UPLOAD)}\n\n'
                                f'{INSTRUCTIONS} beside them says where each one goes and which config lines '
                                'change. The server folder you picked was not changed.')
        try:
            os.startfile(out)       # Windows: show the folder
        except (AttributeError, OSError):
            pass

    def undo(self):
        server = self.state['server'].folder
        if QMessageBox.question(self, 'Undo?', 'Put back the files the last run of setup changed?') != QMessageBox.Yes:
            return
        done = []
        undo(server, log=done.append)
        QMessageBox.information(self, 'Undone', '\n'.join(done) or 'Nothing to undo.')
        self.state['server'] = read_server(server)
        self.initializePage()

    def initializePage(self):
        st = self.state
        plan = plan_bot(st['server'], st['game'], slots=st.get('slots'), bots=st['bots'],
                        bot_name=st['bot_name'], plugin_source=self.plugin.value() or None,
                        track_data=self.track_data.value() or None, train_size=st.get('train_size'),
                        pace_percent=st.get('pace_percent'))
        st['plan'] = plan
        self.tree.clear()
        if plan.changes:
            group = QTreeWidgetItem([f'Changes ({len(plan.changes)})'])
            self.tree.addTopLevelItem(group)
            for c in plan.changes:
                item = QTreeWidgetItem([f'{c.what}   -   {c.file}'])
                group.addChild(item)
                for label, text in (('before', c.before), ('after', c.after)):
                    for line in text.splitlines():
                        child = QTreeWidgetItem([f'{label:6s} | {line}'])
                        child.setFont(0, self.mono)
                        item.addChild(child)
            group.setExpanded(True)
        issues = [f'{BAD} {b}' for b in plan.blockers] + [f'{WARN} {n}' for n in plan.notes]
        self.issues.setText('<br>'.join(issues))
        if plan.ready:
            self.summary.setText(f'{OK} This server already has everything the bot needs.')
        elif plan.blockers:
            self.summary.setText(f'{WARN} {len(plan.changes)} change(s) setup can make, and '
                                 f'{len(plan.blockers)} thing(s) it cannot fix yet.')
        else:
            self.summary.setText(f'{OK} {len(plan.changes)} change(s), all of which setup can make.')
        self.apply_button.setEnabled(bool(plan.changes) and not plan.blockers)
        self.export_button.setEnabled(bool(plan.changes) and not plan.blockers)
        self.undo_button.setEnabled(last_run(st['server'].folder) is not None)


class SetupWizard(QWizard):
    def __init__(self, state=None):
        super().__init__()
        self.state = state or {}
        self.setWindowTitle('Gradient Drift Setup')
        self.setWizardStyle(QWizard.ModernStyle)
        self.setMinimumSize(760, 520)
        for page in (GamePage, ServerPage, TrackPage, CarPage, ChangesPage):
            self.addPage(page(self.state))
        self.setButtonText(QWizard.FinishButton, 'Close')


def run(state=None):
    app = QApplication.instance() or QApplication(sys.argv)
    wizard = SetupWizard(state)
    wizard.show()
    return app.exec()
