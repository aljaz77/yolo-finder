"""Big checkbox menu of detectable classes with categories, search and presets."""
from __future__ import annotations

import json

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget)

from ..core.classes import CATEGORIES, COCO_NAMES, PRESETS
from ..core.config import app_home

COLS = 4


def _user_presets_path():
    return app_home() / "class_presets.json"


def load_user_presets() -> dict:
    try:
        return json.loads(_user_presets_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


class ClassPicker(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.boxes: dict[int, QCheckBox] = {}
        self.groups: list[tuple[QGroupBox, list[int]]] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        top = QHBoxLayout()
        top.addWidget(QLabel("Preset:"))
        self.preset = QComboBox()
        self.preset.setMinimumWidth(260)
        self.preset.activated.connect(self._apply_preset)
        top.addWidget(self.preset)
        b = QPushButton("Save current as preset…")
        b.clicked.connect(self._save_preset)
        top.addWidget(b)
        top.addSpacing(20)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search classes…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        top.addWidget(self.search, 1)
        for txt, val in (("Select all", True), ("Clear", False)):
            bb = QPushButton(txt)
            bb.clicked.connect(lambda _=False, v=val: self.set_all(v))
            top.addWidget(bb)
        lay.addLayout(top)

        self.count = QLabel()
        self.count.setObjectName("hint")
        lay.addWidget(self.count)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        lay.addWidget(self.scroll, 1)
        self.set_names({i: n for i, n in enumerate(COCO_NAMES)})

    # ----- building -----
    def set_names(self, names: dict):
        """Rebuild for a model's class list (COCO gets categories, custom models one group)."""
        sel = self.selected() if self.boxes else []
        self.names = {int(k): v for k, v in names.items()}
        self.boxes.clear()
        self.groups.clear()
        host = QWidget()
        v = QVBoxLayout(host)
        is_coco = [self.names.get(i) for i in range(80)] == COCO_NAMES and len(self.names) == 80
        cats = CATEGORIES if is_coco else [("All classes", sorted(self.names))]
        for title, ids in cats:
            g = QGroupBox(title.replace("&", "&&"))
            grid = QGridLayout(g)
            hdr = QHBoxLayout()
            for txt, val in (("all", True), ("none", False)):
                bb = QPushButton(txt)
                bb.setFlat(True)
                bb.setCursor(Qt.CursorShape.PointingHandCursor)
                bb.setStyleSheet("QPushButton{color:#7fa8ff;padding:0 6px;border:none;background:none}")
                bb.clicked.connect(lambda _=False, ii=ids, vv=val: self._set_ids(ii, vv))
                hdr.addWidget(bb)
            hdr.addStretch(1)
            grid.addLayout(hdr, 0, 0, 1, COLS)
            for k, cid in enumerate(ids):
                cb = QCheckBox(self.names[cid])
                cb.setObjectName("cls")
                cb.toggled.connect(self._on_toggle)
                grid.addWidget(cb, 1 + k // COLS, k % COLS)
                self.boxes[cid] = cb
            v.addWidget(g)
            self.groups.append((g, ids))
        v.addStretch(1)
        self.scroll.setWidget(host)
        self._reload_presets()
        self.set_selected(sel)

    def _reload_presets(self):
        self.preset.blockSignals(True)
        self.preset.clear()
        self.preset.addItem("— choose a preset —")
        for name in PRESETS:
            self.preset.addItem(name)
        user = load_user_presets()
        if user:
            self.preset.insertSeparator(self.preset.count())
            for name in user:
                self.preset.addItem(name)
        self.preset.blockSignals(False)

    # ----- state -----
    def selected(self) -> list[int]:
        return sorted(c for c, b in self.boxes.items() if b.isChecked())

    def set_selected(self, ids):
        ids = set(int(i) for i in ids)
        for c, b in self.boxes.items():
            b.blockSignals(True)
            b.setChecked(c in ids)
            b.blockSignals(False)
        self._on_toggle()

    def set_all(self, val):
        self._set_ids(list(self.boxes), val)

    def _set_ids(self, ids, val):
        for c in ids:
            if c in self.boxes:
                self.boxes[c].blockSignals(True)
                self.boxes[c].setChecked(val)
                self.boxes[c].blockSignals(False)
        self._on_toggle()

    def _on_toggle(self, *_):
        sel = self.selected()
        txt = ", ".join(self.names[c] for c in sel[:12]) + (" …" if len(sel) > 12 else "")
        self.count.setText(f"{len(sel)} of {len(self.boxes)} selected" + (f": {txt}" if sel else ""))
        self.changed.emit()

    def _filter(self, text):
        text = text.lower().strip()
        for g, ids in self.groups:
            any_vis = False
            for c in ids:
                vis = not text or text in self.names[c].lower()
                self.boxes[c].setVisible(vis)
                any_vis |= vis
            g.setVisible(any_vis)

    def _apply_preset(self, idx):
        name = self.preset.itemText(idx)
        ids = PRESETS.get(name)
        if ids is None:
            ids = load_user_presets().get(name)
        if ids is not None:
            self.set_selected(ids)
        self.preset.setCurrentIndex(0)

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if not ok or not name.strip():
            return
        user = load_user_presets()
        user[name.strip()] = self.selected()
        try:
            _user_presets_path().write_text(json.dumps(user, indent=2), encoding="utf-8")
        except OSError as e:
            QMessageBox.warning(self, "Save preset", str(e))
        self._reload_presets()
