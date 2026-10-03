"""Results viewer: event grid, detail view with box toggle, flags/notes, clip export, HTML report."""
from __future__ import annotations

import html
import json
import os
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QAbstractListModel, QModelIndex, QSize, Qt, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListView, QListWidget,
                               QListWidgetItem, QMenu, QMessageBox, QPlainTextEdit, QPushButton, QSplitter,
                               QToolButton, QVBoxLayout, QWidget)

from ..core.media import imread
from ..core.pipeline import RESULTS, write_csv
from ..core.render import ClipWriter, draw_boxes
from .canvas import ZoneCanvas
from .common import bgr_to_qimage, open_path, play_video_at, reveal, run_async

THUMB = QSize(224, 126)


class Results:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.data = json.loads((self.folder / RESULTS).read_text(encoding="utf-8"))
        self.files = {f["id"]: f for f in self.data["files"]}
        self.names = {int(k): v for k, v in self.data.get("class_names", {}).items()}
        self.events = self.data["events"]

    def path(self, rel):
        return str(self.folder / rel) if rel else None

    def save(self):
        tmp = self.folder / (RESULTS + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        os.replace(tmp, self.folder / RESULTS)
        write_csv(self.data, self.folder / "events.csv")


class ThumbCache:
    def __init__(self, on_ready, size=3000):
        self.cache: OrderedDict = OrderedDict()
        self.pending = set()
        self.size, self.on_ready = size, on_ready

    def get(self, path):
        if not path:
            return None
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        if path not in self.pending:
            self.pending.add(path)
            run_async(_load_thumb, path, on_done=self._done, on_error=lambda e, p=path: self.pending.discard(p))
        return None

    def _done(self, res):
        path, qimg = res
        self.pending.discard(path)
        self.cache[path] = QPixmap.fromImage(qimg)
        while len(self.cache) > self.size:
            self.cache.popitem(last=False)
        self.on_ready(path)


def _load_thumb(path):
    img = imread(path)
    if img is None:
        img = np.zeros((THUMB.height(), THUMB.width(), 3), np.uint8)
    h, w = img.shape[:2]
    s = min(THUMB.width() / w, THUMB.height() / h)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return path, bgr_to_qimage(img)


class ItemModel(QAbstractListModel):
    def __init__(self, viewer):
        super().__init__()
        self.v = viewer
        self.items = []   # (event, frame_rec)
        self.placeholder = QPixmap(THUMB)
        self.placeholder.fill(QColor(40, 43, 50))

    def set_items(self, items):
        self.beginResetModel()
        self.items = items
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self.items)

    def thumb_path(self, ev, fr):
        r = self.v.res
        if fr is None:
            if ev.get("thumb"):
                return r.path(ev["thumb"])
            fr = self.v.key_frame_rec(ev)
        if fr is None:
            return None
        return r.path(fr.get("ann") or fr.get("clean"))

    def data(self, idx, role=Qt.ItemDataRole.DisplayRole):
        if not idx.isValid():
            return None
        ev, fr = self.items[idx.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            t = (fr or {}).get("t") or ev.get("ocr_time") or ev.get("start_time")
            tt = t.replace("T", " ")[11:] if t else f"{ev['video_start_s']:.1f}s"
            star = "★ " if ev.get("flag") else ""
            objs = ", ".join(f"{k}" for k in sorted(ev["classes"], key=lambda k: -ev["classes"][k]))
            sub = f"{ev['n_frames']} fr" if fr is None else f"f{fr['f']}"
            return f"{star}{tt}  ·  {objs}\n{Path(self.v.res.files[ev['file_id']]['path']).name}  ·  {sub}"
        if role == Qt.ItemDataRole.DecorationRole:
            pm = self.v.thumbs.get(self.thumb_path(ev, fr))
            return pm or self.placeholder
        if role == Qt.ItemDataRole.ToolTipRole:
            return self.v.describe(ev)
        return None

    def refresh_path(self, path):
        for i, (ev, fr) in enumerate(self.items):
            if self.thumb_path(ev, fr) == path:
                ix = self.index(i)
                self.dataChanged.emit(ix, ix, [Qt.ItemDataRole.DecorationRole])


class ResultsViewer(QWidget):
    status = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.res: Results | None = None
        self.cur_ev = None
        self.cur_fr = None
        self.thumbs = ThumbCache(self._thumb_ready)
        self._build()

    # ---------- UI ----------
    def _build(self):
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        b = QPushButton("Open results folder…")
        b.clicked.connect(self.browse)
        bar.addWidget(b)
        self.folder_lbl = QLabel("No results loaded")
        self.folder_lbl.setObjectName("hint")
        bar.addWidget(self.folder_lbl, 1)
        rep = QPushButton("Export HTML report")
        rep.clicked.connect(self.export_report)
        bar.addWidget(rep)
        of = QPushButton("Open output folder")
        of.clicked.connect(lambda: self.res and open_path(self.res.folder))
        bar.addWidget(of)
        lay.addLayout(bar)

        flt = QHBoxLayout()
        self.cls_btn = QToolButton()
        self.cls_btn.setText("Objects: all ▾")
        self.cls_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.cls_menu = QMenu(self)
        self.cls_btn.setMenu(self.cls_menu)
        flt.addWidget(self.cls_btn)
        self.file_cb = QComboBox()
        self.file_cb.setMinimumWidth(200)
        self.file_cb.currentIndexChanged.connect(self.apply_filter)
        flt.addWidget(self.file_cb)
        self.flag_only = QCheckBox("★ flagged only")
        self.flag_only.toggled.connect(self.apply_filter)
        flt.addWidget(self.flag_only)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search time / file / OCR text / notes…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.apply_filter)
        flt.addWidget(self.search, 1)
        flt.addWidget(QLabel("Sort:"))
        self.sort_cb = QComboBox()
        self.sort_cb.addItems(["Time", "Confidence", "Duration (frames)", "File"])
        self.sort_cb.currentIndexChanged.connect(self.apply_filter)
        flt.addWidget(self.sort_cb)
        flt.addWidget(QLabel("Show:"))
        self.mode_cb = QComboBox()
        self.mode_cb.addItems(["One card per event", "Every saved frame"])
        self.mode_cb.currentIndexChanged.connect(self.apply_filter)
        flt.addWidget(self.mode_cb)
        lay.addLayout(flt)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.grid = QListView()
        self.grid.setViewMode(QListView.ViewMode.IconMode)
        self.grid.setResizeMode(QListView.ResizeMode.Adjust)
        self.grid.setMovement(QListView.Movement.Static)
        self.grid.setUniformItemSizes(True)
        self.grid.setIconSize(THUMB)
        self.grid.setGridSize(QSize(THUMB.width() + 16, THUMB.height() + 52))
        self.grid.setSpacing(4)
        self.grid.setWordWrap(True)
        self.model = ItemModel(self)
        self.grid.setModel(self.model)
        self.grid.selectionModel().currentChanged.connect(self._on_current)
        self.grid.doubleClicked.connect(lambda ix: self._open_current_image())
        split.addWidget(self.grid)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 0, 0, 0)
        tools = QHBoxLayout()
        self.boxes_cb = QCheckBox("Show boxes")
        self.boxes_cb.setChecked(True)
        self.boxes_cb.toggled.connect(self._show_frame)
        tools.addWidget(self.boxes_cb)
        tools.addStretch(1)
        for txt, fn in (("◀ Prev", lambda: self._step(-1)), ("Next ▶", lambda: self._step(1))):
            bb = QPushButton(txt)
            bb.clicked.connect(fn)
            tools.addWidget(bb)
        rl.addLayout(tools)
        self.view = ZoneCanvas()
        self.view.setMinimumHeight(300)
        rl.addWidget(self.view, 3)
        self.strip = QListWidget()
        self.strip.setViewMode(QListWidget.ViewMode.IconMode)
        self.strip.setFlow(QListWidget.Flow.LeftToRight)
        self.strip.setWrapping(False)
        self.strip.setIconSize(QSize(128, 72))
        self.strip.setFixedHeight(112)
        self.strip.setHorizontalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.strip.currentRowChanged.connect(self._on_strip)
        rl.addWidget(self.strip)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rl.addWidget(self.info)
        acts = QHBoxLayout()
        self.flag_btn = QPushButton("★ Flag")
        self.flag_btn.setCheckable(True)
        self.flag_btn.toggled.connect(self._toggle_flag)
        acts.addWidget(self.flag_btn)
        for txt, fn in (("Open image", self._open_current_image), ("Show in folder", self._reveal_current),
                        ("Play saved clip", self._play_clip), ("Open source video at time", self._play_source),
                        ("Export clip…", self._export_clip)):
            bb = QPushButton(txt)
            bb.clicked.connect(fn)
            acts.addWidget(bb)
        acts.addStretch(1)
        rl.addLayout(acts)
        self.note = QPlainTextEdit()
        self.note.setPlaceholderText("Notes for this event (saved automatically)…")
        self.note.setFixedHeight(60)
        self.note.textChanged.connect(self._note_changed)
        rl.addWidget(self.note)
        split.addWidget(right)
        split.setSizes([760, 740])
        lay.addWidget(split, 1)

    # ---------- loading ----------
    def browse(self):
        d = QFileDialog.getExistingDirectory(self, "Open results folder")
        if d:
            self.load(d)

    def load(self, folder):
        try:
            self.res = Results(folder)
        except Exception as e:
            QMessageBox.warning(self, "Open results", f"Could not open results in {folder}:\n{e}")
            return
        r = self.res
        self.folder_lbl.setText(f"{folder}   ·   {len(r.events)} events in {len(r.files)} files"
                                + ("" if r.data.get("complete") else "   (incomplete run)"))
        self.cls_menu.clear()
        classes = sorted({c for e in r.events for c in e["classes"]})
        self.cls_actions = {}
        for c in classes:
            a = QAction(c, self, checkable=True, checked=True)
            a.toggled.connect(self.apply_filter)
            self.cls_menu.addAction(a)
            self.cls_actions[c] = a
        self.cls_menu.addSeparator()
        for txt, val in (("Select all", True), ("Select none", False)):
            a = QAction(txt, self)
            a.triggered.connect(lambda _=False, v=val: self._set_all_cls(v))
            self.cls_menu.addAction(a)
        self.file_cb.blockSignals(True)
        self.file_cb.clear()
        self.file_cb.addItem("All files", None)
        for f in r.data["files"]:
            if f["n_events"]:
                self.file_cb.addItem(f"{Path(f['path']).name} ({f['n_events']})", f["id"])
        self.file_cb.blockSignals(False)
        self.apply_filter()

    def _set_all_cls(self, v):
        for a in self.cls_actions.values():
            a.blockSignals(True)
            a.setChecked(v)
            a.blockSignals(False)
        self.apply_filter()

    def apply_filter(self, *_):
        r = self.res
        if not r:
            return
        want = {c for c, a in self.cls_actions.items() if a.isChecked()}
        self.cls_btn.setText("Objects: all ▾" if len(want) == len(self.cls_actions) else f"Objects: {len(want)} ▾")
        fid = self.file_cb.currentData()
        q = self.search.text().lower().strip()
        evs = []
        for e in r.events:
            if not (set(e["classes"]) & want):
                continue
            if fid is not None and e["file_id"] != fid:
                continue
            if self.flag_only.isChecked() and not e.get("flag"):
                continue
            if q:
                hay = " ".join(str(x) for x in (e.get("start_time"), e.get("ocr_text"), e.get("ocr_time"),
                                                 e.get("note"), r.files[e["file_id"]]["path"], *e["classes"]))
                if q not in hay.lower():
                    continue
            evs.append(e)
        key = {0: lambda e: (e.get("ocr_time") or e.get("start_time") or "", e["file_id"], e["start_frame"]),
               1: lambda e: -e["max_conf"], 2: lambda e: -e["n_frames"],
               3: lambda e: (r.files[e["file_id"]]["path"], e["start_frame"])}[self.sort_cb.currentIndex()]
        evs.sort(key=key)
        if self.mode_cb.currentIndex() == 0:
            items = [(e, None) for e in evs]
        else:
            items = [(e, fr) for e in evs for fr in e["frames"]]
        self.model.set_items(items)
        self.status.emit(f"{len(evs)} events shown")
        if items:
            self.grid.setCurrentIndex(self.model.index(0))
        else:
            self._clear_detail()

    # ---------- detail ----------
    def key_frame_rec(self, ev):
        if not ev["frames"]:
            return None
        return min(ev["frames"], key=lambda fr: abs(fr["f"] - ev["key_frame"]))

    def describe(self, ev):
        r = self.res
        f = r.files[ev["file_id"]]
        lines = [f"<b>Event #{ev['id']}</b> — {html.escape(Path(f['path']).name)}"]
        if ev.get("start_time"):
            lines.append(f"Time: {ev['start_time'].replace('T', ' ')} → {(ev.get('end_time') or '').replace('T', ' ')[11:]}")
        if ev.get("ocr_text"):
            lines.append(f"Burned-in text: <code>{html.escape(ev['ocr_text'])}</code>")
        if f["kind"] == "video":
            lines.append(f"Video position: {ev['video_start_s']:.2f}s – {ev['video_end_s']:.2f}s "
                         f"(frames {ev['start_frame']}–{ev['end_frame']}, {ev['n_frames']} with detections)")
        lines.append("Objects: " + ", ".join(f"{k} ({v} fr)" for k, v in ev["classes"].items())
                     + f" · max confidence {ev['max_conf']:.2f}")
        return "<br>".join(lines)

    def _clear_detail(self):
        self.cur_ev = self.cur_fr = None
        self.strip.clear()
        self.info.setText("")
        self.view.set_image(np.zeros((360, 640, 3), np.uint8))
        self.view.set_detections()

    def _on_current(self, cur, prev):
        if not cur.isValid():
            return
        ev, fr = self.model.items[cur.row()]
        if ev is not self.cur_ev:
            self.cur_ev = ev
            self._fill_strip(ev)
            self.info.setText(self.describe(ev))
            self.flag_btn.blockSignals(True)
            self.flag_btn.setChecked(bool(ev.get("flag")))
            self.flag_btn.blockSignals(False)
            self.note.blockSignals(True)
            self.note.setPlainText(ev.get("note", ""))
            self.note.blockSignals(False)
        target = fr or self.key_frame_rec(ev)
        if target is None:
            self._on_strip(-1)
        else:
            row = ev["frames"].index(target)
            self.strip.setCurrentRow(row)
            self._on_strip(row)

    def _fill_strip(self, ev):
        self.strip.blockSignals(True)
        self.strip.clear()
        for fr in ev["frames"]:
            it = QListWidgetItem(f"f{fr['f']}")
            p = self.res.path(fr.get("ann") or fr.get("clean"))
            pm = self.thumbs.get(p)
            it.setIcon(QIcon(pm) if pm else QIcon(self.model.placeholder))
            it.setData(Qt.ItemDataRole.UserRole, p)
            self.strip.addItem(it)
        self.strip.blockSignals(False)

    def _on_strip(self, row):
        if self.cur_ev is not None and not self.cur_ev["frames"] and self.cur_ev.get("thumb"):
            self.cur_fr = None
            p = self.res.path(self.cur_ev["thumb"])
            self.view.set_image(imread(p) if os.path.exists(p) else np.zeros((360, 640, 3), np.uint8))
            self.view.set_detections()
            return
        if self.cur_ev is None or row < 0 or row >= len(self.cur_ev["frames"]):
            return
        self.cur_fr = self.cur_ev["frames"][row]
        self._show_frame()

    def _show_frame(self, *_):
        fr = self.cur_fr
        if fr is None:
            return
        r = self.res
        clean = r.path(fr.get("clean"))
        if clean and os.path.exists(clean):
            img = imread(clean)
            self.view.set_image(img)
            self.view.set_detections(fr["boxes"] if self.boxes_cb.isChecked() else [], names=r.names)
        else:
            p = r.path(fr.get("ann"))
            img = imread(p) if p and os.path.exists(p) else np.zeros((360, 640, 3), np.uint8)
            self.view.set_image(img)
            self.view.set_detections()
        t = fr.get("t")
        self.status.emit(f"Frame {fr['f']}" + (f" · {t.replace('T', ' ')}" if t else ""))

    def _thumb_ready(self, path):
        self.model.refresh_path(path)
        for i in range(self.strip.count()):
            it = self.strip.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == path:
                it.setIcon(QIcon(self.thumbs.cache[path]))

    def _step(self, d):
        ix = self.grid.currentIndex()
        n = self.model.rowCount()
        if n:
            row = min(max(0, (ix.row() if ix.isValid() else -1) + d), n - 1)
            self.grid.setCurrentIndex(self.model.index(row))

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_F and self.cur_ev is not None:
            self.flag_btn.toggle()
        elif e.key() == Qt.Key.Key_B:
            self.boxes_cb.toggle()
        else:
            super().keyPressEvent(e)

    # ---------- actions ----------
    def _toggle_flag(self, on):
        if self.cur_ev is not None:
            self.cur_ev["flag"] = bool(on)
            self.res.save()
            ix = self.grid.currentIndex()
            self.model.dataChanged.emit(ix, ix)

    def _note_changed(self):
        if self.cur_ev is not None:
            self.cur_ev["note"] = self.note.toPlainText()
            self.res.save()

    def _current_image_path(self):
        if self.cur_fr is None:
            return None
        rel = self.cur_fr.get("ann") if self.boxes_cb.isChecked() else self.cur_fr.get("clean")
        return self.res.path(rel or self.cur_fr.get("clean") or self.cur_fr.get("ann"))

    def _open_current_image(self):
        p = self._current_image_path()
        if p:
            open_path(p)

    def _reveal_current(self):
        p = self._current_image_path()
        if p:
            reveal(p)

    def _play_clip(self):
        if self.cur_ev is None:
            return
        if self.cur_ev.get("clip"):
            open_path(self.res.path(self.cur_ev["clip"]))
        else:
            QMessageBox.information(self, "Clip", "No clip was saved for this event. Use “Export clip…”.")

    def _play_source(self):
        if self.cur_ev is None:
            return
        f = self.res.files[self.cur_ev["file_id"]]
        if f["kind"] != "video":
            open_path(f["path"])
            return
        sec = (self.cur_fr["f"] if self.cur_fr else self.cur_ev["start_frame"]) / (f["fps"] or 30)
        if not play_video_at(f["path"], sec):
            self.status.emit(f"VLC not found: opened the video, seek to {sec:.1f}s manually")

    def _export_clip(self):
        if self.cur_ev is None:
            return
        ev = self.cur_ev
        f = self.res.files[ev["file_id"]]
        if f["kind"] != "video":
            QMessageBox.information(self, "Export clip", "This event comes from a photo.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Export clip")
        form = QFormLayout(dlg)
        pre, post = QDoubleSpinBox(), QDoubleSpinBox()
        for sp, v in ((pre, 3.0), (post, 3.0)):
            sp.setRange(0, 600)
            sp.setSuffix(" s")
            sp.setValue(v)
        ann = QCheckBox("Draw detection boxes")
        ann.setChecked(True)
        whole = QCheckBox("Whole event (otherwise around the current frame only)")
        whole.setChecked(True)
        form.addRow("Seconds before:", pre)
        form.addRow("Seconds after:", post)
        form.addRow(whole)
        form.addRow(ann)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        fps = f["fps"] or 30
        c = self.cur_fr["f"] if self.cur_fr else ev["key_frame"]
        a0, b0 = (ev["start_frame"], ev["end_frame"]) if whole.isChecked() else (c, c)
        a = max(0, int(a0 - pre.value() * fps))
        b = min(f["frames"] - 1, int(b0 + post.value() * fps))
        out_dir = self.res.folder / "clips"
        out_dir.mkdir(exist_ok=True)
        out = out_dir / f"{Path(f['path']).stem}_event{ev['id']}_f{a}-{b}{'_boxes' if ann.isChecked() else ''}.mp4"
        boxes = {fr["f"]: fr["boxes"] for fr in ev["frames"]} if ann.isChecked() else {}
        self.status.emit(f"Exporting clip {out.name} …")
        run_async(export_clip, f["path"], a, b, fps, str(out), boxes, self.res.names,
                  on_done=lambda p: (self.status.emit(f"Clip saved: {p}"), reveal(p)),
                  on_error=lambda e: QMessageBox.warning(self, "Export clip", e))

    def export_report(self):
        if not self.res:
            return
        r = self.res
        evs = [e for e in r.events if e.get("flag")]
        scope = "flagged"
        if not evs:
            evs = [self.model.items[i][0] for i in range(self.model.rowCount())]
            evs = list({id(e): e for e in evs}.values())
            scope = "shown"
        out = r.folder / "report.html"
        out.write_text(build_report(r, evs, scope), encoding="utf-8")
        open_path(out)
        self.status.emit(f"Report with {len(evs)} {scope} events written to {out}")


def export_clip(video, a, b, fps, out, boxes, names):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    w = None
    last = []
    try:
        for f in range(a, b + 1):
            ok, img = cap.read()
            if not ok:
                break
            if w is None:
                w = ClipWriter(out, fps, (img.shape[1], img.shape[0]))
            if boxes:
                if f in boxes:
                    last = boxes[f]
                elif f > max(boxes) or f < min(boxes):
                    last = []
                img = draw_boxes(img, np.array(last).reshape(-1, 6), names)
            w.write(img)
    finally:
        if w:
            w.close()
        cap.release()
    return out


def build_report(r: Results, evs, scope):
    rows = []
    for e in evs:
        f = r.files[e["file_id"]]
        kf = min(e["frames"], key=lambda fr: abs(fr["f"] - e["key_frame"])) if e["frames"] else None
        img = (kf.get("ann") or kf.get("clean")) if kf else e.get("thumb")
        rows.append(f"""<tr><td>{'★' if e.get('flag') else ''} #{e['id']}</td>
<td>{f'<a href="{html.escape(img)}"><img src="{html.escape(img)}"></a>' if img else ''}</td>
<td><b>{html.escape((e.get('ocr_time') or e.get('start_time') or '').replace('T', ' '))}</b><br>
{html.escape(Path(f['path']).name)}<br>video {e['video_start_s']:.1f}s – {e['video_end_s']:.1f}s<br>
<code>{html.escape(e.get('ocr_text') or '')}</code></td>
<td>{html.escape(', '.join(f'{k} ({v})' for k, v in e['classes'].items()))}<br>max conf {e['max_conf']:.2f}</td>
<td>{html.escape(e.get('note') or '').replace(chr(10), '<br>')}</td>
<td>{f'<a href="{html.escape(e["clip"])}">clip</a>' if e.get('clip') else ''}</td></tr>""")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>YOLO Finder report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#222}}table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #ddd;padding:8px;vertical-align:top;text-align:left}}img{{width:360px}}
th{{background:#f3f4f6}}code{{color:#555}}</style></head><body>
<h1>YOLO Finder report</h1>
<p>Results: {html.escape(str(r.folder))}<br>Generated from run {html.escape(r.data.get('created', ''))} ·
{len(evs)} {scope} events · {len(r.files)} source files</p>
<table><tr><th>Event</th><th>Key frame</th><th>Time / source</th><th>Objects</th><th>Notes</th><th></th></tr>
{''.join(rows)}</table></body></html>"""
