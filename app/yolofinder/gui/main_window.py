"""Main application window."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QSlider, QSpinBox, QSplitter,
                               QStackedWidget, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..core import analysis as A
from ..core.config import JobSettings, Zone, app_home, norm
from ..core.detector import KNOWN_MODELS
from ..core.media import find_media, grab_frame, probe
from ..core.render import draw_boxes
from .canvas import ZoneCanvas
from .class_picker import ClassPicker
from .common import bgr_to_pixmap, human_duration, human_size, open_path, run_async, shared_detector
from .viewer import ResultsViewer
from .worker import JobWorker

MODEL_INFO = {
    "yolo11n.pt": "Nano – fastest, least accurate",
    "yolo11s.pt": "Small – fast",
    "yolo11m.pt": "Medium – balanced (recommended)",
    "yolo11l.pt": "Large – more accurate",
    "yolo11x.pt": "Extra large – most accurate, slowest",
}
QUALITY = {
    "Quick scan": ("yolo11s.pt", 960, 2),
    "Balanced": ("yolo11m.pt", 1280, 1),
    "Thorough": ("yolo11x.pt", 1280, 1),
}
PAGES = ["Sources", "Zones & preview", "What to find", "Detection", "Output", "Timestamps", "Run", "Results"]


def hint(text):
    l = QLabel(text)
    l.setObjectName("hint")
    l.setWordWrap(True)
    return l


def h1(text):
    l = QLabel(text)
    l.setObjectName("h1")
    return l


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(1500, 950)
        self.setAcceptDrops(True)
        self.media = []                 # list[MediaInfo]
        self.zones: dict = {}           # norm(path) -> [zone dict]
        self.ts_regions: dict = {}
        self.job_path = None
        self.worker = None
        self.preview_path = None
        self.preview_frame = None
        self._t_start = 0.0
        self._build()
        self._load_last_job()

    # ================= layout =================
    def _build(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        body = QHBoxLayout()
        body.setSpacing(0)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(190)
        for i, p in enumerate(PAGES):
            self.nav.addItem(QListWidgetItem(f"{i + 1}. {p}" if i < 6 else p))
        self.stack = QStackedWidget()
        for build in (self._page_sources, self._page_zones, self._page_targets, self._page_detection,
                      self._page_output, self._page_timestamps, self._page_run, self._page_results):
            w = QWidget()
            lay = QVBoxLayout(w)
            lay.setContentsMargins(18, 14, 18, 10)
            build(lay)
            self.stack.addWidget(w)
        self.nav.currentRowChanged.connect(self._on_page)
        body.addWidget(self.nav)
        body.addWidget(self.stack, 1)
        root.addLayout(body, 1)
        root.addWidget(self._bottom_bar())
        self.setCentralWidget(central)
        self._menus()
        self.nav.setCurrentRow(0)

    def _menus(self):
        m = self.menuBar().addMenu("&File")
        for txt, key, fn in (("New job", QKeySequence.StandardKey.New, self.new_job),
                             ("Open job…", QKeySequence.StandardKey.Open, self.open_job),
                             ("Save job", QKeySequence.StandardKey.Save, self.save_job),
                             ("Save job as…", QKeySequence.StandardKey.SaveAs, lambda: self.save_job(True))):
            a = QAction(txt, self)
            a.setShortcut(key)
            a.triggered.connect(fn)
            m.addAction(a)
        m.addSeparator()
        a = QAction("Open results folder…", self)
        a.triggered.connect(lambda: (self.nav.setCurrentRow(7), self.viewer.browse()))
        m.addAction(a)
        m.addSeparator()
        a = QAction("Exit", self)
        a.triggered.connect(self.close)
        m.addAction(a)
        h = self.menuBar().addMenu("&Help")
        a = QAction("Open app data folder (models, logs)", self)
        a.triggered.connect(lambda: open_path(app_home()))
        h.addAction(a)
        a = QAction("About", self)
        a.triggered.connect(self._about)
        h.addAction(a)

    def _bottom_bar(self):
        f = QFrame()
        f.setStyleSheet("QFrame{background:#1b1d22;border-top:1px solid #30343d}")
        lay = QHBoxLayout(f)
        lay.setContentsMargins(14, 8, 14, 8)
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setMinimumWidth(300)
        lay.addWidget(self.status_lbl, 1)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFixedWidth(360)
        self.bar.setFormat("%p%")
        lay.addWidget(self.bar)
        self.eta_lbl = QLabel("")
        self.eta_lbl.setMinimumWidth(130)
        lay.addWidget(self.eta_lbl)
        self.start_btn = QPushButton("▶  Start analysis")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start)
        lay.addWidget(self.start_btn)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        lay.addWidget(self.stop_btn)
        return f

    def _on_page(self, i):
        self.stack.setCurrentIndex(i)
        if i == 1:
            self._refresh_zone_files()

    def set_status(self, t):
        self.status_lbl.setText(t)

    # ================= pages =================
    def _page_sources(self, lay):
        lay.addWidget(h1("Sources"))
        lay.addWidget(hint("Add folders or files with videos and photos (you can also drag & drop them onto the window). "
                           "Untick files you want to skip."))
        top = QHBoxLayout()
        g = QGroupBox("Inputs")
        gl = QVBoxLayout(g)
        self.inputs = QListWidget()
        self.inputs.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.inputs.setMaximumHeight(120)
        gl.addWidget(self.inputs)
        row = QHBoxLayout()
        for txt, fn in (("Add folder…", self.add_folder), ("Add files…", self.add_files),
                        ("Remove", self.remove_inputs), ("Clear", self.clear_inputs)):
            b = QPushButton(txt)
            b.clicked.connect(fn)
            row.addWidget(b)
        self.recursive = QCheckBox("Include subfolders")
        self.recursive.setChecked(True)
        self.recursive.toggled.connect(self.rescan)
        row.addWidget(self.recursive)
        row.addStretch(1)
        gl.addLayout(row)
        top.addWidget(g, 3)
        g2 = QGroupBox("Output folder")
        g2l = QVBoxLayout(g2)
        self.output = QLineEdit()
        self.output.setPlaceholderText("Where frames, clips and the results file are written")
        g2l.addWidget(self.output)
        r2 = QHBoxLayout()
        b = QPushButton("Browse…")
        b.clicked.connect(self.pick_output)
        r2.addWidget(b)
        b = QPushButton("Open")
        b.clicked.connect(lambda: self.output.text() and os.path.isdir(self.output.text()) and open_path(self.output.text()))
        r2.addWidget(b)
        r2.addStretch(1)
        g2l.addLayout(r2)
        g2l.addWidget(hint("Re-running into the same output folder reuses cached detections, so changing filters, "
                           "zones or classes afterwards is fast."))
        g2l.addStretch(1)
        top.addWidget(g2, 2)
        lay.addLayout(top)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["File", "Type", "Resolution", "FPS", "Duration", "Frames", "Size", "Folder"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.summary = QLabel("No files")
        row.addWidget(self.summary, 1)
        for txt, val in (("Tick all", True), ("Untick all", False)):
            b = QPushButton(txt)
            b.clicked.connect(lambda _=False, v=val: self._tick_all(v))
            row.addWidget(b)
        lay.addLayout(row)

    def _page_zones(self, lay):
        lay.addWidget(h1("Zones & preview"))
        lay.addWidget(hint("Pick a file, scrub to a representative frame and draw areas to IGNORE (e.g. your own parked "
                           "car, a busy road, a flag that moves) or ONLY-HERE areas. A detection counts as inside a zone "
                           "when its bottom-centre point (where it touches the ground) is inside. You can also mark where "
                           "the burned-in timestamp is."))
        split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("Files"))
        self.zone_files = QListWidget()
        self.zone_files.currentRowChanged.connect(self._zone_file_selected)
        ll.addWidget(self.zone_files, 1)
        cg = QGroupBox("Copy this file's zones to…")
        cl = QVBoxLayout(cg)
        for txt, mode in (("All files", "all"), ("Files with same resolution", "res"), ("Files whose name contains…", "name")):
            b = QPushButton(txt)
            b.clicked.connect(lambda _=False, m=mode: self.copy_zones(m))
            cl.addWidget(b)
        ll.addWidget(cg)
        split.addWidget(left)

        mid = QWidget()
        ml = QVBoxLayout(mid)
        ml.setContentsMargins(0, 0, 0, 0)
        tb = QHBoxLayout()
        self.tool_group = QButtonGroup(self)
        for tool, txt in (("pan", "✋ Pan / select"), ("ignore-rect", "⬛ Ignore box"), ("ignore-poly", "⬟ Ignore shape"),
                          ("include-rect", "🟩 Only-here box"), ("include-poly", "⬟ Only-here shape"),
                          ("timestamp", "🕒 Timestamp area")):
            b = QToolButton()
            b.setText(txt)
            b.setCheckable(True)
            b.setProperty("tool", tool)
            self.tool_group.addButton(b)
            tb.addWidget(b)
            if tool == "pan":
                b.setChecked(True)
        self.tool_group.buttonClicked.connect(lambda b: self.canvas.set_tool(b.property("tool")))
        tb.addStretch(1)
        for txt, fn in (("Fit", lambda: self.canvas.fit()), ("Delete selected", lambda: self.canvas.delete_selected()),
                        ("Clear file", lambda: self.canvas.clear_zones())):
            b = QPushButton(txt)
            b.clicked.connect(fn)
            tb.addWidget(b)
        ml.addLayout(tb)
        self.canvas = ZoneCanvas()
        self.canvas.changed.connect(self._zones_changed)
        self.canvas.status.connect(self.set_status)
        ml.addWidget(self.canvas, 1)
        sl = QHBoxLayout()
        self.scrub = QSlider(Qt.Orientation.Horizontal)
        self.scrub.setEnabled(False)
        self.scrub.valueChanged.connect(lambda v: self._scrub_timer.start(120))
        self._scrub_timer = QTimer(self)
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.timeout.connect(self._load_preview_frame)
        sl.addWidget(self.scrub, 1)
        self.scrub_lbl = QLabel("")
        self.scrub_lbl.setMinimumWidth(160)
        sl.addWidget(self.scrub_lbl)
        ml.addLayout(sl)
        tr = QHBoxLayout()
        b = QPushButton("🔍 Test detection on this frame")
        b.clicked.connect(self.test_detection)
        tr.addWidget(b)
        b = QPushButton("🕒 Test timestamp reading")
        b.clicked.connect(self.test_ocr)
        tr.addWidget(b)
        self.test_lbl = QLabel("")
        self.test_lbl.setWordWrap(True)
        tr.addWidget(self.test_lbl, 1)
        ml.addLayout(tr)
        split.addWidget(mid)
        split.setSizes([260, 1100])
        lay.addWidget(split, 1)

    def _page_targets(self, lay):
        lay.addWidget(h1("What to find"))
        lay.addWidget(hint("Tick everything you want reported. Presets are a quick start; you can save your own."))
        self.picker = ClassPicker()
        lay.addWidget(self.picker, 1)

    def _page_detection(self, lay):
        lay.addWidget(h1("Detection"))
        q = QHBoxLayout()
        q.addWidget(QLabel("Quality preset:"))
        for name, (m, sz, st) in QUALITY.items():
            b = QPushButton(name)
            b.setToolTip(f"{m}, image size {sz}, every {st}. frame")
            b.clicked.connect(lambda _=False, v=(m, sz, st): self._apply_quality(*v))
            q.addWidget(b)
        q.addStretch(1)
        lay.addLayout(q)
        grid = QGridLayout()
        g = QGroupBox("Model")
        f = QFormLayout(g)
        mrow = QHBoxLayout()
        self.model_cb = QComboBox()
        for m in KNOWN_MODELS:
            self.model_cb.addItem(f"{m}  –  {MODEL_INFO[m]}", m)
        self.model_cb.currentIndexChanged.connect(self._model_changed)
        mrow.addWidget(self.model_cb, 1)
        b = QPushButton("Custom model…")
        b.clicked.connect(self.pick_model)
        mrow.addWidget(b)
        f.addRow("Model:", mrow)
        self.imgsz = QComboBox()
        for v in (640, 960, 1280, 1600, 1920):
            self.imgsz.addItem(str(v), v)
        f.addRow("Image size:", self.imgsz)
        f.addRow(hint("Larger = finds smaller/farther objects, slower. 1280 is good for 1440p dashcams."))
        self.device = QComboBox()
        self.device.addItems(["auto", "cuda:0", "cpu"])
        f.addRow("Device:", self.device)
        self.half = QCheckBox("Half precision (FP16) on GPU – faster")
        f.addRow(self.half)
        self.batch = QSpinBox()
        self.batch.setRange(1, 64)
        f.addRow("Batch size:", self.batch)
        grid.addWidget(g, 0, 0)

        g = QGroupBox("Sensitivity")
        f = QFormLayout(g)
        crow = QHBoxLayout()
        self.conf_sl = QSlider(Qt.Orientation.Horizontal)
        self.conf_sl.setRange(10, 95)
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.10, 0.95)
        self.conf.setSingleStep(0.05)
        self.conf_sl.valueChanged.connect(lambda v: self.conf.setValue(v / 100))
        self.conf.valueChanged.connect(lambda v: self.conf_sl.setValue(int(round(v * 100))))
        crow.addWidget(self.conf_sl, 1)
        crow.addWidget(self.conf)
        f.addRow("Min confidence:", crow)
        f.addRow(hint("Lower = more detections and more false alarms. 0.3–0.4 is a good start."))
        self.frame_step = QSpinBox()
        self.frame_step.setRange(1, 300)
        self.frame_step.setSuffix(". frame")
        self.frame_step.setPrefix("every ")
        f.addRow("Analyse:", self.frame_step)
        f.addRow(hint("Every frame for time-lapse/parking footage. For normal 30 fps video, every 3rd–5th frame is "
                      "usually enough and 3–5x faster."))
        self.min_box = QDoubleSpinBox()
        self.min_box.setRange(0, 50)
        self.min_box.setSuffix(" % of frame height")
        f.addRow("Ignore objects smaller than:", self.min_box)
        grid.addWidget(g, 0, 1)

        g = QGroupBox("Parked / stationary objects")
        f = QFormLayout(g)
        self.stationary = QCheckBox("Ignore objects that don't move (parked cars etc.)")
        f.addRow(self.stationary)
        self.stat_people = QCheckBox("Also ignore people standing still")
        f.addRow(self.stat_people)
        self.stat_win = QDoubleSpinBox()
        self.stat_win.setRange(0.5, 120)
        self.stat_win.setSuffix(" s (real time)")
        f.addRow("Look ± window:", self.stat_win)
        self.stat_ratio = QDoubleSpinBox()
        self.stat_ratio.setRange(0.1, 1.0)
        self.stat_ratio.setSingleStep(0.05)
        f.addRow("Stationary if present in ≥", self.stat_ratio)
        self.stat_iou = QDoubleSpinBox()
        self.stat_iou.setRange(0.3, 0.98)
        self.stat_iou.setSingleStep(0.05)
        f.addRow("…at the same place (IoU ≥)", self.stat_iou)
        grid.addWidget(g, 1, 0)

        g = QGroupBox("Events")
        f = QFormLayout(g)
        self.gap = QDoubleSpinBox()
        self.gap.setRange(0, 600)
        self.gap.setSuffix(" s (real time)")
        f.addRow("Merge detections closer than:", self.gap)
        f.addRow(hint("Consecutive detections are grouped into one event. The results viewer shows one card per event, "
                      "while every frame is still saved (see Output)."))
        self.anchor = QComboBox()
        self.anchor.addItem("Bottom centre (feet / wheels)", "bottom")
        self.anchor.addItem("Box centre", "center")
        f.addRow("Zone test point:", self.anchor)
        grid.addWidget(g, 1, 1)
        lay.addLayout(grid)
        lay.addStretch(1)

    def _page_output(self, lay):
        lay.addWidget(h1("Output"))
        grid = QGridLayout()
        g = QGroupBox("Still frames")
        f = QFormLayout(g)
        self.save_frames = QCheckBox("Save still frames (full frame, never cropped)")
        f.addRow(self.save_frames)
        self.save_clean = QCheckBox("Clean copy (original pixels) → frames/")
        f.addRow(self.save_clean)
        self.save_ann = QCheckBox("Annotated copy (with boxes) → annotated/")
        f.addRow(self.save_ann)
        self.per_event = QSpinBox()
        self.per_event.setRange(0, 100000)
        self.per_event.setSpecialValueText("all frames")
        f.addRow("Frames per event:", self.per_event)
        f.addRow(hint("0 = save every frame of every event. Set e.g. 10 to keep only 10 evenly spaced frames "
                      "(always including the best one) for long events in normal-speed video."))
        self.jpeg_q = QSpinBox()
        self.jpeg_q.setRange(50, 100)
        f.addRow("JPEG quality:", self.jpeg_q)
        grid.addWidget(g, 0, 0)
        g = QGroupBox("Video clips")
        f = QFormLayout(g)
        self.save_clips = QCheckBox("Save a video clip for every event → clips/")
        f.addRow(self.save_clips)
        self.clip_ann = QCheckBox("Draw boxes in clips")
        f.addRow(self.clip_ann)
        self.clip_pre = QDoubleSpinBox()
        self.clip_pre.setRange(0, 600)
        self.clip_pre.setSuffix(" s of video")
        f.addRow("Seconds before event:", self.clip_pre)
        self.clip_post = QDoubleSpinBox()
        self.clip_post.setRange(0, 600)
        self.clip_post.setSuffix(" s of video")
        f.addRow("Seconds after event:", self.clip_post)
        f.addRow(hint("Tip: untick still frames and tick clips to get only videos. You can also export a clip around "
                      "any frame later from the Results page."))
        grid.addWidget(g, 0, 1)
        g = QGroupBox("Annotation style")
        f = QFormLayout(g)
        self.box_th = QSpinBox()
        self.box_th.setRange(1, 12)
        f.addRow("Box thickness:", self.box_th)
        self.box_lbl = QCheckBox("Show class labels")
        f.addRow(self.box_lbl)
        self.box_conf = QCheckBox("Show confidence")
        f.addRow(self.box_conf)
        grid.addWidget(g, 1, 0)
        lay.addLayout(grid)
        lay.addWidget(hint("Output files are named DATE_TIME_source_fFRAME_objects.jpg so they sort chronologically "
                           "across cameras. results.json and events.csv describe every event."))
        lay.addStretch(1)

    def _page_timestamps(self, lay):
        lay.addWidget(h1("Timestamps"))
        g = QGroupBox("Burned-in date/time (OCR)")
        f = QFormLayout(g)
        self.ocr = QCheckBox("Read the date/time printed on the video (dashcams, CCTV)")
        f.addRow(self.ocr)
        self.date_order = QComboBox()
        for txt, v in (("Automatic", "auto"), ("Day-Month-Year", "DMY"), ("Month-Day-Year", "MDY"), ("Year-Month-Day", "YMD")):
            self.date_order.addItem(txt, v)
        f.addRow("Date order:", self.date_order)
        f.addRow(hint("By default the top and bottom of the frame are searched automatically. If that fails, draw a "
                      "“Timestamp area” on the Zones page. The real-time speed of time-lapse recordings is worked out "
                      "from several readings."))
        b = QPushButton("Test on the preview frame")
        b.clicked.connect(self.test_ocr)
        f.addRow(b)
        lay.addWidget(g)
        g = QGroupBox("Fallbacks")
        f = QFormLayout(g)
        self.fn_time = QCheckBox("Use date/time in the file name (e.g. 2026_1002_123843_xxx.mp4, IMG_20231004_142233.jpg)")
        f.addRow(self.fn_time)
        self.tl_factor = QDoubleSpinBox()
        self.tl_factor.setRange(0, 3600)
        self.tl_factor.setSpecialValueText("auto")
        self.tl_factor.setSuffix(" real s per video s")
        f.addRow("Time-lapse factor:", self.tl_factor)
        f.addRow(hint("Leave on auto unless you know it (e.g. 10 for a 3 fps parking-mode time-lapse played at 30 fps). "
                      "Photos use EXIF time first."))
        lay.addWidget(g)
        lay.addStretch(1)

    def _page_run(self, lay):
        lay.addWidget(h1("Run"))
        top = QHBoxLayout()
        self.run_info = QLabel("Press “Start analysis”.")
        top.addWidget(self.run_info, 1)
        self.file_bar = QProgressBar()
        self.file_bar.setRange(0, 1000)
        self.file_bar.setFormat("file %p%")
        self.file_bar.setFixedWidth(260)
        top.addWidget(self.file_bar)
        lay.addLayout(top)
        split = QSplitter(Qt.Orientation.Vertical)
        self.live = QLabel("Live preview")
        self.live.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live.setMinimumHeight(240)
        self.live.setStyleSheet("background:#121317;color:#555;")
        split.addWidget(self.live)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        split.addWidget(self.log)
        split.setSizes([520, 240])
        lay.addWidget(split, 1)

    def _page_results(self, lay):
        self.viewer = ResultsViewer()
        self.viewer.status.connect(self.set_status)
        lay.addWidget(self.viewer, 1)
        QShortcut(QKeySequence("F"), self.viewer, activated=lambda: self.viewer.flag_btn.toggle())
        QShortcut(QKeySequence("B"), self.viewer, activated=lambda: self.viewer.boxes_cb.toggle())
        QShortcut(QKeySequence(Qt.Key.Key_Right), self.viewer, activated=lambda: self.viewer._step(1))
        QShortcut(QKeySequence(Qt.Key.Key_Left), self.viewer, activated=lambda: self.viewer._step(-1))

    # ================= sources =================
    def add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Add folder with videos/photos")
        if d:
            self._add_inputs([d])

    def add_files(self):
        fs, _ = QFileDialog.getOpenFileNames(self, "Add videos/photos", "",
                                             "Media (*.mp4 *.mov *.avi *.mkv *.m4v *.wmv *.ts *.mts *.webm *.jpg *.jpeg "
                                             "*.png *.bmp *.tif *.tiff *.webp);;All files (*)")
        if fs:
            self._add_inputs(fs)

    def _add_inputs(self, paths):
        have = {norm(self.inputs.item(i).text()) for i in range(self.inputs.count())}
        for p in paths:
            if norm(p) not in have:
                self.inputs.addItem(os.path.abspath(p))
        if not self.output.text() and paths:
            first = Path(paths[0])
            base = first if first.is_dir() else first.parent
            self.output.setText(str(base / "yolo-finder-results"))
        self.rescan()

    def remove_inputs(self):
        for it in self.inputs.selectedItems():
            self.inputs.takeItem(self.inputs.row(it))
        self.rescan()

    def clear_inputs(self):
        self.inputs.clear()
        self.rescan()

    def pick_output(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.output.text())
        if d:
            self.output.setText(d)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self._add_inputs(paths)
            self.nav.setCurrentRow(0)

    def rescan(self, *_):
        inputs = [self.inputs.item(i).text() for i in range(self.inputs.count())]
        excluded = self._excluded()
        self.summary.setText("Scanning…")

        def scan():
            files = find_media(inputs, self.recursive.isChecked(), skip_dirs=[self.output.text()])
            return [probe(f) for f in files]

        run_async(scan, on_done=lambda m: self._fill_table(m, excluded),
                  on_error=lambda e: self.summary.setText("Scan failed: " + e.splitlines()[-1]))

    def _excluded(self):
        out = set(getattr(self, "_pending_excluded", set()))
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it and it.checkState() != Qt.CheckState.Checked:
                out.add(norm(it.data(Qt.ItemDataRole.UserRole)))
        return out

    def _fill_table(self, media, excluded):
        self._pending_excluded = set()
        self.media = media
        self.table.setRowCount(len(media))
        for r, m in enumerate(media):
            it = QTableWidgetItem(os.path.basename(m.path))
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Unchecked if norm(m.path) in excluded else Qt.CheckState.Checked)
            it.setData(Qt.ItemDataRole.UserRole, m.path)
            vals = [m.kind, f"{m.width}×{m.height}", f"{m.fps:.2f}" if m.kind == "video" else "",
                    human_duration(m.duration) if m.kind == "video" else "", str(m.frames) if m.kind == "video" else "",
                    human_size(m.size), os.path.dirname(m.path)]
            self.table.setItem(r, 0, it)
            for c, v in enumerate(vals, 1):
                self.table.setItem(r, c, QTableWidgetItem(v))
        nv = sum(m.kind == "video" for m in media)
        dur = sum(m.duration for m in media)
        self.summary.setText(f"{nv} videos ({human_duration(dur)} total), {len(media) - nv} photos, "
                             f"{human_size(sum(m.size for m in media))}")
        self._refresh_zone_files()

    def _tick_all(self, v):
        for r in range(self.table.rowCount()):
            self.table.item(r, 0).setCheckState(Qt.CheckState.Checked if v else Qt.CheckState.Unchecked)

    # ================= zones =================
    def _refresh_zone_files(self):
        cur = self.preview_path
        self.zone_files.blockSignals(True)
        self.zone_files.clear()
        sel = 0
        for i, m in enumerate(self.media):
            nz = len(self.zones.get(norm(m.path), []))
            ts = " 🕒" if self.ts_regions.get(norm(m.path)) else ""
            self.zone_files.addItem(f"{os.path.basename(m.path)}" + (f"  [{nz} zone{'s' * (nz != 1)}]" if nz else "") + ts)
            if cur and norm(cur) == norm(m.path):
                sel = i
        self.zone_files.blockSignals(False)
        if self.media:
            self.zone_files.setCurrentRow(sel)
            if self.preview_path is None or norm(self.preview_path) != norm(self.media[sel].path):
                self._zone_file_selected(sel)

    def _zone_file_selected(self, row):
        if row < 0 or row >= len(self.media):
            return
        m = self.media[row]
        self.preview_path = m.path
        self.scrub.blockSignals(True)
        self.scrub.setEnabled(m.kind == "video")
        self.scrub.setRange(0, max(0, m.frames - 1))
        self.scrub.setValue(min(self.scrub.value(), max(0, m.frames - 1)) if m.kind == "video" else 0)
        self.scrub.blockSignals(False)
        self._load_preview_frame(reset_zones=True)

    def _load_preview_frame(self, reset_zones=False):
        path = self.preview_path
        if not path:
            return
        idx = self.scrub.value()
        m = next((x for x in self.media if x.path == path), None)

        def done(img):
            if img is None or path != self.preview_path:
                return
            self.preview_frame = img
            self.canvas.set_image(img)
            self.canvas.set_detections()
            if reset_zones:
                self.canvas.set_zones(self.zones.get(norm(path), []))
                self.canvas.set_ts_region(self.ts_regions.get(norm(path)))
            if m and m.kind == "video":
                self.scrub_lbl.setText(f"frame {idx} / {m.frames}  ({human_duration(idx / (m.fps or 30))})")
            else:
                self.scrub_lbl.setText("photo")

        run_async(grab_frame, path, idx, on_done=done)

    def _zones_changed(self):
        if not self.preview_path:
            return
        k = norm(self.preview_path)
        z = self.canvas.get_zones()
        if z:
            self.zones[k] = z
        else:
            self.zones.pop(k, None)
        ts = self.canvas.get_ts_region()
        if ts:
            self.ts_regions[k] = ts
        else:
            self.ts_regions.pop(k, None)
        row = self.zone_files.currentRow()
        if 0 <= row < self.zone_files.count():
            nz = len(z)
            self.zone_files.item(row).setText(os.path.basename(self.preview_path)
                                              + (f"  [{nz} zone{'s' * (nz != 1)}]" if nz else "") + (" 🕒" if ts else ""))

    def copy_zones(self, mode):
        if not self.preview_path:
            return
        src = norm(self.preview_path)
        cur = next(m for m in self.media if norm(m.path) == src)
        if mode == "name":
            pat, ok = QInputDialog.getText(self, "Copy zones", "Copy to files whose name contains:")
            if not ok or not pat:
                return
            targets = [m for m in self.media if pat.lower() in os.path.basename(m.path).lower()]
        elif mode == "res":
            targets = [m for m in self.media if (m.width, m.height) == (cur.width, cur.height)]
        else:
            targets = list(self.media)
        z, ts = self.zones.get(src, []), self.ts_regions.get(src)
        for m in targets:
            k = norm(m.path)
            if z:
                self.zones[k] = [dict(d) for d in z]
            else:
                self.zones.pop(k, None)
            if ts:
                self.ts_regions[k] = list(ts)
        self._refresh_zone_files()
        self.set_status(f"Copied zones to {len(targets)} file(s)")

    def test_detection(self):
        if self.preview_frame is None:
            return
        s = self.collect_settings()
        img = self.preview_frame
        path = self.preview_path
        self.test_lbl.setText("Running detection… (first run loads/downloads the model)")

        def work():
            det = shared_detector(s.model, s.imgsz, s.device, s.half)
            r = det.detect([img])[0]
            D = np.concatenate([np.zeros((len(r), 1), np.float32), r], 1)
            keep = A.basic_mask(D, s.classes, s.conf, s.min_box_pct, img.shape[0])
            zkeep = A.zone_mask(D, s.zones_for(path), img.shape[1], img.shape[0], s.zone_anchor)
            sel = A.basic_mask(D, s.classes, s.conf, 0, img.shape[0])
            return D[keep & zkeep][:, 1:], D[sel & ~zkeep][:, 1:], det.names

        def done(res):
            kept, dropped, names = res
            self.canvas.set_detections(kept, dropped, names)
            parts = {}
            for r in kept:
                parts[names[int(r[5])]] = parts.get(names[int(r[5])], 0) + 1
            self.test_lbl.setText(f"{len(kept)} kept: " + ", ".join(f"{v}× {k}" for k, v in parts.items())
                                  + (f"   ·   {len(dropped)} removed by zones (grey)" if len(dropped) else ""))

        run_async(work, on_done=done, on_error=lambda e: self.test_lbl.setText("Error: " + e.strip().splitlines()[-1]))

    def test_ocr(self):
        if self.preview_frame is None:
            QMessageBox.information(self, "Timestamp", "Add a file and open it on the Zones page first.")
            return
        img, path = self.preview_frame, self.preview_path
        region = self.ts_regions.get(norm(path))
        order = self.date_order.currentData()
        self.test_lbl.setText("Reading timestamp…")

        def work():
            from ..core.timestamps import OcrReader
            if not hasattr(self, "_ocr"):
                self._ocr = OcrReader()
            return self._ocr.read(img, region, order)

        def done(res):
            t, text = res
            msg = f"Read: “{text.strip()}” → {t:%Y-%m-%d %H:%M:%S}" if t else f"No date/time found (text: “{text.strip()}”)"
            self.test_lbl.setText(msg)
            self.set_status(msg)

        run_async(work, on_done=done, on_error=lambda e: self.test_lbl.setText("OCR error: " + e.strip().splitlines()[-1]))

    # ================= model =================
    def pick_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Custom YOLO model", "", "YOLO model (*.pt *.onnx);;All files (*)")
        if p:
            self._set_model(p)

    def _set_model(self, m):
        i = self.model_cb.findData(m)
        if i < 0:
            self.model_cb.addItem(f"{os.path.basename(m)}  –  custom", m)
            i = self.model_cb.count() - 1
        self.model_cb.setCurrentIndex(i)

    def _model_changed(self, *_):
        m = self.model_cb.currentData()
        if m and os.path.isfile(m):
            def names():
                from ultralytics import YOLO
                return {int(k): v for k, v in YOLO(m).names.items()}
            run_async(names, on_done=self.picker.set_names,
                      on_error=lambda e: QMessageBox.warning(self, "Model", "Could not read model classes:\n" + e))
        elif m in KNOWN_MODELS:
            from ..core.classes import COCO_NAMES
            if list(self.picker.names.values()) != COCO_NAMES:
                self.picker.set_names({i: n for i, n in enumerate(COCO_NAMES)})

    def _apply_quality(self, model, imgsz, step):
        self._set_model(model)
        self.imgsz.setCurrentIndex(max(0, self.imgsz.findData(imgsz)))
        self.frame_step.setValue(step)
        self.set_status(f"Preset applied: {model}, {imgsz}px, every {step}. frame")

    # ================= settings <-> UI =================
    def collect_settings(self) -> JobSettings:
        s = JobSettings()
        s.inputs = [self.inputs.item(i).text() for i in range(self.inputs.count())]
        s.recursive = self.recursive.isChecked()
        s.excluded = sorted(self._excluded())
        s.output_dir = self.output.text().strip()
        s.model = self.model_cb.currentData()
        s.imgsz = self.imgsz.currentData()
        s.conf = round(self.conf.value(), 3)
        s.device = self.device.currentText()
        s.half = self.half.isChecked()
        s.batch = self.batch.value()
        s.frame_step = self.frame_step.value()
        s.classes = self.picker.selected()
        s.min_box_pct = self.min_box.value()
        s.ignore_stationary = self.stationary.isChecked()
        s.stationary_people = self.stat_people.isChecked()
        s.stationary_window_s = self.stat_win.value()
        s.stationary_ratio = self.stat_ratio.value()
        s.stationary_iou = self.stat_iou.value()
        s.zone_anchor = self.anchor.currentData()
        s.zones = {k: v for k, v in self.zones.items() if v}
        s.event_gap_s = self.gap.value()
        s.save_frames = self.save_frames.isChecked()
        s.save_clean = self.save_clean.isChecked()
        s.save_annotated = self.save_ann.isChecked()
        s.frames_per_event = self.per_event.value()
        s.jpeg_quality = self.jpeg_q.value()
        s.save_clips = self.save_clips.isChecked()
        s.clip_annotated = self.clip_ann.isChecked()
        s.clip_pre_s = self.clip_pre.value()
        s.clip_post_s = self.clip_post.value()
        s.box_thickness = self.box_th.value()
        s.box_labels = self.box_lbl.isChecked()
        s.box_conf = self.box_conf.isChecked()
        s.ocr_enabled = self.ocr.isChecked()
        s.ts_regions = dict(self.ts_regions)
        s.date_order = self.date_order.currentData()
        s.filename_time = self.fn_time.isChecked()
        s.timelapse_factor = self.tl_factor.value()
        return s

    def apply_settings(self, s: JobSettings):
        self.inputs.clear()
        for p in s.inputs:
            self.inputs.addItem(p)
        self.recursive.blockSignals(True)
        self.recursive.setChecked(s.recursive)
        self.recursive.blockSignals(False)
        self._pending_excluded = {norm(p) for p in s.excluded}
        self.output.setText(s.output_dir)
        self._set_model(s.model)
        self.imgsz.setCurrentIndex(max(0, self.imgsz.findData(s.imgsz)))
        self.conf.setValue(s.conf)
        self.device.setCurrentText(s.device)
        self.half.setChecked(s.half)
        self.batch.setValue(s.batch)
        self.frame_step.setValue(s.frame_step)
        self.picker.set_selected(s.classes)
        self.min_box.setValue(s.min_box_pct)
        self.stationary.setChecked(s.ignore_stationary)
        self.stat_people.setChecked(s.stationary_people)
        self.stat_win.setValue(s.stationary_window_s)
        self.stat_ratio.setValue(s.stationary_ratio)
        self.stat_iou.setValue(s.stationary_iou)
        self.anchor.setCurrentIndex(max(0, self.anchor.findData(s.zone_anchor)))
        self.zones = {k: list(v) for k, v in s.zones.items()}
        self.gap.setValue(s.event_gap_s)
        self.save_frames.setChecked(s.save_frames)
        self.save_clean.setChecked(s.save_clean)
        self.save_ann.setChecked(s.save_annotated)
        self.per_event.setValue(s.frames_per_event)
        self.jpeg_q.setValue(s.jpeg_quality)
        self.save_clips.setChecked(s.save_clips)
        self.clip_ann.setChecked(s.clip_annotated)
        self.clip_pre.setValue(s.clip_pre_s)
        self.clip_post.setValue(s.clip_post_s)
        self.box_th.setValue(s.box_thickness)
        self.box_lbl.setChecked(s.box_labels)
        self.box_conf.setChecked(s.box_conf)
        self.ocr.setChecked(s.ocr_enabled)
        self.ts_regions = dict(s.ts_regions)
        self.date_order.setCurrentIndex(max(0, self.date_order.findData(s.date_order)))
        self.fn_time.setChecked(s.filename_time)
        self.tl_factor.setValue(s.timelapse_factor)
        self.preview_path = None
        self.rescan()

    # ================= job files =================
    def new_job(self):
        self.job_path = None
        self.apply_settings(JobSettings())
        self.setWindowTitle(f"{APP_NAME} {__version__}")

    def open_job(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open job", "", "YOLO Finder job (*.yfjob *.json)")
        if p:
            try:
                self.apply_settings(JobSettings.load(p))
                self.job_path = p
                self.setWindowTitle(f"{APP_NAME} – {os.path.basename(p)}")
            except Exception as e:
                QMessageBox.warning(self, "Open job", str(e))

    def save_job(self, as_new=False):
        p = self.job_path
        if as_new or not p:
            p, _ = QFileDialog.getSaveFileName(self, "Save job", "job.yfjob", "YOLO Finder job (*.yfjob)")
            if not p:
                return
        self.collect_settings().save(p)
        self.job_path = p
        self.setWindowTitle(f"{APP_NAME} – {os.path.basename(p)}")
        self.set_status(f"Saved {p}")

    def _load_last_job(self):
        p = app_home() / "last_job.json"
        try:
            s = JobSettings.load(p) if p.exists() else JobSettings()
        except Exception:
            s = JobSettings()
        self.apply_settings(s)
        geo = QSettings("YoloFinder", "YoloFinder").value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)

    def closeEvent(self, e):
        if self.worker and self.worker.isRunning():
            if QMessageBox.question(self, "Quit", "Analysis is running. Stop it and quit?") != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.worker.cancel()
            self.worker.wait(15000)
        try:
            self.collect_settings().save(app_home() / "last_job.json")
            QSettings("YoloFinder", "YoloFinder").setValue("geometry", self.saveGeometry())
        except Exception:
            pass
        super().closeEvent(e)

    # ================= run =================
    def start(self):
        s = self.collect_settings()
        problems = []
        if not s.inputs:
            problems.append("Add at least one input folder or file (Sources).")
        if not s.output_dir:
            problems.append("Choose an output folder (Sources).")
        if not s.classes:
            problems.append("Select at least one thing to find (What to find).")
        if not (s.save_frames and (s.save_clean or s.save_annotated)) and not s.save_clips:
            problems.append("Nothing would be saved: enable still frames or clips (Output).")
        if problems:
            QMessageBox.warning(self, "Can't start", "\n".join(problems))
            return
        try:
            s.save(app_home() / "last_job.json")
            Path(s.output_dir).mkdir(parents=True, exist_ok=True)
            s.save(Path(s.output_dir) / "job.yfjob")
        except OSError as e:
            QMessageBox.warning(self, "Output folder", str(e))
            return
        self.log.clear()
        self.nav.setCurrentRow(6)
        self.worker = JobWorker(s)
        self.worker.log.connect(self.log.appendPlainText)
        self.worker.progress.connect(self._on_progress)
        self.worker.preview.connect(self._on_preview)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_thread_end)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._t_start = time.time()
        self.bar.setValue(0)
        self.worker.start()

    def stop(self):
        if self.worker:
            self.worker.cancel()
            self.stop_btn.setEnabled(False)
            self.set_status("Stopping after the current batch…")

    def _on_progress(self, overall, file_frac, msg):
        self.bar.setValue(int(overall * 1000))
        self.file_bar.setValue(int(file_frac * 1000))
        self.set_status(msg)
        self.run_info.setText(msg)
        el = time.time() - self._t_start
        if 0.01 < overall < 1:
            self.eta_lbl.setText(f"ETA {human_duration(el / overall - el)}")
        else:
            self.eta_lbl.setText(human_duration(el))

    def _on_preview(self, img, rows, names):
        out = draw_boxes(img, rows, names, 2)
        pm = bgr_to_pixmap(out)
        self.live.setPixmap(pm.scaled(self.live.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))

    def _on_done(self, results_path):
        folder = str(Path(results_path).parent)
        self.viewer.load(folder)
        self.nav.setCurrentRow(7)
        self.set_status(f"Done. Results in {folder}")

    def _on_failed(self, msg):
        QMessageBox.critical(self, "Analysis failed", msg)

    def _on_thread_end(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _about(self):
        QMessageBox.about(self, APP_NAME,
                          f"<b>{APP_NAME} {__version__}</b><br>Batch object detection for videos and photos.<br><br>"
                          f"Detection: Ultralytics YOLO11 (AGPL-3.0)<br>OCR: RapidOCR<br>GUI: Qt for Python<br><br>"
                          f"App data: {app_home()}")
