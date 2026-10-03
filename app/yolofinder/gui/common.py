"""Small Qt helpers shared by the GUI."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import traceback

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtGui import QImage, QPixmap


def bgr_to_qimage(img: np.ndarray) -> QImage:
    if img is None:
        return QImage()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def bgr_to_pixmap(img) -> QPixmap:
    return QPixmap.fromImage(bgr_to_qimage(img))


class _Signals(QObject):
    done = Signal(object)
    error = Signal(str)


class _Task(QRunnable):
    def __init__(self, fn, args, kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = _Signals()

    def run(self):
        try:
            res = self.fn(*self.args, **self.kwargs)
        except Exception:
            self.signals.error.emit(traceback.format_exc())
            return
        self.signals.done.emit(res)


_KEEP = set()


def run_async(fn, *args, on_done=None, on_error=None, **kwargs):
    """Run fn in the global thread pool; callbacks run on the GUI thread."""
    t = _Task(fn, args, kwargs)
    _KEEP.add(t.signals)
    if on_done:
        t.signals.done.connect(on_done)
    if on_error:
        t.signals.error.connect(on_error)
    t.signals.done.connect(lambda *_: _KEEP.discard(t.signals))
    t.signals.error.connect(lambda *_: _KEEP.discard(t.signals))
    QThreadPool.globalInstance().start(t)
    return t


_det_lock = threading.Lock()
_det_cache = {}


def shared_detector(model, imgsz, device, half, log=print):
    """One detector instance reused for previews/tests."""
    from ..core.detector import Detector
    key = (model, int(imgsz), device, bool(half))
    with _det_lock:
        if key not in _det_cache:
            _det_cache.clear()
            _det_cache[key] = Detector(model, imgsz, device, half, log=log)
        return _det_cache[key]


def human_size(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.0f} {u}" if u in ("B", "KB") else f"{n:.1f} {u}"
        n /= 1024
    return ""


def human_duration(s: float) -> str:
    s = int(round(s))
    h, m, sec = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def open_path(path):
    if sys.platform == "win32":
        os.startfile(str(path))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def reveal(path):
    if sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", os.path.normpath(str(path))])
    else:
        open_path(os.path.dirname(str(path)))


def find_vlc():
    for p in (r"C:\Program Files\VideoLAN\VLC\vlc.exe", r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe"):
        if os.path.exists(p):
            return p
    return None


def play_video_at(path, seconds):
    vlc = find_vlc()
    if vlc:
        subprocess.Popen([vlc, f"--start-time={max(0.0, seconds):.2f}", str(path)])
        return True
    open_path(path)
    return False
