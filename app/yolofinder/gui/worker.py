"""Runs a Job in a QThread and forwards callbacks as Qt signals."""
from __future__ import annotations

import time
import traceback

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from ..core.config import JobSettings
from ..core.pipeline import Callbacks, Job


class JobWorker(QThread):
    log = Signal(str)
    progress = Signal(float, float, str)
    preview = Signal(object, object, object)
    failed = Signal(str)
    finished_ok = Signal(str)

    def __init__(self, settings: JobSettings):
        super().__init__()
        self.settings = settings
        self._cancel = False
        self._last_prog = 0.0

    def cancel(self):
        self._cancel = True

    def run(self):
        w = self

        class CB(Callbacks):
            def log(self, msg):
                w.log.emit(msg)

            def progress(self, overall, file_frac, msg):
                now = time.time()
                if now - w._last_prog > 0.15 or overall >= 1.0:
                    w._last_prog = now
                    w.progress.emit(overall, file_frac, msg)

            def preview(self, img, rows, names):
                h, wd = img.shape[:2]
                s = min(1.0, 1280 / max(wd, 1))
                small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
                r = np.array(rows, np.float32).reshape(-1, 6).copy()
                r[:, :4] *= s
                w.preview.emit(small, r, dict(names))

            def cancelled(self):
                return w._cancel

        try:
            res = Job(self.settings, CB()).run()
            self.finished_ok.emit(str(res))
        except Exception as e:
            self.log.emit(traceback.format_exc())
            self.failed.emit(str(e))
