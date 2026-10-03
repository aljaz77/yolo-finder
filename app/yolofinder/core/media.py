"""Finding media files and reading frames."""
from __future__ import annotations

import os
import queue
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg", ".ts", ".mts",
             ".m2ts", ".3gp", ".webm", ".flv", ".h264", ".dav"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def kind_of(path) -> str | None:
    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXT:
        return "video"
    if ext in IMAGE_EXT:
        return "image"
    return None


def find_media(inputs, recursive=True, skip_dirs=()) -> list[str]:
    skip = {os.path.normcase(os.path.abspath(d)) for d in skip_dirs if d}
    out = []
    for inp in inputs:
        p = Path(inp)
        if p.is_file() and kind_of(p):
            out.append(str(p))
        elif p.is_dir():
            walker = os.walk(p) if recursive else [(str(p), [], os.listdir(p))]
            for root, dirs, files in walker:
                if os.path.normcase(os.path.abspath(root)) in skip:
                    dirs[:] = []
                    continue
                dirs[:] = [d for d in dirs if os.path.normcase(os.path.abspath(os.path.join(root, d))) not in skip]
                for f in files:
                    if kind_of(f):
                        out.append(os.path.join(root, f))
    seen, uniq = set(), []
    for f in out:
        k = os.path.normcase(os.path.abspath(f))
        if k not in seen:
            seen.add(k)
            uniq.append(os.path.abspath(f))
    return sorted(uniq, key=natural_key)


@dataclass
class MediaInfo:
    path: str
    kind: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frames: int = 1
    size: int = 0

    @property
    def duration(self) -> float:
        return self.frames / self.fps if self.kind == "video" and self.fps else 0.0


def imread(path):
    """cv2.imread that handles unicode paths on Windows."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path, img, quality=92):
    ext = Path(path).suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        buf.tofile(str(path))
    return ok


def probe(path) -> MediaInfo:
    k = kind_of(path)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    if k == "image":
        img = imread(path)
        h, w = img.shape[:2] if img is not None else (0, 0)
        return MediaInfo(path, k, w, h, 0.0, 1, size)
    cap = cv2.VideoCapture(path)
    info = MediaInfo(path, "video", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), size)
    cap.release()
    return info


def grab_frame(path, index=0):
    """Return a single BGR frame (or the image itself)."""
    if kind_of(path) == "image":
        return imread(path)
    cap = cv2.VideoCapture(path)
    if index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


class FrameReader:
    """Background-thread video decoder yielding (index, frame) for every `step`-th frame."""

    def __init__(self, path, step=1, maxsize=48):
        self.path, self.step = path, max(1, int(step))
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        cap = cv2.VideoCapture(self.path)
        i = 0
        try:
            while not self._stop.is_set():
                if i % self.step == 0:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    self._put((i, fr))
                else:
                    if not cap.grab():
                        break
                i += 1
        finally:
            cap.release()
            self._put(None)

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self.q.put(item, timeout=0.2)
                return
            except queue.Full:
                pass

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            yield item

    def stop(self):
        self._stop.set()
