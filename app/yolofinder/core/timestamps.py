"""Reading time: burned-in OCR timestamps, filename dates, and frame->time models."""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np

_CONFUSE = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "U": "0", "l": "1", "I": "1", "|": "1",
                          "i": "1", "!": "1", "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "G": "6",
                          "g": "9", "q": "9", "T": "7"})
_TOKEN = re.compile(r"[0-9OoDQUlIi|!SsBZzGgqT]*\d[0-9OoDQUlIi|!SsBZzGgqT:./\-]*")

_DATE_YMD = re.compile(r"(?<!\d)(\d{4})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{1,2})(?!\d)")
_DATE_XXY = re.compile(r"(?<!\d)(\d{1,2})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{4})")
_TIME = re.compile(r"(?<!\d)(\d{1,2})\s*[:.]\s*(\d{2})\s*[:.]\s*(\d{2})(?:\s*([AaPp])\.?[Mm])?(?!\d)")


def normalize(text: str) -> str:
    """Fix typical OCR letter/digit confusions inside numeric-looking tokens."""
    return _TOKEN.sub(lambda m: m.group(0).translate(_CONFUSE), text)


def parse_datetime(text: str, order: str = "auto"):
    """Return (datetime|None, normalized text)."""
    t = normalize(text)
    date = None
    m = _DATE_YMD.search(t)
    if m:
        y, a, b = map(int, m.groups())
        date = _mk_date(y, a, b)
        rest = t[:m.start()] + " " + t[m.end():]
    else:
        m = _DATE_XXY.search(t)
        rest = t
        if m:
            a, b, y = map(int, m.groups())
            o = order
            if o == "auto":
                o = "DMY" if a > 12 or b <= 12 else "MDY"
            date = _mk_date(y, b, a) if o in ("DMY", "auto") else _mk_date(y, a, b)
            if date is None:
                date = _mk_date(y, a, b) if o == "DMY" else _mk_date(y, b, a)
            rest = t[:m.start()] + " " + t[m.end():]
    tm = _TIME.search(rest)
    if not tm:
        return None, t
    hh, mm, ss = int(tm.group(1)), int(tm.group(2)), int(tm.group(3))
    ap = tm.group(4)
    if ap:
        if ap.lower() == "p" and hh < 12:
            hh += 12
        if ap.lower() == "a" and hh == 12:
            hh = 0
    if hh > 23 or mm > 59 or ss > 59:
        return None, t
    if date is None:
        return None, t
    return dt.datetime.combine(date, dt.time(hh, mm, ss)), t


def _mk_date(y, mth, d):
    try:
        if not (1990 <= y <= 2100):
            return None
        return dt.date(y, mth, d)
    except ValueError:
        return None


_FN_PATTERNS = [
    re.compile(r"(?<!\d)(\d{4})[-_]?(\d{2})(\d{2})[-_T ]?(\d{2})(\d{2})(\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(\d{4})[-_.](\d{2})[-_.](\d{2})[-_T ](\d{2})[-_.:](\d{2})[-_.:](\d{2})(?!\d)"),
]


def datetime_from_filename(path):
    name = Path(path).stem
    for p in _FN_PATTERNS:
        for m in p.finditer(name):
            try:
                y, mo, d, h, mi, s = map(int, m.groups())
                if 1990 <= y <= 2100:
                    return dt.datetime(y, mo, d, h, mi, s)
            except ValueError:
                continue
    return None


class TimeModel:
    """Maps frame index -> datetime. Built from OCR samples, or a start time + rate."""

    def __init__(self, t0: dt.datetime | None, sec_per_frame: float, source: str):
        self.t0, self.spf, self.source = t0, sec_per_frame, source

    def at(self, frame: int):
        if self.t0 is None:
            return None
        return self.t0 + dt.timedelta(seconds=frame * self.spf)

    @staticmethod
    def fit(samples, fps, fallback_factor=1.0):
        """samples: [(frame, datetime)]. Robust linear fit; returns TimeModel or None."""
        s = sorted((f, t) for f, t in samples if t is not None)
        if not s:
            return None
        if len(s) == 1:
            f, t = s[0]
            spf = fallback_factor / fps
            return TimeModel(t - dt.timedelta(seconds=f * spf), spf, "ocr")
        f = np.array([a for a, _ in s], float)
        t = np.array([(b - s[0][1]).total_seconds() for _, b in s], float)
        slopes = []
        for i in range(len(f)):
            for j in range(i + 1, len(f)):
                if f[j] - f[i] > 0:
                    slopes.append((t[j] - t[i]) / (f[j] - f[i]))
        if not slopes:
            return None
        spf = float(np.median(slopes))
        if spf <= 0:
            spf = fallback_factor / fps
        icpt = float(np.median(t - spf * f))
        resid = np.abs(t - (icpt + spf * f))
        good = resid <= max(2.0, 3 * spf)
        if good.sum() >= 2 and good.sum() < len(f):
            return TimeModel.fit([x for x, g in zip(s, good) if g], fps, fallback_factor)
        return TimeModel(s[0][1] + dt.timedelta(seconds=icpt), spf, "ocr")


class OcrReader:
    """Burned-in timestamp reader using RapidOCR (ONNX, CPU)."""

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.engine = RapidOCR()
        self.strip_cache: dict = {}

    def _ocr(self, img):
        h = img.shape[0]
        if h < 64:
            import cv2
            s = 64 / max(h, 1)
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        res, _ = self.engine(img)
        if not res:
            return ""
        res = sorted(res, key=lambda r: (round(min(p[1] for p in r[0]) / 20), min(p[0] for p in r[0])))
        return " ".join(r[1] for r in res)

    def read(self, frame, region=None, order="auto", key=None):
        """Return (datetime|None, raw text). region: normalized [x1, y1, x2, y2]."""
        h, w = frame.shape[:2]
        if region:
            x1, y1, x2, y2 = region
            crop = frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
            text = self._ocr(crop) if crop.size else ""
            return parse_datetime(text, order)[0], text
        strips = {"bottom": (0.82, 1.0), "top": (0.0, 0.18)}
        order_try = [self.strip_cache[key]] if key in self.strip_cache else []
        order_try += [k for k in strips if k not in order_try]
        last = ""
        for name in order_try:
            a, b = strips[name]
            text = self._ocr(frame[int(a * h):int(b * h)])
            t, _ = parse_datetime(text, order)
            if t is not None:
                if key is not None:
                    self.strip_cache[key] = name
                return t, text
            last = last or text
        return None, last
