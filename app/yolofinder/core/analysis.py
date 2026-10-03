"""Turning raw detections into kept detections and events.

Detections for one media file are a float32 array D with rows
[frame, x1, y1, x2, y2, conf, cls], sorted by frame.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .classes import VEHICLES


def point_in_poly(x, y, poly) -> np.ndarray:
    """Vectorized ray casting. x, y arrays; poly list of (px, py)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    inside = np.zeros(x.shape, bool)
    n = len(poly)
    if n < 3:
        return inside
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi)
        inside ^= cond
        j = i
    return inside


def anchor_points(D, w, h, anchor="bottom"):
    cx = (D[:, 1] + D[:, 3]) / 2 / max(w, 1)
    cy = (D[:, 4] if anchor == "bottom" else (D[:, 2] + D[:, 4]) / 2) / max(h, 1)
    return cx, cy


def zone_mask(D, zones, w, h, anchor="bottom") -> np.ndarray:
    """True for rows that survive ignore/include zones."""
    keep = np.ones(len(D), bool)
    if not len(D) or not zones:
        return keep
    cx, cy = anchor_points(D, w, h, anchor)
    includes = [z for z in zones if z.kind == "include" and len(z.points) >= 3]
    for z in zones:
        if z.kind == "ignore" and len(z.points) >= 3:
            keep &= ~point_in_poly(cx, cy, z.points)
    if includes:
        inc = np.zeros(len(D), bool)
        for z in includes:
            inc |= point_in_poly(cx, cy, z.points)
        keep &= inc
    return keep


def basic_mask(D, classes, conf, min_box_pct, h) -> np.ndarray:
    if not len(D):
        return np.zeros(0, bool)
    m = np.isin(D[:, 6].astype(int), list(classes)) & (D[:, 5] >= conf)
    if min_box_pct > 0:
        m &= (D[:, 4] - D[:, 2]) >= min_box_pct / 100.0 * h
    return m


def _iou_matrix(a, b):
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def _group(cls):
    cls = np.asarray(cls).astype(int)
    return np.where(np.isin(cls, list(VEHICLES)), -1, cls)


def stationary_flags(D, cand, processed, win_frames, iou_thr=0.75, ratio=0.6, people_too=False) -> np.ndarray:
    """Flag candidate rows whose box stays (IoU>thr) in >= ratio of processed frames within +-win_frames."""
    out = np.zeros(len(D), bool)
    if not len(D) or not cand.any():
        return out
    frames = D[:, 0].astype(np.int64)
    proc = np.asarray(sorted(processed), np.int64)
    groups = _group(D[:, 6])
    rows = np.nonzero(cand & (people_too | (D[:, 6].astype(int) != 0)))[0]
    if not len(rows):
        return out
    for f in np.unique(frames[rows]):
        r = rows[frames[rows] == f]
        lo, hi = np.searchsorted(frames, [f - win_frames, f + win_frames + 1])
        w = np.arange(lo, hi)
        w = w[frames[w] != f]
        plo, phi = np.searchsorted(proc, [f - win_frames, f + win_frames + 1])
        denom = (phi - plo) - 1
        if denom <= 0 or not len(w):
            continue
        m = _iou_matrix(D[r, 1:5], D[w, 1:5]) > iou_thr
        m &= groups[r][:, None] == groups[w][None, :]
        wf = frames[w]
        for k, ri in enumerate(r):
            hits = np.unique(wf[m[k]]).size
            out[ri] = hits / denom >= ratio
    return out


@dataclass
class Event:
    file: str
    start: int
    end: int
    frames: list = field(default_factory=list)
    classes: Counter = field(default_factory=Counter)
    max_conf: float = 0.0
    key_frame: int = 0


def key_score(rows):
    area = (rows[:, 3] - rows[:, 1]) * (rows[:, 4] - rows[:, 2])
    weight = np.where(rows[:, 6].astype(int) == 0, 3.0, 1.0)
    return float((area * rows[:, 5] * weight).sum())


def build_events(file, K, gap_frames, names) -> list[Event]:
    """K: kept rows sorted by frame."""
    events: list[Event] = []
    if not len(K):
        return events
    frames = K[:, 0].astype(np.int64)
    uniq = np.unique(frames)
    cur = None
    best = -1.0
    for f in uniq:
        rows = K[frames == f]
        if cur is None or f - cur.end > gap_frames:
            cur = Event(file, int(f), int(f))
            events.append(cur)
            best = -1.0
        cur.end = int(f)
        cur.frames.append(int(f))
        for c in set(rows[:, 6].astype(int).tolist()):
            cur.classes[names.get(c, str(c))] += 1
        cur.max_conf = max(cur.max_conf, float(rows[:, 5].max()))
        s = key_score(rows)
        if s > best:
            best, cur.key_frame = s, int(f)
    return events


def pick_frames(ev: Event, limit: int) -> list[int]:
    if limit <= 0 or len(ev.frames) <= limit:
        return list(ev.frames)
    idx = np.unique(np.round(np.linspace(0, len(ev.frames) - 1, limit)).astype(int))
    chosen = {ev.frames[i] for i in idx}
    if ev.key_frame not in chosen:
        chosen.discard(min(chosen, key=lambda f: abs(f - ev.key_frame)))
        chosen.add(ev.key_frame)
    return sorted(chosen)
