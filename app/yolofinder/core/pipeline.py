"""The batch job: detect -> filter -> group into events -> timestamps -> export frames/clips."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from .. import __version__, APP_NAME
from . import analysis as A
from .config import JobSettings, norm
from .media import FrameReader, find_media, imread, imwrite, probe
from .render import ClipWriter, draw_boxes
from .timestamps import OcrReader, TimeModel, datetime_from_filename

RESULTS = "results.json"


class Cancelled(Exception):
    pass


class Callbacks:
    """Override any of these. All are called from the worker thread."""

    def log(self, msg: str): print(msg)

    def progress(self, overall: float, file_frac: float, msg: str): pass

    def preview(self, img, rows, names): pass

    def cancelled(self) -> bool: return False


def _cache_key(path, info, s: JobSettings):
    st = os.stat(path)
    raw = f"{norm(path)}|{st.st_size}|{int(st.st_mtime)}|{Path(s.model).name}|{s.imgsz}|{s.frame_step}"
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def _exif_time(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            ex = im.getexif()
            v = ex.get_ifd(0x8769).get(36867) or ex.get(306)
            if v:
                return dt.datetime.strptime(str(v).strip()[:19], "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:60]


def _merge_ranges(ranges):
    out = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


class Job:
    def __init__(self, settings: JobSettings, cb: Callbacks | None = None):
        self.s = settings
        self.cb = cb or Callbacks()
        self.detector = None
        self.ocr = None
        self.names = {}
        self.out = Path(settings.output_dir)
        self.results = None

    # ---------- helpers ----------
    def _check(self):
        if self.cb.cancelled():
            raise Cancelled()

    def _get_detector(self):
        if self.detector is None:
            from .detector import Detector
            self.detector = Detector(self.s.model, self.s.imgsz, self.s.device, self.s.half, log=self.cb.log)
            self.names = self.detector.names
        return self.detector

    def _get_ocr(self):
        if self.ocr is None:
            try:
                self.ocr = OcrReader()
            except Exception as e:   # OCR is optional
                self.cb.log(f"OCR unavailable: {e}")
                self.ocr = False
        return self.ocr or None

    # ---------- main ----------
    def run(self) -> Path:
        s = self.s
        if not s.output_dir:
            raise ValueError("No output folder selected")
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / ".cache").mkdir(exist_ok=True)
        excluded = {norm(p) for p in s.excluded}
        files = [f for f in find_media(s.inputs, s.recursive, skip_dirs=[s.output_dir]) if norm(f) not in excluded]
        if not files:
            raise ValueError("No videos or photos found in the selected inputs")
        self.cb.log(f"{APP_NAME} {__version__}: {len(files)} file(s) to process")
        infos = [probe(f) for f in files]
        units = [max(1, i.frames // max(1, s.frame_step)) if i.kind == "video" else 1 for i in infos]
        total = float(sum(units))
        done_units = 0.0

        self.results = {
            "app": APP_NAME, "version": __version__, "created": dt.datetime.now().isoformat(timespec="seconds"),
            "settings": s.to_dict(), "class_names": {}, "files": [], "events": [], "complete": False,
        }
        t_start = time.time()
        status = "complete"
        try:
            for fi, (path, info) in enumerate(zip(files, infos)):
                self._check()
                name = os.path.basename(path)

                def prog(frac, msg, _du=done_units, _u=units[fi]):
                    self.cb.progress((_du + frac * _u) / total, frac, f"[{fi + 1}/{len(files)}] {name}: {msg}")

                self._process_file(fi, path, info, prog)
                done_units += units[fi]
                self._write_results()
        except Cancelled:
            status = "cancelled"
            self.cb.log("Stopped by user. Partial results were saved.")
        self.results["complete"] = status == "complete"
        self.results["elapsed_s"] = round(time.time() - t_start, 1)
        self._write_results()
        n_ev = len(self.results["events"])
        self.cb.log(f"Finished ({status}): {n_ev} event(s) in {len(self.results['files'])} file(s), "
                    f"{self.results['elapsed_s']:.0f}s. Output: {self.out}")
        self.cb.progress(1.0, 1.0, f"Done: {n_ev} events")
        return self.out / RESULTS

    # ---------- per file ----------
    def _process_file(self, fi, path, info, prog):
        s = self.s
        self.cb.log(f"--- {path} ({info.kind}, {info.width}x{info.height}"
                    + (f", {info.frames} frames @ {info.fps:.2f} fps" if info.kind == "video" else "") + ")")
        t0 = time.time()
        D, processed = self._detect(path, info, prog)
        det_s = time.time() - t0

        # time model first: windows and gaps are given in real seconds (matters for time-lapse footage)
        tm = self._time_model(path, info)
        fps = info.fps or 1.0
        spf = tm.spf if (tm and tm.spf > 0) else (s.timelapse_factor if s.timelapse_factor > 0 else 1.0) / fps

        # filtering
        names = self.names or {i: str(i) for i in range(80)}
        keep = A.basic_mask(D, s.classes, s.conf, s.min_box_pct, info.height)
        keep &= A.zone_mask(D, s.zones_for(path), info.width, info.height, s.zone_anchor)
        n_stat = 0
        if info.kind == "video" and s.ignore_stationary and keep.any():
            prog(0.97, "filtering stationary objects")
            win = max(s.frame_step, int(round(s.stationary_window_s / spf)))
            # only kept rows are judged; all cached rows (down to the confidence floor) count as evidence
            stat = A.stationary_flags(D, keep, processed, win, s.stationary_iou, s.stationary_ratio, s.stationary_people)
            n_stat = int((keep & stat).sum())
            keep &= ~stat
        K = D[keep]
        gap = max(s.frame_step, int(round(s.event_gap_s / spf)))
        events = A.build_events(path, K, gap, names)
        self.cb.log(f"{len(D)} raw detections, {len(K)} kept"
                    + (f", {n_stat} stationary dropped" if n_stat else "") + f", {len(events)} event(s)")
        file_rec = {"id": fi, "path": path, "kind": info.kind, "width": info.width, "height": info.height,
                    "fps": info.fps, "frames": info.frames, "detect_seconds": round(det_s, 1),
                    "time_source": tm.source if tm else "none",
                    "start_time": tm.at(0).isoformat(timespec="seconds") if tm and tm.at(0) else None,
                    "sec_per_frame": tm.spf if tm else None, "n_events": len(events)}
        self.results["files"].append(file_rec)
        self.results["class_names"] = {str(k): v for k, v in names.items()}

        # export
        if events:
            self._export(fi, path, info, K, events, tm, prog)

    def _detect(self, path, info, prog):
        s = self.s
        cache = self.out / ".cache" / (_cache_key(path, info, s) + ".npz")
        if cache.exists():
            try:
                z = np.load(cache, allow_pickle=False)
                self.names = {int(k): v for k, v in json.loads(str(z["names"])).items()}
                self.cb.log("Using cached detections (delete the .cache folder to force re-detection)")
                prog(0.95, "cached detections")
                return z["D"], z["processed"].tolist()
            except Exception:
                pass
        det = self._get_detector()
        rows, processed = [], []
        if info.kind == "image":
            img = imread(path)
            if img is None:
                self.cb.log("  could not read image, skipped")
                return np.zeros((0, 7), np.float32), []
            r = det.detect([img])[0]
            rows.append(np.concatenate([np.zeros((len(r), 1), np.float32), r], 1))
            processed = [0]
            self._preview(img, r)
        else:
            reader = FrameReader(path, s.frame_step)
            batch, idxs = [], []
            last_prev = 0.0
            n_exp = max(1, info.frames)
            t0 = time.time()
            try:
                for i, fr in reader:
                    batch.append(fr)
                    idxs.append(i)
                    if len(batch) >= s.batch:
                        self._run_batch(det, batch, idxs, rows, processed)
                        now = time.time()
                        if now - last_prev > 0.4:
                            last_prev = now
                            last = rows[-1] if rows else np.zeros((0, 7))
                            self._preview(batch[-1], last[last[:, 0] == idxs[-1]][:, 1:] if len(last) else last)
                            fps = len(processed) / max(1e-6, now - t0)
                            frac = min(0.95, idxs[-1] / n_exp)
                            prog(frac, f"detecting {idxs[-1]}/{info.frames} ({fps:.0f} fps)")
                        batch, idxs = [], []
                        self._check()
                if batch:
                    self._run_batch(det, batch, idxs, rows, processed)
            finally:
                reader.stop()
        D = np.concatenate(rows, 0) if rows else np.zeros((0, 7), np.float32)
        if len(D):
            D = D[np.argsort(D[:, 0], kind="stable")]
        np.savez_compressed(cache, D=D, processed=np.array(processed, np.int64),
                            names=json.dumps({str(k): v for k, v in self.names.items()}))
        return D, processed

    def _run_batch(self, det, batch, idxs, rows, processed):
        for i, r in zip(idxs, det.detect(batch)):
            processed.append(i)
            if len(r):
                rows.append(np.concatenate([np.full((len(r), 1), i, np.float32), r], 1))

    def _preview(self, img, r):
        s = self.s
        if len(r):
            r = r[np.isin(r[:, 5].astype(int), s.classes) & (r[:, 4] >= s.conf)]
        self.cb.preview(img, r, self.names)

    # ---------- time ----------
    def _time_model(self, path, info):
        s = self.s
        fallback_factor = s.timelapse_factor if s.timelapse_factor > 0 else 1.0
        fps = info.fps or 1.0
        if info.kind == "image":
            t = _exif_time(path)
            src = "exif"
            if t is None and s.ocr_enabled and self._get_ocr():
                img = imread(path)
                if img is not None:
                    t = self.ocr.read(img, s.ts_region_for(path), s.date_order, key=norm(path))[0]
                    src = "ocr"
            if t is None and s.filename_time:
                t, src = datetime_from_filename(path), "filename"
            return TimeModel(t, 0.0, src) if t else None
        if s.ocr_enabled and self._get_ocr():
            n = 6 if info.frames > 60 else 2
            cand = sorted(set(np.linspace(0, max(0, info.frames - 2), n).astype(int).tolist()))
            cap = cv2.VideoCapture(path)
            samples = []
            for f in cand:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, fr = cap.read()
                if ok:
                    t, _ = self.ocr.read(fr, s.ts_region_for(path), s.date_order, key=norm(path))
                    if t:
                        samples.append((f, t))
            cap.release()
            tm = TimeModel.fit(samples, fps, fallback_factor)
            if tm:
                if s.timelapse_factor > 0:
                    tm.spf = s.timelapse_factor / fps
                self.cb.log(f"Burned-in timestamp read in {len(samples)}/{len(cand)} samples: "
                            f"start {tm.at(0):%Y-%m-%d %H:%M:%S}, {tm.spf * fps:.2f} real s per video s")
                return tm
            self.cb.log("No burned-in timestamp found")
        if s.filename_time:
            t = datetime_from_filename(path)
            if t:
                return TimeModel(t, fallback_factor / fps, "filename")
        return None

    # ---------- export ----------
    def _export(self, fi, path, info, K, events, tm, prog):
        s = self.s
        names = self.names
        out = self.out
        stem = _safe(Path(path).stem)
        frames_dir, ann_dir, clip_dir = out / "frames", out / "annotated", out / "clips"
        kf = K[:, 0].astype(np.int64)
        fps = info.fps or 1.0

        save = {}   # frame -> event index
        if s.save_frames:
            for ei, ev in enumerate(events):
                for f in A.pick_frames(ev, s.frames_per_event):
                    save[f] = ei
        key_frames = {ev.key_frame: ei for ei, ev in enumerate(events)}
        clips = []
        if s.save_clips and info.kind == "video":
            pre, post = int(round(s.clip_pre_s * fps)), int(round(s.clip_post_s * fps))
            clips = _merge_ranges([[max(0, e.start - pre), min(info.frames - 1, e.end + post)] for e in events])
        if s.save_frames:
            if s.save_clean:
                frames_dir.mkdir(exist_ok=True)
            if s.save_annotated:
                ann_dir.mkdir(exist_ok=True)
        if clips:
            clip_dir.mkdir(exist_ok=True)

        ev_recs = []
        for ei, ev in enumerate(events):
            ev_recs.append({
                "id": len(self.results["events"]) + ei, "file_id": fi, "start_frame": ev.start, "end_frame": ev.end,
                "start_time": _iso(tm.at(ev.start)) if tm else None, "end_time": _iso(tm.at(ev.end)) if tm else None,
                "video_start_s": round(ev.start / fps, 2) if info.kind == "video" else 0,
                "video_end_s": round(ev.end / fps, 2) if info.kind == "video" else 0,
                "classes": dict(ev.classes), "max_conf": round(ev.max_conf, 3), "key_frame": ev.key_frame,
                "n_frames": len(ev.frames), "frames": [], "clip": None, "ocr_text": "", "flag": False, "note": ""})

        ocr = self._get_ocr() if s.ocr_enabled else None

        thumb_dir = out / "thumbs"
        thumb_dir.mkdir(exist_ok=True)

        def handle(f, img):
            rows = K[kf == f]
            if f in key_frames:
                rec = ev_recs[key_frames[f]]
                if ocr:
                    t, txt = ocr.read(img, s.ts_region_for(path), s.date_order, key=norm(path))
                    rec["ocr_text"] = txt.strip()
                    if t:
                        rec["ocr_time"] = _iso(t)
                # small annotated key-frame image for the viewer (also used in clip-only mode)
                th = draw_boxes(img, rows, names, max(2, s.box_thickness), s.box_labels, False)
                sc = 480 / max(1, th.shape[1])
                th = cv2.resize(th, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
                tname = f"{stem}_f{f:06d}_ev{rec['id']}.jpg"
                imwrite(thumb_dir / tname, th, 85)
                rec["thumb"] = f"thumbs/{tname}"
            if f in save:
                t = tm.at(f) if tm else None
                tstr = t.strftime("%Y-%m-%d_%H-%M-%S_") if t else ""
                labels = "-".join(sorted({names.get(int(c), str(int(c))) for c in rows[:, 6]}))[:40]
                base = f"{tstr}{stem}_f{f:06d}_{_safe(labels)}.jpg"
                rec = {"f": int(f), "t": _iso(t), "clean": None, "ann": None,
                       "boxes": np.round(rows[:, 1:7], 2).tolist()}
                if s.save_clean:
                    imwrite(frames_dir / base, img, s.jpeg_quality)
                    rec["clean"] = f"frames/{base}"
                if s.save_annotated:
                    imwrite(ann_dir / base, draw_boxes(img, rows, names, s.box_thickness, s.box_labels, s.box_conf),
                            s.jpeg_quality)
                    rec["ann"] = f"annotated/{base}"
                ev_recs[save[f]]["frames"].append(rec)

        if info.kind == "image":
            img = imread(path)
            handle(0, img)
        else:
            need_frames = sorted(set(save) | set(key_frames))
            intervals = _merge_ranges([[f, f] for f in need_frames] + [list(c) for c in clips])
            cap = cv2.VideoCapture(path)
            pos = 0
            clip_i, writer = 0, None
            last_rows = np.zeros((0, 7), np.float32)
            n_need = sum(b - a + 1 for a, b in intervals)
            n_done = 0
            try:
                for a, b in intervals:
                    if a - pos > 300:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
                        pos = a
                    while pos < a:
                        if not cap.grab():
                            break
                        pos += 1
                    while pos <= b:
                        ok, img = cap.read()
                        if not ok:
                            break
                        f = pos
                        pos += 1
                        n_done += 1
                        if f in save or f in key_frames:
                            handle(f, img)
                        # clips
                        while clip_i < len(clips) and f > clips[clip_i][1]:
                            clip_i += 1
                        if clip_i < len(clips) and clips[clip_i][0] <= f <= clips[clip_i][1]:
                            c0, c1 = clips[clip_i]
                            if writer is None:
                                t = tm.at(c0) if tm else None
                                tstr = t.strftime("%Y-%m-%d_%H-%M-%S_") if t else ""
                                cname = f"{tstr}{stem}_f{c0:06d}-{c1:06d}.mp4"
                                writer = ClipWriter(clip_dir / cname, fps, (info.width, info.height))
                                for r in ev_recs:
                                    if r["start_frame"] <= c1 and r["end_frame"] >= c0:
                                        r["clip"] = f"clips/{cname}"
                            if s.clip_annotated:
                                cur = K[kf == f]
                                if len(cur) or s.frame_step == 1:
                                    last_rows = cur
                                img = draw_boxes(img, last_rows, names, s.box_thickness, s.box_labels, s.box_conf)
                            writer.write(img)
                            if f == c1:
                                writer.close()
                                writer = None
                                last_rows = np.zeros((0, 7), np.float32)
                        if n_done % 25 == 0:
                            self._check()
                            prog(0.97 + 0.03 * n_done / max(1, n_need), f"exporting {n_done}/{n_need} frames")
            finally:
                if writer:
                    writer.close()
                cap.release()
        self.results["events"].extend(ev_recs)

    def _write_results(self):
        r = self.results
        tmp = self.out / (RESULTS + ".tmp")
        tmp.write_text(json.dumps(r, indent=1), encoding="utf-8")
        os.replace(tmp, self.out / RESULTS)
        write_csv(r, self.out / "events.csv")


def write_csv(r, path):
    files = {f["id"]: f for f in r["files"]}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["event", "file", "start_time", "end_time", "video_start_s", "video_end_s", "start_frame",
                    "end_frame", "frames", "objects", "max_conf", "ocr_text", "clip", "flag", "note"])
        for e in r["events"]:
            w.writerow([e["id"], files[e["file_id"]]["path"], e["start_time"], e["end_time"], e["video_start_s"],
                        e["video_end_s"], e["start_frame"], e["end_frame"], e["n_frames"],
                        " ".join(f"{k}:{v}" for k, v in e["classes"].items()), e["max_conf"], e.get("ocr_text", ""),
                        e.get("clip") or "", e.get("flag", False), e.get("note", "")])


def _iso(t):
    return t.isoformat(timespec="seconds") if t else None
