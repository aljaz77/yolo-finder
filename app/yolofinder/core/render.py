"""Drawing boxes and writing clips."""
from __future__ import annotations

import cv2
import numpy as np

from .classes import color_for


def draw_boxes(img, rows, names, thickness=3, labels=True, conf=True, dim_rows=None):
    """rows: (n, >=7) [frame, x1, y1, x2, y2, conf, cls] or (n, 6) [x1, y1, x2, y2, conf, cls]."""
    out = img.copy()
    h = out.shape[0]
    scale = max(0.5, h / 1440 * 1.1)
    th = max(1, int(thickness))
    if dim_rows is not None:
        for r in _as6(dim_rows):
            x1, y1, x2, y2 = map(int, r[:4])
            cv2.rectangle(out, (x1, y1), (x2, y2), (128, 128, 128), 1)
    for r in _as6(rows):
        x1, y1, x2, y2 = map(int, r[:4])
        c = int(r[5])
        col = color_for(c)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, th)
        if labels:
            txt = names.get(c, str(c)) + (f" {r[4]:.2f}" if conf else "")
            (tw, tht), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, th - 1))
            ty = y1 - 6 if y1 - tht - 10 > 0 else y1 + tht + 8
            cv2.rectangle(out, (x1, ty - tht - 4), (x1 + tw + 6, ty + base), col, -1)
            cv2.putText(out, txt, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), max(1, th - 1), cv2.LINE_AA)
    return out


def _as6(rows):
    if rows is None or len(rows) == 0:
        return []
    rows = np.asarray(rows)
    return rows[:, 1:7] if rows.shape[1] >= 7 else rows


class ClipWriter:
    """H.264 mp4 writer via the ffmpeg binary shipped with imageio-ffmpeg."""

    def __init__(self, path, fps, size):
        import imageio_ffmpeg
        self.w, self.h = size
        w2, h2 = self.w - self.w % 2, self.h - self.h % 2
        self.size2 = (w2, h2)
        self.gen = imageio_ffmpeg.write_frames(
            str(path), (w2, h2), fps=float(fps), codec="libx264", pix_fmt_in="bgr24",
            macro_block_size=1, ffmpeg_log_level="error", quality=None,
            output_params=["-preset", "veryfast", "-crf", "23", "-movflags", "+faststart"])
        self.gen.send(None)

    def write(self, frame):
        if frame.shape[1] != self.size2[0] or frame.shape[0] != self.size2[1]:
            frame = frame[:self.size2[1], :self.size2[0]]
        self.gen.send(np.ascontiguousarray(frame))

    def close(self):
        try:
            self.gen.close()
        except Exception:
            pass
