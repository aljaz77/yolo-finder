"""Thin wrapper around an Ultralytics YOLO model."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import models_dir

KNOWN_MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
CONF_FLOOR = 0.10   # detections are cached down to this confidence so filters can change later


def resolve_model(name: str, log=print) -> str:
    p = Path(name)
    if p.is_file():
        return str(p)
    target = models_dir() / p.name
    if target.exists():
        return str(target)
    log(f"Downloading model {p.name} ...")
    from ultralytics.utils.downloads import attempt_download_asset
    out = Path(attempt_download_asset(str(target)))
    if not out.exists():
        raise FileNotFoundError(f"Could not download model {name}")
    return str(out)


def pick_device(device: str) -> str:
    if device and device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


class Detector:
    def __init__(self, model="yolo11m.pt", imgsz=1280, device="auto", half=True, log=print):
        from ultralytics import YOLO
        self.model_path = resolve_model(model, log)
        self.model = YOLO(self.model_path)
        self.device = pick_device(device)
        self.half = bool(half) and self.device != "cpu"
        self.imgsz = int(imgsz)
        self.names = {int(k): v for k, v in self.model.names.items()}
        from ultralytics.cfg import DEFAULT_CFG_DICT
        if "quantize" in DEFAULT_CFG_DICT:
            self._precision = {"quantize": 16} if self.half else {}
        else:
            self._precision = {"half": self.half}
        log(f"Model {Path(self.model_path).name} on {self.device}{' (fp16)' if self.half else ''}, imgsz {self.imgsz}")

    def detect(self, frames) -> list[np.ndarray]:
        """frames: list of BGR images -> list of (n, 6) arrays [x1, y1, x2, y2, conf, cls]."""
        if not frames:
            return []
        res = self.model.predict(frames, imgsz=self.imgsz, conf=CONF_FLOOR, device=self.device,
                                 verbose=False, **self._precision)
        out = []
        for r in res:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(np.zeros((0, 6), np.float32))
                continue
            arr = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                                  b.cls.cpu().numpy()[:, None]], axis=1).astype(np.float32)
            out.append(arr)
        return out
