"""Job settings (serializable) and application paths."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .classes import DEFAULT_PRESET, PRESETS


def app_home() -> Path:
    """Where models, logs and app settings live. The launcher sets YOLOFINDER_HOME."""
    import sys
    env = os.environ.get("YOLOFINDER_HOME")
    marker = Path(sys.prefix) / "yolofinder_home.txt"     # written by the installer
    if env:
        p = Path(env)
    elif marker.exists():
        p = Path(marker.read_text(encoding="utf-8").strip())
    else:
        p = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "YoloFinder"
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = app_home() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class Zone:
    kind: str = "ignore"          # "ignore" | "include"
    points: list = field(default_factory=list)   # [[x, y], ...] normalized 0..1

    @staticmethod
    def from_dict(d):
        return Zone(kind=d.get("kind", "ignore"), points=[list(p) for p in d.get("points", [])])


@dataclass
class JobSettings:
    # sources
    inputs: list = field(default_factory=list)       # files and/or folders
    recursive: bool = True
    excluded: list = field(default_factory=list)     # files unticked in the list
    output_dir: str = ""

    # detection
    model: str = "yolo11m.pt"
    imgsz: int = 1280
    conf: float = 0.35
    device: str = "auto"                              # auto | cpu | cuda:0 ...
    half: bool = True
    batch: int = 8
    frame_step: int = 1                               # process every Nth frame
    classes: list = field(default_factory=lambda: list(PRESETS[DEFAULT_PRESET]))
    min_box_pct: float = 0.0                          # ignore boxes smaller than this % of frame height

    # filters
    ignore_stationary: bool = True
    stationary_people: bool = False                   # also drop people that stand still
    stationary_window_s: float = 15.0                 # +- seconds of real time
    stationary_iou: float = 0.75
    stationary_ratio: float = 0.6
    zone_anchor: str = "bottom"                       # bottom | center
    zones: dict = field(default_factory=dict)         # file path -> [Zone dict]
    event_gap_s: float = 3.0                          # merge detections closer than this (real seconds)

    # output
    save_frames: bool = True
    save_clean: bool = True
    save_annotated: bool = True
    frames_per_event: int = 0                         # 0 = all frames
    save_clips: bool = False
    clip_annotated: bool = True
    clip_pre_s: float = 2.0
    clip_post_s: float = 2.0
    jpeg_quality: int = 92
    box_thickness: int = 3
    box_labels: bool = True
    box_conf: bool = True

    # timestamps
    ocr_enabled: bool = True
    ts_regions: dict = field(default_factory=dict)    # file path -> [x1, y1, x2, y2] normalized (optional)
    date_order: str = "auto"                          # auto | DMY | MDY | YMD
    filename_time: bool = True
    timelapse_factor: float = 0.0                     # real seconds per video second; 0 = auto/1

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "JobSettings":
        names = {f.name for f in fields(JobSettings)}
        s = JobSettings(**{k: v for k, v in d.items() if k in names})
        return s

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path) -> "JobSettings":
        return JobSettings.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def zones_for(self, path: str) -> list[Zone]:
        return [Zone.from_dict(z) for z in self.zones.get(norm(path), [])]

    def ts_region_for(self, path: str):
        return self.ts_regions.get(norm(path))


def norm(p) -> str:
    return os.path.normcase(os.path.abspath(str(p)))
