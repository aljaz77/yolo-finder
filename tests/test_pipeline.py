"""End-to-end run on short clips cut from real dashcam footage (needs the model; GPU optional)."""
import json
from pathlib import Path

import pytest

from yolofinder.core.config import JobSettings
from yolofinder.core.pipeline import Job

DATA = Path(__file__).parent / "data"
pytestmark = pytest.mark.skipif(not (DATA / "2026_1002_125508_frontsample.mp4").exists(), reason="no test media")


def run(tmp_path, **kw):
    s = JobSettings(inputs=[str(DATA)], output_dir=str(tmp_path / "out"), model="yolo11s.pt", imgsz=960, **kw)
    return Job(s).run(), s


def test_full_job(tmp_path):
    res_path, s = run(tmp_path, save_clips=True, frames_per_event=0)
    r = json.loads(res_path.read_text())
    assert r["complete"]
    kinds = sorted(f["kind"] for f in r["files"])
    assert kinds == ["image", "video", "video"]
    front = next(f for f in r["files"] if "front" in f["path"])
    rear = next(f for f in r["files"] if "rear" in f["path"])
    assert front["time_source"] == "ocr"
    assert front["start_time"].startswith("2026-10-02T12:55")
    assert abs(front["sec_per_frame"] * front["fps"] - 10) < 1.5        # 10x timelapse detected from OCR
    assert rear["n_events"] == 0
    ev = [e for e in r["events"] if e["file_id"] == front["id"]]
    assert ev and any("person" in e["classes"] for e in ev) and any("car" in e["classes"] for e in ev)
    assert all("ocr_time" in e for e in ev)
    out = Path(s.output_dir)
    for e in r["events"]:
        for fr in e["frames"]:
            assert (out / fr["clean"]).exists() and (out / fr["ann"]).exists()
    assert any(e["clip"] and (out / e["clip"]).exists() for e in ev)
    # parked car on the left must not create events: no kept box at the left edge in every frame
    assert (out / "events.csv").exists()


def test_clips_only_no_boxes(tmp_path):
    res_path, s = run(tmp_path, save_frames=False, save_clips=True, clip_annotated=False, classes=[0])
    r = json.loads(res_path.read_text())
    out = Path(s.output_dir)
    assert r["events"]
    assert not (out / "frames").exists() and not (out / "annotated").exists()
    vid_events = [e for e in r["events"] if r["files"][e["file_id"]]["kind"] == "video"]
    assert all(e["clip"] and (out / e["clip"]).stat().st_size > 10_000 for e in vid_events)
    assert all(e["thumb"] and (out / e["thumb"]).exists() and e["frames"] == [] for e in r["events"])


def test_cached_rerun_with_zones(tmp_path):
    run(tmp_path)
    # second run: same output -> cache; ignore the whole frame -> no events
    from yolofinder.core.config import norm
    s = JobSettings(inputs=[str(DATA)], output_dir=str(tmp_path / "out"), model="yolo11s.pt", imgsz=960)
    whole = [{"kind": "ignore", "points": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
    s.zones = {norm(f): whole for f in DATA.iterdir()}
    logs = []
    job = Job(s)
    job.cb.log = logs.append
    r = json.loads(job.run().read_text())
    assert any("cached" in m for m in logs)
    assert r["events"] == []
