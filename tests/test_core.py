import datetime as dt

import numpy as np

from yolofinder.core import analysis as A
from yolofinder.core.config import JobSettings, Zone
from yolofinder.core.timestamps import TimeModel, datetime_from_filename, parse_datetime


def test_parse_dashcam_overlay():
    t, _ = parse_datetime("000KM/H HDR 02-10-2026 12:41:29")
    assert t == dt.datetime(2026, 10, 2, 12, 41, 29)


def test_parse_variants():
    assert parse_datetime("2023/04/05 07:08:09")[0] == dt.datetime(2023, 4, 5, 7, 8, 9)
    assert parse_datetime("12/31/2024 11:59:58 PM")[0] == dt.datetime(2024, 12, 31, 23, 59, 58)
    assert parse_datetime("05.04.2023 O7:O8:O9")[0] == dt.datetime(2023, 4, 5, 7, 8, 9)   # OCR O->0
    assert parse_datetime("04/05/2023 07:08:09", order="MDY")[0] == dt.datetime(2023, 4, 5, 7, 8, 9)
    assert parse_datetime("no time here")[0] is None
    assert parse_datetime("HDR 02-10-202612:44:13")[0] == dt.datetime(2026, 10, 2, 12, 44, 13)  # glued by OCR


def test_filename_dates():
    assert datetime_from_filename("2026_1002_123843_009836PF.MP4") == dt.datetime(2026, 10, 2, 12, 38, 43)
    assert datetime_from_filename("IMG_20231004_142233.jpg") == dt.datetime(2023, 10, 4, 14, 22, 33)
    assert datetime_from_filename("cam1 2024-01-02 03-04-05.mp4") == dt.datetime(2024, 1, 2, 3, 4, 5)
    assert datetime_from_filename("holiday.mp4") is None


def test_time_model_fit_timelapse():
    t0 = dt.datetime(2026, 10, 2, 12, 0, 0)
    # 30 fps timelapse at 10x -> 1/3 s per frame; OCR shows whole seconds, one outlier
    samples = [(f, t0 + dt.timedelta(seconds=int(f / 3))) for f in (0, 300, 900, 1800, 3000)]
    samples.append((1500, t0 + dt.timedelta(hours=5)))
    tm = TimeModel.fit(samples, fps=30)
    assert abs(tm.spf - 1 / 3) < 0.01
    assert abs((tm.at(3600) - (t0 + dt.timedelta(seconds=1200))).total_seconds()) < 2


def test_point_in_poly_and_zones():
    sq = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]
    assert A.point_in_poly([0.25, 0.75], [0.25, 0.25], sq).tolist() == [True, False]
    # rows: frame, x1, y1, x2, y2, conf, cls (100x100 image)
    D = np.array([[0, 10, 10, 20, 40, .9, 0], [0, 60, 60, 70, 90, .9, 0]], np.float32)
    assert A.zone_mask(D, [Zone("ignore", sq)], 100, 100).tolist() == [False, True]
    assert A.zone_mask(D, [Zone("include", sq)], 100, 100).tolist() == [True, False]


def _synthetic():
    rows = []
    for f in range(0, 300):
        rows.append([f, 100, 100, 200, 180, 0.9, 2])           # parked car, every frame
        if 100 <= f < 130:
            x = 300 + (f - 100) * 20                              # moving car
            rows.append([f, x, 300, x + 100, 380, 0.9, 2])
        if f in (200, 201, 202, 260):
            rows.append([f, 500, 100, 540, 200, 0.8, 0])          # person, two events
    return np.array(rows, np.float32)


def test_stationary_filter_and_events():
    D = _synthetic()
    keep = A.basic_mask(D, [0, 2], 0.35, 0, 1000)
    stat = A.stationary_flags(D, keep, list(range(300)), win_frames=30)
    K = D[keep & ~stat]
    assert set(K[:, 6].astype(int)) == {0, 2}
    assert not ((K[:, 1] == 100) & (K[:, 6] == 2)).any(), "parked car must be dropped"
    ev = A.build_events("x", K, gap_frames=10, names={0: "person", 2: "car"})
    assert [(e.start, e.end) for e in ev] == [(100, 129), (200, 202), (260, 260)]
    assert ev[0].classes["car"] == 30


def test_pick_frames_keeps_key_frame():
    e = A.Event("x", 0, 99, frames=list(range(100)), key_frame=37)
    p = A.pick_frames(e, 5)
    assert len(p) == 5 and 37 in p
    assert A.pick_frames(e, 0) == list(range(100))


def test_settings_roundtrip(tmp_path):
    s = JobSettings(inputs=["a"], output_dir="b", classes=[0, 2])
    s.zones["x"] = [{"kind": "ignore", "points": [[0, 0], [1, 0], [1, 1]]}]
    p = tmp_path / "job.json"
    s.save(p)
    s2 = JobSettings.load(p)
    assert s2.to_dict() == s.to_dict()
