"""Drives the real main window end-to-end (offscreen)."""
import json
import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

DATA = Path(__file__).parent / "data"
pytestmark = pytest.mark.skipif(not (DATA / "2026_1002_125508_frontsample.mp4").exists(), reason="no test media")


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    from yolofinder.gui.theme import apply_dark
    apply_dark(a)
    return a


def wait(app, cond, timeout=300):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.02)
        assert time.time() - t0 < timeout, "timeout"


def test_main_window_flow(app, tmp_path, monkeypatch):
    from yolofinder.core.config import JobSettings, app_home
    (app_home() / "last_job.json").unlink(missing_ok=True)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail(f"warning: {a[1:]}"))
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: pytest.fail(f"critical: {a[1:]}"))
    from yolofinder.gui.main_window import MainWindow
    w = MainWindow()
    w.show()

    w._add_inputs([str(DATA)])
    assert w.output.text().endswith("yolo-finder-results")
    w.output.setText(str(tmp_path / "out"))
    wait(app, lambda: w.table.rowCount() == 3)

    # untick the photo
    for r in range(3):
        if w.table.item(r, 0).text() == "photo_person.jpg":
            w.table.item(r, 0).setCheckState(Qt.CheckState.Unchecked)

    # zones page: draw an ignore zone over the left third of the front video
    w.nav.setCurrentRow(1)
    wait(app, lambda: w.zone_files.count() == 3 and w.preview_frame is not None)
    front = next(i for i, m in enumerate(w.media) if "front" in m.path)
    w.zone_files.setCurrentRow(front)
    wait(app, lambda: w.preview_path and "front" in w.preview_path and w.canvas.w == 2560)
    from PySide6.QtGui import QPolygonF
    w.canvas._add_zone("ignore", QPolygonF([QPointF(0, 0), QPointF(800, 0), QPointF(800, 1440), QPointF(0, 1440)]))
    w.canvas.changed.emit()
    assert len(w.zones) == 1
    w.copy_zones("res")
    assert len(w.zones) == 3

    # what to find: preset "People only"
    i = w.picker.preset.findText("People only")
    w.picker.preset.setCurrentIndex(i)
    w.picker._apply_preset(i)
    assert w.picker.selected() == [0]

    w.model_cb.setCurrentIndex(w.model_cb.findData("yolo11s.pt"))
    w.imgsz.setCurrentIndex(w.imgsz.findData(960))
    w.save_clips.setChecked(True)
    w.per_event.setValue(3)

    s = w.collect_settings()
    assert len(s.excluded) == 1 and s.classes == [0] and s.frames_per_event == 3

    w.start()
    wait(app, lambda: w.worker is not None and w.worker.isFinished() and w.stack.currentIndex() == 7, timeout=600)
    r = json.loads((tmp_path / "out" / "results.json").read_text())
    assert r["complete"] and len(r["files"]) == 2
    assert r["events"], "expected person events"
    assert all(set(e["classes"]) == {"person"} for e in r["events"])
    assert all(len(e["frames"]) <= 3 for e in r["events"])
    # zone ignores the left third: no kept person box with bottom-centre in x < 800
    for e in r["events"]:
        for fr in e["frames"]:
            for b in fr["boxes"]:
                assert (b[0] + b[2]) / 2 >= 800
    assert (tmp_path / "out" / "job.yfjob").exists()

    # viewer
    v = w.viewer
    assert v.model.rowCount() == len(r["events"])
    v.mode_cb.setCurrentIndex(1)
    assert v.model.rowCount() == sum(len(e["frames"]) for e in r["events"])
    v.grid.setCurrentIndex(v.model.index(0))
    app.processEvents()
    assert v.cur_fr is not None
    v.flag_btn.setChecked(True)
    v.note.setPlainText("checked by test")
    r2 = json.loads((tmp_path / "out" / "results.json").read_text())
    assert any(e["flag"] and e["note"] == "checked by test" for e in r2["events"])
    v.flag_only.setChecked(True)
    assert v.model.rowCount() >= 1
    from yolofinder.gui.viewer import build_report
    html = build_report(v.res, [e for e in v.res.events if e.get("flag")], "flagged")
    assert "checked by test" in html

    # job save/load round trip through the UI
    jp = tmp_path / "x.yfjob"
    w.collect_settings().save(jp)
    w.apply_settings(JobSettings.load(jp))
    assert w.collect_settings().classes == [0]
    w.close()
