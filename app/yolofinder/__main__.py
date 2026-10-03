"""Entry point.

  python -m yolofinder                      start the GUI
  python -m yolofinder --run job.yfjob      run a saved job headless (prints progress)
  python -m yolofinder --screenshots DIR    GUI self-test: render every page to PNGs and quit
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback

from . import APP_NAME, __version__
from .core.config import app_home


def setup_logging():
    log_dir = app_home() / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(filename=log_dir / "app.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    return log_dir / "app.log"


def run_headless(job_path):
    from .core.config import JobSettings
    from .core.pipeline import Callbacks, Job

    class CB(Callbacks):
        last = -1

        def progress(self, overall, file_frac, msg):
            p = int(overall * 100)
            if p != self.last:
                self.last = p
                print(f"{p:3d}% {msg}", flush=True)

    res = Job(JobSettings.load(job_path), CB()).run()
    print(f"Results: {res}")


def run_gui(screenshots=None, open_results=None):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication, QMessageBox

    from .gui.main_window import MainWindow
    from .gui.theme import apply_dark

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    apply_dark(app)
    from pathlib import Path
    icon = Path(__file__).parent / "icon.png"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))

    def excepthook(t, v, tb):
        msg = "".join(traceback.format_exception(t, v, tb))
        logging.error(msg)
        if not screenshots:
            QMessageBox.critical(None, APP_NAME, f"Unexpected error:\n\n{v}\n\nDetails were written to the log.")
        else:
            print(msg, file=sys.stderr)
    sys.excepthook = excepthook

    w = MainWindow()
    w.show()
    if open_results:
        w.viewer.load(open_results)
        w.nav.setCurrentRow(7)
    if screenshots:
        out = Path(screenshots)
        out.mkdir(parents=True, exist_ok=True)
        pages = list(range(w.stack.count()))

        def shoot(i=0):
            if i >= len(pages):
                app.quit()
                return
            w.nav.setCurrentRow(pages[i])
            QTimer.singleShot(1500, lambda: (w.grab().save(str(out / f"page{pages[i]}.png")), shoot(i + 1)))
        QTimer.singleShot(2500, shoot)
    return app.exec()


def main():
    ap = argparse.ArgumentParser(prog="yolofinder", description=f"{APP_NAME} {__version__}")
    ap.add_argument("--run", metavar="JOB", help="run a saved .yfjob headless")
    ap.add_argument("--screenshots", metavar="DIR", help=argparse.SUPPRESS)
    ap.add_argument("--results", metavar="DIR", help="open a results folder in the viewer")
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args()
    setup_logging()
    logging.info("%s %s starting (%s)", APP_NAME, __version__, sys.executable)
    if args.run:
        run_headless(args.run)
        return 0
    return run_gui(args.screenshots, args.results)


if __name__ == "__main__":
    sys.exit(main())
