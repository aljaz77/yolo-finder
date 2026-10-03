"""Creates a GitHub release for the current version and uploads dist/YoloFinder.exe.

Token is read from stdin (e.g. piped from `git credential fill`), never stored or printed.
  printf "protocol=https\\nhost=github.com\\n\\n" | git credential fill | sed -n 's/^password=//p' | python build/release.py
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "aljaz77/yolo-finder"
sys.path.insert(0, str(ROOT / "app"))
from yolofinder import __version__  # noqa: E402

BODY = r"""**Download `YoloFinder.exe`** (12 MB, portable).

On first start it sets up a private Python + PyTorch environment in `C:\Users\<you>\YoloFinder`
(about 5.5 GB with an NVIDIA GPU, 2.5 GB CPU-only; needs roughly twice that free during setup).
Later starts open the app directly, and a newer exe updates the app automatically.
`YoloFinder.exe --setup` reopens setup (repair / switch GPU-CPU).

Features: batch detection over folders of videos and photos, ignore/only-here zones, 80 object classes
with presets, parked/stationary-object filter, burned-in timestamp OCR (time-lapse aware), clean and
annotated full frames, per-event clips, cached re-runs, and a results viewer with flags, notes, clip export
and HTML report.

Windows may show a SmartScreen warning because the exe is unsigned (More info → Run anyway)."""


def main():
    tok = sys.stdin.read().strip()
    if not tok:
        sys.exit("no token on stdin")
    h = {"Authorization": "Bearer " + tok, "Accept": "application/vnd.github+json", "User-Agent": "yolofinder-release"}
    tag = f"v{__version__}"
    data = json.dumps({"tag_name": tag, "target_commitish": "main", "name": f"YOLO Finder {__version__}",
                       "body": BODY}).encode()
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/releases", data=data, headers=h, method="POST")
    rel = json.load(urllib.request.urlopen(req))
    print("release", rel["html_url"])
    exe = (ROOT / "dist" / "YoloFinder.exe").read_bytes()
    up = urllib.request.Request(
        f"https://uploads.github.com/repos/{REPO}/releases/{rel['id']}/assets?name=YoloFinder.exe",
        data=exe, headers={**h, "Content-Type": "application/octet-stream"}, method="POST")
    a = json.load(urllib.request.urlopen(up, timeout=600))
    print("asset", a["browser_download_url"], a["size"])


if __name__ == "__main__":
    main()
