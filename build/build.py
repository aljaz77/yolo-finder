"""Builds dist/YoloFinder.exe (portable launcher with the app wheel embedded).

  .venv\\Scripts\\python build\\build.py
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
OUT = ROOT / "build" / "out"
DIST = ROOT / "dist"
PY = sys.executable

PYTHON = "3.12"
TORCH_CUDA = ["torch==2.11.0+cu128", "torchvision==0.26.0+cu128"]
TORCH_CPU = ["torch==2.11.0+cpu", "torchvision==0.26.0+cpu"]
DEPS = ["ultralytics==8.4.171", "PySide6==6.11.2", "rapidocr-onnxruntime==1.4.4", "imageio-ffmpeg==0.6.0",
        "opencv-python==5.0.0.93", "numpy==2.5.2", "pillow==12.3.0"]


def run(*a, **kw):
    print(">", " ".join(map(str, a)), flush=True)
    subprocess.run([str(x) for x in a], check=True, **kw)


def main():
    sys.path.insert(0, str(APP))
    from yolofinder import __version__
    shutil.rmtree(OUT, ignore_errors=True)
    payload = OUT / "payload"
    payload.mkdir(parents=True)

    run(PY, ROOT / "build" / "make_icon.py", APP / "yolofinder")
    run(PY, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-w", payload, APP)
    for d in (APP / "build", *APP.glob("*.egg-info")):
        shutil.rmtree(d, ignore_errors=True)
    wheel = next(payload.glob("yolofinder-*.whl"))
    shutil.copy(APP / "yolofinder" / "icon.ico", payload / "icon.ico")
    deps_id = hashlib.sha1(json.dumps([PYTHON, TORCH_CUDA, TORCH_CPU, DEPS]).encode()).hexdigest()[:12]
    (payload / "manifest.json").write_text(json.dumps({
        "version": __version__, "build": hashlib.sha1(wheel.read_bytes()).hexdigest()[:12], "wheel": wheel.name, "python": PYTHON, "torch_cuda": TORCH_CUDA,
        "torch_cpu": TORCH_CPU, "deps": DEPS, "deps_id": deps_id, "default_model": "yolo11m.pt"}, indent=2))

    sep = ";"
    run(PY, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed", "--name", "YoloFinder",
        "--icon", payload / "icon.ico", "--distpath", DIST, "--workpath", OUT / "pyi", "--specpath", OUT,
        "--add-data", f"{payload / 'manifest.json'}{sep}.", "--add-data", f"{wheel}{sep}.",
        "--add-data", f"{payload / 'icon.ico'}{sep}.",
        "--exclude-module", "numpy", "--exclude-module", "PySide6", "--exclude-module", "torch",
        ROOT / "launcher" / "launcher.py")
    exe = DIST / "YoloFinder.exe"
    print(f"\nBuilt {exe} ({exe.stat().st_size / 1e6:.1f} MB), app {__version__}, deps {deps_id}")


if __name__ == "__main__":
    main()
