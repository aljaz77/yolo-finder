"""YOLO Finder portable launcher / installer.

First run: shows a setup window, downloads uv, a private Python, PyTorch (CUDA build when an
NVIDIA GPU is present, else CPU), the other dependencies and the app wheel bundled in this exe.
Later runs: starts the installed app immediately (re-installs the bundled app if it is newer).

  YoloFinder.exe                 start (or set up)
  YoloFinder.exe --setup         show the setup window (repair / change location / reinstall)
  YoloFinder.exe --auto DIR [cuda|cpu] [--shortcut]   unattended install into DIR, no window
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import traceback
import urllib.request
import zipfile
from pathlib import Path

APP = "YOLO Finder"
UV_URL = "https://github.com/astral-sh/uv/releases/download/{ver}/uv-x86_64-pc-windows-msvc.zip"
UV_VERSION = "0.12.22"
UV_FALLBACK = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
TORCH_INDEX = {"cuda": "https://download.pytorch.org/whl/cu128", "cpu": "https://download.pytorch.org/whl/cpu"}
CREATE_NO_WINDOW = 0x08000000


def bundle_dir() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).parent / "payload"))


def manifest() -> dict:
    return json.loads((bundle_dir() / "manifest.json").read_text(encoding="utf-8"))


def config_path() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "YoloFinder" / "launcher.json"


def read_cfg() -> dict:
    try:
        return json.loads(config_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_cfg(cfg):
    config_path().parent.mkdir(parents=True, exist_ok=True)
    config_path().write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def default_dir() -> Path:
    # not under AppData: uv's managed-Python install fails there on some systems (minor-version link error)
    return Path.home() / "YoloFinder"


def has_nvidia() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15, creationflags=CREATE_NO_WINDOW)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return None


def env_python(home: Path, gui=False) -> Path:
    return home / "env" / "Scripts" / ("pythonw.exe" if gui else "python.exe")


def installed_version(home: Path):
    try:
        return json.loads((home / "installed.json").read_text(encoding="utf-8"))
    except Exception:
        return None


def free_gb(path: Path) -> float:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


# ------------------------------------------------------------------ install
class Installer:
    def __init__(self, home: Path, device: str, log, progress, shortcut=True):
        self.home, self.device, self.progress = Path(home), device, progress
        self.shortcut = shortcut
        self.m = manifest()
        self.home.mkdir(parents=True, exist_ok=True)
        self._logf = open(self.home / "install.log", "a", encoding="utf-8")

        def both(msg):
            self._logf.write(msg + "\n")
            self._logf.flush()
            log(msg)
        self.log = both

    def run_cmd(self, args, step):
        self.log(f"> {' '.join(str(a) for a in args)}")
        env = dict(os.environ, UV_CACHE_DIR=str(self.home / "uv-cache"), UV_PYTHON_INSTALL_DIR=str(self.home / "python"),
                   UV_LINK_MODE="copy", PYTHONUTF8="1", UV_NO_PROGRESS="1")
        p = subprocess.Popen([str(a) for a in args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", env=env, creationflags=CREATE_NO_WINDOW)
        for line in p.stdout:
            line = line.rstrip()
            if line:
                self.log("  " + line)
        if p.wait() != 0:
            raise RuntimeError(f"Step failed: {step} (exit code {p.returncode})")

    def download(self, url, dest: Path):
        self.log(f"Downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "YoloFinder-launcher"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while True:
                b = r.read(1 << 16)
                if not b:
                    break
                f.write(b)
                got += len(b)
        self.log(f"  {got / 1e6:.1f} MB")

    def uv(self) -> Path:
        exe = self.home / "bin" / "uv.exe"
        if exe.exists():
            return exe
        exe.parent.mkdir(parents=True, exist_ok=True)
        z = self.home / "bin" / "uv.zip"
        try:
            self.download(UV_URL.format(ver=UV_VERSION), z)
        except Exception as e:
            self.log(f"  pinned uv download failed ({e}), trying latest")
            self.download(UV_FALLBACK, z)
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.endswith("uv.exe"):
                    exe.write_bytes(zf.read(n))
        z.unlink()
        return exe

    def install(self):
        m, home = self.m, self.home
        home.mkdir(parents=True, exist_ok=True)
        steps = 6
        self.progress(0, steps, "Getting the package manager (uv)")
        uv = self.uv()
        py = env_python(home)
        prev = installed_version(home)
        need_env = not py.exists() or not prev or prev.get("deps") != m["deps_id"] or prev.get("device") != self.device
        if need_env:
            self.progress(1, steps, f"Installing Python {m['python']}")
            # --no-bin/--no-registry: keep it portable (no ~/.local/bin shim, no registry entries)
            pyargs = [uv, "python", "install", "--no-bin", "--no-registry", m["python"]]
            try:
                self.run_cmd(pyargs, "python")
            except RuntimeError:
                # antivirus scanning can race uv's extraction; a clean retry normally succeeds
                self.log("Retrying the Python install…")
                shutil.rmtree(home / "python", ignore_errors=True)
                self.run_cmd(pyargs, "python")
            if (home / "env").exists() and (not prev or prev.get("device") != self.device):
                shutil.rmtree(home / "env", ignore_errors=True)
            if not py.exists():
                self.run_cmd([uv, "venv", "--python", m["python"], home / "env"], "venv")
            self.progress(2, steps, f"Installing PyTorch ({'NVIDIA GPU / CUDA' if self.device == 'cuda' else 'CPU'}) "
                                    "– this is the big download (CUDA ≈ 3 GB)")
            torch_pkgs = m["torch_cuda"] if self.device == "cuda" else m["torch_cpu"]
            # uv gives --extra-index-url priority, so torch comes from the PyTorch index, the rest from PyPI
            self.run_cmd([uv, "pip", "install", "--python", py, "--extra-index-url", TORCH_INDEX[self.device],
                          *torch_pkgs], "pytorch")
            self.progress(3, steps, "Installing detection, OCR and GUI libraries")
            self.run_cmd([uv, "pip", "install", "--python", py, "--extra-index-url", TORCH_INDEX[self.device],
                          *torch_pkgs, *m["deps"]], "dependencies")
        self.progress(4, steps, "Installing YOLO Finder")
        wheel = bundle_dir() / m["wheel"]
        self.run_cmd([uv, "pip", "install", "--python", py, "--no-deps", "--reinstall", wheel], "app")
        (home / "env" / "yolofinder_home.txt").write_text(str(home / "data"), encoding="utf-8")
        (home / "data").mkdir(exist_ok=True)
        self.progress(5, steps, "Downloading the default detection model")
        self.run_cmd([py, "-c", "from yolofinder.core.detector import resolve_model;"
                                f"print(resolve_model({m['default_model']!r}))"], "model")
        self.run_cmd([py, "-c", "import torch, PySide6, ultralytics, rapidocr_onnxruntime, cv2;"
                                "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"], "self-test")
        (home / "installed.json").write_text(json.dumps({"app": m["version"], "build": m["build"], "deps": m["deps_id"],
                                                         "device": self.device}), encoding="utf-8")
        # the download cache duplicates everything (~5 GB with CUDA); not needed once installed
        shutil.rmtree(home / "uv-cache", ignore_errors=True)
        self.log("Removed download cache")
        if self.shortcut:
            try:
                make_shortcut(home)
                self.log("Desktop shortcut created")
            except Exception as e:
                self.log(f"Could not create shortcut: {e}")
        cfg = read_cfg()
        cfg["home"] = str(home)
        write_cfg(cfg)
        self.progress(steps, steps, "Done")


def make_shortcut(home: Path):
    target = home / "env" / "Scripts" / "yolofinder-gui.exe"
    exe = Path(sys.executable) if getattr(sys, "frozen", False) else target
    icon = home / "env" / "Lib" / "site-packages" / "yolofinder" / "icon.ico"
    desktop = Path(os.path.expandvars(r"%USERPROFILE%\Desktop"))
    lnk = desktop / "YOLO Finder.lnk"
    ps = (f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}');"
          f"$s.TargetPath='{exe}';$s.IconLocation='{icon}';$s.WorkingDirectory='{exe.parent}';$s.Save()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, creationflags=CREATE_NO_WINDOW,
                   capture_output=True)


def launch(home: Path, extra=()):
    exe = home / "env" / "Scripts" / "yolofinder-gui.exe"
    cmd = [str(exe), *extra] if exe.exists() else [str(env_python(home, gui=True)), "-m", "yolofinder", *extra]
    subprocess.Popen(cmd, cwd=str(home), close_fds=True)


def needs_install(home: Path) -> bool:
    m = manifest()
    inst = installed_version(home)
    return not (inst and env_python(home).exists() and inst.get("build") == m["build"] and inst.get("deps") == m["deps_id"])


# ------------------------------------------------------------------ GUI
def setup_window(home: Path, start_app=True):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    m = manifest()
    root = tk.Tk()
    root.title(f"{APP} {m['version']} – setup")
    root.geometry("760x560")
    root.minsize(640, 460)
    try:
        root.iconbitmap(str(bundle_dir() / "icon.ico"))
    except Exception:
        pass
    q: queue.Queue = queue.Queue()
    gpu = has_nvidia()
    frm = ttk.Frame(root, padding=14)
    frm.pack(fill="both", expand=True)
    ttk.Label(frm, text=f"{APP} {m['version']}", font=("Segoe UI", 16, "bold")).pack(anchor="w")
    ttk.Label(frm, text="This one-time setup downloads Python, PyTorch and the detection libraries into the folder "
                        "below. Nothing is installed system-wide; delete the folder to remove everything.",
              wraplength=720).pack(anchor="w", pady=(4, 10))
    row = ttk.Frame(frm)
    row.pack(fill="x")
    ttk.Label(row, text="Install folder:").pack(side="left")
    dir_var = tk.StringVar(value=str(home))
    ttk.Entry(row, textvariable=dir_var).pack(side="left", fill="x", expand=True, padx=6)
    ttk.Button(row, text="Browse…", command=lambda: dir_var.set(
        filedialog.askdirectory(initialdir=dir_var.get()) or dir_var.get())).pack(side="left")
    dev_var = tk.StringVar(value="cuda" if gpu else "cpu")
    r2 = ttk.Frame(frm)
    r2.pack(fill="x", pady=8)
    ttk.Radiobutton(r2, text=f"NVIDIA GPU (fast){' – detected: ' + gpu if gpu else ' – no NVIDIA GPU detected'}",
                    variable=dev_var, value="cuda").pack(anchor="w")
    ttk.Radiobutton(r2, text="CPU only (works everywhere, much slower, ~3.5 GB smaller)", variable=dev_var,
                    value="cpu").pack(anchor="w")
    sc_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(frm, text="Create a desktop shortcut", variable=sc_var).pack(anchor="w")
    space = ttk.Label(frm, text="")
    space.pack(anchor="w", pady=(6, 0))

    def upd_space(*_):
        try:
            need = 11 if dev_var.get() == "cuda" else 4
            space.config(text=f"Needs about {need} GB free during setup (≈{need // 2 + 1} GB after) · {free_gb(Path(dir_var.get())):.0f} GB free")
        except Exception:
            space.config(text="")
    dir_var.trace_add("write", upd_space)
    dev_var.trace_add("write", upd_space)
    upd_space()
    pb = ttk.Progressbar(frm, maximum=6)
    pb.pack(fill="x", pady=(10, 4))
    step_lbl = ttk.Label(frm, text="")
    step_lbl.pack(anchor="w")
    txt = tk.Text(frm, height=12, bg="#1c1f25", fg="#c8ccd4", insertbackground="white", font=("Consolas", 9))
    txt.pack(fill="both", expand=True, pady=6)
    btns = ttk.Frame(frm)
    btns.pack(fill="x")
    go = ttk.Button(btns, text="Install and start")
    go.pack(side="right")
    ttk.Button(btns, text="Close", command=root.destroy).pack(side="right", padx=6)
    state = {"busy": False}

    def worker(h, dev, sc):
        try:
            Installer(h, dev, lambda s: q.put(("log", s)), lambda i, n, s: q.put(("prog", i, s)), sc).install()
            q.put(("done",))
        except Exception as e:
            q.put(("log", traceback.format_exc()))
            q.put(("fail", str(e)))

    def start():
        h = Path(dir_var.get())
        if dev_var.get() == "cuda" and not gpu and not messagebox.askyesno(
                APP, "No NVIDIA GPU was detected. Install the GPU version anyway?"):
            return
        state["busy"] = True
        go.config(state="disabled")
        threading.Thread(target=worker, args=(h, dev_var.get(), sc_var.get()), daemon=True).start()

    go.config(command=start)

    def poll():
        try:
            while True:
                msg = q.get_nowait()
                if msg[0] == "log":
                    txt.insert("end", msg[1] + "\n")
                    txt.see("end")
                elif msg[0] == "prog":
                    pb["value"] = msg[1]
                    step_lbl.config(text=msg[2])
                elif msg[0] == "done":
                    state["busy"] = False
                    if start_app:
                        launch(Path(dir_var.get()))
                        root.after(800, root.destroy)
                    else:
                        messagebox.showinfo(APP, "Setup finished.")
                        go.config(state="normal")
                elif msg[0] == "fail":
                    state["busy"] = False
                    go.config(state="normal", text="Retry")
                    messagebox.showerror(APP, f"Setup failed:\n{msg[1]}\n\nSee the log for details. "
                                              "Retrying continues where it stopped.")
        except queue.Empty:
            pass
        root.after(100, poll)

    def on_close():
        if state["busy"] and not messagebox.askyesno(APP, "Setup is running. Cancel and close?"):
            return
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    poll()
    if not needs_install(home) and installed_version(home):
        go.config(text="Reinstall / repair")
    root.mainloop()


def main():
    args = sys.argv[1:]
    if args[:1] == ["--auto"]:
        home = Path(args[1])
        rest = [a for a in args[2:] if not a.startswith("--")]
        dev = rest[0] if rest else ("cuda" if has_nvidia() else "cpu")
        if sys.stdout:
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        out = sys.stdout if sys.stdout else open(os.devnull, "w", encoding="utf-8")
        inst = Installer(home, dev, lambda s: print(s, file=out, flush=True), lambda i, n, s: None,
                         shortcut="--shortcut" in args)
        inst.progress = lambda i, n, s: inst.log(f"[{i}/{n}] {s}")
        inst.install()
        return
    cfg = read_cfg()
    home = Path(cfg.get("home") or default_dir())
    inst = installed_version(home)
    if "--setup" not in args and needs_install(home) and inst and env_python(home).exists() \
            and inst.get("deps") == manifest()["deps_id"]:
        # only the app itself changed: update silently (a few seconds), no setup window
        try:
            Installer(home, inst.get("device", "cuda"), lambda s: None, lambda i, n, s: None,
                      shortcut=False).install()
        except Exception:
            pass
    if "--setup" in args or needs_install(home):
        setup_window(home)
        return
    launch(home, args)


if __name__ == "__main__":
    main()
