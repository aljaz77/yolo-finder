# YOLO Finder

Batch object detection for large sets of videos and photos (dashcam, CCTV, trail cams, phone footage).
Pick folders, tick what to look for, draw areas to ignore, press Start. You get full, uncropped frames
(clean + annotated), optional video clips per event, real timestamps read from the burned-in overlay,
and a results viewer to go through everything.

## Using it

Run **`dist\YoloFinder.exe`**. It's portable, so you can copy it anywhere (USB stick, another PC).

* **First start:** a setup window downloads a private Python, PyTorch and the libraries into
  `C:\Users\<you>\YoloFinder`. You can change the folder there. It takes about 6 GB with an NVIDIA GPU, or 2.5 GB CPU-only (it needs roughly twice that free while installing).
  Nothing is installed system-wide; delete that folder to remove everything.
* **Later starts:** the app opens directly. A newer `YoloFinder.exe` updates the app automatically.
* `YoloFinder.exe --setup` reopens the setup window (repair, reinstall, switch GPU/CPU).

### Workflow

1. **Sources:** add folders/files (or drag & drop), choose the output folder, untick files to skip.
2. **Zones & preview:** scrub through a file and draw *Ignore* areas (your own car's bonnet, a busy road, a
   tree that moves) or *Only-here* areas (e.g. around your friend's car). A detection counts as inside when the
   point where it touches the ground (bottom centre of its box) is inside. Copy zones to all files, or to files
   with the same resolution. *Test detection on this frame* shows what would be found; grey boxes are the ones
   the zones removed. You can also mark the *Timestamp area* if auto-detection can't find the overlay.
3. **What to find:** checkbox menu of all 80 classes, grouped, with presets (Security, Theft, Wildlife…). You can
   save your own presets.
4. **Detection:** model size (n/s/m/l/x), image size, confidence, analyse every Nth frame, the
   parked/stationary-object filter, and how detections are merged into events.
5. **Output:** clean frames, annotated frames, or both. *Frames per event* limits how many frames a long event
   saves in normal-speed video; 0 = all. Clips can be saved per event, with or without boxes, with seconds
   before/after. Tick only clips to get videos instead of stills. Box style can be changed too.
6. **Timestamps:** OCR of the burned-in date/time, with file-name and EXIF fallbacks. Time-lapse speed is
   worked out automatically from several readings.
7. **Run:** live preview, log, progress/ETA, Stop. Stopping keeps partial results.
8. **Results:** one card per event, or every saved frame. You can filter by object, file and ★flag, search OCR
   text and notes, and toggle boxes (**B**). Flag events (**F**), add notes, play the saved clip, open the source
   video at that moment (VLC if installed), export a clip around any frame, and export an HTML report of the
   flagged events.

Re-running into the same output folder reuses cached detections. Changing classes, zones, confidence or filters
and running again only takes seconds per file.

### Output folder

```
frames/      full original frames      2026-10-02_14-00-10_<file>_f000253_car-person.jpg
annotated/   same frames with boxes
clips/       mp4 per event (optional)
results.json everything (used by the viewer)
events.csv   one row per event (time, file, video position, objects, OCR text, flag, note)
job.yfjob    the settings used; File → Open job to run it again
```

Command line: `yolofinder --run job.yfjob` runs a saved job without the GUI.

## Development

```
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install ultralytics PySide6 rapidocr-onnxruntime imageio-ffmpeg pytest pyinstaller
set PYTHONPATH=app
.venv\Scripts\python -m yolofinder          (GUI)
.venv\Scripts\python -m pytest tests         (unit, pipeline and GUI tests; uses clips in tests\data)
.venv\Scripts\python build\build.py          (-> dist\YoloFinder.exe)
```

Layout: `app/yolofinder/core` (detection, filtering, events, OCR, export, no GUI), `app/yolofinder/gui` (Qt),
`launcher/` (portable installer exe), `build/` (icon + build script), `tests/`.

## Licences

Detection uses Ultralytics YOLO11 (AGPL-3.0). OCR uses RapidOCR (Apache-2.0). The GUI uses Qt for Python (LGPL).
