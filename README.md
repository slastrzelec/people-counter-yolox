# People Counter

Local, CPU-only person detection and counting — image mode (detect and count
people in a photo) and video mode (track people and count IN/OUT crossings
over a line), switched by a single toggle in the Streamlit UI.

Rebuilt from scratch to replace an earlier Haar-Cascade face counter that was
mislabeled as a general "person counter." This version uses an actual person
detector (YOLOX-nano) plus multi-object tracking, and reports only metrics
that are directly reproducible from the code in this repo (see
[Evaluation](#evaluation) below).

## How it works

- **Detection:** [YOLOX-nano](https://github.com/Megvii-BaseDetection/YOLOX)
  (COCO-pretrained, ONNX export), run locally via ONNX Runtime on CPU. Only
  the "person" class is kept.
- **Tracking (video mode):** [`trackers`](https://github.com/roboflow/trackers)'
  `ByteTrackTracker`, associating detections frame-to-frame by IoU.
- **Line-crossing counting:** a small, pure-Python module
  (`line_counter.py`) that watches which side of a user-drawn line each
  tracked person's centroid is on, and counts a crossing whenever that side
  flips — independent of any CV/ML code, so it's unit-tested with synthetic
  positions, no video or model weights required.

See [`SPEC.md`](SPEC.md) for the full design spec, including the data
security constraints this app was built to satisfy.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`). Pick
**Image** or **Video** mode at the top, adjust the confidence threshold in
the sidebar if needed, and upload a file.

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

All tracking and line-crossing logic is tested with synthetic, injectable
detectors (`FakeDetector` in the test files) — no real video, images, or
model weights are needed to run the suite.

## Evaluation

`evaluate.py` measures real detection accuracy and speed against a labeled
sample: the public
[coco128](https://github.com/ultralytics/assets/releases) subset of COCO
(128 images, YOLO-format labels, class 0 = person — same class ordering the
model uses). It computes true/false positives against ground-truth boxes at
IoU ≥ 0.5 and reports actual precision/recall/F1, plus measured per-image
CPU inference latency. Nothing here is estimated or hand-picked — running
the script regenerates these numbers from scratch.

```bash
curl -L -o coco128.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
unzip coco128.zip -d coco128_raw
python evaluate.py
```

Results from the last run (`conf_thresh=0.35`, `iou_match_thresh=0.5`,
default CPU inference, see `eval_results.json`):

| Metric | Value |
|---|---|
| Precision | 0.825 |
| Recall | 0.520 |
| F1 | 0.638 |
| Mean inference latency (CPU) | 10.5 ms/image |
| Approx. throughput (CPU) | ~95 FPS |

Recall is the honest weak point: YOLOX-nano is the smallest model in the
YOLOX family (optimized for speed, not accuracy), run at 416×416, and
coco128 includes several crowd scenes with many small/overlapping people —
exactly the case a nano-sized model at this resolution struggles with most.
A larger YOLOX variant (s/m/l) or a higher input resolution would trade some
of that ~95 FPS for materially better recall on dense scenes; nano was
chosen here to keep the app usable on CPU-only hosting.

## Data security / privacy

- All inference runs **locally** via the bundled ONNX model — no third-party
  API calls, no uploaded image or video ever leaves the process it's
  handled in.
- Uploaded files are processed in memory (images) or a private per-request
  temp file (video, since OpenCV's `VideoCapture` requires a real file
  path); the temp file is deleted immediately after processing, in a
  `finally` block that runs even if processing fails.
- Nothing uploaded is written to permanent storage, logged, or retained
  between requests.
- Upload size is capped (200 MB) and only image/video file types are
  accepted; unreadable or corrupt files are rejected with an error message
  rather than crashing the app.

## Project structure

```
people-counter-yolox/
├── app.py                  # Streamlit UI (image/video mode toggle)
├── detector.py              # YOLOX-nano ONNX inference (pure, no UI dependency)
├── line_counter.py          # Pure line-crossing counting logic
├── video_counter.py         # Wires detector + tracker + line_counter for video mode
├── evaluate.py               # Real accuracy/speed evaluation against coco128
├── yolox_nano.onnx           # Model weights (Apache 2.0, see License)
├── requirements.txt
├── requirements-dev.txt
├── tests/
│   ├── test_line_counter.py
│   └── test_video_counter.py
├── SPEC.md
└── LICENSE
```

## License

This project's code is MIT-licensed — see [`LICENSE`](LICENSE).

The bundled model weights (`yolox_nano.onnx`) are from
[Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX),
licensed Apache License 2.0, used here unmodified for inference only.
YOLOX-nano was chosen specifically over Apache/AGPL alternatives like
Ultralytics YOLOv8 to keep the whole repo under a permissive, MIT-compatible
license even when deployed as a publicly network-accessible app.
