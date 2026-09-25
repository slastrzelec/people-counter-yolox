# SPEC — People Counter

## Goal

Replace the old Haar-Cascade "face counter" (mislabeled as a person
counter) with an app that actually detects and counts *people*, in two
modes selectable via one toggle:

1. **Image mode** — upload a photo, detect and count people in it.
2. **Video mode** — upload a clip, track people across frames and count
   IN/OUT crossings over a user-drawn line.

## Model choice

- **Detector:** YOLOX-nano, COCO-pretrained, exported to ONNX, run via
  ONNX Runtime (CPU). Chosen over Ultralytics YOLOv8 specifically for its
  license: YOLOX is Apache 2.0, while YOLOv8's weights/code are AGPL-3.0
  unless a commercial license is purchased. AGPL's copyleft is triggered by
  *network use*, not just distribution — since this app is meant to be
  deployed publicly (e.g. Streamlit Community Cloud), an AGPL model would
  force the whole repo to AGPL. YOLOX-nano keeps everything MIT-compatible.
- **Tracker (video mode):** `trackers.ByteTrackTracker` (the maintained
  successor to `supervision.ByteTrack`, which is deprecated as of
  `supervision` 0.28 and scheduled for removal in 0.31).

## Data security / privacy constraints (addressed before implementation)

- No image or video uploaded to the app is ever sent to a third-party API —
  all inference is local, via the bundled ONNX model.
- No uploaded file is persisted to disk beyond what's strictly required to
  process it: images are handled in memory; video needs a real file path
  for OpenCV's `VideoCapture`, so it's written to a private per-request
  temp file that is deleted immediately after processing (in a `finally`
  block, so it's removed even if processing raises an error).
- No logging or retention of uploaded content between requests or sessions.
- Upload size is capped (200 MB) and file type is restricted to expected
  image/video formats; the app validates the file can actually be decoded
  before processing, and shows a clear error instead of crashing on a
  corrupt or mistyped upload.
- These constraints exist specifically because an earlier AI-assisted
  project caused a data-leak-adjacent rejection in a recruitment process —
  see the standing rule: no code gets written before its data-handling
  approach is specified and reviewed.

## Architecture

Pure logic is kept separate from CV/ML code so it can be unit-tested
without real images, video, or model weights:

- `detector.py` — YOLOX-nano pre/postprocessing + inference. Pure function
  boundary: `PersonDetector.detect(bgr_image) -> (N,5) array of
  [x1,y1,x2,y2,score]`.
- `line_counter.py` — line-crossing counting via the sign of a 2D cross
  product between consecutive frames. No CV dependency at all; fully
  covered by synthetic-position unit tests.
- `video_counter.py` — wires `PersonDetector` + `ByteTrackTracker` +
  `LineCounter` together. Only needs an object with a `.detect()` method,
  so tests inject a fake detector instead of loading real model weights.
- `app.py` — Streamlit UI only; no detection/tracking logic lives here.

## Evaluation

Accuracy and speed are measured, not asserted: `evaluate.py` runs the
detector against a real labeled image sample (coco128, person class only)
and computes actual precision/recall/F1 at IoU ≥ 0.5, plus measured CPU
inference latency. See `README.md` → Evaluation for the current numbers and
how to regenerate them.

## Out of scope (for now)

- Re-identification across camera views or after a person leaves and
  re-enters much later (ByteTrack only tracks within a continuous
  presence).
- GPU inference (CPU-only, by design, for simple free-tier deployment).
- Multiple counting lines / zones in one run (a single line, drawn once
  per video, is enough to demonstrate the core capability).
