"""
Streamlit UI: image mode (single-shot person detection) and video mode
(detection + tracking + line-crossing IN/OUT counting), switched by one
toggle as requested.

Privacy / data handling (see SPEC.md):
- All inference runs locally via the bundled ONNX model — no third-party
  API calls, no uploaded data ever leaves this machine/process.
- Uploaded files are processed in memory / a per-run temp file for video
  (OpenCV's VideoCapture needs a real path) and are not persisted anywhere
  once the app finishes handling the request; the temp file is deleted
  immediately after processing.
"""
from __future__ import annotations

import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from detector import PersonDetector
from video_counter import VideoPeopleCounter

MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolox_nano.onnx")
MAX_UPLOAD_MB = 200
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
ALLOWED_VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]

st.set_page_config(page_title="People Counter", page_icon="🧑‍🤝‍🧑", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

:root {
    --pc-primary: #4F46E5;
    --pc-primary-dark: #3730A3;
    --pc-accent: #06B6D4;
    --pc-bg-card: #F8FAFC;
    --pc-border: #E2E8F0;
    --pc-text-muted: #64748B;
}

.pc-hero {
    background: linear-gradient(135deg, var(--pc-primary) 0%, var(--pc-accent) 100%);
    border-radius: 16px;
    padding: 2rem 2.25rem;
    margin-bottom: 1.5rem;
    color: white;
}
.pc-hero h1 {
    margin: 0 0 0.35rem 0;
    font-size: 2rem;
    font-weight: 700;
    color: white;
}
.pc-hero p {
    margin: 0;
    font-size: 1.02rem;
    opacity: 0.92;
}
.pc-badges {
    margin-top: 0.9rem;
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.pc-badge {
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 500;
}

.pc-card {
    background: var(--pc-bg-card);
    border: 1px solid var(--pc-border);
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
}
.pc-card h4 {
    margin-top: 0;
}

div[data-testid="stMetric"] {
    background: var(--pc-bg-card);
    border: 1px solid var(--pc-border);
    border-radius: 10px;
    padding: 0.85rem 1rem 0.6rem 1rem;
}

div[data-testid="stFileUploaderDropzone"] {
    border-radius: 12px;
}

.stButton > button, .stDownloadButton > button {
    border-radius: 8px;
    font-weight: 600;
}
.stButton > button[kind="primary"], .stDownloadButton > button {
    background: var(--pc-primary);
    border-color: var(--pc-primary);
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover {
    background: var(--pc-primary-dark);
    border-color: var(--pc-primary-dark);
}

.pc-footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--pc-border);
    color: var(--pc-text-muted);
    font-size: 0.85rem;
}
</style>
"""


@st.cache_resource
def load_detector(conf_thresh: float) -> PersonDetector:
    return PersonDetector(MODEL_PATH, conf_thresh=conf_thresh)


def draw_boxes(bgr_image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    out = bgr_image.copy()
    for x1, y1, x2, y2, score in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (79, 70, 229), 3)
        label = f"{score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), (79, 70, 229), -1)
        cv2.putText(out, label, (x1 + 4, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return out


def run_image_mode(conf_thresh: float) -> None:
    uploaded = st.file_uploader("Upload an image", type=ALLOWED_IMAGE_TYPES)
    if uploaded is None:
        st.info("Upload a JPG or PNG to count people in it.")
        return

    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File too large (max {MAX_UPLOAD_MB} MB).")
        return

    try:
        pil_img = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("Could not read this file as an image. Please upload a valid JPG or PNG.")
        return

    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    detector = load_detector(conf_thresh)

    t0 = time.time()
    boxes = detector.detect(bgr)
    elapsed_ms = (time.time() - t0) * 1000

    annotated = draw_boxes(bgr, boxes)

    m1, m2 = st.columns(2)
    m1.metric("People detected", len(boxes))
    m2.metric("Inference time (CPU)", f"{elapsed_ms:.0f} ms")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(pil_img, width="stretch")
    with col2:
        st.subheader("Detected")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")

    if len(boxes) > 0:
        with st.expander("Detection details"):
            st.dataframe(
                {
                    "x1": boxes[:, 0].round(1),
                    "y1": boxes[:, 1].round(1),
                    "x2": boxes[:, 2].round(1),
                    "y2": boxes[:, 3].round(1),
                    "confidence": boxes[:, 4].round(3),
                },
                width="stretch",
            )

    _, buf = cv2.imencode(".png", annotated)
    st.download_button("Download annotated image", bytes(buf), file_name="annotated.png", mime="image/png")


def run_video_mode(conf_thresh: float) -> None:
    st.markdown("**Counting line** — where crossings are counted (as % of frame width/height):")
    c1, c2, c3, c4 = st.columns(4)
    lx1 = c1.slider("Line x1 (%)", 0, 100, 50)
    ly1 = c2.slider("Line y1 (%)", 0, 100, 0)
    lx2 = c3.slider("Line x2 (%)", 0, 100, 50)
    ly2 = c4.slider("Line y2 (%)", 0, 100, 100)

    uploaded = st.file_uploader("Upload a video", type=ALLOWED_VIDEO_TYPES)
    if uploaded is None:
        st.info("Upload a short MP4/MOV/AVI clip to count people crossing a line.")
        return

    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File too large (max {MAX_UPLOAD_MB} MB).")
        return

    if not st.button("Process video", type="primary"):
        return

    # OpenCV's VideoCapture needs a real file path; write to a private temp
    # file and always remove it afterward (finally-block), regardless of
    # success or failure, so nothing uploaded is left on disk.
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    tmp_path = None
    out_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            st.error("Could not read this file as a video. Please upload a valid MP4/MOV/AVI/MKV.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        line = (lx1 / 100 * width, ly1 / 100 * height, lx2 / 100 * width, ly2 / 100 * height)
        detector = load_detector(conf_thresh)
        counter = VideoPeopleCounter(detector, line=line)

        out_path = tmp_path + "_out.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        progress = st.progress(0.0)
        status = st.empty()
        frame_i = 0
        t0 = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = counter.process_frame(frame)
            writer.write(result.annotated)
            frame_i += 1
            if n_frames > 0:
                progress.progress(min(frame_i / n_frames, 1.0))
            status.text(f"Frame {frame_i}/{n_frames or '?'} — IN: {result.in_count}  OUT: {result.out_count}")

        writer.release()
        cap.release()
        elapsed = time.time() - t0

        st.success(f"Done in {elapsed:.1f}s ({frame_i} frames).")
        m1, m2, m3 = st.columns(3)
        m1.metric("IN", counter.line_counter.in_count)
        m2.metric("OUT", counter.line_counter.out_count)
        m3.metric("NET", counter.line_counter.net_count)

        with open(out_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
        st.download_button("Download annotated video", video_bytes, file_name="annotated.mp4", mime="video/mp4")

    finally:
        for p in (tmp_path, out_path):
            if p and os.path.exists(p):
                os.remove(p)


def render_hero() -> None:
    st.markdown(
        """
        <div class="pc-hero">
            <h1>🧑‍🤝‍🧑 People Counter</h1>
            <p>Local person detection &amp; line-crossing counting — nothing you upload ever leaves this app.</p>
            <div class="pc-badges">
                <span class="pc-badge">YOLOX-nano · ONNX Runtime</span>
                <span class="pc-badge">Runs 100% on CPU</span>
                <span class="pc-badge">No third-party APIs</span>
                <span class="pc-badge">No stored uploads</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_about() -> None:
    st.markdown(
        """
<div class="pc-card">
<h4>What this is</h4>
People Counter detects and counts people in a photo, or tracks them across a
video and counts how many crossed a line you draw — in either direction
(IN / OUT / NET). It was built to replace an earlier prototype that used a
face detector and mislabeled itself as a "person counter."
</div>

<div class="pc-card">
<h4>How it works</h4>

**Detection** — <a href="https://github.com/Megvii-BaseDetection/YOLOX" target="_blank">YOLOX-nano</a>,
COCO-pretrained, exported to ONNX and run locally via ONNX Runtime. Only the
"person" class is kept.

**Tracking (video mode)** — <code>ByteTrackTracker</code> from the
<a href="https://github.com/roboflow/trackers" target="_blank">trackers</a>
library, associating detections frame-to-frame.

**Counting** — a small, independent module watches which side of your
counting line each tracked person's centroid is on, and counts a crossing
whenever that side flips — so someone walking back and forth is counted
correctly each time, not just once.
</div>

<div class="pc-card">
<h4>Measured accuracy</h4>
Evaluated against a real labeled image sample (62 people-containing images
from the public <em>coco128</em> dataset), matching detections to
ground-truth boxes at IoU ≥ 0.5:

| Metric | Value |
|---|---|
| Precision | 0.825 |
| Recall | 0.520 |
| F1 | 0.638 |
| CPU inference speed | ~95 FPS (10.5 ms/image) |

Recall is the honest weak point — YOLOX-<em>nano</em> is the smallest model
in its family (built for speed over accuracy), and struggles most on dense
crowd scenes with many small, overlapping people. Full numbers are in the
repo's <code>eval_results.json</code>, reproducible via <code>evaluate.py</code>.
</div>

<div class="pc-card">
<h4>Privacy &amp; data handling</h4>

- All inference runs **locally** — no image or video you upload is ever sent to a third-party API.
- Video is written to a private temporary file only because OpenCV needs a
  real file path to read it; that file is deleted immediately after
  processing, even if something goes wrong.
- Nothing you upload is logged, stored, or kept between requests.
- Uploads are capped at 200 MB and validated before processing — a corrupt
  or unsupported file gets a clear error, not a crash.
</div>

<div class="pc-card">
<h4>License</h4>
This project's code is MIT-licensed. The bundled YOLOX-nano weights are
Apache License 2.0 (<a href="https://github.com/Megvii-BaseDetection/YOLOX" target="_blank">Megvii-BaseDetection/YOLOX</a>),
used unmodified for inference only.
</div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_hero()

    tab_detect, tab_about = st.tabs(["🔍 Detect", "ℹ️ About"])

    with tab_detect:
        mode = st.radio("Mode", ["Image", "Video"], horizontal=True)
        conf_thresh = st.sidebar.slider("Detection confidence threshold", 0.1, 0.9, 0.35, 0.05)
        st.sidebar.caption("Lower = more detections (more false positives). Higher = fewer, more confident detections.")

        if mode == "Image":
            run_image_mode(conf_thresh)
        else:
            run_video_mode(conf_thresh)

    with tab_about:
        render_about()

    st.markdown(
        '<div class="pc-footer">People Counter — local YOLOX-nano inference, MIT licensed.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
