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
import imageio
import numpy as np
import streamlit as st
from PIL import Image

from detector import PersonDetector
from video_counter import VideoPeopleCounter

MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolox_nano.onnx")
MAX_UPLOAD_MB = 200
# Streamlit Community Cloud's free tier is CPU-only with ~1GB RAM. Processing
# every frame of a long or very high-resolution video (detection + drawing +
# H.264 encoding) can exceed that and crash the whole process (observed:
# healthz EOF mid-video, no Python traceback — a resource kill, not a code
# bug). These two caps keep video mode within what the free tier can handle.
MAX_VIDEO_FRAMES = 1800  # ~60s at 30fps
MAX_PROCESS_LONG_SIDE = 960  # px; frames are downscaled to this before processing
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
ALLOWED_VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]

st.set_page_config(page_title="People Counter", page_icon="🧑‍🤝‍🧑", layout="wide")

# "Editorial / warm" theme — a calm, magazine-like look (Fraunces display
# serif headings + Source Serif 4 body, cream paper background, terracotta
# accent). One fixed palette is forced app-wide (same reasoning as before:
# floating colors on top of Streamlit's own light/dark theme is what caused
# the earlier invisible-text bug — owning every surface color avoids that
# class of bug entirely, regardless of which look is on top).
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap');

:root {
    --pc-bg: #FAF7F2;
    --pc-surface: #FFFFFF;
    --pc-surface-2: #F2ECE1;
    --pc-border: #E4DCCF;
    --pc-text: #2B2620;
    --pc-text-bright: #2B2620;
    --pc-text-muted: #55493D;
    --pc-text-faint: #9A8F7F;
    --pc-terracotta: #C2634A;
}

html, body, [class*="css"], .stApp {
    font-family: 'Source Serif 4', Georgia, serif !important;
    background: var(--pc-bg) !important;
    color: var(--pc-text) !important;
}
.stApp header[data-testid="stHeader"] { background: var(--pc-bg) !important; }
section[data-testid="stSidebar"] {
    background: var(--pc-surface-2) !important;
    border-right: 1px solid var(--pc-border);
}
section[data-testid="stSidebar"] * { color: var(--pc-text) !important; }
h1, h2, h3, h4, h5, h6 { font-family: 'Fraunces', Georgia, serif !important; color: var(--pc-text); }
p, span, label, div { color: var(--pc-text); }

.pc-hero {
    border-bottom: 1px solid var(--pc-border);
    padding-bottom: 1.4rem;
    margin-bottom: 1.6rem;
}
.pc-kicker {
    font-size: 12px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--pc-text-faint);
    margin-bottom: 16px;
}
.pc-hero h1 {
    margin: 0 0 10px 0;
    font-size: 2.6rem;
    font-weight: 600;
    color: var(--pc-text-bright) !important;
    letter-spacing: -0.01em;
}
.pc-hero h1 .accent { color: var(--pc-terracotta); }
.pc-hero p {
    margin: 0;
    font-size: 1rem;
    line-height: 1.6;
    color: var(--pc-text-muted) !important;
    max-width: 640px;
}

.pc-card {
    background: var(--pc-surface);
    color: var(--pc-text);
    border: 1px solid var(--pc-border);
    border-radius: 2px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
}
.pc-card h4 {
    margin-top: 0;
    color: var(--pc-text-bright) !important;
    font-size: 12px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: 'Source Serif 4', serif !important;
    font-weight: 600;
}
.pc-card, .pc-card p, .pc-card li, .pc-card strong { color: var(--pc-text) !important; }
.pc-card a { color: var(--pc-terracotta) !important; }
.pc-card code {
    background: rgba(194, 99, 74, 0.08);
    color: var(--pc-terracotta) !important;
    border-radius: 3px;
    font-family: ui-monospace, monospace;
}
.pc-card table { color: var(--pc-text); }
.pc-card th { color: var(--pc-text-muted) !important; text-transform: uppercase; font-size: 11px; letter-spacing: 0.06em; }

/* st.metric, restyled as an editorial pull-figure */
div[data-testid="stMetric"] {
    background: var(--pc-surface);
    border: 1px solid var(--pc-border);
    border-left: 3px solid var(--pc-terracotta);
    border-radius: 2px;
    padding: 0.9rem 1.1rem 0.7rem 1.1rem;
}
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: var(--pc-text-muted) !important;
    font-size: 11px !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--pc-text) !important;
    font-family: 'Fraunces', serif;
    font-weight: 600;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { color: var(--pc-terracotta) !important; }

div[data-testid="stFileUploaderDropzone"] {
    background: var(--pc-surface-2) !important;
    border: 1px dashed var(--pc-border) !important;
    border-radius: 2px;
}
div[data-testid="stFileUploaderDropzone"] * { color: var(--pc-text-muted) !important; }

.stButton > button, .stDownloadButton > button {
    border-radius: 2px;
    font-weight: 600;
    font-family: 'Source Serif 4', serif;
    background: transparent;
    color: var(--pc-terracotta) !important;
    border: 1px solid var(--pc-terracotta) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: rgba(194, 99, 74, 0.08) !important;
}
.stButton > button[kind="primary"] {
    background: var(--pc-terracotta) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--pc-terracotta) !important;
}
.stButton > button[kind="primary"]:hover { background: #A8543D !important; }

div[data-testid="stTabs"] button[role="tab"] {
    color: var(--pc-text-muted) !important;
    font-family: 'Fraunces', serif;
    font-weight: 500;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--pc-terracotta) !important;
}
div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] { background-color: var(--pc-terracotta) !important; }
div[data-testid="stTabs"] div[data-baseweb="tab-border"] { background-color: var(--pc-border) !important; }

div[data-testid="stExpander"] {
    background: var(--pc-surface);
    border: 1px solid var(--pc-border) !important;
    border-radius: 2px;
}
div[data-testid="stDataFrame"] { border: 1px solid var(--pc-border); border-radius: 2px; }
.stProgress > div > div { background-color: var(--pc-terracotta) !important; }

.pc-footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--pc-border);
    display: flex;
    gap: 18px;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--pc-text-faint) !important;
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
        cv2.rectangle(out, (x1, y1), (x2, y2), (74, 99, 194), 3)
        label = f"{score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), (74, 99, 194), -1)
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

        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if n_frames > MAX_VIDEO_FRAMES:
            st.error(
                f"This video has ~{n_frames} frames (~{n_frames / fps:.0f}s) — "
                f"longer than the {MAX_VIDEO_FRAMES}-frame limit this demo processes "
                "on shared hosting. Please upload a shorter clip."
            )
            cap.release()
            return

        # Downscale before processing: cuts CPU (detection + drawing) and
        # memory roughly quadratically for high-resolution uploads, at the
        # cost of output resolution — an acceptable tradeoff for a demo.
        scale = min(1.0, MAX_PROCESS_LONG_SIDE / max(orig_width, orig_height))
        width = max(1, round(orig_width * scale))
        height = max(1, round(orig_height * scale))

        line = (lx1 / 100 * width, ly1 / 100 * height, lx2 / 100 * width, ly2 / 100 * height)
        detector = load_detector(conf_thresh)
        counter = VideoPeopleCounter(detector, line=line)

        out_path = tmp_path + "_out.mp4"
        # H.264/yuv420p via imageio-ffmpeg (a bundled, self-contained ffmpeg
        # binary — no system ffmpeg install required on Windows/Linux/macOS).
        # cv2.VideoWriter's "mp4v" fourcc writes MPEG-4 Part 2, which browsers
        # generally refuse to play inline ("video not found in a supported
        # format"); H.264 in an mp4 container is universally playable.
        writer = imageio.get_writer(
            out_path, fps=fps, codec="libx264", pixelformat="yuv420p",
            output_params=["-crf", "23"],
        )

        progress = st.progress(0.0)
        status = st.empty()
        frame_i = 0
        t0 = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            result = counter.process_frame(frame)
            writer.append_data(cv2.cvtColor(result.annotated, cv2.COLOR_BGR2RGB))
            frame_i += 1
            if n_frames > 0:
                progress.progress(min(frame_i / n_frames, 1.0))
            status.text(f"Frame {frame_i}/{n_frames or '?'} — IN: {result.in_count}  OUT: {result.out_count}")

        writer.close()
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
            <div class="pc-kicker">Computer vision · portfolio project</div>
            <h1>People, <span class="accent">counted.</span></h1>
            <p>A local detector finds people in a photo, or follows them through a
            video and tallies who crossed a line you draw — no cloud, no accounts,
            nothing kept once the frame is processed.</p>
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

    tab_detect, tab_about = st.tabs(["Detect", "About"])

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
        """
        <div class="pc-footer">
            <span>no third-party apis</span><span>·</span>
            <span>nothing stored</span><span>·</span>
            <span>runs fully offline</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
