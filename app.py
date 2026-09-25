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

st.set_page_config(page_title="People Counter", layout="wide")


@st.cache_resource
def load_detector(conf_thresh: float) -> PersonDetector:
    return PersonDetector(MODEL_PATH, conf_thresh=conf_thresh)


def draw_boxes(bgr_image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    out = bgr_image.copy()
    for x1, y1, x2, y2, score in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{score:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(pil_img, use_container_width=True)
    with col2:
        st.subheader(f"Detected: {len(boxes)} people")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.metric("People detected", len(boxes))
    st.caption(f"Inference time: {elapsed_ms:.0f} ms (CPU)")

    if len(boxes) > 0:
        st.dataframe(
            {
                "x1": boxes[:, 0].round(1),
                "y1": boxes[:, 1].round(1),
                "x2": boxes[:, 2].round(1),
                "y2": boxes[:, 3].round(1),
                "confidence": boxes[:, 4].round(3),
            },
            use_container_width=True,
        )

    _, buf = cv2.imencode(".png", annotated)
    st.download_button("Download annotated image", bytes(buf), file_name="annotated.png", mime="image/png")


def run_video_mode(conf_thresh: float) -> None:
    st.write("Draw the counting line as two points on the first frame (as fractions of frame width/height).")
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

    if not st.button("Process video"):
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
            st.video(f.read())
            f.seek(0)
            st.download_button("Download annotated video", f.read(), file_name="annotated.mp4", mime="video/mp4")

    finally:
        for p in (tmp_path, out_path):
            if p and os.path.exists(p):
                os.remove(p)


def main() -> None:
    st.title("People Counter")
    st.caption(
        "Local person detection (YOLOX-nano, ONNX Runtime, CPU) — nothing you upload leaves this app. "
        "No accounts, no third-party APIs, no stored copies of your files."
    )

    mode = st.radio("Mode", ["Image", "Video"], horizontal=True)
    conf_thresh = st.sidebar.slider("Detection confidence threshold", 0.1, 0.9, 0.35, 0.05)
    st.sidebar.caption("Lower = more detections (more false positives). Higher = fewer, more confident detections.")

    if mode == "Image":
        run_image_mode(conf_thresh)
    else:
        run_video_mode(conf_thresh)


if __name__ == "__main__":
    main()
