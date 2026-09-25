"""
YOLOX-nano ONNX person detector.

No Streamlit/UI dependency here — pure detection logic, unit-testable
in isolation. Model: YOLOX-nano (Apache 2.0, Megvii-BaseDetection/YOLOX),
COCO-pretrained (80 classes). We only keep class 0 ("person").

Standard YOLOX pre/post-processing (letterbox resize + grid/stride decode),
matching the official YOLOX ONNXRuntime demo.
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort

COCO_PERSON_CLASS_ID = 0
INPUT_SIZE = (416, 416)  # (height, width) — matches yolox_nano.onnx export
_STRIDES = (8, 16, 32)


def preprocess(img: np.ndarray, input_size: tuple[int, int] = INPUT_SIZE):
    """Letterbox-resize + pad an HWC BGR uint8 image to the model's input size.

    Returns (chw_float32_array, scale) where `scale` is needed to map
    predicted boxes back to the original image size.
    """
    if len(img.shape) == 3:
        padded_img = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    else:
        padded_img = np.full(input_size, 114, dtype=np.uint8)

    import cv2

    scale = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * scale), int(img.shape[0] * scale)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: resized_img.shape[0], : resized_img.shape[1]] = resized_img

    padded_img = padded_img.transpose(2, 0, 1)  # HWC -> CHW
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, scale


def _make_grids_and_strides(input_size: tuple[int, int], strides=_STRIDES):
    grids = []
    expanded_strides = []
    hsizes = [input_size[0] // s for s in strides]
    wsizes = [input_size[1] // s for s in strides]
    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))
    return np.concatenate(grids, 1), np.concatenate(expanded_strides, 1)


def decode_outputs(outputs: np.ndarray, input_size: tuple[int, int] = INPUT_SIZE) -> np.ndarray:
    """Decode raw YOLOX head output (grid offsets + log-scale wh) into
    absolute (cx, cy, w, h) in the *model input* pixel space.
    """
    grids, expanded_strides = _make_grids_and_strides(input_size)
    outputs = outputs.copy()
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs


def postprocess(
    decoded: np.ndarray,
    scale: float,
    conf_thresh: float = 0.35,
    nms_thresh: float = 0.45,
    class_id: int = COCO_PERSON_CLASS_ID,
) -> np.ndarray:
    """Filter to one class, apply confidence threshold + NMS.

    Returns an (N, 5) array of [x1, y1, x2, y2, score] in *original image*
    pixel coordinates.
    """
    import cv2

    preds = decoded[0]  # (num_anchors, 85)
    boxes = preds[:, :4]
    obj_conf = preds[:, 4]
    class_conf = preds[:, 5 + class_id]
    scores = obj_conf * class_conf

    keep_mask = scores >= conf_thresh
    if not np.any(keep_mask):
        return np.zeros((0, 5), dtype=np.float32)

    boxes = boxes[keep_mask]
    scores = scores[keep_mask]

    # cx,cy,w,h -> x1,y1,x2,y2 (still in letterboxed/model-input space)
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    idxs = cv2.dnn.NMSBoxes(
        bboxes=[[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)],
        scores=[float(s) for s in scores],
        score_threshold=conf_thresh,
        nms_threshold=nms_thresh,
    )
    if len(idxs) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    idxs = np.array(idxs).reshape(-1)

    out = np.stack([x1[idxs], y1[idxs], x2[idxs], y2[idxs], scores[idxs]], axis=1)
    out[:, :4] /= scale  # back to original image pixel space
    return out.astype(np.float32)


class PersonDetector:
    """Loads yolox_nano.onnx once; call .detect(bgr_image) per frame/image."""

    def __init__(self, model_path: str, conf_thresh: float = 0.35, nms_thresh: float = 0.45):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    def detect(self, bgr_image: np.ndarray) -> np.ndarray:
        """Returns (N, 5) array of [x1, y1, x2, y2, score] for detected persons."""
        inp, scale = preprocess(bgr_image, INPUT_SIZE)
        raw = self.session.run(None, {self.input_name: inp[None, :, :, :]})[0]
        decoded = decode_outputs(raw, INPUT_SIZE)
        return postprocess(decoded, scale, self.conf_thresh, self.nms_thresh)
