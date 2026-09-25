"""
Genuine accuracy/speed evaluation of the bundled YOLOX-nano person detector,
against a real labeled sample (the 128-image "coco128" subset of COCO
train2017, https://github.com/ultralytics/assets — YOLO-format labels,
class 0 = person, same class ordering the model uses).

This produces real, reproducible numbers from actual detector output
matched against ground-truth boxes — not made-up benchmark figures.
Run: python evaluate.py [--data-dir coco128_raw/coco128] [--conf 0.35] [--iou 0.5]

Output: precision / recall / F1 for the person class at IoU>=0.5, plus
mean per-image inference latency (CPU) and derived FPS.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

from detector import PersonDetector

PERSON_CLASS_YOLO_ID = 0  # matches detector.COCO_PERSON_CLASS_ID


def load_yolo_person_boxes(label_path: str, img_w: int, img_h: int) -> np.ndarray:
    """Read a YOLO-format label file, return only person-class boxes as
    (N,4) xyxy pixel coordinates."""
    if not os.path.exists(label_path):
        return np.zeros((0, 4), dtype=np.float32)
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls != PERSON_CLASS_YOLO_ID:
                continue
            cx, cy, w, h = (float(x) for x in parts[1:5])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two (N,4)/(M,4) xyxy box arrays -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ious = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        xx1 = np.maximum(a[i, 0], b[:, 0])
        yy1 = np.maximum(a[i, 1], b[:, 1])
        xx2 = np.minimum(a[i, 2], b[:, 2])
        yy2 = np.minimum(a[i, 3], b[:, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = area_a[i] + area_b - inter
        ious[i] = inter / np.maximum(union, 1e-9)
    return ious


def match_greedy(pred_boxes: np.ndarray, pred_scores: np.ndarray, gt_boxes: np.ndarray, iou_thresh: float):
    """Greedy matching, highest-confidence prediction first. Returns (tp, fp, fn)."""
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    order = np.argsort(-pred_scores)
    ious = iou_matrix(pred_boxes, gt_boxes)
    gt_used = np.zeros(len(gt_boxes), dtype=bool)
    tp = 0
    fp = 0
    for i in order:
        if len(gt_boxes) == 0:
            fp += 1
            continue
        j = np.argmax(ious[i])
        if ious[i, j] >= iou_thresh and not gt_used[j]:
            tp += 1
            gt_used[j] = True
        else:
            fp += 1
    fn = int((~gt_used).sum())
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "coco128_raw", "coco128"))
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "yolox_nano.onnx"))
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "eval_results.json"))
    args = ap.parse_args()

    img_dir = os.path.join(args.data_dir, "images", "train2017")
    label_dir = os.path.join(args.data_dir, "labels", "train2017")
    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    # only evaluate on images that actually have >=1 ground-truth person box
    # (the rest have zero person GT, which is a valid but uninteresting case
    # for precision/recall — we report those separately below as a sanity check)
    detector = PersonDetector(args.model, conf_thresh=args.conf)

    total_tp = total_fp = total_fn = 0
    latencies_ms = []
    n_images_with_person = 0
    n_images_without_person_and_clean = 0  # correctly detected zero people
    n_images_without_person_and_fp = 0

    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, stem + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_boxes = load_yolo_person_boxes(label_path, w, h)

        t0 = time.time()
        preds = detector.detect(img)
        latencies_ms.append((time.time() - t0) * 1000)

        pred_boxes = preds[:, :4] if len(preds) else np.zeros((0, 4), dtype=np.float32)
        pred_scores = preds[:, 4] if len(preds) else np.zeros((0,), dtype=np.float32)

        if len(gt_boxes) == 0:
            if len(pred_boxes) == 0:
                n_images_without_person_and_clean += 1
            else:
                n_images_without_person_and_fp += 1
            continue

        n_images_with_person += 1
        tp, fp, fn = match_greedy(pred_boxes, pred_scores, gt_boxes, args.iou)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    fps = 1000.0 / mean_latency if mean_latency else 0.0

    results = {
        "dataset": "coco128 (ultralytics/assets), train2017 subset, class 0 = person",
        "n_images_total": len(image_paths),
        "n_images_with_person_gt": n_images_with_person,
        "conf_thresh": args.conf,
        "iou_match_thresh": args.iou,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "images_without_person_gt_correctly_empty": n_images_without_person_and_clean,
        "images_without_person_gt_with_false_positive": n_images_without_person_and_fp,
        "mean_inference_latency_ms_cpu": round(mean_latency, 1),
        "approx_fps_cpu": round(fps, 1),
    }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
