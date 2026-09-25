"""
Ties PersonDetector + ByteTrack + LineCounter together for video mode.
No Streamlit dependency — importable and testable on its own (tracking
logic is exercised via a fake/injectable detector in tests, so no real
video file or model weights are needed to test the wiring).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from line_counter import LineCounter


@dataclass
class FrameResult:
    annotated: np.ndarray
    n_tracks: int
    in_count: int
    out_count: int
    crossing_event: str | None  # "in" / "out" / None for this frame


class VideoPeopleCounter:
    """Stateful per-video processor: detect -> track -> count line crossings.

    `detector` only needs a `.detect(bgr_image) -> (N,5) array` method, so
    tests can inject a stub instead of loading real ONNX weights.
    """

    def __init__(self, detector, line: tuple[float, float, float, float]):
        self.detector = detector
        # `minimum_consecutive_frames=2` (the library default) means a track
        # is only reported once it's been matched for 2 frames in a row —
        # this filters out single-frame detector noise. Tentative,
        # not-yet-confirmed tracks come back with tracker_id == -1 and are
        # skipped below, so they never reach the line counter.
        self.tracker = ByteTrackTracker(minimum_consecutive_frames=2)
        self.line_counter = LineCounter(*line)
        self._active_ids: set[int] = set()

    def process_frame(self, bgr_frame: np.ndarray) -> FrameResult:
        boxes = self.detector.detect(bgr_frame)  # (N,5): x1,y1,x2,y2,score
        if len(boxes) == 0:
            detections = sv.Detections.empty()
        else:
            detections = sv.Detections(
                xyxy=boxes[:, :4].astype(np.float32),
                confidence=boxes[:, 4].astype(np.float32),
                class_id=np.zeros(len(boxes), dtype=int),
            )
        tracked = self.tracker.update(detections)

        current_ids = set()
        crossing_event = None
        for xyxy, track_id in zip(tracked.xyxy, tracked.tracker_id):
            track_id = int(track_id)
            if track_id < 0:
                continue  # not yet confirmed as a stable track
            current_ids.add(track_id)
            cx = float((xyxy[0] + xyxy[2]) / 2)
            cy = float((xyxy[1] + xyxy[3]) / 2)
            event = self.line_counter.update(track_id, cx, cy)
            if event is not None:
                crossing_event = event  # last crossing this frame wins for the on-screen flash

        # tracks that vanished this frame: forget them so a reused id later
        # doesn't inherit stale crossing state
        for gone_id in self._active_ids - current_ids:
            self.line_counter.forget(gone_id)
        self._active_ids = current_ids

        annotated = self._draw(bgr_frame.copy(), tracked)
        return FrameResult(
            annotated=annotated,
            n_tracks=len(current_ids),
            in_count=self.line_counter.in_count,
            out_count=self.line_counter.out_count,
            crossing_event=crossing_event,
        )

    def _draw(self, frame: np.ndarray, tracked: "sv.Detections") -> np.ndarray:
        lc = self.line_counter
        cv2.line(frame, (int(lc.x1), int(lc.y1)), (int(lc.x2), int(lc.y2)), (255, 0, 255), 3)

        for xyxy, track_id in zip(tracked.xyxy, tracked.tracker_id):
            if int(track_id) < 0:
                continue
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"#{int(track_id)}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"IN: {lc.in_count}  OUT: {lc.out_count}  NET: {lc.net_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return frame
