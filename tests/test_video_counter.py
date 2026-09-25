import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_counter import VideoPeopleCounter


class FakeDetector:
    """Replays a fixed sequence of (N,5) boxes, one per call — no model needed."""

    def __init__(self, boxes_per_frame):
        self.boxes_per_frame = boxes_per_frame
        self.i = 0

    def detect(self, frame):
        boxes = self.boxes_per_frame[self.i]
        self.i += 1
        return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)


BLANK = np.zeros((200, 300, 3), dtype=np.uint8)


def walking_frames(x_start=40, x_stop=260, step=10, width=40, score=0.9):
    """A single person walking left->right with realistic (overlapping-box)
    motion between frames, so the tracker can actually associate them."""
    return [[[x, 40, x + width, 140, score]] for x in range(x_start, x_stop, step)]


def test_confirmed_track_is_reported_once_matched_two_frames_running():
    detector = FakeDetector(walking_frames())
    counter = VideoPeopleCounter(detector, line=(155, 0, 155, 200))
    # first frame: brand new track, not yet confirmed by the tracker (tracker_id
    # would be -1 internally) -> n_tracks must be 0
    r0 = counter.process_frame(BLANK)
    assert r0.n_tracks == 0
    # second frame: same track matched again -> now confirmed
    r1 = counter.process_frame(BLANK)
    assert r1.n_tracks == 1


def test_walking_across_the_line_counts_exactly_one_crossing():
    detector = FakeDetector(walking_frames())
    counter = VideoPeopleCounter(detector, line=(155, 0, 155, 200))
    results = [counter.process_frame(BLANK) for _ in range(len(detector.boxes_per_frame))]
    total_out = results[-1].out_count
    total_in = results[-1].in_count
    assert total_out + total_in == 1
    events = [r.crossing_event for r in results if r.crossing_event is not None]
    assert len(events) == 1


def test_no_detections_at_all_never_crashes_and_counts_nothing():
    detector = FakeDetector([[] for _ in range(5)])
    counter = VideoPeopleCounter(detector, line=(150, 0, 150, 200))
    for _ in range(5):
        r = counter.process_frame(BLANK)
        assert r.n_tracks == 0
        assert r.in_count == 0
        assert r.out_count == 0
        assert r.crossing_event is None


def test_track_that_leaves_frame_is_forgotten_so_a_reused_id_starts_fresh():
    # person walks across (crosses once), then disappears (empty frames),
    # then a *new* person appears on the far side and walks back across.
    frames = walking_frames(x_start=40, x_stop=260, step=10)
    frames += [[] for _ in range(3)]
    frames += walking_frames(x_start=260, x_stop=40, step=-10)
    detector = FakeDetector(frames)
    counter = VideoPeopleCounter(detector, line=(155, 0, 155, 200))
    results = [counter.process_frame(BLANK) for _ in range(len(frames))]
    # two independent crossings (out, then back in) should both be counted —
    # forget() must have cleared the stale side so the second pass isn't
    # silently dropped as a "no-op" continuation of the first track's state
    assert results[-1].in_count == 1
    assert results[-1].out_count == 1


def test_annotated_frame_has_same_shape_as_input():
    detector = FakeDetector(walking_frames())
    counter = VideoPeopleCounter(detector, line=(155, 0, 155, 200))
    r = counter.process_frame(BLANK)
    assert r.annotated.shape == BLANK.shape
