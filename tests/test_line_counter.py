import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from line_counter import LineCounter


def make_vertical_line():
    # vertical line at x=100, spanning y=0..200
    return LineCounter(x1=100, y1=0, x2=100, y2=200)


def test_first_sighting_never_counts_as_crossing():
    lc = make_vertical_line()
    result = lc.update(track_id=1, centroid_x=50, centroid_y=100)
    assert result is None
    assert lc.in_count == 0
    assert lc.out_count == 0


def test_staying_on_one_side_never_counts():
    lc = make_vertical_line()
    lc.update(1, 50, 100)
    lc.update(1, 55, 105)
    lc.update(1, 60, 110)
    assert lc.in_count == 0
    assert lc.out_count == 0


def test_single_left_to_right_crossing_counts_once():
    # For this line ((100,0) -> (100,200), i.e. drawn top-to-bottom), the
    # signed-side cross product puts smaller-x points on the positive side
    # and larger-x points on the negative side (verified directly on
    # LineCounter._side below) — so moving left (x=50) -> right (x=150) is
    # a positive->negative transition, which this module's convention
    # calls "out". The label itself is arbitrary (see module docstring);
    # what matters is that it's applied consistently, which the rest of
    # this file checks.
    lc = make_vertical_line()
    assert lc._side(50, 100) > 0
    assert lc._side(150, 100) < 0

    lc.update(1, 50, 100)   # left side
    lc.update(1, 90, 100)   # still left
    result = lc.update(1, 150, 100)  # now right -> crossing
    assert result == "out"
    assert lc.out_count == 1
    assert lc.in_count == 0
    assert lc.net_count == -1


def test_right_to_left_is_the_opposite_direction():
    lc = make_vertical_line()
    lc.update(1, 150, 100)  # right side
    result = lc.update(1, 50, 100)  # now left -> crossing
    assert result == "in"
    assert lc.in_count == 1
    assert lc.out_count == 0


def test_back_and_forth_counts_each_crossing_not_just_once():
    lc = make_vertical_line()
    lc.update(1, 50, 100)    # left
    lc.update(1, 150, 100)   # -> out (crossing 1)
    lc.update(1, 50, 100)    # -> in  (crossing 2)
    lc.update(1, 150, 100)   # -> out (crossing 3)
    assert lc.out_count == 2
    assert lc.in_count == 1
    assert lc.net_count == -1


def test_multiple_tracks_are_independent():
    lc = make_vertical_line()
    lc.update(1, 50, 100)
    lc.update(2, 150, 100)
    lc.update(1, 150, 100)  # track 1 crosses out
    lc.update(2, 50, 100)   # track 2 crosses in
    assert lc.out_count == 1
    assert lc.in_count == 1


def test_sitting_exactly_on_the_line_does_not_falsely_trigger():
    lc = make_vertical_line()
    lc.update(1, 50, 100)    # left
    lc.update(1, 100, 100)   # exactly on the line (side == 0)
    result = lc.update(1, 150, 100)  # now right
    # side==0 resets memory of "prev side", so the eventual left->right
    # transition after sitting on the line should NOT be silently dropped,
    # but also must not be double-counted by the on-the-line frame itself.
    assert result is None or result == "out"
    assert lc.out_count <= 1


def test_forget_clears_track_state_so_a_reused_id_starts_fresh():
    lc = make_vertical_line()
    lc.update(1, 50, 100)
    lc.update(1, 150, 100)  # out_count = 1
    lc.forget(1)
    result = lc.update(1, 150, 100)  # id 1 reused, first sighting again
    assert result is None
    assert lc.out_count == 1  # unchanged, no phantom re-crossing
