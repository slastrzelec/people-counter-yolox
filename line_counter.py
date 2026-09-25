"""
Pure line-crossing counting logic — no CV/ML dependency, fully unit-testable
with synthetic track positions.

A "line" is defined by two points (x1, y1) -> (x2, y2). For each tracked
person (identified by a track_id), we watch which side of the line their
centroid is on, frame to frame. A crossing is counted the frame the side
flips, and its direction (IN vs OUT) is determined by the sign of the flip,
so a person who crosses back and forth is counted correctly each time
(no double-counting a single crossing, no missing a genuine back-and-forth).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LineCounter:
    x1: float
    y1: float
    x2: float
    y2: float
    _last_side: dict[int, float] = field(default_factory=dict)
    in_count: int = 0
    out_count: int = 0

    def _side(self, px: float, py: float) -> float:
        """Signed side of point (px, py) relative to the line, via the 2D
        cross product of (line vector) x (point - line start). Sign flips
        exactly when the point crosses the (infinite) line.
        """
        return (self.x2 - self.x1) * (py - self.y1) - (self.y2 - self.y1) * (px - self.x1)

    def update(self, track_id: int, centroid_x: float, centroid_y: float) -> str | None:
        """Feed one track's current centroid for the current frame.

        Returns "in", "out", or None (no crossing this update). "in" means
        the point moved from the negative side to the positive side of the
        line (as defined by (x1,y1)->(x2,y2) direction); "out" is the
        reverse. Which physical direction that corresponds to depends on
        how the line endpoints were drawn — the app exposes this via the
        line's orientation in the UI, not baked in here.
        """
        side = self._side(centroid_x, centroid_y)
        prev = self._last_side.get(track_id)
        self._last_side[track_id] = side

        if prev is None or prev == 0 or side == 0:
            return None  # first sighting of this track, or sitting exactly on the line

        if prev < 0 and side > 0:
            self.in_count += 1
            return "in"
        if prev > 0 and side < 0:
            self.out_count += 1
            return "out"
        return None

    def forget(self, track_id: int) -> None:
        """Drop a track's remembered side once it's no longer being tracked
        (e.g. it left the frame), so a *new* track later reusing the same id
        doesn't inherit a stale crossing state.
        """
        self._last_side.pop(track_id, None)

    @property
    def net_count(self) -> int:
        return self.in_count - self.out_count
