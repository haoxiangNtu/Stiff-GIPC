"""Trajectory reader and interpolator for joint playback demos.

Reads the text-based trajectory format used by Assets/trajectories/*.txt:

    # header comment
    time val1 val2 val3 ...

The first N values (matching num_revolute) are revolute angles in radians;
the remaining values are prismatic distances in metres.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field


@dataclass
class Keyframe:
    time: float
    revolute_angles: list[float] = field(default_factory=list)
    prismatic_dists: list[float] = field(default_factory=list)


class Trajectory:
    """Loads a trajectory text file and provides linear interpolation."""

    def __init__(self, path: str, num_revolute: int, num_prismatic: int):
        self.keyframes: list[Keyframe] = []
        self._times: list[float] = []

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                t = float(parts[0])
                vals = [float(x) for x in parts[1:]]

                kf = Keyframe(time=t)
                kf.revolute_angles = vals[:num_revolute]
                kf.prismatic_dists = vals[num_revolute:num_revolute + num_prismatic]
                self.keyframes.append(kf)
                self._times.append(t)

        if not self.keyframes:
            raise ValueError(f"No keyframes found in {path}")

    @property
    def duration(self) -> float:
        return self._times[-1] - self._times[0]

    def interpolate(self, t: float) -> tuple[list[float], list[float]]:
        """Return (revolute_angles, prismatic_dists) at time t via linear interp."""
        if t <= self._times[0]:
            kf = self.keyframes[0]
            return list(kf.revolute_angles), list(kf.prismatic_dists)
        if t >= self._times[-1]:
            kf = self.keyframes[-1]
            return list(kf.revolute_angles), list(kf.prismatic_dists)

        hi = bisect.bisect_right(self._times, t)
        lo = hi - 1
        kf0, kf1 = self.keyframes[lo], self.keyframes[hi]
        dt = kf1.time - kf0.time + 1e-12
        alpha = max(0.0, min(1.0, (t - kf0.time) / dt))

        rev = [a * (1 - alpha) + b * alpha
               for a, b in zip(kf0.revolute_angles, kf1.revolute_angles)]
        pri = [a * (1 - alpha) + b * alpha
               for a, b in zip(kf0.prismatic_dists, kf1.prismatic_dists)]
        return rev, pri
