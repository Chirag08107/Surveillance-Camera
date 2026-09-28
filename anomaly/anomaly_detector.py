"""
STEP 11 - ANOMALY DETECTION

Two complementary signals, combined:

1. Classifier-based: behaviors the model itself is trained to flag
   (e.g. "Falling", "Loitering") once you've trained it with those
   classes in your dataset - see model/train.py.

2. Statistical: an unsupervised check that needs no labels at all.
   Per track, it keeps a running mean/std of recent motion features
   (velocity, acceleration, pose_movement) and flags frames that are
   an outlier (z-score beyond a threshold) relative to that track's
   own recent normal behavior, or that stay inside a restricted zone
   past a dwell-time limit (loitering, even if the classifier has
   never seen a "Loitering" label).

This lets anomaly detection work from day one (statistical path) and
get sharper once you have a trained classifier with anomaly classes.
"""

from collections import deque, defaultdict
import numpy as np

ANOMALOUS_CLASS_NAMES = {"Falling", "Loitering", "Running", "Fighting"}


class AnomalyDetector:
    def __init__(
        self,
        window: int = 60,
        z_threshold: float = 3.0,
        loiter_seconds: float = 15.0,
    ):
        self.window = window
        self.z_threshold = z_threshold
        self.loiter_seconds = loiter_seconds

        self._velocity_hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
        self._zone_enter_frame: dict[int, int] = {}
        self._frame_count: dict[int, int] = defaultdict(int)

    def _statistical_check(self, track_id: int, velocity: float, acceleration: float) -> bool:
        hist = self._velocity_hist[track_id]
        hist.append(velocity)

        if len(hist) < max(10, self.window // 4):
            return False  # not enough history yet to judge "normal" for this track

        arr = np.array(hist)
        mean, std = arr.mean(), arr.std()

        if std == 0:
            return False

        z = abs(velocity - mean) / std
        return z > self.z_threshold

    def _loiter_check(self, track_id: int, in_restricted_zone: bool, fps: float) -> bool:
        self._frame_count[track_id] += 1

        if not in_restricted_zone:
            self._zone_enter_frame.pop(track_id, None)
            return False

        if track_id not in self._zone_enter_frame:
            self._zone_enter_frame[track_id] = self._frame_count[track_id]
            return False

        frames_inside = self._frame_count[track_id] - self._zone_enter_frame[track_id]
        seconds_inside = frames_inside / fps if fps > 0 else 0

        return seconds_inside >= self.loiter_seconds

    def evaluate(
        self,
        track_id: int,
        velocity: float,
        acceleration: float,
        in_restricted_zone: bool,
        fps: float,
        classifier_label: str | None = None,
        classifier_confidence: float = 0.0,
    ) -> dict:
        """
        Returns:
            {
                "is_anomalous": bool,
                "reasons": [str, ...],
            }
        """

        reasons = []

        if self._statistical_check(track_id, velocity, acceleration):
            reasons.append("motion_outlier")

        if self._loiter_check(track_id, in_restricted_zone, fps):
            reasons.append("loitering_in_restricted_zone")

        if classifier_label in ANOMALOUS_CLASS_NAMES and classifier_confidence >= 0.6:
            reasons.append(f"classifier:{classifier_label}")

        return {"is_anomalous": len(reasons) > 0, "reasons": reasons}

    def drop(self, track_id: int):
        self._velocity_hist.pop(track_id, None)
        self._zone_enter_frame.pop(track_id, None)
        self._frame_count.pop(track_id, None)
