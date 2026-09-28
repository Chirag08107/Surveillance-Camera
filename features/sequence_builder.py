"""
STEP 8 - TEMPORAL SEQUENCE

Buffers each track's per-frame feature vectors into a rolling window
(default 30 frames == the same window size main.py/behaviour.py already
use for pose_history/center_history) so the classifier in model/ sees
behavior over time, not a single frame.
"""

from collections import deque
import numpy as np

from features.feature_vector import FEATURE_DIM


class SequenceBuilder:
    """
    One rolling window of feature vectors per track_id.
    Call `push(track_id, vector)` every frame, and `get_sequence(track_id)`
    whenever you want the current window (e.g. to feed the classifier).
    """

    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.buffers: dict[int, deque] = {}

    def push(self, track_id: int, vector: np.ndarray) -> None:
        if track_id not in self.buffers:
            self.buffers[track_id] = deque(maxlen=self.sequence_length)

        self.buffers[track_id].append(vector)

    def is_ready(self, track_id: int) -> bool:
        """True once we have a full window for this track."""
        return (
            track_id in self.buffers
            and len(self.buffers[track_id]) == self.sequence_length
        )

    def get_sequence(self, track_id: int) -> np.ndarray | None:
        """
        Returns shape (sequence_length, FEATURE_DIM) once full,
        or a zero-padded array (padded at the start) if the track is newer
        than sequence_length frames - lets the pipeline start scoring early
        instead of waiting the full 30 frames.
        """

        if track_id not in self.buffers or len(self.buffers[track_id]) == 0:
            return None

        window = list(self.buffers[track_id])
        pad_len = self.sequence_length - len(window)

        if pad_len > 0:
            pad = [np.zeros(FEATURE_DIM, dtype=np.float32) for _ in range(pad_len)]
            window = pad + window

        return np.stack(window, axis=0)

    def drop(self, track_id: int) -> None:
        """Call when a track disappears (person left frame) to free memory."""
        self.buffers.pop(track_id, None)

    def active_track_ids(self):
        return list(self.buffers.keys())
