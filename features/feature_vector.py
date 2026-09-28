"""
STEP 8 - FEATURE VECTOR

Turns everything behaviour.py already computes per track/per frame
(normalized pose, knee angle, movement stats, velocity, acceleration,
zone status, direction) into a single fixed-length numeric vector that
a neural network can consume.

This module does NOT change behaviour.py / pose.py / main.py. It reads
the same values those files already produce and reshapes them.
"""

import numpy as np

# Order matters - this defines the feature vector schema everywhere
# (dataset generation, training, inference must all agree on this list).
FEATURE_NAMES = [
    "left_knee_angle",
    "right_knee_angle",
    "average_knee_angle",
    "pose_movement",
    "average_movement",
    "maximum_movement",
    "total_movement",
    "center_velocity",
    "center_acceleration",
    "direction_x",       # -1 left, 0 none, 1 right
    "direction_y",       # -1 up, 0 none, 1 down
    "in_restricted_zone",  # 0 / 1
    "pose_confidence_mean",
]

FEATURE_DIM = len(FEATURE_NAMES)


def direction_to_xy(direction: str):
    mapping = {
        "Left": (-1, 0),
        "Right": (1, 0),
        "Up": (0, -1),
        "Down": (0, 1),
        "Stationary": (0, 0),
    }
    return mapping.get(direction, (0, 0))


def build_feature_vector(
    left_knee_angle: float,
    right_knee_angle: float,
    average_knee_angle: float,
    pose_movement: float,
    average_movement: float,
    maximum_movement: float,
    total_movement: float,
    center_velocity: float,
    center_acceleration: float,
    direction: str,
    in_restricted_zone: bool,
    pose_confidence_mean: float,
) -> np.ndarray:
    """
    Assembles one frame's worth of measurements into a fixed-length
    float32 vector, in the exact order of FEATURE_NAMES.
    """

    dx, dy = direction_to_xy(direction)

    vector = np.array(
        [
            left_knee_angle,
            right_knee_angle,
            average_knee_angle,
            pose_movement,
            average_movement,
            maximum_movement,
            total_movement,
            center_velocity,
            center_acceleration,
            dx,
            dy,
            1.0 if in_restricted_zone else 0.0,
            pose_confidence_mean,
        ],
        dtype=np.float32,
    )

    return vector


def normalize_feature_vector(vector: np.ndarray, stats: dict) -> np.ndarray:
    """
    Z-score normalization using precomputed per-feature mean/std.
    `stats` = {"mean": np.ndarray(FEATURE_DIM,), "std": np.ndarray(FEATURE_DIM,)}
    Produced by dataset/dataset_loader.py from the training set and must be
    reused (not recomputed) at inference time.
    """

    mean = stats["mean"]
    std = np.where(stats["std"] == 0, 1.0, stats["std"])

    return (vector - mean) / std
