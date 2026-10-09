"""
Loads the .npz dataset produced by dataset_generator.py, computes
normalization stats, maps labels to binary integers:
    0 = real
    1 = anomaly
and safely splits into train/validation sets without crashing on small datasets.
"""

import numpy as np
from sklearn.model_selection import train_test_split

CLASS_NAMES = ["real", "anomaly"]
LABEL_TO_IDX = {"real": 0, "anomaly": 1}
IDX_TO_LABEL = {0: "real", 1: "anomaly"}


class DatasetBundle:
    def __init__(self, X_train, y_train, X_val, y_val, label_to_idx, norm_stats, labels=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.label_to_idx = label_to_idx
        self.idx_to_label = {v: k for k, v in label_to_idx.items()}
        self.labels = labels if labels is not None else CLASS_NAMES
        self.norm_stats = norm_stats

    @property
    def num_classes(self):
        return len(self.label_to_idx)


def load_dataset(npz_path: str, val_split: float = 0.2, seed: int = 42) -> DatasetBundle:
    """
    Loads behavior_dataset.npz, validates shapes, handles binary label encoding,
    and performs a robust train/val split that does not crash on small datasets.
    """
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y_raw = data["y"]

    # Canonical binary encoding: 0 = real, 1 = anomaly
    if np.issubdtype(y_raw.dtype, np.integer):
        y = y_raw.astype(np.int64)
    else:
        # String labels fallback mapping
        def map_label(lbl):
            s = str(lbl).strip().lower()
            if s in ("1", "anomaly", "anomalous"):
                return 1
            return 0  # "real", "normal", "0"

        y = np.array([map_label(lbl) for lbl in y_raw], dtype=np.int64)

    label_to_idx = {"real": 0, "anomaly": 1}
    idx_to_label = {0: "real", 1: "anomaly"}

    n_samples = len(X)
    if n_samples == 0:
        raise ValueError(f"Dataset in '{npz_path}' is empty.")

    # Train / Validation Split with robust safeguards for small datasets
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_count = min(class_counts) if len(class_counts) > 0 else 0
    can_stratify = len(unique_classes) > 1 and min_count >= 2 and n_samples >= 4

    if n_samples == 1:
        # Extreme edge case: single sequence
        X_train, y_train = X, y
        X_val, y_val = X.copy(), y.copy()
    else:
        # Ensure validation split size is at least 1 sample if n_samples > 1
        effective_val_split = max(val_split, 1.0 / n_samples)
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=effective_val_split,
                random_state=seed,
                stratify=y if can_stratify else None,
            )
        except ValueError:
            # Fallback to unstratified split if stratification fails due to bin size
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=effective_val_split,
                random_state=seed,
                stratify=None,
            )

    # Normalization stats computed exclusively from train split (prevent data leakage)
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std = np.where(std == 0, 1.0, std)  # Prevent division by zero
    norm_stats = {"mean": mean, "std": std}

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        label_to_idx=label_to_idx,
        norm_stats=norm_stats,
        labels=CLASS_NAMES,
    )
