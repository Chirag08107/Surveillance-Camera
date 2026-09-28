"""
Loads the .npz dataset produced by dataset_generator.py, computes
normalization stats, encodes string labels to ints, and splits
train/val.
"""

import numpy as np
from sklearn.model_selection import train_test_split


class DatasetBundle:
    def __init__(self, X_train, y_train, X_val, y_val, label_to_idx, norm_stats):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.label_to_idx = label_to_idx
        self.idx_to_label = {v: k for k, v in label_to_idx.items()}
        self.norm_stats = norm_stats

    @property
    def num_classes(self):
        return len(self.label_to_idx)


def load_dataset(npz_path: str, val_split: float = 0.2, seed: int = 42) -> DatasetBundle:
    data = np.load(npz_path, allow_pickle=True)
    X, y_raw = data["X"], data["y"]

    labels = sorted(set(y_raw.tolist()))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    y = np.array([label_to_idx[label] for label in y_raw], dtype=np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y if len(labels) > 1 else None
    )

    # Normalization stats from TRAIN split only (avoid leakage)
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    norm_stats = {"mean": mean, "std": std}

    X_train = (X_train - mean) / np.where(std == 0, 1.0, std)
    X_val = (X_val - mean) / np.where(std == 0, 1.0, std)

    return DatasetBundle(X_train, y_train, X_val, y_val, label_to_idx, norm_stats)
