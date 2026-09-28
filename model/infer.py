"""
Loads a trained checkpoint (model/checkpoints/) and classifies a single
30-frame sequence coming from features/sequence_builder.py at runtime.

If no checkpoint exists yet (you haven't trained on your own labeled
data), BehaviorClassifier.predict() falls back to the rule-based
Standing/Sitting label already computed by behaviour.py, and reports
`model_loaded = False` so the caller knows it's a fallback and not a
learned prediction.
"""

import json
import os

import numpy as np
import torch

from model.classifier import BehaviorLSTM
from features.feature_vector import FEATURE_DIM, normalize_feature_vector


class BehaviorClassifier:
    def __init__(self, checkpoint_dir: str = "model/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.model_loaded = False
        self.model = None
        self.idx_to_label = {}
        self.norm_stats = None

        self._try_load(checkpoint_dir)

    def _try_load(self, checkpoint_dir: str):
        weights_path = os.path.join(checkpoint_dir, "behavior_lstm.pt")
        labels_path = os.path.join(checkpoint_dir, "labels.json")
        stats_path = os.path.join(checkpoint_dir, "norm_stats.npz")

        if not (os.path.exists(weights_path) and os.path.exists(labels_path) and os.path.exists(stats_path)):
            return  # no trained model yet - stays in fallback mode

        with open(labels_path) as f:
            self.idx_to_label = {int(k): v for k, v in json.load(f).items()}

        stats = np.load(stats_path)
        self.norm_stats = {"mean": stats["mean"], "std": stats["std"]}

        self.model = BehaviorLSTM(input_dim=FEATURE_DIM, num_classes=len(self.idx_to_label))
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()
        self.model_loaded = True

    def predict(self, sequence: np.ndarray, fallback_label: str = "Unknown"):
        """
        sequence: (30, FEATURE_DIM) from SequenceBuilder.get_sequence()
        Returns (label: str, confidence: float, model_loaded: bool)
        """

        if not self.model_loaded:
            return fallback_label, 0.0, False

        normalized = np.stack(
            [normalize_feature_vector(frame, self.norm_stats) for frame in sequence],
            axis=0,
        )

        x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)  # (1, 30, F)
        probs = self.model.predict_proba(x).squeeze(0)

        idx = int(torch.argmax(probs).item())
        confidence = float(probs[idx].item())
        label = self.idx_to_label.get(idx, fallback_label)

        return label, confidence, True
