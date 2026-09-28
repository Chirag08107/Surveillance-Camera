"""
STEP 10 - NEURAL NETWORK BEHAVIOR CLASSIFIER (architecture)

A small bidirectional LSTM over the 30-frame feature sequences built in
features/sequence_builder.py. This is intentionally light (few params)
because the "signal" is already highly processed (angles, velocities,
movement stats) rather than raw pixels - a big transformer is overkill
and would just overfit a small hand-collected dataset.
"""

import torch
import torch.nn as nn


class BehaviorLSTM(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, (h_n, _) = self.lstm(x)

        # concat last-layer forward + backward hidden states
        last_forward = h_n[-2]
        last_backward = h_n[-1]
        combined = torch.cat([last_forward, last_backward], dim=1)

        logits = self.classifier(combined)
        return logits

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
