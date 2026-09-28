"""
Trains BehaviorLSTM on the dataset produced by dataset_generator.py.

Usage:
    python -m model.train --data dataset/behavior_dataset.npz --epochs 40

Outputs:
    model/checkpoints/behavior_lstm.pt   (weights)
    model/checkpoints/labels.json        (idx -> label name)
    model/checkpoints/norm_stats.npz     (mean/std used at train time)
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset.dataset_loader import load_dataset
from model.classifier import BehaviorLSTM
from features.feature_vector import FEATURE_DIM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/behavior_dataset.npz")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", default="model/checkpoints")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bundle = load_dataset(args.data)

    train_ds = TensorDataset(
        torch.tensor(bundle.X_train, dtype=torch.float32),
        torch.tensor(bundle.y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(bundle.X_val, dtype=torch.float32),
        torch.tensor(bundle.y_val, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BehaviorLSTM(input_dim=FEATURE_DIM, num_classes=bundle.num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        train_loss = total_loss / len(train_ds)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_acc={val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "behavior_lstm.pt"))

    with open(os.path.join(args.out_dir, "labels.json"), "w") as f:
        json.dump(bundle.idx_to_label, f, indent=2)

    np.savez(
        os.path.join(args.out_dir, "norm_stats.npz"),
        mean=bundle.norm_stats["mean"],
        std=bundle.norm_stats["std"],
    )

    print(f"Best val accuracy: {best_val_acc:.3f}")
    print(f"Saved checkpoint + labels + norm stats to {args.out_dir}/")


if __name__ == "__main__":
    main()
