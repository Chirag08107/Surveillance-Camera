"""
Trains BehaviorLSTM for binary behavior classification:

    0 = Real
    1 = Anomaly

Handles class imbalance using weighted CrossEntropyLoss.

Usage:
    python -m model.train --data dataset/behavior_dataset.npz --epochs 40

Outputs:
    model/checkpoints/behavior_lstm.pt
    model/checkpoints/labels.json
    model/checkpoints/norm_stats.npz
"""

import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset.dataset_loader import load_dataset
from model.classifier import BehaviorLSTM
from features.feature_vector import FEATURE_DIM


def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy, precision, recall and F1 score
    specifically for the Anomaly class.

    Anomaly = class 1
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_positive = np.sum(
        (y_true == 1) & (y_pred == 1)
    )

    false_positive = np.sum(
        (y_true == 0) & (y_pred == 1)
    )

    false_negative = np.sum(
        (y_true == 1) & (y_pred == 0)
    )

    correct = np.sum(y_true == y_pred)

    total = len(y_true)

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )

    if precision + recall > 0:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    return accuracy, precision, recall, f1


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train Bi-LSTM behavior classifier "
            "(Real vs Anomaly)."
        )
    )

    parser.add_argument(
        "--data",
        default="dataset/behavior_dataset.npz",
        help="Path to .npz dataset",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )

    parser.add_argument(
        "--out-dir",
        default="model/checkpoints",
        help="Output directory for checkpoints",
    )

    args = parser.parse_args()

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    print("=" * 70)

    print(
        "TRAINING BINARY BEHAVIOR CLASSIFIER"
    )

    print(
        "Real (0) vs Anomaly (1)"
    )

    print("=" * 70)

    print("\n" + "!" * 70)

    print(
        "WARNING: Training on a small prototype dataset "
        "(approximately 20 videos)."
    )

    print(
        "High validation accuracy may indicate overfitting "
        "to camera angles or subjects."
    )

    print(
        "Accuracy alone will NOT be used to judge the model."
    )

    print("!" * 70 + "\n")

    # -------------------------------------------------------------
    # LOAD DATASET
    # -------------------------------------------------------------

    bundle = load_dataset(
        args.data
    )

    n_train = len(
        bundle.X_train
    )

    n_val = len(
        bundle.X_val
    )

    print(
        f"Dataset loaded from: {args.data}"
    )

    print(
        f"  Train sequences: {n_train}"
    )

    print(
        f"  Val sequences:   {n_val}"
    )

    # -------------------------------------------------------------
    # CLASS DISTRIBUTION
    # -------------------------------------------------------------

    train_real = int(
        np.sum(bundle.y_train == 0)
    )

    train_anomaly = int(
        np.sum(bundle.y_train == 1)
    )

    val_real = int(
        np.sum(bundle.y_val == 0)
    )

    val_anomaly = int(
        np.sum(bundle.y_val == 1)
    )

    print()

    print(
        "Class breakdown:"
    )

    print(
        f"  Train -> Real: {train_real}, "
        f"Anomaly: {train_anomaly}"
    )

    print(
        f"  Val   -> Real: {val_real}, "
        f"Anomaly: {val_anomaly}"
    )

    # -------------------------------------------------------------
    # CREATE DATASETS
    # -------------------------------------------------------------

    train_ds = TensorDataset(

        torch.tensor(
            bundle.X_train,
            dtype=torch.float32,
        ),

        torch.tensor(
            bundle.y_train,
            dtype=torch.long,
        ),
    )

    val_ds = TensorDataset(

        torch.tensor(
            bundle.X_val,
            dtype=torch.float32,
        ),

        torch.tensor(
            bundle.y_val,
            dtype=torch.long,
        ),
    )

    batch_size = min(
        args.batch_size,
        max(1, n_train),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    # -------------------------------------------------------------
    # SELECT DEVICE
    # -------------------------------------------------------------

    device = "cpu"

    if torch.cuda.is_available():

        try:

            test_tensor = torch.zeros(
                1,
                device="cuda",
            )

            del test_tensor

            device = "cuda"

        except Exception:

            device = "cpu"

    print(
        f"\nUsing compute device: "
        f"{device.upper()}"
    )

    # -------------------------------------------------------------
    # CREATE MODEL
    # -------------------------------------------------------------

    model = BehaviorLSTM(
        input_dim=FEATURE_DIM,
        num_classes=2,
    ).to(device)

    # -------------------------------------------------------------
    # CALCULATE CLASS WEIGHTS
    # -------------------------------------------------------------
    #
    # Your dataset is approximately:
    #
    # Real    = 166
    # Anomaly = 22
    #
    # Therefore Anomaly receives a larger loss weight.
    #
    # Formula:
    #
    # weight(class) =
    #     total_samples /
    #     (number_of_classes * class_count)
    #
    # This prevents the model from simply predicting
    # "Real" for almost everything.
    # -------------------------------------------------------------

    class_counts = np.bincount(
        bundle.y_train,
        minlength=2,
    )

    total_train_samples = (
        class_counts.sum()
    )

    number_of_classes = 2

    class_weights = (
        total_train_samples
        /
        (
            number_of_classes
            * np.maximum(
                class_counts,
                1,
            )
        )
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    print()

    print(
        "Class weights:"
    )

    print(
        f"  Real    (0): "
        f"{class_weights[0].item():.3f}"
    )

    print(
        f"  Anomaly (1): "
        f"{class_weights[1].item():.3f}"
    )

    # -------------------------------------------------------------
    # LOSS + OPTIMIZER
    # -------------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    # -------------------------------------------------------------
    # BEST MODEL TRACKING
    # -------------------------------------------------------------

    best_f1 = -1.0

    best_model_weights = None

    print("\n" + "=" * 70)

    print(
        "STARTING TRAINING"
    )

    print("=" * 70)

    # -------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        # =========================================================
        # TRAIN
        # =========================================================

        model.train()

        total_loss = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )

            optimizer.zero_grad()

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * X_batch.size(0)
            )

        train_loss = (
            total_loss / len(train_ds)
            if len(train_ds) > 0
            else 0.0
        )

        # =========================================================
        # VALIDATION
        # =========================================================

        model.eval()

        validation_predictions = []

        validation_labels = []

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(
                    device
                )

                logits = model(
                    X_batch
                )

                predictions = (
                    logits
                    .argmax(dim=1)
                    .cpu()
                    .numpy()
                )

                labels = (
                    y_batch
                    .cpu()
                    .numpy()
                )

                validation_predictions.extend(
                    predictions
                )

                validation_labels.extend(
                    labels
                )

        (
            val_accuracy,
            anomaly_precision,
            anomaly_recall,
            anomaly_f1,
        ) = calculate_metrics(
            validation_labels,
            validation_predictions,
        )

        print(
            f"Epoch {epoch:3d}/{args.epochs:3d} | "
            f"loss={train_loss:.4f} | "
            f"val_acc={val_accuracy:.3f} | "
            f"anomaly_precision={anomaly_precision:.3f} | "
            f"anomaly_recall={anomaly_recall:.3f} | "
            f"anomaly_f1={anomaly_f1:.3f}"
        )

        # ---------------------------------------------------------
        # SAVE BEST MODEL
        # ---------------------------------------------------------
        #
        # We use anomaly F1 rather than accuracy because the
        # dataset is heavily imbalanced.
        # ---------------------------------------------------------

        if anomaly_f1 >= best_f1:

            best_f1 = anomaly_f1

            best_model_weights = copy.deepcopy(
                model.state_dict()
            )

    # -------------------------------------------------------------
    # SAVE MODEL
    # -------------------------------------------------------------

    weights_path = os.path.join(
        args.out_dir,
        "behavior_lstm.pt",
    )

    if best_model_weights is not None:

        torch.save(
            best_model_weights,
            weights_path,
        )

    else:

        torch.save(
            model.state_dict(),
            weights_path,
        )

    # -------------------------------------------------------------
    # SAVE LABEL MAPPING
    # -------------------------------------------------------------

    canonical_labels = {
        "0": "Real",
        "1": "Anomaly",
    }

    labels_path = os.path.join(
        args.out_dir,
        "labels.json",
    )

    with open(
        labels_path,
        "w",
    ) as file:

        json.dump(
            canonical_labels,
            file,
            indent=2,
        )

    # -------------------------------------------------------------
    # SAVE NORMALIZATION STATISTICS
    # -------------------------------------------------------------

    stats_path = os.path.join(
        args.out_dir,
        "norm_stats.npz",
    )

    np.savez(
        stats_path,
        mean=bundle.norm_stats["mean"],
        std=bundle.norm_stats["std"],
    )

    # -------------------------------------------------------------
    # FINAL OUTPUT
    # -------------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "TRAINING COMPLETE"
    )

    print("=" * 70)

    print(
        f"Best validation Anomaly F1: "
        f"{best_f1:.3f}"
    )

    print(
        f"Saved checkpoint: "
        f"{weights_path}"
    )

    print(
        f"Saved class labels: "
        f"{labels_path}"
    )

    print(
        f"Saved normalization stats: "
        f"{stats_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()