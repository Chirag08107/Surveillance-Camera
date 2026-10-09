"""
Behavior classifier inference using a sliding temporal window.

The trained Bi-LSTM receives:

    30 temporal frames
    x
    13 features per frame

and predicts:

    Real
    Anomaly

Sliding-window inference:

    Frames 1-30   -> Prediction
    Frames 6-35   -> Prediction
    Frames 11-40  -> Prediction
    Frames 16-45  -> Prediction
    ...

The window therefore moves forward by 5 processed frames
between predictions.
"""

import argparse
import contextlib
import io
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import torch

from model.classifier import BehaviorLSTM
from features.feature_vector import (
    FEATURE_DIM,
    normalize_feature_vector,
)


SEQUENCE_LENGTH = 30

DEFAULT_FRAME_STRIDE = 5

DEFAULT_PREDICTION_INTERVAL = 5

DEFAULT_ANOMALY_THRESHOLD = 0.50


class BehaviorClassifier:
    """
    Loads the trained Bi-LSTM model and performs inference.
    """

    def __init__(
        self,
        checkpoint_dir: str = "model/checkpoints",
    ):
        self.checkpoint_dir = checkpoint_dir

        self.model_loaded = False

        self.model = None

        self.idx_to_label = {
            0: "Real",
            1: "Anomaly",
        }

        self.norm_stats = None

        self._try_load(checkpoint_dir)

    def _try_load(
        self,
        checkpoint_dir: str,
    ):
        """
        Load the trained model, labels, and
        normalization statistics.
        """

        weights_path = os.path.join(
            checkpoint_dir,
            "behavior_lstm.pt",
        )

        labels_path = os.path.join(
            checkpoint_dir,
            "labels.json",
        )

        stats_path = os.path.join(
            checkpoint_dir,
            "norm_stats.npz",
        )

        if not os.path.exists(weights_path):

            return

        if not os.path.exists(stats_path):

            return

        # ---------------------------------------------------------
        # Load label mapping
        # ---------------------------------------------------------

        if os.path.exists(labels_path):

            try:

                with open(
                    labels_path,
                    "r",
                ) as file:

                    raw_labels = json.load(file)

                for key, value in raw_labels.items():

                    value_string = (
                        str(value)
                        .strip()
                        .lower()
                    )

                    if value_string in (
                        "1",
                        "anomaly",
                        "anomalous",
                    ):

                        self.idx_to_label[
                            int(key)
                        ] = "Anomaly"

                    else:

                        self.idx_to_label[
                            int(key)
                        ] = "Real"

            except Exception:

                self.idx_to_label = {
                    0: "Real",
                    1: "Anomaly",
                }

        # ---------------------------------------------------------
        # Load normalization statistics
        # ---------------------------------------------------------

        stats = np.load(stats_path)

        self.norm_stats = {
            "mean": stats["mean"],
            "std": stats["std"],
        }

        # ---------------------------------------------------------
        # Create the EXACT architecture used during training
        # ---------------------------------------------------------

        self.model = BehaviorLSTM(
            input_dim=FEATURE_DIM,
            num_classes=2,
        )

        # ---------------------------------------------------------
        # Load trained weights
        # ---------------------------------------------------------

        state_dict = torch.load(
            weights_path,
            map_location="cpu",
        )

        self.model.load_state_dict(
            state_dict
        )

        self.model.eval()

        self.model_loaded = True

    def predict(
        self,
        sequence: np.ndarray,
        fallback_label: str = "Real",
    ) -> tuple[str, float, bool]:
        """
        Predict the class of one 30 x 13 sequence.

        Returns:

            label
            confidence
            model_loaded
        """

        if (
            not self.model_loaded
            or self.norm_stats is None
        ):

            return (
                fallback_label,
                0.0,
                False,
            )

        sequence = np.asarray(
            sequence,
            dtype=np.float32,
        )

        # ---------------------------------------------------------
        # Validate input
        # ---------------------------------------------------------

        if sequence.ndim != 2:

            return (
                fallback_label,
                0.0,
                False,
            )

        if sequence.shape[0] != SEQUENCE_LENGTH:

            return (
                fallback_label,
                0.0,
                False,
            )

        if sequence.shape[1] != FEATURE_DIM:

            return (
                fallback_label,
                0.0,
                False,
            )

        # ---------------------------------------------------------
        # Normalize using training statistics
        # ---------------------------------------------------------

        normalized_sequence = np.stack(
            [
                normalize_feature_vector(
                    frame,
                    self.norm_stats,
                )
                for frame in sequence
            ],
            axis=0,
        )

        # ---------------------------------------------------------
        # Convert to tensor
        #
        # (30, 13)
        #
        # becomes
        #
        # (1, 30, 13)
        # ---------------------------------------------------------

        x = torch.tensor(
            normalized_sequence,
            dtype=torch.float32,
        ).unsqueeze(0)

        # ---------------------------------------------------------
        # Run model
        # ---------------------------------------------------------

        with torch.no_grad():

            probabilities = (
                self.model.predict_proba(x)
            )

        probabilities = probabilities.squeeze(0)

        predicted_index = int(
            torch.argmax(
                probabilities
            ).item()
        )

        confidence = float(
            probabilities[
                predicted_index
            ].item()
        )

        label = self.idx_to_label.get(
            predicted_index,
            (
                "Anomaly"
                if predicted_index == 1
                else "Real"
            ),
        )

        return (
            label,
            confidence,
            True,
        )


class TrackAggregator:
    """
    Stores all predictions made for each track.

    Example:

        Track 7:

        Real
        Real
        Anomaly
        Anomaly
        Anomaly

    Anomaly ratio:

        3 / 5 = 60%
    """

    def __init__(self):

        self.predictions = defaultdict(
            list
        )

    def add_prediction(
        self,
        track_id: int,
        label: str,
        confidence: float,
    ):

        self.predictions[
            track_id
        ].append(
            {
                "label": label,
                "confidence": confidence,
            }
        )

    def get_track_summary(
        self,
        track_id: int,
        anomaly_threshold: float = DEFAULT_ANOMALY_THRESHOLD,
    ):

        results = self.predictions.get(
            track_id,
            [],
        )

        if not results:

            return {
                "real_count": 0,
                "anomaly_count": 0,
                "total": 0,
                "anomaly_ratio": 0.0,
                "average_confidence": 0.0,
                "status": "NORMAL",
            }

        real_count = sum(
            1
            for result in results
            if result["label"] == "Real"
        )

        anomaly_count = sum(
            1
            for result in results
            if result["label"] == "Anomaly"
        )

        total = len(results)

        anomaly_ratio = (
            anomaly_count / total
        )

        average_confidence = (
            sum(
                result["confidence"]
                for result in results
            )
            / total
        )

        status = (
            "SUSPICIOUS"
            if anomaly_ratio >= anomaly_threshold
            else "NORMAL"
        )

        return {
            "real_count": real_count,
            "anomaly_count": anomaly_count,
            "total": total,
            "anomaly_ratio": anomaly_ratio,
            "average_confidence": average_confidence,
            "status": status,
        }

    def get_all_summaries(
        self,
        anomaly_threshold: float = DEFAULT_ANOMALY_THRESHOLD,
    ):

        summaries = {}

        for track_id in self.predictions:

            summaries[
                track_id
            ] = self.get_track_summary(
                track_id,
                anomaly_threshold,
            )

        return summaries


def run_video(
    video_path: str,
    checkpoint_dir: str = "model/checkpoints",
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    sequence_length: int = SEQUENCE_LENGTH,
    prediction_interval: int = DEFAULT_PREDICTION_INTERVAL,
    anomaly_threshold: float = DEFAULT_ANOMALY_THRESHOLD,
):
    """
    Run sliding-window inference on a surveillance video.
    """

    from pose import PoseEstimator

    from behaviour import BehaviorAnalyzer

    from features.feature_vector import (
        build_feature_vector,
    )

    from features.sequence_builder import (
        SequenceBuilder,
    )

    # -------------------------------------------------------------
    # Load classifier
    # -------------------------------------------------------------

    classifier = BehaviorClassifier(
        checkpoint_dir
    )

    if not classifier.model_loaded:

        print(
            "Error: trained model could not "
            f"be loaded from '{checkpoint_dir}'."
        )

        return

    # -------------------------------------------------------------
    # Initialize YOLO pose model
    # -------------------------------------------------------------

    pose_estimator = PoseEstimator()

    behaviour_analyzer = (
        BehaviorAnalyzer()
    )

    sequence_builder = SequenceBuilder(
        sequence_length=sequence_length
    )

    # -------------------------------------------------------------
    # Open video
    # -------------------------------------------------------------

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            f"Error: Could not open "
            f"video '{video_path}'."
        )

        return

    # -------------------------------------------------------------
    # Video properties
    # -------------------------------------------------------------

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0 or np.isnan(fps):

        fps = 30.0

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    effective_fps = (
        fps / frame_stride
    )

    # -------------------------------------------------------------
    # Counters
    # -------------------------------------------------------------

    raw_frame_idx = 0

    processed_frames = 0

    predictions = []

    # -------------------------------------------------------------
    # Per-track sliding-window state
    # -------------------------------------------------------------

    frames_since_prediction = (
        defaultdict(int)
    )

    has_predicted = defaultdict(
        bool
    )

    # -------------------------------------------------------------
    # Aggregate predictions
    # -------------------------------------------------------------

    track_aggregator = (
        TrackAggregator()
    )

    # -------------------------------------------------------------
    # Information
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "BEHAVIOR CLASSIFIER - SLIDING WINDOW INFERENCE"
    )
    print("=" * 70)

    print(
        f"Video:                {video_path}"
    )

    print(
        f"Original FPS:         {fps:.2f}"
    )

    print(
        f"Frame stride:         {frame_stride}"
    )

    print(
        f"Effective FPS:        {effective_fps:.2f}"
    )

    print(
        f"Sequence length:      {sequence_length}"
    )

    print(
        f"Prediction interval:  {prediction_interval}"
    )

    print(
        f"Total frames:         {total_frames}"
    )

    print("-" * 70)

    # -------------------------------------------------------------
    # Main video loop
    #
    # IMPORTANT:
    #
    # We use the actual video frame count as a hard upper bound.
    #
    # This prevents a decoder that behaves unexpectedly from
    # causing an endless loop.
    # -------------------------------------------------------------

    while True:

        # ---------------------------------------------------------
        # Hard stop based on video frame count
        # ---------------------------------------------------------

        if (
            total_frames > 0
            and raw_frame_idx >= total_frames
        ):

            break

        # ---------------------------------------------------------
        # Read next frame
        # ---------------------------------------------------------

        ret, frame = cap.read()

        if not ret or frame is None:

            break

        raw_frame_idx += 1

        # ---------------------------------------------------------
        # Additional emergency safety limit
        #
        # Normally this is never reached.
        # ---------------------------------------------------------

        if (
            total_frames > 0
            and raw_frame_idx
            > total_frames + 10
        ):

            print(
                "\nSafety stop: video decoder "
                "returned more frames than expected."
            )

            break

        # ---------------------------------------------------------
        # Frame sampling
        #
        # Example:
        #
        # frame_stride = 5
        #
        # process:
        #
        # 5, 10, 15, 20, 25, ...
        # ---------------------------------------------------------

        if (
            raw_frame_idx
            % frame_stride
            != 0
        ):

            continue

        processed_frames += 1

        # ---------------------------------------------------------
        # Progress
        # ---------------------------------------------------------

        if (
            processed_frames % 20
            == 0
        ):

            if total_frames > 0:

                progress = (
                    raw_frame_idx
                    / total_frames
                    * 100
                )

                print(
                    f"[Progress] "
                    f"{raw_frame_idx}/{total_frames} "
                    f"frames "
                    f"({progress:.1f}%)"
                )

        # ---------------------------------------------------------
        # SINGLE YOLO INFERENCE
        #
        # This is the important optimization.
        #
        # The YOLO pose model provides:
        #
        #   boxes
        #   track IDs
        #   keypoints
        #   keypoint confidence
        #
        # in one pass.
        # ---------------------------------------------------------

        results = (
            pose_estimator.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                device="cpu",
            )
        )

        if (
            not results
            or len(results) == 0
        ):

            continue

        result = results[0]

        # ---------------------------------------------------------
        # Make sure tracking and pose data exist
        # ---------------------------------------------------------

        if result.boxes is None:

            continue

        if result.boxes.id is None:

            continue

        if result.keypoints is None:

            continue

        # ---------------------------------------------------------
        # Extract tracked boxes
        # ---------------------------------------------------------

        boxes = (
            result.boxes.xyxy
            .cpu()
            .tolist()
        )

        # ---------------------------------------------------------
        # Extract track IDs
        # ---------------------------------------------------------

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        # ---------------------------------------------------------
        # Extract pose keypoints
        # ---------------------------------------------------------

        keypoints = (
            result.keypoints.xy
        )

        # ---------------------------------------------------------
        # Extract keypoint confidence
        # ---------------------------------------------------------

        confidences = (
            result.keypoints.conf
        )

        # ---------------------------------------------------------
        # Process each tracked person
        # ---------------------------------------------------------

        for i, (
            box,
            track_id,
        ) in enumerate(
            zip(
                boxes,
                track_ids,
            )
        ):

            if i >= len(
                keypoints
            ):

                continue

            # -----------------------------------------------------
            # Person pose
            # -----------------------------------------------------

            person = keypoints[i]

            if (
                confidences is not None
                and i < len(confidences)
            ):

                confidence = (
                    confidences[i]
                )

            else:

                confidence = None

            # -----------------------------------------------------
            # Behavior analysis
            # -----------------------------------------------------

            with contextlib.redirect_stdout(
                io.StringIO()
            ):

                (
                    behavior,
                    average_knee_angle,
                ) = (
                    behaviour_analyzer
                    .analyze(
                        track_id,
                        person,
                        confidence,
                        effective_fps,
                    )
                )

            # -----------------------------------------------------
            # Extract knee points
            # -----------------------------------------------------

            left_hip = person[11]

            left_knee = person[13]

            left_ankle = person[15]

            right_hip = person[12]

            right_knee = person[14]

            right_ankle = person[16]

            # -----------------------------------------------------
            # Calculate knee angles
            # -----------------------------------------------------

            left_knee_angle = (
                behaviour_analyzer
                .calculate_angle(
                    left_hip,
                    left_knee,
                    left_ankle,
                )
            )

            right_knee_angle = (
                behaviour_analyzer
                .calculate_angle(
                    right_hip,
                    right_knee,
                    right_ankle,
                )
            )

            # -----------------------------------------------------
            # Movement
            # -----------------------------------------------------

            pose_movement = (
                behaviour_analyzer
                .calculate_pose_movement(
                    track_id
                )
            )

            center_velocity = (
                behaviour_analyzer
                .calculate_center_velocity(
                    track_id,
                    effective_fps,
                )
            )

            center_acceleration = (
                behaviour_analyzer
                .calculate_center_acceleration(
                    track_id,
                    effective_fps,
                )
            )

            # -----------------------------------------------------
            # Pose confidence
            # -----------------------------------------------------

            pose_confidence_mean = 0.0

            if confidence is not None:

                try:

                    confidence_array = (
                        confidence.cpu().numpy()
                        if hasattr(
                            confidence,
                            "cpu",
                        )
                        else np.asarray(
                            confidence
                        )
                    )

                    if len(
                        confidence_array
                    ) > 0:

                        pose_confidence_mean = (
                            float(
                                np.mean(
                                    confidence_array
                                )
                            )
                        )

                except Exception:

                    pose_confidence_mean = 0.0

            # -----------------------------------------------------
            # Build the same 13-feature vector
            # used during training.
            # -----------------------------------------------------

            feature_vector = (
                build_feature_vector(
                    left_knee_angle=(
                        left_knee_angle
                    ),

                    right_knee_angle=(
                        right_knee_angle
                    ),

                    average_knee_angle=(
                        average_knee_angle
                    ),

                    pose_movement=(
                        pose_movement
                    ),

                    average_movement=0.0,

                    maximum_movement=0.0,

                    total_movement=0.0,

                    center_velocity=(
                        center_velocity
                    ),

                    center_acceleration=(
                        center_acceleration
                    ),

                    direction="Stationary",

                    in_restricted_zone=False,

                    pose_confidence_mean=(
                        pose_confidence_mean
                    ),
                )
            )

            # -----------------------------------------------------
            # Add vector to person's temporal buffer
            # -----------------------------------------------------

            sequence_builder.push(
                track_id,
                feature_vector,
            )

            # -----------------------------------------------------
            # Wait until 30 frames are available
            # -----------------------------------------------------

            if not sequence_builder.is_ready(
                track_id
            ):

                continue

            # -----------------------------------------------------
            # Count processed frames since last prediction
            # -----------------------------------------------------

            frames_since_prediction[
                track_id
            ] += 1

            # -----------------------------------------------------
            # First prediction happens immediately
            # -----------------------------------------------------

            should_predict = (
                not has_predicted[
                    track_id
                ]
            )

            # -----------------------------------------------------
            # Subsequent predictions happen every
            # prediction_interval processed frames.
            # -----------------------------------------------------

            if (
                has_predicted[
                    track_id
                ]
                and
                frames_since_prediction[
                    track_id
                ]
                >= prediction_interval
            ):

                should_predict = True

            if not should_predict:

                continue

            # -----------------------------------------------------
            # Get latest 30-frame window
            # -----------------------------------------------------

            sequence = (
                sequence_builder
                .get_sequence(
                    track_id
                )
            )

            if sequence is None:

                continue

            # -----------------------------------------------------
            # Bi-LSTM prediction
            # -----------------------------------------------------

            (
                prediction,
                confidence_score,
                model_loaded,
            ) = classifier.predict(
                sequence
            )

            if not model_loaded:

                continue

            # -----------------------------------------------------
            # Store prediction
            # -----------------------------------------------------

            predictions.append(
                (
                    raw_frame_idx,
                    track_id,
                    prediction,
                    confidence_score,
                )
            )

            track_aggregator.add_prediction(
                track_id,
                prediction,
                confidence_score,
            )

            # -----------------------------------------------------
            # Reset interval
            # -----------------------------------------------------

            frames_since_prediction[
                track_id
            ] = 0

            has_predicted[
                track_id
            ] = True

            # -----------------------------------------------------
            # Print prediction
            # -----------------------------------------------------

            print(
                f"Frame "
                f"{raw_frame_idx:4d} | "
                f"Track "
                f"{track_id:2d} | "
                f"Prediction: "
                f"{prediction:7s} | "
                f"Confidence: "
                f"{confidence_score:.3f}"
            )

    # -------------------------------------------------------------
    # Release video
    # -------------------------------------------------------------

    cap.release()

    # -------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("INFERENCE SUMMARY")
    print("=" * 70)

    print(
        f"Total video frames: "
        f"{total_frames}"
    )

    print(
        f"Frames processed: "
        f"{processed_frames}"
    )

    print(
        f"Predictions made: "
        f"{len(predictions)}"
    )

    real_predictions = sum(
        1
        for (
            _,
            _,
            label,
            _,
        ) in predictions
        if label == "Real"
    )

    anomaly_predictions = sum(
        1
        for (
            _,
            _,
            label,
            _,
        ) in predictions
        if label == "Anomaly"
    )

    print(
        f"Real predictions: "
        f"{real_predictions}"
    )

    print(
        f"Anomaly predictions: "
        f"{anomaly_predictions}"
    )

    print("-" * 70)

    print("TRACK-WISE RESULTS")

    print("-" * 70)

    summaries = (
        track_aggregator
        .get_all_summaries(
            anomaly_threshold
        )
    )

    suspicious_tracks = []

    for track_id in sorted(
        summaries
    ):

        summary = summaries[
            track_id
        ]

        if (
            summary["status"]
            == "SUSPICIOUS"
        ):

            suspicious_tracks.append(
                track_id
            )

        print(
            f"Track {track_id:2d} | "
            f"Real: "
            f"{summary['real_count']:2d} | "
            f"Anomaly: "
            f"{summary['anomaly_count']:2d} | "
            f"Total: "
            f"{summary['total']:2d} | "
            f"Anomaly ratio: "
            f"{summary['anomaly_ratio'] * 100:6.1f}% | "
            f"Avg confidence: "
            f"{summary['average_confidence']:.3f} | "
            f"{summary['status']}"
        )

    print("-" * 70)

    if suspicious_tracks:

        print(
            "Suspicious tracks:",
            suspicious_tracks,
        )

    else:

        print(
            "Suspicious tracks: None"
        )

    print("=" * 70)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run sliding-window inference "
            "using the trained Behavior Bi-LSTM."
        )
    )

    parser.add_argument(
        "--checkpoint-dir",
        default="model/checkpoints",
        help="Path to checkpoint directory",
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to input surveillance video",
    )

    parser.add_argument(
        "--frame-stride",
        type=int,
        default=DEFAULT_FRAME_STRIDE,
        help="Process every Nth video frame",
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
        help="Number of frames per LSTM sequence",
    )

    parser.add_argument(
        "--prediction-interval",
        type=int,
        default=DEFAULT_PREDICTION_INTERVAL,
        help="Processed frames between predictions",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_ANOMALY_THRESHOLD,
        help="Anomaly ratio threshold",
    )

    args = parser.parse_args()

    if args.frame_stride <= 0:

        parser.error(
            "--frame-stride must be greater than 0"
        )

    if args.sequence_length <= 0:

        parser.error(
            "--sequence-length must be greater than 0"
        )

    if args.prediction_interval <= 0:

        parser.error(
            "--prediction-interval must be greater than 0"
        )

    if not (
        0.0
        <= args.threshold
        <= 1.0
    ):

        parser.error(
            "--threshold must be between 0 and 1"
        )

    run_video(
        video_path=args.video,
        checkpoint_dir=args.checkpoint_dir,
        frame_stride=args.frame_stride,
        sequence_length=args.sequence_length,
        prediction_interval=(
            args.prediction_interval
        ),
        anomaly_threshold=args.threshold,
    )


if __name__ == "__main__":

    main()