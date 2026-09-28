"""
STEP 9 - DATASET GENERATION

Runs the SAME pipeline as main.py (YOLO tracking + pose + behaviour
feature extraction + sequence building) over one or more labeled videos,
and saves (sequence, label) pairs to disk as an .npz dataset that
model/train.py can load.

There is no shortcut around real labeled data: point --label at videos
that actually show that behavior (e.g. a walking clip, a falling clip,
a loitering clip). This script does the mechanical work of turning
those videos into training examples; you still have to supply footage
per class.

Usage:
    python -m dataset.dataset_generator \
        --video videos/walking_1.mp4 --label Walking \
        --video videos/falling_1.mp4 --label Falling \
        --out dataset/behavior_dataset.npz
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO

from pose import PoseEstimator
from behaviour import BehaviorAnalyzer
from features.feature_vector import build_feature_vector
from features.sequence_builder import SequenceBuilder

SEQUENCE_LENGTH = 30


def extract_sequences_from_video(video_path: str, label: str, detector_weights="yolo26n.pt"):
    """
    Runs detection+pose+behaviour analysis on one video and returns a list
    of (sequence, label) pairs - one finished 30-frame sequence per track,
    taken every SEQUENCE_LENGTH frames (non-overlapping windows) so the
    same motion isn't duplicated across examples.
    """

    model = YOLO(detector_weights)
    pose_estimator = PoseEstimator()
    behaviour_analyzer = BehaviorAnalyzer()
    seq_builder = SequenceBuilder(sequence_length=SEQUENCE_LENGTH)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    examples = []
    frames_since_flush = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False)
        _, keypoints, pose_confidences, pose_boxes = pose_estimator.process(frame)

        result = results[0]
        if result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().tolist()
        track_ids = result.boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            pose_index = behaviour_analyzer.match_pose_to_person(box, pose_boxes)
            if pose_index is None:
                continue

            person = keypoints[pose_index]
            confidence = pose_confidences[pose_index]

            behavior, avg_knee_angle = behaviour_analyzer.analyze(
                track_id, person, confidence, fps
            )

            left_hip, left_knee, left_ankle = person[11], person[13], person[15]
            right_hip, right_knee, right_ankle = person[12], person[14], person[16]
            left_knee_angle = behaviour_analyzer.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = behaviour_analyzer.calculate_angle(right_hip, right_knee, right_ankle)

            movement = behaviour_analyzer.calculate_pose_movement(track_id)
            avg_move, max_move, total_move = behaviour_analyzer.get_movement_features(track_id)
            velocity = behaviour_analyzer.calculate_center_velocity(track_id, fps)
            acceleration = behaviour_analyzer.calculate_center_acceleration(track_id, fps)

            vector = build_feature_vector(
                left_knee_angle=left_knee_angle,
                right_knee_angle=right_knee_angle,
                average_knee_angle=avg_knee_angle,
                pose_movement=movement,
                average_movement=avg_move,
                maximum_movement=max_move,
                total_movement=total_move,
                center_velocity=velocity,
                center_acceleration=acceleration,
                direction="Stationary",  # direction lives in main.py's loop; optional here
                in_restricted_zone=False,
                pose_confidence_mean=float(np.mean(confidence.cpu().numpy())) if confidence is not None else 0.0,
            )

            seq_builder.push(track_id, vector)
            frames_since_flush.setdefault(track_id, 0)
            frames_since_flush[track_id] += 1

            if seq_builder.is_ready(track_id) and frames_since_flush[track_id] >= SEQUENCE_LENGTH:
                sequence = seq_builder.get_sequence(track_id)
                examples.append((sequence, label))
                frames_since_flush[track_id] = 0

    cap.release()
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate a labeled behavior sequence dataset.")
    parser.add_argument("--video", action="append", required=True, help="Path to a video, repeatable")
    parser.add_argument("--label", action="append", required=True, help="Label for the preceding --video, repeatable")
    parser.add_argument("--out", default="dataset/behavior_dataset.npz")
    args = parser.parse_args()

    if len(args.video) != len(args.label):
        raise SystemExit("Every --video needs a matching --label (same order/count).")

    all_sequences = []
    all_labels = []

    for video_path, label in zip(args.video, args.label):
        print(f"Processing {video_path} -> label '{label}'")
        examples = extract_sequences_from_video(video_path, label)
        print(f"  {len(examples)} sequences extracted")
        for seq, lbl in examples:
            all_sequences.append(seq)
            all_labels.append(lbl)

    if not all_sequences:
        raise SystemExit("No sequences extracted - check that videos contain detectable people.")

    X = np.stack(all_sequences, axis=0)  # (N, 30, FEATURE_DIM)
    y = np.array(all_labels)

    np.savez_compressed(args.out, X=X, y=y)
    print(f"Saved {X.shape[0]} labeled sequences to {args.out}")


if __name__ == "__main__":
    main()
