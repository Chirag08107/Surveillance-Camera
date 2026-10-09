"""
Fast dataset generator for behavior classification.

Classes:
    real    -> 0
    anomaly -> 1

The generator:
    - Samples video frames using --frame-stride
    - Uses YOLO Pose tracking in a single pass
    - Uses ByteTrack for faster tracking
    - Extracts 13-feature vectors
    - Builds 30-frame temporal sequences
    - Saves the dataset as a compressed .npz file
"""

import argparse
import contextlib
import io
import os
from pathlib import Path

import cv2
import numpy as np

from pose import PoseEstimator
from behaviour import BehaviorAnalyzer
from features.feature_vector import build_feature_vector
from features.sequence_builder import SequenceBuilder


SEQUENCE_LENGTH = 30

SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
}


CLASS_CONFIG = {
    "real": {
        "label": 0,
        "fallback_dirs": ["real", "normal"],
    },
    "anomaly": {
        "label": 1,
        "fallback_dirs": ["anomaly", "anomalous"],
    },
}


def find_video_files(directory: str) -> list[str]:
    """
    Find all supported video files in a directory.
    """

    if not os.path.isdir(directory):
        return []

    video_files = []

    for entry in sorted(os.listdir(directory)):
        full_path = os.path.join(directory, entry)

        if not os.path.isfile(full_path):
            continue

        extension = Path(entry).suffix.lower()

        if extension in SUPPORTED_EXTENSIONS:
            video_files.append(full_path)

    return video_files


def resolve_class_dir(raw_dir: str, class_key: str) -> str | None:
    """
    Find the directory corresponding to a class.

    Supports:
        real / normal
        anomaly / anomalous
    """

    for dirname in CLASS_CONFIG[class_key]["fallback_dirs"]:
        candidate = os.path.join(raw_dir, dirname)

        if os.path.isdir(candidate):
            return candidate

    return None


def extract_sequences_from_video(
    video_path: str,
    label: int,
    pose_estimator: PoseEstimator,
    frame_stride: int = 5,
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[list[tuple[np.ndarray, int]], int, int]:
    """
    Extract temporal feature sequences from one video.

    Returns:
        examples:
            List of (sequence, label)

        processed_frames:
            Number of sampled frames actually processed

        total_video_frames:
            Number of frames reported by the video
    """

    if frame_stride < 1:
        frame_stride = 1

    if sequence_length < 1:
        sequence_length = SEQUENCE_LENGTH

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  [!] Could not open video: {video_path}")
        return [], 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Because we process only every Nth frame,
    # the effective time interval between processed frames increases.
    effective_fps = max(1.0, fps / frame_stride)

    behavior_analyzer = BehaviorAnalyzer()

    sequence_builder = SequenceBuilder(
        sequence_length=sequence_length
    )

    examples = []

    frames_since_flush = {}

    raw_frame_idx = 0
    processed_frames = 0

    try:

        while True:

            ret, frame = cap.read()

            if not ret or frame is None:
                break

            raw_frame_idx += 1

            # Safety guard against broken video metadata/decoders.
            if total_frames > 0 and raw_frame_idx > total_frames + 20:
                break

            # Process only every Nth frame.
            if frame_stride > 1:

                if raw_frame_idx % frame_stride != 0:
                    continue

            processed_frames += 1

            # ---------------------------------------------------------
            # YOLO POSE TRACKING
            # ---------------------------------------------------------
            #
            # ByteTrack is explicitly selected here.
            #
            # This avoids the expensive GMC optical-flow processing
            # that can occur with the default tracker.
            #
            # Pose model gives us:
            #   - person bounding boxes
            #   - track IDs
            #   - pose keypoints
            #
            # in one inference pass.
            # ---------------------------------------------------------

            results = pose_estimator.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            if not results:
                continue

            result = results[0]

            if result.boxes is None:
                continue

            if result.boxes.id is None:
                continue

            if result.keypoints is None:
                continue

            boxes = result.boxes.xyxy.cpu().tolist()

            track_ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )

            keypoints = result.keypoints.xy

            confidences = result.keypoints.conf

            # ---------------------------------------------------------
            # PROCESS EVERY TRACKED PERSON
            # ---------------------------------------------------------

            for i, track_id in enumerate(track_ids):

                if i >= len(keypoints):
                    continue

                person = keypoints[i]

                if (
                    confidences is not None
                    and i < len(confidences)
                ):
                    conf = confidences[i]
                else:
                    conf = None

                # -----------------------------------------------------
                # BEHAVIOR ANALYSIS
                # -----------------------------------------------------
                #
                # BehaviorAnalyzer contains diagnostic print
                # statements. They are suppressed here so that
                # dataset generation does not flood the terminal.
                # -----------------------------------------------------

                try:

                    with contextlib.redirect_stdout(
                        io.StringIO()
                    ):

                        behavior, avg_knee_angle = (
                            behavior_analyzer.analyze(
                                track_id,
                                person,
                                conf,
                                effective_fps,
                            )
                        )

                except Exception:
                    continue

                # -----------------------------------------------------
                # KNEE ANGLES
                # -----------------------------------------------------

                try:

                    left_hip = person[11]
                    left_knee = person[13]
                    left_ankle = person[15]

                    right_hip = person[12]
                    right_knee = person[14]
                    right_ankle = person[16]

                    left_knee_angle = (
                        behavior_analyzer.calculate_angle(
                            left_hip,
                            left_knee,
                            left_ankle,
                        )
                    )

                    right_knee_angle = (
                        behavior_analyzer.calculate_angle(
                            right_hip,
                            right_knee,
                            right_ankle,
                        )
                    )

                except Exception:
                    continue

                # -----------------------------------------------------
                # MOVEMENT FEATURES
                # -----------------------------------------------------

                try:

                    movement = (
                        behavior_analyzer.calculate_pose_movement(
                            track_id
                        )
                    )

                    (
                        avg_move,
                        max_move,
                        total_move,
                    ) = (
                        behavior_analyzer.get_movement_features(
                            track_id
                        )
                    )

                    velocity = (
                        behavior_analyzer.calculate_center_velocity(
                            track_id,
                            effective_fps,
                        )
                    )

                    acceleration = (
                        behavior_analyzer.calculate_center_acceleration(
                            track_id,
                            effective_fps,
                        )
                    )

                except Exception:
                    continue

                # -----------------------------------------------------
                # POSE CONFIDENCE
                # -----------------------------------------------------

                pose_conf_mean = 0.0

                if conf is not None:

                    try:

                        if hasattr(conf, "cpu"):
                            conf_np = conf.cpu().numpy()
                        else:
                            conf_np = np.asarray(conf)

                        if len(conf_np) > 0:
                            pose_conf_mean = float(
                                np.mean(conf_np)
                            )

                    except Exception:
                        pose_conf_mean = 0.0

                # -----------------------------------------------------
                # FEATURE VECTOR
                # -----------------------------------------------------
                #
                # 13 features:
                #
                # 1  left knee angle
                # 2  right knee angle
                # 3  average knee angle
                # 4  pose movement
                # 5  average movement
                # 6  maximum movement
                # 7  total movement
                # 8  center velocity
                # 9  center acceleration
                # 10 direction X
                # 11 direction Y
                # 12 restricted zone
                # 13 pose confidence
                # -----------------------------------------------------

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

                    # Zone/direction are not currently available
                    # from this standalone dataset generator.
                    direction="Stationary",

                    in_restricted_zone=False,

                    pose_confidence_mean=pose_conf_mean,
                )

                # -----------------------------------------------------
                # ADD FEATURE VECTOR TO TEMPORAL SEQUENCE
                # -----------------------------------------------------

                sequence_builder.push(
                    track_id,
                    vector,
                )

                frames_since_flush.setdefault(
                    track_id,
                    0,
                )

                frames_since_flush[track_id] += 1

                # -----------------------------------------------------
                # WHEN 30 FRAMES ARE READY
                # -----------------------------------------------------

                if (
                    sequence_builder.is_ready(track_id)
                    and
                    frames_since_flush[track_id]
                    >= sequence_length
                ):

                    sequence = (
                        sequence_builder.get_sequence(
                            track_id
                        )
                    )

                    if (
                        sequence is not None
                        and sequence.shape
                        == (sequence_length, 13)
                    ):

                        examples.append(
                            (
                                sequence,
                                label,
                            )
                        )

                    frames_since_flush[track_id] = 0

    finally:

        cap.release()

    # -------------------------------------------------------------
    # FALLBACK FOR SHORT VIDEOS
    # -------------------------------------------------------------
    #
    # If a video did not produce a complete 30-frame sequence,
    # attempt to use the available sequence if enough frames exist.
    # -------------------------------------------------------------

    if len(examples) == 0:

        for track_id in sequence_builder.active_track_ids():

            try:

                buffer_length = len(
                    sequence_builder.buffers.get(
                        track_id,
                        [],
                    )
                )

                minimum_samples = max(
                    10,
                    sequence_length // 3,
                )

                if buffer_length < minimum_samples:
                    continue

                sequence = (
                    sequence_builder.get_sequence(
                        track_id
                    )
                )

                if (
                    sequence is not None
                    and sequence.shape
                    == (sequence_length, 13)
                ):

                    examples.append(
                        (
                            sequence,
                            label,
                        )
                    )

                    break

            except Exception:
                continue

    return (
        examples,
        processed_frames,
        raw_frame_idx,
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Fast behavior dataset generator "
            "(Real vs Anomaly)"
        )
    )

    parser.add_argument(
        "--raw-dir",
        default="dataset/raw",
        help=(
            "Path to raw videos "
            "(default: dataset/raw)"
        ),
    )

    parser.add_argument(
        "--out",
        default="dataset/behavior_dataset.npz",
        help=(
            "Output dataset path "
            "(default: dataset/behavior_dataset.npz)"
        ),
    )

    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help=(
            "Process every Nth frame "
            "(default: 5)"
        ),
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
        help=(
            "Temporal sequence length "
            "(default: 30)"
        ),
    )

    parser.add_argument(
        "--video",
        action="append",
        default=None,
        help=(
            "Optional individual video path. "
            "Can be supplied multiple times."
        ),
    )

    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help=(
            "Label corresponding to --video. "
            "Use 0/real or 1/anomaly."
        ),
    )

    args = parser.parse_args()

    if args.frame_stride < 1:
        raise SystemExit(
            "Error: --frame-stride must be >= 1."
        )

    if args.sequence_length < 1:
        raise SystemExit(
            "Error: --sequence-length must be >= 1."
        )

    output_directory = os.path.dirname(
        os.path.abspath(args.out)
    )

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    print("=" * 70)

    print(
        "FAST BEHAVIOR DATASET GENERATION"
    )

    print(
        "Real (0) vs Anomaly (1)"
    )

    print("=" * 70)

    print(
        f"Frame stride     : {args.frame_stride}"
    )

    print(
        f"Sequence length  : {args.sequence_length}"
    )

    print(
        f"Output dataset   : {args.out}"
    )

    print("=" * 70)

    # -------------------------------------------------------------
    # INITIALIZE YOLO POSE MODEL ONCE
    # -------------------------------------------------------------

    print(
        "\nInitializing YOLO Pose Estimator..."
    )

    pose_estimator = PoseEstimator()

    print(
        "YOLO Pose Estimator initialized.\n"
    )

    all_sequences = []
    all_labels = []

    counts_by_class = {

        "real": {
            "videos": 0,
            "sequences": 0,
            "frames": 0,
            "label": 0,
        },

        "anomaly": {
            "videos": 0,
            "sequences": 0,
            "frames": 0,
            "label": 1,
        },

    }

    # -------------------------------------------------------------
    # BUILD VIDEO LIST
    # -------------------------------------------------------------

    if args.video is not None:

        if (
            args.label is None
            or len(args.video) != len(args.label)
        ):

            raise SystemExit(
                "Error: Every --video needs a matching --label."
            )

        video_items = []

        for video_path, raw_label in zip(
            args.video,
            args.label,
        ):

            raw_label = str(raw_label).lower()

            if raw_label in (
                "1",
                "anomaly",
                "anomalous",
            ):

                label = 1
                class_name = "anomaly"

            else:

                label = 0
                class_name = "real"

            video_items.append(
                (
                    video_path,
                    class_name,
                    label,
                )
            )

    else:

        video_items = []

        for class_name in [
            "real",
            "anomaly",
        ]:

            class_directory = resolve_class_dir(
                args.raw_dir,
                class_name,
            )

            if class_directory is None:

                print(
                    f"[!] Directory for '{class_name}' "
                    f"not found under '{args.raw_dir}'."
                )

                continue

            video_files = find_video_files(
                class_directory
            )

            label = CLASS_CONFIG[
                class_name
            ]["label"]

            for video_path in video_files:

                video_items.append(
                    (
                        video_path,
                        class_name,
                        label,
                    )
                )

    total_videos = len(video_items)

    print(
        f"Discovered {total_videos} video files.\n"
    )

    # -------------------------------------------------------------
    # PROCESS VIDEOS
    # -------------------------------------------------------------

    for index, (
        video_path,
        class_name,
        label,
    ) in enumerate(video_items, 1):

        filename = os.path.basename(
            video_path
        )

        print(
            f"[{index:2d}/{total_videos:2d}] "
            f"{filename:<25} "
            f"Class: {class_name}"
        )

        (
            examples,
            processed_frames,
            total_raw_frames,
        ) = extract_sequences_from_video(

            video_path=video_path,

            label=label,

            pose_estimator=pose_estimator,

            frame_stride=args.frame_stride,

            sequence_length=args.sequence_length,
        )

        number_of_sequences = len(
            examples
        )

        counts_by_class[
            class_name
        ]["videos"] += 1

        counts_by_class[
            class_name
        ]["sequences"] += number_of_sequences

        counts_by_class[
            class_name
        ]["frames"] += processed_frames

        for sequence, sequence_label in examples:

            all_sequences.append(
                sequence
            )

            all_labels.append(
                sequence_label
            )

        print(
            f"       Processed: "
            f"{processed_frames} sampled frames "
            f"from {total_raw_frames} total"
        )

        print(
            f"       Sequences: "
            f"{number_of_sequences}\n"
        )

    # -------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------

    print("=" * 70)

    print(
        "DATASET GENERATION FINAL SUMMARY"
    )

    print("=" * 70)

    total_sequences = len(
        all_sequences
    )

    total_processed_frames = sum(
        stats["frames"]
        for stats in counts_by_class.values()
    )

    for class_name, stats in counts_by_class.items():

        print(
            f"Class '{class_name}' "
            f"(label {stats['label']}): "
            f"{stats['videos']} videos, "
            f"{stats['frames']} sampled frames, "
            f"{stats['sequences']} sequences"
        )

    print("-" * 70)

    print(
        f"Total videos processed : "
        f"{total_videos}"
    )

    print(
        f"Total sampled frames   : "
        f"{total_processed_frames}"
    )

    print(
        f"Total sequences        : "
        f"{total_sequences}"
    )

    # -------------------------------------------------------------
    # NO DATA CHECK
    # -------------------------------------------------------------

    if not all_sequences:

        raise SystemExit(
            "\n[!] No sequences were extracted.\n"
            "Make sure the videos contain detectable people "
            "and pose keypoints."
        )

    # -------------------------------------------------------------
    # CONVERT TO NUMPY ARRAYS
    # -------------------------------------------------------------

    X = np.stack(
        all_sequences,
        axis=0,
    ).astype(
        np.float32
    )

    y = np.array(
        all_labels,
        dtype=np.int64,
    )

    labels = np.array(
        [
            "real",
            "anomaly",
        ]
    )

    # -------------------------------------------------------------
    # SAVE DATASET
    # -------------------------------------------------------------

    np.savez_compressed(
        args.out,
        X=X,
        y=y,
        labels=labels,
    )

    print("=" * 70)

    print(
        f"Dataset saved to: {args.out}"
    )

    print()

    print(
        f"X shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )

    print("=" * 70)

    # -------------------------------------------------------------
    # PROTOTYPE DATASET WARNING
    # -------------------------------------------------------------

    print()

    print("!" * 70)

    print(
        "WARNING:"
    )

    print(
        "This is currently a small prototype dataset."
    )

    print(
        "High validation accuracy may indicate overfitting "
        "to camera angles or subjects."
    )

    print(
        "For production deployment, collect more diverse footage."
    )

    print("!" * 70)


if __name__ == "__main__":
    main()