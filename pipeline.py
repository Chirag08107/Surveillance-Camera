"""
STEP 17 - FINAL INTEGRATED SURVEILLANCE PIPELINE

Runs the surveillance CV pipeline and sends structured results to the
FastAPI backend.

Pipeline:

    Video
      ↓
    YOLO detection + tracking
      ↓
    YOLO pose estimation + tracking
      ↓
    Behavior feature extraction
      ↓
    30-frame temporal sequence
      ↓
    Bi-LSTM behavior classifier
      ↓
    Anomaly detector
      ↓
    HTTP POST
      ↓
    FastAPI
      ↓
    PostgreSQL + WebSocket

Performance:
    Frames are sampled using --frame-stride.
    Default: every 5th original frame.

Example:

    uvicorn backend.main:app --reload --port 8000

Then in another terminal:

    python pipeline.py --video dataset/raw/anomaly/1.mp4 --headless
"""

import argparse
import math
import time

import cv2
import numpy as np
import requests
from ultralytics import YOLO

from pose import PoseEstimator
from behaviour import BehaviorAnalyzer
from features.feature_vector import build_feature_vector
from features.sequence_builder import SequenceBuilder
from model.infer import BehaviorClassifier
from anomaly.anomaly_detector import AnomalyDetector


# ---------------------------------------------------------------------------
# Optional ANPR
# ---------------------------------------------------------------------------

try:
    from anpr.plate_reader import PlateReader

    PLATE_READER_AVAILABLE = True

except ImportError:
    PLATE_READER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"

FRAME_STRIDE = 5
SEQUENCE_LENGTH = 30

PROGRESS_INTERVAL = 20

# Avoid sending the same anomaly event for the same track continuously.
ANOMALY_EVENT_COOLDOWN = 30


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def calculate_direction(points):
    """
    Determines the dominant movement direction from the last two
    center points.

    Returns:
        Stationary
        Right
        Left
        Down
        Up
    """

    if len(points) < 2:
        return "Stationary"

    px, py = points[-2]
    cx, cy = points[-1]

    dx = cx - px
    dy = cy - py

    distance = math.hypot(dx, dy)

    if distance < 3:
        return "Stationary"

    if abs(dx) > abs(dy):
        return "Right" if dx > 0 else "Left"

    return "Down" if dy > 0 else "Up"


def safe_post_frame(backend_url, payload):
    """
    Sends one batch of detections/events to FastAPI.

    Returns:
        True  -> request succeeded
        False -> backend unavailable
    """

    try:
        response = requests.post(
            f"{backend_url}/api/ingest/frame",
            json=payload,
            timeout=5,
        )

        response.raise_for_status()
        return True

    except requests.exceptions.RequestException as exc:
        print(f"[WARN] Backend unreachable, skipping ingest: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    video_path: str,
    backend_url: str,
    headless: bool,
    frame_stride: int = FRAME_STRIDE,
):
    print("=" * 70)
    print("INTEGRATED SURVEILLANCE PIPELINE")
    print("=" * 70)

    print(f"Video:             {video_path}")
    print(f"Frame stride:      {frame_stride}")
    print(f"Backend:           {backend_url}")
    print(f"Headless:          {headless}")
    print("-" * 70)

    # -----------------------------------------------------------------------
    # Load models/components
    # -----------------------------------------------------------------------

    print("Loading YOLO detection model...")
    detection_model = YOLO("yolo26n.pt")

    print("Loading YOLO pose model...")
    pose_estimator = PoseEstimator()

    print("Initializing behavior analyzer...")
    behaviour_analyzer = BehaviorAnalyzer()

    print("Initializing sequence builder...")
    sequence_builder = SequenceBuilder(
        sequence_length=SEQUENCE_LENGTH
    )

    print("Loading behavior classifier...")
    classifier = BehaviorClassifier()

    print("Initializing anomaly detector...")
    anomaly_detector = AnomalyDetector()

    plate_reader = None

    if PLATE_READER_AVAILABLE:
        try:
            plate_reader = PlateReader(gpu=False)
        except Exception as exc:
            print(f"[WARN] ANPR initialization failed: {exc}")
            plate_reader = None

    # -----------------------------------------------------------------------
    # Open video
    # -----------------------------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    effective_fps = fps / frame_stride

    print(f"Original FPS:      {fps:.2f}")
    print(f"Effective FPS:     {effective_fps:.2f}")
    print(f"Total frames:      {total_frames}")
    print("-" * 70)

    # -----------------------------------------------------------------------
    # Restricted zone
    # -----------------------------------------------------------------------

    restricted_zone = [
        (400, 200),
        (700, 200),
        (700, 450),
        (400, 450),
    ]

    zone_points = np.array(
        restricted_zone,
        dtype=np.int32,
    )

    # -----------------------------------------------------------------------
    # Tracking state
    # -----------------------------------------------------------------------

    person_zone_status = {}

    track_history = {}

    # Track when an anomaly event was last sent.
    anomaly_event_last_sent = {}

    # Statistics
    frames_processed = 0
    detections_sent = 0
    zone_events_sent = 0
    anomaly_events_sent = 0
    backend_successes = 0
    backend_failures = 0

    raw_frame_idx = 0

    start_time = time.time()

    print(f"Behavior classifier loaded: {classifier.model_loaded}")
    print(f"ANPR available:             {plate_reader is not None}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Main video loop
    # -----------------------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        raw_frame_idx += 1

        # ---------------------------------------------------------------
        # Frame sampling
        #
        # Only process every Nth frame.
        # This dramatically reduces YOLO CPU workload.
        # ---------------------------------------------------------------

        if (raw_frame_idx - 1) % frame_stride != 0:
            continue

        frames_processed += 1

        # ---------------------------------------------------------------
        # YOLO object detection + tracking
        # ---------------------------------------------------------------

        detection_results = detection_model.track(
            frame,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        if not detection_results:
            continue

        detection_result = detection_results[0]

        # ---------------------------------------------------------------
        # YOLO pose estimation + tracking
        #
        # This is executed only on sampled frames.
        # ---------------------------------------------------------------

        pose_results = pose_estimator.model.track(
            frame,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        if not pose_results:
            continue

        pose_result = pose_results[0]

        # ---------------------------------------------------------------
        # Draw restricted zone
        # ---------------------------------------------------------------

        cv2.polylines(
            frame,
            [zone_points],
            True,
            (0, 0, 255),
            2,
        )

        detections_batch = []
        zone_events_batch = []
        anomaly_events_batch = []

        # ---------------------------------------------------------------
        # Check whether object tracking produced tracks
        # ---------------------------------------------------------------

        if (
            detection_result.boxes is None
            or detection_result.boxes.id is None
        ):
            continue

        boxes = detection_result.boxes.xyxy.cpu().tolist()

        track_ids = (
            detection_result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        class_ids = (
            detection_result.boxes.cls
            .int()
            .cpu()
            .tolist()
        )

        object_confidences = (
            detection_result.boxes.conf
            .cpu()
            .tolist()
        )

        # ---------------------------------------------------------------
        # Pose information
        # ---------------------------------------------------------------

        pose_keypoints = None
        pose_confidences = None
        pose_boxes = None

        if pose_result.keypoints is not None:

            pose_keypoints = pose_result.keypoints.xy

            pose_confidences = pose_result.keypoints.conf

            if pose_result.boxes is not None:
                pose_boxes = pose_result.boxes.xyxy

        # ---------------------------------------------------------------
        # Process every detected object
        # ---------------------------------------------------------------

        for (
            box,
            track_id,
            class_id,
            object_confidence,
        ) in zip(
            boxes,
            track_ids,
            class_ids,
            object_confidences,
        ):

            x1, y1, x2, y2 = box

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            class_name = detection_result.names[class_id]

            # -----------------------------------------------------------
            # Restricted-zone detection
            # -----------------------------------------------------------

            inside = cv2.pointPolygonTest(
                zone_points,
                (center_x, center_y),
                False,
            )

            currently_inside = inside >= 0

            previously_inside = person_zone_status.get(
                track_id,
                False,
            )

            if currently_inside and not previously_inside:

                zone_events_batch.append(
                    {
                        "track_id": track_id,
                        "event_type": "enter",
                    }
                )

            elif not currently_inside and previously_inside:

                zone_events_batch.append(
                    {
                        "track_id": track_id,
                        "event_type": "exit",
                    }
                )

            person_zone_status[track_id] = currently_inside

            # -----------------------------------------------------------
            # Track history
            # -----------------------------------------------------------

            track_history.setdefault(track_id, [])

            track_history[track_id].append(
                (center_x, center_y)
            )

            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)

            points = track_history[track_id]

            direction = calculate_direction(points)

            # -----------------------------------------------------------
            # Default behavior values
            # -----------------------------------------------------------

            behavior_label = "Unknown"
            behavior_confidence = 0.0

            velocity = 0.0
            acceleration = 0.0

            # -----------------------------------------------------------
            # Pose → behavior analysis
            #
            # Only people need pose/behavior analysis.
            # -----------------------------------------------------------

            if (
                class_name == "person"
                and pose_keypoints is not None
                and pose_boxes is not None
            ):

                pose_index = behaviour_analyzer.match_pose_to_person(
                    box,
                    pose_boxes,
                )

                if pose_index is not None:

                    person = pose_keypoints[pose_index]

                    confidence = None

                    if pose_confidences is not None:
                        confidence = pose_confidences[pose_index]

                    # ---------------------------------------------------
                    # Rule-based behavior
                    # ---------------------------------------------------

                    rule_behavior, avg_knee_angle = (
                        behaviour_analyzer.analyze(
                            track_id,
                            person,
                            confidence,
                            effective_fps,
                        )
                    )

                    # ---------------------------------------------------
                    # Knee angles
                    # ---------------------------------------------------

                    left_hip = person[11]
                    left_knee = person[13]
                    left_ankle = person[15]

                    right_hip = person[12]
                    right_knee = person[14]
                    right_ankle = person[16]

                    left_angle = (
                        behaviour_analyzer.calculate_angle(
                            left_hip,
                            left_knee,
                            left_ankle,
                        )
                    )

                    right_angle = (
                        behaviour_analyzer.calculate_angle(
                            right_hip,
                            right_knee,
                            right_ankle,
                        )
                    )

                    # ---------------------------------------------------
                    # Movement features
                    # ---------------------------------------------------

                    movement = (
                        behaviour_analyzer.calculate_pose_movement(
                            track_id
                        )
                    )

                    (
                        avg_move,
                        max_move,
                        total_move,
                    ) = behaviour_analyzer.get_movement_features(
                        track_id
                    )

                    velocity = (
                        behaviour_analyzer.calculate_center_velocity(
                            track_id,
                            effective_fps,
                        )
                    )

                    acceleration = (
                        behaviour_analyzer.calculate_center_acceleration(
                            track_id,
                            effective_fps,
                        )
                    )

                    # ---------------------------------------------------
                    # Pose confidence
                    # ---------------------------------------------------

                    if confidence is not None:

                        conf_array = (
                            confidence
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        conf_mean = float(
                            np.mean(conf_array)
                        )

                    else:
                        conf_mean = 0.0

                    # ---------------------------------------------------
                    # Build ML feature vector
                    # ---------------------------------------------------

                    vector = build_feature_vector(
                        left_knee_angle=left_angle,
                        right_knee_angle=right_angle,
                        average_knee_angle=avg_knee_angle,
                        pose_movement=movement,
                        average_movement=avg_move,
                        maximum_movement=max_move,
                        total_movement=total_move,
                        center_velocity=velocity,
                        center_acceleration=acceleration,
                        direction=direction,
                        in_restricted_zone=currently_inside,
                        pose_confidence_mean=conf_mean,
                    )

                    # ---------------------------------------------------
                    # Add vector to temporal sequence
                    # ---------------------------------------------------

                    sequence_builder.push(
                        track_id,
                        vector,
                    )

                    sequence = sequence_builder.get_sequence(
                        track_id
                    )

                    # ---------------------------------------------------
                    # Bi-LSTM prediction
                    # ---------------------------------------------------

                    if sequence is not None:

                        (
                            behavior_label,
                            behavior_confidence,
                            model_used,
                        ) = classifier.predict(
                            sequence,
                            fallback_label=rule_behavior,
                        )

                        if not model_used:
                            behavior_label = rule_behavior

                        # ------------------------------------------------
                        # Print ML prediction
                        # ------------------------------------------------

                        print(
                            f"Frame {raw_frame_idx:4d} | "
                            f"Track {track_id:3d} | "
                            f"Prediction: "
                            f"{behavior_label:7s} | "
                            f"Confidence: "
                            f"{behavior_confidence:.3f}"
                        )

                        # ------------------------------------------------
                        # Anomaly detector
                        # ------------------------------------------------

                        anomaly_result = (
                            anomaly_detector.evaluate(
                                track_id=track_id,
                                velocity=velocity,
                                acceleration=acceleration,
                                in_restricted_zone=currently_inside,
                                fps=effective_fps,
                                classifier_label=behavior_label,
                                classifier_confidence=behavior_confidence,
                            )
                        )

                        if anomaly_result["is_anomalous"]:

                            current_processed_frame = frames_processed

                            last_sent = (
                                anomaly_event_last_sent.get(
                                    track_id,
                                    -ANOMALY_EVENT_COOLDOWN,
                                )
                            )

                            if (
                                current_processed_frame
                                - last_sent
                                >= ANOMALY_EVENT_COOLDOWN
                            ):

                                anomaly_events_batch.append(
                                    {
                                        "track_id": track_id,
                                        "reasons": anomaly_result[
                                            "reasons"
                                        ],
                                    }
                                )

                                anomaly_event_last_sent[
                                    track_id
                                ] = current_processed_frame

                    # ---------------------------------------------------
                    # Draw behavior label
                    # ---------------------------------------------------

                    frame = (
                        behaviour_analyzer.draw_behaviour(
                            frame,
                            behavior_label,
                            (
                                center_x,
                                int(y1) - 10,
                            ),
                        )
                    )

            # -----------------------------------------------------------
            # ANPR
            # -----------------------------------------------------------

            if (
                plate_reader is not None
                and plate_reader.is_vehicle(class_id)
            ):

                try:

                    plate = plate_reader.read_plate(
                        frame,
                        box,
                    )

                    if plate:

                        cv2.putText(
                            frame,
                            plate["text"],
                            (
                                int(x1),
                                int(y2) + 20,
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2,
                        )

                except Exception as exc:

                    print(
                        f"[WARN] ANPR error for track "
                        f"{track_id}: {exc}"
                    )

            # -----------------------------------------------------------
            # Detection database record
            # -----------------------------------------------------------

            detections_batch.append(
                {
                    "track_id": track_id,
                    "class_name": class_name,
                    "confidence": object_confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "center_x": center_x,
                    "center_y": center_y,
                    "behavior": behavior_label,
                    "behavior_confidence": behavior_confidence,
                    "in_restricted_zone": currently_inside,
                    "direction": direction,
                    "velocity": velocity,
                    "acceleration": acceleration,
                }
            )

            # -----------------------------------------------------------
            # Draw tracking information
            # -----------------------------------------------------------

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                frame,
                f"ID:{track_id}",
                (
                    int(x1),
                    int(y1) - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # -------------------------------------------------------------------
        # Send this frame's data to FastAPI
        # -------------------------------------------------------------------

        if (
            detections_batch
            or zone_events_batch
            or anomaly_events_batch
        ):

            payload = {
                "detections": detections_batch,
                "zone_events": zone_events_batch,
                "anomaly_events": anomaly_events_batch,
            }

            success = safe_post_frame(
                backend_url,
                payload,
            )

            if success:

                backend_successes += 1

                detections_sent += len(
                    detections_batch
                )

                zone_events_sent += len(
                    zone_events_batch
                )

                anomaly_events_sent += len(
                    anomaly_events_batch
                )

            else:
                backend_failures += 1

        # -------------------------------------------------------------------
        # Progress
        # -------------------------------------------------------------------

        if (
            frames_processed % PROGRESS_INTERVAL == 0
            or raw_frame_idx >= total_frames
        ):

            percentage = (
                (raw_frame_idx / total_frames) * 100
                if total_frames > 0
                else 0
            )

            elapsed = time.time() - start_time

            print(
                f"[Progress] "
                f"{raw_frame_idx}/{total_frames} "
                f"frames "
                f"({percentage:.1f}%) | "
                f"Processed: {frames_processed} | "
                f"Elapsed: {elapsed:.1f}s"
            )

        # -------------------------------------------------------------------
        # Optional display
        # -------------------------------------------------------------------

        if not headless:

            cv2.imshow(
                "Surveillance - Integrated Pipeline",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    cap.release()

    if not headless:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------

    print()
    print("=" * 70)
    print("INTEGRATED PIPELINE SUMMARY")
    print("=" * 70)

    print(f"Total video frames:       {total_frames}")
    print(f"Frames processed:         {frames_processed}")
    print(f"Frame stride:             {frame_stride}")
    print(f"Elapsed time:             {elapsed:.2f} seconds")
    print(f"Detections sent:          {detections_sent}")
    print(f"Zone events sent:         {zone_events_sent}")
    print(f"Anomaly events sent:      {anomaly_events_sent}")
    print(f"Successful backend calls: {backend_successes}")
    print(f"Failed backend calls:     {backend_failures}")

    print("-" * 70)

    if backend_failures == 0:
        print("Backend communication: SUCCESS")
    else:
        print("Backend communication: SOME REQUESTS FAILED")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Integrated AI surveillance pipeline"
    )

    parser.add_argument(
        "--video",
        default="videos/test.mp4",
        help="Path to input surveillance video",
    )

    parser.add_argument(
        "--backend-url",
        default=DEFAULT_BACKEND_URL,
        help="FastAPI backend URL",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without an OpenCV display window",
    )

    parser.add_argument(
        "--frame-stride",
        type=int,
        default=FRAME_STRIDE,
        help="Process every Nth frame",
    )

    args = parser.parse_args()

    if args.frame_stride < 1:
        raise ValueError(
            "--frame-stride must be >= 1"
        )

    run(
        video_path=args.video,
        backend_url=args.backend_url,
        headless=args.headless,
        frame_stride=args.frame_stride,
    )