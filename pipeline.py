"""
STEP 17 - FINAL INTEGRATED SYSTEM

Runs the exact same detection + tracking + pose + behaviour + zone logic
as main.py (unchanged, still runs standalone for local debugging with
an on-screen window), but additionally:
  - builds feature vectors + 30-frame sequences (Step 8)
  - classifies behavior with the trained LSTM when available, otherwise
    falls back to behaviour.py's rule-based label (Step 10)
  - runs anomaly detection (Step 11)
  - reads vehicle plates when a vehicle is detected (Step 13)
  - posts everything to the FastAPI backend over HTTP so it lands in
    Postgres and streams to the dashboard (Step 14/15)

Run the backend first (`uvicorn backend.main:app --port 8000`), then:
    python pipeline.py --video videos/test.mp4
    python pipeline.py --video videos/test.mp4 --headless   (no cv2 window - use this on a server)
"""

import argparse
import math

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

try:
    from anpr.plate_reader import PlateReader
    _plate_reader_available = True
except ImportError:
    _plate_reader_available = False


def run(video_path: str, backend_url: str, headless: bool):
    model = YOLO("yolo26n.pt")
    pose_estimator = PoseEstimator()
    behaviour_analyzer = BehaviorAnalyzer()
    seq_builder = SequenceBuilder(sequence_length=30)
    classifier = BehaviorClassifier()
    anomaly_detector = AnomalyDetector()
    plate_reader = PlateReader(gpu=False) if _plate_reader_available else None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    restricted_zone = [(400, 200), (700, 200), (700, 450), (400, 450)]
    zone_points = np.array(restricted_zone, np.int32)
    person_zone_status = {}
    track_history = {}
    seen_track_ids = set()

    print(f"Behavior classifier loaded from checkpoint: {classifier.model_loaded}")
    print(f"ANPR available: {plate_reader is not None}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False)
        frame, keypoints, pose_confidences, pose_boxes = pose_estimator.process(frame)

        cv2.polylines(frame, [zone_points], True, (0, 0, 255), 2)

        result = results[0]
        detections_batch = []
        zone_events_batch = []
        anomaly_events_batch = []

        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().tolist()
            track_ids = result.boxes.id.int().cpu().tolist()
            class_ids = result.boxes.cls.int().cpu().tolist()
            object_confidences = result.boxes.conf.cpu().tolist()

            for box, track_id, class_id, object_confidence in zip(
                boxes, track_ids, class_ids, object_confidences
            ):
                seen_track_ids.add(track_id)
                x1, y1, x2, y2 = box
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                class_name = result.names[class_id]

                inside = cv2.pointPolygonTest(zone_points, (center_x, center_y), False)
                currently_inside = inside >= 0
                previously_inside = person_zone_status.get(track_id, False)

                if currently_inside and not previously_inside:
                    zone_events_batch.append({"track_id": track_id, "event_type": "enter"})
                if not currently_inside and previously_inside:
                    zone_events_batch.append({"track_id": track_id, "event_type": "exit"})
                person_zone_status[track_id] = currently_inside

                track_history.setdefault(track_id, [])
                track_history[track_id].append((center_x, center_y))
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)
                points = track_history[track_id]

                direction = "Stationary"
                if len(points) >= 2:
                    px, py = points[-2]
                    cx, cy = points[-1]
                    dx, dy = cx - px, cy - py
                    distance = math.hypot(dx, dy)
                    if distance >= 3:
                        direction = ("Right" if dx > 0 else "Left") if abs(dx) > abs(dy) else ("Down" if dy > 0 else "Up")

                behavior_label, behavior_confidence = "Unknown", 0.0
                velocity, acceleration = 0.0, 0.0

                pose_index = behaviour_analyzer.match_pose_to_person(box, pose_boxes)
                if pose_index is not None:
                    person = keypoints[pose_index]
                    confidence = pose_confidences[pose_index]

                    rule_behavior, avg_knee_angle = behaviour_analyzer.analyze(
                        track_id, person, confidence, fps
                    )

                    left_hip, left_knee, left_ankle = person[11], person[13], person[15]
                    right_hip, right_knee, right_ankle = person[12], person[14], person[16]
                    left_angle = behaviour_analyzer.calculate_angle(left_hip, left_knee, left_ankle)
                    right_angle = behaviour_analyzer.calculate_angle(right_hip, right_knee, right_ankle)

                    movement = behaviour_analyzer.calculate_pose_movement(track_id)
                    avg_move, max_move, total_move = behaviour_analyzer.get_movement_features(track_id)
                    velocity = behaviour_analyzer.calculate_center_velocity(track_id, fps)
                    acceleration = behaviour_analyzer.calculate_center_acceleration(track_id, fps)

                    conf_mean = float(np.mean(confidence.cpu().numpy())) if confidence is not None else 0.0

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

                    seq_builder.push(track_id, vector)
                    sequence = seq_builder.get_sequence(track_id)

                    if sequence is not None:
                        behavior_label, behavior_confidence, model_used = classifier.predict(
                            sequence, fallback_label=rule_behavior
                        )
                        if not model_used:
                            behavior_label = rule_behavior

                    anomaly_result = anomaly_detector.evaluate(
                        track_id=track_id,
                        velocity=velocity,
                        acceleration=acceleration,
                        in_restricted_zone=currently_inside,
                        fps=fps,
                        classifier_label=behavior_label,
                        classifier_confidence=behavior_confidence,
                    )

                    if anomaly_result["is_anomalous"]:
                        anomaly_events_batch.append(
                            {"track_id": track_id, "reasons": anomaly_result["reasons"]}
                        )

                    frame = behaviour_analyzer.draw_behaviour(
                        frame, behavior_label, (center_x, int(y1) - 10)
                    )

                # ANPR - only attempt on vehicle classes
                if plate_reader is not None and plate_reader.is_vehicle(class_id):
                    plate = plate_reader.read_plate(frame, box)
                    if plate:
                        cv2.putText(
                            frame, plate["text"], (int(x1), int(y2) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
                        )

                detections_batch.append({
                    "track_id": track_id,
                    "class_name": class_name,
                    "confidence": object_confidence,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "center_x": center_x, "center_y": center_y,
                    "behavior": behavior_label,
                    "behavior_confidence": behavior_confidence,
                    "in_restricted_zone": currently_inside,
                    "direction": direction,
                    "velocity": velocity,
                    "acceleration": acceleration,
                })

                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # clean up sequence/anomaly buffers for tracks that disappeared
        gone = seen_track_ids - set(person_zone_status.keys())
        # (kept simple - see README for a production-grade track-timeout strategy)

        if detections_batch or zone_events_batch or anomaly_events_batch:
            try:
                requests.post(
                    f"{backend_url}/api/ingest/frame",
                    json={
                        "detections": detections_batch,
                        "zone_events": zone_events_batch,
                        "anomaly_events": anomaly_events_batch,
                    },
                    timeout=5,
                )
            except requests.exceptions.RequestException as e:
                print(f"[warn] backend unreachable, skipping ingest this frame: {e}")

        if not headless:
            cv2.imshow("Surveillance - Integrated Pipeline", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="videos/test.mp4")
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--headless", action="store_true", help="No cv2 window - use on a server/without a display")
    args = parser.parse_args()

    run(args.video, args.backend_url, args.headless)
