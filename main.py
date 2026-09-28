import cv2
from ultralytics import YOLO
import math
import numpy as np
from pose import PoseEstimator
from behaviour import BehaviorAnalyzer

model = YOLO("yolo26n.pt")

cap = cv2.VideoCapture("videos/test.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video FPS: {fps}")

pose_estimator = PoseEstimator()
behaviour_analyzer = BehaviorAnalyzer()

track_history = {}

restricted_zone = [
    (400,200),
    (700,200),
    (700,450),
    (400,450)
]

person_zone_status = {}

zone_points = np.array(restricted_zone,np.int32)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # OBJECT TRACKING
    results = model.track(
        frame,
        persist=True
    )

    # POSE ESTIMATION
    frame, keypoints, pose_confidences, pose_boxes = (
        pose_estimator.process(frame)
    )

    # DRAW RESTRICTED ZONE
    cv2.polylines(
        frame,
        [zone_points],
        True,
        (0, 0, 255),
        2
    )

    result = results[0]

    # TRACKING INFORMATION
    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().tolist()

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        class_ids = (
            result.boxes.cls
            .int()
            .cpu()
            .tolist()
        )

        object_confidences = (
            result.boxes.conf
            .cpu()
            .tolist()
        )

        # PROCESS EACH TRACKED OBJECT
        for box, track_id, class_id, object_confidence in zip(
            boxes,
            track_ids,
            class_ids,
            object_confidences
        ):

            x1, y1, x2, y2 = box

            # MATCH TRACKING BOX TO POSE
            pose_index = (
                behaviour_analyzer.match_pose_to_person(
                    box,
                    pose_boxes
                )
            )

            # BEHAVIOR ANALYSIS
            if pose_index is not None:

                person = keypoints[pose_index]

                person_confidence = pose_confidences[pose_index]

                behavior, knee_angle = (
                    behaviour_analyzer.analyze(
                        track_id,
                        person,
                        person_confidence,
                        fps
                    )
                )

                # Find shoulder position
                left_shoulder = person[5]
                right_shoulder = person[6]

                text_x = int(
                    (left_shoulder[0] + right_shoulder[0]) / 2
                )

                text_y = int(
                    (left_shoulder[1] + right_shoulder[1]) / 2
                )

                frame = (
                    behaviour_analyzer.draw_behaviour(
                        frame,
                        behavior,
                        (text_x, text_y)
                    )
                )

                print(
                    f"ID: {track_id} | "
                    f"Behaviour: {behavior} | "
                    f"Knee Angle: {knee_angle:.2f}"
                )

            # PERSON CENTER
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # RESTRICTED ZONE
            inside = cv2.pointPolygonTest(
                zone_points,
                (center_x, center_y),
                False
            )

            currently_inside = inside >= 0

            previously_inside = (
                person_zone_status.get(
                    track_id,
                    False
                )
            )

            if currently_inside and not previously_inside:

                print(
                    f"Enter Event: "
                    f"Person {track_id} "
                    f"entered restricted zone"
                )

            if not currently_inside and previously_inside:

                print(
                    f"Exit Event: "
                    f"Person {track_id} "
                    f"left restricted zone"
                )

            person_zone_status[track_id] = currently_inside

            # TRACK HISTORY
            track_history.setdefault(
                track_id,
                []
            )

            track_history[track_id].append(
                (center_x, center_y)
            )

            if len(track_history[track_id]) > 30:

                track_history[track_id].pop(0)

            points = track_history[track_id]

            # MOVEMENT / DIRECTION
            if len(points) >= 2:

                previous_x, previous_y = points[-2]

                current_x, current_y = points[-1]

                dx = current_x - previous_x

                dy = current_y - previous_y

                distance = math.hypot(dx, dy)

                movement_threshold = 3

                if distance < movement_threshold:

                    direction = "Stationary"

                elif abs(dx) > abs(dy):

                    if dx > 0:
                        direction = "Right"
                    else:
                        direction = "Left"

                else:

                    if dy > 0:
                        direction = "Down"
                    else:
                        direction = "Up"

                print(
                    f"ID: {track_id} | "
                    f"Movement: {distance:.2f} pixels/frame | "
                    f"Direction: {direction}"
                )

            # DRAW TRACK TRAIL
            for i in range(1, len(points)):

                cv2.line(
                    frame,
                    points[i - 1],
                    points[i],
                    (0, 255, 0),
                    2
                )

            # OBJECT INFORMATION
            class_name = result.names[class_id]

            print(
                f"ID: {track_id} | "
                f"Object: {class_name} | "
                f"Confidence: {object_confidence:.2f} | "
                f"Center: ({center_x}, {center_y})"
            )

            # Center point
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 255, 0),
                -1
            )

            # Tracking ID
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # DISPLAY FINAL FRAME
    cv2.imshow(
        "Surveillance Camera",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()