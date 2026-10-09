import math
import cv2
import numpy as np

class BehaviorAnalyzer:

    def __init__(self):
        self.pose_history = {}
        self.center_history = {}
        self.velocity_history = {}

    def calculate_angle(self, point1, point2, point3):

        x1, y1 = point1
        x2, y2 = point2
        x3, y3 = point3

        angle1 = math.atan2(y1 - y2, x1 - x2)
        angle2 = math.atan2(y3 - y2, x3 - x2)

        angle = abs(
            math.degrees(angle1 - angle2)
        )

        if angle > 180:
            angle = 360 - angle

        return angle

    def analyze(self, track_id, person, confidence, fps):

        if track_id not in self.pose_history:
            self.pose_history[track_id] = []

        normalized_pose = self.normalize_pose(
            person,
            confidence
        )

        if normalized_pose is None:
            return "Unknown", 0

        self.pose_history[track_id].append(
            normalized_pose
        )

        if len(self.pose_history[track_id]) > 30:
            self.pose_history[track_id].pop(0)

        body_center = self.get_body_center(
            person,
            confidence
        )

        if body_center is not None:

            if track_id not in self.center_history:
                self.center_history[track_id] = []

            self.center_history[track_id].append(
                body_center
            )

            if len(self.center_history[track_id]) > 30:
                self.center_history[track_id].pop(0)

        movement = self.calculate_pose_movement(
            track_id
        )

        average_movement, maximum_movement, total_movement = (
            self.get_movement_features(track_id)
        )

        velocity = self.calculate_center_velocity(
            track_id,
            fps
        )

        acceleration = self.calculate_center_acceleration(
            track_id,
            fps
        )

        left_hip = person[11]
        left_knee = person[13]
        left_ankle = person[15]

        right_hip = person[12]
        right_knee = person[14]
        right_ankle = person[16]

        left_knee_angle = self.calculate_angle(
            left_hip,
            left_knee,
            left_ankle
        )

        right_knee_angle = self.calculate_angle(
            right_hip,
            right_knee,
            right_ankle
        )

        average_knee_angle = (
            left_knee_angle +
            right_knee_angle
        ) / 2

        if average_knee_angle > 150:
            behavior = "Standing"

        elif average_knee_angle < 120:
            behavior = "Sitting"

        else:
            behavior = "Unknown"

        print(
            f"ID: {track_id} | "
            f"Average: {average_movement:.2f} | "
            f"Maximum: {maximum_movement:.2f} | "
            f"Total: {total_movement:.2f}"
        )

        print(
            f"ID: {track_id} | "
            f"Pose Movement: {movement:.2f}"
        )

        print(
            f"ID: {track_id} | "
            f"Center Velocity: {velocity:.2f} pixels/sec | "
            f"Acceleration: {acceleration:.2f} px/s²"
        )

        return behavior, average_knee_angle

    def draw_behaviour(self,frame,behaviour,position) :
        x,y = position

        cv2.putText(
            frame,
            behaviour,
            (x,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        return frame

    def get_box_center(self,box):
        x1,y1,x2,y2 = box

        center_x = (x1+x2)/2
        center_y = (y1+y2)/2

        return center_x,center_y

    def match_pose_to_person(self,tracking_box,pose_boxes):

        tracking_center = self.get_box_center(tracking_box)

        best_index = None
        best_distance = float("inf")

        for i, pose_box in enumerate(pose_boxes):

            pose_center = self.get_box_center(
                pose_box
            )

            dx = (
                tracking_center[0]
                - pose_center[0]
            )

            dy = (
                tracking_center[1]
                - pose_center[1]
            )

            distance = math.hypot(dx, dy)

            if distance < best_distance:

                best_distance = distance
                best_index = i

        return best_index

    def get_hip_center(self,person):
        left_hip = person[11]
        right_hip = person[12]

        x = (left_hip[0] + right_hip[0]) / 2
        y = (left_hip[1]+right_hip[1]) / 2

        return x,y

    def calculate_pose_movement(self, track_id):

        history = self.pose_history.get(track_id, [])

        if len(history) < 2:
            return 0

        previous_pose = history[-2]
        current_pose = history[-1]

        total_distance = 0
        valid_points = 0

        for i in range(len(previous_pose)):

            previous_x = previous_pose[i][0]
            previous_y = previous_pose[i][1]

            current_x = current_pose[i][0]
            current_y = current_pose[i][1]

            dx = current_x - previous_x
            dy = current_y - previous_y

            distance = math.hypot(dx, dy)

            total_distance += distance
            valid_points += 1

        if valid_points == 0:
            return 0

        average_distance = (
            total_distance / valid_points
        )

        return average_distance

    def normalize_pose(self,person,confidence):
        confidence_threshold = 0.5

        left_hip = person[11]
        right_hip = person[12]

        left_hip = person[11]
        right_hip = person[12]

        left_hip_valid = (
            confidence[11] >= confidence_threshold
        )

        right_hip_valid = (
            confidence[12] >= confidence_threshold
        )

        if left_hip_valid and right_hip_valid:

            hip_center_x = (
                left_hip[0] + right_hip[0]
            ) / 2

            hip_center_y = (
                left_hip[1] + right_hip[1]
            ) / 2

        elif left_hip_valid:

            hip_center_x = left_hip[0]
            hip_center_y = left_hip[1]

        elif right_hip_valid:

            hip_center_x = right_hip[0]
            hip_center_y = right_hip[1]

        else:

            return None

        x_coordinates = person[:,0]
        y_coordinates = person[:,1]

        min_x = x_coordinates.min()
        max_x = x_coordinates.max()

        min_y = y_coordinates.min()
        max_y = y_coordinates.max()

        width = max_x - min_x
        height = max_y - min_y

        scale = max(width, height)

        if scale == 0:
            scale = 1

        normalized_pose = []

        for i, point in enumerate(person):

            x = point[0]
            y = point[1]

            if confidence[i] < confidence_threshold:

                normalized_pose.append([0, 0])

                continue

            normalized_x = (
                x - hip_center_x
            ) / scale

            normalized_y = (
                y - hip_center_y
            ) / scale

            normalized_pose.append(
                [normalized_x, normalized_y]
            )

        return normalized_pose

    def get_movement_features(self, track_id):

        history = self.pose_history.get(track_id, [])

        if len(history) < 2:
            return 0, 0, 0

        movements = []

        for i in range(1, len(history)):

            previous_pose = history[i - 1]
            current_pose = history[i]

            total_distance = 0
            valid_points = 0

            for j in range(len(previous_pose)):

                previous_x = previous_pose[j][0]
                previous_y = previous_pose[j][1]

                current_x = current_pose[j][0]
                current_y = current_pose[j][1]

                dx = current_x - previous_x
                dy = current_y - previous_y

                distance = math.hypot(dx, dy)

                total_distance += distance
                valid_points += 1

            if valid_points > 0:

                average_distance = (
                    total_distance / valid_points
                )

                movements.append(average_distance)

        if len(movements) == 0:
            return 0, 0, 0

        average_movement = (
            sum(movements) / len(movements)
        )

        maximum_movement = max(movements)

        total_movement = sum(movements)

        return (
            average_movement,
            maximum_movement,
            total_movement
        )

    def get_body_center(self,person,confidence):
        confidence_threshold = 0.5

        left_hip = person[11]
        right_hip = person[12]

        left_valid = confidence[11] >= confidence_threshold
        right_valid = confidence[12] >= confidence_threshold

        if left_valid and right_valid:

            center_x = (
                left_hip[0] + right_hip[0]
            ) / 2

            center_y = (
                left_hip[1] + right_hip[1]
            ) / 2

        elif left_valid:

            center_x = left_hip[0]
            center_y = left_hip[1]

        elif right_valid:

            center_x = right_hip[0]
            center_y = right_hip[1]

        else:
            return None

        return center_x, center_y

    def calculate_center_displacement(self, track_id):

        history = self.center_history.get(track_id, [])

        if len(history) < 2:
            return 0

        previous_x, previous_y = history[-2]
        current_x, current_y = history[-1]

        dx = current_x - previous_x
        dy = current_y - previous_y

        distance = math.hypot(dx, dy)

        return distance

    def calculate_center_velocity(self,track_id,fps):
        displacement = self.calculate_center_displacement(track_id)

        if fps <=0:
            return 0

        velocity = displacement * fps

        return velocity

    def calculate_center_velocity_vector(self, track_id, fps):

        history = self.center_history.get(track_id, [])

        if len(history) < 2:
            return 0, 0

        previous_x, previous_y = history[-2]
        current_x, current_y = history[-1]

        dx = current_x - previous_x
        dy = current_y - previous_y

        if fps <= 0:
            return 0, 0

        vx = dx * fps
        vy = dy * fps

        return vx, vy

    def calculate_center_acceleration(
            self,
            track_id,
            fps
        ):

        vx, vy = self.calculate_center_velocity_vector(
            track_id,
            fps
        )

        if track_id not in self.velocity_history:
            self.velocity_history[track_id] = []

        self.velocity_history[track_id].append(
            (vx, vy)
        )

        if len(self.velocity_history[track_id]) > 30:
            self.velocity_history[track_id].pop(0)

        history = self.velocity_history[track_id]

        if len(history) < 2:
            return 0

        previous_vx, previous_vy = history[-2]

        current_vx, current_vy = history[-1]

        delta_vx = current_vx - previous_vx
        delta_vy = current_vy - previous_vy

        acceleration = math.hypot(
            delta_vx,
            delta_vy
        ) * fps

        return acceleration