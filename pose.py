from ultralytics import YOLO

class PoseEstimator:
    def __init__(self):
        self.model = YOLO("yolo26n-pose.pt")

    def estimate(self,frame):
        results = self.model(frame)
        result = results[0]
        return result.keypoints.xy

    def process(self, frame):
        results = self.model(frame)

        result = results[0]
        annotated_frame = results[0].plot()
        keypoints = result.keypoints.xy
        confidence = result.keypoints.conf
        boxes = result.boxes.xyxy

        return annotated_frame,keypoints,confidence, boxes
