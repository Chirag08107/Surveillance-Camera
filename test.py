import cv2
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")

cap = cv2.VideoCapture("videos/test.mp4")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    result = results[0]

    keypoints = result.keypoints.xy

    if keypoints is not None:

        for person in keypoints:

            points = person.cpu().tolist()

            print("Person keypoints:")
            print(points)

    cv2.imshow("Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()