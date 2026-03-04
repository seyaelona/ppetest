import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model (downloads automatically first time)
model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model.track(frame, persist=True, classes=[0])  # class 0 = person

    # Draw boxes on frame
    annotated_frame = results[0].plot()

    cv2.imshow("Person Tracker", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
