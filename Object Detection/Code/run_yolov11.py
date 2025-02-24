import cv2
import time
import torch
from ultralytics import YOLO

# Load the YOLOv11 model
model = YOLO("C:/Users/Christian/Documents/University/ROB10/Code/runs/detect/train7/weights/best.pt")  # Change path if needed

# Open the webcam (0 is default camera)
cap = cv2.VideoCapture(0)

# Set webcam resolution (optional)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

prev_time = 0  # For FPS calculation

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLOv11 inference
    results = model(frame)

    # Process results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            if conf < 0.75:  # Skip boxes with low confidence
                continue
            cls = int(box.cls[0])  # Class ID
            label = f"{model.names[cls]} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("YOLOv11 Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
