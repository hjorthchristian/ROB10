import cv2
import time
import torch
from ultralytics import YOLO
import os


# Load the YOLOv11 model
model = YOLO("C:/Users/Christian/Documents/University/ROB10/Code/runs/detect/train7/weights/best.pt")  # Change path if needed
cardboard_folder_path = "C:/Users/Christian/Documents/University/ROB10/Code/cardboard_boxes"

for file in os.listdir(cardboard_folder_path):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        frame = cv2.imread(os.path.join(cardboard_folder_path, file))
        
        if frame is None:
            print("Failed to grab frame.")
            continue
    

    # Run YOLOv11 inference
    results = model(frame)

    # Process results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            if conf < 0.45:  # Skip boxes with low confidence
                continue
            cls = int(box.cls[0])  # Class ID
            label = f"{model.names[cls]} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Show the frame
    cv2.imshow("YOLOv11 Object Detection", frame)

    key = cv2.waitKey(0)
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources

cv2.destroyAllWindows()
