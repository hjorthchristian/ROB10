import cv2
import time
import torch
import numpy as np
from ultralytics import FastSAM

# Load the FastSAM model
model = FastSAM("FastSAM-s.pt")

# Open the webcam (0 is default camera)
cap = cv2.VideoCapture(0)

# Set webcam resolution (optional)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

prev_time = time.time()  # For FPS calculation

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert BGR frame to RGB as FastSAM expects RGB input
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run FastSAM inference
    results = model(rgb_frame, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Process results: FastSAM returns masks and boxes
    if results and results[0].masks is not None:
        masks = results[0].masks.data  # Segmentation masks
        boxes = results[0].boxes.xyxy  # Bounding boxes
        scores = results[0].boxes.conf  # Confidence scores
        classes = results[0].boxes.cls  # Class IDs

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = scores[idx].item()
            cls_id = int(classes[idx].item())
            label = f"{model.names[cls_id]} {conf:.2f}"

            if conf < 0.8:  # Filter low confidence detections
                continue

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Apply segmentation mask (optional)
            mask = masks[idx].cpu().numpy().astype('uint8') * 255  # Convert to 8-bit
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))  # Resize mask to match frame

            # Ensure mask has 3 channels for blending
            if len(mask.shape) == 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Apply color map for visualization
            colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

            # Blend mask with original frame
            frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)

    # FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("FastSAM Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
