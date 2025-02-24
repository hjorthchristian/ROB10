
import os
from ultralytics import YOLO

def train_yolo(data_yaml, model_path, imgsz=640, epochs=50, batch_size=16, device='cuda'):
  
    model = YOLO(model_path)  # Load YOLOv11 model
    
    model.train(
        data=data_yaml,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch_size,
        device=device,
        workers=4  # Number of CPU workers for loading data
    )
    
    print("Training complete!")

if __name__ == "__main__":
    base_path = "C:/Users/Christian/Downloads/yolo_style_dataset_carton"
    data_yaml = "C:/Users/Christian/Documents/University/ROB10/Code/dataset.yaml"  
    model_path = "yolo11s.pt"  
    print("Training YOLOv11 model...")
    train_yolo(
        data_yaml=data_yaml,
        model_path=model_path,
        imgsz=640,
        epochs=50,
        batch_size=16,
        device='cuda'  # or 'cpu
    )
    print("Training complete!")
