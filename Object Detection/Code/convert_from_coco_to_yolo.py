import os
from pathlib import Path
from ultralytics.data.converter import convert_coco


def main():
    base_path = Path("C:/Users/Christian/Downloads")
    extract_path = base_path / "coco_style_oneclass"
    output_path = base_path / "yolo_style_dataset"
    
    
    convert_coco(
        labels_dir=extract_path / "annotations/instances_train2017",
    )
    
    convert_coco(
        labels_dir=extract_path / "annotations/instances_val2017",
    )
    
    print("COCO to YOLO conversion complete!")

if __name__ == "__main__":
    main()
