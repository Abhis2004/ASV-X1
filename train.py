import os
import sys
from pathlib import Path
from ultralytics import YOLO

# Import dataset preparer if needed
from prepare_dataset import prepare_merged_dataset

def main():
    print("=" * 60)
    print("ASV-X1 DRONE MODEL TRAINING (YOLOv8s)")
    print("=" * 60)

    project_root = Path(__file__).parent.resolve()
    dataset_yaml = project_root / "dataset" / "merged_dataset" / "merged_data.yaml"

    # Step 1: Ensure dataset is prepared and merged
    if not dataset_yaml.exists():
        print(f"[INFO] Merged dataset YAML not found at {dataset_yaml}")
        print("[INFO] Preparing merged VisDrone + UAVDT dataset...")
        prepare_merged_dataset(
            base_dir=project_root / "dataset",
            output_dir=project_root / "dataset" / "merged_dataset"
        )
    else:
        print(f"[INFO] Using existing merged dataset configuration: {dataset_yaml}")

    # Step 2: Initialize YOLOv8s Model
    print("\n[INFO] Initializing YOLOv8s model...")
    model = YOLO("yolov8s.pt")

    # Step 3: Start Training with Specified Drone Hyperparameters
    print("\n[INFO] Starting training pipeline...")
    results = model.train(
        data=str(dataset_yaml.resolve().as_posix()),
        epochs=150,
        imgsz=640,
        batch=16,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        close_mosaic=15,
        degrees=10.0,
        scale=0.5,
        mosaic=1.0,
        mixup=0.15,
        project=str((project_root / "runs" / "detect").resolve().as_posix()),
        name="yolov8s_visdrone_uavdt",
        exist_ok=True,
    )

    print("=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"Best model saved to: runs/detect/yolov8s_visdrone_uavdt/weights/best.pt")
    print("=" * 60)

if __name__ == "__main__":
    main()
