"""
YOLO Training Script using Ultralytics
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List

import yaml
from ultralytics import YOLO


def prepare_yolo_dataset(
    images_root: str,
    labels_root: str,
    splits_dir: str,
    output_dir: str,
) -> str:
    """
    Prepare YOLO dataset.yaml file
    
    Expected structure:
    - images_root/
        - BLF2001/image1.jpg
        - BLF2002/image2.jpg
    - labels_root/
        - BLF2001/image1.txt  (YOLO format: class x_center y_center width height)
        - BLF2002/image2.txt
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset.yaml
    dataset_config = {
        "path": str(Path(images_root).parent.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "cattle"},
    }
    
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f)
    
    print(f"Created dataset config: {yaml_path}")
    return str(yaml_path)


def create_yolo_labels_from_images(
    images_root: str,
    output_labels_dir: str,
):
    """
    Create placeholder YOLO labels (full image bounding box)
    This is a temporary solution until you have real annotations
    """
    from pathlib import Path
    
    images_root = Path(images_root)
    output_dir = Path(output_labels_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    count = 0
    for img_path in images_root.rglob("*"):
        if img_path.suffix.lower() in extensions:
            # Create label file with same structure
            rel_path = img_path.relative_to(images_root)
            label_path = output_dir / rel_path.with_suffix(".txt")
            label_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Placeholder: full image box (will need real annotations)
            # Format: class x_center y_center width height (normalized 0-1)
            with open(label_path, "w") as f:
                f.write("0 0.5 0.5 0.9 0.9\n")  # Cattle roughly centered
            
            count += 1
    
    print(f"Created {count} placeholder labels in {output_dir}")
    print("NOTE: Replace these with real annotations for training!")


def train(config_path: str):
    """Main YOLO training function"""
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("YOLO TRAINING")
    print("=" * 60)
    
    # Initialize model
    model_name = config["model"]["name"]
    print(f"Model: {model_name}")
    
    if config["model"]["pretrained"]:
        # Load pretrained model
        model = YOLO(f"{model_name}.pt")
        print("Loaded pretrained weights")
    else:
        # Train from scratch
        model = YOLO(f"{model_name}.yaml")
        print("Training from scratch")
    
    # Check if dataset.yaml exists
    dataset_yaml = Path("data/yolo_dataset/dataset.yaml")
    
    if not dataset_yaml.exists():
        print("\n⚠️  No YOLO dataset found!")
        print("Creating placeholder dataset structure...")
        
        # Create placeholder labels
        create_yolo_labels_from_images(
            images_root=config["data"]["images_root"],
            output_labels_dir=config["data"]["labels_root"],
        )
        
        # Prepare dataset.yaml
        prepare_yolo_dataset(
            images_root=config["data"]["images_root"],
            labels_root=config["data"]["labels_root"],
            splits_dir="data/splits",
            output_dir="data/yolo_dataset",
        )
        
        dataset_yaml = Path("data/yolo_dataset/dataset.yaml")
    
    # Output directory
    output_dir = Path(config["output"]["save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    print("\nStarting training...")
    
    results = model.train(
        data=str(dataset_yaml),
        epochs=config["training"]["epochs"],
        batch=config["training"]["batch_size"],
        imgsz=config["training"]["img_size"],
        patience=config["training"]["patience"],
        project=str(output_dir),
        name=config["output"]["name"],
        
        # Optimizer
        optimizer="AdamW",
        lr0=config["training"]["optimizer"]["lr"],
        weight_decay=config["training"]["optimizer"]["weight_decay"],
        
        # Augmentation
        hsv_h=config["training"]["augmentation"]["hsv_h"],
        hsv_s=config["training"]["augmentation"]["hsv_s"],
        hsv_v=config["training"]["augmentation"]["hsv_v"],
        degrees=config["training"]["augmentation"]["degrees"],
        translate=config["training"]["augmentation"]["translate"],
        scale=config["training"]["augmentation"]["scale"],
        flipud=config["training"]["augmentation"]["flipud"],
        fliplr=config["training"]["augmentation"]["fliplr"],
        mosaic=config["training"]["augmentation"]["mosaic"],
        mixup=config["training"]["augmentation"]["mixup"],
        
        # Other
        verbose=True,
        exist_ok=True,
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
    
    return results


def train_on_roboflow_dataset(
    dataset_url: str = "https://universe.roboflow.com/cattledetector/cattlesegment-60nea",
    model_name: str = "yolov8m",
    epochs: int = 100,
):
    """
    Train YOLO on Roboflow CattleSegment dataset
    
    Steps:
    1. Go to the Roboflow URL
    2. Download in YOLO format
    3. Place in data/roboflow/
    4. Run this function
    """
    print("=" * 60)
    print("TRAINING ON ROBOFLOW DATASET")
    print("=" * 60)
    print(f"\nDataset URL: {dataset_url}")
    print("\nInstructions:")
    print("1. Visit the URL above")
    print("2. Click 'Download Dataset'")
    print("3. Select 'YOLOv8' format")
    print("4. Extract to data/roboflow/")
    print("5. Run this script again")
    
    roboflow_dir = Path("data/roboflow")
    dataset_yaml = roboflow_dir / "data.yaml"
    
    if not dataset_yaml.exists():
        print(f"\n❌ Dataset not found at {roboflow_dir}")
        print("Please download and extract the dataset first.")
        return
    
    # Train
    model = YOLO(f"{model_name}.pt")
    
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=16,
        imgsz=640,
        project="models/yolo",
        name="roboflow_cattle",
        exist_ok=True,
    )
    
    print("\nTraining complete!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/yolo_config.yaml", help="Config file path")
    parser.add_argument("--roboflow", action="store_true", help="Train on Roboflow dataset")
    args = parser.parse_args()
    
    if args.roboflow:
        train_on_roboflow_dataset()
    else:
        train(args.config)
