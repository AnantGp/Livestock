"""
Data Preparation Script
Organizes images into train/val/test splits for YOLO and U-Net training
"""

import os
import sys
import shutil
import random
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
from tqdm import tqdm


def find_all_images(root_dir: str) -> List[str]:
    """Find all images recursively"""
    root = Path(root_dir)
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    images = []
    for ext in extensions:
        images.extend(root.rglob(f"*{ext}"))
        images.extend(root.rglob(f"*{ext.upper()}"))
    
    return [str(p) for p in images]


def create_train_val_test_split(
    image_paths: List[str],
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    copy_files: bool = False,
) -> Dict[str, List[str]]:
    """
    Create train/val/test splits
    
    Args:
        image_paths: List of image paths
        output_dir: Output directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
        copy_files: Whether to copy files (True) or just create text files (False)
    """
    random.seed(seed)
    
    # Shuffle
    paths = image_paths.copy()
    random.shuffle(paths)
    
    # Calculate split sizes
    n = len(paths)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    # Split
    train_paths = paths[:n_train]
    val_paths = paths[n_train:n_train + n_val]
    test_paths = paths[n_train + n_val:]
    
    splits = {
        "train": train_paths,
        "val": val_paths,
        "test": test_paths,
    }
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as text files
    for split_name, split_paths in splits.items():
        txt_path = output_dir / f"{split_name}.txt"
        with open(txt_path, "w") as f:
            f.write("\n".join(split_paths))
        print(f"Created {txt_path} with {len(split_paths)} images")
    
    # Optionally copy files
    if copy_files:
        for split_name, split_paths in splits.items():
            split_dir = output_dir / split_name / "images"
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for src_path in tqdm(split_paths, desc=f"Copying {split_name}"):
                dst_path = split_dir / Path(src_path).name
                shutil.copy2(src_path, dst_path)
    
    return splits


def organize_for_yolo(
    images_root: str,
    labels_root: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    """
    Organize data into YOLO format:
    
    output_dir/
        images/
            train/
            val/
            test/
        labels/
            train/
            val/
            test/
        dataset.yaml
    """
    images_root = Path(images_root)
    labels_root = Path(labels_root)
    output_dir = Path(output_dir)
    
    # Find all images with corresponding labels
    valid_pairs = []
    
    # Handle flat structure (like our labeling folder)
    if images_root.name == "images" and images_root.parent.name == "yolo":
        # Flat structure: data/labeling/yolo/images/*.jpg
        for img_path in images_root.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                label_path = labels_root / f"{img_path.stem}.txt"
                if label_path.exists():
                    valid_pairs.append((str(img_path), str(label_path)))
    else:
        # Nested structure: images/BLF2001/*.jpg
        for img_path in find_all_images(str(images_root)):
            img_path = Path(img_path)
            # Try direct filename match first
            label_path = labels_root / f"{img_path.stem}.txt"
            if label_path.exists():
                valid_pairs.append((str(img_path), str(label_path)))
            else:
                # Try with relative path
                rel_path = img_path.relative_to(images_root)
                label_path = labels_root / rel_path.with_suffix(".txt")
                if label_path.exists():
                    valid_pairs.append((str(img_path), str(label_path)))
    
    print(f"Found {len(valid_pairs)} image-label pairs")
    
    if len(valid_pairs) == 0:
        print("No valid pairs found. Creating placeholder labels...")
        # Create placeholder labels
        for img_path in find_all_images(str(images_root))[:1000]:  # Limit for speed
            img_path = Path(img_path)
            rel_path = img_path.relative_to(images_root)
            label_path = labels_root / rel_path.with_suffix(".txt")
            label_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Placeholder: full image bbox
            with open(label_path, "w") as f:
                f.write("0 0.5 0.5 0.9 0.9\n")
            
            valid_pairs.append((str(img_path), str(label_path)))
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(valid_pairs)
    
    n = len(valid_pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    splits = {
        "train": valid_pairs[:n_train],
        "val": valid_pairs[n_train:n_train + n_val],
        "test": valid_pairs[n_train + n_val:],
    }
    
    # Create directory structure and copy files
    for split_name, pairs in splits.items():
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_src, lbl_src in tqdm(pairs, desc=f"Creating {split_name}"):
            img_dst = img_dir / Path(img_src).name
            lbl_dst = lbl_dir / Path(lbl_src).name
            
            shutil.copy2(img_src, img_dst)
            shutil.copy2(lbl_src, lbl_dst)
    
    # Create dataset.yaml
    dataset_yaml = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "cattle"},
    }
    
    import yaml
    with open(output_dir / "dataset.yaml", "w") as f:
        yaml.dump(dataset_yaml, f)
    
    print(f"\nYOLO dataset created at {output_dir}")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val: {len(splits['val'])}")
    print(f"  Test: {len(splits['test'])}")
    
    return output_dir / "dataset.yaml"


def create_sku_based_split(
    images_root: str,
    csv_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    """
    Create splits based on SKU (animal-level split)
    This ensures the same animal doesn't appear in both train and test
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    skus = df["sku"].tolist()
    
    # Shuffle SKUs
    random.seed(42)
    random.shuffle(skus)
    
    # Split SKUs
    n = len(skus)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_skus = set(skus[:n_train])
    val_skus = set(skus[n_train:n_train + n_val])
    test_skus = set(skus[n_train + n_val:])
    
    # Find images for each split
    images_root = Path(images_root)
    splits = {"train": [], "val": [], "test": []}
    
    for sku in tqdm(skus, desc="Organizing by SKU"):
        # Find folder (handle both "BLF2001" and "BLF 2001" formats)
        folder = images_root / sku
        if not folder.exists():
            folder = images_root / f"{sku[:3]} {sku[3:]}"
        
        if not folder.exists():
            continue
        
        # Get images
        images = find_all_images(str(folder))
        
        # Assign to split
        if sku in train_skus:
            splits["train"].extend(images)
        elif sku in val_skus:
            splits["val"].extend(images)
        else:
            splits["test"].extend(images)
    
    # Save splits
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, paths in splits.items():
        txt_path = output_dir / f"{split_name}.txt"
        with open(txt_path, "w") as f:
            f.write("\n".join(paths))
        print(f"Created {txt_path} with {len(paths)} images")
    
    # Save SKU splits for reference
    with open(output_dir / "sku_splits.txt", "w") as f:
        f.write(f"Train SKUs ({len(train_skus)}): {sorted(train_skus)}\n")
        f.write(f"Val SKUs ({len(val_skus)}): {sorted(val_skus)}\n")
        f.write(f"Test SKUs ({len(test_skus)}): {sorted(test_skus)}\n")
    
    return splits


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument("--images", default="images", help="Images root directory")
    parser.add_argument("--csv", default="dataset.csv", help="CSV metadata file")
    parser.add_argument("--output", default="data/splits", help="Output directory")
    parser.add_argument("--yolo", action="store_true", help="Create YOLO format dataset")
    parser.add_argument("--sku-split", action="store_true", help="Split by SKU (animal-level)")
    
    args = parser.parse_args()
    
    if args.yolo:
        organize_for_yolo(
            images_root=args.images,
            labels_root="data/labels",
            output_dir="data/yolo_dataset",
        )
    elif args.sku_split:
        create_sku_based_split(
            images_root=args.images,
            csv_path=args.csv,
            output_dir=args.output,
        )
    else:
        # Simple split
        images = find_all_images(args.images)
        print(f"Found {len(images)} images")
        
        create_train_val_test_split(
            image_paths=images,
            output_dir=args.output,
        )


if __name__ == "__main__":
    main()
