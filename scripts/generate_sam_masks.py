"""
Generate high-quality segmentation masks using SAM (Segment Anything Model)
Then use these masks to retrain U-Net for better accuracy (>90% IoU)

This is knowledge distillation: SAM (large, slow) → U-Net (small, fast)
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_sam_model(model_type: str = "vit_b", device: str = "cuda"):
    """Load SAM model"""
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        print("Installing segment-anything...")
        import subprocess
        subprocess.run(["pip", "install", "segment-anything"], check=True)
        from segment_anything import sam_model_registry, SamPredictor
    
    checkpoint_map = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth",
    }
    
    checkpoint_path = Path("models/sam") / checkpoint_map[model_type]
    
    if not checkpoint_path.exists():
        print(f"Downloading SAM checkpoint to {checkpoint_path}...")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://dl.fbaipublicfiles.com/segment_anything/{checkpoint_map[model_type]}"
        import urllib.request
        urllib.request.urlretrieve(url, checkpoint_path)
    
    print(f"Loading SAM ({model_type})...")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device)
    sam.eval()
    
    predictor = SamPredictor(sam)
    print(f"SAM loaded on {device}")
    return predictor


def load_yolo_model(model_path: str = "models/yolo/exp/weights/best.pt"):
    """Load YOLO model for cattle detection"""
    from ultralytics import YOLO
    
    if Path(model_path).exists():
        print(f"Loading YOLO from {model_path}")
        model = YOLO(model_path)
    else:
        print("Loading pretrained YOLOv8n...")
        model = YOLO("yolov8n.pt")
    
    return model


def generate_sam_mask(
    image: np.ndarray,
    predictor,
    box: np.ndarray = None,
    multimask: bool = True,
) -> np.ndarray:
    """
    Generate segmentation mask using SAM
    
    Args:
        image: RGB image (H, W, 3)
        predictor: SAM predictor
        box: Optional bounding box [x1, y1, x2, y2]
        multimask: If True, returns best of 3 masks
    
    Returns:
        Binary mask (H, W)
    """
    predictor.set_image(image)
    h, w = image.shape[:2]
    
    if box is not None:
        # Expand box by 10% for better context
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        expand_x = box_w * 0.1
        expand_y = box_h * 0.1
        expanded_box = np.array([
            max(0, box[0] - expand_x),
            max(0, box[1] - expand_y),
            min(w, box[2] + expand_x),
            min(h, box[3] + expand_y)
        ])
        
        # Use box + center point for better results
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        point_coords = np.array([[center_x, center_y]])
        point_labels = np.array([1])  # Foreground
        
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=expanded_box,
            multimask_output=True,
        )
        
        # Select LARGEST mask (cattle should be the main object)
        mask_areas = [m.sum() for m in masks]
        best_idx = np.argmax(mask_areas)
        mask = masks[best_idx]
    else:
        # No box - use automatic mask generation with multiple points
        # Sample 5 points in center region
        center_points = np.array([
            [w // 2, h // 2],           # Center
            [w // 2, h // 3],           # Upper center
            [w // 2, 2 * h // 3],       # Lower center
            [w // 3, h // 2],           # Left center
            [2 * w // 3, h // 2],       # Right center
        ])
        
        best_mask = None
        best_area = 0
        
        for point in center_points:
            masks, scores, _ = predictor.predict(
                point_coords=np.array([point]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
            # Get largest mask from this point
            for m in masks:
                area = m.sum()
                # Prefer masks with 15-60% coverage (typical cattle size)
                coverage = area / (h * w)
                if 0.15 <= coverage <= 0.60 and area > best_area:
                    best_area = area
                    best_mask = m
        
        # Fallback if no good mask found
        if best_mask is None:
            masks, scores, _ = predictor.predict(
                point_coords=np.array([[w // 2, h // 2]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
            mask_areas = [m.sum() for m in masks]
            best_idx = np.argmax(mask_areas)
            best_mask = masks[best_idx]
        
        mask = best_mask
    
    return mask.astype(np.uint8)


def refine_mask_morphology(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply morphological operations to clean up mask"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Keep only largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        # Find largest component (excluding background)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8)
    
    return mask


def process_dataset(
    image_dir: str,
    output_dir: str,
    sam_model_type: str = "vit_b",
    use_yolo: bool = True,
    max_images: int = None,
    device: str = None,
):
    """
    Generate SAM masks for all images in dataset
    
    Args:
        image_dir: Directory containing images (with subfolders per cattle)
        output_dir: Directory to save masks
        sam_model_type: SAM model variant (vit_b, vit_l, vit_h)
        use_yolo: Use YOLO to detect cattle box first
        max_images: Limit number of images (for testing)
        device: cuda or cpu
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load models
    sam_predictor = load_sam_model(sam_model_type, device)
    yolo_model = load_yolo_model() if use_yolo else None
    
    # Find all images
    image_dir = Path(image_dir)
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(image_dir.rglob(ext))
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"\nProcessing {len(image_paths)} images...")
    
    # Stats
    stats = {
        "total": len(image_paths),
        "success": 0,
        "failed": 0,
        "avg_coverage": 0,
    }
    coverages = []
    
    # Process each image
    for img_path in tqdm(image_paths, desc="Generating SAM masks"):
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                stats["failed"] += 1
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Detect cattle with YOLO
            box = None
            if yolo_model is not None:
                results = yolo_model(image_rgb, verbose=False)
                if len(results) > 0 and len(results[0].boxes) > 0:
                    # Get highest confidence box
                    boxes = results[0].boxes
                    best_idx = boxes.conf.argmax()
                    box = boxes.xyxy[best_idx].cpu().numpy()
            
            # Generate SAM mask
            mask = generate_sam_mask(image_rgb, sam_predictor, box)
            
            # Refine mask
            mask = refine_mask_morphology(mask)
            
            # Calculate coverage
            coverage = mask.sum() / mask.size * 100
            coverages.append(coverage)
            
            # Skip if coverage is too low (failed segmentation)
            if coverage < 5.0:
                stats["failed"] += 1
                continue
            
            # Save mask
            # Maintain folder structure
            relative_path = img_path.relative_to(image_dir)
            mask_path = output_path / relative_path.parent / f"{img_path.stem}_mask.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as binary mask (0 or 255)
            cv2.imwrite(str(mask_path), mask * 255)
            
            # Also save a visualization
            viz_path = output_path / relative_path.parent / f"{img_path.stem}_viz.jpg"
            viz = image.copy()
            viz[mask == 1] = viz[mask == 1] * 0.5 + np.array([0, 255, 0]) * 0.5
            cv2.imwrite(str(viz_path), viz)
            
            stats["success"] += 1
            
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            stats["failed"] += 1
            continue
    
    # Calculate stats
    stats["avg_coverage"] = np.mean(coverages) if coverages else 0
    
    # Save stats
    stats_path = output_path / "generation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 60)
    print("SAM MASK GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total images: {stats['total']}")
    print(f"Successful: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"Average coverage: {stats['avg_coverage']:.1f}%")
    print(f"Masks saved to: {output_path}")
    print("=" * 60)
    
    return stats


def create_training_split(
    mask_dir: str,
    image_dir: str,
    output_dir: str = "data/sam_splits",
):
    """
    Create train/val/test split files for training with SAM masks
    """
    mask_dir = Path(mask_dir)
    image_dir = Path(image_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all mask files
    mask_files = list(mask_dir.rglob("*_mask.png"))
    
    # Create image-mask pairs
    pairs = []
    for mask_path in mask_files:
        # Find corresponding image
        relative = mask_path.relative_to(mask_dir)
        img_name = mask_path.stem.replace("_mask", "")
        
        # Try different extensions
        for ext in [".jpg", ".jpeg", ".png"]:
            img_path = image_dir / relative.parent / f"{img_name}{ext}"
            if img_path.exists():
                pairs.append((str(img_path), str(mask_path)))
                break
    
    print(f"Found {len(pairs)} image-mask pairs")
    
    # Shuffle and split (70/15/15)
    np.random.seed(42)
    np.random.shuffle(pairs)
    
    n = len(pairs)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]
    
    # Save split files
    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        with open(output_path / f"{split_name}.txt", "w") as f:
            for img, mask in split_pairs:
                f.write(f"{img}\t{mask}\n")
        print(f"{split_name}: {len(split_pairs)} pairs")
    
    print(f"\nSplit files saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SAM masks for U-Net training")
    parser.add_argument("--images", default="images", help="Image directory")
    parser.add_argument("--output", default="data/sam_masks", help="Output directory for masks")
    parser.add_argument("--sam-model", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--no-yolo", action="store_true", help="Don't use YOLO for detection")
    parser.add_argument("--max-images", type=int, help="Limit number of images")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--create-splits", action="store_true", help="Create train/val/test splits")
    
    args = parser.parse_args()
    
    # Generate masks
    stats = process_dataset(
        image_dir=args.images,
        output_dir=args.output,
        sam_model_type=args.sam_model,
        use_yolo=not args.no_yolo,
        max_images=args.max_images,
        device=args.device,
    )
    
    # Create splits if requested
    if args.create_splits:
        create_training_split(
            mask_dir=args.output,
            image_dir=args.images,
            output_dir="data/sam_splits",
        )
