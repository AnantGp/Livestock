"""
Generate masks using SAM's AUTOMATIC mask generation mode
This doesn't use prompts - SAM finds all objects automatically
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json


def load_sam_automatic():
    """Load SAM for automatic mask generation"""
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        print("Installing segment-anything...")
        import subprocess
        subprocess.run(["pip", "install", "segment-anything"], check=True)
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    
    checkpoint_path = Path("models/sam/sam_vit_b_01ec64.pth")
    
    if not checkpoint_path.exists():
        print(f"Downloading SAM checkpoint...")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            checkpoint_path
        )
    
    print("Loading SAM automatic mask generator...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))
    sam.to(device)
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=500,  # Minimum pixels
    )
    
    print(f"SAM automatic mask generator ready on {device}")
    return mask_generator


def process_image(image_path: str, mask_generator, output_dir: Path):
    """Generate mask for one image using automatic mode"""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Generate all masks
    masks = mask_generator.generate(image_rgb)
    
    if not masks:
        return None
    
    # Sort by area, pick largest (should be cattle)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    # Find best mask (largest with coverage 15-60%)
    best_mask = None
    for m in masks[:5]:  # Check top 5
        coverage = m['area'] / (h * w)
        if 0.15 <= coverage <= 0.60:
            best_mask = m['segmentation']
            break
    
    # Fallback to largest
    if best_mask is None and len(masks) > 0:
        best_mask = masks[0]['segmentation']
    
    if best_mask is None:
        return None
    
    # Save mask
    relative_path = Path(image_path).relative_to("images")
    mask_path = output_dir / relative_path.parent / f"{Path(image_path).stem}_mask.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    
    mask = best_mask.astype(np.uint8) * 255
    cv2.imwrite(str(mask_path), mask)
    
    # Save viz
    viz_path = output_dir / relative_path.parent / f"{Path(image_path).stem}_viz.jpg"
    viz = image.copy()
    viz[best_mask] = viz[best_mask] * 0.5 + np.array([0, 255, 0]) * 0.5
    cv2.imwrite(str(viz_path), viz)
    
    coverage = best_mask.sum() / (h * w) * 100
    return str(image_path), str(mask_path), coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="images", help="Image directory")
    parser.add_argument("--output", default="data/sam_masks_auto", help="Output directory")
    parser.add_argument("--max-images", type=int, help="Limit images")
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load SAM
    mask_generator = load_sam_automatic()
    
    # Find images
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(Path(args.images).rglob(ext))
    
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    
    print(f"\nProcessing {len(image_paths)} images with SAM automatic mode...")
    
    # Process
    results = []
    coverages = []
    
    for img_path in tqdm(image_paths, desc="Generating masks"):
        try:
            result = process_image(str(img_path), mask_generator, output_path)
            if result:
                img, mask, cov = result
                results.append((img, mask))
                coverages.append(cov)
        except Exception as e:
            print(f"\nError on {img_path}: {e}")
            continue
    
    # Stats
    stats = {
        "total": len(image_paths),
        "success": len(results),
        "failed": len(image_paths) - len(results),
        "avg_coverage": float(np.mean(coverages)) if coverages else 0,
    }
    
    with open(output_path / "generation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Success: {stats['success']}/{stats['total']}")
    print(f"Average coverage: {stats['avg_coverage']:.1f}%")
    print(f"Masks saved to: {output_path}")
    print(f"{'='*60}")
    
    # Create splits
    print("\nCreating train/val/test splits...")
    np.random.seed(42)
    np.random.shuffle(results)
    
    n = len(results)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    splits = {
        "train": results[:n_train],
        "val": results[n_train:n_train + n_val],
        "test": results[n_train + n_val:],
    }
    
    splits_dir = Path("data/sam_auto_splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, pairs in splits.items():
        with open(splits_dir / f"{split_name}.txt", "w") as f:
            for img, mask in pairs:
                f.write(f"{img}\t{mask}\n")
        print(f"{split_name}: {len(pairs)} pairs")
    
    print(f"\nSplits saved to: {splits_dir}")


if __name__ == "__main__":
    main()
