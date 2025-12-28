"""
Convert segmentation masks to YOLO bounding box format
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm


def mask_to_bbox(mask: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Convert binary mask to YOLO bounding box format
    
    Args:
        mask: Binary mask (H, W) with 1 for object, 0 for background
    
    Returns:
        (x_center, y_center, width, height) normalized to [0, 1]
    """
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get bounding box of largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Normalize to [0, 1]
    img_h, img_w = mask.shape
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h
    
    return x_center, y_center, width, height


def mask_to_polygon(mask: np.ndarray, simplify: bool = True) -> List[List[float]]:
    """
    Convert binary mask to YOLO polygon format
    
    Args:
        mask: Binary mask
        simplify: Whether to simplify the polygon
    
    Returns:
        List of normalized polygon points
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    img_h, img_w = mask.shape
    polygons = []
    
    for contour in contours:
        if simplify:
            # Simplify contour
            epsilon = 0.01 * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Flatten and normalize
        points = contour.flatten().tolist()
        normalized = []
        for i in range(0, len(points), 2):
            normalized.append(points[i] / img_w)
            normalized.append(points[i + 1] / img_h)
        
        polygons.append(normalized)
    
    return polygons


def convert_masks_to_yolo(
    masks_dir: str,
    output_dir: str,
    class_id: int = 0,
    use_polygons: bool = False,
):
    """
    Convert mask images to YOLO label format
    
    Args:
        masks_dir: Directory containing mask images
        output_dir: Output directory for YOLO labels
        class_id: Class ID for the object
        use_polygons: Use polygon format instead of bboxes
    """
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all mask images
    mask_files = list(masks_dir.rglob("*.png")) + list(masks_dir.rglob("*.jpg"))
    
    print(f"Converting {len(mask_files)} masks to YOLO format")
    
    for mask_path in tqdm(mask_files):
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            continue
        
        # Binarize
        mask = (mask > 127).astype(np.uint8)
        
        # Convert
        if use_polygons:
            annotations = mask_to_polygon(mask)
            if annotations:
                # Create label file
                label_path = output_dir / mask_path.with_suffix(".txt").name
                with open(label_path, "w") as f:
                    for polygon in annotations:
                        line = f"{class_id} " + " ".join(map(str, polygon))
                        f.write(line + "\n")
        else:
            bbox = mask_to_bbox(mask)
            if bbox:
                x_center, y_center, width, height = bbox
                
                # Create label file
                label_path = output_dir / mask_path.with_suffix(".txt").name
                with open(label_path, "w") as f:
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def convert_coco_to_yolo(
    coco_json_path: str,
    output_dir: str,
):
    """
    Convert COCO format annotations to YOLO format
    
    Args:
        coco_json_path: Path to COCO JSON file
        output_dir: Output directory for YOLO labels
    """
    import json
    
    with open(coco_json_path, "r") as f:
        coco = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build mappings
    image_info = {img["id"]: img for img in coco["images"]}
    category_map = {cat["id"]: i for i, cat in enumerate(coco["categories"])}
    
    # Group annotations by image
    ann_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in ann_by_image:
            ann_by_image[img_id] = []
        ann_by_image[img_id].append(ann)
    
    # Convert each image
    for img_id, img_info in tqdm(image_info.items()):
        img_w, img_h = img_info["width"], img_info["height"]
        file_name = Path(img_info["file_name"]).stem
        
        annotations = ann_by_image.get(img_id, [])
        
        label_path = output_dir / f"{file_name}.txt"
        with open(label_path, "w") as f:
            for ann in annotations:
                # Get class ID
                class_id = category_map.get(ann["category_id"], 0)
                
                # Convert bbox from COCO [x, y, w, h] to YOLO [x_center, y_center, w, h]
                x, y, w, h = ann["bbox"]
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                width = w / img_w
                height = h / img_h
                
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    print(f"Converted {len(image_info)} images to YOLO format")


def generate_pseudo_labels_with_sam(
    images_dir: str,
    output_dir: str,
    sam_checkpoint: str = "sam_vit_h_4b8939.pth",
    device: str = "cpu",
):
    """
    Generate pseudo labels using SAM (Segment Anything Model)
    
    NOTE: Requires segment-anything package
    pip install segment-anything
    
    Args:
        images_dir: Directory with images
        output_dir: Output directory for labels
        sam_checkpoint: Path to SAM checkpoint
        device: Device to use
    """
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        print("Please install segment-anything: pip install segment-anything")
        return
    
    # Load SAM
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    extensions = {".jpg", ".jpeg", ".png"}
    images = [p for p in images_dir.rglob("*") if p.suffix.lower() in extensions]
    
    print(f"Processing {len(images)} images with SAM")
    
    for img_path in tqdm(images):
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image
        predictor.set_image(image)
        
        # Use center point as prompt (assume cattle in center)
        h, w = image.shape[:2]
        center_point = np.array([[w // 2, h // 2]])
        center_label = np.array([1])  # Foreground
        
        # Predict
        masks, scores, _ = predictor.predict(
            point_coords=center_point,
            point_labels=center_label,
            multimask_output=True,
        )
        
        # Use best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        
        # Convert to YOLO format
        bbox = mask_to_bbox(mask.astype(np.uint8))
        
        if bbox:
            x_center, y_center, width, height = bbox
            label_path = output_dir / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                f.write(f"0 {x_center} {y_center} {width} {height}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert masks to YOLO format")
    parser.add_argument("--masks", type=str, help="Masks directory")
    parser.add_argument("--coco", type=str, help="COCO JSON file")
    parser.add_argument("--output", default="data/labels", help="Output directory")
    parser.add_argument("--polygons", action="store_true", help="Use polygon format")
    
    args = parser.parse_args()
    
    if args.coco:
        convert_coco_to_yolo(args.coco, args.output)
    elif args.masks:
        convert_masks_to_yolo(args.masks, args.output, use_polygons=args.polygons)
    else:
        print("Please provide --masks or --coco argument")
