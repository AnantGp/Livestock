"""
Generate placeholder labels for testing the training pipeline
Creates YOLO boxes and U-Net masks automatically
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def create_placeholder_yolo_labels(images_dir: str, labels_dir: str):
    """Create placeholder YOLO labels (centered boxes)"""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
    
    print(f"Creating {len(image_files)} YOLO placeholder labels...")
    
    for img_path in tqdm(image_files):
        # Create label file
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        # Placeholder: cattle roughly in center, taking up 70-90% of image
        x_center = 0.5 + np.random.uniform(-0.1, 0.1)
        y_center = 0.5 + np.random.uniform(-0.1, 0.1)
        width = np.random.uniform(0.7, 0.9)
        height = np.random.uniform(0.7, 0.9)
        
        # Ensure bounds
        x_center = np.clip(x_center, width/2, 1 - width/2)
        y_center = np.clip(y_center, height/2, 1 - height/2)
        
        with open(label_path, "w") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"Created labels in {labels_dir}")


def create_placeholder_masks(images_dir: str, masks_dir: str):
    """Create placeholder masks using GrabCut + thresholding"""
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir) 
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
    
    print(f"Creating {len(image_files)} U-Net placeholder masks...")
    
    for img_path in tqdm(image_files):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Try GrabCut first
        mask = np.zeros((h, w), np.uint8)
        
        # Define rectangle for GrabCut (assume cattle in center 80%)
        margin_w = int(w * 0.1)
        margin_h = int(h * 0.1)
        rect = (margin_w, margin_h, w - 2*margin_w, h - 2*margin_h)
        
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
            
            # If mask is too small or large, fallback to simple method
            coverage = mask.sum() / mask.size
            if coverage < 0.1 or coverage > 0.95:
                raise Exception("GrabCut failed")
                
        except:
            # Fallback: simple thresholding + morphology
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Adaptive threshold
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_MEAN_C, cv2.THRESH_BINARY, 15, 8)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Find largest contour
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                mask = np.zeros((h, w), np.uint8)
                cv2.fillPoly(mask, [largest], 1)
            else:
                # Last resort: ellipse in center
                mask = np.zeros((h, w), np.uint8)
                center = (w//2, h//2)
                axes = (int(w*0.3), int(h*0.35))
                cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
        
        # Save mask (0=background, 255=cattle)
        mask_path = masks_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(mask_path), mask * 255)
    
    print(f"Created masks in {masks_dir}")


if __name__ == "__main__":
    # Create placeholder labels for YOLO
    create_placeholder_yolo_labels("data/labeling/yolo/images", "data/labels")
    
    # Create placeholder masks for U-Net
    create_placeholder_masks("data/labeling/unet/images", "data/masks")
    
    print("\nPlaceholder labels created! Now you can:")
    print("1. python3 scripts/prepare_data.py --yolo")
    print("2. python3 src/training/train_yolo.py")  
    print("3. python3 src/training/train_unet.py")