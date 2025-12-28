"""
Dataset classes for YOLO and U-Net training
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentationDataset(Dataset):
    """Dataset for U-Net segmentation training"""
    
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        img_size: int = 256,
        augment: bool = True,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment
        
        # Augmentation pipeline
        if augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                ], p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Binarize mask (0 = background, 1 = cattle)
        mask = (mask > 127).astype(np.uint8)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"].long()
        
        return image, mask


class COCOSegmentationDataset(Dataset):
    """Dataset for COCO format annotations"""
    
    def __init__(
        self,
        images_dir: str,
        annotations_file: str,
        img_size: int = 256,
        augment: bool = True,
    ):
        self.images_dir = Path(images_dir)
        self.img_size = img_size
        
        # Load COCO annotations
        with open(annotations_file, "r") as f:
            self.coco = json.load(f)
        
        # Build image id to annotations mapping
        self.img_id_to_anns = {}
        for ann in self.coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)
        
        # Filter images that have annotations
        self.images = [
            img for img in self.coco["images"]
            if img["id"] in self.img_id_to_anns
        ]
        
        # Augmentation
        if augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def __len__(self) -> int:
        return len(self.images)
    
    def _create_mask_from_annotations(self, img_info: dict, annotations: List[dict]) -> np.ndarray:
        """Create binary mask from COCO polygon annotations"""
        h, w = img_info["height"], img_info["width"]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for ann in annotations:
            if "segmentation" in ann:
                for seg in ann["segmentation"]:
                    # Convert polygon to mask
                    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
        
        return mask
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_info = self.images[idx]
        
        # Load image
        img_path = self.images_dir / img_info["file_name"]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create mask from annotations
        annotations = self.img_id_to_anns[img_info["id"]]
        mask = self._create_mask_from_annotations(img_info, annotations)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"].long()
        
        return image, mask


class PseudoMaskDataset(Dataset):
    """
    Dataset that generates pseudo-masks using GrabCut
    For training when you don't have manual annotations
    """
    
    def __init__(
        self,
        image_paths: List[str],
        img_size: int = 256,
        augment: bool = True,
        cache_masks: bool = True,
        cache_dir: str = "data/pseudo_masks",
    ):
        self.image_paths = image_paths
        self.img_size = img_size
        self.cache_masks = cache_masks
        self.cache_dir = Path(cache_dir)
        
        if cache_masks:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Augmentation
        if augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def _generate_pseudo_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate pseudo-mask using GrabCut"""
        h, w = image.shape[:2]
        
        # Initialize mask
        mask = np.zeros((h, w), np.uint8)
        
        # Define rectangle (assume cattle in center)
        margin = int(min(h, w) * 0.1)
        rect = (margin, margin, w - 2 * margin, h - 2 * margin)
        
        # GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        except:
            # Fallback to simple threshold
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return mask
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Check cache
        cache_path = self.cache_dir / f"{Path(img_path).stem}_mask.png"
        
        if self.cache_masks and cache_path.exists():
            mask = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = self._generate_pseudo_mask(image)
            if self.cache_masks:
                cv2.imwrite(str(cache_path), mask * 255)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"].long()
        
        return image, mask


def create_data_splits(
    image_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Create train/val/test splits"""
    import random
    random.seed(seed)
    
    # Find all images
    image_dir = Path(image_dir)
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    all_images = []
    for ext in extensions:
        all_images.extend(image_dir.rglob(f"*{ext}"))
        all_images.extend(image_dir.rglob(f"*{ext.upper()}"))
    
    all_images = [str(p) for p in all_images]
    random.shuffle(all_images)
    
    # Split
    n = len(all_images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train:n_train + n_val],
        "test": all_images[n_train + n_val:],
    }
    
    # Save splits
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, paths in splits.items():
        with open(output_dir / f"{split_name}.txt", "w") as f:
            f.write("\n".join(paths))
    
    print(f"Created splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    return splits


# Test
if __name__ == "__main__":
    # Test PseudoMaskDataset
    test_images = ["/Users/anant/Desktop/Projects/LIVESTOCK/images/BLF2001/image1.jpg"]
    if os.path.exists(test_images[0]):
        dataset = PseudoMaskDataset(test_images, augment=False)
        img, mask = dataset[0]
        print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
