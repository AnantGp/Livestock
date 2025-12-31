"""
Evaluate U-Net model on test set
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.unet import get_unet_model
from src.data.dataset import SegmentationDataset, PseudoMaskDataset


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> dict:
    """Calculate IoU per class and mean IoU"""
    pred = pred.argmax(dim=1)
    ious = {}
    
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        if union > 0:
            ious[f"class_{cls}"] = (intersection / union).item()
        else:
            ious[f"class_{cls}"] = 0.0
    
    ious["mean"] = np.mean([ious[f"class_{cls}"] for cls in range(num_classes)])
    return ious


def evaluate(model_path: str, test_file: str = "data/splits/test.txt", use_pseudo_masks: bool = True):
    """Evaluate model on test set"""
    
    print("=" * 60)
    print("U-NET TEST EVALUATION")
    print("=" * 60)
    
    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    config = checkpoint.get("config", {})
    model = get_unet_model(
        encoder_name=config.get("model", {}).get("encoder", "resnet34"),
        num_classes=config.get("model", {}).get("num_classes", 2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Model: U-Net with {config.get('model', {}).get('encoder', 'resnet34')} encoder")
    print(f"Training Val IoU: {checkpoint.get('best_iou', 'N/A'):.4f}")
    
    # Load test split
    test_file = Path(test_file)
    if not test_file.exists():
        print(f"Error: {test_file} not found!")
        return
    
    with open(test_file, "r") as f:
        test_images = [l.strip() for l in f if l.strip()]
    
    print(f"\nTest images: {len(test_images)}")
    
    # Create dataset
    if use_pseudo_masks:
        print("Using pseudo-masks (GrabCut)")
        img_size = config.get("training", {}).get("img_size", 256)
        test_dataset = PseudoMaskDataset(test_images, img_size, augment=False)
    else:
        print("Using annotated masks")
        # Implement if you have real masks
        raise NotImplementedError("Annotated masks not implemented yet")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # Evaluate
    print("\nRunning evaluation...")
    all_ious = []
    class_0_ious = []  # Background
    class_1_ious = []  # Cattle
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            iou_dict = calculate_iou(outputs, masks)
            
            all_ious.append(iou_dict["mean"])
            class_0_ious.append(iou_dict["class_0"])
            class_1_ious.append(iou_dict["class_1"])
    
    # Calculate statistics
    mean_iou = np.mean(all_ious)
    std_iou = np.std(all_ious)
    mean_bg_iou = np.mean(class_0_ious)
    mean_cattle_iou = np.mean(class_1_ious)
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"Background IoU: {mean_bg_iou:.4f}")
    print(f"Cattle IoU: {mean_cattle_iou:.4f}")
    print(f"Test samples: {len(test_images)}")
    print("=" * 60)
    
    return {
        "mean_iou": mean_iou,
        "std_iou": std_iou,
        "background_iou": mean_bg_iou,
        "cattle_iou": mean_cattle_iou,
        "num_samples": len(test_images),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/unet/best.pth", help="Model checkpoint path")
    parser.add_argument("--test-file", default="data/splits/test.txt", help="Test split file")
    parser.add_argument("--no-pseudo", action="store_true", help="Don't use pseudo-masks")
    
    args = parser.parse_args()
    
    evaluate(args.model, args.test_file, use_pseudo_masks=not args.no_pseudo)
