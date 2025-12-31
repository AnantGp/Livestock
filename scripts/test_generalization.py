"""
Test U-Net on Cattle Breeds folder (unseen data)
This tests generalization to completely new images
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
from src.data.dataset import PseudoMaskDataset


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


def test_on_cattle_breeds(model_path: str, cattle_breeds_dir: str = "images/Cattle Breeds"):
    """Test model on Cattle Breeds folder"""
    
    print("=" * 60)
    print("U-NET GENERALIZATION TEST - CATTLE BREEDS")
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
    
    print(f"Model trained IoU: {checkpoint.get('best_iou', 'N/A'):.4f}")
    
    # Find all images in Cattle Breeds
    cattle_breeds_path = Path(cattle_breeds_dir)
    if not cattle_breeds_path.exists():
        print(f"Error: {cattle_breeds_dir} not found!")
        return
    
    # Get images from all subfolders
    all_images = []
    breed_counts = {}
    
    for breed_folder in cattle_breeds_path.iterdir():
        if breed_folder.is_dir() and not breed_folder.name.startswith('.'):
            breed_images = list(breed_folder.glob("*.jpg")) + list(breed_folder.glob("*.png"))
            all_images.extend(breed_images)
            breed_counts[breed_folder.name] = len(breed_images)
    
    print(f"\nCattle Breeds found:")
    for breed, count in breed_counts.items():
        print(f"  {breed}: {count} images")
    print(f"Total: {len(all_images)} images")
    
    # Convert to string paths
    image_paths = [str(p) for p in all_images]
    
    # Create dataset
    img_size = config.get("training", {}).get("img_size", 256)
    test_dataset = PseudoMaskDataset(image_paths, img_size, augment=False)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # Evaluate
    print("\nRunning evaluation on Cattle Breeds (unseen data)...")
    all_ious = []
    cattle_ious = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            iou_dict = calculate_iou(outputs, masks)
            
            all_ious.append(iou_dict["mean"])
            cattle_ious.append(iou_dict["class_1"])
    
    # Calculate statistics
    mean_iou = np.mean(all_ious)
    std_iou = np.std(all_ious)
    mean_cattle_iou = np.mean(cattle_ious)
    
    # Print results
    print("\n" + "=" * 60)
    print("GENERALIZATION TEST RESULTS")
    print("=" * 60)
    print(f"Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"Cattle IoU: {mean_cattle_iou:.4f}")
    print(f"Test samples: {len(all_images)} (from 5 breeds)")
    print("=" * 60)
    
    # Compare with training performance
    train_iou = checkpoint.get('best_iou', 0)
    if train_iou:
        drop = (train_iou - mean_iou) * 100
        print(f"\nComparison:")
        print(f"  Training Val IoU: {train_iou:.4f}")
        print(f"  Cattle Breeds IoU: {mean_iou:.4f}")
        print(f"  Performance drop: {drop:.2f}%")
        
        if drop < 5:
            print("  ✅ Excellent generalization!")
        elif drop < 10:
            print("  ✅ Good generalization")
        elif drop < 15:
            print("  ⚠️ Moderate generalization")
        else:
            print("  ❌ Poor generalization - consider adding more diverse training data")
    
    return {
        "mean_iou": mean_iou,
        "std_iou": std_iou,
        "cattle_iou": mean_cattle_iou,
        "num_samples": len(all_images),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/unet/best.pth", help="Model checkpoint path")
    parser.add_argument("--cattle-breeds", default="images/Cattle Breeds", help="Cattle Breeds directory")
    
    args = parser.parse_args()
    
    test_on_cattle_breeds(args.model, args.cattle_breeds)
