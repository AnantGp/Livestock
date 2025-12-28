"""
U-Net Training Script
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.unet import get_unet_model, get_loss_function
from src.data.dataset import SegmentationDataset, PseudoMaskDataset, create_data_splits


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> float:
    """Calculate mean IoU"""
    pred = pred.argmax(dim=1)
    ious = []
    
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        if union > 0:
            ious.append((intersection / union).item())
    
    return np.mean(ious) if ious else 0.0


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_iou += calculate_iou(outputs, masks)
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    n = len(loader)
    return {"loss": total_loss / n, "iou": total_iou / n}


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """Validate model"""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            total_iou += calculate_iou(outputs, masks)
    
    n = len(loader)
    return {"loss": total_loss / n, "iou": total_iou / n}


def train(config_path: str):
    """Main training function"""
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("U-NET TRAINING")
    print("=" * 60)
    
    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")
    
    # Create output directory
    output_dir = Path(config["output"]["save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create data splits
    splits_dir = Path("data/splits")
    if not (splits_dir / "train.txt").exists():
        print("Creating data splits...")
        create_data_splits(
            image_dir=config["data"]["images_root"],
            output_dir=str(splits_dir),
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )
    
    # Load splits
    with open(splits_dir / "train.txt", "r") as f:
        train_images = [l.strip() for l in f if l.strip()]
    with open(splits_dir / "val.txt", "r") as f:
        val_images = [l.strip() for l in f if l.strip()]
    
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")
    
    # Check if we have masks
    masks_root = Path(config["data"].get("masks_root", "data/annotations/masks"))
    
    if masks_root.exists() and any(masks_root.iterdir()):
        # Use real masks
        print("Using annotated masks")
        train_masks = [str(masks_root / Path(p).name.replace(".jpg", "_mask.png")) for p in train_images]
        val_masks = [str(masks_root / Path(p).name.replace(".jpg", "_mask.png")) for p in val_images]
        
        train_dataset = SegmentationDataset(train_images, train_masks, config["training"]["img_size"], augment=True)
        val_dataset = SegmentationDataset(val_images, val_masks, config["training"]["img_size"], augment=False)
    else:
        # Use pseudo-masks (auto-generated)
        print("Using pseudo-masks (GrabCut)")
        train_dataset = PseudoMaskDataset(train_images, config["training"]["img_size"], augment=True)
        val_dataset = PseudoMaskDataset(val_images, config["training"]["img_size"], augment=False)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # Model
    model = get_unet_model(
        encoder_name=config["model"]["encoder"],
        encoder_weights="imagenet" if config["model"]["pretrained"] else None,
        num_classes=config["model"]["num_classes"],
    )
    model = model.to(device)
    
    print(f"Model: U-Net with {config['model']['encoder']} encoder")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    loss_config = config["training"]["loss"]
    criterion = get_loss_function(
        loss_config["name"],
        ce_weight=loss_config.get("ce_weight", 0.5),
        dice_weight=loss_config.get("dice_weight", 0.5),
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["optimizer"]["lr"],
        weight_decay=config["training"]["optimizer"]["weight_decay"],
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
        eta_min=1e-6,
    )
    
    # Training loop
    best_iou = 0.0
    patience_counter = 0
    patience = config["training"]["patience"]
    
    print("\nStarting training...")
    
    for epoch in range(config["training"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # Print metrics
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Train IoU: {train_metrics['iou']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val IoU: {val_metrics['iou']:.4f}")
        
        # Save best model
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            patience_counter = 0
            
            if config["output"]["save_best"]:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "best_iou": best_iou,
                    "epoch": epoch,
                }, output_dir / "best.pth")
                print(f"  ✓ Saved best model (IoU: {best_iou:.4f})")
        else:
            patience_counter += 1
        
        # Save last model
        if config["output"]["save_last"]:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
            }, output_dir / "last.pth")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best IoU: {best_iou:.4f}")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unet_config.yaml", help="Config file path")
    args = parser.parse_args()
    
    train(args.config)
