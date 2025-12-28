"""
U-Net Model Definition
Using segmentation_models_pytorch for pre-trained encoders
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def get_unet_model(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    num_classes: int = 2,
) -> nn.Module:
    """
    Create U-Net model with pre-trained encoder
    
    Args:
        encoder_name: Backbone encoder (resnet34, resnet50, efficientnet-b0, etc.)
        encoder_weights: Pre-trained weights ("imagenet" or None)
        in_channels: Number of input channels (3 for RGB)
        num_classes: Number of output classes (2 for binary: background + cattle)
    
    Returns:
        U-Net model
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # We'll use softmax/sigmoid in loss
    )
    return model


def get_unetplusplus_model(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    num_classes: int = 2,
) -> nn.Module:
    """U-Net++ for better segmentation accuracy"""
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )
    return model


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: [B, C, H, W], target: [B, H, W]
        pred = torch.softmax(pred, dim=1)
        
        # One-hot encode target
        num_classes = pred.shape[1]
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        
        # Calculate Dice per class
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        # Average over classes and batch
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined Cross-Entropy + Dice Loss"""
    
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def get_loss_function(loss_name: str = "combined", **kwargs) -> nn.Module:
    """Get loss function by name"""
    if loss_name == "dice":
        return DiceLoss()
    elif loss_name == "ce":
        return nn.CrossEntropyLoss()
    elif loss_name == "combined":
        return CombinedLoss(
            ce_weight=kwargs.get("ce_weight", 0.5),
            dice_weight=kwargs.get("dice_weight", 0.5)
        )
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


# Test
if __name__ == "__main__":
    model = get_unet_model()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
