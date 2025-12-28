# Cattle Detection & Segmentation Pipeline
# YOLO + U-Net + LLM Interpretation

## Project Structure
```
LIVESTOCK/
├── data/
│   ├── raw/                    # Raw images (your current images/, yt_images/)
│   ├── annotations/
│   │   ├── coco/              # COCO format (for U-Net)
│   │   └── yolo/              # YOLO format (for detection)
│   └── splits/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
├── models/
│   ├── yolo/                  # YOLO weights
│   └── unet/                  # U-Net weights
├── src/
│   ├── data/                  # Data utilities
│   ├── models/                # Model definitions
│   ├── training/              # Training scripts
│   └── inference/             # Inference pipeline
├── configs/                   # Configuration files
├── outputs/                   # Inference outputs
└── scripts/                   # Utility scripts
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Pre-trained Models (optional, for zero-shot)
```bash
python scripts/download_models.py
```

### 3. Prepare Data for Training
```bash
# If you have COCO annotations (masks)
python scripts/prepare_data.py --format coco --input annotations/coco

# Convert masks to YOLO boxes
python scripts/masks_to_yolo.py --input annotations/coco --output annotations/yolo
```

### 4. Train Models
```bash
# Train YOLO
python src/training/train_yolo.py --config configs/yolo_config.yaml

# Train U-Net
python src/training/train_unet.py --config configs/unet_config.yaml
```

### 5. Run Inference
```bash
# Single image
python src/inference/pipeline.py --image path/to/image.jpg

# Batch
python src/inference/pipeline.py --folder images/BLF2001/
```

## Pipeline Architecture
```
INPUT → YOLO (detect) → U-Net (segment) → CSV (metadata) → LLM (explain)
```
