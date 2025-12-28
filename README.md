# Cattle Analysis Pipeline

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Input Image   │ ──▶ │  YOLO Detection  │ ──▶ │ U-Net Segment.   │ ──▶ │  LLM Interpret  │
└─────────────────┘     └──────────────────┘     └──────────────────┘     └─────────────────┘
                                 │                        │                        │
                                 │                        │                        │
                                 ▼                        ▼                        ▼
                        Bounding Boxes            Segmentation Mask         Natural Language
                        + Confidence              + Body Coverage           Analysis Report
                                                         │
                                                         │
                                                         ▼
                                                 ┌──────────────────┐
                                                 │   CSV Metadata   │
                                                 │  (breed, weight, │
                                                 │   age, price)    │
                                                 └──────────────────┘
```

## Project Structure

```
LIVESTOCK/
├── configs/
│   ├── yolo_config.yaml       # YOLO training config
│   ├── unet_config.yaml       # U-Net training config
│   └── pipeline_config.yaml   # Inference pipeline config
├── src/
│   ├── models/
│   │   └── unet.py           # U-Net model definition
│   ├── data/
│   │   └── dataset.py        # PyTorch datasets
│   ├── training/
│   │   ├── train_yolo.py     # YOLO training script
│   │   └── train_unet.py     # U-Net training script
│   └── inference/
│       ├── pipeline.py       # Full inference pipeline
│       └── llm_interpreter.py # LLM interpretation
├── scripts/
│   ├── prepare_data.py       # Data preparation
│   └── masks_to_yolo.py      # Convert masks to YOLO format
├── data/
│   ├── splits/               # Train/val/test splits
│   ├── labels/               # YOLO format labels
│   └── yolo_dataset/         # Organized YOLO dataset
├── models/                    # Saved model weights
├── outputs/                   # Pipeline outputs
├── images/                    # Source images (513 SKU folders)
├── yt_images/                 # YouTube frame images
├── dataset.csv               # Cattle metadata
├── demo.py                   # Quick demo script
└── requirements.txt          # Dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Component Tests

```bash
python demo.py --test
```

### 3. Run Demo Pipeline

```bash
python demo.py
```

### 4. Process Specific Image

```bash
python src/inference/pipeline.py --image images/BLF2001/image1.jpg
```

### 5. Process All Images for an SKU

```bash
python src/inference/pipeline.py --folder BLF2001 --output outputs
```

## Training

### Prepare Data

```bash
# Create train/val/test splits (animal-level, so same cattle doesn't appear in multiple sets)
python scripts/prepare_data.py --sku-split --csv dataset.csv --images images --output data/splits

# For YOLO format (requires labels)
python scripts/prepare_data.py --yolo --images images
```

### Train YOLO (Detection)

```bash
python src/training/train_yolo.py --config configs/yolo_config.yaml
```

### Train U-Net (Segmentation)

```bash
python src/training/train_unet.py --config configs/unet_config.yaml
```

## Dataset Statistics

- **513** unique cattle (SKUs)
- **2,056** curated images
- **15,843** YouTube frame images
- **17,899** total images

### Metadata Fields
- `sku`: Unique identifier (BLF2001, BLF2002, etc.)
- `sex`: MALE_BULL (507) or FEMALE_HEIFER (6)
- `breed`: MURRAH, NIL-RAVI, etc.
- `color`: BLACK, GREY, etc.
- `weight_in_kg`: Weight in kilograms
- `height_in_inch`: Height in inches
- `age_in_year`: Age in years
- `teeth`: Dental development
- `price`: Price in local currency
- `feed`: Diet information

## Models

### YOLO (Detection)
- **Model**: YOLOv8m (medium)
- **Input**: 640×640 RGB
- **Output**: Bounding boxes + confidence scores
- **Classes**: cattle (1 class)

### U-Net (Segmentation)
- **Encoder**: ResNet34 (ImageNet pretrained)
- **Input**: 256×256 RGB
- **Output**: 2-class segmentation mask (background/cattle)
- **Loss**: Combined CrossEntropy + Dice

### LLM (Interpretation)
- **Model**: BLIP-large (Salesforce/blip-image-captioning-large)
- **Function**: Generate natural language descriptions and health assessments
- **Fallback**: Rule-based SimpleInterpreter when LLM unavailable

## Annotation Workflow

If you want to train with your own annotations:

### Option 1: Use CVAT (Computer Vision Annotation Tool)
1. Install CVAT: `docker-compose up -d` (see cvat.ai)
2. Import images
3. Annotate with polygons (segmentation) or rectangles (detection)
4. Export as COCO JSON or YOLO TXT

### Option 2: Use Label Studio
1. Install: `pip install label-studio`
2. Run: `label-studio start`
3. Create project with semantic segmentation template
4. Annotate and export

### Option 3: Use Roboflow
1. Upload images to Roboflow
2. Annotate online
3. Download in YOLO format

### Convert Annotations
```bash
# COCO to YOLO
python scripts/masks_to_yolo.py --coco annotations.json --output data/labels

# Mask images to YOLO
python scripts/masks_to_yolo.py --masks data/masks --output data/labels
```

## Hardware Requirements

- **Training**: GPU recommended (8GB+ VRAM)
- **Inference**: CPU or Apple Silicon (MPS) supported
- **Memory**: 16GB+ RAM recommended for training

## Performance Benchmarks

| Component | Device | Time per Image |
|-----------|--------|----------------|
| YOLO Detection | MPS | ~50ms |
| U-Net Segmentation | MPS | ~30ms |
| LLM Interpretation | CPU | ~500ms |
| **Total Pipeline** | **MPS** | **~600ms** |

## Future Enhancements

1. **Multi-animal tracking**: Track individual cattle across video frames
2. **Behavior analysis**: Detect mounting, feeding, resting behaviors
3. **Health monitoring**: Body condition scoring from segmentation
4. **Growth prediction**: ML model for weight gain forecasting
5. **Real-time processing**: Optimize for live video feeds
