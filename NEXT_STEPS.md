# Ready to Label - Quick Start

## ✅ DONE: Labeling subsets created

- **YOLO**: 600 images in `data/labeling/yolo/images/`
- **U-Net**: 150 images in `data/labeling/unet/images/`

## Next: Pick your labeling tool and follow the steps

---

## Option A: Roboflow (Easiest for YOLO)

### 1) Upload YOLO images
- Go to roboflow.com → Create Project → Object Detection
- Upload all files from: `data/labeling/yolo/images/`
- Class name: `cattle`

### 2) Draw bounding boxes
- Use their web interface to draw boxes around cattle
- Export → YOLOv8 format
- Download the exported labels

### 3) Place labels in your repo
- Unzip the download
- Copy all `.txt` files from `train/labels/` + `valid/labels/` → `data/labels/`

---

## Option B: CVAT (Best for teams/masks)

### 1) Install CVAT locally
```bash
git clone https://github.com/opencv/cvat
cd cvat
docker-compose up -d
# Open http://localhost:8080
```

### 2A) For YOLO boxes
- Create Task → Upload `data/labeling/yolo/images/`
- Labels: Add `cattle` class
- Annotate with rectangles
- Actions → Export → YOLO 1.1 format
- Place exported `.txt` files in `data/labels/`

### 2B) For U-Net masks  
- Create Task → Upload `data/labeling/unet/images/`
- Labels: Add `cattle` class  
- Annotate with polygons or masks
- Actions → Export → COCO 1.0 format
- Convert: `python3 scripts/masks_to_yolo.py --coco annotations.json --output data/masks`

---

## Option C: Label Studio (Flexible)

### 1) Install
```bash
pip install label-studio
label-studio start
# Open http://localhost:8080
```

### 2) For boxes: Use Object Detection template
### 3) For masks: Use Semantic Segmentation template

Export and convert as needed.

---

## After labeling → Train

### Once you have `data/labels/` (YOLO):
```bash
python3 scripts/prepare_data.py --yolo --images images
python3 src/training/train_yolo.py --config configs/yolo_config.yaml
```

### Once you have `data/masks/` (U-Net):  
```bash
python3 src/training/train_unet.py --config configs/unet_config.yaml
```

---

## ⚡ Fastest path (recommended)

1. **Start with Roboflow** for the 600 YOLO images (boxes are faster to draw than masks)
2. **Label 100-200 of those same images** with masks in CVAT/Label Studio  
3. **Train both models**
4. **Run full pipeline**: `python3 demo.py`

The 600 images are sampled across all your SKUs, so you'll get good coverage of breeds/lighting/angles.