# Labeling Guide (YOLO boxes + U-Net masks)

This repo currently has **images + metadata** but does **not** contain human annotations yet.
To fine-tune models on your domain, you should label:

- **YOLO (detection)**: ~300–800 images with **bounding boxes** around cattle.
- **U-Net (segmentation)**: ~100–200 images with **pixel masks** for cattle vs background.

The fastest workflow is: **label masks → derive boxes from masks** (optional), or label boxes directly.

## 0) Create the labeling subsets

### YOLO subset (recommended 600)

```bash
python3 scripts/sample_for_labeling.py --task yolo --n 600 --per-sku-max 3
```

Outputs:
- `data/labeling/yolo/images/` (flat folder of sampled images)
- `data/labeling/yolo/manifest.csv` (maps each sampled file back to original SKU/path)

### U-Net subset (recommended 150)

```bash
python3 scripts/sample_for_labeling.py --task unet --n 150 --per-sku-max 2
```

Outputs:
- `data/labeling/unet/images/`
- `data/labeling/unet/manifest.csv`

If you want to include YouTube frames too:

```bash
python3 scripts/sample_for_labeling.py --task yolo --n 600 --include-yt
python3 scripts/sample_for_labeling.py --task unet --n 150 --include-yt
```

## 1) Label YOLO bounding boxes

### Target folder format

After labeling, place YOLO labels here:

- `data/labels/`
  - one `.txt` per image
  - filename must match the image filename (stem)

Example:
- image: `data/labeling/yolo/images/BLF2646__00012.jpg`
- label: `data/labels/BLF2646__00012.txt`

YOLO txt content (1 class `cattle`):

```
0 x_center y_center width height
```

All values are normalized to `[0,1]`.

### Tools (pick one)

- **Roboflow**: easiest for YOLO export.
- **CVAT**: great for teams, can export YOLO.
- **Label Studio**: works well, but YOLO export may need a converter depending on setup.

## 2) Label U-Net segmentation masks

### Target format (simple and robust)

Save binary masks as PNG:
- `data/masks/`
  - one mask per image
  - same stem as image
  - background = 0 (black), cattle = 255 (white)

Example:
- image: `data/labeling/unet/images/BLF2646__00012.jpg`
- mask:  `data/masks/BLF2646__00012.png`

### Tools

- **CVAT**: export masks or COCO polygons.
- **Label Studio**: export masks (or polygons).

## 3) Convert masks → YOLO boxes (optional but useful)

If you label masks first, you can generate YOLO labels from them:

```bash
python3 scripts/masks_to_yolo.py --masks data/masks --output data/labels
```

This will create `data/labels/*.txt` files.

If you exported COCO JSON instead:

```bash
python3 scripts/masks_to_yolo.py --coco path/to/annotations.json --output data/labels
```

## 4) Create a YOLO training dataset folder

Once `data/labels/` exists, build the YOLO dataset structure:

```bash
python3 scripts/prepare_data.py --yolo --images images
```

This creates:
- `data/yolo_dataset/images/{train,val,test}/`
- `data/yolo_dataset/labels/{train,val,test}/`
- `data/yolo_dataset/dataset.yaml`

## 5) Train

### YOLO

```bash
python3 src/training/train_yolo.py --config configs/yolo_config.yaml
```

### U-Net

If `data/masks/` exists, U-Net uses real masks.
If not, it falls back to pseudo-masks (GrabCut) but that is **not** a substitute for real labels.

```bash
python3 src/training/train_unet.py --config configs/unet_config.yaml
```

## 6) Run inference pipeline

```bash
python3 src/inference/pipeline.py --image images/BLF2646/BLF2646_3.jpg --output outputs
```

Notes:
- The pipeline can join with `dataset.csv` based on the SKU folder name.
- The LLM report is optional; if it’s enabled, it may run on CPU on macOS.
