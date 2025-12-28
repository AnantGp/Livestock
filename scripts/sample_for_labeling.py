"""\
Sample a manageable subset of images for labeling.

Goals
- Pick N images across SKUs to avoid overfitting to a few animals.
- Copy them into a flat folder for easier labeling in CVAT/Label Studio/Roboflow.
- Emit a manifest.csv so you can map labeled files back to original paths/SKUs.

Typical usage
- YOLO boxes:  python3 scripts/sample_for_labeling.py --task yolo --n 600
- U-Net masks: python3 scripts/sample_for_labeling.py --task unet --n 150

Notes
- By default samples from images/ only. Add --include-yt to also sample yt_images/.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class ImageRecord:
    sku: str
    src_path: Path
    source_root: str  # "images" or "yt_images"


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def _normalize_sku(folder_name: str) -> str:
    # Handles "BLF 2340" vs "BLF2340"
    return folder_name.replace(" ", "").strip().upper()


def _iter_sku_folders(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for child in root.iterdir():
        if child.is_dir():
            yield child


def collect_images(images_root: Path, source_root_name: str) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    for sku_folder in _iter_sku_folders(images_root):
        sku = _normalize_sku(sku_folder.name)
        for p in sku_folder.iterdir():
            if _is_image(p):
                records.append(ImageRecord(sku=sku, src_path=p, source_root=source_root_name))
    return records


def sample_balanced(
    records: Sequence[ImageRecord],
    n_total: int,
    per_sku_max: int,
    seed: int,
) -> List[ImageRecord]:
    rng = random.Random(seed)

    by_sku: Dict[str, List[ImageRecord]] = {}
    for r in records:
        by_sku.setdefault(r.sku, []).append(r)

    # Shuffle within each SKU
    for sku in by_sku:
        rng.shuffle(by_sku[sku])

    skus = list(by_sku.keys())
    rng.shuffle(skus)

    sampled: List[ImageRecord] = []

    # Round-robin to spread across SKUs
    round_idx = 0
    while len(sampled) < n_total:
        progressed = False
        for sku in skus:
            pool = by_sku[sku]
            # Count already taken for this SKU
            taken_for_sku = sum(1 for r in sampled if r.sku == sku)
            if taken_for_sku >= per_sku_max:
                continue
            if round_idx < len(pool):
                sampled.append(pool[round_idx])
                progressed = True
                if len(sampled) >= n_total:
                    break
        if not progressed:
            break
        round_idx += 1

    return sampled[:n_total]


def ensure_empty_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_and_manifest(
    sampled: Sequence[ImageRecord],
    out_images_dir: Path,
    manifest_path: Path,
) -> None:
    ensure_empty_dir(out_images_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Flat output names: {sku}__{i:05d}{ext}
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dest_file", "sku", "source_root", "src_path"],
        )
        writer.writeheader()

        for i, rec in enumerate(sampled):
            dest_name = f"{rec.sku}__{i:05d}{rec.src_path.suffix.lower()}"
            dest_path = out_images_dir / dest_name
            shutil.copy2(rec.src_path, dest_path)
            writer.writerow(
                {
                    "dest_file": dest_name,
                    "sku": rec.sku,
                    "source_root": rec.source_root,
                    "src_path": str(rec.src_path),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample images for labeling")
    parser.add_argument("--task", choices=["yolo", "unet"], required=True)
    parser.add_argument("--n", type=int, required=True, help="Number of images to sample")
    parser.add_argument("--per-sku-max", type=int, default=3, help="Max images per SKU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-yt", action="store_true", help="Also sample from yt_images/")
    parser.add_argument("--images-root", default="images")
    parser.add_argument("--yt-root", default="yt_images")
    parser.add_argument("--out-dir", default="data/labeling")

    args = parser.parse_args()

    images_root = Path(args.images_root)
    yt_root = Path(args.yt_root)
    out_dir = Path(args.out_dir) / args.task
    out_images_dir = out_dir / "images"
    manifest_path = out_dir / "manifest.csv"

    records: List[ImageRecord] = []
    records.extend(collect_images(images_root, "images"))
    if args.include_yt:
        records.extend(collect_images(yt_root, "yt_images"))

    if not records:
        raise SystemExit("No images found to sample. Check images_root/yt_root.")

    sampled = sample_balanced(records, n_total=args.n, per_sku_max=args.per_sku_max, seed=args.seed)
    copy_and_manifest(sampled, out_images_dir=out_images_dir, manifest_path=manifest_path)

    print(f"Sampled {len(sampled)} images")
    print(f"Images:   {out_images_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
