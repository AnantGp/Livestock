"""Image EDA for livestock/cattle datasets.

What it does (useful before training UNet/YOLO):
- Counts images, file types
- Detects unreadable/corrupt images
- Computes width/height/aspect ratio distributions
- Estimates brightness and blur (if OpenCV available)
- Finds exact duplicates via file hash
- Saves a small report (JSON + CSV) + plots

Usage:
  python3 eda_images.py --input images --out eda_out_images
  python3 eda_images.py --input yt_images --out eda_out_yt --max-images 5000

Notes:
- For very large folders, use --max-images to sample.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore

    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_PLT = True
except Exception:
    HAS_PLT = False


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageRow:
    path: str
    ext: str
    width: int
    height: int
    aspect: float
    mode: str
    channels: int
    file_size_bytes: int
    sha256_16: str
    brightness_mean: Optional[float]
    blur_laplacian_var: Optional[float]


@dataclass
class Summary:
    input: str
    total_files_found: int
    analyzed_files: int
    unreadable_files: int
    extensions: Dict[str, int]
    width_min: int
    width_p50: int
    width_p95: int
    width_max: int
    height_min: int
    height_p50: int
    height_p95: int
    height_max: int
    aspect_min: float
    aspect_p50: float
    aspect_p95: float
    aspect_max: float
    file_size_mb_p50: float
    file_size_mb_p95: float
    duplicates_exact_groups: int
    duplicates_exact_files: int


def iter_images(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    return files


def sha256_first16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def safe_open_image(path: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        img.load()
        return img
    except Exception:
        return None


def pil_to_gray_np(img: Image.Image) -> np.ndarray:
    gray = img.convert("L")
    arr = np.asarray(gray)
    return arr


def brightness_mean(gray: np.ndarray) -> float:
    return float(gray.mean() / 255.0)


def blur_laplacian_var(gray: np.ndarray) -> float:
    if not HAS_CV2:
        raise RuntimeError("OpenCV not available")
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def percentile_int(values: np.ndarray, p: float) -> int:
    return int(np.percentile(values, p))


def percentile_float(values: np.ndarray, p: float) -> float:
    return float(np.percentile(values, p))


def save_hist(values: np.ndarray, out_path: Path, title: str, xlabel: str, bins: int = 50) -> None:
    if not HAS_PLT:
        return
    plt.figure(figsize=(8, 4.5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EDA on an image folder")
    parser.add_argument("--input", required=True, help="Folder containing images")
    parser.add_argument("--out", required=True, help="Output folder for report")
    parser.add_argument("--max-images", type=int, default=0, help="If >0, analyze a random sample of this size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    args = parser.parse_args()

    in_dir = Path(args.input)
    if not in_dir.is_absolute():
        # interpret relative to cwd
        in_dir = (Path.cwd() / in_dir).resolve()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = iter_images(in_dir)
    total_files = len(all_files)

    if total_files == 0:
        print(f"No images found in: {in_dir}")
        return 2

    random.seed(args.seed)
    files = all_files
    if args.max_images and args.max_images > 0 and args.max_images < total_files:
        files = random.sample(all_files, args.max_images)

    ext_counts: Dict[str, int] = {}
    rows: List[ImageRow] = []
    unreadable: List[str] = []

    for p in files:
        ext = p.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

        img = safe_open_image(p)
        if img is None:
            unreadable.append(str(p))
            continue

        w, h = img.size
        aspect = (w / h) if h else 0.0
        mode = img.mode
        channels = len(img.getbands())
        size_bytes = p.stat().st_size

        gray = pil_to_gray_np(img)
        bmean = brightness_mean(gray)

        blvar: Optional[float] = None
        if HAS_CV2:
            try:
                blvar = blur_laplacian_var(gray)
            except Exception:
                blvar = None

        digest16 = sha256_first16(p)

        rows.append(
            ImageRow(
                path=str(p),
                ext=ext,
                width=w,
                height=h,
                aspect=float(aspect),
                mode=mode,
                channels=channels,
                file_size_bytes=size_bytes,
                sha256_16=digest16,
                brightness_mean=bmean,
                blur_laplacian_var=blvar,
            )
        )

    # Save per-image CSV
    csv_path = out_dir / "images.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        if rows:
            writer.writeheader()
            for r in rows:
                writer.writerow(asdict(r))

    # Duplicates (exact)
    dup_map: Dict[str, List[str]] = {}
    for r in rows:
        dup_map.setdefault(r.sha256_16, []).append(r.path)
    dup_groups = {k: v for k, v in dup_map.items() if len(v) > 1}

    # Summary stats
    widths = np.array([r.width for r in rows], dtype=np.int32) if rows else np.array([], dtype=np.int32)
    heights = np.array([r.height for r in rows], dtype=np.int32) if rows else np.array([], dtype=np.int32)
    aspects = np.array([r.aspect for r in rows], dtype=np.float32) if rows else np.array([], dtype=np.float32)
    sizes_mb = np.array([r.file_size_bytes / (1024 * 1024) for r in rows], dtype=np.float32) if rows else np.array([], dtype=np.float32)

    summary = Summary(
        input=str(in_dir),
        total_files_found=total_files,
        analyzed_files=len(rows),
        unreadable_files=len(unreadable),
        extensions=dict(sorted(ext_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        width_min=int(widths.min()) if len(widths) else 0,
        width_p50=percentile_int(widths, 50) if len(widths) else 0,
        width_p95=percentile_int(widths, 95) if len(widths) else 0,
        width_max=int(widths.max()) if len(widths) else 0,
        height_min=int(heights.min()) if len(heights) else 0,
        height_p50=percentile_int(heights, 50) if len(heights) else 0,
        height_p95=percentile_int(heights, 95) if len(heights) else 0,
        height_max=int(heights.max()) if len(heights) else 0,
        aspect_min=float(aspects.min()) if len(aspects) else 0.0,
        aspect_p50=percentile_float(aspects, 50) if len(aspects) else 0.0,
        aspect_p95=percentile_float(aspects, 95) if len(aspects) else 0.0,
        aspect_max=float(aspects.max()) if len(aspects) else 0.0,
        file_size_mb_p50=percentile_float(sizes_mb, 50) if len(sizes_mb) else 0.0,
        file_size_mb_p95=percentile_float(sizes_mb, 95) if len(sizes_mb) else 0.0,
        duplicates_exact_groups=len(dup_groups),
        duplicates_exact_files=sum(len(v) for v in dup_groups.values()),
    )

    # Save JSON summary
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(asdict(summary), indent=2))

    # Save unreadable list
    if unreadable:
        (out_dir / "unreadable.txt").write_text("\n".join(unreadable) + "\n")

    # Save duplicates list
    if dup_groups:
        (out_dir / "duplicates_exact.json").write_text(json.dumps(dup_groups, indent=2))

    # Plots
    if len(rows) and HAS_PLT:
        save_hist(widths, out_dir / "width_hist.png", "Width distribution", "width")
        save_hist(heights, out_dir / "height_hist.png", "Height distribution", "height")
        save_hist(aspects, out_dir / "aspect_hist.png", "Aspect ratio distribution", "w/h")
        save_hist(sizes_mb, out_dir / "filesize_hist.png", "File size distribution (MB)", "MB")

        bmeans = np.array([r.brightness_mean for r in rows if r.brightness_mean is not None], dtype=np.float32)
        if len(bmeans):
            save_hist(bmeans, out_dir / "brightness_hist.png", "Brightness (mean, 0..1)", "brightness")

        blvars = np.array([r.blur_laplacian_var for r in rows if r.blur_laplacian_var is not None], dtype=np.float32)
        if len(blvars):
            save_hist(blvars, out_dir / "blur_hist.png", "Blur (Laplacian variance)", "var")

    # Print quick summary
    print("\n=== IMAGE EDA SUMMARY ===")
    print(f"Input: {summary.input}")
    print(f"Found: {summary.total_files_found} | Analyzed: {summary.analyzed_files} | Unreadable: {summary.unreadable_files}")
    print(f"Extensions: {summary.extensions}")
    print(f"Width:  min/p50/p95/max = {summary.width_min}/{summary.width_p50}/{summary.width_p95}/{summary.width_max}")
    print(f"Height: min/p50/p95/max = {summary.height_min}/{summary.height_p50}/{summary.height_p95}/{summary.height_max}")
    print(f"Aspect: min/p50/p95/max = {summary.aspect_min:.3f}/{summary.aspect_p50:.3f}/{summary.aspect_p95:.3f}/{summary.aspect_max:.3f}")
    print(f"File size (MB): p50/p95 = {summary.file_size_mb_p50:.2f}/{summary.file_size_mb_p95:.2f}")
    print(f"Exact duplicates: groups={summary.duplicates_exact_groups}, files={summary.duplicates_exact_files}")
    print(f"Report saved to: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
