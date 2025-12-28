"""
EDA + Validation for dataset.csv ↔ image folders
Checks: missing values, duplicates, outliers, image count mismatches
"""

import argparse
import ast
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def norm_sku(s: str) -> str:
    """Normalize SKUs: 'BLF 2340' -> 'BLF2340'"""
    return str(s).strip().replace(" ", "")


def safe_parse_feed(x) -> List[str]:
    """Parse feed column (stringified list) into actual list"""
    if pd.isna(x) or str(x).strip() == "":
        return []
    s = str(x).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(i).strip() for i in v]
    except Exception:
        pass
    return [p.strip() for p in s.split(",") if p.strip()]


def count_images_in_dir(path: str) -> int:
    """Count image files in a directory"""
    if not os.path.isdir(path):
        return 0
    n = 0
    for root, _, files in os.walk(path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTS:
                n += 1
    return n


def find_folder_for_sku(sku_raw: str, sku_norm: str, roots: List[str]) -> Optional[str]:
    """Try to find folder matching SKU (with/without spaces)"""
    for root in roots:
        # Try raw first (with spaces)
        p1 = os.path.join(root, sku_raw)
        if os.path.isdir(p1):
            return p1
        # Try normalized (no spaces)
        p2 = os.path.join(root, sku_norm)
        if os.path.isdir(p2):
            return p2
    return None


@dataclass
class ValidationIssue:
    type: str
    count: int
    examples: List[str]


def add_issue(issues: List[ValidationIssue], typ: str, mask: pd.Series, df: pd.DataFrame, example_col: str = "sku", k: int = 5):
    idx = df.index[mask.fillna(False)]
    if len(idx) == 0:
        return
    examples = df.loc[idx, example_col].astype(str).head(k).tolist()
    issues.append(ValidationIssue(type=typ, count=int(len(idx)), examples=examples))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset.csv", help="Path to dataset.csv")
    ap.add_argument("--images-root", default="images", help="Root folder for per-SKU images")
    ap.add_argument("--yt-images-root", default="yt_images", help="Root folder for per-SKU youtube images")
    ap.add_argument("--out", default="eda_out_tabular", help="Output folder for reports")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading CSV...")
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    # Check required columns
    required = [
        "sku", "sex", "color", "breed", "feed", "age_in_year", "teeth",
        "height_in_inch", "weight_in_kg", "price", "size",
        "images_count", "yt_images_count", "total_images"
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns: {missing_cols}")
        # Continue anyway with available columns

    # Normalize SKU + parse feed
    df["sku_raw"] = df["sku"].astype(str)
    df["sku_norm"] = df["sku"].map(norm_sku)
    
    if "feed" in df.columns:
        df["feed_list"] = df["feed"].map(safe_parse_feed)
        df["feed_count"] = df["feed_list"].map(len)

    # Basic derived checks
    if all(c in df.columns for c in ["images_count", "yt_images_count", "total_images"]):
        df["total_images_calc"] = df["images_count"].fillna(0) + df["yt_images_count"].fillna(0)
        df["total_images_mismatch"] = df["total_images_calc"] != df["total_images"]
    else:
        df["total_images_mismatch"] = False

    # Check duplicates
    df["sku_dup"] = df["sku_norm"].duplicated(keep=False)

    # Numeric sanity checks
    def out_of_range(col: str, lo: Optional[float], hi: Optional[float]) -> pd.Series:
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        s = pd.to_numeric(df[col], errors="coerce")
        m = pd.Series(False, index=df.index)
        if lo is not None:
            m |= s < lo
        if hi is not None:
            m |= s > hi
        return m

    issues: List[ValidationIssue] = []
    add_issue(issues, "TOTAL_IMAGES_MISMATCH", df["total_images_mismatch"], df, "sku_raw")
    add_issue(issues, "DUPLICATE_SKU", df["sku_dup"], df, "sku_raw")

    # Domain-specific ranges (adjust as needed)
    add_issue(issues, "AGE_OUT_OF_RANGE", out_of_range("age_in_year", 0, 30), df, "sku_raw")
    add_issue(issues, "TEETH_OUT_OF_RANGE", out_of_range("teeth", 0, 16), df, "sku_raw")
    add_issue(issues, "HEIGHT_OUT_OF_RANGE_INCH", out_of_range("height_in_inch", 20, 80), df, "sku_raw")
    add_issue(issues, "WEIGHT_OUT_OF_RANGE_KG", out_of_range("weight_in_kg", 50, 1200), df, "sku_raw")
    add_issue(issues, "PRICE_OUT_OF_RANGE", out_of_range("price", 1000, 10_000_000), df, "sku_raw")

    # Image folder consistency checks
    print("\nChecking image folders...")
    img_counts_fs = []
    yt_counts_fs = []
    folder_found = []
    
    for _, row in df.iterrows():
        sku_raw = str(row["sku_raw"])
        sku_norm = str(row["sku_norm"])
        
        # Find images folder
        img_folder = find_folder_for_sku(sku_raw, sku_norm, [args.images_root])
        img_count = count_images_in_dir(img_folder) if img_folder else 0
        img_counts_fs.append(img_count)
        
        # Find yt_images folder
        yt_folder = find_folder_for_sku(sku_raw, sku_norm, [args.yt_images_root])
        yt_count = count_images_in_dir(yt_folder) if yt_folder else 0
        yt_counts_fs.append(yt_count)
        
        folder_found.append(img_folder is not None or yt_folder is not None)

    df["images_count_fs"] = img_counts_fs
    df["yt_images_count_fs"] = yt_counts_fs
    df["folder_found"] = folder_found
    
    if "images_count" in df.columns:
        df["images_count_fs_mismatch"] = df["images_count_fs"] != df["images_count"].fillna(0).astype(int)
        add_issue(issues, "IMAGES_COUNT_FS_MISMATCH", df["images_count_fs_mismatch"], df, "sku_raw")
    
    if "yt_images_count" in df.columns:
        df["yt_images_count_fs_mismatch"] = df["yt_images_count_fs"] != df["yt_images_count"].fillna(0).astype(int)
        add_issue(issues, "YT_IMAGES_COUNT_FS_MISMATCH", df["yt_images_count_fs_mismatch"], df, "sku_raw")
    
    add_issue(issues, "NO_FOLDER_FOUND", ~df["folder_found"], df, "sku_raw")

    # Build summary
    summary: Dict = {
        "rows": int(len(df)),
        "unique_sku": int(df["sku_norm"].nunique()),
        "missing_values": df.isnull().sum().to_dict(),
    }
    
    # Category counts
    for col in ["sex", "color", "size"]:
        if col in df.columns:
            summary[f"{col}_counts"] = df[col].value_counts(dropna=False).to_dict()
    
    if "breed" in df.columns:
        summary["breed_counts_top10"] = df["breed"].value_counts(dropna=False).head(10).to_dict()

    # Numeric stats
    numeric_cols = ["age_in_year", "teeth", "height_in_inch", "weight_in_kg", "price", "total_images"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    if numeric_cols:
        summary["numeric_describe"] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").describe().to_dict()

    # Image stats
    summary["total_images_in_fs"] = int(df["images_count_fs"].sum())
    summary["total_yt_images_in_fs"] = int(df["yt_images_count_fs"].sum())
    summary["skus_with_folders"] = int(df["folder_found"].sum())
    summary["skus_without_folders"] = int((~df["folder_found"]).sum())

    summary["issues"] = [asdict(i) for i in issues]

    # Save outputs
    json_path = os.path.join(args.out, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(args.out, "dataset_enriched.csv")
    df.to_csv(csv_path, index=False)

    # Print report
    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"\nRows: {summary['rows']}")
    print(f"Unique SKUs: {summary['unique_sku']}")
    print(f"\nImage folders found: {summary['skus_with_folders']}")
    print(f"Image folders missing: {summary['skus_without_folders']}")
    print(f"Total images (filesystem): {summary['total_images_in_fs']}")
    print(f"Total YT images (filesystem): {summary['total_yt_images_in_fs']}")
    
    if "sex_counts" in summary:
        print(f"\nSex distribution: {summary['sex_counts']}")
    if "size_counts" in summary:
        print(f"Size distribution: {summary['size_counts']}")
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for i in issues:
            print(f"  - {i.type}: {i.count} rows (examples: {i.examples[:3]})")
    else:
        print("\n✅ No issues detected!")
    
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
