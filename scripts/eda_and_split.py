"""
Dataset EDA and Train/Val/Test Split for Cattle Analysis

This script:
1. Performs comprehensive EDA
2. Creates stratified train/val/test splits
3. Validates data integrity
4. Saves split information for reproducibility
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import os


def load_dataset(csv_path: str = "dataset.csv") -> pd.DataFrame:
    """Load and validate dataset"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")
    return df


def perform_eda(df: pd.DataFrame, output_dir: str = "outputs/eda"):
    """Comprehensive Exploratory Data Analysis"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    
    # 1. Dataset Overview
    print("\n--- DATASET OVERVIEW ---")
    print(f"Total records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # 2. Categorical Analysis
    print("\n--- CATEGORICAL DISTRIBUTIONS ---")
    categorical_cols = ['sex', 'color', 'breed', 'size']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(categorical_cols):
        counts = df[col].value_counts()
        print(f"\n{col.upper()}:")
        print(counts)
        
        # Plot
        ax = axes[i]
        bars = ax.bar(range(len(counts)), counts.values, color=plt.cm.Set3(np.linspace(0, 1, len(counts))))
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha='right')
        ax.set_title(f'{col.upper()} Distribution')
        ax.set_ylabel('Count')
        
        # Add count labels
        for bar, count in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   str(count), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/categorical_distributions.png", dpi=150)
    plt.close()
    print(f"\nSaved: {output_dir}/categorical_distributions.png")
    
    # 3. Numerical Analysis
    print("\n--- NUMERICAL DISTRIBUTIONS ---")
    numerical_cols = ['age_in_year', 'teeth', 'height_in_inch', 'weight_in_kg', 'price']
    
    print(df[numerical_cols].describe().round(2))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        ax = axes[i]
        ax.hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.1f}')
        ax.axvline(df[col].median(), color='green', linestyle='--', label=f'Median: {df[col].median():.1f}')
        ax.set_title(f'{col} Distribution')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.legend()
    
    # Hide the 6th subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/numerical_distributions.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/numerical_distributions.png")
    
    # 4. Weight Analysis by Breed
    print("\n--- WEIGHT BY BREED ---")
    weight_by_breed = df.groupby('breed')['weight_in_kg'].agg(['mean', 'std', 'min', 'max', 'count'])
    print(weight_by_breed.round(1).sort_values('count', ascending=False))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    breed_order = df.groupby('breed')['weight_in_kg'].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x='breed', y='weight_in_kg', order=breed_order, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title('Weight Distribution by Breed')
    ax.set_ylabel('Weight (kg)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/weight_by_breed.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/weight_by_breed.png")
    
    # 5. Correlation Analysis
    print("\n--- CORRELATIONS ---")
    corr_cols = ['height_in_inch', 'weight_in_kg', 'price', 'age_in_year']
    corr_matrix = df[corr_cols].corr()
    print(corr_matrix.round(3))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, ax=ax, fmt='.3f')
    ax.set_title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/correlation_matrix.png")
    
    # 6. Height vs Weight Scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    for breed in df['breed'].unique():
        breed_data = df[df['breed'] == breed]
        ax.scatter(breed_data['height_in_inch'], breed_data['weight_in_kg'], 
                  label=f"{breed} (n={len(breed_data)})", alpha=0.7)
    
    # Add regression line
    z = np.polyfit(df['height_in_inch'], df['weight_in_kg'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['height_in_inch'].min(), df['height_in_inch'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Regression: y={z[0]:.1f}x + {z[1]:.1f}')
    
    ax.set_xlabel('Height (inches)')
    ax.set_ylabel('Weight (kg)')
    ax.set_title('Height vs Weight by Breed')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/height_vs_weight.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/height_vs_weight.png")
    
    # 7. Image availability check
    print("\n--- IMAGE AVAILABILITY ---")
    images_dir = Path('images')
    image_counts = []
    
    for sku in df['sku']:
        sku_dir = images_dir / sku
        if sku_dir.exists():
            count = len(list(sku_dir.glob('*.jpg')) + list(sku_dir.glob('*.png')) + list(sku_dir.glob('*.jpeg')))
        else:
            count = 0
        image_counts.append(count)
    
    df['actual_images'] = image_counts
    print(f"Total SKUs with images: {sum(1 for c in image_counts if c > 0)}")
    print(f"Total images found: {sum(image_counts)}")
    print(f"Average images per SKU: {np.mean(image_counts):.1f}")
    
    # 8. Class Imbalance Analysis
    print("\n--- CLASS IMBALANCE ANALYSIS ---")
    
    for col in ['breed', 'size', 'color']:
        counts = df[col].value_counts()
        imbalance_ratio = counts.max() / counts.min()
        print(f"{col}: max/min ratio = {imbalance_ratio:.1f}x")
    
    return df


def create_stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_col: str = 'breed',
    random_state: int = 42,
    output_dir: str = "data/splits"
) -> dict:
    """
    Create stratified train/val/test splits
    
    Stratification ensures each split has similar distribution of breeds
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print("=" * 70)
    print(f"Ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    print(f"Stratifying by: {stratify_col}")
    
    # For small classes (< 5 samples), we can't stratify properly
    # Combine rare classes for stratification
    df_copy = df.copy()
    class_counts = df_copy[stratify_col].value_counts()
    
    # Need at least 5 samples to split into train/val/test
    min_samples_needed = 5
    rare_classes = class_counts[class_counts < min_samples_needed].index.tolist()
    
    if rare_classes:
        print(f"\nRare classes (< {min_samples_needed} samples): {rare_classes}")
        print("These will be grouped with the most common class for stratification")
        
        # Instead of creating 'OTHER', merge with the most common class
        most_common = class_counts.idxmax()
        df_copy['stratify_group'] = df_copy[stratify_col].apply(
            lambda x: most_common if x in rare_classes else x
        )
        stratify_col_actual = 'stratify_group'
    else:
        stratify_col_actual = stratify_col
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df_copy,
        test_size=(val_ratio + test_ratio),
        stratify=df_copy[stratify_col_actual],
        random_state=random_state
    )
    
    # Second split: val vs test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df[stratify_col_actual],
        random_state=random_state
    )
    
    # Remove temporary column
    if 'stratify_group' in train_df.columns:
        train_df = train_df.drop(columns=['stratify_group'])
        val_df = val_df.drop(columns=['stratify_group'])
        test_df = test_df.drop(columns=['stratify_group'])
    
    # Print split statistics
    print(f"\n--- SPLIT SIZES ---")
    print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Print breed distribution in each split
    print(f"\n--- BREED DISTRIBUTION ---")
    print(f"{'Breed':<18} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print("-" * 50)
    
    for breed in df['breed'].unique():
        train_count = len(train_df[train_df['breed'] == breed])
        val_count = len(val_df[val_df['breed'] == breed])
        test_count = len(test_df[test_df['breed'] == breed])
        total_count = train_count + val_count + test_count
        print(f"{breed:<18} {train_count:<8} {val_count:<8} {test_count:<8} {total_count:<8}")
    
    # Save splits
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
    
    # Save split info
    split_info = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "stratify_col": stratify_col,
        "random_state": random_state,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "train_skus": train_df['sku'].tolist(),
        "val_skus": val_df['sku'].tolist(),
        "test_skus": test_df['sku'].tolist(),
    }
    
    with open(f"{output_dir}/split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\nSaved splits to {output_dir}/")
    print(f"  - train.csv ({len(train_df)} records)")
    print(f"  - val.csv ({len(val_df)} records)")
    print(f"  - test.csv ({len(test_df)} records)")
    print(f"  - split_info.json")
    
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "info": split_info
    }


def validate_image_splits(split_info: dict, images_dir: str = "images"):
    """Validate that all images exist for the splits"""
    print("\n" + "=" * 70)
    print("VALIDATING IMAGE SPLITS")
    print("=" * 70)
    
    images_path = Path(images_dir)
    
    for split_name, skus in [("train", split_info["train_skus"]), 
                             ("val", split_info["val_skus"]), 
                             ("test", split_info["test_skus"])]:
        total_images = 0
        missing_skus = []
        
        for sku in skus:
            sku_dir = images_path / sku
            if sku_dir.exists():
                img_count = len(list(sku_dir.glob('*.jpg')) + 
                              list(sku_dir.glob('*.png')) + 
                              list(sku_dir.glob('*.jpeg')))
                total_images += img_count
            else:
                missing_skus.append(sku)
        
        print(f"\n{split_name.upper()}:")
        print(f"  SKUs: {len(skus)}")
        print(f"  Total images: {total_images}")
        print(f"  Missing SKU folders: {len(missing_skus)}")
        if missing_skus:
            print(f"  First 5 missing: {missing_skus[:5]}")


def create_yolo_splits(split_info: dict, images_dir: str = "images", output_dir: str = "data/yolo_splits"):
    """Create YOLO-format data splits with image paths"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("CREATING YOLO DATA SPLITS")
    print("=" * 70)
    
    images_path = Path(images_dir)
    
    for split_name, skus in [("train", split_info["train_skus"]), 
                             ("val", split_info["val_skus"]), 
                             ("test", split_info["test_skus"])]:
        image_paths = []
        
        for sku in skus:
            sku_dir = images_path / sku
            if sku_dir.exists():
                for img_path in sku_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        image_paths.append(str(img_path))
        
        # Save image list
        with open(f"{output_dir}/{split_name}.txt", "w") as f:
            f.write("\n".join(image_paths))
        
        print(f"{split_name}: {len(image_paths)} images → {output_dir}/{split_name}.txt")
    
    # Create data.yaml for YOLO
    data_yaml = f"""# Cattle Detection Dataset
path: {Path(output_dir).absolute()}
train: train.txt
val: val.txt
test: test.txt

# Classes
names:
  0: cattle

nc: 1
"""
    
    with open(f"{output_dir}/data.yaml", "w") as f:
        f.write(data_yaml)
    
    print(f"\nSaved YOLO config: {output_dir}/data.yaml")


def main():
    """Main function to run EDA and create splits"""
    
    # Load data
    df = load_dataset("dataset.csv")
    
    # Perform EDA
    df = perform_eda(df, output_dir="outputs/eda")
    
    # Create stratified splits
    splits = create_stratified_split(
        df,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify_col='breed',
        random_state=42
    )
    
    # Validate image availability
    validate_image_splits(splits["info"])
    
    # Create YOLO splits
    create_yolo_splits(splits["info"])
    
    print("\n" + "=" * 70)
    print("EDA AND SPLIT CREATION COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review EDA outputs in outputs/eda/")
    print("2. Use data/splits/ for model training")
    print("3. Use data/yolo_splits/ for YOLO training")


if __name__ == "__main__":
    main()
