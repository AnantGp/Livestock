"""
Quick Demo Script
Run the complete pipeline on a sample image
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def run_demo():
    """Run pipeline demo on sample images"""
    
    print("=" * 60)
    print("CATTLE ANALYSIS PIPELINE - DEMO")
    print("=" * 60)
    
    # Find a sample image
    images_root = Path("images")
    sample_image = None
    
    for folder in images_root.iterdir():
        if folder.is_dir():
            for img in folder.iterdir():
                if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    sample_image = img
                    break
        if sample_image:
            break
    
    if sample_image is None:
        print("No sample image found in images/")
        return
    
    print(f"\nSample image: {sample_image}")
    
    # Import pipeline
    from src.inference.pipeline import CattlePipeline
    
    # Create pipeline
    pipeline = CattlePipeline("configs/pipeline_config.yaml")
    
    # Process
    results = pipeline.process(
        str(sample_image),
        visualize=True,
        save_output=True,
        output_dir="outputs/demo",
    )
    
    # Print results
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    
    if results.get("detection"):
        print(f"Detection: {results['detection']['num_detections']} cattle found")
    
    if results.get("segmentation"):
        print(f"Segmentation: {results['segmentation']['coverage_percent']:.1f}% coverage")
    
    if results.get("metadata"):
        meta = results["metadata"]
        print(f"Metadata: {meta.get('breed', 'N/A')} | {meta.get('weight_in_kg', 'N/A')}kg")
    
    if results.get("interpretation"):
        interp = results["interpretation"]
        if "full_report" in interp:
            print("\n" + interp["full_report"])
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


def test_components():
    """Test individual pipeline components"""
    
    print("\n" + "=" * 60)
    print("COMPONENT TESTS")
    print("=" * 60)
    
    # Test 1: U-Net model
    print("\n[1] Testing U-Net model...")
    try:
        from src.models.unet import get_unet_model
        import torch
        
        model = get_unet_model()
        x = torch.randn(1, 3, 256, 256)
        y = model(x)
        print(f"    ✓ U-Net: input {x.shape} → output {y.shape}")
    except Exception as e:
        print(f"    ✗ U-Net: {e}")
    
    # Test 2: Dataset
    print("\n[2] Testing Dataset...")
    try:
        images = []
        for folder in Path("images").iterdir():
            if folder.is_dir():
                images.extend([str(p) for p in folder.iterdir() if p.suffix.lower() == ".jpg"])
                if len(images) >= 10:
                    break
        
        print(f"    ✓ Found {len(images)} sample images")
    except Exception as e:
        print(f"    ✗ Dataset: {e}")
    
    # Test 3: CSV Lookup
    print("\n[3] Testing CSV metadata lookup...")
    try:
        import pandas as pd
        
        df = pd.read_csv("dataset.csv")
        sample_sku = df["sku"].iloc[0]
        sample_data = df[df["sku"] == sample_sku].iloc[0].to_dict()
        
        print(f"    ✓ CSV: {len(df)} records")
        print(f"    ✓ Sample: {sample_sku} - {sample_data.get('breed', 'N/A')}")
    except Exception as e:
        print(f"    ✗ CSV: {e}")
    
    # Test 4: YOLO
    print("\n[4] Testing YOLO...")
    try:
        from ultralytics import YOLO
        
        model = YOLO("yolov8n.pt")  # Smallest model
        print("    ✓ YOLO: Model loaded successfully")
    except Exception as e:
        print(f"    ✗ YOLO: {e}")
    
    # Test 5: LLM Interpreter
    print("\n[5] Testing LLM Interpreter...")
    try:
        from src.inference.llm_interpreter import SimpleInterpreter
        
        interpreter = SimpleInterpreter()
        result = interpreter.analyze_cattle(
            detection_results={"boxes": [[10, 10, 100, 100]], "confidences": [0.95]},
            segmentation_results={"coverage_percent": 35.0},
            metadata={"sku": "BLF2001", "breed": "MURRAH"},
        )
        print(f"    ✓ SimpleInterpreter: {result['metadata_summary']}")
    except Exception as e:
        print(f"    ✗ LLM Interpreter: {e}")
    
    print("\n" + "=" * 60)
    print("Component tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run component tests")
    args = parser.parse_args()
    
    if args.test:
        test_components()
    else:
        run_demo()
