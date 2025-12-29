"""
Complete Inference Pipeline
YOLO Detection → U-Net Segmentation → CSV Metadata Lookup → LLM Interpretation
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
import yaml

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class CattlePipeline:
    """
    Complete pipeline for cattle analysis:
    1. YOLO: Detect cattle in image
    2. U-Net: Segment detected cattle
    3. CSV: Lookup metadata by SKU
    4. LLM: Generate interpretation
    """
    
    def __init__(self, config_path: str = "configs/pipeline_config.yaml", use_llm: bool = True):
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Override LLM setting if specified
        self.use_llm = use_llm
        
        # Set device
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        print(f"Pipeline device: {self.device}")
        print(f"LLM interpretation: {'enabled' if use_llm else 'disabled'}")
        
        # Initialize components (lazy loading)
        self.yolo_model = None
        self.unet_model = None
        self.llm_interpreter = None
        self.metadata_df = None
    
    def load_yolo(self):
        """Load YOLO model"""
        if self.yolo_model is not None:
            return
        
        from ultralytics import YOLO
        
        model_path = self.config.get("pipeline", {}).get("yolo_model", "models/yolo/best.pt")
        
        if Path(model_path).exists():
            print(f"Loading YOLO from {model_path}")
            self.yolo_model = YOLO(model_path)
        else:
            # Use pretrained model
            print("Using pretrained YOLOv8m")
            self.yolo_model = YOLO("yolov8m.pt")
    
    def load_unet(self):
        """Load U-Net model"""
        if self.unet_model is not None:
            return
        
        from src.models.unet import get_unet_model
        
        model_path = self.config.get("pipeline", {}).get("unet_model", "models/unet/best.pth")
        
        if Path(model_path).exists():
            print(f"Loading U-Net from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            config = checkpoint.get("config", {})
            self.unet_model = get_unet_model(
                encoder_name=config.get("model", {}).get("encoder", "resnet34"),
                num_classes=config.get("model", {}).get("num_classes", 2),
            )
            self.unet_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print("Creating new U-Net model (no pretrained weights)")
            self.unet_model = get_unet_model()
        
        self.unet_model = self.unet_model.to(self.device)
        self.unet_model.eval()
    
    def load_llm(self):
        """Load LLM interpreter"""
        if self.llm_interpreter is not None:
            return
        
        # Check if LLM is disabled
        if not self.use_llm:
            from src.inference.llm_interpreter import SimpleInterpreter
            self.llm_interpreter = SimpleInterpreter()
            return
        
        llm_config = self.config.get("pipeline", {}).get("llm", {})
        
        if llm_config.get("enabled", True):
            from src.inference.llm_interpreter import LLMInterpreter
            self.llm_interpreter = LLMInterpreter(
                model_name=llm_config.get("model", "Salesforce/blip-image-captioning-large"),
                device=self.device if self.device != "mps" else "cpu",  # BLIP may not work on MPS
            )
            self.llm_interpreter.load()
        else:
            from src.inference.llm_interpreter import SimpleInterpreter
            self.llm_interpreter = SimpleInterpreter()
    
    def load_metadata(self):
        """Load CSV metadata"""
        if self.metadata_df is not None:
            return
        
        csv_path = self.config.get("pipeline", {}).get("metadata_csv", "dataset.csv")
        
        if Path(csv_path).exists():
            self.metadata_df = pd.read_csv(csv_path)
            print(f"Loaded metadata: {len(self.metadata_df)} records")
        else:
            print(f"Warning: CSV not found at {csv_path}")
            self.metadata_df = pd.DataFrame()
    
    def load_all(self):
        """Load all models"""
        print("Loading pipeline components...")
        self.load_yolo()
        self.load_unet()
        self.load_llm()
        self.load_metadata()
        print("Pipeline ready!")
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        YOLO Detection
        
        Args:
            image: BGR image (OpenCV format)
        
        Returns:
            Detection results with boxes, confidences, classes
        """
        self.load_yolo()
        
        # Run YOLO
        results = self.yolo_model(image, verbose=False)
        
        # Parse results
        boxes = []
        confidences = []
        class_ids = []
        
        for r in results:
            for box in r.boxes:
                boxes.append(box.xyxy[0].cpu().numpy())  # [x1, y1, x2, y2]
                confidences.append(float(box.conf[0]))
                class_ids.append(int(box.cls[0]))
        
        return {
            "boxes": boxes,
            "confidences": confidences,
            "class_ids": class_ids,
            "num_detections": len(boxes),
        }
    
    def segment(self, image: np.ndarray, box: List[float] = None) -> Dict[str, Any]:
        """
        U-Net Segmentation
        
        Args:
            image: BGR image
            box: Optional bounding box to crop [x1, y1, x2, y2]
        
        Returns:
            Segmentation results with mask and coverage
        """
        self.load_unet()
        
        full_h, full_w = image.shape[:2]

        # Crop if box provided
        x1 = y1 = x2 = y2 = None
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            # Clamp to image bounds
            x1 = max(0, min(x1, full_w - 1))
            x2 = max(0, min(x2, full_w))
            y1 = max(0, min(y1, full_h - 1))
            y2 = max(0, min(y2, full_h))
            if x2 <= x1 or y2 <= y1:
                crop = image
                x1 = y1 = x2 = y2 = None
            else:
                crop = image[y1:y2, x1:x2]
        else:
            crop = image
        
        # Preprocess
        img_size = self.config.get("pipeline", {}).get("unet_img_size", 256)
        original_size = crop.shape[:2]
        
        # Resize
        resized = cv2.resize(crop, (img_size, img_size))
        
        # Normalize
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        normalized = (normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # To tensor
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
        tensor = tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.unet_model(tensor)
            pred = torch.softmax(output, dim=1)
            mask = pred.argmax(dim=1).squeeze().cpu().numpy()
        
        # Resize mask back to crop size
        mask_crop = cv2.resize(
            mask.astype(np.uint8),
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # Coverage inside the crop (kept as coverage_percent for backwards compatibility)
        coverage_in_box = (mask_crop > 0).sum() / mask_crop.size * 100

        # Re-project crop mask back into full image coordinates
        if x1 is not None:
            mask_full = np.zeros((full_h, full_w), dtype=np.uint8)
            mask_full[y1:y2, x1:x2] = mask_crop
        else:
            mask_full = mask_crop

        coverage_full = (mask_full > 0).sum() / mask_full.size * 100

        return {
            "mask": mask_full,
            "mask_crop": mask_crop,
            "coverage_percent": coverage_in_box,
            "coverage_full_percent": coverage_full,
            "original_size": original_size,
            "box": [x1, y1, x2, y2] if x1 is not None else None,
        }
    
    def lookup_metadata(self, sku: str) -> Optional[Dict]:
        """
        Lookup cattle metadata by SKU
        
        Args:
            sku: Cattle SKU (e.g., "BLF2001")
        
        Returns:
            Metadata dict or None
        """
        self.load_metadata()
        
        if self.metadata_df.empty:
            return None
        
        # Normalize SKU
        sku_normalized = sku.replace(" ", "").upper()
        
        # Search
        matches = self.metadata_df[
            self.metadata_df["sku"].str.replace(" ", "").str.upper() == sku_normalized
        ]
        
        if len(matches) > 0:
            return matches.iloc[0].to_dict()
        
        return None
    
    def interpret(
        self,
        image: Image.Image,
        detection_results: Dict,
        segmentation_results: Dict,
        metadata: Optional[Dict],
    ) -> Dict[str, Any]:
        """
        LLM Interpretation
        
        Args:
            image: PIL Image
            detection_results: YOLO output
            segmentation_results: U-Net output
            metadata: CSV metadata
        
        Returns:
            Interpretation report
        """
        self.load_llm()
        
        if hasattr(self.llm_interpreter, 'analyze_cattle'):
            return self.llm_interpreter.analyze_cattle(
                image=image,
                detection_results=detection_results,
                segmentation_results=segmentation_results,
                metadata=metadata,
            )
        else:
            return {"error": "Interpreter not available"}
    
    def process(
        self,
        image_path: str,
        sku: str = None,
        visualize: bool = True,
        save_output: bool = False,
        output_dir: str = "outputs",
    ) -> Dict[str, Any]:
        """
        Full pipeline processing
        
        Args:
            image_path: Path to image
            sku: Optional SKU for metadata lookup
            visualize: Whether to create visualization
            save_output: Whether to save outputs
            output_dir: Output directory
        
        Returns:
            Complete analysis results
        """
        print(f"\nProcessing: {image_path}")
        
        # Load image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return {"error": f"Could not load image: {image_path}"}
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Detect SKU from path if not provided
        if sku is None:
            sku = Path(image_path).parent.name
        
        results = {
            "image_path": image_path,
            "sku": sku,
            "detection": None,
            "segmentation": None,
            "metadata": None,
            "interpretation": None,
        }
        
        # 1. Detection
        print("  [1/4] Running YOLO detection...")
        results["detection"] = self.detect(image_bgr)
        print(f"        Found {results['detection']['num_detections']} cattle")
        
        # 2. Segmentation (on best detection or full image)
        print("  [2/4] Running U-Net segmentation...")
        if results["detection"]["boxes"]:
            # Use highest confidence detection
            best_idx = np.argmax(results["detection"]["confidences"])
            best_box = results["detection"]["boxes"][best_idx]
            results["segmentation"] = self.segment(image_bgr, best_box)
        else:
            results["segmentation"] = self.segment(image_bgr)
        print(f"        Coverage (in box): {results['segmentation']['coverage_percent']:.1f}%")
        if results["segmentation"].get("coverage_full_percent") is not None:
            print(f"        Coverage (full frame): {results['segmentation']['coverage_full_percent']:.1f}%")
        
        # 3. Metadata lookup
        print("  [3/4] Looking up metadata...")
        results["metadata"] = self.lookup_metadata(sku)
        if results["metadata"]:
            print(f"        Found: {results['metadata'].get('breed', 'N/A')} / {results['metadata'].get('weight_in_kg', 'N/A')}kg")
        else:
            print("        No metadata found")
        
        # 4. Interpretation
        print("  [4/4] Generating interpretation...")
        results["interpretation"] = self.interpret(
            image_pil,
            results["detection"],
            results["segmentation"],
            results["metadata"],
        )
        
        # Visualization
        if visualize or save_output:
            vis_image = self._create_visualization(
                image_bgr,
                results["detection"],
                results["segmentation"],
            )
            results["visualization"] = vis_image
            
            if save_output:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                vis_path = output_path / f"{Path(image_path).stem}_analysis.jpg"
                cv2.imwrite(str(vis_path), vis_image)
                print(f"  Saved: {vis_path}")
        
        return results
    
    def _create_visualization(
        self,
        image: np.ndarray,
        detection: Dict,
        segmentation: Dict,
    ) -> np.ndarray:
        """Create visualization with detection boxes and segmentation overlay"""
        vis = image.copy()
        
        # Draw segmentation overlay
        mask = segmentation.get("mask")
        if mask is not None:
            # `segment()` returns a full-size mask when a YOLO crop is used.
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            
            # Create colored overlay
            overlay = np.zeros_like(vis)
            overlay[mask > 0] = [0, 255, 0]  # Green for cattle
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        # Draw detection boxes
        for i, box in enumerate(detection.get("boxes", [])):
            x1, y1, x2, y2 = map(int, box)
            conf = detection["confidences"][i]
            
            # Box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Label
            label = f"Cattle {conf:.0%}"
            cv2.putText(vis, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 0, 255), 2)
        
        return vis
    
    def process_batch(
        self,
        image_paths: List[str],
        output_dir: str = "outputs",
    ) -> List[Dict]:
        """Process multiple images"""
        results = []

        for path in image_paths:
            result = self.process(
                path,
                visualize=True,
                save_output=True,
                output_dir=output_dir,
            )
            results.append(result)

        return results
    def process_sku_folder(
        self,
        sku: str,
        images_root: str = "images",
        output_dir: str = "outputs",
        max_images: int = 5,
    ) -> List[Dict]:
        """Process all images for a specific SKU"""
        # Find folder
        sku_folder = Path(images_root) / sku
        
        if not sku_folder.exists():
            # Try with space
            sku_folder = Path(images_root) / f"{sku[:3]} {sku[3:]}"
        
        if not sku_folder.exists():
            return [{"error": f"Folder not found for SKU: {sku}"}]
        
        # Get images
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        images = [p for p in sku_folder.iterdir() if p.suffix.lower() in extensions]
        images = images[:max_images]
        
        print(f"Processing {len(images)} images for SKU: {sku}")
        
        results = []
        for img_path in images:
            result = self.process(
                str(img_path),
                sku=sku,
                visualize=True,
                save_output=True,
                output_dir=f"{output_dir}/{sku}",
            )
            results.append(result)
        
        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Cattle Analysis Pipeline")
    parser.add_argument("--image", type=str, help="Path to image")
    parser.add_argument("--sku", type=str, help="Cattle SKU")
    parser.add_argument("--folder", type=str, help="Process all images in SKU folder")
    parser.add_argument("--config", default="configs/pipeline_config.yaml", help="Config file")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--no-vis", action="store_true", help="Skip visualization")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = CattlePipeline(args.config)
    pipeline.load_all()
    
    if args.folder:
        # Process folder
        results = pipeline.process_sku_folder(
            args.folder,
            output_dir=args.output,
        )
    elif args.image:
        # Process single image
        results = pipeline.process(
            args.image,
            sku=args.sku,
            visualize=not args.no_vis,
            save_output=True,
            output_dir=args.output,
        )
        
        # Print interpretation
        if "interpretation" in results and results["interpretation"]:
            print("\n" + results["interpretation"].get("full_report", ""))
    else:
        # Demo mode
        print("\n" + "=" * 60)
        print("CATTLE ANALYSIS PIPELINE - DEMO")
        print("=" * 60)
        print("\nUsage:")
        print("  python pipeline.py --image path/to/image.jpg")
        print("  python pipeline.py --image path/to/image.jpg --sku BLF2001")
        print("  python pipeline.py --folder BLF2001")
        print("\nExample:")
        print("  python src/inference/pipeline.py --folder BLF2001 --output outputs")


if __name__ == "__main__":
    main()
