"""
Benchmarking Analysis: Segmentation Approaches for Cattle Analysis

Compare three approaches:
1. U-Net + LLM (semantic segmentation)
2. SAM + LLM (Segment Anything Model - instance segmentation)
3. Mask R-CNN + LLM (detection + instance segmentation)

Metrics:
- Inference time (ms)
- GPU memory usage (MB)
- Segmentation IoU (if ground truth available)
- Coverage accuracy
"""

import torch
import time
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
import gc


@dataclass
class BenchmarkResult:
    """Store benchmark results for a single run"""
    method: str
    image_path: str
    inference_time_ms: float
    gpu_memory_mb: float
    num_instances: int
    total_coverage_percent: float
    success: bool
    error: Optional[str] = None
    

class SegmentationBenchmark:
    """Benchmark different segmentation approaches"""
    
    def __init__(self, device: str = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        self.results: List[BenchmarkResult] = []
        
        # Models (lazy loaded)
        self._unet = None
        self._sam = None
        self._maskrcnn = None
        
    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB"""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def _clear_gpu_cache(self):
        """Clear GPU cache between runs"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    # ==================== U-Net ====================
    def load_unet(self, model_path: str = "models/unet/best.pth"):
        """Load U-Net model"""
        from src.models.unet import get_unet_model
        
        print("Loading U-Net...")
        
        if Path(model_path).exists():
            # weights_only=False needed for PyTorch 2.6+ (checkpoint contains numpy arrays)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Get config from checkpoint if available
            config = checkpoint.get("config", {})
            self._unet = get_unet_model(
                encoder_name=config.get("model", {}).get("encoder", "resnet34"),
                num_classes=config.get("model", {}).get("num_classes", 2),
            )
            
            # Load state dict from checkpoint (training saves as dict with "model_state_dict" key)
            if "model_state_dict" in checkpoint:
                self._unet.load_state_dict(checkpoint["model_state_dict"])
            else:
                # Fallback for old format where state_dict was saved directly
                self._unet.load_state_dict(checkpoint)
        else:
            print(f"Warning: Model not found at {model_path}, using untrained model")
            self._unet = get_unet_model()
        
        self._unet.to(self.device)
        self._unet.eval()
        print(f"U-Net loaded on {self.device}")
        
    def run_unet(self, image: Image.Image, threshold: float = 0.5) -> Dict:
        """Run U-Net segmentation on full image"""
        import torchvision.transforms as T
        
        if self._unet is None:
            self.load_unet()
        
        # Preprocess
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self._unet(input_tensor)
            if isinstance(output, dict):
                output = output.get('out', output.get('logits', list(output.values())[0]))
            
            # Handle both single-class (sigmoid) and multi-class (softmax) outputs
            if output.shape[1] == 1:
                # Single channel - use sigmoid
                probs = torch.sigmoid(output).squeeze().cpu().numpy()
            else:
                # Multi-class (e.g., 2 classes: background, foreground) - use softmax
                probs = torch.softmax(output, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
        
        # Create mask
        mask = (probs > threshold).astype(np.uint8)
        
        # Resize mask to original image size
        from PIL import Image as PILImage
        mask_pil = PILImage.fromarray(mask * 255)
        mask_pil = mask_pil.resize(image.size, PILImage.NEAREST)
        mask = np.array(mask_pil) > 127
        
        coverage = mask.sum() / mask.size * 100
        
        return {
            "mask": mask,
            "num_instances": 1,  # U-Net gives semantic mask (all cattle = 1)
            "coverage_percent": coverage,
        }
    
    # ==================== SAM ====================
    def load_sam(self, model_type: str = "vit_b"):
        """Load Segment Anything Model"""
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError:
            print("Installing segment-anything...")
            import subprocess
            subprocess.run(["pip", "install", "segment-anything"], check=True)
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        
        print(f"Loading SAM ({model_type})...")
        
        # Download checkpoint if needed
        checkpoint_map = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth",
        }
        
        checkpoint_path = Path("models/sam") / checkpoint_map[model_type]
        
        if not checkpoint_path.exists():
            print(f"SAM checkpoint not found at {checkpoint_path}")
            print("Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
            # Try to download
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://dl.fbaipublicfiles.com/segment_anything/{checkpoint_map[model_type]}"
            print(f"Downloading from {url}...")
            import urllib.request
            urllib.request.urlretrieve(url, checkpoint_path)
        
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        sam.to(self.device)
        
        self._sam = SamAutomaticMaskGenerator(
            sam,
            points_per_side=16,  # Reduce for speed
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=1000,  # Filter small regions
        )
        
        print(f"SAM loaded on {self.device}")
    
    def run_sam(self, image: Image.Image) -> Dict:
        """Run SAM automatic mask generation"""
        if self._sam is None:
            self.load_sam()
        
        # Convert to numpy
        image_np = np.array(image)
        
        # Generate masks
        masks = self._sam.generate(image_np)
        
        # Filter for cattle-sized masks (heuristic: >5% and <80% of image)
        image_area = image_np.shape[0] * image_np.shape[1]
        cattle_masks = []
        
        for m in masks:
            mask_area = m["area"]
            area_ratio = mask_area / image_area
            if 0.05 < area_ratio < 0.80:
                cattle_masks.append(m)
        
        # Sort by area (largest first)
        cattle_masks.sort(key=lambda x: x["area"], reverse=True)
        
        # Combine masks
        if cattle_masks:
            combined_mask = np.zeros(image_np.shape[:2], dtype=bool)
            for m in cattle_masks[:5]:  # Top 5 largest
                combined_mask |= m["segmentation"]
            coverage = combined_mask.sum() / combined_mask.size * 100
        else:
            combined_mask = np.zeros(image_np.shape[:2], dtype=bool)
            coverage = 0.0
        
        return {
            "mask": combined_mask,
            "num_instances": len(cattle_masks),
            "coverage_percent": coverage,
            "all_masks": cattle_masks,
        }
    
    # ==================== Mask R-CNN ====================
    def load_maskrcnn(self):
        """Load Mask R-CNN (pretrained on COCO)"""
        from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
        
        print("Loading Mask R-CNN...")
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self._maskrcnn = maskrcnn_resnet50_fpn(weights=weights)
        self._maskrcnn.to(self.device)
        self._maskrcnn.eval()
        
        self._maskrcnn_transforms = weights.transforms()
        
        # COCO class for cattle: 21 = cow
        self._cow_class_id = 21
        
        print(f"Mask R-CNN loaded on {self.device}")
    
    def run_maskrcnn(self, image: Image.Image, conf_threshold: float = 0.5) -> Dict:
        """Run Mask R-CNN detection + segmentation"""
        if self._maskrcnn is None:
            self.load_maskrcnn()
        
        # Preprocess
        import torchvision.transforms.functional as F
        image_tensor = F.to_tensor(image).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self._maskrcnn([image_tensor])[0]
        
        # Filter for cattle (class 21 in COCO)
        cattle_indices = []
        for i, (label, score) in enumerate(zip(predictions["labels"], predictions["scores"])):
            # COCO: 21 = cow, 22 = elephant, 23 = bear, etc.
            # Also accept 20 (sheep) and 19 (horse) as similar animals
            if label.item() in [21, 20, 19] and score.item() > conf_threshold:
                cattle_indices.append(i)
        
        # Combine masks
        if cattle_indices:
            masks = predictions["masks"][cattle_indices].cpu().numpy()
            combined_mask = np.zeros(masks.shape[2:], dtype=bool)
            for i in range(masks.shape[0]):
                combined_mask |= (masks[i, 0] > 0.5)
            coverage = combined_mask.sum() / combined_mask.size * 100
            
            boxes = predictions["boxes"][cattle_indices].cpu().numpy()
            scores = predictions["scores"][cattle_indices].cpu().numpy()
        else:
            combined_mask = np.zeros((image.height, image.width), dtype=bool)
            coverage = 0.0
            boxes = np.array([])
            scores = np.array([])
        
        return {
            "mask": combined_mask,
            "num_instances": len(cattle_indices),
            "coverage_percent": coverage,
            "boxes": boxes,
            "scores": scores,
        }
    
    # ==================== Benchmark Runner ====================
    def benchmark_image(self, image_path: str, methods: List[str] = None) -> List[BenchmarkResult]:
        """Benchmark all methods on a single image"""
        if methods is None:
            methods = ["unet", "sam", "maskrcnn"]
        
        image = Image.open(image_path).convert("RGB")
        results = []
        
        for method in methods:
            self._clear_gpu_cache()
            
            mem_before = self._get_gpu_memory()
            start_time = time.perf_counter()
            
            try:
                if method == "unet":
                    output = self.run_unet(image)
                elif method == "sam":
                    output = self.run_sam(image)
                elif method == "maskrcnn":
                    output = self.run_maskrcnn(image)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                end_time = time.perf_counter()
                mem_after = self._get_gpu_memory()
                
                result = BenchmarkResult(
                    method=method,
                    image_path=image_path,
                    inference_time_ms=(end_time - start_time) * 1000,
                    gpu_memory_mb=mem_after - mem_before,
                    num_instances=output["num_instances"],
                    total_coverage_percent=output["coverage_percent"],
                    success=True,
                )
                
            except Exception as e:
                end_time = time.perf_counter()
                result = BenchmarkResult(
                    method=method,
                    image_path=image_path,
                    inference_time_ms=(end_time - start_time) * 1000,
                    gpu_memory_mb=0,
                    num_instances=0,
                    total_coverage_percent=0,
                    success=False,
                    error=str(e),
                )
            
            results.append(result)
            self.results.append(result)
            
        return results
    
    def benchmark_dataset(self, image_dir: str, methods: List[str] = None, max_images: int = None) -> Dict:
        """Benchmark on a directory of images"""
        image_paths = list(Path(image_dir).rglob("*.jpg")) + \
                     list(Path(image_dir).rglob("*.png")) + \
                     list(Path(image_dir).rglob("*.jpeg"))
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        print(f"Benchmarking {len(image_paths)} images...")
        
        for i, path in enumerate(image_paths):
            print(f"  [{i+1}/{len(image_paths)}] {path.name}")
            self.benchmark_image(str(path), methods)
        
        return self.get_summary()
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.results:
            return {}
        
        summary = {}
        
        for method in ["unet", "sam", "maskrcnn"]:
            method_results = [r for r in self.results if r.method == method and r.success]
            
            if not method_results:
                continue
            
            times = [r.inference_time_ms for r in method_results]
            coverage = [r.total_coverage_percent for r in method_results]
            instances = [r.num_instances for r in method_results]
            memory = [r.gpu_memory_mb for r in method_results]
            
            summary[method] = {
                "count": len(method_results),
                "success_rate": len(method_results) / len([r for r in self.results if r.method == method]),
                "inference_time_ms": {
                    "mean": np.mean(times),
                    "std": np.std(times),
                    "min": np.min(times),
                    "max": np.max(times),
                },
                "coverage_percent": {
                    "mean": np.mean(coverage),
                    "std": np.std(coverage),
                },
                "num_instances": {
                    "mean": np.mean(instances),
                },
                "gpu_memory_mb": {
                    "mean": np.mean(memory),
                    "max": np.max(memory),
                },
            }
        
        return summary
    
    def save_results(self, output_path: str = "outputs/benchmark_results.json"):
        """Save results to JSON"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "results": [asdict(r) for r in self.results],
            "summary": self.get_summary(),
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Results saved to {output_path}")
    
    def print_summary(self):
        """Print formatted summary"""
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("SEGMENTATION BENCHMARK RESULTS")
        print("=" * 70)
        
        print(f"\n{'Method':<12} {'Time (ms)':<15} {'Coverage %':<15} {'Instances':<12} {'Memory (MB)':<12}")
        print("-" * 70)
        
        for method, stats in summary.items():
            time_str = f"{stats['inference_time_ms']['mean']:.1f} ± {stats['inference_time_ms']['std']:.1f}"
            cov_str = f"{stats['coverage_percent']['mean']:.1f} ± {stats['coverage_percent']['std']:.1f}"
            inst_str = f"{stats['num_instances']['mean']:.1f}"
            mem_str = f"{stats['gpu_memory_mb']['mean']:.1f}"
            
            print(f"{method:<12} {time_str:<15} {cov_str:<15} {inst_str:<12} {mem_str:<12}")
        
        print("=" * 70)


def run_benchmark(
    image_dir: str = "images",
    output_path: str = "outputs/benchmark_results.json",
    methods: List[str] = None,
    max_images: int = 20,
):
    """Run full benchmark"""
    benchmark = SegmentationBenchmark()
    benchmark.benchmark_dataset(image_dir, methods=methods, max_images=max_images)
    benchmark.print_summary()
    benchmark.save_results(output_path)
    return benchmark.get_summary()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark segmentation methods")
    parser.add_argument("--images", type=str, default="images", help="Image directory")
    parser.add_argument("--output", type=str, default="outputs/benchmark_results.json")
    parser.add_argument("--methods", type=str, nargs="+", default=["unet", "sam", "maskrcnn"])
    parser.add_argument("--max-images", type=int, default=20, help="Max images to test")
    
    args = parser.parse_args()
    
    run_benchmark(
        image_dir=args.images,
        output_path=args.output,
        methods=args.methods,
        max_images=args.max_images,
    )
