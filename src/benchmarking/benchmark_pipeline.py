"""
Full Pipeline Benchmark: Compare Segmentation + LLM approaches

Approaches:
1. U-Net + LLM (semantic segmentation)
2. SAM + LLM (Segment Anything Model)  
3. Mask R-CNN + LLM (instance detection + segmentation)

Metrics:
- End-to-end inference time
- GPU memory usage
- Segmentation quality (IoU, coverage)
- LLM response quality (manual evaluation needed)
"""

import torch
import time
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional
import json
from dataclasses import dataclass, asdict
import gc


@dataclass
class PipelineResult:
    """Store full pipeline results"""
    method: str
    image_path: str
    
    # Timing
    segmentation_time_ms: float
    llm_time_ms: float
    total_time_ms: float
    
    # Memory
    peak_gpu_memory_mb: float
    
    # Segmentation quality
    num_instances: int
    coverage_percent: float
    
    # LLM output
    llm_response_length: int
    has_bcs: bool
    has_breed: bool
    has_weight: bool
    
    success: bool
    error: Optional[str] = None


class PipelineBenchmark:
    """Benchmark full segmentation + LLM pipelines"""
    
    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.results: List[PipelineResult] = []
        
        # LLM (shared across all methods)
        self._llm = None
        
    def _get_gpu_memory(self) -> float:
        if self.device == "cuda":
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return 0.0
    
    def _reset_gpu_memory(self):
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        gc.collect()
    
    def load_llm(self):
        """Load LLM interpreter"""
        try:
            from src.inference.llm_interpreter import Qwen2VLInterpreter
            self._llm = Qwen2VLInterpreter(device=self.device)
            self._llm.load()
        except Exception as e:
            print(f"Warning: Could not load LLM: {e}")
            self._llm = None
    
    def run_llm_analysis(
        self, 
        image: Image.Image, 
        segmentation_results: Dict
    ) -> Dict:
        """Run LLM analysis on segmented image"""
        if self._llm is None:
            return {
                "response": "",
                "has_bcs": False,
                "has_breed": False,
                "has_weight": False,
            }
        
        result = self._llm.analyze_cattle(
            image=image,
            segmentation_results=segmentation_results,
        )
        
        response = result.get("health_assessment", "")
        
        return {
            "response": response,
            "response_length": len(response),
            "has_bcs": "body condition" in response.lower() or "bcs" in response.lower(),
            "has_breed": "breed" in response.lower(),
            "has_weight": "kg" in response.lower() or "weight" in response.lower(),
        }
    
    # ==================== Method 1: U-Net + LLM ====================
    def run_unet_pipeline(self, image_path: str) -> PipelineResult:
        """Run U-Net segmentation + LLM"""
        from src.benchmarking.benchmark_segmentation import SegmentationBenchmark
        
        self._reset_gpu_memory()
        image = Image.open(image_path).convert("RGB")
        
        # Segmentation
        seg_benchmark = SegmentationBenchmark(device=self.device)
        seg_start = time.perf_counter()
        seg_result = seg_benchmark.run_unet(image)
        seg_time = (time.perf_counter() - seg_start) * 1000
        
        # LLM
        llm_start = time.perf_counter()
        llm_result = self.run_llm_analysis(image, {
            "coverage_percent": seg_result["coverage_percent"],
            "mask": seg_result["mask"],
        })
        llm_time = (time.perf_counter() - llm_start) * 1000
        
        return PipelineResult(
            method="unet_llm",
            image_path=image_path,
            segmentation_time_ms=seg_time,
            llm_time_ms=llm_time,
            total_time_ms=seg_time + llm_time,
            peak_gpu_memory_mb=self._get_gpu_memory(),
            num_instances=seg_result["num_instances"],
            coverage_percent=seg_result["coverage_percent"],
            llm_response_length=llm_result.get("response_length", 0),
            has_bcs=llm_result.get("has_bcs", False),
            has_breed=llm_result.get("has_breed", False),
            has_weight=llm_result.get("has_weight", False),
            success=True,
        )
    
    # ==================== Method 2: SAM + LLM ====================
    def run_sam_pipeline(self, image_path: str) -> PipelineResult:
        """Run SAM segmentation + LLM"""
        from src.benchmarking.benchmark_segmentation import SegmentationBenchmark
        
        self._reset_gpu_memory()
        image = Image.open(image_path).convert("RGB")
        
        # Segmentation
        seg_benchmark = SegmentationBenchmark(device=self.device)
        seg_start = time.perf_counter()
        seg_result = seg_benchmark.run_sam(image)
        seg_time = (time.perf_counter() - seg_start) * 1000
        
        # LLM
        llm_start = time.perf_counter()
        llm_result = self.run_llm_analysis(image, {
            "coverage_percent": seg_result["coverage_percent"],
            "mask": seg_result["mask"],
        })
        llm_time = (time.perf_counter() - llm_start) * 1000
        
        return PipelineResult(
            method="sam_llm",
            image_path=image_path,
            segmentation_time_ms=seg_time,
            llm_time_ms=llm_time,
            total_time_ms=seg_time + llm_time,
            peak_gpu_memory_mb=self._get_gpu_memory(),
            num_instances=seg_result["num_instances"],
            coverage_percent=seg_result["coverage_percent"],
            llm_response_length=llm_result.get("response_length", 0),
            has_bcs=llm_result.get("has_bcs", False),
            has_breed=llm_result.get("has_breed", False),
            has_weight=llm_result.get("has_weight", False),
            success=True,
        )
    
    # ==================== Method 3: Mask R-CNN + LLM ====================
    def run_maskrcnn_pipeline(self, image_path: str) -> PipelineResult:
        """Run Mask R-CNN detection/segmentation + LLM"""
        from src.benchmarking.benchmark_segmentation import SegmentationBenchmark
        
        self._reset_gpu_memory()
        image = Image.open(image_path).convert("RGB")
        
        # Detection + Segmentation
        seg_benchmark = SegmentationBenchmark(device=self.device)
        seg_start = time.perf_counter()
        seg_result = seg_benchmark.run_maskrcnn(image)
        seg_time = (time.perf_counter() - seg_start) * 1000
        
        # LLM
        llm_start = time.perf_counter()
        llm_result = self.run_llm_analysis(image, {
            "coverage_percent": seg_result["coverage_percent"],
            "mask": seg_result["mask"],
            "boxes": seg_result.get("boxes", []),
        })
        llm_time = (time.perf_counter() - llm_start) * 1000
        
        return PipelineResult(
            method="maskrcnn_llm",
            image_path=image_path,
            segmentation_time_ms=seg_time,
            llm_time_ms=llm_time,
            total_time_ms=seg_time + llm_time,
            peak_gpu_memory_mb=self._get_gpu_memory(),
            num_instances=seg_result["num_instances"],
            coverage_percent=seg_result["coverage_percent"],
            llm_response_length=llm_result.get("response_length", 0),
            has_bcs=llm_result.get("has_bcs", False),
            has_breed=llm_result.get("has_breed", False),
            has_weight=llm_result.get("has_weight", False),
            success=True,
        )
    
    # ==================== Benchmark Runner ====================
    def benchmark_image(self, image_path: str, methods: List[str] = None) -> List[PipelineResult]:
        """Benchmark all pipeline methods on a single image"""
        if methods is None:
            methods = ["unet_llm", "sam_llm", "maskrcnn_llm"]
        
        results = []
        
        for method in methods:
            try:
                if method == "unet_llm":
                    result = self.run_unet_pipeline(image_path)
                elif method == "sam_llm":
                    result = self.run_sam_pipeline(image_path)
                elif method == "maskrcnn_llm":
                    result = self.run_maskrcnn_pipeline(image_path)
                else:
                    continue
                    
                results.append(result)
                self.results.append(result)
                
            except Exception as e:
                result = PipelineResult(
                    method=method,
                    image_path=image_path,
                    segmentation_time_ms=0,
                    llm_time_ms=0,
                    total_time_ms=0,
                    peak_gpu_memory_mb=0,
                    num_instances=0,
                    coverage_percent=0,
                    llm_response_length=0,
                    has_bcs=False,
                    has_breed=False,
                    has_weight=False,
                    success=False,
                    error=str(e),
                )
                results.append(result)
                self.results.append(result)
        
        return results
    
    def benchmark_dataset(
        self, 
        image_dir: str, 
        methods: List[str] = None, 
        max_images: int = None
    ):
        """Benchmark on directory of images"""
        image_paths = list(Path(image_dir).rglob("*.jpg")) + \
                     list(Path(image_dir).rglob("*.png")) + \
                     list(Path(image_dir).rglob("*.jpeg"))
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        # Load LLM once
        print("Loading LLM...")
        self.load_llm()
        
        print(f"\nBenchmarking {len(image_paths)} images with methods: {methods}")
        
        for i, path in enumerate(image_paths):
            print(f"\n[{i+1}/{len(image_paths)}] {path.name}")
            self.benchmark_image(str(path), methods)
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        summary = {}
        
        for method in ["unet_llm", "sam_llm", "maskrcnn_llm"]:
            method_results = [r for r in self.results if r.method == method and r.success]
            
            if not method_results:
                continue
            
            summary[method] = {
                "count": len(method_results),
                "segmentation_time_ms": {
                    "mean": np.mean([r.segmentation_time_ms for r in method_results]),
                    "std": np.std([r.segmentation_time_ms for r in method_results]),
                },
                "llm_time_ms": {
                    "mean": np.mean([r.llm_time_ms for r in method_results]),
                    "std": np.std([r.llm_time_ms for r in method_results]),
                },
                "total_time_ms": {
                    "mean": np.mean([r.total_time_ms for r in method_results]),
                    "std": np.std([r.total_time_ms for r in method_results]),
                },
                "coverage_percent": {
                    "mean": np.mean([r.coverage_percent for r in method_results]),
                    "std": np.std([r.coverage_percent for r in method_results]),
                },
                "peak_gpu_memory_mb": {
                    "mean": np.mean([r.peak_gpu_memory_mb for r in method_results]),
                    "max": np.max([r.peak_gpu_memory_mb for r in method_results]),
                },
                "llm_quality": {
                    "has_bcs_rate": np.mean([r.has_bcs for r in method_results]),
                    "has_breed_rate": np.mean([r.has_breed for r in method_results]),
                    "has_weight_rate": np.mean([r.has_weight for r in method_results]),
                },
            }
        
        return summary
    
    def print_comparison_table(self):
        """Print formatted comparison table"""
        summary = self.get_summary()
        
        print("\n" + "=" * 90)
        print("PIPELINE BENCHMARK: Segmentation + LLM")
        print("=" * 90)
        
        print(f"\n{'Method':<15} {'Seg (ms)':<12} {'LLM (ms)':<12} {'Total (ms)':<12} {'Coverage %':<12} {'Memory (MB)':<12}")
        print("-" * 90)
        
        for method, stats in summary.items():
            seg_str = f"{stats['segmentation_time_ms']['mean']:.0f}"
            llm_str = f"{stats['llm_time_ms']['mean']:.0f}"
            total_str = f"{stats['total_time_ms']['mean']:.0f}"
            cov_str = f"{stats['coverage_percent']['mean']:.1f}"
            mem_str = f"{stats['peak_gpu_memory_mb']['mean']:.0f}"
            
            print(f"{method:<15} {seg_str:<12} {llm_str:<12} {total_str:<12} {cov_str:<12} {mem_str:<12}")
        
        print("\n" + "-" * 90)
        print("LLM Output Quality:")
        print(f"{'Method':<15} {'Has BCS':<12} {'Has Breed':<12} {'Has Weight':<12}")
        print("-" * 50)
        
        for method, stats in summary.items():
            bcs = f"{stats['llm_quality']['has_bcs_rate']*100:.0f}%"
            breed = f"{stats['llm_quality']['has_breed_rate']*100:.0f}%"
            weight = f"{stats['llm_quality']['has_weight_rate']*100:.0f}%"
            print(f"{method:<15} {bcs:<12} {breed:<12} {weight:<12}")
        
        print("=" * 90)
    
    def save_results(self, output_path: str = "outputs/pipeline_benchmark.json"):
        """Save results to JSON"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "results": [asdict(r) for r in self.results],
            "summary": self.get_summary(),
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\nResults saved to {output_path}")


def run_pipeline_benchmark(
    image_dir: str = "images",
    output_path: str = "outputs/pipeline_benchmark.json",
    methods: List[str] = None,
    max_images: int = 10,
):
    """Run full pipeline benchmark"""
    if methods is None:
        methods = ["unet_llm", "sam_llm", "maskrcnn_llm"]
    
    benchmark = PipelineBenchmark()
    benchmark.benchmark_dataset(image_dir, methods=methods, max_images=max_images)
    benchmark.print_comparison_table()
    benchmark.save_results(output_path)
    
    return benchmark.get_summary()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark segmentation + LLM pipelines")
    parser.add_argument("--images", type=str, default="images", help="Image directory")
    parser.add_argument("--output", type=str, default="outputs/pipeline_benchmark.json")
    parser.add_argument("--methods", type=str, nargs="+", 
                       default=["unet_llm", "sam_llm", "maskrcnn_llm"],
                       help="Methods to benchmark")
    parser.add_argument("--max-images", type=int, default=10)
    
    args = parser.parse_args()
    
    run_pipeline_benchmark(
        image_dir=args.images,
        output_path=args.output,
        methods=args.methods,
        max_images=args.max_images,
    )
