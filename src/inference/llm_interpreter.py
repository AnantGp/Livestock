"""
LLM Interpreter for Cattle Analysis
Uses BLIP for image captioning and interpretation
"""

import torch
from PIL import Image
from typing import Dict, List, Optional, Any
from pathlib import Path


class LLMInterpreter:
    """
    LLM-based interpreter for cattle image analysis
    Combines detection/segmentation results with metadata to generate explanations
    """
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-large",
        device: str = None,
    ):
        self.model_name = model_name
        
        # Set device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self.model = None
        self.processor = None
    
    def load(self):
        """Load BLIP model"""
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        print(f"Loading {self.model_name}...")
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        
        # Move to device (note: MPS may have issues with some models)
        if self.device == "mps":
            try:
                self.model = self.model.to(self.device)
            except:
                print("MPS not supported for BLIP, using CPU")
                self.device = "cpu"
                self.model = self.model.to(self.device)
        else:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def generate_caption(
        self,
        image: Image.Image,
        prompt: str = None,
        max_length: int = 100,
    ) -> str:
        """Generate caption for image"""
        if self.model is None:
            self.load()
        
        if prompt:
            # Conditional generation
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        else:
            # Unconditional generation
            inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=3,
            )
        
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    
    def analyze_cattle(
        self,
        image: Image.Image,
        detection_results: Dict = None,
        segmentation_results: Dict = None,
        metadata: Dict = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive cattle analysis combining all available information
        
        Args:
            image: PIL Image
            detection_results: YOLO detection output
            segmentation_results: UNet segmentation output
            metadata: CSV metadata for this cattle
        
        Returns:
            Analysis report
        """
        report = {
            "visual_description": "",
            "detection_summary": "",
            "segmentation_summary": "",
            "metadata_summary": "",
            "health_assessment": "",
            "recommendations": [],
        }
        
        # Visual description from BLIP
        report["visual_description"] = self.generate_caption(
            image,
            prompt="This image shows a"
        )
        
        # Specific cattle analysis
        cattle_description = self.generate_caption(
            image,
            prompt="The cattle in this image appears to be"
        )
        
        # Detection summary
        if detection_results:
            num_detections = len(detection_results.get("boxes", []))
            confidences = detection_results.get("confidences", [])
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            
            report["detection_summary"] = (
                f"Detected {num_detections} cattle with average confidence {avg_conf:.1%}. "
            )
        
        # Segmentation summary
        if segmentation_results:
            coverage = segmentation_results.get("coverage_percent", 0)
            report["segmentation_summary"] = (
                f"Cattle body covers {coverage:.1f}% of the image. "
            )
            
            if coverage < 20:
                report["recommendations"].append(
                    "Consider closer framing for better analysis"
                )
        
        # Metadata summary
        if metadata:
            report["metadata_summary"] = self._format_metadata(metadata)
            
            # Health assessment based on metadata
            report["health_assessment"] = self._assess_health(metadata)
            
            # Add recommendations based on metadata
            report["recommendations"].extend(self._generate_recommendations(metadata))
        
        # Combine into natural language report
        report["full_report"] = self._generate_full_report(
            report, cattle_description
        )
        
        return report
    
    def _format_metadata(self, metadata: Dict) -> str:
        """Format metadata into readable summary"""
        parts = []
        
        if metadata.get("sku"):
            parts.append(f"ID: {metadata['sku']}")
        if metadata.get("breed"):
            parts.append(f"Breed: {metadata['breed']}")
        if metadata.get("sex"):
            sex_map = {"MALE_BULL": "Bull", "FEMALE_HEIFER": "Heifer", "FEMALE_COW": "Cow"}
            parts.append(f"Sex: {sex_map.get(metadata['sex'], metadata['sex'])}")
        if metadata.get("color"):
            parts.append(f"Color: {metadata['color']}")
        if metadata.get("age_in_year"):
            parts.append(f"Age: {metadata['age_in_year']} years")
        if metadata.get("weight_in_kg"):
            parts.append(f"Weight: {metadata['weight_in_kg']} kg")
        if metadata.get("height_in_inch"):
            parts.append(f"Height: {metadata['height_in_inch']} inches")
        if metadata.get("teeth"):
            parts.append(f"Teeth: {metadata['teeth']}")
        
        return " | ".join(parts)
    
    def _assess_health(self, metadata: Dict) -> str:
        """Generate health assessment from metadata"""
        assessments = []
        
        # Weight assessment
        weight = metadata.get("weight_in_kg")
        age = metadata.get("age_in_year", 0)
        
        if weight and age:
            # Simple heuristic for weight assessment
            expected_weight_per_year = 200  # kg (rough estimate)
            expected = age * expected_weight_per_year
            
            if weight >= expected * 0.9:
                assessments.append("Weight appears appropriate for age")
            elif weight >= expected * 0.7:
                assessments.append("Weight is slightly below average for age")
            else:
                assessments.append("Weight may indicate underfeeding or health issues")
        
        # Teeth assessment (indicator of age/health)
        teeth = metadata.get("teeth")
        if teeth:
            assessments.append(f"Dental development: {teeth}")
        
        return ". ".join(assessments) if assessments else "Insufficient data for health assessment"
    
    def _generate_recommendations(self, metadata: Dict) -> List[str]:
        """Generate recommendations based on metadata"""
        recommendations = []
        
        # Feed recommendations
        feed = metadata.get("feed", "").lower()
        if "silage" in feed:
            recommendations.append("Current diet includes silage - ensure proper fermentation")
        if "hay" in feed:
            recommendations.append("Supplement hay diet with minerals if not already done")
        
        # Age-based recommendations
        age = metadata.get("age_in_year", 0)
        if age < 1:
            recommendations.append("Young cattle - monitor growth rate closely")
        elif age > 5:
            recommendations.append("Mature cattle - regular health checkups recommended")
        
        return recommendations
    
    def _generate_full_report(
        self,
        report: Dict,
        cattle_description: str,
    ) -> str:
        """Generate comprehensive natural language report"""
        
        lines = [
            "=" * 50,
            "CATTLE ANALYSIS REPORT",
            "=" * 50,
            "",
            f"Visual Analysis: {cattle_description}",
            "",
        ]
        
        if report["metadata_summary"]:
            lines.extend([
                "Animal Information:",
                f"  {report['metadata_summary']}",
                "",
            ])
        
        if report["detection_summary"]:
            lines.extend([
                "Detection Results:",
                f"  {report['detection_summary']}",
                "",
            ])
        
        if report["segmentation_summary"]:
            lines.extend([
                "Segmentation Results:",
                f"  {report['segmentation_summary']}",
                "",
            ])
        
        if report["health_assessment"]:
            lines.extend([
                "Health Assessment:",
                f"  {report['health_assessment']}",
                "",
            ])
        
        if report["recommendations"]:
            lines.extend([
                "Recommendations:",
            ])
            for rec in report["recommendations"]:
                lines.append(f"  • {rec}")
            lines.append("")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)


# Simpler prompt-based interpretation without heavy LLM
class SimpleInterpreter:
    """
    Rule-based interpreter as fallback when LLM is not available
    """
    
    def analyze_cattle(
        self,
        image=None,  # Ignored, for API compatibility with LLMInterpreter
        detection_results: Dict = None,
        segmentation_results: Dict = None,
        metadata: Dict = None,
    ) -> Dict[str, Any]:
        """Generate analysis without image captioning"""
        
        report = {
            "detection_summary": "",
            "segmentation_summary": "",
            "metadata_summary": "",
            "health_assessment": "",
            "recommendations": [],
        }
        
        # Detection summary
        if detection_results:
            boxes = detection_results.get("boxes", [])
            confs = detection_results.get("confidences", [])
            
            if boxes:
                avg_conf = sum(confs) / len(confs)
                report["detection_summary"] = (
                    f"Detected {len(boxes)} cattle. "
                    f"Average confidence: {avg_conf:.1%}. "
                )
        
        # Segmentation summary
        if segmentation_results:
            coverage = segmentation_results.get("coverage_percent", 0)
            report["segmentation_summary"] = (
                f"Cattle body covers {coverage:.1f}% of frame. "
            )
        
        # Metadata summary
        if metadata:
            parts = []
            if metadata.get("breed"):
                parts.append(f"{metadata['breed']} breed")
            if metadata.get("sex"):
                sex = "bull" if "MALE" in str(metadata.get("sex", "")) else "cow/heifer"
                parts.append(sex)
            if metadata.get("weight_in_kg"):
                parts.append(f"{metadata['weight_in_kg']}kg")
            if metadata.get("age_in_year"):
                parts.append(f"{metadata['age_in_year']}yr old")
            
            report["metadata_summary"] = ", ".join(parts)
        
        return report


class Qwen2VLInterpreter:
    """
    Qwen2.5-VL based interpreter for detailed cattle health analysis.
    This model can actually reason about cattle health, body condition, breed, etc.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = None,
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.llm_config = llm_config or {}
        
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self.model = None
        self.processor = None
    
    def load(self):
        """Load Qwen2.5-VL model"""
        from transformers import AutoProcessor
        
        print(f"Loading {self.model_name}...")
        print("This may take a few minutes on first run...")
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        # Qwen2.5-VL-7B is too large for many ~15GB GPUs in fp16 once other models
        # (YOLO/U-Net) are resident. Default to 4-bit on CUDA unless explicitly disabled.
        use_cuda = self.device == "cuda"
        cfg = self.llm_config or {}
        load_in_4bit = bool(cfg.get("load_in_4bit", True if use_cuda else False))
        load_in_8bit = bool(cfg.get("load_in_8bit", False))

        quantization_config = None
        if use_cuda and (load_in_4bit or load_in_8bit):
            try:
                from transformers import BitsAndBytesConfig

                compute_dtype_name = str(cfg.get("bnb_4bit_compute_dtype", "float16")).lower()
                if compute_dtype_name in {"bf16", "bfloat16"}:
                    compute_dtype = torch.bfloat16
                else:
                    compute_dtype = torch.float16

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_quant_type=str(cfg.get("bnb_4bit_quant_type", "nf4")),
                    bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
                    bnb_4bit_compute_dtype=compute_dtype,
                )
            except Exception as e:
                print(f"Warning: bitsandbytes quantization requested but unavailable ({e}). Falling back to fp16.")
                quantization_config = None

        # Prefer the newer class name when available (AutoModelForVision2Seq is deprecated).
        model_cls = None
        try:
            from transformers import AutoModelForImageTextToText as _Model
            model_cls = _Model
        except Exception:
            from transformers import AutoModelForVision2Seq as _Model
            model_cls = _Model

        self.model = model_cls.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            device_map="auto" if use_cuda else None,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print(f"Qwen2.5-VL loaded on {self.device}")
    
    def _get_weight_reference(self) -> str:
        """Get weight reference data learned from the dataset"""
        try:
            from src.inference.weight_predictor import get_weight_predictor
            predictor = get_weight_predictor(self.llm_config.get("metadata_csv", "dataset.csv"))
            return predictor.get_breed_context_for_llm()
        except Exception as e:
            # Fallback hardcoded reference if predictor fails
            return """WEIGHT REFERENCE DATA (from actual cattle measurements):
- LOCAL: 200-270 kg typical (range: 150-510 kg, most common breed)
- SAHIWAL: 250-310 kg typical (range: 205-698 kg)
- SINDHI: 250-320 kg typical (range: 210-546 kg)
- RED_CHITTAGONG: 200-260 kg typical (range: 170-286 kg)
- HOSTINE_CROSS: 330-470 kg typical (range: 202-545 kg)
- BRAHMA: 600-750 kg typical (large breed)

SIZE CATEGORIES:
- MINIMUM: 150-293 kg (mean: 208)
- MEDIUM: 176-511 kg (mean: 257)
- LARGE: 256-606 kg (mean: 376)
- EXTRA_LARGE: 315-816 kg (mean: 530)"""

    def analyze_cattle(
        self,
        image: Image.Image,
        detection_results: Dict = None,
        segmentation_results: Dict = None,
        metadata: Dict = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive cattle analysis using Qwen2.5-VL vision-language model.
        """
        if self.model is None:
            self.load()
        
        # Build context from detection/segmentation/metadata
        context_parts = []
        
        if detection_results:
            num_det = len(detection_results.get("boxes", []))
            confs = detection_results.get("confidences", [])
            avg_conf = sum(confs) / len(confs) if confs else 0
            context_parts.append(f"Detection: {num_det} cattle detected (confidence: {avg_conf:.1%})")
        
        if segmentation_results:
            coverage = segmentation_results.get("coverage_percent", 0)
            context_parts.append(f"Body coverage: {coverage:.1f}% of frame")
        
        if metadata:
            meta_parts = []
            if metadata.get("breed"):
                meta_parts.append(f"Breed: {metadata['breed']}")
            if metadata.get("sex"):
                meta_parts.append(f"Sex: {metadata['sex']}")
            if metadata.get("weight_in_kg"):
                meta_parts.append(f"Weight: {metadata['weight_in_kg']}kg")
            if metadata.get("age_in_year"):
                meta_parts.append(f"Age: {metadata['age_in_year']} years")
            if metadata.get("height_in_inch"):
                meta_parts.append(f"Height: {metadata['height_in_inch']} inches")
            if meta_parts:
                context_parts.append("Known data: " + ", ".join(meta_parts))
        
        # Add learned weight reference from dataset
        weight_reference = self._get_weight_reference()
        
        context = "\n".join(context_parts) if context_parts else ""
        
        # Build the prompt for cattle health analysis
        prompt = f"""Analyze this cattle image and provide a detailed health assessment.

{context}

{weight_reference}

IMPORTANT GUIDELINES:
- Use the WEIGHT REFERENCE DATA above (from real cattle measurements) to guide your weight estimates.
- Indian indigenous cattle (LOCAL, SAHIWAL, SINDHI, RED_CHITTAGONG) are typically 180-350 kg.
- Only BRAHMA and HOLSTEIN_CROSS breeds commonly exceed 400 kg in this dataset.
- Weight estimation from images is inherently uncertain. Provide a range, not a single number.

Please evaluate:
1. Body Condition Score (BCS) on 1-5 scale (1=emaciated, 3=ideal, 5=obese)
   - Base this on visible ribs, spine, hip bones, and muscle/fat cover
2. Breed type (describe physical features: hump, horns, coat color, ear shape)
   - Try to match to one of the known breeds: LOCAL, SAHIWAL, SINDHI, HOSTINE_CROSS, RED_CHITTAGONG, BRAHMA
3. Estimated weight range
   - Use the WEIGHT REFERENCE DATA to give realistic ranges for the identified breed type
   - State confidence level (low/medium/high)
4. Health indicators (coat condition, posture, alertness, eyes, any discharge)
5. Any visible concerns or abnormalities
6. Overall health assessment

Be specific and use the reference data for accurate weight estimation."""

        # Prepare inputs for Qwen2.5-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Process with Qwen2.5-VL
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generation can OOM on ~15GB GPUs, especially if earlier vision models
        # are still resident. Retry with a lower-memory config, and if it still
        # fails, degrade gracefully instead of crashing the whole pipeline.
        max_new_tokens = int(self.llm_config.get("max_new_tokens", 512))
        do_sample = bool(self.llm_config.get("do_sample", True))
        temperature = float(self.llm_config.get("temperature", 0.7))
        use_cache = bool(self.llm_config.get("use_cache", True))

        oom_fallback_enabled = bool(self.llm_config.get("oom_fallback", True))
        oom_fallback_max_new_tokens = int(self.llm_config.get("oom_fallback_max_new_tokens", 256))

        def _try_generate(gen_kwargs: Dict[str, Any]):
            with torch.no_grad():
                return self.model.generate(**inputs, **gen_kwargs)

        try:
            generated_ids = _try_generate(
                {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "temperature": temperature,
                    "use_cache": use_cache,
                }
            )
        except torch.OutOfMemoryError as e:
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            if not oom_fallback_enabled:
                raise

            print(
                "Warning: CUDA OOM during LLM generation. Retrying with a lower-memory configuration "
                "(fewer tokens, no KV cache, greedy decoding)."
            )

            try:
                generated_ids = _try_generate(
                    {
                        "max_new_tokens": min(max_new_tokens, oom_fallback_max_new_tokens),
                        "do_sample": False,
                        "temperature": 0.0,
                        "use_cache": False,
                    }
                )
            except torch.OutOfMemoryError:
                if self.device == "cuda" and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

                # Return an actionable error payload instead of throwing.
                return {
                    "error": (
                        "CUDA out of memory while generating the LLM interpretation. "
                        "Try enabling 4-bit quantization (install bitsandbytes), reducing max_new_tokens, "
                        "or run with --no-llm."
                    ),
                    "visual_description": "",
                    "detection_summary": "",
                    "segmentation_summary": "",
                    "metadata_summary": "",
                    "health_assessment": "",
                    "recommendations": [],
                    "full_report": (
                        "=" * 60
                        + "\nCATTLE HEALTH ANALYSIS REPORT (Qwen2.5-VL)\n"
                        + "=" * 60
                        + "\n\n"
                        + "LLM interpretation unavailable due to CUDA OOM.\n"
                        + "Suggestions:\n"
                        + "- Install/enable bitsandbytes 4-bit quantization\n"
                        + "- Lower pipeline.llm.max_new_tokens in configs/pipeline_config.yaml\n"
                        + "- Run demo.py with --no-llm\n\n"
                        + "=" * 60
                    ),
                }
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        # Build report
        report = {
            "visual_description": response,
            "detection_summary": "",
            "segmentation_summary": "",
            "metadata_summary": "",
            "health_assessment": response,
            "recommendations": [],
        }
        
        if detection_results:
            num_det = len(detection_results.get("boxes", []))
            confs = detection_results.get("confidences", [])
            avg_conf = sum(confs) / len(confs) if confs else 0
            report["detection_summary"] = f"Detected {num_det} cattle (avg conf: {avg_conf:.1%})"
        
        if segmentation_results:
            coverage = segmentation_results.get("coverage_percent", 0)
            report["segmentation_summary"] = f"Body coverage: {coverage:.1f}%"
        
        if metadata:
            parts = []
            for key in ["sku", "breed", "sex", "weight_in_kg", "age_in_year"]:
                if metadata.get(key):
                    parts.append(f"{key}: {metadata[key]}")
            report["metadata_summary"] = " | ".join(parts)
        
        # Generate full report
        report["full_report"] = self._generate_full_report(report)
        
        return report
    
    def _generate_full_report(self, report: Dict) -> str:
        """Generate formatted report"""
        lines = [
            "=" * 60,
            "CATTLE HEALTH ANALYSIS REPORT (Qwen2.5-VL)",
            "=" * 60,
            "",
        ]
        
        if report["metadata_summary"]:
            lines.extend([
                "Known Animal Data:",
                f"  {report['metadata_summary']}",
                "",
            ])
        
        if report["detection_summary"]:
            lines.extend([
                "Detection:",
                f"  {report['detection_summary']}",
                "",
            ])
        
        if report["segmentation_summary"]:
            lines.extend([
                "Segmentation:",
                f"  {report['segmentation_summary']}",
                "",
            ])
        
        lines.extend([
            "Health Analysis:",
            "-" * 40,
            report["health_assessment"],
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)


# Test
if __name__ == "__main__":
    # Test simple interpreter
    interpreter = SimpleInterpreter()
    
    result = interpreter.analyze_cattle(
        detection_results={"boxes": [[10, 10, 100, 100]], "confidences": [0.95]},
        segmentation_results={"coverage_percent": 35.5},
        metadata={
            "sku": "BLF2001",
            "breed": "MURRAH",
            "sex": "MALE_BULL",
            "weight_in_kg": 450,
            "age_in_year": 2,
        }
    )
    
    print("Detection:", result["detection_summary"])
    print("Segmentation:", result["segmentation_summary"])
    print("Metadata:", result["metadata_summary"])
