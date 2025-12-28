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
