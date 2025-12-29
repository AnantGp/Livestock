"""
Weight Predictor - Learn from metadata to improve weight estimation

Uses the dataset's actual breed/height/weight data to make better predictions
than pure LLM guessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import pickle


class WeightPredictor:
    """
    Predicts cattle weight using learned patterns from metadata.
    
    Features used:
    - Breed → weight range lookup
    - Size category → weight range
    - Height (if available) → regression
    - Segmentation coverage → relative size estimation
    """
    
    def __init__(self, metadata_csv: str = "dataset.csv"):
        self.metadata_csv = metadata_csv
        self.df = None
        self.breed_stats = {}
        self.size_stats = {}
        self.height_weight_model = None
        self.overall_stats = {}
        
    def load_and_learn(self):
        """Load metadata and learn weight patterns"""
        csv_path = Path(self.metadata_csv)
        if not csv_path.exists():
            print(f"Warning: Metadata CSV not found at {csv_path}")
            return False
            
        self.df = pd.read_csv(csv_path)
        
        # Learn breed → weight mapping
        self.breed_stats = {}
        for breed in self.df['breed'].unique():
            breed_data = self.df[self.df['breed'] == breed]['weight_in_kg']
            self.breed_stats[breed.upper()] = {
                'mean': breed_data.mean(),
                'std': breed_data.std(),
                'min': breed_data.min(),
                'max': breed_data.max(),
                'count': len(breed_data),
                'q25': breed_data.quantile(0.25),
                'q75': breed_data.quantile(0.75),
            }
        
        # Learn size category → weight mapping  
        self.size_stats = {}
        for size in self.df['size'].unique():
            size_data = self.df[self.df['size'] == size]['weight_in_kg']
            self.size_stats[size.upper()] = {
                'mean': size_data.mean(),
                'std': size_data.std(),
                'min': size_data.min(),
                'max': size_data.max(),
                'count': len(size_data),
            }
        
        # Overall stats
        self.overall_stats = {
            'mean': self.df['weight_in_kg'].mean(),
            'std': self.df['weight_in_kg'].std(),
            'min': self.df['weight_in_kg'].min(),
            'max': self.df['weight_in_kg'].max(),
        }
        
        # Simple linear model: weight = a * height + b
        valid_data = self.df.dropna(subset=['height_in_inch', 'weight_in_kg'])
        if len(valid_data) > 10:
            heights = valid_data['height_in_inch'].values
            weights = valid_data['weight_in_kg'].values
            # Simple linear regression
            A = np.vstack([heights, np.ones(len(heights))]).T
            self.height_weight_model = np.linalg.lstsq(A, weights, rcond=None)[0]
            
        print(f"WeightPredictor: Learned from {len(self.df)} records")
        print(f"  Breeds: {list(self.breed_stats.keys())}")
        print(f"  Sizes: {list(self.size_stats.keys())}")
        
        return True
    
    def predict_from_breed(self, breed: str) -> Optional[Dict]:
        """Get weight range based on breed"""
        breed_upper = breed.upper().strip()
        
        # Try exact match
        if breed_upper in self.breed_stats:
            stats = self.breed_stats[breed_upper]
            return {
                'source': 'breed_exact',
                'breed': breed_upper,
                'mean': stats['mean'],
                'range': (stats['q25'], stats['q75']),
                'full_range': (stats['min'], stats['max']),
                'confidence': min(0.9, 0.5 + stats['count'] / 100),  # More data = higher confidence
            }
        
        # Try partial match
        for known_breed, stats in self.breed_stats.items():
            if breed_upper in known_breed or known_breed in breed_upper:
                return {
                    'source': 'breed_partial',
                    'breed': known_breed,
                    'mean': stats['mean'],
                    'range': (stats['q25'], stats['q75']),
                    'full_range': (stats['min'], stats['max']),
                    'confidence': min(0.7, 0.3 + stats['count'] / 100),
                }
        
        # Map common breed names to our categories
        breed_mapping = {
            # Indian zebu breeds → LOCAL or specific
            'GIR': 'LOCAL',
            'THARPARKAR': 'LOCAL', 
            'KANKREJ': 'LOCAL',
            'ONGOLE': 'LOCAL',
            'HARIANA': 'LOCAL',
            'DEONI': 'LOCAL',
            'KANGAYAM': 'LOCAL',
            'HALLIKAR': 'LOCAL',
            'AMRITMAHAL': 'LOCAL',
            'KHILLARI': 'LOCAL',
            'DANGI': 'LOCAL',
            'NIMARI': 'LOCAL',
            'MALVI': 'LOCAL',
            'GAOLAO': 'LOCAL',
            'MEWATI': 'LOCAL',
            'NAGORI': 'LOCAL',
            'RATHI': 'LOCAL',
            'THAR': 'LOCAL',
            'ZEBU': 'LOCAL',
            'INDIAN': 'LOCAL',
            # European breeds → HOLSTEIN_CROSS or larger
            'HOLSTEIN': 'HOSTINE_CROSS',
            'FRIESIAN': 'HOSTINE_CROSS',
            'JERSEY': 'HOSTINE_CROSS',
            'ANGUS': 'BRAHMA',  # Large beef breed
            'HEREFORD': 'BRAHMA',
            'CHAROLAIS': 'BRAHMA',
            'LIMOUSIN': 'BRAHMA',
            'SIMMENTAL': 'BRAHMA',
        }
        
        for pattern, mapped_breed in breed_mapping.items():
            if pattern in breed_upper:
                if mapped_breed in self.breed_stats:
                    stats = self.breed_stats[mapped_breed]
                    return {
                        'source': 'breed_mapped',
                        'breed': f"{breed} → {mapped_breed}",
                        'mean': stats['mean'],
                        'range': (stats['q25'], stats['q75']),
                        'full_range': (stats['min'], stats['max']),
                        'confidence': 0.5,
                    }
        
        return None
    
    def predict_from_height(self, height_inches: float) -> Optional[Dict]:
        """Predict weight from height using learned regression"""
        if self.height_weight_model is None:
            return None
            
        a, b = self.height_weight_model
        predicted_weight = a * height_inches + b
        
        # Estimate uncertainty (rough)
        uncertainty = 30 + abs(height_inches - 46) * 3  # More uncertain far from mean height
        
        return {
            'source': 'height_regression',
            'height_inches': height_inches,
            'mean': predicted_weight,
            'range': (predicted_weight - uncertainty, predicted_weight + uncertainty),
            'confidence': 0.7,
        }
    
    def predict_from_size_category(self, size: str) -> Optional[Dict]:
        """Get weight range based on size category"""
        size_upper = size.upper().strip()
        
        if size_upper in self.size_stats:
            stats = self.size_stats[size_upper]
            return {
                'source': 'size_category',
                'size': size_upper,
                'mean': stats['mean'],
                'range': (stats['min'], stats['max']),
                'confidence': 0.6,
            }
        return None
    
    def predict_from_segmentation(self, coverage_percent: float, image_area: int = None) -> Optional[Dict]:
        """
        Estimate relative size from segmentation coverage.
        
        Rough heuristic:
        - < 20% coverage → likely smaller animal or distant shot
        - 20-40% coverage → medium animal, typical framing
        - > 40% coverage → larger animal or close shot
        """
        if coverage_percent <= 0:
            return None
            
        # Map coverage to size category (rough heuristic)
        if coverage_percent < 20:
            size_est = 'MINIMUM'
        elif coverage_percent < 30:
            size_est = 'MEDIUM'
        elif coverage_percent < 40:
            size_est = 'LARGE'
        else:
            size_est = 'EXTRA_LARGE'
        
        if size_est in self.size_stats:
            stats = self.size_stats[size_est]
            return {
                'source': 'segmentation_heuristic',
                'coverage_percent': coverage_percent,
                'estimated_size': size_est,
                'mean': stats['mean'],
                'range': (stats['min'], stats['max']),
                'confidence': 0.3,  # Low confidence - just a heuristic
            }
        return None
    
    def predict(
        self,
        breed: str = None,
        height_inches: float = None,
        size_category: str = None,
        segmentation_coverage: float = None,
    ) -> Dict:
        """
        Combine all available signals to predict weight.
        
        Returns combined prediction with confidence-weighted average.
        """
        predictions = []
        
        if breed:
            pred = self.predict_from_breed(breed)
            if pred:
                predictions.append(pred)
        
        if height_inches and height_inches > 0:
            pred = self.predict_from_height(height_inches)
            if pred:
                predictions.append(pred)
        
        if size_category:
            pred = self.predict_from_size_category(size_category)
            if pred:
                predictions.append(pred)
        
        if segmentation_coverage and segmentation_coverage > 0:
            pred = self.predict_from_segmentation(segmentation_coverage)
            if pred:
                predictions.append(pred)
        
        if not predictions:
            # Fallback to overall dataset stats
            return {
                'source': 'dataset_overall',
                'mean': self.overall_stats['mean'],
                'range': (self.overall_stats['min'], self.overall_stats['max']),
                'confidence': 0.2,
                'note': 'No specific features matched, using overall dataset statistics',
            }
        
        # Confidence-weighted average
        total_conf = sum(p['confidence'] for p in predictions)
        weighted_mean = sum(p['mean'] * p['confidence'] for p in predictions) / total_conf
        
        # Range: take widest reasonable range
        all_mins = [p['range'][0] for p in predictions]
        all_maxs = [p['range'][1] for p in predictions]
        
        return {
            'source': 'combined',
            'sources_used': [p['source'] for p in predictions],
            'mean': round(weighted_mean, 1),
            'range': (round(min(all_mins), 1), round(max(all_maxs), 1)),
            'confidence': min(0.95, total_conf / len(predictions)),
            'individual_predictions': predictions,
        }
    
    def get_breed_context_for_llm(self) -> str:
        """Generate context string for LLM prompt with learned weight ranges"""
        lines = [
            "WEIGHT REFERENCE DATA (from actual cattle measurements in this dataset):",
        ]
        
        # Sort by count (most common first)
        sorted_breeds = sorted(
            self.breed_stats.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        for breed, stats in sorted_breeds:
            lines.append(
                f"- {breed}: {stats['q25']:.0f}-{stats['q75']:.0f} kg typical "
                f"(range: {stats['min']:.0f}-{stats['max']:.0f} kg, n={stats['count']})"
            )
        
        lines.append("")
        lines.append("SIZE CATEGORIES:")
        for size in ['MINIMUM', 'MEDIUM', 'LARGE', 'EXTRA_LARGE']:
            if size in self.size_stats:
                stats = self.size_stats[size]
                lines.append(f"- {size}: {stats['min']:.0f}-{stats['max']:.0f} kg (mean: {stats['mean']:.0f})")
        
        return "\n".join(lines)


# Singleton instance
_predictor = None

def get_weight_predictor(metadata_csv: str = "dataset.csv") -> WeightPredictor:
    """Get or create weight predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = WeightPredictor(metadata_csv)
        _predictor.load_and_learn()
    return _predictor


if __name__ == "__main__":
    # Test the predictor
    predictor = WeightPredictor("dataset.csv")
    predictor.load_and_learn()
    
    print("\n=== TEST PREDICTIONS ===")
    
    # Test breed-based
    print("\nBreed: SAHIWAL")
    print(predictor.predict(breed="SAHIWAL"))
    
    print("\nBreed: Holstein (mapped)")
    print(predictor.predict(breed="Holstein"))
    
    print("\nBreed: Gir (mapped to LOCAL)")
    print(predictor.predict(breed="Gir"))
    
    # Test height-based
    print("\nHeight: 48 inches")
    print(predictor.predict(height_inches=48))
    
    # Test combined
    print("\nCombined: breed=LOCAL, height=45, coverage=35%")
    print(predictor.predict(breed="LOCAL", height_inches=45, segmentation_coverage=35))
    
    # Print LLM context
    print("\n=== LLM CONTEXT ===")
    print(predictor.get_breed_context_for_llm())
