"""Uncertainty sampling strategies for active learning."""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class UncertaintyStrategy(Enum):
    """Uncertainty sampling strategies."""
    MARGIN = "margin"
    ENTROPY = "entropy"
    LOW_CONF = "low_conf"


@dataclass
class SampleScore:
    """Represents a sample with uncertainty score."""
    image_id: str
    page: int
    score: float
    strategy: str
    metadata: Dict[str, Any]


def load_predictions(pred_path: str) -> Dict[str, Any]:
    """Load predictions from COCO format JSON."""
    with open(pred_path, 'r') as f:
        return json.load(f)


def compute_margin_score(class_probs: List[float]) -> float:
    """Compute margin score (difference between top-1 and top-2 probabilities).
    
    Lower margin = more uncertain.
    """
    if len(class_probs) < 2:
        return 0.0
    
    sorted_probs = sorted(class_probs, reverse=True)
    return sorted_probs[0] - sorted_probs[1]


def compute_entropy_score(class_probs: List[float]) -> float:
    """Compute entropy score over class probabilities.
    
    Higher entropy = more uncertain.
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    probs = np.array(class_probs) + eps
    probs = probs / np.sum(probs)  # Normalize
    
    entropy = -np.sum(probs * np.log(probs))
    return entropy


def compute_low_conf_score(confidence: float, threshold: float = 0.5) -> float:
    """Compute low confidence score.
    
    Higher score for lower confidence.
    """
    return max(0, threshold - confidence)


def score_samples(
    predictions: Dict[str, Any],
    strategy: UncertaintyStrategy,
    **kwargs
) -> List[SampleScore]:
    """Score samples based on uncertainty strategy.
    
    Args:
        predictions: COCO format predictions
        strategy: Uncertainty sampling strategy
        **kwargs: Additional parameters for specific strategies
        
    Returns:
        List of SampleScore objects sorted by uncertainty (highest first)
    """
    samples = []
    
    # Group predictions by image
    pred_by_image = {}
    for ann in predictions['annotations']:
        img_id = ann['image_id']
        if img_id not in pred_by_image:
            pred_by_image[img_id] = []
        pred_by_image[img_id].append(ann)
    
    # Create image ID to filename mapping
    image_id_to_filename = {img['id']: img['file_name'] for img in predictions['images']}
    
    for img_id, img_preds in pred_by_image.items():
        if not img_preds:
            continue
        
        # Get image info
        filename = image_id_to_filename[img_id]
        page = img_id  # Assuming image ID corresponds to page number
        
        # Compute uncertainty score for this image
        if strategy == UncertaintyStrategy.MARGIN:
            # Use the prediction with lowest margin
            min_margin = float('inf')
            best_pred = None
            
            for pred in img_preds:
                # Simulate class probabilities from confidence score
                # In practice, you'd have actual class probabilities
                conf = pred.get('score', 0.0)
                class_probs = [conf] + [(1 - conf) / (len(pred_by_image) - 1)] * (len(pred_by_image) - 1)
                margin = compute_margin_score(class_probs)
                
                if margin < min_margin:
                    min_margin = margin
                    best_pred = pred
            
            score = min_margin
            metadata = {
                'num_predictions': len(img_preds),
                'best_prediction': best_pred
            }
            
        elif strategy == UncertaintyStrategy.ENTROPY:
            # Use the prediction with highest entropy
            max_entropy = 0.0
            best_pred = None
            
            for pred in img_preds:
                conf = pred.get('score', 0.0)
                class_probs = [conf] + [(1 - conf) / (len(pred_by_image) - 1)] * (len(pred_by_image) - 1)
                entropy = compute_entropy_score(class_probs)
                
                if entropy > max_entropy:
                    max_entropy = entropy
                    best_pred = pred
            
            score = max_entropy
            metadata = {
                'num_predictions': len(img_preds),
                'best_prediction': best_pred
            }
            
        elif strategy == UncertaintyStrategy.LOW_CONF:
            # Use the prediction with lowest confidence
            min_conf = float('inf')
            best_pred = None
            
            for pred in img_preds:
                conf = pred.get('score', 0.0)
                if conf < min_conf:
                    min_conf = conf
                    best_pred = pred
            
            threshold = kwargs.get('threshold', 0.5)
            score = compute_low_conf_score(min_conf, threshold)
            metadata = {
                'num_predictions': len(img_preds),
                'min_confidence': min_conf,
                'best_prediction': best_pred
            }
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        sample = SampleScore(
            image_id=str(img_id),
            page=page,
            score=score,
            strategy=strategy.value,
            metadata=metadata
        )
        samples.append(sample)
    
    # Sort by uncertainty (highest first)
    samples.sort(key=lambda x: x.score, reverse=True)
    
    return samples


def select_samples(
    predictions_path: str,
    strategy: str,
    top_k: int = 500,
    output_path: Optional[str] = None,
    **kwargs
) -> List[SampleScore]:
    """Select most uncertain samples for annotation.
    
    Args:
        predictions_path: Path to predictions COCO JSON
        strategy: Uncertainty strategy ("margin", "entropy", "low_conf")
        top_k: Number of top samples to select
        output_path: Optional path to save selection results
        **kwargs: Additional parameters for specific strategies
        
    Returns:
        List of selected SampleScore objects
    """
    print(f"[INFO] Selecting samples using {strategy} strategy")
    
    # Load predictions
    predictions = load_predictions(predictions_path)
    
    # Convert strategy string to enum
    try:
        strategy_enum = UncertaintyStrategy(strategy)
    except ValueError:
        raise ValueError(f"Unknown strategy: {strategy}. Must be one of: {[s.value for s in UncertaintyStrategy]}")
    
    # Score samples
    samples = score_samples(predictions, strategy_enum, **kwargs)
    
    # Select top-k samples
    selected = samples[:top_k]
    
    print(f"[INFO] Selected {len(selected)} samples out of {len(samples)} total")
    
    # Save results if output path provided
    if output_path:
        output_data = {
            'strategy': strategy,
            'top_k': top_k,
            'total_samples': len(samples),
            'selected_samples': len(selected),
            'samples': [
                {
                    'image_id': s.image_id,
                    'page': s.page,
                    'score': s.score,
                    'strategy': s.strategy,
                    'metadata': s.metadata
                }
                for s in selected
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"[INFO] Saved selection results to {output_path}")
    
    return selected


def main():
    """Command line interface for uncertainty sampling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Select samples for active learning")
    parser.add_argument("--pred", required=True, help="Predictions COCO JSON file")
    parser.add_argument("--strategy", choices=["margin", "entropy", "low_conf"], 
                       default="margin", help="Uncertainty sampling strategy")
    parser.add_argument("--top", type=int, default=500, help="Number of top samples to select")
    parser.add_argument("--out", help="Output JSON file for selection results")
    parser.add_argument("--threshold", type=float, default=0.5, 
                       help="Confidence threshold for low_conf strategy")
    
    args = parser.parse_args()
    
    selected = select_samples(
        args.pred, args.strategy, args.top, args.out, 
        threshold=args.threshold
    )
    
    print(f"[INFO] Selected {len(selected)} samples for annotation")


if __name__ == "__main__":
    main()
