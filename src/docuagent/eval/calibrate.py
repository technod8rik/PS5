"""Detector calibration for better confidence-precision alignment."""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class CalibrationParams:
    """Calibration parameters."""
    method: str
    temperature: Optional[float] = None
    bias: Optional[float] = None
    scale: Optional[float] = None
    isotonic_regressor: Optional[object] = None
    logistic_regressor: Optional[object] = None


def load_predictions(pred_path: str) -> Dict[str, Any]:
    """Load predictions from COCO format JSON."""
    with open(pred_path, 'r') as f:
        return json.load(f)


def load_ground_truth(gt_path: str) -> Dict[str, Any]:
    """Load ground truth from COCO format JSON."""
    with open(gt_path, 'r') as f:
        return json.load(f)


def match_predictions_to_gt(
    predictions: Dict[str, Any],
    ground_truth: Dict[str, Any],
    iou_threshold: float = 0.5
) -> Tuple[List[float], List[bool]]:
    """Match predictions to ground truth and return confidence scores and correctness.
    
    Args:
        predictions: COCO format predictions
        ground_truth: COCO format ground truth
        iou_threshold: IoU threshold for matching
        
    Returns:
        confidences: List of confidence scores
        is_correct: List of boolean correctness labels
    """
    # Create image ID to filename mapping
    gt_image_id_to_filename = {img['id']: img['file_name'] for img in ground_truth['images']}
    pred_image_id_to_filename = {img['id']: img['file_name'] for img in predictions['images']}
    
    # Create filename to GT annotations mapping
    gt_by_filename = {}
    for ann in ground_truth['annotations']:
        img_id = ann['image_id']
        filename = gt_image_id_to_filename[img_id]
        if filename not in gt_by_filename:
            gt_by_filename[filename] = []
        gt_by_filename[filename].append(ann)
    
    # Create filename to predictions mapping
    pred_by_filename = {}
    for ann in predictions['annotations']:
        img_id = ann['image_id']
        filename = pred_image_id_to_filename[img_id]
        if filename not in pred_by_filename:
            pred_by_filename[filename] = []
        pred_by_filename[filename].append(ann)
    
    confidences = []
    is_correct = []
    
    # Process each image
    for filename in pred_by_filename:
        if filename not in gt_by_filename:
            continue
        
        gt_anns = gt_by_filename[filename]
        pred_anns = pred_by_filename[filename]
        
        # Track which GT annotations have been matched
        gt_matched = [False] * len(gt_anns)
        
        # Find matches for each prediction
        for pred_ann in pred_anns:
            confidence = pred_ann.get('score', 0.0)
            confidences.append(confidence)
            
            # Find best matching GT annotation
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt_ann in enumerate(gt_anns):
                if gt_matched[i]:
                    continue
                
                # Check class match
                if pred_ann['category_id'] != gt_ann['category_id']:
                    continue
                
                # Compute IoU
                iou = compute_iou(pred_ann['bbox'], gt_ann['bbox'])
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            # Determine if prediction is correct
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                is_correct.append(True)
                gt_matched[best_gt_idx] = True
            else:
                is_correct.append(False)
    
    return confidences, is_correct


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Compute IoU between two bounding boxes in COCO format [x, y, w, h]."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Convert to [x1, y1, x2, y2] format
    box1 = [x1, y1, x1 + w1, y1 + h1]
    box2 = [x2, y2, x2 + w2, y2 + h2]
    
    # Compute intersection
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Compute union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def calibrate_temperature_scaling(
    confidences: List[float],
    is_correct: List[bool]
) -> CalibrationParams:
    """Calibrate using temperature scaling.
    
    Args:
        confidences: List of confidence scores
        is_correct: List of correctness labels
        
    Returns:
        CalibrationParams with temperature parameter
    """
    # Convert to numpy arrays
    confs = np.array(confidences)
    correct = np.array(is_correct, dtype=float)
    
    # Temperature scaling: p_calibrated = sigmoid(logit(p) / T)
    # We need to find T that minimizes negative log likelihood
    
    def temperature_scaling_loss(temp):
        if temp <= 0:
            return float('inf')
        
        # Apply temperature scaling
        logits = np.log(confs / (1 - confs + 1e-8))  # Convert to logits
        scaled_logits = logits / temp
        probs = 1 / (1 + np.exp(-scaled_logits))  # Sigmoid
        
        # Compute negative log likelihood
        probs = np.clip(probs, 1e-8, 1 - 1e-8)  # Avoid log(0)
        nll = -np.mean(correct * np.log(probs) + (1 - correct) * np.log(1 - probs))
        
        return nll
    
    # Find optimal temperature using grid search
    temps = np.logspace(-2, 2, 100)
    losses = [temperature_scaling_loss(t) for t in temps]
    best_temp = temps[np.argmin(losses)]
    
    return CalibrationParams(
        method="temperature",
        temperature=best_temp
    )


def calibrate_platt_scaling(
    confidences: List[float],
    is_correct: List[bool]
) -> CalibrationParams:
    """Calibrate using Platt scaling (logistic regression).
    
    Args:
        confidences: List of confidence scores
        is_correct: List of correctness labels
        
    Returns:
        CalibrationParams with logistic regression parameters
    """
    # Convert to numpy arrays
    confs = np.array(confidences).reshape(-1, 1)
    correct = np.array(is_correct, dtype=int)
    
    # Fit logistic regression
    regressor = LogisticRegression()
    regressor.fit(confs, correct)
    
    return CalibrationParams(
        method="platt",
        bias=regressor.intercept_[0],
        scale=regressor.coef_[0][0],
        logistic_regressor=regressor
    )


def calibrate_isotonic_regression(
    confidences: List[float],
    is_correct: List[bool]
) -> CalibrationParams:
    """Calibrate using isotonic regression.
    
    Args:
        confidences: List of confidence scores
        is_correct: List of correctness labels
        
    Returns:
        CalibrationParams with isotonic regressor
    """
    # Convert to numpy arrays
    confs = np.array(confidences)
    correct = np.array(is_correct, dtype=float)
    
    # Fit isotonic regression
    regressor = IsotonicRegression(out_of_bounds='clip')
    regressor.fit(confs, correct)
    
    return CalibrationParams(
        method="isotonic",
        isotonic_regressor=regressor
    )


def calibrate_detector(
    val_preds_path: str,
    gt_path: str,
    method: str = "temperature",
    iou_threshold: float = 0.5
) -> CalibrationParams:
    """Calibrate detector confidence scores.
    
    Args:
        val_preds_path: Path to validation predictions COCO JSON
        gt_path: Path to ground truth COCO JSON
        method: Calibration method ("temperature", "platt", "isotonic")
        iou_threshold: IoU threshold for matching
        
    Returns:
        CalibrationParams object
    """
    print(f"[INFO] Calibrating detector using {method} scaling")
    
    # Load data
    predictions = load_predictions(val_preds_path)
    ground_truth = load_ground_truth(gt_path)
    
    # Match predictions to ground truth
    confidences, is_correct = match_predictions_to_gt(
        predictions, ground_truth, iou_threshold
    )
    
    print(f"[INFO] Matched {len(confidences)} predictions")
    print(f"[INFO] Correct predictions: {sum(is_correct)} / {len(is_correct)} ({sum(is_correct)/len(is_correct)*100:.1f}%)")
    
    # Calibrate based on method
    if method == "temperature":
        params = calibrate_temperature_scaling(confidences, is_correct)
    elif method == "platt":
        params = calibrate_platt_scaling(confidences, is_correct)
    elif method == "isotonic":
        params = calibrate_isotonic_regression(confidences, is_correct)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    print(f"[INFO] Calibration completed")
    if params.temperature:
        print(f"[INFO] Temperature: {params.temperature:.3f}")
    if params.bias is not None and params.scale is not None:
        print(f"[INFO] Platt scaling: bias={params.bias:.3f}, scale={params.scale:.3f}")
    
    return params


def apply_calibration(
    confidence: float,
    params: CalibrationParams
) -> float:
    """Apply calibration to a confidence score.
    
    Args:
        confidence: Original confidence score
        params: Calibration parameters
        
    Returns:
        Calibrated confidence score
    """
    if params.method == "temperature":
        if params.temperature is None:
            return confidence
        
        # Apply temperature scaling
        logit = np.log(confidence / (1 - confidence + 1e-8))
        scaled_logit = logit / params.temperature
        calibrated = 1 / (1 + np.exp(-scaled_logit))
        
        return float(np.clip(calibrated, 0, 1))
    
    elif params.method == "platt":
        if params.logistic_regressor is None:
            return confidence
        
        # Apply Platt scaling
        calibrated = params.logistic_regressor.predict_proba([[confidence]])[0][1]
        return float(calibrated)
    
    elif params.method == "isotonic":
        if params.isotonic_regressor is None:
            return confidence
        
        # Apply isotonic regression
        calibrated = params.isotonic_regressor.predict([confidence])[0]
        return float(np.clip(calibrated, 0, 1))
    
    else:
        return confidence


def save_calibration_params(params: CalibrationParams, output_path: str) -> None:
    """Save calibration parameters to JSON file."""
    # Convert to serializable format
    params_dict = {
        'method': params.method,
        'temperature': params.temperature,
        'bias': params.bias,
        'scale': params.scale
    }
    
    # Note: We don't save the sklearn regressors as they're not JSON serializable
    # In practice, you'd use pickle or joblib for that
    
    with open(output_path, 'w') as f:
        json.dump(params_dict, f, indent=2)
    
    print(f"[INFO] Calibration parameters saved to {output_path}")


def load_calibration_params(params_path: str) -> CalibrationParams:
    """Load calibration parameters from JSON file."""
    with open(params_path, 'r') as f:
        params_dict = json.load(f)
    
    return CalibrationParams(
        method=params_dict['method'],
        temperature=params_dict.get('temperature'),
        bias=params_dict.get('bias'),
        scale=params_dict.get('scale')
    )


def compute_ece(
    confidences: List[float],
    is_correct: List[bool],
    n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE).
    
    Args:
        confidences: List of confidence scores
        is_correct: List of correctness labels
        n_bins: Number of bins for ECE computation
        
    Returns:
        ECE value
    """
    # Convert to numpy arrays
    confs = np.array(confidences)
    correct = np.array(is_correct, dtype=float)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (confs > bin_lower) & (confs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Compute accuracy and confidence in this bin
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = confs[in_bin].mean()
            
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


def main():
    """Command line interface for calibration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calibrate detector confidence scores")
    parser.add_argument("--val-preds", required=True, help="Validation predictions COCO JSON")
    parser.add_argument("--gt", required=True, help="Ground truth COCO JSON")
    parser.add_argument("--out", required=True, help="Output calibration parameters file")
    parser.add_argument("--method", choices=["temperature", "platt", "isotonic"], 
                       default="temperature", help="Calibration method")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching")
    
    args = parser.parse_args()
    
    # Calibrate detector
    params = calibrate_detector(
        args.val_preds, args.gt, args.method, args.iou_threshold
    )
    
    # Save parameters
    save_calibration_params(params, args.out)
    
    print(f"[INFO] Calibration complete. Parameters saved to {args.out}")


if __name__ == "__main__":
    main()
