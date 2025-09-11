"""Error mining for active learning - harvest FP/FN from evaluation results."""

import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
from dataclasses import dataclass

from ..eval.coco_eval import compute_iou


@dataclass
class ErrorCase:
    """Represents an error case (FP or FN)."""
    type: str  # "FP" or "FN"
    page: int
    class_gt: str
    class_pred: str
    score: float
    iou: float
    bbox_gt: Tuple[int, int, int, int]
    bbox_pred: Tuple[int, int, int, int]
    image_path: str
    crop_path: str


def load_coco_annotations(coco_path: str) -> Dict[str, Any]:
    """Load COCO format annotations."""
    with open(coco_path, 'r') as f:
        return json.load(f)


def match_predictions_to_gt(
    gt_annotations: Dict[str, Any],
    pred_annotations: Dict[str, Any],
    iou_threshold: float = 0.5
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Match predictions to ground truth annotations.
    
    Returns:
        matched_pairs: List of (gt, pred) pairs that match
        false_negatives: List of unmatched GT annotations
        false_positives: List of unmatched predictions
    """
    # Create image ID to filename mapping
    image_id_to_filename = {img['id']: img['file_name'] for img in gt_annotations['images']}
    
    # Group annotations by image
    gt_by_image = {}
    for ann in gt_annotations['annotations']:
        img_id = ann['image_id']
        if img_id not in gt_by_image:
            gt_by_image[img_id] = []
        gt_by_image[img_id].append(ann)
    
    pred_by_image = {}
    for ann in pred_annotations['annotations']:
        img_id = ann['image_id']
        if img_id not in pred_by_image:
            pred_by_image[img_id] = []
        pred_by_image[img_id].append(ann)
    
    matched_pairs = []
    false_negatives = []
    false_positives = []
    
    # Process each image
    for img_id in gt_by_image:
        gt_anns = gt_by_image[img_id]
        pred_anns = pred_by_image.get(img_id, [])
        
        # Track which annotations have been matched
        gt_matched = [False] * len(gt_anns)
        pred_matched = [False] * len(pred_anns)
        
        # Find matches
        for i, gt_ann in enumerate(gt_anns):
            best_iou = 0
            best_pred_idx = -1
            
            for j, pred_ann in enumerate(pred_anns):
                if pred_matched[j]:
                    continue
                
                # Check class match
                if gt_ann['category_id'] != pred_ann['category_id']:
                    continue
                
                # Compute IoU
                iou = compute_iou(gt_ann['bbox'], pred_ann['bbox'])
                
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_pred_idx = j
            
            if best_pred_idx >= 0:
                # Found a match
                matched_pairs.append((gt_ann, pred_anns[best_pred_idx]))
                gt_matched[i] = True
                pred_matched[best_pred_idx] = True
            else:
                # False negative
                false_negatives.append(gt_ann)
        
        # Remaining predictions are false positives
        for j, pred_ann in enumerate(pred_anns):
            if not pred_matched[j]:
                false_positives.append(pred_ann)
    
    return matched_pairs, false_negatives, false_positives


def create_error_crops(
    error_cases: List[ErrorCase],
    images_dir: str,
    output_dir: str
) -> None:
    """Create cropped images for error cases."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for case in error_cases:
        # Load image
        image_path = Path(images_dir) / case.image_path
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load image: {image_path}")
            continue
        
        # Use GT bbox for FN, pred bbox for FP
        if case.type == "FN":
            bbox = case.bbox_gt
        else:
            bbox = case.bbox_pred
        
        # Extract crop
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Clamp to image bounds
        h_img, w_img = image.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        if w <= 0 or h <= 0:
            print(f"Warning: Invalid bbox for {case.image_path}: {bbox}")
            continue
        
        crop = image[y:y+h, x:x+w]
        
        # Save crop
        crop_filename = f"{case.type}_{case.class_gt}_p{case.page:02d}_{case.score:.3f}.jpg"
        crop_path = output_path / crop_filename
        cv2.imwrite(str(crop_path), crop)
        
        # Update case with actual crop path
        case.crop_path = str(crop_path)


def mine_errors(
    gt_coco: str,
    pred_coco: str,
    images_dir: str,
    output_dir: str,
    iou_threshold: float = 0.5
) -> List[ErrorCase]:
    """Mine false positives and false negatives from evaluation results.
    
    Args:
        gt_coco: Path to ground truth COCO JSON
        pred_coco: Path to predictions COCO JSON
        images_dir: Directory containing images
        output_dir: Directory to save error crops and audit
        iou_threshold: IoU threshold for matching
        
    Returns:
        List of ErrorCase objects
    """
    print(f"[INFO] Mining errors from {gt_coco} and {pred_coco}")
    
    # Load annotations
    gt_annotations = load_coco_annotations(gt_coco)
    pred_annotations = load_coco_annotations(pred_coco)
    
    # Match predictions to ground truth
    matched_pairs, false_negatives, false_positives = match_predictions_to_gt(
        gt_annotations, pred_annotations, iou_threshold
    )
    
    print(f"[INFO] Found {len(matched_pairs)} matches, {len(false_negatives)} FNs, {len(false_positives)} FPs")
    
    # Create error cases
    error_cases = []
    
    # Create image ID to filename mapping
    image_id_to_filename = {img['id']: img['file_name'] for img in gt_annotations['images']}
    category_id_to_name = {cat['id']: cat['name'] for cat in gt_annotations['categories']}
    
    # Process false negatives
    for fn_ann in false_negatives:
        img_id = fn_ann['image_id']
        page = img_id  # Assuming image ID corresponds to page number
        
        case = ErrorCase(
            type="FN",
            page=page,
            class_gt=category_id_to_name[fn_ann['category_id']],
            class_pred="N/A",
            score=0.0,
            iou=0.0,
            bbox_gt=tuple(fn_ann['bbox']),
            bbox_pred=(0, 0, 0, 0),
            image_path=image_id_to_filename[img_id],
            crop_path=""
        )
        error_cases.append(case)
    
    # Process false positives
    for fp_ann in false_positives:
        img_id = fp_ann['image_id']
        page = img_id  # Assuming image ID corresponds to page number
        
        case = ErrorCase(
            type="FP",
            page=page,
            class_gt="N/A",
            class_pred=category_id_to_name[fp_ann['category_id']],
            score=fp_ann.get('score', 0.0),
            iou=0.0,
            bbox_gt=(0, 0, 0, 0),
            bbox_pred=tuple(fp_ann['bbox']),
            image_path=image_id_to_filename[img_id],
            crop_path=""
        )
        error_cases.append(case)
    
    # Create error crops
    create_error_crops(error_cases, images_dir, output_dir)
    
    # Save audit index
    audit_path = Path(output_dir) / "audit_index.csv"
    with open(audit_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'type', 'page', 'class_gt', 'class_pred', 'score', 
            'iou', 'image_path', 'crop_path'
        ])
        
        for case in error_cases:
            writer.writerow([
                case.type, case.page, case.class_gt, case.class_pred,
                case.score, case.iou, case.image_path, case.crop_path
            ])
    
    print(f"[INFO] Saved {len(error_cases)} error cases to {output_dir}")
    print(f"[INFO] Audit index saved to {audit_path}")
    
    return error_cases


def main():
    """Command line interface for error mining."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mine errors from evaluation results")
    parser.add_argument("--gt", required=True, help="Ground truth COCO JSON file")
    parser.add_argument("--pred", required=True, help="Predictions COCO JSON file")
    parser.add_argument("--images", required=True, help="Images directory")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching")
    
    args = parser.parse_args()
    
    error_cases = mine_errors(
        args.gt, args.pred, args.images, args.out, args.iou_threshold
    )
    
    print(f"[INFO] Error mining complete. Found {len(error_cases)} error cases.")


if __name__ == "__main__":
    main()
