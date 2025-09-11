"""Visualization utilities for bounding box overlays and error analysis."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np
from collections import defaultdict


def draw_boxes(image: np.ndarray, boxes: List[Dict[str, Any]], 
               class_colors: Optional[Dict[str, Tuple[int, int, int]]] = None) -> np.ndarray:
    """Draw bounding boxes on image with class colors.
    
    Args:
        image: Input image
        boxes: List of box dictionaries with 'bbox', 'class', 'score' keys
        class_colors: Optional color mapping for classes
        
    Returns:
        Image with drawn boxes
    """
    if class_colors is None:
        class_colors = {
            'Text': (0, 255, 0),      # Green
            'Title': (255, 0, 0),     # Blue
            'List': (0, 0, 255),      # Red
            'Table': (255, 255, 0),   # Cyan
            'Figure': (255, 0, 255)   # Magenta
        }
    
    # Create overlay
    overlay = image.copy()
    
    for box in boxes:
        bbox = box['bbox']
        class_name = box.get('class', 'Text')
        score = box.get('score', 1.0)
        
        # Get color
        color = class_colors.get(class_name, (128, 128, 128))
        
        # Draw rectangle
        x, y, w, h = bbox
        cv2.rectangle(overlay, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        
        # Draw label
        label = f"{class_name}: {score:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        # Draw label background
        cv2.rectangle(overlay, (int(x), int(y) - label_size[1] - 10), 
                     (int(x + label_size[0]), int(y)), color, -1)
        
        # Draw label text
        cv2.putText(overlay, label, (int(x), int(y) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return overlay


def save_page_overlay(image_path: str, predictions: List[Dict[str, Any]], 
                     out_path: str, class_colors: Optional[Dict[str, Tuple[int, int, int]]] = None) -> None:
    """Save page overlay with predictions.
    
    Args:
        image_path: Path to input image
        predictions: List of predictions for this page
        out_path: Output path for overlay image
        class_colors: Optional color mapping for classes
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARN] Could not load image: {image_path}")
        return
    
    # Draw boxes
    overlay = draw_boxes(image, predictions, class_colors)
    
    # Save overlay
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(out_path), overlay)
    print(f"[INFO] Overlay saved to {out_path}")


def create_error_overlays(gt_json: str, pred_json: str, images_dir: str, 
                         output_dir: str, top_k: int = 150) -> None:
    """Create error overlays for false positives and false negatives.
    
    Args:
        gt_json: Path to ground truth COCO JSON
        pred_json: Path to predictions COCO JSON
        images_dir: Directory containing images
        output_dir: Output directory for overlays
        top_k: Number of top errors to visualize
    """
    # Load data
    with open(gt_json, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    with open(pred_json, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    
    # Create output directories
    output_path = Path(output_dir)
    (output_path / "false_positives").mkdir(parents=True, exist_ok=True)
    (output_path / "false_negatives").mkdir(parents=True, exist_ok=True)
    
    # Create image ID to info mapping
    images = {img['id']: img for img in gt_data['images']}
    
    # Group annotations by image
    gt_by_image = defaultdict(list)
    pred_by_image = defaultdict(list)
    
    for ann in gt_data['annotations']:
        gt_by_image[ann['image_id']].append(ann)
    
    for ann in pred_data['annotations']:
        pred_by_image[ann['image_id']].append(ann)
    
    # Find errors
    false_positives = []
    false_negatives = []
    
    for img_id, img_info in images.items():
        img_gt = gt_by_image.get(img_id, [])
        img_pred = pred_by_image.get(img_id, [])
        
        # Find false positives (predicted but not in GT)
        for pred_ann in img_pred:
            is_fp = True
            for gt_ann in img_gt:
                if compute_bbox_iou(pred_ann['bbox'], gt_ann['bbox']) > 0.5:
                    is_fp = False
                    break
            
            if is_fp:
                false_positives.append({
                    'image_id': img_id,
                    'image_name': img_info['file_name'],
                    'annotation': pred_ann
                })
        
        # Find false negatives (in GT but not predicted)
        for gt_ann in img_gt:
            is_fn = True
            for pred_ann in img_pred:
                if compute_bbox_iou(gt_ann['bbox'], pred_ann['bbox']) > 0.5:
                    is_fn = False
                    break
            
            if is_fn:
                false_negatives.append({
                    'image_id': img_id,
                    'image_name': img_info['file_name'],
                    'annotation': gt_ann
                })
    
    # Sort by confidence/score
    false_positives.sort(key=lambda x: x['annotation'].get('score', 0), reverse=True)
    false_negatives.sort(key=lambda x: x['annotation'].get('score', 0), reverse=True)
    
    # Create overlays for top errors
    create_fp_overlays(false_positives[:top_k], images_dir, output_path / "false_positives")
    create_fn_overlays(false_negatives[:top_k], images_dir, output_path / "false_negatives")
    
    print(f"[INFO] Created {min(len(false_positives), top_k)} false positive overlays")
    print(f"[INFO] Created {min(len(false_negatives), top_k)} false negative overlays")


def create_fp_overlays(false_positives: List[Dict[str, Any]], images_dir: str, output_dir: Path) -> None:
    """Create false positive overlays.
    
    Args:
        false_positives: List of false positive errors
        images_dir: Directory containing images
        output_dir: Output directory for overlays
    """
    for i, fp in enumerate(false_positives):
        img_path = Path(images_dir) / fp['image_name']
        if not img_path.exists():
            continue
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Draw the false positive box
        overlay = draw_boxes(image, [fp['annotation']])
        
        # Save overlay
        out_path = output_dir / f"fp_{i:03d}_{fp['image_name']}"
        cv2.imwrite(str(out_path), overlay)


def create_fn_overlays(false_negatives: List[Dict[str, Any]], images_dir: str, output_dir: Path) -> None:
    """Create false negative overlays.
    
    Args:
        false_negatives: List of false negative errors
        images_dir: Directory containing images
        output_dir: Output directory for overlays
    """
    for i, fn in enumerate(false_negatives):
        img_path = Path(images_dir) / fn['image_name']
        if not img_path.exists():
            continue
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Draw the false negative box
        overlay = draw_boxes(image, [fn['annotation']])
        
        # Save overlay
        out_path = output_dir / f"fn_{i:03d}_{fn['image_name']}"
        cv2.imwrite(str(out_path), overlay)


def compute_bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Compute IoU between two bounding boxes.
    
    Args:
        bbox1: First bounding box [x, y, w, h]
        bbox2: Second bounding box [x, y, w, h]
        
    Returns:
        IoU value
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Compute intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
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


def create_class_distribution_plot(class_counts: Dict[str, int], output_path: str) -> None:
    """Create class distribution plot.
    
    Args:
        class_counts: Dictionary of class counts
        output_path: Output path for plot
    """
    try:
        import matplotlib.pyplot as plt
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(classes, counts)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Class distribution plot saved to {output_path}")
        
    except ImportError:
        print("[WARN] matplotlib not available, skipping plot generation")


def create_confidence_histogram(scores: List[float], output_path: str) -> None:
    """Create confidence score histogram.
    
    Args:
        scores: List of confidence scores
        output_path: Output path for histogram
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Score Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Confidence histogram saved to {output_path}")
        
    except ImportError:
        print("[WARN] matplotlib not available, skipping histogram generation")
