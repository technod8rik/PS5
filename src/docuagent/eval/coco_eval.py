"""COCO-style evaluation metrics for layout detection."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    PYOCOCO_AVAILABLE = True
except ImportError:
    PYOCOCO_AVAILABLE = False


def load_coco_data(coco_json: str) -> Dict[str, Any]:
    """Load COCO data from JSON file.
    
    Args:
        coco_json: Path to COCO JSON file
        
    Returns:
        COCO data dictionary
    """
    with open(coco_json, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_predictions_to_coco(predictions: List[Dict[str, Any]], 
                               images: List[Dict[str, Any]], 
                               categories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert predictions to COCO format.
    
    Args:
        predictions: List of predictions
        images: List of image information
        categories: List of category information
        
    Returns:
        COCO format annotations
    """
    # Create image name to ID mapping
    img_name_to_id = {img['file_name']: img['id'] for img in images}
    
    coco_annotations = []
    ann_id = 0
    
    for pred in predictions:
        # Get image ID
        img_name = pred.get('image_name', '')
        if img_name not in img_name_to_id:
            continue
        
        img_id = img_name_to_id[img_name]
        
        # Convert bbox
        bbox = pred['bbox']
        x, y, w, h = bbox
        
        # Create COCO annotation
        ann = {
            'id': ann_id,
            'image_id': img_id,
            'category_id': pred['category_id'],
            'bbox': [x, y, w, h],
            'area': w * h,
            'iscrowd': 0,
            'score': pred.get('score', 1.0)
        }
        coco_annotations.append(ann)
        ann_id += 1
    
    return coco_annotations


def compute_coco_metrics(gt_json: str, pred_json: str, output_dir: str) -> Dict[str, Any]:
    """Compute COCO-style metrics.
    
    Args:
        gt_json: Path to ground truth COCO JSON
        pred_json: Path to predictions COCO JSON
        output_dir: Output directory for results
        
    Returns:
        Evaluation metrics
    """
    if not PYOCOCO_AVAILABLE:
        raise ImportError("pycocotools not available. Install with: pip install pycocotools")
    
    # Load ground truth
    coco_gt = COCO(gt_json)
    
    # Load predictions
    with open(pred_json, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    
    # Convert predictions to COCO format if needed
    if isinstance(pred_data, list):
        # Predictions are in list format, convert to COCO
        gt_data = load_coco_data(gt_json)
        coco_annotations = convert_predictions_to_coco(
            pred_data, gt_data['images'], gt_data['categories']
        )
        
        # Save converted predictions
        pred_coco_file = Path(output_dir) / "predictions_coco.json"
        pred_coco_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(pred_coco_file, 'w', encoding='utf-8') as f:
            json.dump(coco_annotations, f, indent=2, ensure_ascii=False)
        
        pred_json = str(pred_coco_file)
    
    # Load predictions
    coco_dt = coco_gt.loadRes(pred_json)
    
    # Create COCO evaluator
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract metrics
    metrics = {
        'mAP_0.5': float(coco_eval.stats[1]),  # mAP@0.5
        'mAP_0.5_0.95': float(coco_eval.stats[0]),  # mAP@0.5:0.95
        'mAP_small': float(coco_eval.stats[3]),  # mAP for small objects
        'mAP_medium': float(coco_eval.stats[4]),  # mAP for medium objects
        'mAP_large': float(coco_eval.stats[5]),  # mAP for large objects
        'AR_1': float(coco_eval.stats[6]),  # AR@1
        'AR_10': float(coco_eval.stats[7]),  # AR@10
        'AR_100': float(coco_eval.stats[8]),  # AR@100
        'AR_small': float(coco_eval.stats[9]),  # AR for small objects
        'AR_medium': float(coco_eval.stats[10]),  # AR for medium objects
        'AR_large': float(coco_eval.stats[11])  # AR for large objects
    }
    
    # Per-class metrics
    per_class_metrics = {}
    for cat_id in coco_gt.getCatIds():
        cat_info = coco_gt.loadCats(cat_id)[0]
        cat_name = cat_info['name']
        
        # Get per-class precision and recall
        coco_eval.params.catIds = [cat_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        
        per_class_metrics[cat_name] = {
            'mAP_0.5': float(coco_eval.stats[1]),
            'mAP_0.5_0.95': float(coco_eval.stats[0]),
            'precision': float(coco_eval.stats[2]),
            'recall': float(coco_eval.stats[3])
        }
    
    metrics['per_class'] = per_class_metrics
    
    # Save detailed results
    results_file = Path(output_dir) / "coco_metrics.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] COCO metrics computed and saved to {results_file}")
    print(f"[INFO] mAP@0.5: {metrics['mAP_0.5']:.3f}")
    print(f"[INFO] mAP@0.5:0.95: {metrics['mAP_0.5_0.95']:.3f}")
    
    return metrics


def compute_confusion_matrix(gt_json: str, pred_json: str, output_dir: str) -> Dict[str, Any]:
    """Compute confusion matrix for class predictions.
    
    Args:
        gt_json: Path to ground truth COCO JSON
        pred_json: Path to predictions COCO JSON
        output_dir: Output directory for results
        
    Returns:
        Confusion matrix data
    """
    # Load data
    gt_data = load_coco_data(gt_json)
    pred_data = load_coco_data(pred_json)
    
    # Create category mappings
    cat_id_to_name = {cat['id']: cat['name'] for cat in gt_data['categories']}
    cat_name_to_id = {cat['name']: cat['id'] for cat in gt_data['categories']}
    
    # Create image ID to predictions mapping
    img_predictions = {}
    for pred in pred_data['annotations']:
        img_id = pred['image_id']
        if img_id not in img_predictions:
            img_predictions[img_id] = []
        img_predictions[img_id].append(pred)
    
    # Compute IoU-based matching
    confusion_data = {
        'confusion_matrix': {},
        'class_names': list(cat_name_to_id.keys()),
        'total_gt': 0,
        'total_pred': 0,
        'matched': 0,
        'false_positives': 0,
        'false_negatives': 0
    }
    
    # Initialize confusion matrix
    class_names = list(cat_name_to_id.keys())
    for gt_class in class_names:
        confusion_data['confusion_matrix'][gt_class] = {}
        for pred_class in class_names:
            confusion_data['confusion_matrix'][gt_class][pred_class] = 0
    
    # Process each image
    for img in gt_data['images']:
        img_id = img['id']
        
        # Get ground truth annotations
        gt_anns = [ann for ann in gt_data['annotations'] if ann['image_id'] == img_id]
        
        # Get predictions
        pred_anns = img_predictions.get(img_id, [])
        
        confusion_data['total_gt'] += len(gt_anns)
        confusion_data['total_pred'] += len(pred_anns)
        
        # Match predictions to ground truth
        matched_gt = set()
        matched_pred = set()
        
        for pred_ann in pred_anns:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_ann in enumerate(gt_anns):
                if gt_idx in matched_gt:
                    continue
                
                # Compute IoU
                iou = compute_bbox_iou(pred_ann['bbox'], gt_ann['bbox'])
                
                if iou > best_iou and iou > 0.5:  # IoU threshold
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                # Match found
                gt_ann = gt_anns[best_gt_idx]
                gt_class = cat_id_to_name[gt_ann['category_id']]
                pred_class = cat_id_to_name[pred_ann['category_id']]
                
                confusion_data['confusion_matrix'][gt_class][pred_class] += 1
                confusion_data['matched'] += 1
                
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_ann)
        
        # Count false positives and false negatives
        confusion_data['false_positives'] += len(pred_anns) - len(matched_pred)
        confusion_data['false_negatives'] += len(gt_anns) - len(matched_gt)
    
    # Save confusion matrix
    confusion_file = Path(output_dir) / "confusion_matrix.json"
    with open(confusion_file, 'w', encoding='utf-8') as f:
        json.dump(confusion_data, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Confusion matrix saved to {confusion_file}")
    
    return confusion_data


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


def generate_coco_report(metrics: Dict[str, Any], output_dir: str) -> str:
    """Generate COCO evaluation report.
    
    Args:
        metrics: Evaluation metrics
        output_dir: Output directory
        
    Returns:
        Path to report file
    """
    report_file = Path(output_dir) / "coco_report.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# COCO Evaluation Report\n\n")
        
        # Overall metrics
        f.write("## Overall Metrics\n\n")
        f.write(f"- **mAP@0.5**: {metrics['mAP_0.5']:.3f}\n")
        f.write(f"- **mAP@0.5:0.95**: {metrics['mAP_0.5_0.95']:.3f}\n")
        f.write(f"- **mAP (small)**: {metrics['mAP_small']:.3f}\n")
        f.write(f"- **mAP (medium)**: {metrics['mAP_medium']:.3f}\n")
        f.write(f"- **mAP (large)**: {metrics['mAP_large']:.3f}\n")
        f.write(f"- **AR@1**: {metrics['AR_1']:.3f}\n")
        f.write(f"- **AR@10**: {metrics['AR_10']:.3f}\n")
        f.write(f"- **AR@100**: {metrics['AR_100']:.3f}\n\n")
        
        # Per-class metrics
        if 'per_class' in metrics:
            f.write("## Per-Class Metrics\n\n")
            f.write("| Class | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |\n")
            f.write("|-------|---------|--------------|-----------|--------|\n")
            
            for class_name, class_metrics in metrics['per_class'].items():
                f.write(f"| {class_name} | {class_metrics['mAP_0.5']:.3f} | "
                       f"{class_metrics['mAP_0.5_0.95']:.3f} | "
                       f"{class_metrics['precision']:.3f} | "
                       f"{class_metrics['recall']:.3f} |\n")
    
    print(f"[INFO] COCO report saved to {report_file}")
    
    return str(report_file)
