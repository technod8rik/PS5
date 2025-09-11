"""Text evaluation metrics (CER/WER) for OCR results."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np

try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False

from ..ocr_lang import OCRLang
from ..layout import LayoutBox


def compute_cer_wer(gt_text: str, pred_text: str) -> Tuple[float, float]:
    """Compute Character Error Rate (CER) and Word Error Rate (WER).
    
    Args:
        gt_text: Ground truth text
        pred_text: Predicted text
        
    Returns:
        Tuple of (CER, WER)
    """
    if not JIWER_AVAILABLE:
        raise ImportError("jiwer not available. Install with: pip install jiwer")
    
    # Compute CER
    cer = jiwer.character_error_rate(gt_text, pred_text)
    
    # Compute WER
    wer = jiwer.word_error_rate(gt_text, pred_text)
    
    return float(cer), float(wer)


def align_predictions_to_gt(gt_annotations: List[Dict[str, Any]], 
                           pred_annotations: List[Dict[str, Any]], 
                           iou_threshold: float = 0.5) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Align predictions to ground truth annotations based on IoU.
    
    Args:
        gt_annotations: Ground truth annotations
        pred_annotations: Predicted annotations
        iou_threshold: IoU threshold for matching
        
    Returns:
        List of matched (gt, pred) pairs
    """
    matches = []
    matched_pred = set()
    
    for gt_ann in gt_annotations:
        best_iou = 0
        best_pred_idx = -1
        
        for pred_idx, pred_ann in enumerate(pred_annotations):
            if pred_idx in matched_pred:
                continue
            
            # Compute IoU
            iou = compute_bbox_iou(gt_ann['bbox'], pred_ann['bbox'])
            
            if iou > best_iou and iou > iou_threshold:
                best_iou = iou
                best_pred_idx = pred_idx
        
        if best_pred_idx >= 0:
            matches.append((gt_ann, pred_annotations[best_pred_idx]))
            matched_pred.add(best_pred_idx)
    
    return matches


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


def evaluate_text_regions(gt_json: str, pred_json: str, images_dir: str, 
                         output_dir: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate text regions using CER/WER metrics.
    
    Args:
        gt_json: Path to ground truth COCO JSON
        pred_json: Path to predictions COCO JSON
        images_dir: Directory containing images
        output_dir: Output directory for results
        cfg: Configuration dictionary
        
    Returns:
        Evaluation metrics
    """
    if not JIWER_AVAILABLE:
        raise ImportError("jiwer not available. Install with: pip install jiwer")
    
    # Load data
    with open(gt_json, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    with open(pred_json, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    
    # Create image ID to info mapping
    images = {img['id']: img for img in gt_data['images']}
    
    # Filter text annotations (Text, Title, List)
    text_classes = ['Text', 'Title', 'List']
    text_class_ids = {cat['id'] for cat in gt_data['categories'] 
                     if cat['name'] in text_classes}
    
    gt_text_anns = [ann for ann in gt_data['annotations'] 
                   if ann['category_id'] in text_class_ids]
    pred_text_anns = [ann for ann in pred_data['annotations'] 
                     if ann['category_id'] in text_class_ids]
    
    # Initialize OCR
    ocr_lang = OCRLang(cfg)
    
    # Process each image
    all_cer = []
    all_wer = []
    per_class_metrics = {class_name: {'cer': [], 'wer': []} for class_name in text_classes}
    
    for img in gt_data['images']:
        img_id = img['id']
        img_path = Path(images_dir) / img['file_name']
        
        if not img_path.exists():
            continue
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Get annotations for this image
        img_gt_anns = [ann for ann in gt_text_anns if ann['image_id'] == img_id]
        img_pred_anns = [ann for ann in pred_text_anns if ann['image_id'] == img_id]
        
        if not img_gt_anns:
            continue
        
        # Align predictions to ground truth
        matches = align_predictions_to_gt(img_gt_anns, img_pred_anns)
        
        # Process each match
        for gt_ann, pred_ann in matches:
            # Get ground truth text from annotation
            gt_text = gt_ann.get('text', '')
            
            if not gt_text:
                # Extract text using OCR on ground truth region
                bbox = gt_ann['bbox']
                x, y, w, h = bbox
                crop = image[y:y+h, x:x+w]
                
                if crop.size > 0:
                    # Create layout box for OCR
                    layout_box = LayoutBox(
                        class_name=gt_ann.get('class', 'Text'),
                        bbox=(x, y, w, h),
                        score=1.0
                    )
                    
                    # Run OCR
                    elements = ocr_lang.run(image, [layout_box], 0)
                    if elements and elements[0].content:
                        gt_text = elements[0].content
            
            # Get predicted text
            pred_text = pred_ann.get('text', '')
            
            if not pred_text:
                # Extract text using OCR on predicted region
                bbox = pred_ann['bbox']
                x, y, w, h = bbox
                crop = image[y:y+h, x:x+w]
                
                if crop.size > 0:
                    # Create layout box for OCR
                    layout_box = LayoutBox(
                        class_name=pred_ann.get('class', 'Text'),
                        bbox=(x, y, w, h),
                        score=pred_ann.get('score', 1.0)
                    )
                    
                    # Run OCR
                    elements = ocr_lang.run(image, [layout_box], 0)
                    if elements and elements[0].content:
                        pred_text = elements[0].content
            
            # Compute metrics
            if gt_text and pred_text:
                cer, wer = compute_cer_wer(gt_text, pred_text)
                all_cer.append(cer)
                all_wer.append(wer)
                
                # Per-class metrics
                class_name = gt_ann.get('class', 'Text')
                if class_name in per_class_metrics:
                    per_class_metrics[class_name]['cer'].append(cer)
                    per_class_metrics[class_name]['wer'].append(wer)
    
    # Compute overall metrics
    overall_metrics = {
        'mean_cer': float(np.mean(all_cer)) if all_cer else 0.0,
        'mean_wer': float(np.mean(all_wer)) if all_wer else 0.0,
        'std_cer': float(np.std(all_cer)) if all_cer else 0.0,
        'std_wer': float(np.std(all_wer)) if all_wer else 0.0,
        'total_samples': len(all_cer)
    }
    
    # Compute per-class metrics
    for class_name, metrics in per_class_metrics.items():
        if metrics['cer']:
            metrics['mean_cer'] = float(np.mean(metrics['cer']))
            metrics['mean_wer'] = float(np.mean(metrics['wer']))
            metrics['std_cer'] = float(np.std(metrics['cer']))
            metrics['std_wer'] = float(np.std(metrics['wer']))
            metrics['count'] = len(metrics['cer'])
        else:
            metrics['mean_cer'] = 0.0
            metrics['mean_wer'] = 0.0
            metrics['std_cer'] = 0.0
            metrics['std_wer'] = 0.0
            metrics['count'] = 0
    
    # Combine metrics
    results = {
        'overall': overall_metrics,
        'per_class': per_class_metrics
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "text_metrics.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Text evaluation completed. Mean CER: {overall_metrics['mean_cer']:.3f}, "
          f"Mean WER: {overall_metrics['mean_wer']:.3f}")
    
    return results


def generate_text_report(metrics: Dict[str, Any], output_dir: str) -> str:
    """Generate text evaluation report.
    
    Args:
        metrics: Evaluation metrics
        output_dir: Output directory
        
    Returns:
        Path to report file
    """
    report_file = Path(output_dir) / "text_report.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Text Evaluation Report\n\n")
        
        # Overall metrics
        overall = metrics['overall']
        f.write("## Overall Metrics\n\n")
        f.write(f"- **Mean CER**: {overall['mean_cer']:.3f} ± {overall['std_cer']:.3f}\n")
        f.write(f"- **Mean WER**: {overall['mean_wer']:.3f} ± {overall['std_wer']:.3f}\n")
        f.write(f"- **Total Samples**: {overall['total_samples']}\n\n")
        
        # Per-class metrics
        f.write("## Per-Class Metrics\n\n")
        f.write("| Class | Mean CER | Mean WER | Count |\n")
        f.write("|-------|----------|----------|-------|\n")
        
        for class_name, class_metrics in metrics['per_class'].items():
            f.write(f"| {class_name} | {class_metrics['mean_cer']:.3f} | "
                   f"{class_metrics['mean_wer']:.3f} | {class_metrics['count']} |\n")
    
    print(f"[INFO] Text report saved to {report_file}")
    
    return str(report_file)
