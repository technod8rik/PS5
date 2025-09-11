"""YOLOv10 training and evaluation utilities."""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import yaml

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    YOLO = None
    torch = None


def prepare_yolo_dataset(coco_json: str, images_dir: str, work_dir: str, 
                        class_names: List[str]) -> str:
    """Prepare YOLO dataset from COCO format.
    
    Args:
        coco_json: Path to COCO JSON file
        images_dir: Directory containing images
        work_dir: Working directory for YOLO dataset
        class_names: List of class names
        
    Returns:
        Path to data.yaml file
    """
    if YOLO is None:
        raise ImportError("ultralytics not available. Install with: pip install ultralytics")
    
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    
    # Create YOLO directory structure
    (work_path / "images").mkdir(exist_ok=True)
    (work_path / "labels").mkdir(exist_ok=True)
    
    # Load COCO data
    with open(coco_json, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # Create label mapping
    label_map = {name: idx for idx, name in enumerate(class_names)}
    
    # Create image ID to info mapping
    images = {img['id']: img for img in coco_data['images']}
    
    # Create category ID to name mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Process annotations
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in images:
            continue
            
        img_info = images[image_id]
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Get category name
        cat_id = ann['category_id']
        if cat_id not in categories:
            continue
        cat_name = categories[cat_id]
        
        if cat_name not in label_map:
            continue
        
        # Convert bbox to YOLO format
        bbox = ann['bbox']  # COCO format: [x, y, w, h]
        x, y, w, h = bbox
        
        if w <= 0 or h <= 0:
            continue
        
        # Convert to YOLO format (normalized center coordinates)
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        norm_width = w / img_width
        norm_height = h / img_height
        
        # Create label file
        img_name = Path(img_info['file_name']).stem
        label_file = work_path / "labels" / f"{img_name}.txt"
        
        # Append to label file
        with open(label_file, 'a', encoding='utf-8') as f:
            class_id = label_map[cat_name]
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
    
    # Copy images
    for img_info in coco_data['images']:
        src_path = Path(images_dir) / img_info['file_name']
        dst_path = work_path / "images" / img_info['file_name']
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
    
    # Create data.yaml
    data_yaml = {
        'path': str(work_path.absolute()),
        'train': 'images',
        'val': 'images',
        'test': 'images',
        'nc': len(class_names),
        'names': class_names
    }
    
    data_yaml_path = work_path / "data.yaml"
    with open(data_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"[INFO] Prepared YOLO dataset: {work_path}")
    print(f"[INFO] Found {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
    
    return str(data_yaml_path)


def train_yolo(data_yaml: str, model: str, imgsz: int, epochs: int, batch: int, 
               lr0: float, project: str, name: str, device: str = "auto") -> str:
    """Train YOLO model.
    
    Args:
        data_yaml: Path to data.yaml file
        model: Model name or path (e.g., 'yolov10n.pt')
        imgsz: Image size
        epochs: Number of epochs
        batch: Batch size
        lr0: Initial learning rate
        project: Project directory
        name: Run name
        device: Device to use ('auto', 'cpu', 'cuda')
        
    Returns:
        Path to best weights
    """
    if YOLO is None:
        raise ImportError("ultralytics not available. Install with: pip install ultralytics")
    
    # Load model
    model = YOLO(model)
    
    # Train model
    results = model.train(
        data=data_yaml,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        lr0=lr0,
        project=project,
        name=name,
        device=device,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True
    )
    
    # Get best weights path
    best_weights = Path(project) / name / "weights" / "best.pt"
    
    if best_weights.exists():
        print(f"[INFO] Training completed. Best weights: {best_weights}")
        return str(best_weights)
    else:
        print(f"[WARN] Best weights not found at {best_weights}")
        return ""


def eval_yolo(weights: str, data_yaml: str, project: str, name: str) -> Dict[str, Any]:
    """Evaluate YOLO model.
    
    Args:
        weights: Path to model weights
        data_yaml: Path to data.yaml file
        project: Project directory
        name: Run name
        
    Returns:
        Evaluation metrics
    """
    if YOLO is None:
        raise ImportError("ultralytics not available. Install with: pip install ultralytics")
    
    # Load model
    model = YOLO(weights)
    
    # Evaluate model
    results = model.val(
        data=data_yaml,
        project=project,
        name=name,
        save=True,
        save_txt=True,
        save_json=True,
        plots=True,
        verbose=True
    )
    
    # Extract metrics
    metrics = {
        'mAP50': float(results.box.map50),
        'mAP50-95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr),
        'f1': float(results.box.f1)
    }
    
    # Per-class metrics
    if hasattr(results.box, 'maps') and results.box.maps is not None:
        class_names = model.names
        per_class_metrics = {}
        for i, (class_id, class_name) in enumerate(class_names.items()):
            if i < len(results.box.maps):
                per_class_metrics[class_name] = {
                    'mAP50': float(results.box.maps[i]),
                    'precision': float(results.box.p[i]) if i < len(results.box.p) else 0.0,
                    'recall': float(results.box.r[i]) if i < len(results.box.r) else 0.0
                }
        metrics['per_class'] = per_class_metrics
    
    print(f"[INFO] Evaluation completed. mAP50: {metrics['mAP50']:.3f}, mAP50-95: {metrics['mAP50-95']:.3f}")
    
    return metrics


def export_yolo(weights: str, fmt: str = "onnx") -> str:
    """Export YOLO model to different formats.
    
    Args:
        weights: Path to model weights
        fmt: Export format ('onnx', 'torchscript', 'tflite', etc.)
        
    Returns:
        Path to exported model
    """
    if YOLO is None:
        raise ImportError("ultralytics not available. Install with: pip install ultralytics")
    
    # Load model
    model = YOLO(weights)
    
    # Export model
    exported_path = model.export(format=fmt)
    
    print(f"[INFO] Model exported to {exported_path}")
    
    return exported_path


def predict_yolo(weights: str, images_dir: str, output_dir: str, 
                conf_threshold: float = 0.25, iou_threshold: float = 0.5) -> str:
    """Run inference on images using trained YOLO model.
    
    Args:
        weights: Path to model weights
        images_dir: Directory containing images
        output_dir: Output directory for predictions
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        
    Returns:
        Path to predictions JSON file
    """
    if YOLO is None:
        raise ImportError("ultralytics not available. Install with: pip install ultralytics")
    
    # Load model
    model = YOLO(weights)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    results = model.predict(
        source=images_dir,
        conf=conf_threshold,
        iou=iou_threshold,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(output_path),
        name="predictions"
    )
    
    # Convert results to COCO format
    predictions = []
    for result in results:
        if result.boxes is not None:
            img_path = Path(result.path)
            img_name = img_path.name
            
            for box in result.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Convert to COCO format
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                
                prediction = {
                    'image_name': img_name,
                    'category_id': cls,
                    'bbox': [x, y, w, h],
                    'score': float(conf)
                }
                predictions.append(prediction)
    
    # Save predictions
    predictions_file = output_path / "predictions.json"
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Predictions saved to {predictions_file}")
    
    return str(predictions_file)


def save_best_weights(weights_path: str, output_dir: str, model_name: str = "doclayout_yolov10_best.pt") -> str:
    """Save best weights to specified directory.
    
    Args:
        weights_path: Path to best weights
        output_dir: Output directory
        model_name: Name for saved weights
        
    Returns:
        Path to saved weights
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_weights = output_path / model_name
    shutil.copy2(weights_path, saved_weights)
    
    print(f"[INFO] Best weights saved to {saved_weights}")
    
    return str(saved_weights)
