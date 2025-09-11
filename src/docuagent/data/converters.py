"""COCO ↔ YOLO format converters for layout detection data."""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np


def load_label_map(class_names: List[str]) -> Dict[str, int]:
    """Create stable label mapping from class names to IDs.
    
    Args:
        class_names: List of class names in order
        
    Returns:
        Dictionary mapping class names to IDs
    """
    return {name: idx for idx, name in enumerate(class_names)}


def validate_hbb(box: List[float], img_width: int, img_height: int) -> List[float]:
    """Validate and clip HBB to image bounds.
    
    Args:
        box: HBB as [x, y, w, h]
        img_width: Image width
        img_height: Image height
        
    Returns:
        Validated and clipped HBB
    """
    x, y, w, h = box
    
    # Ensure positive dimensions
    w = max(0, w)
    h = max(0, h)
    
    # Clip to image bounds
    x = max(0, min(x, img_width - w))
    y = max(0, min(y, img_height - h))
    w = min(w, img_width - x)
    h = min(h, img_height - y)
    
    return [x, y, w, h]


def coco_to_yolo(coco_json_path: str, out_dir: str, class_names: List[str]) -> None:
    """Convert COCO format to YOLO format.
    
    Args:
        coco_json_path: Path to COCO JSON file
        out_dir: Output directory for YOLO format
        class_names: List of class names in order
    """
    coco_path = Path(coco_json_path)
    out_path = Path(out_dir)
    
    # Create output directories
    (out_path / "images").mkdir(parents=True, exist_ok=True)
    (out_path / "labels").mkdir(parents=True, exist_ok=True)
    
    # Load COCO data
    with open(coco_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # Create label mapping
    label_map = load_label_map(class_names)
    
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
        
        # Validate and clip bbox
        bbox = validate_hbb(bbox, img_width, img_height)
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
        label_file = out_path / "labels" / f"{img_name}.txt"
        
        # Append to label file
        with open(label_file, 'a', encoding='utf-8') as f:
            class_id = label_map[cat_name]
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
    
    # Copy images
    for img_info in coco_data['images']:
        src_path = coco_path.parent / img_info['file_name']
        dst_path = out_path / "images" / img_info['file_name']
        
        if src_path.exists():
            # Create symlink or copy
            try:
                dst_path.symlink_to(src_path.absolute())
            except OSError:
                # Fallback to copy if symlink fails
                import shutil
                shutil.copy2(src_path, dst_path)
    
    # Create data.yaml
    data_yaml = {
        'path': str(out_path.absolute()),
        'train': 'images',
        'val': 'images',
        'test': 'images',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(out_path / "data.yaml", 'w', encoding='utf-8') as f:
        import yaml
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"[INFO] Converted COCO to YOLO format in {out_path}")
    print(f"[INFO] Found {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")


def yolo_to_coco(images_dir: str, labels_dir: str, out_json_path: str, class_names: List[str]) -> None:
    """Convert YOLO format to COCO format.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO label files
        out_json_path: Output COCO JSON file path
        class_names: List of class names in order
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    out_path = Path(out_json_path)
    
    # Create label mapping
    label_map = {idx: name for idx, name in enumerate(class_names)}
    
    # Initialize COCO structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": idx, "name": name} for idx, name in enumerate(class_names)]
    }
    
    image_id = 0
    ann_id = 0
    
    # Process each image
    for img_file in images_path.glob("*"):
        if not img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue
        
        # Get image dimensions
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        height, width = img.shape[:2]
        
        # Add image info
        image_info = {
            "id": image_id,
            "file_name": img_file.name,
            "width": width,
            "height": height
        }
        coco_data["images"].append(image_info)
        
        # Process corresponding label file
        label_file = labels_path / f"{img_file.stem}.txt"
        if label_file.exists():
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    norm_width = float(parts[3])
                    norm_height = float(parts[4])
                    
                    if class_id not in label_map:
                        continue
                    
                    # Convert back to COCO format
                    x = (x_center - norm_width / 2) * width
                    y = (y_center - norm_height / 2) * height
                    w = norm_width * width
                    h = norm_height * height
                    
                    # Validate bbox
                    bbox = validate_hbb([x, y, w, h], width, height)
                    x, y, w, h = bbox
                    
                    if w <= 0 or h <= 0:
                        continue
                    
                    # Add annotation
                    ann = {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    }
                    coco_data["annotations"].append(ann)
                    ann_id += 1
        
        image_id += 1
    
    # Save COCO JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Converted YOLO to COCO format: {out_path}")
    print(f"[INFO] Found {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")


def validate_conversion(coco_json_path: str, class_names: List[str]) -> Dict[str, Any]:
    """Validate COCO to YOLO to COCO conversion.
    
    Args:
        coco_json_path: Path to original COCO JSON
        class_names: List of class names
        
    Returns:
        Validation results
    """
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        original = json.load(f)
    
    # Count original annotations
    original_counts = {}
    for ann in original['annotations']:
        cat_id = ann['category_id']
        cat_name = next(cat['name'] for cat in original['categories'] if cat['id'] == cat_id)
        original_counts[cat_name] = original_counts.get(cat_name, 0) + 1
    
    return {
        "original_images": len(original['images']),
        "original_annotations": len(original['annotations']),
        "original_counts": original_counts,
        "class_names": class_names
    }
