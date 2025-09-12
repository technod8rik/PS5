"""
Custom per-image JSON ingester for PS5 dataset.

This module handles the conversion from per-image JSON format to YOLO/COCO formats
with proper ID remapping and data validation.
"""

import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import cv2
import numpy as np
from collections import Counter, defaultdict
import yaml


def ingest_perimage_json(
    src_dir: str,
    out_root: str,
    split: Tuple[float, float, float] = (0.9, 0.1, 0.0),
    id_base: Union[str, int] = "auto",
    id_to_name: Optional[Dict[int, str]] = None,
    seed: int = 42,
    make_coco: bool = True,
    make_yolo: bool = True,
) -> Dict:
    """
    Ingest per-image JSON format to YOLO/COCO formats.
    
    Args:
        src_dir: Source directory containing images and JSON files
        out_root: Output root directory
        split: Train/val/test split ratios (default: 0.9, 0.1, 0.0)
        id_base: ID base for remapping ("auto" or 0/1)
        id_to_name: Optional mapping from original ID to class name
        seed: Random seed for deterministic splits
        make_coco: Whether to create COCO format files
        make_yolo: Whether to create YOLO format files
        
    Returns:
        Dict with ingestion statistics and paths
    """
    src_path = Path(src_dir)
    out_path = Path(out_root)
    
    if not src_path.exists():
        raise ValueError(f"Source directory not found: {src_dir}")
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"[INGEST] Starting ingestion from {src_dir}")
    print(f"[INGEST] Output directory: {out_root}")
    
    # Find all image files and their corresponding JSON files
    image_files = []
    json_files = []
    
    for ext in ['.png', '.jpg', '.jpeg']:
        for img_file in src_path.glob(f"*{ext}"):
            json_file = img_file.with_suffix('.json')
            if json_file.exists():
                image_files.append(img_file)
                json_files.append(json_file)
            else:
                print(f"[WARN] Missing JSON for {img_file.name}")
    
    print(f"[INGEST] Found {len(image_files)} image-JSON pairs")
    
    if not image_files:
        raise ValueError("No valid image-JSON pairs found")
    
    # Process all files to collect data and determine ID mapping
    all_data = []
    id_counts = Counter()
    id_names = defaultdict(list)
    total_boxes = 0
    dropped_boxes = 0
    
    for img_file, json_file in zip(image_files, json_files):
        try:
            # Load JSON
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Load image to get dimensions
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"[WARN] Could not load image {img_file.name}")
                continue
            
            h, w = img.shape[:2]
            
            # Process annotations
            annotations = []
            for ann in data.get('annotations', []):
                bbox = ann.get('bbox', [])
                if len(bbox) != 4:
                    continue
                
                x, y, w_box, h_box = bbox
                category_id = ann.get('category_id')
                category_name = ann.get('category_name', '')
                
                if category_id is None:
                    continue
                
                # Convert to pixel coordinates
                x_px = x
                y_px = y
                w_px = w_box
                h_px = h_box
                
                # Clamp to image bounds
                x_px = max(0, min(x_px, w - 1))
                y_px = max(0, min(y_px, h - 1))
                w_px = max(0, min(w_px, w - x_px))
                h_px = max(0, min(h_px, h - y_px))
                
                # Drop degenerate boxes
                if w_px <= 1 or h_px <= 1:
                    dropped_boxes += 1
                    continue
                
                # Convert to YOLO format (normalized center coordinates)
                xc = (x_px + w_px / 2) / w
                yc = (y_px + h_px / 2) / h
                w_norm = w_px / w
                h_norm = h_px / h
                
                # Clamp to [0, 1]
                xc = max(0, min(1, xc))
                yc = max(0, min(1, yc))
                w_norm = max(0, min(1, w_norm))
                h_norm = max(0, min(1, h_norm))
                
                annotation = {
                    'bbox': [x, y, w_box, h_box],  # Original bbox
                    'bbox_yolo': [xc, yc, w_norm, h_norm],  # YOLO format
                    'bbox_coco': [int(x_px), int(y_px), int(w_px), int(h_px)],  # COCO format
                    'category_id': category_id,
                    'category_name': category_name
                }
                
                annotations.append(annotation)
                id_counts[category_id] += 1
                if category_name:
                    id_names[category_id].append(category_name)
                total_boxes += 1
            
            # Store image data
            image_data = {
                'image_file': img_file,
                'json_file': json_file,
                'width': w,
                'height': h,
                'annotations': annotations,
                'corruption': data.get('corruption', {})
            }
            all_data.append(image_data)
            
        except Exception as e:
            print(f"[WARN] Error processing {img_file.name}: {e}")
            continue
    
    print(f"[INGEST] Processed {len(all_data)} images")
    print(f"[INGEST] Total boxes: {total_boxes}, Dropped: {dropped_boxes}")
    
    # Determine ID mapping
    original_ids = sorted(id_counts.keys())
    print(f"[INGEST] Found category IDs: {original_ids}")
    
    if id_base == "auto":
        # Determine if IDs start from 0 or 1
        if 0 in original_ids:
            id_base = 0
        else:
            id_base = 1
    else:
        id_base = int(id_base)
    
    # Create ID mapping
    id_mapping = {}
    for i, orig_id in enumerate(original_ids):
        id_mapping[orig_id] = i
    
    print(f"[INGEST] ID mapping: {id_mapping}")
    
    # Determine class names
    class_names = []
    for orig_id in original_ids:
        if id_to_name and orig_id in id_to_name:
            name = id_to_name[orig_id]
        elif orig_id in id_names and id_names[orig_id]:
            # Use most frequent name
            name = Counter(id_names[orig_id]).most_common(1)[0][0]
        else:
            name = f"class_{orig_id}"
        class_names.append(name)
    
    print(f"[INGEST] Class names: {class_names}")
    
    # Create output directories
    if make_yolo:
        yolo_path = out_path / "yolo"
        yolo_path.mkdir(parents=True, exist_ok=True)
        (yolo_path / "images" / "train").mkdir(parents=True, exist_ok=True)
        (yolo_path / "images" / "val").mkdir(parents=True, exist_ok=True)
        (yolo_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (yolo_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    if make_coco:
        coco_path = out_path / "coco"
        coco_path.mkdir(parents=True, exist_ok=True)
    
    # Create splits
    image_stems = [data['image_file'].stem for data in all_data]
    random.shuffle(image_stems)
    
    n_train = int(len(image_stems) * split[0])
    n_val = int(len(image_stems) * split[1])
    
    train_stems = set(image_stems[:n_train])
    val_stems = set(image_stems[n_train:n_train + n_val])
    test_stems = set(image_stems[n_train + n_val:])
    
    print(f"[INGEST] Split: {len(train_stems)} train, {len(val_stems)} val, {len(test_stems)} test")
    
    # Process images and create outputs
    train_data = []
    val_data = []
    test_data = []
    
    for data in all_data:
        stem = data['image_file'].stem
        if stem in train_stems:
            split_name = 'train'
            train_data.append(data)
        elif stem in val_stems:
            split_name = 'val'
            val_data.append(data)
        else:
            split_name = 'test'
            test_data.append(data)
        
        if make_yolo:
            # Copy image
            img_dst = yolo_path / "images" / split_name / data['image_file'].name
            shutil.copy2(data['image_file'], img_dst)
            
            # Create YOLO label file
            label_dst = yolo_path / "labels" / split_name / f"{data['image_file'].stem}.txt"
            with open(label_dst, 'w') as f:
                for ann in data['annotations']:
                    train_id = id_mapping[ann['category_id']]
                    xc, yc, w_norm, h_norm = ann['bbox_yolo']
                    f.write(f"{train_id} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    
    # Create data.yaml
    if make_yolo:
        data_yaml = {
            'path': str(yolo_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test' if test_stems else 'images/val',
            'nc': len(class_names),
            'names': class_names
        }
        
        with open(yolo_path / "data.yaml", 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"[INGEST] Created data.yaml with {len(class_names)} classes")
    
    # Create COCO format
    if make_coco:
        for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            if not split_data:
                continue
            
            coco_data = {
                'images': [],
                'annotations': [],
                'categories': []
            }
            
            # Add categories
            for i, name in enumerate(class_names):
                coco_data['categories'].append({
                    'id': i,
                    'name': name,
                    'supercategory': 'document'
                })
            
            # Add images and annotations
            ann_id = 0
            for img_data in split_data:
                # Add image
                img_info = {
                    'id': len(coco_data['images']),
                    'file_name': img_data['image_file'].name,
                    'width': img_data['width'],
                    'height': img_data['height']
                }
                
                # Add corruption info if available
                if img_data['corruption']:
                    img_info['file_attributes'] = img_data['corruption']
                
                coco_data['images'].append(img_info)
                
                # Add annotations
                for ann in img_data['annotations']:
                    train_id = id_mapping[ann['category_id']]
                    x, y, w, h = ann['bbox_coco']
                    
                    coco_data['annotations'].append({
                        'id': ann_id,
                        'image_id': img_info['id'],
                        'category_id': train_id,
                        'bbox': [x, y, w, h],
                        'area': w * h,
                        'iscrowd': 0
                    })
                    ann_id += 1
            
            # Save COCO file
            coco_file = coco_path / f"{split_name}.json"
            with open(coco_file, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            print(f"[INGEST] Created {coco_file} with {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    
    # Create category mapping file
    cat_map = {
        'orig_to_train': {str(k): v for k, v in id_mapping.items()},
        'names': {str(v): class_names[v] for v in range(len(class_names))},
        'stats': {
            'images': len(all_data),
            'boxes': total_boxes,
            'dropped_boxes': dropped_boxes
        }
    }
    
    with open(out_path / "cat_map.json", 'w') as f:
        json.dump(cat_map, f, indent=2)
    
    # Create debug overlays
    debug_path = out_path / "debug" / "overlay_samples"
    debug_path.mkdir(parents=True, exist_ok=True)
    
    # Sample 20 random images for overlay
    sample_data = random.sample(all_data, min(20, len(all_data)))
    
    for i, data in enumerate(sample_data):
        img = cv2.imread(str(data['image_file']))
        if img is None:
            continue
        
        # Draw bounding boxes
        for ann in data['annotations']:
            train_id = id_mapping[ann['category_id']]
            x, y, w, h = ann['bbox_coco']
            
            # Color based on class
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)][train_id % 5]
            
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{train_id}:{class_names[train_id]}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save overlay
        overlay_file = debug_path / f"overlay_{i:02d}_{data['image_file'].name}"
        cv2.imwrite(str(overlay_file), img)
    
    print(f"[INGEST] Created {len(sample_data)} debug overlays in {debug_path}")
    
    # Return results
    results = {
        'images_processed': len(all_data),
        'total_boxes': total_boxes,
        'dropped_boxes': dropped_boxes,
        'class_names': class_names,
        'id_mapping': id_mapping,
        'split_counts': {
            'train': len(train_data),
            'val': len(val_data),
            'test': len(test_data)
        }
    }
    
    if make_yolo:
        results['yolo_path'] = str(yolo_path)
        results['data_yaml'] = str(yolo_path / "data.yaml")
    
    if make_coco:
        results['coco_path'] = str(coco_path)
        results['coco_files'] = [str(coco_path / f"{split}.json") for split in ['train', 'val', 'test'] if (coco_path / f"{split}.json").exists()]
    
    results['cat_map'] = str(out_path / "cat_map.json")
    results['debug_overlays'] = str(debug_path)
    
    print(f"[INGEST] ✅ Ingestion complete!")
    print(f"[INGEST] YOLO data.yaml: {results.get('data_yaml', 'N/A')}")
    print(f"[INGEST] COCO files: {results.get('coco_files', [])}")
    print(f"[INGEST] Category mapping: {results['cat_map']}")
    
    return results
