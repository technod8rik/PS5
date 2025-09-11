"""Data sampling utilities for creating balanced subsets."""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict


def balanced_subset(coco_json: str, out_json: str, per_class: int = 200) -> None:
    """Create a balanced subset with specified number of samples per class.
    
    Args:
        coco_json: Path to input COCO JSON file
        out_json: Path to output COCO JSON file
        per_class: Number of samples per class
    """
    # Load COCO data
    with open(coco_json, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # Create category ID to name mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Group annotations by category
    ann_by_category = defaultdict(list)
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        if cat_id in categories:
            ann_by_category[cat_id].append(ann)
    
    # Sample annotations per category
    sampled_annotations = []
    sampled_image_ids = set()
    
    for cat_id, annotations in ann_by_category.items():
        cat_name = categories[cat_id]
        n_samples = min(per_class, len(annotations))
        
        # Randomly sample annotations
        sampled = random.sample(annotations, n_samples)
        sampled_annotations.extend(sampled)
        
        # Collect image IDs
        for ann in sampled:
            sampled_image_ids.add(ann['image_id'])
        
        print(f"[INFO] Sampled {n_samples} annotations for {cat_name}")
    
    # Filter images to only include those with sampled annotations
    sampled_images = [img for img in coco_data['images'] if img['id'] in sampled_image_ids]
    
    # Create subset COCO data
    subset_data = {
        'images': sampled_images,
        'annotations': sampled_annotations,
        'categories': coco_data['categories']
    }
    
    # Save subset
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(subset_data, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print(f"[INFO] Created balanced subset: {out_json}")
    print(f"[INFO] Images: {len(sampled_images)}")
    print(f"[INFO] Annotations: {len(sampled_annotations)}")
    
    # Print per-class counts
    class_counts = defaultdict(int)
    for ann in sampled_annotations:
        cat_id = ann['category_id']
        class_counts[categories[cat_id]] += 1
    
    for cat_name, count in class_counts.items():
        print(f"[INFO] {cat_name}: {count} annotations")


def stratified_subset(coco_json: str, out_json: str, per_class: int = 200, 
                     strat_key: str = "language") -> None:
    """Create a stratified subset with balanced samples per class and stratification key.
    
    Args:
        coco_json: Path to input COCO JSON file
        out_json: Path to output COCO JSON file
        per_class: Number of samples per class per stratification group
        strat_key: Key to stratify on (e.g., 'language', 'doctype')
    """
    # Load COCO data
    with open(coco_json, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # Create category ID to name mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Create image ID to image mapping
    images = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by category and stratification key
    ann_by_strat = defaultdict(lambda: defaultdict(list))
    
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        if cat_id not in categories:
            continue
        
        image_id = ann['image_id']
        if image_id not in images:
            continue
        
        img = images[image_id]
        strat_value = img.get(strat_key, "unknown")
        
        ann_by_strat[strat_value][cat_id].append(ann)
    
    # Sample annotations per category per stratification group
    sampled_annotations = []
    sampled_image_ids = set()
    
    for strat_value, cat_annotations in ann_by_strat.items():
        print(f"[INFO] Processing {strat_key}={strat_value}")
        
        for cat_id, annotations in cat_annotations.items():
            cat_name = categories[cat_id]
            n_samples = min(per_class, len(annotations))
            
            # Randomly sample annotations
            sampled = random.sample(annotations, n_samples)
            sampled_annotations.extend(sampled)
            
            # Collect image IDs
            for ann in sampled:
                sampled_image_ids.add(ann['image_id'])
            
            print(f"[INFO]   {cat_name}: {n_samples} annotations")
    
    # Filter images to only include those with sampled annotations
    sampled_images = [img for img in coco_data['images'] if img['id'] in sampled_image_ids]
    
    # Create subset COCO data
    subset_data = {
        'images': sampled_images,
        'annotations': sampled_annotations,
        'categories': coco_data['categories']
    }
    
    # Save subset
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(subset_data, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print(f"[INFO] Created stratified subset: {out_json}")
    print(f"[INFO] Images: {len(sampled_images)}")
    print(f"[INFO] Annotations: {len(sampled_annotations)}")
    
    # Print per-class counts
    class_counts = defaultdict(int)
    for ann in sampled_annotations:
        cat_id = ann['category_id']
        class_counts[categories[cat_id]] += 1
    
    for cat_name, count in class_counts.items():
        print(f"[INFO] {cat_name}: {count} annotations")


def random_subset(coco_json: str, out_json: str, max_images: int = 1000) -> None:
    """Create a random subset with specified maximum number of images.
    
    Args:
        coco_json: Path to input COCO JSON file
        out_json: Path to output COCO JSON file
        max_images: Maximum number of images to include
    """
    # Load COCO data
    with open(coco_json, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # Randomly sample images
    all_images = coco_data['images'].copy()
    random.shuffle(all_images)
    
    n_samples = min(max_images, len(all_images))
    sampled_images = all_images[:n_samples]
    sampled_image_ids = {img['id'] for img in sampled_images}
    
    # Filter annotations to only include those from sampled images
    sampled_annotations = [ann for ann in coco_data['annotations'] 
                          if ann['image_id'] in sampled_image_ids]
    
    # Create subset COCO data
    subset_data = {
        'images': sampled_images,
        'annotations': sampled_annotations,
        'categories': coco_data['categories']
    }
    
    # Save subset
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(subset_data, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Created random subset: {out_json}")
    print(f"[INFO] Images: {len(sampled_images)}")
    print(f"[INFO] Annotations: {len(sampled_annotations)}")


def analyze_dataset(coco_json: str) -> Dict[str, Any]:
    """Analyze dataset statistics.
    
    Args:
        coco_json: Path to COCO JSON file
        
    Returns:
        Dataset statistics
    """
    # Load COCO data
    with open(coco_json, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # Basic statistics
    n_images = len(coco_data['images'])
    n_annotations = len(coco_data['annotations'])
    n_categories = len(coco_data['categories'])
    
    # Per-category counts
    cat_counts = defaultdict(int)
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        cat_counts[cat_id] += 1
    
    # Category names
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    cat_name_counts = {categories[cat_id]: count for cat_id, count in cat_counts.items()}
    
    # Image dimensions
    widths = [img['width'] for img in coco_data['images']]
    heights = [img['height'] for img in coco_data['images']]
    
    # Annotation areas
    areas = [ann['area'] for ann in coco_data['annotations']]
    
    stats = {
        'n_images': n_images,
        'n_annotations': n_annotations,
        'n_categories': n_categories,
        'category_counts': cat_name_counts,
        'image_width': {
            'min': min(widths) if widths else 0,
            'max': max(widths) if widths else 0,
            'mean': sum(widths) / len(widths) if widths else 0
        },
        'image_height': {
            'min': min(heights) if heights else 0,
            'max': max(heights) if heights else 0,
            'mean': sum(heights) / len(heights) if heights else 0
        },
        'annotation_area': {
            'min': min(areas) if areas else 0,
            'max': max(areas) if areas else 0,
            'mean': sum(areas) / len(areas) if areas else 0
        }
    }
    
    return stats
