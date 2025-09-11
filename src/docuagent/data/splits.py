"""Data splitting utilities for training, validation, and test sets."""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


def stratified_split(coco_json: str, out_root: str, train: float = 0.8, val: float = 0.1, 
                    test: float = 0.1, strat_keys: Tuple[str, ...] = ("language", "doctype")) -> None:
    """Create stratified train/val/test splits from COCO JSON.
    
    Args:
        coco_json: Path to COCO JSON file
        out_root: Output directory for splits
        train: Training set proportion
        val: Validation set proportion
        test: Test set proportion
        strat_keys: Keys to stratify on (language, doctype, etc.)
    """
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("Train, val, and test proportions must sum to 1.0")
    
    # Load COCO data
    with open(coco_json, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # Create output directories
    out_path = Path(out_root)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Group images by stratification keys
    image_groups = defaultdict(list)
    
    for img in coco_data['images']:
        # Extract stratification values
        strat_values = []
        for key in strat_keys:
            if key in img:
                strat_values.append(str(img[key]))
            else:
                strat_values.append("unknown")
        
        strat_key = tuple(strat_values)
        image_groups[strat_key].append(img)
    
    print(f"[INFO] Found {len(image_groups)} stratification groups")
    
    # Split each group
    train_images = []
    val_images = []
    test_images = []
    
    for group_key, images in image_groups.items():
        if len(images) < 3:
            # Small groups go to train
            train_images.extend(images)
            continue
        
        # Shuffle group
        random.shuffle(images)
        
        # Calculate split indices
        n = len(images)
        train_end = int(n * train)
        val_end = train_end + int(n * val)
        
        # Split group
        train_images.extend(images[:train_end])
        val_images.extend(images[train_end:val_end])
        test_images.extend(images[val_end:])
    
    # Create splits
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    # Create COCO files for each split
    for split_name, images in splits.items():
        if not images:
            print(f"[WARN] No images in {split_name} split")
            continue
        
        # Get image IDs
        image_ids = {img['id'] for img in images}
        
        # Filter annotations
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids]
        
        # Create split COCO data
        split_data = {
            'images': images,
            'annotations': annotations,
            'categories': coco_data['categories']
        }
        
        # Save split
        split_file = out_path / f"{split_name}.json"
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] {split_name}: {len(images)} images, {len(annotations)} annotations")
    
    # Create summary
    summary = {
        'total_images': len(coco_data['images']),
        'total_annotations': len(coco_data['annotations']),
        'splits': {
            'train': len(splits['train']),
            'val': len(splits['val']),
            'test': len(splits['test'])
        },
        'stratification_keys': list(strat_keys)
    }
    
    with open(out_path / "split_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Splits saved to {out_path}")


def uniform_split(coco_json: str, out_root: str, train: float = 0.8, val: float = 0.1, 
                 test: float = 0.1) -> None:
    """Create uniform random train/val/test splits from COCO JSON.
    
    Args:
        coco_json: Path to COCO JSON file
        out_root: Output directory for splits
        train: Training set proportion
        val: Validation set proportion
        test: Test set proportion
    """
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("Train, val, and test proportions must sum to 1.0")
    
    # Load COCO data
    with open(coco_json, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # Create output directories
    out_path = Path(out_root)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Shuffle images
    images = coco_data['images'].copy()
    random.shuffle(images)
    
    # Calculate split indices
    n = len(images)
    train_end = int(n * train)
    val_end = train_end + int(n * val)
    
    # Split images
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]
    
    # Create splits
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    # Create COCO files for each split
    for split_name, images in splits.items():
        if not images:
            print(f"[WARN] No images in {split_name} split")
            continue
        
        # Get image IDs
        image_ids = {img['id'] for img in images}
        
        # Filter annotations
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids]
        
        # Create split COCO data
        split_data = {
            'images': images,
            'annotations': annotations,
            'categories': coco_data['categories']
        }
        
        # Save split
        split_file = out_path / f"{split_name}.json"
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] {split_name}: {len(images)} images, {len(annotations)} annotations")
    
    print(f"[INFO] Uniform splits saved to {out_path}")


def page_level_split(coco_json: str, out_root: str, train: float = 0.8, val: float = 0.1, 
                    test: float = 0.1) -> None:
    """Create page-level splits (keeps all annotations from same page together).
    
    Args:
        coco_json: Path to COCO JSON file
        out_root: Output directory for splits
        train: Training set proportion
        val: Validation set proportion
        test: Test set proportion
    """
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("Train, val, and test proportions must sum to 1.0")
    
    # Load COCO data
    with open(coco_json, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # Create output directories
    out_path = Path(out_root)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Group images by page (assuming page info is available)
    page_groups = defaultdict(list)
    
    for img in coco_data['images']:
        # Try to extract page number from filename or metadata
        page_num = 0
        if 'page' in img:
            page_num = img['page']
        elif 'file_name' in img:
            # Try to extract page number from filename
            import re
            match = re.search(r'page[_\s]*(\d+)', img['file_name'], re.IGNORECASE)
            if match:
                page_num = int(match.group(1))
        
        page_groups[page_num].append(img)
    
    # Convert to list and shuffle
    pages = list(page_groups.values())
    random.shuffle(pages)
    
    # Calculate split indices
    n_pages = len(pages)
    train_end = int(n_pages * train)
    val_end = train_end + int(n_pages * val)
    
    # Split pages
    train_pages = pages[:train_end]
    val_pages = pages[train_end:val_end]
    test_pages = pages[val_end:]
    
    # Flatten page groups
    splits = {
        'train': [img for page in train_pages for img in page],
        'val': [img for page in val_pages for img in page],
        'test': [img for page in test_pages for img in page]
    }
    
    # Create COCO files for each split
    for split_name, images in splits.items():
        if not images:
            print(f"[WARN] No images in {split_name} split")
            continue
        
        # Get image IDs
        image_ids = {img['id'] for img in images}
        
        # Filter annotations
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids]
        
        # Create split COCO data
        split_data = {
            'images': images,
            'annotations': annotations,
            'categories': coco_data['categories']
        }
        
        # Save split
        split_file = out_path / f"{split_name}.json"
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] {split_name}: {len(images)} images, {len(annotations)} annotations")
    
    print(f"[INFO] Page-level splits saved to {out_path}")
