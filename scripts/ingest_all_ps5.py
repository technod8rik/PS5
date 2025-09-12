#!/usr/bin/env python3
"""
PS5 Dataset Ingester: Convert per-image JSON + PNG to YOLO/COCO formats

The names order defines YOLO class indices; we lock: [Text, Title, List, Table, Figure].
Original IDs {1,2,5,3,4} are remapped to {0,1,4,2,3} by id but stored as {1→0,2→1,3→2,4→3,5→4} for training consistency.

How resume works (.done index) and how to re-run clean (delete .done and output dirs).
What to do if a new category_id appears later (rerun with an updated --map).

Usage:
    python scripts/ingest_all_ps5.py \
      --src "/home/akshar/PS5/data/PS 5 Intelligent Multilingual Document Understanding/extracted_data/train" \
      --out data/ps5_ingested \
      --split 0.9 0.1 0.0 \
      --seed 42 \
      --resume \
      --map "1:Text,2:Title,3:List,4:Table,5:Figure"
"""

import argparse
import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest PS5 dataset to YOLO/COCO formats")
    parser.add_argument("--src", required=True, help="Source directory with images and JSON files")
    parser.add_argument("--out", required=True, help="Output root directory")
    parser.add_argument("--split", nargs=3, type=float, default=[0.9, 0.1, 0.0], 
                       help="Train/val/test split ratios")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits")
    parser.add_argument("--resume", action="store_true", help="Resume from .done index")
    parser.add_argument("--map", default="1:Text,2:Title,3:List,4:Table,5:Figure",
                       help="Original ID to name mapping")
    return parser.parse_args()


def parse_id_mapping(map_str: str) -> Tuple[Dict[int, str], Dict[int, int]]:
    """Parse ID mapping string and create remapping to contiguous indices.
    
    Returns:
        orig_to_name: {orig_id: name}
        orig_to_train: {orig_id: train_id} where train_id is 0-based contiguous
    """
    # Parse mapping string
    orig_to_name = {}
    for pair in map_str.split(","):
        orig_id, name = pair.strip().split(":")
        orig_to_name[int(orig_id)] = name.strip()
    
    # Define canonical order
    canonical_order = ["Text", "Title", "List", "Table", "Figure"]
    
    # Create remapping: orig_id -> train_id (0-based contiguous)
    orig_to_train = {}
    for orig_id, name in orig_to_name.items():
        if name in canonical_order:
            train_id = canonical_order.index(name)
            orig_to_train[orig_id] = train_id
        else:
            raise ValueError(f"Unknown class name: {name}. Must be one of {canonical_order}")
    
    return orig_to_name, orig_to_train


def find_image_json_pairs(src_dir: Path) -> List[Tuple[Path, Path]]:
    """Find all matching image-JSON pairs in source directory."""
    pairs = []
    
    # Find all JSON files
    json_files = list(src_dir.glob("*.json"))
    
    for json_file in json_files:
        # Look for corresponding image file (case-insensitive)
        stem = json_file.stem
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
            img_file = src_dir / f"{stem}{ext}"
            if img_file.exists():
                pairs.append((img_file, json_file))
                break
        else:
            print(f"[WARN] No image found for {json_file}")
    
    return pairs


def clamp_bbox(bbox: List[float], img_w: int, img_h: int) -> Optional[List[float]]:
    """Clamp bounding box to image bounds and check for degeneracy.
    
    Args:
        bbox: [x, y, w, h] in pixels
        img_w, img_h: Image dimensions
    
    Returns:
        Clamped bbox or None if degenerate
    """
    x, y, w, h = bbox
    
    # Clamp to image bounds
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    
    # Check for degeneracy
    if w <= 1 or h <= 1:
        return None
    
    return [x, y, w, h]


def hbb_to_yolo(bbox: List[float], img_w: int, img_h: int) -> List[float]:
    """Convert HBB [x,y,w,h] to YOLO normalized [xc,yc,w,h]."""
    x, y, w, h = bbox
    xc = (x + w/2) / img_w
    yc = (y + h/2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return [xc, yc, w_norm, h_norm]


def create_overlay(img_path: Path, annotations: List[Dict], orig_to_train: Dict[int, int], 
                   class_names: List[str], output_path: Path):
    """Create debug overlay with colored bounding boxes."""
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        return
    
    # Color palette (BGR format for OpenCV)
    colors = [
        (255, 0, 0),    # Red - Text
        (0, 255, 0),    # Green - Title  
        (0, 0, 255),    # Blue - List
        (255, 255, 0),  # Cyan - Table
        (255, 0, 255),  # Magenta - Figure
    ]
    
    # Draw boxes
    for ann in annotations:
        bbox = ann["bbox"]
        orig_id = ann["category_id"]
        
        if orig_id in orig_to_train:
            train_id = orig_to_train[orig_id]
            color = colors[train_id % len(colors)]
            
            # Draw rectangle
            x, y, w, h = map(int, bbox)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{class_names[train_id]}:{orig_id}"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Save overlay
    cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def process_image_json_pair(img_path: Path, json_path: Path, orig_to_train: Dict[int, int],
                           class_names: List[str], output_dirs: Dict[str, Path],
                           coco_data: Dict, stats: Dict, overlay_dir: Path) -> bool:
    """Process a single image-JSON pair."""
    try:
        # Load JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Load image to get dimensions
        img = Image.open(img_path)
        img_w, img_h = img.size
        
        # Process annotations
        yolo_lines = []
        coco_annotations = []
        
        for ann in data.get("annotations", []):
            bbox = ann["bbox"]
            orig_id = ann["category_id"]
            
            if orig_id not in orig_to_train:
                continue
            
            # Clamp bbox
            clamped_bbox = clamp_bbox(bbox, img_w, img_h)
            if clamped_bbox is None:
                stats["dropped_boxes"] += 1
                continue
            
            # Convert to YOLO format
            yolo_bbox = hbb_to_yolo(clamped_bbox, img_w, img_h)
            train_id = orig_to_train[orig_id]
            
            # YOLO line
            yolo_lines.append(f"{train_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}")
            
            # COCO annotation
            coco_ann = {
                "id": len(coco_data["annotations"]) + len(coco_annotations) + 1,
                "image_id": coco_data["images"][-1]["id"] if coco_data["images"] else 1,
                "category_id": train_id,
                "bbox": [int(x) for x in clamped_bbox],  # Integer HBB for COCO
                "area": int(clamped_bbox[2] * clamped_bbox[3]),
                "iscrowd": 0
            }
            coco_annotations.append(coco_ann)
        
        # Add COCO image
        coco_img = {
            "id": len(coco_data["images"]) + 1,
            "file_name": img_path.name,
            "width": img_w,
            "height": img_h,
            "file_attributes": data.get("corruption", {})
        }
        coco_data["images"].append(coco_img)
        coco_data["annotations"].extend(coco_annotations)
        
        # Write YOLO label file
        stem = img_path.stem
        split = "train"  # Will be determined by split logic
        label_file = output_dirs["yolo_labels"] / split / f"{stem}.txt"
        label_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        # Copy image to YOLO images
        img_dest = output_dirs["yolo_images"] / split / img_path.name
        img_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, img_dest)
        
        # Create overlay (randomly sample ~20)
        if random.random() < 0.1:  # 10% chance
            overlay_path = overlay_dir / f"{stem}_overlay.jpg"
            create_overlay(img_path, data.get("annotations", []), orig_to_train, class_names, overlay_path)
        
        stats["boxes"] += len(yolo_lines)
        stats["images"] += 1
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to process {img_path}: {e}")
        stats["skipped_pairs"] += 1
        return False


def create_data_yaml(output_dir: Path, class_names: List[str], nc: int):
    """Create YOLO data.yaml file."""
    data_yaml = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/val", 
        "test": "images/test",
        "nc": nc,
        "names": class_names
    }
    
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(f"path: {data_yaml['path']}\n")
        f.write(f"train: {data_yaml['train']}\n")
        f.write(f"val: {data_yaml['val']}\n")
        f.write(f"test: {data_yaml['test']}\n")
        f.write(f"nc: {data_yaml['nc']}\n")
        f.write(f"names: {data_yaml['names']}\n")


def create_cat_map(orig_to_name: Dict[int, str], orig_to_train: Dict[int, int], 
                   class_names: List[str], stats: Dict, output_path: Path):
    """Create category mapping JSON file."""
    cat_map = {
        "orig_to_train": {str(k): v for k, v in orig_to_train.items()},
        "names": {str(i): name for i, name in enumerate(class_names)},
        "stats": stats
    }
    
    with open(output_path, 'w') as f:
        json.dump(cat_map, f, indent=2)


def create_coco_json(coco_data: Dict, class_names: List[str], output_path: Path):
    """Create COCO JSON file."""
    # Add categories
    categories = []
    for i, name in enumerate(class_names):
        categories.append({
            "id": i,
            "name": name,
            "supercategory": "document"
        })
    
    coco_data["categories"] = categories
    coco_data["info"] = {
        "description": "PS5 Document Layout Dataset",
        "version": "1.0",
        "year": 2024
    }
    
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)


def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Parse mappings
    orig_to_name, orig_to_train = parse_id_mapping(args.map)
    class_names = ["Text", "Title", "List", "Table", "Figure"]
    
    # Setup paths
    src_dir = Path(args.src)
    out_root = Path(args.out)
    
    # Create output directories
    output_dirs = {
        "yolo_images": out_root / "yolo" / "images",
        "yolo_labels": out_root / "yolo" / "labels", 
        "coco": out_root / "coco",
        "debug": out_root / "debug" / "overlays"
    }
    
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Resume logic
    done_file = out_root / ".done"
    processed_stems = set()
    if args.resume and done_file.exists():
        with open(done_file, 'r') as f:
            processed_stems = set(line.strip() for line in f)
        print(f"[INFO] Resuming: {len(processed_stems)} files already processed")
    
    # Find image-JSON pairs
    pairs = find_image_json_pairs(src_dir)
    print(f"[INFO] Found {len(pairs)} image-JSON pairs")
    
    # Filter out already processed
    if args.resume:
        pairs = [(img, json) for img, json in pairs if img.stem not in processed_stems]
        print(f"[INFO] {len(pairs)} pairs remaining to process")
    
    # Initialize stats and COCO data
    stats = {"images": 0, "boxes": 0, "dropped_boxes": 0, "skipped_pairs": 0}
    coco_data = {"images": [], "annotations": []}
    
    # Process pairs
    with open(done_file, 'a') as done_f:
        for img_path, json_path in tqdm(pairs, desc="Processing"):
            if args.resume and img_path.stem in processed_stems:
                continue
                
            success = process_image_json_pair(
                img_path, json_path, orig_to_train, class_names, 
                output_dirs, coco_data, stats, output_dirs["debug"]
            )
            
            if success:
                done_f.write(f"{img_path.stem}\n")
                done_f.flush()
    
    # Create splits (simple deterministic split)
    all_images = list((out_root / "yolo" / "images" / "train").glob("*.png"))
    random.shuffle(all_images)
    
    n_train = int(len(all_images) * args.split[0])
    n_val = int(len(all_images) * args.split[1])
    
    # Move images to correct splits
    for i, img_path in enumerate(all_images):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        
        # Move image
        new_img_path = out_root / "yolo" / "images" / split / img_path.name
        new_img_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img_path), str(new_img_path))
        
        # Move corresponding label
        label_path = out_root / "yolo" / "labels" / "train" / f"{img_path.stem}.txt"
        if label_path.exists():
            new_label_path = out_root / "yolo" / "labels" / split / f"{img_path.stem}.txt"
            new_label_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(label_path), str(new_label_path))
    
    # Create output files
    create_data_yaml(out_root / "yolo", class_names, len(class_names))
    create_cat_map(orig_to_name, orig_to_train, class_names, stats, out_root / "cat_map.json")
    create_coco_json(coco_data, class_names, out_root / "coco" / "train.json")
    
    # Create val COCO (subset of train for now)
    val_coco = {"images": [], "annotations": []}
    val_images = list((out_root / "yolo" / "images" / "val").glob("*.png"))
    val_img_names = {img.name for img in val_images}
    
    for img in coco_data["images"]:
        if img["file_name"] in val_img_names:
            val_coco["images"].append(img)
    
    # Filter annotations for val images
    val_img_ids = {img["id"] for img in val_coco["images"]}
    for ann in coco_data["annotations"]:
        if ann["image_id"] in val_img_ids:
            val_coco["annotations"].append(ann)
    
    create_coco_json(val_coco, class_names, out_root / "coco" / "val.json")
    
    # Print summary
    print(f"\n[SUMMARY]")
    print(f"Images processed: {stats['images']}")
    print(f"Boxes: {stats['boxes']}")
    print(f"Dropped boxes: {stats['dropped_boxes']}")
    print(f"Skipped pairs: {stats['skipped_pairs']}")
    print(f"Class mapping: {orig_to_train}")
    print(f"\nOutput files:")
    print(f"  data.yaml: {out_root / 'yolo' / 'data.yaml'}")
    print(f"  cat_map.json: {out_root / 'cat_map.json'}")
    print(f"  COCO train: {out_root / 'coco' / 'train.json'}")
    print(f"  COCO val: {out_root / 'coco' / 'val.json'}")
    print(f"  Overlays: {output_dirs['debug']}")


if __name__ == "__main__":
    main()
