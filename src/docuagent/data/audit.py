"""
Dataset audit module for pre-upload data quality checks.

This module provides comprehensive auditing of training datasets including:
- Image and label counts
- Image quality checks
- Duplicate detection
- Label validation
- Split leakage detection
- Class imbalance analysis
- Filename hygiene
"""

import os
import json
import csv
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
import cv2
import numpy as np
from PIL import Image, ExifTags
import imagehash
from tqdm import tqdm

from ..config import load_config


def audit_dataset(
    images_dir: str,
    labels: str,
    schema: str = "yolo",
    class_names: List[str] = None,
    fix: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive dataset audit with optional auto-fixing.
    
    Args:
        images_dir: Path to images directory
        labels: Path to labels (YOLO dir or COCO json)
        schema: "yolo" or "coco"
        class_names: List of class names
        fix: Whether to auto-fix issues
        
    Returns:
        Dict with audit results and statistics
    """
    images_path = Path(images_dir)
    labels_path = Path(labels)
    
    if not images_path.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    
    if not labels_path.exists():
        raise ValueError(f"Labels path not found: {labels}")
    
    # Initialize results
    results = {
        "counts": {},
        "image_sanity": {},
        "duplicates": [],
        "label_sanity": {},
        "split_leakage": {},
        "imbalance": {},
        "filename_hygiene": {},
        "fixes_applied": []
    }
    
    # Load class names if not provided
    if class_names is None:
        class_names = ["Text", "Title", "List", "Table", "Figure"]
    
    # 1. Count images and labels
    print("[AUDIT] Counting images and labels...")
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.jpeg")) + list(images_path.glob("*.png"))
    results["counts"]["images"] = len(image_files)
    
    if schema == "yolo":
        label_files = list(labels_path.glob("*.txt"))
        results["counts"]["labels"] = len(label_files)
    else:  # COCO
        with open(labels_path, 'r') as f:
            coco_data = json.load(f)
        results["counts"]["labels"] = len(coco_data.get("annotations", []))
    
    # 2. Image sanity checks
    print("[AUDIT] Checking image sanity...")
    results["image_sanity"] = _check_image_sanity(image_files, fix)
    
    # 3. Duplicate detection
    print("[AUDIT] Detecting duplicates...")
    results["duplicates"] = _detect_duplicates(image_files)
    
    # 4. Label sanity checks
    print("[AUDIT] Checking label sanity...")
    if schema == "yolo":
        results["label_sanity"] = _check_yolo_labels(labels_path, class_names, fix)
    else:
        results["label_sanity"] = _check_coco_labels(labels_path, class_names, fix)
    
    # 5. Split leakage detection
    print("[AUDIT] Checking for split leakage...")
    results["split_leakage"] = _check_split_leakage(images_path, labels_path, schema)
    
    # 6. Class imbalance analysis
    print("[AUDIT] Analyzing class imbalance...")
    results["imbalance"] = _analyze_class_imbalance(labels_path, schema, class_names)
    
    # 7. Filename hygiene
    print("[AUDIT] Checking filename hygiene...")
    results["filename_hygiene"] = _check_filename_hygiene(image_files, fix)
    
    return results


def _check_image_sanity(image_files: List[Path], fix: bool) -> Dict[str, Any]:
    """Check image quality and integrity."""
    sanity = {
        "unreadable": [],
        "extreme_aspect_ratio": [],
        "tiny_dims": [],
        "corrupt_exif": [],
        "fixed": []
    }
    
    for img_path in tqdm(image_files, desc="Checking images"):
        try:
            # Try to read image
            img = cv2.imread(str(img_path))
            if img is None:
                sanity["unreadable"].append(str(img_path))
                continue
            
            h, w = img.shape[:2]
            
            # Check dimensions
            if min(h, w) < 400:
                sanity["tiny_dims"].append({
                    "path": str(img_path),
                    "dims": (w, h)
                })
            
            # Check aspect ratio
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 10:  # Extreme aspect ratio
                sanity["extreme_aspect_ratio"].append({
                    "path": str(img_path),
                    "aspect_ratio": aspect_ratio
                })
            
            # Check EXIF
            try:
                with Image.open(img_path) as pil_img:
                    exif = pil_img._getexif()
                    if exif is not None:
                        # Check for common EXIF corruption
                        for tag_id, value in exif.items():
                            if tag_id in ExifTags.TAGS:
                                tag = ExifTags.TAGS[tag_id]
                                if isinstance(value, bytes) and len(value) > 10000:
                                    sanity["corrupt_exif"].append(str(img_path))
                                    break
            except Exception:
                sanity["corrupt_exif"].append(str(img_path))
                
        except Exception as e:
            sanity["unreadable"].append({
                "path": str(img_path),
                "error": str(e)
            })
    
    return sanity


def _detect_duplicates(image_files: List[Path]) -> List[Dict[str, Any]]:
    """Detect near-duplicate images using perceptual hashing."""
    duplicates = []
    hashes = {}
    
    for img_path in tqdm(image_files, desc="Computing hashes"):
        try:
            img = Image.open(img_path)
            img_hash = imagehash.phash(img)
            hashes[str(img_path)] = img_hash
        except Exception:
            continue
    
    # Find duplicates
    for path1, hash1 in hashes.items():
        for path2, hash2 in hashes.items():
            if path1 >= path2:  # Avoid duplicates and self-comparison
                continue
            
            hamming_dist = hash1 - hash2
            if hamming_dist <= 8:  # Near-duplicate threshold
                duplicates.append({
                    "path1": path1,
                    "path2": path2,
                    "hamming_distance": hamming_dist
                })
    
    return duplicates


def _check_yolo_labels(labels_path: Path, class_names: List[str], fix: bool) -> Dict[str, Any]:
    """Check YOLO label files for sanity."""
    sanity = {
        "invalid_coords": [],
        "out_of_bounds": [],
        "unknown_classes": [],
        "empty_labels": [],
        "fixed": []
    }
    
    class_to_id = {name: i for i, name in enumerate(class_names)}
    
    for label_file in tqdm(labels_path.glob("*.txt"), desc="Checking YOLO labels"):
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                sanity["empty_labels"].append(str(label_file))
                continue
            
            for line_num, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    
                    # Check class ID
                    if class_id not in range(len(class_names)):
                        sanity["unknown_classes"].append({
                            "file": str(label_file),
                            "line": line_num + 1,
                            "class_id": class_id
                        })
                        if fix:
                            # Remove invalid class
                            lines[line_num] = ""
                            sanity["fixed"].append(f"Removed invalid class {class_id} from {label_file}")
                    
                    # Check coordinates
                    if x < 0 or y < 0 or w <= 0 or h <= 0:
                        sanity["invalid_coords"].append({
                            "file": str(label_file),
                            "line": line_num + 1,
                            "coords": (x, y, w, h)
                        })
                        if fix:
                            # Clamp coordinates
                            x = max(0, x)
                            y = max(0, y)
                            w = max(0.001, w)
                            h = max(0.001, h)
                            lines[line_num] = f"{class_id} {x} {y} {w} {h}\n"
                            sanity["fixed"].append(f"Fixed coordinates in {label_file}")
                    
                    # Check bounds
                    if x + w > 1.0 or y + h > 1.0:
                        sanity["out_of_bounds"].append({
                            "file": str(label_file),
                            "line": line_num + 1,
                            "coords": (x, y, w, h)
                        })
                        if fix:
                            # Clamp to bounds
                            x = min(x, 1.0 - w)
                            y = min(y, 1.0 - h)
                            lines[line_num] = f"{class_id} {x} {y} {w} {h}\n"
                            sanity["fixed"].append(f"Clamped bounds in {label_file}")
                
                except ValueError:
                    continue
            
            # Write fixed file
            if fix and sanity["fixed"]:
                with open(label_file, 'w') as f:
                    f.writelines([line for line in lines if line.strip()])
                    
        except Exception as e:
            print(f"Error checking {label_file}: {e}")
    
    return sanity


def _check_coco_labels(labels_path: Path, class_names: List[str], fix: bool) -> Dict[str, Any]:
    """Check COCO label file for sanity."""
    sanity = {
        "invalid_coords": [],
        "out_of_bounds": [],
        "unknown_classes": [],
        "empty_labels": [],
        "fixed": []
    }
    
    try:
        with open(labels_path, 'r') as f:
            coco_data = json.load(f)
        
        # Check categories
        category_names = {cat["name"] for cat in coco_data.get("categories", [])}
        for class_name in class_names:
            if class_name not in category_names:
                sanity["unknown_classes"].append({
                    "class": class_name,
                    "reason": "Not in COCO categories"
                })
        
        # Check annotations
        for ann in coco_data.get("annotations", []):
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue
            
            x, y, w, h = bbox
            
            # Check coordinates
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                sanity["invalid_coords"].append({
                    "annotation_id": ann.get("id"),
                    "coords": (x, y, w, h)
                })
                if fix:
                    # Fix coordinates
                    ann["bbox"] = [max(0, x), max(0, y), max(1, w), max(1, h)]
                    sanity["fixed"].append(f"Fixed annotation {ann.get('id')}")
            
            # Check category
            category_id = ann.get("category_id")
            if category_id is None:
                sanity["unknown_classes"].append({
                    "annotation_id": ann.get("id"),
                    "reason": "Missing category_id"
                })
        
        # Write fixed file
        if fix and sanity["fixed"]:
            with open(labels_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
                
    except Exception as e:
        print(f"Error checking COCO labels: {e}")
    
    return sanity


def _check_split_leakage(images_path: Path, labels_path: Path, schema: str) -> Dict[str, Any]:
    """Check for image leakage across train/val/test splits."""
    leakage = {
        "found": False,
        "duplicate_stems": [],
        "splits_checked": []
    }
    
    # Look for split directories
    split_dirs = []
    for split in ["train", "val", "test"]:
        split_dir = images_path.parent / split
        if split_dir.exists():
            split_dirs.append(split)
            leakage["splits_checked"].append(split)
    
    if len(split_dirs) < 2:
        return leakage
    
    # Collect image stems from each split
    split_stems = {}
    for split in split_dirs:
        split_dir = images_path.parent / split
        stems = set()
        for img_file in split_dir.glob("*.jpg"):
            stems.add(img_file.stem)
        for img_file in split_dir.glob("*.png"):
            stems.add(img_file.stem)
        split_stems[split] = stems
    
    # Check for overlaps
    for split1 in split_dirs:
        for split2 in split_dirs:
            if split1 >= split2:
                continue
            
            overlap = split_stems[split1] & split_stems[split2]
            if overlap:
                leakage["found"] = True
                leakage["duplicate_stems"].append({
                    "split1": split1,
                    "split2": split2,
                    "count": len(overlap),
                    "stems": list(overlap)[:10]  # First 10 examples
                })
    
    return leakage


def _analyze_class_imbalance(labels_path: Path, schema: str, class_names: List[str]) -> Dict[str, Any]:
    """Analyze class distribution and imbalance."""
    imbalance = {
        "class_counts": {},
        "class_percentages": {},
        "imbalanced_classes": [],
        "total_boxes": 0
    }
    
    class_counts = Counter()
    
    if schema == "yolo":
        for label_file in labels_path.glob("*.txt"):
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            if 0 <= class_id < len(class_names):
                                class_counts[class_names[class_id]] += 1
            except Exception:
                continue
    else:  # COCO
        try:
            with open(labels_path, 'r') as f:
                coco_data = json.load(f)
            
            # Map category IDs to names
            cat_id_to_name = {}
            for cat in coco_data.get("categories", []):
                cat_id_to_name[cat["id"]] = cat["name"]
            
            for ann in coco_data.get("annotations", []):
                cat_id = ann.get("category_id")
                if cat_id in cat_id_to_name:
                    class_counts[cat_id_to_name[cat_id]] += 1
        except Exception:
            pass
    
    total_boxes = sum(class_counts.values())
    imbalance["total_boxes"] = total_boxes
    
    for class_name in class_names:
        count = class_counts.get(class_name, 0)
        percentage = (count / total_boxes * 100) if total_boxes > 0 else 0
        
        imbalance["class_counts"][class_name] = count
        imbalance["class_percentages"][class_name] = percentage
        
        if percentage < 5.0:  # Flag classes with < 5%
            imbalance["imbalanced_classes"].append({
                "class": class_name,
                "count": count,
                "percentage": percentage
            })
    
    return imbalance


def _check_filename_hygiene(image_files: List[Path], fix: bool) -> Dict[str, Any]:
    """Check filename hygiene and suggest fixes."""
    hygiene = {
        "non_ascii": [],
        "spaces": [],
        "uppercase": [],
        "suggestions": [],
        "rename_map": []
    }
    
    for img_path in image_files:
        filename = img_path.name
        issues = []
        
        # Check for non-ASCII characters
        if not filename.isascii():
            issues.append("non_ascii")
            hygiene["non_ascii"].append(str(img_path))
        
        # Check for spaces
        if " " in filename:
            issues.append("spaces")
            hygiene["spaces"].append(str(img_path))
        
        # Check for uppercase
        if filename != filename.lower():
            issues.append("uppercase")
            hygiene["uppercase"].append(str(img_path))
        
        if issues:
            # Generate suggestion
            suggested_name = filename.lower().replace(" ", "_")
            # Remove non-ASCII characters
            suggested_name = "".join(c for c in suggested_name if c.isascii() and c.isalnum() or c in "._-")
            
            hygiene["suggestions"].append({
                "original": str(img_path),
                "suggested": str(img_path.parent / suggested_name),
                "issues": issues
            })
            
            hygiene["rename_map"].append({
                "original": str(img_path),
                "suggested": str(img_path.parent / suggested_name)
            })
    
    return hygiene


def save_audit_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Save audit results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main results
    with open(output_dir / "audit_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save duplicates CSV
    if results["duplicates"]:
        with open(output_dir / "duplicates.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["path1", "path2", "hamming_distance"])
            for dup in results["duplicates"]:
                writer.writerow([dup["path1"], dup["path2"], dup["hamming_distance"]])
    
    # Save rename map CSV
    if results["filename_hygiene"]["rename_map"]:
        with open(output_dir / "rename_map.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["original", "suggested"])
            for rename in results["filename_hygiene"]["rename_map"]:
                writer.writerow([rename["original"], rename["suggested"]])
    
    print(f"[AUDIT] Results saved to {output_dir}")
