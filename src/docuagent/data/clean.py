"""
Data cleaning and fixing module.

This module provides automatic fixing of common data issues:
- Broken images and labels
- Invalid coordinates
- Filename normalization
- Data validation
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm


def apply_auto_fixes(
    audit_result: Dict[str, Any],
    images_dir: str,
    labels: str,
    schema: str,
    out_dir: str
) -> Dict[str, Any]:
    """
    Apply automatic fixes based on audit results.
    
    Args:
        audit_result: Results from dataset audit
        images_dir: Path to images directory
        labels: Path to labels (YOLO dir or COCO json)
        schema: "yolo" or "coco"
        out_dir: Output directory for cleaned data
        
    Returns:
        Dict with cleaning results and changelog
    """
    images_path = Path(images_dir)
    labels_path = Path(labels)
    output_path = Path(out_dir)
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    quarantine_dir = output_path / "_quarantine"
    quarantine_dir.mkdir(exist_ok=True)
    
    # Initialize results
    results = {
        "cleaned_images": 0,
        "cleaned_labels": 0,
        "quarantined_files": [],
        "fixes_applied": [],
        "changelog": []
    }
    
    print(f"[CLEAN] Applying fixes to {images_dir}")
    print(f"[CLEAN] Output directory: {out_dir}")
    
    # 1. Clean images
    print("[CLEAN] Cleaning images...")
    image_results = _clean_images(images_path, output_path, quarantine_dir, audit_result)
    results.update(image_results)
    
    # 2. Clean labels
    print("[CLEAN] Cleaning labels...")
    if schema == "yolo":
        label_results = _clean_yolo_labels(labels_path, output_path, quarantine_dir, audit_result)
    else:
        label_results = _clean_coco_labels(labels_path, output_path, quarantine_dir, audit_result)
    results.update(label_results)
    
    # 3. Normalize filenames
    print("[CLEAN] Normalizing filenames...")
    filename_results = _normalize_filenames(output_path, audit_result)
    results.update(filename_results)
    
    # 4. Generate changelog
    print("[CLEAN] Generating changelog...")
    changelog = _generate_changelog(results)
    results["changelog"] = changelog
    
    # Save changelog
    with open(output_path / "CHANGELOG.md", 'w') as f:
        f.write(changelog)
    
    print(f"[CLEAN] Cleaning complete. Results saved to {out_dir}")
    return results


def _clean_images(
    images_path: Path,
    output_path: Path,
    quarantine_dir: Path,
    audit_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Clean image files."""
    results = {
        "cleaned_images": 0,
        "quarantined_files": [],
        "fixes_applied": []
    }
    
    # Get list of images to process
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.jpeg")) + list(images_path.glob("*.png"))
    
    # Get problematic images from audit
    problematic_images = set()
    for issue_type in ["unreadable", "extreme_aspect_ratio", "tiny_dims", "corrupt_exif"]:
        if issue_type in audit_result.get("image_sanity", {}):
            for item in audit_result["image_sanity"][issue_type]:
                if isinstance(item, dict) and "path" in item:
                    problematic_images.add(item["path"])
                elif isinstance(item, str):
                    problematic_images.add(item)
    
    # Process each image
    for img_path in tqdm(image_files, desc="Cleaning images"):
        try:
            # Check if image is problematic
            if str(img_path) in problematic_images:
                # Try to fix the image
                fixed = _fix_image(img_path, output_path)
                if fixed:
                    results["fixes_applied"].append(f"Fixed image: {img_path.name}")
                    results["cleaned_images"] += 1
                else:
                    # Move to quarantine
                    quarantine_path = quarantine_dir / img_path.name
                    shutil.move(str(img_path), str(quarantine_path))
                    results["quarantined_files"].append(str(quarantine_path))
                    results["fixes_applied"].append(f"Quarantined broken image: {img_path.name}")
            else:
                # Copy good image
                output_img_path = output_path / img_path.name
                shutil.copy2(str(img_path), str(output_img_path))
                results["cleaned_images"] += 1
        
        except Exception as e:
            print(f"[WARN] Error processing {img_path}: {e}")
            # Move to quarantine
            try:
                quarantine_path = quarantine_dir / img_path.name
                shutil.move(str(img_path), str(quarantine_path))
                results["quarantined_files"].append(str(quarantine_path))
            except Exception:
                pass
    
    return results


def _fix_image(img_path: Path, output_path: Path) -> bool:
    """Try to fix a problematic image."""
    try:
        # Try to read image
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        
        # Fix common issues
        h, w = img.shape[:2]
        
        # Fix extreme aspect ratios
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 10:
            # Resize to reasonable aspect ratio
            if w > h:
                new_w = min(w, h * 5)
                new_h = h
            else:
                new_w = w
                new_h = min(h, w * 5)
            
            img = cv2.resize(img, (new_w, new_h))
        
        # Fix tiny dimensions
        if min(h, w) < 400:
            scale_factor = 400 / min(h, w)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            img = cv2.resize(img, (new_w, new_h))
        
        # Fix EXIF issues by re-encoding
        output_img_path = output_path / img_path.name
        success = cv2.imwrite(str(output_img_path), img)
        
        return success
    
    except Exception as e:
        print(f"[WARN] Could not fix image {img_path}: {e}")
        return False


def _clean_yolo_labels(
    labels_path: Path,
    output_path: Path,
    quarantine_dir: Path,
    audit_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Clean YOLO label files."""
    results = {
        "cleaned_labels": 0,
        "quarantined_files": [],
        "fixes_applied": []
    }
    
    # Create labels output directory
    labels_output_dir = output_path / "labels"
    labels_output_dir.mkdir(exist_ok=True)
    
    # Get problematic labels from audit
    problematic_labels = set()
    label_sanity = audit_result.get("label_sanity", {})
    for issue_type in ["invalid_coords", "out_of_bounds", "unknown_classes", "empty_labels"]:
        if issue_type in label_sanity:
            for item in label_sanity[issue_type]:
                if isinstance(item, dict) and "file" in item:
                    problematic_labels.add(item["file"])
                elif isinstance(item, str):
                    problematic_labels.add(item)
    
    # Process each label file
    for label_file in tqdm(labels_path.glob("*.txt"), desc="Cleaning YOLO labels"):
        try:
            # Check if label is problematic
            if str(label_file) in problematic_labels:
                # Try to fix the label
                fixed = _fix_yolo_label(label_file, labels_output_dir)
                if fixed:
                    results["fixes_applied"].append(f"Fixed label: {label_file.name}")
                    results["cleaned_labels"] += 1
                else:
                    # Move to quarantine
                    quarantine_path = quarantine_dir / label_file.name
                    shutil.move(str(label_file), str(quarantine_path))
                    results["quarantined_files"].append(str(quarantine_path))
                    results["fixes_applied"].append(f"Quarantined broken label: {label_file.name}")
            else:
                # Copy good label
                output_label_path = labels_output_dir / label_file.name
                shutil.copy2(str(label_file), str(output_label_path))
                results["cleaned_labels"] += 1
        
        except Exception as e:
            print(f"[WARN] Error processing {label_file}: {e}")
            # Move to quarantine
            try:
                quarantine_path = quarantine_dir / label_file.name
                shutil.move(str(label_file), str(quarantine_path))
                results["quarantined_files"].append(str(quarantine_path))
            except Exception:
                pass
    
    return results


def _fix_yolo_label(label_file: Path, output_dir: Path) -> bool:
    """Try to fix a problematic YOLO label file."""
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return False
        
        fixed_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                
                # Fix coordinates
                x = max(0, min(1.0, x))
                y = max(0, min(1.0, y))
                w = max(0.001, min(1.0, w))
                h = max(0.001, min(1.0, h))
                
                # Ensure bbox is within bounds
                if x + w > 1.0:
                    w = 1.0 - x
                if y + h > 1.0:
                    h = 1.0 - y
                
                # Only keep valid bboxes
                if w > 0.001 and h > 0.001:
                    fixed_lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            
            except ValueError:
                continue
        
        if fixed_lines:
            output_path = output_dir / label_file.name
            with open(output_path, 'w') as f:
                f.writelines(fixed_lines)
            return True
        
        return False
    
    except Exception as e:
        print(f"[WARN] Could not fix label {label_file}: {e}")
        return False


def _clean_coco_labels(
    labels_path: Path,
    output_path: Path,
    quarantine_dir: Path,
    audit_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Clean COCO label file."""
    results = {
        "cleaned_labels": 0,
        "quarantined_files": [],
        "fixes_applied": []
    }
    
    try:
        with open(labels_path, 'r') as f:
            coco_data = json.load(f)
        
        # Fix annotations
        fixed_annotations = []
        for ann in coco_data.get("annotations", []):
            bbox = ann.get("bbox", [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                
                # Fix coordinates
                x = max(0, x)
                y = max(0, y)
                w = max(1, w)
                h = max(1, h)
                
                # Ensure bbox is within bounds (assuming image size)
                # This is a simplified check - in practice, you'd need image dimensions
                if x + w > 10000:  # Reasonable upper bound
                    w = 10000 - x
                if y + h > 10000:
                    h = 10000 - y
                
                # Only keep valid bboxes
                if w > 0 and h > 0:
                    ann["bbox"] = [x, y, w, h]
                    fixed_annotations.append(ann)
        
        # Update annotations
        coco_data["annotations"] = fixed_annotations
        
        # Save fixed file
        output_file = output_path / labels_path.name
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        results["cleaned_labels"] = 1
        results["fixes_applied"].append(f"Fixed COCO labels: {labels_path.name}")
        
    except Exception as e:
        print(f"[WARN] Could not fix COCO labels {labels_path}: {e}")
        # Move to quarantine
        try:
            quarantine_path = quarantine_dir / labels_path.name
            shutil.move(str(labels_path), str(quarantine_path))
            results["quarantined_files"].append(str(quarantine_path))
        except Exception:
            pass
    
    return results


def _normalize_filenames(
    output_path: Path,
    audit_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Normalize filenames based on audit results."""
    results = {
        "normalized_files": 0,
        "fixes_applied": []
    }
    
    # Get filename suggestions from audit
    filename_hygiene = audit_result.get("filename_hygiene", {})
    suggestions = filename_hygiene.get("suggestions", [])
    
    if not suggestions:
        return results
    
    # Apply filename normalization
    for suggestion in suggestions:
        try:
            original_path = Path(suggestion["original"])
            suggested_path = Path(suggestion["suggested"])
            
            # Check if original file exists in output directory
            output_original = output_path / original_path.name
            if output_original.exists():
                # Rename file
                output_suggested = output_path / suggested_path.name
                output_original.rename(output_suggested)
                results["normalized_files"] += 1
                results["fixes_applied"].append(f"Renamed: {original_path.name} -> {suggested_path.name}")
                
                # Update corresponding label file if it exists
                label_file = output_path / "labels" / original_path.with_suffix(".txt").name
                if label_file.exists():
                    label_suggested = output_path / "labels" / suggested_path.with_suffix(".txt").name
                    label_file.rename(label_suggested)
                    results["fixes_applied"].append(f"Renamed label: {label_file.name} -> {label_suggested.name}")
        
        except Exception as e:
            print(f"[WARN] Could not normalize filename {suggestion['original']}: {e}")
    
    return results


def _generate_changelog(results: Dict[str, Any]) -> str:
    """Generate a changelog of applied fixes."""
    changelog_lines = [
        "# Data Cleaning Changelog",
        "",
        f"**Cleaned images:** {results['cleaned_images']}",
        f"**Cleaned labels:** {results['cleaned_labels']}",
        f"**Quarantined files:** {len(results['quarantined_files'])}",
        f"**Total fixes applied:** {len(results['fixes_applied'])}",
        "",
        "## Fixes Applied",
        ""
    ]
    
    # Group fixes by type
    fix_groups = {
        "Image fixes": [],
        "Label fixes": [],
        "Filename fixes": [],
        "Quarantined files": []
    }
    
    for fix in results["fixes_applied"]:
        if "image" in fix.lower():
            fix_groups["Image fixes"].append(fix)
        elif "label" in fix.lower():
            fix_groups["Label fixes"].append(fix)
        elif "rename" in fix.lower():
            fix_groups["Filename fixes"].append(fix)
        else:
            fix_groups["Image fixes"].append(fix)
    
    # Add quarantined files
    for file_path in results["quarantined_files"]:
        fix_groups["Quarantined files"].append(f"Quarantined: {Path(file_path).name}")
    
    # Write changelog
    for group_name, fixes in fix_groups.items():
        if fixes:
            changelog_lines.extend([
                f"### {group_name}",
                ""
            ])
            for fix in fixes:
                changelog_lines.append(f"- {fix}")
            changelog_lines.append("")
    
    return "\n".join(changelog_lines)


def validate_cleaned_data(
    cleaned_dir: str,
    schema: str,
    class_names: List[str] = None
) -> Dict[str, Any]:
    """
    Validate cleaned data for remaining issues.
    
    Args:
        cleaned_dir: Path to cleaned data directory
        schema: "yolo" or "coco"
        class_names: List of class names
        
    Returns:
        Dict with validation results
    """
    if class_names is None:
        class_names = ["Text", "Title", "List", "Table", "Figure"]
    
    cleaned_path = Path(cleaned_dir)
    if not cleaned_path.exists():
        raise ValueError(f"Cleaned directory not found: {cleaned_dir}")
    
    results = {
        "validation_passed": True,
        "issues_found": [],
        "summary": {
            "total_images": 0,
            "total_labels": 0,
            "valid_images": 0,
            "valid_labels": 0
        }
    }
    
    # Validate images
    image_files = list(cleaned_path.glob("*.jpg")) + list(cleaned_path.glob("*.jpeg")) + list(cleaned_path.glob("*.png"))
    results["summary"]["total_images"] = len(image_files)
    
    for img_path in image_files:
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                results["summary"]["valid_images"] += 1
            else:
                results["issues_found"].append(f"Invalid image: {img_path.name}")
                results["validation_passed"] = False
        except Exception as e:
            results["issues_found"].append(f"Error reading image {img_path.name}: {e}")
            results["validation_passed"] = False
    
    # Validate labels
    if schema == "yolo":
        labels_dir = cleaned_path / "labels"
        if labels_dir.exists():
            label_files = list(labels_dir.glob("*.txt"))
            results["summary"]["total_labels"] = len(label_files)
            
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    valid_lines = 0
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            try:
                                class_id = int(parts[0])
                                x, y, w, h = map(float, parts[1:5])
                                
                                if (0 <= class_id < len(class_names) and
                                    0 <= x <= 1 and 0 <= y <= 1 and
                                    0 < w <= 1 and 0 < h <= 1 and
                                    x + w <= 1 and y + h <= 1):
                                    valid_lines += 1
                            except ValueError:
                                continue
                    
                    if valid_lines > 0:
                        results["summary"]["valid_labels"] += 1
                    else:
                        results["issues_found"].append(f"Empty or invalid label: {label_file.name}")
                        results["validation_passed"] = False
                
                except Exception as e:
                    results["issues_found"].append(f"Error reading label {label_file.name}: {e}")
                    results["validation_passed"] = False
    
    return results
