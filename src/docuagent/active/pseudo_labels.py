"""Semi-supervised pseudo-labeling for active learning."""

import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import hashlib
import time

from ..data.converters import coco_to_yolo


@dataclass
class PseudoLabel:
    """Represents a pseudo-label with provenance information."""
    image_id: str
    page: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    source_model: str
    epoch: int
    threshold: float
    timestamp: str


def load_predictions(pred_path: str) -> Dict[str, Any]:
    """Load predictions from COCO format JSON."""
    with open(pred_path, 'r') as f:
        return json.load(f)


def validate_bbox(bbox: List[float], image_width: int, image_height: int) -> Tuple[int, int, int, int]:
    """Validate and clamp bounding box to image bounds.
    
    Args:
        bbox: [x, y, w, h] in COCO format
        image_width: Image width
        image_height: Image height
        
    Returns:
        Validated bbox as (x, y, w, h)
    """
    x, y, w, h = bbox
    
    # Clamp to image bounds
    x = max(0, min(int(x), image_width - 1))
    y = max(0, min(int(y), image_height - 1))
    w = min(int(w), image_width - x)
    h = min(int(h), image_height - y)
    
    # Ensure positive dimensions
    w = max(1, w)
    h = max(1, h)
    
    return (x, y, w, h)


def create_pseudo_labels(
    predictions_path: str,
    images_dir: str,
    output_dir: str,
    confidence_threshold: float = 0.6,
    source_model: str = "unknown",
    epoch: int = 0
) -> List[PseudoLabel]:
    """Create pseudo-labels from high-confidence predictions.
    
    Args:
        predictions_path: Path to predictions COCO JSON
        images_dir: Directory containing images
        output_dir: Directory to save pseudo-labels
        confidence_threshold: Minimum confidence for pseudo-labels
        source_model: Name/hash of source model
        epoch: Training epoch when predictions were made
        
    Returns:
        List of PseudoLabel objects
    """
    print(f"[INFO] Creating pseudo-labels with threshold {confidence_threshold}")
    
    # Load predictions
    predictions = load_predictions(predictions_path)
    
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    images_out = output_path / "images"
    labels_out = output_path / "labels"
    images_out.mkdir(exist_ok=True)
    labels_out.mkdir(exist_ok=True)
    
    # Create image ID to filename mapping
    image_id_to_filename = {img['id']: img['file_name'] for img in predictions['images']}
    image_id_to_info = {img['id']: img for img in predictions['images']}
    category_id_to_name = {cat['id']: cat['name'] for cat in predictions['categories']}
    
    pseudo_labels = []
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Process each prediction
    for ann in predictions['annotations']:
        confidence = ann.get('score', 0.0)
        
        # Skip low confidence predictions
        if confidence < confidence_threshold:
            continue
        
        img_id = ann['image_id']
        img_info = image_id_to_info[img_id]
        filename = image_id_to_filename[img_id]
        class_name = category_id_to_name[ann['category_id']]
        
        # Validate bbox
        bbox = validate_bbox(
            ann['bbox'], 
            img_info['width'], 
            img_info['height']
        )
        
        # Create pseudo-label
        pseudo_label = PseudoLabel(
            image_id=str(img_id),
            page=img_id,  # Assuming image ID corresponds to page number
            class_name=class_name,
            bbox=bbox,
            confidence=confidence,
            source_model=source_model,
            epoch=epoch,
            threshold=confidence_threshold,
            timestamp=timestamp
        )
        pseudo_labels.append(pseudo_label)
    
    print(f"[INFO] Created {len(pseudo_labels)} pseudo-labels from {len(predictions['annotations'])} predictions")
    
    # Save pseudo-labels in YOLO format
    save_yolo_labels(pseudo_labels, images_dir, images_out, labels_out)
    
    # Save manifest
    save_manifest(pseudo_labels, output_path, source_model, epoch, confidence_threshold)
    
    return pseudo_labels


def save_yolo_labels(
    pseudo_labels: List[PseudoLabel],
    images_dir: str,
    images_out: Path,
    labels_out: Path
) -> None:
    """Save pseudo-labels in YOLO format."""
    # Group labels by image
    labels_by_image = {}
    for label in pseudo_labels:
        img_id = label.image_id
        if img_id not in labels_by_image:
            labels_by_image[img_id] = []
        labels_by_image[img_id].append(label)
    
    # Create class mapping
    classes = sorted(set(label.class_name for label in pseudo_labels))
    class_to_id = {cls: i for i, cls in enumerate(classes)}
    
    # Process each image
    for img_id, labels in labels_by_image.items():
        # Find corresponding image file
        image_files = list(Path(images_dir).glob(f"*{img_id}*"))
        if not image_files:
            print(f"Warning: No image found for ID {img_id}")
            continue
        
        image_file = image_files[0]
        
        # Copy image
        image_dest = images_out / image_file.name
        shutil.copy2(image_file, image_dest)
        
        # Create YOLO label file
        label_file = labels_out / f"{image_file.stem}.txt"
        
        with open(label_file, 'w') as f:
            for label in labels:
                # Convert COCO bbox to YOLO format
                x, y, w, h = label.bbox
                
                # Get image dimensions (assume from first label's metadata)
                # In practice, you'd load the actual image
                img_width = 1000  # Default assumption
                img_height = 1000  # Default assumption
                
                # Convert to YOLO format (normalized center coordinates)
                center_x = (x + w / 2) / img_width
                center_y = (y + h / 2) / img_height
                norm_w = w / img_width
                norm_h = h / img_height
                
                class_id = class_to_id[label.class_name]
                
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")


def save_manifest(
    pseudo_labels: List[PseudoLabel],
    output_path: Path,
    source_model: str,
    epoch: int,
    threshold: float
) -> None:
    """Save manifest with provenance information."""
    manifest = {
        'source_model': source_model,
        'epoch': epoch,
        'threshold': threshold,
        'total_labels': len(pseudo_labels),
        'created_at': pseudo_labels[0].timestamp if pseudo_labels else time.strftime("%Y-%m-%d %H:%M:%S"),
        'classes': sorted(set(label.class_name for label in pseudo_labels)),
        'confidence_stats': {
            'min': min(label.confidence for label in pseudo_labels) if pseudo_labels else 0,
            'max': max(label.confidence for label in pseudo_labels) if pseudo_labels else 0,
            'mean': sum(label.confidence for label in pseudo_labels) / len(pseudo_labels) if pseudo_labels else 0
        },
        'labels': [
            {
                'image_id': label.image_id,
                'page': label.page,
                'class_name': label.class_name,
                'bbox': list(label.bbox),
                'confidence': label.confidence,
                'source_model': label.source_model,
                'epoch': label.epoch,
                'threshold': label.threshold,
                'timestamp': label.timestamp
            }
            for label in pseudo_labels
        ]
    }
    
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"[INFO] Saved manifest to {manifest_path}")


def merge_with_existing_dataset(
    pseudo_labels_dir: str,
    existing_dataset_dir: str,
    output_dir: str
) -> None:
    """Merge pseudo-labels with existing dataset.
    
    Args:
        pseudo_labels_dir: Directory containing pseudo-labels
        existing_dataset_dir: Directory containing existing dataset
        output_dir: Directory to save merged dataset
    """
    print(f"[INFO] Merging pseudo-labels with existing dataset")
    
    pseudo_path = Path(pseudo_labels_dir)
    existing_path = Path(existing_dataset_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "labels").mkdir(parents=True, exist_ok=True)
    
    # Copy existing dataset
    if existing_path.exists():
        for subdir in ["images", "labels"]:
            src = existing_path / subdir
            dst = output_path / subdir
            if src.exists():
                shutil.copytree(src, dst, dirs_exist_ok=True)
    
    # Copy pseudo-labels
    for subdir in ["images", "labels"]:
        src = pseudo_path / subdir
        dst = output_path / subdir
        if src.exists():
            for file in src.iterdir():
                if file.is_file():
                    shutil.copy2(file, dst / file.name)
    
    # Merge manifests
    existing_manifest = existing_path / "manifest.json"
    pseudo_manifest = pseudo_path / "manifest.json"
    output_manifest = output_path / "manifest.json"
    
    merged_manifest = {
        'merged_at': time.strftime("%Y-%m-%d %H:%M:%S"),
        'sources': []
    }
    
    if existing_manifest.exists():
        with open(existing_manifest, 'r') as f:
            existing_data = json.load(f)
        merged_manifest['sources'].append(existing_data)
    
    if pseudo_manifest.exists():
        with open(pseudo_manifest, 'r') as f:
            pseudo_data = json.load(f)
        merged_manifest['sources'].append(pseudo_data)
    
    with open(output_manifest, 'w') as f:
        json.dump(merged_manifest, f, indent=2)
    
    print(f"[INFO] Merged dataset saved to {output_path}")


def main():
    """Command line interface for pseudo-labeling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate pseudo-labels from predictions")
    parser.add_argument("--pred", required=True, help="Predictions COCO JSON file")
    parser.add_argument("--images", required=True, help="Images directory")
    parser.add_argument("--out", required=True, help="Output directory for pseudo-labels")
    parser.add_argument("--thr", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--model", default="unknown", help="Source model name/hash")
    parser.add_argument("--epoch", type=int, default=0, help="Training epoch")
    
    args = parser.parse_args()
    
    pseudo_labels = create_pseudo_labels(
        args.pred, args.images, args.out, args.thr, args.model, args.epoch
    )
    
    print(f"[INFO] Generated {len(pseudo_labels)} pseudo-labels")


if __name__ == "__main__":
    main()
