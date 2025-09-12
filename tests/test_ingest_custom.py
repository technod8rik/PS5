"""
Tests for custom per-image JSON ingester.
"""

import json
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2
from unittest.mock import patch

from docuagent.data.ingest_custom import ingest_perimage_json


def test_ingest_perimage_json():
    """Test the custom per-image JSON ingester."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        src_dir = Path(temp_dir) / "src"
        src_dir.mkdir()
        
        out_dir = Path(temp_dir) / "out"
        
        # Create 2 fake images and JSONs
        test_data = [
            {
                "image_name": "doc_00000.png",
                "json_name": "doc_00000.json",
                "width": 800,
                "height": 600,
                "annotations": [
                    {"bbox": [100, 200, 300, 150], "category_id": 1, "category_name": "Text"},
                    {"bbox": [50, 50, 400, 100], "category_id": 2, "category_name": "Title"}
                ]
            },
            {
                "image_name": "doc_00001.png", 
                "json_name": "doc_00001.json",
                "width": 1000,
                "height": 800,
                "annotations": [
                    {"bbox": [200, 300, 250, 200], "category_id": 1, "category_name": "Text"},
                    {"bbox": [150, 100, 500, 80], "category_id": 3, "category_name": "List"}
                ]
            }
        ]
        
        # Create images and JSON files
        for data in test_data:
            # Create fake image
            img = np.random.randint(0, 255, (data["height"], data["width"], 3), dtype=np.uint8)
            img_path = src_dir / data["image_name"]
            cv2.imwrite(str(img_path), img)
            
            # Create JSON file
            json_data = {
                "file_name": data["image_name"],
                "annotations": data["annotations"],
                "corruption": {"type": "none", "severity": 0}
            }
            json_path = src_dir / data["json_name"]
            with open(json_path, 'w') as f:
                json.dump(json_data, f)
        
        # Run ingestion
        results = ingest_perimage_json(
            src_dir=str(src_dir),
            out_root=str(out_dir),
            split=(0.5, 0.5, 0.0),  # 50/50 split
            id_base="auto",
            seed=42
        )
        
        # Assertions
        assert results["images_processed"] == 2
        assert results["total_boxes"] == 4
        assert results["dropped_boxes"] == 0
        assert len(results["class_names"]) == 3  # IDs 1, 2, 3 -> 0, 1, 2
        assert results["id_mapping"] == {1: 0, 2: 1, 3: 2}
        assert results["class_names"] == ["class_1", "class_2", "class_3"]
        
        # Check YOLO output
        yolo_path = Path(results["yolo_path"])
        assert yolo_path.exists()
        assert (yolo_path / "data.yaml").exists()
        assert (yolo_path / "images" / "train").exists()
        assert (yolo_path / "images" / "val").exists()
        assert (yolo_path / "labels" / "train").exists()
        assert (yolo_path / "labels" / "val").exists()
        
        # Check data.yaml
        with open(yolo_path / "data.yaml", 'r') as f:
            data_yaml = f.read()
            assert "nc: 3" in data_yaml
            assert "names:" in data_yaml
        
        # Check YOLO label files
        label_files = list((yolo_path / "labels" / "train").glob("*.txt")) + list((yolo_path / "labels" / "val").glob("*.txt"))
        assert len(label_files) == 2
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    assert len(parts) == 5
                    cls, xc, yc, w, h = parts
                    assert 0 <= int(cls) < 3
                    assert 0 <= float(xc) <= 1
                    assert 0 <= float(yc) <= 1
                    assert 0 <= float(w) <= 1
                    assert 0 <= float(h) <= 1
        
        # Check COCO output
        coco_path = Path(results["coco_path"])
        assert coco_path.exists()
        assert (coco_path / "train.json").exists()
        assert (coco_path / "val.json").exists()
        
        # Check category mapping
        cat_map_path = Path(results["cat_map"])
        assert cat_map_path.exists()
        with open(cat_map_path, 'r') as f:
            cat_map = json.load(f)
            assert cat_map["orig_to_train"] == {"1": 0, "2": 1, "3": 2}
            assert cat_map["stats"]["images"] == 2
            assert cat_map["stats"]["boxes"] == 4
        
        # Check debug overlays
        debug_path = Path(results["debug_overlays"])
        assert debug_path.exists()
        overlay_files = list(debug_path.glob("*.png"))
        assert len(overlay_files) == 2
        
        print("✅ All tests passed!")


def test_id_remapping():
    """Test ID remapping with different scenarios."""
    with tempfile.TemporaryDirectory() as temp_dir:
        src_dir = Path(temp_dir) / "src"
        src_dir.mkdir()
        out_dir = Path(temp_dir) / "out"
        
        # Test with IDs starting from 0
        test_data = {
            "image_name": "test.png",
            "json_name": "test.json", 
            "width": 400,
            "height": 300,
            "annotations": [
                {"bbox": [50, 50, 100, 80], "category_id": 0, "category_name": "Class0"},
                {"bbox": [200, 100, 150, 120], "category_id": 2, "category_name": "Class2"}
            ]
        }
        
        # Create test files
        img = np.random.randint(0, 255, (test_data["height"], test_data["width"], 3), dtype=np.uint8)
        cv2.imwrite(str(src_dir / test_data["image_name"]), img)
        
        json_data = {
            "file_name": test_data["image_name"],
            "annotations": test_data["annotations"],
            "corruption": {"type": "none", "severity": 0}
        }
        with open(src_dir / test_data["json_name"], 'w') as f:
            json.dump(json_data, f)
        
        # Test with id_base="auto"
        results = ingest_perimage_json(
            src_dir=str(src_dir),
            out_root=str(out_dir),
            id_base="auto"
        )
        
        # Should map 0->0, 2->1
        assert results["id_mapping"] == {0: 0, 2: 1}
        assert results["class_names"] == ["Class0", "class_2"]
        
        print("✅ ID remapping tests passed!")


if __name__ == "__main__":
    test_ingest_perimage_json()
    test_id_remapping()
