"""Test data converters functionality."""

import json
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docuagent.data.converters import coco_to_yolo, yolo_to_coco, validate_hbb, load_label_map


def test_load_label_map():
    """Test label mapping creation."""
    class_names = ["Text", "Title", "List", "Table", "Figure"]
    label_map = load_label_map(class_names)
    
    expected = {"Text": 0, "Title": 1, "List": 2, "Table": 3, "Figure": 4}
    assert label_map == expected


def test_validate_hbb():
    """Test HBB validation and clipping."""
    # Valid bbox
    bbox = [10, 20, 100, 50]
    validated = validate_hbb(bbox, 200, 300)
    assert validated == [10, 20, 100, 50]
    
    # Clipped bbox
    bbox = [150, 250, 100, 100]  # Would exceed image bounds
    validated = validate_hbb(bbox, 200, 300)
    assert validated[0] == 100  # x clipped
    assert validated[1] == 200  # y clipped
    assert validated[2] == 100  # w clipped
    assert validated[3] == 100  # h clipped
    
    # Negative dimensions
    bbox = [10, 20, -50, 30]
    validated = validate_hbb(bbox, 200, 300)
    assert validated[2] == 0  # w set to 0


def test_coco_to_yolo_conversion():
    """Test COCO to YOLO conversion."""
    # Create test COCO data
    coco_data = {
        "images": [
            {"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 200, 150, 100],
                "area": 15000,
                "iscrowd": 0
            }
        ],
        "categories": [
            {"id": 1, "name": "Text"}
        ]
    }
    
    class_names = ["Text", "Title", "List", "Table", "Figure"]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save test COCO data
        coco_file = Path(temp_dir) / "test.json"
        with open(coco_file, 'w') as f:
            json.dump(coco_data, f)
        
        # Create test image
        images_dir = Path(temp_dir) / "images"
        images_dir.mkdir()
        (images_dir / "test.jpg").write_bytes(b"fake image data")
        
        # Convert
        yolo_dir = Path(temp_dir) / "yolo"
        coco_to_yolo(str(coco_file), str(yolo_dir), class_names)
        
        # Check output
        assert (yolo_dir / "data.yaml").exists()
        assert (yolo_dir / "images" / "test.jpg").exists()
        assert (yolo_dir / "labels" / "test.txt").exists()
        
        # Check label file content
        label_file = yolo_dir / "labels" / "test.txt"
        with open(label_file, 'r') as f:
            content = f.read().strip()
            assert content == "0 0.2734375 0.3125 0.234375 0.20833333333333334"


def test_yolo_to_coco_conversion():
    """Test YOLO to COCO conversion."""
    class_names = ["Text", "Title", "List", "Table", "Figure"]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test YOLO data
        images_dir = Path(temp_dir) / "images"
        labels_dir = Path(temp_dir) / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        
        # Create test image
        (images_dir / "test.jpg").write_bytes(b"fake image data")
        
        # Create label file
        label_file = labels_dir / "test.txt"
        with open(label_file, 'w') as f:
            f.write("0 0.2734375 0.3125 0.234375 0.20833333333333334\n")
        
        # Convert
        output_file = Path(temp_dir) / "output.json"
        yolo_to_coco(str(images_dir), str(labels_dir), str(output_file), class_names)
        
        # Check output
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            coco_data = json.load(f)
        
        assert "images" in coco_data
        assert "annotations" in coco_data
        assert "categories" in coco_data
        assert len(coco_data["images"]) == 1
        assert len(coco_data["annotations"]) == 1
        assert len(coco_data["categories"]) == 5


def test_roundtrip_conversion():
    """Test COCO -> YOLO -> COCO roundtrip."""
    # Create test COCO data
    coco_data = {
        "images": [
            {"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 200, 150, 100],
                "area": 15000,
                "iscrowd": 0
            }
        ],
        "categories": [
            {"id": 1, "name": "Text"}
        ]
    }
    
    class_names = ["Text", "Title", "List", "Table", "Figure"]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save original COCO data
        original_file = Path(temp_dir) / "original.json"
        with open(original_file, 'w') as f:
            json.dump(coco_data, f)
        
        # Create test image
        images_dir = Path(temp_dir) / "images"
        images_dir.mkdir()
        (images_dir / "test.jpg").write_bytes(b"fake image data")
        
        # COCO -> YOLO
        yolo_dir = Path(temp_dir) / "yolo"
        coco_to_yolo(str(original_file), str(images_dir), str(yolo_dir), class_names)
        
        # YOLO -> COCO
        output_file = Path(temp_dir) / "output.json"
        yolo_to_coco(str(yolo_dir / "images"), str(yolo_dir / "labels"), str(output_file), class_names)
        
        # Check that we have the same number of annotations
        with open(output_file, 'r') as f:
            output_data = json.load(f)
        
        assert len(output_data["annotations"]) == len(coco_data["annotations"])
        assert len(output_data["images"]) == len(coco_data["images"])


if __name__ == "__main__":
    test_load_label_map()
    test_validate_hbb()
    test_coco_to_yolo_conversion()
    test_yolo_to_coco_conversion()
    test_roundtrip_conversion()
    print("All converter tests passed!")
