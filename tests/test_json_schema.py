"""Test JSON schema validation."""

import json
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docuagent.compile_json import validate_json_schema, to_standard_json
from docuagent.ocr_lang import Element


def test_valid_json_schema():
    """Test valid JSON schema."""
    # Create test elements
    elements = [
        Element(
            page=0,
            cls="Text",
            bbox=(10, 20, 100, 30),
            content="Sample text",
            language="en",
            meta={"confidence": 0.95}
        ),
        Element(
            page=0,
            cls="Table",
            bbox=(10, 60, 200, 100),
            content="Table with 3 columns and 5 rows",
            language=None,
            meta={"source": "vlm_description"}
        )
    ]
    
    # Convert to JSON
    json_data = to_standard_json("test_document", elements)
    
    # Validate schema
    assert validate_json_schema(json_data), "Valid JSON should pass validation"
    
    # Check required fields
    assert "document_id" in json_data
    assert "elements" in json_data
    assert json_data["document_id"] == "test_document"
    assert len(json_data["elements"]) == 2
    
    # Check element structure
    element = json_data["elements"][0]
    assert element["class"] == "Text"
    assert element["bbox"] == [10, 20, 100, 30]
    assert element["content"] == "Sample text"
    assert element["language"] == "en"
    assert element["page"] == 0


def test_invalid_json_schema():
    """Test invalid JSON schema."""
    # Missing required fields
    invalid_data = {"document_id": "test"}
    assert not validate_json_schema(invalid_data), "Missing elements should fail validation"
    
    # Invalid bbox format
    invalid_data = {
        "document_id": "test",
        "elements": [{
            "class": "Text",
            "bbox": "invalid",  # Should be list
            "content": "text",
            "language": "en",
            "page": 0
        }]
    }
    assert not validate_json_schema(invalid_data), "Invalid bbox format should fail validation"
    
    # Negative bbox values
    invalid_data = {
        "document_id": "test",
        "elements": [{
            "class": "Text",
            "bbox": [-1, 20, 100, 30],  # Negative x
            "content": "text",
            "language": "en",
            "page": 0
        }]
    }
    assert not validate_json_schema(invalid_data), "Negative bbox values should fail validation"


def test_bbox_format():
    """Test bbox format validation."""
    # Valid bbox
    valid_bboxes = [
        [0, 0, 100, 50],
        [10, 20, 200, 300],
        [0, 0, 1, 1]
    ]
    
    for bbox in valid_bboxes:
        data = {
            "document_id": "test",
            "elements": [{
                "class": "Text",
                "bbox": bbox,
                "content": "text",
                "language": "en",
                "page": 0
            }]
        }
        assert validate_json_schema(data), f"Valid bbox {bbox} should pass validation"
    
    # Invalid bbox
    invalid_bboxes = [
        "not_a_list",
        [1, 2, 3],  # Wrong length
        [1, 2, 3, 4, 5],  # Wrong length
        [1.5, 2, 3, 4],  # Not integers
        ["1", "2", "3", "4"],  # Not integers
    ]
    
    for bbox in invalid_bboxes:
        data = {
            "document_id": "test",
            "elements": [{
                "class": "Text",
                "bbox": bbox,
                "content": "text",
                "language": "en",
                "page": 0
            }]
        }
        assert not validate_json_schema(data), f"Invalid bbox {bbox} should fail validation"


if __name__ == "__main__":
    test_valid_json_schema()
    test_invalid_json_schema()
    test_bbox_format()
    print("All JSON schema tests passed!")
