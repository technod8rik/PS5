"""JSON compilation for standardized document output."""

import json
from pathlib import Path
from typing import List, Dict, Any

from .ocr_lang import Element


def to_standard_json(document_id: str, elements: List[Element]) -> Dict[str, Any]:
    """Convert elements to standardized JSON format.
    
    Args:
        document_id: Document identifier
        elements: List of processed elements
        
    Returns:
        Standardized JSON structure
    """
    # Sort elements by reading order (page, y, x)
    sorted_elements = sorted(elements, key=lambda e: (e.page, e.bbox[1], e.bbox[0]))
    
    # Convert elements to JSON format
    json_elements = []
    for element in sorted_elements:
        json_element = {
            "class": element.cls,
            "bbox": list(element.bbox),  # Convert tuple to list
            "content": element.content,
            "language": element.language,
            "page": element.page
        }
        
        # Add metadata if present
        if element.meta:
            json_element["meta"] = element.meta
        
        json_elements.append(json_element)
    
    return {
        "document_id": document_id,
        "elements": json_elements
    }


def save_json(payload: Dict[str, Any], out_path: Path) -> None:
    """Save JSON payload to file.
    
    Args:
        payload: JSON data to save
        out_path: Output file path
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] Saved JSON to {out_path}")


def load_json(json_path: Path) -> Dict[str, Any]:
    """Load JSON from file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Loaded JSON data
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_json_schema(payload: Dict[str, Any]) -> bool:
    """Validate JSON schema for required fields.
    
    Args:
        payload: JSON data to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["document_id", "elements"]
    
    # Check top-level fields
    for field in required_fields:
        if field not in payload:
            print(f"[ERROR] Missing required field: {field}")
            return False
    
    # Check elements structure
    if not isinstance(payload["elements"], list):
        print("[ERROR] 'elements' must be a list")
        return False
    
    # Check each element
    for i, element in enumerate(payload["elements"]):
        if not isinstance(element, dict):
            print(f"[ERROR] Element {i} is not a dictionary")
            return False
        
        element_required = ["class", "bbox", "content", "language", "page"]
        for field in element_required:
            if field not in element:
                print(f"[ERROR] Element {i} missing required field: {field}")
                return False
        
        # Validate bbox format
        bbox = element["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            print(f"[ERROR] Element {i} bbox must be a list of 4 integers")
            return False
        
        if not all(isinstance(x, int) for x in bbox):
            print(f"[ERROR] Element {i} bbox must contain only integers")
            return False
        
        # Validate bbox values (should be non-negative)
        if any(x < 0 for x in bbox):
            print(f"[ERROR] Element {i} bbox contains negative values")
            return False
    
    return True


def merge_partial_results(partial_paths: List[Path]) -> Dict[str, Any]:
    """Merge multiple partial JSON results into a single document.
    
    Args:
        partial_paths: List of paths to partial JSON files
        
    Returns:
        Merged JSON data
    """
    all_elements = []
    document_id = None
    
    for path in partial_paths:
        if not path.exists():
            print(f"[WARN] Partial file not found: {path}")
            continue
        
        partial_data = load_json(path)
        
        if document_id is None:
            document_id = partial_data.get("document_id", "merged_document")
        
        elements = partial_data.get("elements", [])
        all_elements.extend(elements)
    
    return to_standard_json(document_id or "merged_document", all_elements)
