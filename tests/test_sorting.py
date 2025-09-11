"""Test reading order sorting."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docuagent.ocr_lang import Element


def test_reading_order_sorting():
    """Test that elements are sorted by reading order (page, y, x)."""
    # Create test elements with different positions
    elements = [
        Element(page=0, cls="Text", bbox=(100, 50, 80, 20), content="Bottom right", language="en", meta={}),
        Element(page=0, cls="Text", bbox=(10, 10, 60, 20), content="Top left", language="en", meta={}),
        Element(page=0, cls="Text", bbox=(80, 10, 60, 20), content="Top right", language="en", meta={}),
        Element(page=0, cls="Text", bbox=(10, 50, 60, 20), content="Bottom left", language="en", meta={}),
        Element(page=1, cls="Text", bbox=(10, 10, 60, 20), content="Page 1 top", language="en", meta={}),
        Element(page=0, cls="Text", bbox=(10, 30, 60, 20), content="Middle", language="en", meta={}),
    ]
    
    # Sort by reading order
    sorted_elements = sorted(elements, key=lambda e: (e.page, e.bbox[1], e.bbox[0]))
    
    # Check page order first
    pages = [e.page for e in sorted_elements]
    assert pages == [0, 0, 0, 0, 0, 1], f"Pages not sorted correctly: {pages}"
    
    # Check y-coordinate order within page 0
    page_0_elements = [e for e in sorted_elements if e.page == 0]
    y_coords = [e.bbox[1] for e in page_0_elements]
    assert y_coords == [10, 10, 30, 50, 50], f"Y coordinates not sorted correctly: {y_coords}"
    
    # Check x-coordinate order for same y-coordinates
    # Elements at y=10 should be sorted by x
    y_10_elements = [e for e in page_0_elements if e.bbox[1] == 10]
    x_coords = [e.bbox[0] for e in y_10_elements]
    assert x_coords == [10, 80], f"X coordinates not sorted correctly: {x_coords}"
    
    # Check content order
    content_order = [e.content for e in sorted_elements]
    expected_order = ["Top left", "Top right", "Middle", "Bottom left", "Bottom right", "Page 1 top"]
    assert content_order == expected_order, f"Content order incorrect: {content_order}"


def test_identical_positions():
    """Test sorting with identical positions."""
    elements = [
        Element(page=0, cls="Text", bbox=(10, 10, 60, 20), content="First", language="en", meta={}),
        Element(page=0, cls="Text", bbox=(10, 10, 60, 20), content="Second", language="en", meta={}),
        Element(page=0, cls="Text", bbox=(10, 10, 60, 20), content="Third", language="en", meta={}),
    ]
    
    # Sort should maintain stable order for identical positions
    sorted_elements = sorted(elements, key=lambda e: (e.page, e.bbox[1], e.bbox[0]))
    
    content_order = [e.content for e in sorted_elements]
    assert content_order == ["First", "Second", "Third"], f"Stable sort failed: {content_order}"


def test_multiple_pages():
    """Test sorting across multiple pages."""
    elements = [
        Element(page=2, cls="Text", bbox=(10, 10, 60, 20), content="Page 2", language="en", meta={}),
        Element(page=0, cls="Text", bbox=(10, 10, 60, 20), content="Page 0", language="en", meta={}),
        Element(page=1, cls="Text", bbox=(10, 10, 60, 20), content="Page 1", language="en", meta={}),
    ]
    
    sorted_elements = sorted(elements, key=lambda e: (e.page, e.bbox[1], e.bbox[0]))
    
    content_order = [e.content for e in sorted_elements]
    assert content_order == ["Page 0", "Page 1", "Page 2"], f"Multi-page sort failed: {content_order}"


def test_edge_cases():
    """Test edge cases for sorting."""
    # Empty list
    elements = []
    sorted_elements = sorted(elements, key=lambda e: (e.page, e.bbox[1], e.bbox[0]))
    assert len(sorted_elements) == 0
    
    # Single element
    elements = [Element(page=0, cls="Text", bbox=(10, 10, 60, 20), content="Single", language="en", meta={})]
    sorted_elements = sorted(elements, key=lambda e: (e.page, e.bbox[1], e.bbox[0]))
    assert len(sorted_elements) == 1
    assert sorted_elements[0].content == "Single"
    
    # Zero coordinates
    elements = [
        Element(page=0, cls="Text", bbox=(0, 0, 60, 20), content="Origin", language="en", meta={}),
        Element(page=0, cls="Text", bbox=(10, 0, 60, 20), content="Right of origin", language="en", meta={}),
    ]
    sorted_elements = sorted(elements, key=lambda e: (e.page, e.bbox[1], e.bbox[0]))
    content_order = [e.content for e in sorted_elements]
    assert content_order == ["Origin", "Right of origin"]


if __name__ == "__main__":
    test_reading_order_sorting()
    test_identical_positions()
    test_multiple_pages()
    test_edge_cases()
    print("All sorting tests passed!")
