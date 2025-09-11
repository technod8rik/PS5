"""Command-line interface for DocuAgent."""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Tuple

from .config import load_config
from .preprocessing import DocumentPreprocessor
from .layout import LayoutDetector
from .ocr_lang import OCRLang, Element
from .describe import Describer
from .compile_json import to_standard_json, save_json, load_json
from .utils import set_deterministic_seeds, ensure_dir, get_file_stem


def cmd_process(args):
    """End-to-end document processing."""
    print(f"[INFO] Processing document: {args.input}")
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Set deterministic seeds
    set_deterministic_seeds(42)
    
    # Parse page range
    page_range = None
    if args.pages:
        try:
            start, end = map(int, args.pages.split('-'))
            page_range = (start, end)
        except ValueError:
            print("[ERROR] Invalid page range format. Use 'start-end' (e.g., '0-10')")
            return 1
    
    # Initialize components
    preprocessor = DocumentPreprocessor(dpi=cfg["dpi"])
    layout_detector = LayoutDetector(cfg)
    ocr_lang = OCRLang(cfg)
    describer = Describer(cfg)
    
    # Process document
    try:
        # Preprocess
        print("[INFO] Preprocessing document...")
        images = preprocessor.process_document(args.input, page_range=page_range)
        print(f"[INFO] Processed {len(images)} pages")
        
        # Save processed images if requested
        if args.save_images:
            output_dir = Path(args.out) / f"{get_file_stem(args.input)}.pages"
            saved_paths = preprocessor.save_processed_images(images, output_dir, get_file_stem(args.input))
            print(f"[INFO] Saved processed images to {output_dir}")
        
        # Process each page
        all_elements = []
        for page_idx, image in enumerate(images):
            print(f"[INFO] Processing page {page_idx + 1}/{len(images)}")
            
            # Layout detection
            layout_boxes = layout_detector.detect(image, page_idx)
            print(f"[INFO] Detected {len(layout_boxes)} layout elements")
            
            # OCR and language ID
            elements = ocr_lang.run(image, layout_boxes, page_idx)
            print(f"[INFO] Extracted {len(elements)} text elements")
            
            all_elements.extend(elements)
        
        # Generate descriptions for non-text elements
        print("[INFO] Generating descriptions for non-text elements...")
        for page_idx, image in enumerate(images):
            page_elements = [e for e in all_elements if e.page == page_idx]
            if any(e.cls in ["Table", "Figure"] for e in page_elements):
                updated_elements = describer.describe_elements(image, page_elements)
                # Update elements in the main list
                for i, element in enumerate(all_elements):
                    if element.page == page_idx:
                        for updated in updated_elements:
                            if (updated.page == element.page and 
                                updated.bbox == element.bbox and 
                                updated.cls == element.cls):
                                all_elements[i] = updated
                                break
        
        # Compile JSON
        print("[INFO] Compiling JSON output...")
        document_id = get_file_stem(args.input)
        json_data = to_standard_json(document_id, all_elements)
        
        # Save JSON
        output_path = Path(args.out) / f"{document_id}.elements.json"
        ensure_dir(output_path.parent)
        save_json(json_data, output_path)
        
        print(f"[INFO] Processing complete. Output saved to {output_path}")
        print(f"[INFO] Total elements: {len(all_elements)}")
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        return 1


def cmd_layout(args):
    """Layout detection only."""
    print(f"[INFO] Running layout detection on: {args.input}")
    
    cfg = load_config(args.config)
    preprocessor = DocumentPreprocessor(dpi=cfg["dpi"])
    layout_detector = LayoutDetector(cfg)
    
    try:
        images = preprocessor.process_document(args.input)
        
        for page_idx, image in enumerate(images):
            print(f"[INFO] Detecting layout on page {page_idx + 1}")
            layout_boxes = layout_detector.detect(image, page_idx)
            print(f"[INFO] Detected {len(layout_boxes)} elements")
        
        print("[INFO] Layout detection complete. Check debug directory for overlays.")
        return 0
        
    except Exception as e:
        print(f"[ERROR] Layout detection failed: {e}")
        return 1


def cmd_ocr(args):
    """OCR and language ID only."""
    print(f"[INFO] Running OCR on: {args.input}")
    
    cfg = load_config(args.config)
    preprocessor = DocumentPreprocessor(dpi=cfg["dpi"])
    layout_detector = LayoutDetector(cfg)
    ocr_lang = OCRLang(cfg)
    
    try:
        images = preprocessor.process_document(args.input)
        
        all_elements = []
        for page_idx, image in enumerate(images):
            print(f"[INFO] Processing page {page_idx + 1}")
            
            # Layout detection
            layout_boxes = layout_detector.detect(image, page_idx)
            
            # OCR
            elements = ocr_lang.run(image, layout_boxes, page_idx)
            all_elements.extend(elements)
        
        print(f"[INFO] OCR complete. Extracted {len(all_elements)} elements.")
        return 0
        
    except Exception as e:
        print(f"[ERROR] OCR failed: {e}")
        return 1


def cmd_describe(args):
    """Description generation only."""
    print(f"[INFO] Running description generation on: {args.input}")
    
    cfg = load_config(args.config)
    preprocessor = DocumentPreprocessor(dpi=cfg["dpi"])
    layout_detector = LayoutDetector(cfg)
    ocr_lang = OCRLang(cfg)
    describer = Describer(cfg)
    
    try:
        images = preprocessor.process_document(args.input)
        
        all_elements = []
        for page_idx, image in enumerate(images):
            print(f"[INFO] Processing page {page_idx + 1}")
            
            # Layout detection
            layout_boxes = layout_detector.detect(image, page_idx)
            
            # OCR
            elements = ocr_lang.run(image, layout_boxes, page_idx)
            all_elements.extend(elements)
        
        # Generate descriptions
        print("[INFO] Generating descriptions...")
        for page_idx, image in enumerate(images):
            page_elements = [e for e in all_elements if e.page == page_idx]
            if any(e.cls in ["Table", "Figure"] for e in page_elements):
                updated_elements = describer.describe_elements(image, page_elements)
                # Update elements
                for i, element in enumerate(all_elements):
                    if element.page == page_idx:
                        for updated in updated_elements:
                            if (updated.page == element.page and 
                                updated.bbox == element.bbox and 
                                updated.cls == element.cls):
                                all_elements[i] = updated
                                break
        
        print("[INFO] Description generation complete.")
        return 0
        
    except Exception as e:
        print(f"[ERROR] Description generation failed: {e}")
        return 1


def cmd_json(args):
    """JSON compilation only."""
    print(f"[INFO] Compiling JSON from: {args.input}")
    
    try:
        # Load existing JSON or process from scratch
        if Path(args.input).suffix == '.json':
            json_data = load_json(Path(args.input))
        else:
            # Process document first
            return cmd_process(args)
        
        # Save compiled JSON
        output_path = Path(args.out) / f"{get_file_stem(args.input)}.elements.json"
        ensure_dir(output_path.parent)
        save_json(json_data, output_path)
        
        print(f"[INFO] JSON compilation complete. Output saved to {output_path}")
        return 0
        
    except Exception as e:
        print(f"[ERROR] JSON compilation failed: {e}")
        return 1


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DocuAgent: Layout-aware document processing")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Process command
    process_parser = subparsers.add_parser("process", help="End-to-end document processing")
    process_parser.add_argument("input", help="Input document (PDF or image)")
    process_parser.add_argument("--out", required=True, help="Output directory")
    process_parser.add_argument("--config", default="configs/default.yaml", help="Configuration file")
    process_parser.add_argument("--device", help="Override device (cuda/cpu/auto)")
    process_parser.add_argument("--pages", help="Page range (e.g., '0-10')")
    process_parser.add_argument("--save-images", action="store_true", help="Save processed images")
    process_parser.set_defaults(func=cmd_process)
    
    # Layout command
    layout_parser = subparsers.add_parser("layout", help="Layout detection only")
    layout_parser.add_argument("input", help="Input document")
    layout_parser.add_argument("--config", default="configs/default.yaml", help="Configuration file")
    layout_parser.add_argument("--device", help="Override device")
    layout_parser.set_defaults(func=cmd_layout)
    
    # OCR command
    ocr_parser = subparsers.add_parser("ocr", help="OCR and language ID only")
    ocr_parser.add_argument("input", help="Input document")
    ocr_parser.add_argument("--config", default="configs/default.yaml", help="Configuration file")
    ocr_parser.add_argument("--device", help="Override device")
    ocr_parser.set_defaults(func=cmd_ocr)
    
    # Describe command
    describe_parser = subparsers.add_parser("describe", help="Description generation only")
    describe_parser.add_argument("input", help="Input document")
    describe_parser.add_argument("--config", default="configs/default.yaml", help="Configuration file")
    describe_parser.add_argument("--device", help="Override device")
    describe_parser.set_defaults(func=cmd_describe)
    
    # JSON command
    json_parser = subparsers.add_parser("json", help="JSON compilation only")
    json_parser.add_argument("input", help="Input document or JSON file")
    json_parser.add_argument("--out", required=True, help="Output directory")
    json_parser.add_argument("--config", default="configs/default.yaml", help="Configuration file")
    json_parser.set_defaults(func=cmd_json)
    
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    try:
        args = parse_args(argv)
        
        # Override device if specified
        if hasattr(args, 'device') and args.device:
            import os
            os.environ['DOCUAGENT_DEVICE'] = args.device
        
        return args.func(args)
        
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
