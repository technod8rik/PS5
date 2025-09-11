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

# Import new modules
from .data.converters import coco_to_yolo, yolo_to_coco
from .data.splits import stratified_split, uniform_split
from .data.sampler import balanced_subset, stratified_subset, random_subset
from .training.yolo_train import prepare_yolo_dataset, train_yolo, eval_yolo, export_yolo, save_best_weights
from .training.ppdoc_train import get_ppdoc_instructions, check_ppdoc_availability
from .eval.coco_eval import compute_coco_metrics, compute_confusion_matrix, generate_coco_report
from .eval.text_eval import evaluate_text_regions, generate_text_report
from .eval.langid_eval import evaluate_language_id, generate_langid_report
from .eval.desc_eval import evaluate_descriptions, generate_desc_report
from .viz.overlays import create_error_overlays, save_page_overlay
from .viz.report import generate_evaluation_report, generate_html_report


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


# Data commands
def cmd_data_coco_to_yolo(args):
    """Convert COCO format to YOLO format."""
    print(f"[INFO] Converting COCO to YOLO: {args.coco}")
    
    try:
        coco_to_yolo(args.coco, args.out, args.classes)
        print("[INFO] COCO to YOLO conversion completed")
        return 0
    except Exception as e:
        print(f"[ERROR] COCO to YOLO conversion failed: {e}")
        return 1


def cmd_data_yolo_to_coco(args):
    """Convert YOLO format to COCO format."""
    print(f"[INFO] Converting YOLO to COCO: {args.images}")
    
    try:
        yolo_to_coco(args.images, args.labels, args.out, args.classes)
        print("[INFO] YOLO to COCO conversion completed")
        return 0
    except Exception as e:
        print(f"[ERROR] YOLO to COCO conversion failed: {e}")
        return 1


def cmd_data_split(args):
    """Split dataset into train/val/test."""
    print(f"[INFO] Splitting dataset: {args.coco}")
    
    try:
        if args.stratify:
            stratified_split(args.coco, args.out, args.train, args.val, args.test)
        else:
            uniform_split(args.coco, args.out, args.train, args.val, args.test)
        print("[INFO] Dataset splitting completed")
        return 0
    except Exception as e:
        print(f"[ERROR] Dataset splitting failed: {e}")
        return 1


def cmd_data_subset(args):
    """Create balanced subset of dataset."""
    print(f"[INFO] Creating subset: {args.coco}")
    
    try:
        if args.stratify:
            stratified_subset(args.coco, args.out, args.per_class, args.stratify_key)
        else:
            balanced_subset(args.coco, args.out, args.per_class)
        print("[INFO] Subset creation completed")
        return 0
    except Exception as e:
        print(f"[ERROR] Subset creation failed: {e}")
        return 1


# Training commands
def cmd_train_yolo(args):
    """Train YOLO model."""
    print(f"[INFO] Training YOLO model: {args.coco}")
    
    try:
        cfg = load_config(args.config)
        
        # Prepare dataset
        work_dir = Path(args.out) / "yolo_dataset"
        data_yaml = prepare_yolo_dataset(args.coco, args.images, str(work_dir), cfg["train"]["classes"])
        
        # Train model
        weights = train_yolo(
            data_yaml, 
            args.model, 
            args.imgsz, 
            args.epochs, 
            args.batch, 
            args.lr0, 
            args.project, 
            args.name,
            cfg["device"]
        )
        
        # Save best weights
        if weights:
            best_weights = save_best_weights(weights, cfg["paths"]["weights_dir"])
            print(f"[INFO] Best weights saved to {best_weights}")
        
        print("[INFO] YOLO training completed")
        return 0
    except Exception as e:
        print(f"[ERROR] YOLO training failed: {e}")
        return 1


def cmd_train_ppdoc(args):
    """Train PP-DocLayout model (stub)."""
    print("[INFO] PP-DocLayout training not yet implemented")
    
    try:
        instructions = get_ppdoc_instructions()
        print(instructions)
        return 0
    except Exception as e:
        print(f"[ERROR] PP-DocLayout training failed: {e}")
        return 1


# Evaluation commands
def cmd_eval_coco(args):
    """Evaluate COCO metrics."""
    print(f"[INFO] Evaluating COCO metrics: {args.gt}")
    
    try:
        metrics = compute_coco_metrics(args.gt, args.pred, args.out)
        
        # Generate report
        report_file = generate_coco_report(metrics, args.out)
        print(f"[INFO] COCO evaluation completed. Report: {report_file}")
        return 0
    except Exception as e:
        print(f"[ERROR] COCO evaluation failed: {e}")
        return 1


def cmd_eval_text(args):
    """Evaluate text processing metrics."""
    print(f"[INFO] Evaluating text metrics: {args.gt}")
    
    try:
        cfg = load_config(args.config)
        metrics = evaluate_text_regions(args.gt, args.pred, args.images, args.out, cfg)
        
        # Generate report
        report_file = generate_text_report(metrics, args.out)
        print(f"[INFO] Text evaluation completed. Report: {report_file}")
        return 0
    except Exception as e:
        print(f"[ERROR] Text evaluation failed: {e}")
        return 1


def cmd_eval_langid(args):
    """Evaluate language ID metrics."""
    print(f"[INFO] Evaluating language ID metrics: {args.csv}")
    
    try:
        cfg = load_config(args.config)
        metrics = evaluate_language_id(args.csv, cfg["langid"]["model_path"], args.out)
        
        # Generate report
        report_file = generate_langid_report(metrics, args.out)
        print(f"[INFO] Language ID evaluation completed. Report: {report_file}")
        return 0
    except Exception as e:
        print(f"[ERROR] Language ID evaluation failed: {e}")
        return 1


def cmd_eval_desc(args):
    """Evaluate description quality metrics."""
    print(f"[INFO] Evaluating description metrics: {args.pred}")
    
    try:
        metrics = evaluate_descriptions(args.pred, args.refs, args.out)
        
        # Generate report
        report_file = generate_desc_report(metrics, args.out)
        print(f"[INFO] Description evaluation completed. Report: {report_file}")
        return 0
    except Exception as e:
        print(f"[ERROR] Description evaluation failed: {e}")
        return 1


# Visualization commands
def cmd_viz_overlays(args):
    """Create error overlays."""
    print(f"[INFO] Creating overlays: {args.images}")
    
    try:
        cfg = load_config(args.config)
        create_error_overlays(args.gt, args.pred, args.images, args.out, cfg["eval"]["topk_errors"])
        print("[INFO] Overlay creation completed")
        return 0
    except Exception as e:
        print(f"[ERROR] Overlay creation failed: {e}")
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
    
    # Data commands
    data_parser = subparsers.add_parser("data", help="Data processing commands")
    data_subparsers = data_parser.add_subparsers(dest="data_cmd", required=True)
    
    # COCO to YOLO
    coco_to_yolo_parser = data_subparsers.add_parser("coco-to-yolo", help="Convert COCO to YOLO format")
    coco_to_yolo_parser.add_argument("--coco", required=True, help="COCO JSON file")
    coco_to_yolo_parser.add_argument("--images", required=True, help="Images directory")
    coco_to_yolo_parser.add_argument("--out", required=True, help="Output directory")
    coco_to_yolo_parser.add_argument("--classes", nargs="+", required=True, help="Class names")
    coco_to_yolo_parser.set_defaults(func=cmd_data_coco_to_yolo)
    
    # YOLO to COCO
    yolo_to_coco_parser = data_subparsers.add_parser("yolo-to-coco", help="Convert YOLO to COCO format")
    yolo_to_coco_parser.add_argument("--images", required=True, help="Images directory")
    yolo_to_coco_parser.add_argument("--labels", required=True, help="Labels directory")
    yolo_to_coco_parser.add_argument("--out", required=True, help="Output JSON file")
    yolo_to_coco_parser.add_argument("--classes", nargs="+", required=True, help="Class names")
    yolo_to_coco_parser.set_defaults(func=cmd_data_yolo_to_coco)
    
    # Split dataset
    split_parser = data_subparsers.add_parser("split", help="Split dataset into train/val/test")
    split_parser.add_argument("--coco", required=True, help="COCO JSON file")
    split_parser.add_argument("--out", required=True, help="Output directory")
    split_parser.add_argument("--train", type=float, default=0.8, help="Training proportion")
    split_parser.add_argument("--val", type=float, default=0.1, help="Validation proportion")
    split_parser.add_argument("--test", type=float, default=0.1, help="Test proportion")
    split_parser.add_argument("--stratify", action="store_true", help="Use stratified splitting")
    split_parser.set_defaults(func=cmd_data_split)
    
    # Create subset
    subset_parser = data_subparsers.add_parser("subset", help="Create balanced subset")
    subset_parser.add_argument("--coco", required=True, help="COCO JSON file")
    subset_parser.add_argument("--out", required=True, help="Output JSON file")
    subset_parser.add_argument("--per-class", type=int, default=200, help="Samples per class")
    subset_parser.add_argument("--stratify", action="store_true", help="Use stratified sampling")
    subset_parser.add_argument("--stratify-key", default="language", help="Key to stratify on")
    subset_parser.set_defaults(func=cmd_data_subset)
    
    # Training commands
    train_parser = subparsers.add_parser("train", help="Training commands")
    train_subparsers = train_parser.add_subparsers(dest="train_cmd", required=True)
    
    # YOLO training
    yolo_train_parser = train_subparsers.add_parser("yolo", help="Train YOLO model")
    yolo_train_parser.add_argument("--coco", required=True, help="Training COCO JSON")
    yolo_train_parser.add_argument("--images", required=True, help="Images directory")
    yolo_train_parser.add_argument("--val", help="Validation COCO JSON")
    yolo_train_parser.add_argument("--out", required=True, help="Output directory")
    yolo_train_parser.add_argument("--config", default="configs/default.yaml", help="Configuration file")
    yolo_train_parser.add_argument("--model", default="yolov10n.pt", help="YOLO model")
    yolo_train_parser.add_argument("--imgsz", type=int, default=960, help="Image size")
    yolo_train_parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    yolo_train_parser.add_argument("--batch", type=int, default=16, help="Batch size")
    yolo_train_parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    yolo_train_parser.add_argument("--project", default="runs", help="Project directory")
    yolo_train_parser.add_argument("--name", default="doclayout", help="Run name")
    yolo_train_parser.set_defaults(func=cmd_train_yolo)
    
    # PP-DocLayout training
    ppdoc_train_parser = train_subparsers.add_parser("ppdoc", help="Train PP-DocLayout model")
    ppdoc_train_parser.add_argument("--config", default="configs/default.yaml", help="Configuration file")
    ppdoc_train_parser.set_defaults(func=cmd_train_ppdoc)
    
    # Evaluation commands
    eval_parser = subparsers.add_parser("eval", help="Evaluation commands")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_cmd", required=True)
    
    # COCO evaluation
    coco_eval_parser = eval_subparsers.add_parser("coco", help="Evaluate COCO metrics")
    coco_eval_parser.add_argument("--gt", required=True, help="Ground truth COCO JSON")
    coco_eval_parser.add_argument("--pred", required=True, help="Predictions COCO JSON")
    coco_eval_parser.add_argument("--out", required=True, help="Output directory")
    coco_eval_parser.set_defaults(func=cmd_eval_coco)
    
    # Text evaluation
    text_eval_parser = eval_subparsers.add_parser("text", help="Evaluate text metrics")
    text_eval_parser.add_argument("--gt", required=True, help="Ground truth COCO JSON")
    text_eval_parser.add_argument("--pred", required=True, help="Predictions COCO JSON")
    text_eval_parser.add_argument("--images", required=True, help="Images directory")
    text_eval_parser.add_argument("--out", required=True, help="Output directory")
    text_eval_parser.add_argument("--config", default="configs/default.yaml", help="Configuration file")
    text_eval_parser.set_defaults(func=cmd_eval_text)
    
    # Language ID evaluation
    langid_eval_parser = eval_subparsers.add_parser("langid", help="Evaluate language ID metrics")
    langid_eval_parser.add_argument("--csv", required=True, help="Language labels CSV")
    langid_eval_parser.add_argument("--out", required=True, help="Output directory")
    langid_eval_parser.add_argument("--config", default="configs/default.yaml", help="Configuration file")
    langid_eval_parser.set_defaults(func=cmd_eval_langid)
    
    # Description evaluation
    desc_eval_parser = eval_subparsers.add_parser("desc", help="Evaluate description metrics")
    desc_eval_parser.add_argument("--pred", required=True, help="Predictions JSON")
    desc_eval_parser.add_argument("--refs", required=True, help="Reference descriptions JSON")
    desc_eval_parser.add_argument("--out", required=True, help="Output directory")
    desc_eval_parser.set_defaults(func=cmd_eval_desc)
    
    # Visualization commands
    viz_parser = subparsers.add_parser("viz", help="Visualization commands")
    viz_subparsers = viz_parser.add_subparsers(dest="viz_cmd", required=True)
    
    # Error overlays
    overlays_parser = viz_subparsers.add_parser("overlays", help="Create error overlays")
    overlays_parser.add_argument("--images", required=True, help="Images directory")
    overlays_parser.add_argument("--gt", required=True, help="Ground truth COCO JSON")
    overlays_parser.add_argument("--pred", required=True, help="Predictions COCO JSON")
    overlays_parser.add_argument("--out", required=True, help="Output directory")
    overlays_parser.add_argument("--config", default="configs/default.yaml", help="Configuration file")
    overlays_parser.set_defaults(func=cmd_viz_overlays)
    
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    try:
        args = parse_args(argv)
        
        # Override device if specified
        if hasattr(args, 'device') and args.device:
            import os
            os.environ['DOCUAGENT_DEVICE'] = args.device
        
        # Handle nested command structure
        if hasattr(args, 'data_cmd') and args.data_cmd:
            return args.func(args)
        elif hasattr(args, 'train_cmd') and args.train_cmd:
            return args.func(args)
        elif hasattr(args, 'eval_cmd') and args.eval_cmd:
            return args.func(args)
        elif hasattr(args, 'viz_cmd') and args.viz_cmd:
            return args.func(args)
        else:
            return args.func(args)
        
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
