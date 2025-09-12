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


def cmd_data_ingest_ps5(args):
    """Ingest PS5 dataset to YOLO/COCO formats."""
    import subprocess
    import sys
    from pathlib import Path
    
    # Get the script path relative to the project root
    script_path = Path(__file__).parent.parent.parent / "scripts" / "ingest_all_ps5.py"
    
    if not script_path.exists():
        print(f"[ERROR] PS5 ingest script not found at {script_path}")
        return 1
    
    # Build command arguments
    cmd_args = [
        sys.executable, str(script_path),
        "--src", args.src,
        "--out", args.out,
        "--split", str(args.split[0]), str(args.split[1]), str(args.split[2]),
        "--seed", str(args.seed),
        "--map", args.map
    ]
    
    if args.resume:
        cmd_args.append("--resume")
    
    print(f"[INFO] Running PS5 dataset ingestion...")
    print(f"[INFO] Command: {' '.join(cmd_args)}")
    
    try:
        result = subprocess.run(cmd_args, check=True)
        print("[INFO] PS5 dataset ingestion completed successfully")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] PS5 dataset ingestion failed with exit code {e.returncode}")
        return 1
    except Exception as e:
        print(f"[ERROR] PS5 dataset ingestion failed: {e}")
        return 1


# Training commands
def cmd_train_yolo(args):
    """Train YOLO model."""
    from .training.yolo_train import prepare_yolo_dataset, train_yolo, save_best_weights
    from .config import load_config
    from pathlib import Path
    import tempfile
    import shutil
    
    try:
        # Load configuration
        cfg = load_config(args.config)
        
        # Determine data source
        if args.data:
            # Direct YOLO data.yaml provided
            data_yaml = args.data
            print(f"[TRAIN] Using YOLO data.yaml: {data_yaml}")
        elif args.coco and args.images:
            # Convert COCO to YOLO first
            print(f"[TRAIN] Converting COCO to YOLO format...")
            work_dir = args.out or "temp_yolo"
            data_yaml = prepare_yolo_dataset(
                coco_json=args.coco,
                images_dir=args.images,
                work_dir=work_dir,
                class_names=cfg["layout"]["classes"]
            )
        else:
            print("❌ Either --data or (--coco and --images) must be provided")
            return 1
        
        # Train model
        print(f"[TRAIN] Starting YOLO training...")
        results = train_yolo(
            data_yaml=data_yaml,
            model=args.model,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            lr0=args.lr0,
            project=args.project,
            name=args.name
        )
        
        # Save best weights
        best_weights = save_best_weights(
            weights_path=results["best_weights"],
            output_dir=cfg["paths"]["weights_dir"],
            model_name="doclayout_yolov10_best.pt"
        )
        
        print(f"[TRAIN] Training completed. Best weights saved to {best_weights}")
        return 0
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return 1


def cmd_train_yolo_from_coco(args):
    """Train YOLO model from COCO format."""
    from .training.yolo_train import prepare_yolo_dataset, train_yolo, save_best_weights
    from .config import load_config
    from pathlib import Path
    import tempfile
    import shutil
    
    try:
        # Load configuration
        cfg = load_config(args.config)
        
        # Convert COCO to YOLO format
        print(f"[TRAIN] Converting COCO to YOLO format...")
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare dataset
            data_yaml = prepare_yolo_dataset(
                coco_json=args.coco_train,
                images_dir=args.images,
                work_dir=temp_dir,
                class_names=cfg["layout"]["classes"]
            )
            
            # Train model
            print(f"[TRAIN] Starting YOLO training...")
            results = train_yolo(
                data_yaml=data_yaml,
                model=args.model,
                imgsz=args.imgsz,
                epochs=args.epochs,
                batch=args.batch,
                lr0=0.01,
                project=args.project,
                name=args.name
            )
            
            # Save best weights
            best_weights = save_best_weights(
                weights_path=results["best_weights"],
                output_dir=cfg["paths"]["weights_dir"],
                model_name="doclayout_yolov10_best.pt"
            )
            
            print(f"[TRAIN] Training completed. Best weights saved to {best_weights}")
            return 0
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
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
    
    # PS5 dataset ingest
    ps5_ingest_parser = data_subparsers.add_parser("ingest-ps5", help="Ingest PS5 dataset to YOLO/COCO formats")
    ps5_ingest_parser.add_argument("--src", required=True, help="Source directory with images and JSON files")
    ps5_ingest_parser.add_argument("--out", required=True, help="Output root directory")
    ps5_ingest_parser.add_argument("--split", nargs=3, type=float, default=[0.9, 0.1, 0.0], 
                                   help="Train/val/test split ratios")
    ps5_ingest_parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits")
    ps5_ingest_parser.add_argument("--resume", action="store_true", help="Resume from .done index")
    ps5_ingest_parser.add_argument("--map", default="1:Text,2:Title,3:List,4:Table,5:Figure",
                                   help="Original ID to name mapping")
    ps5_ingest_parser.set_defaults(func=cmd_data_ingest_ps5)
    
    # Training commands
    train_parser = subparsers.add_parser("train", help="Training commands")
    train_subparsers = train_parser.add_subparsers(dest="train_cmd", required=True)
    
    # YOLO training
    yolo_train_parser = train_subparsers.add_parser("yolo", help="Train YOLO model")
    yolo_train_parser.add_argument("--coco", help="Training COCO JSON")
    yolo_train_parser.add_argument("--data", help="YOLO data.yaml file")
    yolo_train_parser.add_argument("--images", help="Images directory")
    yolo_train_parser.add_argument("--val", help="Validation COCO JSON")
    yolo_train_parser.add_argument("--out", help="Output directory")
    yolo_train_parser.add_argument("--config", default="configs/default.yaml", help="Configuration file")
    yolo_train_parser.add_argument("--model", default="yolov10n.pt", help="YOLO model")
    yolo_train_parser.add_argument("--imgsz", type=int, default=960, help="Image size")
    yolo_train_parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    yolo_train_parser.add_argument("--batch", type=int, default=16, help="Batch size")
    yolo_train_parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    yolo_train_parser.add_argument("--project", default="runs", help="Project directory")
    yolo_train_parser.add_argument("--name", default="doclayout", help="Run name")
    yolo_train_parser.add_argument("--device", help="Device (cuda/cpu/auto)")
    yolo_train_parser.set_defaults(func=cmd_train_yolo)
    
    # YOLO training from COCO
    yolo_coco_parser = train_subparsers.add_parser("yolo-from-coco", help="Train YOLO model from COCO format")
    yolo_coco_parser.add_argument("--coco-train", required=True, help="COCO training JSON")
    yolo_coco_parser.add_argument("--coco-val", required=True, help="COCO validation JSON")
    yolo_coco_parser.add_argument("--images", required=True, help="Images directory")
    yolo_coco_parser.add_argument("--imgsz", type=int, default=960, help="Image size")
    yolo_coco_parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    yolo_coco_parser.add_argument("--batch", type=int, default=16, help="Batch size")
    yolo_coco_parser.add_argument("--model", default="yolov10n.pt", help="YOLO model")
    yolo_coco_parser.add_argument("--project", default="runs/doclayout", help="Project directory")
    yolo_coco_parser.add_argument("--name", default="seed", help="Run name")
    yolo_coco_parser.add_argument("--device", help="Device (cuda/cpu/auto)")
    yolo_coco_parser.set_defaults(func=cmd_train_yolo_from_coco)
    
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
    
    # Calibration evaluation
    calibrate_parser = eval_subparsers.add_parser("calibrate", help="Calibrate detector confidence scores")
    calibrate_parser.add_argument("--val-preds", required=True, help="Validation predictions COCO JSON")
    calibrate_parser.add_argument("--gt", required=True, help="Ground truth COCO JSON")
    calibrate_parser.add_argument("--out", required=True, help="Output calibration parameters file")
    calibrate_parser.add_argument("--method", choices=["temperature", "platt", "isotonic"], 
                                 default="temperature", help="Calibration method")
    calibrate_parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching")
    calibrate_parser.set_defaults(func=cmd_eval_calibrate)
    
    # Description QA
    desc_qa_parser = eval_subparsers.add_parser("desc-qa", help="Quality assurance for descriptions")
    desc_qa_parser.add_argument("--elements", required=True, help="Elements JSON file")
    desc_qa_parser.add_argument("--out", required=True, help="Output directory")
    desc_qa_parser.add_argument("--page-lang", help="Page dominant language")
    desc_qa_parser.add_argument("--allowed-langs", nargs='+', help="Allowed languages")
    desc_qa_parser.set_defaults(func=cmd_eval_desc_qa)
    
    # Active learning commands
    active_parser = subparsers.add_parser("active", help="Active learning commands")
    active_subparsers = active_parser.add_subparsers(dest="active_cmd", required=True)
    
    # Error mining
    mine_errors_parser = active_subparsers.add_parser("mine-errors", help="Mine errors from evaluation results")
    mine_errors_parser.add_argument("--gt", required=True, help="Ground truth COCO JSON file")
    mine_errors_parser.add_argument("--pred", required=True, help="Predictions COCO JSON file")
    mine_errors_parser.add_argument("--images", required=True, help="Images directory")
    mine_errors_parser.add_argument("--out", required=True, help="Output directory")
    mine_errors_parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching")
    mine_errors_parser.set_defaults(func=cmd_active_mine_errors)
    
    # Sample selection
    select_parser = active_subparsers.add_parser("select", help="Select samples for annotation")
    select_parser.add_argument("--pred", required=True, help="Predictions COCO JSON file")
    select_parser.add_argument("--strategy", choices=["margin", "entropy", "low_conf"], 
                              default="margin", help="Uncertainty sampling strategy")
    select_parser.add_argument("--top", type=int, default=500, help="Number of top samples to select")
    select_parser.add_argument("--out", help="Output JSON file for selection results")
    select_parser.add_argument("--threshold", type=float, default=0.5, 
                              help="Confidence threshold for low_conf strategy")
    select_parser.set_defaults(func=cmd_active_select)
    
    # Pseudo-labeling
    pseudo_parser = active_subparsers.add_parser("pseudo-labels", help="Generate pseudo-labels from predictions")
    pseudo_parser.add_argument("--pred", required=True, help="Predictions COCO JSON file")
    pseudo_parser.add_argument("--images", required=True, help="Images directory")
    pseudo_parser.add_argument("--out", required=True, help="Output directory for pseudo-labels")
    pseudo_parser.add_argument("--thr", type=float, default=0.6, help="Confidence threshold")
    pseudo_parser.add_argument("--model", default="unknown", help="Source model name/hash")
    pseudo_parser.add_argument("--epoch", type=int, default=0, help="Training epoch")
    pseudo_parser.set_defaults(func=cmd_active_pseudo_labels)
    
    # Autopilot
    autopilot_parser = active_subparsers.add_parser("autopilot", help="Run active learning autopilot")
    autopilot_parser.add_argument("--coco", required=True, help="Training COCO JSON file")
    autopilot_parser.add_argument("--val", required=True, help="Validation COCO JSON file")
    autopilot_parser.add_argument("--images", required=True, help="Images directory")
    autopilot_parser.add_argument("--cycles", type=int, default=2, help="Number of cycles")
    autopilot_parser.add_argument("--quota", type=int, default=500, help="Samples per cycle")
    autopilot_parser.add_argument("--pseudo-thr", type=float, default=0.6, help="Pseudo-label threshold")
    autopilot_parser.add_argument("--strategy", choices=["margin", "entropy", "low_conf"], 
                                 default="margin", help="Uncertainty strategy")
    autopilot_parser.add_argument("--project", required=True, help="Project directory")
    autopilot_parser.add_argument("--name", help="Project name")
    autopilot_parser.add_argument("--retrain-epochs", type=int, default=20, help="Retrain epochs")
    autopilot_parser.set_defaults(func=cmd_active_autopilot)
    
    # Performance commands
    perf_parser = subparsers.add_parser("perf", help="Performance commands")
    perf_subparsers = perf_parser.add_subparsers(dest="perf_cmd", required=True)
    
    # Benchmark
    bench_parser = perf_subparsers.add_parser("bench", help="Run performance benchmark")
    bench_parser.add_argument("--pdf", required=True, help="PDF file to benchmark")
    bench_parser.add_argument("--config", default="configs/default.yaml", help="Configuration file")
    bench_parser.add_argument("--out", help="Output JSON file for results")
    bench_parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    bench_parser.set_defaults(func=cmd_perf_bench)
    
    # Cache management
    cache_parser = perf_subparsers.add_parser("cache", help="Cache management")
    cache_parser.add_argument("--cache-dir", default=".cache", help="Cache directory")
    cache_parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    cache_parser.add_argument("--clear", help="Clear cache (ocr, vlm, or all)")
    cache_parser.add_argument("--max-size", type=int, default=1000, help="Maximum cache size in MB")
    cache_parser.set_defaults(func=cmd_perf_cache)
    
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
    
    # Pre-upload check command
    preupload_parser = subparsers.add_parser('preupload', help='Pre-upload data quality check')
    preupload_parser.add_argument('--images', required=True, help='Path to images directory')
    preupload_parser.add_argument('--labels', required=True, help='Path to labels (YOLO dir or COCO json)')
    preupload_parser.add_argument('--schema', choices=['yolo', 'coco'], required=True, help='Dataset schema')
    preupload_parser.add_argument('--classes', nargs='+', default=['Text', 'Title', 'List', 'Table', 'Figure'], help='Class names')
    preupload_parser.add_argument('--fix', action='store_true', help='Apply automatic fixes')
    preupload_parser.add_argument('--out', default='data_clean', help='Output directory for cleaned data')
    preupload_parser.add_argument('--config', help='Path to config file')
    preupload_parser.add_argument('--sample-rate', type=float, default=0.03, help='Fraction of images to sample for RTL/rotation analysis')
    preupload_parser.add_argument('--no-pii', action='store_true', help='Skip PII scanning')
    preupload_parser.add_argument('--no-rtl', action='store_true', help='Skip RTL/rotation analysis')
    preupload_parser.add_argument('--no-license', action='store_true', help='Skip license checking')
    preupload_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    preupload_parser.set_defaults(func=cmd_preupload_check)
    
    # Dry run command
    dryrun_parser = subparsers.add_parser('dry-run', help='Dry run training data validation')
    dryrun_parser.add_argument('--coco', help='Path to COCO JSON file')
    dryrun_parser.add_argument('--images', help='Path to images directory')
    dryrun_parser.add_argument('--data-yaml', help='Path to YOLO data.yaml file')
    dryrun_parser.add_argument('--n', type=int, default=32, help='Number of images to test')
    dryrun_parser.set_defaults(func=cmd_dry_run)
    
    # Custom ingest command
    ingest_parser = subparsers.add_parser('ingest-custom', help='Ingest per-image JSON format to YOLO/COCO')
    ingest_parser.add_argument('--dir', required=True, help='Source directory with images and JSON files')
    ingest_parser.add_argument('--out', required=True, help='Output directory')
    ingest_parser.add_argument('--split', nargs=3, type=float, default=[0.9, 0.1, 0.0], help='Train/val/test split ratios')
    ingest_parser.add_argument('--id-base', default='auto', help='ID base for remapping (auto, 0, or 1)')
    ingest_parser.add_argument('--id-to-name', help='ID to name mapping (e.g., "1:Text 2:Title 3:List 4:Table 5:Figure")')
    ingest_parser.add_argument('--seed', type=int, default=42, help='Random seed for splits')
    ingest_parser.add_argument('--no-coco', action='store_true', help='Skip COCO format creation')
    ingest_parser.add_argument('--no-yolo', action='store_true', help='Skip YOLO format creation')
    ingest_parser.set_defaults(func=cmd_ingest_custom)
    
    return parser.parse_args(argv)


# New command functions for active learning and performance

def cmd_eval_calibrate(args):
    """Calibrate detector confidence scores."""
    from .eval.calibrate import calibrate_detector, save_calibration_params
    
    try:
        params = calibrate_detector(
            args.val_preds, args.gt, args.method, args.iou_threshold
        )
        save_calibration_params(params, args.out)
        print(f"[INFO] Calibration completed. Parameters saved to {args.out}")
        return 0
    except Exception as e:
        print(f"[ERROR] Calibration failed: {e}")
        return 1


def cmd_eval_desc_qa(args):
    """Quality assurance for descriptions."""
    from .eval.qa_descriptions import qa_descriptions
    
    try:
        result = qa_descriptions(
            args.elements, args.out, args.page_lang, args.allowed_langs
        )
        print(f"[INFO] QA completed: {result.flagged_elements}/{result.total_elements} elements flagged")
        return 0
    except Exception as e:
        print(f"[ERROR] Description QA failed: {e}")
        return 1


def cmd_active_mine_errors(args):
    """Mine errors from evaluation results."""
    from .active.mine_errors import mine_errors
    
    try:
        error_cases = mine_errors(
            args.gt, args.pred, args.images, args.out, args.iou_threshold
        )
        print(f"[INFO] Error mining completed. Found {len(error_cases)} error cases")
        return 0
    except Exception as e:
        print(f"[ERROR] Error mining failed: {e}")
        return 1


def cmd_active_select(args):
    """Select samples for annotation."""
    from .active.uncertainty import select_samples
    
    try:
        selected = select_samples(
            args.pred, args.strategy, args.top, args.out, threshold=args.threshold
        )
        print(f"[INFO] Selected {len(selected)} samples for annotation")
        return 0
    except Exception as e:
        print(f"[ERROR] Sample selection failed: {e}")
        return 1


def cmd_active_pseudo_labels(args):
    """Generate pseudo-labels from predictions."""
    from .active.pseudo_labels import create_pseudo_labels
    
    try:
        pseudo_labels = create_pseudo_labels(
            args.pred, args.images, args.out, args.thr, args.model, args.epoch
        )
        print(f"[INFO] Generated {len(pseudo_labels)} pseudo-labels")
        return 0
    except Exception as e:
        print(f"[ERROR] Pseudo-labeling failed: {e}")
        return 1


def cmd_active_autopilot(args):
    """Run active learning autopilot."""
    from .active.autopilot import run_autopilot
    
    try:
        results = run_autopilot(
            args.coco, args.val, args.images, args.project,
            args.cycles, args.quota, args.pseudo_thr, args.strategy, args.retrain_epochs
        )
        print(f"[INFO] Autopilot completed {len(results)} cycles")
        return 0
    except Exception as e:
        print(f"[ERROR] Autopilot failed: {e}")
        return 1


def cmd_perf_bench(args):
    """Run performance benchmark."""
    from .perf.parallel import process_pipeline
    
    try:
        # This would call the actual benchmark function
        print(f"[INFO] Running benchmark on {args.pdf}")
        print(f"[INFO] Using {args.workers} workers")
        # In practice, you'd call the benchmark function here
        return 0
    except Exception as e:
        print(f"[ERROR] Benchmark failed: {e}")
        return 1


def cmd_perf_cache(args):
    """Cache management."""
    from .perf.cache import PersistentCache, OCRCache, VLMCache
    
    try:
        cache = PersistentCache(args.cache_dir, args.max_size)
        
        if args.stats:
            stats = cache.get_stats()
            print("Cache Statistics:")
            print(f"  Total entries: {stats['total_entries']}")
            print(f"  Total size: {stats['total_size_mb']:.1f} MB")
            print(f"  Average access count: {stats['avg_access_count']:.1f}")
            
            print("\nSize by prefix:")
            for prefix, info in stats['size_by_prefix'].items():
                print(f"  {prefix}: {info['count']} entries, {info['size'] / (1024*1024):.1f} MB")
        
        if args.clear:
            if args.clear == "all":
                cleared = cache.clear()
            elif args.clear == "ocr":
                ocr_cache = OCRCache(args.cache_dir)
                cleared = ocr_cache.clear_ocr_cache()
            elif args.clear == "vlm":
                vlm_cache = VLMCache(args.cache_dir)
                cleared = vlm_cache.clear_vlm_cache()
            else:
                print(f"Unknown cache type: {args.clear}")
                return 1
            
            print(f"Cleared {cleared} cache entries")
        
        return 0
    except Exception as e:
        print(f"[ERROR] Cache management failed: {e}")
        return 1


def cmd_preupload_check(args):
    """Pre-upload data quality check."""
    from .data.audit import audit_dataset, save_audit_results
    from .data.pii_scan import scan_pii, generate_pii_report
    from .data.rtl_rot_check import check_rtl_rotation, generate_rtl_report
    from .data.license_check import check_licenses, generate_license_report
    from .data.clean import apply_auto_fixes, validate_cleaned_data
    from .data.report import generate_preupload_report
    import time
    
    try:
        print("🔍 Starting pre-upload data quality check...")
        start_time = time.time()
        
        # 1. Dataset Audit
        print("1️⃣ Running dataset audit...")
        audit_results = audit_dataset(
            images_dir=args.images,
            labels=args.labels,
            schema=args.schema,
            class_names=args.classes,
            fix=args.fix
        )
        
        # Save audit results
        output_dir = Path(args.out)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_audit_results(audit_results, output_dir)
        
        print(f"✅ Audit complete. Found {len(audit_results.get('duplicates', []))} duplicates")
        
        # 2. PII Scanning
        if not args.no_pii:
            print("\n2️⃣ Scanning for PII...")
            pii_results = scan_pii(
                filenames=[str(f) for f in Path(args.images).glob("*.jpg")],
                output_path=str(output_dir / "pii_findings.json")
            )
            print(f"✅ PII scan complete. Found {pii_results['summary']['total_findings']} PII instances")
        else:
            print("\n2️⃣ Skipping PII scan")
            pii_results = {"summary": {"total_findings": 0, "files_with_pii": 0}}
        
        # 3. RTL/Rotation Analysis
        if not args.no_rtl:
            print("\n3️⃣ Analyzing RTL and rotation...")
            rtl_results = check_rtl_rotation(
                images_dir=args.images,
                sample_rate=args.sample_rate,
                output_path=str(output_dir / "rtl_rot_report.json"),
                config_path=args.config
            )
            print(f"✅ RTL/rotation analysis complete. "
                  f"RTL: {rtl_results['summary']['rtl_percentage']:.1f}%, "
                  f"Skewed: {rtl_results['summary']['skewed_percentage']:.1f}%")
        else:
            print("\n3️⃣ Skipping RTL/rotation analysis")
            rtl_results = {"summary": {"rtl_percentage": 0, "skewed_percentage": 0}}
        
        # 4. License Check
        if not args.no_license:
            print("\n4️⃣ Checking licenses...")
            license_results = check_licenses(
                data_root=args.images,
                output_path=str(output_dir / "license_report.json")
            )
            print(f"✅ License check complete. "
                  f"Licensed folders: {license_results['summary']['licensed_folders']}")
        else:
            print("\n4️⃣ Skipping license check")
            license_results = {"compliance_status": {"has_license": False}}
        
        # 5. Data Cleaning (if requested)
        if args.fix:
            print("\n5️⃣ Applying automatic fixes...")
            clean_results = apply_auto_fixes(
                audit_result=audit_results,
                images_dir=args.images,
                labels=args.labels,
                schema=args.schema,
                out_dir=args.out
            )
            print(f"✅ Cleaning complete. "
                  f"Cleaned: {clean_results['cleaned_images']} images, "
                  f"{clean_results['cleaned_labels']} labels")
            
            # Validate cleaned data
            print("\n6️⃣ Validating cleaned data...")
            validation_results = validate_cleaned_data(
                cleaned_dir=args.out,
                schema=args.schema,
                class_names=args.classes
            )
            if validation_results["validation_passed"]:
                print("✅ Validation passed")
            else:
                print(f"⚠️ Validation failed: {len(validation_results['issues_found'])} issues found")
        else:
            print("\n5️⃣ Skipping data cleaning (use --fix to enable)")
        
        # 6. Generate Report
        print("\n7️⃣ Generating comprehensive report...")
        report_path = generate_preupload_report(
            audit_results=audit_results,
            pii_results=pii_results,
            rtl_results=rtl_results,
            license_results=license_results,
            output_dir=args.out,
            sample_images=[str(f) for f in Path(args.images).glob("*.jpg")][:20]
        )
        print(f"✅ Report generated: {report_path}")
        
        # 7. Summary
        elapsed_time = time.time() - start_time
        print(f"\n🎉 Pre-upload check complete in {elapsed_time:.1f}s")
        
        # Determine overall status
        critical_issues = (
            len(audit_results.get("image_sanity", {}).get("unreadable", [])) +
            len(audit_results.get("label_sanity", {}).get("invalid_coords", [])) +
            pii_results["summary"]["total_findings"]
        )
        
        needs_attention = (
            rtl_results["summary"]["rtl_percentage"] > 20 or
            rtl_results["summary"]["skewed_percentage"] > 10 or
            not license_results["compliance_status"]["has_license"]
        )
        
        if critical_issues == 0 and not needs_attention:
            print("✅ Status: READY FOR UPLOAD")
            return 0
        else:
            print("⚠️ Status: NEEDS ATTENTION")
            if critical_issues > 0:
                print(f"   - {critical_issues} critical issues found")
            if needs_attention:
                print("   - Dataset needs preprocessing or compliance review")
            return 1
        
    except Exception as e:
        print(f"❌ Error during pre-upload check: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_dry_run(args):
    """Dry run training data validation."""
    from .training.yolo_train import dry_run
    from .data.converters import coco_to_yolo
    from pathlib import Path
    import tempfile
    import shutil
    
    try:
        if args.data_yaml:
            # Direct YOLO data.yaml provided
            data_yaml = args.data_yaml
        elif args.coco and args.images:
            # Convert COCO to YOLO first
            print("Converting COCO to YOLO format...")
            with tempfile.TemporaryDirectory() as temp_dir:
                coco_to_yolo(
                    coco_json_path=args.coco,
                    out_dir=temp_dir,
                    class_names=["Text", "Title", "List", "Table", "Figure"]
                )
                data_yaml = Path(temp_dir) / "data.yaml"
                if not data_yaml.exists():
                    print("❌ Failed to create data.yaml")
                    return 1
        else:
            print("❌ Either --data-yaml or (--coco and --images) must be provided")
            return 1
        
        # Run dry run
        results = dry_run(data_yaml, args.n)
        
        if results["status"] == "OK":
            print("✅ Dry run passed")
            return 0
        else:
            print("❌ Dry run failed")
            for error in results["errors"]:
                print(f"   - {error}")
            return 1
            
    except Exception as e:
        print(f"❌ Dry run failed: {e}")
        return 1


def cmd_ingest_custom(args):
    """Ingest per-image JSON format to YOLO/COCO."""
    from .data.ingest_custom import ingest_perimage_json
    
    try:
        # Parse ID to name mapping if provided
        id_to_name = None
        if args.id_to_name:
            id_to_name = {}
            for mapping in args.id_to_name.split():
                if ':' in mapping:
                    id_str, name = mapping.split(':', 1)
                    id_to_name[int(id_str)] = name
        
        # Parse ID base
        id_base = args.id_base
        if id_base not in ['auto']:
            try:
                id_base = int(id_base)
            except ValueError:
                print(f"❌ Invalid id-base: {args.id_base}")
                return 1
        
        # Run ingestion
        results = ingest_perimage_json(
            src_dir=args.dir,
            out_root=args.out,
            split=tuple(args.split),
            id_base=id_base,
            id_to_name=id_to_name,
            seed=args.seed,
            make_coco=not args.no_coco,
            make_yolo=not args.no_yolo
        )
        
        # Print summary
        print(f"\n📊 Ingestion Summary:")
        print(f"   Images processed: {results['images_processed']}")
        print(f"   Total boxes: {results['total_boxes']}")
        print(f"   Dropped boxes: {results['dropped_boxes']}")
        print(f"   Class names: {results['class_names']}")
        print(f"   ID mapping: {results['id_mapping']}")
        print(f"   Split counts: {results['split_counts']}")
        
        if 'data_yaml' in results:
            print(f"   YOLO data.yaml: {results['data_yaml']}")
        
        if 'coco_files' in results:
            print(f"   COCO files: {results['coco_files']}")
        
        print(f"   Category mapping: {results['cat_map']}")
        print(f"   Debug overlays: {results['debug_overlays']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


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
