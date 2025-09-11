# DocuAgent: Layout-Aware Document Processing

A modular pipeline for layout-aware document processing with OCR, language identification, and VLM-based descriptions.

## Features

- **Layout Detection**: PP-DocLayout-L and YOLOv10 backends for detecting document elements
- **OCR + Language ID**: PaddleOCR with fastText language identification
- **VLM Descriptions**: Qwen2-VL for describing tables, figures, and charts
- **Standardized JSON**: Clean output format preserving layout fidelity and reading order
- **Modular Pipeline**: Pluggable components for preprocessing, layout detection, OCR, and description
- **CLI Interface**: Easy-to-use command-line tools for end-to-end and stepwise processing

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### End-to-End Processing

Process a document with all pipeline steps:

```bash
python cli.py process input.pdf --out ./output --config configs/default.yaml
```

### Stepwise Processing

Run individual pipeline components:

```bash
# Layout detection only
python cli.py layout input.pdf --config configs/default.yaml

# OCR and language ID only
python cli.py ocr input.pdf --config configs/default.yaml

# Description generation only
python cli.py describe input.pdf --config configs/default.yaml

# JSON compilation only
python cli.py json input.pdf --out ./output
```

### Command Options

- `--out`: Output directory for results
- `--config`: Configuration file (default: `configs/default.yaml`)
- `--device`: Override device (cuda/cpu/auto)
- `--pages`: Page range (e.g., `0-10`)
- `--save-images`: Save processed images

## Configuration

The `configs/default.yaml` file contains all configuration options:

```yaml
device: "auto"  # "cuda" | "cpu" | "auto"
dpi: 220

layout:
  model: "pp_doclayout_l"  # "pp_doclayout_l" | "yolov10"
  conf_threshold: 0.25
  classes: ["Text", "Title", "List", "Table", "Figure"]

ocr:
  lang: "ml"  # multilingual

langid:
  model_path: ".cache/lid.176.bin"

vlm:
  model: "Qwen/Qwen2-VL-7B-Instruct"

describe:
  batch_size: 2

io:
  cache_dir: ".cache"
  debug_dir: "./debug"

runtime:
  num_workers: 4
```

## Output Format

The pipeline generates a standardized JSON file with the following structure:

```json
{
  "document_id": "example.pdf",
  "elements": [
    {
      "class": "Text",
      "bbox": [x, y, w, h],
      "content": "Extracted text or description",
      "language": "en",
      "page": 0
    }
  ]
}
```

## Pipeline Components

1. **Preprocessing**: PDF rasterization, deskewing, and denoising
2. **Layout Detection**: Element detection with HBB bounding boxes
3. **OCR + Language ID**: Text extraction with language identification
4. **Description**: VLM-based descriptions for non-text elements
5. **JSON Compilation**: Standardized output format

## Development

### Running Tests

```bash
python tests/test_json_schema.py
python tests/test_sorting.py
```

### Package Structure

```
src/
  docuagent/
    __init__.py
    config.py          # Configuration management
    preprocessing.py   # Document preprocessing
    layout.py          # Layout detection
    ocr_lang.py        # OCR and language ID
    describe.py        # VLM descriptions
    compile_json.py    # JSON compilation
    utils.py           # Utility functions
    cli.py             # Command-line interface
configs/
  default.yaml        # Default configuration
tests/
  test_json_schema.py # JSON validation tests
  test_sorting.py     # Reading order tests
```

## Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)
- Sufficient disk space for model downloads

## Notes

- First run downloads required models (fastText language ID, VLM models)
- Models are cached in the `.cache` directory
- Debug outputs are saved to the `debug` directory
- The pipeline gracefully handles missing models with clear error messages

## Training and Evaluation

### Data Processing

Convert between COCO and YOLO formats:

```bash
# Convert COCO to YOLO
python cli.py data coco-to-yolo --coco data/anno.json --images data/images --out data/yolo --classes Text Title List Table Figure

# Convert YOLO to COCO
python cli.py data yolo-to-coco --images data/yolo/images --labels data/yolo/labels --out data/coco.json --classes Text Title List Table Figure

# Split dataset
python cli.py data split --coco data/anno.json --out data/splits --train 0.8 --val 0.1 --test 0.1

# Create balanced subset
python cli.py data subset --coco data/anno.json --out data/subset.json --per-class 200
```

### Model Training

Train layout detection models:

```bash
# Train YOLO model
python cli.py train yolo --coco data/splits/train.json --images data/images --val data/splits/val.json \
  --imgsz 960 --epochs 60 --batch 16 --model yolov10n.pt --project runs/doclayout --name seed

# PP-DocLayout training (stub)
python cli.py train ppdoc --config configs/default.yaml
```

### Evaluation

Evaluate model performance:

```bash
# COCO metrics
python cli.py eval coco --gt data/splits/val.json --pred runs/doclayout/seed/preds_val.json --out out/eval

# Text processing metrics
python cli.py eval text --gt data/splits/val.json --pred runs/doclayout/seed/preds_val.json --images data/images --out out/eval

# Language ID metrics
python cli.py eval langid --csv data/langid_val.csv --out out/eval

# Description quality metrics
python cli.py eval desc --pred out/sample.elements.json --refs data/descriptions_refs.json --out out/eval
```

### Visualization

Create error analysis overlays:

```bash
# Error overlays
python cli.py viz overlays --images data/images --gt data/splits/val.json --pred runs/doclayout/seed/preds_val.json --out debug/overlays
```

## Integration with RAG

This pipeline extends the existing RAG starter (`starter.py`) by providing enhanced document processing capabilities. The generated JSON can be directly used with the existing FAISS-based search functionality.

## Training Workflow

1. **Prepare Data**: Convert your annotations to COCO format
2. **Split Dataset**: Create train/val/test splits
3. **Train Model**: Train YOLO or PP-DocLayout model
4. **Evaluate**: Run comprehensive evaluation metrics
5. **Deploy**: Use trained weights in production pipeline

## Evaluation Metrics

- **Layout Detection**: mAP@0.5, mAP@0.5:0.95, per-class AP
- **Text Processing**: Character Error Rate (CER), Word Error Rate (WER)
- **Language ID**: Accuracy, confusion matrix, per-language metrics
- **Description Quality**: BLEU score, BERTScore for VLM outputs
