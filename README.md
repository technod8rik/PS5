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

## Integration with RAG

This pipeline extends the existing RAG starter (`starter.py`) by providing enhanced document processing capabilities. The generated JSON can be directly used with the existing FAISS-based search functionality.
