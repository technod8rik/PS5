# PS5 Dataset Training Runbook

This runbook provides exact commands to ingest and train the PS5 dataset using the DocuAgent pipeline.

## Environment & Dependencies (One Time Setup)

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install -U pip wheel
pip install -r requirements.txt

# Install PyTorch (if needed)
bash scripts/install_torch.sh

# Install fastText fallback (if needed)
bash scripts/install_fasttext_fallback.sh || true
```

## Dataset Ingestion (Already Done - Reference Only)

The PS5 dataset has already been ingested. For reference, here's the command that was used:

```bash
python scripts/ingest_all_ps5.py \
  --src "/home/akshar/PS5/data/PS 5 Intelligent Multilingual Document Understanding/extracted_data/train" \
  --out data/ps5_ingested \
  --split 0.9 0.1 0.0 \
  --seed 42 \
  --resume \
  --map "1:Text,2:Title,3:List,4:Table,5:Figure"
```

**Dataset Statistics:**
- Images processed: 4,000
- Train images: 3,600 (90%)
- Val images: 400 (10%)
- Total boxes: 40,665
- Class mapping: {1→0, 2→1, 3→2, 4→3, 5→4}

## Training (Easy Launcher)

Use the automated training script with device detection and OOM fallback:

```bash
# Make script executable
chmod +x scripts/train_ps5_easy.sh

# Run training with default parameters
IMGSZ=1280 EPOCHS=80 BATCH=16 YOLO_MODEL=yolov10s.pt DEVICE=0 bash scripts/train_ps5_easy.sh
```

**Alternative: Manual Training**

```bash
# Direct CLI training (if you prefer manual control)
python3 cli.py train yolo \
  --data data/ps5_ingested/yolo/data.yaml \
  --imgsz 1280 \
  --epochs 80 \
  --batch 16 \
  --model yolov10s.pt \
  --project runs/doclayout \
  --name ps5_seed \
  --device 0
```

## Where Results Land

After successful training, check these locations:

```
Best weights:       runs/doclayout/ps5_seed/weights/best.pt
Last weights:       runs/doclayout/ps5_seed/weights/last.pt
Metrics & curves:   runs/doclayout/ps5_seed/
Val predictions:    runs/doclayout/ps5_seed_val/
Runtime copy:       weights/doclayout_yolov10_best.pt
Training logs:      runs/doclayout/ps5_seed/train.log
```

## Pipeline Smoke Test

Test the trained model with a sample document:

```bash
python3 cli.py process samples/multilang_10p.pdf --out out --config configs/default.yaml
```

## Important Notes

### Class Configuration
- **Class order locked:** `names: [Text, Title, List, Table, Figure]` (nc=5)
- **ID mapping:** {1→0, 2→1, 3→2, 4→3, 5→4}
- **Dataset:** 4,000 images with 40,665 total annotations

### Memory Management
- **If OOM occurs:** The training script automatically retries with `IMGSZ=960 BATCH=8`
- **Manual fallback:** Set `IMGSZ=960 BATCH=8` in environment variables
- **Device detection:** Script automatically detects CUDA availability

### Troubleshooting

1. **"python: command not found"**: Always use `python3`
2. **"ModuleNotFoundError"**: Ensure virtual environment is activated
3. **CUDA errors**: Set `DEVICE=cpu` to force CPU training
4. **OOM errors**: Reduce `IMGSZ` and `BATCH` parameters

### File Recovery

If you need to restore any files moved by the cleanup script:

```bash
# Check what was moved
cat cleanup_checks_report.md

# Restore files (if needed)
bash .trash/<timestamp>/restore.sh
```

## Quick Commands Summary

```bash
# Setup (one time)
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel && pip install -r requirements.txt

# Train (main command)
chmod +x scripts/train_ps5_easy.sh
IMGSZ=1280 EPOCHS=80 BATCH=16 YOLO_MODEL=yolov10s.pt bash scripts/train_ps5_easy.sh

# Test trained model
python3 cli.py process samples/multilang_10p.pdf --out out --config configs/default.yaml
```
