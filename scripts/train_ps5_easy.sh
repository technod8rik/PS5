#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# PS5 Dataset Training Script
# Handles Python/venv bootstrap, dataset validation, training with OOM fallback
# =============================================================================

# Config (env overrides supported)
DATA_YAML="data/ps5_ingested/yolo/data.yaml"
if [ -f "data/ps5_clean/yolo/data.yaml" ]; then
  DATA_YAML="data/ps5_clean/yolo/data.yaml"
fi

IMGSZ=${IMGSZ:-1280}  # set IMGSZ=960 if VRAM tight
EPOCHS=${EPOCHS:-80}
BATCH=${BATCH:-16}
YOLO_MODEL=${YOLO_MODEL:-yolov10s.pt}
PROJECT=${PROJECT:-runs/doclayout}
RUN_NAME=${RUN_NAME:-ps5_seed}

# --- Device detection (env override respected)
if [ -z "${DEVICE:-}" ] || [ "${DEVICE:-}" = "auto" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    # If torch is available, count CUDA devices
    CUDA_COUNT=$(python3 - <<'PY' 2>/dev/null || echo 0
import sys
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)
    if [ "${CUDA_COUNT:-0}" -ge 1 ]; then
      DEVICE="0"   # default to first GPU
    else
      DEVICE="cpu"
    fi
  else
    DEVICE="cpu"
  fi
fi
echo "[info] Using DEVICE=$DEVICE"

echo "===================================================="
echo "PS5 Dataset Training Script"
echo "===================================================="

# =============================================================================
# Python/venv bootstrap (idempotent)
# =============================================================================

echo "[bootstrap] Detecting Python..."
if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ python3 not found. Please install with:"
  echo "   sudo apt-get install -y python3 python3-venv python3-pip"
  exit 1
fi

echo "[bootstrap] Setting up virtual environment..."
if [ ! -d ".venv" ]; then
  echo "[bootstrap] Creating .venv..."
  python3 -m venv .venv
fi

echo "[bootstrap] Activating .venv..."
source .venv/bin/activate

echo "[bootstrap] Upgrading pip..."
pip install -U pip wheel

echo "[bootstrap] Installing dependencies..."
if [ -f "requirements.txt" ]; then
  echo "[bootstrap] Installing from requirements.txt..."
  pip install -r requirements.txt
else
  echo "[bootstrap] Installing minimal dependencies..."
  pip install -U ultralytics opencv-python tqdm ruamel.yaml PyYAML
fi

echo "[bootstrap] Verifying ultralytics..."
python3 - <<'PY'
import ultralytics, sys
print('ultralytics', ultralytics.__version__)
PY

# =============================================================================
# Dataset sanity printouts
# =============================================================================

echo ""
echo "===================================================="
echo "Dataset Validation"
echo "===================================================="

echo "USING DATA_YAML: $DATA_YAML"
echo ""
echo "Data YAML contents:"
sed -n '1,40p' "$DATA_YAML"

echo ""
echo "Counting images..."
TRAIN_DIR="$(dirname "$DATA_YAML")/images/train"
VAL_DIR="$(dirname "$DATA_YAML")/images/val"
TRAIN_N=$(find "$TRAIN_DIR" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l || true)
VAL_N=$(find "$VAL_DIR" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l || true)
echo "[info] images: train=${TRAIN_N}  val=${VAL_N}  (expected ~3600 / ~400)"

# =============================================================================
# Optional dry-run (non-fatal if missing)
# =============================================================================

echo ""
echo "===================================================="
echo "Dry-run Validation"
echo "===================================================="

if python3 cli.py train dry-run --help >/dev/null 2>&1; then
  echo "[dry-run] Running validation..."
  python3 cli.py train dry-run --coco data/ps5_ingested/coco/train.json --images "$(dirname "$DATA_YAML")/images" || true
else
  echo "[dry-run] Skipping (dry-run command not available)"
fi

# =============================================================================
# Training (with OOM fallback) and logging
# =============================================================================

echo ""
echo "===================================================="
echo "Training Phase"
echo "===================================================="

mkdir -p "$PROJECT/$RUN_NAME"
LOGFILE="$PROJECT/$RUN_NAME/train.log"
echo "[train] imgsz=$IMGSZ batch=$BATCH epochs=$EPOCHS model=$YOLO_MODEL device=$DEVICE" | tee -a "$LOGFILE"

echo "[train] Starting primary training attempt..."
echo "[train] imgsz=$IMGSZ batch=$BATCH epochs=$EPOCHS model=$YOLO_MODEL device=$DEVICE" | tee -a "$LOGFILE"
set +e
python3 cli.py train yolo \
  --data "$DATA_YAML" \
  --imgsz "$IMGSZ" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --model "$YOLO_MODEL" \
  --project "$PROJECT" \
  --name "$RUN_NAME" \
  --device "$DEVICE" 2>&1 | tee -a "$LOGFILE"
RC=${PIPESTATUS[0]}
set -e

# OOM/backoff strategy
if [ "$RC" -ne 0 ]; then
  echo ""
  echo "[warn] Training failed (likely OOM). Retrying smaller..."
  IMGSZ=960
  BATCH=8
  RETRY_NAME="${RUN_NAME}_retry"
  RETRY_LOGFILE="$PROJECT/$RETRY_NAME/train.log"
  mkdir -p "$PROJECT/$RETRY_NAME"
  
  echo "[train] imgsz=$IMGSZ batch=$BATCH epochs=$EPOCHS model=$YOLO_MODEL device=$DEVICE (retry)" | tee -a "$RETRY_LOGFILE"
  
  set +e
  python3 cli.py train yolo \
    --data "$DATA_YAML" \
    --imgsz "$IMGSZ" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --model "$YOLO_MODEL" \
    --project "$PROJECT" \
    --name "$RETRY_NAME" \
    --device "$DEVICE" 2>&1 | tee -a "$RETRY_LOGFILE"
  RC=${PIPESTATUS[0]}
  set -e
  
  if [ "$RC" -eq 0 ]; then
    RUN_NAME="$RETRY_NAME"
    echo "[info] Retry training succeeded!"
  else
    echo "[error] Both training attempts failed. Check logs for details."
    echo "[error] Primary log: $LOGFILE"
    echo "[error] Retry log: $RETRY_LOGFILE"
  fi
fi

# =============================================================================
# Post-training: export, validate, wire weights (non-fatal if tools missing)
# =============================================================================

echo ""
echo "===================================================="
echo "Post-Training Processing"
echo "===================================================="

FINAL_NAME="$RUN_NAME"
BEST_PT="$PROJECT/$FINAL_NAME/weights/best.pt"
LAST_PT="$PROJECT/$FINAL_NAME/weights/last.pt"

echo "[info] BEST: $BEST_PT"
echo "[info] LAST: $LAST_PT"

if [ -f "$BEST_PT" ]; then
  echo "[export] Exporting to ONNX..."
  yolo export model="$BEST_PT" format=onnx || true
  
  echo "[val] Running validation with JSON output..."
  yolo val model="$BEST_PT" data="$DATA_YAML" save_json=True project="$PROJECT" name="${FINAL_NAME}_val" || true
  
  echo "[wire] Copying weights to runtime location..."
  mkdir -p weights
  cp -f "$BEST_PT" weights/doclayout_yolov10_best.pt || true
  echo "[info] Copied best weights to weights/doclayout_yolov10_best.pt"
else
  echo "[warn] Best weights not found at $BEST_PT"
fi

# =============================================================================
# Summary + next command
# =============================================================================

echo ""
echo "===================================================="
echo "Training Summary"
echo "===================================================="
echo "Training complete (name: $FINAL_NAME)"
echo "Weights (best):  $BEST_PT"
echo "Weights (last):  $LAST_PT"
echo "Logs & metrics:  $PROJECT/$FINAL_NAME/"
echo "Val preds JSON:  $PROJECT/${FINAL_NAME}_val/"
echo "Runtime weight:  weights/doclayout_yolov10_best.pt"
echo "Smoke test:      python3 cli.py process samples/multilang_10p.pdf --out out --config configs/default.yaml"
echo "===================================================="

# Deactivate venv
deactivate || true

echo ""
echo "✅ Training script completed!"
