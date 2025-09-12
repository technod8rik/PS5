#!/usr/bin/env bash
set -euo pipefail

# ---- Paths
OUT_ROOT="data/ps5_ingested"
CLEAN_ROOT="data/ps5_clean"   # optional; prefer if preupload cleaned
PROJECT="runs/doclayout"
RUN_NAME="ps5_seed"

# Prefer cleaned data.yaml if it exists
DATA_YAML="$OUT_ROOT/yolo/data.yaml"
if [ -f "$CLEAN_ROOT/yolo/data.yaml" ]; then
  DATA_YAML="$CLEAN_ROOT/yolo/data.yaml"
fi

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

echo "USING DATA_YAML: $DATA_YAML"

# ---- Quick split verification (prints counts; does not fail build)
echo "[info] Counting images..."
TRAIN_N=$(find "$(dirname "$DATA_YAML")/images/train" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l || true)
VAL_N=$(find "$(dirname "$DATA_YAML")/images/val"   -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l || true)
echo "[info] images: train=${TRAIN_N}  val=${VAL_N}  (expected ~3600 / ~400)"

# ---- Dry-run to catch loader/shape issues
echo "[info] Running dry-run validation..."
python3 cli.py dry-run \
  --coco "$OUT_ROOT/coco/train.json" \
  --images "$(dirname "$DATA_YAML")/images" || true

# ---- Training hyperparams (tweak if low VRAM)
IMGSZ="${IMGSZ:-1280}"      # set IMGSZ=960 in env to reduce memory
EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-16}"
YOLO_MODEL="${YOLO_MODEL:-yolov10s.pt}"  # switch to yolov10m.pt if ample GPU

echo "[info] Starting training with:"
echo "  IMGSZ: $IMGSZ"
echo "  EPOCHS: $EPOCHS"
echo "  BATCH: $BATCH"
echo "  MODEL: $YOLO_MODEL"
echo "  PROJECT: $PROJECT"
echo "  NAME: $RUN_NAME"

python3 cli.py train yolo \
  --data "$DATA_YAML" \
  --imgsz "$IMGSZ" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --model "$YOLO_MODEL" \
  --project "$PROJECT" \
  --name "$RUN_NAME" \
  --device "$DEVICE"

BEST_PT="$PROJECT/$RUN_NAME/weights/best.pt"
LAST_PT="$PROJECT/$RUN_NAME/weights/last.pt"
echo "[info] BEST: $BEST_PT"
echo "[info] LAST: $LAST_PT"

# ---- Optional: export ONNX
echo "[info] Exporting to ONNX..."
yolo export model="$BEST_PT" format=onnx || true

# ---- Optional: evaluate on val and save JSON predictions
echo "[info] Running validation with JSON output..."
yolo val model="$BEST_PT" data="$DATA_YAML" save_json=True project="$PROJECT" name="${RUN_NAME}_val" || true

# ---- Wire weights into runtime
mkdir -p weights
cp -f "$BEST_PT" weights/doclayout_yolov10_best.pt
echo "[info] Copied best weights to weights/doclayout_yolov10_best.pt"

# ---- Print where to look
echo "===================================================="
echo "Training complete."
echo "Weights:        $BEST_PT"
echo "Logs/metrics:   $PROJECT/$RUN_NAME/"
echo "Val preds JSON: $PROJECT/${RUN_NAME}_val/"
echo "Runtime weight: weights/doclayout_yolov10_best.pt"
echo "Run pipeline:   python cli.py process samples/multilang_10p.pdf --out out --config configs/default.yaml"
echo "===================================================="
