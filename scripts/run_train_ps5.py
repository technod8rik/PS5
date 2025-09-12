#!/usr/bin/env python3
"""
Convenience runner for PS5 training script.
Expands environment variables and spawns the bash script.
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    # Get script directory
    script_dir = Path(__file__).parent
    train_script = script_dir / "train_ps5.sh"
    
    if not train_script.exists():
        print(f"[ERROR] Training script not found: {train_script}")
        return 1
    
    # Print environment variables being used
    print("[INFO] Training configuration:")
    print(f"  IMGSZ: {os.getenv('IMGSZ', '1280')}")
    print(f"  EPOCHS: {os.getenv('EPOCHS', '80')}")
    print(f"  BATCH: {os.getenv('BATCH', '16')}")
    print(f"  YOLO_MODEL: {os.getenv('YOLO_MODEL', 'yolov10s.pt')}")
    print()
    
    # Build command
    cmd = [str(train_script)]
    
    print(f"[INFO] Running: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        # Run the training script
        result = subprocess.run(cmd, check=True)
        print("=" * 60)
        print("[INFO] Training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print("=" * 60)
        print(f"[ERROR] Training failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
