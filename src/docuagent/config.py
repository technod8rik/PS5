"""Configuration management with YAML loading and environment overrides."""

import os
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load configuration from YAML file with defaults and environment overrides.
    
    Args:
        config_path: Path to YAML config file. If None, uses configs/default.yaml
        
    Returns:
        Configuration dictionary with all required keys
    """
    # Default configuration
    defaults = {
        "device": "auto",
        "dpi": 220,
        "layout": {
            "model": "pp_doclayout_l",
            "conf_threshold": 0.25,
            "classes": ["Text", "Title", "List", "Table", "Figure"]
        },
        "ocr": {
            "lang": "ml"
        },
        "langid": {
            "model_path": ".cache/lid.176.bin"
        },
        "vlm": {
            "model": "Qwen/Qwen2-VL-7B-Instruct"
        },
        "describe": {
            "batch_size": 2
        },
        "io": {
            "cache_dir": ".cache",
            "debug_dir": "./debug"
        },
        "runtime": {
            "num_workers": 4
        }
    }
    
    # Load from YAML if provided
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f) or {}
                # Deep merge with defaults
                defaults = _deep_merge(defaults, yaml_config)
    
    # Apply environment overrides
    env_overrides = {
        "device": os.getenv("DOCUAGENT_DEVICE"),
        "dpi": os.getenv("DOCUAGENT_DPI"),
        "layout.model": os.getenv("DOCUAGENT_LAYOUT_MODEL"),
        "layout.conf_threshold": os.getenv("DOCUAGENT_LAYOUT_CONF_THRESHOLD"),
        "ocr.lang": os.getenv("DOCUAGENT_OCR_LANG"),
        "langid.model_path": os.getenv("DOCUAGENT_LANGID_MODEL_PATH"),
        "vlm.model": os.getenv("DOCUAGENT_VLM_MODEL"),
        "describe.batch_size": os.getenv("DOCUAGENT_DESCRIBE_BATCH_SIZE"),
        "io.cache_dir": os.getenv("DOCUAGENT_CACHE_DIR"),
        "io.debug_dir": os.getenv("DOCUAGENT_DEBUG_DIR"),
        "runtime.num_workers": os.getenv("DOCUAGENT_NUM_WORKERS")
    }
    
    for key, value in env_overrides.items():
        if value is not None:
            _set_nested_value(defaults, key, value)
    
    # Auto-detect device if set to "auto"
    if defaults["device"] == "auto":
        try:
            import torch
            defaults["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            defaults["device"] = "cpu"
    
    # Convert string numbers to appropriate types
    if isinstance(defaults.get("dpi"), str):
        defaults["dpi"] = int(defaults["dpi"])
    if isinstance(defaults.get("layout", {}).get("conf_threshold"), str):
        defaults["layout"]["conf_threshold"] = float(defaults["layout"]["conf_threshold"])
    if isinstance(defaults.get("describe", {}).get("batch_size"), str):
        defaults["describe"]["batch_size"] = int(defaults["describe"]["batch_size"])
    if isinstance(defaults.get("runtime", {}).get("num_workers"), str):
        defaults["runtime"]["num_workers"] = int(defaults["runtime"]["num_workers"])
    
    return defaults


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_nested_value(d: Dict[str, Any], key: str, value: Any) -> None:
    """Set a nested dictionary value using dot notation."""
    keys = key.split('.')
    current = d
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


def write_lock(cfg: Dict[str, Any], lock_path: Union[str, Path]) -> str:
    """Write lock configuration and return SHA256 hash.
    
    Args:
        cfg: Configuration dictionary
        lock_path: Path to save lock file
        
    Returns:
        SHA256 hash of the lock file
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create lock configuration
    lock_config = {
        "dataset": {
            "classes": cfg.get("layout", {}).get("classes", ["Text", "Title", "List", "Table", "Figure"]),
            "imgsz": cfg.get("train", {}).get("imgsz", 960),
            "schema": "yolo"
        },
        "training": {
            "backend": cfg.get("train", {}).get("backend", "yolov10"),
            "model": cfg.get("train", {}).get("yolo_model", "yolov10n.pt"),
            "epochs": cfg.get("train", {}).get("epochs", 60),
            "batch": cfg.get("train", {}).get("batch", 16),
            "lr0": cfg.get("train", {}).get("lr0", 0.01),
            "weight_decay": 0.0005,
            "momentum": 0.937,
            "warmup_epochs": 3,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1
        },
        "detection": {
            "conf_thr": cfg.get("train", {}).get("conf_thr", 0.25),
            "iou_nms": cfg.get("train", {}).get("iou_nms", 0.5),
            "max_det": 300,
            "agnostic_nms": False
        },
        "augmentation": {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0,
            "copy_paste": 0.0
        },
        "seeds": {
            "python_hash_seed": 0,
            "numpy_seed": 42,
            "torch_seed": 42,
            "paddle_seed": 42,
            "random_seed": 42
        },
        "paths": {
            "data_root": cfg.get("train", {}).get("data_root", "./data"),
            "weights_dir": cfg.get("paths", {}).get("weights_dir", "./weights"),
            "runs_dir": cfg.get("paths", {}).get("runs_dir", "./runs"),
            "debug_dir": cfg.get("paths", {}).get("debug_dir", "./debug"),
            "cache_dir": cfg.get("io", {}).get("cache_dir", ".cache")
        },
        "environment": {
            "device": cfg.get("device", "auto"),
            "num_workers": cfg.get("runtime", {}).get("num_workers", 4),
            "pin_memory": True,
            "deterministic": True
        },
        "evaluation": {
            "val_every": cfg.get("train", {}).get("val_every", 1),
            "save_period": 10,
            "plots": True,
            "save_json": True,
            "save_hybrid": False,
            "verbose": True
        },
        "logging": {
            "project": "docuagent",
            "name": "layout_detection",
            "exist_ok": True,
            "resume": False,
            "save_dir": "./runs",
            "log_level": "INFO"
        }
    }
    
    # Write lock file
    with open(lock_path, 'w') as f:
        yaml.dump(lock_config, f, default_flow_style=False, sort_keys=False)
    
    # Calculate SHA256 hash
    with open(lock_path, 'rb') as f:
        content = f.read()
        sha256_hash = hashlib.sha256(content).hexdigest()
    
    # Update lock file with hash
    with open(lock_path, 'r') as f:
        content = f.read()
    
    # Replace placeholder with actual hash
    content = content.replace("# SHA256: [will be calculated and printed during training]", f"# SHA256: {sha256_hash}")
    
    with open(lock_path, 'w') as f:
        f.write(content)
    
    return sha256_hash


def set_deterministic_seeds(lock_config: Dict[str, Any]) -> None:
    """Set all random seeds for reproducibility.
    
    Args:
        lock_config: Lock configuration dictionary
    """
    seeds = lock_config.get("seeds", {})
    
    # Set Python hash seed
    if "python_hash_seed" in seeds:
        os.environ["PYTHONHASHSEED"] = str(seeds["python_hash_seed"])
    
    # Set NumPy seed
    try:
        import numpy as np
        np.random.seed(seeds.get("numpy_seed", 42))
    except ImportError:
        pass
    
    # Set PyTorch seed
    try:
        import torch
        torch.manual_seed(seeds.get("torch_seed", 42))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seeds.get("torch_seed", 42))
            torch.cuda.manual_seed_all(seeds.get("torch_seed", 42))
    except ImportError:
        pass
    
    # Set PaddlePaddle seed
    try:
        import paddle
        paddle.seed(seeds.get("paddle_seed", 42))
    except ImportError:
        pass
    
    # Set Python random seed
    import random
    random.seed(seeds.get("random_seed", 42))
