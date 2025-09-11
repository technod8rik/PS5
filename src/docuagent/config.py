"""Configuration management with YAML loading and environment overrides."""

import os
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
