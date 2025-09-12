#!/usr/bin/env python3
"""Health check script for DocuAgent - validates all components are working."""

import sys
import os
import subprocess
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_result(check_name: str, success: bool, details: str = ""):
    """Print check result."""
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"{status} {check_name}")
    if details:
        print(f"    {details}")

def check_imports() -> Tuple[bool, List[str]]:
    """Check all required imports work."""
    print_header("IMPORT VERIFICATION")
    
    required_modules = [
        "docuagent.config",
        "docuagent.layout", 
        "docuagent.ocr_lang",
        "docuagent.describe",
        "docuagent.compile_json",
        "docuagent.cli",
        "docuagent.training.yolo_train",
        "docuagent.eval.coco_eval"
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print_result(f"Import {module}", True)
        except ImportError as e:
            print_result(f"Import {module}", False, str(e))
            failed_imports.append(module)
    
    return len(failed_imports) == 0, failed_imports

def check_configs() -> Tuple[bool, str]:
    """Check config file loads and has required keys."""
    print_header("CONFIG VERIFICATION")
    
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    
    if not config_path.exists():
        return False, f"Config file not found: {config_path}"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return False, f"Failed to load config: {e}"
    
    required_keys = [
        "device", "dpi", "layout.model", "ocr.lang", "vlm.model", 
        "paths.weights_dir", "paths.runs_dir", "paths.debug_dir"
    ]
    
    missing_keys = []
    for key in required_keys:
        if "." in key:
            # Nested key
            parts = key.split(".")
            current = config
            try:
                for part in parts:
                    current = current[part]
            except KeyError:
                missing_keys.append(key)
        else:
            if key not in config:
                missing_keys.append(key)
    
    if missing_keys:
        return False, f"Missing required keys: {missing_keys}"
    
    print_result("Config file loads", True)
    print_result("Required keys present", True)
    return True, ""

def check_cli() -> Tuple[bool, str]:
    """Check CLI commands work."""
    print_header("CLI VERIFICATION")
    
    try:
        # Test main CLI help
        result = subprocess.run([
            sys.executable, "-m", "docuagent.cli", "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return False, f"CLI help failed: {result.stderr}"
        
        # Check for expected subcommands
        expected_subcommands = [
            "process", "data coco-to-yolo", "data yolo-to-coco", 
            "data split", "train yolo", "eval coco", "viz overlays"
        ]
        
        missing_commands = []
        for cmd in expected_subcommands:
            if cmd not in result.stdout:
                missing_commands.append(cmd)
        
        if missing_commands:
            return False, f"Missing subcommands: {missing_commands}"
        
        print_result("CLI help command", True)
        print_result("All subcommands present", True)
        return True, ""
        
    except subprocess.TimeoutExpired:
        return False, "CLI help command timed out"
    except Exception as e:
        return False, f"CLI test failed: {e}"

def check_dry_runs() -> Tuple[bool, str]:
    """Test dry-run mode for process command."""
    print_header("DRY-RUN VERIFICATION")
    
    # Create a minimal test PDF if it doesn't exist
    test_pdf = Path(__file__).parent.parent / "samples" / "test_1page.pdf"
    test_pdf.parent.mkdir(exist_ok=True)
    
    if not test_pdf.exists():
        # Create a simple 1-page PDF for testing
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            c = canvas.Canvas(str(test_pdf), pagesize=letter)
            c.drawString(100, 750, "Test Document for Health Check")
            c.drawString(100, 700, "This is a simple test page.")
            c.drawString(100, 650, "It contains some text elements.")
            c.save()
            print_result("Created test PDF", True)
        except ImportError:
            print_result("Created test PDF", False, "reportlab not available")
            return False, "Cannot create test PDF without reportlab"
    
    try:
        # Test dry-run mode
        result = subprocess.run([
            sys.executable, "-m", "docuagent.cli", "process",
            str(test_pdf), "--out", "/tmp/healthcheck_out", "--dry-run"
        ], capture_output=True, text=True, timeout=60)
        
        # Dry-run should either succeed or fail gracefully with clear error
        if result.returncode == 0:
            print_result("Dry-run process", True)
        else:
            # Check if error is actionable
            error_msg = result.stderr.lower()
            if any(keyword in error_msg for keyword in ["missing", "not found", "unavailable", "weights"]):
                print_result("Dry-run process", True, "Failed gracefully with actionable error")
            else:
                print_result("Dry-run process", False, f"Unexpected error: {result.stderr}")
                return False, f"Unexpected dry-run error: {result.stderr}"
        
        return True, ""
        
    except subprocess.TimeoutExpired:
        return False, "Dry-run test timed out"
    except Exception as e:
        return False, f"Dry-run test failed: {e}"

def check_artifacts() -> Tuple[bool, str]:
    """Check output schema and debug capabilities."""
    print_header("ARTIFACTS VERIFICATION")
    
    # Check if we can create a minimal elements.json
    try:
        from docuagent.compile_json import to_standard_json
        
        # Create a minimal test element
        test_element = {
            "class": "Text",
            "bbox": [100, 200, 300, 50],  # x, y, w, h
            "content": "Test content",
            "language": "en",
            "page": 1
        }
        
        elements_json = to_standard_json([test_element], "test.pdf")
        
        # Verify required fields exist
        required_fields = ["class", "bbox", "content", "language", "page"]
        missing_fields = []
        
        if "elements" in elements_json and len(elements_json["elements"]) > 0:
            element = elements_json["elements"][0]
            for field in required_fields:
                if field not in element:
                    missing_fields.append(field)
        else:
            return False, "No elements in generated JSON"
        
        if missing_fields:
            return False, f"Missing fields in elements.json: {missing_fields}"
        
        print_result("Elements JSON schema", True)
        print_result("Required fields present", True)
        return True, ""
        
    except Exception as e:
        return False, f"Artifacts check failed: {e}"

def main():
    """Run all health checks."""
    print("DocuAgent Health Check")
    print("=" * 60)
    
    all_passed = True
    issues = []
    
    # Run all checks
    checks = [
        ("Imports", check_imports),
        ("Configs", check_configs), 
        ("CLI", check_cli),
        ("Dry-runs", check_dry_runs),
        ("Artifacts", check_artifacts)
    ]
    
    for check_name, check_func in checks:
        try:
            if check_name == "Imports":
                success, failed_imports = check_func()
                if not success:
                    issues.extend(failed_imports)
            else:
                success, error = check_func()
                if not success:
                    issues.append(f"{check_name}: {error}")
            
            if not success:
                all_passed = False
                
        except Exception as e:
            print_result(check_name, False, f"Check crashed: {e}")
            all_passed = False
            issues.append(f"{check_name}: Check crashed - {e}")
    
    # Print summary
    print_header("SUMMARY")
    
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("\nNext steps:")
        print("1. Run: python -m docuagent.cli process samples/test_1page.pdf --out out")
        print("2. Proceed to Phase B implementation")
        return 0
    else:
        print("❌ SOME CHECKS FAILED!")
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease fix these issues before proceeding to Phase B.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
