#!/usr/bin/env python3
"""
Safe cleaner for old "check" files and debug artifacts.
Moves candidates to .trash/<timestamp>/ instead of deleting them.
"""

import argparse
import os
import shutil
import time
from pathlib import Path
from typing import List, Set


def get_candidates() -> List[Path]:
    """Find candidate files to move (non-destructive)."""
    repo_root = Path.cwd()
    candidates = []
    
    # Hard whitelist - never move these
    whitelist = {
        "scripts/ingest_all_ps5.py",
        "scripts/train_ps5_easy.sh", 
        "scripts/train_ps5.sh",
        "scripts/install_torch.sh",
        "scripts/install_fasttext_fallback.sh",
        "src",
        "configs", 
        "cli.py",
        "README.md",
        "LICENSE",
        ".github",
        "Makefile",
        "Dockerfile",
        "requirements.txt",
        "pyproject.toml",
        "weights",
        "runs",
        "data"
    }
    
    # Pattern matching for candidates
    patterns = [
        # Scripts with check/debug in name
        "scripts/*check*.py",
        "scripts/*healthcheck*.py", 
        "scripts/*preupload*check*.py",
        "scripts/*probe*.py",
        "scripts/*debug*.py",
        # Top-level check files
        "*check*.py",
        "*healthcheck*.py",
        # Jupyter checkpoints and Python caches
        "**/.ipynb_checkpoints/**",
        "**/__pycache__/**",
        "**/*.pyc"
    ]
    
    for pattern in patterns:
        if "**" in pattern:
            # Use glob for recursive patterns
            for path in repo_root.glob(pattern):
                if path.is_file() or path.is_dir():
                    # Check if any parent matches whitelist
                    if not any(str(path).startswith(str(repo_root / w)) for w in whitelist):
                        candidates.append(path)
        else:
            # Use glob for simple patterns
            for path in repo_root.glob(pattern):
                if path.is_file() or path.is_dir():
                    # Check if path matches whitelist
                    rel_path = path.relative_to(repo_root)
                    if str(rel_path) not in whitelist:
                        candidates.append(path)
    
    return sorted(set(candidates))


def generate_report(candidates: List[Path]) -> str:
    """Generate markdown report of candidates."""
    report = ["# Cleanup Checks Report", "", f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    
    if not candidates:
        report.append("No candidate files found.")
        return "\n".join(report)
    
    report.append(f"Found {len(candidates)} candidate files/directories:")
    report.append("")
    
    for i, path in enumerate(candidates, 1):
        rel_path = path.relative_to(Path.cwd())
        reason = "check/debug pattern" if any(x in str(path).lower() for x in ["check", "debug", "probe"]) else "cache/checkpoint"
        report.append(f"{i}. `{rel_path}` - {reason}")
    
    return "\n".join(report)


def create_restore_script(candidates: List[Path], trash_dir: Path) -> str:
    """Create restore script to undo the moves."""
    script_lines = [
        "#!/bin/bash",
        "# Restore script for cleanup_checks.py",
        f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "set -e",
        "",
        "echo 'Restoring files from cleanup...'"
    ]
    
    for path in candidates:
        rel_path = path.relative_to(Path.cwd())
        trash_path = trash_dir / rel_path
        script_lines.append(f"mv '{trash_path}' '{rel_path}'")
    
    script_lines.extend([
        "",
        "echo 'Restore complete!'",
        "echo 'You can now remove this restore script and the trash directory.'"
    ])
    
    return "\n".join(script_lines)


def main():
    parser = argparse.ArgumentParser(description="Clean up old check files and debug artifacts")
    parser.add_argument("--apply", action="store_true", help="Actually move files (default: report only)")
    args = parser.parse_args()
    
    print("🔍 Scanning for candidate files...")
    candidates = get_candidates()
    
    if not candidates:
        print("✅ No candidate files found.")
        return
    
    # Generate report
    report_content = generate_report(candidates)
    report_path = Path("cleanup_checks_report.md")
    report_path.write_text(report_content)
    print(f"📄 Report written to: {report_path}")
    
    if not args.apply:
        print(f"\n📋 Found {len(candidates)} candidate files:")
        for path in candidates:
            print(f"  - {path.relative_to(Path.cwd())}")
        print(f"\n💡 To move these files, run: python {__file__} --apply")
        return
    
    # Create trash directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    trash_dir = Path(f".trash/{timestamp}")
    trash_dir.mkdir(parents=True, exist_ok=True)
    print(f"🗑️  Moving files to: {trash_dir}")
    
    # Move files
    moved_count = 0
    for path in candidates:
        try:
            rel_path = path.relative_to(Path.cwd())
            dest_path = trash_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            if path.is_file():
                shutil.move(str(path), str(dest_path))
            elif path.is_dir():
                shutil.move(str(path), str(dest_path))
            
            moved_count += 1
            print(f"  ✓ Moved: {rel_path}")
        except Exception as e:
            print(f"  ✗ Failed to move {path}: {e}")
    
    # Create restore script
    restore_script_content = create_restore_script(candidates, trash_dir)
    restore_script_path = trash_dir / "restore.sh"
    restore_script_path.write_text(restore_script_content)
    restore_script_path.chmod(0o755)
    
    print(f"\n✅ Moved {moved_count} files to {trash_dir}")
    print(f"📄 Report: {report_path}")
    print(f"🔄 Restore script: {restore_script_path}")
    print(f"\n💡 To restore files: bash {restore_script_path}")


if __name__ == "__main__":
    main()
