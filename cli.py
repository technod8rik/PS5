#!/usr/bin/env python3
"""Main CLI entry point for DocuAgent."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from docuagent.cli import main

if __name__ == "__main__":
    sys.exit(main())
