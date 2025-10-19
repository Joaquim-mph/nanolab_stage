#!/usr/bin/env python3
"""
Nanolab Pipeline CLI - Entry Point

Modern command-line interface for the nanolab data processing pipeline.

Usage:
    python nanolab-pipeline.py <command> [options]

    # Or make it executable:
    chmod +x nanolab-pipeline.py
    ./nanolab-pipeline.py <command> [options]

Commands:
    stage       - Stage raw CSV files to Parquet
    preprocess  - Segment voltage sweeps
    --help      - Show help message
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cli.main import app

if __name__ == "__main__":
    app()
