#!/usr/bin/env python3
"""
Startup script for MHRAS CLI.

This script provides command-line access to MHRAS management functions.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cli import cli

if __name__ == "__main__":
    cli()
