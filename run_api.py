#!/usr/bin/env python3
"""
Startup script for MHRAS API server.

This script initializes and runs the FastAPI application with all
integrated components.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import run_api_server

if __name__ == "__main__":
    run_api_server()
