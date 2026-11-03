"""
conftest.py — Shared pytest configuration for the test cases directory.
"""
import sys
from pathlib import Path

# Ensure the project root is on sys.path for all tests in this directory
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
