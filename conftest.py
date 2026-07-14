"""Make the repo root importable so `pytest` works without `python -m`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
