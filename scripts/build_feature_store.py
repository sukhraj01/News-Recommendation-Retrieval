#!/usr/bin/env python3
"""Thin CLI entrypoint for `make data` — rebuilds the entire feature store."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.download import RawDataMissingError
from src.pipeline.orchestrator import build_all
from src.pipeline.validators import SchemaValidationError

if __name__ == "__main__":
    try:
        build_all()
    except (RawDataMissingError, SchemaValidationError) as e:
        print(f"\n✗ Feature store build failed: {e}", file=sys.stderr)
        sys.exit(1)
