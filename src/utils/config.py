"""Central path and constant definitions for the data pipeline.

Single source of truth so no module hardcodes a path or a magic count.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MIND_RAW_DIR = RAW_DIR / "mind"
EBNERD_RAW_DIR = RAW_DIR / "ebnerd"

ID_DELIMITER = ":"

MIND_SPLITS = ("train", "dev")
EBNERD_SPLITS = ("train", "validation")

# ADR-001 empirical evidence table — hardcoded as a regression guard against
# silent parsing bugs (row counts must match what was directly inspected).
MIND_SMALL_EVIDENCE = {
    "train": {"impressions": 156_965, "users": 50_000},
    "dev": {"impressions": 73_152, "users": 50_000},
}
EBNERD_DEMO_EVIDENCE = {
    "train": {"impressions": 24_724, "users": 1_590},
    "validation": {"impressions": 25_356, "users": 1_562},
}
