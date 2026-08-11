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

# MINDlarge evidence table, per Wu et al. (2020) Table 2 / Section 3.2
# (ACL Anthology 2020.acl-main.331 -- ACL2020_MIND.pdf is not present in
# this repo; the paper was fetched directly via WebFetch and Table 2/
# Section 3.2 text-extracted with pypdf, not taken from memory). "samples"
# in the paper == one row per impression_id in behaviors.tsv, matching
# this project's own "impressions" row-count convention (see
# MIND_SMALL_EVIDENCE above). Corpus-wide totals (161,013 articles;
# 1,000,000 users) are reported once for the whole dataset, not
# per-split -- MINDlarge's train/dev/test each ship their own news.tsv
# with overlapping but non-identical article sets, so only the sample
# (impression) counts are checked per-split; article/user totals are
# checked as an approximate union across all three splits.
MIND_LARGE_EVIDENCE = {
    "train": {"impressions": 2_186_683},
    "dev": {"impressions": 365_200},
    "test": {"impressions": 2_341_619},
}
MIND_LARGE_TOTAL_ARTICLES = 161_013
MIND_LARGE_TOTAL_USERS = 1_000_000
