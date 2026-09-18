#!/usr/bin/env python3
"""Permanent pre-upload gate (ADR-015, 2026-09-17 addendum): verify that a
MIND prediction-generation pipeline's per-impression history lookup
actually recovers real user history, at full scale -- not just on a
hand-sampled handful of impressions.

Why this exists: submission 930353 (MINDlarge_test, real Codabench score
0.5589 vs. 0.6868 local) shipped with every one of its 2,370,727
impressions scored on an EMPTY history, because a hand-rolled reader keyed
its query_by_user dict by the raw user_id while the lookup used the
prefixed one. The prediction file was still well-formed (a valid
permutation per impression, right line count, no duplicates) -- format
validation (`a2_validate_mind_prediction.py`) cannot see this class of
bug, because a fully history-blind run is not malformed, just wrong. A
10-impression spot check caught it only because it happened to compare
against real history explicitly; this script makes that check cheap
enough (no model, no GPU, seconds to run) to be a mandatory step before
every future prediction upload, not just this one.

Method: compute the real, non-empty-history rate TWO independent ways over
the SAME split and assert they agree:
  (a) ground truth, directly from the raw behaviors.tsv `history` column,
      with no query_by_user dict or join involved at all;
  (b) the pipeline's own path: build_mind_test -> user_history.parquet ->
      query_by_user -> joined back onto every real impression's user_id,
      the exact join a generation script performs.
A real pipeline bug in the join/keying layer moves (b) far below (a)
without touching (a) at all -- exactly what happened here (ground truth
~large majority non-cold-start; the broken pipeline's join rate was 0%).

Usage:
    python3 scripts/a2_check_mind_history_coverage.py \
        --mind-test-zip $HOME/a2/raw/MINDlarge_test.zip \
        --data-dir $HOME/a2/data_large
"""
import argparse
import sys
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

from src.pipeline.orchestrator import build_mind_test  # noqa: E402
from src.submission.mind_format import read_raw_impressions  # noqa: E402


def ground_truth_hit_rate(zip_path: Path) -> tuple[float, int, int]:
    """Directly from behaviors.tsv's own `history` column -- no parquet,
    no query_by_user, no prefix_id. One row per impression, matching
    read_raw_impressions' population exactly (same file, same rows)."""
    z = zipfile.ZipFile(zip_path)
    member = [m for m in z.namelist() if m.endswith("/behaviors.tsv") or m == "behaviors.tsv"][0]
    n_total = n_hit = 0
    with z.open(member) as f:
        for ln in f:
            r = ln.decode("utf-8").strip("\n").split("\t")
            n_total += 1
            if r[3]:  # non-empty history field
                n_hit += 1
    return n_hit / n_total, n_hit, n_total


def pipeline_hit_rate(zip_path: Path, data_dir: Path) -> tuple[float, int, int]:
    """The real generation path: build_mind_test's parquet -> query_by_user
    -> joined onto every real impression via read_raw_impressions, exactly
    as a2_generate_mind_test_predictions.py does it."""
    build_mind_test(zip_path, data_dir)
    test_history = pd.read_parquet(data_dir / "test" / "user_history.parquet")
    query_by_user = dict(zip(test_history["user_id"], test_history["article_ids"]))

    rows = read_raw_impressions(zip_path, has_labels=False)
    n_hit = sum(1 for r in rows if len(query_by_user.get(r["user_id"], [])) > 0)
    return n_hit / len(rows), n_hit, len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mind-test-zip", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="scratch dir for build_mind_test's parquet output (reused if already built)")
    ap.add_argument("--max-abs-diff", type=float, default=0.005,
                    help="max allowed |ground_truth - pipeline| hit-rate difference "
                         "before this exits non-zero (default: 0.5 percentage points)")
    args = ap.parse_args()

    gt_rate, gt_hit, gt_total = ground_truth_hit_rate(args.mind_test_zip)
    print(f"ground truth (raw behaviors.tsv):  {gt_hit:,}/{gt_total:,} ({gt_rate:.2%}) "
          f"impressions have non-empty history", flush=True)

    pl_rate, pl_hit, pl_total = pipeline_hit_rate(args.mind_test_zip, args.data_dir)
    print(f"pipeline (build_mind_test join):   {pl_hit:,}/{pl_total:,} ({pl_rate:.2%}) "
          f"impressions map to non-empty history", flush=True)

    diff = abs(gt_rate - pl_rate)
    print(f"|difference|: {diff:.4%}", flush=True)
    if gt_total != pl_total:
        sys.exit(f"FATAL: row counts disagree ({gt_total:,} vs {pl_total:,}) -- "
                 f"read_raw_impressions and the raw TSV scan saw a different number "
                 f"of impressions. Investigate before trusting either rate.")
    if diff > args.max_abs_diff:
        sys.exit(f"FATAL: pipeline hit-rate diverges from ground truth by {diff:.4%}, "
                 f"exceeding --max-abs-diff {args.max_abs_diff:.4%}. The query_by_user "
                 f"join is very likely broken (this is exactly how ADR-015's 2026-09-17 "
                 f"bug presented: ground truth was the normal MIND rate, the broken "
                 f"pipeline's rate was ~0%). Do not proceed to GPU scoring or upload.")
    print("OK: pipeline history coverage matches ground truth within tolerance.", flush=True)


if __name__ == "__main__":
    main()
