#!/usr/bin/env python3
"""Candidate G (2026-08-21 second session, see PROJECT_STATE.md's "Current
Objective"): motivated directly by a real measurement, not speculation —
Candidate F (`run_learned_combiner_experiment.py`) lost overall but posted
a genuine CI-clear WIN on the cold cohort specifically (0.5926, 95% CI
0.5869-0.5985, vs. baseline cold 0.5737, CI 0.5682-0.5792 — the first
candidate this session to beat the baseline anywhere), while losing warm
(0.6309 vs. baseline warm 0.6439). This candidate asks the natural next
question: does routing by cohort — the deployed embedding scorer for warm
users (where it already wins), Candidate F's fitted combiner for cold
users (where it wins instead) — beat the baseline *overall*, by combining
each method's own best segment rather than picking one method for
everyone?

Reuses Candidate F's already-fitted model exactly as recorded in
`experiments/candidate_f_learned_combiner_mind_small_2026-08-21/config.json`
(scaler mean/scale, coefficients, intercept) — no retraining, since lbfgs
is deterministic and the point is to test a *routing* idea, not a new
model. Only the dev split is touched; MINDsmall-train's role here is
solely to reproduce the identical train-only popularity prior F's features
depend on (`build_train_popularity`, ADR-007/009), not to fit anything new.

Usage:
    poetry run python scripts/run_gated_cohort_experiment.py
"""
import gc
import json
import sys
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from run_bm25_experiment import COLD_THRESHOLD, dataset_paths
from run_learned_combiner_experiment import (
    FEATURE_NAMES,
    _build_side,
    _impression_feature_matrix,
    _train_paths,
)
from run_ranking_eval import _train_impressions_path
from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.ranking_metrics import (
    build_train_popularity,
    mrr,
    ndcg_at_k,
    rank_candidates,
    ranking_metric_ci,
    safe_auc,
)
from src.retrieval.features import HistoryProfile

N_BOOTSTRAP = 2000
SEED = 0
METRIC_COLUMNS = ["auc", "mrr", "ndcg5", "ndcg10"]
BASELINE_AUC_OVERALL = 0.63399337223849
CANDIDATE_F_CONFIG = (
    _REPO_ROOT / "experiments" / "candidate_f_learned_combiner_mind_small_2026-08-21" / "config.json"
)


def paired_metric_diff_ci(
    df: pd.DataFrame, col_a: str, col_b: str, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED
) -> tuple[float, float, float]:
    """Paired bootstrap for `mean(col_a) - mean(col_b)` computed over the
    SAME impressions/users (same per-user sum/count reduction
    `ranking_metric_ci` uses, so this is directly comparable to every other
    CI in this project), resampling users once per replicate and applying
    both metrics to the identical resampled users.

    This is the statistically appropriate test for "does A beat B" when A
    and B are two scores over the *same* impressions (here: gated vs.
    baseline) — it isolates the actual paired difference, whereas comparing
    two independently-computed marginal CIs (as every other candidate this
    session has been judged by) discards the shared variance between A and
    B and can show overlapping CIs even when the paired difference is real
    (or vice versa). Returns `(point_diff, ci_low, ci_high)`.
    """
    d = df.dropna(subset=[col_a, col_b])
    grouped = d.groupby("user_id")[[col_a, col_b]].agg(["sum", "count"])
    a_sum = grouped[(col_a, "sum")].to_numpy(dtype=np.float64)
    a_count = grouped[(col_a, "count")].to_numpy(dtype=np.float64)
    b_sum = grouped[(col_b, "sum")].to_numpy(dtype=np.float64)
    b_count = grouped[(col_b, "count")].to_numpy(dtype=np.float64)

    def stat_fn(idx: np.ndarray) -> float:
        a = a_sum[idx].sum() / a_count[idx].sum()
        b = b_sum[idx].sum() / b_count[idx].sum()
        return a - b

    point = (a_sum.sum() / a_count.sum()) - (b_sum.sum() / b_count.sum())
    ci_low, ci_high = bootstrap_ci(len(a_sum), stat_fn, n_bootstrap, seed)
    return float(point), ci_low, ci_high


def run(bundle: str = "small") -> tuple[dict, dict]:
    f_config = json.loads(CANDIDATE_F_CONFIG.read_text())
    assert f_config["feature_names"] == FEATURE_NAMES, "Candidate F's feature order must match this script's"
    scaler_mean = np.array(f_config["scaler_mean"])
    scaler_scale = np.array(f_config["scaler_scale"])
    coef = np.array(f_config["coefficients"])
    intercept = f_config["intercept"]

    dev_paths = dataset_paths("mind", bundle)
    train_paths = _train_paths(bundle)

    # Read only the columns this step actually needs. At MINDlarge scale
    # (83.5M exploded train rows) a bare `pd.read_parquet(...)` pulls in
    # every column, including four always-null EB-NeRD-only float64 fields
    # (`session_id`/`dwell_time`/`scroll_percentage`/`is_front_page`, per
    # ADR-002) and a full-precision `impression_time` — real, wasted memory
    # at this row count. A 2026-08-21 run without this restriction, PLUS
    # keeping the resulting frame resident for the rest of `run()` instead
    # of releasing it, drove this machine (8GB RAM) into heavy swapping —
    # caught live, not from a pre-flight estimate that turned out wrong;
    # fixed at the root here, not papered over with a bigger machine.
    train_articles_n = pq.ParquetFile(train_paths["articles"]).metadata.num_rows
    train_impressions_for_popularity = pd.read_parquet(
        _train_impressions_path("mind", bundle), columns=["article_id", "clicked"]
    )
    popularity = build_train_popularity(train_impressions_for_popularity, n_catalog=train_articles_n, alpha=1.0)
    del train_impressions_for_popularity
    gc.collect()

    t0 = time.time()
    dev_side = _build_side(dev_paths)
    build_s = time.time() - t0

    cohort_by_user = {
        uid: ("cold" if hlen < COLD_THRESHOLD else "warm")
        for uid, hlen in dev_side["history_len_by_user"].items()
    }
    empty_profile = HistoryProfile()

    impressions_sorted = dev_side["impressions"].sort_values(["user_id", "impression_id"], kind="stable")

    t0 = time.time()
    rows: list[dict] = []
    for (user_id, impression_id), group in impressions_sorted.groupby(
        ["user_id", "impression_id"], sort=False
    ):
        candidate_ids = group["article_id"].tolist()
        clicked = group["clicked"].to_numpy(dtype=bool)
        cohort = cohort_by_user.get(user_id, "cold")

        embedq = dev_side["embed_query_by_user"].get(user_id, None)
        embed_scores = dev_side["embed_scorer"].score(embedq, candidate_ids)

        if cohort == "warm":
            scores = embed_scores  # deployed baseline, unchanged for warm users
        else:
            bm25q = dev_side["bm25_query_by_user"].get(user_id, [])
            recq = dev_side["recency_query_by_user"].get(user_id, None)
            profile = dev_side["profile_by_user"].get(user_id, empty_profile)
            bm25_scores = dev_side["bm25_scorer"].score(bm25q, candidate_ids)
            recency_scores = dev_side["recency_scorer"].score(recq, candidate_ids)
            X = _impression_feature_matrix(
                candidate_ids, bm25_scores, embed_scores, recency_scores, profile,
                dev_side["category_lookup"], dev_side["subcategory_lookup"], dev_side["entity_set_lookup"], popularity,
            )
            scores = ((X - scaler_mean) / scaler_scale) @ coef + intercept

        order = rank_candidates(scores, impression_id, seed=SEED)
        ranked_clicked = clicked[order]
        rows.append({
            "impression_id": impression_id,
            "user_id": user_id,
            "cohort": cohort,
            "auc": safe_auc(scores, clicked),
            "mrr": mrr(ranked_clicked),
            "ndcg5": ndcg_at_k(ranked_clicked, 5),
            "ndcg10": ndcg_at_k(ranked_clicked, 10),
            # Baseline (embed-only) AUC for the SAME impression, computed in
            # the same pass — for warm impressions this is bit-identical to
            # "auc" above (gated == baseline there by construction); for cold
            # impressions it's the deployed baseline's own AUC. Kept
            # per-impression so a proper PAIRED bootstrap (below) can test
            # "does gating actually beat baseline," not just compare two
            # separately-computed marginal CIs, which is the wrong test for
            # two scores built from the same underlying impressions.
            "auc_baseline": safe_auc(embed_scores, clicked),
        })
    eval_s = time.time() - t0

    per_impression = pd.DataFrame(rows)

    metric_results: dict[str, dict] = {}
    for metric in METRIC_COLUMNS:
        slices = {
            "overall": per_impression,
            "warm": per_impression[per_impression["cohort"] == "warm"],
            "cold": per_impression[per_impression["cohort"] == "cold"],
        }
        metric_results[metric] = {
            name: asdict(ranking_metric_ci(df, metric, n_bootstrap=N_BOOTSTRAP, seed=SEED))
            for name, df in slices.items()
        }

    paired_point, paired_ci_low, paired_ci_high = paired_metric_diff_ci(per_impression, "auc", "auc_baseline")
    results_paired = {
        "metric": "auc_overall_gated_minus_baseline",
        "point_diff": paired_point,
        "ci_low": paired_ci_low,
        "ci_high": paired_ci_high,
        "ci_excludes_zero": bool(paired_ci_low > 0),
    }

    n_cold_users = sum(1 for c in cohort_by_user.values() if c == "cold")
    config = {
        "dataset": "mind",
        "bundle": bundle,
        "method": "cohort_gated_embed_warm_combiner_cold",
        "candidate": "G",
        "warm_method": "embed (deployed baseline, unchanged)",
        "cold_method": "candidate_f_learned_combiner (fitted coefficients reused, not retrained)",
        "source_model_config": str(CANDIDATE_F_CONFIG.relative_to(_REPO_ROOT)),
        "cold_threshold": COLD_THRESHOLD,
        "n_bootstrap": N_BOOTSTRAP,
        "n_users": len(dev_side["history_len_by_user"]),
        "n_warm_users": len(dev_side["history_len_by_user"]) - n_cold_users,
        "n_cold_users": n_cold_users,
        "n_impressions": len(per_impression),
        "index_build_seconds": round(build_s, 2),
        "eval_seconds": round(eval_s, 2),
        "baseline_auc_overall_embed": BASELINE_AUC_OVERALL,
        "date": str(date.today()),
    }
    results = {"metrics": metric_results, "paired_vs_baseline": results_paired}
    return config, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle", default="small",
        help="'small' (default, matches Candidate F's training scale) or 'large' — MINDlarge-scale "
             "verification of the SAME MINDsmall-fitted model (no retraining; see ADR-010's condition).",
    )
    args = parser.parse_args()
    config, results = run(bundle=args.bundle)

    out_dir = _REPO_ROOT / "experiments" / f"candidate_g_gated_cohort_mind_{args.bundle}_{date.today().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"Wrote {out_dir}/config.json and results.json")
    print(json.dumps(config, indent=2))
    for metric, slices in results["metrics"].items():
        print(f"\n{metric}")
        for name, r in slices.items():
            print(f"  {name}: {r['metric']:.4f} (95% CI: {r['ci_low']:.4f}-{r['ci_high']:.4f}), "
                  f"n_impressions={r['n_impressions']}, n_users={r['n_users']}, n_skipped={r['n_skipped']}")

    p = results["paired_vs_baseline"]
    print(f"\npaired bootstrap (gated - baseline, overall AUC): {p['point_diff']:+.4f} "
          f"(95% CI: {p['ci_low']:+.4f} to {p['ci_high']:+.4f}), "
          f"CI excludes zero: {p['ci_excludes_zero']}")
