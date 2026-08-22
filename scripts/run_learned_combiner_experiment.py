#!/usr/bin/env python3
"""Candidate F (2026-08-21 second session, see PROJECT_STATE.md's "Current
Objective"): does a supervised logistic-regression combiner over real,
diverse features beat the deployed MIND embedding baseline (AUC 0.6340,
95% CI 0.6319-0.6361) on MINDsmall-dev? Genuinely different mechanism from
every earlier candidate — instead of a hand-picked blend weight (Candidate
B's untuned 50/50) or a single alternative signal (A, D, E), this *learns*
how much to trust each signal from MINDsmall-train's real click labels.

Seven features, one row per (impression, candidate):

| Feature | Source | Why |
|---|---|---|
| `bm25` | `BM25Scorer` (ADR-006) | lexical relevance |
| `embed_cos` | `EmbeddingScorer` (ADR-008), unweighted mean-pooled query | semantic relevance |
| `recency_embed_cos` | `build_user_embedding_query_recency`, decay=0.9 (Candidate C) | recency, as an ADDITIONAL feature this time, not a full replacement — Candidate C's own "Conditions for Revisiting" named exactly this as the next thing worth trying |
| `category_match` | `features.py`, fraction of history sharing the candidate's category | symbolic identity signal (Candidate D) |
| `subcategory_match` | `features.py` | symbolic identity signal (Candidate D) |
| `entity_overlap_log1p` | `features.py`, log1p of weighted entity overlap | symbolic identity signal (Candidate D) |
| `log_popularity` | `build_train_popularity` (ADR-007/009, train-split only) | item-side prior (Candidate E) |

Every feature is itself something already tried and lost standalone (A/B/C)
or is being tried standalone in this same session (D/E) — the question
this candidate asks is specifically whether a *learned* combination beats
every untuned one, not whether any single new signal exists.

Trained on MINDsmall-**train** (`StandardScaler` + `LogisticRegression`,
`lbfgs`, untuned defaults — no hidden hyperparameter search, same "no
hidden tuning" stance ADR-009 states explicitly), never touching dev
labels; evaluated on MINDsmall-**dev** via the same Q4 harness/bootstrap
CI every other candidate uses. Train-side and dev-side each get their own
BM25/embedding index built from *that split's own* article catalog (same
convention every other script in this project follows) — only the fitted
linear weights transfer from train to dev, not any index or vector.

Fitted coefficients are written to `config.json` in full (not just
"trust the pickle") so the exact scoring function is reproducible from the
numbers alone, consistent with this project's reproducibility standard.

Usage:
    poetry run python scripts/run_learned_combiner_experiment.py
"""
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from run_bm25_experiment import COLD_THRESHOLD, dataset_paths
from run_ranking_eval import _train_impressions_path
from src.evaluation.ranking_metrics import (
    build_train_popularity,
    mrr,
    ndcg_at_k,
    rank_candidates,
    ranking_metric_ci,
    safe_auc,
)
from src.retrieval.embed import (
    DEFAULT_MODEL,
    build_embedding_index,
    build_user_embedding_query,
    build_user_embedding_query_recency,
    model_slug,
)
from src.retrieval.features import (
    HistoryProfile,
    build_history_profile,
    build_lookup_tables,
    category_match_score,
    entity_overlap_count,
    subcategory_match_score,
)
from src.retrieval.index import build_index
from src.retrieval.query import build_user_query
from src.retrieval.score import BM25Scorer, EmbeddingScorer
from src.utils.config import PROCESSED_DIR

N_BOOTSTRAP = 2000
SEED = 0
DECAY = 0.9  # matches Candidate C's already-tested value, not re-searched
METRIC_COLUMNS = ["auc", "mrr", "ndcg5", "ndcg10"]
BASELINE_AUC_OVERALL = 0.63399337223849
FEATURE_NAMES = [
    "bm25", "embed_cos", "recency_embed_cos",
    "category_match", "subcategory_match", "entity_overlap_log1p", "log_popularity",
]


def _train_paths(bundle: str) -> dict:
    base = PROCESSED_DIR / "mind" / bundle / "train"
    return {"articles": base / "articles.parquet", "history": base / "user_history.parquet",
            "impressions": base / "impressions.parquet"}


def _build_side(paths: dict) -> dict:
    """Everything needed to compute per-impression feature matrices for one
    split: that split's own BM25/embedding indices, per-user queries
    (BM25/embed/recency-embed), symbolic history profiles, and lookups."""
    articles = pd.read_parquet(paths["articles"])
    history = pd.read_parquet(paths["history"])
    impressions = pd.read_parquet(paths["impressions"])

    bm25_scorer = BM25Scorer(build_index(articles))

    cache_path = paths["articles"].parent / "embeddings" / f"{model_slug(DEFAULT_MODEL)}.npy"
    embed_index = build_embedding_index(articles, model_name=DEFAULT_MODEL, cache_path=cache_path)
    embed_scorer = EmbeddingScorer(embed_index)
    recency_scorer = EmbeddingScorer(embed_index)  # separate instance -> separate identity cache

    text_lookup = dict(zip(articles["article_id"], articles["title"].fillna("") + " " + articles["abstract"].fillna("")))
    vector_lookup = dict(zip(embed_index.article_ids, embed_index.vectors))
    category_lookup, subcategory_lookup, entity_set_lookup = build_lookup_tables(articles)

    bm25_query_by_user, embed_query_by_user, recency_query_by_user = {}, {}, {}
    profile_by_user, history_len_by_user = {}, {}
    for row in history.itertuples(index=False):
        bm25_query_by_user[row.user_id] = build_user_query(row.article_ids, text_lookup)
        embed_query_by_user[row.user_id] = build_user_embedding_query(row.article_ids, vector_lookup)
        recency_query_by_user[row.user_id] = build_user_embedding_query_recency(
            row.article_ids, vector_lookup, decay=DECAY
        )
        profile_by_user[row.user_id] = build_history_profile(
            row.article_ids, category_lookup, subcategory_lookup, entity_set_lookup
        )
        history_len_by_user[row.user_id] = len(row.article_ids)

    return {
        "impressions": impressions, "bm25_scorer": bm25_scorer, "embed_scorer": embed_scorer,
        "recency_scorer": recency_scorer, "category_lookup": category_lookup,
        "subcategory_lookup": subcategory_lookup, "entity_set_lookup": entity_set_lookup,
        "bm25_query_by_user": bm25_query_by_user, "embed_query_by_user": embed_query_by_user,
        "recency_query_by_user": recency_query_by_user, "profile_by_user": profile_by_user,
        "history_len_by_user": history_len_by_user, "n_articles": len(articles),
    }


def _impression_feature_matrix(
    candidate_ids: list[str], bm25_scores, embed_scores, recency_scores,
    profile: HistoryProfile, category_lookup, subcategory_lookup, entity_set_lookup, popularity,
) -> np.ndarray:
    k = len(candidate_ids)
    cat = np.fromiter(
        (category_match_score(category_lookup.get(c), profile) for c in candidate_ids), dtype=np.float64, count=k
    )
    subcat = np.fromiter(
        (subcategory_match_score(subcategory_lookup.get(c), profile) for c in candidate_ids), dtype=np.float64, count=k
    )
    ent = np.fromiter(
        (np.log1p(entity_overlap_count(entity_set_lookup.get(c, frozenset()), profile)) for c in candidate_ids),
        dtype=np.float64, count=k,
    )
    pop = np.fromiter((np.log(popularity[c]) for c in candidate_ids), dtype=np.float64, count=k)
    return np.column_stack([bm25_scores, embed_scores, recency_scores, cat, subcat, ent, pop])


def _build_training_matrix(side: dict, popularity: dict) -> tuple[np.ndarray, np.ndarray]:
    impressions_sorted = side["impressions"].sort_values(["user_id", "impression_id"], kind="stable")
    n = len(impressions_sorted)
    X = np.empty((n, len(FEATURE_NAMES)), dtype=np.float64)
    y = np.empty(n, dtype=bool)
    empty_profile = HistoryProfile()
    pos = 0
    for (user_id, impression_id), group in impressions_sorted.groupby(["user_id", "impression_id"], sort=False):
        candidate_ids = group["article_id"].tolist()
        k = len(candidate_ids)
        clicked = group["clicked"].to_numpy(dtype=bool)
        bm25q = side["bm25_query_by_user"].get(user_id, [])
        embedq = side["embed_query_by_user"].get(user_id, None)
        recq = side["recency_query_by_user"].get(user_id, None)
        profile = side["profile_by_user"].get(user_id, empty_profile)

        bm25_scores = side["bm25_scorer"].score(bm25q, candidate_ids)
        embed_scores = side["embed_scorer"].score(embedq, candidate_ids)
        recency_scores = side["recency_scorer"].score(recq, candidate_ids)

        X[pos:pos + k] = _impression_feature_matrix(
            candidate_ids, bm25_scores, embed_scores, recency_scores, profile,
            side["category_lookup"], side["subcategory_lookup"], side["entity_set_lookup"], popularity,
        )
        y[pos:pos + k] = clicked
        pos += k
    assert pos == n
    return X, y


def _evaluate(side: dict, popularity: dict, scaler_mean, scaler_scale, coef, intercept, cold_threshold: int) -> pd.DataFrame:
    impressions_sorted = side["impressions"].sort_values(["user_id", "impression_id"], kind="stable")
    cohort_by_user = {
        uid: ("cold" if hlen < cold_threshold else "warm")
        for uid, hlen in side["history_len_by_user"].items()
    }
    empty_profile = HistoryProfile()
    rows = []
    for (user_id, impression_id), group in impressions_sorted.groupby(["user_id", "impression_id"], sort=False):
        candidate_ids = group["article_id"].tolist()
        clicked = group["clicked"].to_numpy(dtype=bool)
        cohort = cohort_by_user.get(user_id, "cold")
        bm25q = side["bm25_query_by_user"].get(user_id, [])
        embedq = side["embed_query_by_user"].get(user_id, None)
        recq = side["recency_query_by_user"].get(user_id, None)
        profile = side["profile_by_user"].get(user_id, empty_profile)

        bm25_scores = side["bm25_scorer"].score(bm25q, candidate_ids)
        embed_scores = side["embed_scorer"].score(embedq, candidate_ids)
        recency_scores = side["recency_scorer"].score(recq, candidate_ids)

        X = _impression_feature_matrix(
            candidate_ids, bm25_scores, embed_scores, recency_scores, profile,
            side["category_lookup"], side["subcategory_lookup"], side["entity_set_lookup"], popularity,
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
        })
    return pd.DataFrame(rows)


def run(bundle: str = "small") -> tuple[dict, dict]:
    dev_paths = dataset_paths("mind", bundle)
    train_paths = _train_paths(bundle)

    t0 = time.time()
    train_side = _build_side(train_paths)
    dev_side = _build_side(dev_paths)
    build_s = time.time() - t0

    train_impressions_for_popularity = pd.read_parquet(_train_impressions_path("mind", bundle))
    popularity = build_train_popularity(
        train_impressions_for_popularity, n_catalog=train_side["n_articles"], alpha=1.0
    )

    t0 = time.time()
    X_train, y_train = _build_training_matrix(train_side, popularity)
    feature_build_s = time.time() - t0

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=0)
    t0 = time.time()
    model.fit(X_train_scaled, y_train)
    fit_s = time.time() - t0

    coef = model.coef_[0]
    intercept = float(model.intercept_[0])

    t0 = time.time()
    per_impression = _evaluate(
        dev_side, popularity, scaler.mean_, scaler.scale_, coef, intercept, COLD_THRESHOLD
    )
    eval_s = time.time() - t0

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

    n_cold_users = sum(1 for c in dev_side["history_len_by_user"].values() if c < COLD_THRESHOLD)
    config = {
        "dataset": "mind",
        "bundle": bundle,
        "method": "learned_combiner_logreg",
        "candidate": "F",
        "feature_names": FEATURE_NAMES,
        "recency_decay": DECAY,
        "n_train_rows": int(len(y_train)),
        "n_train_impressions": int(train_side["impressions"]["impression_id"].nunique()),
        "train_click_rate": float(y_train.mean()),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": coef.tolist(),
        "intercept": intercept,
        "cold_threshold": COLD_THRESHOLD,
        "n_bootstrap": N_BOOTSTRAP,
        "n_users": len(dev_side["history_len_by_user"]),
        "n_warm_users": len(dev_side["history_len_by_user"]) - n_cold_users,
        "n_cold_users": n_cold_users,
        "n_impressions": len(per_impression),
        "index_build_seconds": round(build_s, 2),
        "feature_build_seconds": round(feature_build_s, 2),
        "fit_seconds": round(fit_s, 2),
        "eval_seconds": round(eval_s, 2),
        "baseline_auc_overall_embed": BASELINE_AUC_OVERALL,
        "date": str(date.today()),
    }
    results = {"metrics": metric_results}
    return config, results


if __name__ == "__main__":
    config, results = run()

    out_dir = _REPO_ROOT / "experiments" / f"candidate_f_learned_combiner_mind_small_{date.today().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"Wrote {out_dir}/config.json and results.json")
    print(json.dumps({k: v for k, v in config.items() if k not in ("scaler_mean", "scaler_scale")}, indent=2))
    print("\nfitted coefficients (standardized features):")
    for name, c in zip(FEATURE_NAMES, config["coefficients"]):
        print(f"  {name}: {c:+.4f}")
    for metric, slices in results["metrics"].items():
        print(f"\n{metric}")
        for name, r in slices.items():
            print(f"  {name}: {r['metric']:.4f} (95% CI: {r['ci_low']:.4f}-{r['ci_high']:.4f}), "
                  f"n_impressions={r['n_impressions']}, n_users={r['n_users']}, n_skipped={r['n_skipped']}")
