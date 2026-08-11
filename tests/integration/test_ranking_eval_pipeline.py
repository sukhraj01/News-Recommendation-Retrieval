"""End-to-end Q4 ranking harness: score and rank the real candidates
already listed in each impression (not top-K from the whole corpus, unlike
`test_retrieval_pipeline.py`'s recall@K), then compute AUC/MRR/nDCG/
diversity/novelty against real EB-NeRD-demo-validation data (the fast
tier — same `processed_dir` fixture as the rest of the integration suite,
no extra opt-in build needed).
"""
import json

import numpy as np
import pandas as pd

from scripts.run_bm25_experiment import COLD_THRESHOLD
from src.evaluation.ranking_metrics import (
    build_train_popularity,
    coverage,
    intra_list_diversity,
    mrr,
    ndcg_at_k,
    novelty,
    rank_candidates,
    ranking_metric_ci,
    safe_auc,
)
from src.retrieval.index import build_index
from src.retrieval.query import build_user_query
from src.retrieval.score import BM25Scorer


def _run_harness(articles, history, impressions, train_impressions):
    index = build_index(articles)
    scorer = BM25Scorer(index)
    text_lookup = dict(zip(articles["article_id"],
                            articles["title"].fillna("") + " " + articles["abstract"].fillna("")))
    category_lookup = dict(zip(articles["article_id"], articles["category"]))
    popularity = build_train_popularity(train_impressions, n_catalog=len(articles), alpha=1.0)

    query_by_user, history_len_by_user = {}, {}
    for row in history.itertuples(index=False):
        query_by_user[row.user_id] = build_user_query(row.article_ids, text_lookup)
        history_len_by_user[row.user_id] = len(row.article_ids)
    cohort_by_user = {u: ("cold" if h < COLD_THRESHOLD else "warm") for u, h in history_len_by_user.items()}

    rows = []
    top10_by_impression = []
    for (user_id, impression_id), group in impressions.sort_values(
        ["user_id", "impression_id"], kind="stable"
    ).groupby(["user_id", "impression_id"], sort=False):
        candidate_ids = group["article_id"].tolist()
        clicked = group["clicked"].to_numpy(dtype=bool)
        query = query_by_user.get(user_id, [])
        scores = scorer.score(query, candidate_ids)
        order = rank_candidates(scores, impression_id, seed=0)
        ranked_ids = [candidate_ids[i] for i in order]
        ranked_clicked = clicked[order]
        top_ids = ranked_ids[:min(10, len(ranked_ids))]
        top_cats = [category_lookup.get(a) for a in top_ids]

        rows.append({
            "impression_id": impression_id,
            "user_id": user_id,
            "cohort": cohort_by_user.get(user_id, "cold"),
            "auc": safe_auc(scores, clicked),
            "mrr": mrr(ranked_clicked),
            "ndcg5": ndcg_at_k(ranked_clicked, 5),
            "ndcg10": ndcg_at_k(ranked_clicked, 10),
            "diversity10": intra_list_diversity(top_cats),
            "novelty10": novelty(top_ids, popularity),
        })
        top10_by_impression.append(top_ids)

    return pd.DataFrame(rows), top10_by_impression, cohort_by_user


def test_ranking_harness_end_to_end_on_ebnerd_demo_validation(processed_dir):
    base = processed_dir / "ebnerd" / "demo"
    articles = pd.read_parquet(base / "articles.parquet")
    history = pd.read_parquet(base / "validation" / "user_history.parquet")
    impressions = pd.read_parquet(base / "validation" / "impressions.parquet")
    train_impressions = pd.read_parquet(base / "train" / "impressions.parquet")

    per_impression, top10_by_impression, cohort_by_user = _run_harness(
        articles, history, impressions, train_impressions
    )

    # Real EB-NeRD data has one impression_id per row-group here.
    assert len(per_impression) == impressions["impression_id"].nunique()

    # Metric ranges: AUC/nDCG/MRR in [0,1] where defined; diversity in [0,1].
    for col in ["auc", "ndcg5", "ndcg10", "mrr", "diversity10"]:
        valid = per_impression[col].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all() and (valid <= 1).all(), col

    # Novelty is a finite, non-negative self-information value (smoothing
    # prevents -log2(0) = inf; see ADR-002's train/validation article-set
    # gap this smoothing exists to handle).
    assert np.isfinite(per_impression["novelty10"]).all()
    assert (per_impression["novelty10"] >= 0).all()

    # ADR-005: ebnerd_demo has zero cold-start users by construction.
    assert set(cohort_by_user.values()) == {"warm"}
    assert (per_impression["cohort"] == "warm").all()

    # ranking_metric_ci produces a sane, JSON-serializable result.
    result = ranking_metric_ci(per_impression, "auc", n_bootstrap=200, seed=0)
    assert result.n_impressions == len(per_impression)
    assert 0 <= result.ci_low <= result.metric <= result.ci_high <= 1
    json.dumps({"auc": result.__dict__})  # must not raise

    # Coverage is well-defined and bounded.
    cov = coverage(top10_by_impression, catalog_size=len(articles))
    assert 0 < cov <= 1
