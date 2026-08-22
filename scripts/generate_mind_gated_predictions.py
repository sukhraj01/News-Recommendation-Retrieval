#!/usr/bin/env python3
"""Generate official-format MINDlarge_test Codabench predictions using the
cohort-gated scorer verified at MINDlarge scale (ADR-010 + its 2026-08-21
MINDlarge-verification addendum): the deployed `EmbeddingScorer` unchanged
for warm users (history length >= `COLD_THRESHOLD`), Candidate F's
already-fitted logistic-regression combiner (MINDsmall-trained, reused
exactly — not retrained here or on Kaggle) for cold users. This is the
actual second MIND submission, not a local screen — routed through
`src/submission/mind_format.py::write_predictions` unchanged, the same
module every prior real submission from this project used, so the output
format is identical by construction, not re-derived.

`GatedScorer` implements the same `Scorer` protocol
(`src/retrieval/score.py`) every other method in this project does —
`write_predictions` never needed to change to support this; only a new
`Scorer` implementation and a richer per-user query bundle (`GatedQuery`,
carrying whichever precomputed pieces that user's cohort actually needs)
were required. Warm users' `GatedQuery` skips building a BM25 query/
recency-weighted vector/symbolic profile at all (never used for warm
scoring) — real, deliberate savings across ~86% of users, not just an
unused-field convenience.

MINDlarge_test has no labels (`has_labels=False`) — Codabench's own
private ground truth scores it, same as every prior real submission.

Usage:
    poetry run python scripts/generate_mind_gated_predictions.py
"""
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import numpy as np
import pandas as pd

from run_bm25_experiment import COLD_THRESHOLD
from run_learned_combiner_experiment import FEATURE_NAMES, _impression_feature_matrix
from src.evaluation.ranking_metrics import build_train_popularity
from src.retrieval.embed import (
    DEFAULT_MODEL,
    build_embedding_index,
    build_user_embedding_query,
    build_user_embedding_query_recency,
    model_slug,
)
from src.retrieval.features import HistoryProfile, build_history_profile, build_lookup_tables
from src.retrieval.index import build_index
from src.retrieval.query import build_user_query
from src.retrieval.score import BM25Scorer, EmbeddingScorer
from src.submission.mind_format import write_predictions
from src.utils.config import MIND_RAW_DIR, PROCESSED_DIR

DECAY = 0.9  # matches Candidate C/F's already-tested value, not re-searched
CANDIDATE_F_CONFIG = (
    _REPO_ROOT / "experiments" / "candidate_f_learned_combiner_mind_small_2026-08-21" / "config.json"
)
EMPTY_PROFILE = HistoryProfile()


@dataclass
class GatedQuery:
    cohort: str  # "warm" or "cold" — decided once, per user, at query-build time
    embed_query: np.ndarray | None
    bm25_query: list  # only populated for cold users; [] for warm (never read)
    recency_query: np.ndarray | None  # only populated for cold users; None for warm
    profile: HistoryProfile  # only populated for cold users; EMPTY_PROFILE for warm


class GatedScorer:
    """Implements `src/retrieval/score.py::Scorer`'s `score(query,
    candidate_ids) -> np.ndarray` contract — the same interface
    `write_predictions` already expects from BM25Scorer/EmbeddingScorer,
    so no change to the shared submission-writing code was needed."""

    def __init__(
        self, embed_scorer, bm25_scorer, recency_scorer,
        category_lookup, subcategory_lookup, entity_set_lookup, popularity,
        scaler_mean, scaler_scale, coef, intercept,
    ):
        self.embed_scorer = embed_scorer
        self.bm25_scorer = bm25_scorer
        self.recency_scorer = recency_scorer
        self.category_lookup = category_lookup
        self.subcategory_lookup = subcategory_lookup
        self.entity_set_lookup = entity_set_lookup
        self.popularity = popularity
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale
        self.coef = coef
        self.intercept = intercept

    def score(self, query: GatedQuery, candidate_ids) -> np.ndarray:
        embed_scores = self.embed_scorer.score(query.embed_query, candidate_ids)
        if query.cohort == "warm":
            return embed_scores

        bm25_scores = self.bm25_scorer.score(query.bm25_query, candidate_ids)
        recency_scores = self.recency_scorer.score(query.recency_query, candidate_ids)

        # A candidate absent from this split's own article corpus (a real,
        # documented MIND data quirk — e.g. MINDlarge_test's "N89741", per
        # `score.py::_lookup_scores`'s own docstring) scores -inf from every
        # index-backed feature (bm25/embed/recency all share the same
        # id_to_col, so they go missing together). Combining several
        # already -inf features through this linear model's mixed-sign
        # coefficients can produce NaN (-inf * positive_coef + -inf *
        # negative_coef = -inf + +inf = NaN), not just -inf — caught via a
        # real `RuntimeWarning` on a MINDsmall-dev smoke test before this
        # ever reached the actual MINDlarge_test run. Detected explicitly
        # and forced to -inf (ranks last, deterministic — the same
        # convention `score.py` already uses for this exact situation),
        # not left to incidental NaN-sort behavior.
        missing = np.isneginf(bm25_scores) | np.isneginf(embed_scores) | np.isneginf(recency_scores)

        X = _impression_feature_matrix(
            candidate_ids, bm25_scores, embed_scores, recency_scores, query.profile,
            self.category_lookup, self.subcategory_lookup, self.entity_set_lookup, self.popularity,
        )
        with np.errstate(invalid="ignore"):
            scores = ((X - self.scaler_mean) / self.scaler_scale) @ self.coef + self.intercept
        scores[missing] = -np.inf
        return scores


def build_query_by_user(
    history: pd.DataFrame, vector_lookup, text_lookup,
    category_lookup, subcategory_lookup, entity_set_lookup,
) -> dict[str, GatedQuery]:
    query_by_user: dict[str, GatedQuery] = {}
    for row in history.itertuples(index=False):
        cohort = "cold" if len(row.article_ids) < COLD_THRESHOLD else "warm"
        query_by_user[row.user_id] = GatedQuery(
            cohort=cohort,
            embed_query=build_user_embedding_query(row.article_ids, vector_lookup),
            bm25_query=build_user_query(row.article_ids, text_lookup) if cohort == "cold" else [],
            recency_query=(
                build_user_embedding_query_recency(row.article_ids, vector_lookup, decay=DECAY)
                if cohort == "cold" else None
            ),
            profile=(
                build_history_profile(row.article_ids, category_lookup, subcategory_lookup, entity_set_lookup)
                if cohort == "cold" else EMPTY_PROFILE
            ),
        )
    return query_by_user


def main() -> None:
    base = PROCESSED_DIR / "mind" / "large" / "test"
    train_base = PROCESSED_DIR / "mind" / "large" / "train"
    zip_path = MIND_RAW_DIR / "MINDlarge_test.zip"

    articles = pd.read_parquet(base / "articles.parquet")
    history = pd.read_parquet(base / "user_history.parquet")
    print(f"corpus: {len(articles)} articles, {len(history)} users with history", flush=True)

    t0 = time.time()

    # Train-split-only popularity (ADR-007/009's anti-gaming requirement),
    # column-restricted read — the exact bug fixed this session after it
    # drove the local machine into heavy swapping on a naive full-column
    # read at this row count (see ADR-010's MINDlarge-verification addendum).
    train_articles_n = len(pd.read_parquet(train_base / "articles.parquet", columns=["article_id"]))
    train_impressions = pd.read_parquet(
        train_base / "impressions.parquet", columns=["article_id", "clicked"]
    )
    popularity = build_train_popularity(train_impressions, n_catalog=train_articles_n, alpha=1.0)
    del train_impressions

    bm25_scorer = BM25Scorer(build_index(articles))
    cache_path = base / "embeddings" / f"{model_slug(DEFAULT_MODEL)}.npy"
    embed_index = build_embedding_index(articles, model_name=DEFAULT_MODEL, cache_path=cache_path)
    embed_scorer = EmbeddingScorer(embed_index)
    recency_scorer = EmbeddingScorer(embed_index)  # separate instance -> separate identity cache

    text_lookup = dict(zip(
        articles["article_id"], articles["title"].fillna("") + " " + articles["abstract"].fillna("")
    ))
    vector_lookup = dict(zip(embed_index.article_ids, embed_index.vectors))
    category_lookup, subcategory_lookup, entity_set_lookup = build_lookup_tables(articles)

    f_config = json.loads(CANDIDATE_F_CONFIG.read_text())
    assert f_config["feature_names"] == FEATURE_NAMES, "Candidate F's feature order must match this script's"
    scaler_mean = np.array(f_config["scaler_mean"])
    scaler_scale = np.array(f_config["scaler_scale"])
    coef = np.array(f_config["coefficients"])
    intercept = f_config["intercept"]

    query_by_user = build_query_by_user(
        history, vector_lookup, text_lookup, category_lookup, subcategory_lookup, entity_set_lookup
    )
    empty_query = GatedQuery(
        cohort="cold", embed_query=None, bm25_query=[], recency_query=None, profile=EMPTY_PROFILE
    )
    scorer = GatedScorer(
        embed_scorer, bm25_scorer, recency_scorer,
        category_lookup, subcategory_lookup, entity_set_lookup, popularity,
        scaler_mean, scaler_scale, coef, intercept,
    )
    n_cold = sum(1 for q in query_by_user.values() if q.cohort == "cold")
    print(
        f"index + query build: {time.time() - t0:.1f}s "
        f"({n_cold}/{len(query_by_user)} users cold, {n_cold / len(query_by_user):.1%})",
        flush=True,
    )

    out_dir = _REPO_ROOT / "submissions" / "mind_large_test_gated_cohort"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n = write_predictions(
        out_dir / "prediction.txt", zip_path, scorer, query_by_user, empty_query,
        has_labels=False, seed=0,
    )
    print(f"wrote {n} prediction lines in {time.time() - t0:.1f}s -> {out_dir / 'prediction.txt'}", flush=True)


if __name__ == "__main__":
    main()
