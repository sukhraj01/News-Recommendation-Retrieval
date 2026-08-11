"""Q4 per-impression ranking metrics: AUC, MRR, nDCG@5, nDCG@10 (Q4.1),
intra-list diversity, novelty, catalog coverage (Q4.2), with bootstrap CI
(Q4.4) built on the same per-user resampling pattern as
`metrics.py::recall_at_k` (Q2's recall@K).

Structurally different question from recall@K: recall@K asks whether the
true clicked article appears anywhere in a top-K retrieved from the WHOLE
corpus. These metrics instead score and rank the candidates ALREADY listed
in a single impression (real data: MIND-dev median 23/impression, EB-NeRD
median 9-12), then compare that ranking against the click labels. See
ADR-007 for the tie-breaking rule, the K=10 choice for diversity/novelty,
and why coverage doesn't get a bootstrap CI.
"""
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .bootstrap import bootstrap_ratio_ci


def _stable_seed(impression_id: str, seed: int) -> int:
    """hashlib, not Python's built-in `hash()` — `hash()` is salted per
    process (PYTHONHASHSEED) by default and would silently break
    reproducibility across runs/machines, which this project's "same code +
    same data = same results" principle (CLAUDE.md) cannot tolerate."""
    digest = hashlib.sha256(f"{seed}:{impression_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def rank_candidates(scores: np.ndarray, impression_id: str, seed: int = 0) -> np.ndarray:
    """Returns the permutation of positional indices (into `scores`, and
    correspondingly into whatever candidate_ids/clicked arrays share that
    position) ranking candidates by score descending. Ties broken by a
    deterministic per-impression-seeded pseudo-random permutation — see
    ADR-007 for why (cold-start users produce a total tie across every
    candidate in every impression, since an empty query scores everything
    0; a fixed ordering, whether the impression's given candidate order or
    article_id, risks silently smuggling an unverified bias into exactly
    the cold-user cohort Q4.3 is supposed to characterize honestly).
    """
    rng = np.random.default_rng(_stable_seed(impression_id, seed))
    tiebreak = rng.random(len(scores))
    return np.lexsort((tiebreak, -np.asarray(scores, dtype=np.float64)))


def ndcg_at_k(ranked_clicked: np.ndarray, k: int) -> float:
    """`ranked_clicked`: bool array already in tie-broken rank order.
    Binary relevance means the DCG gain term is just `rel` (2**rel - 1 ==
    rel for rel in {0,1}), so no graded-relevance formula is needed. NaN if
    the impression has zero positives (IDCG undefined) — should not occur
    in this project's current data (verified directly for MIND-dev and
    EB-NeRD-demo-validation) but handled defensively since ebnerd_small/
    future data isn't guaranteed to match."""
    rel = ranked_clicked[:k].astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    dcg = float(np.sum(rel * discounts))

    ideal_rel = np.sort(ranked_clicked.astype(np.float64))[::-1][:k]
    idcg = float(np.sum(ideal_rel * (1.0 / np.log2(np.arange(2, len(ideal_rel) + 2)))))
    if idcg == 0:
        return float("nan")
    return dcg / idcg


def mrr(ranked_clicked: np.ndarray) -> float:
    """Reciprocal rank of the first clicked candidate (1-indexed), 0.0 if
    none — the standard MRR convention (no hit = zero credit, not
    undefined). Correct under multi-click impressions (confirmed real in
    this project's data, up to 24 clicks in one MIND impression) since only
    the *first* hit matters for MRR by definition."""
    hits = np.flatnonzero(ranked_clicked)
    return 1.0 / (hits[0] + 1) if hits.size else 0.0


def safe_auc(scores: np.ndarray, clicked: np.ndarray) -> float:
    """NaN for degenerate single-class impressions (all-clicked or
    all-unclicked) — AUC is undefined without both classes present. Not
    observed in this project's current data (0 such impressions across
    MIND-dev and EB-NeRD-demo-validation, checked directly) but must be
    handled defensively since ebnerd_small is unverified until checked.
    Does NOT use `rank_candidates`'s tie-break — `roc_auc_score`'s own
    rank-average handling of tied scores is already the textbook-correct
    AUC treatment (Mann-Whitney U); reapplying a randomized break would
    diverge from the standard definition, not improve it."""
    clicked = np.asarray(clicked, dtype=bool)
    if clicked.all() or not clicked.any():
        return float("nan")
    return float(roc_auc_score(clicked, scores))


def intra_list_diversity(categories: Sequence[str]) -> float:
    """Standard average pairwise category-dissimilarity over a ranked list
    (fraction of candidate pairs with differing category). NaN for lists
    shorter than 2 (undefined — no pairs to compare)."""
    n = len(categories)
    if n < 2:
        return float("nan")
    n_diff = 0
    n_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            n_pairs += 1
            if categories[i] != categories[j]:
                n_diff += 1
    return n_diff / n_pairs


def build_train_popularity(
    train_impressions: pd.DataFrame, n_catalog: int, alpha: float = 1.0
) -> dict[str, float]:
    """Article click-probability estimated from the TRAIN split only — per
    Q9's anti-gaming requirement, using validation/dev-split popularity
    would leak future-click information into a metric meant to reward
    recommending items the system couldn't have known were popular yet at
    serving time.

    Laplace (add-alpha) smoothing is required, not cosmetic: ADR-002
    measured 32.9% (MIND) / 45.7% (EB-NeRD) of validation-candidate
    articles never appear in train at all — unsmoothed popularity would
    give these p=0, and -log2(0) = inf would poison any average that
    includes them. Smoothing gives a small floor probability instead,
    which reads correctly as "very novel," not "broken."
    """
    clicks = train_impressions.loc[train_impressions["clicked"], "article_id"].value_counts()
    total = float(clicks.sum())
    denom = total + alpha * n_catalog
    floor = alpha / denom
    popularity: dict[str, float] = defaultdict(lambda: floor)
    for article_id, count in clicks.items():
        popularity[article_id] = (count + alpha) / denom
    return popularity


def novelty(article_ids: Sequence[str], popularity: dict[str, float]) -> float:
    """Mean self-information (-log2 popularity) over a ranked list — higher
    means less popular in train, i.e. more novel."""
    if not article_ids:
        return float("nan")
    return float(np.mean([-np.log2(popularity[a]) for a in article_ids]))


def coverage(top_k_by_impression: Iterable[Sequence[str]], catalog_size: int) -> float:
    """Catalog coverage: fraction of the corpus that appears in the union
    of every impression's top-K ranked candidates. A single point estimate
    over the full run, deliberately without a bootstrap CI — see ADR-007:
    coverage is a set-union statistic, and with-replacement bootstrap
    resampling of a union is mechanically biased low (resampling the same
    user twice adds nothing new to the union, unlike a mean, where a
    repeat correctly reweights the average), so a naive CI here would look
    precise while actually being an artifact of the resampling mechanism,
    not real uncertainty."""
    seen: set[str] = set()
    for ids in top_k_by_impression:
        seen.update(ids)
    return len(seen) / catalog_size if catalog_size else float("nan")


@dataclass
class RankingMetricResult:
    metric: float
    ci_low: float
    ci_high: float
    n_impressions: int
    n_users: int
    n_skipped: int = 0  # e.g. degenerate-AUC impressions dropped, not silently averaged in


def ranking_metric_ci(
    per_impression: pd.DataFrame,
    value_col: str,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> RankingMetricResult:
    """`per_impression` has columns `user_id` and `value_col` (one row per
    impression). Rows with NaN in `value_col` (e.g. degenerate-AUC skips)
    are dropped and counted in `n_skipped`, not silently averaged in. One
    generic wrapper reused identically for every per-impression scalar
    metric (AUC, MRR, nDCG@5, nDCG@10, diversity@10, novelty@10) — no
    per-metric bootstrap code duplicated anywhere, reusing
    `bootstrap.py::bootstrap_ratio_ci`, the same substrate
    `metrics.py::recall_at_k` uses for Q2's recall@K.
    """
    total = len(per_impression)
    df = per_impression.dropna(subset=[value_col])
    n_skipped = total - len(df)

    if df.empty:
        return RankingMetricResult(
            metric=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
            n_impressions=0, n_users=0, n_skipped=n_skipped,
        )

    grouped = df.groupby("user_id")[value_col].agg(["sum", "count"])
    user_sum = grouped["sum"].to_numpy(dtype=np.float64)
    user_count = grouped["count"].to_numpy(dtype=np.float64)
    ci_low, ci_high = bootstrap_ratio_ci(user_sum, user_count, n_bootstrap, seed)

    return RankingMetricResult(
        metric=float(df[value_col].mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        n_impressions=len(df),
        n_users=len(grouped),
        n_skipped=n_skipped,
    )
