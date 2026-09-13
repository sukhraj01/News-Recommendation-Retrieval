"""A2 Q1 — MIND click-history feature engineering (Candidate L).

`features.py` already has unweighted symbolic history matching
(`category_match_score`/`subcategory_match_score`, every history item
weighted equally). This module adds the two things Q1 asks for that did not
exist: **recency-weighted** history affinity, and **click count** as an
engineered statistic.

Recency weighting reuses `embed.py::build_user_embedding_query_recency`'s
exact convention (geometric decay by position, most-recent-last, decay=0.9)
rather than inventing a second definition of "recency" for MIND — that
function already states the assumption this relies on: MIND's history list
order is "presumed chronological, not independently verified" (ADR-002).
That assumption is inherited here, not re-argued.

Dataset limitation, stated plainly rather than worked around: MIND provides
no per-click timestamps (only a static ordered list) and no publish
timestamps or session boundaries (`src/datasets/mind.py` hardcodes
`article_published_time`/`session_id`/`dwell_time`/`scroll_percentage`/
`is_front_page` to null — confirmed by direct measurement against the real
processed data, not assumed). So "recency-weighted" here means
position-based decay, and freshness/session features are not built for
MIND at all — EB-NeRD already has both (ADR-013's `article_age_h`,
`session_position`/`session_start_gap_h`).
"""
import numpy as np

DECAY = 0.9  # matches build_user_embedding_query_recency and Candidate C/F


def click_count(article_ids: list[str]) -> int:
    """Raw history length. Q1's simplest engineered statistic — also the
    quantity `COLD_THRESHOLD` cohort assignment already uses project-wide,
    surfaced here as a named, tested function rather than an inline `len()`.
    """
    return len(article_ids)


def recency_weights(n: int, decay: float = DECAY) -> np.ndarray:
    """Geometric decay weights for a history of length `n`, index `n-1`
    (last) assumed most recent: `weights[i] = decay ** (n - 1 - i)`, so the
    most-recent resolvable item gets weight 1.0 and older ones decay
    geometrically. Byte-for-byte the same formula as
    `build_user_embedding_query_recency` (embed.py) — one recency
    definition for MIND, not two that could silently drift apart.
    """
    if n == 0:
        return np.array([], dtype=np.float64)
    return np.array([decay ** (n - 1 - i) for i in range(n)], dtype=np.float64)


def recency_weighted_affinity(
    article_ids: list[str], field_lookup: dict[str, str], target: str | None, decay: float = DECAY,
) -> float:
    """Recency-weighted fraction of history matching `target` — the
    weighted counterpart to `features.py::category_match_score` /
    `subcategory_match_score`, which give every history item equal weight
    regardless of position. `field_lookup` maps article_id -> category (or
    subcategory); pass either to get the corresponding affinity.

    Returns a value in [0, 1], same range as the unweighted score, so the
    two are directly comparable as features. 0.0 for an empty history or an
    unknown/absent target — never NaN, matching every other score function
    in this project's convention.
    """
    if not target or not article_ids:
        return 0.0
    weights = recency_weights(len(article_ids), decay)
    total = float(weights.sum())
    if total == 0.0:
        return 0.0
    matched = sum(w for aid, w in zip(article_ids, weights) if field_lookup.get(aid) == target)
    return matched / total


FEATURE_NAMES = [
    "log_click_count",
    "category_match",
    "subcategory_match",
    "recency_category_affinity",
    "recency_subcategory_affinity",
]


def build_feature_row(
    candidate_id: str, article_ids: list[str], category_lookup: dict[str, str],
    subcategory_lookup: dict[str, str], profile, decay: float = DECAY,
) -> list[float]:
    """One candidate's feature vector, in `FEATURE_NAMES` order. `profile`
    is a `features.py::HistoryProfile` (built once per user, reused across
    that user's candidates) — only `category_match_score`/
    `subcategory_match_score` need it; the recency features recompute
    directly from `article_ids` since position information is exactly what
    a `HistoryProfile`'s counters discard.

    IMPORTANT, documented rather than silently accepted: `log_click_count`
    is identical for every candidate in the same impression (it depends
    only on the user, not the candidate). A linear combiner's per-impression
    RANKING (AUC/MRR/nDCG@K, all rank-order-only) is provably invariant to
    an additive per-impression-constant term — this feature cannot move a
    ranking metric under `LogisticRegression`, no matter its fitted
    coefficient. It is included because Q1 asks for click count explicitly,
    and `scripts/run_mind_history_features_experiment.py` verifies this
    claim empirically (zero-variance-within-impression check, and an
    ablation confirming its removal changes ranking metrics by exactly
    nothing) rather than asserting it from theory alone.
    """
    from .features import category_match_score, subcategory_match_score

    return [
        np.log1p(click_count(article_ids)),
        category_match_score(category_lookup.get(candidate_id), profile),
        subcategory_match_score(subcategory_lookup.get(candidate_id), profile),
        recency_weighted_affinity(article_ids, category_lookup, category_lookup.get(candidate_id), decay),
        recency_weighted_affinity(article_ids, subcategory_lookup, subcategory_lookup.get(candidate_id), decay),
    ]
