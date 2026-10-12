"""Feature groups over the bandit's linear vector, for the mechanism ablation (ADR-017 addendum).

The vector is `LinearFeatureTransformer.feature_names` (100 columns, ending in `bias`) plus the two
hybrid columns (`lgb_prob`, `lgb_centred`). Every non-bias column belongs to exactly one group
(tested), so "drop one group" and "keep only one group" are clean partitions. The task named five
groups; the leftover columns (context, session, history-popularity) are a sixth, so nothing is left
unassigned.
"""
from __future__ import annotations

HYBRID_NAMES = ["lgb_prob", "lgb_centred"]

GROUPS: dict[str, list[str]] = {
    "lightgbm_score": list(HYBRID_NAMES),
    "freshness": ["article_age_h", "is_fresh_6h", "age_vs_user_appetite",
                  "article_age_rank_in_imp", "article_age_minus_imp_min"],
    "article_metadata": ["sentiment_score", "is_premium", "title_len", "body_len", "n_topics", "n_entities"],
    "user_history": [
        "lt_cat_affinity", "lt_subcat_affinity", "lt_topic_affinity", "lt_embed_sim", "lt_cat_rank",
        "lt_sentiment_distance", "st_cat_affinity", "st_subcat_affinity", "st_topic_affinity", "st_embed_sim",
        "st_cat_rank", "is_last_clicked_category", "in_recent_categories", "cat_affinity_drift",
        "embed_sim_drift", "topic_affinity_drift", "log_history_len", "user_mean_read_time",
        "user_median_read_time", "user_mean_scroll", "user_distinct_categories", "user_clicks_last_24h",
        "user_mean_article_age_h", "hours_since_last_click", "user_history_span_h",
        "lt_cat_affinity_rank_in_imp", "st_cat_affinity_rank_in_imp", "lt_embed_sim_rank_in_imp",
        "lt_embed_sim_minus_imp_max",
    ],
    "context_session_popularity": [
        "log_history_popularity", "log_history_popularity_decayed", "history_popularity_rank_in_imp",
        "n_candidates", "hour_of_day", "day_of_week", "is_weekend", "is_front_page", "is_sso_user",
        "is_subscriber", "user_age", "context_read_time", "context_scroll_percentage",
        "context_category_match", "context_topic_overlap", "context_embed_sim", "session_position",
        "session_start_gap_h",
    ],
}
ONEHOT_PREFIXES = ("category_code=", "article_type=", "sentiment_label=")


def assign_groups(feature_names: list[str]) -> dict[str, list[int]]:
    """group -> column indices into `feature_names` (which must include the hybrid names).
    `bias` is deliberately in no group (always kept)."""
    out: dict[str, list[int]] = {g: [] for g in [*GROUPS, "onehot_categoricals"]}
    lookup = {n: g for g, ns in GROUPS.items() for n in ns}
    for i, n in enumerate(feature_names):
        if n == "bias":
            continue
        if n.startswith(ONEHOT_PREFIXES):
            out["onehot_categoricals"].append(i)
        elif n in lookup:
            out[lookup[n]].append(i)
        else:
            raise KeyError(f"feature {n!r} is in no group")
    return out


def group_mask(feature_names: list[str], *, drop: list[str] | None = None, keep_only: list[str] | None = None):
    """0/1 mask over `feature_names`. `drop`: zero those groups. `keep_only`: zero every group not listed
    (bias always kept)."""
    import numpy as np

    groups = assign_groups(feature_names)
    mask = np.ones(len(feature_names))
    if keep_only is not None:
        for g, idx in groups.items():
            if g not in keep_only:
                mask[idx] = 0.0
    for g in drop or []:
        mask[groups[g]] = 0.0
    return mask
