"""Unified schema contracts for `articles`, `impressions`, `user_history`.

Single source of truth per ADR-002: mandatory core fields non-null for both
datasets, optional fields explicitly marked as belonging to one dataset only
(always null for the other, by construction — not a data-quality issue).
This module only declares the contract; src/pipeline/validators.py enforces
it and src/datasets/*.py must produce dataframes conforming to it.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    mandatory: bool
    dataset_only: str | None = None  # "mind" | "ebnerd" | None (present in both)


ARTICLES_SCHEMA: dict[str, FieldSpec] = {
    "article_id": FieldSpec(mandatory=True),
    "dataset": FieldSpec(mandatory=True),
    "title": FieldSpec(mandatory=True),
    "abstract": FieldSpec(mandatory=True),
    "body": FieldSpec(mandatory=False, dataset_only="ebnerd"),
    "category": FieldSpec(mandatory=True),
    "subcategory": FieldSpec(mandatory=False),
    "entities": FieldSpec(mandatory=False),
    "topics": FieldSpec(mandatory=False, dataset_only="ebnerd"),
    "article_published_time": FieldSpec(mandatory=False, dataset_only="ebnerd"),
}

IMPRESSIONS_SCHEMA: dict[str, FieldSpec] = {
    "impression_id": FieldSpec(mandatory=True),
    "dataset": FieldSpec(mandatory=True),
    "user_id": FieldSpec(mandatory=True),
    "article_id": FieldSpec(mandatory=True),
    "impression_time": FieldSpec(mandatory=True),
    "clicked": FieldSpec(mandatory=True),
    "session_id": FieldSpec(mandatory=False, dataset_only="ebnerd"),
    "dwell_time": FieldSpec(mandatory=False, dataset_only="ebnerd"),
    "scroll_percentage": FieldSpec(mandatory=False, dataset_only="ebnerd"),
    "is_front_page": FieldSpec(mandatory=False, dataset_only="ebnerd"),
}

# MINDlarge_test candidates: same shape as impressions minus `clicked`
# (labels don't exist — the true hidden test set). Kept as a separate
# schema/table (`candidates`), never merged into `impressions`, so the
# unconditionally-mandatory `clicked` field never needs a null exception.
CANDIDATES_SCHEMA: dict[str, FieldSpec] = {
    k: v for k, v in IMPRESSIONS_SCHEMA.items() if k != "clicked"
}

USER_HISTORY_SCHEMA: dict[str, FieldSpec] = {
    "user_id": FieldSpec(mandatory=True),
    "dataset": FieldSpec(mandatory=True),
    "article_ids": FieldSpec(mandatory=True),
    "click_times": FieldSpec(mandatory=False, dataset_only="ebnerd"),
    "read_times": FieldSpec(mandatory=False, dataset_only="ebnerd"),
}
