"""EB-NeRD -> unified schema (ADR-002) parsing.

Reads exact named members from the zip only (articles.parquet at the top
level, {split}/behaviors.parquet, {split}/history.parquet) — the zip also
contains __MACOSX/ junk metadata entries that must never be globbed.
"""
import json
from pathlib import Path

import pandas as pd

from src.utils.ids import prefix_id
from src.utils.io import read_zip_parquet

DATASET = "ebnerd"


def _combine_ebnerd_entities(ner_clusters, entity_groups) -> str:
    def _to_list(v):
        return [] if v is None else list(v)

    combined = {
        "ner_clusters": _to_list(ner_clusters),
        "entity_groups": _to_list(entity_groups),
    }
    return json.dumps(combined, sort_keys=True)


def parse_ebnerd_articles(zip_path: Path) -> pd.DataFrame:
    raw = read_zip_parquet(zip_path, "articles.parquet")

    entities = [
        _combine_ebnerd_entities(n, e)
        for n, e in zip(raw["ner_clusters"], raw["entity_groups"])
    ]

    return pd.DataFrame({
        "article_id": [prefix_id(DATASET, i) for i in raw["article_id"]],
        "dataset": DATASET,
        "title": raw["title"],
        "abstract": raw["subtitle"],
        "body": raw["body"],
        "category": raw["category_str"],
        "subcategory": raw["subcategory"],
        "entities": entities,
        "topics": raw["topics"],
        "article_published_time": raw["published_time"],
    })


def _explode_impressions(behaviors: pd.DataFrame, split: str) -> pd.DataFrame:
    is_front_page = behaviors["article_id"].isna()

    rows = []
    for impression_id, user_id, impression_time, inview, clicked_ids, \
        read_time, scroll_pct, session_id, front_page in zip(
        behaviors["impression_id"], behaviors["user_id"], behaviors["impression_time"],
        behaviors["article_ids_inview"], behaviors["article_ids_clicked"],
        behaviors["read_time"], behaviors["scroll_percentage"],
        behaviors["session_id"], is_front_page,
    ):
        clicked_set = set(clicked_ids)
        for candidate in inview:
            rows.append({
                "impression_id": prefix_id(DATASET, impression_id, split),
                "dataset": DATASET,
                "user_id": prefix_id(DATASET, user_id),
                "article_id": prefix_id(DATASET, candidate),
                "impression_time": impression_time,
                "clicked": candidate in clicked_set,
                "session_id": str(session_id),
                "dwell_time": read_time,
                "scroll_percentage": scroll_pct,
                "is_front_page": bool(front_page),
            })
    return pd.DataFrame(rows)


def _parse_history(zip_path: Path, split: str) -> pd.DataFrame:
    raw = read_zip_parquet(zip_path, f"{split}/history.parquet")
    return pd.DataFrame({
        "user_id": [prefix_id(DATASET, u) for u in raw["user_id"]],
        "dataset": DATASET,
        "article_ids": [
            [prefix_id(DATASET, a) for a in ids] for ids in raw["article_id_fixed"]
        ],
        "click_times": list(raw["impression_time_fixed"]),
        "read_times": list(raw["read_time_fixed"]),
    })


def parse_ebnerd_split(zip_path: Path, split: str) -> dict[str, pd.DataFrame]:
    behaviors = read_zip_parquet(zip_path, f"{split}/behaviors.parquet")
    impressions = _explode_impressions(behaviors, split)
    user_history = _parse_history(zip_path, split)
    return {"impressions": impressions, "user_history": user_history}
