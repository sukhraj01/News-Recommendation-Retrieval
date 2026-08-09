"""MIND -> unified schema (ADR-002) parsing.

Internal zip folder name is assumed to equal the zip file's stem
(confirmed for every MIND bundle on disk: MINDsmall_train.zip contains
MINDsmall_train/, etc.) so callers only need to pass a zip path.
"""
import json
from pathlib import Path

import pandas as pd

from src.utils.ids import prefix_id
from src.utils.io import read_zip_tsv

DATASET = "mind"

_NEWS_COLUMNS = [
    "id", "category", "subcategory", "title", "abstract",
    "url", "title_entities", "abstract_entities",
]
_BEHAVIORS_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]
_TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def _combine_mind_entities(title_entities: str | float, abstract_entities: str | float) -> str:
    def _load(s):
        if s is None or (isinstance(s, float) and pd.isna(s)) or s == "":
            return []
        return json.loads(s)

    combined = {
        "title_entities": _load(title_entities),
        "abstract_entities": _load(abstract_entities),
    }
    return json.dumps(combined, sort_keys=True)


def _parse_news_tsv(zip_path: Path, folder: str) -> pd.DataFrame:
    raw = read_zip_tsv(zip_path, f"{folder}/news.tsv", _NEWS_COLUMNS)

    entities = [
        _combine_mind_entities(t, a)
        for t, a in zip(raw["title_entities"], raw["abstract_entities"])
    ]

    return pd.DataFrame({
        "article_id": [prefix_id(DATASET, i) for i in raw["id"]],
        "dataset": DATASET,
        "title": raw["title"],
        "abstract": raw["abstract"].fillna(""),
        "body": None,
        "category": raw["category"],
        "subcategory": raw["subcategory"],
        "entities": entities,
        "topics": None,
        "article_published_time": pd.NaT,
    })


def _parse_history(history: str | float) -> list[str]:
    if history is None or (isinstance(history, float) and pd.isna(history)):
        return []
    return [prefix_id(DATASET, a) for a in history.split(" ")]


def _build_user_history(behaviors: pd.DataFrame) -> pd.DataFrame:
    """MIND's `history` is a static per-user snapshot repeated on every
    impression row (verified 0-variance across 33,617 MINDsmall users).
    Re-assert the invariant here rather than trust it blindly — it was
    never checked at MINDlarge scale."""
    per_user = behaviors.groupby("user_id")["history"].apply(
        lambda s: s.fillna("").unique()
    )
    bad_users = per_user[per_user.map(len) > 1]
    if len(bad_users):
        raise ValueError(
            f"MIND static-history invariant violated for {len(bad_users)} user(s): "
            f"history varies across that user's own impression rows "
            f"(first offender: {bad_users.index[0]!r})"
        )

    first_per_user = behaviors.drop_duplicates(subset="user_id", keep="first")
    return pd.DataFrame({
        "user_id": [prefix_id(DATASET, u) for u in first_per_user["user_id"]],
        "dataset": DATASET,
        "article_ids": [_parse_history(h) for h in first_per_user["history"]],
        "click_times": None,
        "read_times": None,
    })


def _explode_impressions(
    behaviors: pd.DataFrame, split: str, has_labels: bool
) -> pd.DataFrame:
    rows = []
    for impression_id, user_id, impression_time, tokens in zip(
        behaviors["impression_id"], behaviors["user_id"],
        behaviors["impression_time"], behaviors["impressions"],
    ):
        for token in tokens.split(" "):
            if has_labels:
                article_raw, label = token.rsplit("-", 1)
                clicked = label == "1"
            else:
                article_raw, clicked = token, None
            row = {
                "impression_id": prefix_id(DATASET, impression_id, split),
                "dataset": DATASET,
                "user_id": prefix_id(DATASET, user_id),
                "article_id": prefix_id(DATASET, article_raw),
                "impression_time": impression_time,
            }
            if has_labels:
                row["clicked"] = clicked
            rows.append(row)

    df = pd.DataFrame(rows)
    for col in ("session_id", "dwell_time", "scroll_percentage", "is_front_page"):
        df[col] = None
    return df


def _parse_behaviors_tsv(
    zip_path: Path, folder: str, split: str, has_labels: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = read_zip_tsv(zip_path, f"{folder}/behaviors.tsv", _BEHAVIORS_COLUMNS)
    raw["impression_time"] = pd.to_datetime(raw["time"], format=_TIME_FORMAT)

    interactions = _explode_impressions(raw, split, has_labels)
    user_history = _build_user_history(raw)
    return interactions, user_history


def parse_mind_split(zip_path: Path, split: str) -> dict[str, pd.DataFrame]:
    """Parse a labeled MIND split (train/dev) into articles/impressions/user_history."""
    folder = zip_path.stem
    articles = _parse_news_tsv(zip_path, folder)
    impressions, user_history = _parse_behaviors_tsv(zip_path, folder, split, has_labels=True)
    return {"articles": articles, "impressions": impressions, "user_history": user_history}


def parse_mind_test_candidates(zip_path: Path, split: str = "test") -> dict[str, pd.DataFrame]:
    """Parse MINDlarge_test: unlabeled candidates, no `clicked` column, never
    merged into `impressions` (ADR-002's `clicked` field is unconditionally
    mandatory; this table exists precisely so that constraint never needs an
    exception)."""
    folder = zip_path.stem
    articles = _parse_news_tsv(zip_path, folder)
    candidates, _user_history = _parse_behaviors_tsv(zip_path, folder, split, has_labels=False)
    return {"articles": articles, "candidates": candidates}
