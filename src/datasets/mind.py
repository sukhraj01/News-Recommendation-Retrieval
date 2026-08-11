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
    """Vectorized via `DataFrame.explode` rather than a per-token Python
    loop building a list of dicts. The loop form was measured to hang (heavy
    CPU + swap growth, no progress after 7+ minutes) on MINDlarge_train's
    ~2.2M impressions x ~37 avg candidates (~80M+ exploded rows) on an 8GB
    machine — it never showed a problem at MINDsmall's ~14x-smaller scale
    (156,965 impressions). Same category of fix as ADR-006's BM25
    `get_scores()` rewrite: a naive per-token Python loop that was fine at
    MINDsmall scale doesn't survive MINDlarge. Verified exact-match against
    the original loop implementation on real MINDsmall data before being
    trusted (see `tests/unit/test_mind_parsing.py`).

    ID columns use `category` dtype: at MINDlarge_train scale, the raw
    object-dtype form projected to ~27GB (measured directly, not estimated)
    because pandas' object dtype repeats each ~37x-duplicated prefixed
    string as a distinct Python object per row instead of storing each
    unique value once. Categorical dictionary-encoding cut that to ~2.2GB
    (12.5x, measured on the full MINDsmall_train sample and projected) —
    the actual fix for the resource ceiling, not just the loop rewrite.
    The four EB-NeRD-only null columns get the same treatment: a bare
    `df[col] = None` materializes a full object-dtype array (~140MB per
    column at MINDsmall_train scale, confirmed by direct measurement,
    despite every element being the same singleton) where an empty
    categorical costs ~6MB and round-trips through parquet as float64 NaN.

    Categories are built from the RAW (unprefixed) id values, THEN the
    prefix is applied only to the small category array — never to the full
    exploded row count. Doing it the other way (build the full ~40-char
    prefixed string per row, THEN `.astype("category")`) was measured to
    hang at real MINDlarge_train scale (~81M rows): profiling the stuck
    process showed nearly all its time inside pandas' `map_infer_mask`,
    because Series `+` string concatenation on object dtype is an
    elementwise Python loop, not a vectorized C op — it was paying that
    cost 81M times over instead of ~2.2M times (impression_id's real
    cardinality) or less (~700K users, ~160K articles). `pd.Categorical()`
    itself (factorize) IS fast, C-level, near-linear — confirmed by timing
    this version directly against the naive one, not assumed.

    Article/label splitting happens on the RAW ~2.2M-row `impressions`
    column (one Python-level pass per impression, splitting all of that
    impression's tokens at once), never on the ~81M-row exploded Series.
    An earlier version split space-separated tokens first, exploded, THEN
    ran `.str.rsplit("-", n=1, expand=True)` on the exploded (81M-row)
    Series — profiling showed that call was *also* `map_infer_mask`
    underneath (pandas' `.str` accessor methods call the Python string
    method per element via a Cython loop, not a vectorized C op), paying
    the same 81M-vs-2.2M cost multiplier this whole function exists to
    avoid. `DataFrame.explode` on pre-split list columns IS the fast,
    C-level part — confirmed by profiling exactly where CPU time went
    before assuming a fix helped, not just by reasoning about it."""
    def _split_tokens(tokens: str) -> tuple[list[str], list[bool]] | list[str]:
        if has_labels:
            articles: list[str] = []
            clicked: list[bool] = []
            for token in tokens.split(" "):
                article_raw, label = token.rsplit("-", 1)
                articles.append(article_raw)
                clicked.append(label == "1")
            return articles, clicked
        return tokens.split(" ")

    exploded = behaviors[["impression_id", "user_id", "impression_time"]].copy()
    if has_labels:
        split_cols = [_split_tokens(t) for t in behaviors["impressions"]]
        exploded["article_raw"] = [a for a, _ in split_cols]
        exploded["clicked_raw"] = [c for _, c in split_cols]
        exploded = exploded.explode(["article_raw", "clicked_raw"], ignore_index=True)
        article_raw = exploded.pop("article_raw")
        clicked = exploded.pop("clicked_raw").to_numpy()
    else:
        exploded["article_raw"] = [_split_tokens(t) for t in behaviors["impressions"]]
        exploded = exploded.explode("article_raw", ignore_index=True)
        article_raw = exploded.pop("article_raw")
        clicked = None

    def _prefixed_categorical(raw_values, *qualifiers: str) -> pd.Categorical:
        cat = pd.Categorical(raw_values)
        prefixed_categories = [prefix_id(DATASET, c, *qualifiers) for c in cat.categories]
        return pd.Categorical.from_codes(cat.codes, categories=prefixed_categories)

    n = len(exploded)
    df = pd.DataFrame({
        "impression_id": _prefixed_categorical(exploded["impression_id"], split),
        "dataset": pd.Series([DATASET] * n, dtype="category"),
        "user_id": _prefixed_categorical(exploded["user_id"]),
        "article_id": _prefixed_categorical(article_raw),
        "impression_time": exploded["impression_time"].to_numpy(),
    })
    if has_labels:
        df["clicked"] = clicked
    for col in ("session_id", "dwell_time", "scroll_percentage", "is_front_page"):
        df[col] = pd.array([None] * n, dtype="category")
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
    exception). `user_history` is still returned (unlike `clicked`, it does
    not depend on labels being present) — query construction for the blind
    test split needs it exactly like train/dev do."""
    folder = zip_path.stem
    articles = _parse_news_tsv(zip_path, folder)
    candidates, user_history = _parse_behaviors_tsv(zip_path, folder, split, has_labels=False)
    return {"articles": articles, "candidates": candidates, "user_history": user_history}
