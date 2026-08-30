"""EB-NeRD learning-to-rank feature engineering (Candidate K, ADR-013).

Why this module exists
----------------------
Every EB-NeRD scorer this project has shipped so far (BM25 per ADR-005/006,
MiniLM embeddings per ADR-008, the provided `contrastive_vector` artifact per
ADR-008's addendum 2) scores a candidate purely by *content similarity to the
user's history*. On the real Codabench test set that tops out at 0.5404 AUC —
below the challenge's own naive "most clicks" popularity baseline (0.5970) and
barely above its "in view rate" baseline (0.5450).

The reason is structural, not a bug: EB-NeRD's in-view candidate lists are
drawn from the same front page at the same moment, so every candidate is
already topically adjacent and recently published. Content similarity has
almost nothing left to discriminate on. What actually separates a click from a
non-click is *behavioural and temporal* — how fresh the article is, how well it
matches the categories this user keeps returning to, whether it matches what
they were reading twenty minutes ago, how popular it already was.

Per RecSys Challenge 2024's own "Common Themes" (arXiv:2409.20483), every top
solution was a GBDT over exactly this kind of engineered tabular feature set.
This module builds that feature set.

Long-term vs. short-term
------------------------
The feature families mirror the hierarchical-user-interest idea BlackPearl (2nd
place, 88.15 AUC) describes in its abstract: a *long-term* stable interest
profile aggregated over the user's whole 21-day history, and a *short-term*
fast-moving profile that exponentially down-weights older clicks. Both are
computed for category, subcategory, topic and embedding space, so the model can
learn its own blend rather than us fixing one.

Note that BlackPearl's full text was not obtainable (ACM DOI
10.1145/3687151.3687163 returns 403), so this decomposition is reconstructed
from their abstract and is a working hypothesis, not a reproduction of their
methodology. See ADR-013.

Leakage discipline
------------------
This is the axis the real competition turned on: the winning team's 88.64 AUC
fell to 76.99 once the organizers ablated features carrying future information.
Every feature here is derived from exactly three sources, all of which are
available at prediction time for the impression being scored:

1. The split's own ``history.parquet`` — EB-NeRD constructs this as a 21-day
   window that ends strictly before the behaviors window begins (verified
   empirically for both splits; see ADR-013's audit table).
2. Static article metadata that exists at publication time (category, topics,
   sentiment, type, premium flag, title/body text).
3. The impression's own request-time context (timestamp, device, session,
   in-view list size, subscriber flags).

Three raw fields are deliberately *never* read here:
``total_inviews`` / ``total_pageviews`` / ``total_read_time`` (article-lifetime
aggregates — ADR-009 already measured what these leak), and ``next_read_time``
/ ``next_scroll_percentage`` (post-click outcomes of the very click being
predicted). Popularity is computed from the *history* file only, never from the
behaviors file, because behaviors-window click counts would not exist at
serving time — and do not exist at all in the unlabelled test set.

The module reads raw EB-NeRD parquet directly rather than the processed feature
store, because the store (by ADR-002's unified-schema design) drops the
dataset-specific context columns this model needs — ``device_type``, ``age``,
``is_subscriber``, ``sentiment_score``, ``article_type``, ``premium``. Reading
raw keeps one identical code path across train / validation / test, which is
what the Kaggle test-set run needs.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import read_zip_member_bytes

# Article-lifetime aggregates and post-click outcomes. Named here explicitly so
# the exclusion is greppable and testable, not just a comment. See ADR-009.
FORBIDDEN_RAW_COLUMNS: frozenset[str] = frozenset(
    {
        "total_inviews",
        "total_pageviews",
        "total_read_time",
        "next_read_time",
        "next_scroll_percentage",
    }
)

# Half-life (hours) for the short-term exponential decay. 24h chosen a priori
# from the news domain's item half-life (ADR-005's framing), not tuned against
# validation — tuning it would be a hyperparameter search this project's
# first-attempt discipline (ADR-009/ADR-010) reserves for a later pass.
SHORT_TERM_HALF_LIFE_HOURS: float = 24.0

# How many trailing history clicks count as "the recent sequence".
LAST_N_CLICKS: int = 5


def _read_zip_parquet(
    zip_path: str | Path, name: str, columns: list[str] | None = None
) -> pd.DataFrame:
    """Read one parquet member out of an EB-NeRD zip.

    Two responsibilities, both load-bearing:

    1. **Leakage guard.** Requesting a quarantined column raises rather than
       silently returning it, so the contract is enforced at the I/O boundary
       instead of relying on every call site to remember. See ADR-009/ADR-013.
    2. **Member-path normalisation.** Delegates to
       `src/utils/io.py::read_zip_member_bytes` rather than calling
       `ZipFile.read` directly, because `ebnerd_testset.zip` wraps every member
       in an extra top-level directory — a bare `zf.read("test/behaviors.parquet")`
       raises `KeyError` there. That resolution logic already exists and is
       already exercised by the Part-2 test-set run; duplicating it here would
       let the two drift.
    """
    if columns is not None:
        bad = FORBIDDEN_RAW_COLUMNS.intersection(columns)
        if bad:
            raise ValueError(
                f"Refusing to read serving-time-unavailable column(s) {sorted(bad)} "
                f"from {name}; see ADR-009/ADR-013."
            )
    return pd.read_parquet(io.BytesIO(read_zip_member_bytes(Path(zip_path), name)), columns=columns)


def _available_columns(zip_path: str | Path, name: str) -> set[str]:
    """Column names in a zipped parquet member, read from its schema only.

    Used to degrade gracefully when an optional column is absent. The blind test
    set is not identical in shape to the training bundles (it carries no
    `article_ids_clicked`, and its history table is not guaranteed to carry
    every engagement column), and discovering that by exception after a 1.5 GB
    download and a long profile build is a poor trade for one schema read.
    """
    import pyarrow.parquet as pq

    buf = io.BytesIO(read_zip_member_bytes(Path(zip_path), name))
    # `schema_arrow`, not `schema`: the Parquet physical schema flattens list
    # columns into nested leaf paths (`article_id_fixed.list.item`), so a
    # membership test against it never matches the logical column name.
    return set(pq.ParquetFile(buf).schema_arrow.names)


def _to_epoch_seconds(values: pd.Series) -> np.ndarray:
    """Convert a datetime Series to float64 epoch seconds, NaT -> NaN.

    EB-NeRD's parquet stores timestamps at microsecond resolution. Dividing the
    raw int64 view by 1e9 (the reflex for nanosecond-backed pandas datetimes)
    under-scales by 1000x, which does not raise — it just makes every article-age
    and time-since-last-click feature a near-constant. Casting to an explicit
    ``datetime64[s]`` states the unit instead of assuming it.
    """
    as_seconds = values.to_numpy(dtype="datetime64[s]").astype(np.float64)
    as_seconds[values.isna().to_numpy()] = np.nan
    return as_seconds


@dataclass(frozen=True)
class ArticleTable:
    """Dense, position-indexed article metadata.

    Every array is indexed by *article row position*, and `id_to_pos` maps a raw
    integer `article_id` to that position. Dense layout is affordable because
    EB-NeRD's metadata cardinalities are all small (26 categories, 174
    subcategories, 78 topics), which is what makes the whole per-candidate
    feature build a fancy-index rather than a Python loop.
    """

    id_to_pos: dict[int, int]
    category: np.ndarray  # int16, code into `category_vocab`
    subcategory: np.ndarray  # bool (n_articles, n_subcategories)
    topics: np.ndarray  # bool (n_articles, n_topics)
    published_at: np.ndarray  # float64 epoch seconds; NaN when unknown
    sentiment_score: np.ndarray  # float32
    sentiment_label: np.ndarray  # int8
    article_type: np.ndarray  # int8
    premium: np.ndarray  # bool
    title_len: np.ndarray  # int32 (characters)
    body_len: np.ndarray  # int32 (characters)
    n_topics: np.ndarray  # int8
    n_entities: np.ndarray  # int8
    embeddings: np.ndarray | None  # float32 (n_articles, dim), L2-normalised
    # Every categorical vocabulary is carried explicitly so it can be persisted
    # at training time and forced back on at inference. A vocabulary derived
    # independently on each corpus would renumber the codes -- the model would
    # then read "sport" where it learned "krimi", with no error raised.
    category_vocab: list[str]
    subcategory_vocab: list
    topic_vocab: list
    article_type_vocab: list[str]
    sentiment_label_vocab: list[str]

    @property
    def n_subcategories(self) -> int:
        return len(self.subcategory_vocab)

    @property
    def n_topic_terms(self) -> int:
        return len(self.topic_vocab)

    def vocabularies(self) -> dict:
        """Serializable snapshot, written beside the model at training time."""
        return {
            "category": list(self.category_vocab),
            "subcategory": [int(v) if isinstance(v, (int, np.integer)) else v
                            for v in self.subcategory_vocab],
            "topic": list(self.topic_vocab),
            "article_type": list(self.article_type_vocab),
            "sentiment_label": list(self.sentiment_label_vocab),
        }


def load_article_table(
    zip_path: str | Path,
    *,
    embeddings_npy: str | Path | None = None,
    embeddings_json: str | Path | None = None,
    category_vocab: list[str] | None = None,
    subcategory_vocab: list[int] | None = None,
    topic_vocab: list[str] | None = None,
    article_type_vocab: list[str] | None = None,
) -> ArticleTable:
    """Build the dense article table from a raw EB-NeRD zip.

    The `*_vocab` arguments exist so the test-set run can be forced onto the
    *training* vocabularies. Letting the test set define its own codes would
    silently shift the meaning of every categorical column between fit and
    predict — a correctness bug that would not raise, only quietly degrade.
    """
    art = _read_zip_parquet(
        zip_path,
        "articles.parquet",
        columns=[
            "article_id",
            "title",
            "body",
            "published_time",
            "premium",
            "article_type",
            "ner_clusters",
            "topics",
            "category_str",
            "subcategory",
            "sentiment_score",
            "sentiment_label",
        ],
    )

    n = len(art)
    ids = art["article_id"].to_numpy().astype(np.int64)
    id_to_pos = {int(a): i for i, a in enumerate(ids)}

    cat_str = art["category_str"].fillna("__unknown__").astype(str)
    if category_vocab is None:
        category_vocab = sorted(cat_str.unique())
    cat_index = {c: i for i, c in enumerate(category_vocab)}
    # Unseen categories map to -1 and are treated as "no affinity" downstream.
    category = cat_str.map(lambda c: cat_index.get(c, -1)).to_numpy().astype(np.int16)

    def _multi_hot(series: pd.Series, vocab: list) -> tuple[np.ndarray, list]:
        if vocab is None:
            seen: dict = {}
            for lst in series:
                if lst is None:
                    continue
                for v in lst:
                    seen[v] = True
            vocab = sorted(seen)
        index = {v: i for i, v in enumerate(vocab)}
        out = np.zeros((n, max(len(vocab), 1)), dtype=bool)
        for row, lst in enumerate(series):
            if lst is None:
                continue
            for v in lst:
                j = index.get(v)
                if j is not None:
                    out[row, j] = True
        return out, vocab

    subcategory, subcategory_vocab = _multi_hot(art["subcategory"], subcategory_vocab)
    topics, topic_vocab = _multi_hot(art["topics"], topic_vocab)

    published = pd.to_datetime(art["published_time"], errors="coerce")
    # Convert via an explicit second-resolution cast rather than dividing raw
    # int64 nanoseconds: EB-NeRD stores these as datetime64[*us*], so a /1e9
    # divide silently under-scales by 1000x and turns every downstream age
    # feature into a near-constant. Casting states the target unit instead of
    # assuming it.
    published_at = _to_epoch_seconds(published)

    # Fixed by EB-NeRD's own schema, so it needs no corpus-derived vocabulary.
    sent_label_vocab = ["Negative", "Neutral", "Positive"]
    sent_label_index = {v: i for i, v in enumerate(sent_label_vocab)}
    sentiment_label = (
        art["sentiment_label"].map(lambda v: sent_label_index.get(v, -1)).to_numpy().astype(np.int8)
    )

    if article_type_vocab is None:
        article_type_vocab = sorted(art["article_type"].fillna("__unknown__").astype(str).unique())
    type_index = {v: i for i, v in enumerate(article_type_vocab)}
    article_type = (
        art["article_type"]
        .fillna("__unknown__")
        .astype(str)
        .map(lambda v: type_index.get(v, -1))
        .to_numpy()
        .astype(np.int8)
    )

    title_len = art["title"].fillna("").astype(str).str.len().to_numpy().astype(np.int32)
    body_len = art["body"].fillna("").astype(str).str.len().to_numpy().astype(np.int32)
    n_topics = np.array(
        [0 if L is None else min(len(L), 127) for L in art["topics"]], dtype=np.int8
    )
    n_entities = np.array(
        [0 if L is None else min(len(L), 127) for L in art["ner_clusters"]], dtype=np.int8
    )

    embeddings = None
    if embeddings_npy is not None and embeddings_json is not None:
        embeddings = _load_aligned_embeddings(embeddings_npy, embeddings_json, id_to_pos, n)

    return ArticleTable(
        id_to_pos=id_to_pos,
        category=category,
        subcategory=subcategory,
        topics=topics,
        published_at=published_at,
        sentiment_score=art["sentiment_score"].to_numpy().astype(np.float32),
        sentiment_label=sentiment_label,
        article_type=article_type,
        premium=art["premium"].fillna(False).to_numpy().astype(bool),
        title_len=title_len,
        body_len=body_len,
        n_topics=n_topics,
        n_entities=n_entities,
        embeddings=embeddings,
        category_vocab=list(category_vocab),
        subcategory_vocab=list(subcategory_vocab),
        topic_vocab=list(topic_vocab),
        article_type_vocab=list(article_type_vocab),
        sentiment_label_vocab=list(sent_label_vocab),
    )


def _load_aligned_embeddings(
    npy_path: str | Path, json_path: str | Path, id_to_pos: dict[int, int], n: int
) -> np.ndarray:
    """Load cached article embeddings and re-align them to article-table order.

    The cache on disk is keyed by this project's prefixed string ids
    (``ebnerd:9738663``) in its own order; the article table is keyed by raw
    integer id in parquet order. Re-aligning rather than assuming a shared order
    is the difference between real similarity features and noise.
    """
    meta = json.loads(Path(json_path).read_text())
    vectors = np.load(npy_path)
    aligned = np.zeros((n, vectors.shape[1]), dtype=np.float32)
    for row, key in enumerate(meta["article_ids"]):
        raw = int(str(key).split(":")[-1])
        pos = id_to_pos.get(raw)
        if pos is not None:
            aligned[pos] = vectors[row]
    norms = np.linalg.norm(aligned, axis=1, keepdims=True)
    np.divide(aligned, np.maximum(norms, 1e-9), out=aligned)
    return aligned


@dataclass(frozen=True)
class UserHistoryRaw:
    """Per-user raw history, retained so a short-term profile can be recomputed
    at an arbitrary reference time -- specifically, each impression's OWN
    timestamp, rather than one timestamp shared by the whole split.

    An earlier version of this module computed `st_*` (decay-weighted) profiles
    once per user against a single split-wide reference time (the close of the
    history window). That made "short-term interest" a static per-user snapshot,
    identical across every impression that user has in the split regardless of
    whether it fell on day 1 or day 7 of the window -- defeating the premise of
    a *short-term* signal, which is supposed to track forward through time.
    Fixed by keeping the raw per-click data here and recomputing the decay in
    `compute_short_term_for_impressions`, called once per impression.

    `kpos`/`kts` are ragged (one array per user, variable length) because
    history length varies by user (measured on ebnerd_small: mean 160, p95 545,
    max 1896 clicks) -- a dense matrix here would be mostly padding. Both arrays
    are already time-sorted ascending (verified directly against the raw data,
    100% of a 3000-user sample), which lets `compute_short_term_for_impressions`
    use `np.searchsorted` to cheaply bound the work per impression to only the
    clicks recent enough to matter, rather than rescanning full histories that
    can run into the thousands.
    """

    kpos: list  # list[np.ndarray[int64]] -- article-table row positions, known articles only
    kts: list  # list[np.ndarray[float64]] -- epoch-second click times, ascending


@dataclass(frozen=True)
class UserProfiles:
    """Per-user long-term interest profile plus raw history for on-demand
    short-term profiles, built from history only.

    Every matrix is row-indexed by *user row position*; `id_to_pos` maps raw
    integer `user_id` to that position. `lt_` matrices weight every history
    click equally (stable interest) and are L1-normalised per user so they read
    as probability-like affinities comparable across users with different
    history lengths. The `st_*` (short-term, decay-weighted) matrices this
    dataclass used to carry are gone -- see `UserHistoryRaw`'s docstring for
    why they had to become per-impression instead of per-user.
    """

    id_to_pos: dict[int, int]
    raw: UserHistoryRaw
    lt_category: np.ndarray
    lt_subcategory: np.ndarray
    lt_topics: np.ndarray
    lt_embedding: np.ndarray | None
    history_len: np.ndarray
    mean_read_time: np.ndarray
    median_read_time: np.ndarray
    mean_scroll: np.ndarray
    last_click_at: np.ndarray  # epoch seconds
    first_click_at: np.ndarray
    last_category: np.ndarray  # int16 code
    recent_categories: np.ndarray  # bool (n_users, n_categories) over last N clicks
    mean_sentiment: np.ndarray
    mean_article_age_at_click: np.ndarray  # seconds; the user's freshness appetite
    distinct_categories: np.ndarray


def build_user_profiles(
    zip_path: str | Path,
    split: str,
    art: ArticleTable,
    *,
    last_n: int = LAST_N_CLICKS,
) -> UserProfiles:
    """Aggregate each user's 21-day history into a long-term profile plus raw
    history for on-demand short-term profiles.

    Long-term weighting (engagement = 1 + log1p(read_time)) has no notion of
    "now" at all, so it is correctly computed once per user, here. Short-term
    (decay) weighting DOES depend on "now" -- see `UserHistoryRaw` and
    `compute_short_term_for_impressions` for why that has to be computed per
    impression instead of once per user.
    """
    required = ["user_id", "article_id_fixed", "impression_time_fixed"]
    optional = ["read_time_fixed", "scroll_percentage_fixed"]
    present = _available_columns(zip_path, f"{split}/history.parquet")
    missing_required = [c for c in required if c not in present]
    if missing_required:
        raise ValueError(
            f"{split}/history.parquet is missing required column(s) {missing_required}; "
            f"cannot build user profiles."
        )
    hist = _read_zip_parquet(
        zip_path,
        f"{split}/history.parquet",
        columns=required + [c for c in optional if c in present],
    )
    # Engagement columns are optional: absent ones become all-NaN so the
    # downstream features stay present (LightGBM handles NaN natively) rather
    # than the feature vector silently changing width between train and test.
    for column in optional:
        if column not in hist.columns:
            hist[column] = [np.full(len(a), np.nan, dtype=np.float64)
                            for a in hist["article_id_fixed"]]

    n_users = len(hist)
    n_cat = len(art.category_vocab)
    n_sub = art.subcategory.shape[1]
    n_top = art.topics.shape[1]
    emb_dim = art.embeddings.shape[1] if art.embeddings is not None else 0

    id_to_pos = {int(u): i for i, u in enumerate(hist["user_id"].to_numpy())}

    lt_category = np.zeros((n_users, n_cat), dtype=np.float32)
    lt_subcategory = np.zeros((n_users, n_sub), dtype=np.float32)
    lt_topics = np.zeros((n_users, n_top), dtype=np.float32)
    lt_embedding = np.zeros((n_users, emb_dim), dtype=np.float32) if emb_dim else None
    recent_categories = np.zeros((n_users, n_cat), dtype=bool)

    history_len = np.zeros(n_users, dtype=np.int32)
    mean_read_time = np.full(n_users, np.nan, dtype=np.float32)
    median_read_time = np.full(n_users, np.nan, dtype=np.float32)
    mean_scroll = np.full(n_users, np.nan, dtype=np.float32)
    last_click_at = np.full(n_users, np.nan, dtype=np.float64)
    first_click_at = np.full(n_users, np.nan, dtype=np.float64)
    last_category = np.full(n_users, -1, dtype=np.int16)
    mean_sentiment = np.full(n_users, np.nan, dtype=np.float32)
    mean_article_age = np.full(n_users, np.nan, dtype=np.float32)
    distinct_categories = np.zeros(n_users, dtype=np.int16)

    raw_kpos: list = [np.empty(0, dtype=np.int64) for _ in range(n_users)]
    raw_kts: list = [np.empty(0, dtype=np.float64) for _ in range(n_users)]

    art_pos_lookup = art.id_to_pos

    for row in range(n_users):
        arts = hist["article_id_fixed"].iloc[row]
        times = hist["impression_time_fixed"].iloc[row]
        reads = hist["read_time_fixed"].iloc[row]
        scrolls = hist["scroll_percentage_fixed"].iloc[row]
        if len(arts) == 0:
            continue

        pos = np.fromiter(
            (art_pos_lookup.get(int(a), -1) for a in arts), dtype=np.int64, count=len(arts)
        )
        known = pos >= 0
        ts = np.asarray(times, dtype="datetime64[s]").astype(np.float64)

        history_len[row] = len(arts)
        first_click_at[row] = ts[0]
        last_click_at[row] = ts[-1]
        reads_arr = np.asarray(reads, dtype=np.float64)
        # An all-NaN slice is a legitimate state here, not an anomaly: the blind
        # test set's history may carry no engagement columns at all, in which
        # case these stay NaN by design and LightGBM handles them natively.
        # Guarding explicitly rather than letting numpy warn on every user.
        if reads_arr.size and not np.all(np.isnan(reads_arr)):
            mean_read_time[row] = np.nanmean(reads_arr)
            median_read_time[row] = np.nanmedian(reads_arr)
        scroll_arr = np.asarray(scrolls, dtype=np.float64)
        if scroll_arr.size and not np.all(np.isnan(scroll_arr)):
            mean_scroll[row] = np.nanmean(scroll_arr)

        if not known.any():
            continue

        kpos = pos[known]
        kts = ts[known]
        # Already time-sorted ascending (article_id_fixed/impression_time_fixed
        # come this way from EB-NeRD; verified directly, 100% of a 3000-user
        # sample) -- `compute_short_term_for_impressions` relies on this for a
        # cheap `searchsorted` truncation.
        raw_kpos[row] = kpos
        raw_kts[row] = kts

        # Long-term weights engagement as well as count: a click the user spent
        # real time on is stronger evidence of interest than a bounce. read_time
        # lives in the history window, so it is strictly past information.
        kreads = np.nan_to_num(reads_arr[known], nan=0.0)
        engagement = (1.0 + np.log1p(np.clip(kreads, 0, 600))).astype(np.float32)

        kcat = art.category[kpos].astype(np.int64)
        valid_cat = kcat >= 0
        if valid_cat.any():
            np.add.at(lt_category[row], kcat[valid_cat], engagement[valid_cat])
            distinct_categories[row] = len(np.unique(kcat[valid_cat]))
            last_category[row] = kcat[valid_cat][-1]
            recent_categories[row, np.unique(kcat[valid_cat][-last_n:])] = True

        if n_sub:
            lt_subcategory[row] = (art.subcategory[kpos] * engagement[:, None]).sum(axis=0)
        if n_top:
            lt_topics[row] = (art.topics[kpos] * engagement[:, None]).sum(axis=0)
        if art.embeddings is not None:
            lt_embedding[row] = (art.embeddings[kpos] * engagement[:, None]).sum(axis=0)

        mean_sentiment[row] = float(np.nanmean(art.sentiment_score[kpos]))
        pub = art.published_at[kpos]
        age = kts - pub
        if np.isfinite(age).any():
            mean_article_age[row] = float(np.nanmean(age[np.isfinite(age)]))

    def _l1(mat: np.ndarray) -> np.ndarray:
        total = mat.sum(axis=1, keepdims=True)
        return mat / np.maximum(total, 1e-9)

    def _l2(mat: np.ndarray | None) -> np.ndarray | None:
        if mat is None:
            return None
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(norms, 1e-9)

    return UserProfiles(
        id_to_pos=id_to_pos,
        raw=UserHistoryRaw(kpos=raw_kpos, kts=raw_kts),
        lt_category=_l1(lt_category),
        lt_subcategory=_l1(lt_subcategory),
        lt_topics=_l1(lt_topics),
        lt_embedding=_l2(lt_embedding),
        history_len=history_len,
        mean_read_time=mean_read_time,
        median_read_time=median_read_time,
        mean_scroll=mean_scroll,
        last_click_at=last_click_at,
        first_click_at=first_click_at,
        last_category=last_category,
        recent_categories=recent_categories,
        mean_sentiment=mean_sentiment,
        mean_article_age_at_click=mean_article_age,
        distinct_categories=distinct_categories,
    )


# Clicks older than this, relative to the impression being scored, are excluded
# from the short-term decay sum entirely rather than merely down-weighted to
# near-zero. At the 24h half-life, 7 days = 7 half-lives -> a contribution of
# at most 2^-7 = 0.78% of the freshest possible click's weight -- below the
# float32 precision this project already truncates history-popularity counts
# to, so the truncation costs a bound on runtime, not measurable signal. Chosen
# a priori from the decay math, not tuned against validation.
SHORT_TERM_LOOKBACK_HOURS: float = 24.0 * 7.0


def compute_short_term_features(
    prof: UserProfiles,
    art: ArticleTable,
    *,
    imp_time: np.ndarray,
    user_pos: np.ndarray,
    group_starts: np.ndarray,
    group_sizes: np.ndarray,
    cand_cat: np.ndarray,
    apos_safe: np.ndarray,
    a_known: np.ndarray,
    half_life_hours: float = SHORT_TERM_HALF_LIFE_HOURS,
    lookback_hours: float = SHORT_TERM_LOOKBACK_HOURS,
) -> dict[str, np.ndarray]:
    """Per-IMPRESSION (not per-user) short-term profile, written directly into
    candidate-row-length arrays.

    This is the fix for the bug `UserHistoryRaw`'s docstring explains: decay
    weight is now computed against each impression's own `imp_time`, so a
    user's short-term profile genuinely moves forward as their impressions
    span the split, instead of being frozen at one split-wide reference.

    Deliberately does NOT return a `(n_impressions, dim)` matrix (an earlier
    draft did). At `ebnerd_large` scale that would be real: the 384-dim
    embedding matrix alone projects to several GB even at a capped
    `--train-sample-impressions`, on top of the training feature matrix
    itself -- a meaningful fraction of the 40G Ada job budget for a
    short-lived intermediate. Instead, each impression's decayed
    category/subcategory/topic/embedding vector is a transient local, used
    immediately to compute that impression's candidates' scalar features and
    then discarded -- the same total compute, a fraction of the peak memory.

    `np.searchsorted` on the user's time-sorted history bounds each
    impression's work to only clicks within `lookback_hours`, which is what
    keeps this affordable despite being called once per impression instead of
    once per user (~15x more calls, measured on ebnerd_small: 232,887
    impressions vs 15,143 users) against histories up to ~1,900 clicks long.
    """
    total = apos_safe.shape[0]
    n_cat = len(art.category_vocab)
    decay_lambda = np.log(2.0) / (half_life_hours * 3600.0)
    lookback_s = lookback_hours * 3600.0

    st_cat_affinity = np.zeros(total, dtype=np.float32)
    st_subcat_affinity = np.zeros(total, dtype=np.float32)
    st_topic_affinity = np.zeros(total, dtype=np.float32)
    st_embed_sim = np.full(total, np.nan, dtype=np.float32)
    st_cat_rank = np.full(total, np.nan, dtype=np.float32)
    clicks_last_24h = np.full(total, np.nan, dtype=np.float32)

    for i in range(len(user_pos)):
        start, size = int(group_starts[i]), int(group_sizes[i])
        sl = slice(start, start + size)
        u = user_pos[i]
        if u < 0:
            continue
        kts_full = prof.raw.kts[u]
        if kts_full.size == 0:
            clicks_last_24h[sl] = 0.0
            continue
        ref = imp_time[i]
        # `hi` is a hard `< ref` cut, not just a lower-bound truncation: it
        # excludes any history click at or after this impression's own time.
        # `history.parquet` is verified (separately, on real data) to close
        # strictly before any impression's `behaviors.parquet` window opens,
        # so in practice every element of `kts_full` already satisfies this --
        # but that is an external invariant this function used to lean on
        # without checking. A synthetic history containing a click after `ref`
        # was fed through the pre-fix version of this function directly and it
        # returned a non-zero, non-NaN affinity for a candidate matching that
        # future click's category -- not a leak that has occurred against real
        # EB-NeRD data, but a real gap: the function was not safe on its own
        # terms, only safe given how its one real caller happens to be used.
        # Explicit bounds make it correct for any input, not just today's data.
        lo = np.searchsorted(kts_full, ref - lookback_s, side="left")
        hi = np.searchsorted(kts_full, ref, side="left")
        lo24 = np.searchsorted(kts_full, ref - 86400.0, side="left")
        clicks_last_24h[sl] = float(hi - lo24)
        if lo >= hi:
            continue
        kpos = prof.raw.kpos[u][lo:hi]
        kts = kts_full[lo:hi]
        # `kts` is now guaranteed strictly < ref (enforced by `hi` above), so
        # `ref - kts` is always positive -- no clamp needed, and the absence
        # of one means a future violation would show up as a visible bug
        # (a negative argument to exp) rather than being silently absorbed.
        decay = np.exp(-decay_lambda * (ref - kts)).astype(np.float32)

        cats_slice = cand_cat[sl]
        aknown_slice = a_known[sl]
        valid_c = aknown_slice & (cats_slice >= 0)

        kcat = art.category[kpos]
        valid_h = kcat >= 0
        cat_vec = np.zeros(n_cat, dtype=np.float32)
        if valid_h.any():
            np.add.at(cat_vec, kcat[valid_h], decay[valid_h])
        cat_total = float(cat_vec.sum())
        if cat_total > 0 and valid_c.any():
            cat_vec_norm = cat_vec / cat_total
            st_cat_affinity[sl][valid_c] = cat_vec_norm[cats_slice[valid_c]]
            rank = np.argsort(np.argsort(-cat_vec_norm)).astype(np.float32)
            st_cat_rank[sl][valid_c] = rank[cats_slice[valid_c]]
        elif valid_c.any():
            st_cat_rank[sl][valid_c] = np.nan  # a user with zero decayed mass has no ranking

        if art.subcategory.shape[1] and aknown_slice.any():
            sub_vec = (art.subcategory[kpos] * decay[:, None]).sum(axis=0)
            total_sub = float(sub_vec.sum())
            if total_sub > 0:
                sub_vec = sub_vec / total_sub
                subhits = art.subcategory[apos_safe[sl]][aknown_slice].astype(np.float32)
                st_subcat_affinity[sl][aknown_slice] = subhits @ sub_vec

        if art.topics.shape[1] and aknown_slice.any():
            top_vec = (art.topics[kpos] * decay[:, None]).sum(axis=0)
            total_top = float(top_vec.sum())
            if total_top > 0:
                top_vec = top_vec / total_top
                tophits = art.topics[apos_safe[sl]][aknown_slice].astype(np.float32)
                st_topic_affinity[sl][aknown_slice] = tophits @ top_vec

        if art.embeddings is not None and aknown_slice.any():
            emb_vec = (art.embeddings[kpos] * decay[:, None]).sum(axis=0)
            norm = float(np.linalg.norm(emb_vec))
            if norm > 0:
                emb_vec = emb_vec / norm
                cand_embs = art.embeddings[apos_safe[sl]][aknown_slice]
                st_embed_sim[sl][aknown_slice] = cand_embs @ emb_vec

    return {
        "st_cat_affinity": st_cat_affinity,
        "st_subcat_affinity": st_subcat_affinity,
        "st_topic_affinity": st_topic_affinity,
        "st_embed_sim": st_embed_sim,
        "st_cat_rank": st_cat_rank,
        "clicks_last_24h": clicks_last_24h,
    }


def add_session_position_columns(behaviors: pd.DataFrame) -> pd.DataFrame:
    """Adds `session_position` (1-indexed, causal) and `session_start_gap_h`.

    Computed on the split's FULL impression-level table, once, before any
    chunking -- a session can span a chunk boundary in the streamed validation
    path, and getting this right requires seeing every impression in the
    session, not just the ones in the current chunk.

    Both outputs are causal by construction:

    - `session_position` = this impression's rank within its own
      `(user_id, session_id)` group after sorting by `impression_time` (ties
      broken by `impression_id` for determinism, matching ADR-007's tie-break
      discipline elsewhere) -- rank at position i never depends on any
      impression that sorts after it.
    - `session_start_gap_h` = `impression_time - group_min(impression_time)`.
      The group minimum is, by definition, at or before every member of the
      group, so referencing it from any impression in that group never reaches
      into the future -- unlike the bug this function's neighbours just fixed,
      there's no per-impression decay math here, so no analogous mistake to
      make.
    """
    behaviors = behaviors.copy()
    order = behaviors.sort_values(["user_id", "session_id", "impression_time", "impression_id"]).index
    grouped = behaviors.loc[order].groupby(["user_id", "session_id"])
    position = grouped.cumcount() + 1
    start = grouped["impression_time"].transform("min")
    gap_h = (behaviors.loc[order, "impression_time"] - start).dt.total_seconds() / 3600.0

    behaviors["session_position"] = position.reindex(behaviors.index)
    behaviors["session_start_gap_h"] = gap_h.reindex(behaviors.index)
    return behaviors


def build_history_popularity(
    zip_path: str | Path,
    split: str,
    art: ArticleTable,
    *,
    half_life_hours: float = SHORT_TERM_HALF_LIFE_HOURS,
) -> tuple[np.ndarray, np.ndarray]:
    """Article popularity counted from the *history* window only.

    This is the leak-safe substitute for the article-lifetime aggregates ADR-009
    quarantined. Counting clicks from `behaviors` instead would be doubly
    invalid: those clicks are the labels being predicted, and the real test set
    has no labels at all, so the feature could not be computed at serving time.

    Returns `(raw_count, decayed_count)`, both indexed by article row position.
    """
    hist = _read_zip_parquet(
        zip_path, f"{split}/history.parquet", columns=["article_id_fixed", "impression_time_fixed"]
    )

    n_articles = len(art.category)
    counts = np.zeros(n_articles, dtype=np.float32)
    decayed = np.zeros(n_articles, dtype=np.float32)

    all_last = np.array(
        [t[-1] for t in hist["impression_time_fixed"] if len(t)], dtype="datetime64[us]"
    )
    reference = all_last.max().astype("datetime64[s]").astype(np.float64)
    decay_lambda = np.log(2.0) / (half_life_hours * 3600.0)

    lookup = art.id_to_pos
    for arts, times in zip(hist["article_id_fixed"], hist["impression_time_fixed"]):
        if len(arts) == 0:
            continue
        pos = np.fromiter(
            (lookup.get(int(a), -1) for a in arts), dtype=np.int64, count=len(arts)
        )
        known = pos >= 0
        if not known.any():
            continue
        kpos = pos[known]
        kts = np.asarray(times, dtype="datetime64[s]").astype(np.float64)[known]
        np.add.at(counts, kpos, 1.0)
        np.add.at(
            decayed,
            kpos,
            np.exp(-decay_lambda * np.maximum(reference - kts, 0.0)).astype(np.float32),
        )

    return counts, decayed


# Feature names in the exact column order `build_feature_frame` emits.
# Grouped by family; ADR-013's audit table walks these one by one and names the
# serving-time source that makes each of them legitimate.
LONG_TERM_FEATURES = [
    "lt_cat_affinity",
    "lt_subcat_affinity",
    "lt_topic_affinity",
    "lt_embed_sim",
    "lt_cat_rank",
    "lt_sentiment_distance",
]
SHORT_TERM_FEATURES = [
    "st_cat_affinity",
    "st_subcat_affinity",
    "st_topic_affinity",
    "st_embed_sim",
    "st_cat_rank",
    "is_last_clicked_category",
    "in_recent_categories",
]
INTEREST_DRIFT_FEATURES = [
    "cat_affinity_drift",
    "embed_sim_drift",
    "topic_affinity_drift",
]
USER_FEATURES = [
    "log_history_len",
    "user_mean_read_time",
    "user_median_read_time",
    "user_mean_scroll",
    "user_distinct_categories",
    "user_clicks_last_24h",
    "user_mean_article_age_h",
    "hours_since_last_click",
    "user_history_span_h",
]
ITEM_FEATURES = [
    "article_age_h",
    "is_fresh_6h",
    "sentiment_score",
    "sentiment_label",
    "article_type",
    "is_premium",
    "title_len",
    "body_len",
    "n_topics",
    "n_entities",
    "log_history_popularity",
    "log_history_popularity_decayed",
    "category_code",
    "age_vs_user_appetite",
]
CONTEXT_FEATURES = [
    "n_candidates",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_front_page",
    "device_type",
    "is_sso_user",
    "is_subscriber",
    "user_age",
    "gender",
    "context_read_time",
    "context_scroll_percentage",
    "position_in_view",
    "relative_position_in_view",
]
# Within-impression normalisations. A GBDT sees each candidate row independently;
# these are what let it reason "this one is the freshest *of the ones on offer*"
# rather than "this one is 3.2 hours old", which is the comparison the ranking
# task actually turns on.
RELATIVE_FEATURES = [
    "lt_cat_affinity_rank_in_imp",
    "st_cat_affinity_rank_in_imp",
    "lt_embed_sim_rank_in_imp",
    "article_age_rank_in_imp",
    "history_popularity_rank_in_imp",
    "article_age_minus_imp_min",
    "lt_embed_sim_minus_imp_max",
]
# What the candidate matches against the article the user was ACTIVELY reading
# when this impression was served (`behaviors.article_id`, non-null for 29.1%
# of impressions on ebnerd_small validation, measured directly). More
# immediate than any history-derived signal -- "what are you reading right
# now" rather than "what have you read in the last 24h" -- and was missed in
# the first pass despite being part of the same impression row every other
# context feature already reads from (leak-safety tier: source C).
CONTEXT_ARTICLE_FEATURES = [
    "context_category_match",
    "context_topic_overlap",
    "context_embed_sim",
]
# Causal within-session structure -- named explicitly in the original feature
# brief ("session device/context") but only `device_type` was actually built
# in the first pass. 44.8% of (user, session) pairs on ebnerd_small validation
# have more than one impression, so this has real prevalence, not just
# theoretical relevance. See `add_session_position_columns` for the causality
# argument (both features only ever reference information at or before the
# impression they're attached to).
SESSION_FEATURES = [
    "session_position",
    "session_start_gap_h",
]

FEATURE_NAMES: list[str] = (
    LONG_TERM_FEATURES
    + SHORT_TERM_FEATURES
    + INTEREST_DRIFT_FEATURES
    + USER_FEATURES
    + ITEM_FEATURES
    + CONTEXT_FEATURES
    + CONTEXT_ARTICLE_FEATURES
    + SESSION_FEATURES
    + RELATIVE_FEATURES
)

# Columns LightGBM should treat as unordered categoricals rather than numeric.
CATEGORICAL_FEATURES: list[str] = [
    "sentiment_label",
    "article_type",
    "category_code",
    "device_type",
    "gender",
]

BEHAVIOR_COLUMNS: list[str] = [
    "impression_id",
    "user_id",
    "impression_time",
    "article_id",
    "article_ids_inview",
    "read_time",
    "scroll_percentage",
    "device_type",
    "is_sso_user",
    "is_subscriber",
    "gender",
    "age",
    "session_id",
]


def load_test_behaviors(zip_path: str | Path, split: str) -> pd.DataFrame:
    """Read a test split's behaviors, tolerating any `BEHAVIOR_COLUMNS` field
    the real bundle turns out not to carry.

    Lives here (not in `generate_ebnerd_gbdt_predictions.py`, where it
    originated) so tests can import it without pulling in `lightgbm` --
    that script imports lightgbm at module level, and loading lightgbm
    alongside a torch-based test in the same pytest process has already
    segfaulted this suite once (see `per_impression_auc`'s move to
    `src/evaluation/ranking_metrics.py` for the first occurrence).

    `article_ids_clicked` is deliberately not requested at all -- it doesn't
    exist in the blind test set, which is the whole point of the test set.

    `article_id` (the "context article the user was reading" field) was
    originally assumed present, since train/validation both carry it --
    checked directly against `ebnerd_testset`'s real schema and confirmed
    genuinely absent there (every other `BEHAVIOR_COLUMNS` field IS present).
    Rather than special-case that one column, this reads whatever
    `BEHAVIOR_COLUMNS` fields actually exist and NaN-fills any that don't --
    the same degrade-gracefully pattern used for optional history columns
    above, applied here because behaviors can differ by bundle the same way.
    A NaN-filled `article_id` column makes every downstream read (`.isna()`,
    the context-article-match features) behave exactly as it already does for
    an individual null row -- "no known context article" -- the honest state
    here, not an error condition.
    """
    present = _available_columns(zip_path, f"{split}/behaviors.parquet")
    missing = [c for c in BEHAVIOR_COLUMNS if c not in present]
    to_read = [c for c in BEHAVIOR_COLUMNS if c in present]
    if missing:
        print(f"      behaviors is missing column(s) {missing} -- NaN-filling", flush=True)

    raw = read_zip_member_bytes(Path(zip_path), f"{split}/behaviors.parquet")
    beh = pd.read_parquet(io.BytesIO(raw), columns=to_read)
    for column in missing:
        beh[column] = np.nan
    beh = beh[BEHAVIOR_COLUMNS]  # restore BEHAVIOR_COLUMNS' order regardless of what was missing
    # Must run on the FULL split before any caller chunks it -- a session can
    # span a chunk boundary.
    return add_session_position_columns(beh)


def _rank_within_group(values: np.ndarray, group_starts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Percentile rank of each value within its impression group (0=lowest, 1=highest).

    Implemented over the flat exploded array using each group's contiguous slice,
    which is valid because `build_feature_frame` explodes in impression order and
    never reorders. NaNs rank as 0.5 (neutral) so a missing value does not read as
    "worst in the list".
    """
    out = np.full(values.shape[0], 0.5, dtype=np.float32)
    for start, size in zip(group_starts, sizes):
        if size <= 1:
            continue
        chunk = values[start : start + size]
        finite = np.isfinite(chunk)
        if finite.sum() <= 1:
            continue
        order = np.argsort(np.argsort(np.where(finite, chunk, -np.inf)))
        out[start : start + size] = (order / (size - 1)).astype(np.float32)
    return out


def build_feature_frame(
    behaviors: pd.DataFrame,
    art: ArticleTable,
    prof: UserProfiles,
    popularity: tuple[np.ndarray, np.ndarray],
    *,
    with_labels: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Explode in-view candidates and compute the full per-candidate feature matrix.

    Returns `(X, meta)` where `X` is float32 `(n_candidate_rows, len(FEATURE_NAMES))`
    in `FEATURE_NAMES` order, and `meta` carries `impression_id`, `group_sizes`,
    and (when `with_labels`) the binary `label`.

    `behaviors` is taken as a slice so the caller can chunk; the test set is far
    too large to explode in one piece. `session_position`/`session_start_gap_h`
    must already be present as columns on `behaviors` (via
    `add_session_position_columns`, called once on the FULL split before any
    chunking) -- computing them per-chunk here would be wrong whenever a
    session spans a chunk boundary.
    """
    inview = behaviors["article_ids_inview"].to_numpy()
    sizes = np.array([len(v) for v in inview], dtype=np.int64)
    total = int(sizes.sum())
    group_starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)

    # Flat candidate article ids, and the per-impression row each one belongs to.
    cand_ids = np.concatenate([np.asarray(v, dtype=np.int64) for v in inview])
    imp_row = np.repeat(np.arange(len(behaviors), dtype=np.int64), sizes)
    position = np.concatenate([np.arange(s, dtype=np.int64) for s in sizes])

    lookup = art.id_to_pos
    apos = np.fromiter((lookup.get(int(a), -1) for a in cand_ids), dtype=np.int64, count=total)
    a_known = apos >= 0
    apos_safe = np.where(a_known, apos, 0)

    user_ids = behaviors["user_id"].to_numpy()
    upos_by_imp = np.fromiter(
        (prof.id_to_pos.get(int(u), -1) for u in user_ids), dtype=np.int64, count=len(behaviors)
    )
    upos = upos_by_imp[imp_row]
    u_known = upos >= 0
    upos_safe = np.where(u_known, upos, 0)

    imp_time = _to_epoch_seconds(pd.to_datetime(behaviors["impression_time"]))
    imp_time_flat = imp_time[imp_row]

    X = np.full((total, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    col = {name: i for i, name in enumerate(FEATURE_NAMES)}

    def put(name: str, values: np.ndarray) -> None:
        X[:, col[name]] = values.astype(np.float32, copy=False)

    cand_cat = art.category[apos_safe].astype(np.int64)
    cat_valid = a_known & (cand_cat >= 0)
    cat_safe = np.where(cat_valid, cand_cat, 0)
    valid = cat_valid & u_known

    # ---- Long-term affinities ---------------------------------------------
    def _affinity(matrix: np.ndarray) -> np.ndarray:
        out = np.zeros(total, dtype=np.float32)
        out[valid] = matrix[upos_safe[valid], cat_safe[valid]]
        return out

    put("lt_cat_affinity", _affinity(prof.lt_category))

    def _multi_hot_affinity(user_mat: np.ndarray, item_mat: np.ndarray) -> np.ndarray:
        out = np.zeros(total, dtype=np.float32)
        if item_mat.shape[1] == 0:
            return out
        sel = np.flatnonzero(a_known & u_known)
        for lo in range(0, sel.size, 500_000):
            idx = sel[lo : lo + 500_000]
            out[idx] = np.einsum(
                "ij,ij->i", user_mat[upos_safe[idx]], item_mat[apos_safe[idx]].astype(np.float32)
            )
        return out

    put("lt_subcat_affinity", _multi_hot_affinity(prof.lt_subcategory, art.subcategory))
    put("lt_topic_affinity", _multi_hot_affinity(prof.lt_topics, art.topics))

    def _embed_sim(user_mat: np.ndarray | None) -> np.ndarray:
        out = np.full(total, np.nan, dtype=np.float32)
        if user_mat is None or art.embeddings is None:
            return out
        sel = np.flatnonzero(a_known & u_known)
        for lo in range(0, sel.size, 500_000):
            idx = sel[lo : lo + 500_000]
            out[idx] = np.einsum(
                "ij,ij->i", user_mat[upos_safe[idx]], art.embeddings[apos_safe[idx]]
            )
        return out

    lt_sim = _embed_sim(prof.lt_embedding)
    put("lt_embed_sim", lt_sim)

    # Rank of the candidate's category within the user's own long-term preference ordering.
    def _cat_rank(matrix: np.ndarray) -> np.ndarray:
        ranks = np.argsort(np.argsort(-matrix, axis=1), axis=1).astype(np.float32)
        out = np.full(total, np.nan, dtype=np.float32)
        out[valid] = ranks[upos_safe[valid], cat_safe[valid]]
        return out

    put("lt_cat_rank", _cat_rank(prof.lt_category))

    cand_sent = np.where(a_known, art.sentiment_score[apos_safe], np.nan)
    user_sent = np.where(u_known, prof.mean_sentiment[upos_safe], np.nan)
    put("lt_sentiment_distance", np.abs(cand_sent - user_sent))

    last_cat = np.where(u_known, prof.last_category[upos_safe], -1)
    put("is_last_clicked_category", (cat_valid & (last_cat == cand_cat)).astype(np.float32))
    recent_hit = np.zeros(total, dtype=np.float32)
    recent_hit[valid] = prof.recent_categories[upos_safe[valid], cat_safe[valid]]
    put("in_recent_categories", recent_hit)

    # ---- Short-term affinities (per-impression, memory-disciplined) -------
    # See `compute_short_term_features`'s docstring for why this is computed
    # here rather than read off a per-user matrix -- the earlier per-user
    # version made "short-term interest" static across a user's whole split,
    # defeating the point of a short-term signal.
    st = compute_short_term_features(
        prof, art,
        imp_time=imp_time, user_pos=upos_by_imp,
        group_starts=group_starts, group_sizes=sizes,
        cand_cat=cand_cat, apos_safe=apos_safe, a_known=a_known,
    )
    put("st_cat_affinity", st["st_cat_affinity"])
    put("st_subcat_affinity", st["st_subcat_affinity"])
    put("st_topic_affinity", st["st_topic_affinity"])
    st_sim = st["st_embed_sim"]
    put("st_embed_sim", st_sim)
    put("st_cat_rank", st["st_cat_rank"])
    put("user_clicks_last_24h", st["clicks_last_24h"])

    put("cat_affinity_drift", X[:, col["st_cat_affinity"]] - X[:, col["lt_cat_affinity"]])
    put("embed_sim_drift", st_sim - lt_sim)
    put(
        "topic_affinity_drift",
        X[:, col["st_topic_affinity"]] - X[:, col["lt_topic_affinity"]],
    )

    # ---- User-level ------------------------------------------------------
    def _user_col(values: np.ndarray) -> np.ndarray:
        out = np.full(total, np.nan, dtype=np.float32)
        out[u_known] = values[upos_safe[u_known]]
        return out

    put("log_history_len", np.log1p(_user_col(prof.history_len.astype(np.float32))))
    put("user_mean_read_time", _user_col(prof.mean_read_time))
    put("user_median_read_time", _user_col(prof.median_read_time))
    put("user_mean_scroll", _user_col(prof.mean_scroll))
    put("user_distinct_categories", _user_col(prof.distinct_categories.astype(np.float32)))
    put("user_mean_article_age_h", _user_col(prof.mean_article_age_at_click) / 3600.0)

    last_click = np.full(total, np.nan, dtype=np.float64)
    last_click[u_known] = prof.last_click_at[upos_safe[u_known]]
    hours_since = (imp_time_flat - last_click) / 3600.0
    put("hours_since_last_click", hours_since)

    first_click = np.full(total, np.nan, dtype=np.float64)
    first_click[u_known] = prof.first_click_at[upos_safe[u_known]]
    put("user_history_span_h", (last_click - first_click) / 3600.0)

    # ---- Item-level ------------------------------------------------------
    pub = np.where(a_known, art.published_at[apos_safe], np.nan)
    age_h = (imp_time_flat - pub) / 3600.0
    put("article_age_h", age_h)
    put("is_fresh_6h", (age_h < 6.0).astype(np.float32))
    put("sentiment_score", cand_sent)
    put("sentiment_label", np.where(a_known, art.sentiment_label[apos_safe], -1))
    put("article_type", np.where(a_known, art.article_type[apos_safe], -1))
    put("is_premium", np.where(a_known, art.premium[apos_safe], False).astype(np.float32))
    put("title_len", np.where(a_known, art.title_len[apos_safe], np.nan))
    put("body_len", np.where(a_known, art.body_len[apos_safe], np.nan))
    put("n_topics", np.where(a_known, art.n_topics[apos_safe], np.nan))
    put("n_entities", np.where(a_known, art.n_entities[apos_safe], np.nan))
    put("category_code", np.where(cat_valid, cand_cat, -1))

    pop_raw, pop_dec = popularity
    put("log_history_popularity", np.log1p(np.where(a_known, pop_raw[apos_safe], 0.0)))
    put("log_history_popularity_decayed", np.log1p(np.where(a_known, pop_dec[apos_safe], 0.0)))
    put("age_vs_user_appetite", age_h - X[:, col["user_mean_article_age_h"]])

    # ---- Impression context ---------------------------------------------
    put("n_candidates", sizes[imp_row].astype(np.float32))
    dt = pd.to_datetime(behaviors["impression_time"])
    put("hour_of_day", dt.dt.hour.to_numpy()[imp_row])
    dow = dt.dt.dayofweek.to_numpy()
    put("day_of_week", dow[imp_row])
    put("is_weekend", (dow >= 5).astype(np.float32)[imp_row])
    context_article_null = behaviors["article_id"].isna().to_numpy()
    put("is_front_page", context_article_null.astype(np.float32)[imp_row])
    put("device_type", behaviors["device_type"].fillna(-1).to_numpy()[imp_row])
    put("is_sso_user", behaviors["is_sso_user"].fillna(False).to_numpy().astype(np.float32)[imp_row])
    put(
        "is_subscriber",
        behaviors["is_subscriber"].fillna(False).to_numpy().astype(np.float32)[imp_row],
    )
    put("user_age", behaviors["age"].to_numpy(dtype=np.float64)[imp_row])
    gender = behaviors["gender"].fillna(-1).to_numpy(dtype=np.float64)
    put("gender", gender[imp_row])
    put("context_read_time", behaviors["read_time"].to_numpy(dtype=np.float64)[imp_row])
    put(
        "context_scroll_percentage",
        behaviors["scroll_percentage"].to_numpy(dtype=np.float64)[imp_row],
    )
    put("position_in_view", position.astype(np.float32))
    put("relative_position_in_view", position / np.maximum(sizes[imp_row] - 1, 1))

    # ---- Context-article match (what the user is reading RIGHT NOW) -------
    # `behaviors.article_id`, non-null for 29.1% of impressions on
    # ebnerd_small validation (measured directly). Part of THIS impression's
    # own logged request -- same leak-safety tier as device_type -- but a more
    # immediate signal than anything derived from history, since it reflects
    # the article actually on screen at request time, not something read
    # minutes-to-days ago.
    ctx_id = behaviors["article_id"].to_numpy()
    ctx_apos = np.fromiter(
        (art.id_to_pos.get(int(a), -1) if not (a is None or a != a) else -1 for a in ctx_id),
        dtype=np.int64, count=len(behaviors),
    )
    ctx_known_by_imp = ctx_apos >= 0
    ctx_apos_safe_by_imp = np.where(ctx_known_by_imp, ctx_apos, 0)
    ctx_known = ctx_known_by_imp[imp_row]
    ctx_apos_safe = ctx_apos_safe_by_imp[imp_row]
    ctx_valid = ctx_known & a_known

    ctx_cat = art.category[ctx_apos_safe]
    put(
        "context_category_match",
        (ctx_valid & (ctx_cat >= 0) & (ctx_cat == cand_cat)).astype(np.float32),
    )

    def _overlap(item_mat: np.ndarray) -> np.ndarray:
        out = np.zeros(total, dtype=np.float32)
        if item_mat.shape[1] == 0:
            return out
        sel = np.flatnonzero(ctx_valid)
        for lo in range(0, sel.size, 500_000):
            idx = sel[lo : lo + 500_000]
            out[idx] = np.einsum(
                "ij,ij->i",
                item_mat[ctx_apos_safe[idx]].astype(np.float32),
                item_mat[apos_safe[idx]].astype(np.float32),
            )
        return out

    put("context_topic_overlap", _overlap(art.topics))

    ctx_sim = np.full(total, np.nan, dtype=np.float32)
    if art.embeddings is not None:
        sel = np.flatnonzero(ctx_valid)
        for lo in range(0, sel.size, 500_000):
            idx = sel[lo : lo + 500_000]
            ctx_sim[idx] = np.einsum(
                "ij,ij->i", art.embeddings[ctx_apos_safe[idx]], art.embeddings[apos_safe[idx]]
            )
    put("context_embed_sim", ctx_sim)

    # ---- Session (causal; see add_session_position_columns) ---------------
    if "session_position" not in behaviors.columns or "session_start_gap_h" not in behaviors.columns:
        raise ValueError(
            "behaviors is missing session_position/session_start_gap_h -- call "
            "add_session_position_columns on the FULL split before chunking. "
            "See ADR-013."
        )
    put("session_position", behaviors["session_position"].to_numpy(dtype=np.float64)[imp_row])
    put("session_start_gap_h", behaviors["session_start_gap_h"].to_numpy(dtype=np.float64)[imp_row])

    # ---- Within-impression relative features ------------------------------
    put(
        "lt_cat_affinity_rank_in_imp",
        _rank_within_group(X[:, col["lt_cat_affinity"]], group_starts, sizes),
    )
    put(
        "st_cat_affinity_rank_in_imp",
        _rank_within_group(X[:, col["st_cat_affinity"]], group_starts, sizes),
    )
    put("lt_embed_sim_rank_in_imp", _rank_within_group(lt_sim, group_starts, sizes))
    put("article_age_rank_in_imp", _rank_within_group(age_h, group_starts, sizes))
    put(
        "history_popularity_rank_in_imp",
        _rank_within_group(X[:, col["log_history_popularity"]], group_starts, sizes),
    )

    imp_min_age = pd.Series(age_h).groupby(imp_row).transform("min").to_numpy()
    put("article_age_minus_imp_min", age_h - imp_min_age)
    imp_max_sim = pd.Series(lt_sim).groupby(imp_row).transform("max").to_numpy()
    put("lt_embed_sim_minus_imp_max", lt_sim - imp_max_sim)

    meta: dict[str, np.ndarray] = {
        "impression_id": behaviors["impression_id"].to_numpy(),
        "group_sizes": sizes,
        "candidate_article_id": cand_ids,
    }
    if with_labels:
        clicked = behaviors["article_ids_clicked"].to_numpy()
        label = np.zeros(total, dtype=np.int8)
        for row, clicks in enumerate(clicked):
            if clicks is None or len(clicks) == 0:
                continue
            cset = set(int(c) for c in clicks)
            start = group_starts[row]
            for j, a in enumerate(inview[row]):
                if int(a) in cset:
                    label[start + j] = 1
        meta["label"] = label

    return X, meta
