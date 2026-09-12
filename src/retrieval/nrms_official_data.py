"""Input semantics of the two official NRMS data loaders, ported (ADR-015, Option B).

The model port (`nrms_official.py`) is only a faithful reproduction if the
inputs it sees are built the way the official loaders build them. Each
function below cites the official behaviour it reproduces:

  microsoft/recommenders `newsrec/io/mind_iterator.py` + `newsrec_utils.py` (MIND)
  ebnerd-benchmark        `ebrec/utils/_behaviors.py`                      (EB-NeRD)

Exact RNG streams cannot be matched across TF/polars and numpy, so sampling
reproduces the *distribution* (which items can be drawn, with or without
replacement, padded how), not the identical draws.
"""
import random
import re
from typing import Sequence

import numpy as np

_MIND_TOKEN_RE = re.compile(r"[\w]+|[.,!?;|]")


def mind_word_tokenize(sent) -> list[str]:
    """recommenders `word_tokenize`: regex `[\\w]+|[.,!?;|]` over the
    lowercased sentence; a non-str input gives []."""
    if isinstance(sent, str):
        return _MIND_TOKEN_RE.findall(sent.lower())
    return []


def mind_encode_text(text, word_dict: dict[str, int], size: int) -> np.ndarray:
    """MINDIterator.init_news: the first `size` tokens, each mapped through
    `word_dict`, with unknown words -> 0; right-padded with 0."""
    out = np.zeros(size, dtype=np.int32)
    toks = mind_word_tokenize(text)
    for i in range(min(size, len(toks))):
        out[i] = word_dict.get(toks[i], 0)
    return out


def build_mind_news_tokens(
    news_ids: Sequence[str], titles: Sequence, abstracts: Sequence | None,
    word_dict: dict[str, int], title_size: int, abstract_size: int = 0,
) -> tuple[dict[str, int], np.ndarray]:
    """Returns `(nid2index, tokens)`. Row 0 of `tokens` is the official
    dummy news (all zeros, used for history padding and newsample padding),
    and real news start at index 1, as in MINDIterator.

    `abstract_size > 0` is the MIND *treatment* (ADR-015): each row becomes
    [title segment (title_size) | abstract segment (abstract_size)]. The
    title segment is encoded exactly as the control's, so the only
    difference between arms is the added abstract tokens."""
    width = title_size + abstract_size
    tokens = np.zeros((len(news_ids) + 1, width), dtype=np.int32)
    nid2index: dict[str, int] = {}
    for i, nid in enumerate(news_ids, start=1):
        nid2index[nid] = i
        tokens[i, :title_size] = mind_encode_text(titles[i - 1], word_dict, title_size)
        if abstract_size:
            tokens[i, title_size:] = mind_encode_text(abstracts[i - 1], word_dict, abstract_size)
    return nid2index, tokens


def left_pad_recent(ids: Sequence[int], size: int, pad: int = 0) -> list[int]:
    """Both official loaders keep the MOST RECENT `size` history items,
    left-padded: MIND `[0]*(his_size-len(h)) + h[-his_size:]`, EB-NeRD
    `list.tail(history_size)` with left padding."""
    ids = list(ids)[-size:] if size > 0 else []
    return [pad] * (size - len(ids)) + ids


def mind_newsample(negs: Sequence[int], ratio: int, rng: random.Random) -> list[int]:
    """recommenders `newsample`: fewer negatives than `ratio` -> pad with
    dummy news 0 (NOT resampling); otherwise sample `ratio` without
    replacement."""
    negs = list(negs)
    if ratio > len(negs):
        return negs + [0] * (ratio - len(negs))
    return rng.sample(negs, ratio)


def to_epoch_seconds(ts) -> np.ndarray:
    """Unit-agnostic datetime -> float seconds since 1970-01-01 UTC.

    Exists because of a real bug (ADR-015, 2026-09-12). `astype("int64")`
    returns the column's *storage* unit, and pandas >= 2 keeps a pyarrow
    `timestamp[us]` column as `datetime64[us]`. Dividing that by 1e9
    (assuming nanoseconds) made every EB-NeRD publish time ~1000x too small,
    so every candidate aged ~53 years, and the freshness treatment became a
    constant shift per impression, which is a silent no-op under softmax.
    Subtracting the epoch and dividing by a Timedelta is correct for any
    stored unit."""
    import pandas as pd

    s = pd.to_datetime(pd.Series(ts))
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return ((s - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)).to_numpy(dtype=np.float64)


def ebnerd_sample_negatives(
    negs: Sequence[int], npratio: int, rng: np.random.Generator,
) -> list[int] | None:
    """ebnerd-benchmark `sampling_strategy_wu2019(..., with_replacement=True)`:
    `npratio` negatives drawn with replacement from the in-view list minus
    the clicked articles. Returns None when there is no negative to draw
    from; the caller drops that sample rather than invent one."""
    negs = list(negs)
    if not negs:
        return None
    return [negs[j] for j in rng.integers(0, len(negs), size=npratio)]
