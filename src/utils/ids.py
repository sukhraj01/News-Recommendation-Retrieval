"""Dataset-aware ID prefixing.

article_id is stable across a dataset's own splits (ADR-002's cold-start
counts confirm dev-window articles overlap with train), so it is prefixed
by dataset only. impression_id and user_id are split-qualified: MINDsmall's
train/dev impression_id ranges both restart at 1, and EB-NeRD's train/
validation impression_id ranges overlap heavily (confirmed empirically) even
though they didn't collide in the demo bundle — split-qualifying both
datasets uniformly is the only safe, single-code-path-consistent choice.
"""
from .config import ID_DELIMITER


def prefix_id(dataset: str, raw_id, *qualifiers: str) -> str:
    """Build a globally-unique, dataset-prefixed ID.

    prefix_id("mind", "N55528") -> "mind:N55528"
    prefix_id("mind", 1, "train") -> "mind:train:1"
    """
    parts = [dataset, *qualifiers, str(raw_id)]
    return ID_DELIMITER.join(parts)
