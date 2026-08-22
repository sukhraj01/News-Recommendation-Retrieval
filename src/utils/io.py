"""Zip-aware raw file readers and a parquet write wrapper.

Raw MIND/EB-NeRD bundles stay zipped on disk (data/raw/, gitignored);
nothing is extracted. EB-NeRD zips also contain __MACOSX/ junk metadata
entries — callers must read the exact named member, never glob the zip.

`read_zip_member_bytes`'s fallback addendum (2026-08-12, Part 2 execution):
`ebnerd_small.zip`/`ebnerd_demo.zip` both pack members at a flat path
(`articles.parquet`, `{split}/behaviors.parquet`) and every EB-NeRD caller
in this project (`src/datasets/ebnerd.py`, `src/submission/ebnerd_format.py`)
was written and tested against that layout. The real `ebnerd_testset.zip`
does not follow it — running Part 2 for real on Kaggle surfaced that its
members are wrapped in an extra top-level directory
(`ebnerd_testset/test/behaviors.parquet`, not `test/behaviors.parquet`),
which raised a `KeyError` on the first real read. Rather than hardcoding
that one prefix (an `articles_large_only.zip`-shaped bundle could easily
use a different one, and guessing would just move the same fragility
elsewhere), the exact-path lookup is tried first (unchanged, fast, and
still what every existing fixture/test exercises) and only falls back to
a suffix search across the real namelist (excluding `__MACOSX/` junk) if
that fails — so every caller's already-correct flat member names keep
working unmodified against either packaging convention.

Second fallback addendum (2026-08-21, MINDlarge Kaggle verification, ADR-010):
the opposite mismatch. `src/datasets/mind.py` always requests
`f"{zip_path.stem}/news.tsv"` (e.g. `MINDlarge_train/news.tsv`) because
every MIND zip downloaded from the official source packs members inside a
folder named after the zip itself — confirmed directly at this project's
very first session and true of every local MIND zip since. The Hugging
Face mirror this project's own `download.py` uses as its download source
does NOT follow that convention for `MINDlarge_train.zip`: real download,
inspected directly, packs `news.tsv`/`behaviors.tsv`/etc. flat at the zip
root with no folder prefix at all — the suffix fallback above can't match
this (it requires a `/` before `member_name`, which a genuinely flat entry
never has), so it raised `KeyError` with zero suffix candidates on a real
Kaggle run. Same fix philosophy as the first addendum: try the caller's
existing exact/suffix-based lookups first (unchanged), and only fall back
further, to an exact basename match among flat (no `/`) entries, if both
fail — never guessing which convention a *new* bundle will use, just
handling the ones actually observed in the wild so far.
"""
import zipfile
from pathlib import Path

import pandas as pd


def read_zip_member_bytes(zip_path: Path, member_name: str) -> bytes:
    with zipfile.ZipFile(zip_path) as z:
        try:
            with z.open(member_name) as f:
                return f.read()
        except KeyError:
            pass

        namelist = [n for n in z.namelist() if not n.startswith("__MACOSX/")]
        suffix_candidates = [n for n in namelist if n.endswith("/" + member_name)]
        basename = member_name.rsplit("/", 1)[-1]
        flat_candidates = [n for n in namelist if n == basename]
        candidates = suffix_candidates or flat_candidates

        if len(candidates) != 1:
            raise KeyError(
                f"{member_name!r} not found in {zip_path} as an exact "
                f"member name, and neither fallback found exactly 1 "
                f"candidate (suffix: {suffix_candidates}, flat: {flat_candidates})"
            ) from None
        with z.open(candidates[0]) as f:
            return f.read()


def read_zip_tsv(zip_path: Path, member_name: str, names: list[str]) -> pd.DataFrame:
    import io

    raw = read_zip_member_bytes(zip_path, member_name)
    return pd.read_csv(
        io.BytesIO(raw),
        sep="\t",
        header=None,
        names=names,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )


def read_zip_parquet(zip_path: Path, member_name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """`columns=None` reads every column (default, unchanged behavior).
    Passing an explicit subset avoids materializing columns a caller never
    reads — real cost at real scale (see `ebnerd_format.py`'s addendum)."""
    import io

    raw = read_zip_member_bytes(zip_path, member_name)
    return pd.read_parquet(io.BytesIO(raw), columns=columns)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame the caller has already deterministically sorted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
