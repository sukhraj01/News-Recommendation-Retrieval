"""Zip-aware raw file readers and a parquet write wrapper.

Raw MIND/EB-NeRD bundles stay zipped on disk (data/raw/, gitignored);
nothing is extracted. EB-NeRD zips also contain __MACOSX/ junk metadata
entries — callers must read the exact named member, never glob the zip.
"""
import zipfile
from pathlib import Path

import pandas as pd


def read_zip_member_bytes(zip_path: Path, member_name: str) -> bytes:
    with zipfile.ZipFile(zip_path) as z:
        with z.open(member_name) as f:
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


def read_zip_parquet(zip_path: Path, member_name: str) -> pd.DataFrame:
    import io

    raw = read_zip_member_bytes(zip_path, member_name)
    return pd.read_parquet(io.BytesIO(raw))


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame the caller has already deterministically sorted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
