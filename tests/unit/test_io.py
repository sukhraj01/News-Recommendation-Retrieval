"""Unit tests for src/utils/io.py, focused on read_zip_member_bytes's
wrapped-top-level-directory fallback (2026-08-12 addendum) — discovered
against the real ebnerd_testset.zip, which packs members at
`ebnerd_testset/test/behaviors.parquet` rather than the flat
`test/behaviors.parquet` layout every existing EB-NeRD caller assumes."""
import zipfile

import pytest

from src.utils.io import read_zip_member_bytes


def _make_zip(path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for name, content in members.items():
            z.writestr(name, content)


def test_exact_match_is_used_when_present(tmp_path):
    zip_path = tmp_path / "flat.zip"
    _make_zip(zip_path, {"test/behaviors.parquet": b"flat-content"})
    assert read_zip_member_bytes(zip_path, "test/behaviors.parquet") == b"flat-content"


def test_falls_back_to_wrapped_top_level_directory(tmp_path):
    zip_path = tmp_path / "wrapped.zip"
    _make_zip(zip_path, {"ebnerd_testset/test/behaviors.parquet": b"wrapped-content"})
    assert read_zip_member_bytes(zip_path, "test/behaviors.parquet") == b"wrapped-content"


def test_fallback_ignores_macosx_junk(tmp_path):
    zip_path = tmp_path / "wrapped_with_junk.zip"
    _make_zip(zip_path, {
        "ebnerd_testset/test/behaviors.parquet": b"real-content",
        "__MACOSX/ebnerd_testset/test/._behaviors.parquet": b"junk",
    })
    assert read_zip_member_bytes(zip_path, "test/behaviors.parquet") == b"real-content"


def test_ambiguous_fallback_raises(tmp_path):
    zip_path = tmp_path / "ambiguous.zip"
    _make_zip(zip_path, {
        "a/test/behaviors.parquet": b"one",
        "b/test/behaviors.parquet": b"two",
    })
    with pytest.raises(KeyError):
        read_zip_member_bytes(zip_path, "test/behaviors.parquet")


def test_missing_member_raises(tmp_path):
    zip_path = tmp_path / "empty.zip"
    _make_zip(zip_path, {"unrelated.txt": b"x"})
    with pytest.raises(KeyError):
        read_zip_member_bytes(zip_path, "test/behaviors.parquet")
