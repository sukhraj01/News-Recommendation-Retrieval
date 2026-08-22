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


def test_falls_back_to_flat_root_when_expected_folder_is_absent(tmp_path):
    # 2026-08-21 addendum: the Hugging Face MIND mirror's MINDlarge_train.zip
    # packs news.tsv/behaviors.tsv flat at the root, with no
    # `MINDlarge_train/` folder — the opposite mismatch from the
    # wrapped-extra-directory case above.
    zip_path = tmp_path / "flat_root.zip"
    _make_zip(zip_path, {"news.tsv": b"flat-root-content"})
    assert read_zip_member_bytes(zip_path, "MINDlarge_train/news.tsv") == b"flat-root-content"


def test_suffix_fallback_preferred_over_flat_root_fallback(tmp_path):
    # If a suffix match exists, it wins even if a same-named flat entry is
    # also present - preserves the pre-existing, already-tested behavior
    # rather than changing precedence.
    zip_path = tmp_path / "both.zip"
    _make_zip(zip_path, {
        "ebnerd_testset/test/behaviors.parquet": b"suffix-content",
        "behaviors.parquet": b"flat-content",
    })
    assert read_zip_member_bytes(zip_path, "test/behaviors.parquet") == b"suffix-content"


def test_flat_root_fallback_ignores_macosx_junk(tmp_path):
    zip_path = tmp_path / "flat_with_junk.zip"
    _make_zip(zip_path, {
        "news.tsv": b"real-content",
        "__MACOSX/._news.tsv": b"junk",
    })
    assert read_zip_member_bytes(zip_path, "MINDlarge_train/news.tsv") == b"real-content"
