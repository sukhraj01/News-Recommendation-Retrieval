"""Unit coverage for `src/pipeline/download.py::download_mind_bundle`'s
retry/resume behavior -- added after a real failure downloading
MINDlarge_train.zip (531MB) over a throttled connection hit
`ContentTooShortError` with the original single-shot `urlretrieve`
implementation (no retry, no resume)."""
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.download import download_mind_bundle


class _FakeResponse:
    """Minimal stand-in for `http.client.HTTPResponse` as used via
    `urlopen(...) as response`: needs `.status` and `.read(size)`."""

    def __init__(self, chunks: list[bytes], status: int = 200):
        self._chunks = list(chunks)
        self.status = status

    def read(self, _size: int) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_download_mind_bundle_returns_existing_file_without_network_call(tmp_path):
    dest = tmp_path / "MINDsmall_train.zip"
    dest.write_bytes(b"already here")

    with patch("urllib.request.urlopen") as mock_urlopen:
        result = download_mind_bundle("MINDsmall_train.zip", tmp_path)

    assert result == dest
    mock_urlopen.assert_not_called()


def test_download_mind_bundle_unknown_filename_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Unknown MIND bundle"):
        download_mind_bundle("not_a_real_bundle.zip", tmp_path)


def test_download_mind_bundle_succeeds_in_one_pass(tmp_path):
    with patch("urllib.request.urlopen", return_value=_FakeResponse([b"hello ", b"world"])):
        result = download_mind_bundle("MINDsmall_train.zip", tmp_path)

    assert result == tmp_path / "MINDsmall_train.zip"
    assert result.read_bytes() == b"hello world"
    assert not (tmp_path / "MINDsmall_train.zip.part").exists()  # renamed away, not left behind


def test_download_mind_bundle_resumes_after_a_dropped_connection(tmp_path):
    """First attempt writes some bytes then the connection drops; the
    second attempt must request a Range starting from what was already
    written, and the final file must be the full, uncorrupted concatenation
    -- not a duplicate or a gap."""
    call_log = []

    def fake_urlopen(request, timeout=60):
        call_log.append(request.get_header("Range"))
        if len(call_log) == 1:
            return _FakeResponse([b"hello "])  # then simulate a drop below
        return _FakeResponse([b"world"], status=206)

    real_read = _FakeResponse.read
    call_count = {"n": 0}

    def flaky_read(self, size):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise ConnectionError("simulated drop")
        return real_read(self, size)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
         patch.object(_FakeResponse, "read", flaky_read), \
         patch("time.sleep"):
        result = download_mind_bundle("MINDsmall_train.zip", tmp_path)

    assert result.read_bytes() == b"hello world"
    assert call_log[0] is None  # first attempt: no Range header (nothing written yet)
    assert call_log[1] == "bytes=6-"  # second attempt: resumes from byte 6 ("hello " is 6 bytes)


def test_download_mind_bundle_raises_after_max_retries_exhausted(tmp_path):
    with patch("urllib.request.urlopen", side_effect=ConnectionError("always fails")), \
         patch("time.sleep"):
        with pytest.raises(ConnectionError):
            download_mind_bundle("MINDsmall_train.zip", tmp_path, max_retries=2)


def test_download_mind_bundle_restarts_if_server_ignores_range_header(tmp_path):
    """A server that doesn't support resume returns 200 (full content from
    byte 0) even when a Range header was sent -- must overwrite, not append
    the full response onto the existing partial bytes (which would
    duplicate/corrupt the file)."""
    part_path = tmp_path / "MINDsmall_train.zip.part"
    part_path.write_bytes(b"stale partial ")

    with patch("urllib.request.urlopen", return_value=_FakeResponse([b"full content"], status=200)):
        result = download_mind_bundle("MINDsmall_train.zip", tmp_path)

    assert result.read_bytes() == b"full content"  # not "stale partial full content"
