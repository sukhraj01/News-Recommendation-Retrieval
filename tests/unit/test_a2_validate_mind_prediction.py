"""Unit tests for scripts/a2_validate_mind_prediction.py -- the real
validator run against the 2,370,727-line MINDlarge_test submission
(ADR-015's 2026-09-17 addendum) before it was trusted."""
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "a2_validate_mind_prediction.py"


def _run(tmp_path, lines: list[str]) -> str:
    p = tmp_path / "prediction.txt"
    p.write_text("\n".join(lines) + "\n")
    return subprocess.run([sys.executable, str(SCRIPT), str(p)], capture_output=True, text=True).stdout


def test_valid_file_reports_zero_problems(tmp_path):
    out = _run(tmp_path, ["1 [3,1,2]", "2 [1]", "3 [2,1]"])
    assert "total lines checked: 3" in out
    assert "unique impression ids: 3" in out
    assert "duplicates: 0" in out
    assert "malformed/non-permutation: 0" in out


def test_duplicate_impression_id_detected(tmp_path):
    out = _run(tmp_path, ["1 [1]", "1 [1]"])
    assert "duplicates: 1" in out


def test_non_permutation_detected(tmp_path):
    out = _run(tmp_path, ["1 [1,1,3]"])
    assert "malformed/non-permutation: 1" in out
    assert "NOT A PERMUTATION" in out


def test_malformed_line_detected(tmp_path):
    out = _run(tmp_path, ["not a valid line at all"])
    assert "malformed/non-permutation: 1" in out
    assert "MALFORMED" in out


def test_spaced_rank_list_rejected(tmp_path):
    """The official format is JSON with no internal spaces
    (`separators=(",", ":")`); a rank list with a stray space is malformed,
    not silently tolerated."""
    out = _run(tmp_path, ["1 [1, 2]"])
    assert "malformed/non-permutation: 1" in out
