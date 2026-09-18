"""Unit tests for scripts/a2_check_mind_history_coverage.py -- the
permanent pre-upload history-coverage gate added after ADR-015's
2026-09-17 bug (a broken query_by_user join silently scored every
MINDlarge_test impression with empty history; real Codabench score
0.5589 vs. 0.6868 local)."""
import importlib.util
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
UNLABELED_ZIP = FIXTURES / "MINDlarge_test_sample.zip"
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "a2_check_mind_history_coverage.py"

spec = importlib.util.spec_from_file_location("a2_check_mind_history_coverage", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_ground_truth_hit_rate_matches_fixture():
    # Fixture behaviors.tsv: U3 has real history "N1 N2"; U4 has none.
    rate, hit, total = mod.ground_truth_hit_rate(UNLABELED_ZIP)
    assert (hit, total) == (1, 2)
    assert rate == 0.5


def test_pipeline_hit_rate_matches_ground_truth_on_a_correct_pipeline(tmp_path):
    """The real regression guard: build_mind_test's own join must recover
    the SAME rate ground truth does, on real (fixture) data, not just
    produce *a* number."""
    gt_rate, _, _ = mod.ground_truth_hit_rate(UNLABELED_ZIP)
    pl_rate, hit, total = mod.pipeline_hit_rate(UNLABELED_ZIP, tmp_path / "data")
    assert (hit, total) == (1, 2)
    assert pl_rate == gt_rate


def test_cli_exits_zero_and_reports_ok_on_a_correct_pipeline(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--mind-test-zip", str(UNLABELED_ZIP), "--data-dir", str(tmp_path / "data")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: pipeline history coverage matches ground truth" in result.stdout


def test_cli_fails_when_max_abs_diff_is_impossibly_tight(tmp_path):
    """Sanity-inverts the gate: an impossibly strict tolerance must still
    make the script fail loudly, proving the threshold check is wired to
    something real rather than always printing OK."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--mind-test-zip", str(UNLABELED_ZIP), "--data-dir", str(tmp_path / "data"),
         "--max-abs-diff", "-1"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "FATAL" in result.stderr  # sys.exit(str) writes its message to stderr
