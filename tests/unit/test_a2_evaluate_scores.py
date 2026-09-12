"""Unit tests for scripts/a2_evaluate_scores.py: the A/B pairing guard and
paired-difference behaviour (ADR-015)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "a2_evaluate_scores.py"


def _scores(n_imp: int, shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_imp):
        labels = [1, 0, 0, 0, 0]
        base = rng.normal(size=5)
        base[0] += shift  # shift > 0 lifts the clicked item's score
        rows.append({"impression_id": i, "user_id": f"U{i % 40}", "labels": labels,
                     "scores": base.tolist()})
    return pd.DataFrame(rows)


def _run(tmp_path, ctrl: pd.DataFrame, treat: pd.DataFrame | None):
    c = tmp_path / "c.parquet"
    ctrl.to_parquet(c)
    args = [sys.executable, str(SCRIPT), "--control", str(c), "--out", str(tmp_path / "r.json")]
    if treat is not None:
        t = tmp_path / "t.parquet"
        treat.to_parquet(t)
        args += ["--treatment", str(t)]
    return subprocess.run(args, capture_output=True, text=True)


def test_identical_arms_give_exactly_zero_difference(tmp_path):
    df = _scores(200)
    res = _run(tmp_path, df, df)
    assert res.returncode == 0, res.stderr
    paired = json.loads((tmp_path / "r.json").read_text())["paired_treatment_minus_control"]
    for m in ("auc", "mrr", "ndcg5", "ndcg10"):
        assert paired[m]["diff"] == 0.0
        assert paired[m]["ci_low"] == 0.0 and paired[m]["ci_high"] == 0.0
        assert paired[m]["significant"] is False


def test_real_improvement_is_detected_as_significant(tmp_path):
    ctrl = _scores(400, shift=0.0, seed=1)
    treat = ctrl.copy()
    treat["scores"] = [list(np.asarray(s) + np.array([3.0, 0, 0, 0, 0])) for s in ctrl["scores"]]
    res = _run(tmp_path, ctrl, treat)
    assert res.returncode == 0, res.stderr
    auc = json.loads((tmp_path / "r.json").read_text())["paired_treatment_minus_control"]["auc"]
    assert auc["diff"] > 0 and auc["ci_low"] > 0 and auc["significant"]


def test_refuses_to_pair_arms_over_different_impressions(tmp_path):
    ctrl = _scores(100)
    treat = _scores(100).iloc[:90]
    res = _run(tmp_path, ctrl, treat)
    assert res.returncode != 0
    assert "different impressions" in (res.stderr + res.stdout)


def test_refuses_to_pair_when_candidate_labels_differ(tmp_path):
    ctrl = _scores(50)
    treat = ctrl.copy()
    treat.at[0, "labels"] = [0, 1, 0, 0, 0]
    res = _run(tmp_path, ctrl, treat)
    assert res.returncode != 0
    assert "labels differ" in (res.stderr + res.stdout)


def _mind_fixture(tmp_path, n_imp: int = 30):
    """A tiny MIND-format dev zip, A1-style processed articles and train
    impressions, and a scores frame aligned with the zip's raw candidate order."""
    import zipfile

    rng = np.random.default_rng(0)
    cats = ["news", "sports", "finance"]
    art_ids = [f"N{i}" for i in range(12)]
    lines, rows = [], []
    for i in range(1, n_imp + 1):
        cands = list(rng.choice(art_ids, size=6, replace=False))
        labels = [1] + [0] * 5
        lines.append(f"{i}\tU{i % 7}\t11/15/2019 8:00:00 AM\tN1 N2\t"
                     + " ".join(f"{c}-{lab}" for c, lab in zip(cands, labels)))
        rows.append({"impression_id": i, "user_id": f"U{i % 7}", "labels": labels,
                     "scores": rng.normal(size=6).tolist()})
    zpath = tmp_path / "dev.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("MINDlarge_dev/behaviors.tsv", "\n".join(lines) + "\n")
    arts = pd.DataFrame({"article_id": [f"mind:{a}" for a in art_ids],
                         "category": [cats[i % 3] for i in range(12)]})
    arts.to_parquet(tmp_path / "articles.parquet")
    train = pd.DataFrame({"article_id": [f"mind:N{i % 4}" for i in range(40)],
                          "clicked": [i % 2 == 0 for i in range(40)]})
    train.to_parquet(tmp_path / "train.parquet")
    return zpath, pd.DataFrame(rows)


def _guardrail_args(tmp_path, zpath):
    return ["--dataset", "mind", "--source-zip", str(zpath),
            "--articles", str(tmp_path / "articles.parquet"),
            "--train-impressions", str(tmp_path / "train.parquet")]


def test_guardrails_computed_and_self_pair_has_no_regression(tmp_path):
    zpath, df = _mind_fixture(tmp_path)
    c = tmp_path / "c.parquet"
    df.to_parquet(c)
    res = subprocess.run([sys.executable, str(SCRIPT), "--control", str(c), "--treatment", str(c),
                          "--out", str(tmp_path / "r.json")] + _guardrail_args(tmp_path, zpath),
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    rep = json.loads((tmp_path / "r.json").read_text())
    assert rep["guardrails"] is True and rep["tiebreak_prefix"] == "mind:dev:"
    for g in ("diversity10", "novelty10"):
        assert 0 <= rep["control"][g]["value"] or g == "novelty10"
        assert rep["paired_treatment_minus_control"][g]["diff"] == 0.0
        assert rep["paired_treatment_minus_control"][g]["regressed"] is False
    assert 0 < rep["control"]["coverage10"]["value"] <= 1
    assert rep["paired_treatment_minus_control"]["coverage10"]["diff"] == 0.0


def test_guardrails_refuse_when_runner_labels_disagree_with_raw_zip(tmp_path):
    zpath, df = _mind_fixture(tmp_path)
    df.at[0, "labels"] = [0, 1, 0, 0, 0, 0]  # runner claims a different click than the raw log
    c = tmp_path / "c.parquet"
    df.to_parquet(c)
    res = subprocess.run([sys.executable, str(SCRIPT), "--control", str(c),
                          "--out", str(tmp_path / "r.json")] + _guardrail_args(tmp_path, zpath),
                         capture_output=True, text=True)
    assert res.returncode != 0
    assert "do not match the raw zip" in (res.stderr + res.stdout)


def _slice_inputs():
    """cands: impression -> (ids, labels); popularity: train-click probabilities."""
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "scripts"))
    from a2_evaluate_scores import assign_slices  # noqa: E402

    # A1 is the most-clicked article, A4 the least; A9 was never clicked in train.
    popularity = {"a:A1": 0.4, "a:A2": 0.3, "a:A3": 0.2, "a:A4": 0.1}
    cands = {
        1: (["a:A1", "a:A9"], [1, 0]),   # clicked the most popular -> head
        2: (["a:A4", "a:A9"], [1, 0]),   # clicked the least popular -> tail
        3: (["a:A9", "a:A1"], [1, 0]),   # clicked an unseen-in-train article -> tail
        4: (["a:A1", "a:A9"], [0, 0]),   # no positive -> none
    }
    pi = pd.DataFrame({"impression_id": [1, 2, 3, 4], "user_id": ["U1", "U2", "U3", "U4"]})
    return assign_slices, pi, cands, popularity


def test_head_tail_uses_first_clicked_article_and_train_popularity():
    assign_slices, pi, cands, popularity = _slice_inputs()
    out = assign_slices(pi, cands, popularity, head_frac=0.25, hist_len=None, id_prefix="a:")
    assert list(out["slice_pop"]) == ["head", "tail", "tail", "none"]


def test_unseen_in_train_articles_fall_in_tail_not_head():
    """The smoothed floor must not promote an unclicked article into the head."""
    assign_slices, pi, cands, popularity = _slice_inputs()
    out = assign_slices(pi, cands, popularity, head_frac=0.99, hist_len=None, id_prefix="a:")
    assert out.loc[out["impression_id"] == 3, "slice_pop"].item() == "tail"


def test_head_frac_widens_the_head_set():
    assign_slices, pi, cands, popularity = _slice_inputs()
    narrow = assign_slices(pi, cands, popularity, 0.25, None, "a:")["slice_pop"].tolist()
    wide = assign_slices(pi, cands, popularity, 1.0, None, "a:")["slice_pop"].tolist()
    assert narrow[1] == "tail" and wide[1] == "head"  # impression 2 flips


def test_warm_cold_uses_a1_threshold_and_prefixed_ids():
    assign_slices, pi, cands, popularity = _slice_inputs()
    hist = {"a:U1": 4, "a:U2": 5, "a:U3": 50}  # U4 absent -> unknown
    out = assign_slices(pi, cands, popularity, 0.25, hist, "a:")
    assert list(out["slice_cohort"]) == ["cold", "warm", "warm", "unknown"]


@pytest.mark.parametrize("n", [60])
def test_control_only_reports_all_metrics(tmp_path, n):
    res = _run(tmp_path, _scores(n), None)
    assert res.returncode == 0, res.stderr
    rep = json.loads((tmp_path / "r.json").read_text())
    assert set(rep["control"]) >= {"auc", "mrr", "ndcg5", "ndcg10"}
    assert "paired_treatment_minus_control" not in rep
