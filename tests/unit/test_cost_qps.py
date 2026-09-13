"""Regression test for a real bug: `cost_qps.py` once priced GPU-measured MIND
latency against a CPU-only instance (and, before that, the reverse) by always
selecting "the newest MIND profile" for both pricing rows regardless of which
device produced it. Fixed by filtering on `setup.nrms_device`."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import cost_qps  # noqa: E402


def _write_profile(dir_, name: str, device: str, mean_ms: float, timestamp: str) -> Path:
    data = {
        "dataset": "mind", "timestamp": timestamp, "setup": {"nrms_device": device},
        "per_request": {"n_requests": 10,
                        "request_total_ms": {"mean": mean_ms, "p50": mean_ms, "p90": mean_ms,
                                            "p99": mean_ms * 1.5, "max": mean_ms * 2}},
        "end_to_end_impressions_per_s": 100.0, "index_memory": {"total_mb": 1.0},
    }
    path = dir_ / name
    path.write_text(json.dumps(data))
    return path


def test_newest_filters_by_device_not_just_recency(tmp_path, monkeypatch):
    monkeypatch.setattr(cost_qps, "_RESULTS", tmp_path)
    # The GPU profile is NEWER than the CPU one -- a plain "most recent" pick
    # would wrongly return it when asked for the CPU-device profile.
    _write_profile(tmp_path, "a_cpu_profile.json", "cpu", mean_ms=33.0, timestamp="2026-01-01T00:00:00")
    _write_profile(tmp_path, "b_gpu_profile.json", "cuda", mean_ms=20.0, timestamp="2026-01-02T00:00:00")

    cpu = cost_qps.newest("*_profile.json", device="cpu")
    gpu = cost_qps.newest("*_profile.json", device="cuda")

    assert cpu["setup"]["nrms_device"] == "cpu"
    assert cpu["per_request"]["request_total_ms"]["mean"] == 33.0
    assert gpu["setup"]["nrms_device"] == "cuda"
    assert gpu["per_request"]["request_total_ms"]["mean"] == 20.0


def test_newest_without_device_filter_still_picks_most_recent(tmp_path, monkeypatch):
    monkeypatch.setattr(cost_qps, "_RESULTS", tmp_path)
    _write_profile(tmp_path, "a.json", "cpu", mean_ms=33.0, timestamp="2026-01-01T00:00:00")
    _write_profile(tmp_path, "b.json", "cuda", mean_ms=20.0, timestamp="2026-01-02T00:00:00")

    newest = cost_qps.newest("*.json")
    assert newest["setup"]["nrms_device"] == "cuda"  # the later timestamp, no filter applied


def test_newest_returns_none_when_no_profile_matches_the_requested_device(tmp_path, monkeypatch):
    monkeypatch.setattr(cost_qps, "_RESULTS", tmp_path)
    _write_profile(tmp_path, "a.json", "cpu", mean_ms=33.0, timestamp="2026-01-01T00:00:00")
    assert cost_qps.newest("*.json", device="cuda") is None


def test_analyse_never_mixes_a_profiles_device_with_the_wrong_instance():
    """The two output rows for the two instances must each carry the request
    latency FROM THE PROFILE PASSED IN -- analyse() itself has no device
    concept, so this pins that main()'s caller-side selection is what keeps
    them honest, by checking analyse is a pure function of its `profile` arg."""
    gpu_profile = {"dataset": "mind", "per_request": {"n_requests": 1,
                   "request_total_ms": {"mean": 20.0, "p99": 33.0}}}
    row = cost_qps.analyse(gpu_profile, sla_ms=100.0, instance="g4dn.xlarge")
    assert row["request_mean_ms"] == 20.0
    assert row["instance"] == "g4dn.xlarge"
