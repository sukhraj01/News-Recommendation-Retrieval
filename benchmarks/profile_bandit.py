#!/usr/bin/env python3
"""Per-decision latency / throughput of the bandit layer vs LightGBM-greedy (ADR-017).

One impression per call, no batching -- the same single-request convention as
`profile_ebnerd.py`'s `per_request` block, so numbers read straight across.

What is timed (wall clock, `time.perf_counter`, via the shared `StageTimer`):
  lgbm_greedy.predict_pick   booster.predict(raw 63-col rows) + argmax   <- the static baseline's decision
  <policy>.select            posterior-mean (+ UCB / sampled-theta) scoring of the candidates
  <policy>.update            one Sherman-Morrison rank-1 update (O(d^2))
  <policy>.decision          select + update, i.e. the per-impression online cost
  hybrid_*.total             lgbm predict + hybrid select + update -- what a hybrid costs in production,
                             because it needs the ranker's score as an input

Not timed (identical for every policy, so it cancels): the 65-feature build (0.2 ms/impression,
profile_ebnerd.py). Thompson's Cholesky is re-factorised only when the posterior changed, so
`lints.select` after a per-impression update includes it (reported) and an hourly-batch deployment
amortises it (reported separately as `*.select_cached_posterior`).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "scripts"))

import numpy as np
import pandas as pd

from benchmarks.timing import (StageTimer, git_dirty, git_rev, hardware_label, hardware_spec, markdown_table,
                               peak_rss_gb, stats_to_rows, write_results)
from bandit_prepare_stream import K_DIR, load_boosters
from run_bandit_experiment import load_stream, make_policy
from run_ebnerd_gbdt_experiment import _embeddings_paths, load_behaviors
from src.retrieval.ebnerd_features import (FEATURE_NAMES, build_feature_frame, build_history_popularity,
                                           build_user_profiles, load_article_table)

WARMUP = 300


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-impressions", type=int, default=5000)
    ap.add_argument("--tag", default="a3-bandit-layer")
    args = ap.parse_args()
    n = args.n_impressions
    zip_path = _REPO / "data" / "raw" / "ebnerd" / "ebnerd_small.zip"
    emb_npy, emb_json = _embeddings_paths("small")

    # ---- setup (not timed as decisions): raw features for the first n chronological impressions ----
    t0 = time.time()
    art = load_article_table(zip_path, embeddings_npy=emb_npy, embeddings_json=emb_json)
    prof = build_user_profiles(zip_path, "validation", art)
    pop = build_history_popularity(zip_path, "validation", art)
    beh = load_behaviors(zip_path, "validation")
    beh["_t"] = pd.to_datetime(beh["impression_time"])
    beh = beh.sort_values(["_t", "impression_id"], kind="stable").reset_index(drop=True).iloc[:n].reset_index(drop=True)
    X_raw, meta = build_feature_frame(beh, art, prof, pop, with_labels=True)
    boosters, cols = load_boosters()
    booster, keep = boosters["K_rank_nopos"], cols["K_rank_nopos"]
    sizes = meta["group_sizes"]; starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    X63 = np.ascontiguousarray(X_raw[:, keep])
    stream = load_stream(limit=n)
    assert np.array_equal(stream.sizes, sizes), "stream/raw misalignment"
    d0 = stream.base_dim
    setup_s = time.time() - t0

    timer = StageTimer()
    # LightGBM-greedy: predict + argmax per impression
    for i in range(n):
        rows = X63[starts[i]:starts[i] + sizes[i]]
        t = time.perf_counter()
        k = int(np.argmax(booster.predict(rows)))
        timer.record("lgbm_greedy.decision (predict+argmax)", time.perf_counter() - t)
    lgbm_samples = list(timer._samples["lgbm_greedy.decision (predict+argmax)"])

    specs = {
        "linucb": ("linucb", {"alpha": 0.03, "lam": 10}),
        "lints": ("lints", {"v": 0.03, "lam": 10}),
        "hybrid_linucb": ("hybrid_linucb", {"alpha": 0.003, "lam": 100, "w": 0.0}),
        "hybrid_lints": ("hybrid_lints", {"v": 0.1, "lam": 100, "w": 0.0}),
    }
    per_request: dict = {}
    for tag, (kind, params) in specs.items():
        pol = make_policy(kind, params, d0, 0)
        sel, upd = [], []
        for i in range(n):
            ctx, cand = stream.context(i), stream.candidates(i)
            t = time.perf_counter(); k = pol.select(ctx, cand); sel.append(time.perf_counter() - t)
            arm, r = stream.arm(i, k), float(stream.label[stream.offsets[i] + k])
            t = time.perf_counter(); pol.update(arm, r); upd.append(time.perf_counter() - t)
        sel, upd = np.array(sel[WARMUP:]), np.array(upd[WARMUP:])
        for name, arr in ((f"{tag}.select", sel), (f"{tag}.update", upd), (f"{tag}.decision (select+update)", sel + upd)):
            for v in arr:
                timer.record(name, float(v))
        if kind.endswith("lints"):  # cached-posterior select (hourly deployment): same state, no update in between
            ctx, cand = stream.context(n - 1), stream.candidates(n - 1)
            pol.select(ctx, cand)
            for _ in range(2000):
                t = time.perf_counter(); pol.select(ctx, cand); timer.record(f"{tag}.select_cached_posterior", time.perf_counter() - t)
        if tag.startswith("hybrid"):
            lg = np.array(lgbm_samples[WARMUP:])
            for v in lg + sel + upd:
                timer.record(f"{tag}.total (lgbm predict+select+update)", float(v))
        tot = (sel + upd) * 1e3
        per_request[tag] = {"n_requests": len(tot), "decision_ms": {"mean": float(tot.mean()), "p50": float(np.percentile(tot, 50)),
                            "p90": float(np.percentile(tot, 90)), "p99": float(np.percentile(tot, 99)), "max": float(tot.max())},
                            "select_ms_p50": float(np.percentile(sel * 1e3, 50)), "update_ms_p50": float(np.percentile(upd * 1e3, 50)),
                            "decisions_per_s": float(1.0 / (sel + upd).mean()), "dim": d0 + (2 if tag.startswith("hybrid") else 0)}
    stats = timer.summary()
    lg = np.array(lgbm_samples[WARMUP:]) * 1e3
    per_request["lgbm_greedy"] = {"n_requests": len(lg), "decision_ms": {"mean": float(lg.mean()), "p50": float(np.percentile(lg, 50)),
                                  "p90": float(np.percentile(lg, 90)), "p99": float(np.percentile(lg, 99)), "max": float(lg.max())},
                                  "decisions_per_s": float(1e3 / lg.mean())}
    payload = {
        "benchmark": "profile_bandit", "dataset": "ebnerd", "bundle": "small", "split": "validation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_rev": git_rev(), "git_dirty": git_dirty(), "tag": args.tag,
        "hardware_label": hardware_label(), "hardware": hardware_spec(),
        "sample": {"impressions": n, "warmup_discarded": WARMUP, "sampling": "first n chronological impressions", "mean_candidates": float(sizes.mean())},
        "model": {"path": str(K_DIR / "model_K_rank_nopos.txt"), "n_features": int(booster.num_feature()), "n_trees": int(booster.num_trees())},
        "setup": {"setup_s": setup_s, "linear_dim_d0": d0, "note": "BLAS threads are NOT pinned here; LightGBM uses its default threads, as in profile_ebnerd.py"},
        "per_request": per_request, "stages": stats_to_rows(stats), "peak_rss_gb": round(peak_rss_gb(), 2),
    }
    path = write_results("ebnerd_small_bandit_profile", payload)
    print(markdown_table(stats, unit="decision"))
    print("\nper-request decision latency (ms):")
    for k, v in per_request.items():
        d = v["decision_ms"]; print(f"  {k:16s} p50 {d['p50']:.4f}  p90 {d['p90']:.4f}  p99 {d['p99']:.4f}  {v['decisions_per_s']:,.0f}/s")
    print("->", path)


if __name__ == "__main__":
    main()
