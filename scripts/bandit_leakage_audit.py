#!/usr/bin/env python3
"""Leakage / provenance audit for the bandit gains (ADR-017). TUNING PERIOD ONLY.

1. Label-permutation control: shuffle which candidate is "clicked" inside each tuning
   impression. Every policy must fall to ~random; any residual edge = a leak in the pipeline.
2. Single-feature scan: top-1 CTR of picking argmax of each lone feature. A feature that
   alone matches LightGBM would flag a leak.
3. Full-information static ridge: fit pointwise on the first half of the tuning period
   WITH all labels, evaluate frozen on the second half. Separates "online exploration"
   from "any model fitted on recent validation-window data beats a ranker fitted before it".
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "scripts"))
import numpy as np

import run_bandit_experiment as rbe
from src.bandit.runner import run_prequential
from src.bandit.stream import ImpressionStream

stream = rbe.load_stream()
tune, _, _ = rbe.split_periods(stream)
d0 = tune.base_dim
meta = json.loads((rbe.STREAM_DIR / "meta.json").read_text())
names = meta["feature_names"]
out: dict = {}

# ---- 1. label permutation ---------------------------------------------------
rng = np.random.default_rng(123)
lab = np.zeros_like(tune.label)
for i in range(len(tune)):
    r = tune.rows(i)
    lab[r.start + rng.integers(tune.sizes[i])] = 1
perm = ImpressionStream(features=tune.features, lgb_score=tune.lgb_score, label=lab, sizes=tune.sizes,
                        time=tune.time, user_id=tune.user_id, article_id=tune.article_id,
                        history_len=tune.history_len, base_dim=d0)
winners = json.loads((rbe.OUT_DIR / "tune.json").read_text())["winners"]
perm_out = {"random_expected_ctr": float(np.mean(1.0 / tune.sizes))}
for key in ("hybrid_linucb|per_impression", "hybrid_lints|per_impression", "linucb|per_impression", "lgbm_greedy"):
    kind, params, mode = (key.split("|") + ["per_impression"])[0], winners.get(key, {}).get("params", {}), \
        (key.split("|") + ["per_impression"])[1]
    pol = rbe.make_policy(kind, params, d0, 0)
    perm_out[key] = float(run_prequential(pol, perm, update_interval_s=rbe.MODES[mode]).reward.mean())
out["label_permutation"] = perm_out
print("1. label-permutation:", perm_out, flush=True)

# ---- 2. single-feature scan ---------------------------------------------------
H = np.asarray(tune.features)
rep = lambda v: np.repeat(v, tune.sizes)
scan = []
for j in range(d0 + 2):
    col = H[:, j].astype(np.float64)
    for sign in (1, -1):
        x = sign * col
        mx = np.maximum.reduceat(x, tune.offsets)
        eq = (x == rep(mx)).astype(np.float64)
        hit = np.add.reduceat(eq * tune.label, tune.offsets) / np.add.reduceat(eq, tune.offsets)
        scan.append((float(hit.mean()), (names + ["lgb_prob", "lgb_centred"])[j] if j < d0 else ["lgb_prob", "lgb_centred"][j - d0], sign))
scan.sort(reverse=True)
out["single_feature_top1"] = [{"top1_ctr": a, "feature": b, "sign": c} for a, b, c in scan[:8]]
lgb_ctr = float(np.mean([tune.label[tune.rows(i)][int(np.argmax(tune.lgb_score[tune.rows(i)]))] for i in range(len(tune))]))
out["lgb_top1_ctr_tune"] = lgb_ctr
print("2. best single features (top1 CTR):", [(round(a, 4), b, c) for a, b, c in scan[:8]], " | LGB:", round(lgb_ctr, 4), flush=True)

# ---- 3. full-information static ridge ----------------------------------------
half = len(tune) // 2
a = tune.window(0, half); b = tune.window(half, len(tune))
def top1(strm, theta, dim):
    pick = np.empty(len(strm), dtype=int); s = np.asarray(strm.features)[:, :dim].astype(np.float64) @ theta
    return float(np.mean([strm.label[strm.rows(i)][int(np.argmax(s[strm.rows(i)]))] for i in range(len(strm))]))
res3 = {"lgb_on_second_half": top1(b, np.eye(d0 + 2)[d0 + 1] * 0 + 0, d0 + 2) if False else float(np.mean(
    [b.label[b.rows(i)][int(np.argmax(b.lgb_score[b.rows(i)]))] for i in range(len(b))]))}
for dim, tag in ((d0, "linear_only"), (d0 + 2, "hybrid")):
    Xa = np.asarray(a.features)[:, :dim].astype(np.float64); ya = a.label.astype(np.float64)
    for lam in (1.0, 100.0, 1000.0):
        theta = np.linalg.solve(Xa.T @ Xa + lam * np.eye(dim), Xa.T @ ya)
        res3[f"static_ridge_{tag}_lam{lam:g}"] = top1(b, theta, dim)
out["full_information_static_ridge"] = res3
print("3. static full-info ridge (fit 1st half of tune, eval 2nd half):", res3, flush=True)
(rbe.OUT_DIR / "leakage_audit.json").write_text(json.dumps(out, indent=1))
