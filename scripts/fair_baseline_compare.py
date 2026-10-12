#!/usr/bin/env python3
"""Fair-baseline comparison (ADR-018 addendum): retrained GBDT with causal trailing-CTR features vs the
forgetting hybrid bandit, per feedback delay D, on held-out days 3-6.

Per D (all selection on days 0-2 only, then frozen):
  original      LightGBM-greedy, K_rank_nopos (ADR-017 stream score)
  gbdt+ctr      LightGBM retrained with trailing-CTR features computed with delay D (scripts/trailing_ctr_gbdt.py)
  blend         p_lgb + beta * smoothed trailing CTR (window in {30m,2h,6h,24h}, beta grid) with delay D
  forgetting    hybrid LinUCB with exponential forgetting; gamma/lambda re-tuned UNDER delay D; reward released
                to the policy D seconds after the pick (runner.delay_s)
Information asymmetry, stated up front: the GBDT features and the blend see the labels of EVERY candidate
of every impression older than D (full information); the bandit only ever sees the reward of its own picks.
Timing convention: trailing features use observations with obs_time < t - D (strict, so a row's own label can
never enter); the bandit receives a pick's reward once pick_time + D <= decision_time (inclusive). The difference
is at most second-level ties.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "scripts"))
import numpy as np

import run_bandit_experiment as rbe
import run_drift_experiment as rde
from src.bandit.evaluation import factorize_users, final_window_mask, ratio_ci
from src.bandit.runner import run_prequential
from src.bandit.trailing import smoothed_ctr, trailing_counts
from src.evaluation.ranking_metrics import per_impression_auc

OUT = rde.OUT
CACHE = _REPO / "data" / "processed" / "ebnerd" / "small" / "trailing_cache"
DELAYS = [0, 900, 1800, 3600]
BLEND_WINDOWS = {"30min": 1800, "2h": 7200, "6h": 21600, "24h": 86400}
BETAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
HYBRID_GRID = [(g, l) for g in (0.998, 0.999, 0.9995, 0.9999) for l in (1, 10, 100)]

stream = rbe.load_stream()
tune, held, n_tune = rbe.split_periods(stream)
d0 = stream.base_dim
users_h, nu_h = factorize_users(held.user_id)
base_params = json.loads((OUT / "tune_forget.json").read_text())["winners"]["hybrid_linucb|per_impression|forget"]["params"]

# ---- timeline (train + validation), identical construction to trailing_ctr_gbdt.py -------------------
tr, va = np.load(CACHE / "tr_meta.npz"), np.load(CACHE / "va_meta.npz")
sizes_all = np.concatenate([tr["sizes"], va["sizes"]])
art_all = np.concatenate([tr["article"], va["article"]])
_, art_idx = np.unique(art_all, return_inverse=True)
t_imp = np.concatenate([tr["time"], va["time"]])
t_row = np.repeat(np.round(t_imp - t_imp.min()).astype(np.int64), sizes_all)
y_all = np.concatenate([tr["label"], va["label"]])
n_tr_rows = len(tr["label"])
lgb_prob = np.asarray(stream.features[:, d0], dtype=np.float64)


def hits(scores):
    return rde.topk_hit(np.asarray(scores, dtype=np.float64), stream)


def ci(num, den=1.0, mask=None):
    v, lo, hi, n = ratio_ci(num, den, users_h, nu_h, mask=mask)
    return {"value": v, "ci_low": lo, "ci_high": hi, "n": n}


# ---- bandit tuning/final under delay (fork pool) ---------------------------------------------------------
_S = None


def _bandit_task(args):
    D, gamma, lam, which = args
    s = tune if which == "tune" else held
    p = {**base_params, "gamma": gamma, "lam": lam}
    res = run_prequential(rbe.make_policy("hybrid_linucb", p, d0, 0), s, delay_s=float(D) if D > 0 else None)
    return args, res.reward


def tune_and_run_bandit(D, workers=7):
    with mp.get_context("fork").Pool(workers) as pool:
        rows = pool.map(_bandit_task, [(D, g, l, "tune") for g, l in HYBRID_GRID])
    scores = {(a[1], a[2]): float(r.mean()) for a, r in rows}
    (g, l) = max(scores, key=scores.get)
    _, rew = _bandit_task((D, g, l, "held"))
    return {"gamma": g, "lam": l, "tuning_ctr": scores[(g, l)], "grid_tuning_ctr": {f"{a}|{b}": v for (a, b), v in scores.items()}}, rew


# ---- per-D comparison ------------------------------------------------------------------------------------
lgb_orig_hit = hits(stream.lgb_score)
auc_orig = per_impression_auc(stream.lgb_score.astype(np.float64), stream.label, stream.sizes)
fw = final_window_mask(len(held), 0.2)
result = {"delays_s": DELAYS, "n_held_out": len(held), "tune_n": n_tune, "per_delay": {}}
for D in DELAYS:
    t0 = time.time()
    score_ctr = np.load(rbe.STREAM_DIR / f"gbdt_ctr_D{D}_score.npy")
    hit_ctr = hits(score_ctr)
    # blend: choose (window, beta) on days 0-2
    best = (-1.0, None, None)
    for wn, ws in BLEND_WINDOWS.items():
        c, s = trailing_counts(art_idx, t_row, y_all, delay_s=D, window_s=ws)
        ctr = smoothed_ctr(c, s)[n_tr_rows:]
        for beta in BETAS:
            h = hits(lgb_prob + beta * ctr)
            tc = float(h[:n_tune].mean())
            if tc > best[0]:
                best = (tc, (wn, beta), h)
    blend_hit = best[2]
    bandit_meta, bandit_rew = tune_and_run_bandit(D)
    H = slice(n_tune, None)
    arrays = {"original": lgb_orig_hit[H], "gbdt_plus_trailing_ctr": hit_ctr[H], "popularity_blend": blend_hit[H], "forgetting_hybrid": bandit_rew}
    entry = {"blend_selected_on_tuning": {"window": best[1][0], "beta": best[1][1], "tuning_ctr": best[0]},
             "forgetting_hybrid_selected_on_tuning": bandit_meta, "policies": {}}
    for name, arr in arrays.items():
        e = {}
        for wn, m in (("overall", None), ("final 20%", fw)):
            w = {"p_at_1": ci(arr, 1.0, m)}
            for ref in ("original", "forgetting_hybrid"):
                if name != ref:
                    w[f"delta_vs_{ref}"] = ci(arr - arrays[ref], 1.0, m)
            e[wn] = w
        entry["policies"][name] = e
    # AUC (rankers only): days 3-6, paired
    auc_ctr = per_impression_auc(score_ctr.astype(np.float64), stream.label, stream.sizes)[n_tune:]
    a0 = auc_orig[n_tune:]
    valid = np.isfinite(a0) & np.isfinite(auc_ctr)
    entry["auc_days_3_6"] = {"original": ci(np.where(valid, a0, 0.0), valid.astype(float)),
                             "gbdt_plus_trailing_ctr": ci(np.where(valid, auc_ctr, 0.0), valid.astype(float)),
                             "paired_delta": ci(np.where(valid, auc_ctr - a0, 0.0), valid.astype(float))}
    # fraction of impressions on which gbdt+ctr and the bandit disagree is not needed; keep per-day P@1 for the story
    day = ((held.time - held.time[0]) // 86400).astype(int)
    entry["per_day_p_at_1"] = {name: [float(arr[day == d].mean()) for d in range(day.max() + 1)] for name, arr in arrays.items()}
    result["per_delay"][str(D)] = entry
    f = lambda d: f"{d['value']:+.4f} [{d['ci_low']:+.4f},{d['ci_high']:+.4f}]"
    print(f"D={D}s ({time.time() - t0:.0f}s)  P@1: " + "  ".join(f"{n} {entry['policies'][n]['overall']['p_at_1']['value']:.4f}" for n in arrays)
          + f" | gbdt+ctr vs hybrid {f(entry['policies']['gbdt_plus_trailing_ctr']['overall']['delta_vs_forgetting_hybrid'])}", flush=True)
(OUT / "fair_baseline.json").write_text(json.dumps(result, indent=1))
