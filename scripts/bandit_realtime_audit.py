#!/usr/bin/env python3
"""Audit of the forgetting hybrid's large gain (ADR-017 addendum).

The discounted-update variants beat LightGBM-greedy by ~+0.047 precision@1, far more than the original
hybrid's +0.011. That is the "suspiciously good" case, so before believing it:

1. Label-permutation control on the forgetting winners (TUNING period): must collapse to random.
2. Real-time-popularity baseline (no bandit at all): rank in-view articles by trailing-window clicks /
   smoothed CTR computed from the labels of strictly EARLIER impressions (FULL information, i.e. more
   than a bandit sees). If this reproduces the gain, the forgetting hybrid is learning "what is being
   clicked right now", a signal Candidate K cannot have by design (ADR-013 uses history-window popularity
   only). Blend weight and window are chosen on days 0-2, frozen, scored on days 3-6.
3. Feedback-delay sensitivity: the per-impression mode assumes a click is learnable the instant it happens.
   Re-run the frozen winners with the reward becoming available only `delay` seconds later. POST-HOC
   evaluation of frozen policies on days 3-6, no re-tuning.
4. Coverage facts: how often the clicked article had recent clicks before, and how often the same user
   had already clicked it.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "scripts"))
import numpy as np

import run_bandit_experiment as rbe
import run_drift_experiment as rde
from src.bandit.evaluation import factorize_users, ratio_ci
from src.bandit.runner import run_prequential
from src.bandit.stream import ImpressionStream

OUT = rde.OUT
stream = rbe.load_stream()
tune, held, n_tune = rbe.split_periods(stream)
d0 = stream.base_dim
winners = json.loads((OUT / "tune_forget.json").read_text())["winners"]
base_w = json.loads((rbe.OUT_DIR / "tune.json").read_text())["winners"]
ref_npz = np.load(rbe.OUT_DIR / "heldout_runs.npz")
out: dict = {}


def ci(num, users, n_users, mask=None):
    v, lo, hi, n = ratio_ci(num, 1.0, users, n_users, mask=mask)
    return {"value": v, "ci_low": lo, "ci_high": hi, "n": n}


# ---- 1. label permutation on the forgetting winners (tuning period) -----------------------------
rng = np.random.default_rng(321)
lab = np.zeros_like(tune.label)
for i in range(len(tune)):
    lab[tune.rows(i).start + rng.integers(tune.sizes[i])] = 1
perm = ImpressionStream(features=tune.features, lgb_score=tune.lgb_score, label=lab, sizes=tune.sizes, time=tune.time,
                        user_id=tune.user_id, article_id=tune.article_id, history_len=tune.history_len, base_dim=d0)
p1 = {"random_expected": float(np.mean(1.0 / tune.sizes))}
for key in ("hybrid_linucb|per_impression|forget", "hybrid_linucb|per_impression|window", "hybrid_linucb|per_impression|reset",
            "hybrid_lints|per_impression|forget", "hybrid_linucb|hourly|forget"):
    w = winners[key]
    pol = rbe.make_policy(w["kind"], w["params"], d0, 0)
    p1[key] = float(run_prequential(pol, perm, update_interval_s=rbe.MODES[w["mode"]]).reward.mean())
out["1_label_permutation_tuning_period"] = p1
print("1.", p1, flush=True)

# ---- 2. real-time popularity baselines with full information on earlier impressions ---------------
WINDOWS = {"30min": 1800.0, "2h": 7200.0, "6h": 21600.0, "24h": 86400.0}
N = len(stream)


def trailing_counts(window_s):
    clicks_q: dict = defaultdict(deque); shows_q: dict = defaultdict(deque)
    c = np.zeros(len(stream.article_id)); s = np.zeros(len(stream.article_id))
    for i in range(N):
        t = stream.time[i]; r = stream.rows(i)
        for k, a in enumerate(stream.article_id[r]):
            a = int(a)
            sq, cq = shows_q[a], clicks_q[a]
            while sq and sq[0] < t - window_s: sq.popleft()
            while cq and cq[0] < t - window_s: cq.popleft()
            c[r.start + k], s[r.start + k] = len(cq), len(sq)
        for k, a in enumerate(stream.article_id[r]):  # only AFTER scoring: labels of impression i become history
            a = int(a)
            shows_q[a].append(t)
            if stream.label[r.start + k]:
                clicks_q[a].append(t)
    return c, s


users_h, nu_h = factorize_users(held.user_id)
lgb_hit_full = rde.topk_hit(stream.lgb_score.astype(np.float64), stream)
lgb_prob = np.asarray(stream.features[:, d0], dtype=np.float64)
p0, m = 0.1, 5.0
base_cols = {}
rt = {"blend_grid_tuning_ctr": {}, "frozen": {}}
best = (-1, None)
for wn, ws in WINDOWS.items():
    c, s = trailing_counts(ws)
    ctr = (c + m * p0) / (s + m)
    base_cols[wn] = (c, s, ctr)
    # pure recent-clicks, ties -> LightGBM
    pure = c + 1e-3 * lgb_prob
    hit_pure = rde.topk_hit(pure, stream)
    rt["frozen"][f"pure_recent_clicks_{wn}"] = {"tuning_ctr": float(hit_pure[:n_tune].mean()), "heldout_ctr": ci(hit_pure[n_tune:], users_h, nu_h),
                                                "delta_vs_lgbm": ci(hit_pure[n_tune:] - lgb_hit_full[n_tune:], users_h, nu_h)}
    for beta in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        hit = rde.topk_hit(lgb_prob + beta * ctr, stream)
        tc = float(hit[:n_tune].mean())
        rt["blend_grid_tuning_ctr"][f"{wn}|beta={beta}"] = tc
        if tc > best[0]:
            best = (tc, (wn, beta), hit)
wn, beta = best[1]
hit = best[2]
rt["selected_on_tuning"] = {"window": wn, "beta": beta, "tuning_ctr": best[0]}
rt["frozen"]["blend_lgbm_prob_plus_beta_x_smoothed_recent_ctr"] = {
    "window": wn, "beta": beta, "heldout_ctr": ci(hit[n_tune:], users_h, nu_h),
    "delta_vs_lgbm": ci(hit[n_tune:] - lgb_hit_full[n_tune:], users_h, nu_h)}
fw = rde.final_window_mask(len(held), 0.2)
rt["frozen"]["blend_delta_vs_lgbm_final20"] = ci(hit[n_tune:] - lgb_hit_full[n_tune:], users_h, nu_h, fw)
out["2_realtime_popularity_baseline"] = rt
print("2.", rt["selected_on_tuning"], rt["frozen"]["blend_lgbm_prob_plus_beta_x_smoothed_recent_ctr"], flush=True)

# ---- 4. coverage facts on the held-out period ------------------------------------------------
c30, s30, _ = base_cols["30min"]; c2h, _, _ = base_cols["2h"]
clicked_c30 = np.where(stream.label > 0, c30, np.inf); clicked_c2h = np.where(stream.label > 0, c2h, np.inf)
min30 = np.minimum.reduceat(clicked_c30, stream.offsets)[n_tune:]
min2h = np.minimum.reduceat(clicked_c2h, stream.offsets)[n_tune:]
seen_user = defaultdict(set); repeat = np.zeros(N, dtype=bool)
for i in range(N):
    r = stream.rows(i); u = int(stream.user_id[i])
    cl = [int(a) for a in stream.article_id[r][stream.label[r] > 0]]
    repeat[i] = any(a in seen_user[u] for a in cl)
    seen_user[u].update(cl)
out["4_coverage_heldout"] = {
    "clicked_article_had_a_click_in_previous_30min": float((min30 >= 1).mean()),
    "clicked_article_had_a_click_in_previous_2h": float((min2h >= 1).mean()),
    "same_user_had_clicked_this_article_before": float(repeat[n_tune:].mean())}
print("4.", out["4_coverage_heldout"], flush=True)

# ---- 3. feedback-delay sensitivity (frozen policies, held-out) -------------------------------------
lgb_r = ref_npz["lgbm_greedy|per_impression__reward"][0]
specs = {"hybrid_linucb non-forgetting": ("hybrid_linucb", base_w["hybrid_linucb|per_impression"]["params"]),
         "hybrid_linucb forgetting (gamma winner)": ("hybrid_linucb", winners["hybrid_linucb|per_impression|forget"]["params"]),
         "hybrid_linucb window (winner)": ("hybrid_linucb", winners["hybrid_linucb|per_impression|window"]["params"])}
dl = {}
for name, (kind, params) in specs.items():
    dl[name] = {}
    for delay in (0.0, 60.0, 300.0, 900.0, 1800.0, 3600.0, 7200.0):
        res = run_prequential(rbe.make_policy(kind, params, d0, 0), held, delay_s=delay if delay > 0 else None)
        dl[name][f"{int(delay)}s"] = {"ctr": ci(res.reward, users_h, nu_h), "delta_vs_lgbm": ci(res.reward - lgb_r, users_h, nu_h)}
        print("3.", name, int(delay), round(res.reward.mean(), 4), flush=True)
out["3_feedback_delay_sensitivity_heldout_frozen"] = dl
(OUT / "realtime_audit.json").write_text(json.dumps(out, indent=1))
