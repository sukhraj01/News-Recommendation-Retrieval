#!/usr/bin/env python3
"""Drift hypothesis + mechanism analysis for the hybrid bandit (ADR-017 addendum).

Stages (all reuse the ADR-017 stream, frozen LightGBM score, tuning = days 0-2, held-out = days 3-6,
user-level bootstrap n=2000; nothing here is tuned on days 3-6):

  tune     discounted-update variants (exponential forgetting / sliding window / periodic reset) of the
           frozen hybrid winners; only the forgetting parameter and lambda are searched, on days 0-2.
  final    freeze winners, run once on days 3-6 (5 seeds for Thompson); CIs + paired deltas vs
           LightGBM-greedy and vs the non-forgetting hybrid, per day and final-20%.
  diag     drift diagnosis: weight movement of the non-forgetting hybrid, ranker staleness, composition
           check, full-information day-to-day transfer matrix, article-age slice of the final 20%.
  groups   feature-group ablation (drop one group / keep LightGBM score + one group), tuned on days 0-2,
           frozen, run once on days 3-6.

`diag`'s transfer matrix fits an OFFLINE full-information ridge on held-out-period labels. That is a
diagnostic of how stable the optimal correction is, not a policy and not a tuning step.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

import numpy as np

import run_bandit_experiment as rbe
from src.bandit.evaluation import factorize_users, final_window_mask, ratio_ci
from src.bandit.groups import GROUPS, assign_groups
from src.bandit.runner import run_prequential

OUT = rbe.OUT_DIR / "drift"
FAMILIES = ["hybrid_linucb", "hybrid_lints"]
MODES = ["per_impression", "hourly"]
# Widened ONCE after the first pass (kept in drift/tune_forget_first_pass.json) put window=1000, reset=5000 and
# lambda in {10, 1000} on its edges. First pass: window [1e3..6e4], reset [5e3..6e4], lambda [10,100,1000].
VARIANT_GRID = {
    "forget": ("gamma", [0.99999, 0.99995, 0.9999, 0.9995, 0.999, 0.995]),
    "window": ("window", [100, 300, 1000, 3000, 10000, 30000, 60000]),
    "reset": ("reset_every", [1000, 2500, 5000, 15000, 30000, 60000]),
}
LAMBDAS = [1, 3, 10, 100, 1000, 3000]
REF_KEYS = [f"{f}|{m}" for f in FAMILIES for m in MODES]
GROUP_NAMES = list(GROUPS) + ["onehot_categoricals"]
FIVE_NON_LGB = [g for g in GROUP_NAMES if g != "lightgbm_score"]


def ci_row(num, den, users, n_users, mask=None):
    v, lo, hi, n = ratio_ci(num, den, users, n_users, mask=mask)
    return {"value": v, "ci_low": lo, "ci_high": hi, "n": n}


def windows_for(held):
    n = len(held)
    day = ((held.time - held.time[0]) // 86400).astype(int)
    w = {"overall": np.ones(n, dtype=bool)}
    for d in range(day.max() + 1):
        w[f"day {d + 3}"] = day == d
    w["final 20%"] = final_window_mask(n, 0.2)
    return w


def best_per_group(rows, keyfn):
    best = {}
    for r in sorted(rows, key=lambda r: -r["tune_ctr"]):
        best.setdefault(keyfn(r), r)
    return best


# ------------------------------------------------------------------ tune / final: forgetting
def stage_tune(tune, workers):
    base = json.loads((rbe.OUT_DIR / "tune.json").read_text())["winners"]
    tasks, meta = [], {}
    for fam, mode in itertools.product(FAMILIES, MODES):
        w = base[f"{fam}|{mode}"]
        for variant, (pname, values) in VARIANT_GRID.items():
            for val, lam in itertools.product(values, LAMBDAS):
                params = {**w["params"], "lam": lam, pname: val}
                tasks.append((fam, params, mode, rbe.TUNE_SEED, False))
                meta[(fam, json.dumps(params, sort_keys=True), mode)] = variant
    print(f"[tune] {len(tasks)} runs on {len(tune):,} impressions", flush=True)
    t0 = time.time()
    rows = []
    for task, arm, reward, dt in rbe._pool_map(tasks, tune, workers):
        fam, params, mode, _, _ = task
        rows.append({"kind": fam, "mode": mode, "variant": meta[(fam, json.dumps(params, sort_keys=True), mode)],
                     "params": params, "tune_ctr": float(reward.mean())})
    print(f"[tune] done in {time.time() - t0:.0f}s", flush=True)
    winners = best_per_group(rows, lambda r: f"{r['kind']}|{r['mode']}|{r['variant']}")
    for key, w in winners.items():
        pname, values = VARIANT_GRID[w["variant"]]
        w["on_param_grid_edge"] = bool(w["params"][pname] in (values[0], values[-1]))
        w["on_lambda_grid_edge"] = bool(w["params"]["lam"] in (LAMBDAS[0], LAMBDAS[-1]))
    refs = {k: base[k]["tune_ctr"] for k in REF_KEYS}
    return {"grid": rows, "winners": winners, "non_forgetting_reference_tune_ctr": refs,
            "variant_grid": VARIANT_GRID, "lambdas": LAMBDAS}


def heldout_runs(held, plan, workers):
    """plan: list of (key, kind, params, mode). Returns key -> (arm (S,N), reward (S,N))."""
    tasks = []
    for key, kind, params, mode in plan:
        seeds = rbe.FINAL_SEEDS if kind in rbe.STOCHASTIC else [0]
        tasks += [(kind, {**params, "_key": key}, mode, s, False) for s in seeds]
    out: dict = {}
    for task, arm, reward, dt in rbe._pool_map(tasks, held, workers):
        kind, params, mode, seed, _ = task
        out.setdefault(params["_key"], []).append((seed, arm, reward))
    res = {}
    for key, lst in out.items():
        lst.sort(key=lambda t: t[0])
        res[key] = (np.stack([a for _, a, _ in lst]), np.stack([r for _, _, r in lst]))
    return res


def stage_final(held, tune_out, workers):
    plan = []
    for key, w in tune_out["winners"].items():
        p = {k: v for k, v in w["params"].items()}
        plan.append((key, w["kind"], p, w["mode"]))
    print(f"[final] {len(plan)} frozen variants on {len(held):,} held-out impressions", flush=True)
    runs = heldout_runs(held, plan, workers)
    ref_npz = np.load(rbe.OUT_DIR / "heldout_runs.npz")
    ref = {k: ref_npz[f"{k}__reward"].mean(axis=0) for k in REF_KEYS}
    lgb = ref_npz["lgbm_greedy|per_impression__reward"][0]
    users, n_users = factorize_users(held.user_id)
    wins = windows_for(held)
    report = {"windows": {k: int(m.sum()) for k, m in wins.items()}, "policies": {}}
    for key in [*REF_KEYS, *runs]:
        if key in ref:
            rew, seed_std, params = ref[key], None, "frozen ADR-017 winner"
            fam_mode = key
        else:
            r = runs[key][1]
            rew, seed_std, params = r.mean(axis=0), (float(r.mean(axis=1).std(ddof=1)) if len(r) > 1 else 0.0), tune_out["winners"][key]["params"]
            fam_mode = "|".join(key.split("|")[:2])
        entry = {"params": params, "seed_std": seed_std, "windows": {}}
        for wn, m in wins.items():
            e = {"ctr": ci_row(rew, 1.0, users, n_users, m), "lgbm_ctr": float(lgb[m].mean()),
                 "delta_vs_lgbm": ci_row(rew - lgb, 1.0, users, n_users, m)}
            if key not in ref:
                e["delta_vs_nonforgetting_hybrid"] = ci_row(rew - ref[fam_mode], 1.0, users, n_users, m)
            entry["windows"][wn] = e
        report["policies"][key] = entry
    return report, runs


# ------------------------------------------------------------------ diag
def topk_hit(scores, stream):
    """Per-impression expected top-1 hit of argmax(scores), ties averaged."""
    mx = np.maximum.reduceat(scores, stream.offsets)
    eq = (scores == np.repeat(mx, stream.sizes)).astype(np.float64)
    return np.add.reduceat(eq * stream.label, stream.offsets) / np.add.reduceat(eq, stream.offsets)


def stage_diag(stream, held, workers):
    from run_ebnerd_gbdt_experiment import _embeddings_paths
    from src.retrieval.ebnerd_features import load_article_table

    out: dict = {}
    names = rbe._feature_names()
    d0 = stream.base_dim
    users, n_users = factorize_users(held.user_id)
    hday = ((held.time - held.time[0]) // 86400).astype(int)
    ref_npz = np.load(rbe.OUT_DIR / "heldout_runs.npz")
    lgb_r = ref_npz["lgbm_greedy|per_impression__reward"][0]
    winners = json.loads((rbe.OUT_DIR / "tune.json").read_text())["winners"]

    # 5a -- weight movement of the non-forgetting hybrid (re-run; must reproduce the stored rewards exactly)
    wdiag = {}
    for mode in MODES:
        w = winners[f"hybrid_linucb|{mode}"]
        pol = rbe.make_policy("hybrid_linucb", w["params"], d0, 0)
        snaps = {}

        def hook(i, p, snaps=snaps):
            if i == len(held) - 1 or hday[i + 1] != hday[i]:
                snaps[int(hday[i])] = p.model.theta.copy()

        res = run_prequential(pol, held, update_interval_s=rbe.MODES[mode], hook=hook)
        ref_rew = ref_npz[f"hybrid_linucb|{mode}__reward"][0]
        assert np.array_equal(res.reward, ref_rew), "diagnostic re-run does not reproduce the stored held-out run"
        days = sorted(snaps)
        th = np.stack([snaps[d] for d in days])
        cos = [float(th[i] @ th[i + 1] / (np.linalg.norm(th[i]) * np.linalg.norm(th[i + 1]))) for i in range(len(days) - 1)]
        rel = [float(np.linalg.norm(th[i + 1] - th[i]) / np.linalg.norm(th[i])) for i in range(len(days) - 1)]
        tops = []
        for i in range(len(days) - 1):
            dlt = th[i + 1] - th[i]
            idx = np.argsort(-np.abs(dlt))[:10]
            tops.append([{"feature": names[j], "delta": float(dlt[j]), "theta_before": float(th[i][j]), "theta_after": float(th[i + 1][j])} for j in idx])
        wdiag[mode] = {"params": w["params"], "reproduces_stored_run": True,
                       "cosine_consecutive_day_ends": dict(zip([f"day {days[i]+3}->{days[i+1]+3}" for i in range(len(days) - 1)], cos)),
                       "relative_change_norm": dict(zip([f"day {days[i]+3}->{days[i+1]+3}" for i in range(len(days) - 1)], rel)),
                       "theta_norm_by_day": dict(zip([f"day {d + 3}" for d in days], [float(np.linalg.norm(t)) for t in th])),
                       "theta_lgb_prob_by_day": dict(zip([f"day {d + 3}" for d in days], [float(t[d0]) for t in th])),
                       "top10_changes": dict(zip([f"day {days[i]+3}->{days[i+1]+3}" for i in range(len(days) - 1)], tops))}
    out["5a_weight_movement"] = wdiag

    # 5b -- is the ranker itself changing? plus the list-size composition check
    expected_random = np.array([(1.0 / held.sizes[hday == d]).mean() for d in range(hday.max() + 1)])
    med = float(np.median(held.sizes))
    b = {"per_day": {}, "median_candidate_set": med}
    for d in range(hday.max() + 1):
        m = hday == d
        e = {"lgbm_ctr": ci_row(lgb_r, 1.0, users, n_users, m), "expected_random_ctr": float(expected_random[d]),
             "lgbm_lift_over_expected_random": float(lgb_r[m].mean() / expected_random[d]),
             "mean_candidates": float(held.sizes[m].mean()), "share_large_sets": float((held.sizes[m] > med).mean())}
        for mode in MODES:
            h = ref_npz[f"hybrid_linucb|{mode}__reward"][0]
            e[f"hybrid_linucb_{mode}_ctr"] = float(h[m].mean())
            e[f"hybrid_linucb_{mode}_lift_over_expected_random"] = float(h[m].mean() / expected_random[d])
        for nm, sm in (("small", held.sizes <= med), ("large", held.sizes > med)):
            mm = m & sm
            e[f"lgbm_ctr_{nm}_sets"] = float(lgb_r[mm].mean())
            e[f"hybrid_linucb_per_impression_ctr_{nm}_sets"] = float(ref_npz["hybrid_linucb|per_impression__reward"][0][mm].mean())
            e[f"n_{nm}_sets"] = int(mm.sum())
        b["per_day"][f"day {d + 3}"] = e
    # unpaired day6-vs-day3 ranker change, user bootstrap on each side not paired: report difference of the two point values
    b["lgbm_day6_minus_day3"] = b["per_day"]["day 6"]["lgbm_ctr"]["value"] - b["per_day"]["day 3"]["lgbm_ctr"]["value"]
    b["hybrid_linucb_per_impression_day6_minus_day3"] = (b["per_day"]["day 6"]["hybrid_linucb_per_impression_ctr"]
                                                         - b["per_day"]["day 3"]["hybrid_linucb_per_impression_ctr"])
    out["5b_ranker_staleness"] = b

    # 5c -- article-age bucket of the CLICKED article (evaluation-side partition; policies never see it)
    zip_path = _REPO / "data" / "raw" / "ebnerd" / "ebnerd_small.zip"
    emb_npy, emb_json = _embeddings_paths("small")
    art = load_article_table(zip_path, embeddings_npy=emb_npy, embeddings_json=emb_json)
    pos = np.fromiter((art.id_to_pos.get(int(a), -1) for a in held.article_id), dtype=np.int64, count=len(held.article_id))
    pub = np.where(pos >= 0, art.published_at[np.maximum(pos, 0)], np.nan)
    age_row = (np.repeat(held.time, held.sizes) - pub) / 3600.0
    age_row = np.where(held.label > 0, age_row, np.inf)
    min_age = np.minimum.reduceat(age_row, held.offsets)
    unknown = ~np.isfinite(min_age)
    edges = [(0, 6, "<6h"), (6, 24, "6-24h"), (24, 72, "24-72h"), (72, np.inf, ">=72h")]
    buckets = {nm: (min_age >= lo) & (min_age < hi) & ~unknown for lo, hi, nm in edges}
    buckets["unknown"] = unknown
    fw = final_window_mask(len(held), 0.2)
    fam_rew = {k: ref_npz[f"{k}__reward"].mean(axis=0) for k in REF_KEYS}
    age = {"share_of_final20": {k: float((m & fw).sum() / fw.sum()) for k, m in buckets.items()},
           "share_of_heldout": {k: float(m.mean()) for k, m in buckets.items()}, "rows": {}}
    for scope, sm in (("final 20%", fw), ("held-out overall", np.ones(len(held), bool))):
        for k, m in buckets.items():
            mm = m & sm
            if mm.sum() == 0:
                continue
            row = {"n": int(mm.sum()), "lgbm_ctr": ci_row(lgb_r, 1.0, users, n_users, mm)}
            for rk in ("hybrid_linucb|per_impression", "hybrid_linucb|hourly"):
                row[rk] = {"ctr": ci_row(fam_rew[rk], 1.0, users, n_users, mm),
                           "delta_vs_lgbm": ci_row(fam_rew[rk] - lgb_r, 1.0, users, n_users, mm)}
            age["rows"][f"{scope} | {k}"] = row
    out["5c_age_bucket_slices"] = age

    # full-information transfer matrix (offline diagnostic): ridge fit on the first half of day i, tested on the second half of day j
    day_all = ((stream.time - stream.time[0]) // 86400).astype(int)
    halves = {}
    for d in range(day_all.max() + 1):
        idx = np.flatnonzero(day_all == d)
        halves[(d, 0)], halves[(d, 1)] = idx[: len(idx) // 2], idx[len(idx) // 2:]
    lam = 100.0
    thetas, sub = {}, {}
    for key, idx in halves.items():
        sub[key] = stream.window(int(idx[0]), int(idx[-1]) + 1)
    for d in range(day_all.max() + 1):
        s = sub[(d, 0)]
        X = np.asarray(s.features, dtype=np.float64)
        thetas[d] = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ s.label.astype(np.float64))
    n_days = day_all.max() + 1
    M = np.zeros((n_days, n_days)); L = np.zeros(n_days)
    for j in range(n_days):
        t = sub[(j, 1)]
        Xt = np.asarray(t.features, dtype=np.float64)
        L[j] = topk_hit(t.lgb_score.astype(np.float64), t).mean()
        for i in range(n_days):
            M[i, j] = topk_hit(Xt @ thetas[i], t).mean() - L[j]
    cosm = np.array([[thetas[i] @ thetas[j] / (np.linalg.norm(thetas[i]) * np.linalg.norm(thetas[j])) for j in range(n_days)] for i in range(n_days)])
    out["full_information_transfer"] = {
        "description": "rows = ridge fitted on the FIRST half of stream day i with all labels (offline diagnostic, not a policy); columns = tested on the SECOND half of day j; entry = top-1 precision minus LightGBM-greedy's on the same impressions",
        "lgbm_ctr_second_half_by_day": L.tolist(), "delta_matrix": M.tolist(), "theta_cosine_matrix": cosm.tolist(),
        "ridge_lambda": lam, "days": list(range(n_days))}
    return out


# ------------------------------------------------------------------ groups
def group_specs():
    specs = {"full (no ablation)": {}}
    for g in GROUP_NAMES:
        specs[f"drop {g}"] = {"drop_groups": [g]}
    specs["keep only lightgbm_score"] = {"keep_only": ["lightgbm_score"]}
    for g in FIVE_NON_LGB:
        specs[f"keep lightgbm_score + {g}"] = {"keep_only": ["lightgbm_score", g]}
    return specs


def stage_groups(tune, held, workers, base, extra=False):
    specs = group_specs()
    if extra:  # follow-up after the main ablation: do the two article-descriptor groups together reproduce the gain?
        specs = {"full (no ablation)": {}, "keep lightgbm_score + onehot_categoricals + article_metadata":
                 {"keep_only": ["lightgbm_score", "onehot_categoricals", "article_metadata"]}}
    tasks, meta = [], {}
    for sname, sp in specs.items():
        if base == "plain":  # non-forgetting hybrid LinUCB
            combos = [({"alpha": a, "lam": l, "w": w}, m) for m, a, l, w in itertools.product(MODES, [0.0, 0.01], [10, 100, 1000], [0.0, 1.0])]
        else:  # exponential-forgetting hybrid LinUCB (alpha fixed at the ADR-017 winner's 0.003: alpha is flat)
            combos = [({"alpha": 0.003, "lam": l, "w": w, "gamma": g}, m)
                      for m, g, l, w in itertools.product(MODES, [0.998, 0.999, 0.9995], [1, 10, 100], [0.0, 1.0])]
        for base_params, mode in combos:
            tasks.append(("hybrid_linucb", {**base_params, **sp, "_spec": sname}, mode, rbe.TUNE_SEED, False))
    print(f"[groups] tuning {len(tasks)} runs", flush=True)
    rows = []
    for task, arm, reward, dt in rbe._pool_map(tasks, tune, workers):
        _, params, mode, _, _ = task
        rows.append({"spec": params["_spec"], "mode": mode, "params": {k: v for k, v in params.items() if k != "_spec"},
                     "tune_ctr": float(reward.mean())})
    winners = best_per_group(rows, lambda r: f"{r['spec']}|{r['mode']}")
    plan = [(k, "hybrid_linucb", w["params"], w["mode"]) for k, w in winners.items()]
    print(f"[groups] held-out: {len(plan)} frozen runs", flush=True)
    runs = heldout_runs(held, plan, workers)
    ref_npz = np.load(rbe.OUT_DIR / "heldout_runs.npz")
    lgb = ref_npz["lgbm_greedy|per_impression__reward"][0]
    users, n_users = factorize_users(held.user_id)
    wins = {k: v for k, v in windows_for(held).items() if k in ("overall", "final 20%")}
    names = rbe._feature_names()
    sizes = {g: len(i) for g, i in assign_groups(names).items()}
    rep = {"group_sizes": sizes, "rows": {}}
    for key, (arm, rew2) in runs.items():
        rew = rew2.mean(axis=0)
        sname, mode = key.rsplit("|", 1)
        full = runs[f"full (no ablation)|{mode}"][1].mean(axis=0)
        e = {"spec": sname, "mode": mode, "params": winners[key]["params"], "tune_ctr": winners[key]["tune_ctr"], "windows": {}}
        for wn, m in wins.items():
            e["windows"][wn] = {"ctr": ci_row(rew, 1.0, users, n_users, m), "delta_vs_lgbm": ci_row(rew - lgb, 1.0, users, n_users, m),
                                "delta_vs_full_hybrid": ci_row(rew - full, 1.0, users, n_users, m)}
        rep["rows"][key] = e
    return {"tune_grid": rows, "winners": winners}, rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tune", "final", "diag", "groups"], required=True)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--extra", action="store_true", help="groups stage: only the follow-up spec (+ full reference)")
    ap.add_argument("--base", choices=["plain", "forget"], default="plain", help="groups stage: which hybrid to ablate")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stream = rbe.load_stream()
    tune, held, _ = rbe.split_periods(stream)
    print(f"tune {len(tune):,} / held-out {len(held):,} impressions", flush=True)
    if args.stage == "tune":
        (OUT / "tune_forget.json").write_text(json.dumps(stage_tune(tune, args.workers), indent=1))
    elif args.stage == "final":
        tune_out = json.loads((OUT / "tune_forget.json").read_text())
        report, runs = stage_final(held, tune_out, args.workers)
        (OUT / "forget_heldout.json").write_text(json.dumps(report, indent=1))
        np.savez_compressed(OUT / "forget_heldout_runs.npz", **{f"{k}__reward": v[1] for k, v in runs.items()})
    elif args.stage == "diag":
        (OUT / "diagnosis.json").write_text(json.dumps(stage_diag(stream, held, args.workers), indent=1))
    elif args.stage == "groups":
        tuned, rep = stage_groups(tune, held, args.workers, args.base, args.extra)
        (OUT / f"groups_{args.base}{'_extra' if args.extra else ''}_tune.json").write_text(json.dumps(tuned, indent=1))
        (OUT / f"groups_{args.base}{'_extra' if args.extra else ''}_heldout.json").write_text(json.dumps(rep, indent=1))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
