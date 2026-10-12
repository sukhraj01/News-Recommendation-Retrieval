#!/usr/bin/env python3
"""Offline bandit experiment on the ebnerd_small validation stream (ADR-017).

Stages
  tune   grid search on the TUNING period (stream days 0-2); one run per config.
  final  freeze the per-(family, update-mode) winners; run ONCE on the held-out period
         (days 3-6) from a fresh state; seeds 0-4 for stochastic policies; full metrics.
  sim    same frozen policies against the LightGBM-derived click simulator (SIMULATED).
Run `--dry-run` first: it shrinks every stage to a few thousand impressions.

Reads data/processed/ebnerd/small/bandit_stream/ (see bandit_prepare_stream.py).
Writes results/bandit_ebnerd_small/.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker; parallelism is across runs

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import numpy as np

from src.bandit.evaluation import (
    N_BOOTSTRAP, factorize_users, final_window_mask, ratio_ci, simulated_pseudo_regret, slice_masks,
)
from src.bandit.policies import (
    EpsilonGreedy, LightGBMGreedy, LinThompson, LinUCB, PopularityGreedy, RandomPolicy, hybrid_prior_theta,
)
from src.bandit.runner import run_prequential
from src.bandit.stream import ImpressionStream

STREAM_DIR = _REPO / "data" / "processed" / "ebnerd" / "small" / "bandit_stream"
OUT_DIR = _REPO / "results" / "bandit_ebnerd_small"
TUNE_DAYS = 3
HOUR = 3600.0
COLD_ARTICLE_N, COLD_USER_N, LOW_HIST_N = 100, 5, 20
TUNE_SEED, FINAL_SEEDS = 1000, [0, 1, 2, 3, 4]
MODES = {"per_impression": None, "hourly": HOUR}

# Grid widened once after the first pass put several winners on its upper edge
# (lam=100, v=0.1, m=20, eps=0.01). Widening used the TUNING period only; the first-pass
# winners are kept in results/.../tune_first_pass.json for the audit trail.
GRID = {
    "popularity_greedy": dict(window_h=[6, 24], m=[1, 5, 20, 100]),
    "eps_greedy_lgbm": dict(epsilon=[0.003, 0.01, 0.03, 0.1, 0.2]),
    "linucb": dict(alpha=[0.0, 0.003, 0.01, 0.03, 0.1], lam=[1, 10, 100, 1000]),
    "lints": dict(v=[0.003, 0.01, 0.03, 0.1, 0.3], lam=[1, 10, 100, 1000]),
    "hybrid_linucb": dict(alpha=[0.0, 0.003, 0.01, 0.03, 0.1], lam=[1, 10, 100, 1000], w=[0.0, 1.0]),
    "hybrid_lints": dict(v=[0.003, 0.01, 0.03, 0.1, 0.3], lam=[1, 10, 100, 1000], w=[0.0, 1.0]),
}
STOCHASTIC = {"random", "popularity_greedy", "eps_greedy_lgbm", "lints", "hybrid_lints"}
STATIC = {"random", "lgbm_greedy", "eps_greedy_lgbm"}  # no learning -> update mode is irrelevant


def load_stream(limit: int | None = None) -> ImpressionStream:
    meta = json.loads((STREAM_DIR / "meta.json").read_text())
    imp = np.load(STREAM_DIR / "impressions.npz")
    n = len(imp["sizes"])
    full = ImpressionStream(
        features=np.load(STREAM_DIR / "features.npy", mmap_mode="r"),
        lgb_score=np.load(STREAM_DIR / "lgb_score.npy"), label=np.load(STREAM_DIR / "label.npy"),
        sizes=imp["sizes"], time=imp["time"], user_id=imp["user_id"],
        article_id=np.load(STREAM_DIR / "article_id.npy"), history_len=imp["history_len"],
        base_dim=meta["base_dim"], sim_prob=np.load(STREAM_DIR / "sim_prob.npy"),
        prior_exposure=np.load(STREAM_DIR / "prior_exposure.npy"),
    )
    return full if limit is None else full.window(0, min(limit, n))


def split_periods(stream: ImpressionStream) -> tuple[ImpressionStream, ImpressionStream, int]:
    n_tune = int(np.searchsorted(stream.time, stream.time[0] + TUNE_DAYS * 86400, side="left"))
    return stream.window(0, n_tune), stream.window(n_tune, len(stream)), n_tune


def make_policy(kind: str, p: dict, d0: int, seed: int):
    if kind == "random":
        return RandomPolicy(seed)
    if kind == "lgbm_greedy":
        return LightGBMGreedy()
    if kind == "eps_greedy_lgbm":
        return EpsilonGreedy(p["epsilon"], seed)
    if kind == "popularity_greedy":
        return PopularityGreedy(p["window_h"] * 3600.0, p["m"], seed)
    if kind == "linucb":
        return LinUCB(d0, p["alpha"], p["lam"])
    if kind == "lints":
        return LinThompson(d0, p["v"], p["lam"], seed=seed)
    if kind in ("hybrid_linucb", "hybrid_lints"):
        prior = hybrid_prior_theta(d0 + 2, d0, p["w"])
        if kind == "hybrid_linucb":
            return LinUCB(d0 + 2, p["alpha"], p["lam"], prior, name=kind)
        return LinThompson(d0 + 2, p["v"], p["lam"], prior, seed=seed, name=kind)
    raise KeyError(kind)


# ---- worker plumbing: the stream lives in a module global inherited by fork ----
_STREAM: ImpressionStream | None = None
_D0 = 0


def _run_task(task):
    kind, params, mode, seed, simulated = task
    policy = make_policy(kind, params, _D0, seed)
    t0 = time.time()
    res = run_prequential(policy, _STREAM, update_interval_s=MODES[mode], simulated=simulated, env_seed=seed)
    return task, res.arm, res.reward, time.time() - t0


def _pool_map(tasks, stream, workers):
    global _STREAM, _D0
    _STREAM, _D0 = stream, stream.base_dim
    if workers <= 1:
        return [_run_task(t) for t in tasks]
    with mp.get_context("fork").Pool(workers) as pool:
        return list(pool.imap_unordered(_run_task, tasks, chunksize=1))


def configs(kind: str):
    keys = list(GRID[kind])
    for vals in itertools.product(*GRID[kind].values()):
        yield dict(zip(keys, vals))


def stage_tune(tune: ImpressionStream, workers: int, dry: bool) -> dict:
    tasks = []
    for kind in GRID:
        cfgs = list(configs(kind))
        for p in (cfgs[:: max(1, len(cfgs) // 3)] if dry else cfgs):  # dry run: ~3 spread-out configs per family
            for mode in (["per_impression"] if kind in STATIC else MODES):
                tasks.append((kind, p, mode, TUNE_SEED, False))
    print(f"[tune] {len(tasks)} runs on {len(tune):,} impressions, {workers} workers", flush=True)
    t0 = time.time()
    rows = []
    for task, arm, reward, dt in _pool_map(tasks, tune, workers):
        kind, p, mode, seed, _ = task
        rows.append({"kind": kind, "params": p, "mode": mode, "tune_ctr": float(reward.mean()), "run_s": dt})
    rows.sort(key=lambda r: (r["kind"], r["mode"], -r["tune_ctr"]))
    print(f"[tune] done in {time.time()-t0:.0f}s", flush=True)
    # winners: best tuning CTR per (kind, mode)
    winners: dict[str, dict] = {}
    for r in rows:
        key = f"{r['kind']}|{r['mode']}"
        if key not in winners:
            winners[key] = r
    # LightGBM-greedy / random reference on the tuning period (no hyper-parameters)
    refs = {}
    for kind in ("random", "lgbm_greedy"):
        _, _, rw, _ = _pool_map([(kind, {}, "per_impression", TUNE_SEED, False)], tune, 1)[0]
        refs[kind] = float(rw.mean())
    return {"grid": rows, "winners": winners, "reference_tune_ctr": refs, "n_tune_impressions": len(tune)}


def stage_final(held: ImpressionStream, tune_out: dict, workers: int):
    tasks = []
    plan: list[tuple[str, dict, str]] = [("random", {}, "per_impression"), ("lgbm_greedy", {}, "per_impression")]
    for key, w in tune_out["winners"].items():
        plan.append((w["kind"], w["params"], w["mode"]))
    for kind, params, mode in plan:
        seeds = FINAL_SEEDS if kind in STOCHASTIC else [0]
        for s in seeds:
            tasks.append((kind, params, mode, s, False))
    print(f"[final] {len(tasks)} runs on {len(held):,} held-out impressions", flush=True)
    out = _pool_map(tasks, held, workers)
    runs: dict[str, dict] = {}
    spec_of = {f"{k}|{m}": (k, p, m) for k, p, m in plan}
    for task, arm, reward, dt in out:
        kind, params, mode, seed, _ = task
        key = f"{kind}|{mode}"
        runs.setdefault(key, {"kind": kind, "params": params, "mode": mode, "seeds": [], "arm": [], "reward": [], "run_s": []})
        runs[key]["seeds"].append(seed); runs[key]["arm"].append(arm); runs[key]["reward"].append(reward)
        runs[key]["run_s"].append(dt)
    for r in runs.values():
        order = np.argsort(r["seeds"])
        r["seeds"] = [r["seeds"][i] for i in order]
        r["arm"] = np.stack([r["arm"][i] for i in order]); r["reward"] = np.stack([r["reward"][i] for i in order])
    return runs


def _fmt(ci):
    return {"value": ci[0], "ci_low": ci[1], "ci_high": ci[2], "n": ci[3]}


def analyse(held: ImpressionStream, runs: dict) -> dict:
    users, n_users = factorize_users(held.user_id)
    n = len(held)
    oracle = held.oracle_reward()
    lgb_top1 = held.lgb_top1()
    base_key = "lgbm_greedy|per_impression"
    base_r = runs[base_key]["reward"][0]
    fw = final_window_mask(n, 0.2)
    masks = slice_masks(held, cold_article_n=COLD_ARTICLE_N, cold_user_n=COLD_USER_N, low_hist_n=LOW_HIST_N)
    masks["ALL"] = np.ones(n, dtype=bool)
    day = ((held.time - held.time[0]) // 86400).astype(int)
    res: dict = {"n_held_out_impressions": n, "n_users": n_users, "oracle_ctr": float(oracle.mean()),
                 "slice_sizes": {k: int(m.sum()) for k, m in masks.items()}, "policies": {}}
    for key, r in runs.items():
        rew = r["reward"].mean(axis=0)            # seed-averaged per-impression reward
        per_seed_ctr = r["reward"].mean(axis=1)
        entry: dict = {"kind": r["kind"], "mode": r["mode"], "params": r["params"], "n_seeds": len(r["seeds"]),
                       "ctr": _fmt(ratio_ci(rew, 1.0, users, n_users)),
                       "ctr_seed_std": float(per_seed_ctr.std(ddof=1)) if len(per_seed_ctr) > 1 else 0.0,
                       "mean_regret_per_impression": _fmt(ratio_ci(oracle - rew, 1.0, users, n_users)),
                       "cumulative_regret_total": float((oracle - rew).sum()),
                       "final_window_ctr": _fmt(ratio_ci(rew, 1.0, users, n_users, mask=fw)),
                       "daily_ctr": [float(rew[day == d].mean()) for d in range(day.max() + 1)],
                       "run_s_mean": float(np.mean(r["run_s"]))}
        if key != base_key:
            entry["paired_vs_lgbm_greedy"] = _fmt(ratio_ci(rew - base_r, 1.0, users, n_users))
            entry["paired_vs_lgbm_greedy_final_window"] = _fmt(ratio_ci(rew - base_r, 1.0, users, n_users, mask=fw))
        # slices: CTR and paired diff vs LightGBM-greedy
        sl = {}
        for name, m in masks.items():
            if m.sum() == 0:
                sl[name] = {"n": 0}
                continue
            e = {"ctr": _fmt(ratio_ci(rew, 1.0, users, n_users, mask=m))}
            if key != base_key:
                e["paired_vs_lgbm_greedy"] = _fmt(ratio_ci(rew - base_r, 1.0, users, n_users, mask=m))
            sl[name] = e
        entry["slices"] = sl
        # exploration cost: where the pick differs from the LightGBM top-1
        differs = (r["arm"] != lgb_top1[None, :]).astype(np.float64).mean(axis=0)  # seed-averaged indicator
        d_rew = (r["reward"] * (r["arm"] != lgb_top1[None, :])).mean(axis=0)
        d_lgb = base_r * differs
        entry["exploration"] = {
            "pick_differs_from_lgbm_top1_rate": float(differs.mean()),
            "ctr_of_differing_picks": _fmt(ratio_ci(d_rew, differs, users, n_users)),
            "ctr_of_lgbm_pick_on_same_impressions": _fmt(ratio_ci(d_lgb, differs, users, n_users)),
            "paired_diff_on_differing_impressions": _fmt(ratio_ci(d_rew - d_lgb, differs, users, n_users)),
        }
        res["policies"][key] = entry
    return res


def stage_sim(held: ImpressionStream, tune_out: dict, workers: int) -> dict:
    users_n = len(held)
    plan = [("random", {}, "per_impression"), ("lgbm_greedy", {}, "per_impression")]
    plan += [(w["kind"], w["params"], w["mode"]) for w in tune_out["winners"].values()]
    tasks = [(k, p, m, s, True) for k, p, m in plan for s in FINAL_SEEDS]
    print(f"[sim] {len(tasks)} simulated runs", flush=True)
    out = _pool_map(tasks, held, workers)
    agg: dict[str, dict] = {}
    for task, arm, reward, dt in out:
        kind, params, mode, seed, _ = task
        key = f"{kind}|{mode}"
        a = agg.setdefault(key, {"pseudo": [], "ctr": []})
        a["pseudo"].append(float(simulated_pseudo_regret(held, arm).sum()))
        a["ctr"].append(float(reward.mean()))
    best = np.maximum.reduceat(held.sim_prob.astype(np.float64), held.offsets)
    return {"SIMULATED": True, "n_impressions": users_n, "n_seeds": len(FINAL_SEEDS),
            "simulator_oracle_expected_ctr": float(best.mean()),
            "policies": {k: {"cum_pseudo_regret_mean": float(np.mean(v["pseudo"])),
                             "cum_pseudo_regret_std": float(np.std(v["pseudo"], ddof=1)),
                             "sim_ctr_mean": float(np.mean(v["ctr"])), "sim_ctr_std": float(np.std(v["ctr"], ddof=1))}
                         for k, v in agg.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tune", "final", "sim", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    out_dir = OUT_DIR / ("dry_run" if args.dry_run else "")
    out_dir.mkdir(parents=True, exist_ok=True)

    stream = load_stream(limit=8000 if args.dry_run else None)
    if args.dry_run:  # tune/held-out split by impression count in the dry run
        n_tune = len(stream) // 2
        tune, held = stream.window(0, n_tune), stream.window(n_tune, len(stream))
    else:
        tune, held, n_tune = split_periods(stream)
    print(f"stream {len(stream):,} imps; tune {len(tune):,}; held-out {len(held):,}; d0={stream.base_dim}", flush=True)

    tune_path = out_dir / "tune.json"
    if args.stage in ("tune", "all"):
        tune_out = stage_tune(tune, args.workers, args.dry_run)
        tune_path.write_text(json.dumps(tune_out, indent=1))
    else:
        tune_out = json.loads(tune_path.read_text())
    if args.stage in ("final", "all"):
        runs = stage_final(held, tune_out, args.workers)
        analysis = analyse(held, runs)
        analysis["config"] = {"tune_days": TUNE_DAYS, "n_bootstrap": N_BOOTSTRAP, "resample_unit": "user",
                              "cold_article_n": COLD_ARTICLE_N, "low_history_n": LOW_HIST_N,
                              "final_seeds": FINAL_SEEDS, "tune_seed": TUNE_SEED}
        (out_dir / "heldout.json").write_text(json.dumps(analysis, indent=1))
        np.savez_compressed(out_dir / "heldout_runs.npz", **{
            f"{k}__{name}": r[name] for k, r in runs.items() for name in ("arm", "reward")})
    if args.stage in ("sim", "all"):
        sim = stage_sim(held, tune_out, args.workers)
        (out_dir / "simulated.json").write_text(json.dumps(sim, indent=1))
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
