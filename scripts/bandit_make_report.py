#!/usr/bin/env python3
"""Render results/bandit_ebnerd_small/*.json into markdown tables for ADR-017 (no new numbers)."""
from __future__ import annotations
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results" / "bandit_ebnerd_small"
H = json.loads((R / "heldout.json").read_text())
T = json.loads((R / "tune.json").read_text())
S = json.loads((R / "simulated.json").read_text())
P = H["policies"]
ORDER = ["random|per_impression", "popularity_greedy|per_impression", "popularity_greedy|hourly",
         "lgbm_greedy|per_impression", "eps_greedy_lgbm|per_impression",
         "linucb|per_impression", "linucb|hourly", "lints|per_impression", "lints|hourly",
         "hybrid_linucb|per_impression", "hybrid_linucb|hourly", "hybrid_lints|per_impression", "hybrid_lints|hourly"]
LABEL = lambda k: k.replace("|per_impression", " (per-impression)").replace("|hourly", " (hourly)") if "lgbm_greedy" not in k and "random" not in k and "eps" not in k else k.split("|")[0]
ci = lambda d, f="{:.4f}": f"{f.format(d['value'])} [{f.format(d['ci_low'])}, {f.format(d['ci_high'])}]"
dci = lambda d: f"{d['value']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"
sig = lambda d: "win" if d["ci_low"] > 0 else ("LOSS" if d["ci_high"] < 0 else "n.s.")
out = []

out.append("### Tuned hyper-parameters (selected on the tuning period; frozen)\n")
out.append("| policy (update mode) | frozen hyper-parameters | tuning-period CTR |\n|---|---|---:|")
for k in ORDER:
    if k in T["winners"]:
        w = T["winners"][k]; out.append(f"| {LABEL(k)} | `{json.dumps(w['params'])}` | {w['tune_ctr']:.4f} |")
out.append(f"\nReferences on the tuning period: random {T['reference_tune_ctr']['random']:.4f}, LightGBM-greedy {T['reference_tune_ctr']['lgbm_greedy']:.4f}.\n")

out.append("### Held-out primary results (days 3-6; %d impressions, %d users; user-bootstrap 95%% CI, n=2000)\n" % (H["n_held_out_impressions"], H["n_users"]))
out.append("| policy | top-1 CTR | mean regret / imp | cumulative regret | final-window (last 20%) CTR | paired Δ CTR vs LightGBM-greedy | verdict |\n|---|---|---|---:|---|---|---|")
for k in ORDER:
    v = P[k]; pd_ = v.get("paired_vs_lgbm_greedy")
    out.append(f"| {LABEL(k)} | {ci(v['ctr'])} | {v['mean_regret_per_impression']['value']:.4f} | {v['cumulative_regret_total']:,.0f} | {ci(v['final_window_ctr'])} | "
               + (f"{dci(pd_)} | {sig(pd_)} |" if pd_ else "- | reference |"))
out.append("\nSeed-to-seed std of CTR (5 seeds, stochastic policies): " + ", ".join(
    f"{LABEL(k)} {P[k]['ctr_seed_std']:.4f}" for k in ORDER if P[k]['n_seeds'] > 1) + ".\n")

out.append("### Learning check: CTR by held-out day, and paired Δ vs LightGBM-greedy in the last 20%\n")
out.append("| policy | day 3 | day 4 | day 5 | day 6 | Δ vs LightGBM, final window |\n|---|---:|---:|---:|---:|---|")
for k in ORDER:
    v = P[k]; d = v["daily_ctr"]; fw = v.get("paired_vs_lgbm_greedy_final_window")
    out.append(f"| {LABEL(k)} | " + " | ".join(f"{x:.4f}" for x in d) + f" | {dci(fw) if fw else '-'} |")

out.append("\n### Slices (CTR [CI]; Δ vs LightGBM-greedy [CI]) for the policies that matter\n")
KEYS = ["lgbm_greedy|per_impression", "linucb|per_impression", "hybrid_linucb|per_impression", "hybrid_linucb|hourly", "hybrid_lints|hourly"]
names = [n for n in P[KEYS[0]]["slices"] if n != "ALL"]
out.append("| slice (n impressions) | " + " | ".join(LABEL(k) for k in KEYS) + " |\n|---|" + "---|" * len(KEYS))
for n in names:
    size = H["slice_sizes"][n]
    cells = []
    for k in KEYS:
        s = P[k]["slices"][n]
        if "ctr" not in s:
            cells.append("empty"); continue
        c = f"{s['ctr']['value']:.4f} [{s['ctr']['ci_low']:.4f}, {s['ctr']['ci_high']:.4f}]"
        if "paired_vs_lgbm_greedy" in s:
            d = s["paired_vs_lgbm_greedy"]; c += f"<br>Δ {d['value']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] {sig(d)}"
        cells.append(c)
    out.append(f"| {n} (n={size:,}) | " + " | ".join(cells) + " |")

out.append("\n### Exploration cost: where the pick differs from the LightGBM top-1\n")
out.append("| policy | picks differing from LightGBM top-1 | CTR of the differing picks | CTR of LightGBM's pick, same impressions | paired difference | verdict |\n|---|---:|---|---|---|---|")
for k in ORDER:
    if k.startswith("lgbm_greedy"): continue
    e = P[k]["exploration"]; d = e["paired_diff_on_differing_impressions"]
    out.append(f"| {LABEL(k)} | {e['pick_differs_from_lgbm_top1_rate']*100:.1f}% | {ci(e['ctr_of_differing_picks'])} | {ci(e['ctr_of_lgbm_pick_on_same_impressions'])} | {dci(d)} | {sig(d)} |")

out.append("\n### SIMULATED click model (K_cls probabilities normalised per impression; Bernoulli rewards; 5 seeds; held-out stream) -- NOT log-replay\n")
out.append(f"Simulator oracle expected CTR (always pick the max-probability arm): {S['simulator_oracle_expected_ctr']:.4f}.\n")
out.append("| policy | cumulative expected regret (mean ± std) | simulated CTR (mean ± std) |\n|---|---:|---:|")
for k in ORDER:
    if k in S["policies"]:
        s = S["policies"][k]; out.append(f"| {LABEL(k)} | {s['cum_pseudo_regret_mean']:,.0f} ± {s['cum_pseudo_regret_std']:,.0f} | {s['sim_ctr_mean']:.4f} ± {s['sim_ctr_std']:.4f} |")
(R / "tables.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
