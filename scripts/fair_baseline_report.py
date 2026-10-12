#!/usr/bin/env python3
"""Render drift/fair_baseline*.json + trailing_gbdt_training.json into markdown (no new numbers)."""
import json
from pathlib import Path
D = Path(__file__).resolve().parents[1] / "results" / "bandit_ebnerd_small" / "drift"
J = lambda n: json.loads((D / n).read_text())
R, T, A = J("fair_baseline.json"), J("trailing_gbdt_training.json"), J("fair_baseline_audit.json")
ci = lambda d: f"{d['value']:.4f} [{d['ci_low']:.4f}, {d['ci_high']:.4f}]"
dci = lambda d: f"{d['value']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"
verdict = lambda d: "beats" if d["ci_low"] > 0 else ("loses to" if d["ci_high"] < 0 else "matches (n.s.)")
NAME = {"original": "LightGBM-greedy (original)", "gbdt_plus_trailing_ctr": "LightGBM + trailing CTR (retrained)",
        "popularity_blend": "popularity blend (no learning)", "forgetting_hybrid": "forgetting hybrid LinUCB"}
out = ["### Retrained GBDT: training summary (all validation days; the models never saw validation)\n",
       "| D | best iteration | stream AUC (all 7 days) | top-1 (all 7 days) | share of gain from the 6 trailing features |\n|---|---:|---:|---:|---:|"]
for d, v in T["per_delay"].items():
    out.append(f"| {int(d)//60} min | {v['best_iteration']} | {v['stream_auc_all_validation']:.4f} | {v['top1_all_validation']:.4f} | {v['trailing_gain_share']:.3f} |")
out.append("\nOriginal ranker on the same stream: AUC 0.7588, top-1 0.3168.\n")
for d, e in R["per_delay"].items():
    out.append(f"### D = {int(d)//60} min — held-out days 3-6 (n = {R['n_held_out']:,}); selections made on days 0-2\n")
    out.append(f"Blend chosen on days 0-2: {e['blend_selected_on_tuning']['window']} window, beta = {e['blend_selected_on_tuning']['beta']}. "
               f"Forgetting hybrid re-tuned under this delay: gamma = {e['forgetting_hybrid_selected_on_tuning']['gamma']}, lambda = {e['forgetting_hybrid_selected_on_tuning']['lam']}.\n")
    out.append("| policy | P@1 [CI] | Δ vs original [CI] | Δ vs forgetting hybrid [CI] | final-20% P@1 | final-20% Δ vs original | final-20% Δ vs hybrid |\n|---|---|---|---|---|---|---|")
    for k, nm in NAME.items():
        p = e["policies"][k]; o, f = p["overall"], p["final 20%"]
        g = lambda w, key: dci(w[key]) if key in w else "-"
        out.append(f"| {nm} | {ci(o['p_at_1'])} | {g(o,'delta_vs_original')} | {g(o,'delta_vs_forgetting_hybrid')} | {f['p_at_1']['value']:.4f} | {g(f,'delta_vs_original')} | {g(f,'delta_vs_forgetting_hybrid')} |")
    a = e["auc_days_3_6"]
    out.append(f"\nStream AUC on days 3-6: original {ci(a['original'])}, retrained {ci(a['gbdt_plus_trailing_ctr'])}, paired Δ {dci(a['paired_delta'])}. "
               f"Per-day P@1 (days 3-6): " + "; ".join(f"{NAME[k].split(' (')[0]} " + "/".join(f"{x:.3f}" for x in v) for k, v in e["per_day_p_at_1"].items()) + ".\n")
out.append("### Verdict per D (retrained GBDT vs forgetting hybrid, paired, overall / final 20%)\n\n| D | overall | final 20% |\n|---|---|---|")
for d, e in R["per_delay"].items():
    p = e["policies"]["gbdt_plus_trailing_ctr"]
    o, f = p["overall"]["delta_vs_forgetting_hybrid"], p["final 20%"]["delta_vs_forgetting_hybrid"]
    out.append(f"| {int(d)//60} min | **{verdict(o)}** the hybrid ({dci(o)}) | **{verdict(f)}** the hybrid ({dci(f)}) |")
out.append("\n### Causality audit of the trained models (end to end)\n\n| D | rows with t ≤ T0+D checked | max \\|score change\\| on them | later rows whose score changed | placebo top-1 (permuted-label features) | real top-1 | original ranker top-1 |\n|---|---:|---:|---:|---:|---:|---:|")
for d, v in A.items():
    out.append(f"| {d} | {v['rows_checked_unaffected']:,} | {v['max_abs_score_diff_unaffected']:.1e} | {v['rows_after_cutoff_that_changed']:.3f} | {v['placebo_top1_all_validation']:.4f} | {v['real_top1_all_validation']:.4f} | {v['original_ranker_top1_all_validation']:.4f} |")
(D / "fair_baseline_tables.md").write_text("\n".join(out) + "\n"); print("\n".join(out))
