#!/usr/bin/env python3
"""Render results/bandit_ebnerd_small/drift/*.json into markdown tables (no new numbers)."""
from __future__ import annotations
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "results" / "bandit_ebnerd_small" / "drift"
J = lambda n: json.loads((D / n).read_text())
ci = lambda d, f="{:.4f}": f"{f.format(d['value'])} [{f.format(d['ci_low'])}, {f.format(d['ci_high'])}]"
dci = lambda d: f"{d['value']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"
tag = lambda d: "win" if d["ci_low"] > 0 else ("LOSS" if d["ci_high"] < 0 else "n.s.")
out = []

# --- tuning ---
T, T1 = J("tune_forget.json"), J("tune_forget_first_pass.json")
out.append("### Tuning-period winners (days 0-2; widened grid, first-pass winner in brackets)\n")
out.append("| policy | variant | frozen parameters | tuning CTR | first pass |\n|---|---|---|---:|---|")
for k, w in sorted(T["winners"].items(), key=lambda kv: (kv[1]["mode"], kv[1]["kind"], kv[1]["variant"])):
    p = {a: b for a, b in w["params"].items() if a in ("gamma", "window", "reset_every", "lam", "alpha", "v", "w")}
    f1 = T1["winners"][k]; p1 = {a: b for a, b in f1["params"].items() if a in ("gamma", "window", "reset_every", "lam")}
    out.append(f"| {w['kind']} ({w['mode']}) | {w['variant']} | `{json.dumps(p)}` | {w['tune_ctr']:.4f} | {f1['tune_ctr']:.4f} `{json.dumps(p1)}` |")
refs = T["non_forgetting_reference_tune_ctr"]
out.append("\nNon-forgetting references (tuning CTR): " + ", ".join(f"{k} {v:.4f}" for k, v in refs.items()) + ".\n")

# --- held-out ---
H = J("forget_heldout.json")
wins = ["overall", "day 3", "day 4", "day 5", "day 6", "final 20%"]
out.append("### Held-out (days 3-6): precision@1 [95% user-bootstrap CI]\n")
out.append("Window sizes (impressions): " + ", ".join(f"{k} {v:,}" for k, v in H["windows"].items()) + ".\n")
out.append("| policy | " + " | ".join(wins) + " |\n|---|" + "---|" * len(wins))
lg = H["policies"]["hybrid_linucb|per_impression"]["windows"]
out.append("| LightGBM-greedy | " + " | ".join(f"{lg[w]['lgbm_ctr']:.4f}" for w in wins) + " |")
order = [k for k in H["policies"] if k.count("|") == 1] + [k for k in H["policies"] if k.count("|") == 2]
nice = lambda k: k.replace("|per_impression", " per-imp.").replace("|hourly", " hourly").replace("|", " ")
for k in order:
    out.append(f"| {nice(k)} | " + " | ".join(ci(H["policies"][k]["windows"][w]["ctr"]) for w in wins) + " |")
out.append("\n### Held-out: paired Δ vs LightGBM-greedy [CI]\n")
out.append("| policy | " + " | ".join(wins) + " |\n|---|" + "---|" * len(wins))
for k in order:
    out.append(f"| {nice(k)} | " + " | ".join(dci(H["policies"][k]["windows"][w]["delta_vs_lgbm"]) for w in wins) + " |")
out.append("\n### Held-out: paired Δ vs the non-forgetting hybrid of the same family and update mode [CI]\n")
out.append("| policy | " + " | ".join(wins) + " |\n|---|" + "---|" * len(wins))
for k in order:
    if k.count("|") == 2:
        out.append(f"| {nice(k)} | " + " | ".join(dci(H["policies"][k]["windows"][w]["delta_vs_nonforgetting_hybrid"]) for w in wins) + " |")

# --- diagnosis ---
G = J("diagnosis.json")
out.append("\n### 5a. Non-forgetting hybrid LinUCB: weight movement at the end of each held-out day\n")
for mode, v in G["5a_weight_movement"].items():
    out.append(f"**{mode}** — cosine between consecutive day-end θ: " + ", ".join(f"{k}: {x:.3f}" for k, x in v["cosine_consecutive_day_ends"].items())
               + "; relative ‖Δθ‖/‖θ‖: " + ", ".join(f"{k}: {x:.2f}" for k, x in v["relative_change_norm"].items())
               + "; ‖θ‖: " + ", ".join(f"{k}: {x:.2f}" for k, x in v["theta_norm_by_day"].items())
               + "; θ[lgb_prob]: " + ", ".join(f"{k}: {x:.3f}" for k, x in v["theta_lgb_prob_by_day"].items()) + ".\n")
    out.append("| transition | top-10 features by \\|Δθ\\| (signed Δ) |\n|---|---|")
    for tr, lst in v["top10_changes"].items():
        out.append(f"| {tr} | " + ", ".join(f"{x['feature']} ({x['delta']:+.3f})" for x in lst) + " |")
    out.append("")
B = G["5b_ranker_staleness"]["per_day"]
out.append("### 5b. Is the ranker itself changing across days 3-6?\n")
out.append("| day | LightGBM P@1 [CI] | expected-random P@1 | LightGBM lift over random | mean candidates | share of large sets | LightGBM small / large sets | non-forgetting hybrid small / large sets |\n|---|---|---:|---:|---:|---:|---|---|")
for d, e in B.items():
    out.append(f"| {d} | {ci(e['lgbm_ctr'])} | {e['expected_random_ctr']:.4f} | {e['lgbm_lift_over_expected_random']:.3f}× | {e['mean_candidates']:.2f} | {e['share_large_sets']:.3f} | "
               f"{e['lgbm_ctr_small_sets']:.4f} / {e['lgbm_ctr_large_sets']:.4f} | {e['hybrid_linucb_per_impression_ctr_small_sets']:.4f} / {e['hybrid_linucb_per_impression_ctr_large_sets']:.4f} |")
out.append(f"\nLightGBM day 6 − day 3: {G['5b_ranker_staleness']['lgbm_day6_minus_day3']:+.4f}; non-forgetting hybrid (per-imp.) day 6 − day 3: {G['5b_ranker_staleness']['hybrid_linucb_per_impression_day6_minus_day3']:+.4f}.\n")
A = G["5c_age_bucket_slices"]
out.append("### 5c. Final 20% (and whole held-out) by age of the CLICKED article at impression time\n")
out.append("| scope \\| bucket | n | LightGBM P@1 | hybrid per-imp. Δ vs LightGBM | hybrid hourly Δ vs LightGBM |\n|---|---:|---:|---|---|")
for k, r in A["rows"].items():
    out.append(f"| {k} | {r['n']:,} | {r['lgbm_ctr']['value']:.4f} | {dci(r['hybrid_linucb|per_impression']['delta_vs_lgbm'])} {tag(r['hybrid_linucb|per_impression']['delta_vs_lgbm'])} | {dci(r['hybrid_linucb|hourly']['delta_vs_lgbm'])} {tag(r['hybrid_linucb|hourly']['delta_vs_lgbm'])} |")
X = G["full_information_transfer"]
out.append("\n### Full-information transfer (OFFLINE diagnostic, not a policy): ridge fitted on the first half of stream day i, tested on the second half of day j; entry = P@1 minus LightGBM's on those impressions\n")
out.append("| fit day \\ test day | " + " | ".join(str(j) for j in X["days"]) + " |\n|---|" + "---:|" * len(X["days"]))
for i, row in enumerate(X["delta_matrix"]):
    out.append(f"| {i} | " + " | ".join(f"{v:+.4f}" for v in row) + " |")
out.append("\nθ cosine range across all day pairs: " + f"{min(min(r) for r in X['theta_cosine_matrix']):.3f} – 1.000 (stream days 0-6).\n")

# --- realtime audit ---
R = J("realtime_audit.json")
out.append("### Audit 1 — label permutation (tuning period): precision@1 of the frozen forgetting winners\n")
out.append(", ".join(f"{k}: {v:.4f}" for k, v in R["1_label_permutation_tuning_period"].items()) + ".\n")
out.append("### Audit 2 — real-time popularity baseline (no learning; full information about EARLIER impressions)\n")
fz = R["2_realtime_popularity_baseline"]["frozen"]
out.append("| scorer | held-out P@1 [CI] | Δ vs LightGBM [CI] |\n|---|---|---|")
for k, v in fz.items():
    if "delta_vs_lgbm" in v and "heldout_ctr" in v:
        out.append(f"| {k}{' ('+v['window']+', β='+str(v['beta'])+')' if 'window' in v else ''} | {ci(v['heldout_ctr'])} | {dci(v['delta_vs_lgbm'])} |")
out.append(f"\nBlend selected on days 0-2: {R['2_realtime_popularity_baseline']['selected_on_tuning']}. Coverage on held-out: "
           + ", ".join(f"{k.replace('_', ' ')}: {v:.3f}" for k, v in R["4_coverage_heldout"].items()) + ".\n")
out.append("### Audit 3 — feedback-delay sensitivity (frozen policies, days 3-6; POST-HOC, no re-tuning)\n")
delays = list(next(iter(R["3_feedback_delay_sensitivity_heldout_frozen"].values())))
out.append("| policy | " + " | ".join(delays) + " |\n|---|" + "---|" * len(delays))
for name, v in R["3_feedback_delay_sensitivity_heldout_frozen"].items():
    out.append(f"| {name} (P@1; Δ vs LightGBM) | " + " | ".join(f"{v[d]['ctr']['value']:.4f}; {v[d]['delta_vs_lgbm']['value']:+.4f}" for d in delays) + " |")

# --- groups ---
for base, title in (("plain", "non-forgetting hybrid LinUCB"), ("forget", "exponential-forgetting hybrid LinUCB (per-impression and hourly)")):
    Gp = J(f"groups_{base}_heldout.json")
    out.append(f"\n### 6. Feature-group ablation — {title}\n")
    out.append("Group sizes (columns): " + ", ".join(f"{k} {v}" for k, v in Gp["group_sizes"].items()) + ". Every spec is re-tuned on days 0-2, frozen, run once on days 3-6.\n")
    out.append("| spec | mode | P@1 [CI] | Δ vs LightGBM [CI] | Δ vs full hybrid [CI] | final-20% Δ vs LightGBM |\n|---|---|---|---|---|---|")
    rows = sorted(Gp["rows"].values(), key=lambda e: (e["mode"], -e["windows"]["overall"]["ctr"]["value"]))
    for e in rows:
        w = e["windows"]
        out.append(f"| {e['spec']} | {e['mode']} | {ci(w['overall']['ctr'])} | {dci(w['overall']['delta_vs_lgbm'])} | {dci(w['overall']['delta_vs_full_hybrid'])} | {dci(w['final 20%']['delta_vs_lgbm'])} |")
Ge = J("groups_forget_extra_heldout.json")
out.append("\n**Follow-up (forgetting hybrid):** LightGBM score + one-hots + article metadata only\n")
out.append("| spec | mode | P@1 [CI] | Δ vs LightGBM [CI] | Δ vs full [CI] |\n|---|---|---|---|---|")
for e in Ge["rows"].values():
    if e["spec"] != "full (no ablation)":
        w = e["windows"]["overall"]
        out.append(f"| {e['spec']} | {e['mode']} | {ci(w['ctr'])} | {dci(w['delta_vs_lgbm'])} | {dci(w['delta_vs_full_hybrid'])} |")

# --- latency ---
import glob
lat = json.load(open(sorted(glob.glob(str(D.parents[2] / "benchmarks" / "results" / "*ebnerd_small_bandit_profile.json")))[-1]))
out.append("\n### 7. Latency (ms per decision; Apple M3 / 8 GiB; 4,700 timed decisions; single run)\n")
out.append("| policy | select+update p50 | p90 | p99 | select p50 | update p50 | decisions/s |\n|---|---:|---:|---:|---:|---:|---:|")
for k, v in lat["per_request"].items():
    d = v["decision_ms"]
    out.append(f"| {k} | {d['p50']:.4f} | {d['p90']:.4f} | {d['p99']:.4f} | {v.get('select_ms_p50', float('nan')):.4f} | {v.get('update_ms_p50', float('nan')):.4f} | {v['decisions_per_s']:,.0f} |")
(D / "tables.md").write_text("\n".join(out) + "\n")
print(len(out), "blocks ->", D / "tables.md")
