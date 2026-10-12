# ADR-018 — Drift hypothesis, discounted updates, and what the hybrid bandit's correction uses

**Date:** 2026-10-12
**Status:** Decided (results measured; conclusions bounded by the limitations below)
**Severity:** Medium (experiment only; nothing deployed changes)
**Builds on:** ADR-017 (stream, features, frozen LightGBM score, tuning = days 0-2, held-out = days 3-6, user-bootstrap n=2000).
**Untouched:** the 0.7542 artifacts, `submissions/`, the training pipeline. Nothing in this ADR used data beyond ADR-017's stream.

---

# Engineering question

ADR-017 left one result unexplained: the hybrid bandit's lead over LightGBM-greedy shrank from
+0.0156 (held-out day 1) to +0.0024 [−0.0004, +0.0056] in the final 20 %, and its mechanism was
unknown ("about half the gain survives removing static article metadata"). Two follow-ups:

1. **Drift:** does a bandit that forgets (exponential discount / sliding window / periodic reset)
   hold its lead, and is "drift" really what was going on?
2. **Mechanism:** what is the learned correction actually using?

# Summary of what was measured (read this first)

| Question | Answer (measured) |
|---|---|
| Does forgetting recover the final-window lead? | **Yes, and much more.** Per-impression updates: exponential-forgetting hybrid LinUCB **+0.0457 [+0.0408, +0.0507]** over LightGBM-greedy in the final 20 % (non-forgetting: +0.0024 n.s.); overall **0.3558 vs 0.3089 precision@1, +0.0469 [+0.0448, +0.0492]**, i.e. +0.0363 [+0.0340, +0.0386] over the non-forgetting hybrid. The lead is *flat-to-growing* across days (+0.045, +0.050, +0.046, +0.046). Sliding window (+0.0409) and periodic reset (+0.0302) also win, in that order. With hourly batch updates the same variants give +0.0206 (forgetting), +0.0140 (window), +0.0125 (reset). |
| Was the drift hypothesis supported? | **Partly, and not in the form stated.** What forgetting exploits is *fast* non-stationarity (memory ≈ 1,000 observations ≈ 45 minutes of tuning-period traffic), not slow drift of the optimal weights: a full-information linear correction is stable across the week (θ cosines 0.95–0.99; day-to-day transfer gains ≤ 0.016, not diagonal-dominated), and the ranker is **not** going stale (its lift over a random pick is flat, 2.67×→2.70×). The shrinking lead of the non-forgetting hybrid is a flat bandit next to a ranker whose precision rises (partly list-size mix), not a degrading model. |
| What carries the gain? | **Article descriptors, not users.** For the forgetting hybrid, LightGBM score + one-hot category/type/sentiment + static article metadata alone reproduces **91 %** of the gain (+0.0428 of +0.0469); dropping the 41 one-hots costs −0.0164, dropping metadata −0.0090; the 29 user-history features contribute nothing (dropping them is +0.0034 *better*). A non-learning "LightGBM probability + trailing 2-hour CTR" blend, using the labels of earlier impressions, already gets **+0.0326 [+0.0307, +0.0345]**. The correction is largely *real-time article/section popularity* — a signal Candidate K lacks by design (ADR-013 uses history-window popularity only). |
| How much of it survives realistic feedback latency? | **Little, beyond ~15 min.** Frozen forgetting policy, reward available only `delay` after the impression: +0.0469 (0), +0.0399 (5 min), +0.0267 (15 min), +0.0112 (30 min), **−0.0040 [−0.0061, −0.0017] (60 min)**. Non-forgetting hybrid: +0.0106 → +0.0075 (15 min) → +0.0001 (60 min). The headline per-impression numbers assume near-instant click feedback. |

> **Superseded in part by Addendum 2 (end of this ADR):** the 60-minute row above used hyper-parameters frozen at D = 0; re-tuned under delay the hybrid is +0.0049 at 60 min, and the retrained GBDT with trailing-CTR features beats the forgetting hybrid at every D tested.

**Therefore the honest reading of the +0.047 is not "a better learner beats the ranker". It is
"the ranker has no real-time popularity signal, and a fast-forgetting online layer supplies one
if clicks are fed back within minutes".** The fair comparison — a GBDT retrained with trailing-CTR
features — was *not* run (see Alternatives / next steps).

# Method

**New update rules** (`src/bandit/policies.py::_SharedLinear`, all default-off; with defaults the code
path and every ADR-017 number are unchanged — a test pins it):

* **Exponential forgetting** `A ← γA + xxᵀ, b ← γb + r·x`. Derivation: (γA)⁻¹ = A⁻¹/γ, so the scaling
  is one O(d²) pass, then the usual rank-1 Sherman–Morrison add. *Deviation from the literal rule,
  and why:* applied literally, the ridge prior decays too — in a direction that never sees data,
  A → 0 and A⁻¹ → ∞ (a test demonstrates a coordinate's precision falling below 10⁻⁶·λ). So every
  `floor_every = 256` steps the prior mass lost since the last restore, λ(1−γ^256)·I (and λ·θ₀ in b),
  is added back and A⁻¹ re-derived by direct inversion — amortised O(d³/256) < O(d²) per step. This
  approximates the standard weighted-ridge form A ← γA + xxᵀ + (1−γ)λI to within 256 steps. Tests compare
  A, A⁻¹ and θ against the direct recursion/inverse at every step, with and without the restore.
* **Sliding window** of the last W observations: rank-1 add of the new (x, r) and rank-1 *remove* of the
  oldest, A⁻¹ += (A⁻¹x)(A⁻¹x)ᵀ/(1 − xᵀA⁻¹x); the prior λI is never removed so A stays positive definite.
  Tests compare with the direct solve over the exact window at every step, W ∈ {1, 7, 50}.
* **Periodic reset** to the prior every T observations (crude baseline).
* Unit tests also cover: shuffled-future-label invariance of all six new policies in per-impression and
  hourly modes; a feedback-delay path (`delay_s`) that only releases an observation once its delay has
  elapsed; feature masks (a masked column has no effect on any decision); the group partition.
  Full suite: see "Verification".

**Tuning protocol** (unchanged): days 0-2 only, one run per configuration, tuning seed 1000; per
(family, update mode, variant) the best tuning CTR is frozen; the held-out run is made once (5 seeds for
Thompson). The only parameters searched were the forgetting parameter and λ; α, v and the warm-start
weight w stayed at the ADR-017 winners (α, v are flat there).

**Grid events (disclosed, as before).** First pass: γ ∈ {0.99999, 0.99995, 0.9999, 0.9995, 0.999, 0.995};
W ∈ {1e3, 3e3, 1e4, 3e4, 6e4}; T ∈ {5e3, 1.5e4, 3e4, 6e4}; λ ∈ {10, 100, 1000}. Winners W=1000, T=5000 and
λ ∈ {10, 1000} sat on edges, so the grid was widened **once**: W ∈ {100, 300, …}, T ∈ {1000, 2500, …},
λ ∈ {1, 3, 10, 100, 1000, 3000}. Result: the shorter windows (100, 300) and resets (1000, 2500) lost to
W=1000 / T=5000, so those winners are now interior; one winner (per-impression forgetting LinUCB) moved to
λ = 1, the new lower edge (its tuning CTR gained only 0.0010). γ winners were interior in both passes
(0.999 per-impression, 0.9995–0.9999 hourly). First pass kept in `drift/tune_forget_first_pass.json`.

# Findings

1. **Forgetting works, and the final-window lead is recovered and then some.** Tuning-period CTR
   moved from 0.3478 to 0.3722 (per-impression), and the held-out result (tables below) held up:
   forgetting hybrid LinUCB 0.3558 [0.3527, 0.3590] vs LightGBM-greedy 0.3089; every held-out day is
   CI-clear (+0.0451, +0.0500, +0.0462, +0.0462) and so is the final 20 % (+0.0457 [+0.0408, +0.0507]).
   Thompson gives the same picture (+0.0463). Ranking of the three rules: forgetting > window > reset,
   in both update modes. Hourly batching keeps a CI-clear lead in the final 20 % for every variant
   (forgetting +0.0196 [+0.0151, +0.0243]) but shrinks it by 56 % relative to per-impression.
2. **It is not slow drift.** (5a) The non-forgetting hybrid's end-of-day weight vectors are similar
   (cosine 0.922, 0.981, 0.981 per-impression; 0.964, 0.988, 0.989 hourly) and *move toward the
   ranker*: the weight on the calibrated LightGBM probability rises 0.425 → 0.628 as data accumulates,
   consistent with a long-run average washing out the local correction. The weights that move most are
   category/type one-hots (`category_code=11`, `article_type=3`, `category_code=15`, …), i.e.
   section-level popularity. A full-information check (offline ridge fitted on half a day, tested on the
   second half of every other day) shows the *best static* linear correction is worth ≤ +0.016 and is not
   diagonal-dominated; per-day θ cosines are 0.95–0.99 across all pairs. (Diagnostic only: it uses
   held-out-period labels offline and is not a policy.)
3. **The ranker did not get worse; the bandit did not either.** (5b) LightGBM-greedy P@1 rises
   0.3028 → 0.3152 from day 3 to day 6 (+0.0123) while the non-forgetting hybrid stays flat (0.3184 →
   0.3193, +0.0009). LightGBM's *lift over the expected-random baseline* is constant (2.669×, 2.642×,
   2.678×, 2.699×); the expected-random baseline itself rises 0.1135 → 0.1168 because the share of
   large candidate sets falls 0.467 → 0.442, which at constant lift accounts for about 0.009 of the 0.012 rise.
   So the non-forgetting lead shrank because the ranker's measured precision rose with the easier mix
   while the bandit (which averages the whole past) stayed flat — not because either degraded.
4. **Fresh articles do not explain the shrinkage by themselves.** (5c) Slicing by the age of the *clicked*
   article (an evaluation-side partition): over the whole held-out period the non-forgetting hybrid's gain is
   largest on <6 h articles (+0.0132 [+0.0110, +0.0151]), and 24-72 h (+0.0196) with a CI-clear **loss** on
   ≥72 h (−0.0065 [−0.0103, −0.0024]). In the final 20 % the <6 h lead is +0.0039 [+0.0002, +0.0077] (72 % of
   those impressions), 6-24 h −0.0024 n.s., 24-72 h +0.0154, ≥72 h −0.0092 [−0.0182, 0.0000]. The lead has
   shrunk on fresh articles too, not vanished only there. Not done: the same slice for the forgetting variants.
5. **Mechanism = article descriptors tracking what is being clicked right now.** (6) See the ablation tables.
   *Forgetting hybrid* (per-impression, each spec re-tuned on days 0-2): drop one-hots −0.0164, drop
   article_metadata −0.0090, drop freshness −0.0017, drop context/session/popularity −0.0021, drop user_history
   +0.0034; keep-only (LightGBM score + one group): one-hots +0.0293, metadata +0.0170, user_history +0.0136,
   freshness +0.0023, context −0.0001; one-hots + metadata together +0.0428 (91 %). Effects are
   non-additive (the two descriptor groups overlap: both identify "which article/section"). The LightGBM score
   is needed (dropping it: −0.0030 [−0.0059, −0.0004], and the no-forgetting linear model without it is
   −0.0276) — without the ranker's score the forgetting linear model only *matches* LightGBM (0.3059 vs 0.3089).
   *Non-forgetting hybrid, per-impression*: **no group explains it** — every drop-one leaves ≥ 74 % of its +0.0102
   (worst: user_history −0.0027) and the best keep-only is the one-hots at +0.0064 (63 %): the signal is
   redundantly spread, which also corrects ADR-017's "half survives removing static metadata". *Non-forgetting,
   hourly*, is different: user_history is the largest contributor (drop −0.0036 [−0.0051, −0.0022], leaving +0.0018 of
   +0.0054; keep-only user_history retains +0.0043) — the slow model leans on user features, the fast one on article
   descriptors.
6. **Audits.** (i) Label permutation (tuning period): every forgetting variant collapses to random
   (0.1177 / 0.1189 / 0.1201 / 0.1190 / 0.1193 vs 0.1192 expected) — no label path through the pipeline.
   (ii) Real-time popularity baseline: pure trailing-click counts (ties→LightGBM) are *worse* than
   LightGBM (−0.049 … −0.022), but the blend `p_lgb + β·smoothed trailing CTR` (window 2 h, β = 2, chosen on days
   0-2) is +0.0326 [+0.0307, +0.0345] (+0.0234 in the final 20 %), with full information about earlier impressions.
   The forgetting bandit, which sees only its own picks, scores 0.0143 higher still in P@1 (0.3558 vs 0.3415; *not* a paired test).
   Coverage: 95.3 % of held-out clicked articles had ≥ 1 click in the previous 30 minutes (97.3 % in 2 h); only 2.5 % had been clicked
   by the same user before (so this is not a same-user repeat effect). (iii) Feedback delay: see the summary row;
   with a 60-minute delay the forgetting policy is CI-clearly *worse* than LightGBM-greedy (−0.0040) and the
   window variant is worse at 30 minutes (−0.0087).
7. **Latency** (`benchmarks/results/2026-10-12T22-06-36_ebnerd_small_bandit_profile.json`, same harness/format as
   ADR-017, M3 / 8 GiB, single run, 4,700 timed decisions after 300 warm-up): select+update p50 / p99 in ms —
   LightGBM-greedy predict 0.317 / 1.62; hybrid LinUCB 0.060 / 1.24; **+forgetting 0.068 / 1.15**; **+window 0.102 / 1.48**
   (add + remove = two rank-1 updates, update p50 0.083 vs 0.042); +reset 0.067 / 1.01; Thompson+forgetting 0.111 / 2.09.
   Forgetting costs ≈ +0.008 ms p50 over the non-forgetting hybrid (the periodic re-inversion shows up only in the tail);
   the window rule doubles the update cost. p99 values vary run to run on a laptop (the same hybrid LinUCB had a
   0.39 ms p99 in ADR-017's run vs 1.24 here): no CI is claimed, and the hybrid still needs the ranker's score (+0.317 ms p50).

# Limitations

* **The headline depends on near-instant feedback.** EB-NeRD logs no click timestamps, so the real feedback delay
  distribution is unknowable here; the delay curve above is the substitute. The conservative number is the
  hourly-batch result (+0.0206 for forgetting), and below 60 minutes of delay the forgetting policy loses to the ranker.
* **Unequal information in the comparison.** LightGBM-greedy cannot see in-stream clicks by construction
  (ADR-013); the bandit can. The right baseline for "is a bandit needed" is a GBDT retrained with trailing-CTR
  features, which was **not** run. Do not read +0.047 as a bandit-vs-tree result.
* **Replay caveats from ADR-017 all still apply** (selected/click-only logs, non-random logging, top-1 precision rather
  than CTR, one week, a locally retrained small model, not the Ada 0.7542 model). `ebnerd_small` is a user sample, so
  real traffic would see denser popularity feedback — but also different concentration; untested.
* **Tuning and selection.** 3 tuning days; best of 36–42 (parameter × λ) configurations per cell; several tuning surfaces are flat
  or near-flat so frozen values are not individually meaningful. Group ablations re-tune a small grid per spec
  (12–18 configs per mode), smaller than the headline grid, so "drop-one vs full" differences of ≲ 0.002 are within selection noise.
* **Post-hoc analyses.** The delay sweep and the popularity baseline were designed *after* seeing the large gain
  (the baseline's window and β were still chosen on days 0-2). The mechanism claim rests on ablations and a baseline,
  not on an interventional test of the proposed mechanism.
* **Slices** are evaluation-side partitions defined on the label (clicked-article age); CIs are not multiplicity-corrected.
* **Not run:** age slice for forgetting variants; a delay-aware tuning (tuning γ under a delay); discount applied per unit
  time instead of per observation; the Ada-scale model.

# Alternatives considered / next steps

* **GBDT with trailing-CTR features** (retrain Candidate K with `clicks_last_{30min,2h}` / smoothed CTR as features,
  computed causally) — the baseline this result owes. If it recovers most of +0.03–0.047, the production answer is a
  feature, not a bandit.
* **Time-based discounting** (γ per second rather than per observation) so memory is stable when traffic rate changes.
* **Tune under a feedback delay** (the winners above assume none).
* **Ablate the forgetting variant at a finer granularity** (single category one-hots vs. article fingerprints).

# Decision and conditions to revisit

* **Decision.** Keep ADR-017's verdict that plain linear bandits should not ship; **upgrade** the hybrid candidate to the
  exponential-forgetting variant for any further online work, but treat its offline lead as an *upper bound* that holds
  only for sub-15-minute feedback. Prefer the cheapest route to the same signal (trailing-CTR features in the ranker) before
  taking on online state.
* **Revisit if:** the GBDT-with-trailing-CTR baseline matches the forgetting bandit; a delay-aware tuning shows the lead
  survives 30-60 minutes; or a second week / the `ebnerd_large` ranker shows the lead < +0.01.

# Verification

* `tests/unit/test_bandit_drift.py` (17 cases) + `tests/unit/test_bandit.py` (20): SM-with-scaling vs the direct
  recursion at every step, sliding-window vs direct solve, reset, default-path equality, restore-prior floor,
  shuffled-future invariance for all new policies in both update modes and under delayed feedback, mask effect,
  group partition. Full suite: see PROJECT_STATE.md.
* The diagnostic re-run of the non-forgetting hybrid reproduces ADR-017's stored held-out rewards **exactly**
  (`np.array_equal`), so ADR-017's numbers were re-verified, not re-derived.

# Reproduce

```bash
poetry run python scripts/run_drift_experiment.py --stage tune     # ~7 min; writes drift/tune_forget.json
poetry run python scripts/run_drift_experiment.py --stage final    # held-out, once
poetry run python scripts/run_drift_experiment.py --stage diag
poetry run python scripts/run_drift_experiment.py --stage groups --base plain
poetry run python scripts/run_drift_experiment.py --stage groups --base forget
poetry run python scripts/run_drift_experiment.py --stage groups --base forget --extra
poetry run python scripts/bandit_realtime_audit.py                 # ~10 min
poetry run python benchmarks/profile_bandit.py
poetry run python scripts/drift_make_report.py                     # renders the tables below from the JSON
```

---

# Appendix — all tables (rendered from `results/bandit_ebnerd_small/drift/*.json`)

### Tuning-period winners (days 0-2; widened grid, first-pass winner in brackets)

| policy | variant | frozen parameters | tuning CTR | first pass |
|---|---|---|---:|---|
| hybrid_lints (hourly) | forget | `{"v": 0.1, "lam": 10, "w": 1.0, "gamma": 0.9999}` | 0.3476 | 0.3476 `{"lam": 10, "gamma": 0.9999}` |
| hybrid_lints (hourly) | reset | `{"v": 0.1, "lam": 1000, "w": 1.0, "reset_every": 5000}` | 0.3452 | 0.3452 `{"lam": 1000, "reset_every": 5000}` |
| hybrid_lints (hourly) | window | `{"v": 0.1, "lam": 1000, "w": 1.0, "window": 1000}` | 0.3457 | 0.3457 `{"lam": 1000, "window": 1000}` |
| hybrid_linucb (hourly) | forget | `{"alpha": 0.0, "lam": 10, "w": 1.0, "gamma": 0.9995}` | 0.3482 | 0.3482 `{"lam": 10, "gamma": 0.9995}` |
| hybrid_linucb (hourly) | reset | `{"alpha": 0.0, "lam": 1000, "w": 1.0, "reset_every": 5000}` | 0.3449 | 0.3449 `{"lam": 1000, "reset_every": 5000}` |
| hybrid_linucb (hourly) | window | `{"alpha": 0.0, "lam": 1000, "w": 1.0, "window": 1000}` | 0.3458 | 0.3458 `{"lam": 1000, "window": 1000}` |
| hybrid_lints (per_impression) | forget | `{"v": 0.1, "lam": 10, "w": 0.0, "gamma": 0.999}` | 0.3708 | 0.3708 `{"lam": 10, "gamma": 0.999}` |
| hybrid_lints (per_impression) | reset | `{"v": 0.1, "lam": 100, "w": 0.0, "reset_every": 5000}` | 0.3589 | 0.3589 `{"lam": 100, "reset_every": 5000}` |
| hybrid_lints (per_impression) | window | `{"v": 0.1, "lam": 100, "w": 0.0, "window": 3000}` | 0.3642 | 0.3642 `{"lam": 100, "window": 3000}` |
| hybrid_linucb (per_impression) | forget | `{"alpha": 0.003, "lam": 1, "w": 0.0, "gamma": 0.999}` | 0.3722 | 0.3712 `{"lam": 10, "gamma": 0.999}` |
| hybrid_linucb (per_impression) | reset | `{"alpha": 0.003, "lam": 100, "w": 0.0, "reset_every": 5000}` | 0.3597 | 0.3597 `{"lam": 100, "reset_every": 5000}` |
| hybrid_linucb (per_impression) | window | `{"alpha": 0.003, "lam": 3, "w": 0.0, "window": 1000}` | 0.3656 | 0.3646 `{"lam": 10, "window": 1000}` |

Non-forgetting references (tuning CTR): hybrid_linucb|per_impression 0.3478, hybrid_linucb|hourly 0.3420, hybrid_lints|per_impression 0.3472, hybrid_lints|hourly 0.3421.

### Held-out (days 3-6): precision@1 [95% user-bootstrap CI]

Window sizes (impressions): overall 147,492, day 3 35,041, day 4 36,902, day 5 38,292, day 6 37,257, final 20% 29,498.

| policy | overall | day 3 | day 4 | day 5 | day 6 | final 20% |
|---|---|---|---|---|---|---|
| LightGBM-greedy | 0.3089 | 0.3028 | 0.3029 | 0.3140 | 0.3152 | 0.3169 |
| hybrid_linucb per-imp. | 0.3195 [0.3164, 0.3227] | 0.3184 [0.3131, 0.3239] | 0.3198 [0.3146, 0.3250] | 0.3203 [0.3151, 0.3256] | 0.3193 [0.3138, 0.3247] | 0.3193 [0.3131, 0.3258] |
| hybrid_linucb hourly | 0.3143 [0.3110, 0.3175] | 0.3048 [0.2994, 0.3101] | 0.3149 [0.3097, 0.3201] | 0.3199 [0.3146, 0.3253] | 0.3167 [0.3111, 0.3221] | 0.3171 [0.3109, 0.3235] |
| hybrid_lints per-imp. | 0.3195 [0.3164, 0.3227] | 0.3183 [0.3132, 0.3235] | 0.3199 [0.3147, 0.3251] | 0.3205 [0.3151, 0.3260] | 0.3190 [0.3136, 0.3244] | 0.3191 [0.3129, 0.3256] |
| hybrid_lints hourly | 0.3146 [0.3115, 0.3178] | 0.3056 [0.3002, 0.3108] | 0.3153 [0.3101, 0.3203] | 0.3195 [0.3143, 0.3250] | 0.3175 [0.3119, 0.3228] | 0.3176 [0.3114, 0.3239] |
| hybrid_linucb per-imp. forget | 0.3558 [0.3527, 0.3590] | 0.3480 [0.3427, 0.3535] | 0.3529 [0.3475, 0.3582] | 0.3602 [0.3549, 0.3657] | 0.3614 [0.3556, 0.3671] | 0.3626 [0.3561, 0.3690] |
| hybrid_lints per-imp. forget | 0.3552 [0.3523, 0.3581] | 0.3459 [0.3410, 0.3510] | 0.3505 [0.3455, 0.3556] | 0.3618 [0.3568, 0.3671] | 0.3618 [0.3563, 0.3673] | 0.3635 [0.3573, 0.3697] |
| hybrid_linucb per-imp. window | 0.3498 [0.3468, 0.3528] | 0.3412 [0.3360, 0.3468] | 0.3431 [0.3377, 0.3483] | 0.3603 [0.3552, 0.3656] | 0.3536 [0.3479, 0.3590] | 0.3552 [0.3488, 0.3618] |
| hybrid_linucb per-imp. reset | 0.3391 [0.3360, 0.3421] | 0.3332 [0.3278, 0.3386] | 0.3293 [0.3240, 0.3342] | 0.3451 [0.3398, 0.3502] | 0.3483 [0.3426, 0.3538] | 0.3478 [0.3413, 0.3542] |
| hybrid_lints per-imp. window | 0.3484 [0.3455, 0.3514] | 0.3412 [0.3365, 0.3465] | 0.3417 [0.3364, 0.3466] | 0.3542 [0.3491, 0.3594] | 0.3558 [0.3502, 0.3612] | 0.3576 [0.3513, 0.3639] |
| hybrid_lints per-imp. reset | 0.3395 [0.3367, 0.3423] | 0.3322 [0.3274, 0.3370] | 0.3315 [0.3269, 0.3360] | 0.3476 [0.3427, 0.3526] | 0.3458 [0.3406, 0.3510] | 0.3458 [0.3398, 0.3518] |
| hybrid_linucb hourly forget | 0.3295 [0.3265, 0.3326] | 0.3201 [0.3150, 0.3257] | 0.3277 [0.3220, 0.3329] | 0.3325 [0.3275, 0.3382] | 0.3371 [0.3314, 0.3426] | 0.3365 [0.3301, 0.3430] |
| hybrid_lints hourly forget | 0.3236 [0.3205, 0.3268] | 0.3142 [0.3092, 0.3194] | 0.3250 [0.3200, 0.3301] | 0.3200 [0.3149, 0.3254] | 0.3348 [0.3291, 0.3404] | 0.3353 [0.3292, 0.3416] |
| hybrid_linucb hourly window | 0.3229 [0.3198, 0.3260] | 0.3181 [0.3126, 0.3236] | 0.3115 [0.3062, 0.3166] | 0.3288 [0.3236, 0.3341] | 0.3327 [0.3270, 0.3380] | 0.3314 [0.3252, 0.3377] |
| hybrid_lints hourly window | 0.3224 [0.3193, 0.3254] | 0.3172 [0.3117, 0.3226] | 0.3117 [0.3065, 0.3168] | 0.3278 [0.3228, 0.3330] | 0.3324 [0.3269, 0.3376] | 0.3311 [0.3248, 0.3372] |
| hybrid_lints hourly reset | 0.3217 [0.3185, 0.3246] | 0.3179 [0.3125, 0.3231] | 0.3112 [0.3059, 0.3163] | 0.3266 [0.3214, 0.3320] | 0.3305 [0.3250, 0.3359] | 0.3289 [0.3226, 0.3351] |
| hybrid_linucb hourly reset | 0.3214 [0.3182, 0.3244] | 0.3179 [0.3126, 0.3232] | 0.3106 [0.3050, 0.3158] | 0.3261 [0.3208, 0.3316] | 0.3306 [0.3250, 0.3359] | 0.3288 [0.3226, 0.3352] |

### Held-out: paired Δ vs LightGBM-greedy [CI]

| policy | overall | day 3 | day 4 | day 5 | day 6 | final 20% |
|---|---|---|---|---|---|---|
| hybrid_linucb per-imp. | +0.0106 [+0.0090, +0.0123] | +0.0156 [+0.0117, +0.0197] | +0.0169 [+0.0136, +0.0205] | +0.0063 [+0.0030, +0.0097] | +0.0041 [+0.0012, +0.0069] | +0.0024 [-0.0004, +0.0056] |
| hybrid_linucb hourly | +0.0054 [+0.0038, +0.0071] | +0.0020 [-0.0017, +0.0054] | +0.0120 [+0.0087, +0.0155] | +0.0059 [+0.0027, +0.0092] | +0.0016 [-0.0013, +0.0044] | +0.0002 [-0.0027, +0.0034] |
| hybrid_lints per-imp. | +0.0106 [+0.0091, +0.0123] | +0.0155 [+0.0116, +0.0193] | +0.0170 [+0.0137, +0.0204] | +0.0065 [+0.0032, +0.0098] | +0.0038 [+0.0010, +0.0066] | +0.0022 [-0.0007, +0.0053] |
| hybrid_lints hourly | +0.0058 [+0.0043, +0.0074] | +0.0027 [-0.0007, +0.0060] | +0.0123 [+0.0093, +0.0156] | +0.0055 [+0.0023, +0.0088] | +0.0023 [-0.0004, +0.0051] | +0.0007 [-0.0020, +0.0037] |
| hybrid_linucb per-imp. forget | +0.0469 [+0.0448, +0.0492] | +0.0451 [+0.0404, +0.0501] | +0.0500 [+0.0453, +0.0548] | +0.0462 [+0.0416, +0.0507] | +0.0462 [+0.0418, +0.0507] | +0.0457 [+0.0408, +0.0507] |
| hybrid_lints per-imp. forget | +0.0463 [+0.0443, +0.0485] | +0.0430 [+0.0389, +0.0472] | +0.0476 [+0.0432, +0.0521] | +0.0478 [+0.0435, +0.0523] | +0.0466 [+0.0425, +0.0511] | +0.0466 [+0.0418, +0.0516] |
| hybrid_linucb per-imp. window | +0.0409 [+0.0386, +0.0434] | +0.0384 [+0.0335, +0.0435] | +0.0402 [+0.0352, +0.0450] | +0.0463 [+0.0417, +0.0514] | +0.0385 [+0.0337, +0.0432] | +0.0383 [+0.0328, +0.0439] |
| hybrid_linucb per-imp. reset | +0.0302 [+0.0279, +0.0326] | +0.0303 [+0.0258, +0.0352] | +0.0264 [+0.0216, +0.0310] | +0.0311 [+0.0263, +0.0359] | +0.0331 [+0.0287, +0.0377] | +0.0308 [+0.0257, +0.0363] |
| hybrid_lints per-imp. window | +0.0395 [+0.0374, +0.0416] | +0.0383 [+0.0341, +0.0425] | +0.0388 [+0.0344, +0.0431] | +0.0402 [+0.0361, +0.0447] | +0.0407 [+0.0362, +0.0450] | +0.0407 [+0.0357, +0.0457] |
| hybrid_lints per-imp. reset | +0.0306 [+0.0285, +0.0326] | +0.0293 [+0.0250, +0.0336] | +0.0286 [+0.0243, +0.0329] | +0.0336 [+0.0294, +0.0378] | +0.0306 [+0.0265, +0.0349] | +0.0289 [+0.0243, +0.0338] |
| hybrid_linucb hourly forget | +0.0206 [+0.0186, +0.0226] | +0.0173 [+0.0130, +0.0215] | +0.0247 [+0.0206, +0.0290] | +0.0185 [+0.0146, +0.0228] | +0.0219 [+0.0178, +0.0262] | +0.0196 [+0.0151, +0.0243] |
| hybrid_lints hourly forget | +0.0148 [+0.0130, +0.0167] | +0.0114 [+0.0076, +0.0151] | +0.0221 [+0.0186, +0.0257] | +0.0060 [+0.0024, +0.0098] | +0.0196 [+0.0159, +0.0235] | +0.0184 [+0.0141, +0.0226] |
| hybrid_linucb hourly window | +0.0140 [+0.0125, +0.0156] | +0.0153 [+0.0123, +0.0183] | +0.0086 [+0.0055, +0.0118] | +0.0148 [+0.0117, +0.0180] | +0.0175 [+0.0145, +0.0206] | +0.0145 [+0.0112, +0.0178] |
| hybrid_lints hourly window | +0.0135 [+0.0121, +0.0150] | +0.0143 [+0.0116, +0.0172] | +0.0088 [+0.0059, +0.0117] | +0.0138 [+0.0109, +0.0168] | +0.0172 [+0.0143, +0.0201] | +0.0142 [+0.0110, +0.0175] |
| hybrid_lints hourly reset | +0.0128 [+0.0113, +0.0143] | +0.0150 [+0.0120, +0.0181] | +0.0082 [+0.0053, +0.0112] | +0.0126 [+0.0094, +0.0157] | +0.0153 [+0.0123, +0.0183] | +0.0120 [+0.0086, +0.0153] |
| hybrid_linucb hourly reset | +0.0125 [+0.0109, +0.0141] | +0.0151 [+0.0119, +0.0183] | +0.0077 [+0.0045, +0.0108] | +0.0121 [+0.0087, +0.0153] | +0.0154 [+0.0124, +0.0186] | +0.0119 [+0.0083, +0.0154] |

### Held-out: paired Δ vs the non-forgetting hybrid of the same family and update mode [CI]

| policy | overall | day 3 | day 4 | day 5 | day 6 | final 20% |
|---|---|---|---|---|---|---|
| hybrid_linucb per-imp. forget | +0.0363 [+0.0340, +0.0386] | +0.0296 [+0.0251, +0.0341] | +0.0331 [+0.0286, +0.0375] | +0.0399 [+0.0352, +0.0445] | +0.0421 [+0.0375, +0.0467] | +0.0433 [+0.0382, +0.0484] |
| hybrid_lints per-imp. forget | +0.0357 [+0.0338, +0.0378] | +0.0276 [+0.0237, +0.0314] | +0.0306 [+0.0265, +0.0345] | +0.0413 [+0.0371, +0.0456] | +0.0428 [+0.0386, +0.0471] | +0.0444 [+0.0397, +0.0490] |
| hybrid_linucb per-imp. window | +0.0303 [+0.0280, +0.0327] | +0.0228 [+0.0183, +0.0278] | +0.0233 [+0.0181, +0.0279] | +0.0400 [+0.0353, +0.0450] | +0.0343 [+0.0295, +0.0394] | +0.0359 [+0.0303, +0.0415] |
| hybrid_linucb per-imp. reset | +0.0196 [+0.0173, +0.0218] | +0.0148 [+0.0105, +0.0190] | +0.0095 [+0.0047, +0.0139] | +0.0248 [+0.0200, +0.0293] | +0.0290 [+0.0245, +0.0337] | +0.0284 [+0.0232, +0.0337] |
| hybrid_lints per-imp. window | +0.0289 [+0.0270, +0.0309] | +0.0229 [+0.0194, +0.0266] | +0.0217 [+0.0175, +0.0256] | +0.0337 [+0.0296, +0.0378] | +0.0368 [+0.0326, +0.0411] | +0.0385 [+0.0336, +0.0435] |
| hybrid_lints per-imp. reset | +0.0200 [+0.0180, +0.0220] | +0.0138 [+0.0102, +0.0174] | +0.0116 [+0.0077, +0.0153] | +0.0271 [+0.0228, +0.0313] | +0.0268 [+0.0227, +0.0309] | +0.0266 [+0.0220, +0.0314] |
| hybrid_linucb hourly forget | +0.0152 [+0.0133, +0.0172] | +0.0153 [+0.0113, +0.0192] | +0.0128 [+0.0086, +0.0168] | +0.0126 [+0.0087, +0.0165] | +0.0203 [+0.0162, +0.0245] | +0.0194 [+0.0149, +0.0239] |
| hybrid_lints hourly forget | +0.0090 [+0.0075, +0.0106] | +0.0086 [+0.0061, +0.0111] | +0.0098 [+0.0069, +0.0127] | +0.0005 [-0.0028, +0.0036] | +0.0173 [+0.0136, +0.0208] | +0.0177 [+0.0137, +0.0217] |
| hybrid_linucb hourly window | +0.0086 [+0.0068, +0.0104] | +0.0133 [+0.0097, +0.0172] | -0.0034 [-0.0070, +0.0000] | +0.0089 [+0.0055, +0.0123] | +0.0159 [+0.0126, +0.0193] | +0.0143 [+0.0104, +0.0178] |
| hybrid_lints hourly window | +0.0078 [+0.0061, +0.0095] | +0.0116 [+0.0083, +0.0152] | -0.0035 [-0.0069, -0.0004] | +0.0083 [+0.0050, +0.0116] | +0.0149 [+0.0118, +0.0180] | +0.0135 [+0.0098, +0.0169] |
| hybrid_lints hourly reset | +0.0070 [+0.0053, +0.0086] | +0.0123 [+0.0091, +0.0156] | -0.0041 [-0.0073, -0.0009] | +0.0071 [+0.0037, +0.0103] | +0.0130 [+0.0098, +0.0163] | +0.0113 [+0.0075, +0.0150] |
| hybrid_linucb hourly reset | +0.0071 [+0.0053, +0.0088] | +0.0131 [+0.0097, +0.0168] | -0.0043 [-0.0078, -0.0009] | +0.0062 [+0.0026, +0.0096] | +0.0138 [+0.0104, +0.0173] | +0.0117 [+0.0078, +0.0157] |

### 5a. Non-forgetting hybrid LinUCB: weight movement at the end of each held-out day

**per_impression** — cosine between consecutive day-end θ: day 3->4: 0.922, day 4->5: 0.981, day 5->6: 0.981; relative ‖Δθ‖/‖θ‖: day 3->4: 0.47, day 4->5: 0.22, day 5->6: 0.21; ‖θ‖: day 3: 0.50, day 4: 0.59, day 5: 0.63, day 6: 0.67; θ[lgb_prob]: day 3: 0.425, day 4: 0.522, day 5: 0.577, day 6: 0.628.

| transition | top-10 features by \|Δθ\| (signed Δ) |
|---|---|
| day 3->4 | category_code=11 (-0.122), lgb_prob (+0.097), article_type=3 (-0.091), sentiment_score (+0.053), article_type=0 (+0.042), article_age_h (+0.040), article_type=9 (+0.037), article_age_minus_imp_min (+0.034), sentiment_label=1 (+0.032), category_code=15 (+0.032) |
| day 4->5 | lgb_prob (+0.055), category_code=11 (+0.051), category_code=9 (-0.036), article_age_h (+0.035), article_age_minus_imp_min (-0.032), category_code=21 (+0.031), article_type=8 (+0.030), category_code=22 (-0.029), sentiment_score (+0.024), sentiment_label=1 (+0.021) |
| day 5->6 | category_code=15 (-0.073), category_code=11 (+0.054), lgb_prob (+0.052), article_age_minus_imp_min (-0.034), category_code=10 (+0.027), lt_embed_sim (-0.020), sentiment_label=2 (+0.020), sentiment_label=0 (-0.019), lgb_centred (-0.017), cat_affinity_drift (-0.017) |

**hourly** — cosine between consecutive day-end θ: day 3->4: 0.963, day 4->5: 0.988, day 5->6: 0.989; relative ‖Δθ‖/‖θ‖: day 3->4: 0.27, day 4->5: 0.15, day 5->6: 0.15; ‖θ‖: day 3: 0.79, day 4: 0.77, day 5: 0.76, day 6: 0.77; θ[lgb_prob]: day 3: 0.743, day 4: 0.716, day 5: 0.706, day 6: 0.732.

| transition | top-10 features by \|Δθ\| (signed Δ) |
|---|---|
| day 3->4 | article_type=3 (-0.107), sentiment_score (+0.072), article_type=0 (+0.057), category_code=11 (-0.048), article_age_minus_imp_min (+0.047), category_code=15 (+0.041), sentiment_label=1 (+0.039), lt_sentiment_distance (+0.039), article_type=8 (+0.038), category_code=10 (-0.037) |
| day 4->5 | category_code=11 (+0.051), article_type=8 (+0.039), category_code=9 (-0.036), article_age_minus_imp_min (-0.033), article_age_h (+0.031), category_code=22 (-0.028), sentiment_label=1 (+0.025), sentiment_score (+0.024), category_code=21 (+0.024), bias (+0.020) |
| day 5->6 | category_code=15 (-0.070), category_code=11 (+0.047), article_age_minus_imp_min (-0.032), lgb_prob (+0.025), category_code=10 (+0.023), cat_affinity_drift (-0.021), sentiment_label=0 (-0.020), lt_embed_sim (-0.019), sentiment_label=2 (+0.018), lt_embed_sim_minus_imp_max (+0.013) |

### 5b. Is the ranker itself changing across days 3-6?

| day | LightGBM P@1 [CI] | expected-random P@1 | LightGBM lift over random | mean candidates | share of large sets | LightGBM small / large sets | non-forgetting hybrid small / large sets |
|---|---|---:|---:|---:|---:|---|---|
| day 3 | 0.3028 [0.2975, 0.3083] | 0.1135 | 2.669× | 12.54 | 0.467 | 0.3613 / 0.2361 | 0.3725 / 0.2567 |
| day 4 | 0.3029 [0.2978, 0.3082] | 0.1147 | 2.642× | 12.66 | 0.449 | 0.3653 / 0.2263 | 0.3740 / 0.2532 |
| day 5 | 0.3140 [0.3090, 0.3195] | 0.1173 | 2.678× | 12.02 | 0.440 | 0.3744 / 0.2372 | 0.3772 / 0.2480 |
| day 6 | 0.3152 [0.3098, 0.3207] | 0.1168 | 2.699× | 11.99 | 0.442 | 0.3865 / 0.2249 | 0.3899 / 0.2299 |

LightGBM day 6 − day 3: +0.0123; non-forgetting hybrid (per-imp.) day 6 − day 3: +0.0009.

### 5c. Final 20% (and whole held-out) by age of the CLICKED article at impression time

| scope \| bucket | n | LightGBM P@1 | hybrid per-imp. Δ vs LightGBM | hybrid hourly Δ vs LightGBM |
|---|---:|---:|---|---|
| final 20% | <6h | 21,218 | 0.3187 | +0.0039 [+0.0002, +0.0077] win | +0.0012 [-0.0024, +0.0047] n.s. |
| final 20% | 6-24h | 5,826 | 0.4231 | -0.0024 [-0.0099, +0.0047] n.s. | -0.0041 [-0.0122, +0.0036] n.s. |
| final 20% | 24-72h | 1,040 | 0.0452 | +0.0154 [+0.0049, +0.0263] win | +0.0173 [+0.0068, +0.0289] win |
| final 20% | >=72h | 1,414 | 0.0523 | -0.0092 [-0.0182, +0.0000] n.s. | -0.0092 [-0.0188, +0.0007] n.s. |
| held-out overall | <6h | 99,931 | 0.3246 | +0.0132 [+0.0110, +0.0151] win | +0.0070 [+0.0049, +0.0088] win |
| held-out overall | 6-24h | 35,043 | 0.3614 | +0.0048 [+0.0009, +0.0086] win | +0.0004 [-0.0032, +0.0038] n.s. |
| held-out overall | 24-72h | 6,214 | 0.0383 | +0.0196 [+0.0143, +0.0249] win | +0.0214 [+0.0159, +0.0269] win |
| held-out overall | >=72h | 6,304 | 0.0349 | -0.0065 [-0.0103, -0.0024] LOSS | -0.0073 [-0.0113, -0.0033] LOSS |

### Full-information transfer (OFFLINE diagnostic, not a policy): ridge fitted on the first half of stream day i, tested on the second half of day j; entry = P@1 minus LightGBM's on those impressions

| fit day \ test day | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | +0.0073 | +0.0040 | -0.0004 | +0.0008 | +0.0071 | +0.0070 | +0.0024 |
| 1 | +0.0157 | +0.0049 | +0.0039 | +0.0047 | -0.0022 | +0.0123 | +0.0065 |
| 2 | +0.0109 | +0.0044 | +0.0045 | +0.0019 | -0.0047 | +0.0112 | -0.0030 |
| 3 | -0.0012 | +0.0007 | +0.0068 | -0.0023 | -0.0034 | +0.0091 | +0.0056 |
| 4 | +0.0061 | +0.0053 | +0.0029 | +0.0068 | +0.0101 | +0.0038 | -0.0003 |
| 5 | -0.0036 | -0.0006 | +0.0029 | -0.0011 | +0.0074 | -0.0013 | -0.0010 |
| 6 | +0.0093 | +0.0028 | -0.0038 | -0.0030 | -0.0030 | +0.0063 | +0.0085 |

θ cosine range across all day pairs: 0.949 – 1.000 (stream days 0-6).

### Audit 1 — label permutation (tuning period): precision@1 of the frozen forgetting winners

random_expected: 0.1192, hybrid_linucb|per_impression|forget: 0.1177, hybrid_linucb|per_impression|window: 0.1189, hybrid_linucb|per_impression|reset: 0.1201, hybrid_lints|per_impression|forget: 0.1190, hybrid_linucb|hourly|forget: 0.1193.

### Audit 2 — real-time popularity baseline (no learning; full information about EARLIER impressions)

| scorer | held-out P@1 [CI] | Δ vs LightGBM [CI] |
|---|---|---|
| pure_recent_clicks_30min | 0.2596 [0.2564, 0.2628] | -0.0492 [-0.0523, -0.0461] |
| pure_recent_clicks_2h | 0.2732 [0.2701, 0.2766] | -0.0356 [-0.0384, -0.0328] |
| pure_recent_clicks_6h | 0.2872 [0.2841, 0.2903] | -0.0217 [-0.0245, -0.0188] |
| pure_recent_clicks_24h | 0.2721 [0.2693, 0.2749] | -0.0368 [-0.0397, -0.0336] |
| blend_lgbm_prob_plus_beta_x_smoothed_recent_ctr (2h, β=2.0) | 0.3415 [0.3384, 0.3448] | +0.0326 [+0.0307, +0.0345] |

Blend selected on days 0-2: {'window': '2h', 'beta': 2.0, 'tuning_ctr': 0.36000205856620865}. Coverage on held-out: clicked article had a click in previous 30min: 0.953, clicked article had a click in previous 2h: 0.973, same user had clicked this article before: 0.025.

### Audit 3 — feedback-delay sensitivity (frozen policies, days 3-6; POST-HOC, no re-tuning)

| policy | 0s | 60s | 300s | 900s | 1800s | 3600s | 7200s |
|---|---|---|---|---|---|---|---|
| hybrid_linucb non-forgetting (P@1; Δ vs LightGBM) | 0.3195; +0.0106 | 0.3194; +0.0105 | 0.3186; +0.0097 | 0.3163; +0.0075 | 0.3136; +0.0048 | 0.3089; +0.0001 | 0.3036; -0.0053 |
| hybrid_linucb forgetting (gamma winner) (P@1; Δ vs LightGBM) | 0.3558; +0.0469 | 0.3548; +0.0460 | 0.3488; +0.0399 | 0.3355; +0.0267 | 0.3201; +0.0112 | 0.3049; -0.0040 | 0.2895; -0.0194 |
| hybrid_linucb window (winner) (P@1; Δ vs LightGBM) | 0.3498; +0.0409 | 0.3483; +0.0394 | 0.3400; +0.0311 | 0.3226; +0.0137 | 0.3002; -0.0087 | 0.2897; -0.0192 | 0.2699; -0.0390 |

### 6. Feature-group ablation — non-forgetting hybrid LinUCB

Group sizes (columns): lightgbm_score 2, freshness 5, article_metadata 6, user_history 29, context_session_popularity 18, onehot_categoricals 41. Every spec is re-tuned on days 0-2, frozen, run once on days 3-6.

| spec | mode | P@1 [CI] | Δ vs LightGBM [CI] | Δ vs full hybrid [CI] | final-20% Δ vs LightGBM |
|---|---|---|---|---|---|
| drop context_session_popularity | hourly | 0.3162 [0.3130, 0.3195] | +0.0074 [+0.0058, +0.0090] | +0.0020 [+0.0012, +0.0028] | +0.0021 [-0.0009, +0.0053] |
| drop article_metadata | hourly | 0.3157 [0.3127, 0.3188] | +0.0069 [+0.0055, +0.0083] | +0.0015 [+0.0001, +0.0028] | +0.0014 [-0.0011, +0.0040] |
| drop freshness | hourly | 0.3154 [0.3123, 0.3185] | +0.0065 [+0.0051, +0.0080] | +0.0011 [+0.0002, +0.0022] | +0.0017 [-0.0010, +0.0047] |
| full (no ablation) | hourly | 0.3143 [0.3110, 0.3175] | +0.0054 [+0.0038, +0.0071] | +0.0000 [+0.0000, +0.0000] | +0.0002 [-0.0027, +0.0034] |
| keep lightgbm_score + onehot_categoricals | hourly | 0.3142 [0.3112, 0.3176] | +0.0054 [+0.0041, +0.0068] | -0.0000 [-0.0014, +0.0015] | -0.0048 [-0.0070, -0.0026] |
| drop onehot_categoricals | hourly | 0.3135 [0.3104, 0.3166] | +0.0047 [+0.0032, +0.0062] | -0.0007 [-0.0022, +0.0007] | +0.0063 [+0.0032, +0.0095] |
| keep lightgbm_score + user_history | hourly | 0.3132 [0.3101, 0.3165] | +0.0043 [+0.0031, +0.0055] | -0.0011 [-0.0027, +0.0005] | +0.0013 [-0.0013, +0.0042] |
| keep lightgbm_score + article_metadata | hourly | 0.3114 [0.3083, 0.3146] | +0.0025 [+0.0013, +0.0039] | -0.0029 [-0.0044, -0.0013] | +0.0022 [-0.0005, +0.0050] |
| drop user_history | hourly | 0.3106 [0.3075, 0.3137] | +0.0018 [+0.0001, +0.0035] | -0.0036 [-0.0051, -0.0022] | +0.0000 [-0.0028, +0.0031] |
| keep lightgbm_score + context_session_popularity | hourly | 0.3092 [0.3061, 0.3125] | +0.0004 [-0.0001, +0.0008] | -0.0050 [-0.0067, -0.0035] | +0.0011 [-0.0001, +0.0023] |
| keep only lightgbm_score | hourly | 0.3089 [0.3058, 0.3120] | -0.0000 [-0.0000, +0.0000] | -0.0054 [-0.0071, -0.0038] | +0.0000 [+0.0000, +0.0000] |
| keep lightgbm_score + freshness | hourly | 0.3087 [0.3055, 0.3120] | -0.0002 [-0.0011, +0.0008] | -0.0056 [-0.0072, -0.0040] | +0.0044 [+0.0024, +0.0065] |
| drop lightgbm_score | hourly | 0.2666 [0.2638, 0.2695] | -0.0423 [-0.0450, -0.0397] | -0.0477 [-0.0504, -0.0452] | -0.0315 [-0.0375, -0.0253] |
| drop context_session_popularity | per_impression | 0.3200 [0.3168, 0.3232] | +0.0111 [+0.0095, +0.0128] | +0.0009 [-0.0001, +0.0019] | +0.0037 [+0.0008, +0.0069] |
| drop article_metadata | per_impression | 0.3194 [0.3163, 0.3226] | +0.0105 [+0.0090, +0.0122] | +0.0003 [-0.0010, +0.0016] | +0.0025 [-0.0001, +0.0053] |
| full (no ablation) | per_impression | 0.3191 [0.3160, 0.3224] | +0.0102 [+0.0086, +0.0119] | +0.0000 [+0.0000, +0.0000] | +0.0021 [-0.0007, +0.0052] |
| drop freshness | per_impression | 0.3191 [0.3159, 0.3223] | +0.0102 [+0.0087, +0.0118] | -0.0000 [-0.0009, +0.0009] | +0.0015 [-0.0014, +0.0047] |
| drop onehot_categoricals | per_impression | 0.3172 [0.3141, 0.3204] | +0.0084 [+0.0067, +0.0100] | -0.0019 [-0.0034, -0.0003] | +0.0081 [+0.0049, +0.0119] |
| drop user_history | per_impression | 0.3164 [0.3132, 0.3196] | +0.0076 [+0.0059, +0.0093] | -0.0027 [-0.0039, -0.0014] | +0.0012 [-0.0017, +0.0043] |
| keep lightgbm_score + onehot_categoricals | per_impression | 0.3153 [0.3122, 0.3185] | +0.0064 [+0.0051, +0.0078] | -0.0038 [-0.0053, -0.0024] | -0.0047 [-0.0068, -0.0024] |
| keep lightgbm_score + user_history | per_impression | 0.3142 [0.3112, 0.3174] | +0.0054 [+0.0040, +0.0068] | -0.0049 [-0.0065, -0.0032] | +0.0032 [+0.0004, +0.0062] |
| keep lightgbm_score + article_metadata | per_impression | 0.3133 [0.3102, 0.3164] | +0.0044 [+0.0032, +0.0059] | -0.0058 [-0.0074, -0.0041] | +0.0032 [+0.0005, +0.0060] |
| keep lightgbm_score + context_session_popularity | per_impression | 0.3091 [0.3059, 0.3123] | +0.0002 [-0.0003, +0.0007] | -0.0100 [-0.0117, -0.0084] | +0.0012 [-0.0001, +0.0024] |
| keep only lightgbm_score | per_impression | 0.3089 [0.3058, 0.3120] | -0.0000 [-0.0000, +0.0000] | -0.0102 [-0.0119, -0.0086] | +0.0000 [+0.0000, +0.0000] |
| keep lightgbm_score + freshness | per_impression | 0.3089 [0.3057, 0.3121] | -0.0000 [-0.0010, +0.0010] | -0.0102 [-0.0120, -0.0086] | +0.0045 [+0.0025, +0.0066] |
| drop lightgbm_score | per_impression | 0.2813 [0.2784, 0.2843] | -0.0276 [-0.0301, -0.0249] | -0.0378 [-0.0402, -0.0353] | -0.0274 [-0.0334, -0.0215] |

### 6. Feature-group ablation — exponential-forgetting hybrid LinUCB (per-impression and hourly)

Group sizes (columns): lightgbm_score 2, freshness 5, article_metadata 6, user_history 29, context_session_popularity 18, onehot_categoricals 41. Every spec is re-tuned on days 0-2, frozen, run once on days 3-6.

| spec | mode | P@1 [CI] | Δ vs LightGBM [CI] | Δ vs full hybrid [CI] | final-20% Δ vs LightGBM |
|---|---|---|---|---|---|
| full (no ablation) | hourly | 0.3297 [0.3267, 0.3328] | +0.0208 [+0.0188, +0.0229] | +0.0000 [+0.0000, +0.0000] | +0.0211 [+0.0165, +0.0256] |
| drop context_session_popularity | hourly | 0.3289 [0.3257, 0.3320] | +0.0200 [+0.0182, +0.0218] | -0.0008 [-0.0022, +0.0004] | +0.0189 [+0.0147, +0.0232] |
| drop freshness | hourly | 0.3283 [0.3253, 0.3315] | +0.0195 [+0.0177, +0.0213] | -0.0014 [-0.0028, +0.0000] | +0.0194 [+0.0153, +0.0235] |
| drop user_history | hourly | 0.3281 [0.3250, 0.3312] | +0.0192 [+0.0172, +0.0213] | -0.0016 [-0.0031, -0.0002] | +0.0181 [+0.0132, +0.0226] |
| drop article_metadata | hourly | 0.3261 [0.3230, 0.3292] | +0.0172 [+0.0155, +0.0189] | -0.0036 [-0.0053, -0.0021] | +0.0160 [+0.0120, +0.0199] |
| keep lightgbm_score + onehot_categoricals | hourly | 0.3252 [0.3220, 0.3283] | +0.0163 [+0.0145, +0.0181] | -0.0045 [-0.0064, -0.0026] | +0.0170 [+0.0131, +0.0208] |
| drop onehot_categoricals | hourly | 0.3203 [0.3172, 0.3234] | +0.0114 [+0.0097, +0.0132] | -0.0094 [-0.0112, -0.0078] | +0.0114 [+0.0074, +0.0158] |
| keep lightgbm_score + user_history | hourly | 0.3170 [0.3138, 0.3202] | +0.0081 [+0.0066, +0.0096] | -0.0127 [-0.0147, -0.0108] | +0.0045 [+0.0006, +0.0082] |
| keep lightgbm_score + article_metadata | hourly | 0.3136 [0.3106, 0.3166] | +0.0047 [+0.0032, +0.0062] | -0.0161 [-0.0181, -0.0142] | +0.0042 [+0.0010, +0.0077] |
| keep only lightgbm_score | hourly | 0.3079 [0.3048, 0.3110] | -0.0010 [-0.0011, -0.0008] | -0.0218 [-0.0238, -0.0198] | -0.0023 [-0.0029, -0.0018] |
| keep lightgbm_score + freshness | hourly | 0.3074 [0.3042, 0.3105] | -0.0015 [-0.0026, -0.0004] | -0.0223 [-0.0243, -0.0203] | +0.0005 [-0.0019, +0.0029] |
| keep lightgbm_score + context_session_popularity | hourly | 0.3059 [0.3028, 0.3090] | -0.0030 [-0.0038, -0.0021] | -0.0238 [-0.0259, -0.0217] | -0.0055 [-0.0073, -0.0037] |
| drop lightgbm_score | hourly | 0.2539 [0.2511, 0.2568] | -0.0549 [-0.0579, -0.0521] | -0.0758 [-0.0784, -0.0731] | -0.0458 [-0.0522, -0.0397] |
| drop user_history | per_impression | 0.3592 [0.3562, 0.3622] | +0.0503 [+0.0481, +0.0525] | +0.0034 [+0.0016, +0.0051] | +0.0482 [+0.0433, +0.0533] |
| full (no ablation) | per_impression | 0.3558 [0.3527, 0.3590] | +0.0469 [+0.0448, +0.0492] | +0.0000 [+0.0000, +0.0000] | +0.0457 [+0.0408, +0.0507] |
| drop freshness | per_impression | 0.3540 [0.3510, 0.3572] | +0.0452 [+0.0430, +0.0474] | -0.0017 [-0.0033, -0.0002] | +0.0456 [+0.0406, +0.0506] |
| drop context_session_popularity | per_impression | 0.3537 [0.3507, 0.3569] | +0.0448 [+0.0427, +0.0471] | -0.0021 [-0.0035, -0.0007] | +0.0452 [+0.0402, +0.0502] |
| drop article_metadata | per_impression | 0.3468 [0.3440, 0.3499] | +0.0379 [+0.0359, +0.0401] | -0.0090 [-0.0108, -0.0072] | +0.0382 [+0.0333, +0.0434] |
| drop onehot_categoricals | per_impression | 0.3394 [0.3364, 0.3425] | +0.0305 [+0.0283, +0.0327] | -0.0164 [-0.0185, -0.0145] | +0.0317 [+0.0265, +0.0371] |
| keep lightgbm_score + onehot_categoricals | per_impression | 0.3382 [0.3351, 0.3412] | +0.0293 [+0.0274, +0.0313] | -0.0176 [-0.0197, -0.0155] | +0.0260 [+0.0219, +0.0302] |
| keep lightgbm_score + article_metadata | per_impression | 0.3259 [0.3228, 0.3290] | +0.0170 [+0.0151, +0.0190] | -0.0299 [-0.0321, -0.0277] | +0.0191 [+0.0148, +0.0235] |
| keep lightgbm_score + user_history | per_impression | 0.3224 [0.3194, 0.3256] | +0.0136 [+0.0118, +0.0154] | -0.0333 [-0.0356, -0.0311] | +0.0145 [+0.0105, +0.0188] |
| keep lightgbm_score + freshness | per_impression | 0.3112 [0.3081, 0.3143] | +0.0023 [+0.0011, +0.0035] | -0.0446 [-0.0469, -0.0424] | +0.0028 [+0.0000, +0.0056] |
| keep only lightgbm_score | per_impression | 0.3089 [0.3058, 0.3120] | +0.0000 [+0.0000, +0.0000] | -0.0469 [-0.0492, -0.0448] | +0.0000 [+0.0000, +0.0000] |
| keep lightgbm_score + context_session_popularity | per_impression | 0.3088 [0.3057, 0.3120] | -0.0001 [-0.0010, +0.0008] | -0.0470 [-0.0494, -0.0448] | -0.0006 [-0.0028, +0.0014] |
| drop lightgbm_score | per_impression | 0.3059 [0.3031, 0.3088] | -0.0030 [-0.0059, -0.0004] | -0.0499 [-0.0523, -0.0475] | +0.0018 [-0.0044, +0.0081] |

**Follow-up (forgetting hybrid):** LightGBM score + one-hots + article metadata only

| spec | mode | P@1 [CI] | Δ vs LightGBM [CI] | Δ vs full [CI] |
|---|---|---|---|---|
| keep lightgbm_score + onehot_categoricals + article_metadata | hourly | 0.3272 [0.3240, 0.3303] | +0.0183 [+0.0166, +0.0200] | -0.0025 [-0.0042, -0.0009] |
| keep lightgbm_score + onehot_categoricals + article_metadata | per_impression | 0.3517 [0.3486, 0.3548] | +0.0428 [+0.0407, +0.0449] | -0.0041 [-0.0060, -0.0023] |

### 7. Latency (ms per decision; Apple M3 / 8 GiB; 4,700 timed decisions; single run)

| policy | select+update p50 | p90 | p99 | select p50 | update p50 | decisions/s |
|---|---:|---:|---:|---:|---:|---:|
| linucb | 0.0663 | 0.3365 | 3.7423 | 0.0178 | 0.0453 | 3,926 |
| lints | 0.1207 | 0.3042 | 3.3138 | 0.0509 | 0.0415 | 3,219 |
| hybrid_linucb | 0.0601 | 0.1340 | 1.2379 | 0.0177 | 0.0416 | 7,357 |
| hybrid_lints | 0.1040 | 0.1799 | 1.4242 | 0.0627 | 0.0433 | 4,930 |
| hybrid_linucb_forget | 0.0677 | 0.1450 | 1.1491 | 0.0176 | 0.0497 | 6,698 |
| hybrid_linucb_window | 0.1016 | 0.2173 | 1.4813 | 0.0180 | 0.0833 | 5,122 |
| hybrid_linucb_reset | 0.0668 | 0.1336 | 1.0142 | 0.0190 | 0.0458 | 7,339 |
| hybrid_lints_forget | 0.1111 | 0.1973 | 2.0923 | 0.0561 | 0.0505 | 4,768 |
| lgbm_greedy | 0.3170 | 0.8112 | 1.6185 | nan | nan | 2,229 |


---

# Addendum 2 (2026-10-12): the fair baseline — LightGBM retrained with causal trailing-CTR features

Branch `bandit-fair-baseline`. This is the baseline the main text said it owed ("Alternatives / next steps", first bullet).
Same stream, tuning days 0-2, held-out days 3-6, user-bootstrap n = 2000. Candidate K's artifacts, `submissions/` and the
training pipeline are untouched; new models live in `experiments/trailing_ctr_gbdt_2026-10-12/` (git-ignored) and new
score arrays next to the ADR-017 stream.

## Verdict (measured)

| D (feedback delay) | retrained GBDT vs forgetting hybrid, overall days 3-6 | final 20 % | GBDT vs original ranker | hybrid vs original |
|---|---|---|---|---|
| 0 | **beats**: +0.0437 [+0.0410, +0.0464] | +0.0539 [+0.0486, +0.0595] | +0.0906 [+0.0880, +0.0932] | +0.0469 |
| 15 min | **beats**: +0.0393 [+0.0368, +0.0415] | +0.0384 [+0.0333, +0.0435] | +0.0698 [+0.0675, +0.0722] | +0.0305 |
| 30 min | **beats**: +0.0373 [+0.0349, +0.0397] | +0.0318 [+0.0264, +0.0370] | +0.0570 [+0.0548, +0.0591] | +0.0197 |
| 60 min | **beats**: +0.0348 [+0.0325, +0.0372] | +0.0255 [+0.0202, +0.0305] | +0.0397 [+0.0377, +0.0417] | +0.0049 [+0.0028, +0.0069] |

**The retrained GBDT beats the forgetting hybrid at every delay, overall and in the final 20 %, CI-clear in all eight cells.**
It does not "match" it at any D. The gap narrows with delay (0.0437 → 0.0348 overall) but never closes in this range.
Stream AUC (days 3-6): original 0.7560 [0.7540, 0.7579] → retrained 0.8167 [0.8149, 0.8185] at D = 0 (paired +0.0608 [+0.0597, +0.0618]),
0.8021 / 0.7933 at 15 / 30 min, **0.7815 [0.7795, 0.7834] at 60 min (paired +0.0255 [+0.0247, +0.0263])**.
On the full seven-day validation stream the AUCs are 0.8173 / 0.8028 / 0.7943 / 0.7827 vs 0.7588 for the original.

The non-learning popularity blend (re-tuned per D on days 0-2) sits between: +0.0326, +0.0268, +0.0229, +0.0172 over the original,
and it is **better than the forgetting hybrid at D ≥ 30 min** (+0.0032 [+0.0008, +0.0055] at 30 min, +0.0124 [+0.0103, +0.0146] at 60 min) and worse at
D = 0 (−0.0143) and 15 min (−0.0037).

## What was built

* `src/bandit/trailing.py` — for a row at time t, clicks and shows of the same article over `[t − D − W, t − D)`, W ∈ {1 h, 2 h}.
  Six features: smoothed CTR `(clicks + 5·0.1)/(shows + 5)` and impression count for each window, plus each CTR's rank inside the
  impression. The upper bound is **strict**: the brief said "≤ impression_time − D", but with D = 0 a non-strict bound admits the
  row's own label (and same-second siblings), a direct leak, so `<` is used (for D > 0 the two differ only for events at exactly
  t − D to the second). Shows and clicks are windowed identically, so a delayed click is never divided by an undelayed show count.
* `scripts/trailing_ctr_gbdt.py` — one model per D: Candidate K's `K_rank_nopos` recipe unchanged (lambdarank, identical
  `LGB_PARAMS`, 1,500 rounds, early stopping 75 on the **last 24 h of train only**, the same 63 leakage-safe columns — no
  lifetime aggregates, no position features) plus the six features. Trailing statistics run over the concatenated train + validation
  timeline so the first validation hours are seeded by the end of train, as a live system would be. Best iterations 950 / 513 / 426 / 462.
  The six features carry 37.9 % / 30.3 % / 24.8 % / 18.3 % of total gain at D = 0 / 15 / 30 / 60 min.
* `scripts/fair_baseline_compare.py` — per D, everything chosen on days 0-2 and frozen: the blend's (window, β); the forgetting hybrid's
  (γ, λ) **re-tuned under delay D** (12 configs; the runner releases a pick's reward `D` seconds later).

## Causality evidence (the gap is large, so it was checked three ways)

1. **Unit tests** (`tests/unit/test_trailing.py`, 12 cases): equality with a brute-force loop; for D ∈ {0, 15, 30, 60} min, flipping **every** label
   with `obs_time ≥ t − D` leaves the feature at t unchanged (60 random probes each); a row's own label and same-second siblings never enter at D = 0.
2. **End-to-end on the trained models** (`scripts/fair_baseline_audit.py`): permute every validation label from stream day 4 onward, recompute the
   features, re-score with the *same* model. For D = 0 / 15 / 60 min, 1.56 M / 1.56 M / 1.59 M rows with `t ≤ T0 + D` have a maximum absolute score change of **0.0**;
   92–99 % of later rows do change (so the check is not vacuous).
3. **Placebo**: building the trailing features from within-impression-permuted labels (all of train + validation) drops the same models to top-1
   0.3092 / 0.3113 / 0.3194 vs 0.4077 / 0.3859 / 0.3536 with real labels and 0.3168 for the original ranker — the gain comes from label information in earlier
   impressions and nothing else.

## Reading it correctly

* **Information asymmetry, in the GBDT's favour.** The retrained GBDT and the blend use the shows *and* clicks of **every** in-view candidate of every impression older
  than D; the bandit sees only the reward of its own picks (the "pick one arm" formulation of ADR-017). A production system logs all served impressions regardless
  of policy, so the GBDT's information is the realistic one; the bandit's restriction is a property of the formulation, not of reality. A fairer bandit would be handed the same
  full-information log — then it is just a recency-weighted regression, i.e. this GBDT.
* **D = 0 is an upper bound, not a deployable number.** AUC 0.8173 uses labels up to the previous second. It is **not comparable** to the 0.7542 leaderboard score or the
  0.7699 clean literature figure (blind test sets without in-stream labels). The D = 60 min row (AUC 0.7827 over all seven days) is the more conservative reading.
* **Revision of an ADR-018 statement.** The main text says the forgetting policy is "−0.0040 [−0.0061, −0.0017] worse than LightGBM-greedy at a 60-minute delay". That policy's
  hyper-parameters were frozen at D = 0. Re-tuned under the delay (γ = 0.9999, λ = 10) it is **+0.0049 [+0.0028, +0.0069]** over the original at 60 min (and +0.0305 / +0.0197 at 15 / 30 min vs +0.0267 / +0.0112
  before). The earlier row measured "a policy tuned for instant feedback under delay", not "the best bandit under delay"; this section supersedes it. The qualitative conclusion
  (the bandit's lead depends heavily on the delay) stands: +0.0469 → +0.0049.
* The hybrid's re-tuning grid is small (γ ∈ {0.998, 0.999, 0.9995, 0.9999} × λ ∈ {1, 10, 100}) and winners sit at γ = 0.9999 for D = 60 min (an edge); the 0.035 gap to the
  GBDT is much larger than anything a wider grid recovered in the main experiment, but a wider delay-aware grid was not run.

## Limitations (new)

One week of one small-sample dataset; the GBDT trains on a single week (05-18 to 05-25) with no hyper-parameter search (same fixed parameters as K); one seed per model;
trailing features need an in-stream label feed, which the blind-test setting of the leaderboards does not provide (so none of this changes the 0.7542 story); the six features are
article-level only — no user×article or category-level trailing statistics; **a bandit on top of the retrained GBDT (hybrid warm-started from `gbdt_plus_trailing_ctr`) was not run**, so
"does an online layer add anything once the ranker has real-time features" is open.

## Decision (revised, preserving the earlier text above)

ADR-018's condition "revisit if the GBDT-with-trailing-CTR baseline matches the forgetting bandit" is resolved in the other direction: it **beats** it at every D tested.
Revised recommendation: supply real-time popularity to the ranker as **features** (cheap, stateless at serving time beyond a rolling count store, +0.0397 P@1 over the original even at a
60-minute delay); do not ship the forgetting bandit as a substitute for those features. The only open bandit question is incremental: whether a forgetting layer on top of the
retrained ranker adds anything — to be tested before any online work. The earlier "upgrade the hybrid candidate to the forgetting variant" is withdrawn as a *production* recommendation
(it remains the best *bandit*).

## Verification and reproduce

* New tests: 12 (`tests/unit/test_trailing.py`). Full suite: see PROJECT_STATE.md.

```bash
poetry run python scripts/trailing_ctr_gbdt.py            # ~8 min: caches raw matrices, trains 4 models (D = 0, 15, 30, 60 min)
poetry run python scripts/fair_baseline_compare.py        # ~4 min: tuning on days 0-2, one held-out run
poetry run python scripts/fair_baseline_audit.py          # ~15 min: permuted-future-label invariance + placebo on the trained models
poetry run python scripts/fair_baseline_report.py         # renders the tables below
```

## Tables (rendered from `results/bandit_ebnerd_small/drift/fair_baseline*.json`)

### Retrained GBDT: training summary (all validation days; the models never saw validation)

| D | best iteration | stream AUC (all 7 days) | top-1 (all 7 days) | share of gain from the 6 trailing features |
|---|---:|---:|---:|---:|
| 0 min | 950 | 0.8173 | 0.4077 | 0.379 |
| 15 min | 513 | 0.8028 | 0.3859 | 0.303 |
| 30 min | 426 | 0.7943 | 0.3724 | 0.248 |
| 60 min | 462 | 0.7827 | 0.3536 | 0.183 |

Original ranker on the same stream: AUC 0.7588, top-1 0.3168.

### D = 0 min — held-out days 3-6 (n = 147,492); selections made on days 0-2

Blend chosen on days 0-2: 2h window, beta = 2.0. Forgetting hybrid re-tuned under this delay: gamma = 0.999, lambda = 1.

| policy | P@1 [CI] | Δ vs original [CI] | Δ vs forgetting hybrid [CI] | final-20% P@1 | final-20% Δ vs original | final-20% Δ vs hybrid |
|---|---|---|---|---|---|---|
| LightGBM-greedy (original) | 0.3089 [0.3058, 0.3120] | - | -0.0469 [-0.0492, -0.0448] | 0.3169 | - | -0.0457 [-0.0507, -0.0408] |
| LightGBM + trailing CTR (retrained) | 0.3995 [0.3961, 0.4029] | +0.0906 [+0.0880, +0.0932] | +0.0437 [+0.0410, +0.0464] | 0.4166 | +0.0997 [+0.0943, +0.1052] | +0.0539 [+0.0486, +0.0595] |
| popularity blend (no learning) | 0.3415 [0.3384, 0.3448] | +0.0326 [+0.0306, +0.0345] | -0.0143 [-0.0167, -0.0120] | 0.3402 | +0.0233 [+0.0196, +0.0270] | -0.0224 [-0.0273, -0.0176] |
| forgetting hybrid LinUCB | 0.3558 [0.3527, 0.3590] | +0.0469 [+0.0448, +0.0492] | - | 0.3626 | +0.0457 [+0.0408, +0.0507] | - |

Stream AUC on days 3-6: original 0.7560 [0.7540, 0.7579], retrained 0.8167 [0.8149, 0.8185], paired Δ +0.0608 [+0.0597, +0.0618]. Per-day P@1 (days 3-6): LightGBM-greedy 0.303/0.303/0.314/0.315; LightGBM + trailing CTR 0.390/0.390/0.404/0.414; popularity blend 0.337/0.332/0.352/0.344; forgetting hybrid LinUCB 0.348/0.353/0.360/0.361.

### D = 15 min — held-out days 3-6 (n = 147,492); selections made on days 0-2

Blend chosen on days 0-2: 2h window, beta = 2.0. Forgetting hybrid re-tuned under this delay: gamma = 0.9995, lambda = 10.

| policy | P@1 [CI] | Δ vs original [CI] | Δ vs forgetting hybrid [CI] | final-20% P@1 | final-20% Δ vs original | final-20% Δ vs hybrid |
|---|---|---|---|---|---|---|
| LightGBM-greedy (original) | 0.3089 [0.3058, 0.3120] | - | -0.0305 [-0.0327, -0.0285] | 0.3169 | - | -0.0314 [-0.0363, -0.0265] |
| LightGBM + trailing CTR (retrained) | 0.3786 [0.3754, 0.3820] | +0.0698 [+0.0675, +0.0722] | +0.0393 [+0.0368, +0.0415] | 0.3867 | +0.0698 [+0.0648, +0.0750] | +0.0384 [+0.0333, +0.0435] |
| popularity blend (no learning) | 0.3357 [0.3326, 0.3390] | +0.0268 [+0.0249, +0.0287] | -0.0037 [-0.0060, -0.0015] | 0.3367 | +0.0198 [+0.0159, +0.0236] | -0.0116 [-0.0166, -0.0064] |
| forgetting hybrid LinUCB | 0.3394 [0.3365, 0.3426] | +0.0305 [+0.0285, +0.0327] | - | 0.3483 | +0.0314 [+0.0265, +0.0363] | - |

Stream AUC on days 3-6: original 0.7560 [0.7540, 0.7579], retrained 0.8021 [0.8002, 0.8039], paired Δ +0.0461 [+0.0451, +0.0471]. Per-day P@1 (days 3-6): LightGBM-greedy 0.303/0.303/0.314/0.315; LightGBM + trailing CTR 0.373/0.371/0.385/0.385; popularity blend 0.332/0.327/0.345/0.339; forgetting hybrid LinUCB 0.326/0.341/0.342/0.347.

### D = 30 min — held-out days 3-6 (n = 147,492); selections made on days 0-2

Blend chosen on days 0-2: 2h window, beta = 2.0. Forgetting hybrid re-tuned under this delay: gamma = 0.9995, lambda = 10.

| policy | P@1 [CI] | Δ vs original [CI] | Δ vs forgetting hybrid [CI] | final-20% P@1 | final-20% Δ vs original | final-20% Δ vs hybrid |
|---|---|---|---|---|---|---|
| LightGBM-greedy (original) | 0.3089 [0.3058, 0.3120] | - | -0.0197 [-0.0219, -0.0175] | 0.3169 | - | -0.0249 [-0.0300, -0.0200] |
| LightGBM + trailing CTR (retrained) | 0.3659 [0.3627, 0.3691] | +0.0570 [+0.0548, +0.0591] | +0.0373 [+0.0349, +0.0397] | 0.3736 | +0.0567 [+0.0520, +0.0615] | +0.0318 [+0.0264, +0.0370] |
| popularity blend (no learning) | 0.3317 [0.3286, 0.3351] | +0.0229 [+0.0209, +0.0247] | +0.0032 [+0.0008, +0.0055] | 0.3343 | +0.0174 [+0.0136, +0.0212] | -0.0075 [-0.0124, -0.0027] |
| forgetting hybrid LinUCB | 0.3286 [0.3255, 0.3316] | +0.0197 [+0.0175, +0.0219] | - | 0.3418 | +0.0249 [+0.0200, +0.0300] | - |

Stream AUC on days 3-6: original 0.7560 [0.7540, 0.7579], retrained 0.7933 [0.7914, 0.7951], paired Δ +0.0374 [+0.0364, +0.0383]. Per-day P@1 (days 3-6): LightGBM-greedy 0.303/0.303/0.314/0.315; LightGBM + trailing CTR 0.360/0.362/0.371/0.370; popularity blend 0.329/0.323/0.340/0.334; forgetting hybrid LinUCB 0.312/0.328/0.332/0.341.

### D = 60 min — held-out days 3-6 (n = 147,492); selections made on days 0-2

Blend chosen on days 0-2: 2h window, beta = 1.0. Forgetting hybrid re-tuned under this delay: gamma = 0.9999, lambda = 10.

| policy | P@1 [CI] | Δ vs original [CI] | Δ vs forgetting hybrid [CI] | final-20% P@1 | final-20% Δ vs original | final-20% Δ vs hybrid |
|---|---|---|---|---|---|---|
| LightGBM-greedy (original) | 0.3089 [0.3058, 0.3120] | - | -0.0049 [-0.0069, -0.0028] | 0.3169 | - | -0.0132 [-0.0176, -0.0087] |
| LightGBM + trailing CTR (retrained) | 0.3485 [0.3453, 0.3519] | +0.0397 [+0.0377, +0.0417] | +0.0348 [+0.0325, +0.0372] | 0.3556 | +0.0387 [+0.0345, +0.0432] | +0.0255 [+0.0202, +0.0305] |
| popularity blend (no learning) | 0.3261 [0.3231, 0.3294] | +0.0172 [+0.0158, +0.0186] | +0.0124 [+0.0103, +0.0146] | 0.3299 | +0.0130 [+0.0103, +0.0158] | -0.0002 [-0.0049, +0.0044] |
| forgetting hybrid LinUCB | 0.3137 [0.3105, 0.3168] | +0.0049 [+0.0028, +0.0069] | - | 0.3301 | +0.0132 [+0.0087, +0.0176] | - |

Stream AUC on days 3-6: original 0.7560 [0.7540, 0.7579], retrained 0.7815 [0.7795, 0.7834], paired Δ +0.0255 [+0.0247, +0.0263]. Per-day P@1 (days 3-6): LightGBM-greedy 0.303/0.303/0.314/0.315; LightGBM + trailing CTR 0.344/0.340/0.357/0.352; popularity blend 0.322/0.317/0.335/0.329; forgetting hybrid LinUCB 0.289/0.321/0.315/0.328.

### Verdict per D (retrained GBDT vs forgetting hybrid, paired, overall / final 20%)

| D | overall | final 20% |
|---|---|---|
| 0 min | **beats** the hybrid (+0.0437 [+0.0410, +0.0464]) | **beats** the hybrid (+0.0539 [+0.0486, +0.0595]) |
| 15 min | **beats** the hybrid (+0.0393 [+0.0368, +0.0415]) | **beats** the hybrid (+0.0384 [+0.0333, +0.0435]) |
| 30 min | **beats** the hybrid (+0.0373 [+0.0349, +0.0397]) | **beats** the hybrid (+0.0318 [+0.0264, +0.0370]) |
| 60 min | **beats** the hybrid (+0.0348 [+0.0325, +0.0372]) | **beats** the hybrid (+0.0255 [+0.0202, +0.0305]) |

### Causality audit of the trained models (end to end)

| D | rows with t ≤ T0+D checked | max \|score change\| on them | later rows whose score changed | placebo top-1 (permuted-label features) | real top-1 | original ranker top-1 |
|---|---:|---:|---:|---:|---:|---:|
| D=0 | 1,555,133 | 0.0e+00 | 0.994 | 0.3092 | 0.4077 | 0.3168 |
| D=900 | 1,564,487 | 0.0e+00 | 0.989 | 0.3113 | 0.3859 | 0.3168 |
| D=3600 | 1,588,275 | 0.0e+00 | 0.917 | 0.3194 | 0.3536 | 0.3168 |
