# ADR-017 — Offline contextual-bandit layer over the EB-NeRD LightGBM ranker

**Date:** 2026-10-12
**Status:** Decided (pre-registration written before any bandit result; results appended below, 2026-10-12)
**Severity:** Medium (experiment only; nothing deployed changes)

> **Numbering note.** The task asked for `docs/ADR-016_bandit_experiment.md`.
> `decisions/ADR-016-mind-click-history-features.md` already exists and the project's ADRs live
> in `decisions/`, so this record is **ADR-017**, in `decisions/`. Nothing else was renamed.

---

# Engineering Question

Candidate K (LightGBM lambdarank, ADR-013) is a static ranker: trained once, scored forever.
Does an online-learning layer on top of it — a contextual bandit that learns from the one
click/no-click it observes per decision — improve top-1 click-through, and what does the
exploration cost?

**Hard constraints.** The 0.7542 leaderboard pipeline, `submissions/`, and every existing
artifact are untouched. This package only *reads* K's frozen boosters.

# What is actually being replayed (Phase 0 findings, measured)

| Fact | Value | Source |
|---|---|---|
| Data in use | `ebnerd_small` validation window: **244,647 impressions, 15,342 users, 2.93 M candidate rows** | `scripts/bandit_prepare_stream.py` |
| The "0.7542" model | Trained on Ada on `ebnerd_large`; **its weights are not available locally** (PROJECT_STATE, benchmarks note). The bandit layer therefore sits on the **locally retrained `ebnerd_small` Candidate K (`K_rank_nopos`, 63 features, validation AUC 0.7588)**, reproduced to 6 decimals on the chronological stream (0.758808). Conclusions transfer to "the same recipe at small scale", not to the Ada model byte-for-byte. | ADR-013, `experiments/candidate_k_gbdt_ebnerd_small_2026-09-12/` |
| Stream | 2023-05-25 07:00:02 → 2023-06-01 06:59:59, 7 × 24 h days (impressions/day 32.6k, 33.4k, 31.2k, 35.0k, 36.9k, 38.3k, 37.3k). The raw parquet is **not** time-sorted; the stream is sorted by `(impression_time, impression_id)`. | measured |
| Train window (K's fit data) | 2023-05-18 → 2023-05-25 07:00 (232,887 impressions). Early stopping used its last 24 h; validation was never used for any K decision. | ADR-013 |
| Candidates | `article_ids_inview` (mean 12.0, median 9, min 5, max 99); `article_ids_clicked` has 1 click in 99.4 % of impressions, 2–5 in 0.6 %. **Every impression has ≥ 1 click** (the logs only contain impressions with a click-through). | measured |
| Is in-view order the display position? | **No — not recoverable.** P(clicked item is at position j \| list size n) is flat (n=5: 0.199/0.202/0.196/0.203/0.199; n=12: 0.079–0.088 across positions) and only 0.14 % of lists are sorted by article id. The order is effectively shuffled. | measured |
| Consequence for Step 12 (SNIPS) | A position-based propensity model has nothing to model: position carries no click signal. See "SNIPS skipped" below. | |
| Side-finding | ADR-013 cites clicked-vs-non-clicked mean position 5.12 vs 7.92 as "far too strong to absorb silently". Within a list the click position is uniform, so that gap is a **list-size composition effect** (non-clicked rows come disproportionately from long lists), not a position effect. It is consistent with K_rank vs K_rank_nopos differing by only −0.0007 AUC. ADR-013 is not edited (history preserved); this note is the correction. | measured |

**Feature columns (65, `FEATURE_NAMES` order).** Long-term (6): `lt_cat_affinity, lt_subcat_affinity,
lt_topic_affinity, lt_embed_sim, lt_cat_rank, lt_sentiment_distance`. Short-term (7): `st_cat_affinity,
st_subcat_affinity, st_topic_affinity, st_embed_sim, st_cat_rank, is_last_clicked_category,
in_recent_categories`. Drift (3): `cat_affinity_drift, embed_sim_drift, topic_affinity_drift`. User (9):
`log_history_len, user_mean_read_time, user_median_read_time, user_mean_scroll, user_distinct_categories,
user_clicks_last_24h, user_mean_article_age_h, hours_since_last_click, user_history_span_h`. Item (14):
`article_age_h, is_fresh_6h, sentiment_score, sentiment_label, article_type, is_premium, title_len, body_len,
n_topics, n_entities, log_history_popularity, log_history_popularity_decayed, category_code,
age_vs_user_appetite`. Context (14): `n_candidates, hour_of_day, day_of_week, is_weekend, is_front_page,
device_type, is_sso_user, is_subscriber, user_age, gender, context_read_time, context_scroll_percentage,
position_in_view, relative_position_in_view`. Context-article (3): `context_category_match,
context_topic_overlap, context_embed_sim`. Session (2): `session_position, session_start_gap_h`. Relative (7):
`lt_cat_affinity_rank_in_imp, st_cat_affinity_rank_in_imp, lt_embed_sim_rank_in_imp, article_age_rank_in_imp,
history_popularity_rank_in_imp, article_age_minus_imp_min, lt_embed_sim_minus_imp_max`.

# Formulation: supervised → bandit conversion

* **Round t** = one impression, in chronological order. **Context** = the user (history-derived
  features), the request (time, device, session, context article) and the in-view candidate set.
* **Arms** = the in-view candidates of that impression (5–99). The policy picks **one**.
* **Reward** = 1 if the picked article is in `article_ids_clicked`, else 0. The reward of every
  *other* candidate is never revealed to the policy (enforced structurally: `BanditEnvironment`
  exposes only `pull(i, k)`; a test pins that exactly one reward is revealed per round).
* **Oracle** = picks a clicked article when one exists (always, here) → reward 1 every round, so
  cumulative regret = N − hits and **CTR = 1 − mean regret** identically. Both are reported
  because the task asks for both, but they carry the same information.

**Interpretation of "CTR" (important).** Because each impression has ≥ 1 click, this "CTR" is the
probability that a single top-1 pick is *the* clicked article — a top-1 precision, not a
click-through rate over all impressions (random = mean 1/n = **0.1170**). It is the right quantity
for "did we put the article the user chose first", but it must not be compared with a site CTR.

**Exposure-bias caveat.** Labels exist only for articles that the production system chose to put
in view, and that logging policy was **not random**: the in-view set is the output of Ekstra
Bladet's own selection, and impressions without any click are absent. The replay therefore
measures "which of the shown articles is clicked", never the value of articles that were not shown.
Any policy can only choose among already-selected articles, which compresses every gap.

# Protocol

* **Prequential**: for t = 1..N: `select` (current state) → `pull` (reveal one reward) → `update`.
  `update_interval_s=None` applies the update immediately; `3600` buffers observations and applies
  them at the next clock-hour boundary (a production-like refresh). Both are reported.
  An update contains only rewards of impressions strictly earlier than the decision it precedes
  (tests: `test_no_future_leakage_*`).
* **Periods (fixed before looking at any bandit result).** Stream days 0–2 (97,155 impressions,
  05-25 07:00 → 05-28 07:00) are the **tuning period**; days 3–6 (147,492 impressions,
  05-28 07:00 → 06-01 07:00) are the **held-out period**. All hyper-parameters are selected on
  the tuning period only (one run per configuration), then frozen. The held-out run starts every
  policy from a **fresh state** (no carry-over of tuning-period learning), so the held-out numbers
  are a clean cold-start-per-policy measurement with no tuning leakage.
* **Selection rule**: highest tuning-period CTR; the full grid is reported. Stochastic policies
  use tuning seed 1000; held-out seeds are 0–4 (disjoint).
* **Resampling unit**: **users** (15,342; 2,000 replicates; `bootstrap_ci` from
  `src/evaluation/bootstrap.py`). Day-block bootstrap was rejected: the held-out period has four
  days, too few blocks for a percentile interval. Seed-averaged rewards are used for stochastic
  policies, so the CI reflects *data* variance; the seed-to-seed std is reported separately.
* **Slices (pre-registered).** Cold article = the clicked article has **< 100** prior in-view
  appearances before the impression (train window + earlier stream). `corpus_stats.md` has no
  per-article exposure statistic; N = 100 is chosen as just under the corpus mean exposure per
  article (2.59 M train candidate rows / 20,738 articles ≈ 125). Cold user = history < 5 —
  **empty**: the minimum history length in the validation window is 5, so the pre-registered
  slice has 0 impressions; the substitute is **history < 20** (≈ 5 % of impressions), labelled as
  a substitution. Day slices = first vs last 24 h of the held-out period. Candidate-set size =
  ≤ median (9) vs > median.

# Policies and why a shared parameter vector

1. Random. 2. Popularity-greedy (trailing-window CTR from the policy's *own* observed pulls — the
   only CTR evidence bandit feedback permits). 3. LightGBM-greedy (frozen `K_rank_nopos`).
4. ε-greedy over the LightGBM score. 5. LinUCB, shared θ. 6. Linear Thompson sampling, shared θ.
7. Hybrid: LinUCB / Thompson on the same features **plus** the LightGBM calibrated softmax
   probability and centred score, with the posterior mean initialised at `θ₀ = w·e_prob`
   (`b = λθ₀`), so before any data the hybrid ranks exactly like LightGBM-greedy and learns a
   residual. `w` ∈ {0, 1} is a tuned hyper-parameter (w = 0 is the no-warm-start ablation).

**Why shared θ, not disjoint per-arm LinUCB.** The arm set is the article catalogue, and articles
are born and die within days (`article_age_h` is the top feature by gain, 464 k vs 220 k for the
next). Disjoint LinUCB keeps a d×d matrix per article, so each article starts with zero data and
the cold-start problem becomes the *only* problem; memory is also O(#articles·d²) (4.7 k shown
articles × 100² × 8 B ≈ 380 MB on `small`, tens of GB at `ebnerd_large`'s catalogue). A shared θ
over joint (user, article, context) features scores a never-seen article from its metadata on
first sight. Cost: a *linear* reward model in d = 100 features cannot represent the interactions
the tree ensemble uses, and user-constant features (device, subscriber, …) cannot influence which
candidate wins under a shared θ (they cancel in the argmax) — the hybrid exists to recover the
ranker's non-linear signal.

**Update cost.** Sherman–Morrison keeps A⁻¹ per observation in O(d²) instead of O(d³) for re-inversion;
`A` is kept in parallel, A⁻¹ is symmetrised on every update and re-derived by direct inversion every
20 000 updates to cancel float drift (test: SM inverse equals `np.linalg.inv` to 1e-9).

**Linear feature vector (d₀ = 100).** The 63 leakage-safe columns minus the 5 user-constant /
position columns the linear model cannot use, i.e. 58 standardised numerics (signed-log if skew > 2,
clip to P1–P99, z-score, NaN → 0) + one-hot `category_code` (26), `article_type` (12),
`sentiment_label` (3) + bias. Scalers are fitted on the **last 24 h of the train window** only.
`position_in_view` / `relative_position_in_view` are withheld, matching `K_rank_nopos`, so
bandit-vs-LightGBM differs in policy, not in information. `d₀ = 100`, not the task's "≈ 65": one-hot
expansion of three categorical item attributes is what a linear model needs; update cost stays
~10⁴ flops.

**Leakage guard.** `assert_leakage_safe` (called at transformer fit and in the stream builder)
fails if any source field is in `FORBIDDEN_RAW_COLUMNS` (`total_inviews`, `total_pageviews`,
`total_read_time`, `next_read_time`, `next_scroll_percentage`) or is not in the audited
`FEATURE_NAMES`. Tested.

# Results

All numbers below are produced by `scripts/run_bandit_experiment.py` (+ `bandit_make_report.py`,
which only renders JSON into tables) and live in `results/bandit_ebnerd_small/`. Reproduce:

```bash
poetry run python scripts/bandit_prepare_stream.py            # ~110 s; K's frozen boosters -> stream arrays
poetry run python scripts/run_bandit_experiment.py --dry-run  # always first
poetry run python scripts/run_bandit_experiment.py --stage tune     # ~8 min, 7 workers
poetry run python scripts/run_bandit_experiment.py --stage final    # held-out, run once
poetry run python scripts/run_bandit_experiment.py --stage sim
poetry run python scripts/bandit_make_report.py
```

**Protocol deviations / events, in order, all disclosed:**
1. The first stream build had a time-unit bug (datetime64[us] read as ns → epoch seconds off by 1000×)
   that would have silently broken hourly batching and the day slices. Caught before any bandit run
   (the stream's date range printed as 1970); fixed and regenerated.
2. The first tuning pass put five winners on its grid's upper edge (λ=100, v=0.1, m=20, ε=0.01). The
   grid was widened once (tuning period only; λ→1000, v→0.3, m→100, ε→0.003). Winners did not change
   except ε (0.01→0.003, the floor: ε→0 *is* LightGBM-greedy). First-pass winners are kept in
   `tune_first_pass.json`.
3. Because the task says to audit anything that looks too good, three leakage/provenance audits were
   run on the **tuning period only** before the held-out run (below). The held-out set was run once.

## Primary results (held-out, log-replay)

### Tuned hyper-parameters (selected on the tuning period; frozen)

| policy (update mode) | frozen hyper-parameters | tuning-period CTR |
|---|---|---:|
| popularity_greedy (per-impression) | `{"window_h": 24, "m": 5}` | 0.2408 |
| popularity_greedy (hourly) | `{"window_h": 6, "m": 20}` | 0.1860 |
| eps_greedy_lgbm | `{"epsilon": 0.003}` | 0.3283 |
| linucb (per-impression) | `{"alpha": 0.0, "lam": 100}` | 0.3004 |
| linucb (hourly) | `{"alpha": 0.03, "lam": 10}` | 0.2872 |
| lints (per-impression) | `{"v": 0.003, "lam": 10}` | 0.3001 |
| lints (hourly) | `{"v": 0.03, "lam": 10}` | 0.2877 |
| hybrid_linucb (per-impression) | `{"alpha": 0.003, "lam": 100, "w": 0.0}` | 0.3478 |
| hybrid_linucb (hourly) | `{"alpha": 0.0, "lam": 100, "w": 1.0}` | 0.3420 |
| hybrid_lints (per-impression) | `{"v": 0.1, "lam": 100, "w": 0.0}` | 0.3472 |
| hybrid_lints (hourly) | `{"v": 0.1, "lam": 100, "w": 1.0}` | 0.3421 |

References on the tuning period: random 0.1194, LightGBM-greedy 0.3289.

### Held-out primary results (days 3-6; 147492 impressions, 13261 users; user-bootstrap 95% CI, n=2000)

| policy | top-1 CTR | mean regret / imp | cumulative regret | final-window (last 20%) CTR | paired Δ CTR vs LightGBM-greedy | verdict |
|---|---|---|---:|---|---|---|
| random | 0.1157 [0.1148, 0.1166] | 0.8843 | 130,426 | 0.1173 [0.1156, 0.1191] | -0.1932 [-0.1964, -0.1899] | LOSS |
| popularity_greedy (per-impression) | 0.2303 [0.2283, 0.2324] | 0.7697 | 113,521 | 0.2316 [0.2275, 0.2358] | -0.0785 [-0.0814, -0.0756] | LOSS |
| popularity_greedy (hourly) | 0.1808 [0.1791, 0.1825] | 0.8192 | 120,818 | 0.1943 [0.1906, 0.1979] | -0.1280 [-0.1309, -0.1250] | LOSS |
| lgbm_greedy | 0.3089 [0.3058, 0.3120] | 0.6911 | 101,936 | 0.3169 [0.3106, 0.3232] | - | reference |
| eps_greedy_lgbm | 0.3084 [0.3053, 0.3115] | 0.6916 | 102,009 | 0.3163 [0.3100, 0.3226] | -0.0005 [-0.0006, -0.0004] | LOSS |
| linucb (per-impression) | 0.2813 [0.2784, 0.2843] | 0.7187 | 106,002 | 0.2895 [0.2833, 0.2959] | -0.0276 [-0.0301, -0.0249] | LOSS |
| linucb (hourly) | 0.2662 [0.2635, 0.2692] | 0.7338 | 108,223 | 0.2829 [0.2768, 0.2892] | -0.0426 [-0.0453, -0.0400] | LOSS |
| lints (per-impression) | 0.2810 [0.2781, 0.2839] | 0.7190 | 106,053 | 0.2894 [0.2833, 0.2959] | -0.0279 [-0.0304, -0.0254] | LOSS |
| lints (hourly) | 0.2683 [0.2655, 0.2712] | 0.7317 | 107,920 | 0.2836 [0.2776, 0.2898] | -0.0406 [-0.0430, -0.0380] | LOSS |
| hybrid_linucb (per-impression) | 0.3195 [0.3164, 0.3227] | 0.6805 | 100,370 | 0.3193 [0.3131, 0.3258] | +0.0106 [+0.0090, +0.0123] | win |
| hybrid_linucb (hourly) | 0.3143 [0.3110, 0.3175] | 0.6857 | 101,139 | 0.3171 [0.3109, 0.3235] | +0.0054 [+0.0038, +0.0071] | win |
| hybrid_lints (per-impression) | 0.3195 [0.3164, 0.3227] | 0.6805 | 100,372 | 0.3191 [0.3129, 0.3256] | +0.0106 [+0.0091, +0.0123] | win |
| hybrid_lints (hourly) | 0.3146 [0.3115, 0.3178] | 0.6854 | 101,085 | 0.3176 [0.3114, 0.3239] | +0.0058 [+0.0043, +0.0074] | win |

Seed-to-seed std of CTR (5 seeds, stochastic policies): random 0.0009, popularity_greedy (per-impression) 0.0017, popularity_greedy (hourly) 0.0038, eps_greedy_lgbm 0.0000, lints (per-impression) 0.0004, lints (hourly) 0.0008, hybrid_lints (per-impression) 0.0003, hybrid_lints (hourly) 0.0001.

### Learning check: CTR by held-out day, and paired Δ vs LightGBM-greedy in the last 20%

| policy | day 3 | day 4 | day 5 | day 6 | Δ vs LightGBM, final window |
|---|---:|---:|---:|---:|---|
| random | 0.1134 | 0.1146 | 0.1165 | 0.1181 | -0.1996 [-0.2062, -0.1930] |
| popularity_greedy (per-impression) | 0.2225 | 0.2302 | 0.2386 | 0.2293 | -0.0853 [-0.0918, -0.0791] |
| popularity_greedy (hourly) | 0.1691 | 0.1770 | 0.1870 | 0.1894 | -0.1226 [-0.1290, -0.1162] |
| lgbm_greedy | 0.3028 | 0.3029 | 0.3140 | 0.3152 | - |
| eps_greedy_lgbm | 0.3023 | 0.3024 | 0.3135 | 0.3147 | -0.0006 [-0.0008, -0.0004] |
| linucb (per-impression) | 0.2710 | 0.2777 | 0.2879 | 0.2878 | -0.0274 [-0.0334, -0.0215] |
| linucb (hourly) | 0.2391 | 0.2584 | 0.2817 | 0.2837 | -0.0340 [-0.0399, -0.0280] |
| lints (per-impression) | 0.2716 | 0.2756 | 0.2883 | 0.2876 | -0.0275 [-0.0335, -0.0216] |
| lints (hourly) | 0.2452 | 0.2622 | 0.2813 | 0.2828 | -0.0333 [-0.0392, -0.0275] |
| hybrid_linucb (per-impression) | 0.3184 | 0.3198 | 0.3203 | 0.3193 | +0.0024 [-0.0004, +0.0056] |
| hybrid_linucb (hourly) | 0.3048 | 0.3149 | 0.3199 | 0.3167 | +0.0002 [-0.0027, +0.0034] |
| hybrid_lints (per-impression) | 0.3183 | 0.3199 | 0.3205 | 0.3190 | +0.0022 [-0.0007, +0.0053] |
| hybrid_lints (hourly) | 0.3056 | 0.3153 | 0.3195 | 0.3175 | +0.0007 [-0.0020, +0.0037] |

### Slices (CTR [CI]; Δ vs LightGBM-greedy [CI]) for the policies that matter

| slice (n impressions) | lgbm_greedy | linucb (per-impression) | hybrid_linucb (per-impression) | hybrid_linucb (hourly) | hybrid_lints (hourly) |
|---|---|---|---|---|---|
| cold_article (clicked article <100 prior exposures) (n=3,892) | 0.2109 [0.1975, 0.2242] | 0.1973 [0.1841, 0.2101]<br>Δ -0.0136 [-0.0274, -0.0008] LOSS | 0.2258 [0.2118, 0.2397]<br>Δ +0.0149 [+0.0064, +0.0240] win | 0.2194 [0.2056, 0.2333]<br>Δ +0.0085 [+0.0008, +0.0167] win | 0.2190 [0.2052, 0.2323]<br>Δ +0.0080 [+0.0006, +0.0160] win |
| warm_article (>= 100) (n=143,600) | 0.3115 [0.3084, 0.3145] | 0.2836 [0.2804, 0.2868]<br>Δ -0.0279 [-0.0307, -0.0253] LOSS | 0.3220 [0.3188, 0.3252]<br>Δ +0.0105 [+0.0088, +0.0123] win | 0.3168 [0.3137, 0.3201]<br>Δ +0.0053 [+0.0038, +0.0070] win | 0.3172 [0.3141, 0.3205]<br>Δ +0.0057 [+0.0041, +0.0074] win |
| cold_user (history < 5) (n=0) | empty | empty | empty | empty | empty |
| low_history_user (history < 20) (n=7,288) | 0.2859 [0.2749, 0.2967] | 0.2409 [0.2303, 0.2520]<br>Δ -0.0450 [-0.0555, -0.0343] LOSS | 0.2961 [0.2849, 0.3081]<br>Δ +0.0102 [+0.0022, +0.0178] win | 0.2917 [0.2803, 0.3035]<br>Δ +0.0058 [-0.0016, +0.0132] n.s. | 0.2914 [0.2801, 0.3033]<br>Δ +0.0055 [-0.0014, +0.0128] n.s. |
| higher_history_user (history >= 20) (n=140,204) | 0.3101 [0.3070, 0.3133] | 0.2834 [0.2801, 0.2865]<br>Δ -0.0267 [-0.0294, -0.0241] LOSS | 0.3207 [0.3173, 0.3240]<br>Δ +0.0106 [+0.0090, +0.0124] win | 0.3154 [0.3120, 0.3186]<br>Δ +0.0054 [+0.0038, +0.0070] win | 0.3158 [0.3124, 0.3190]<br>Δ +0.0058 [+0.0042, +0.0074] win |
| first_day (n=35,041) | 0.3028 [0.2975, 0.3083] | 0.2710 [0.2662, 0.2762]<br>Δ -0.0318 [-0.0377, -0.0262] LOSS | 0.3184 [0.3131, 0.3239]<br>Δ +0.0156 [+0.0117, +0.0197] win | 0.3048 [0.2994, 0.3101]<br>Δ +0.0020 [-0.0017, +0.0054] n.s. | 0.3056 [0.3002, 0.3108]<br>Δ +0.0027 [-0.0007, +0.0060] n.s. |
| last_day (n=37,257) | 0.3152 [0.3098, 0.3207] | 0.2878 [0.2821, 0.2936]<br>Δ -0.0274 [-0.0323, -0.0227] LOSS | 0.3193 [0.3138, 0.3247]<br>Δ +0.0041 [+0.0012, +0.0069] win | 0.3167 [0.3111, 0.3221]<br>Δ +0.0016 [-0.0013, +0.0044] n.s. | 0.3175 [0.3119, 0.3228]<br>Δ +0.0023 [-0.0004, +0.0051] n.s. |
| small_candidate_set (<= 9) (n=81,258) | 0.3722 [0.3683, 0.3762] | 0.3537 [0.3496, 0.3578]<br>Δ -0.0186 [-0.0223, -0.0150] LOSS | 0.3786 [0.3747, 0.3827]<br>Δ +0.0064 [+0.0040, +0.0088] win | 0.3742 [0.3703, 0.3781]<br>Δ +0.0020 [-0.0002, +0.0042] n.s. | 0.3744 [0.3704, 0.3783]<br>Δ +0.0022 [-0.0001, +0.0043] n.s. |
| large_candidate_set (> 9) (n=66,234) | 0.2311 [0.2274, 0.2351] | 0.1925 [0.1891, 0.1960]<br>Δ -0.0386 [-0.0427, -0.0349] LOSS | 0.2470 [0.2430, 0.2507]<br>Δ +0.0158 [+0.0133, +0.0183] win | 0.2407 [0.2366, 0.2446]<br>Δ +0.0096 [+0.0070, +0.0120] win | 0.2413 [0.2373, 0.2451]<br>Δ +0.0102 [+0.0077, +0.0125] win |

### Exploration cost: where the pick differs from the LightGBM top-1

| policy | picks differing from LightGBM top-1 | CTR of the differing picks | CTR of LightGBM's pick, same impressions | paired difference | verdict |
|---|---:|---|---|---|---|
| random | 88.4% | 0.0856 [0.0847, 0.0866] | 0.3041 [0.3010, 0.3073] | -0.2185 [-0.2221, -0.2148] | LOSS |
| popularity_greedy (per-impression) | 80.0% | 0.1769 [0.1751, 0.1788] | 0.2751 [0.2723, 0.2779] | -0.0982 [-0.1019, -0.0944] | LOSS |
| popularity_greedy (hourly) | 84.5% | 0.1369 [0.1355, 0.1384] | 0.2884 [0.2855, 0.2914] | -0.1515 [-0.1548, -0.1480] | LOSS |
| eps_greedy_lgbm | 0.3% | 0.0944 [0.0815, 0.1067] | 0.2826 [0.2629, 0.3030] | -0.1882 [-0.2156, -0.1630] | LOSS |
| linucb (per-impression) | 59.0% | 0.1910 [0.1883, 0.1938] | 0.2377 [0.2348, 0.2408] | -0.0467 [-0.0513, -0.0424] | LOSS |
| linucb (hourly) | 61.0% | 0.1761 [0.1734, 0.1790] | 0.2461 [0.2430, 0.2490] | -0.0699 [-0.0742, -0.0653] | LOSS |
| lints (per-impression) | 59.2% | 0.1912 [0.1886, 0.1938] | 0.2384 [0.2355, 0.2413] | -0.0472 [-0.0514, -0.0430] | LOSS |
| lints (hourly) | 60.7% | 0.1786 [0.1762, 0.1811] | 0.2455 [0.2426, 0.2485] | -0.0669 [-0.0708, -0.0626] | LOSS |
| hybrid_linucb (per-impression) | 25.7% | 0.2367 [0.2320, 0.2409] | 0.1953 [0.1913, 0.1998] | +0.0413 [+0.0341, +0.0478] | win |
| hybrid_linucb (hourly) | 24.1% | 0.2197 [0.2153, 0.2240] | 0.1973 [0.1931, 0.2017] | +0.0224 [+0.0155, +0.0291] | win |
| hybrid_lints (per-impression) | 25.7% | 0.2370 [0.2328, 0.2411] | 0.1958 [0.1918, 0.1996] | +0.0412 [+0.0350, +0.0478] | win |
| hybrid_lints (hourly) | 24.2% | 0.2197 [0.2156, 0.2238] | 0.1958 [0.1918, 0.1998] | +0.0239 [+0.0174, +0.0307] | win |

### SIMULATED click model (K_cls probabilities normalised per impression; Bernoulli rewards; 5 seeds; held-out stream) -- NOT log-replay

Simulator oracle expected CTR (always pick the max-probability arm): 0.3347.

| policy | cumulative expected regret (mean ± std) | simulated CTR (mean ± std) |
|---|---:|---:|
| random | 32,319 ± 40 | 0.1155 ± 0.0005 |
| popularity_greedy (per-impression) | 18,506 ± 594 | 0.2089 ± 0.0050 |
| popularity_greedy (hourly) | 23,024 ± 325 | 0.1786 ± 0.0029 |
| lgbm_greedy | 2,346 ± 0 | 0.3181 ± 0.0009 |
| eps_greedy_lgbm | 2,432 ± 4 | 0.3175 ± 0.0009 |
| linucb (per-impression) | 9,833 ± 320 | 0.2674 ± 0.0020 |
| linucb (hourly) | 10,669 ± 220 | 0.2617 ± 0.0016 |
| lints (per-impression) | 9,849 ± 372 | 0.2674 ± 0.0024 |
| lints (hourly) | 10,566 ± 287 | 0.2625 ± 0.0021 |
| hybrid_linucb (per-impression) | 2,295 ± 23 | 0.3184 ± 0.0009 |
| hybrid_linucb (hourly) | 2,353 ± 51 | 0.3180 ± 0.0008 |
| hybrid_lints (per-impression) | 2,308 ± 34 | 0.3183 ± 0.0009 |
| hybrid_lints (hourly) | 2,363 ± 45 | 0.3180 ± 0.0008 |

## How to read the tables

* **CTR here is top-1 precision** (see Formulation): random = 0.1157, oracle = 1.000. Mean regret
  per impression = 1 − CTR exactly, so the two columns carry one piece of information.
* "win / LOSS / n.s." = whether the paired 95 % user-bootstrap CI of the difference to
  LightGBM-greedy excludes zero. CIs reflect *data* variance (users resampled); seed variance of the
  stochastic policies is separate and tiny (≤ 0.0004 for the learning bandits, 0.0038 for hourly popularity).
* Slice rows share one set of runs; the slice CIs are not corrected for multiple comparisons, and
  differences *between* slices (e.g. cold vs warm) were not separately tested.

## Leakage and provenance audits (tuning period only; `results/bandit_ebnerd_small/leakage_audit.json`, `mechanism_ablation.json`)

The task says to audit before reporting if a bandit looks suspiciously good. The hybrid's lead over
LightGBM-greedy is modest (+0.0106, 3.4 % relative), not large, but the *shape* was odd (the lead
is biggest on day 1 and shrinks), so the audits were run anyway.

| Audit | Result | Reading |
|---|---|---|
| Label permutation (which candidate is "clicked" shuffled inside each tuning impression) | hybrid LinUCB 0.1172, hybrid TS 0.1185, LinUCB 0.1190, LightGBM-greedy 0.1168 vs random-expected 0.1192 | everything collapses to random → no label path through the features/pipeline |
| Single-feature top-1 scan (argmax of one feature alone) | best lone non-LightGBM feature: `lt_topic_affinity` 0.180; next 0.174, 0.170; LightGBM 0.329 | no individual feature is a leak-sized predictor |
| Full-information static ridge, fit on the first half of the tuning period with *all* candidate labels, frozen on the second half | linear-only 0.2837, hybrid 0.3229 vs LightGBM 0.3176 | the linear-only model is capacity-limited **even with full information** (so its bandit loss is not an exploration cost); hybrid gains only +0.005 when static |
| Mechanism ablation of the winning hybrid (zero feature groups) | full 0.3478; zero {title_len, body_len, sentiment_score, n_topics, n_entities, is_premium} 0.3460; also zero category/type/sentiment one-hots 0.3390; LightGBM columns + bias only 0.3289 (= LightGBM, as it must be) vs LightGBM 0.3289 | the gain needs the non-LightGBM features; ≈ half of it survives removing all static article metadata, so it is not solely a "pseudo article-ID" effect. Mechanism is **not** fully pinned down |
| Unit tests | shuffled-future labels leave all 8 policies' past decisions identical (per-impression and hourly); a deliberate future-peeking policy *is* caught by the same test; only the chosen arm's reward is revealed | no future information reaches a decision by construction |

Why the lead shrinks (hypotheses, **not tested**): (a) LightGBM was fitted on the week *before* the
stream, so a ridge fitted on the first hours of the validation week quickly learns a correction for
train→validation shift, and the shift fades or LightGBM's own accuracy rises later in the week
(LightGBM-greedy CTR rises 0.303 → 0.315 across the held-out days); (b) the ridge has no forgetting,
so its early advantage is diluted as it averages over the week. A discounted/windowed update is the
obvious next experiment. Neither is claimed as the cause.

## Findings

1. **Plain shared-θ linear bandits lose clearly to LightGBM-greedy** (LinUCB −0.0276, Thompson
   −0.0279 per-impression; −0.043 / −0.041 hourly; all CI-clear). The full-information control above
   says why: a linear model on these features tops out at 0.284 vs 0.318 even with every label
   visible, so this is model capacity (trees capture interactions a shared linear θ cannot, and
   user-constant features cancel in a shared argmax), not a bandit-feedback or exploration cost.
2. **The hybrid is the only bandit that wins**: +0.0106 [+0.0090, +0.0123] (LinUCB) and +0.0106
   [+0.0091, +0.0123] (Thompson) per-impression; +0.0054 and +0.0058 with hourly batches. Cumulative
   regret 100,370 vs 101,936 (−1.5 %). It is the production-realistic variant, and in no reported slice or day
   is it CI-clearly worse than the ranker it sits on.
3. **Exploration does not pay for itself here; the gain is greedy residual learning.** The tuning
   surface is flat in the exploration parameter (hybrid LinUCB α from 0 to 0.1: 0.3471–0.3478, inside
   the ±0.0015 tuning noise; Thompson v: 0.3463–0.3472), the winners have α ≈ 0, and ε-greedy over
   LightGBM degrades monotonically with ε (0.3283 → 0.2863 for ε 0.003 → 0.2) and is a CI-clear
   (tiny) loss at its tuned ε. Where the hybrid disagrees with LightGBM (25.7 % of impressions), its
   picks win 0.2367 vs 0.1953 for LightGBM's picks on the same impressions (+0.0413 [+0.0341,
   +0.0478]); those are exploitation re-rankings, not exploration. Plain LinUCB/Thompson disagree on
   59 % of impressions and lose −0.047 there.
4. **Update frequency matters: hourly batching roughly halves the gain** (+0.0106 → +0.0054/+0.0058)
   and removes the first-day win (+0.0020 [−0.0017, +0.0054], n.s.). A production refresh cadence of
   one hour costs about half of the per-impression benefit.
5. **No visible "learning curve" in the final window.** The lead is largest on held-out day 1
   (+0.0156) and in the final 20 % of the stream it is +0.0024 [−0.0004, +0.0056] (n.s.) — i.e. the
   bandit gains fast and then stops improving while LightGBM catches up. The final-window CTR
   (0.3193) is not higher than the first-day CTR (0.3184). The task's "final-window CTR to show
   learning" therefore shows *fast adaptation then a plateau*, not continued improvement.
6. **Cold articles benefit more**: hybrid LinUCB +0.0149 [+0.0064, +0.0240] on the 3,892 impressions
   whose clicked article had < 100 prior exposures (7 % relative) vs +0.0105 on warm articles;
   plain LinUCB loses on both (−0.0136 / −0.0279). Cold users (history < 5) do not exist in this
   window (minimum history = 5); the substituted low-history slice (< 20, n = 7,288) shows +0.0102
   [+0.0022, +0.0178] per-impression and an n.s. +0.0058 hourly. Large candidate sets gain more
   (+0.0158 vs +0.0064) — more arms, more room for a correction.
7. **Simulator (SIMULATED, not log-replay)**: hybrids ≈ LightGBM-greedy (cumulative expected regret
   2,295 ± 23 / 2,308 ± 34 vs 2,346), i.e. −1.8 % to −2.2 %; plain linear bandits ≈ 4× worse (9,833).
   The simulator is built from the sibling `K_cls` booster, which shares features and training data
   with the ranker, so it *favours* the static ranker by construction; it is a sanity check on bandit
   mechanics (regret ordering matches log-replay), not independent evidence.

## Latency (`benchmarks/results/2026-10-12T19-14-16_ebnerd_small_bandit_profile.json`)

Apple M3 / 8 GiB, one impression per call, 4,700 timed decisions after 300 warm-up, d₀ = 100
(hybrid d = 102). Feature build (0.21 ms/impression, `profile_ebnerd.py`) is common to all policies and not included.

| Decision | p50 (ms) | p90 | p99 | decisions/s (mean) |
|---|---:|---:|---:|---:|
| LightGBM-greedy (predict + argmax) | 0.300 | 0.780 | 1.495 | 2,384 |
| LinUCB select | 0.017 | 0.037 | 0.059 | 43,969 |
| LinUCB update (Sherman–Morrison) | 0.039 | 0.096 | 0.717 | 11,542 |
| LinUCB select + update | 0.057 | 0.133 | 0.800 | 9,142 |
| Thompson select (fresh posterior: Cholesky) | 0.053 | 0.080 | 0.352 | 12,730 |
| Thompson select (cached posterior, hourly deployment) | 0.009 | 0.015 | 0.027 | 91,113 |
| Thompson select + update | 0.095 | 0.178 | 0.871 | 6,806 |
| Hybrid LinUCB select + update | 0.057 | 0.132 | 0.392 | 10,962 |
| **Hybrid LinUCB total (LightGBM predict + select + update)** | **0.380** | 0.874 | 1.854 | 1,958 |
| Hybrid Thompson total | 0.445 | 0.962 | 1.985 | 1,705 |

Reading: the bandit's own work is cheaper than the booster's (p50 0.057 vs 0.300 ms), but the
hybrid *needs* the booster's score, so its production cost is the booster plus ≈ 0.08–0.15 ms — about
+27 % (LinUCB) / +48 % (Thompson) at p50 over LightGBM-greedy alone. The update is O(d²) and not the
bottleneck. Thompson's Cholesky is the expensive part and is amortised when updates are batched.
Single-run measurement on a laptop: p99/max (update p99 0.7 ms) are noisy and no CI is claimed.

## SNIPS / counterfactual estimate: skipped, with reasons

Skipped because no defensible propensity exists in this data, not for effort. (1) Step 2 shows
in-view order is not display position — click probability is flat across positions — so a
position-based propensity has nothing to model. (2) Inverse-propensity weighting needs the logging
policy's probability of the *action being evaluated*. Here the production system chose the *set*
shown, and the user — not a policy — chose the click; a fitted "propensity of being shown" model
describes the set selection, not the pick. EB-NeRD logs no propensities. (3) The shown set is
(near-)deterministic given the context, so positivity/overlap fails for any article that was not
shown; clipped weights would only hide that. Replay on the logged candidates (what this ADR does) is
unbiased only for the question "which of the already-shown articles is clicked".

## Limitations

* **Exposure bias / non-random logging.** Only articles the production system showed can be chosen;
  impressions without a click are absent (every impression has ≥ 1 click). All gaps are compressed,
  and "CTR" is top-1 precision, not site CTR.
* **Label = click only.** No dwell time, satisfaction, or editorial value; a click-maximiser can
  drift toward click-bait. `read_time`/`scroll` are not part of the reward.
* **No real-time feedback in the data.** The logged outcome of impression *t* is available to the
  replay immediately; in production, clicks arrive with delay, and the hourly mode only partly models that.
* **Scale and scope.** One dataset, one week (4 held-out days); the ranker is the locally retrained
  `ebnerd_small` Candidate K (validation AUC 0.7588), **not** the Ada `ebnerd_large` model behind the
  0.7542 leaderboard score (weights unavailable). Tuning picked the best of 8–40 noisy
  configurations per (family, update mode) on a flat surface, so the frozen hyper-parameters are not individually
  meaningful (winner's curse); only the family-level conclusions are.
* **Replay is not a policy-value estimate under feedback loops.** In deployment the bandit's picks
  change what users read and what future data looks like; replay cannot show that.
* **CI scope.** User bootstrap ignores correlation across days; the held-out period is too short for
  day-block resampling.
* **Mechanism of the hybrid's gain is unproven** (audit above) and the decay of its lead is unexplained.

## What an online A/B test would need

* **Unit and split.** Randomise by **user** (sticky bucket), not impression: bandit state is
  shared, and a user sees many impressions. Start 5 % bandit / 5 % LightGBM-greedy holdout with the rest
  on control, ramp to 50/50 after guardrails clear. Keep a permanent 1–5 % LightGBM-greedy holdout.
* **Sample size (inference, from this run's variance).** Top-1-precision σ² ≈ p(1−p) ≈ 0.21; detecting a
  +0.005 absolute lift at 80 % power, α = 0.05 needs ≈ 2·0.21·(2.8)²/0.005² ≈ 132 k impressions **per arm**
  before a design-effect inflation for repeat users (likely 1.5–2×). The hourly-batch lift (≈ +0.005)
  is right at that boundary.
* **Primary metric.** Online CTR / clicked-article rank, plus read-time-weighted clicks to resist click-bait.
* **Guardrails** (the project already has them, ADR-007): intra-list **diversity@10**, **novelty@10**,
  catalogue **coverage@10**, share of impressions whose shown item is cold (< 100 exposures) and
  per-category exposure share vs. editorial targets; **latency** (p99 decision ≤ SLA; hybrid adds
  ≈ 0.1 ms p50 on top of the booster here) and error rate; stability (θ norm, share of picks ≠
  ranker top-1 — 25 % here).
* **Safety.** Cap the bandit's deviation from the ranker (e.g. only re-rank within the ranker's top-k),
  kill-switch to LightGBM-greedy, and log the **propensity of each served action** (this dataset has
  none) — that is what makes a later SNIPS/doubly-robust evaluation defensible.

## Alternatives considered

* **Disjoint per-article LinUCB** — rejected (memory, cold start for every article); see above.
* **Neural / non-linear contextual bandit** — not run: the hybrid already needs the booster; a
  learned residual on the booster's embedding is the natural follow-up, outside the 6-hour box.
* **Discounted / sliding-window ridge** — not run; the top follow-up given finding 5.
* **Day-block bootstrap** — rejected (4 blocks).

## Decision and conditions to revisit

* **Decision.** Treat the hybrid bandit as a *candidate online layer*, not a replacement: it is
  CI-clear better in log-replay (+0.0106 / +0.0054) at +27 % decision latency, but the evidence is one
  week of one small model on selected, click-only logs. Do not ship plain linear bandits. Do not spend
  engineering on exploration bonuses; spend it on update cadence and forgetting.
* **Revisit if:** (a) a discounted-update variant sustains the lead into the final window;
  (b) the experiment is repeated on the Ada `ebnerd_large` ranker or a second period and the lead
  is < +0.003; (c) propensities become available; (d) an online A/B contradicts replay.

## References

* Li, Chu, Langford, Schapire (2010), *A contextual-bandit approach to personalized news article
  recommendation* (LinUCB; hybrid LinUCB; the offline replay method).
* Agrawal & Goyal (2013), *Thompson sampling for contextual bandits with linear payoffs*.
* Li, Chu, Langford, Wang (2011), *Unbiased offline evaluation of contextual-bandit-based news
  article recommendation algorithms* (replay estimator; requires a uniformly random logging policy —
  which EB-NeRD is not, see Limitations).
* Sherman & Morrison (1950), rank-1 inverse update.
