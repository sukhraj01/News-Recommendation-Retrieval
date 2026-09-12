# ADR-013 — Candidate K: LightGBM Learning-to-Rank over Engineered EB-NeRD Features

> **Purpose**
>
> An Architecture Decision Record (ADR) captures **why** an engineering
> decision was made.
>
> It records the explored alternatives, the evidence considered, the
> trade-offs accepted, and the reasoning behind the final decision.
>
> Implementation may evolve over time, but the engineering reasoning
> should remain traceable.
>
> When reasoning changes, update the ADR instead of rewriting history.

---

**Date:** 2026-08-26
**Status: RESOLVED — real, confirmed Codabench win.** Submission 907863 (2026-08-29 23:01) scored **0.7542** on the real EB-NeRD leaderboard, **+0.2138 over the previous deployed submission (0.5404)** and **+0.1572 over the challenge's own popularity baseline (0.5970)** — the first time this project's EB-NeRD line has beaten either. Within 0.0157 AUC of the honest, leakage-free literature ceiling (0.7699). 0.80 was **not** reached — stated plainly, not rounded up. Local `ebnerd_large` validation (K_rank_nopos, the shipped arm): 0.7590 (95% CI 0.7588–0.7593, full 12,566,385-impression validation set) → real 0.7542, a small honest compression (−0.0048), the first EB-NeRD/MIND candidate in this project whose local win didn't evaporate or thin to nothing on the real test set. Assumption A1 (BlackPearl long/short-term decomposition) stays falsified at this scale: short-term interest contributes 0.8% of gain, freshness 33.3%. Full trail: local `ebnerd_small` fix + audit (2026-08-27 addendum) → real `ebnerd_large` training, five distinct Ada engineering incidents hit and fixed (2026-08-29 addendum) → real Codabench result (2026-08-30 addendum).
**Severity:** High (this is the project's last planned EB-NeRD intervention)

---

# Engineering Question

This project's EB-NeRD leaderboard score is **0.5404 AUC** (submission 888045,
MiniLM embeddings; the `contrastive_vector` resubmission 896072 scored the same
to four decimals). That is **below the challenge's own naive popularity
baseline** and barely above its in-view-rate baseline. Can a GBDT over
engineered behavioural/temporal features — the approach the RecSys Challenge
2024 report says every top team actually used — close a meaningful part of that
gap, without importing the leakage that inflated the real competition's
headline numbers?

---

# Reference Points (corrected against the primary source)

All figures below are from the organizers' own report, *RecSys Challenge 2024:
Balancing Accuracy and Editorial Values in News Recommendations*
([arXiv:2409.20483](https://arxiv.org/html/2409.20483v1)), read directly this
session. Two numbers the session brief carried forward turned out to be wrong
and are corrected here rather than propagated:

| Reference | AUC | Note |
|---|---|---|
| Winner `:D`, as submitted | 89.24 | Headline leaderboard score |
| Winner `:D`, leaky-feature ablation, with | 88.64 | Organizers' own ablation arm |
| Winner `:D`, leaky-feature ablation, **without** | **76.99** | −11.65. "Even with an AUC of 76.99, :D would be in the top 10% of teams." |
| 2nd place, **BlackPearl** | **88.15** | **Not 82.20.** No published "clean" BlackPearl number exists — the organizers ran the leakage ablation only on the winner. |
| 3rd, Tom3TK | 87.07 | |
| Top academic team, **FeatureSalad** | **85.13** | Pure LightGBM Ranker; full text + public code available |
| Baseline — most clicks (popularity) | **59.70** | This project is currently *below* this |
| Baseline — read-time consumption | 59.49 | |
| Baseline — in-view rate | 54.50 | This project's 54.04 sits just under this |
| Baseline — random | 49.98 | |

**Correction 1 — BlackPearl scored 88.15, not 82.20.** The 82.20 figure in the
session brief does not appear anywhere in the challenge report. Since the
organizers only ablated the winner, there is no published leakage-free
BlackPearl score to target. The honest clean-performance anchor in the
literature is therefore **76.99** (the winner with future information removed),
not 82.20.

**Correction 2 — the brief assumed Candidate H2 used LightGBM or CatBoost.**
It used neither: `scripts/run_gbdt_combiner_experiment.py` uses
`sklearn.ensemble.HistGradientBoostingClassifier`, explicitly chosen at the
time to avoid "a new dependency and macOS OpenMP install risk". That risk was
re-tested this session and no longer exists — LightGBM 4.7.0 installs from a
prebuilt `macosx_12_0_arm64` wheel and `lambdarank` trains correctly. Since
H2's model has no ranking objective at all, reusing it would have forced the
pointwise framing; LightGBM is adopted instead.

**Sourcing caveat.** BlackPearl's full paper (ACM DOI
[10.1145/3687151.3687163](https://dl.acm.org/doi/10.1145/3687151.3687163))
could not be retrieved — the ACM full-text endpoint returns HTTP 403, as does
the ResearchGate mirror. The long-term/short-term feature decomposition below
is reconstructed from their abstract and is a **working hypothesis, not a
reproduction of their methodology**. The better-grounded reference actually
obtained is FeatureSalad's LightGBM Ranker paper (85.13 AUC, top academic
team), whose approach — "user, article, and session information, as well as
their interactions... embeddings created from users' browsing history and news
article texts, and their cosine similarities used as additional features" —
matches the feature families built here.

---

# Decision Scope

- [ ] Entire project
- [x] Experiment only (new candidate)
- [ ] Temporary / Prototype
- [x] Assignment-specific
- [x] General reusable pattern (the feature module is split-agnostic and reused unchanged for train / validation / test)

**Affected Components:** New — `src/retrieval/ebnerd_features.py`,
`scripts/run_ebnerd_gbdt_experiment.py`, `tests/unit/test_ebnerd_features.py`.
No existing scorer, schema, or pipeline module is modified.

---

# Context: why content similarity plateaus at ~0.54

EB-NeRD's in-view candidate lists are drawn from the same front page at the
same moment. Every candidate is therefore already topically adjacent and
recently published, which leaves content similarity — the only signal BM25
(ADR-005/006), MiniLM (ADR-008) and `contrastive_vector` (ADR-008 addendum 2)
use — with very little to discriminate on. Measured directly on `ebnerd_small`
validation this session: the median candidate is **3.7 hours old** and 58% of
candidates are under 6 hours old.

What separates a click from a non-click in this setting is behavioural and
temporal: article freshness *relative to the other candidates on offer*, how
well the item matches the categories this user returns to, whether it matches
what they were reading twenty minutes ago, and how much traction it already
had. None of those are expressible as a cosine similarity.

---

# Design Space Exploration

## Option 1 — More/better embeddings (rejected)

Swap MiniLM for a stronger encoder, or blend both. ADR-008's addendum 2 already
measured the ceiling here: EB-NeRD's own `contrastive_vector` artifact beat
MiniLM CI-clear on local validation, and on the real leaderboard the difference
was **+0.0005** (0.5402 vs 0.5397 mean AUC over the per-date breakdown). The
lever is exhausted; the problem is not encoder quality.

## Option 2 — Neural news recommender (NRMS-style), as Candidate J did on MIND (rejected)

Candidate J is this project's one real leaderboard win, so the temptation is
real. Rejected on evidence: the challenge report's "Common Themes" states most
solutions were GBDT ensembles, and the top *academic* team reached 85.13 with a
plain LightGBM Ranker. A neural architecture is also far more expensive to
iterate inside a 4-day budget, and the EB-NeRD gap here is a *feature* gap, not
a representation gap — NRMS would still see only title text.

## Option 3 — GBDT over engineered long/short-term features (chosen)

Matches what actually won, is cheap to iterate, handles NaN natively (which
matters: `age`/`postcode`/`gender` are 93–98% missing, `scroll_percentage` ~70%),
and treats categorical fields without one-hot blowup.

**Two arms, so the result is attributable:**

- **K-rank** — `objective="lambdarank"`, one group per impression. Verified
  precondition: 100% of `ebnerd_small` impressions have ≥1 click (mean 1.004),
  so every group is well-formed for a listwise loss.
- **K-cls** — `objective="binary"` on the identical matrix, i.e. H2's pointwise
  framing. Isolates whether any gain comes from the listwise objective or
  merely from having better features.

---

# The Feature Set

60 features in six families. Long-term profiles weight every history click by
engagement (`1 + log1p(read_time)`); short-term profiles weight by exponential
recency decay with a **24-hour half-life**, chosen a priori from the news
domain's item half-life (ADR-005's framing) and deliberately **not tuned**
against validation.

---

# Leakage Audit — every feature, individually

The standard applied is ADR-007's: a feature is admissible only if its value
could be computed from information that exists **at the moment the impression
is served**. Sources are abbreviated:

- **H** = the split's own `history.parquet`. EB-NeRD constructs this as a
  21-day window ending strictly before the behaviors window opens. **Verified
  empirically this session, not assumed** — `train` history spans
  `2023-04-27 07:00:00 … 2023-05-18 06:59:59` against behaviors
  `2023-05-18 07:00:01 …`; `validation` history spans
  `2023-05-04 07:00:00 … 2023-05-25 06:59:59` against behaviors
  `2023-05-25 07:00:02 …`. `history_max < behaviors_min` holds for both.
- **A** = static article metadata fixed at publication.
- **C** = the impression's own request-time context.

| # | Feature | Src | Available at serving time because… |
|---|---|---|---|
| 1 | `lt_cat_affinity` | H+A | Category distribution over the user's closed past history window. |
| 2 | `lt_subcat_affinity` | H+A | As above, subcategory multi-hot. |
| 3 | `lt_topic_affinity` | H+A | As above, topic multi-hot. |
| 4 | `lt_embed_sim` | H+A | Cosine of candidate embedding against engagement-weighted history mean; embeddings are a pure function of article text. |
| 5 | `lt_cat_rank` | H+A | Rank of candidate category within the user's own past preference ordering. |
| 6 | `lt_sentiment_distance` | H+A | Candidate sentiment vs. user's past mean sentiment; sentiment is a published-time article attribute. |
| 7 | `st_cat_affinity` | H+A | Recency-decayed variant of #1; decay reference is the history window's close, which precedes every impression. |
| 8 | `st_subcat_affinity` | H+A | As #7. |
| 9 | `st_topic_affinity` | H+A | As #7. |
| 10 | `st_embed_sim` | H+A | As #7. |
| 11 | `st_cat_rank` | H+A | As #7. |
| 12 | `is_last_clicked_category` | H+A | The user's final *history* click, not any behaviors-window click. |
| 13 | `in_recent_categories` | H+A | Categories of the last 5 history clicks. |
| 14 | `cat_affinity_drift` | H+A | Arithmetic difference of #7 and #1. No new source. |
| 15 | `embed_sim_drift` | H+A | Difference of #10 and #4. |
| 16 | `topic_affinity_drift` | H+A | Difference of #9 and #3. |
| 17 | `log_history_len` | H | Count of past clicks. |
| 18 | `user_mean_read_time` | H | Read times inside the closed history window. |
| 19 | `user_median_read_time` | H | As #18. |
| 20 | `user_mean_scroll` | H | As #18. |
| 21 | `user_distinct_categories` | H+A | As #18. |
| 22 | `user_clicks_last_24h` | H | Clicks in the final 24h *of the history window*. |
| 23 | `user_mean_article_age_h` | H+A | The user's past appetite for freshness. |
| 24 | `hours_since_last_click` | H+C | Impression timestamp minus last history click. Both known at request time. |
| 25 | `user_history_span_h` | H | First-to-last history click. |
| 26 | `article_age_h` | A+C | Impression time minus publication time. Verified non-negative in tests. |
| 27 | `is_fresh_6h` | A+C | Threshold on #26. |
| 28 | `sentiment_score` | A | Published-time attribute. |
| 29 | `sentiment_label` | A | Published-time attribute. |
| 30 | `article_type` | A | Published-time attribute. |
| 31 | `is_premium` | A | Published-time attribute. |
| 32 | `title_len` | A | Published-time attribute. |
| 33 | `body_len` | A | Published-time attribute. |
| 34 | `n_topics` | A | Published-time attribute. |
| 35 | `n_entities` | A | Published-time attribute (`ner_clusters`). |
| 36 | `log_history_popularity` | H+A | **See the note below — this is the one that had to be designed, not just checked.** |
| 37 | `log_history_popularity_decayed` | H+A | As #36, recency-weighted. |
| 38 | `category_code` | A | Published-time attribute. |
| 39 | `age_vs_user_appetite` | H+A+C | #26 minus #23. |
| 40 | `n_candidates` | C | Length of the in-view list being served. |
| 41 | `hour_of_day` | C | Request timestamp. |
| 42 | `day_of_week` | C | Request timestamp. |
| 43 | `is_weekend` | C | Request timestamp. |
| 44 | `is_front_page` | C | Whether the request had a context article. |
| 45 | `device_type` | C | Request attribute. |
| 46 | `is_sso_user` | C | Request attribute. |
| 47 | `is_subscriber` | C | Request attribute. |
| 48 | `user_age` | C | Account attribute (97% null; LightGBM handles natively). |
| 49 | `gender` | C | Account attribute (93% null). |
| 50 | `context_read_time` | C | **Flagged — see caveat below.** |
| 51 | `context_scroll_percentage` | C | **Flagged — see caveat below.** |
| 52 | `position_in_view` | C | **Flagged — see caveat below.** |
| 53 | `relative_position_in_view` | C | As #52. |
| 54–58 | `*_rank_in_imp` (5 features) | — | Rank of an already-admitted feature among the *other candidates in the same in-view list*. Uses only values already computed for that request. |
| 59 | `article_age_minus_imp_min` | A+C | Candidate age minus the freshest candidate's age in the same request. |
| 60 | `lt_embed_sim_minus_imp_max` | H+A+C | Same construction on #4. |

## Fields deliberately never read

Enforced in code, not just documented: `FORBIDDEN_RAW_COLUMNS` in
`src/retrieval/ebnerd_features.py` causes `_read_zip_parquet` to **raise** if
any of these is requested, and `tests/unit/test_ebnerd_features.py`
parametrizes a test over every one of them.

| Field | Why excluded |
|---|---|
| `total_inviews` | Article-lifetime exposure aggregate — includes exposure occurring *after* the impression being scored. ADR-009 already measured what this leaks. |
| `total_pageviews` | As above. |
| `total_read_time` | As above. |
| `next_read_time` | Post-click outcome of the very click being predicted. Pure label leakage. |
| `next_scroll_percentage` | As above. |

## The popularity feature — the one that needed design, not just a check

Article popularity is the single strongest non-content signal in news, and it
is also exactly where the real competition leaked. Three constructions were
considered:

1. **Clicks counted from the behaviors window.** Rejected outright. Those
   clicks *are* the labels, and the blind test set has no labels at all — the
   feature could not be computed at serving time even in principle.
2. **Causal expanding-window in-view counts** (how often the article appeared
   in-view before time *T*). Genuinely leak-free and computable on the test
   set, but requires strict chronological streaming and adds real complexity.
   Deferred, not rejected — a candidate for a second pass.
3. **Clicks counted from the `history` file only** (chosen). The history window
   closes before the behaviors window opens, so this is strictly past
   information, it is computed identically for train / validation / test, and
   it needs no labels. Its weakness is honest and expected: articles published
   after the history window — the item-cold-start case that dominates news —
   score zero. That is a real limitation, and the model is given
   `article_age_h` to compensate. A unit test asserts an
   after-the-window article gets exactly zero popularity, which would fail
   immediately if the source were ever switched to behaviors.

## Three flagged features (admitted, but with a caveat on record)

- **`context_read_time` / `context_scroll_percentage`** (#50, #51) describe the
  page the user was on *when the in-view list was logged*. They are provided in
  EB-NeRD's test set, so using them is within the competition's rules and they
  are not future information about the *candidate*. But a page's total read
  time is only fully known once the user leaves it, so in a strict production
  serving loop this value would be partially unknown at request time. Admitted
  as request-time context, flagged as the weakest such claim in this table.
- **`position_in_view` / `relative_position_in_view`** (#52, #53) encode the
  order of the in-view list, which may reflect the publisher's own ranking.
  This is available at prediction time and is standard practice, but it means
  part of any gain could be "learning Ekstra Bladet's existing ranker" rather
  than learning user interest. Measured directly on `ebnerd_small` train:
  clicked candidates sit at mean position 5.12 vs 7.92 for non-clicked, so the
  signal is real and substantial. **Because this is a position-bias confound
  rather than a temporal leak, it will be reported as an explicit ablation arm
  (with and without #52/#53) rather than silently absorbed into the headline
  number.**

---

# Assumptions and Unknowns

- **A1.** The long/short-term split reconstructed from BlackPearl's abstract is
  a hypothesis. If `st_*` features show negligible gain, that falsifies the
  reconstruction, not the GBDT approach.
- **A2.** The 24h short-term half-life is an a-priori choice, untuned.
- **A3.** Short-term decay references the history window's close rather than
  each impression's own timestamp (a ~170x cost saving). The varying part is
  recovered by `hours_since_last_click`. Assumed to cost a scale factor, not
  signal — unverified.
- **U1.** Whether local validation gains transfer to the blind test set. This
  project has now seen a CI-clear local win compress to nothing **twice**
  (Candidate G on MIND, `contrastive_vector` on EB-NeRD). Any result here must
  be reported with that prior stated.

---

# Environment defect found while building this (recorded because it invalidates measurements)

The project virtualenv was running **numpy 1.26.4 on CPython 3.14.0**. No
`cp314` wheel of numpy 1.26.4 exists (confirmed: `pip download numpy==1.26.4
--only-binary=:all:` on this interpreter reports the oldest available version
as 2.3.2), so the installed copy had been built from source against an
interpreter it was never tested against.

It produced provably wrong results. Captured directly:

```
len(row_is_stop)      = 278139
row_is_stop.sum()     = 240288      <- WRONG
flatnonzero(~mask)    = 240288
flatnonzero(mask)     = 240288      <- both branches equal: impossible
np.repeat(is_stop_imp, sizes_tr).sum()  = 37851   <- CORRECT (same expression, recomputed)
sizes_tr[is_stop_imp].sum()             = 37851   <- CORRECT
```

A stored boolean array disagreed with a fresh recomputation of the identical
expression, with no intervening mutation, and the result flipped between runs.
It never reproduced in a short isolated script — only inside the longer-running
process, which is consistent with miscompiled vectorised paths from an
ABI-mismatched source build.

**Resolution:** the venv was rebuilt on **Python 3.11** (`/opt/homebrew/bin/python3.11`),
which `pyproject.toml` already declares (`python = "^3.11"`) and for which
numpy 1.26.4 ships an official tested wheel. Every pinned version is unchanged,
so results stay comparable to earlier work.

**Open item, flagged not resolved:** it is not established when this venv was
created, so it is not known whether any previously recorded numeric result in
this project was produced under the defective build. This ADR does not claim
prior results are wrong — it records that they have not been re-verified under
a supported interpreter.

---

# Addendum (2026-08-30) — Real Codabench Result: a Genuine Win

Submission ID **907863**, uploaded 2026-08-29 23:01, `K_rank_nopos` trained on
the real `ebnerd_large` bundle. Screenshots:
`submissions/ebnerd_testset_gbdt_k/leaderboard_screenshot_{upload,rank}.png`.

## The result, against every reference point from this ADR's opening table

| Reference | AUC | Δ from this result |
|---|---|---|
| **This submission (Score column)** | **0.7542** | — |
| This submission (per-date breakdown mean, `*` = 50% of testset) | 0.7535 | −0.0007 (same tie the MIND converter's MRR discrepancy already established — a metric-scope difference, not a bug) |
| Previous deployed EB-NeRD submission (888045) | 0.5404 | **+0.2138** |
| Challenge's own "most clicks" baseline | 0.5970 | **+0.1572** |
| Challenge's own "in-view rate" baseline | 0.5450 | +0.2092 |
| **Honest literature ceiling** (winner, leakage-ablated) | 0.7699 | **−0.0157** |
| BlackPearl (2nd place, as submitted — not leakage-ablated) | 0.8815 | −0.1273 |
| Local `ebnerd_large` validation (this candidate) | 0.7590 | −0.0048 |
| The session's original 0.80 target | 0.80 | **−0.0458 (0.80 was NOT reached)** |

**0.80 was not reached. Stated plainly, not rounded up** — the session brief's
own instruction. What was reached instead is more defensible than 0.80 would
have been on its own terms: **within 0.0157 of the honest, leakage-free
literature ceiling** this project explicitly adopted as the real target after
finding the originally-floated 0.85 required features later shown to leak
future information (this ADR's opening section). No leaky feature was used
here either — every feature in this candidate passed the same audit
(ADR-009-style) the rest of this project holds to. Closing to within 2 points
of AUC of the real 2nd-place-caliber result, using only genuinely
leakage-free signal, is the honest headline, not "we hit 0.75 instead of
0.80."

## Local-to-real transfer: the first EB-NeRD candidate that didn't compress badly

This project has now measured local-to-real transfer three times on EB-NeRD/MIND
candidates with a CI-clear local win:

| Candidate | Local win | Real result | Compression |
|---|---|---|---|
| MIND Candidate G (cohort-gated combiner) | +0.0027 (MINDsmall) / +0.0019 (MINDlarge) | −0.0003 | Evaporated to flat |
| EB-NeRD contrastive vector | +0.0023 (ebnerd_small) | +0.0005 | Thinned to near-nothing |
| **EB-NeRD Candidate K (this ADR)** | **local ebnerd_large 0.7590** | **real 0.7542** | **−0.0048 — a real but small, honest compression** |

This is the first time in this project a strong local EB-NeRD/MIND result
transferred to the real leaderboard largely intact. The most likely reason,
consistent with everything else in this ADR: this candidate's edge comes from
structural, item-level, leak-safe signal (article freshness, immediate
context) rather than a subtle statistical pattern (a cohort-gating rule, a
marginal embedding-artifact edge) that a real blind test set was always
likely to wash out.

## What remains open, stated rather than hidden

- The **4.5-point gap to the honest 0.7699 ceiling** is real and unclosed.
  Reasonable next moves, not attempted this session given the time budget:
  causal expanding-window in-view popularity (flagged as unexplored in the
  original design-space section), entity-overlap features (this project's own
  MIND-era `entity_overlap_count` pattern, never ported to EB-NeRD), or a
  second stacked model over these predictions.
- A fourth submission was mid-flight (`Running prediction.zip`, ID 907906)
  when these screenshots were taken — not yet resolved at time of writing;
  see PROJECT_STATE for whatever it turns out to be.
- `--train-sample-impressions 2500000` was a memory-driven cap, not a
  data-sufficiency finding — ADR-013's own real-scale result (10.7x more data
  bought +0.0002) suggests using the full ~12M train impressions would likely
  not move this further, but that is inference from a different scale's
  result, not measured directly at the full unsampled scale.

# Addendum (2026-08-29) — Real ebnerd_large Training + Codabench Submission

Trained on Ada (SLURM, IIIT-H) after the local `ebnerd_small` work above.
Real bundle scale, measured directly (not projected): **12,063,890 train
impressions, 12,566,385 validation impressions** — both close to
`ebnerd_testset`'s own 13.5M scale. Training used a deterministic subsample of
**2,500,000** train impressions (whole impressions, not candidates, so every
lambdarank group stays intact) — capped for memory, not desire; see the OOM
section below. Validation was **not** subsampled — the full 12,566,385
impressions were scored via the streaming path, so the numbers below are a
real full-scale measurement, not an estimate.

## Result: scale did not move the needle

| Arm | ebnerd_small AUC | ebnerd_large AUC | Δ | ebnerd_large 95% CI |
|---|---|---|---|---|
| K_cls | 0.7597 | 0.7603 | +0.0006 | 0.7601–0.7605 |
| **K_rank_nopos (shipped)** | 0.7588 | **0.7590** | **+0.0002** | 0.7588–0.7593 |
| K_rank | 0.7581 | 0.7586 | +0.0005 | 0.7584–0.7589 |

**~10.7x more training data bought essentially nothing.** The CIs are now
extremely tight (width ~0.0005, from 12.57M validation impressions), so this
is a precise null result, not noise masking a real gain. Consistent with the
feature-importance story: the model's edge comes from item-level freshness and
immediate context, not from learning richer per-user patterns — signal that
doesn't get meaningfully deeper with more users or more history per user. This
is the honest answer to "does scale help here," and it's no, for this feature
set. Worth stating plainly rather than treating the Ada run as validated-by-
scale; it was validated by the local ebnerd_small run, and the large run
mainly confirms that finding transfers rather than improves on it.

Feature importance at real scale, `K_rank_nopos`:

| Family | ebnerd_small | ebnerd_large |
|---|---|---|
| Freshness | 31.2% | 33.3% |
| Long-term (`lt_*`) | 15.9% | 14.1% |
| Short-term | 1.9% | **0.8%** |
| Context-article | 4.2% | 4.5% |
| Session | 0.6% | 0.4% |

Short-term interest went down further with more data, not up — A1 is now
falsified at a well-powered scale, not just a small-sample one.

## Real engineering incidents this run surfaced (all fixed, not routed around)

Five real, distinct problems, each found from actual job failures on Ada, not
anticipated in advance:

1. **A real OOM**, SLURM-confirmed (`oom-kill` event), at the original
   `--train-sample-impressions 4000000`. Root cause: `fit_X`/`stop_X` (the
   un-split 65-feature train matrix, ~11.5GB) were never freed, and the arm
   that withholds two features (`K_rank_nopos`) needs a column-sliced COPY
   (numpy always copies for fancy-indexed column selection) that coexisted
   with them — ~22.7GB just from those four arrays. Fixed three ways: the
   code now frees `fit_X`/`stop_X` once the arm loop finishes (confirmed by
   grep they're never read again), `--train-sample-impressions` was cut to
   2,500,000, and `--mem-per-cpu` was bumped 4G→8G. Re-run cleared all three
   arms successfully.
2. **A missing-column crash in the real blind test set.** `ebnerd_testset`'s
   `behaviors.parquet` has no `article_id` field at all (checked directly
   against the real schema, not assumed) — every other `BEHAVIOR_COLUMNS`
   field is present. `load_test_behaviors` now degrades this the same way
   `build_user_profiles` already degrades missing optional history columns:
   NaN-fill and continue, which makes every downstream context-article
   feature read as "no known context article" — the honest state, not an
   error. Verified against the real gap (not just unit-tested): re-ran the
   actual scoring script on `ebnerd_small` with `article_id` stripped and
   confirmed via the official evaluator it still produces sensible output
   (AUC 0.7508, a small expected drop from 0.7524 with the column present).
3. **A second segfault from the same root cause as before** (lightgbm +
   torch imported in the same pytest process) — this time via a new test
   file importing the scoring script. Fixed the same way: `load_test_
   behaviors` now lives in `src/retrieval/ebnerd_features.py` (no lightgbm
   dependency), not in the lightgbm-importing script.
4. **Cluster-side friction, not a code bug**: a GPU-holding job got
   `scancel`'d twice by the account's own uid during a multi-hour throttled
   S3 download with the GPU sitting idle the whole time — consistent with a
   shared-cluster fair-use reclaim. Fixed by splitting into a CPU-only
   download stage and a GPU stage that starts using the GPU within seconds
   of allocation, plus a `$HOME`-persistent shared cache (`/ssd_scratch` is
   node-local, so a job landing on a different node than the download
   couldn't see it) so no later job re-hits S3 for the same bytes.
5. **A `#SBATCH` ordering bug caught before it shipped**: an early draft of
   the standalone scoring script placed `export` statements before the
   `#SBATCH` block, which would have caused SLURM to silently ignore every
   resource request. Caught by explicitly checking directive placement
   before submitting, not discovered by a failed job.

## Submission

Arm shipped: **K_rank_nopos** (per the engineer's decision — statistically
indistinguishable from the best arm, immune to the position-bias objection,
and withholding position costs nothing since it's CI-clear better without it).

- `submissions/ebnerd_testset_gbdt_k/prediction.zip` — 13,536,710 lines,
  matching the real test set's impression count exactly. 219MB zipped.
- Real cost: testset article encode ~15 min, user-profile build ~13 min
  (807,677 users), scoring loop ~124 min at ~1,780–1,870 impressions/sec.
- Uploaded to Codabench (competition 2469) by the engineer — real leaderboard
  score pending at time of writing.

# Addendum (2026-08-27) — Per-Impression Short-Term Fix + Dataset Re-Audit

Prompted by the engineer asking whether recency/timestamps had actually been
used properly, given the earlier result reported short-term interest
contributing almost nothing (2.1% of gain).

## Bug found: short-term profiles were static per user, not per impression

The `st_*` (decay-weighted) features were computed once per user in
`build_user_profiles`, against a single reference time shared by the whole
split (the close of the history window). Every impression that user had in
the split — whether on day 1 or day 7 of the window — got an *identical*
short-term profile. `article_age_h` and `hours_since_last_click` were already
correctly per-impression (using each candidate row's own `imp_time`); only the
category/subcategory/topic/embedding decay profiles and `user_clicks_last_24h`
had this defect.

**Why this matters for the earlier finding.** The prior conclusion — "short-
term interest contributes almost nothing, Assumption A1 is falsified" —
assumed the short-term features were actually measuring short-term dynamics.
If they were frozen per user instead, low importance could mean either "short-
term interest genuinely doesn't help" or "this particular construction never
captured real short-term signal." The two are not distinguishable from the
pre-fix result. Re-run below with the fix in place.

**Fix:** `st_*` features are now computed once per impression
(`compute_short_term_features`), referenced to that impression's own
`imp_time`, so a user's short-term profile genuinely moves forward as their
impressions span the split. Verified directly: `st_cat_affinity` for the same
user/category now differs across an early vs. late synthetic impression
(`test_short_term_affinity_varies_across_a_users_own_impressions`), which was
impossible before this fix by construction.

**Memory discipline preserved.** A naive fix would return one
`(n_impressions, dim)` matrix per feature — at `ebnerd_large`'s capped
4,000,000-impression training sample, the 384-dim embedding matrix alone
projects to ~6GB, a meaningful bite out of the 40G Ada budget for a
short-lived intermediate. Instead, each impression's decayed vector is a
transient local: computed, used immediately to write that impression's
candidates' scalar features, and discarded. Peak RSS measured directly on the
full `ebnerd_small` re-run: **426MB**, confirming the design holds in practice,
not just in theory.

**Cost:** `SHORT_TERM_LOOKBACK_HOURS = 168` (7 days = 7 half-lives at the
24h half-life, contributing ≤0.78% of the freshest click's weight past that
point) bounds each impression's work via `np.searchsorted` on the user's
time-sorted history (verified 100% sorted across a 3000-user sample), rather
than rescanning full histories that run up to ~1,900 clicks.

## Dataset re-audit: two real misses found, one hypothesis rejected

- **Context-article match — added.** `behaviors.article_id`, non-null for
  **29.1%** of impressions on ebnerd_small validation (measured directly), is
  the article the user was *actively reading* when the impression was served
  — a more immediate signal than anything history-derived, and part of the
  same impression row every other context feature already reads (same leak
  tier as `device_type`). New features: `context_category_match`,
  `context_topic_overlap`, `context_embed_sim`.
- **Causal session structure — added.** The original brief named "session
  device/context" as a short-term signal; only `device_type` had actually
  been built. **44.8%** of `(user, session)` pairs on ebnerd_small validation
  have more than one impression (mean 2.01/session) — real prevalence, not a
  rare edge case. New features: `session_position` (1-indexed, causal —
  ranks only by information at or before the current impression),
  `session_start_gap_h` (elapsed time since the session's first impression,
  which is by construction always at or before every member of the group).
  Both computed once on the full split before any chunking (a session can
  span a chunk boundary in the streamed validation path).
- **`last_modified_time` — audited and explicitly rejected, not silently
  skipped.** Checked as a candidate freshness feature; found to be **100%**
  different from `published_time` for every article, median gap **~150
  days**. That is a bulk pipeline artifact (a batch reprocessing job), not
  editorial revision — confirmed directly by cross-referencing the earlier
  negative-age data-artifact finding: the 9 validation articles with
  `article_age_h < 0` were all touched within the same ~30-minute window on
  2023-06-29, unrelated to their actual publish dates. Not used as a feature:
  it isn't a genuine freshness signal, and using it risks being a soft proxy
  for the article-lifetime popularity signals ADR-009 already quarantined
  (if reprocessing priority correlates with traffic).

`FEATURE_NAMES` grows from 60 to 65. Full re-run on `ebnerd_small` below.

## A second, sharper question closed a real gap: was the fix itself leak-safe?

Asked directly after the fix landed: does the per-impression short-term
computation ever reach into the future, and could the model be "gaming" the
data rather than learning something real? Worth re-verifying on its own,
since this was fresh code, not yet independently re-audited after being
written.

**Found a real gap, not just a theoretical one.** `compute_short_term_features`
used `np.searchsorted` to trim the *old* end of the decay lookback window, but
never checked the *new* end — nothing excluded a history click at or after the
impression's own reference time. Constructed the adversarial case directly
against the function (a synthetic history with a click 300 seconds after the
test impression's time) and confirmed it: the future click was counted with
maximal decay weight (as if it were the freshest possible signal), producing a
non-zero affinity that should have been exactly zero.

**Why this never manifested against real data.** `history.parquet` is
verified, separately, to close strictly before any impression's
`behaviors.parquet` window opens (that's what
`test_history_window_closes_before_behaviors_window_opens` checks). Given that
invariant, every element of `kts_full` was already `< ref` for every real row
— so the function's OUTPUT on real EB-NeRD data was correct throughout. But
the function's *correctness* depended entirely on an external invariant it
never checked itself — safe today, only because nothing (yet) violates the
assumption it silently assumed.

**Fixed:** an explicit upper bound (`hi = np.searchsorted(kts_full, ref,
side="left")`) now excludes any click at or after `ref`, and the same
`clicks_last_24h` computation is bounded the same way. The `np.maximum(...,
0.0)` clamp that used to silently absorb a violation is removed — `kts` is now
provably `< ref` by construction, so a future regression would surface as a
visible bug (a negative argument to `exp`) rather than being quietly clamped
away.

**Verified the fix changes nothing on real data** (as expected, since the
invariant already held there) and **verified it closes the gap on the
adversarial case** (two new tests:
`test_short_term_never_counts_a_click_at_or_after_the_reference_time`,
`test_clicks_last_24h_excludes_a_click_at_or_after_the_reference_time`).
Re-ran the full `ebnerd_small` experiment after the fix — see below for
whether the numbers moved.

`build_history_popularity` was checked for the same class of bug and found
safe by a different, stronger property: its reference point (`all_last.max()`)
is the maximum *within the same array* being filtered, so `reference >= kts`
holds by construction for every element, not by relying on a separate file's
temporal boundary. No fix needed there.

## Results after the fix — ebnerd_small (2026-08-27)

| Arm | AUC before fix | AUC after fix | Δ | 95% CI (after) |
|---|---|---|---|---|
| K_cls | 0.7528 | **0.7597** | +0.0069 | 0.7579–0.7615 |
| K_rank_nopos (ships) | 0.7524 | **0.7588** | +0.0064 | 0.7571–0.7606 |
| K_rank | 0.7514 | **0.7581** | +0.0067 | 0.7564–0.7600 |

All three: CI-clear improvement over both the old numbers and every baseline
(paired vs. embed_sim: +0.2158 for K_rank_nopos). `K_rank_vs_K_rank_nopos` is
now -0.0007 (still CI-clear, position still net-negative to include).

**Short-term interest, re-measured with the bug fixed:**

| Family | Before fix | After fix |
|---|---|---|
| Freshness | 33.5% | 31.2% |
| Long-term (`lt_*`) | 15.3% | 15.9% |
| **Short-term (`st_*` + `user_clicks_last_24h`)** | 2.1% | **1.9%** |
| Drift (`st_ − lt_`) | 0.2% | 0.5% |

Short-term did not recover once genuinely per-impression — it went slightly
*down*. **This closes the loop the fix was meant to close**: the earlier low
importance was not an artifact of the static-per-user construction, since that
construction has now been replaced and the result didn't change direction.
Assumption A1 (the BlackPearl-style long/short-term decomposition) is falsified
with more confidence than before, not less.

**The new features, individually:**

| Feature | Gain rank | Share |
|---|---|---|
| `context_embed_sim` | **#13** | **2.37%** — strongest single addition, beats every `st_*` feature combined |
| `session_start_gap_h` | #32 | 0.38% |
| `context_topic_overlap` | #37 | 0.30% |
| `session_position` | #47 | 0.20% |
| `context_category_match` | #59 | 0.06% — the binary match is too coarse; the continuous `context_embed_sim` captures what it misses |

Combined: context-article features 4.2% of gain, session features 0.6% —
modest individually, but real (not noise), and together they account for a
meaningful share of the measured +0.0064–0.0069 AUC improvement. The pattern
holds from the first pass: *immediate* signals (what's being read right now,
which candidates are freshest in this exact list) beat *aggregated-over-time*
signals (short-term decay, long-term affinity) on this task.

**Peak RSS during the full ebnerd_small run: 426MB** — confirms the
memory-disciplined design (no `(n_impressions, dim)` matrix ever materialized)
holds in practice, not just on paper. Feature-build time roughly doubled
(20s → 40s for train) from the added per-impression loop work, still fully
tractable.


# Results — ebnerd_small (2026-08-26)

`experiments/candidate_k_gbdt_ebnerd_small_2026-08-26/`. Trained on
`ebnerd_small` train (232,887 impressions / 2,585,747 candidate rows), evaluated
on `ebnerd_small` validation (244,647 impressions, 15,342 users, 0 degenerate
impressions skipped). Bootstrap CIs are ADR-007's, 2000 replicates, resampled
over users.

| Arm | AUC | 95% CI | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|
| **K-cls** (binary) | **0.7528** | 0.7511–0.7548 | 0.5224 | 0.5869 | 0.6253 |
| **K-rank-nopos** (lambdarank, position withheld) | **0.7524** | 0.7507–0.7543 | 0.5242 | 0.5882 | 0.6261 |
| **K-rank** (lambdarank) | **0.7514** | 0.7496–0.7533 | 0.5225 | 0.5870 | 0.6247 |
| embed_sim (deployed method's local equivalent) | 0.5430 | 0.5415–0.5444 | 0.3437 | 0.3804 | 0.4591 |
| random | 0.4993 | 0.4979–0.5006 | 0.3129 | 0.3447 | 0.4297 |
| history-window popularity | 0.4269 | 0.4252–0.4288 | 0.2596 | 0.2805 | 0.3804 |

Paired bootstrap (ADR-010's `paired_metric_diff_ci`, same impressions both sides):

| Comparison | ΔAUC | 95% CI | Verdict |
|---|---|---|---|
| K-rank vs. embed_sim | **+0.2084** | +0.2061, +0.2107 | CI-clear |
| K-cls vs. embed_sim | **+0.2099** | +0.2077, +0.2122 | CI-clear |
| K-rank vs. random | +0.2521 | +0.2498, +0.2543 | CI-clear |
| K-rank vs. K-rank-nopos | **−0.0011** | −0.0014, −0.0007 | CI-clear — position features *hurt* |

## The harness is measuring the same thing as prior work

`embed_sim` scored **0.5430**, which reproduces ADR-008 addendum 2's recorded
MiniLM figure on this exact split **to four decimal places**, and `random`
landed at 0.4993. Those two numbers are the reason the 0.75 can be taken
seriously: they show this evaluation path is the same measuring instrument
every earlier candidate was judged with, not a new one that happens to read
high.

## What actually drove it — and what did not

Gain shares for K-rank (`feature_importance.json`, 2,596,949 total gain):

| Family | Share of gain | Best member (overall rank) |
|---|---|---|
| **Freshness** (`article_age_h`, `article_age_rank_in_imp`, `article_age_minus_imp_min`, `is_fresh_6h`, `age_vs_user_appetite`) | **33.5%** | `article_age_h` (**#1**) |
| **Long-term interest** (`lt_*`) | 15.3% | `lt_topic_affinity` (#8) |
| Embedding similarity | 3.3% | `lt_embed_sim` (#16) |
| **Short-term interest** (`st_*`) | **2.1%** | `st_topic_affinity` (#27) |
| History popularity | 1.4% | `history_popularity_rank_in_imp` (#26) |
| Position (flagged) | 0.6% | `relative_position_in_view` (#31) |
| **Interest drift** (`st_* − lt_*`) | **0.2%** | `topic_affinity_drift` (#46) |

**Assumption A1 is falsified.** The long-term half of the BlackPearl-derived
decomposition carries real weight (15.3%), but the short-term half contributes
2.1% and the drift features 0.2% — `is_last_clicked_category` was never chosen
for a single split (gain exactly 0). The hierarchical long/short-term interest
structure reconstructed from BlackPearl's abstract is **not** what produces this
result on `ebnerd_small`.

What produces it is **article freshness judged relative to the other candidates
in the same in-view list**. That confirms the structural diagnosis this ADR
opened with — content similarity cannot discriminate among candidates that are
all topically adjacent, whereas *which of these particular items is freshest*
can — but it arrives through item/context features, not through user-interest
modelling. Stated plainly because it is evidence against the hypothesis the
session set out to test, not a footnote to it.

## The three flagged features did not carry the result

The caveats registered above were checked against measured importance rather
than left as assertions:

- `context_read_time` 1.4% (#19), `context_scroll_percentage` 0.3% (#37).
- Position features 0.6% combined — and withholding them **improved** AUC by a
  CI-clear +0.0011. The gain is therefore not "learned Ekstra Bladet's ranker";
  K-rank-nopos is the arm to quote if a reviewer objects, and it costs nothing.

## The popularity arm is NOT the challenge's popularity baseline

This must not be conflated. The challenge's "most clicks" baseline scores
**59.70** on the real test set. The `popularity` arm here scores **0.4269** —
*below random* — because it is a deliberately different construction: clicks
counted over the 21-day **history window** (the only leak-safe source available
at serving time, per the design note above). Articles with high history-window
click counts are by definition *older* articles, and in a news feed older is
anti-predictive. So this row measures "how good is stale popularity as a
ranker" (answer: worse than nothing), not "how good is the challenge baseline".
The two numbers are not comparable and are not compared.

This also explains the feature's low importance (1.4%): the model gets a far
better freshness-aware signal from `article_age_*`. Design-space option 2
(causal expanding-window in-view counts) is now the more interesting unexplored
direction, since it would capture *current* traction rather than stale traction.

## Objective choice: no meaningful difference

K-cls (pointwise binary, H2's framing) scored 0.7528 vs. K-rank's 0.7514 — a
difference of 0.0014, smaller than either CI's width. The listwise objective did
**not** justify itself on this data. Worth recording because the pre-registered
expectation was the opposite; with one positive per impression and ~12
candidates, the pointwise loss evidently has enough signal.

## Honest positioning against the reference points

| Reference | AUC | Candidate K vs. it |
|---|---|---|
| This project's deployed EB-NeRD submission (real Codabench) | 0.5404 | local validation is +0.21 higher — **but see the caveat below** |
| Challenge "most clicks" baseline (real test set) | 0.5970 | local validation is above it |
| Challenge winner, leakage-ablated (real test set) | 0.7699 | **0.7514–0.7528 local is just below this** |
| BlackPearl, 2nd place, as submitted (real test set) | 0.8815 | well below |

**0.80 was not reached, and this table does not claim it was.** More
importantly, **every Candidate K number here is local `ebnerd_small` validation,
while every reference number is the real blind test set.** They are not the same
measurement and putting them in one table is a positioning aid, not a
like-for-like comparison. This project has twice watched a CI-clear local win
compress at real test scale — Candidate G (+0.0027 local → −0.0003 real) and the
EB-NeRD contrastive vector (+0.0023 local → +0.0005 real). The prior here is
that the real leaderboard number will be **lower than 0.75**, possibly
substantially. The margin being +0.21 rather than +0.003 is the reason it is
still worth submitting; it is not a reason to expect it to survive intact.

## Submission pipeline validated end to end (not assumed)

`generate_ebnerd_gbdt_predictions.py` was dry-run against `ebnerd_small`'s
validation split treated as unlabelled — the same code path the blind test set
will take — and the resulting file was scored with the bundled
`evaluation/official/evaluate.py`:

| Metric | Official evaluator | This project's harness |
|---|---|---|
| **AUC** | **0.7524** | **0.752443** — exact |
| nDCG@5 | 0.5882 | 0.5882 — exact |
| nDCG@10 | 0.6261 | 0.6261 — exact |
| MRR | 0.5236 | 0.5242 | 

The MRR gap is the already-documented metric-definition difference (the official
script sums 1/rank over *all* clicked items; this project's `mrr` is
first-hit-only), not a converter defect — the same discrepancy recorded for the
MIND converter. Throughput: 8,443 impressions/s, which puts the real 13,536,710-
impression test set at roughly 27 minutes.

**One real defect caught by this dry run.** The two Codabench competitions
disagree on the filename inside the zip: EB-NeRD (competition 2469) requires a
root-level **`predictions.txt`** — that is what submission 888045 (Score 0.5404)
contained — while MIND requires the singular `prediction.txt`, which is also
what the bundled `evaluate.py` opens. The script now asserts the EB-NeRD form
explicitly rather than relying on it being right.

## Data artifact: candidates published after the impression showing them

The leakage integration test initially failed on a strict "age must never be
negative" assertion. Measured across the full bundle rather than assumed away:

| Split | Negative-age rows | Distinct articles | Worst | CTR in those rows vs overall |
|---|---|---|---|---|
| train | 641 / 2,585,747 (**0.025%**) | 4 | −0.7 h | 6.40% vs 9.04% |
| validation | 61 / 2,928,942 (**0.002%**) | 9 | −43.1 h | 1.64% vs 8.39% |

Thirteen articles across 5.5M candidate rows carry a `published_time` later than
an impression that showed them — most plausibly a revision/republish timestamp
rather than original publication.

**Not treated as an exploitable leak, on evidence:** click-through in those rows
is *lower* than baseline in both splits, so the anomaly is anti-correlated with
the label rather than predictive of it.

**Left untransformed, deliberately.** Clipping negatives to zero would make
exactly these rows look maximally *fresh* — the one direction the model rewards,
given freshness is 33.5% of total gain. Silently "fixing" the artifact would
therefore inject a small bias in the worst possible direction. The test now
bounds the rate (<0.1%) and the median age instead, so a genuine regression in
timestamp handling still fails loudly while the known artifact does not.

## Cost

Feature build 20s (train) / 21s (validation); training 42s (K-rank), 84s
(K-cls), 47s (K-rank-nopos). The dominant cost was the *evaluation* loop
(~18 min), which called `roc_auc_score` once per impression per arm in Python.
`per_impression_auc_fast` (Mann-Whitney identity, exact-match tested against
`safe_auc` including ties, multi-click and degenerate impressions) exists to
make the same measurement affordable at `ebnerd_large` scale.

---

# Revisit Conditions

Resolved this session:

- ~~If K-rank does not clear the local `popularity` baseline...~~ **Cleared it by
  +0.3245 (CI-clear).** Note the popularity arm is not the challenge baseline —
  see the results section.
- ~~If the position-bias ablation shows most of the gain comes from
  `position_in_view`...~~ **It does not.** Position contributes 0.6% of gain and
  withholding it *improves* AUC by +0.0011. No restatement needed; K-rank-nopos
  is quotable as-is.

Still open:

- **If local gains again fail to transfer to Codabench (a third occurrence),**
  that pattern becomes a primary finding for the design note in its own right —
  and would be the strongest evidence this project has produced that its local
  validation protocol systematically over-reads.
- **A1 is falsified, so the short-term/drift features (9 of 60) are now
  candidates for removal**, not extension. Any further feature work should go
  into freshness and within-impression relative signals, which is where the gain
  actually is — and into design-space option 2 (causal expanding-window in-view
  counts), the one leak-safe popularity construction not yet tried.
- **Whether `ebnerd_large` moves this at all.** The feature set is now known to
  lean on item/context signals rather than per-user history depth, which is
  precisely the kind of signal that may *not* benefit much from more users.

---

# References

- Kruse et al., *EB-NeRD: A Large-Scale Dataset for News Recommendation* — [arXiv:2410.03432](https://arxiv.org/html/2410.03432)
- *RecSys Challenge 2024: Balancing Accuracy and Editorial Values in News Recommendations* — [arXiv:2409.20483](https://arxiv.org/html/2409.20483v1)
- *Leveraging LightGBM Ranker for Efficient Large-Scale News Recommendation Systems* (FeatureSalad, 85.13 AUC) — [ACM 10.1145/3687151.3687156](https://dl.acm.org/doi/10.1145/3687151.3687156); code: [recsyspolimi/recsys-challenge-2024-ekstrabladet](https://github.com/recsyspolimi/recsys-challenge-2024-ekstrabladet)
- *Large Scale Hierarchical User Interest Modeling for Click-through Rate Prediction* (BlackPearl, 88.15 AUC) — [ACM 10.1145/3687151.3687163](https://dl.acm.org/doi/10.1145/3687151.3687163) — **full text not retrievable (HTTP 403)**
- Internal: ADR-007 (evaluation harness, train-only novelty), ADR-008 + addenda (embeddings), ADR-009 (leaky-feature ablation), ADR-010 (paired bootstrap, candidate-search discipline), ADR-012 (Candidate J)

---

# Addendum (2026-09-12, later) — A2 Q9: what the serving-time-unavailable features are worth

A2 Q9 requires metrics **with and without features unavailable at serving time**. This ADR
flagged `context_read_time` / `context_scroll_percentage` as "the weakest such claim in this
table" — EB-NeRD ships them in the test set and they are not future information about the
candidate, but a page's total read time is only fully known once the user *leaves* the page,
so a strict production loop would not have the final value at request time. The flag was
never quantified. A fourth arm, **`K_rank_noctx`**, now quantifies it.

This is a different category from ADR-009's `total_inviews` / `total_pageviews` /
`total_read_time`, which are forbidden outright and never read. These two are
admitted-but-flagged, which is exactly the case Q9 asks to be measured rather than argued.

| Arm | AUC | 95% CI | MRR | nDCG@5 | nDCG@10 | best_iter |
|---|---:|---|---:|---:|---:|---:|
| `K_rank` (**with**) | 0.7581 | 0.7564–0.7600 | 0.5299 | 0.5940 | 0.6309 | 563 |
| `K_rank_noctx` (**without**) | 0.7570 | 0.7552–0.7588 | 0.5290 | 0.5927 | 0.6298 | 246 |

**Paired bootstrap, identical impressions: +0.0011 AUC (95% CI +0.0008 to +0.0015).**

Reading, stated so the number is not oversold in either direction:

- The flagged features **do** help, CI-clear. The effect is real.
- It is also **negligible against the model's margin**: without them the ranker still scores
  0.7570, i.e. **+0.2140 over `embed_sim`** (CI-clear). The headline EB-NeRD result does not
  rest on features that are shaky at serving time, which is the substantive Q9 answer.
- **Contrast with the position ablation in this ADR**: withholding `position_in_view` /
  `relative_position_in_view` *improves* AUC by 0.0007. So of the two flagged groups, one is
  worth +0.0011 and the other is worth −0.0007. **Neither is load-bearing**; freshness is
  (`article_age_h` remains the top feature by gain at 398,688).
- `K_rank_noctx` early-stopped at **246 iterations against `K_rank`'s 563** — with two fewer
  usable signals the model plateaus sooner, which is consistent with a small real
  contribution rather than noise.

**Reproducibility evidence, obtained for free.** Adding the fourth arm left the other arms
**bit-identical** to the three-arm run: `K_rank` 0.758123 and `K_rank_nopos` 0.758808 in
both, exactly equal. The arms are independent and the pipeline is deterministic under a
fixed seed on CPU.

**Scope limit.** Measured on `ebnerd_small` validation (244,647 impressions), not on the
`ebnerd_large` test set that produced the 0.7542 Codabench score. The same ablation at that
scale is not run: the bundle is ~4.6 GB, EB-NeRD's S3 is ~18 KB/s from Ada, and Ada's `$HOME`
has ~4.9 GB free (ADR-015). Recorded as a bounded claim, not extrapolated.

Run cost: 510 s wall, peak RSS 1.79 GB, four arms. (The earlier three-arm run took 1,074 s;
the difference is OS page-cache warmth from that run, not a speedup — `article_table_s`
rounds to 0 here.) Artifacts: `experiments/candidate_k_gbdt_ebnerd_small_2026-09-12_q9/`.

---

# Addendum (2026-09-12) — Candidate K retrained at 65 features, and stored durably

**Why.** A2's Q4/Q5 need Candidate K's scores, and no trained K survived: Ada's `$HOME` no
longer holds the `ebnerd_large` booster (checked directly, 2026-09-11), and the only local
model was the **pre-correction 60-feature** one from 2026-08-26 that ADR-014 profiled. So K
was retrained locally on `ebnerd_small` with the current code.

**Result — the corrected feature set reproduces this ADR's post-correction numbers.**
65 features, including all five the correction added (`context_category_match`,
`context_topic_overlap`, `context_embed_sim`, `session_position`, `session_start_gap_h`).

| Arm | AUC | 95% CI | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|
| K_cls | 0.7597 | 0.7579–0.7615 | 0.5300 | 0.5939 | 0.6314 |
| K_rank_nopos | 0.7588 | 0.7571–0.7606 | 0.5295 | 0.5939 | 0.6306 |
| K_rank | 0.7581 | 0.7564–0.7600 | 0.5299 | 0.5940 | 0.6309 |
| embed_sim | 0.5430 | 0.5415–0.5444 | 0.3437 | 0.3804 | 0.4591 |
| random | 0.4993 | 0.4979–0.5006 | 0.3129 | 0.3447 | 0.4297 |
| popularity | 0.4269 | 0.4252–0.4288 | 0.2596 | 0.2805 | 0.3804 |

This lands in the 0.7581–0.7597 band this ADR recorded after the recency fix, and the three
baselines reproduce their A1 values exactly (`embed_sim` 0.5430 matches ADR-008 addendum 2;
`random` 0.4993; `popularity` 0.4269). The harness is the same instrument as before.

**Position bias, re-measured.** `K_rank` vs `K_rank_nopos` is **−0.0007 (95% CI −0.0011 to
−0.0003)**: withholding `position_in_view`/`relative_position_in_view` still *helps*,
CI-clear, in the same direction as the original +0.0011 though smaller. `K_rank_nopos`
therefore remains the quotable arm.

**Feature gain still says freshness, not long/short-term interest.** `article_age_h` is the
top feature by a wide margin, followed by `article_age_minus_imp_min` and
`article_age_rank_in_imp`. Unchanged conclusion from this ADR's third finding.

**Cost:** 1,073.7 s (17.9 min) wall, peak RSS 2.07 GB on the 8 GB M3 — comfortably local, no
cluster needed.

**Durability (the failure this addendum exists to prevent).** The `ebnerd_large` booster was
lost because it existed only on Ada. This model is kept in **three** places:
`experiments/candidate_k_gbdt_ebnerd_small_2026-09-12/` (gitignored but on local disk),
`~/a2_model_artifacts/candidate_k_ebnerd_small_2026-09-12/`, and Ada
`$HOME/a2/artifacts/candidate_k_ebnerd_small_2026-09-12/`, sha256-verified across copies.

**Consequence for ADR-014.** Its stated caveat — that the EB-NeRD profile used the
60-feature pre-correction booster — can now be closed: the 65-feature model exists locally.
Re-profiling EB-NeRD against it is A2 Q4 work.

**Not retrained: `ebnerd_large`.** That needs the ~4.6 GB bundle, and EB-NeRD's S3 runs at
~18 KB/s from Ada while Ada's `$HOME` has ~4.9 GB free. It would mean downloading locally and
pushing, plus staging on a compute node's `/ssd_scratch`. Deferred, and recorded here rather
than silently skipped.
