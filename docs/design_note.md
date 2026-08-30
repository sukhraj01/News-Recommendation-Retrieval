# Design Note — Lexical vs. Semantic Retrieval for News Recommendation

**CS4.406 Information Retrieval & Extraction — Assignment 1**

---

## 1. What We Built

News recommendation differs structurally from catalog domains like retail
or streaming: item relevance half-life is hours, not months, so item
cold-start is the *default* state, not an edge case (Wu et al. 2020;
Kruse et al. 2024). This motivates content-based candidate retrieval —
lexical (BM25) and semantic (embeddings) — over collaborative filtering.

This project implements both retrieval methods, plus one shared evaluation
harness, over MIND and EB-NeRD under a unified schema and temporal-split
protocol (ADR-001, ADR-002), so the two methods and two datasets are
compared on identical footing rather than via method-specific pipelines.
Both methods were pushed through to real Codabench leaderboard submissions
on both competitions (§3.5). Beyond that required baseline, each dataset's
weakest link was then chased with a dedicated follow-on candidate — a
trainable NRMS-lite encoder for MIND (§3.6) and a LightGBM learning-to-rank
model over engineered behavioural/temporal features for EB-NeRD (§3.7) —
both of which became this project's real leaderboard wins.

## 2. Choices We Made

**Data & splits.** Official train/validation splits are used as-is
(7-day windows) — the only feasible option given data budgets (ADR-001).
A unified three-table schema (articles, impressions, user_history) with
mandatory core + dataset-specific optional fields lets one retrieval and
evaluation code path run against both datasets (ADR-002).

**Lexical retrieval.** BM25Okapi over unweighted concatenated
title+abstract query text (full click history, stopwords removed), scored
as a precomputed sparse weight matrix rather than `rank_bm25`'s naive
`get_scores()` — the latter measured at ~10hr projected vs. ~80s actual at
MINDsmall-dev scale, a real infeasibility, not a style preference
(ADR-005, ADR-006).

**Semantic retrieval.** One embedding model computed over both datasets
(rejecting EB-NeRD's provided embeddings + a separate MIND model, to keep
the single-code-path comparison valid) — `paraphrase-multilingual-MiniLM-L12-v2`,
chosen over `multilingual-e5-small` after a same-category vs.
different-category cosine-similarity discrimination check: MiniLM's gap
(0.069, 2.4x ratio) is real and usable; e5-small's `passage:`-prefixed
convention compresses nearly all pairs into a narrow 0.75–0.78 band (gap
0.017, ratio 1.02x) — e5's asymmetric query/passage objective doesn't fit
this project's symmetric user-profile-vs-article use case (ADR-008). ANN
backend is brute-force cosine similarity — 0.99ms/query at MINDsmall
scale, 2.44ms/query at MINDlarge's 255,990 users — FAISS stays unjustified
at this scale (ADR-008 + Addendum; revisited at 10x scale, §5). User
representation is mean-pooled, L2-renormalized embeddings of full click
history; zero-history users get an explicit all-zero vector, mirroring
BM25's own cold-start reporting posture.

Rejecting EB-NeRD's provided embeddings (above) was originally an argued
trade-off, not a measured one. An isolated follow-up comparison (ADR-008
Addendum, 2026-08-19) measured it directly on `ebnerd_small` validation,
reusing the identical user-representation and evaluation code with only the
embedding source swapped: EB-NeRD's provided `contrastive_vector` artifact
is a CI-clear win on recall@K (0.58%/1.20%/2.47% vs. MiniLM's
0.14%/0.43%/1.21%) and on every Q4 accuracy metric (AUC 0.5453 vs. 0.5430,
MRR 0.3527 vs. 0.3437, nDCG@5 0.3870 vs. 0.3804, nDCG@10 0.4660 vs. 0.4591),
but a CI-clear *loss* on Diversity@10 (0.780 vs. 0.789) and a tie on
Novelty@10 — a real trade-off, not a blanket verdict, and one that applies
to EB-NeRD only (the artifact doesn't cover MIND, so it can't replace this
project's single cross-dataset embedding space either way).

**Evaluation.** Two distinct questions are measured, not conflated:
*recall@K* (does retrieval surface the clicked article out of the *whole*
corpus, K ∈ {50,100,200}) and *Q4's ranking metrics* — AUC, MRR, nDCG@5,
nDCG@10, intra-list diversity, novelty, catalog coverage — computed over
the candidates *already listed in each impression*. Ties are broken by a
deterministic per-impression-seeded pseudo-random permutation (cold users
produce exact ties); K=10 for diversity/novelty. Every metric reports 95%
bootstrap CIs (n=2000), sliced by warm/cold user (history length ≥5 =
warm), except catalog coverage, reported as a point estimate only — a
set-union statistic is mechanically biased under with-replacement
resampling (ADR-007).

## 3. Observations

### 3.1 Recall@K — candidate generation over the whole corpus

| Dataset (split) | Method | recall@50 | recall@100 | recall@200 |
|---|---|---|---|---|
| MIND-small dev, overall | BM25 | 0.73% | 1.50% | 2.62% |
| MIND-small dev, overall | Embed | 0.90% | 1.60% | 2.78% |
| MIND-small dev, warm | BM25 | 0.73% | 1.54% | 2.73% |
| MIND-small dev, warm | Embed | 0.93% | 1.67% | 2.92% |
| MIND-small dev, cold | BM25 | 0.72% | 1.19% | 1.78% |
| MIND-small dev, cold | Embed | 0.65% | 1.10% | 1.81% |
| EB-NeRD-demo val (warm only\*) | BM25 | 1.01% | 2.13% | 4.02% |
| EB-NeRD-demo val (warm only\*) | Embed | 0.32% | 0.94% | 2.57% |
| EB-NeRD-small val (warm only\*) | BM25 | 0.72% | 1.44% | 2.77% |
| EB-NeRD-small val | Embed | 0.14% | 0.43% | 1.21% |

\*EB-NeRD's active-user-filtered bundles have zero cold-start users
(min history length = 5) by construction. Source:
`experiments/{bm25,embed}_{mind,ebnerd,ebnerd_small}_2026-08-10/results.json`.
Not run at MINDlarge scale — flagged in §6, not approximated.

### 3.2 Q4 ranking metrics — AUC / MRR / nDCG (bootstrap 95% CI)

| Dataset | Method | AUC overall (CI) | AUC warm | AUC cold |
|---|---|---|---|---|
| MIND-small dev | BM25 | 0.569 (0.567–0.571) | 0.577 | 0.524 |
| MIND-small dev | Embed | 0.634 (0.632–0.636) | 0.644 | 0.574 |
| MINDlarge dev | BM25 | 0.570 (0.569–0.571) | 0.578 | 0.522 |
| MINDlarge dev | Embed | 0.634 (0.633–0.634) | 0.643 | 0.575 |
| EB-NeRD-demo val | BM25 | 0.533 (0.528–0.537) | — (=overall) | n/a |
| EB-NeRD-demo val | Embed | 0.544 (0.539–0.549) | — | n/a |
| EB-NeRD-small val | BM25 | 0.529 (0.527–0.530) | — | n/a |
| EB-NeRD-small val | Embed | 0.543 (0.541–0.544) | — | n/a |

On MIND (both scales), embeddings' AUC CI does not overlap BM25's on
either cohort. The warm/cold gap is *not* similar between methods: BM25's
is Δ0.052–0.056, embeddings' is Δ0.068–0.070 — embeddings lift warm AUC by
~0.065–0.067 over BM25 but cold by only ~0.050–0.053, widening the
relative warm/cold disparity rather than closing it.

On EB-NeRD, embeddings' AUC CI also does not overlap BM25's — a modest
(~0.011–0.014) edge — despite BM25 winning recall@K by 1.6–2.3x on the
*same* corpus (§3.1). These measure different things: recall@K asks
whether the click is anywhere in the whole catalog; AUC asks how well the
method ranks EB-NeRD's already-curated, much shorter candidate lists
(median 9–12 vs. MIND's 23). A plausible (not isolated) explanation:
EB-NeRD's long median history (81–93 articles) produces a diffuse
mean-pooled query that struggles against the whole catalog but still
discriminates within a pre-curated short list.

MRR/nDCG@5/nDCG@10: embeddings lead BM25 by a similar margin on MIND (e.g.
MINDlarge nDCG@10: 0.390 vs. 0.350) and a smaller, mixed margin on EB-NeRD
(e.g. EB-NeRD-small nDCG@10: 0.459 vs. 0.454). Source:
`experiments/ranking_{bm25,embed}_mind{,_large}_2026-08-1{0,1}/results.json`.

### 3.3 Diversity / novelty / coverage (K=10, point estimate for coverage)

| Dataset | Method | Diversity@10 | Novelty@10 | Coverage@10 |
|---|---|---|---|---|
| MIND-small dev | BM25 | 0.837 | 16.30 | 0.083 |
| MIND-small dev | Embed | 0.827 | 16.15 | 0.078 |
| MINDlarge dev | BM25 | 0.836 | 18.09 | 0.067 |
| MINDlarge dev | Embed | 0.828 | 17.76 | 0.065 |
| EB-NeRD-small val | BM25 | 0.795 | 17.17 | 0.206 |
| EB-NeRD-small val | Embed | 0.789 | 17.19 | 0.205 |

BM25 and embeddings sit within ~1 point of each other on every
beyond-accuracy metric, on every corpus — accuracy differs meaningfully
between methods, beyond-accuracy behavior barely does (n up to 376,471
impressions, so even a 1-point gap is not automatically noise, but it is
not the large, consistent effect the AUC results show).

### 3.4 System cost

BM25 at MINDlarge scale: 1.70s index build, 23.5MB sparse weight matrix,
~9.8 min projected full retrieval for 255,990 users (ADR-006 Addendum).
Embeddings at the same scale: 264.4s to encode+cache all 72,023 articles,
~10.4 min projected full retrieval (ADR-008 Addendum). Both stay local on
an 8GB machine at this scale — only EB-NeRD's *blind test* split (13.5M
impressions, no local ground truth) required Kaggle.

### 3.5 Leaderboard results (Q5)

| Competition | Method | Score | Submission ID | Date |
|---|---|---|---|---|
| MIND (competitions/13967) | Embed (MiniLM) | 0.6195 | 886468 | 2026-08-12 13:01 |
| MIND (competitions/13967) | Cohort-gated combiner (§ADR-010 Addendum) | 0.6192 | 896696 | 2026-08-22 |
| MIND (competitions/13967) | **NRMS-lite, Candidate J (§ADR-012)** | **0.6462** | 901961 | 2026-08-26 06:58 |
| MIND (competitions/13967) | NRMS-lite, Candidate J, corrected catalog (§ADR-012) | 0.6462 (identical) | not separately recorded\*\* | 2026-08-26 |
| EB-NeRD (competitions/2469) | Embed (MiniLM) | 0.5404 | 888045 | 2026-08-13 23:08 |
| EB-NeRD (competitions/2469) | Contrastive vector (§ADR-008 Addendum) | 0.5404 | 896072 | 2026-08-21 13:01 |
| EB-NeRD (competitions/2469) | **GBDT ranker, Candidate K (§3.7/ADR-013)** | **0.7542** | 907863 | 2026-08-29 23:01 |

\*\*The corrected-catalog resubmission's score was confirmed by the engineer directly on the
Codabench leaderboard but, unlike every other submission in this table, no distinct submission
ID or screenshot was captured for it (`submissions/mind_large_test_nrms_lite_corrected/` holds
only the prediction files, not a screenshot) — recorded here as a real, minor gap in this
project's own screenshotting discipline, not smoothed over (ADR-012).

Both submissions used the embedding method — the clear winner on local
dev-set AUC for both datasets (§3.2). EB-NeRD's first leaderboard score
(0.5404) sits close to this project's own local validation AUC for
embeddings on `ebnerd_small` (0.5430) — an informal coherence check that
the submission pipeline behaves consistently between local validation and
the real held-out test set, not a claim the two numbers are defined
identically (column headers were cropped from the available screenshots,
so Score≈AUC is inference from the numeric match, not a confirmed label —
screenshots in `submissions/{mind_large_test_embed,ebnerd_testset_embed}/`).

**Second MIND submission — cohort-gated combiner, essentially flat, not
the local win.** Local validation predicted a real, CI-clear win at every
stage checked before submitting: +0.0027 on MINDsmall-dev (95% CI +0.0018
to +0.0034), +0.0019 on a MINDlarge-dev re-verification (95% CI +0.0015 to
+0.0023, ADR-010's addendum). The real leaderboard result was **0.6192
vs. 886468's 0.6195 — a -0.0003 difference**, essentially flat, not the
win either local check predicted. No defect was found in the submission
pipeline (format verified with the same discipline as every prior
submission — exact line count, zero malformed permutations, all 32
`N89741`-affected impressions individually spot-checked). The most likely
explanation is a genuine population difference: MIND's official dev and
test splits are different, non-overlapping calendar weeks (ADR-001), and
this method's edge was built around properties specific to the
MINDlarge-dev population (a warm/cold cohort mix, coefficients fit on
MINDsmall-train's click patterns) rather than a large, population-
independent effect. This is the same category of finding as EB-NeRD's
contrastive-vector submission below — a CI-clear local win compressing
substantially at real blind-test scale — now observed twice, on two
different datasets, worth reading as a real pattern about this project's
validation methodology rather than two unrelated surprises (ADR-010's
2026-08-22 addendum).

**Third MIND submission — NRMS-lite (Candidate J), a real leaderboard
win.** Unlike the cohort-gated combiner above, this candidate is a
genuinely different architecture (§3.6): trainable self-attention title
and click-history encoders (Wu et al. 2019), GloVe-initialized, trained
on a dedicated GPU (institutional HPC access obtained specifically for
this, after the earlier candidate search had been closed for lack of
compute). Real trail across three checkpoints: MINDsmall-dev 0.6391 (95%
CI 0.6370–0.6412, vs. baseline 0.6340) → MINDlarge-dev re-verification
0.6579 (95% CI 0.6569–0.6588, vs. baseline 0.6335) — a margin that
**widened**, not compressed, the only candidate in this project where
that happened → real Codabench leaderboard **0.6462**. Some compression
did occur at the final, real step (0.6579 → 0.6462), consistent with
this project's repeated finding below that local numbers don't fully
transfer to the real blind test — but unlike the cohort-gated combiner
or the EB-NeRD contrastive result, the result stayed a real, substantial
win rather than compressing to flat: **+0.0267 over the original
baseline submission, +0.0270 over the cohort-gated combiner.** A
narrow-impact bug (32 of 2,370,727 impressions, 0.0013% — the scoring
catalog initially included train+dev articles rather than test's own
only, misapplying MIND's documented `N89741` missing-candidate handling)
was found via this project's standard spot-check discipline and fixed;
practical impact was initially assessed as negligible (32/2,370,727,
0.0013%) from checking only the one previously-documented example, before
deciding to submit the original prediction set rather than block on a
rerun. A corrected re-run was later diffed directly against the original,
not re-estimated: the real measured impact was **2,087/2,370,727 (0.088%,
~65x the initial estimate)** — a real correction to the "negligible" claim,
not a confirmation of it. Given the larger real number, the engineer
resubmitted the corrected prediction set as a fourth MIND entry; it scored
**0.6462 — identical to the original submission**, empirically closing the
loop that the catalog bug, while real and worth fixing at the root, never
put the submitted result in question. Full detail, including the
literature-motivated hypothesis this candidate tested (a trainable text/
history encoder plus pretrained embeddings, missing from every prior MIND
attempt) and the real HPC/infrastructure work involved, is in ADR-012.

**Second EB-NeRD submission — contrastive vector on the real test set.**
Both EB-NeRD submissions round to an identical Score column (0.5404).
Before drawing any conclusion from that, the two `predictions.txt` files
were diffed directly rather than assumed distinct: different SHA-256/
CRC-32, and 99.48% of the 13,536,710 impressions carry a different ranking
for the same impression ID (0 ID misalignments; both files independently
pass the exact line-count/malformed-permutation check the submission
pipeline itself uses before packaging). These are genuinely two
independent scoring runs, not a duplicate upload — `load_contrastive_index`
was correctly substituted for `build_embedding_index` in the test-set run,
per ADR-008's Addendum.

Codabench's per-submission detail view (AUC/MRR/nDCG@5/nDCG@10, grouped by
date, covering the stated "50% of the testset") resolves the tie the
rounded Score column hides:

| Date | MiniLM AUC | Contrastive AUC |
|---|---|---|
| 2023-06-01 | 0.5374 | 0.5597 |
| 2023-06-02 | 0.5422 | 0.5533 |
| 2023-06-03 | 0.5755 | 0.5504 |
| 2023-06-04 | 0.5444 | 0.5226 |
| 2023-06-05 | 0.5296 | 0.5314 |
| 2023-06-06 | 0.5330 | 0.5489 |
| 2023-06-07 | 0.5316 | 0.5231 |
| 2023-06-08 | 0.5240 | 0.5323 |
| **MEAN** | **0.5397** | **0.5402** |

Contrastive leads on 5 of 8 days and on the mean, but the margin is thin —
mean AUC **+0.0005**, with the per-day sign flipping three times (MiniLM
ahead on 06-03, 06-04, 06-07). MRR/nDCG@5/nDCG@10 point the same direction
on their means (MRR 0.3553 vs. 0.3452; nDCG@5 0.3898 vs. 0.3825; nDCG@10
0.4695 vs. 0.4615 — not tabulated per-day here for space), but the AUC
margin is the one worth stating honestly: it is far smaller than the
**~0.0023 CI-clear gap** (0.5453 vs. 0.5430, 95% CI 0.5435–0.5471) ADR-008's
Addendum measured on `ebnerd_small` local validation. A margin this thin on
a single 50%-of-testset sample is consistent with a real but small local
edge, not with a dramatically larger real-test-set win — a second data
point in the same direction as local validation, not confirmation of a
large effect.

### 3.6 MIND candidate search: ten approaches, and what finally worked

Before Candidate J (§3.5) — the eventual real leaderboard win — ten
other approaches were screened on MINDsmall-dev against the deployed
baseline (AUC 0.634, §3.2), all scoring on top of *frozen* sentence
embeddings with a shallow head. Only G's cohort-gated routing cleared a
CI-clear win overall (and didn't hold at real leaderboard scale, §3.5),
and only F's raw combiner cleared one on the cold cohort alone. Full
methodology, evidence, and interpretation for each candidate is in
ADR-010 (D–H), ADR-011 (I), and ADR-012 (J).

| Candidate | Method | Result vs. baseline |
|---|---|---|
| A | Entity embeddings (TransE, confidence-weighted pooling) | 0.553, CI-clear loss |
| B | Untuned 50/50 BM25+embedding hybrid | 0.626, CI-clear loss |
| C | Recency-weighted (decay=0.9) embedding query | 0.627, CI-clear loss |
| D | Symbolic category/entity overlap score | 0.613, CI-clear loss |
| E | Train-split popularity only, no personalization | 0.532, CI-clear loss |
| F | 7-feature `LogisticRegression` combiner | 0.626 overall (loss); 0.593 cold (CI-clear win) |
| G | Cohort-gated routing (deployed embed + F for cold) | 0.637 local (CI-clear win); 0.6192 real leaderboard (flat, §3.5) |
| H1 | `LogisticRegression`, class-balanced | 0.632, CI-clear loss |
| H2 | `HistGradientBoostingClassifier` | 0.599, CI-clear loss |
| I | Attention re-ranker, 5 epochs (first trained model) | 0.623, CI-clear loss |
| I-long | Same, 30 epochs (holdout AUC climbed to 0.637) | 0.621 dev, worse than the 5-epoch run |
| I-pop | I-long + an unnormalized popularity feature | 0.513, confounded/inconclusive (see ADR-011) |
| **J** | **Trainable title+history encoders, GloVe-initialized (Wu et al. 2019)** | **0.658 MINDlarge-dev (CI-clear win); 0.6462 real leaderboard (real win, §3.5)** |

Every candidate through I shared the same structural ceiling: a shallow
head over *frozen* embeddings, never a trainable text/history encoder.
Candidate J tested that specific hypothesis directly — and, distinctly
from every other candidate here, needed dedicated GPU compute (obtained
after this search had already been closed once for lack of it) to do so.
The result: the only candidate in this list whose local win *widened*
rather than compressed moving to larger/real scale, and the only one
whose real leaderboard result was a substantial win rather than flat.

### 3.7 EB-NeRD candidate search: Candidate K (GBDT learning-to-rank)

EB-NeRD's deployed leaderboard score (0.5404, §3.5) sits *below* the
challenge's own popularity baseline (0.5970) — content-similarity retrieval
plateaus because in-view candidates are drawn from the same front page at
the same moment, so they are already topically adjacent and recently
published, leaving little for cosine similarity to discriminate on
(measured: median candidate age 3.7h, 58% under 6h). Unlike MIND's search
(§3.6, ten scored variants), EB-NeRD's search is a single candidate —
Candidate K, a LightGBM learning-to-rank model over 65 engineered
behavioural/temporal features — because the RecSys Challenge 2024
organizers' own report states most real competitors used GBDT ensembles
over engineered features, not neural architectures, and this project's own
Option-1 (better embeddings) and Option-2 (NRMS-style, as Candidate J did
for MIND) alternatives were both rejected on evidence before building K
(ADR-013's Design Space Exploration).

**A real per-impression bug was found and fixed mid-session.** The
short-term (recency-decayed) features were initially computed once per
*user* against the whole split's reference time, not once per *impression*
— effectively static, not capturing genuine short-term dynamics. Prompted
by the engineer questioning whether recency was used properly, the fix
(`compute_short_term_features`, referenced to each impression's own
timestamp) was verified to change behavior on a synthetic case where it
previously couldn't, and a second, independently-found leak-safety gap in
the same code path (no upper bound excluding a future-dated history click)
was closed before re-running. Local `ebnerd_small` validation moved from
**0.7514–0.7528 (pre-fix) to 0.7581–0.7597 (post-fix)**, a CI-clear
improvement — not because recency stopped mattering, but because the
feature now actually measured it (ADR-013's 2026-08-27 Addendum).

**Trained at real `ebnerd_large` scale on Ada** (IIIT-H's SLURM HPC
cluster): 12,063,890 train / 12,566,385 validation impressions — the full
validation set, scored via a streaming path, not a sample. Result: local
validation AUC **0.7590 (95% CI 0.7588–0.7593)**, essentially unmoved from
the `ebnerd_small` screen despite ~10.7x more training data — a real null
result on scale for this feature set, consistent with the finding below
that the model's edge is item-level (freshness, immediate context) rather
than deepening per-user history modelling (ADR-013's 2026-08-29 Addendum).

**Real Codabench result: submission 907863, Score 0.7542** (§3.5) —
**+0.2138** over the previously deployed EB-NeRD submission and **+0.1572**
over the challenge's own popularity baseline, the first time this
project's EB-NeRD line has beaten either. The compression from local
`ebnerd_large` validation (0.7590 → 0.7542, −0.0048) is small and honest —
the first EB-NeRD/MIND candidate in this project whose CI-clear local win
transferred to the real leaderboard largely intact, rather than evaporating
(MIND's cohort-gated combiner, §3.5) or thinning to near-nothing (EB-NeRD's
contrastive vector, §3.5). The most likely reason: this candidate's edge
comes from structural, leak-safe item-level signal, not a subtle pattern
specific to the validation population (ADR-013's 2026-08-30 Addendum).

**What actually drove the result.** Feature-importance at real
`ebnerd_large` scale: freshness features (led by `article_age_h`, the
single top feature throughout) contribute **33.3%** of model gain;
long-term interest 14.1%; **short-term interest only 0.8%** (down from an
already-small 1.9% pre-large-scale, confirming the direction rather than
reversing it). The BlackPearl-derived long/short-term hierarchical
interest hypothesis this candidate was originally built to test is
therefore **falsified**, with more confidence after the recency fix and at
scale, not less: what wins is *which candidate is freshest relative to the
others in the same in-view list* and *what the user is reading right now*
(`context_embed_sim`, the single strongest of the newly-added context
features), not modelled long/short-term user interest. Withholding
`position_in_view`/`relative_position_in_view` *improved* AUC by a
CI-clear +0.0011, so the result is not "learned Ekstra Bladet's own
ranker" — the shipped arm (`K_rank_nopos`) withholds them and is quotable
without that caveat.

## 4. Anti-Gaming and Leakage (Q9)

Two separate Q9 obligations, both addressed directly rather than only
described:

**Serving-time-unavailable features — measured, not just avoided.**
Novelty's popularity signal is computed from the **train split only**,
Laplace-smoothed, never from the evaluation split (ADR-007). More
specifically, EB-NeRD's raw `articles.parquet` carries three lifetime
aggregate fields — `total_inviews`, `total_pageviews`, `total_read_time`
— that are unavailable at serving time (they include exposure/clicks that
happen after any given impression). `src/datasets/ebnerd.py` never reads
them into the unified schema (grep-verified). To go beyond documenting the
exclusion, `scripts/run_leakage_ablation.py` measures what including them
would have bought: an untuned 50/50 blend of the (already-deployed)
embedding score with a min-max-normalized leak signal built from these
three fields, evaluated on `ebnerd_small` validation (ADR-009):

| Metric | Without leaky features | With leaky features (50/50 blend) | CIs overlap? |
|---|---|---|---|
| AUC | 0.5430 (0.5415–0.5445) | 0.5662 (0.5646–0.5678) | No |
| MRR | 0.3437 (0.3421–0.3452) | 0.3638 (0.3623–0.3653) | No |
| nDCG@5 | 0.3804 (0.3784–0.3823) | 0.4034 (0.4016–0.4051) | No |
| nDCG@10 | 0.4591 (0.4574–0.4607) | 0.4783 (0.4767–0.4799) | No |

Every metric moves by a CI-clear margin from exposure to these three
fields alone (AUC +0.023 absolute, ~4.3% relative), with no tuning of the
blend weight. ~48% of articles were missing at least one raw field and
were median-imputed rather than zero-filled — since low-exposure articles
plausibly have below-median true popularity, this measured gap is more
likely a conservative *underestimate* of the leak's real effect than an
overestimate. This is direct, measured evidence — not just a stated rule
— for why these fields are absent from the schema entirely (ADR-009).

**Leakage-boundary test.** `tests/integration/test_leakage.py` asserts no
user's history click occurs at or after their own earliest impression
time. **This test exists and passes for EB-NeRD**, run on both the demo
train and validation splits. **It is explicitly skip-marked for MIND**
(`test_mind_no_impression_precedes_its_own_history_cutoff`), with an
inline reason: MIND provides only a static per-user article-ID list for
history, with no per-click timestamps, so there is no way to
independently verify the temporal boundary for MIND from the data itself.
This is a real, acknowledged gap in test coverage for MIND, not an
oversight silently left for a grader to find — MIND's temporal safety
instead rests on the dataset's own documented split construction (ADR-001),
not on an executable check in this repository.

## 5. Where It Breaks at 10×

Every real scaling failure found in this project so far was invisible at
the previously-exercised scale and appeared at roughly a 5–14x jump, not
a 2x one — a pattern, not a one-off:

- **MINDsmall → MINDlarge** (~5x users: 50,000 → 255,990; ~14x exploded
  impression-candidate rows) hung indefinitely in a per-token Python loop
  never exercised at MINDsmall's row counts; the fix's first version
  projected to **~27GB** on this 8GB machine (measured directly, not
  estimated) from one Python object per repeated ID string, fixed via
  categorical dtype (12.5x measured reduction, ~2.2GB); a second, separate
  bottleneck (`series.map()` on categorical data, >1hr with zero progress)
  was found only by profiling the actually-stuck process.
- **MINDlarge_test (2.37M impressions) → ebnerd_testset (13.5M
  impressions, 5.7x further)** surfaced a third, structurally identical
  failure mode: a converter that materialized a full `list[dict]` before
  writing anything projected to ~16GB, invisible at the smaller scale.
  Fixed by making the read path a generator, consumed one row at a time.
- **`ebnerd_small` (232,887 train impressions) → real `ebnerd_large`
  training on Ada (12,063,890 train impressions, ~52x further) hit a
  real, SLURM-confirmed OOM** — a fourth occurrence of the same shape.
  Root cause: the un-split 65-feature train matrix (`fit_X`/`stop_X`,
  ~11.5GB) was never freed once training finished, and the arm withholding
  two features (`K_rank_nopos`) required a column-sliced *copy* (numpy
  always copies for fancy-indexed column selection) that coexisted with
  it — ~22.7GB from four arrays alone, invisible at `ebnerd_small`'s
  scale. Fixed three ways: the code now frees `fit_X`/`stop_X` once the
  arm loop finishes (grep-confirmed they're never read again),
  `--train-sample-impressions` was cut from 4,000,000 to 2,500,000, and
  `--mem-per-cpu` was raised 4G→8G. Getting this run onto Ada also
  surfaced four further real, non-modelling engineering incidents in the
  same session (a missing `article_id` column in the blind test set, a
  lightgbm+torch import segfault, a GPU-idle job-cancellation pattern
  from a throttled download, and one `#SBATCH`-ordering bug caught before
  it shipped) — see ADR-013's 2026-08-29 Addendum for the full account.

**Projecting one more 10x from the largest scale currently verified**
(MINDlarge: 255,990 users / 72,023 articles; EB-NeRD: 13.5M impressions)
is inference, not measurement, but the historical pattern above makes two
things predictable: (1) BM25/embedding *retrieval time* scales roughly
linearly with user count, so 10x users alone projects to ~98 min (BM25)
and ~104 min (embeddings) — still locally feasible on time alone; but
(2) brute-force cosine similarity is O(users × articles) — if the
*corpus* also grows 10x alongside users (plausible at real news-scale,
not guaranteed), that's 100x compute, not 10x, projecting ~10.4 min to
roughly **17 hours** — this would force revisiting ADR-008's "FAISS
unjustified" call, not just re-running the same code longer. (3) Every
real bug found so far shared one shape — a Python-object-per-row or
full-materialization pattern that only breaks past a scale nothing had
exercised yet. Nothing in this codebase guarantees another unaudited path
doesn't share that shape at a further 10x; this is a documented risk, not
a resolved one.

## 6. Limitations and Open Questions

- **EB-NeRD's real result (0.7542, §3.7) did not reach this project's own
  internally-discussed 0.80 target — stated plainly, not rounded up.**
  That target itself was set with reference to a literature figure that
  turned out to be leakage-dependent: the RecSys Challenge 2024
  organizers' own report (arXiv:2409.20483) shows the winning team scored
  89.24 as submitted (88.64 in their own ablation-arm-with-features), but
  dropped to **76.99** once the organizers removed features later found
  to leak future information — an 11.65-point collapse. That leakage-free
  76.99 is the honest ceiling this project can defensibly compare
  against, not the unablated 80+ figures; this project's own 0.7542 sits
  **0.0157 below that honest ceiling**, closing most of the gap to a
  real, leakage-audited top-tier result using only features that passed
  the same per-feature leakage audit this project holds every candidate
  to (ADR-013's opening Reference Points table and 2026-08-30 Addendum).
  No published leakage-free score exists for 2nd place (BlackPearl, 88.15
  as submitted) — the organizers only ablated the winner.
- **True (zero-history) cold-start is a shared ceiling, not something
  either method solves.** MIND's cold-cohort recall@200 is nearly
  identical between methods (BM25 1.78% vs. embed 1.81%).
- **EB-NeRD's cold cohort is structurally empty** in both `demo` and
  `small` bundles (min history length = 5, by construction) — no
  warm/cold comparison is possible for EB-NeRD in this project;
  `ebnerd_large` was never downloaded/verified and might differ.
- **Recall@K was never run at MINDlarge scale** (§3.1) — only Q4's
  ranking metrics were.
- **The Q9 leakage-boundary test is real but partial** (§4): it exists
  and passes for EB-NeRD, and is explicitly skip-marked for MIND because
  MIND's history has no per-click timestamps to check against — stated
  here directly rather than left implicit in the skip reason alone.
- **The Q9 ablation's magnitude is specific to an untuned 50/50 blend**
  (§4) — it demonstrates a real, CI-clear effect and a defensible lower
  bound, not the maximum a calibrated leaky model could extract.
- **The encoder choice is verified via a category-based discrimination
  proxy, not a curated near-duplicate/paraphrase set** — a real but weak
  signal for true topical relevance (ADR-008, Confidence: Medium-High).
- **The hand-built Danish+English stopword list** used in BM25 query
  construction has not been validated against an NLP resource (ADR-005).
- **The EB-NeRD recall-vs-AUC disagreement is not fully explained**
  (§3.2): BM25 wins whole-corpus recall by 1.6–2.3x; embeddings edge out
  BM25 on the already-curated candidate-list AUC. The diffuse-mean-pooled-
  query explanation offered is plausible but not isolated by a controlled
  experiment.
- **Rejecting EB-NeRD's provided embeddings has a now-measured cost, not
  just an argued one.** ADR-008's Addendum shows the provided
  `contrastive_vector` artifact is a CI-clear win on `ebnerd_small`
  recall@K and every Q4 accuracy metric, but a CI-clear *loss* on
  Diversity@10 and a tie on Novelty@10 — a real trade-off, not resolved in
  either direction here. No public documentation of that artifact's
  training methodology (base model, objective, text fields) was found
  anywhere, including inside the archive itself. Whether to also submit a
  second, EB-NeRD-only leaderboard entry using it is an open decision for
  the engineer, not something this note resolves.
- **Local CI-clear validation wins compress at real blind-test scale —
  but do not always vanish, and one predicted the real result correctly
  in direction, not just optimistically.** MIND's cohort-gated combiner
  (§3.5) showed a CI-clear win at both a MINDsmall-dev screen (+0.0027)
  and a MINDlarge-dev re-verification (+0.0019), then landed essentially
  flat on the real leaderboard (-0.0003, 896696 vs. 886468). EB-NeRD's
  contrastive-vector submission (§3.5) showed the same pattern in
  miniature: a +0.0023 CI-clear local win compressed to +0.0005 mean AUC
  on the real test set. **Candidate J (§3.5/§3.6) is a third, genuinely
  different data point: a local win that *widened* moving from
  MINDsmall-dev to MINDlarge-dev (+0.0051 → +0.0244), then compressed at
  the final real-leaderboard step (0.6579 → 0.6462) like the other two —
  but stayed a real, substantial win (+0.0267 over the original
  submission) rather than compressing to flat.** None of the three cases
  traced to a pipeline defect (all format-verified with this project's
  standard discipline, including a real narrow-impact bug found and fixed
  in Candidate J's own submission pipeline via the established spot-check
  process). The leading explanation for the two flat cases is that the
  method's edge was built around properties specific to the validation
  population that the official, temporally disjoint test split doesn't
  guarantee will hold; Candidate J's more robust transfer is plausibly
  because a trainable encoder learns more generalizable structure than a
  small combiner's fitted coefficients over fixed features — inference,
  not independently verified by a controlled ablation. With `N=3`, this
  is a clearer, still-not-fully-quantified pattern about this project's
  validation methodology: local wins reliably compress at real scale, but
  the *architecture* behind the win appears to matter for whether that
  compression reaches zero (ADR-010's 2026-08-22 addendum, ADR-012's
  2026-08-25/26 addenda). **Candidate K (§3.7) is a fourth data point,
  and the cleanest yet**: local `ebnerd_large` win 0.7590 → real 0.7542,
  a −0.0048 compression, an order of magnitude smaller than any prior
  case. Its edge comes from structural, leak-safe item-level signal
  (freshness, immediate context) rather than a cohort-specific rule or a
  marginal embedding-artifact edge — strengthening, not just repeating,
  the inference that architecture/signal type affects how much a local
  win survives real-test-set transfer (ADR-013's 2026-08-30 Addendum).
- **The BlackPearl-derived long/short-term interest hypothesis (A1,
  ADR-013) is falsified, not just unconfirmed.** Short-term interest
  features contribute only 0.8% of Candidate K's model gain at real
  `ebnerd_large` scale (down from an already-small 1.9% before a
  per-impression recency bug was fixed) — the fix made short-term
  features *more* correctly computed, not more important. Freshness
  (33.3%) and immediate context (`context_embed_sim`) drive the result
  instead. Reported because it contradicts the hypothesis the candidate
  was built to test, not smoothed into a generic "features helped."
- **A numpy/CPython version-mismatch defect was found and fixed
  (ADR-013), and it is not known whether any earlier recorded number in
  this project was computed under it.** The project virtualenv had numpy
  1.26.4 source-built against CPython 3.14 (no released wheel for that
  combination), and was measured producing provably wrong array results
  (a stored boolean array disagreeing with a fresh recomputation of the
  identical expression). Rebuilt on Python 3.11, which `pyproject.toml`
  already declares; the full test suite passed afterward. This ADR does
  not claim any prior number is wrong — only that it has not been
  re-verified under a supported interpreter, which is itself worth
  stating rather than leaving implicit.
