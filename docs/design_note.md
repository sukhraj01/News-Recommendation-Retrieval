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
on both competitions (§3.5).

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
| EB-NeRD (competitions/2469) | Embed (MiniLM) | 0.5404 | 888045 | 2026-08-13 23:08 |
| EB-NeRD (competitions/2469) | Contrastive vector (§ADR-008 Addendum) | 0.5404 | 896072 | 2026-08-21 13:01 |

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

### 3.6 MIND candidate search: eight other approaches, before G

Before committing to Candidate G (above), eight other approaches were
screened on MINDsmall-dev against the deployed baseline (AUC 0.634,
§3.2). Only G's cohort-gated routing cleared a CI-clear win overall, and
only F's raw combiner cleared one on the cold cohort alone. Full
methodology, evidence, and interpretation for each candidate is in
ADR-010 (D–H) and ADR-011 (I).

| Candidate | Method | Result vs. baseline |
|---|---|---|
| A | Entity embeddings (TransE, confidence-weighted pooling) | 0.553, CI-clear loss |
| B | Untuned 50/50 BM25+embedding hybrid | 0.626, CI-clear loss |
| C | Recency-weighted (decay=0.9) embedding query | 0.627, CI-clear loss |
| D | Symbolic category/entity overlap score | 0.613, CI-clear loss |
| E | Train-split popularity only, no personalization | 0.532, CI-clear loss |
| F | 7-feature `LogisticRegression` combiner | 0.626 overall (loss); 0.593 cold (CI-clear win) |
| H1 | `LogisticRegression`, class-balanced | 0.632, CI-clear loss |
| H2 | `HistGradientBoostingClassifier` | 0.599, CI-clear loss |
| I | Attention re-ranker, 5 epochs (first trained model) | 0.623, CI-clear loss |
| I-long | Same, 30 epochs (holdout AUC climbed to 0.637) | 0.621 dev, worse than the 5-epoch run |
| I-pop | I-long + an unnormalized popularity feature | 0.513, confounded/inconclusive (see ADR-011) |

No untuned single alternative signal beats the deployed baseline
outright; the only real wins found across either round (F's cold cohort,
G's routing) come from combining signals around `log_popularity` — and
even G's win didn't hold at real leaderboard scale (§3.5 above).

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
- **Local CI-clear validation wins have not reliably transferred to real
  blind-test leaderboard results, twice.** MIND's cohort-gated combiner
  (§3.5) showed a CI-clear win at both a MINDsmall-dev screen (+0.0027)
  and a MINDlarge-dev re-verification (+0.0019), then landed essentially
  flat on the real leaderboard (-0.0003, 896696 vs. 886468). EB-NeRD's
  contrastive-vector submission (§3.5) showed the same pattern in
  miniature: a +0.0023 CI-clear local win compressed to +0.0005 mean AUC
  on the real test set. Neither case traced to a pipeline defect (both
  format-verified with this project's standard discipline); the leading
  explanation in both is that the method's edge was built around
  properties specific to the validation population (a cohort mix, a
  small model's fitted coefficients) that the official, temporally
  disjoint test split doesn't guarantee will hold. With `N=2`, this is a
  real, named pattern about this project's validation methodology, not
  yet a quantified relationship between local and real effect sizes
  (ADR-010's 2026-08-22 addendum).
