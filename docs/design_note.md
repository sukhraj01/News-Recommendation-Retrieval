<!--
Editorial note (delete before Moodle submission): the assignment brief/PDF
itself is not present anywhere in this repository — every project doc
references "Q1"-"Q9" by number, but the literal spec (required sections,
per-section page budget) was never captured verbatim. The structure below
is inferred from how ADR-001 through ADR-008 and PROJECT_STATE.md already
organize the project around those question numbers. Every number in this
note is pulled directly from a committed experiment/config/results file or
ADR, cited inline as `path/to/file` — none are re-derived or approximated.
One real gap is flagged explicitly where it occurs (recall@K was never run
at MINDlarge scale) rather than filled with a plausible-looking number.
-->

# Design Note — Lexical vs. Semantic Retrieval for News Recommendation

**CS4.406 Information Retrieval & Extraction — Assignment 1**

---

## 1. Problem and Approach

News recommendation differs structurally from catalog domains like
retail or streaming: item relevance half-life is hours, not months, so
item cold-start is the *default* state, not an edge case (Wu et al. 2020;
Kruse et al. 2024). This motivates content-based candidate retrieval —
lexical (BM25) and semantic (embeddings) — over collaborative filtering.
This project implements both retrieval methods, plus a shared evaluation
harness, over MIND and EB-NeRD under one unified schema and temporal-split
protocol (ADR-001, ADR-002), so the two methods and two datasets are
compared on identical footing rather than method-specific pipelines.

## 2. Methodology

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
(0.069, 2.4x ratio) is real and usable; e5-small's own `passage:`-prefixed
convention compresses nearly all pairs into a narrow 0.75–0.78 band (gap
0.017, ratio 1.02x) — e5's asymmetric query/passage objective doesn't fit
this project's symmetric user-profile-vs-article use case
(`decisions/ADR-008-semantic-retrieval-design.md`, lines ~150-177). ANN
backend is brute-force cosine similarity — 0.99ms/query at MINDsmall
scale, 2.44ms/query (~10.4 min full-run projected) at MINDlarge's 255,990
users — FAISS stays unjustified at this scale (ADR-008 + Addendum).
User representation is mean-pooled, L2-renormalized embeddings of full
click history; zero-history users get an explicit all-zero vector, mirror-
ing BM25's own cold-start reporting posture rather than an unbenchmarked
fallback.

**Evaluation.** Two distinct questions are measured, not conflated:
*recall@K* (does retrieval surface the clicked article out of the *whole*
corpus, K ∈ {50,100,200}) and *Q4's ranking metrics* — AUC, MRR, nDCG@5,
nDCG@10, intra-list diversity, novelty, catalog coverage — computed over
the candidates *already listed in each impression* (not full-corpus
top-K). Ties are broken by a deterministic per-impression-seeded pseudo-
random permutation (needed because cold users produce exact ties); K=10
for diversity/novelty (anchored to nDCG@10); novelty's popularity signal
is Laplace-smoothed and computed from the **train split only**, never the
evaluation split, satisfying the assignment's anti-gaming requirement
(Q9) — `total_inviews`/`total_pageviews`/`total_read-time` are never read
anywhere in this pipeline (grepped, not asserted from memory). Every
metric reports 95% bootstrap CIs (n_bootstrap=2000,
`experiments/ranking_bm25_mind_2026-08-10/config.json`), sliced by
warm/cold user (history length ≥ 5 = warm, per EB-NeRD's own
active-user-filter threshold), except catalog coverage, which is a
set-union statistic mechanically biased under with-replacement
resampling and is reported as a point estimate only (ADR-007).

## 3. Results

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
(min history length = 5) by construction — not measured as zero, undefined
(`experiments/bm25_ebnerd_small_2026-08-10/results.json`, `notes`).

Source: `experiments/{bm25,embed}_{mind,ebnerd,ebnerd_small}_2026-08-10/results.json`.

**Recall@K was never computed at MINDlarge scale** — only Q4's
candidate-list ranking metrics were (§3.2). If a full-corpus recall
number at that scale is required, it does not exist in this project and
is not approximated here.

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

On MIND (both scales), embeddings' AUC CI does not overlap BM25's —
embeddings win outright, on warm *and* cold cohorts, not just cold. This
runs against the Day-1 working hypothesis ("BM25 favors warm/entity-heavy
MIND"). The warm/cold gap is *not* similar between methods, though: BM25's
is Δ0.052–0.056, embeddings' is Δ0.068–0.070 (unrounded
`auc.warm.metric`−`auc.cold.metric` from
`experiments/ranking_{bm25,embed}_mind{,_large}_2026-08-1{0,1}/results.json`:
BM25 0.0524/0.0557, embed 0.0701/0.0682) — embeddings widen the relative
warm/cold disparity by ~0.015–0.016, because they lift warm AUC by
~0.065–0.067 over BM25 (0.5766→0.6439 small, 0.5777→0.6431 large) versus
only ~0.050–0.053 on cold (0.5242→0.5737 small, 0.5220→0.5749 large).
Embeddings raise both cohorts, but warm users benefit somewhat more —
they do not specifically close the cold-start gap, and by this measure
edge toward widening it rather than leaving it unchanged.

On EB-NeRD, embeddings' AUC CI also does not overlap BM25's — a real,
if modest (~0.011–0.014), edge — despite BM25 winning recall@K by
1.6–2.3x on the *same* corpus (§3.1). These are different measurements:
recall@K asks whether retrieval finds the click anywhere in the whole
catalog; Q4's AUC asks how well the method ranks EB-NeRD's already-curated,
much shorter candidate lists (median 9–12 vs. MIND's 23). A plausible
(not yet isolated) explanation: EB-NeRD's long median history (81–93
articles) produces a diffuse mean-pooled query that struggles against the
*whole* catalog but still discriminates adequately within a pre-curated
short list.

MRR/nDCG@5/nDCG@10 (same source files): embeddings lead BM25 by a similar
margin on MIND (e.g. MINDlarge nDCG@10: 0.390 vs. 0.350) and by a smaller,
mixed margin on EB-NeRD (e.g. EB-NeRD-small nDCG@10: 0.459 vs. 0.454).

### 3.3 Diversity / novelty / coverage (K=10, point estimate for coverage)

| Dataset | Method | Diversity@10 | Novelty@10 | Coverage@10 |
|---|---|---|---|---|
| MIND-small dev | BM25 | 0.837 | 16.30 | 0.083 |
| MIND-small dev | Embed | 0.827 | 16.15 | 0.078 |
| MINDlarge dev | BM25 | 0.836 | 18.09 | 0.067 |
| MINDlarge dev | Embed | 0.828 | 17.76 | 0.065 |
| EB-NeRD-small val | BM25 | 0.795 | 17.17 | 0.206 |
| EB-NeRD-small val | Embed | 0.789 | 17.19 | 0.205 |

BM25 and embeddings sit within ~1 percentage point of each other on every
beyond-accuracy metric, on every corpus — accuracy differs meaningfully
between methods, beyond-accuracy behavior barely does. Given the large
sample sizes here (n up to 376,471 impressions), even a 1-point gap can
be statistically non-trivial, but it is not the kind of large,
consistent effect the AUC results show — this is exactly the pattern
ADR-007's multi-metric design was built to surface, not a null result
smoothed over.

Source: `experiments/ranking_{bm25,embed}_{mind,ebnerd_small,mind_large}_2026-08-1{0,1}/results.json`.

### 3.4 System cost

BM25 at MINDlarge scale: 1.70s index build, 23.5MB sparse weight matrix,
~9.8 min projected full retrieval for 255,990 users (`decisions/ADR-006-bm25-variant.md`,
Addendum). Embeddings at the same scale: 264.4s to encode+cache all
72,023 articles, ~10.4 min projected full retrieval (`decisions/ADR-008-semantic-retrieval-design.md`,
Addendum). Both stay local on an 8GB machine — no cloud GPU migration was
needed for either method at MINDlarge scale; only EB-NeRD's *blind test*
split (13.5M impressions, no local ground truth) required Kaggle.

## 4. Leaderboard Results (Q5/Q6)

| Competition | Score | Submission ID | Date |
|---|---|---|---|
| MIND (competitions/13967) | 0.6195 | 886468 | 2026-08-12 13:01 |
| EB-NeRD (competitions/2469) | 0.5404 | 888045 | 2026-08-13 23:08 |

Both submissions used the embedding method — the clear winner on local
dev-set AUC for both datasets (§3.2). EB-NeRD's leaderboard score (0.5404)
sits close to this project's own local validation AUC for embeddings on
`ebnerd_small` (0.5430,
`experiments/ranking_embed_ebnerd_small_2026-08-10/results.json`) — an
informal coherence check that the submission pipeline (candidate-order
preservation, format conversion) behaves consistently between local
validation and the real held-out test set, not a claim that the two
numbers are defined identically (`PROJECT_STATE.md`, Leaderboard
Submission row — column headers were cropped from the available
screenshots, so the Score≈AUC mapping is inference from the numeric
match, not a confirmed label).

## 5. Limitations and Open Questions

- **True (zero-history) cold-start is a shared ceiling, not something
  either method solves.** MIND's cold-cohort recall@200 is nearly
  identical between methods (BM25 1.78% vs. embed 1.81%) — both degrade to
  a structural miss for users with no vocabulary to match and no vector to
  compute.
- **EB-NeRD's cold cohort is structurally empty** in both the `demo` and
  `small` bundles (min history length = 5, by construction of the
  active-user filter) — no warm/cold comparison is possible for EB-NeRD in
  this project; `ebnerd_large` was never downloaded/verified and might
  behave differently.
- **Recall@K was never run at MINDlarge scale** (§3.1) — the MINDlarge
  results in this note are Q4 ranking-metrics only.
- **The encoder choice is verified via a category-based discrimination
  proxy, not a curated near-duplicate/paraphrase set** — category is a
  real but weak, noisy signal for true topical relevance (ADR-008,
  Decision Confidence: Medium-High, not High).
- **The hand-built Danish+English stopword list** used in BM25 query
  construction has not been validated against an NLP resource — flagged
  as an open revisit trigger in ADR-005, not yet tested.
- **The EB-NeRD result is a genuine disagreement between two evaluation
  questions**, not fully explained: BM25 wins whole-corpus recall by
  1.6–2.3x; embeddings edge out BM25 on the already-curated candidate-list
  AUC. The diffuse-mean-pooled-query explanation offered in §3.2 is
  plausible but not isolated by a controlled experiment.
