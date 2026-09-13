# ADR-014 — Pipeline Performance Profiling & Benchmarking Methodology

> Captures **why** performance is measured the way it is here, and what the
> measurements found. Companion to ADR-006 (BM25 scoring implementation) and
> ADR-008 (semantic retrieval), both of which recorded *macro* timings; this
> ADR establishes the per-stage, per-query substrate those numbers lacked, and
> reconciles against them rather than replacing them.

---

**Date:** 2026-09-04
**Status:** Decided
**Severity:** High — establishes a permanent, enforced workflow step, and names
the component the project should invest optimisation effort in.

---

# Engineering Question

Three questions, deliberately kept apart because they have different answers:

1. **Where does per-query time actually go**, stage by stage, in each of the two
   deployed serving paths (MIND: BM25 / embeddings / hybrid / NRMS; EB-NeRD:
   engineered features / LightGBM)? Not "how long does a full run take" — that
   was already known — but what fraction each component owns.
2. **What does each component cost in latency terms specifically**, measured by
   removing or swapping it? This is a new axis. Every ablation this project has
   run (ADR-009's leaky features, ADR-010/011/012's candidate arms, ADR-013's
   `nopos` arm) measured *accuracy*. None measured latency.
3. **Does the answer to "where should I invest" change with hardware?** The
   project has used three real machines — an 8GB Apple M3, a Kaggle T4, and
   Ada's SLURM GPU nodes — and it must not be assumed the bottleneck is the
   same on all three.

And one process question: **how does performance measurement stop being a
one-off report and become routine**, the way `experiments/` already made
accuracy measurement routine?

---

# Context

## What already existed, and why it was not enough

The project had real, honestly-obtained macro timings:

| Figure | Source |
|---|---|
| BM25 index build 1.70s (MINDlarge-dev, 72,023 articles) | ADR-006 addendum, 2026-08-11 |
| BM25 full retrieval ~9.8 min projected (255,990 users) | ADR-006 addendum |
| Embedding encode 264.4s; throughput 373.6 articles/s | ADR-008 addendum |
| Embedding full retrieval ~10.4 min projected | ADR-008 addendum |

Every one of those is a **whole-stage aggregate**. None can answer "what
fraction of a single query's latency is tokenization?" or "is the ranker or the
feature builder the expensive half?" — and neither can they be *added up*,
because they were measured at different times, under different definitions, and
in one case (the ~9.8 min figure) from a 1,000-user sample whose draw method was
not recorded.

The gap this ADR closes is therefore not "we had no numbers"; it is "the numbers
we had could not be decomposed, composed, or compared across hardware."

## The constraint that shapes the method

The local development machine is an **8GB Apple M3 (Mac15,12, 8 cores)**.
MINDlarge-dev's impressions table alone is 14,085,557 candidate rows, and
PROJECT_STATE's risk register already records real thrashing at that scale. A
benchmark that cannot run without swapping produces numbers that measure the
swap, not the pipeline.

This was confirmed the hard way during this very work: running two
MINDlarge-scale measurements concurrently moved the BM25 scoring stage from
**4.33 ms/user to 10.9 ms/user** — a 2.5x distortion, identical code, identical
data. Serial execution is a correctness requirement here, not tidiness.

---

# Decision Criteria

| Criteria | Importance | Notes |
|---|---|---|
| Measures the code actually deployed | Critical | A profile of a loop the pipeline never runs is worthless. Stage boundaries follow `run_ranking_eval.py` and `run_ebnerd_gbdt_experiment.py` exactly. |
| Real scale | Critical | Full article catalog, real vocabulary, real embedding matrix. Sampling may reduce *queries*, never the *index*. |
| Cross-checkable | High | Manual instrumentation must be corroborated by an independent profiler (cProfile), or a systematic error in one is invisible. |
| Reconciles with prior records | High | New fine-grained numbers must be shown consistent (or explicitly inconsistent) with ADR-006/ADR-008, not silently supersede them. |
| Composable percentages | High | Every row denominated in the same unit, sub-stages excluded from their parent's denominator, so shares sum honestly. |
| Cheap enough to run routinely | High | The habit fails if the check is slow. Separate authoritative (minutes) from tripwire (seconds). |
| Fits in 8GB | Critical | Per CLAUDE.md's Memory Estimation clause. |
| Portable across hardware | High | One script, three machines, self-labelling output. |

---

# Design Space Exploration

## Option 1 — Wall-clock the existing experiment scripts, before and after

**How it works.** Time `run_ranking_eval.py` end to end; diff across changes.

**Optimizes for:** zero new code; measures exactly what is shipped.

**Sacrifices:** no decomposition at all — this is the situation the project was
already in. Cannot answer any of the three engineering questions. A regression
is detectable but not attributable.

**Rejected:** it reproduces the existing gap rather than closing it.

## Option 2 — cProfile alone

**How it works.** Run the harness under `cProfile`, read the function table.

**Optimizes for:** genuine per-function attribution, no manual instrumentation
to get wrong, catches time in places nobody thought to instrument.

**Sacrifices:** `cProfile` instruments every Python call and inflates absolute
times, so its numbers are not latencies. It also reports *functions*, not
*pipeline stages* — `numpy.einsum` appears once, aggregated across three
different stages that call it. Percentages from it answer "which function" not
"which component", and per-query throughput cannot be derived at all.

**Rejected as the primary instrument, adopted as the cross-check.**

## Option 3 — py-spy sampling profiler

**How it works.** Sample the stack of a running process externally.

**Optimizes for:** near-zero overhead; can attach to a live job.

**Sacrifices:** sampling gives a statistical picture, not per-call latency
distributions; p99 and max per stage are not recoverable. Also an extra
dependency, and on macOS it needs elevated privileges to attach.

**Rejected:** the question is per-query latency including tails, which sampling
cannot give directly. Retained as a future option for profiling long Ada jobs
in situ.

## Option 4 — Explicit stage instrumentation, cross-checked by cProfile *(chosen)*

**How it works.** A `StageTimer` context manager wraps each pipeline stage in a
benchmark driver that mirrors the deployed loop structure. `perf_counter` wall
time, warmup excluded and recorded, full sample distributions retained so
percentiles are reportable. A separate `cProfile` pass over a sub-batch provides
independent function-level attribution.

**Optimizes for:** stage-level percentages that compose; per-stage throughput;
latency tails; one substrate reused unchanged across datasets and machines.

**Sacrifices:** the stage boundaries are a human judgement — a wrong boundary
mis-attributes cost. This is exactly why the cProfile cross-check is mandatory
rather than decorative.

**Assumptions:** stage timing overhead (~100ns/call) is negligible against
stage costs (microseconds to tens of milliseconds). Verified: the unattributed
remainder is reported on every run.

---

# Final Decision

## Chosen Option

**Option 4 — explicit stage instrumentation, cross-checked by cProfile**, run at
real corpus scale with systematically-sampled queries, on every hardware spec
available, plus a lightweight `snapshot.py` tripwire wired into a pre-commit
hook to make the measurement routine rather than retrospective.

### Reason

It is the only option that produces composable per-stage percentages AND
per-stage throughput AND latency tails, from one substrate reusable unchanged
across two datasets and three machines. cProfile alone answers "which function";
this answers "which component", which is the question the investment decision
actually turns on. The cProfile pass is retained as a mandatory independent
check, not discarded.

### Decision Date
2026-09-04

### Decision Owner
Engineer (sukhraj01)

### Contributors
Claude Code (design, implementation, benchmarking, ADR drafting)

---

# Measurement Setup

| | Local | Ada (SLURM) |
|---|---|---|
| Machine | Mac15,12, **Apple M3, 8 cores, 8 GiB** | gnode078, **Intel Xeon E5-2640 v4 @ 2.40GHz**, 125.8 GiB |
| GPU | Apple MPS (integrated) | **NVIDIA RTX 2080 Ti**, 11 GiB, driver 580.173.02 |
| Cores used | 8 | 8 (`-c 8`, pinned via `OMP_NUM_THREADS`) |
| torch | 2.13.0 | **2.11.0+cu128** |
| numpy / scipy / pandas / lightgbm / faiss | 1.26.4 / 1.17.1 / 2.3.3 / 4.7.0 / 1.15.0 | identical |

**Kaggle T4 was NOT measured.** There are no Kaggle API credentials on this
machine and the notebook environment is interactive-only. Rather than
substitute a lower-fidelity stand-in (CLAUDE.md's Resource Availability clause),
this is recorded as an explicit gap. PROJECT_STATE holds T4 *macro* figures
(849.1 articles/s encode, 105.3s `query_by_user`) from an earlier session; those
are **not** per-stage numbers and are not presented as if they were.

**The torch version differs between machines** (2.13.0 vs 2.11.0) and cannot be
made to match: the cu128 index's newest cp311 build is 2.11.0, and these nodes
cap at CUDA 12.8. The cross-machine comparison is therefore confounded. The
GPU-vs-CPU comparison **within one Ada node**, one venv, one allocation, is not
— and that is the pair the hardware conclusion rests on.

## Sampling

Full-scale index, systematically-sampled queries. MIND: all 72,023
MINDlarge-dev articles, 1,200 users sampled systematically → 1,771 impressions.
Sample fidelity verified, not assumed: **1.4758 impressions/user sampled vs.
1.4706 in the population.** That ratio governs how often the per-user score
caches hit; sampling impressions independently instead drew 1.005
impressions/user (measured), which would have inflated per-impression BM25 cost
by ~1.47x and pointed the investment answer at the wrong stage.

EB-NeRD: 25,000 of 244,647 `ebnerd_small` validation impressions, systematic
(never a prefix — EB-NeRD clusters a 200,000-row `is_beyond_accuracy` block).

## Serving vs. evaluation

`10_metrics` / `4_metrics` compute AUC/MRR/nDCG. **A production ranker never
does this.** Percentages are therefore reported against a *serving-only*
denominator, with the evaluation stage shown separately. Folding a 22-24%
evaluation-only stage into a "where should I invest in serving" answer would be
straightforwardly misleading.

---

# Results — MIND (MINDlarge-dev, 72,023 articles, 1,170 users / 1,718 impressions)

Percentages are of the **serving path** (metrics excluded).

| Stage | M3 CPU ms | % | Ada Xeon CPU ms | % | Ada 2080 Ti ms | % |
|---|---:|---:|---:|---:|---:|---:|
| `1_query_tokenize` | 0.458 | 1.0% | 0.355 | 0.5% | 0.606 | 4.0% |
| `2_bm25_score_all` | 2.950 | 6.7% | 3.313 | 4.7% | 4.916 | **32.3%** |
| `3_embed_query_build` | 0.222 | 0.5% | 0.103 | 0.1% | 0.133 | 0.9% |
| `4_embed_score_all` | 4.668 | 10.7% | 2.414 | 3.4% | 1.955 | 12.9% |
| `5_bm25_candidate_lookup` | 0.183 | 0.4% | 0.064 | 0.1% | 0.057 | 0.4% |
| `6_embed_candidate_lookup` | 0.087 | 0.2% | 0.043 | 0.1% | 0.042 | 0.3% |
| `7_hybrid_blend` | 0.039 | 0.1% | 0.107 | 0.2% | 0.119 | 0.8% |
| **`8_nrms_inference`** | **34.920** | **79.9%** | **64.243** | **90.7%** | **7.152** | **47.1%** |
| `9_rank_tiebreak` | 0.178 | 0.4% | 0.211 | 0.3% | 0.218 | 1.4% |
| *(`10_metrics`, eval only)* | *1.957* | — | *3.303* | — | *4.854* | — |

| | M3 CPU | Ada Xeon CPU | Ada 2080 Ti |
|---|---:|---:|---:|
| Serving latency | 43.70 ms/imp | 70.85 ms/imp | **15.20 ms/imp** |
| Serving throughput | 22.9 imp/s | 14.1 imp/s | **65.8 imp/s** |
| End-to-end (incl. eval) | 46.70 ms | 75.24 ms | 21.17 ms |
| Peak RSS | **0.77 GB** | 1.98 GB | 1.95 GB |

Local Apple MPS was also measured (50-user smoke): **74.89 ms/impression** for
NRMS — **2.1x SLOWER than the same machine's CPU.** At batch size 1 the MPS
dispatch overhead exceeds the compute it replaces. `embed.py::_default_device`
prefers MPS on this machine, which is the wrong choice for this workload.

**cProfile cross-check.** On M3 CPU it attributes 14.07s of 19.28s (73.0%) of
`run_batch` to `nrms_training.py:242(score)`, against the StageTimer's 76.5%
(all-stage denominator) — two independent instruments agreeing. On the GPU node
the ranking changes and `sklearn.roc_auc_score` (3.21s) overtakes NRMS (2.55s),
which is what motivated separating the evaluation denominator above.

---

# Results — EB-NeRD (ebnerd_small validation, 20,000 impressions, 11.9 candidates/impression)

Ada Xeon CPU. Model: the locally-available `model_K_rank.txt`, **319 trees, 60
features**.

| Stage | ms/impression | % of serving | µs/candidate |
|---|---:|---:|---:|
| **`1_feature_frame_build`** | **0.4757** | **63.5%** | 39.97 |
| ↳ *`1a_short_term_features` (sub-stage)* | *0.3550* | *47.4%* | *29.83* |
| `2_lightgbm_predict` | 0.0596 | **8.0%** | 5.01 |
| `3_rank_tiebreak` | 0.0439 | 5.9% | 3.69 |
| *(`4_metrics`, eval only)* | *0.1704* | *22.7%* | *14.32* |

End-to-end **0.7515 ms/impression = 1,330.6 impressions/s**; peak RSS 1.58 GB;
full-split projection 3.06 min. One-time setup: user profiles **9.06s**
(dominant), popularity 1.37s, behaviours 1.05s, article table 0.71s.

cProfile agrees independently: `build_feature_frame` 2.92s of `run_batch`'s
4.98s, with `compute_short_term_features` 2.11s inside it.

**Model caveat, stated because it bounds the claim.** This is the 2026-08-26
`ebnerd_small` booster, trained **before** ADR-013's correction added
`context_category_match`, `context_topic_overlap`, `context_embed_sim`,
`session_position`, `session_start_gap_h`. The 65-feature `ebnerd_large` model
that actually scored **0.7542** on Codabench was trained on Ada and its weights
were never brought back. Since the LightGBM ablation shows inference cost is
nearly flat in feature count (1.08x for 60→20), the 60-vs-65 difference does not
threaten the conclusion — but the numbers are from the model that exists, not
the model that shipped.

---

# Results — Performance Ablations

Measured on Ada. This is a **new axis**: every prior ablation in this project
(ADR-009, ADR-010, ADR-011, ADR-012, ADR-013's `nopos`) measured accuracy.

| Ablation | Deployed | Alternative | Speedup | Accuracy cost |
|---|---|---|---:|---|
| BM25 scorer | sparse matvec **3.75 ms** | `rank_bm25.get_scores` **9,406.72 ms** | **2,507.9x** | none (max abs diff **5.7e-14**) |
| ANN index | brute-force numpy **5.51 ms** | FAISS IVF (nlist 256, nprobe 8) **0.74 ms** | 7.45x | **recall@100 = 0.6511** |
| ANN index | brute-force numpy **5.51 ms** | FAISS Flat (exact) **46.88 ms** | **0.12x** (8.5x slower) | none |
| NRMS cache (GPU) | uncached **3.749 ms** | cached news vectors **1.319 ms** | 2.84x | none (4.5e-8, float32) |
| NRMS cache (CPU) | uncached **9.826 ms** | cached news vectors **1.399 ms** | **7.02x** | none (**exactly 0.0**) |
| LightGBM features | 60 features **0.0616 ms** | top-20 by gain **0.0569 ms** | 1.08x | mean AUC **−0.0023** |

Notes that change how these read:

- **BM25's 2,508x is why ADR-006's rewrite exists**, now with the price tag:
  projected full MINDlarge-dev retrieval is **16.0 min** sparse vs. **40,133.8
  min (27.9 days)** naive. Exactness re-verified here rather than trusted.
- **FAISS IVF is rejected — again, and now on accuracy, not speed.** 7.45x
  faster, but it agrees with exact brute force on only **65.1%** of the top-100,
  min recall **0.15**, and just **3.1%** of queries get a perfect top-100.
  ADR-008's original "brute force is sufficient" call stands.
- **FAISS Flat is 8.5x SLOWER than a plain numpy matvec** at 72,023 x 384. The
  library adds nothing at this scale; a `vectors @ q` is already the right
  answer. This is the single most counterintuitive number in the exercise.
- **The NRMS cache is a pure win with no modelling change** — same function,
  verified numerically identical on CPU (0.0 difference). Break-even is 632
  impressions on GPU, 1,371 on CPU, against MINDlarge-dev's 376,471 — i.e. it
  amortises after **0.17%** of one split.
- **LightGBM feature count barely affects inference (1.08x).** Cutting 40
  features buys 8% of a stage that is only 8% of the path — ~0.6% end to end,
  for −0.0023 AUC. Not worth doing. The cost is in *building* features, not
  consuming them.

---

# The Bottleneck — stated directly

**MIND: NRMS inference.** 34.920 ms/impression on the local M3, **79.9% of
serving-path latency**, 7.5x the next-largest stage. On the Ada Xeon CPU it is
worse: 64.243 ms, **90.7%**.

**EB-NeRD: feature engineering — specifically the short-term feature block.**
`build_feature_frame` is 63.5% of the serving path while LightGBM inference is
8.0%. Inside it, `compute_short_term_features` alone is **0.3550 ms/impression =
47.4% of the whole path, ~75% of all feature building.**

The two datasets do **not** share a bottleneck, and assuming they did would have
sent optimisation effort to the wrong place in one of them.

## The sharpest single finding

The short-term feature block is **the most expensive component in the EB-NeRD
path and contributes nothing to the top 20 features by gain.** Its best-ranked
member ranks **#27** by the deployed model's own recorded gain importance; none
of `st_cat_affinity`, `st_subcat_affinity`, `st_topic_affinity`, `st_embed_sim`,
`st_cat_rank`, `st_cat_affinity_rank_in_imp` appears in the top 20.

That is a 47.4% latency cost sitting on the lowest-value features in the model.
It is the clearest cost/value mismatch this profiling found — and it is *not*
an instruction to delete them: ADR-013's addendum records that fixing the
short-term computation (it had been static per-user) was part of what moved
local `ebnerd_small` from 0.7514 to 0.7581. The correct next step is to measure
the AUC cost of removing them, which this ADR does not do.

## Does the answer change with hardware? Yes — measurably

| | NRMS share of serving path |
|---|---:|
| Ada Xeon CPU | **90.7%** |
| Local M3 CPU | **79.9%** |
| Ada RTX 2080 Ti | **47.1%** |

Same code, same data, same sample. On the internally-controlled pair (one node,
one venv, one allocation), moving NRMS to the GPU makes it **8.98x faster**
(64.243 → 7.152 ms) and drops its share from 90.7% to 47.1%, at which point
`bm25_score_all` (32.3%) becomes a genuine second bottleneck rather than a
rounding error.

**So the investment advice is hardware-dependent, and stating it without the
hardware would be wrong:**

- **CPU-only deployment → optimise NRMS.** It is 80-91% of the path. The cheapest
  available win is the candidate-vector cache: **7.02x on CPU**, no accuracy cost,
  no retraining.
- **GPU deployment → NRMS first (47.1%), then BM25 scoring (32.3%).** The profile
  is flat enough that a second round of NRMS work has much worse marginal return
  than the local numbers suggest.
- **Some stages are *slower* on the cluster.** `bm25_score_all` runs at 0.60x and
  `metrics` at 0.40x of local speed, because the node's 2016-era Xeon E5-2640 v4
  is a weaker single-thread and memory-bandwidth part than the Apple M3. "Move it
  to the HPC cluster" only helps where the GPU does the work; for the CPU-bound
  stages it is a **regression**.

---

# Reconciliation with ADR-006 / ADR-008 — one figure does not reproduce

Per the brief, the new fine-grained numbers reconcile against the existing macro
ones rather than replacing them.

| Quantity | Recorded | Measured now | Verdict |
|---|---|---|---|
| BM25 sparse weight matrix | 23.5 MB (ADR-006) | **23.5 MB** | exact match |
| BM25 index build | 1.70s (ADR-006) | 2.23s (M3), 3.99s (Ada) | same order; build is not a bottleneck either way |
| Embedding matrix | — | 110.6 MB, 384-dim, 72,023 rows | consistent with ADR-008 |
| Embedding full retrieval | ~10.4 min (ADR-008) | **12.25 min** (Ada GPU node) | **reproduces** (18% high) |
| **BM25 full retrieval** | **~9.8 min (ADR-006)** | **15.4–21.4 min** | **does NOT reproduce** |

ADR-006's addendum records "2.31s / 1,000 users → 590.2s (9.8 min)", i.e. **2.31
ms/user**. This instrumentation measures `score_all` at 3.62–4.33 ms/user across
two machines.

**The obvious explanation was tested and refuted.** ADR-006's sample-draw method
was not recorded; since `score_all` cost scales with distinct query terms, and
therefore with history length, a prefix sample could have drawn an unrepresentative
population. `benchmarks/reconcile_adr006.py` measured exactly this on Ada
(job 2687446), same corpus, same index, same function:

| Sample | history len (mean/p50) | query tokens (mean) | `score_all` mean | projected |
|---|---:|---:|---:|---:|
| `head(1000)` prefix | 31.8 / 21 | 828.5 | **4.158 ms** | 17.74 min |
| systematic 1,000 | 24.6 / 14 | 644.3 | **3.618 ms** | 15.44 min |

The prefix sample is **slower**, not faster (ratio 0.87) — so sampling bias
cannot explain an *under*-estimate. The hypothesis is dead.

**Honest status: the ~2x gap is unexplained.** Candidate explanations not yet
tested — the original may have timed a different quantity (e.g. `retrieve_top_k`
with its no-vocabulary-overlap early return), or been run under a materially
different machine/cache state. For planning purposes the measured 15.4–21.4 min
supersedes 9.8 min, and **ADR-006 gets an addendum recording the discrepancy
rather than a silent edit** — the historical reasoning is preserved, per
CLAUDE.md's Decision Reversal clause. Neither figure changes ADR-006's actual
*decision* (BM25 stays local; even 21 min is far short of the >2hr threshold).

---

# The Ongoing Habit

A one-off report was explicitly not what was asked for. The mechanism:

| Piece | What it does |
|---|---|
| `benchmarks/snapshot.py` | ~1.5s tripwire on MINDsmall-dev. `--compare` diffs against the last snapshot with the **same hardware and same config** and **exits nonzero on regression**. |
| `benchmarks/PERFORMANCE_LOG.md` | Auto-appended before/after table per run. Never hand-edited. |
| `benchmarks/results/*.json` | Timestamped, hardware-labelled, one per run — mirroring `experiments/`. |
| `.githooks/pre-commit` | Runs the snapshot **only** when a performance-relevant path is staged; blocks on regression. Bypass: `SKIP_PERF_SNAPSHOT=1`. |
| `make bench-baseline` / `bench-snapshot` / `bench` / `hooks` | The routine, and the full-scale run. |

**A defect found while building the gate, and fixed.** Two identical
back-to-back snapshot runs disagreed by **−44.5%** on `bm25_candidate_lookup`
and **−20.4%** on `embed_query_build` — those stages cost 0.02–0.04
ms/impression and sit in per-call timer noise. A pure percentage threshold would
have fired constantly on nothing and trained everyone to bypass it. A stage now
counts as regressed only if it is **both** >15% relatively **and** >0.25
ms/impression absolutely; sub-floor moves are labelled `noise`.

---

# Accepted Trade-offs

- **Sampled queries, not full splits.** Mitigated by preserving population
  impressions/user (1.4758 vs 1.4706) and by reconciling projections against
  independent macro figures — which is how the ADR-006 discrepancy surfaced at all.
- **NRMS runs randomly-initialised weights.** No trained Candidate J checkpoint
  exists locally; only `config.json`/`results.json` survived. Architecture is the
  real recorded one (embed_dim 300, num_heads 15, max_title_len 20,
  max_history_len 50). Latency depends on tensor shapes and dispatch, not weight
  values, so timings are valid — **no accuracy claim is made from this path.**
- **EB-NeRD uses the 60-feature pre-correction booster** (see above).
- **Kaggle T4 not measured** (no credentials). Explicit gap, not a substitution.
- **torch 2.13.0 vs 2.11.0 across machines** — unavoidable, confounds
  cross-machine only; the GPU-vs-CPU conclusion uses the controlled within-node pair.
- **cProfile absolute times are inflated** and used only for relative attribution.
- **Ada numbers are one node each** (gnode078, gnode047). Node-to-node variance is
  real and unquantified: `score_all` measured 4.86 ms/user on gnode078 within the
  full profile and 3.62 ms/user standalone on gnode047.

---

# Conditions for Revisiting

## Technical Triggers
- [ ] NRMS candidate-vector caching is implemented → re-measure; it should move
      MIND's bottleneck off NRMS on CPU (predicted 7.02x on that stage).
- [ ] Short-term EB-NeRD features are changed or removed → re-measure both
      latency **and** AUC; ADR-013 records they were load-bearing for accuracy.
- [ ] Corpus grows materially (MINDlarge test, `ebnerd_large`) → brute-force
      cosine's 5.51 ms scales linearly with catalog size; the FAISS rejection is
      valid at 72,023 docs and must be re-tested, not assumed, at 10x that.
- [ ] Deployment target changes CPU↔GPU → the whole investment ranking changes.
- [ ] `sklearn.roc_auc_score` enters a serving path (it currently does not).

## Project Triggers
- [ ] Kaggle credentials become available → fill the T4 row.
- [ ] A trained NRMS checkpoint becomes available → re-run with real weights and
      drop the caveat.
- [ ] The 65-feature `ebnerd_large` booster is retrieved from Ada → re-run EB-NeRD.
- [ ] Anyone reproduces or refutes ADR-006's 2.31 ms/user.

---

# Addendum (2026-09-12) — A2 Q4: index memory, request-level p99, cost/QPS, 10×

A2 Q4 asks three things this ADR's original tables could not answer, plus one it could:
**index memory**, **p99 latency for a single user request**, **cost per 1,000 queries at a
target SLA**, and a **10× scaling argument**. The gap was not missing data but the wrong
denominator: every number here was amortised per impression inside batched sweeps, so its
p99 is a tail over *chunks*, not over requests.

## Method changes (`profile_mind.py`, `profile_ebnerd.py`, new `cost_qps.py`)

- **`--per-request`**: each request runs the whole path alone. On MIND that means a *cold*
  request pays the per-user stages (tokenize, BM25 `score_all`, embedding query, embedding
  `score_all`) itself, instead of having them hoisted and amortised across a user's
  impressions. Metrics are excluded, per this ADR's serving-vs-evaluation split.
- **`--measure-memory`**: retained bytes per component via `tracemalloc` on a *separate*
  rebuild, so the timed build's numbers (which this ADR reports) stay uncontaminated.
- **Arm-aware feature mapping**: the shipped `K_rank_nopos` arm has 63 features against the
  run's 65-name list, so the profiler previously refused it. It now filters the recorded
  list by the run's own `arms[arm].withheld`, deriving the column order from the record
  rather than guessing. **The arm that actually shipped is profilable for the first time.**
- **This ADR's standing caveat is closed**: the EB-NeRD profile no longer uses the
  60-feature pre-correction booster. It runs against the **retrained 65-feature** model
  (ADR-013's 2026-09-12 addendum).

## Q4.1 — Index and feature-store memory

| MIND (MINDlarge-dev, 72,023 articles) | MB | | EB-NeRD (`ebnerd_small`, 20,738 articles / 15,342 users) | MB |
|---|---:|---|---|---:|
| BM25 weight matrix | 23.5 | | Article table | 108.2 |
| Embedding matrix (384-dim) | 110.6 | | User profiles | 82.1 |
| NRMS parameters (7.48M × fp32) | 29.9 | | History popularity | 0.2 |
| **Total index** | **164.0** | | **Total feature store** | **190.5** |
| Peak process RSS | 750 MB | | Peak process RSS | 770 MB |

Float32 payloads; Python container overhead and the id→row dicts are excluded, which is why
the process RSS is ~4× the index total.

## Q4.2 — Latency for a single user request (the new number)

| | MIND (M3 CPU) | EB-NeRD `K_rank_nopos` | EB-NeRD `K_rank` |
|---|---:|---:|---:|
| mean | 33.29 ms | 9.21 ms | 10.30 ms |
| p50 | 28.69 ms | 9.50 ms | 9.83 ms |
| p90 | 52.38 ms | 11.87 ms | 14.74 ms |
| **p99** | **94.67 ms** | **22.74 ms** | **27.36 ms** |
| max | 126.28 ms | 42.84 ms | 191.44 ms |
| requests timed | 500 | 2,000 | 2,000 |

**MIND meets a p99 < 100 ms SLA with 1.1× headroom — and its observed max, 126 ms, already
breaches it.** EB-NeRD clears the same SLA with 4.4× headroom.

**Batched vs per-request is a ~30× gap on EB-NeRD** (0.33–0.37 ms/impression amortised vs
~10 ms for one request) and ~1.2× on MIND (40.46 vs 33.29 — MIND's per-user work dominates
either way). Quoting a batched figure as request latency would have understated EB-NeRD's
serving cost by a factor of thirty. Both are reported; neither replaces the other.

Within a MIND request, **NRMS is 26.03 ms of the 33.29 ms mean (78%) and 81.27 ms of the
p99**. The batched sweep's NRMS stage p99 (147.6 ms) is larger than the whole request's p99:
different samples (1,723 impressions vs 500 cold requests) and different cache states, not a
contradiction — stated because the two tables sit next to each other.

## Addendum (2026-09-14) — the excluded MIND GPU row, measured for real

The 2026-09-12 addendum above excluded a MIND-on-`g4dn.xlarge` cost row rather
than estimate it, because the only per-request run at that point used
`--device cpu`. Once the Q3 MIND treatment job released the account's one-GPU
quota, `benchmarks/ada_mind_gpu_per_request.sbatch` ran the identical
`profile_mind.py --per-request` on a real 2080 Ti:

| | mean | p50 | p90 | **p99** | max |
|---|---:|---:|---:|---:|---:|
| MIND, GPU (2080 Ti) | 19.91 ms | 19.29 ms | 26.60 ms | **33.15 ms** | 35.65 ms |
| MIND, CPU (M3), for comparison | 33.29 ms | 28.69 ms | 52.38 ms | 94.67 ms | 126.28 ms |

**GPU headroom against the 100 ms SLA is 3.0×, against the CPU figure's 1.1×.**
NRMS inference drops from 78% of the request (CPU) to 36% (GPU) — consistent
with this ADR's own GPU-vs-CPU finding elsewhere (NRMS 8.98× faster on a
2080 Ti). Real cost, both rows now genuinely measured on their own hardware:

| Instance | mean | p99 | headroom | \$/1k queries (1 worker) |
|---|---:|---:|---:|---:|
| c6i.2xlarge (CPU-measured) | 33.29 ms | 94.67 ms | 1.1× | \$0.0031 |
| g4dn.xlarge (GPU-measured) | 19.91 ms | 33.15 ms | 3.0× | \$0.0007 |

**A second bug caught in the process, same family as the first.** Fixing the
first excluded-row problem (CPU latency priced on a GPU instance) left a
second one: `cost_qps.py` selected "the newest MIND profile" for *both*
pricing rows. Once the GPU profile became the newest file on disk, that same
selection bug would have silently priced GPU latency onto the CPU instance
row too — the identical mistake, mirrored. Fixed by filtering candidate
profiles on their recorded `setup.nrms_device` before selecting the newest
match for each instance, with 4 new unit tests pinning the device-selection
logic (`tests/unit/test_cost_qps.py`) since this script had none before.

## Q4.3 — Cost per 1,000 queries (`benchmarks/cost_qps.py`)

Prices are **AWS on-demand list, us-east-1, captured 2026-09-12**: c6i.2xlarge (8 vCPU)
$0.340/h, g4dn.xlarge (4 vCPU, 1×T4) $0.526/h. Not measured by this project.

| System | Instance | QPS/worker | $/1k (1 worker) | $/1k (all vCPU, optimistic) |
|---|---|---:|---:|---:|
| MIND | c6i.2xlarge | 30.0 | **$0.0031** | $0.0004 |
| EB-NeRD (`nopos`) | c6i.2xlarge | 108.5 | **$0.0009** | $0.0001 |

**The range is the answer, not the lower bound.** The optimistic column assumes perfect
linear scaling across vCPUs; this ADR itself measured a **2.5× slowdown from a single
concurrent job** on one machine, so real throughput is sublinear and the truth sits between
the columns.

**One row is deliberately excluded as invalid.** `cost_qps.py` will also price MIND on
g4dn.xlarge, but that run measured latency with `--device cpu`, so it applies CPU latency to
a GPU instance. This ADR measured NRMS **8.98× faster on a 2080 Ti**, so the GPU figure would
be far off. Fixing it needs a GPU per-request run, which is blocked: QoS `low` allows one GPU
and the MIND treatment (job 2694992) holds it for ~36 h. **Recorded as a gap, not estimated.**

## Q4.4 — What breaks at 10×, per dataset

- **MIND breaks on latency, not memory.** The index is 164 MB and grows with articles and
  vocabulary, not users: MINDlarge-test's 120,959 articles imply ~250 MB, still trivial. But
  p99 headroom is 1.1×, and NRMS is 78% of a request. 10× the load therefore needs ~10×
  the workers, or the NRMS candidate-vector cache this ADR measured at **7.02× on CPU** (no
  accuracy change), or GPU serving at **8.98×**. Of the three, the cache is free.
- **EB-NeRD breaks on memory, and specifically on user profiles.** They are 82.1 MB for
  15,342 users = **5.35 KB/user**, while the article table (108.2 MB) scales with articles.
  At `ebnerd_large`'s **791,582 users** the same rate projects **~4.2 GB** of profiles — from
  a 190 MB feature store to something that no longer fits comfortably beside a model on a
  small instance. Latency is not the constraint here (4.4× headroom).

## Reconciliation with this ADR's original numbers

| Quantity | Originally recorded | Measured now | Verdict |
|---|---|---|---|
| MIND batched, M3 CPU | 43.70 ms/impression | 40.46 ms/impression | reproduces (7% faster) |
| BM25 weight matrix | 23.5 MB | 23.5 MB | exact |
| BM25 full-retrieval projection | 15.4–21.4 min | 20.23 min | inside the range |
| EB-NeRD booster | 60 features (pre-correction) | **65 features** (retrained) | caveat closed |

# Affected Files

**New:** `benchmarks/{timing,loaders,profile_mind,profile_ebnerd,ablations,snapshot,reconcile_adr006}.py`,
`benchmarks/{README,PERFORMANCE_LOG}.md`, `benchmarks/results/`,
`benchmarks/ada_perf_{prep,profile,resume}.sbatch`, `benchmarks/ada_reconcile.sbatch`,
`.githooks/pre-commit`, this ADR.

**Modified:** `Makefile` (bench targets), `PROJECT_STATE.md`.

**Unmodified:** all of `src/`. This ADR measures the pipeline; it does not change it.
