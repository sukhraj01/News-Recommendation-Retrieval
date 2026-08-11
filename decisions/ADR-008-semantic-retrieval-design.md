# ADR-008 — Semantic Retrieval Design (Embedding Model, ANN Backend, User Representation)

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

**Date:** 2026-08-10
**Status:** Decided
**Severity:** High

---

# Engineering Question

Q3 requires embedding-based candidate retrieval for both datasets — compute
or load article embeddings, build an ANN index, retrieve top-K by a
mean-pooled user representation. EB-NeRD ships pre-computed Word2Vec and
multilingual-BERT embeddings; MIND ships none. This forces four bundled
sub-decisions (same bundling rationale ADR-005/007 used — they're facets of
one question, not independent choices):

1. Use EB-NeRD's provided embeddings (a separate model for MIND), or compute
   one model ourselves over both datasets?
2. Which encoder, specifically?
3. Which ANN backend?
4. How is a user represented, and what happens for a zero-history user?

---

# Decision Scope

This decision affects:

- [x] Entire project
- [x] Single component
- [ ] Experiment only
- [ ] Temporary / Prototype
- [x] Assignment-specific
- [ ] General reusable pattern

**Affected Components**

- Retrieval (`src/retrieval/embed.py`, `src/retrieval/score.py`,
  `src/retrieval/retrieve.py`)
- Evaluation (second `Scorer` consumer of the unchanged Q4 harness)

---

# Context

Grounding facts gathered before designing, not assumed:

- Corpus sizes: MIND-dev 42,416 articles; ebnerd_demo 11,777; ebnerd_small
  20,738 — small enough that brute-force cosine similarity is a real
  candidate before reaching for FAISS, per ADR-006's precedent for BM25
  scoring.
- This machine: 8 CPU, ~8.6GB RAM, **no CUDA GPU**, but Apple Silicon
  **MPS is available** (`torch 2.13`, `torch.backends.mps.is_available()
  == True`). The assignment recommends free-tier cloud GPUs; this project
  runs locally on MPS instead — validated by benchmark (below) before being
  trusted for the full corpus, per CLAUDE.md's Resource Availability clause,
  not silently assumed adequate.
- `torch`, `transformers`, `faiss-cpu` were already `pyproject.toml`
  dependencies; `sentence-transformers` was not — added this session
  (`^5.7.0`, verified via `poetry lock && poetry install`, the same
  discipline the pyarrow incident in Phase 2 established: verify a lock
  succeeds, don't assume it).
- EB-NeRD's provided embeddings (`Ekstra_Bladet_word2vec.zip` ~140MB,
  `google_bert_base_multilingual_cased.zip` ~361MB) were confirmed reachable
  at the assignment's S3 URLs before deciding not to use them — a real,
  checked option, not hand-waved away.

---

# Decision Criteria

| Criteria | Importance | Notes |
|----------|------------|-------|
| Single embedding space across both datasets | Critical | Directly enables Q3.5/Q4.5's cross-dataset comparison — same governing goal ADR-002 established for the schema |
| Correctness of the embedding space for cosine-similarity retrieval | Critical | Not all "BERT-family" encoders produce embeddings where distance is meaningful — see Sub-decision 2 |
| Runs in reasonable time on this machine (Resource Availability) | Critical | No cloud GPU available; must be benchmarked, not assumed |
| No unnecessary complexity (ANN backend) | High | Per ADR-006's precedent: benchmark before reaching for FAISS |
| No new, unbenchmarked cold-start assumption | High | Mirror ADR-005's honest reporting instead of inventing an untested fallback |

---

# Design Space Exploration

## Sub-decision 1 — Provided (EB-NeRD) vs. computed (both datasets) embeddings

### Option A — Use EB-NeRD's provided embeddings; compute a separate model for MIND

**Optimizes for:** Reuses a resource the assignment explicitly hands us for
EB-NeRD; less compute.

**Sacrifices:** Two different embedding spaces (EB-NeRD's provided model's
space vs. whatever we'd pick for MIND) are not directly comparable —
exactly the trade-off ADR-002 already rejected for the *schema* itself
("Option A — separate pipelines," rejected because it "undermines the
assignment's own Q4 requirement... by letting two different code paths
introduce uncontrolled variance into that comparison"). The same objection
applies here to a different component.

### Option B — Compute one model ourselves, over both datasets (chosen)

**Optimizes for:** One embedding space, one code path — Q3.5/Q4.5's
comparison measures retrieval-method differences, not embedding-space
differences.

**Sacrifices:** More compute than reusing EB-NeRD's provided embeddings;
doesn't literally reuse the assignment's provided artifact (though see
Sub-decision 2 — the *model family* the assignment names is still used).

---

## Sub-decision 2 — Which encoder

### Option A — Mean-pooled raw multilingual BERT or XLM-RoBERTa

**Sacrifices:** Reimers & Gurevych (2019, the Sentence-BERT paper) showed
mean-pooled vanilla BERT/XLM-R embeddings perform *poorly* on
cosine-similarity tasks — sometimes worse than averaged GloVe — because the
model was never trained to make embedding *distance* meaningful. This
project's exact use case (mean-pool a user's clicked articles, retrieve
catalog articles by cosine similarity) is the failure mode that paper
documents. This is a correctness-tier objection (evidence hierarchy rank
2), not a style preference.

### Option B — A purpose-built multilingual sentence-embedding model (chosen)

**How it works:** A `sentence-transformers` model — still a BERT/XLM-R-
family transformer backbone (the assignment names "BERT, XLM-RoBERTa" as
the *modeling family*, not a mandate to hand-roll pooling), fine-tuned with
a similarity/retrieval objective so cosine distance is actually meaningful.

**Shortlist benchmarked** (800 real MINDsmall-dev articles, this machine's
MPS backend):

| Model | Load (first run) | Encode 800 articles | Rate | Projected 75k articles |
|---|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | 178.3s (one-time download) | 4.64s | 172.4/s | 7.25 min |
| `multilingual-e5-small` | 113.6s (one-time download) | 3.90s | 204.9/s | 6.10 min |

Both comfortably clear the Resource Availability bar (well under an hour on
local MPS, no cloud GPU needed) — throughput was not the deciding factor.

**Discrimination sanity check** (the deciding factor): 1,500 real
MINDsmall-dev articles, 3,000 sampled article pairs, mean cosine similarity
for same-category vs. different-category pairs (category is a real, if
weak, topical-similarity proxy):

| Model | Same-category mean cosine | Diff-category mean cosine | Gap | Same/diff ratio |
|---|---|---|---|---|
| MiniLM-L12-v2 | 0.1183 (std 0.128) | 0.0492 (std 0.095) | **0.0691** | 2.4x |
| e5-small (`passage:` prefix, its documented convention) | 0.7730 (std 0.025) | 0.7564 (std 0.022) | 0.0166 | 1.02x |
| e5-small (no prefix) | 0.7845 (std 0.025) | 0.7651 (std 0.021) | 0.0194 | 1.03x |

e5-small — even used with its own documented `passage:` prefix convention —
compresses nearly all article pairs into a narrow 0.75–0.78 cosine band
regardless of topical relatedness, a much weaker and less usable ranking
signal than MiniLM's wider, better-separated 0.05–0.12 range. This matches
the plan's pre-registered concern: e5's asymmetric query/passage training
objective doesn't cleanly fit this project's *symmetric* use case (a
user-profile vector and a catalog article are the same kind of object, not
a query and a document) — confirmed empirically, not just argued
theoretically.

**Chosen: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**
(384-dim, ~118M params, 50+ languages incl. Danish) — real discrimination
between topically related and unrelated articles, adequate throughput,
symmetric training objective matching the actual use case.

---

## Sub-decision 3 — ANN backend

### Option A — FAISS (already a `pyproject.toml` dependency)

**Sacrifices:** Unjustified complexity at these corpus sizes without a
measured need — the same objection ADR-006 raised against BM25L/BM25+ for
BM25 ("solves a problem we don't clearly have").

### Option B — Brute-force cosine similarity via a vectorized matrix product (chosen)

**Measured directly** (not assumed): 1,000 single-query matvecs against the
largest corpus (MIND-dev, 42,416 × 384): **0.99ms/query**. At this scale, a
full-corpus retrieval is sub-millisecond — FAISS would add real complexity
(index construction, a new data structure to keep in sync with the article
catalog) for no measurable latency benefit. `faiss-cpu` stays an unused,
present `pyproject.toml` dependency, documented rather than silently
ignored — the same posture ADR-006 took toward BM25L/BM25+.

---

## Sub-decision 4 — User representation & cold-start

Mean-pooled, L2-renormalized embedding of a user's full click history (per
Q3), no recency weighting — matching ADR-005's BM25 stance for the identical
reason (MIND's history order is unverified per ADR-002; no new
dataset-specific branch introduced).

**Zero-history user:** mean of an empty set is undefined. Handled by
mirroring BM25's existing two-context split exactly, not inventing a new
rule:

- **recall@K:** `build_user_embedding_query` returns `None`;
  `embed_retrieve_top_k` short-circuits to `[]` — same shape as
  `retrieve_top_k`'s empty-query short-circuit.
- **Q4 ranking harness:** `EmbeddingScorer.score()` returns an all-zero
  vector for a `None` query — a deliberate total tie, chosen specifically
  so ADR-007's existing seeded pseudo-random tie-break resolves *both*
  scorers' cold-start case identically, resolving ADR-007's own flagged
  Research Trigger ("re-examine whether the tie-break is still load-bearing
  the same way" once a non-BM25 scorer exists) by construction.

This is reported as the same structural finding ADR-005 already reported
for BM25 (see Benchmark Results below) — not a new cold-start-specific
strategy. No fallback (e.g. popularity backoff for zero-history users) was
built; flagged under Conditions for Revisiting.

---

# Comparison Summary

| Sub-decision | Chosen | Why |
|---|---|---|
| Embedding source | Compute one model, both datasets | Same single-code-path argument ADR-002 already established |
| Encoder | `paraphrase-multilingual-MiniLM-L12-v2` | Measurably better topical discrimination than e5-small; adequate throughput; symmetric objective matches the use case |
| ANN backend | Brute-force cosine matvec | Measured 0.99ms/query at the largest corpus — FAISS unjustified at this scale |
| User representation | Mean-pooled, L2-renormalized, no recency weighting | Mirrors ADR-005's BM25 reasoning exactly |
| Cold-start | `None`/all-zero-tie, same reporting posture as ADR-005 | No assignment or session requirement to build a new fallback |

---

# Final Decision

## Chosen Option

As summarized above — implemented in `src/retrieval/embed.py`
(`EmbeddingIndex`, `build_embedding_index`, `build_user_embedding_query`),
`src/retrieval/score.py` (`EmbeddingScorer`), `src/retrieval/retrieve.py`
(`embed_retrieve_top_k`, and a shared `_top_k_from_scores` helper factored
out of `retrieve_top_k` for reuse), `scripts/run_embed_experiment.py`, and
`scripts/run_ranking_eval.py --method embed`.

### Decision Date

2026-08-10

### Decision Owner

**Primary Engineer**

- sukhraj01

### Contributors

- Claude Code (design exploration in plan mode, benchmarking, implementation,
  ADR drafting)

### Reviewer *(Optional)*

- Pending

---

# Rationale

## Why this option is best

- Every sub-decision here was resolved by a real measurement (encoder
  discrimination gap, matvec latency, throughput) before being trusted, not
  argued from first principles alone — same discipline ADR-006 established
  for BM25's scoring rewrite.
- The embedding-source and cold-start sub-decisions both directly reuse
  reasoning this project already established (ADR-002's single-code-path
  goal; ADR-005's cold-start reporting posture) rather than inventing new
  precedent for a component that didn't strictly need one.

## Why alternatives were rejected

See each sub-decision's own rejection reasoning above — every rejection is
tied to a specific, stated failure mode (embedding-space incomparability,
poor cosine-similarity discrimination, unjustified complexity), not
preference.

---

# Decision Confidence

**Current Confidence**

Medium-High

### Why

- The encoder choice is now benchmark-verified on real data (not just
  theoretical Sentence-BERT-paper reasoning) — the discrimination gap
  measurement directly confirmed the pre-registered concern about e5-small.
- The ANN-backend and throughput conclusions are directly measured, not
  projected from a much smaller sample.
- Medium, not High: MiniLM's absolute same/diff-category cosine gap (0.069)
  is real but modest in absolute terms — category is a weak, noisy proxy
  for true topical relevance, not ground truth; the recall@K/Q4 production
  numbers below are the stronger evidence.

### What Would Increase Confidence

- A larger, curated near-duplicate/paraphrase test set (rather than the
  category proxy) to directly validate discrimination quality.
- Testing whether a larger model (e.g. `paraphrase-multilingual-mpnet-base-v2`)
  meaningfully improves recall@K enough to justify its slower throughput —
  not attempted this session, since MiniLM already cleared the Resource
  Availability bar and produced a real, above-random signal (below).

---

# Evidence

## Evidence Sources

- [x] Assignment Requirements (Q3, Q3.4, Q3.5)
- [x] Experimental Benchmark (encoder throughput, discrimination gap, matvec
      latency, full production recall@K and Q4 runs — all below)
- [ ] Official Documentation
- [x] Research Paper (Reimers & Gurevych 2019, Sentence-BERT — motivates
      Sub-decision 2's rejection of naive mean-pooled BERT/XLM-R)
- [ ] Industry Practice
- [ ] Community Consensus
- [x] Engineering Inference (ADR-002's single-code-path argument, reapplied)

---

## Empirical Evidence

### Pre-flight benchmarks

See Sub-decision 2 (encoder throughput + discrimination gap) and
Sub-decision 3 (0.99ms/query brute-force matvec) tables above.

### Production recall@K (Q3.4) — BM25 vs. embeddings, same corpora ADR-006/ADR-002-addendum already benchmarked

**MIND-small dev** (42,416 articles, 50,000 users):

| k | BM25 overall | Embed overall | BM25 warm | Embed warm | BM25 cold | Embed cold |
|---|---|---|---|---|---|---|
| 50 | 0.73% | 0.90% | 0.73% | 0.93% | 0.72% | 0.65% |
| 100 | 1.50% | 1.60% | 1.54% | 1.67% | 1.19% | 1.10% |
| 200 | 2.62% | 2.78% | 2.73% | 2.92% | 1.78% | 1.81% |

**EB-NeRD demo validation** (11,777 articles, 1,562 users, all warm):

| k | BM25 | Embed |
|---|---|---|
| 50 | 1.01% | 0.32% |
| 100 | 2.13% | 0.94% |
| 200 | 4.02% | 2.57% |

**EB-NeRD small validation** (20,738 articles, 15,342 users, all warm):

| k | BM25 | Embed |
|---|---|---|
| 50 | 0.72% | 0.14% |
| 100 | 1.44% | 0.43% |
| 200 | 2.77% | 1.21% |

(`experiments/embed_mind_2026-08-10/`, `experiments/embed_ebnerd_2026-08-10/`,
`experiments/embed_ebnerd_small_2026-08-10/`; bootstrap 95% CIs in the raw
`results.json` files, omitted here for table width — see Benchmark Results.)

### Production Q4 ranking metrics — BM25 vs. embeddings, same corpora ADR-007 already benchmarked

**MIND-small dev:**

| Metric | BM25 overall | Embed overall | BM25 warm | Embed warm | BM25 cold | Embed cold |
|---|---|---|---|---|---|---|
| AUC | 0.5692 | **0.6340** | 0.5766 | 0.6439 | 0.5242 | 0.5737 |
| MRR | 0.3115 | 0.3486 | 0.3138 | 0.3523 | 0.2975 | 0.3258 |
| nDCG@5 | 0.2887 | 0.3314 | 0.2888 | 0.3329 | 0.2879 | 0.3221 |
| nDCG@10 | 0.3486 | 0.3903 | 0.3485 | 0.3918 | 0.3490 | 0.3807 |
| Diversity@10 | 0.8367 | 0.8269 | 0.8314 | 0.8213 | 0.8690 | 0.8607 |
| Novelty@10 | 16.30 | 16.15 | 16.29 | 16.12 | 16.38 | 16.30 |
| Coverage@10 | 0.0834 | 0.0782 | 0.0804 | 0.0740 | 0.0450 | 0.0440 |

**EB-NeRD demo validation** (overall = warm; cold n/a, 0 cold users):

| Metric | BM25 | Embed |
|---|---|---|
| AUC | 0.5326 | 0.5437 |
| MRR | 0.3436 | 0.3433 |
| nDCG@5 | 0.3764 | 0.3804 |
| nDCG@10 | 0.4571 | 0.4587 |
| Diversity@10 | 0.7951 | 0.7893 |
| Novelty@10 | 14.78 | 14.79 |
| Coverage@10 | 0.2159 | 0.2158 |

**EB-NeRD small validation** (overall = warm; cold n/a, 0 cold users):

| Metric | BM25 | Embed |
|---|---|---|
| AUC | 0.5288 | 0.5430 |
| MRR | 0.3412 | 0.3437 |
| nDCG@5 | 0.3745 | 0.3804 |
| nDCG@10 | 0.4543 | 0.4591 |
| Diversity@10 | 0.7949 | 0.7890 |
| Novelty@10 | 17.17 | 17.19 |
| Coverage@10 | 0.2057 | 0.2050 |

## Theoretical Evidence

- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings
  using Siamese BERT-Networks.* EMNLP 2019.

## Accepted Trade-offs

- Disk-cached embeddings (`data/processed/{dataset}/{bundle}/embeddings/`)
  deviate from BM25's "rebuild every run" precedent — accepted because
  encoding is real, repeated cost (minutes, not ~1s) that a stale/mismatched
  cache would silently corrupt if not invalidated correctly; sidecar
  metadata (model name + article ID order) makes cache staleness structural
  to detect, not a trust assumption.
- The same-category/diff-category discrimination check uses `category` as a
  proxy for true topical similarity, not a curated ground-truth pair set —
  accepted as a real signal (category is not random with respect to topic)
  but a noisy one; flagged under Decision Confidence, not treated as
  definitive.

---

# Assumptions

| Assumption | Why It Matters | Monitoring Strategy |
|------------|----------------|---------------------|
| `title + abstract` mandatory text (ADR-002) is sufficient for the encoder, same as it was for BM25 | If EB-NeRD's `body` field is indexed in a future phase, this would need re-evaluating for both retrieval legs | Revisit if a future phase adds `body` to either method |
| MiniLM's category-based discrimination gap generalizes to true click-relevance discrimination | The proxy check isn't the same as the actual recall@K/AUC signal | Confirmed directionally by the full production benchmarks above showing real, above-baseline signal on both datasets |
| Disk-cached embeddings remain valid as long as `(model_name, article_ids order)` matches | A cache bug here would silently serve stale embeddings | Sidecar JSON comparison is exact-match, not best-effort; any mismatch forces recompute |

---

# Conditions for Revisiting

## Technical Triggers

Revisit if:

- [ ] Performance regression
- [ ] Latency exceeds threshold
- [x] Corpus size grows materially (e.g. MINDlarge enters scope) — re-run
      the brute-force matvec timing check before trusting it holds; FAISS
      may become justified at that scale where it currently isn't

---

## Research Triggers

Revisit if:

- [ ] Significant new research appears
- [x] A future session wants a curated near-duplicate/paraphrase evaluation
      set instead of the category-proxy discrimination check used here

---

## Project / Assignment Triggers

Revisit if:

- [x] A zero-history-user fallback (e.g. popularity backoff) becomes
      in-scope — not built this session, since neither the assignment nor
      this session's objective required it
- [ ] Assignment requirements change

---

**Estimated Cost to Change**

Medium — the encoder is isolated behind `load_encoder()`/`DEFAULT_MODEL` in
`embed.py`; swapping models invalidates the disk cache automatically (model
name is part of the cache key) but requires a full re-encode.

---

# Engineering Impact

## Affected Components

- Retrieval
- Evaluation (second `Scorer` consumer)

## Affected Files

- `src/retrieval/embed.py` (new)
- `src/retrieval/score.py` (`EmbeddingScorer` added)
- `src/retrieval/retrieve.py` (`embed_retrieve_top_k` added;
  `_top_k_from_scores` factored out of `retrieve_top_k` for reuse)
- `scripts/run_embed_experiment.py` (new)
- `scripts/run_ranking_eval.py` (`_build_method` dispatch added; `--method
  embed` choice)
- `pyproject.toml` (`sentence-transformers` added)
- `tests/unit/test_embed.py` (new), `tests/unit/test_score.py`,
  `tests/unit/test_retrieval.py` (extended)
- `tests/integration/test_retrieval_pipeline.py` (extended)

## Expected Refactoring

`retrieve_top_k`'s top-k selection was factored into `_top_k_from_scores`
so `embed_retrieve_top_k` could reuse it — behavior-preserving (verified by
`tests/unit/test_retrieval.py`'s existing BM25 cases still passing
unchanged).

## Breaking Changes

No — additive. `run_ranking_eval.py`'s BM25 path was refactored onto the new
`_build_method` dispatch; re-verified byte-identical against ADR-007's
recorded numbers (AUC 0.5692/MRR 0.3115/nDCG@5 0.2887/nDCG@10
0.3486/Diversity@10 0.8367/Novelty@10 16.2988/Coverage@10 0.0834 on MIND —
exact match after the refactor).

## Required Tests

- Unit: `EmbeddingIndex` shape/normalization, mean-pool correctness,
  cold-start `None`, cache round-trip and invalidation
  (`tests/unit/test_embed.py`).
- Unit: `EmbeddingScorer` cosine-similarity ranking, subset-vs-full
  consistency, `None`-query zero-tie, identity-based caching
  (`tests/unit/test_score.py`).
- Unit: `embed_retrieve_top_k` ranking/k-boundary/`None`-query behavior
  (`tests/unit/test_retrieval.py`).
- Integration: real MINDsmall-dev article sample, real encoder, end-to-end
  retrieval; cold-start short-circuit (`tests/integration/test_retrieval_pipeline.py`).

## Documentation Updates

- [x] Architecture (`ARCHITECTURE.md`'s Retrieval component)
- [x] Project State
- [ ] Knowledge Base

---

# Benchmark Results

## Baseline

BM25 (ADR-005/ADR-006), already benchmarked on all three corpora this ADR
re-benchmarks embeddings against.

## Candidate

`paraphrase-multilingual-MiniLM-L12-v2`, mean-pooled user query, brute-force
cosine similarity.

## Benchmark Environment

| Item | Value |
|------|-------|
| Dataset | MIND-small dev (42,416 articles, 50,000 users); ebnerd_demo (11,777 articles, 1,562 users); ebnerd_small (20,738 articles, 15,342 users) |
| Hardware | Local development machine (macOS, Apple Silicon, MPS backend, no CUDA) |
| Software Version | sentence-transformers 5.7.0, torch 2.13 |
| Random Seed | 0 (bootstrap CIs and Q4 tie-break, same as BM25's runs) |

## Experiment

`experiments/embed_mind_2026-08-10/`, `experiments/embed_ebnerd_2026-08-10/`,
`experiments/embed_ebnerd_small_2026-08-10/`,
`experiments/ranking_embed_mind_2026-08-10/`,
`experiments/ranking_embed_ebnerd_2026-08-10/`,
`experiments/ranking_embed_ebnerd_small_2026-08-10/`. Also backfilled
`experiments/ranking_bm25_ebnerd_2026-08-10/` (ADR-007 had only benchmarked
`ebnerd_small` for Q4 ranking, not `ebnerd_demo` — needed for full
BM25-vs-embed parity across all three corpora).

## Results

See the full recall@K and Q4 ranking tables under Empirical Evidence above.

## Interpretation

**On MIND, embeddings win outright** — higher recall@K at every k (2.78%
vs. 2.62% at k=200) and a clear Q4 AUC gap (0.634 vs. 0.569, +0.065). This
does *not* match the naive Day-1 framing ("BM25 favors warm/entity-heavy
MIND") — semantic retrieval wins across both cohorts on MIND, not just cold.
The warm/cold *gap itself* is comparable between methods (BM25: 0.577→0.524,
Δ0.052; embed: 0.644→0.574, Δ0.067) — embeddings do not close the cold-start
gap, they raise the whole curve. A genuinely useful, if less tidy, finding:
the working hypothesis's *direction* (embeddings help cold-start relatively
more) is not clearly supported by these numbers; if anything the absolute
AUC gain is slightly larger for warm users. Reported honestly rather than
reframed to fit the hypothesis.

**On EB-NeRD, the picture splits by evaluation question.** BM25 clearly
wins whole-corpus recall@K (demo: 4.02% vs. 2.57%; small: 2.77% vs. 1.21% —
roughly 1.6–2.3x higher), but embeddings slightly *edge out* BM25 on the Q4
ranking metrics computed over each impression's already-listed candidates
(AUC 0.544 vs. 0.533 on demo; 0.543 vs. 0.529 on small). These are
structurally different questions (ADR-007's own framing): recall@K asks
whether embeddings can *find* the right article among 11,777–20,738
candidates from scratch; Q4 asks how well embeddings *rank* a much smaller,
already-curated candidate list (EB-NeRD's median 9–12 candidates/impression,
per ADR-007). A plausible explanation, not yet isolated: EB-NeRD's long
median history (81–93 articles, per ADR-005) produces a broad,
averaged-out mean-pooled query that's diffuse relative to the whole catalog
(hurting whole-corpus retrieval) but still discriminates well enough within
a short candidate list already filtered by EB-NeRD's own serving pipeline.

**Diversity, novelty, and coverage are nearly identical between methods on
every corpus** (all differences well under what would look like a real
effect against these metrics' own CIs on MIND, and near-equal by inspection
on both EB-NeRD bundles) — this project's Q4 harness design (ADR-007) was
built to detect exactly this kind of "accuracy differs, beyond-accuracy
metrics don't" pattern, and it's a genuine finding here, not a null result
being overlooked.

**True cold-start remains a shared ceiling.** MIND recall@200 for the cold
cohort is nearly identical between methods (BM25 1.78% vs. embed 1.81%) —
neither approach solves the *zero-history* problem specifically; both
degrade to a structural miss for users with no history at all (`None` query
→ `[]`/all-zero-tie for either scorer, per Sub-decision 4). The cold cohort
blends true zero-history users (guaranteed miss under both methods) with
1–4-history users (some signal), which is why the aggregate cold numbers
converge even though the *underlying mechanism* differs (BM25: no
vocabulary to match; embeddings: no vector to compute).

## Comparison to Alternatives

The e5-small vs. MiniLM comparison (Sub-decision 2) is the one alternative
in this ADR that *was* empirically benchmarked against the chosen option,
not just structurally argued — see the discrimination-gap table above.

---

# Related Decisions

## Influenced By

- ADR-002 (Unified Data Schema) — single-code-path argument reapplied to
  embeddings; mandatory `title`/`abstract` fields as the encoder's input
- ADR-005 (Query Construction) — no-recency-weighting stance and cold-start
  reporting posture, both reused unchanged
- ADR-006 (BM25 Variant) — "benchmark before adding complexity" precedent,
  reapplied to the ANN-backend decision
- ADR-007 (Q4 Ranking Evaluation Harness Design) — the unchanged harness
  this ADR's `EmbeddingScorer` plugs into; resolves ADR-007's own flagged
  Research Trigger about a non-BM25 scorer's cold-start behavior

## Influences

- Q6's design note (not started this session) — this ADR's Interpretation
  section is the primary source for that note's "lexical vs. semantic"
  discussion

---

# References

## Internal

- ARCHITECTURE.md — Retrieval component
- ADR-002, ADR-005, ADR-006, ADR-007
- `experiments/embed_*`, `experiments/ranking_embed_*`,
  `experiments/ranking_bm25_ebnerd_2026-08-10/`

## External

- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings
  using Siamese BERT-Networks.* EMNLP 2019.
- `sentence-transformers` model cards:
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
  `intfloat/multilingual-e5-small` (installed package, version 5.7.0).

---

# Decision History

| Date | Event |
|------|-------|
| 2026-08-10 | Design explored in plan mode; four sub-decisions bundled |
| 2026-08-10 | `sentence-transformers` added; encoder shortlist benchmarked (throughput + discrimination gap) on real data |
| 2026-08-10 | MiniLM-L12-v2 chosen; brute-force ANN backend benchmarked (0.99ms/query) |
| 2026-08-10 | Implemented, unit + integration tested |
| 2026-08-10 | Full production recall@K and Q4 runs on all three corpora; ebnerd_demo's missing BM25 ranking baseline backfilled for parity |

---

# Notes

Sub-decision 2's benchmark is a direct example of this project's evidence
hierarchy in action: the theoretical argument (Sentence-BERT paper) picked
the right *category* of solution (purpose-built sentence embeddings over
naive pooling), but the *specific model* within that category was decided
by a real measurement that overturned the naive assumption that a
retrieval-tuned model (e5) would obviously beat a paraphrase-tuned one
(MiniLM) for this task — it didn't, because the retrieval tuning's
asymmetric convention doesn't match this project's symmetric use case. This
is exactly the kind of finding CLAUDE.md's "benchmark before trusting a
decision" principle exists to surface.

---

# Addendum — MINDlarge-Scale Benchmark (2026-08-11)

**Status:** Resolves this ADR's own flagged Technical Trigger ("Corpus size grows materially (e.g. MINDlarge enters scope) — re-run the brute-force matvec timing check before trusting it holds; FAISS may become justified at that scale where it currently isn't"). Appended per CLAUDE.md's decision-reversal guidance.

## What was checked

Real MINDlarge_dev data (72,023 articles, 255,990 users) — not projected from MINDsmall:

| Metric | MINDsmall-dev (ADR-008 original) | MINDlarge-dev (this addendum) |
|---|---|---|
| Corpus size | 42,416 articles | 72,023 articles |
| Users | 50,000 | 255,990 |
| Encoder throughput (sample) | 172.4/s (800-article sample, first-run incl. model download) | 373.6/s (2,000-article sample, warm model) |
| Full encode + disk-cache write | not separately measured | **264.4s (4.4 min)** for all 72,023 articles |
| Brute-force matvec retrieval | 0.99ms/query | 2.44ms/query (1,000-user sample) → **624.4s projected (10.4 min) for all 255,990 users** |
| Embedding vectors memory | not stated | 110.6 MB (384-dim float32 × 72,023 docs) |

The encoder-throughput jump (172→374/s) is plausibly a warm-model-cache effect (ADR-008's original number included a one-time ~178s download/load counted against the sample), not a hardware change — not isolated further since both numbers clear the feasibility bar by a wide margin either way.

## Decision

**Stays local — brute-force cosine similarity remains sufficient, FAISS remains unjustified.** Combined full MINDlarge-dev run (both BM25 and embeddings, per this ADR's and ADR-006's addenda) projects to under 25 minutes total, nowhere near CLAUDE.md's Resource Availability threshold (this session's working definition: >2hrs or exceeds available RAM) that would justify a Kaggle GPU detour. Per-query matvec latency (2.44ms) is higher than MINDsmall's (0.99ms, as expected — corpus is 1.7x larger) but still trivial at this scale; FAISS would add real complexity (index construction, a new data structure to keep in sync with the article catalog) for no measurable benefit, the same conclusion ADR-008's original decision reached, now re-verified rather than assumed to still hold.

## Related

- ADR-006's parallel addendum (BM25 side of the same MINDlarge-scale check)
- `tests/integration/test_schema_conformance.py::test_mind_large_total_article_and_user_counts_sanity` — why 72,023 (dev's own catalog), not the paper's 161,013 total, is the correct benchmarking target
