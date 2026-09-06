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

---

# Addendum — Contrastive-Vector-vs-MiniLM Isolated Comparison (2026-08-19)

**Status:** Does not reverse this ADR's decision (Sub-decision 1: compute one
model ourselves, over both datasets). Sub-decision 1's original objection —
using EB-NeRD's provided artifact for EB-NeRD and a separate model for MIND
breaks the single-embedding-space comparison Q3.5/Q4.5 depends on — still
holds and was never in question here. What this addendum resolves is
narrower: that rejection's *accuracy cost*, specifically, was argued but
never measured. It now has been, on `ebnerd_small` validation only (this
artifact doesn't cover MIND at all, so the single-code-path problem Option A
was rejected for isn't even avoidable outside EB-NeRD). Appended per
CLAUDE.md's decision-reversal guidance — history preserved, not overwritten.

## What was checked

Same eval harness, same `ebnerd_small` validation split, same warm/cold
slicing, same bootstrap CI machinery as this ADR's original MiniLM numbers —
the only variable changed is the embedding source. New code
(`scripts/run_contrastive_vector_experiment.py`) adds exactly one new
function, `load_contrastive_index`, which builds an `EmbeddingIndex` (the
same dataclass this ADR's `embed.py` defines) from EB-NeRD's provided
`Ekstra_Bladet_contrastive_vector.zip` instead of encoding text with MiniLM.
`build_user_embedding_query`, `EmbeddingScorer`, `embed_retrieve_top_k`,
`recall_at_k`, and every Q4 ranking metric are imported and reused with zero
modification — `embed.py`, `score.py`, `retrieve.py`,
`run_embed_experiment.py`, `run_ranking_eval.py`, and this ADR's own decision
are all untouched.

**Artifact, inspected directly rather than assumed:** a single
`Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet`, 125,541 rows
(EB-NeRD's full article catalog — this project's `ebnerd_small` corpus,
20,738 articles, is a subset), 768-dim vectors, 100% coverage of
`ebnerd_small`'s local corpus (0 missing). Format confirmed against
`ebnerd-benchmark`'s own reproducibility scripts before writing any loading
code (`examples/reproducibility_scripts/ebnerd_nrms_docvec.py`:
`create_article_id_to_value_mapping(df=df_articles,
value_col=df_articles.columns[-1])` — an article-id column plus the vector
as the last column), not guessed from the filename. **No public
documentation of the training methodology (base model, contrastive
objective, which text fields) was found anywhere** — not in the paper, not
in `ebnerd-benchmark`'s README, not in the archive itself (no accompanying
README/txt/md/json shipped alongside `contrastive_vector.parquet`). This is
a real evidence gap, reported as one rather than papered over with an
inferred guess: this project can confirm *what* the artifact is (per-article
768-dim vectors, full-catalog coverage) and *how well it performs*
(below), but not authoritatively *how it was produced*.

**Execution:** local sandbox network to EB-NeRD's S3 bucket proved
unreliable for the 341MB artifact (throughput 4-170KB/s across three
attempts, one ending in a mid-transfer connection reset after several
hours, never completing) — a real, named resource constraint per CLAUDE.md's
Resource Availability clause, not silently worked around. Moved to Kaggle
(reliable network for the same download), reusing the project's existing
Kaggle-relay pattern (`notebooks/ebnerd_contrastive_vector_kaggle_run.py` +
`ebnerd_contrastive_vector_src_bundle.zip`, mirroring Part 0/Part 2's
EB-NeRD Codabench notebooks): the notebook rebuilds `ebnerd_small` via this
project's own unmodified `build_ebnerd_bundle`, guaranteeing byte-identical
schema to what produced the MiniLM baseline, before running the comparison.
A real bug was caught by a local smoke test (synthetic 90%-coverage
artifact) before this went to Kaggle: `score.py`'s unmodified
`_lookup_scores` scores an uncovered candidate as `-inf` by design, which
`sklearn.roc_auc_score` rejects outright — a path MiniLM's 100%-by-
construction coverage never exercised. Fixed locally in the new script only
(`_finite_scores_for_auc`, substitutes each impression's own minimum finite
score minus 1, preserving `-inf`'s existing "ranks last" behavior without
introducing an actual infinity) — not a change to `score.py`'s contract. In
the event, real coverage came back at 100%, so this path never actually
fired on the real data — but it would have crashed the run without the fix.

## Results

`experiments/contrastive_vector_ebnerd_small_2026-08-18/{config,results}.json`,
compared against this ADR's own `experiments/embed_ebnerd_small_2026-08-10/`
and `experiments/ranking_embed_ebnerd_small_2026-08-10/` MiniLM numbers:

| Metric | MiniLM (this ADR) | Contrastive vector | CI-clear winner? |
|---|---|---|---|
| recall@50 | 0.14% | 0.58% | **Contrastive** |
| recall@100 | 0.43% | 1.20% | **Contrastive** |
| recall@200 | 1.21% | 2.47% | **Contrastive** |
| AUC | 0.5430 | 0.5453 (95% CI 0.5435–0.5471) | **Contrastive**, but narrow — the CI floor clears the MiniLM point estimate by only 0.0005 |
| MRR | 0.3437 | 0.3527 (95% CI 0.3510–0.3544) | **Contrastive** |
| nDCG@5 | 0.3804 | 0.3870 (95% CI 0.3849–0.3891) | **Contrastive** |
| nDCG@10 | 0.4591 | 0.4660 (95% CI 0.4641–0.4678) | **Contrastive** |
| Diversity@10 | 0.7890 | 0.7800 (95% CI 0.7784–0.7817) | **MiniLM** — contrastive's CI sits entirely below MiniLM's point estimate |
| Novelty@10 | 17.19 | 17.218 (95% CI 17.205–17.231) | **No clear winner** — MiniLM's point estimate falls inside contrastive's own CI |
| Coverage@10 | 0.2050 | 0.2043 | Point estimate only, no CI (ADR-007) — near-identical, consistent with this ADR's own "beyond-accuracy metrics barely differ between methods" pattern |

`ebnerd_small`'s cold cohort is structurally empty (min history length = 5,
same as every other result in this ADR) — no warm/cold split possible;
`overall` and `warm` are identical, `cold` is `NaN` throughout, as expected.

## Interpretation

**This is a real, CI-clear accuracy win for the provided artifact on
`ebnerd_small`, not a marginal one** — recall@K roughly triples to quadruples
across all three K values, and every Q4 ranking-accuracy metric (AUC, MRR,
nDCG@5, nDCG@10) clears MiniLM with a non-overlapping CI. **It is not,
however, a blanket "the provided artifact is better."** Diversity@10 is a
CI-clear *loss* — the contrastive vector's top-10 rankings are measurably
less category-diverse than MiniLM's, plausibly consistent with a model
trained to sharply separate near-duplicate/related content (better at
finding the *single most relevant* article, worse at surfacing varied ones
in the same top-K) — plausible, not isolated by a controlled test here.
Novelty ties. The honest summary is a real accuracy-vs-diversity trade-off,
not a strictly dominant option in either direction.

**Sub-decision 1's original rejection reasoning is not undermined by this
result.** The accuracy gain is real and was worth measuring, but the
artifact only exists for EB-NeRD — adopting it would still fracture the
single-embedding-space property Q3.5/Q4.5's cross-dataset comparison
depends on, exactly the objection Option A was rejected for. This addendum
answers "what would it cost us to reject the provided artifact" (answer: a
real, now-quantified amount of ranking accuracy on EB-NeRD specifically,
traded for cross-dataset comparability and a diversity edge) — it does not
answer "should we use it instead," which remains a genuinely separate
decision (a second, EB-NeRD-only leaderboard submission) for the engineer to
make with this evidence in hand, not one this ADR resolves unilaterally.

## Conditions for Revisiting (this addendum's own)

- If a second, EB-NeRD-only Codabench submission using this artifact is
  decided on, that is a new decision point (separate ADR or explicit
  addendum here), not an automatic consequence of this result.
- The training-methodology documentation gap (noted above) means this
  result's generalization beyond `ebnerd_small` validation — e.g. to
  `ebnerd_large` or the real held-out test set — is unverified; the
  artifact's own coverage of `ebnerd_large`/testset-era articles was not
  checked here.
- Diversity's CI-clear loss was not investigated beyond a plausible
  hypothesis (sharper same/different discrimination trading off against
  intra-list variety) — a controlled follow-up (e.g. comparing the two
  models' cosine-similarity distributions the way Sub-decision 2's original
  discrimination check did) would be needed to confirm it, not just assert
  it.

## Related

- `scripts/run_contrastive_vector_experiment.py`,
  `notebooks/ebnerd_contrastive_vector_kaggle_run.py`,
  `notebooks/ebnerd_contrastive_vector_src_bundle.zip`
- `experiments/contrastive_vector_ebnerd_small_2026-08-18/`
- `knowledge/ai-usage-log/2026-08-18_contrastive-vector-adr008-addendum.md`
- `docs/design_note.md`/`.tex` §2 and §6 (updated to reflect this measured
  trade-off)

---

# Addendum — Second EB-NeRD Submission: Contrastive Vector on the Real Test Set (2026-08-21)

**Status:** Does not reverse this ADR's decision. This is the direct
follow-up the prior addendum's own Conditions for Revisiting flagged as a
"genuinely separate decision... for the engineer to make" — the engineer
made it, submitting `notebooks/ebnerd_contrastive_vector_testset_kaggle_run.py`'s
output (`load_contrastive_index` swapped in for `build_embedding_index`,
everything else byte-identical to the known-working `ebnerd_part2_kaggle_test_run.py`
reference) to Codabench competition 2469 as a second, independent entry
alongside the existing MiniLM submission (888045).

## What was checked

The new submission (896072, `prediction_contrastive.zip`) and the original
MiniLM submission (888045) both round to an identical leaderboard Score
(0.5404) — before treating that as meaningful (e.g. "the wrong file was
uploaded"), this was verified rather than assumed:

- **Checksums:** different SHA-256/CRC-32/MD5 for both the zip and the
  extracted `predictions.txt` — not a duplicate upload.
- **Line-by-line diff** (13,536,710 lines each): 0 impression-ID
  misalignments (both files walk `ebnerd_testset.zip` in identical row
  order); 99.48% of lines carry a genuinely different ranking for the same
  impression ID; the remaining 0.52% match at the background rate expected
  for impressions with too few candidates for more than one/few possible
  permutations. This is the fingerprint of two independent scoring runs
  over different embedding spaces, not a partial fallback to MiniLM.
- **Cell 8's own validation logic**, re-run locally against the downloaded
  file (line count, malformed-permutation check): clean, 13,536,710/13,536,710,
  0 malformed — confirming the Kaggle run completed and packaged correctly.
  Note: the actual Kaggle-side Cell 8 output was never relayed/logged for
  this run, unlike every prior run in this project — a process gap, not a
  correctness one (caught by re-deriving the same check locally instead).
- **Submission-ID-to-detail-table attribution was genuinely ambiguous at
  first** — Codabench's per-submission detail page displays no submission
  ID, and an initial pass surfaced three candidate per-day tables (one,
  MEAN AUC 0.5123, didn't match either submission and was traced to a
  stray click on an unrelated competitor's row while browsing the public
  leaderboard). Resolved only after the engineer re-opened each submission's
  detail view individually and confirmed which table belonged to which ID
  — not inferred from screenshot timing or engineer recollection alone.

## Results

Codabench's per-submission "Ranking Metrics: Grouped by Selected Dates"
view (covers the stated "50% of the testset", 8 dates: 2023-06-01 through
2023-06-08):

| Date | MiniLM AUC | Contrastive AUC | MiniLM MRR | Contrastive MRR | MiniLM nDCG@5 | Contrastive nDCG@5 | MiniLM nDCG@10 | Contrastive nDCG@10 |
|---|---|---|---|---|---|---|---|---|
| 2023-06-01 | 0.5374 | 0.5597 | 0.3422 | 0.3588 | 0.3818 | 0.3992 | 0.4597 | 0.4758 |
| 2023-06-02 | 0.5422 | 0.5533 | 0.3404 | 0.3610 | 0.3794 | 0.3989 | 0.4579 | 0.4775 |
| 2023-06-03 | 0.5755 | 0.5504 | 0.3693 | 0.3607 | 0.4102 | 0.3967 | 0.4839 | 0.4744 |
| 2023-06-04 | 0.5444 | 0.5226 | 0.3447 | 0.3384 | 0.3818 | 0.3703 | 0.4607 | 0.4533 |
| 2023-06-05 | 0.5296 | 0.5314 | 0.3374 | 0.3461 | 0.3752 | 0.3817 | 0.4555 | 0.4620 |
| 2023-06-06 | 0.5330 | 0.5489 | 0.3350 | 0.3672 | 0.3713 | 0.3989 | 0.4523 | 0.4771 |
| 2023-06-07 | 0.5316 | 0.5231 | 0.3486 | 0.3507 | 0.3836 | 0.3828 | 0.4637 | 0.4645 |
| 2023-06-08 | 0.5240 | 0.5323 | 0.3442 | 0.3592 | 0.3765 | 0.3903 | 0.4587 | 0.4714 |
| **MEAN** | **0.5397** | **0.5402** | **0.3452** | **0.3553** | **0.3825** | **0.3898** | **0.4615** | **0.4695** |

No confidence intervals are available here — this is Codabench's own
reported per-day aggregate, not a bootstrap this project controls.

## Interpretation

**Contrastive leads on the mean and on every non-AUC metric shown, but the
AUC margin is thin, not the win the local screen predicted.** Mean AUC
+0.0005 (0.5402 vs. 0.5397), with the day-to-day sign flipping three times
(MiniLM ahead on 06-03, 06-04, 06-07) — nothing like a clean, one-sided
result. This is **far smaller than the ~0.0023 CI-clear gap** (0.5453 vs.
0.5430, 95% CI 0.5435–0.5471) the prior addendum measured on `ebnerd_small`
local validation. MRR/nDCG@5/nDCG@10 means all favor contrastive by a
larger relative margin than AUC does, so the qualitative direction from
local validation does hold up here — but the magnitude does not transfer
1:1, and AUC (the metric the leaderboard Score itself appears to track,
per §3.5's coherence check) is the one that matters most for the "did this
submission actually win" question the engineer asked.

**Plausible explanation, not confirmed:** the local `ebnerd_small` edge
was itself small (+0.0023) relative to its own CI width, measured on a
244,647-impression validation split with different users/articles/time
period than the real `ebnerd_testset` blind split. A margin that thin is
consistent with washing out or narrowing further on a different,
much-larger population — ordinary sampling behavior, not evidence of a
pipeline defect. No bug was found in the submission pipeline itself (see
"What was checked" above); the smaller real-world margin is the honest
result, not an artifact of a mistake.

## Conditions for Revisiting

- If a third method or a tuned variant of the contrastive vector is tried,
  this real-test-set result (not the `ebnerd_small` screen alone) is now
  the correct baseline to beat, being the closer proxy for genuine
  out-of-distribution generalization.
- The per-day breakdown covers only "50% of the testset" per Codabench's
  own footnote — the other 50%'s numbers were never obtained; if Codabench
  exposes them (e.g. after competition close), re-checking the full-set
  mean against this 8-day, half-set mean would be worth doing before
  treating +0.0005 as final.
- Whether Codabench's Score column is exactly AUC, and over which exact
  subset, remains inference (per §3.5's own caveat) — not re-litigated
  here.

## Related

- `submissions/ebnerd_testset_embed/prediction.zip` (888045),
  `prediction_contrastive.zip` (896072, downloaded to
  `~/Downloads/prediction_contrastive.zip` — not yet moved into
  `submissions/`, gitignored either way per Q8)
- `notebooks/ebnerd_contrastive_vector_testset_kaggle_run.py`,
  `ebnerd_contrastive_vector_testset_src_bundle.zip`
- `knowledge/ai-usage-log/2026-08-19_contrastive-vector-testset-submission.md`
  (the run's preparation), `knowledge/ai-usage-log/2026-08-21_contrastive-vector-submission-verification.md`
  (this addendum's own verification session)
- `docs/design_note.md`/`.tex` §3.5 (updated with the same per-day AUC
  table and honest margin comparison) and §6 (stale "remains an open
  decision" bullet updated to reflect this result)
- Prior addendum (2026-08-19) above, whose "Conditions for Revisiting" this
  one directly resolves

---

# Addendum — MIND Entity-Embedding and BM25+Embedding Hybrid Screens (2026-08-21)

**Status:** Does not reverse this ADR's decision. Both are pre-decision
screens run per PROJECT_STATE.md's "before committing to a real MIND
second-submission attempt" objective — a check for a real win before
spending one of the Codabench competition's submission slots (confirmed
this session, via the competition's own API:
`max_submissions_per_day: 10`, `max_submissions_per_person: 999` on
competition 13967 — generous, not the binding constraint here). Baseline
throughout: MINDsmall-dev, deployed MiniLM `EmbeddingScorer`, AUC 0.6340
(95% CI 0.6319-0.6361) — reconfirmed bit-identical this session by rerunning
`run_ranking_eval.py --dataset mind --method embed` before either candidate.

## What was checked

**Candidate A — MIND entity embeddings.** New module
`src/retrieval/entities.py`: `parse_entity_mentions` reads the unified
schema's `entities` column (already populated for MIND by
`src/datasets/mind.py::_combine_mind_entities`, previously unused
downstream); `build_article_entity_vector` confidence-weights and mean-pools
each article's linked Wikidata entities' vectors, pulled from MIND's own
`entity_embedding.vec` (100-dim TransE); `build_entity_index` assembles one
`EmbeddingIndex` row per article (reusing the dataclass unchanged), zero-
filling articles with no resolvable entity. Real article coverage measured
directly (not assumed) before evaluating, per the objective's own ordering:
**86.1%** of MINDsmall-dev's 42,416 articles resolved to a non-zero vector.
Scoring/ranking reuses `EmbeddingScorer`/`build_user_embedding_query`
unchanged — the only new logic is vector construction.
(`scripts/run_entity_embedding_experiment.py`)

**Candidate B — BM25+embedding hybrid.** New script
`scripts/run_hybrid_experiment.py`: per impression, both `BM25Scorer` and
`EmbeddingScorer` raw scores are min-max normalized within that impression
and blended at a flat, untuned 50/50 weight — same design ADR-009's leaky-
feature ablation used (`_minmax`, reused directly, not duplicated). All
three arms (bm25-only, embed-only, hybrid) scored in one pass over the same
impression groups/tie-break/bootstrap seed, for a fair paired comparison.

## Results

| Candidate | Overall AUC | 95% CI | vs. baseline (0.6340, CI 0.6319-0.6361) |
|---|---|---|---|
| A — entity embeddings (MIND-only) | 0.5525 | 0.5503-0.5546 | **CI-clear loss** — no overlap |
| B — BM25+embed hybrid (50/50) | 0.6263 | 0.6242-0.6284 | **CI-clear loss** — no overlap (upper bound 0.6284 < baseline lower bound 0.6319) |
| B — BM25 alone (context) | 0.5692 | 0.5670-0.5714 | CI-clear loss, as already known from this ADR's own BM25-vs-semantic comparison |

Warm/cold slices for both candidates follow the same direction as overall
(no crossover) — see `experiments/candidate_a_entity_embed_mind_small_2026-08-21/`
and `experiments/candidate_b_hybrid_mind_small_2026-08-21/` for the full
per-cohort AUC/MRR/nDCG@5/nDCG@10 tables.

## Interpretation

**Neither candidate is a real win — both are real, CI-clear losses**, not
noise or a wash. For A: 86.1% coverage is high enough that low coverage
alone doesn't explain the gap; a TransE knowledge-graph embedding trained
for graph-structural similarity (entity co-occurrence/relations) is not
optimized for the same "topically similar text" notion MiniLM's
sentence-embedding objective targets, and entity mentions alone discard
everything in an article's text that isn't a linked named entity — a
narrower, structurally weaker signal than title+abstract text, consistent
with the measured result. For B: blending in BM25's much weaker raw signal
(0.5692) at an equal, untuned 50/50 weight *dilutes* the stronger embedding
signal rather than complementing it — the hybrid sits between the two
inputs, closer to embed-only but still measurably below it, meaning the
blend has no floor-raising effect here; an asymmetric, tuned weight favoring
embeddings might behave differently, but that is a different, larger
question this cheap untuned screen wasn't sized to answer (same "test
whether it helps at all first" framing this ADR's leaky-feature-ablation
precedent used).

## Conditions for Revisiting

- Candidate A: only worth another look if a genuinely different entity-use
  strategy is tried (e.g. concatenating entity vectors to MiniLM's rather
  than replacing it, or filtering to high-confidence mentions only) — this
  screen tested entity vectors as a full replacement, not an addition.
- Candidate B: an asymmetric, validation-tuned blend weight (rather than
  untuned 50/50) is the natural next experiment if hybrid retrieval is
  revisited — untested here by design.

## Related

- `src/retrieval/entities.py`, `tests/unit/test_entities.py`
- `scripts/run_entity_embedding_experiment.py`, `scripts/run_hybrid_experiment.py`
- `experiments/candidate_a_entity_embed_mind_small_2026-08-21/`,
  `experiments/candidate_b_hybrid_mind_small_2026-08-21/`
- `knowledge/ai-usage-log/2026-08-21_mind-candidate-improvements-local-validation.md`
- See ADR-005's own 2026-08-21 addendum for Candidate C (recency-weighted
  history), tested separately since it's a query/user-representation
  concern, not a semantic-retrieval-design one.

---

## Addendum — 2026-09-04: FAISS re-tested at MINDlarge scale; rejection upheld on accuracy

This ADR rejected FAISS on a latency argument (brute force measured 0.99 ms/query
at 42,416 docs, so an ANN index was unjustified). ADR-014 re-tested that at
**72,023 docs** and added the measurement this ADR never made — what the
approximation actually costs in retrieval quality.

| Arm | Latency/query | Index build | recall@100 vs exact |
|---|---:|---:|---:|
| brute-force numpy (deployed) | **5.51 ms** | 0s (no index) | 1.000 (ground truth) |
| FAISS IVF (nlist 256, nprobe 8) | **0.74 ms** | 1.34s | **0.6511** |
| FAISS Flat (exact) | **46.88 ms** | 0.08s | 1.000 |

**Rejection upheld, and now on stronger grounds than before.** IVF is 7.45x
faster but agrees with exact search on only **65.1%** of the top-100 — min
recall **0.15**, and only **3.1%** of queries return a perfect top-100. Trading
a third of retrieval quality to save 4.8 ms on a stage that is 10.7% of the
serving path (M3) is not a trade worth making.

**A genuinely counterintuitive second result: FAISS Flat is 8.5x SLOWER than a
plain numpy matvec** (46.88 ms vs 5.51 ms) for identical exact results. At
72,023 x 384 with L2-normalised rows, `index.vectors @ query` is already the
right implementation; FAISS's exact index adds a dependency and loses. This
strengthens rather than merely confirms the original call.

**Reproduces from this ADR:** full retrieval projected **12.25 min** (Ada) against
the recorded ~10.4 min — 18% high, same order, consistent. Embedding matrix
110.6 MB, 384-dim, 72,023 rows.

**Revisit trigger added:** the FAISS rejection is validated at 72,023 docs.
Brute-force cost scales linearly with catalog size, so at ~10x (MINDlarge test,
`ebnerd_large`) this must be re-measured, not assumed. See
`decisions/ADR-014-performance-benchmarking-methodology.md`.
