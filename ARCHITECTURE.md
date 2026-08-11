# System Architecture

**Last Updated:** 2026-08-10
**Status:** Under Construction — Data Pipeline (now including MINDlarge), Feature Store, lexical Retrieval (BM25), semantic Retrieval (embeddings, Phase 4), Evaluation (recall@K + Q4's ranking metrics, both methods), and the MIND Codabench submission-format converter implemented; MIND Part 4 (blind test submission) and design note (Q6) not yet started

---

# System Overview

A reproducible pipeline comparing lexical (BM25) and semantic (embedding-based) retrieval for news recommendation, evaluated on MIND and EB-NeRD under a unified schema and shared temporal-split protocol (ADR-001, ADR-002). Data Pipeline, Feature Store, lexical Retrieval, semantic Retrieval, and Evaluation are all implemented, tested, and benchmarked against each other (ADR-008).

---

# Design Goals

This architecture is intentionally optimized for:

- Reproducibility
- Modularity
- Comparability of retrieval methods
- Extensibility
- Evaluation rigor
- Engineering clarity
- Ease of experimentation

It is intentionally **not** optimized for:

- Production deployment
- Distributed execution
- Online inference
- Ultra-low latency
- Massive-scale serving

These trade-offs are intentional because they align with the goals of this project.

---

# Architectural Principles

Every architectural decision should follow these principles.

1. Simplicity before optimization.
2. Reproducibility before performance.
3. Evidence before intuition.
4. Modular components with explicit interfaces.
5. Benchmark before replacing an existing component.
6. Minimize coupling between subsystems.
7. Prefer reversible decisions early.
8. Optimize only after bottlenecks are measured.
9. Document decisions before they become tribal knowledge.

---

# System Data Flow

```text
Raw Data (MIND / EB-NeRD)
        │
        ▼
 Download
        │
        ▼
 Parse & Clean
        │
        ▼
 Temporal Split
        │
        ▼
 Feature Store
        │
        ▼
 Index Construction
     ├── BM25
     └── ANN
        │
        ▼
 Query Construction
        │
        ▼
 Retrieval
     ├── Lexical
     └── Semantic
        │
        ▼
 Evaluation
        │
        ├── Metrics
        ├── Experiments
        ├── Benchmarks
        ├── Leaderboard Submission
        └── Design Note
```

---

# Components

## Data Pipeline

### Purpose

Parses raw MIND (TSV, zipped) and EB-NeRD (parquet, zipped) bundles into
ADR-002's unified schema, preserving ADR-001's official train/dev/validation
split boundaries without re-partitioning. Never extracts zips to disk.

### Responsibilities

- Presence-check raw bundles under `data/raw/`, fail loudly with instructions if absent (`src/pipeline/download.py::ensure_raw_data`)
- Parse MIND `news.tsv`/`behaviors.tsv` and EB-NeRD `articles.parquet`/`behaviors.parquet`/`history.parquet` into the unified `articles`/`impressions`/`user_history` shape (`src/datasets/mind.py`, `src/datasets/ebnerd.py`)
- Validate every table against ADR-002's schema contract before and after writing (`src/pipeline/validators.py`)
- Write one parquet file per (dataset, bundle, split, table), deterministically sorted

### Dependencies

- pandas, pyarrow (parquet read/write — see ADR-002 amendment note below)

### Primary Artifacts

- `data/processed/{dataset}/{bundle}/{split}/{table}.parquet` (gitignored, rebuilt by `make data`)
- `data/processed/ebnerd/{bundle}/articles.parquet` (shared across EB-NeRD splits — see Known Limitations)

### Public Interface

- `src.pipeline.orchestrator.build_all(raw_dir, processed_dir, include_mind_large=False) -> None`
- `src.datasets.mind.parse_mind_split(zip_path, split) -> dict[str, DataFrame]`
- `src.datasets.mind.parse_mind_test_candidates(zip_path, split) -> dict[str, DataFrame]`
- `src.datasets.ebnerd.parse_ebnerd_articles(zip_path) -> DataFrame`, `parse_ebnerd_split(zip_path, split) -> dict[str, DataFrame]`
- CLI: `make data` → `scripts/build_feature_store.py`

### Consumers

- Feature Store (same artifacts — the pipeline writes directly to the feature-store layout; there is no separate transformation step between them in the current design)

### Current Assumptions

- Internal MIND zip folder name equals the zip's own filename stem (verified across all 5 MIND bundles on disk)
- `user_id`/`article_id` are stable identities across a dataset's own splits (verified: MIND train/dev share 5,943/50,000 users; EB-NeRD train/validation share 1,217/~1,590) — only `impression_id` is split-qualified, since it genuinely restarts per file
- MIND's per-user `history` field is a static snapshot (0 variance across all rows for a user) — re-asserted at build time, not just trusted from ADR-002's original 33,617-user sample

### Known Limitations

- `ebnerd_small`/`ebnerd_large` unverified against this design (built strictly against `ebnerd_demo`'s confirmed schema)
- EB-NeRD's shared `articles.parquet` (one file, not per-split) is a real asymmetry with MIND's per-split `news.tsv` — referential-integrity checks need dataset-aware logic, not one uniform check
- MINDlarge is a slow tier, excluded from `make data`'s default scope (`include_mind_large=True` required)

### Related Decisions

- ADR-001
- ADR-002 (amendment: `pyarrow` constraint corrected from `^12.0.0` to `^22.0.0` — the original pin predates any Python 3.14 wheel and had been silently dropped from pyproject.toml, which would have broken EB-NeRD parquet parsing entirely; caught and fixed during Phase 2 implementation)

---

## Feature Store

### Purpose

The on-disk parquet layout produced by the Data Pipeline, conforming to ADR-002's `articles`/`impressions`/`user_history` schema. Currently the pipeline writes this layout directly — there is no separate feature-store build step distinct from parsing.

### Responsibilities

- Persist parsed tables in a layout that keeps train/dev/validation/test physically separate (never a single file with a `split` column), so a leakage bug can't merge them silently
- Provide a schema every downstream retrieval/evaluation component can depend on without dataset-specific branching

### Dependencies

- Data Pipeline

### Primary Artifacts

- `data/processed/mind/{small,large}/{train,dev,test}/{articles,impressions,user_history,candidates}.parquet`
- `data/processed/ebnerd/{bundle}/articles.parquet`, `data/processed/ebnerd/{bundle}/{train,validation}/{impressions,user_history}.parquet`

### Public Interface

- Direct `pd.read_parquet()` against the layout above
- `src.pipeline.schema.ARTICLES_SCHEMA` / `IMPRESSIONS_SCHEMA` / `USER_HISTORY_SCHEMA` / `CANDIDATES_SCHEMA` as the schema contract

### Consumers

- Retrieval (BM25/semantic indexing over `articles`)
- Evaluation (labels from `impressions.clicked`)

### Current Assumptions

- Mandatory core fields (title, abstract, category, impression timestamp, clicked label) are sufficient for Phase 2–3 retrieval — verified for BM25/semantic candidate scoring, not yet verified against Q4's diversity/coverage metrics (per ADR-002)

### Known Limitations

- No feature APIs beyond raw parquet reads yet (e.g. no recency-weighted history helper) — deferred to the retrieval phase, where the first consumer determines the actual interface needed

### Related Decisions

- ADR-002

---

## Retrieval

### Purpose

Given a user's click history, retrieve the top-K candidate articles a
downstream ranker would consider. Two legs, both implemented: lexical
(BM25, Phase 3) and semantic (embeddings, Phase 4) — sharing the
`Scorer` Protocol, `src/evaluation/metrics.py`'s recall@K, and the Q4
ranking harness, per ADR-007's design intent.

### Responsibilities

- Lexical retrieval (BM25, implemented — ADR-005/ADR-006)
- Semantic retrieval (embeddings, implemented — ADR-008)
- Candidate generation: build a per-user query from history — token bag
  for BM25 (ADR-005), mean-pooled L2-normalized vector for embeddings
  (ADR-008) — score it against the corresponding index, return the top-K
  article IDs

### Dependencies

- Feature Store (`articles`, `user_history` tables)
- `rank_bm25` (assignment-mandated library; used for model fitting only —
  see ADR-006 for why scoring itself doesn't call its `get_scores()`)
- `scipy.sparse`, `numpy` (vectorized BM25 scoring)
- `sentence-transformers`, `torch` (embedding computation — ADR-008;
  `faiss-cpu` remains a present-but-unused dependency, ANN backend is
  brute-force cosine similarity, see ADR-008)

### Primary Artifacts

- In-memory `BM25Index` (`src/retrieval/index.py`) — not persisted to disk;
  rebuilt per experiment run (index build is ~1s even at MINDsmall-dev
  scale, so persistence wasn't worth the complexity in this phase)
- Disk-cached `EmbeddingIndex` vectors (`src/retrieval/embed.py`) under
  `data/processed/{dataset}/{bundle}/embeddings/{model_slug}.npy` + a JSON
  sidecar (model name, article ID order) for cache invalidation — persisted
  because encoding is real, repeated cost (minutes, not BM25's ~1s), unlike
  BM25's rebuild-every-run precedent (ADR-008 explains the deviation)
- Candidate rankings (top-K article IDs per user, computed in-memory during
  an experiment run, not persisted independently of `experiments/*/results.json`)

### Public Interface

- `src.retrieval.tokenize.tokenize(text: str) -> list[str]`
- `src.retrieval.index.build_index(articles: pd.DataFrame) -> BM25Index`
- `src.retrieval.query.build_user_query(article_ids: list[str], text_lookup: dict[str, str]) -> list[str]`
- `src.retrieval.retrieve.retrieve_top_k(index: BM25Index, query_tokens: list[str], k: int) -> list[str]`
- `src.retrieval.embed.build_embedding_index(articles, model_name, cache_path, batch_size, device) -> EmbeddingIndex`
- `src.retrieval.embed.build_user_embedding_query(article_ids: list[str], vector_lookup: dict[str, np.ndarray]) -> np.ndarray | None`
- `src.retrieval.retrieve.embed_retrieve_top_k(index: EmbeddingIndex, query_vec: np.ndarray | None, k: int) -> list[str]`
- `src.retrieval.score.{BM25Scorer, EmbeddingScorer}` — both implement the `Scorer` Protocol
- `src.evaluation.metrics.recall_at_k(hits: pd.DataFrame, n_bootstrap: int, seed: int) -> RecallResult`
- CLI: `poetry run python scripts/run_bm25_experiment.py --dataset {mind,ebnerd} [--bundle ...]`
- CLI: `poetry run python scripts/run_embed_experiment.py --dataset {mind,ebnerd} [--bundle ...]`

### Consumers

- Evaluation (`src.evaluation.metrics.recall_at_k`; `scripts/run_ranking_eval.py --method {bm25,embed}`)
- `experiments/{bm25,embed}_{dataset}_{date}/` (config.json + results.json)

### Current Assumptions

- Query construction doesn't use MIND's history-list order, for either
  method (ADR-005, ADR-008; inherited from ADR-002's unverified-ordering
  risk)
- Each split's own article catalog (not a fixed train-time catalog) is the
  correct index universe at query time (ADR-005, reused unchanged by ADR-008)
- BM25Okapi (not BM25L/BM25+) is appropriate because title+abstract is
  short, schema-bounded text for both datasets — revisit if a future phase
  indexes EB-NeRD's `body` field (ADR-006)
- `paraphrase-multilingual-MiniLM-L12-v2` embeddings, computed from
  `title + abstract` only, are sufficient for both languages — benchmark-
  verified (throughput + a category-based discrimination check), not
  assumed (ADR-008)
- Brute-force cosine similarity is sufficient at these corpus sizes
  (measured 0.99ms/query at 42,416 docs) — revisit if MINDlarge enters
  scope (ADR-008)

### Known Limitations

- The vectorized BM25 scorer (`index.py`'s sparse weight matrix) duplicates
  `rank_bm25.BM25Okapi`'s formula rather than calling its `get_scores()`
  directly — verified exact-match on real data, but a future `rank_bm25`
  version bump could silently desync the two (ADR-006)
- ~~Neither BM25 nor the embedding pipeline is yet benchmarked at MINDlarge
  scale~~ **Resolved (2026-08-11): both benchmarked directly against real
  MINDlarge-dev data (72,023 articles, 255,990 users) — BM25's sparse
  matrix is 23.5MB, full retrieval ~9.8 min; embeddings' full encode+cache
  is 264s, full retrieval ~10.4 min. Both stay local, no Kaggle migration
  needed — see ADR-006/ADR-008 addenda.** Getting the MINDlarge *data
  pipeline itself* (not this scoring layer) to build at all required three
  real algorithmic fixes to `src/datasets/mind.py`/`validators.py` — see
  this file's Architecture Changelog and PROJECT_STATE.md's 2026-08-11
  session notes.
- EB-NeRD's cold-start cohort is empty by construction in both the `demo`
  and `small` bundles (min history length = 5) — the warm/cold comparison
  this phase produces is only meaningful for MIND, for either retrieval
  method (ADR-005)
- Absolute recall@200 is modest for both methods (MIND: BM25 2.62%/embed
  2.78%; EB-NeRD: BM25 4.02%/embed 2.57% on demo) — expected for baseline
  retrieval methods per this project's own Day-1 framing, not evidence of a
  bug (verified via random-baseline comparison, ADR-006/ADR-008)
- Neither method has a zero-history-user fallback (e.g. popularity
  backoff) — both structurally cannot retrieve for true cold-start users;
  reported as a shared finding (ADR-005, ADR-008), not built around, since
  no requirement called for one this session

### Related Decisions

- ADR-005 (Query Construction)
- ADR-006 (BM25 Variant)
- ADR-008 (Semantic Retrieval Design)

---

## Evaluation

### Purpose

Two structurally different questions, both answered against the same
Feature Store: (1) recall@K — is the true clicked article anywhere in a
top-K retrieved from the WHOLE corpus (Q2/Q3's lexical/semantic candidate
generation); (2) Q4's ranking metrics — given the candidates ALREADY
LISTED in a real impression, score and rank them, then evaluate that
ranking against the click labels (AUC, MRR, nDCG@5, nDCG@10,
diversity/novelty/coverage). Both share one bootstrap-CI substrate and one
warm/cold cohort definition, so results are comparable across metrics, not
just internally consistent within each.

### Responsibilities

- Recall@K with bootstrap 95% CI, resampling per user
  (`src/evaluation/metrics.py`)
- Q4 per-impression ranking metrics — AUC, MRR, nDCG@5, nDCG@10,
  intra-list diversity, novelty (train-split-only popularity, per Q9's
  anti-gaming requirement), catalog coverage — with warm/cold slicing and
  bootstrap 95% CI (`src/evaluation/ranking_metrics.py`)
- Shared per-user bootstrap resampling substrate used by both of the above
  (`src/evaluation/bootstrap.py`) — extracted so CI computation isn't
  duplicated per metric family (ADR-007)
- Deterministic tie-breaking for ranking a fixed candidate list (needed
  because a cold-start user's empty query scores every candidate
  identically — a top-K-of-everything retrieval never needs to resolve
  this, but a full ranking over a small candidate list does) — ADR-007

### Dependencies

- Retrieval (`src/retrieval/score.py`'s `Scorer` Protocol / `BM25Scorer` —
  Evaluation never constructs scores itself, only consumes them)
- Feature Store (`impressions.clicked` for labels, `articles.category` for
  diversity, train-split `impressions` for novelty's popularity signal)
- scikit-learn (`roc_auc_score`), numpy, scipy

### Primary Artifacts

- `experiments/bm25_{dataset}_{date}/{config,results}.json` — recall@K
- `experiments/ranking_{method}_{dataset}_{date}/{config,results}.json` —
  Q4 ranking metrics (`method` is currently always `bm25`; Phase 4 adds a
  second value without changing this artifact shape)

### Public Interface

- `src.evaluation.metrics.recall_at_k(hits, n_bootstrap, seed) -> RecallResult`
- `src.evaluation.bootstrap.bootstrap_ci(n_units, stat_fn, n_bootstrap, seed) -> (ci_low, ci_high)`
- `src.evaluation.bootstrap.bootstrap_ratio_ci(user_sum, user_count, n_bootstrap, seed) -> (ci_low, ci_high)`
- `src.evaluation.ranking_metrics.rank_candidates(scores, impression_id, seed) -> np.ndarray` (tie-broken rank order)
- `src.evaluation.ranking_metrics.{ndcg_at_k, mrr, safe_auc, intra_list_diversity, novelty}` — per-impression metric functions
- `src.evaluation.ranking_metrics.build_train_popularity(train_impressions, n_catalog, alpha) -> dict[str, float]`
- `src.evaluation.ranking_metrics.coverage(top_k_by_impression, catalog_size) -> float`
- `src.evaluation.ranking_metrics.ranking_metric_ci(per_impression, value_col, n_bootstrap, seed) -> RankingMetricResult`
- `src.retrieval.score.Scorer` (Protocol) / `src.retrieval.score.score_all` / `src.retrieval.score.{BM25Scorer, EmbeddingScorer}` — the generic scoring seam Evaluation's ranking harness consumes; BM25 is the first implementation, `EmbeddingScorer` (ADR-008) the second, plugged in without changing the harness itself
- CLI: `poetry run python scripts/run_bm25_experiment.py --dataset {mind,ebnerd} [--bundle {small,demo}]`
- CLI: `poetry run python scripts/run_ranking_eval.py --dataset {mind,ebnerd} [--bundle ...] [--method {bm25,embed}]`

### Consumers

- `experiments/` (benchmark artifacts feeding the design note, Q6)

### Current Assumptions

- A cold-start user's scorer output is uniformly identical across every
  candidate — verified true for BM25 (an empty query scores everything 0)
  **and now for embeddings** (`EmbeddingScorer` returns an explicit
  all-zero vector for a `None` query, by construction — ADR-008 chose this
  specifically so ADR-007's tie-break stays the single mechanism resolving
  cold-start ties for either scorer). Resolves ADR-007's own flagged
  Research Trigger.
- `category` is 100% non-null for any dataset this harness runs against
  (verified for MIND, `ebnerd_demo`, `ebnerd_small`; `intra_list_diversity`
  has no null-handling branch)
- Novelty popularity must come from the train split only, never the
  evaluation split (Q9 anti-gaming) — structural in
  `build_train_popularity`'s signature, not a convention callers must
  remember

### Known Limitations

- Coverage has no bootstrap CI, by design (ADR-007): it's a set-union
  statistic, and with-replacement bootstrap resampling of a union is
  mechanically biased low, not just noisy — reported as a single point
  estimate instead of a misleading interval
- K=10 for diversity/novelty is a judgment call (anchored to the
  already-required nDCG@10 cutoff, not independently justified) — absolute
  diversity/novelty values are K-dependent
- EB-NeRD's cold cohort is empty by construction at both the `demo` and
  `small` tiers (ADR-002 addendum) — the warm/cold ranking-metrics
  comparison is, like recall@K's, only meaningful for MIND with this
  project's current data

### Related Decisions

- ADR-005 (Query Construction) — warm/cold cohort threshold, reused
  identically by both recall@K and the ranking harness
- ADR-006 (BM25 Variant) — recall@K's scoring method
- ADR-007 (Q4 Ranking Evaluation Harness Design) — tie-breaking,
  diversity/novelty K, novelty/coverage definitions, coverage-CI omission
- ADR-008 (Semantic Retrieval Design) — second `Scorer` consumer;
  resolves ADR-007's cold-start Research Trigger

---

## Submission

### Purpose

Converts a `Scorer`'s per-impression output into the official MIND
Codabench prediction format (`impression_id [rank_1,...,rank_N]`, ranks not
scores, candidates in the impression's original order) — the format
`evaluation/official/evaluate.py` (the real script Codabench runs) actually
parses, not the CSV format the assignment README originally guessed.

### Responsibilities

- Re-read the raw MIND zip directly to recover each impression's original
  candidate order — `_write_table`'s deterministic sort (alphabetical by
  `article_id`, per ADR-002) destroys this order once it's in the
  processed feature store, so the raw zip is the only place it survives
- Invert `src.evaluation.ranking_metrics.rank_candidates`'s score-order
  permutation into a per-original-position rank list (the quantity the
  official format wants)
- Build a local ground-truth file (`impression_id [label_1,...]`) from a
  labeled split's own real click labels, for dev-set validation only —
  Codabench's actual private test-set ground truth is never available to
  this project

### Dependencies

- Retrieval (`src.retrieval.score.Scorer` — never constructs scores
  itself, only consumes them, identical posture to Evaluation)
- Evaluation (`src.evaluation.ranking_metrics.rank_candidates` — reused
  unchanged, same tie-break rule ADR-007 established)

### Primary Artifacts

- `submissions/mind_large_{split}_{method}/{prediction,truth}.txt`

### Public Interface

- `src.submission.mind_format.read_raw_impressions(zip_path, has_labels) -> list[dict]`
- `src.submission.mind_format.ranks_for_impression(scores, impression_id, seed) -> list[int]`
- `src.submission.mind_format.write_predictions(out_path, zip_path, scorer, query_by_user, empty_query, has_labels, seed) -> int`
- `src.submission.mind_format.write_truth_file(out_path, zip_path) -> int`
- CLI: `poetry run python scripts/generate_mind_predictions.py --split {train,dev,test} --method {bm25,embed}`

### Consumers

- `evaluation/official/evaluate.py` (external, the real Codabench scorer) — validated locally against it for both methods on MINDlarge_dev (2026-08-11): AUC/nDCG match the project's own `ranking_metrics.py` almost exactly; MRR disagrees by a real, fully-explained margin (official sums 1/rank over every clicked item rather than crediting only the first hit — verified directly, not assumed, against real prediction/truth data with 28.72% multi-click impressions)

### Current Assumptions

- The raw zip's row order is the order Codabench's own ground truth file
  follows — not independently verifiable (Codabench's private ground
  truth isn't available), but it's the dataset's own native order and the
  only order this project can construct predictions in
- A labeled split's own real labels are an adequate stand-in for
  Codabench's private ground truth for *validating the converter itself*
  (format correctness, rank inversion, ordering) — not a claim about
  matching the actual leaderboard score, which depends on the blind test
  set

### Known Limitations

- Only validated end-to-end against MINDlarge_dev (labeled) — MINDlarge_test
  (blind, Part 4) has no local ground truth to validate against by
  construction; Part 3's dev-set validation is what stands in for it
- `write_predictions`/`write_truth_file` re-read the raw zip's
  `behaviors.tsv` fully into memory each call — fine at MINDlarge scale
  (measured), would need revisiting if a future dataset's raw behaviors
  file were materially larger

### Related Decisions

- ADR-002 (Unified Data Schema) — the deterministic-sort requirement this
  module works around
- ADR-005 (Query Construction) — cold-start empty-query contract, reused
  via each method's existing `query_by_user`/`empty_query` construction
- ADR-007 (Q4 Ranking Evaluation Harness Design) — tie-break rule reused
  unchanged via `rank_candidates`

---

# Decision Traceability

| Component | Governing ADRs |
|------------|----------------|
| Data Pipeline | ADR-001 (Dataset Processing) |
| Feature Store | ADR-002 (Feature Representation) |
| Retrieval | ADR-005 (Query Construction), ADR-006 (BM25 Variant, + MINDlarge-scale addendum), ADR-008 (Semantic Retrieval Design, + MINDlarge-scale addendum) |
| Evaluation | ADR-007 (Q4 Ranking Evaluation Harness Design), ADR-008 (second Scorer consumer) |
| Submission | ADR-002 (works around the deterministic-sort requirement), ADR-005/ADR-007 (reuses cold-start contract and tie-break rule unchanged) |

---

# Component Interfaces

## Data Pipeline → Feature Store

| Aspect | Description |
|---------|-------------|
| Data Format | |
| Contract | |
| Assumptions | |

---

## Feature Store → Retrieval

| Aspect | Description |
|---------|-------------|
| Data Format | Direct `pd.read_parquet()` against the Feature Store's on-disk layout — no intermediate loader abstraction was needed (per that section's own note: "determined by Retrieval's first real consumer") |
| Contract | Retrieval depends only on ADR-002's mandatory fields (`title`, `abstract` from `articles`; `article_ids` from `user_history`; `user_id`, `article_id`, `clicked` from `impressions`) — never an optional, dataset-only field, preserving the single-code-path goal |
| Assumptions | Each split's own `articles.parquet` is the correct index universe at query time (MIND: per-split file; EB-NeRD: shared bundle-level file — see ADR-005) |

---

## Retrieval → Evaluation

| Aspect | Description |
|---------|-------------|
| Data Format | Two shapes, one per evaluation question. recall@K: `top_k_by_user: dict[str, list[str]]` (article IDs, computed by `retrieve_top_k` against the whole corpus) joined against `impressions.clicked` to produce a `hits` DataFrame (`user_id`, `hit: bool`). Ranking metrics: a `Scorer.score(query, candidate_ids) -> np.ndarray` call per impression (candidates already fixed by the `impressions` table's own rows for that `impression_id`), never a corpus-wide retrieval |
| Contract | Evaluation never constructs scores itself — it only calls `retrieve_top_k`/`embed_retrieve_top_k` (recall@K) or a `Scorer` (ranking metrics) and consumes the output. The `Scorer` Protocol (`src/retrieval/score.py`) is the seam: `BM25Scorer` and `EmbeddingScorer` (ADR-008) both implement it; `scripts/run_ranking_eval.py`'s `_build_method` is the only place that dispatches by method name, the scoring loop itself does not |
| Assumptions | A cold-start user's query (empty history) produces a score vector Evaluation can still consume without erroring — `retrieve_top_k` returns `[]`; `score_all`/`BM25Scorer` return an all-zero vector, which `rank_candidates`'s tie-break (ADR-007) then resolves into a valid, reproducible ranking |

---

# Cross-Cutting Concerns

These concerns affect every component.

## Configuration

- Centralized configuration
- Environment-independent execution

## Logging

- Structured logging
- Execution traceability

## Reproducibility

- Random seed management
- Dataset versioning
- Configuration versioning

## Experiment Tracking

- Saved configurations
- Metrics
- Artifacts
- Benchmark history

## Testing

- Unit tests
- Integration tests
- Reproducibility tests

## Benchmarking

- Baseline comparison
- Statistical significance
- Slice evaluation

## Error Handling

- Explicit failures
- Helpful diagnostics
- Graceful recovery

## Documentation

- Architecture
- ADRs
- Project State
- Knowledge Base

---

# Architectural Assumptions

| Assumption | Rationale | Validation | Risk |
|------------|-----------|-----------|------|
| | | | |

---

# Scalability & Known Limitations

## Current Design Limits

Performance

- ...

Memory

- ...

Storage

- ...

Accuracy

- ...

---

## Expected Bottlenecks at 10×

- ...
- ...

---

## Migration Strategy

Monitor

- ...

Trigger

- ...

Potential Replacement

- ...

Related ADR

- ADR-XXX

---

# Architecture Changelog

## 2026-08-11 — MINDlarge built + benchmarked; Q5 submission-format converter implemented

### What Changed

- Fixed three real, MINDlarge-scale-only bugs in `src/datasets/mind.py`
  (`_explode_impressions`) and `src/pipeline/validators.py` (`_is_null`) —
  a naive per-token Python loop that hung indefinitely, a categorical-dtype
  fix for a ~27GB memory projection, and two separate `map_infer_mask`
  performance bugs (pandas `.map()`/`.str.rsplit()` being elementwise
  Python loops at 81M-row scale instead of the ~2.2M rows actually needed)
  — all found by profiling the actual stuck process, not by inspection.
  Full details in PROJECT_STATE.md's 2026-08-11 session notes.
- Built and row-count-verified MINDlarge (train/dev/test) via
  `include_mind_large=True` for the first time — new `MIND_LARGE_EVIDENCE`/
  `MIND_LARGE_TOTAL_ARTICLES`/`MIND_LARGE_TOTAL_USERS` in
  `src/utils/config.py`, new `@pytest.mark.slow` tests in
  `tests/integration/test_schema_conformance.py`.
- Added `src/submission/mind_format.py` (new component — see
  ARCHITECTURE.md's Submission section) and
  `scripts/generate_mind_predictions.py`; 7 new unit tests
  (`tests/unit/test_mind_format.py`).
- Benchmarked BM25 and embeddings directly against real MINDlarge-dev data
  (ADR-006/ADR-008 addenda) — both stay local, no Kaggle migration needed.
- Fixed `tests/integration/test_pipeline_end_to_end.py::test_default_build_does_not_touch_mindlarge`,
  which was checking the shared real `processed_dir` fixture (now
  legitimately containing `mind/large`) — moved to an isolated `tmp_path`
  build so it checks `build_all()`'s own default-argument behavior.

### Why

Implements Part 1-3 of Q5 (MIND Codabench submission): build the real-scale
corpus, benchmark before trusting anything runs at that scale (CLAUDE.md's
benchmarking philosophy), and validate the submission-format converter
against the real official scoring script before ever generating a blind
test submission.

### Impact

- MINDlarge is no longer an unverified, never-exercised code path.
- The converter's correctness is verified against real ground truth, not
  just fixtures: AUC/nDCG match `ranking_metrics.py`'s own numbers almost
  exactly on MINDlarge-dev for both methods; the one real MRR disagreement
  is a fully-explained metric-definition difference (verified with
  certainty by recomputing both formulas directly), not a bug.
- Embeddings win clearly on MINDlarge-dev (AUC 0.6335 vs. BM25's 0.5699) —
  consistent with ADR-008's original MINDsmall finding.

### Related ADR

- ADR-006 (addendum), ADR-008 (addendum)

### Benchmark

- See ADR-006/ADR-008's addenda for the full MINDlarge-scale numbers;
  PROJECT_STATE.md's 2026-08-11 session notes for the data-pipeline fixes'
  own before/after measurements.

---

## 2026-08-10 — Phase 4: Semantic Retrieval implemented

### What Changed

- Added `src/retrieval/embed.py` (`EmbeddingIndex`, disk-cached
  `build_embedding_index`, `build_user_embedding_query`), `EmbeddingScorer`
  in `src/retrieval/score.py`, `embed_retrieve_top_k` in
  `src/retrieval/retrieve.py` (factored a shared `_top_k_from_scores`
  helper out of `retrieve_top_k` for reuse)
- Added `scripts/run_embed_experiment.py`; generalized
  `scripts/run_ranking_eval.py`'s previously-BM25-hardcoded dispatch into
  `_build_method`, adding `--method embed`
- Added `sentence-transformers` (`^5.7.0`) to `pyproject.toml`, verified via
  `poetry lock && poetry install`
- Added `decisions/ADR-008-semantic-retrieval-design.md`
- Added `tests/unit/test_embed.py`; extended `test_score.py`,
  `test_retrieval.py`, `tests/integration/test_retrieval_pipeline.py`
- Ran recall@K and the Q4 ranking harness for embeddings against the same
  three corpora BM25 already covers (MINDsmall-dev, ebnerd_demo-validation,
  ebnerd_small-validation); backfilled a missing
  `ranking_bm25_ebnerd_2026-08-10` (demo bundle) run for full parity

### Why

Implements Q3 (semantic candidate generation) and completes Q3.5/Q4.5 (the
BM25-vs-semantic comparison), the second `Scorer` consumer ADR-007's design
was built to accommodate without requiring changes to the harness itself.

### Impact

- Encoder chosen (`paraphrase-multilingual-MiniLM-L12-v2` over
  `multilingual-e5-small`) by a real discrimination-gap benchmark, not
  assumed from the assignment's example model list — see ADR-008
- ANN backend confirmed brute-force-sufficient (0.99ms/query at 42,416
  docs) — FAISS stays an unused, documented dependency
- BM25-vs-semantic comparison produced a genuine, non-obvious finding: the
  Day-1 hypothesis ("BM25 favors warm/entity-heavy, embeddings favor
  cold-start") does not hold on MIND (embeddings win on both cohorts), and
  BM25 vs. embeddings disagree by *evaluation question* on EB-NeRD
  (BM25 wins recall@K, embeddings edge out BM25 on Q4 ranking) — see
  ADR-008's Interpretation and PROJECT_STATE.md's Benchmarking Status
- `run_ranking_eval.py`'s BM25 path refactored onto the new `_build_method`
  dispatch; re-verified byte-identical against ADR-007's previously
  recorded BM25 numbers — behavior-preserving, not just assumed so
- Resolves ADR-007's own flagged Research Trigger about a non-BM25 scorer's
  cold-start behavior: `EmbeddingScorer` deliberately produces the same
  all-zero total tie BM25 does, keeping ADR-007's tie-break the single
  mechanism for either scorer
- Full test suite: 145 passed, 1 skipped (MIND leakage, expected), 1
  deselected (`slow` MINDlarge test, expected) — no regressions

### Related ADR

- ADR-008

### Benchmark

- See ADR-008's Benchmark Results for encoder selection, ANN-backend
  timing, and the full recall@K/Q4-ranking BM25-vs-semantic tables

---

## 2026-08-10 — ebnerd_small Verification + Q4 Ranking Evaluation Harness implemented

### What Changed

- Downloaded, built, and schema-verified `ebnerd_small` (20,738 articles;
  15,143/15,342 train/validation users; 2.59M/2.93M impressions) —
  resolves the `ebnerd_small` open question ADR-001/ADR-002/ADR-005 all
  flagged; documented as an ADR-002 addendum
- Added `--bundle` to `scripts/run_bm25_experiment.py`; re-ran the BM25
  benchmark against `ebnerd_small` (`experiments/bm25_ebnerd_small_2026-08-10/`)
- Extracted `src/retrieval/score.py` (`score_all`, `Scorer` Protocol,
  `BM25Scorer`) from logic previously inlined in `retrieve_top_k`; added
  `id_to_col` to `BM25Index` — the generic scoring seam Q4's harness (and
  Phase 4's future embedding scorer) consumes
- Extracted `src/evaluation/bootstrap.py` from `recall_at_k`'s previously-
  inlined bootstrap logic (pinned behavior-identical by a new test written
  before the refactor)
- Added `src/evaluation/ranking_metrics.py` (AUC, MRR, nDCG@5, nDCG@10,
  intra-list diversity, novelty, coverage, deterministic tie-breaking) and
  `scripts/run_ranking_eval.py`, producing
  `experiments/ranking_bm25_{dataset}_{date}/{config,results}.json`
- Added `decisions/ADR-007-ranking-evaluation-design.md` (tie-break rule,
  diversity/novelty K=10, train-only novelty popularity, coverage-CI
  omission) and an ADR-002 addendum
- Added `tests/unit/{test_score,test_bootstrap,test_ranking_metrics,test_metrics}.py`
  and `tests/integration/test_ranking_eval_pipeline.py`; extended
  `tests/integration/test_schema_conformance.py` for `ebnerd/small`

### Why

Resolves this session's two objectives: verifying `ebnerd_small` against
the schema/loaders built and tested only against `ebnerd_demo` so far
(ADR-002's own flagged open question), and building Q4's ranking-metrics
harness — a structurally different question from recall@K (rank the
candidates already listed in an impression, rather than retrieve top-K
from the whole corpus) that the assignment requires alongside it.

### Impact

- `ebnerd_small`'s cold cohort is empty by construction (min history = 5),
  same as `ebnerd_demo` — confirms this is a property of EB-NeRD's
  active-user-filtered bundle construction, not a demo-only artifact
  (ADR-002 addendum)
- Full Q4 benchmark on both datasets
  (`experiments/ranking_bm25_mind_2026-08-10/`,
  `experiments/ranking_bm25_ebnerd_small_2026-08-10/`): MIND AUC 0.5692
  overall (warm 0.5766 / cold 0.5242 — confirms the warm>cold pattern
  recall@K already found); EB-NeRD-small AUC 0.5288. nDCG@5/@10 show the
  opposite dataset ordering from AUC (EB-NeRD higher) — traced to
  candidate-list-length differences (MIND median 23 vs. EB-NeRD median
  9–12), a genuine finding about why Q4 mandates multiple metrics rather
  than any single one (see ADR-007's Interpretation)
- `retrieve_top_k`'s refactor onto the extracted `score_all` is behavior-
  preserving (existing `tests/unit/test_retrieval.py` passes unchanged);
  `recall_at_k`'s bootstrap refactor is pinned byte-identical by a new
  test written before the refactor

### Related ADR

- ADR-002 (addendum), ADR-005 (addendum to Conditions for Revisiting),
  ADR-007

### Benchmark

- See ADR-002's addendum for the `ebnerd_small` recall@K comparison
  against `ebnerd_demo`; ADR-007's Benchmark Results for the full Q4
  ranking-metrics table on both datasets

---

## 2026-08-10 — Phase 3: BM25 Lexical Retrieval implemented

### What Changed

- Implemented `src/retrieval/{tokenize,index,query,retrieve}.py` and `src/evaluation/metrics.py`
- Added `scripts/run_bm25_experiment.py`, producing `experiments/bm25_{dataset}_{date}/{config,results}.json`
- Added `decisions/ADR-005-query-construction.md` and `decisions/ADR-006-bm25-variant.md`
- Added unit tests (`tests/unit/test_query_construction.py`, `tests/unit/test_retrieval.py`) and an integration test (`tests/integration/test_retrieval_pipeline.py`)

### Why

Implements Phase 3's BM25 baseline per ADR-005/ADR-006, turning the assignment's query-construction and BM25-variant open questions (PROJECT_STATE) into a working, benchmarked implementation.

### Impact

- Two real defects were found and fixed only by running the real pipeline at real scale, not by code review or unit tests alone (both documented in ADR-006 and ADR-005 respectively, not silently folded into the implementation as if planned from the start):
  - `rank_bm25.get_scores()`'s per-query-token Python loop was measured infeasible at MINDsmall-dev scale (200 users: projected 10+ hours); replaced with a sparse-matrix scorer reproducing the identical formula (verified exact-match against `rank_bm25` on real data), cutting the full 50,000-user MIND-dev run to ~80 seconds.
  - EB-NeRD's long per-user histories (up to 1,459 articles), concatenated unweighted, produced queries whose term-count mass was dominated by high-frequency Danish/English function words — measured to push recall@50/100 *below* the random-retrieval baseline before stopword removal was added to the tokenizer.
- Full benchmark results: MIND-small dev recall@200 = 2.62% (warm 2.73%, cold 1.78%); EB-NeRD validation recall@200 = 4.02% (cold cohort empty by construction — EB-NeRD demo's active-user filter guarantees history length >= 5). Both clear their respective random baselines by a real margin (5.6x / 2.4x), confirming genuine signal.

### Related ADR

- ADR-005, ADR-006

### Benchmark

- See ADR-006's Benchmark Results for the full recall@K table and the performance/correctness verification of the vectorized BM25 scorer.

---

## 2026-08-09 — Phase 2: Data Pipeline / Feature Store implemented

### What Changed

- Implemented `src/datasets/{mind,ebnerd}.py`, `src/pipeline/{schema,validators,download,orchestrator}.py`, `src/utils/{config,ids,io}.py`
- `make data` now builds the full feature store end-to-end from raw MINDsmall + ebnerd_demo zips
- Added `make test-reproducibility` and `make clean-data` (previously referenced by README but missing)
- Added unit (29), integration (48), and reproducibility test suites

### Why

Turns ADR-001/ADR-002 from paper decisions into a working, tested pipeline, per CLAUDE.md's engineering lifecycle.

### Impact

- Row counts verified to match ADR-001's evidence table exactly (MINDsmall train=156,965/50,000 users, dev=73,152/50,000; ebnerd_demo train=24,724/1,590, validation=25,356/1,562)
- Corrected an in-progress, uncommitted `pyproject.toml` change that had silently dropped `pyarrow` entirely (would have broken all EB-NeRD parsing) — see Data Pipeline's Related Decisions note
- Corrected the implementation plan's original ID scheme mid-build: `user_id` must NOT be split-qualified (verified real user overlap across MIND/EB-NeRD splits), only `impression_id` should be

### Related ADR

- ADR-001, ADR-002

### Benchmark

- Not applicable — this phase is pure ETL correctness/reproducibility, no retrieval-performance claims yet

---

# Related Decisions

| ADR | Status | Impact |
|-----|--------|--------|
| ADR-XXX | Decided | |

---

# Future Evolution

## Known Future Work

- Planned architectural improvements
- Components already scheduled

---

## Research Questions

- Questions requiring experiments
- Competing architectural ideas
- Unknown trade-offs

---

## Technical Debt

- Areas intentionally deferred
- Coupling that should eventually be removed
- Temporary implementations

---

# Related Documentation

## Engineering

- PROJECT_STATE.md
- ENGINEERING_WORKFLOW.md

## Decisions

- decisions/

## Knowledge Base

The project maintains an evolving knowledge repository for research, notes, and references.

```text
knowledge/
├── assignment/
│   ├── assignment.md
│   └── grading_notes.md
│
├── retrieval/
│   ├── bm25.md
│   ├── bm25_variants.md
│   ├── semantic_retrieval.md
│   └── faiss.md
│
├── evaluation/
│   ├── metrics.md
│   ├── benchmarking.md
│   └── statistical_testing.md
│
├── research/
│   ├── papers/
│   └── reading_notes/
│
└── industry/
    ├── production_practices.md
    └── case_studies.md
```

This directory acts as the team's internal engineering wiki. It captures research, benchmarking notes, implementation references, paper summaries, and industry practices so knowledge accumulates throughout the project instead of being rediscovered.

---

# Architecture Health Checklist

When modifying this architecture, verify:

- [ ] Interfaces remain explicit.
- [ ] Components remain loosely coupled.
- [ ] Component interfaces remain backward compatible or are appropriately versioned.
- [ ] New decisions are captured in ADRs.
- [ ] Benchmarks exist for performance-sensitive changes.
- [ ] Assumptions remain valid.
- [ ] Changelog is updated.
- [ ] Documentation reflects the current implementation.