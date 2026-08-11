# ADR-002 — Unified Data Schema for MIND and EB-NeRD

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

**Date:** 2026-08-09
**Status:** Decided
**Severity:** High

---

# Engineering Question

What single schema can represent articles, impressions, and user history for both MIND and EB-NeRD, so that the data pipeline, BM25 index, semantic index, and evaluation harness can operate over one code path instead of two — without silently losing information that either dataset actually contains?

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

- Data Pipeline
- Feature Store
- Retrieval (BM25 + semantic both consume the articles table's text fields)
- Evaluation (consumes the impressions table's `clicked` label)

---

# Context

MIND and EB-NeRD are structurally different at the field level (full inspection recorded separately; summarized in Evidence below):

- MIND ships no article-level publish timestamp and no per-click timestamps in user history — only impression-level timestamps exist.
- EB-NeRD ships both, plus full article body text, engagement metadata, and (sparse) demographics.
- Both provide the minimum text needed for BM25/semantic retrieval (title + a short summary field) and both provide *some* form of entity annotation and category label — but in incompatible shapes.

Without a unified schema, the retrieval and evaluation code would need a dataset-specific branch at every layer, doubling the surface area to test and making Q4's cross-dataset BM25-vs-semantic comparison harder to trust (are we comparing retrieval methods, or comparing retrieval methods entangled with two different data-loading code paths?).

---

# Decision Criteria

| Criteria | Importance | Notes |
|----------|------------|-------|
| Single code path for BM25/semantic retrieval | Critical | Directly enables Q4's cross-dataset comparison |
| No silent information loss | Critical | Losing a field silently (vs. documenting the loss) is a correctness risk, not just an engineering-style preference |
| Null-handling clarity | High | Downstream code must be able to tell "field doesn't exist for this dataset" from "field exists but is missing for this row" |
| Engineering complexity | Medium | More tables/joins = more surface area |
| Schema bloat | Low-Medium | Acceptable cost if it buys correctness and a single code path |

---

# Design Space Exploration

## Option A — Separate pipelines per dataset

### How it Works

Build independent loaders, feature stores, and retrieval code for MIND and EB-NeRD, each shaped natively around its own schema.

### Optimizes For

Zero normalization work; every field stays in its native shape.

### Sacrifices

A single BM25/semantic retrieval implementation cannot run against both datasets without dataset-specific branching baked into the retrieval and evaluation code itself — the exact thing Q4's cross-dataset comparison depends on being controlled for. Doubles the code (and test) surface.

### Assumptions

That retrieval/evaluation logic can tolerate being duplicated or heavily branched without correctness drift between the two copies.

### Complexity

Low per-dataset, high in aggregate (two of everything).

### Maturity

★★★☆☆ — common in ad hoc research code, less common in anything meant to be compared rigorously.

---

## Option B — Unified schema that preserves all fields from both

### How it Works

One schema per table, but with every field either dataset exposes, unioned together — including MIND-specific and EB-NeRD-specific quirks as first-class columns with no distinction from universal fields.

### Optimizes For

Maximal information preservation, structurally.

### Sacrifices

No signal about *which* fields are structurally guaranteed (present for every row of every dataset) versus dataset-specific (always null for one dataset by construction). Code consuming the schema has to rediscover, empirically, which nulls are "sometimes missing" and which are "always missing for this dataset" — exactly the ambiguity a schema should remove.

### Assumptions

That downstream code will correctly infer field applicability from null patterns rather than from the schema itself.

### Complexity

Medium — same table count as Option C, but without the mandatory/optional signal doing useful work.

### Maturity

★★☆☆☆ — a real anti-pattern in schema design (the "everything nullable, nothing documented" table).

---

## Option C — Unified schema with mandatory core fields + explicitly-marked dataset-specific optional fields

### How it Works

Three tables (`articles`, `impressions`, `user_history`), each with a small set of fields guaranteed present and non-null-by-construction across both datasets, plus explicitly-labeled optional fields that are populated for one dataset and always null for the other. A `dataset` column on every table makes the source explicit and queryable, rather than left to null-pattern inference.

### Optimizes For

A single retrieval/evaluation code path that only depends on mandatory fields; optional fields are available for dataset-aware feature engineering without being load-bearing for the core pipeline.

### Sacrifices

Some columns are always null for one dataset (e.g., `article_published_time` is always null for MIND) — minor storage/schema bloat, but with the applicability documented rather than implicit.

### Assumptions

That the mandatory-field subset (title, abstract/subtitle, category, impression timestamp, click label) is sufficient to implement BM25 and semantic retrieval plus standard evaluation metrics without needing any dataset-specific field. Verified against the schema inspection — true for BM25/semantic candidate scoring; not tested yet against Q4's diversity/coverage metrics, which is why this ADR's confidence is Medium, not High (see below).

### Complexity

Medium. One join key (`article_id`, `user_id`) per table, `dataset` as a discriminator column, explicit optional-field documentation.

### Maturity

★★★★☆ — standard practice for multi-source feature stores (mandatory core schema + source-specific extension columns).

---

# Comparison Summary

| Option | Single Retrieval Code Path | Information Loss | Null-Handling Clarity | Complexity | Decision |
|---------|----------|------------|------------------|-----------|----------|
| A — Separate pipelines | No | None | N/A (no shared schema) | High (2x surface) | |
| B — Unify everything, undifferentiated | Yes | None structurally, but ambiguous in practice | Poor | Medium | |
| C — Mandatory core + marked optional | Yes | Documented, not silent | Good | Medium | ✅ |

---

# Final Decision

## Chosen Option

**Option C** — unified schema across three tables (`articles`, `impressions`, `user_history`), each with a mandatory core and explicitly dataset-marked optional fields.

### Reason

Option A defeats the purpose of comparing retrieval methods across datasets in Q4 by letting the comparison get entangled with two different data-loading implementations. Option B preserves the same information as Option C but throws away the one thing that makes a schema useful: telling a reader (or a future test) which fields they can rely on unconditionally versus which are dataset-contingent. Option C is the only option where "does this pipeline work for both datasets" is answerable by inspecting the schema itself, not by re-deriving it from null-pattern behavior at runtime.

### Decision Date

2026-08-09

### Decision Owner

**Primary Engineer**

- sukhraj01

### Contributors

- Claude Code (schema inspection, ADR drafting)

### Reviewer *(Optional)*

- Pending

---

# Rationale

## Why this option is best

- Enables a single BM25/semantic retrieval code path keyed on `article_id` + text fields (`title`, `abstract`) that never needs to know which dataset it's operating on.
- Preserves all information from both sources by keeping dataset-specific fields as explicit optional columns, rather than dropping them (Option A) or hiding their applicability (Option B).
- Directly enables Q4's cross-dataset comparison, since retrieval and evaluation code depend only on the mandatory core — the comparison measures retrieval-method differences, not data-loading differences.
- `article_published_time` being explicitly optional (present for EB-NeRD, always null for MIND) documents a real, load-bearing constraint rather than a schema afterthought: no recency feature can be computed for MIND from article metadata, full stop. This was the central finding of the schema inspection and needs to be structurally visible, not just remembered.

## Why alternatives were rejected

### Option A (separate pipelines)

Rejected because it undermines the assignment's own Q4 requirement — a rigorous BM25-vs-semantic comparison across both datasets — by letting two different code paths introduce uncontrolled variance into that comparison.

### Option B (unify everything, undifferentiated)

Rejected because it solves the wrong problem. The engineering risk here was never "can we fit all fields into one table" (trivially yes) — it's "can code consuming this schema know what it can depend on without inspecting actual data first." Option B doesn't answer that question; Option C does, for the same storage cost.

---

# Decision Confidence

**Current Confidence**

Medium

### Why

- The mandatory-core fields (title, abstract, category, impression timestamp, clicked label) are verified sufficient for BM25 and semantic candidate scoring — this was directly checked against both datasets' actual schemas.
- Not yet verified against Q4's diversity/novelty/coverage metrics, which may need `category`/`subcategory`/`topics` in ways this schema hasn't been stress-tested for.
- The EB-NeRD `subcategory` field has no string label anywhere in the inspected data — this schema currently carries it as an opaque, dataset-local integer list rather than a resolved value, which is a known gap, not a solved one.
- `ebnerd_small`/`ebnerd_large` haven't been inspected — if their schema differs from `ebnerd_demo`'s (unlikely, but unverified), this schema would need revision.

### What Would Increase Confidence

- Implementing Q4's diversity/coverage metrics against this schema and confirming no additional mandatory field is needed.
- Inspecting `ebnerd_small`/`ebnerd_large` to confirm identical column structure to `ebnerd_demo`.
- Resolving whether an EB-NeRD subcategory string mapping exists anywhere in the release (not found in `articles.parquet` itself).

---

# Evidence

## Evidence Sources

- [x] Assignment Requirements (Q4 requires a cross-dataset BM25-vs-semantic comparison, which motivates a single schema)
- [x] Experimental Benchmark (direct inspection of on-disk MIND + EB-NeRD files)
- [x] Official Documentation (MIND/EB-NeRD paper field descriptions)
- [ ] Research Paper
- [ ] Industry Practice
- [ ] Community Consensus
- [ ] Engineering Inference

---

## Empirical Evidence

**Articles schema comparison** (full detail in schema-inspection notes; corrected here against two errors in the original draft):

| Field | MIND | EB-NeRD | Unified treatment |
|---|---|---|---|
| Text | `title` + `abstract` (5.2% empty) | `title` + `subtitle` (0% null) | mandatory: `title`, `abstract` |
| Body | absent | `body` (0% null) | optional, EB-NeRD only |
| Publish time | absent (no date field anywhere in `news.tsv`'s 8 columns) | `published_time` (2000-10-02 → 2023-06-08) | optional, EB-NeRD only |
| Category | `category`, 17 unique flat strings | `category_str`, 25 unique | mandatory: `category` |
| Subcategory | `subcategory`, 264 unique **flat strings** | `subcategory`, **list[int], no string label found** | present in both, but MIND's is the usable string form — **not** EB-NeRD-exclusive as originally drafted |
| Entities | `title_entities`/`abstract_entities`, Wikidata-linked, **73.0% nonempty** | `ner_clusters`/`entity_groups`, **85.7% nonempty** | present in both, structurally incompatible — kept as raw dataset-specific passthrough, **not** EB-NeRD-exclusive as originally drafted |
| Topics | absent | `topics` (free-text list) | optional, EB-NeRD only |

**Impressions schema:** MIND encodes candidate+label as fused pairs (`ArticleID-0/1`) per impression; EB-NeRD encodes as separate `article_ids_inview` / `article_ids_clicked` lists. Unified schema below explodes both into one row per (impression, candidate) pair — a standard long-format representation that both encodings losslessly reduce to.

**Cold-start, quantified** (motivates why `article_published_time` nullability matters, not just a schema nicety): 13,956/42,416 (32.9%) of MIND dev-window articles never appeared in train; 1,251/2,738 (45.7%) of EB-NeRD validation-window candidate articles never appeared in train.

**User history:** MIND history verified empirically fixed per user (checked 33,617 MIND-small users with >1 impression row; 0 had a history string that varied across their own rows — confirms history is a static pre-log-window snapshot). MIND's official documentation describes history entries as listed in chronological order; **this project has not independently verified that ordering** (no per-click timestamps exist in MIND to check list order against) — treated as documentation-tier evidence, not benchmarked fact.

## Theoretical Evidence

- Wu et al. (2020), MIND — field definitions for `news.tsv` / `behaviors.tsv`.
- Kruse et al. (2024), EB-NeRD — field definitions and active-user filtering methodology.

## Accepted Trade-offs

- **EB-NeRD's `context_article_id` (which article the user was reading when the impression fired, ~29.8% populated) is reduced to a single `is_front_page: bool` in the unified `impressions` table.** This is a real information loss, not a free derivation — the *specific* context article is discarded, only the front-page/article-page distinction survives. Accepted because no MIND equivalent exists to unify against, and the binary distinction is what's likely to matter for query construction; the raw `article_id` context field remains available in EB-NeRD's native files if finer-grained analysis is ever needed.
- **EB-NeRD's `scroll_percentage_fixed` (per-click scroll depth in history) is not carried into the unified `user_history` table** — only `read_times` is. Accepted for the same reason: no MIND equivalent, and read time is the primary engagement signal; scroll depth can be pulled from native EB-NeRD files if needed later.
- **`article_ids` in `user_history` is documented as "ordered in EB-NeRD (verified), presumed ordered in MIND (per MIND's documentation, not independently verified)"** rather than flatly "unordered in MIND" — the original framing overstated what was actually confirmed. Any code that depends on MIND history order should treat it as lower-confidence than EB-NeRD's.

---

# Unified Schema

## `articles`

| Field | Type | Mandatory? | Notes |
|---|---|---|---|
| `article_id` | str | required | dataset-prefixed to avoid collision (e.g. `mind:N55528`, `ebnerd:9738663`) |
| `dataset` | str (`mind`\|`ebnerd`) | required | renamed from `source` for consistency with `impressions`/`user_history` (see Notes) |
| `title` | str | required | |
| `abstract` | str | required (column always present; **5.2% null in practice for MIND** — see Known Limitations) | MIND `abstract` / EB-NeRD `subtitle` |
| `body` | str | optional — EB-NeRD only | always null for MIND |
| `category` | str | required | MIND `category` / EB-NeRD `category_str` |
| `subcategory` | raw passthrough (str for MIND, list[int] for EB-NeRD) | optional | present in both, **not** EB-NeRD-exclusive; no unified type possible until an EB-NeRD subcategory string mapping is found (unresolved — see Decision Confidence) |
| `entities` | raw passthrough JSON (dataset-specific shape) | optional | present in both (73.0% / 85.7% nonempty respectively), **not** EB-NeRD-exclusive; structurally incompatible, not force-unified |
| `topics` | list[str] | optional — EB-NeRD only | always null for MIND |
| `article_published_time` | datetime | optional — EB-NeRD only | **always null for MIND, by construction — not a data quality issue** |

## `impressions`

One row per (impression, candidate article) pair — both MIND's fused pairs and EB-NeRD's inview/clicked lists are exploded into this long format.

| Field | Type | Mandatory? | Notes |
|---|---|---|---|
| `impression_id` | str | required | dataset-prefixed |
| `dataset` | str (`mind`\|`ebnerd`) | required | |
| `user_id` | str | required | dataset-prefixed |
| `article_id` | str | required | the candidate article being scored in this row |
| `impression_time` | datetime | required | |
| `clicked` | bool | required | |
| `session_id` | str | optional — EB-NeRD only | |
| `dwell_time` | float | optional — EB-NeRD only | EB-NeRD `read_time` |
| `scroll_percentage` | float | optional — EB-NeRD only | |
| `is_front_page` | bool | optional — EB-NeRD only | derived from EB-NeRD's raw `article_id` (context) column being null; **lossy** — see Accepted Trade-offs |

## `user_history`

| Field | Type | Mandatory? | Notes |
|---|---|---|---|
| `user_id` | str | required | dataset-prefixed |
| `dataset` | str (`mind`\|`ebnerd`) | required | |
| `article_ids` | list[str] | required | ordered in EB-NeRD (verified); presumed ordered in MIND per its documentation, **not independently verified** — see Empirical Evidence |
| `click_times` | list[datetime] | optional — EB-NeRD only | always null for MIND, by construction |
| `read_times` | list[float] | optional — EB-NeRD only | |

---

# Assumptions

| Assumption | Why It Matters | Monitoring Strategy |
|------------|----------------|---------------------|
| Null fields are acceptable for dataset-specific columns (not backfilled/imputed) | Imputing a fake `article_published_time` for MIND would silently fabricate signal that doesn't exist | Any feature/model using an optional field must explicitly branch on `dataset`, never assume presence |
| MIND's history list order is chronological | Query construction (e.g. recency-weighted history features) depends on this for MIND specifically | Flagged as documentation-tier, not verified — do not build a MIND recency feature that would break silently if this assumption is wrong; revisit before implementing Q3.5's recency-weighting hypothesis from Phase 1A |
| Mandatory core fields are sufficient for Phase 2–3 (BM25 + semantic retrieval) implementation | If wrong, retrieval code would need to fall back on optional fields, breaking the single-code-path goal | Re-verify once BM25/semantic implementation starts; not yet verified against Q4's diversity/coverage metrics specifically |
| `ebnerd_small`/`ebnerd_large` share `ebnerd_demo`'s exact column structure | This schema was only inspected against the demo bundle | Re-run the same schema inspection against `ebnerd_small`/`ebnerd_large` before relying on them for final leaderboard submission |

---

# Conditions for Revisiting

## Technical Triggers

Revisit if:

- [x] A field currently dropped (`context_article_id`, `scroll_percentage_fixed`) turns out to be needed for a later feature
- [ ] Performance regression
- [ ] Latency exceeds threshold

---

## Research Triggers

Revisit if:

- [x] ~~`ebnerd_small`/`ebnerd_large` reveal a different schema than `ebnerd_demo`~~ **Resolved (2026-08-10 addendum): `ebnerd_small` schema is identical to `ebnerd_demo`; `ebnerd_large` remains unverified.**
- [ ] Significant new research appears

---

## Project / Assignment Triggers

Revisit if:

- [x] Q4's diversity/novelty/coverage metrics need a field not currently in the mandatory core (e.g., a resolved `subcategory` string)
- [x] A usable EB-NeRD subcategory-to-string mapping is found, allowing `subcategory` to become a true unified field instead of raw passthrough
- [ ] Assignment requirements change

---

**Estimated Cost to Change**

Easy — additive (new optional columns) for most cases; changing a mandatory field would touch the retrieval/evaluation code paths that depend on it.

---

# Engineering Impact

## Affected Components

- Data Pipeline
- Feature Store
- Retrieval
- Evaluation

---

## Affected Files

- src/datasets/ (per-dataset loaders producing the unified schema)
- src/pipeline/ (schema validation)
- tests/unit/ (schema conformance tests)
- tests/integration/ (cross-dataset pipeline tests)

---

## Expected Refactoring

None yet — this precedes implementation. It's the contract `src/datasets/` loaders must produce.

---

## Breaking Changes

No — no implementation exists yet.

---

## Required Tests

- Unit: every mandatory field is non-null for every row, for both datasets.
- Unit: every optional field marked "EB-NeRD only" is null for every MIND row, and vice versa (catches a loader bug that accidentally populates a field that shouldn't exist for a dataset).
- Integration: `impressions` row count after exploding equals the sum of candidate-list lengths in the raw files (regression guard against silent row loss/duplication during the explode step).
- Integration: every `article_id` referenced in `impressions`/`user_history` exists in `articles` (referential integrity across the unified tables).

---

## Expected Benchmarks

Not applicable — this decision is about information preservation and code-path unification, not measured retrieval performance.

---

## Documentation Updates

- [x] Architecture (ARCHITECTURE.md — Feature Store's "Related Decisions" already stubs ADR-002)
- [x] Project State
- [ ] Knowledge Base

---

# Benchmark Results

Not applicable — see "Expected Benchmarks" above.

---

# Related Decisions

## Influenced By

- ADR-001 (Temporal Split Strategy) — this schema assumes official train/dev/validation files are loaded as-is, not re-split, per ADR-001's decision

## Influences

- Future ADR on embedding model choice (mandatory text fields `title`+`abstract` define the minimum text available to encode for both datasets)
- Future ADR on BM25 variant (same mandatory-text constraint)

---

# References

## Internal

- ARCHITECTURE.md — System Data Flow (Feature Store stage)
- PROJECT_STATE.md — Phase 1A Learning Progress, Phase 1B.2 schema inspection findings
- ADR-001-temporal-split-strategy.md
- `data/raw/mind/`, `data/raw/ebnerd/` — inspected directly for this ADR

## External

- Wu et al. (2020). *MIND: A Large-scale Dataset for News Recommendation.* ACL Anthology 2020.acl-main.331.
- Kruse et al. (2024). *EB-NeRD: A Large-Scale Dataset for News Recommendation.* arXiv:2410.03432 / ACM RecSys Challenge 2024.

---

# Decision History

| Date | Event |
|------|-------|
| 2026-08-09 | Schema inspection performed against MINDsmall + ebnerd_demo |
| 2026-08-09 | Alternatives documented |
| 2026-08-09 | Decision made; two field-attribution errors (subcategory, entities) corrected against inspection evidence during drafting |

---

# Notes

**Corrections made during drafting, against the proposed schema:**

1. `subcategory` was proposed as "EB-NeRD only." Inspection shows the opposite emphasis is more accurate: MIND has `subcategory` as a clean flat string (264 values); EB-NeRD's `subcategory` is a `list[int]` with no string label found anywhere in `articles.parquet`. Both datasets have the field; only MIND's is directly usable as text. Documented as present-in-both, raw passthrough, pending resolution of an EB-NeRD subcategory string mapping.
2. `entities` was proposed as "EB-NeRD only." Both datasets have populated entity fields (MIND 73.0% nonempty, EB-NeRD 85.7% nonempty) — they're structurally incompatible with each other (Wikidata-linked vs. Danish NER clusters), not exclusive to one dataset. Documented as present-in-both, raw passthrough.
3. `source` (articles table) renamed to `dataset` for consistency with the `impressions`/`user_history` tables, which already used `dataset` in the original proposal.
4. The **discrepancy resolution** (assignment's 2.7M users/600M+ impressions vs. paper's ~1M/37M active-user-filtered subset vs. `ebnerd_demo`'s directly-measured 1,590 train-window users/1,562 validation-window users, 24,724+25,356 impressions) is carried forward from the Phase 1B.2 schema analysis at **medium confidence** — it is inference from the papers' stated methodology plus `ebnerd_demo`'s structural match to that methodology (verified in ADR-001: 21-day-history/7-day-window cadence), not a direct inspection of `ebnerd_small`/`ebnerd_large`, which remain undownloaded. Flagged as a condition for revisiting above.

---

# Addendum — `ebnerd_small` Verification (2026-08-10)

**Status:** Resolves the open question this ADR flagged above ("`ebnerd_small`/`ebnerd_large` haven't been inspected" / "Re-run the same schema inspection against `ebnerd_small`/`ebnerd_large` before relying on them"). Per CLAUDE.md's decision-reversal guidance, this is appended rather than rewriting the original decision — the original schema design was correct; what was unverified is now verified.

## What was checked

`ebnerd_small.zip` (84,135,301 bytes) was downloaded from the assignment's S3 bucket
(`https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip`) — publicly
accessible with no registration/auth wall, which is worth noting since
`src/pipeline/download.py`'s module docstring currently describes EB-NeRD as
"consent/terms-gated (registration required)"; that claim is now known to be
inaccurate for this specific S3 distribution (not corrected in the download
module itself as part of this addendum — flagged here as a minor,
non-blocking documentation drift, since `ensure_raw_data()`'s actual behavior,
manual placement under `data/raw/ebnerd/`, still works regardless of why).

1. **Raw zip structure** — inspected via `unzip -l` before trusting
   `src/datasets/ebnerd.py`'s existing parse code against it: identical
   member layout to `ebnerd_demo.zip` (top-level `articles.parquet`,
   `{train,validation}/behaviors.parquet`, `{train,validation}/history.parquet`,
   plus the same `__MACOSX/` junk entries the loader already knows to
   ignore). No structural surprises.
2. **Schema conformance** — built via the existing, already-generic
   `build_ebnerd_bundle()` (zero code changes needed, confirming ADR-002's
   own design goal: the unified schema doesn't care which EB-NeRD bundle
   feeds it) and validated against `ARTICLES_SCHEMA`/`IMPRESSIONS_SCHEMA`/
   `USER_HISTORY_SCHEMA` — **passes, identical to `ebnerd_demo`**. Referential
   integrity (every `article_id` in `impressions`/`user_history` exists in
   `articles`) and the impressions-row-count-vs-raw-candidate-count
   regression check both pass. New tests:
   `tests/integration/test_schema_conformance.py::test_ebnerd_small_*`
   (mirrors the existing `ebnerd_demo` parametrization).
3. **Temporal structure** — validation window `2023-05-25 → 2023-06-01`
   (identical 7-day window to `ebnerd_demo`'s validation split, per
   ADR-001), confirming the 21-day-history/7-day-window cadence generalizes
   beyond the demo bundle, not just an artifact of demo's specific sampling.
4. **Cold-start distribution** — the open question ADR-005 specifically
   flagged as a revisit trigger: **`ebnerd_small`'s validation split also
   has zero users below the `<5` history-length threshold** (min history =
   5, exactly like `ebnerd_demo`). This is not a demo-only artifact — it's
   structural to how EB-NeRD's `small`/`demo` bundles are both sampled from
   the paper's active-user-filtered population (5–1,000 clicks). **The
   warm/cold BM25 comparison remains genuinely unavailable for EB-NeRD**,
   confirmed at a second, larger, independently-downloaded bundle rather
   than assumed to generalize from one data point.
5. **Recall benchmark** (`experiments/bm25_ebnerd_small_2026-08-10/`),
   compared against the existing demo numbers (ADR-006):

   | Metric | ebnerd_demo (ADR-006) | ebnerd_small | Change |
   |---|---|---|---|
   | Corpus size | 11,777 articles | 20,738 articles | 1.76x larger |
   | Validation users | 1,562 | 15,342 | 9.8x larger |
   | Positive impressions | 25,356 | 245,622 | 9.7x larger |
   | recall@50 | 1.01% (0.85–1.17%) | 0.72% (0.68–0.76%) | lower |
   | recall@100 | 2.13% (1.90–2.37%) | 1.44% (1.38–1.49%) | lower |
   | recall@200 | 4.02% (3.69–4.39%) | 2.77% (2.69–2.85%) | lower |
   | Random baseline @200 | 1.70% | 0.96% | — |
   | Lift over random @200 | 2.37x | **2.87x** | *higher* |

## Interpretation

Absolute recall@K is lower on `ebnerd_small` than `ebnerd_demo`, but this
tracks the corpus growing 1.76x (recall@fixed-K over a bigger haystack is
mechanically harder, not a sign the retrieval got worse) — the
lift-over-random figure, which normalizes for corpus size, is actually
*higher* on `ebnerd_small` (2.87x vs. 2.37x), consistent with the same BM25
signal holding up, or even strengthening slightly, at 10x the user/impression
scale. This is a real, reportable finding (both numbers ship, not just the
favorable one), not a discrepancy that needed explaining away.

## Confidence impact

This addendum moves several of ADR-002's own "Decision Confidence" caveats
from unverified to verified: `ebnerd_small` shares `ebnerd_demo`'s exact
column structure (previously listed as "haven't been inspected"). ADR-002's
overall confidence should be read as **Medium → Medium-High** as a result —
still not High, because the Q4 diversity/coverage stress-test and the
EB-NeRD subcategory-string-mapping gap (both called out in the original
Decision Confidence section) remain genuinely open, unrelated to this
addendum.

## Related

- Resolves the `ebnerd_small`/`ebnerd_large` schema-verification open
  question in this ADR's own "Conditions for Revisiting" and in
  `PROJECT_STATE.md`'s "Open Engineering Questions."
- Confirms ADR-005's cold-start-threshold finding ("EB-NeRD has no cold
  users") is a property of the active-user-filtered `demo`/`small`
  bundles, not a `demo`-only artifact — ADR-005's own revisit trigger for
  this is now resolved the same way.
- `experiments/bm25_ebnerd_small_2026-08-10/{config,results}.json`
