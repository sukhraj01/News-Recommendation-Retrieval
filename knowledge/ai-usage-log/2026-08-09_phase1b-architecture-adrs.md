# AI Usage Log — 2026-08-09 — Phase 1A Confirmation + Phase 1B Architecture ADRs

Per CLAUDE.md's "Prompt & Session Logging" section. This session's prompts are
logged retroactively (the CLAUDE.md requirement was added after this session
had already happened) but verbatim and complete — reconstructed from the
actual session transcript, not summarized.

This is the earliest recorded session for this project (commits
`ed2844f`…`592cf93`, 2026-08-09 17:36–18:12) — it precedes the
`2026-08-10_ebnerd-small-q4-harness.md`, `2026-08-10_phase4-semantic-retrieval-design.md`,
and `2026-08-11_mindlarge-codabench-submission.md` logs already in this
directory. Dated accordingly so the log directory stays chronologically
honest.

Tool: Claude Code (Sonnet 5).

---

## Prompt 1 (session start)

```
I've completed Phase 1A (build mental model of recommendation systems).

Read:
1. CLAUDE.md (operating model)
2. PROJECT_STATE.md (project state, including my Phase 1A learning notes)

Confirm you've read both and understand:
- Where we are (Phase 1A complete)
- What we learned (mental model grounded in MIND + EB-NeRD papers)
- What's next (Phase 1B: Architecture exploration — temporal split + unified schema ADRs first)

Then I'll give you Phase 1B objective.
```

**Outcome:** Claude read `CLAUDE.md` and `PROJECT_STATE.md`, summarized Phase 1A's mental-model findings and confirmed understanding of the Phase 1B objective. No code or docs written — confirmation only.

---

## Prompt 2

```
Current Objective: Phase 1B — Architecture Exploration

Foundation laid (Phase 1A complete). Now we make architecture decisions grounded in what we learned.

Two parallel ADRs to create:

1. ADR-001: Temporal Split Strategy
   - Question: Last 7 days vs. last 14 days as test boundary?
   - Why it matters: Affects cold-start severity, leakage risk, metric reliability
   - Research: MIND splits by date; EB-NeRD uses 21-day history + 7-day forward window
   - Test: What does leakage look like for each choice? Any risk?
   - Decision needed: Pick one with justification

2. ADR-002: Unified Data Schema (for both MIND + EB-NeRD)
   - Question: What fields do we need in articles, behaviors, impressions?
   - Why it matters: Constrains feature store, retrieval, evaluation
   - Research: MIND schema vs. EB-NeRD schema — what's common?
   - Discrepancy to resolve: Verify actual EB-NeRD bundle stats (demo vs. paper's filtered subset)
   - Decision needed: Unified schema that works for both datasets

Definition of Done:
- Two ADRs created in /decisions/ (ADR-001-temporal-split.md, ADR-002-unified-schema.md)
- Each ADR has: question, alternatives (2-3), rationale for choice, assumptions
- PROJECT_STATE.md updated with "Phase 1B Complete"
- Git commit: "Phase 1B: Define temporal split strategy and unified data schema"

This unblocks Phase 2 (data pipeline implementation).

Let's start with temporal split: should we use last 7 days or last 14 days as the test boundary?
```

**Outcome:** Claude inspected the actual downloaded data (MINDsmall, MINDlarge, `ebnerd_demo` — zip contents, date ranges, schemas, impression/user counts) rather than reasoning from the papers alone, then presented an analysis and recommendation (7-day windows; 14 days is arithmetically infeasible given the on-disk data budgets) before writing anything to disk. No files written yet — analysis and discussion only.

---

## Prompt 3

```
Current Objective: Phase 1B.1 — Write ADR-001: Temporal Split Strategy

You have the data inspection findings. Write ADR-001 now.

Context:
- MIND-small actual: train (6d: Nov 9-14) + dev (1d: Nov 15) = 7d total
- MIND-large actual: train (6d) + dev (1d) + test (7d) = 14d total, test is official
- EB-NeRD-demo actual: train (7d: May 18-25) + val (7d: May 25-Jun 1) = 14d total
- EB-NeRD-demo history: exactly 21 days prior to each window (structural leakage prevention)
- MIND news.tsv: no per-article timestamps; only impression-level timestamps
- EB-NeRD: article timestamps + impression timestamps both present

Key finding: dataset creators converged on ~7-day windows independently. This isn't 
coincidence — it's the data budget that's actually available.

Task: Write /decisions/ADR-001-temporal-split-strategy.md

Use the ADR template from /decisions/ADR-TEMPLATE.md

Structure:
1. **Question:** Use official train/dev/val splits, or re-carve from raw logs?

2. **Context:** Explain why this matters (leakage prevention, evaluation integrity)

3. **Alternatives Considered:**
   - Option A: Use official splits as-is (7-day test windows)
   - Option B: Re-carve our own longer test window (e.g., 14 days)
   - Option C: Variable per dataset (7 days for MIND-small, 14 for MIND-large)

4. **Decision:** Adopt official train/val splits as-is. Use MIND-dev and EB-NeRD-val 
   as our local test sets (7-day windows for both).

5. **Rationale:**
   - Data budget: 14-day test window is arithmetically impossible with demo/small bundles
   - Evidence tier: matches official dataset documentation (tier 4 in evidence hierarchy)
   - Leakage prevention: EB-NeRD's 21-day history window is structural, confirmed empirically
   - Convergence: both dataset creators independently chose ~7-day windows (not coincidence)

6. **Assumptions:**
   - 7-day forward window is sufficient for meaningful offline evaluation
   - Official splits reflect the dataset creators' own leakage-prevention strategy
   - MIND's true test set (hidden labels) and EB-NeRD's true test set (hidden labels) 
     are leaderboard-only; our "test" is the official val/dev sets

7. **Trade-offs:**
   - Pro: Aligned with official dataset design; simpler (no custom split code)
   - Con: Smaller test set than might be ideal (but only option we have)

8. **Conditions for Revisiting:**
   - If we move to MIND-large (larger data budget), reconsider holding out full 7-day test
   - If evaluation metrics suggest insufficient test coverage, revisit

9. **Reversibility:** Very easy (just load different files). Very low cost.

10. **Data Inspection Evidence:**
    Include the actual file inspection summary:
    - MIND-small: 6d train / 1d dev (156,965 / 73,152 impressions)
    - EB-NeRD-demo: 7d train / 7d val (both May 18–Jun 1)
    - Timestamp granularity differs (impression-level for MIND, per-article for EB-NeRD)

Definition of Done:
- ADR-001 written and saved to /decisions/ADR-001-temporal-split-strategy.md
- Evidence is grounded in actual data inspection, not papers alone
- Trade-offs are explicit
- Assumptions are testable

Then: git add /decisions/ADR-001-temporal-split-strategy.md && git commit -m "ADR-001: Temporal split strategy — adopt official train/val boundaries"

After that, report back. Next is ADR-002 (unified schema).
```

**Outcome (AI-generated):** `decisions/ADR-001-temporal-split-strategy.md` written in full by Claude, following the ADR template, populated with the empirical evidence gathered in Prompt 2. Committed as `9343191`. No human edits to the file content.

---

## Prompt 4

```
Current Objective: Phase 1B.2 — Analyze Data Schema (Before Writing ADR-002)

We have ADR-001 (temporal split) done. Now we need a unified data schema 
that works for both MIND and EB-NeRD across our entire pipeline.

Task: Inspect both datasets and report findings. Don't write ADR-002 yet — 
just analyze and present the landscape.

Datasets to inspect:
- MIND-small: /data/raw/mind/MINDsmall_train.zip, MINDsmall_dev.zip
- EB-NeRD-demo: /data/raw/ebnerd/ebnerd_demo.zip

For each, extract and examine:

1. **Articles schema**
   - MIND: news.tsv — fields and sample rows
   - EB-NeRD: articles.parquet — fields and sample rows
   - Question: What's common? What's different?
   - Question: Which fields are mandatory for BM25 (text) and semantic (embeddings)?

2. **Behaviors/Impressions schema**
   - MIND: train/behaviors.tsv, dev/behaviors.tsv — fields and sample rows
   - EB-NeRD: train/behaviors.parquet, validation/behaviors.parquet — fields and sample rows
   - Question: What represents an "impression" in each?
   - Question: What represents "clicked" vs "displayed but not clicked"?

3. **User history schema**
   - MIND: extracted from behaviors (per-user click sequence)
   - EB-NeRD: history.parquet — fields and sample rows
   - Question: Are per-click timestamps available?
   - Question: How is "in-view" vs "clicked" recorded?

4. **Unified schema proposal**
   - What single schema could represent articles from both datasets?
   - What single schema could represent impressions/behaviors from both?
   - What single schema could represent user history from both?
   - Which fields are dataset-specific (keep separate) vs. universal (normalize)?

5. **Resolve outstanding discrepancy**
   - Assignment PDF says EB-NeRD is "~2.7M users, 600M+ impressions"
   - Paper reports "~1M users, 37M impressions" for active-user-filtered subset
   - ebnerd_demo shows: 1,590 users, 24.7k train + 25.3k val impressions
   - What's the relationship? Full dataset vs. filtered? Demo vs. small vs. large?

Output format:

Report (not code, not ADR yet):
- Schema comparison table (articles: MIND vs EB-NeRD)
- Schema comparison table (impressions: MIND vs EB-NeRD)
- Schema comparison table (user history: MIND vs EB-NeRD)
- Unified schema proposal (three tables: articles, impressions, user_history)
- Resolution of the 2.7M/600M vs 1M/37M discrepancy
- Flags/unknowns (e.g., "MIND provides no per-click timestamps; EB-NeRD does")

Definition of Done:
- You've inspected actual files and extracted their schemas
- You've proposed a unified representation
- You've flagged dataset-specific quirks (e.g., no MIND article timestamps)
- You've resolved the user/impression count discrepancy

Then: report back with the findings. I'll write ADR-002 based on your analysis.
```

**Outcome:** Analysis only, no files written. Claude extracted actual MIND/EB-NeRD data (unzipped to scratchpad, used a throwaway venv with pandas/pyarrow to read the parquet files), produced schema comparison tables, quantified item cold-start empirically (32.9% MIND, 45.7% EB-NeRD), verified MIND's per-user history is fixed (not incrementally growing) across 33,617 multi-impression users, and gave a medium-confidence resolution of the user/impression-count discrepancy. Presented as a chat report per the prompt's explicit "don't write ADR-002 yet" instruction.

---

## Prompt 5

```
Current Objective: Phase 1B.2 — Write ADR-002: Unified Data Schema

You have the schema analysis above. Write ADR-002 now.

Use the analysis to populate:

1. **Question:** What unified schema can represent articles, impressions, and 
   user history for both MIND and EB-NeRD without losing critical information?

2. **Context:** Both datasets exist, but with different field structures:
   - MIND: no article-level timestamps, unordered per-user click history
   - EB-NeRD: article timestamps present, per-click timestamps in history
   - Both required for BM25 (text fields) and semantic (embeddings/entities)

3. **Alternatives Considered:**
   - Option A: Separate pipelines per dataset (lose cross-dataset comparison)
   - Option B: Unified schema that preserves all fields from both (bloated, null-heavy)
   - Option C: Unified schema with dataset-specific optional fields (clean, handles both)

4. **Decision:** Option C — Unified schema with mandatory core fields + 
   dataset-specific optional fields.

5. **Rationale:**
   - Enables single BM25/semantic retrieval code path for both datasets
   - Preserves all information from both sources (no lossy mapping)
   - Allows Q4 cross-dataset comparison (BM25 vs semantic on MIND vs EB-NeRD)
   - Fields like "article_published_time" are optional (MIND: null, EB-NeRD: present)

6. **Unified Schema Proposal** (use the analysis above):
   
   **articles table:**

article_id (required, str)
title (required, str)
abstract (required, str)
body (optional, str) — EB-NeRD only
category (required, str)
subcategory (optional, str) — EB-NeRD only
entities (optional, list[str]) — EB-NeRD only
topics (optional, list[str]) — EB-NeRD only
article_published_time (optional, datetime) — EB-NeRD only
source (required, str) — MIND or EB-NeRD

   
   **impressions table:**

impression_id (required, str)
user_id (required, str)
article_id (required, str)
session_id (optional, str) — EB-NeRD only
impression_time (required, datetime)
clicked (required, bool)
dwell_time (optional, float) — EB-NeRD only
scroll_percentage (optional, float) — EB-NeRD only
is_front_page (optional, bool) — EB-NeRD only
dataset (required, str) — MIND or EB-NeRD

   
   **user_history table:**

user_id (required, str)
article_ids (required, list[str]) — ordered in EB-NeRD, unordered in MIND
click_times (optional, list[datetime]) — EB-NeRD only
read_times (optional, list[float]) — EB-NeRD only
dataset (required, str) — MIND or EB-NeRD


7. **Assumptions:**
   - Null fields are acceptable for dataset-specific columns
   - "unordered" click history in MIND is still usable (order is reconstructed at query time if needed)
   - All fields extracted from schema analysis above are sufficient for Phase 2-3 implementation

8. **Trade-offs:**
   - Pro: Single schema, cross-dataset code
   - Con: Some fields always null for certain datasets (minor schema bloat)

9. **Conditions for Revisiting:**
   - If a new field becomes critical for evaluation/retrieval
   - If ebnerd_small/ebnerd_large reveal different schema than ebnerd_demo

10. **Reversibility:** Easy (add/remove columns as needed)

11. **Evidence:**
    - Schema comparison table from analysis above
    - Resolved discrepancy: assignment's 2.7M/600M refers to full dataset across all bundles; 
      paper's 1M/37M is active-user-filtered subset; ebnerd_demo is 1.6k users / 50k impressions

Task: Write /decisions/ADR-002-unified-data-schema.md

Use ADR-TEMPLATE.md structure.

Definition of Done:
- ADR-002 written and saved
- Unified schema explicitly defined (three tables as above)
- Dataset-specific fields clearly marked
- Discrepancy resolution documented
- Flags for null-handling documented

Then: git add /decisions/ADR-002-unified-data-schema.md && git commit -m "ADR-002: Unified data schema for MIND and EB-NeRD"
```

**Outcome (AI-generated, with corrections flagged to the human before writing):** `decisions/ADR-002-unified-data-schema.md` written by Claude. Before writing, Claude flagged and corrected two factual errors in the prompt's proposed schema against its own verified data (`subcategory` and `entities` are present in *both* datasets, not EB-NeRD-exclusive as proposed), unified the `source`/`dataset` field-naming inconsistency, and downgraded an unverified claim ("unordered in MIND") to a documented assumption. Committed as `b3fc951`. Note: this file shows as modified again after creation per the harness's own file-watch note — the diff appeared content-equivalent (formatting), not a substantive rewrite.

---

## Prompt 6

```
Current Objective: Phase 1B Final — Close Phase 1B (Architecture Foundation)

We have:
- ADR-001: Temporal Split Strategy (commit 9343191)
- ADR-002: Unified Data Schema (commit b3fc951)

Task: Update PROJECT_STATE.md to close Phase 1B, then make the final commit.

Update PROJECT_STATE.md:

1. Change "Current Phase" to: "Phase 2 — Data Pipeline Implementation"

2. Update "Current Objective" to: "Build reproducible data pipeline (download → parse → split → feature store)"

3. In the "Component Status Summary" table, update:
   - Data Pipeline: ⏳ Not Started → 🔄 In Progress
   - Keep Lexical/Semantic/Evaluation/Benchmarking as ⏳ Not Started

4. Add to "Recent Decisions" table:
   - ADR-001 | Temporal Split Strategy | Decided | Use official train/val splits as-is
   - ADR-002 | Unified Data Schema | Decided | Three tables (articles, impressions, user_history) with mandatory core + optional fields

5. Update "Learning Progress" section:
   Add a new date entry:

   **August 9, 2026 — Phase 1B: Architecture Exploration Complete (Decisions ADR-001 & ADR-002)**
   
   - Explored temporal split (7 vs 14 days) using actual on-disk data inspection
   - Decision: adopt official train/val splits (7-day windows) — only feasible option given data budgets
   - Inspected schemas for MIND and EB-NeRD (corrected two field-attribution errors in the process)
   - Decision: unified three-table schema with mandatory core + dataset-specific optional fields
   - Both decisions enable single BM25/semantic code path for cross-dataset Q4 comparison
   - Confidence: Medium (verified for retrieval; not yet checked against Q4 diversity/coverage metrics)

6. Update "Next Steps" section to:
   - [ ] Implement data pipeline (download + parse + split + feature store build)
   - [ ] Run temporal-split leakage tests
   - [ ] Verify unified schema works for both datasets in practice

Definition of Done:
- PROJECT_STATE.md updated with Phase 1B closure
- Reflects both ADRs (ADR-001, ADR-002)
- Ready to transition to Phase 2

Then: git add PROJECT_STATE.md && git commit -m "Phase 1B Complete: Architecture decisions (ADR-001, ADR-002)

Temporal split strategy: adopt official train/val boundaries (7-day windows).
Unified schema: three tables with mandatory core fields + dataset-specific optional.
Both decisions enable single retrieval code path for MIND + EB-NeRD.

Next phase: Data pipeline implementation (download, parse, split, feature store).
"
```

**Outcome (AI-generated, minor unrequested fix flagged):** `PROJECT_STATE.md` updated per the six numbered instructions. Claude also lightly updated the (already-stale) EB-NeRD discrepancy entry under "Open Engineering Questions" to reflect ADR-002's resolution — not explicitly requested, flagged inline in the response rather than done silently. Session Notes log entries were left untouched (append-only, point-in-time record). Committed as `592cf93` with the exact message specified in the prompt.

---

## Prompt 7 (this entry)

```
we have to store the prompts i made during our chat i hvae added the point in the claude.md file as well and htere is the knowledge folder. i missed it out during u convo nowihave addd it so please maage he time as its part of the name of the file and dont name it later than the alread prent as you are the phase one chat
```

**Outcome (AI-generated):** This log file — `knowledge/ai-usage-log/2026-08-09_phase1b-architecture-adrs.md` — created retroactively to cover Prompts 1–7. Dated 2026-08-09 to match this session's actual commit timestamps (`ed2844f` through `592cf93`, 2026-08-09 17:36–18:12) and to sort before the three later-session logs already present (`2026-08-10_ebnerd-small-q4-harness.md`, `2026-08-10_phase4-semantic-retrieval-design.md`, `2026-08-11_mindlarge-codabench-submission.md`), per the engineer's instruction not to date it later than what's already there.

---

## Summary: AI-generated vs. human-written/edited

- **AI-generated in full:** `decisions/ADR-001-temporal-split-strategy.md`, `decisions/ADR-002-unified-data-schema.md`, all `PROJECT_STATE.md` edits made this session, this log file.
- **Human-written:** every prompt above (the engineer's objectives, structure, and explicit decisions — e.g., the choice of Option C in ADR-002, the exact commit messages).
- **Human-in-the-loop corrections surfaced by Claude, not silently applied:** the `subcategory`/`entities` dataset-attribution errors in the ADR-002 prompt's proposed schema, the `source`/`dataset` naming inconsistency, and the unverified "unordered in MIND" claim — all flagged in Claude's response before the file was written, per this project's evidence-over-consistency standard.
- **No code was written this session** — Phase 1B was architecture/documentation only (ADRs + project-state update); data inspection was done with throwaway scratch scripts, not committed to the repo.
