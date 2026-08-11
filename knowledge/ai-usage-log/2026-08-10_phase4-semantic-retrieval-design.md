# AI Usage Log — 2026-08-10 — Phase 4 Semantic Retrieval Design + Housekeeping

Per CLAUDE.md's "Prompt & Session Logging" section. Written live, as the
session proceeds — not reconstructed afterward.

Tool: Claude Code (Sonnet 5).

---

## Prompt 1 (session start)

```
Current Objective: Close open housekeeping items, then Phase 4 — Semantic
Retrieval (embeddings) design + implementation, wired into the Q4 harness.

Foundation: BM25 (Phase 3) and the Q4 ranking harness (Scorer interface,
ranking_metrics.py) are implemented, benchmarked, and documented (ADR-005,
006, 007, ADR-002 addendum) against MINDsmall-dev, ebnerd_demo-validation,
and ebnerd_small. CLAUDE.md now has a "Resource Availability" clause and a
"Prompt & Session Logging" requirement — both must be honored this session,
not just noted.

Read first: CLAUDE.md (including the two new sections), PROJECT_STATE.md,
ARCHITECTURE.md, ADR-002 (+ addendum), ADR-005, ADR-006, ADR-007,
Assignment1_v1.pdf Q3 and Q9.

Part 0 — Housekeeping (do first, five minutes, no loose ends):
- Confirm knowledge/ai-usage-log/ exists and this session's prompts are
  being written to it as we go, not reconstructed at the end.
- Register both Codabench competitions now if not already done — zero
  dependency on anything else, cheap to forget under deadline pressure.
- Update PROJECT_STATE.md's Deliverables Checklist (Q7) to reflect current
  state honestly before starting new work.

Part 1 — Phase 4 design (plan mode, before any embedding code):

The central decision: EB-NeRD ships pre-computed Word2Vec + multilingual
BERT embeddings; MIND ships none. Deciding how to handle this split is the
first real design question — write it up as ADR-008, decision-criteria
table first, same pattern as ADR-005/006:

- Option A: use EB-NeRD's provided embeddings as-is, compute a separate
  model for MIND. Sacrifices single-code-path consistency (ADR-002's
  governing goal) — two different embedding spaces, not directly comparable
  across datasets.
- Option B: compute one cross-lingual model (e.g. XLM-R-based sentence
  embeddings) over both datasets' title+abstract text, ignoring the
  provided EB-NeRD embeddings. Single code path, but discards a resource
  the assignment explicitly hands you and costs more compute.
- Whichever is chosen, this is exactly where the CLAUDE.md Resource
  Availability clause applies: if the chosen model is too slow/heavy on
  free-tier Kaggle GPU, do not silently drop to a smaller/weaker model.
  Stop, name the constraint, benchmark actual throughput on a small sample
  first (like the BM25 scoring benchmark did), and bring the trade-off back
  as a decision, not a quiet substitution.

Also decide and document in the same ADR:
- ANN backend: given corpus sizes (11,777-42,416 articles), brute-force
  cosine similarity via a vectorized matrix product is the BM25 lesson
  repeating itself — benchmark it before reaching for FAISS, don't assume
  FAISS is needed just because it's the "proper" answer.
- User representation: mean-pooled embeddings of history articles (per Q3),
  and what a zero-history user produces (structurally the same cold-start
  gap BM25 hit — decide whether this is the finding that motivates a
  cold-start-specific strategy, or just reported the same way ADR-005 did).

Part 2 — Implementation:
- src/retrieval/embed.py (embedding computation/loading), src/retrieval/
  ann_index.py (or reuse score.py's Scorer interface — embeddings should be
  the second consumer of the same interface BM25 already implements, per
  ADR-007's design intent).
- Benchmark embedding computation time on MIND's real corpus before running
  the full pipeline (same discipline as ADR-006's pre-flight check).
- Run recall@{50,100,200} (Q3.4) and the full Q4 ranking harness against
  embeddings on both datasets, same as BM25 already has.

Part 3 — The actual A/B test (Q3.5 / Q4.5):
- Compare BM25 vs. semantic recall and ranking metrics side by side, sliced
  by the existing warm/cold cohort, on both datasets. This is the
  comparison the whole project has been building toward — report it
  honestly, including if the working hypothesis (BM25 favors warm/entity-
  heavy, embeddings favor cold-start/paraphrase-heavy) doesn't hold.

Deliverable: ADR-008; embedding + ANN code, tested; Q4 harness run on
embeddings for both datasets; comparison table (BM25 vs. semantic, warm vs.
cold) in PROJECT_STATE.md; ARCHITECTURE.md updated; full test suite green.

Do not start Q5 (leaderboard submission) or Q6 (design note) this session —
both depend on having real semantic numbers to compare against first.
```

---

## AI-generated vs. human-written

**Human-written this session:**
- The session-start prompt above (engineer's own task specification).

**AI-generated this session** (Claude Code; plan-mode approval gate used
before implementation began, per CLAUDE.md):
- `PROJECT_STATE.md` (Deliverables Checklist, Component Status Summary,
  Recent Decisions, Open Engineering Questions, Benchmarking Status,
  Next Actions, Session Notes)
- `decisions/ADR-008-semantic-retrieval-design.md` (new)
- `src/retrieval/embed.py` (new)
- `src/retrieval/score.py` (`EmbeddingScorer` added)
- `src/retrieval/retrieve.py` (`embed_retrieve_top_k`, `_top_k_from_scores` extracted)
- `scripts/run_embed_experiment.py` (new)
- `scripts/run_ranking_eval.py` (`_build_method` dispatch, `--method embed`)
- `pyproject.toml` (`sentence-transformers` dependency added)
- `tests/unit/test_embed.py` (new); `tests/unit/test_score.py`,
  `tests/unit/test_retrieval.py` (extended)
- `tests/integration/test_retrieval_pipeline.py` (extended)
- `ARCHITECTURE.md` (Retrieval/Evaluation components, Decision
  Traceability, Component Interfaces, Architecture Changelog)
- This log file

**Not human-edited beyond the plan-approval gate** — the engineer approved
the plan in `/Users/test01/.claude/plans/wild-drifting-parasol.md` before
any implementation began; no further inline code review edits were made
this session beyond that approval.

**Still outstanding, flagged to the engineer, not resolved by Claude Code:**
- Codabench registration for both competitions (requires the engineer's own
  account/login).
- `README.md`'s usage docs for `run_embed_experiment.py` (not updated this
  session).
