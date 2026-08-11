# AI Usage Log — 2026-08-10 — ebnerd_small Verification + Q4 Ranking Evaluation Harness

Per CLAUDE.md's "Prompt & Session Logging" section. This session's prompts
are logged retroactively (the CLAUDE.md requirement was added mid-session,
by the engineer directly, not by Claude Code) but verbatim and complete —
reconstructed from the actual session transcript, not summarized.

Tool: Claude Code (Sonnet 5).

---

## Prompt 1 (session start)

```
Current Objective: EB-NeRD `small` Verification + Q4 Ranking/Evaluation Harness

Foundation: BM25 retrieval (Phase 3) is implemented, benchmarked, and documented
(ADR-005, ADR-006), but only against MINDsmall-dev and ebnerd_demo-validation.
Assignment1_v1.pdf's Part 0 labels `ebnerd_demo` "for quick iteration" and
`ebnerd_small` "for final training" — separate tiers, not interchangeable.
Q4 of the assignment requires metrics recall@K doesn't cover.

Read first: CLAUDE.md, PROJECT_STATE.md, ARCHITECTURE.md, ADR-001, ADR-002,
ADR-005, ADR-006, Assignment1_v1.pdf (especially Q4, Q5, Q9).

Two objectives this session, in order. Do NOT start Phase 4 (semantic
retrieval) yet — the Q4 harness needs to exist and be validated against a
known method before a second method plugs into it.

1. EB-NeRD `small` verification (resolves ADR-002's open question)
   - Download ebnerd_small.zip, verify its schema against src/datasets/ebnerd.py
     (built/tested only against demo so far — don't assume it just works)
   - Check whether ebnerd_small's validation split has any users below ADR-005's
     cold-start threshold (<5 history) — demo had zero, which blocked the
     warm/cold comparison for EB-NeRD entirely
   - If schema matches, re-run scripts/run_bm25_experiment.py against ebnerd_small,
     compare against the demo numbers in experiments/bm25_ebnerd_2026-08-10/
   - Document as an ADR-002 addendum (resolves the flagged open question, don't
     spin up a new ADR for this alone)

2. Q4 Evaluation Harness — a NEW per-impression scoring path
   - Design in plan mode first. recall@K (already built) answers "is the true
     article in my top-K retrieved from the whole corpus." Q4 asks something
     different: for every candidate already listed in an impression, produce a
     relevance score, rank it against the impression's other candidates, and
     score against the click label.
   - Required: AUC, MRR, nDCG@5, nDCG@10 (Q4.1); diversity/novelty/coverage
     (Q4.2); bootstrap 95% CI (Q4.4); warm/cold slicing (Q4.3, reuse the
     existing cohort). Extend src/evaluation/metrics.py's CI pattern, don't
     duplicate it.
   - Build it generic — a scoring function over (user_query, candidate_article)
     pairs — so BM25 slots in now and embeddings slot in later without
     rewriting the harness. Phase 4 is the second consumer, not the first.
   - Run against BM25 on both datasets (ebnerd_small this time, not just demo).
     Write an ADR for any real design decision inside the harness (nDCG
     tie-breaking, how novelty/coverage are defined for this project).

Deliverable: ADR-002 addendum; new ADR for the harness if warranted;
src/evaluation/ranking_metrics.py (or similar), tested; harness run against
BM25 on MINDsmall-dev + ebnerd_small; PROJECT_STATE.md and ARCHITECTURE.md
updated.

Start with: plan mode — pull real impression-candidate-count and click-rate
stats (you already have some from Phase 3) to ground the nDCG/AUC design in
actual data shape before picking an implementation, same as Phase 3's plan did.


please be careful not to take any short term workaround
```

## Prompt 2

```
yes, "Prompt & Session Logging" is intentional, I gave you that text last turn for Q7.4 compliance also made nanother addition abotu resource avialbalitly go hceck it out
```

(Sent in response to Claude flagging an unexpected CLAUDE.md diff — the
"Prompt & Session Logging" and "Resource Availability" sections — that
Claude had not authored and wanted confirmation on before proceeding.)

---

## AI-generated vs. human-written

**Human-written this session:**
- `CLAUDE.md`'s "Prompt & Session Logging" and "Resource Availability"
  sections (added directly by the engineer, outside the Claude Code
  session, mid-way through — confirmed via Prompt 2 above).

**AI-generated this session** (Claude Code; reviewed and approved by the
engineer via the plan-mode approval gate before implementation began):
- `decisions/ADR-002-unified-data-schema.md` (addendum section)
- `decisions/ADR-005-query-construction.md` (Conditions-for-Revisiting edit)
- `decisions/ADR-007-ranking-evaluation-design.md` (new)
- `src/pipeline/orchestrator.py` (`include_ebnerd_small` flag)
- `src/retrieval/index.py` (`id_to_col` field)
- `src/retrieval/retrieve.py` (refactor onto extracted `score_all`)
- `src/retrieval/score.py` (new)
- `src/evaluation/bootstrap.py` (new)
- `src/evaluation/metrics.py` (refactor onto `bootstrap_ratio_ci`)
- `src/evaluation/ranking_metrics.py` (new)
- `scripts/run_bm25_experiment.py` (`--bundle` flag)
- `scripts/run_ranking_eval.py` (new)
- `tests/unit/{test_score,test_bootstrap,test_metrics,test_ranking_metrics}.py` (new)
- `tests/integration/test_ranking_eval_pipeline.py` (new)
- `tests/integration/test_schema_conformance.py` (ebnerd/small cases added)
- `ARCHITECTURE.md`, `PROJECT_STATE.md` (Evaluation section, changelog, status tables, session notes)
- This log file
