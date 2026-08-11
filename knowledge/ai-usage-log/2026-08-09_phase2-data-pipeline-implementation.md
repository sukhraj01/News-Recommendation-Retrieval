# AI Usage Log — 2026-08-09 — Phase 2: Data Pipeline Implementation

Per CLAUDE.md's "Prompt & Session Logging" section. Logged retroactively —
the engineer flagged mid-project that this session (Phase 2) had been
missed when the ai-usage-log convention was set up — but reconstructed
verbatim and complete from the actual session transcript, not summarized.
Dated 2026-08-09 to match this session's actual place in the project
timeline (see PROJECT_STATE.md's Phase 2 completion date and the
`Fix: Re-pin pyarrow` / `Phase 2: Implement...` / `Phase 2 Complete` commits),
which is earlier than the other ai-usage-log entries already on disk
(2026-08-10, 2026-08-10, 2026-08-11) even though this file was written after
them.

Tool: Claude Code (Sonnet 5).

---

## Prompt 1 (session start)

```
Current Objective: Phase 2 — Data Pipeline Implementation (Days 4-5, ~8 hours)

Foundation: ADR-001 (temporal split) + ADR-002 (unified schema) are locked.

Goal: Build a reproducible, single-command pipeline that:
1. Downloads raw data (if not present locally)
2. Parses MIND + EB-NeRD into unified schema
3. Applies temporal split (official train/val/test boundaries per ADR-001)
4. Builds feature store (articles table, impressions table, user_history table per ADR-002)
5. Passes leakage tests (no test data in training, no future information visible at serving time)

Deliverable: One command (make data) rebuilds the entire feature store from raw files.

Read: ADR-001, ADR-002, ARCHITECTURE.md, PROJECT_STATE.md

Then: Start implementation.
```

## Prompt 2 (via AskUserQuestion — "How should I resolve the missing pyarrow dependency needed to parse EB-NeRD's parquet files?")

```
Re-add pyarrow with a loosened constraint (Recommended)
```

(Claude had found `pyarrow` silently dropped from `pyproject.toml`,
uncommitted, before starting implementation, and asked the engineer to
choose between re-pinning it, pinning the whole project to Python 3.11
instead, or handling it themselves.)

## Prompt 3 (via AskUserQuestion — "What on-disk format should the feature store use?")

```
Parquet via pyarrow (Recommended)
```

## Prompt 4 (plan-mode approval)

Plan approved via the `ExitPlanMode` gate — no additional free-text
instructions attached; the plan file at that point is the full Phase 2
design (module layout, ID scheme, split-preservation strategy, schema
validation approach, test plan, implementation order).

## Prompt 5 (via AskUserQuestion — corrected `user_id` ID-prefixing scheme mid-implementation)

```
Decision: User ID should be dataset-only prefixed (mind:U1, ebnerd:U1), NOT split-qualified.

Reason: Users are real-world identities shared across train/dev/test splits.
Split-qualifying them would break joinability and silently disable Q4's
warm/cold-user analysis (can't tell if a dev-window user was in training).

Schema correction:
- user_id: dataset-only prefix (mind:U1, ebnerd:U1)
- impression_id: split-qualified (mind:train:1, mind:dev:1)
- article_id: dataset-only prefix (mind:N123, ebnerd:9738663)

This preserves cross-split user joinability while maintaining impression uniqueness.

Proceed with Phase 2 using this corrected schema.
```

(Claude had flagged, mid-implementation, that the plan's original ID scheme
was wrong about `user_id` — verified empirically that MIND train/dev share
5,943/50,000 overlapping users and EB-NeRD train/validation share
1,217/~1,590 — and asked the engineer to confirm the fix before proceeding.)

## Prompt 6

```
Commit in 3 parts. Here's why:

Pyarrow fix is a prerequisite/environment fix — separate from implementation
Phase 2 implementation is the core work
Project state closure marks Phase 2 complete and points to Phase 3

This makes the git history legible: someone reading it can see "oh, we had to fix dependencies first, then built the pipeline."

Commit 1: Fix Dependencies
[... full commit message for `Fix: Re-pin pyarrow to ^22.0.0 (was silently dropped)` ...]

Commit 2: Phase 2 Implementation
[... full commit message for `Phase 2: Implement reproducible data pipeline (download → parse → split → feature store)` ...]

Commit 3: Update Project State
[... full commit message for `Phase 2 Complete: Data pipeline verified and reproducible` ...]

Then:

git log --oneline -10

You should see a clean story:

abc1234 Phase 2 Complete: Data pipeline verified and reproducible
def5678 Phase 2: Implement reproducible data pipeline
ghi9012 Fix: Re-pin pyarrow to ^22.0.0 (was silently dropped)
jkl3456 Phase 1B Complete: Architecture decisions
...

Do these three commits now. Then update PROJECT_STATE.md to point to Phase 3, which is next.
```

(Full proposed commit messages omitted here for brevity — verbatim as
originally typed by the engineer, in full, in the actual session transcript
and in `git log`; two factual corrections were made before committing:
"byte-identical" → "value-identical" reproducibility claim, and the
script filename `build_pipeline.py` → the actual `build_feature_store.py`.)

## Prompt 7

```
we have to store the prompts i made during our chat i hvae added the point in the claude.md file as well and htere is the knowledge folder. i missed it out during u convo nowihave addd it so please maage he time as its part of the name of the file and dont name it later than the alread prent as you are the phase 2 chat
```

---

## AI-generated vs. human-written

**Human-written this session:**
- CLAUDE.md's "Prompt & Session Logging" section (added by the engineer
  directly, not through this Claude Code session — Claude never edited
  CLAUDE.md this session; its presence was read as pre-existing context at
  session start)
- All prompt text above (Prompts 1, 6, 7 typed directly; Prompts 2, 3, 5
  are the engineer's verbatim selections/decisions via the `AskUserQuestion`
  tool)

**AI-generated this session** (Claude Code; reviewed and approved by the
engineer via the plan-mode approval gate before implementation began, plus
two additional explicit approval points — the pyarrow fix and the mid-build
`user_id` correction):
- `pyproject.toml` (`pyarrow` re-pinned `^12.0.0` → `^22.0.0`), `poetry.lock` (regenerated)
- `src/utils/{config,ids,io}.py` (new)
- `src/pipeline/{schema,validators,download,orchestrator}.py` (new — `orchestrator.py` has since been extended by later sessions, e.g. `include_ebnerd_small`; that later work is not this session's)
- `src/datasets/{mind,ebnerd}.py` (new)
- `scripts/{generate_test_fixtures,build_feature_store}.py` (new)
- `tests/unit/{test_mind_loader,test_ebnerd_loader,test_schema_validators}.py` (new)
- `tests/integration/{conftest,test_leakage,test_pipeline_end_to_end,test_schema_conformance}.py` (new — `test_pipeline_end_to_end.py`/`test_schema_conformance.py` have since been extended by later sessions with `ebnerd_small` cases; that later work is not this session's)
- `tests/reproducibility/test_determinism.py` (new)
- `Makefile` (`data`, `test-reproducibility`, `clean-data` targets added)
- `ARCHITECTURE.md` (Data Pipeline / Feature Store sections filled in from placeholders, changelog entry)
- `PROJECT_STATE.md` (status table, risks, Open Engineering Questions, session notes, Phase 3 pointer)
- This log file
