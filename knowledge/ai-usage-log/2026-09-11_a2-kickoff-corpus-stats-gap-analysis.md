# AI Usage Log — 2026-09-11 — Assignment 2 Kickoff: Corpus Stats + Gap Analysis

## Prompts (verbatim, in order)

### Prompt 1

> Current Objective: Assignment 2 has arrived . It builds directly on Assignment 1's existing pipeline. Before any new engineering, close one small pending item, then verify a gap analysis against the real codebase.
>
> Context you need: Read PROJECT_STATE.md and the ADRs (especially ADR-011/012 for MIND's NRMS-lite/Candidate J, ADR-013 for EB-NeRD's GBDT/Candidate K, ADR-014 for yesterday's latency/throughput profiling) for full A1 history — don't take my summary below as authoritative, verify it against what's actually committed. I'll place A2.pdf at the repo root — read it directly for the exact wording; the summary below is my own reading of it, not a substitute.
>
> 0. Do this first, standalone, quick: Compute and report real corpus statistics for both datasets — vocabulary size (pre/post stopword removal), total token count, average tokens per document, per-split corpus size (articles/impressions/users), BM25 index size. This was asked cold in a written quiz with no prep; get it into a small, memorable reference table (5-10 anchor numbers, not a document to study) so it's never a surprise again.
>
> A2's actual asks, and where A1 already covers them (verify each against real code, don't trust this list blindly):
>
> Q1 (click-history/session features): recency-weighted history, category affinity, freshness already exist (Candidate K's 65 features; Candidate J's native history encoding). Gap: session-level features (within-session patterns, position bias) aren't built — and if dwell/read-time is added, it needs the same leakage scrutiny ADR-009 already gave EB-NeRD's other lifetime-aggregate fields, since read-time is exactly that kind of serving-time-unavailable signal.
> Q2 (two-stage re-ranker): functionally exists (Candidate K = GBDT, Candidate J = neural), already on both leaderboards. Verify it's framed correctly as retrieve-then-rank in the report, and pull MRR/nDCG@5/nDCG@10 for both candidates, not just AUC — currently AUC is the only number reported prominently for J/K.
> Q3 (reproduce the official baseline, then beat it) — the real gap. Everything built so far is a custom architecture inspired by the literature, not a literal reproduction of ebnerd-benchmark's NRMS or the official MIND baseline. A2 wants that reproduced first as the trustworthy reference point. Do this before anything else below — the ablation and CI in Q3 need a real baseline to be measured against, not our own candidate standing in for one.
> Q4 (serving/scale): largely done via ADR-014 — carry it forward, don't redo. Check specifically whether p99 (not just mean) latency was captured, add an explicit index-memory-footprint number, and reframe existing throughput numbers as cost/QPS at a target SLA (p99 < 100ms) — that specific framing doesn't exist yet even though the underlying numbers do.
> Q5 (extended eval): full metric suite + bootstrap CIs already exist for baseline retrieval methods, needs extending to the two-stage pipeline. Warm/cold slicing exists; head-vs-tail (popular vs. unpopular articles) is a new slice, not built yet.
> Q6 (design note): not written as its own 6-page PDF yet — mostly assembly from A1's note + ADR-014 once Q3's gap is closed.
> Q7-Q9 (deliverables/git/anti-gaming): already close to compliant. Double check .gitignore covers *.pt/*.ckpt now that NRMS produces real checkpoints, per A2's explicit callout.
>
> Two framing requirements from the professor, not just the written brief:
>
> He wants A/B-testing language specifically: name control (reproduced baseline) vs. treatment (improved re-ranker) explicitly, use the paired bootstrap 95% CI as the significance test (that's what it already functionally is), and name guardrail metrics (diversity/novelty/coverage, and now latency) that aren't allowed to regress even if the primary metric wins. This is a reporting/framing addition, not new engineering — weave it into however Q3/Q6 get written.
> Team goal: rank < 10 on both leaderboards. The assignment itself states grading is never on rank — this is the team's own added ambition on top of the graded requirements, not a substitute for them. Don't chase it at the expense of the actual deliverables above; only push further once Q1-Q6 are solid.

### Prompt 2

> Decisions locked in:
>
> Fix the environment as you proposed — go ahead.
> Baseline reproduction: Option A (official TF/Keras code for both). Timebox it — one focused Ada session. If it's not training cleanly by then, stop, fall back to Option B, and document why in the ADR rather than quietly extending the deadline on it.
> Treatment: EB-NeRD = official NRMS + freshness feature, exactly as you proposed, with K unchanged as the Q2 re-ranker. MIND = official NRMS + pretrained GloVe embeddings instead of random init (the lever flagged but never taken when J was first built) — ablation is GloVe-init vs. random-init, everything else held fixed. If this beats reproduced NRMS, it replaces J as the MIND leaderboard submission.
> Q2: keep leaderboard submissions ranking each impression's real candidate list, add the true-retrieval-ceiling evaluation (top-K from A1's retriever, ~2.7%) as a separate, honestly-reported result.

## What was AI-generated vs. human-written/edited

- **AI-generated, no human edits yet:** `scripts/compute_corpus_stats.py`, `docs/corpus_stats.md`, and this log's non-verbatim sections. All numbers were computed this session using the project's own `tokenize`/`build_index`. They match the 2026-08-31 artifact exactly on docs/vocab/tokens, and ADR-006's 23.5 MB matrix. The run used a scratch Python 3.11 venv pinned to `poetry.lock` versions (numpy 1.26.4, scipy 1.17.1, pandas 2.3.3, pyarrow 22.0.0, rank-bm25 0.2.2), because the project's poetry 3.11 venv no longer exists on disk. Peak RSS was 1.07 GB.
- **Environment incident, caused by AI:** a `poetry run` probe auto-created an empty `...-py3.14` venv. It was verified empty (created 14:01 today) and removed.
- **Prompt 2 execution (all AI-written, no human edits yet):**
  - Local env: `poetry env use python3.11 && poetry install`. Result: Python 3.11.14,
    numpy 1.26.4, torch 2.13.0, lightgbm 4.7.0. `make test` gave 328 passed, 1 skipped,
    14 deselected.
  - New files: `decisions/ADR-015-a2-official-baseline-reproduction.md`,
    `scripts/a2_official_nrms_train.sbatch`, `scripts/a2_ebnerd_official_nrms.py`,
    `scripts/a2_mind_official_nrms.py`.
  - **Ada jobs started by Claude this session (the only ones it may act on):**
    - 2694345: prep job. FAILED instantly because `/share1` is not mounted on compute nodes.
    - 2694346: 1-CPU diagnostic that confirmed the `/share1` finding. COMPLETED.
    - 2694353: combined official-NRMS control job (`"mind ebnerd"`) on gnode087. Claude
      cancelled it after 4h00m. The data step had failed (missing `mkdir`, now fixed) and
      the env install was stalled on compute-node PyPI (~0.3 MB/s, then no progress for
      ~1 h). A GPU sat idle for ~4 h because the progress watcher was capped at 30 min and
      the session paused. This was an AI process error: the watcher should have had a hard
      idle-GPU cutoff.
  - **Timebox applied (engineer's pre-stated rule, not a new decision):** Option A stopped and
    Option B adopted, recorded in the ADR-015 addendum. Measured PyPI rates: Ada compute node
    ~0.3 MB/s (then stalled), Ada login node ~0.09 MB/s, local Mac ~0.37 MB/s. Every route
    to the ~3 GB TF+CUDA env was multi-hour.
  - **Option B implementation (AI-written, no human edits yet):**
    - `src/retrieval/nrms_official.py`: layer-for-layer port of the official Keras NRMS.
    - `src/retrieval/nrms_official_data.py`: official loader semantics.
    - `tests/unit/test_nrms_official{,_data}.py`: 20 tests, all passing.
    - `scripts/a2_nrms_official_run.py`, `scripts/a2_nrms_official.sbatch`,
      `scripts/a2_ebnerd_prepare_tokens.py`.
    - Local MINDsmall smoke tests: control and treatment both passed, and the two scored
      the identical 300 impressions.
    - More Ada jobs started by Claude: 2694462 (perf_bench venv check on a compute node,
      COMPLETED), **2694501 (MIND control, full run)**, **2694505 (EB-NeRD control)**, and
      **2694506 (EB-NeRD treatment)**. The MIND treatment was held until the control's
      epoch time was known (3-day wall limit), then submitted as **2694529** once epoch 1
      measured 63 min of training and gave a monitor AUC of 0.6692.
      Each full job was submitted only after sha256 checks of its inputs and code matched
      the local copies.
  - **Two AI-introduced bugs, both caught by the local smoke tests before any EB-NeRD GPU
    run.** Details in the ADR-015 addendum.
    - A `.to_numpy()` call on an ndarray in the EB-NeRD early-stop split.
    - A publish-time units error (microseconds treated as nanoseconds) that made the
      freshness treatment a silent no-op. It was exposed by the paired test returning
      exactly +0.0000.
    - Fixed, with a regression test and two runtime guards. The first plausibility bound
      (year 2000) was also wrong: the catalog has 2 genuine pre-2000 articles.
  - Guardrails (diversity/novelty@10, coverage@10) were added to `a2_evaluate_scores.py` and
    validated on real smoke outputs for both datasets; 7 tests pass. Reading MIND epoch
    progress through `py-spy` failed because compute nodes block ptrace (no elevation
    attempted). A background watcher with a 4 h cap tracks jobs 2694501/2694505/2694506.
  - Engineer's answer to the MIND treatment question: **title+abstract input** (ablation:
    title-only control vs title+abstract).
  - Data pushed to Ada `$HOME/a2/raw` and sha256-verified against local A1 copies:
    `ebnerd_small.zip`, `MINDlarge_train.zip`, `MINDlarge_dev.zip`.
  - Raised with the engineer, not acted on: the MIND treatment as specified ("official
    NRMS + GloVe vs random init") is the control itself. Official recommenders NRMS is
    already GloVe-initialised, and ADR-012 already ran that lever on J.
- **Gap analysis (AI research, not code):** verified against committed results.json, profile JSONs, `src/`, ADR-012/013/014, the local MIND paper PDF, the ebnerd-benchmark repo (`args_nrms.py`, `ebnerd_nrms.py`, `pyproject.toml`), the EB-NeRD arXiv paper's Table 3, and recommenders' `nrms_MIND.ipynb`. No pipeline code was changed.
