# AI Usage Log — 2026-08-16 — Design Note Verification & Q7 Checklist Closeout (Ablation, ADR-009, Leakage Test, Test-Fixture Reproducibility Fix)

## Prompts (verbatim, in order)

### Prompt 1

> Current Objective: Verify §4's anti-gaming ablation is real, not just well-written prose, before treating the design note as final.
>
> Confirm scripts/run_leakage_ablation.py exists, run it fresh, and check the output matches the numbers in the note (AUC 0.5430 vs 0.5662, etc.) exactly — not approximately.
> Confirm decisions/ADR-009-*.md exists and actually documents this ablation, not a placeholder.
> Confirm tests/integration/test_leakage.py still passes for EB-NeRD right now (not just "passed at some point").
> If all three check out, tell me explicitly "verified, numbers match" — if anything doesn't match or doesn't exist, tell me exactly what's missing rather than silently fixing it.

### Prompt 2

> Current Objective: Close out the last two unverified items on the Q7 deliverable checklist before considering the repo submission-ready.
>
> Show me .gitignore's actual contents and confirm it excludes *.zip, *.pt, *.ckpt, __pycache__/, data/.
> Run git log --all --pretty=format: --name-only --diff-filter=A | sort -u (or equivalent) and check file sizes for anything that shouldn't be in history — report any large files already committed, don't just check the current working tree.
> Confirm knowledge/ai-usage-log/ has an entry covering this session's design-note verification work (the leaky-feature ablation rerun and leakage test rerun just done) — if it doesn't, add one now, verbatim per the existing log format, not summarized.
> Give me a final one-line status per Q7 item (1-4) — done or not — so I have a clear submission checklist.

### Prompt 3

> Current Objective: Close the one open Q7 item — fresh-clone test failures from untracked tests/fixtures/*.zip files — without violating Q8's "ignore *.zip" policy by adding a gitignore exception.
>
> Confirm scripts/generate_test_fixtures.py actually produces the exact three files the failing tests need (MINDsmall_train_sample.zip, MINDlarge_test_sample.zip, ebnerd_demo_sample.zip) — check by running it fresh and diffing output filenames/paths against what test_ebnerd_loader.py/test_mind_loader.py/test_mind_format.py actually read.
> Wire fixture generation into the reproduce path so a fresh clone doesn't need the zips committed — either as a make target the test suite depends on, or a documented setup step in README's one-command reproduce section. Pick whichever the Makefile's existing structure makes more natural, and tell me which you chose and why.
> Verify the fix for real: in a clean checkout (or git stash/temp clone) with the fixtures absent, run the actual test suite end to end and confirm it now passes without any zip files pre-existing.
> Update the Q7 checklist status for item 1 to Done only after step 3 actually passes — not before.
> Also note the missing submissions/ebnerd_small_validation_embed/ screenshots you flagged — confirm again that's not one of the two required competitions (mind_large_test_embed, ebnerd_testset_embed) and doesn't block submission, then leave it as-is unless I say otherwise.

## What was AI-generated vs. human-written/edited

- **AI-executed, this session, nothing pre-computed or recalled from memory:**
  - `scripts/run_leakage_ablation.py --dataset ebnerd --bundle small` re-run fresh
    (`experiments/ablation_leaky_features_ebnerd_small_2026-08-16/`); every value
    (AUC/MRR/nDCG@5/nDCG@10, both arms, all four CIs) compared digit-for-digit
    against `docs/design_note.md`'s §4 table — exact match, not approximate.
  - `tests/integration/test_leakage.py` run live via `pytest -v`: 7 passed,
    1 skipped (MIND, documented reason), confirming EB-NeRD's leakage-boundary
    test passes right now, not just historically.
  - `decisions/ADR-009-leaky-feature-ablation.md` read in full (338 lines) and
    grepped for the same four metric values, confirming it documents the real
    ablation (Context/Design Space/Rationale/Evidence/Interpretation), not a
    template placeholder.
  - `.gitignore` read and checked against Q8's required pattern list.
  - `git rev-list --objects --all | git cat-file --batch-check=...` run against
    the entire repo history (not the working tree) to size every blob ever
    committed: largest is `poetry.lock` at 340,338 bytes; zero blobs exceed
    1MB anywhere in history; `.git/` itself is 2.6MB total.
  - This log file itself.
- **AI-discovered in Prompt 2, fixed in Prompt 3, verified rather than assumed:**
  `tests/fixtures/*.zip` (`MINDsmall_train_sample.zip`, `MINDlarge_test_sample.zip`,
  `ebnerd_demo_sample.zip`) are not tracked in git — caught by the blanket
  `*.zip` gitignore rule, which only has a `!notebooks/*.zip` exception, not a
  `tests/fixtures/` one. Fixed without a gitignore exception (would have
  weakened Q8's policy) — instead:
  - Confirmed `scripts/generate_test_fixtures.py` (pre-existing, not new)
    already produces exactly the three files needed, at the exact paths
    `test_ebnerd_loader.py`/`test_mind_loader.py`/`test_mind_format.py`/
    `test_ebnerd_format.py` read — verified by moving the existing fixtures
    aside, running the generator fresh, and diffing filenames/sizes, then
    running those four test files (46 tests) against the freshly-generated
    output.
  - Added a `fixtures` target to `Makefile`, made `test`/`test-unit` depend on
    it (`test-integration`/`test-reproducibility` excluded — neither suite
    reads `tests/fixtures/`, confirmed by grep). Chose a Make-target
    dependency over a README-documented manual step because the Makefile
    already has this exact declarative structure and a Make prerequisite
    can't be silently skipped the way a documented step can; a file-timestamp-
    gated multi-output rule (the more "correct" Make idiom) was rejected
    because this repo's `make --version` is GNU Make 3.81 (macOS default),
    which doesn't support the grouped-target (`&:`) syntax that needs.
  - Updated `README.md`'s "Verify Installation" and "Run individual suites"
    sections, which previously called `pytest` directly (bypassing the new
    Make dependency) — changed to call the `make` targets instead, with the
    reasoning stated inline so a reader doesn't have to guess why.
  - Verified end to end for real: deleted all three fixture zips, ran
    `make test` (the full suite, not just the fixture-dependent subset) with
    them absent beforehand — 180 passed, 1 expected skip, 6 deselected, zero
    failures, fixtures regenerated automatically. PROJECT_STATE.md's
    Deliverables Checklist row 1 updated to Complete only after this passed,
    per the prompt's explicit instruction not to mark it done first.
  - Re-checked `submissions/ebnerd_small_validation_embed/` (flagged in
    Prompt 2 as missing its screenshots): confirmed again it holds only
    `prediction.txt`/`truth.txt`, is not one of the two required competition
    submissions (`mind_large_test_embed`, `ebnerd_testset_embed`, both of
    which still have their screenshots present and git-trackable, re-verified
    via `git add -n`), and left it untouched per instruction.
- **Human-originated:** all three prompts above (verbatim), and every
  underlying number/decision this session verified rather than produced (the
  ablation's original run, ADR-009's original authorship, the design note's
  original restructure — all from the 2026-08-14 session, unchanged here).
