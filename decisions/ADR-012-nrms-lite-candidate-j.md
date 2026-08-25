# ADR-012 — NRMS-Lite: Trainable Title + History Encoders (Candidate J)

**Date:** 2026-08-22 (decision) / 2026-08-24 (real results)
**Status:** Decided — real, CI-clear WIN against the deployed baseline (0.6391 vs. 0.634, GloVe-initialized). First real win this project's MIND candidate search has produced from a genuinely new architecture. MINDlarge re-verification and any Codabench submission decision remain open for the engineer (see 2026-08-24 addendum).

See `decisions/ADR-011-neural-attention-reranker.md`'s own addendum for
the short cross-reference from the neural-candidate-search line's anchor
ADR; this document carries the full context, environment history
(Kaggle → Ada), and real results.

---

## Candidate J — NRMS-lite (title encoder + click-history encoder)

**Context.** Candidates H1 (linear), H2 (tree), I / I-long / I-pop
(attention over frozen sentence embeddings) all stalled at real
MINDsmall-dev AUC ~0.61-0.62, flat against the deployed embedding-
similarity baseline (local Q4 AUC 0.634; Codabench score 0.6195,
submission 886468). Before committing further engineering time, published
benchmark numbers for the standard architectures built for this exact
dataset were checked: NAML/LSTUR/NRMS score ~0.64-0.66 AUC on MINDsmall in
the literature (DIGAT paper's benchmark table: NAML 0.6612, LSTUR 0.6587,
NRMS 0.6563; a second source reports NAML 0.655, LSTUR 0.6438, NRMS
0.6483). A combined variant (Co-NAML-LSTUR) reports +1.55-2.45% AUC over
single models. This means a 0.70 target floated as "achievable" is above
what even careful, published implementations of the field's standard
models reach on this dataset -- treated here as a claim to double-check
(what split/metric is actually being reported), not an assumed bar.

**Root-cause hypothesis for the H1/H2/I/I-long/I-pop plateau.** All five
prior attempts scored on top of frozen, pre-computed sentence embeddings
with a shallow head (linear, tree, or attention-based readout). None
included the two components every NAML/LSTUR/NRMS model shares: (1) a
trainable encoder over the article title text itself, jointly optimized
for the click task, and (2) an explicit sequence model over the user's
click history (not a static mean-pooled feature). This is offered as a
real, falsifiable explanation for the plateau -- Candidate J tests it
directly by being architecturally different from all five prior attempts
in exactly these two respects, not just "one more model class."

**Decision.** Implement a lightweight, from-scratch NRMS reproduction (Wu
et al. 2019) -- self-attention title encoder + additive-attention pooling,
self-attention history encoder + additive-attention pooling, dot-product
scoring, trained end-to-end with NRMS's own negative-sampling recipe
(K=4, softmax cross-entropy). Word embeddings are randomly initialized
(no pretrained GloVe was staged in this project's Kaggle datasets;
flagged as a known handicap, not silently absorbed -- see Trade-offs).
Evaluated with the exact same per-impression-averaged AUC/MRR/nDCG@5/
nDCG@10 methodology as the official `evaluate.py`, so it is directly
comparable to every number already recorded for H1/H2/I and the baseline.

**Evidence (real, from this project).**

| Candidate | Real MINDsmall-dev AUC | Notes |
|---|---|---|
| Baseline (embed similarity, deployed) | 0.634 local (CI 0.6319-0.6361) / 0.6195 Codabench | submission 886468 |
| H1 (linear) | 0.6319 (CI 0.6297-0.6340) | CI-clear loss (ADR-011) |
| H2 (tree) | 0.5985 (CI 0.5961-0.6006) | CI-clear loss, large (ADR-011) |
| I (attention, 5 epoch) | 0.6233 | clean negative |
| I-long (attention, 30 epoch) | 0.6214 | clean negative; internal holdout AUC misleadingly kept climbing (0.6067→0.6372) while real dev AUC fell |
| I-pop (attention + popularity, 30 epoch) | 0.5133 | confounded — unnormalized `log_popularity` feature, not a clean comparison |
| J, no GloVe (Ada, 2026-08-23) | 0.6242 (CI 0.6220-0.6262) | real run, but GloVe download failed both times this attempt — random init only, see 2026-08-24 addendum |
| **J, with GloVe (Ada, 2026-08-24)** | **0.6391 (CI 0.6370-0.6412)** | **real, CI-clear WIN vs. baseline** — see 2026-08-24 addendum |
| Literature band, standard NAML/LSTUR/NRMS | 0.64–0.66 | Wu et al. via DIGAT benchmark table; not this project's own number |

**Interpretation.** See the 2026-08-24 addendum below for the full,
real-data interpretation — the run environment moved from Kaggle to Ada
(2026-08-22 addendum) before any real Kaggle run happened, so every number
above is a real Ada result, not a Kaggle one. Short version: with GloVe,
J is a real, CI-clear win against the deployed baseline (first one this
project's candidate search has produced from a new architecture, not a
combiner) and lands just short of the literature band's low end (0.6391
vs. 0.64, within noise of clearing it). Without GloVe, J is flat against
I/I-long — confirming the missing-architecture-components hypothesis
alone, absent pretrained embeddings, does not resolve the plateau; GloVe
was the actual lever, exactly as this document's own pre-registered
priority-order guess anticipated.

**Trade-offs accepted going in, not discovered after the fact.**
- No pretrained embeddings (time risk of downloading/staging a multi-GB
  file this close to the deadline was judged not worth it).
- Smaller embedding dim (128 vs. paper's 300) and fewer epochs (3 default)
  to fit a single Kaggle GPU session within a few hours.
- Vocabulary built from train+dev title text only (no label information
  used — token coverage, not leakage, same posture as BM25's corpus-wide
  index).

**Conditions to revisit.** If J lands meaningfully below 0.64 even after
verifying no implementation bug, the honest conclusion is that the
missing-architecture-components hypothesis was only partially right, and
the remaining gap is compute/data (pretrained embeddings, more epochs,
more careful hyperparameter search) rather than a one-shot architecture
fix — worth stating plainly rather than continuing to chase it against a
hard deadline.

---

## 2026-08-22 Addendum: Reconsidering the Kaggle Trade-offs (Ada Available)

**Context.** The original decision above (and the trade-offs section)
explicitly justified three choices by Kaggle's time/network constraints —
not by architecture reasoning. The engineer subsequently gained access to
Ada, IIIT-H's SLURM cluster (`research` account, `low`/`medium` QoS,
GTX 1080 Ti / RTX 2080 Ti GPUs). Per CLAUDE.md's Decision Reversal clause,
constraint-driven trade-offs are worth re-examining once the constraint
that motivated them changes — this section does that, without deleting or
overwriting the original reasoning above (which stays correct as a record
of what was decided under the original constraint).

**One correction to the initial framing, checked before proceeding:** Ada
was initially described as having "no time limit." Per the real user guide
(`hpc.iiit.ac.in/wiki`, fetched and read directly, not assumed), the
`research` account under `low`/`medium` QoS actually has a real **4-day
wall-clock cap per job** — not infinite. Not a practical blocker for this
job (its real per-epoch cost is expected to be minutes, not days), but the
premise itself doesn't hold and shouldn't be treated as though it does.

**Three trade-offs reconsidered:**

1. **No pretrained word embeddings** (was: "none staged in this project's
   Kaggle datasets, downloading a multi-GB file adds real time risk").
   Ada removes that specific risk. `scripts/mind_nrms_lite_ada_run.py` (and
   the Kaggle script, updated in step) now load GloVe (`glove.6B.300d.txt`)
   if available, initializing (not freezing) the word-embedding layer —
   still degrades gracefully to random init if GloVe isn't staged/
   reachable, so this stays an enhancement, not a new hard dependency.
2. **`embed_dim=128`/`num_heads=8`** (was: "traded down for Kaggle time
   budget"). Bumped to `embed_dim=300` (matches GloVe's dimensionality
   exactly) / `num_heads=15` (20-dim/head, evenly divides 300). Flagged
   honestly: this is an engineering choice for divisibility + GloVe
   alignment, not a verified reproduction of Wu et al.'s own exact head
   configuration — the original draft's "15-16-head" recollection was
   itself hedged as approximate, not checked against the primary paper.
3. **`NRMS_EPOCHS` default of 3** (was: "sized to fit a single Kaggle GPU
   session"). Replaced with real early stopping (patience=3 non-improving
   epochs) against a generous epoch ceiling (30), rather than a fixed low
   count acting as the real governor. Because both scripts checkpoint on
   the REAL MINDsmall-dev AUC every epoch (not an internal train-side
   holdout), running more epochs is safe against the I/I-long overfitting
   trap by construction — the worst case is wasted compute, not a worse
   reported result, since the best checkpoint is always what gets kept and
   reported.

**One thing NOT changed, and why:** `NEG_K=4` (negative sampling count) —
this is a direct match to the NRMS paper's own training recipe, not a
resource-driven trade-off, so there was nothing to reconsider here.

**A methodological point being surfaced deliberately, not smoothed over
under "maximize AUC" pressure:** checkpointing directly on MINDsmall-dev
(rather than an internal train-side holdout, the stricter discipline
Candidate I used) means the reported dev AUC is mildly optimistic — the
epoch selected is the one that happened to do best on the exact set being
reported against. This was already true of the original 3-epoch design;
raising the epoch ceiling increases how many "looks" at dev happen before
selecting the best one, which mildly widens the same effect. Not
disqualifying (every candidate in this project's search reports this way),
but worth remembering when Candidate J's real number comes back, not
something to only notice if the number is disappointing.

**Not changed in this addendum, deliberately deferred:** batch size,
learning rate, and `NEG_K` were left at their original values — the
Kaggle-vs-Ada resource change speaks specifically to *time and pretrained-
data availability*, not to optimization hyperparameters, and changing
those without a specific reason would be scope creep dressed up as
"maximizing AUC," not a defensible engineering decision.

**Where this leaves the evidence table above:** the `[FILL IN]`s for
Candidate J still wait on a real run — this addendum changes what
configuration that real run will use, not the "no estimates, only real
Kaggle/Ada output" rule the original decision already established.

---

## 2026-08-24 Addendum: Real Ada Results — Candidate J Is a Real Win

**Status:** Resolves the `[FILL IN]`s left open above with real, from-this-
project data. First real, CI-clear win this project's MIND candidate
search has produced from a genuinely new architecture (not a combiner
stacking existing scorers, like Candidate G). Whether to pursue a real
Codabench submission is left open for the engineer — this addendum
reports the result, it does not decide that.

### What actually happened getting here (worth recording, not just the number)

Getting a real run took two attempts, both real data points:

1. **First Ada run (job 2675355, 2026-08-23):** GloVe failed to download
   — `ContentTooShortError`, connection dropped at 160MB/822MB —
   because `_ensure_glove`'s original implementation used
   `urllib.request.urlretrieve`, a single-shot call with no retry/resume.
   The script's graceful-degradation design worked exactly as intended
   (random init, not a crash): **0.6242 (95% CI 0.6220-0.6262)**, 236,344
   train examples, best at epoch 8/11 (early-stopped, patience=3).
2. **Root-cause fix, not a retry-and-hope:** `_ensure_glove` rewritten to
   shell out to `wget -c --tries=10 --waitretry=15 --retry-connrefused`
   (resumable, auto-retrying) instead of `urlretrieve`. This is the actual
   fix for the observed failure mode — a dropped connection now resumes
   from where it stopped instead of failing outright.
3. **Second Ada run (job 2675573, 2026-08-24):** GloVe download succeeded
   this time (9h15m — `downloads.cs.stanford.edu` is genuinely,
   persistently throttled from this cluster's network path, ~15-50KB/s
   sustained, not a one-off blip; not fixable from this side, the retry
   logic just outlasted it). **20,638/21,319 vocab words (96.8%)** matched
   real GloVe vectors. Result: **0.6391 (95% CI 0.6370-0.6412)**, best at
   epoch 3/6 (early-stopped, patience=3).

A real, separate infrastructure finding surfaced along the way, worth
recording since it will recur: **`/ssd_scratch` is local to each Ada
compute node, not shared across the cluster.** Data staged on one node
(via an interactive `srun` session) was invisible to a later `sbatch` job
that landed on a different node — `mind_nrms_lite_ada_run.py` now
auto-downloads the MIND zips itself (via `src/pipeline/download.py`,
idempotent) rather than assuming pre-staged data persists, for the same
reason GloVe already worked this way.

### Results

| Run | GloVe | Train examples | Best epoch | Overall AUC | 95% CI |
|---|---|---|---|---|---|
| J, no GloVe (job 2675355) | No (download failed) | 236,344 | 8/11 | 0.6242 | 0.6220-0.6262 |
| **J, with GloVe (job 2675573)** | **Yes, 96.8% coverage** | 236,344 | 3/6 | **0.6391** | **0.6370-0.6412** |

Full Q4 metric suite for the adopted (GloVe) run's best epoch, verified
directly against `results.json` (not just the console paste):

| Metric | Value | 95% CI |
|---|---|---|
| AUC | 0.6391 | 0.6370-0.6412 |
| MRR | 0.3434 | 0.3409-0.3459 |
| nDCG@5 | 0.3241 | 0.3213-0.3267 |
| nDCG@10 | 0.3898 | 0.3873-0.3923 |

All from `n_impressions=73,152`, `n_users=50,000`, `n_skipped=0` — the
full real MINDsmall-dev split, no subsetting.

Full config/results: `experiments/candidate_j_nrms_lite_ada_2026-08-24/{config,results}.json`
(the no-GloVe run's raw JSON was overwritten on Ada by the second run
sharing the same `--results-dir` — a real housekeeping miss, not repeated
next time — but its complete console output is preserved verbatim in this
session's `knowledge/ai-usage-log/` entry, so no real data was lost).

### Interpretation

**J with GloVe is a real, CI-clear win against the deployed baseline.**
The baseline's own established CI (0.6319-0.6361, per ADR-011's
Comparison Summary) and J's CI (0.6370-0.6412) do not overlap — J's lower
bound (0.6370) sits above the baseline's upper bound (0.6361). This is
the first time in this project's entire MIND candidate search (A through
J, across two sessions) that a genuinely new architecture — not a
combiner stacking already-deployed scorers, which is what Candidate G
was — has produced a real, CI-clear win. It also clearly beats every
other neural candidate tried (I: 0.6233, I-long: 0.6214, J-no-GloVe:
0.6242) by a real margin, and lands just 0.0009 below the literature
band's low end (0.64) — within noise of clearing it, not clearly short of
it, unlike every prior candidate in this search.

**The no-GloVe vs. GloVe comparison directly confirms the priority-order
guess this document made before either run happened.** Without GloVe, J
(0.6242) is statistically indistinguishable from I (0.6233) — the
architecture change alone (trainable title + history encoders replacing
frozen embeddings) did not resolve the plateau. With GloVe, the same
architecture jumps to 0.6391 — a real, attributable +0.015 AUC change
from one specific, isolated intervention (holding architecture, epochs-
run pattern, and every other hyperparameter constant). This is real
evidence, not inference, that pretrained embeddings were the dominant
missing ingredient, not the architecture components alone — exactly the
(a)-ranked suspect this document flagged as most likely before the data
came in.

**One caveat held onto deliberately now that the result is good, not just
when the 2026-08-22 addendum flagged it while the outcome was still
unknown:** both runs checkpoint on the REAL MINDsmall-dev AUC every
epoch, not an internal train-side holdout (Candidate I's stricter
discipline). The reported number is the epoch that happened to do best on
the exact set being reported against, which is mildly optimistic — a real
methodological point, not disqualifying (every candidate in this search
reports this way, and the CI-clear margin over baseline is large enough
that this caveat is very unlikely to explain away the win entirely), but
worth stating plainly rather than only when convenient.

### Conditions to revisit (updated from the original, now resolved)

The original document's condition — "if J lands meaningfully below 0.64,
the missing-architecture hypothesis was only partially right" — did not
trigger; J landed essentially at the literature band's boundary, with
GloVe identified as the specific lever that mattered. Remaining open
questions for the engineer, not resolved by this addendum:

- **MINDlarge re-verification** before any real Codabench submission —
  standing project practice (ADR-010) for every candidate considered for
  deployment, not yet done for J.
- **Whether to pursue a real second/replacement MIND Codabench
  submission** given this local win — an explicit engineer decision, per
  this project's established pattern (Candidate G's MINDlarge
  verification and submission were both explicit engineer calls, not
  automatic).
- The dev-checkpoint-selection optimism caveat above means a MINDlarge
  re-verification is also the cleanest way to confirm this isn't
  compressing at larger scale the way Candidate G's local win did at real
  blind-test time — the same category of check this project has applied
  consistently, not a new bar invented for this result specifically.

### Related

- `experiments/candidate_j_nrms_lite_ada_2026-08-24/{config,results}.json`
- `scripts/mind_nrms_lite_ada_run.py`, `scripts/mind_nrms_lite_ada.sbatch`
- `src/retrieval/nrms.py`, `src/retrieval/nrms_training.py`,
  `tests/unit/test_nrms.py`, `tests/unit/test_nrms_training.py`

---

## 2026-08-25 Addendum: MINDlarge-Dev Re-Verification — the Win Widens, Not Compresses

**Status:** Resolves the prior addendum's open "MINDlarge re-verification"
condition. Real result: **0.6579 (95% CI 0.6569-0.6588) vs. the real
MINDlarge-dev baseline of 0.6335** — a **+0.0244 AUC** margin, larger than
the MINDsmall-dev win (+0.0051), not smaller. This is the opposite of
every other local-win-checked-at-scale result in this project's history.
Whether to generate real MINDlarge_test predictions and pursue a
Codabench resubmission remains open for the engineer — not decided here.

### A real bug caught before it corrupted this write-up

The script's console output printed "vs. deployed baseline (local Q4
AUC): 0.6340" — but that's a hardcoded constant left over from the
MINDsmall runs, not the right number for a `--bundle large` run. The real
MINDlarge-dev baseline, independently verified elsewhere in this project
against the official `evaluate.py` (within 0.0002), is **0.6335** — close
to 0.634 by coincidence, not the same measurement. Caught by manually
cross-checking against `PROJECT_STATE.md` before writing this addendum,
not by the script itself — fixed at the root
(`scripts/mind_nrms_lite_ada_run.py`, `BASELINE_LOCAL_Q4_AUC_BY_BUNDLE`)
so a future run gets this right automatically instead of depending on
whoever reads the output to catch it.

### What was done

Same model, same training/eval code as the MINDsmall run (`--bundle
large` added to `mind_nrms_lite_ada_run.py` rather than a new script —
only which zips get downloaded/parsed differs). Real infrastructure
issues hit and fixed along the way, each from a real error, not
anticipated in advance:

- `download_mind_bundle` (the project's shared MIND downloader, used by
  `make data` too) hit the same `ContentTooShortError` the GloVe download
  hit at MINDsmall scale — `MINDlarge_train.zip` (531MB) dropped mid-
  transfer, and `urlretrieve` has no retry/resume. Fixed at the root with
  a pure-`urllib` Range-header retry/resume loop (not a `wget` subprocess
  like the GloVe fix — this function also runs in `make data` on
  whichever machine has the repo cloned, and `wget` isn't preinstalled on
  macOS, confirmed on this project's own dev machine). New
  `tests/unit/test_download.py`, 6 tests, including the resume-from-
  correct-byte-offset case specifically, not just "it eventually works."
- The engineer's Ada account was upgraded `low` → `medium` QoS mid-
  session (confirmed via `sacctmgr`) — broke both `sbatch` scripts
  outright (`Invalid qos specification`) until fixed.
- An initial version pinned the MINDlarge job to `gnode007` specifically,
  to reuse a GloVe file already cached there from the MINDsmall run
  (avoiding a second ~9h throttled download). **This backfired in
  practice, not just in theory:** `gnode007` stayed busy long enough that
  the queue wait itself exceeded what a fresh GloVe download would have
  cost on any free node. Reverted to unpinned; GloVe re-downloaded from
  scratch on whichever node the job landed on (successfully, via the
  already-proven resumable download).
- The benchmark step (`--max-train-examples 3000`, matching this
  project's own benchmark-before-commit discipline) confirmed the real
  MINDlarge feature store builds cleanly with no memory errors (101,527
  train articles, 72,023 dev articles — the dev count matches this
  project's previously recorded MINDlarge-dev figure exactly) before the
  full job was ever submitted, but an SSH disconnection killed that
  interactive session before it produced a timing number. Not repeated:
  the real full run went through `sbatch` instead, immune to client-side
  connection drops by design.

### Results

Real MINDlarge-dev, best epoch (1 of 4 run, early-stopped patience=3),
`n_impressions=376,471` (the full real split), `n_users=255,990`,
`n_skipped=0`:

| Metric | Value | 95% CI |
|---|---|---|
| AUC | 0.6579 | 0.6569-0.6588 |
| MRR | 0.3581 | 0.3570-0.3592 |
| nDCG@5 | 0.3411 | 0.3399-0.3422 |
| nDCG@10 | 0.4064 | 0.4053-0.4074 |

GloVe: 25,170/26,293 vocab words (95.7% coverage). 3,383,656 real training
examples (14.3x MINDsmall's 236,344 — consistent with the ~14.2x train-
impression scale ratio this project already had on record). Real per-
epoch timing: ~117 min/epoch, consistently across all 4 epochs run —
close to the ~110 min/epoch projected before the run (within 7%), a real
validation of that projection method, not just a lucky guess. Total job
wall time roughly 9.25h (GloVe download) + 7.83h (4 epochs of train+eval)
≈ 17h, comfortably inside the 3-day budget.

Full config/results:
`experiments/candidate_j_nrms_lite_ada_mindlarge_2026-08-25/{config,results}.json`.

### Interpretation

**The win is real and larger at MINDlarge scale, not smaller.** Every
prior instance of "check a local win at bigger/real scale" in this
project went the other direction: Candidate G's MINDsmall-dev win of
+0.0027 shrank to +0.0019 at MINDlarge-dev and to essentially flat
(-0.0003) at the real Codabench blind test; the EB-NeRD contrastive-
vector result did the same, thinning from a CI-clear local gap to a much
thinner real-test edge. Candidate J's MINDsmall-dev margin (+0.0051)
instead widened nearly 5x to +0.0244 at MINDlarge-dev. This is real,
measured evidence — not a hoped-for pattern — that this result behaves
differently from every prior candidate this project has checked at scale.

**A plausible explanation, flagged as inference, not proven:** Candidate
G was a combiner over already-fixed, precomputed scalar features — more
data doesn't give it anything new to learn, since the ceiling was the
feature set itself (ADR-010's own finding). Candidate J has a trainable
title encoder and a trainable history encoder with real capacity to use
more data — MINDlarge's ~14x larger real training set is exactly the kind
of resource a model like this can actually benefit from in a way a fixed-
feature combiner structurally cannot. This is a real, falsifiable
explanation consistent with the data, not independently verified by a
controlled ablation here.

**The dev-checkpoint-selection caveat, carried forward again:** this
result still checkpoints directly on the real dev split, same caveat as
before. Given the margin here (+0.0244) is roughly 5x the MINDsmall
result's own margin, this caveat is even less likely to explain the win
away than it already was — but it's the same category of caveat, stated
for the same reason: not smoothing it over because the news is good.

### Conditions to revisit (updated — MINDlarge condition now resolved)

- ~~MINDlarge re-verification before any real Codabench submission~~
  **Resolved by this addendum:** real, CI-clear win, margin widened not
  compressed.
- **Whether to generate real MINDlarge_test predictions and pursue a
  Codabench resubmission — still open, for the engineer.** Given this
  result is the first in this project's history to widen rather than
  compress at scale, the honest expectation going into a real blind-test
  submission is more optimistic than it was for Candidate G or the
  EB-NeRD result — but "more optimistic" is not "certain," and this
  project's own standing discipline is to find out for real rather than
  assume.

### Related

- `experiments/candidate_j_nrms_lite_ada_mindlarge_2026-08-25/{config,results}.json`
- `scripts/mind_nrms_lite_ada_run.py` (`--bundle large`),
  `scripts/mind_nrms_lite_ada_large.sbatch`
- `src/pipeline/download.py` (retry/resume fix), `tests/unit/test_download.py`
