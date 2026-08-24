# ADR-011 — Neural Candidate-Aware Attention Re-ranker (Candidate I)

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

**Date:** 2026-08-22
**Status:** Decided (not adopted — real, CI-clear loss at the MINDsmall-dev checkpoint)
**Severity:** High

---

# Engineering Question

ADR-010's Candidate H addendum concluded that two model classes (linear —
Candidate F/H1; nonlinear tree ensemble — Candidate H2) trained over a
fixed set of *precomputed scalar* similarity scores (BM25, mean-pooled
embedding cosine, symbolic overlap, popularity) both lost against the
deployed embedding baseline, and inferred the likely ceiling was the
**feature set** — collapsing history to fixed scalars — not the
combiner's model class. This ADR asks the direct follow-up: does a small,
end-to-end-trained model that operates on the **raw embedding sequence**
itself (candidate-aware attention pooling, the core NRMS/NAML mechanism,
minus the text encoder) do meaningfully better? Per the engineer's
explicit instruction, this is a checkpointed question — report back at
MINDsmall-dev with a real, substantial result or stop, not a fire-and-
scale decision.

---

# Decision Scope

- [ ] Entire project
- [x] Single component
- [x] Experiment only
- [x] Assignment-specific
- [ ] General reusable pattern

**Affected Components:** New standalone module/script only
(`src/retrieval/rerank.py`, `scripts/run_attention_reranker_experiment.py`).
No production pipeline, schema, or deployed `EmbeddingScorer` code path
modified — same isolation discipline every candidate in ADR-010 used.
This is, however, the **first neural-network training** in this project —
every prior `torch` usage (`embed.py`) is frozen `sentence-transformers`
inference only.

---

# Context

MINDsmall-train's real click labels have now been used to fit three
qualitatively different model classes: a linear combiner (F/H1), a
gradient-boosted tree ensemble (H2), both over 7-8 precomputed scalar
features, and now (this ADR) a small attention network operating directly
on the 384-dim frozen MiniLM embeddings themselves — closer to what MIND's
own published NAML/NRMS/LSTUR baselines do (attention-pooled user
representations, though those also fine-tune a text encoder end-to-end,
explicitly out of scope here per the engineer's own framing).

---

# Decision Criteria

| Criteria | Importance | Notes |
|---|---|---|
| Genuinely different mechanism from every prior candidate | Critical | Operates on raw embeddings, not derived scalars — the one thing not yet tried |
| Fast to iterate (minutes/hours, not days) | Critical | Explicit engineer constraint given timeline |
| No hidden tuning | High | One documented hyperparameter set (d_attn=64, lr=1e-3, 5 epochs), consistent with every other candidate's "untuned first attempt" stance |
| Correct dev discipline | Critical | MINDsmall-dev touched exactly once, at the end; a train-internal 95/5 user holdout used for early stopping |
| Statistically appropriate comparison | Critical | Paired bootstrap vs. baseline (`paired_metric_diff_ci`), per ADR-010's own established requirement for this shape of comparison |
| Report back at the MINDsmall-dev checkpoint, win or lose | Critical | Explicit engineer instruction — no MINDlarge run regardless of outcome |

---

# Design Space Exploration

## Option 1 — Per-impression single-head attention, dot-product scoring (chosen)

### How it Works

`src/retrieval/rerank.py`:

- `build_user_history_vectors`: the RAW (unpooled) per-article history
  embedding sequence — no equivalent existed anywhere in the project
  before this (confirmed by research; every existing function mean-pools).
- `AttentionScorer(nn.Module)`: single-head scaled dot-product attention,
  **per impression, no padding/masking**. `Wq, Wk: Linear(384, 64,
  bias=False)`. For each candidate: `attn = softmax((Wq(candidate) @
  Wk(history).T) / sqrt(64))` over the history dimension, `pooled = attn @
  history` (a *candidate-specific* user vector — the actual "candidate-
  aware pooling" this ADR is testing), `score = candidate · pooled` (plain
  dot product, not an MLP head — isolates exactly one new variable,
  attention pooling itself, from a second one, a learned nonlinear scoring
  head — same "one hypothesis per candidate" discipline H1/H2 used).
  Zero-history users get an explicit all-zero score (the same cold-start
  convention every other scorer in this project uses for a `None`/absent
  query), not a special case invented for this model.
- `AttentionRerankScorer`: `Scorer`-protocol wrapper for the unchanged Q4
  harness — no whole-corpus caching (unlike BM25Scorer/EmbeddingScorer),
  since candidates score independently given the history; a missing
  candidate id scores `-inf`, matching `score.py::_lookup_scores`'s
  established convention.

### Training

- Loss: `BCEWithLogitsLoss` per candidate, `pos_weight` computed directly
  from MINDsmall-train's real 4.04% click rate (23.72) — one principled,
  data-derived imbalance correction, same spirit as H1's
  `class_weight="balanced"`.
- Optimizer: Adam, `lr=1e-3`, gradient accumulation over 64-impression
  minibatches (no padded batch tensor — see below).
- Model selection: deterministic 95/5 **user-level** holdout carved from
  MINDsmall-train (seed=0), evaluated after each of 5 epochs; best-
  holdout-AUC checkpoint kept. MINDsmall-dev untouched until the one final
  evaluation.
- Reproducibility: `torch.manual_seed(0)`, seeded holdout split and
  per-epoch shuffling.

### A real bug the required benchmark step caught before the full run

Per CLAUDE.md's benchmarking philosophy (measure before committing to a
long run), a small-subset benchmark (3,000 training impressions, 1 epoch)
was run before committing to the real 5-epoch/149,107-impression job. It
crashed: `RuntimeError: element 0 of tensors does not require grad and
does not have a grad_fn`. Root cause: a zero-history user's score is a
constant `torch.zeros(...)` (the cold-start convention above) — that
tensor has no gradient path to `Wq`/`Wk` at all, so `loss.backward()` has
nothing to differentiate whenever a minibatch contains one. Fixed by
skipping zero-history impressions during *training* only (`_train_one_epoch`,
`scripts/run_attention_reranker_experiment.py`) — a correctness fix, not
an approximation: since the model's output for these users is always
exactly 0 regardless of the learned weights, they carry zero gradient
information for this architecture either way; they are still scored
normally (0/tie) at evaluation time, matching every other cold-start
method in this project. The same class of bug (a benchmark step catching
a real issue before a long run) has now happened at least twice in this
project — ADR-010's MINDlarge addendum's NaN-on-`-inf`-inputs bug, and
this one — reinforcing that the benchmark-before-commit step is not
theatre.

### Optimizes For

Testing the specific, previously-unexplored hypothesis (raw embeddings +
a learned, candidate-conditioned pooling function) that ADR-010's addendum
identified as the most plausible remaining lever, at minimal implementation
risk (no padding/masking bugs possible, since each impression is scored
with its own real shapes).

### Sacrifices

- No padded batching means each impression is its own Python-level
  forward/backward call — real but small throughput cost, verified
  acceptable by the benchmark step (measured 1,505 impressions/sec on this
  machine's CPU).
- Dot-product scoring (not an MLP head) and single-head attention (not
  multi-head, as NRMS itself uses) are both deliberate simplifications for
  a first, fast attempt — documented as open levers below, not built.
- Text encoder frozen (per the engineer's explicit scope) — the model
  cannot learn anything about token-level content the MiniLM embedding
  space doesn't already separate.

### Complexity

Low-medium — one new module (~140 lines), one new script (~330 lines),
10 new unit tests. First neural training in the project, so it also
establishes (rather than reuses) the checkpoint-saving/reproducibility
convention.

---

# Comparison Summary (MINDsmall-dev, vs. baseline 0.6340, CI 0.6319-0.6361)

| Candidate | Mechanism | Overall AUC | 95% CI | Paired vs. baseline |
|---|---|---|---|---|
| F | Learned combiner (7 scalar features, LogisticRegression) | 0.6255 | 0.6234-0.6276 | Loss |
| H1 | Learned combiner (8 scalar features, LogisticRegression, balanced) | 0.6319 | 0.6297-0.6340 | **-0.0021** (95% CI -0.0042 to -0.0002) — CI-clear loss |
| H2 | Learned combiner (8 scalar features, HistGradientBoostingClassifier) | 0.5985 | 0.5961-0.6006 | **-0.0355** (95% CI -0.0381 to -0.0331) — CI-clear loss, large |
| **I** | **Neural candidate-aware attention over raw embeddings** | **0.6233** | **0.6212-0.6254** | **-0.0107** (95% CI -0.0126 to -0.0088) — CI-clear loss |

Full per-cohort/per-metric breakdown:
`experiments/candidate_i_attention_reranker_mind_small_2026-08-22/{config,results}.json`.

---

# Final Decision

## Chosen Option

**Option 1, as designed and run — not adopted.** Per the engineer's
explicit checkpoint instruction, this result is reported and the session
stops here rather than iterating further or scaling to MINDlarge.

### Reason

The result is a real, CI-clear **loss** (paired -0.0107, 95% CI -0.0126 to
-0.0088) — worse than H1's already-small loss, though smaller than H2's.
It does not clear the engineer's own bar ("a real, meaningfully-larger win
... not just CI-clear, actually substantial"). The instruction is explicit
that a result short of that bar means stop and report, not iterate
further within this session.

---

# Rationale

## Why the result came out this way (interpretation, not certain — flagged where inference)

- **Cold cohort is flat, not a win** (0.5727 vs. baseline's 0.5737 — no
  meaningful difference), unlike F/H1's real cold wins. Plausible
  explanation: F/H1's cold-cohort improvement was driven overwhelmingly by
  `log_popularity` (by far the dominant fitted coefficient in both) — a
  pure item-side prior needing no history at all. Candidate I has **no
  popularity signal whatsoever** — only the raw embedding sequence — so it
  structurally cannot reproduce that specific win. This is a real,
  identifiable difference in inputs, not a mysterious regression.
- **Holdout AUC was still climbing at the 5-epoch cap, not plateaued**
  (0.6067 → 0.6096 → 0.6137 → 0.6181 → 0.6205, a steady increase every
  single epoch). This is a genuine open question, not resolved here: the
  reported number may understate this architecture's real ceiling, since
  training was capped at 5 epochs specifically to honor "fast to iterate"
  and the report-back-within-a-day instruction, not because the model had
  converged. Whether more epochs would close the gap, plateau short of it,
  or start overfitting the holdout is unknown — flagged as the most
  concrete, cheap next step if this line is revisited (see Conditions for
  Revisiting), not assumed either way.
- **Warm cohort also lost** (0.6316 vs. baseline warm's 0.6439) — the
  mean-pooled `EmbeddingScorer` baseline (already a strong, unweighted
  signal) was not beaten even where the attention mechanism has the most
  history to work with. Combined with the still-climbing holdout curve,
  the most defensible reading is "undertrained, not proven futile" rather
  than "the mechanism doesn't work" — a real distinction this ADR is
  keeping explicit rather than collapsing into a single verdict.

## Why alternatives (documented, not built) were not tried in this pass

- **MLP scoring head instead of a dot product:** would conflate two
  hypotheses (attention pooling AND a nonlinear head) in one result,
  contrary to this project's "isolate one variable" discipline; a natural
  next step only once the dot-product version's own ceiling is understood.
- **Multi-head attention (NRMS's actual mechanism):** more capacity, more
  parameters, more training time — reasonable next lever if single-head
  clearly plateaus, not attempted here to keep the first pass small and
  fast per the explicit scope constraint.
- **More than 5 epochs:** directly implied by the still-climbing holdout
  curve as the cheapest, most likely-informative next experiment — not run
  here because the engineer's own checkpoint instruction was to report
  back at this point, not to keep iterating unilaterally.
- **Adding `log_popularity` as an extra input alongside the embeddings:**
  would directly test the cold-cohort hypothesis above; cheap, but again a
  new iteration beyond this checkpoint's scope.

---

# Decision Confidence

**Current Confidence:** Medium

### Why

- The paired-bootstrap result itself is solid (2000 replicates, same
  tested methodology as every other candidate, CI excludes zero cleanly).
- Not High: the still-climbing holdout curve means this specific number is
  a lower bound on the architecture's real ceiling at this hyperparameter
  setting, not necessarily its converged performance — the "does this
  mechanism work at all" question is not fully closed by one 5-epoch run,
  only "does it work *this well, this fast*."

### What Would Increase Confidence

- Training to convergence (more epochs, watching for the holdout curve to
  actually plateau or reverse) before drawing a final conclusion about the
  mechanism itself, as opposed to this specific budget-capped attempt.
- A multi-head or MLP-head variant, to isolate whether the dot-product/
  single-head simplifications specifically are what's limiting it.

---

# Evidence

## Evidence Sources

- [x] Assignment Requirements (checkpoint framing, bootstrap-CI rigor)
- [x] Experimental Benchmark (real MINDsmall-dev result, real training-time
  benchmark before the full run)
- [ ] Official Documentation
- [x] Research Paper (NRMS/NAML's candidate-aware-attention mechanism,
  adapted — not fine-tuning the text encoder, per explicit scope)
- [ ] Industry Practice
- [ ] Community Consensus
- [x] Engineering Inference (cold-cohort/popularity-feature explanation,
  undertraining hypothesis — both flagged as inference, not verified fact)

## Empirical Evidence

See Comparison Summary above; full config/results in
`experiments/candidate_i_attention_reranker_mind_small_2026-08-22/`.

## Accepted Trade-offs

- No padding/masking (per-impression forward passes) trades some raw
  throughput for eliminating an entire class of correctness bugs on this
  project's first neural training — verified acceptable via the benchmark
  step (1,505 impressions/sec), not merely assumed.
- Skipping zero-history impressions during training (not evaluation) is a
  correctness fix following directly from the model's own architecture
  (constant, gradient-free output for those rows), not an approximation
  or a leakage risk.

---

# Assumptions

| Assumption | Why It Matters | Monitoring Strategy |
|---|---|---|
| 5 epochs is enough to draw a "did this mechanism help" conclusion | The holdout curve was still rising at epoch 5 — this assumption is the weakest link in this ADR's conclusion | Re-run with more epochs (e.g. 15-20) and check whether the holdout curve plateaus, before treating this as a settled negative result rather than a budget-capped one |
| MINDsmall-train/dev's real click-label quality and scale are sufficient to fit a 384→64-dim attention projection meaningfully | If not, more data (MINDlarge-train) rather than more epochs might be the real lever — untested here | Would need a MINDlarge-train fit to check, out of scope per this checkpoint's own stop condition |

---

# Conditions for Revisiting

## Technical Triggers

- [x] Train past 5 epochs and check whether the holdout AUC curve
  plateaus, reverses, or keeps climbing — the single most informative,
  cheapest next data point (same code, just a higher `--epochs`).
- [ ] Try an MLP scoring head and/or multi-head attention, once the
  dot-product/single-head ceiling is actually understood via the epoch
  experiment above — not before, to keep changes isolated.
- [ ] Add `log_popularity` as an auxiliary input, to test directly whether
  the missing cold-cohort win is explained by its absence (this ADR's own
  stated hypothesis, not yet verified).

## Research Triggers

- [ ] If a real win is eventually found at MINDsmall-dev scale, MINDlarge
  re-verification remains the standing project practice (ADR-010) before
  any real submission — not reached in this pass.

## Project / Assignment Triggers

- [x] Whether to spend further session time on this line at all, given
  three candidates (H1, H2, I) have now all lost against the deployed
  baseline — an explicit scope/time decision for the engineer, not
  resolved automatically by this ADR.

**Estimated Cost to Change:** Easy to iterate further (same script,
`--epochs` flag already exists); the deployed `EmbeddingScorer` code path
was never touched, so reverting/ignoring this line entirely costs nothing.

---

# Engineering Impact

## Affected Files

- `src/retrieval/rerank.py`, `tests/unit/test_rerank.py`
- `scripts/run_attention_reranker_experiment.py`
- `experiments/candidate_i_attention_reranker_mind_small_2026-08-22/{config,results}.json`

## Breaking Changes

None — no production code path touched; first neural-training convention
established but not wired into any deployed path.

## Required Tests

10 new unit tests (`test_rerank.py`): sequence-order/skip-missing
correctness for `build_user_history_vectors`, shape/cold-start/gradient-
flow correctness for `AttentionScorer`, missing-candidate/-inf and
empty-history/zero-score correctness for `AttentionRerankScorer`. Full
existing suite (193 tests) reconfirmed passing after the addition.

---

# Related Decisions

## Influenced By

- ADR-010 and its Candidate H addendum — the "feature set is the ceiling"
  finding this ADR directly tests
- ADR-008 (Semantic Retrieval Design) — the frozen MiniLM embeddings and
  cache this candidate's only input source

## Influences

- Any future MIND candidate-search work — three model classes (linear,
  tree, attention) have now all been tried and lost; the next real lever,
  per this ADR's own Conditions for Revisiting, is most plausibly more
  training (epochs) or more input (popularity feature), not a fourth new
  model class

---

# References

## Internal

- `experiments/candidate_i_attention_reranker_mind_small_2026-08-22/`
- `knowledge/ai-usage-log/2026-08-22_gbdt-combiner-candidate-h.md` (this
  candidate's prompt is logged there, as a continuation of the same
  session)
- `/Users/test01/.claude/plans/rosy-twirling-ripple.md` (this candidate's
  approved plan)
- `decisions/ADR-010-mind-second-submission-candidate-search.md`

---

# Decision History

| Date | Event |
|---|---|
| 2026-08-22 | Candidate I designed via Plan Mode, approved by engineer |
| 2026-08-22 | Benchmark step caught a real zero-history-gradient bug before the full run; fixed at the root |
| 2026-08-22 | Real MINDsmall-dev run: CI-clear loss, -0.0107 paired vs. baseline; not adopted; reported back at the checkpoint per the engineer's explicit instruction, no MINDlarge run attempted |
| 2026-08-22 | Addendum: 30-epoch and +popularity re-tests, both real losses; session closed per explicit hard-stop instruction |

---

# Addendum — Two Bounded Follow-up Checks: More Epochs, Popularity Feature (2026-08-22)

**Status:** Resolves the open questions this ADR's own "Decision Confidence"
section flagged (whether 5 epochs was enough; whether adding popularity
fixes the cold-cohort miss). Both checks are real, CI-clear losses.
**Per the engineer's explicit instruction, this closes the MIND
candidate-search line for this session — no further candidates, no
MINDlarge, time redirected to the design note/Q7 checklist.**

## What was checked

Two independent extensions of the same base architecture, both trained on
MINDsmall-train and evaluated once on MINDsmall-dev, same harness/paired-
bootstrap discipline as the base ADR:

- **I-long**: identical architecture/code, `--epochs 30` (vs. the original
  5) — directly tests whether the still-climbing holdout curve flagged as
  an open question meant the original result understated the ceiling.
- **I-pop**: `--epochs 30 --use-popularity` — adds one new learnable
  scalar (`AttentionScorer.pop_weight`) multiplying a candidate's
  train-split-only log-popularity (`build_train_popularity`, reused
  unchanged — same non-leaky construction as every other candidate),
  summed with the existing attention-derived score. Zero-history
  impressions, previously skipped during training (no gradient path
  without this term), are no longer skipped once popularity is enabled —
  they now get a real, popularity-only learnable score instead of a flat
  tie, and correctly contribute a gradient. `src/retrieval/rerank.py` and
  `scripts/run_attention_reranker_experiment.py` both extended (not
  replaced); `tests/unit/test_rerank.py` gained 3 new tests for the
  popularity path (196 total, all passing).

## Results

| Run | Overall AUC | 95% CI | Paired vs. baseline (0.6340) | Best epoch |
|---|---|---|---|---|
| Original (5 epochs, no popularity) | 0.6233 | 0.6212-0.6254 | -0.0107 (95% CI -0.0126 to -0.0088) | 4 |
| **I-long** (30 epochs, no popularity) | 0.6214 | 0.6192-0.6235 | **-0.0126** (95% CI -0.0147 to -0.0104) | 29 |
| **I-pop** (30 epochs, + popularity) | 0.5133 | 0.5110-0.5156 | **-0.1207** (95% CI -0.1237 to -0.1178) | 24 |

Full epoch-by-epoch curves and per-cohort breakdowns:
`experiments/candidate_i_attention_reranker_mind_small_2026-08-22_e30/`
and `..._e30_pop/`.

## I-long: a clean, real answer — more epochs does not help

The holdout AUC climbed **every single epoch, all 30, never plateauing or
declining** (epoch 0: 0.6067 → epoch 29: 0.6372 — a real, substantial
internal-metric improvement, and by epoch 29 it exceeds the deployed
baseline's own overall AUC of 0.6340). Despite that, the actual
MINDsmall-**dev** result for the epoch-29 checkpoint (0.6214) is *lower*
than the epoch-4 checkpoint's dev result from the original 5-epoch run
(0.6233) — confirmed as a real, not noise-level, divergence: both runs
share an identical seeded RNG trajectory through epoch 4 (epoch 4's
holdout AUC matches to 6 decimal places across both runs), so this is the
same deterministic training process diverging further from real dev
generalization the longer it runs, not two independently-seeded results
being compared.

**This is a genuine, useful negative result, not an "undertrained"
caveat anymore.** It also reproduces, one level earlier in the pipeline,
the same qualitative pattern this project already documented twice
between local validation and real blind-test performance (EB-NeRD's
contrastive-vector submission, MIND's Candidate G leaderboard result,
ADR-010's 2026-08-22 addendum): **a validation metric that improves
monotonically does not guarantee the real target metric improves with
it** — here, an internal train-derived holdout (same time window as fit
users, just different users) versus the officially held-out MINDsmall-dev
split. Training longer did not close the gap to baseline; if anything the
real, held-out number got slightly worse.

## I-pop: a confounded result, not a clean test of the hypothesis

The raw number is dramatic — AUC 0.5133 (paired -0.1207, 95% CI -0.1237 to
-0.1178), collapsed to barely-above-chance, worse than every other
candidate tried in this entire two-session search including the
weakest standalone signals (Candidate A's entity embeddings, 0.5525;
Candidate E's popularity-only baseline, 0.5318). **This should not be read
as "adding popularity makes things worse" as a clean scientific finding —
inspection of the run surfaced a specific, identifiable implementation gap
that confounds the result:**

- The holdout AUC was already **0.4733 at epoch 0** — below chance from
  the very first epoch, not a gradual overfit building up over 30 epochs.
  This points to an early-training instability, not slow degradation.
- The fitted `pop_weight` (-0.1047) is **negative** — the opposite sign
  from every other candidate's fitted popularity coefficient in this
  project (Candidate F: +0.744 standardized; Candidate H1: +0.660
  standardized) — both of which found popular articles are *more* likely
  to be clicked, the intuitive direction. A flipped sign here is a strong
  signal something is mechanically wrong, not just "popularity doesn't
  transfer to this architecture."
- **Root cause (identified after the fact, not caught before running —
  a real process gap):** unlike every combiner candidate in this project
  (F/H1/H2), which fit `StandardScaler` over *all* features including
  `log_popularity` before training, this implementation feeds *raw,
  unnormalized* `log_popularity` directly into the model. Given
  MINDsmall's Laplace-smoothed popularity floor (~51,282-article corpus,
  `alpha=1`), unseen-in-train articles score `log(popularity) ≈ -12.5`,
  while genuinely popular articles score roughly `-3` to `-5` — a raw
  range spanning close to 10 units, heavily non-zero-centered. The
  attention term it's added to is a dot product of L2-normalized vectors,
  naturally bounded near `[-1, 1]` and typically much smaller early in
  training. A single shared scalar weight, fit jointly via one BCE loss
  (with `pos_weight=23.72` amplifying gradients on the rare positive
  class) has to simultaneously find the right sign/magnitude for
  popularity *and* counteract that feature's own much larger raw scale —
  a much harder joint-optimization problem than F/H1's setting, where
  `StandardScaler` had already done that normalization before the linear
  model ever saw the feature. This is a plausible, evidence-consistent
  explanation (matches every symptom above), not independently re-verified
  by an actual fix-and-rerun — deliberately not attempted, per the
  engineer's explicit hard-stop instruction for this round.

**Conclusion: I-pop, as implemented, does not cleanly answer "does adding
popularity fix the cold-cohort gap."** It answers a narrower, less useful
question ("does adding an unnormalized raw log-popularity term to this
specific architecture work") negatively, and even that answer is
confounded by an identifiable, fixable implementation gap rather than
strong evidence about the underlying signal itself.

## Final Decision (this addendum)

**Neither follow-up changes the base ADR's non-adoption decision.**
I-long provides real, clean evidence that more training is not the fix.
I-pop's result is real but confounded, and is reported as inconclusive
rather than as a second clean negative — an important distinction to
preserve for anyone revisiting this line, so a future session doesn't
mistake "we ran it once and it collapsed" for "we cleanly tested this and
it doesn't work."

**Per the engineer's explicit instruction, this closes the round: no
further epochs, no re-run with a standardized popularity feature, no new
candidate, no MINDlarge, this session.** If revisited in a future session,
the concrete, well-identified next step is a single fix (z-score
`log_popularity`, matching F/H1/H2's own convention, before combining it
with the attention score) — not a redesign — but that is explicitly
future work, not undertaken here.

## Related

- `experiments/candidate_i_attention_reranker_mind_small_2026-08-22_e30/{config,results}.json`
- `experiments/candidate_i_attention_reranker_mind_small_2026-08-22_e30_pop/{config,results}.json`
- `knowledge/ai-usage-log/2026-08-22_gbdt-combiner-candidate-h.md` (prompt 4)

---

# Addendum — Candidate J Reopens This Line With a Real Win (2026-08-24)

**Status:** This ADR's own closing line ("three model classes ... have now
all been tried and lost") no longer holds. The engineer explicitly
reopened the MIND candidate-search line this session (new Ada HPC compute
access removed the resource constraints that motivated the prior
hard-stop), and Candidate J — a genuinely different architecture from I/
I-long/I-pop, not a further extension of this ADR's own attention
mechanism — produced this project's first real, CI-clear win.

**What changed:** where I/I-long/I-pop all scored a shallow attention head
on top of *frozen* sentence embeddings, Candidate J replaces that with a
trainable title encoder and a trainable history-sequence encoder (the
NRMS architecture proper, Wu et al. 2019), plus GloVe pretrained word
embeddings. Full design, environment history (Kaggle → Ada), and results
are in **`decisions/ADR-012-nrms-lite-candidate-j.md`** — not duplicated
here in full, per this project's practice of keeping one canonical home
per decision and cross-referencing rather than copying.

**Headline result:** 0.6391 (95% CI 0.6370-0.6412) vs. this ADR's own
established baseline CI (0.6319-0.6361) — no overlap, a real win. Also
confirms this ADR's own "Conditions for Revisiting" Research Trigger
("if a real win is eventually found... MINDlarge re-verification remains
standing project practice") is now the live open question, not a
hypothetical.

**This ADR's own I/I-long/I-pop conclusions are unchanged and still
correct as a record of what was tested and why it lost** — Candidate J
does not retroactively fix or explain away those results, it tests a
different, additional hypothesis (trainable encoders + pretrained
embeddings) that this ADR's own "Conditions for Revisiting" section had
already flagged as the next lever, not something this ADR got wrong.

See ADR-012 for the full interpretation, including the honest caveat
about dev-set checkpoint selection and the still-open MINDlarge/
Codabench-submission questions.
