<!--
Fill in every [FILL IN] after running mind_nrms_lite_kaggle_run.py on
Kaggle and pasting back its Cell 7 summary block. Do not estimate or guess
any number below -- leave [FILL IN] literally in place until the real run
reports back, consistent with this project's evidence-hierarchy rule
(CLAUDE.md: benchmarks collected during this project outrank inference).
-->

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
| Baseline (embed similarity, deployed) | 0.634 local / 0.6195 Codabench | submission 886468 |
| H1 (linear) | [FILL IN if recorded elsewhere] | |
| H2 (tree) | [FILL IN if recorded elsewhere] | |
| I (attention, 5 epoch) | 0.6233 | clean negative |
| I-long (attention, 30 epoch) | 0.6214 | clean negative; internal holdout AUC misleadingly kept climbing (0.6067→0.6372) while real dev AUC fell |
| I-pop (attention + popularity, 30 epoch) | 0.5133 | confounded — unnormalized `log_popularity` feature, not a clean comparison |
| **J (NRMS-lite)** | **[FILL IN from Kaggle Cell 7 output]** | epochs=[FILL IN], vocab_size=[FILL IN] |
| Literature band, standard NAML/LSTUR/NRMS | 0.64–0.66 | Wu et al. via DIGAT benchmark table; not this project's own number |

**Interpretation.** [FILL IN after the run: did J clear the baseline?
Did it land inside, above, or below the 0.64-0.66 literature band? If it
underperforms the literature band despite the architectural fix, the next
suspects, in priority order, are: (a) no pretrained word embeddings —
biggest single lever, ~1-2 AUC points in published ablations; (b) only
3 epochs on MINDsmall vs. more extensive tuning in published work; (c)
embed_dim=128 / 8 heads is smaller than the paper's 300-dim / 15-16-head
setup, traded down for Kaggle time budget.]

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
