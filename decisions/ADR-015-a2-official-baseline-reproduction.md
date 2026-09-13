# ADR-015 — A2 Q3: Reproduce the Official NRMS Baselines, Then One Principled Change

**Date:** 2026-09-11
**Status:** Decided (the plan). In progress (execution). One open item: the MIND treatment definition, see "Open".
**Decision owner:** Engineer (sukhraj01). **Contributors:** Claude Code (research, plan, implementation).

---

## Context

A2 Q3 reads: *"Reproduce the official/starter baseline (e.g., NRMS from the `ebnerd-benchmark`
repo, or the MIND baseline) on both datasets. Improve with one principled change. Ablation.
Claimed gains must ship a paired bootstrap 95% CI that excludes zero."*

Nothing in A1 is a reproduction of an official baseline. Candidate J (ADR-012) is our own
PyTorch "NRMS-lite", and Candidate K (ADR-013) is a GBDT. The gap check on 2026-09-11 found
the reference points cut in opposite directions:

| Dataset | Official NRMS, published | Our best, real leaderboard | Source |
|---|---|---|---|
| MIND (test) | **AUC 67.76**, MRR 33.05, nDCG@5 35.94, nDCG@10 41.63 | J: 0.6462 | MIND paper Table 3 (`data/ACL2020_MIND.pdf`) |
| EB-NeRD (test) | **AUC 61.03**, MRR 39.75, nDCG@5 44.45, nDCG@10 51.24 | K: 0.7542 | EB-NeRD paper Table 3 (arXiv 2410.03432) |

So on MIND the faithful official baseline is expected to *beat* our current submission.
J also deviates from the canonical config: title length 20 vs 30, 15 vs 20 heads, lr 1e-3 vs
1e-4, batch 64 vs 32.

## Alternatives considered

1. **Option A (chosen): run the official TF/Keras code as-is.** EB-NeRD uses `ebnerd-benchmark`'s
   `NRMSModel` with `args_nrms.py` defaults. MIND uses `microsoft/recommenders`' `NRMSModel`
   with the official `MINDlarge_utils` (GloVe-initialised `embedding.npy`, `nrms.yaml`).
   This is literally what the brief names, and it is reviewer-proof. Cost: a separate TF env.
2. **Option B: align our PyTorch NRMS to the canonical config** and check it against the
   published numbers. Faster, but it is our code standing in for the official baseline.
   Kept as the fallback.

## Decision

Option A, **timeboxed to one focused Ada session.** If both official models are not
training cleanly by the end of that session, stop, fall back to Option B, and record why in
an addendum here. The deadline is not to be extended quietly.

## Minimal, documented adaptations (required for the paired CI; not model changes)

The official EB-NeRD script trains on train+validation and predicts only the unlabeled test
set. A paired bootstrap needs labeled, per-impression scores for control and treatment on
the *same* impressions. Therefore:

- **EB-NeRD:** train on `ebnerd_small/train` only. Keep the official "last day of training
  data" early-stopping split. Score all of `ebnerd_small/validation` (244,647 impressions,
  the split Candidate K was also evaluated on). Model code, hparams, embeddings, and
  sampling (`sampling_strategy_wu2019`, npratio 4) stay unchanged. `ebnerd_small` is also
  the official script's own default `datasplit`.
- **MIND:** train on MINDlarge train, evaluate on MINDlarge dev (376,471 impressions, the
  same split J's 0.6579 and the embedding baseline's 0.6335 were measured on). Official
  `nrms.yaml` hparams. Per-impression scores are dumped for our bootstrap. MINDlarge test
  predictions are produced only for the model that ships.
- Metrics are computed by *our* harness (`src/evaluation/`) on the dumped scores. The
  official evaluator's numbers are also logged, as a cross-check.

## A/B framing (for the report)

- **Control** = the reproduced official NRMS. **Treatment** = control + exactly one change.
- **Primary metric:** AUC. **Significance test:** paired bootstrap 95% CI on the per-impression
  difference (the `src/evaluation/bootstrap.py` substrate, as in ADR-010's Candidate G).
- **Guardrails** (must not regress, CI-checked): diversity@10, novelty@10, coverage@10 (point
  estimate, per ADR-007), and p99 serving latency (ADR-014 method).

## Treatments

- **EB-NeRD:** official NRMS + a freshness signal (candidate age at impression time). This is
  motivated by ADR-013, where freshness carries 33.3% of K's gain and `article_age_h` is the
  top feature. Ablation: with vs. without the freshness input, everything else fixed.
  Candidate K is unchanged and serves as the Q2 re-ranker.
- **MIND (decided 2026-09-11): official NRMS with title+abstract news input.** Ablation:
  title-only (= control) vs title+abstract, everything else fixed (GloVe init, hparams,
  epochs, seed). Literature basis: the MIND paper's own news-representation table reports
  Title 66.22 → Title+Abs+Body+Cat 67.09 → 67.38 (AMV).

### Decision reversal on the MIND treatment (preserved)

The engineer first specified MIND treatment = "official NRMS + pretrained GloVe instead of
random init". It was re-decided on this evidence:
- The official `recommenders` NRMS is *already* GloVe-initialised. Its notebook says "a
  embedding matrix is initted from pretrained glove embeddings", loading
  `utils/embedding.npy`.
- ADR-012 already ran exactly this lever on J (no GloVe 0.6242 vs GloVe 0.6391).

"GloVe vs random" would therefore compare the control against a *degraded* control, and
leave Q3.2's "improve" unmet on MIND. Options put to the engineer: title+abstract
(chosen), pretrained-LM news vectors, or GloVe-vs-random reported as a component ablation.

### EB-NeRD freshness: design

Facts from the official code (`ebrec/models/newsrec/nrms.py`, `dataloader.py`):
- The scorer is `dot(user_vec, news_vec)`, with sigmoid for scoring and softmax over the
  npratio group for training.
- The loaders pass no timestamp columns.

Treatment: add a per-candidate input `age = log1p(max(0, impression_time −
published_time) in hours)`, fused late as `score = dot(u, n) + w·age + b`, where `w` and
`b` are learned scalars. Both the training and scorer graphs get the input, and a loader
subclass supplies it alongside `pred_input_title`. The news and user encoders are untouched,
so the ablation isolates freshness alone.

The clip at 0 is required by `log1p` and applies to the 13 known negative-age articles
(ADR-013: 0.002% of validation rows, lower-than-baseline CTR, so not exploitable). This
departs from ADR-013's choice to leave ages untransformed for the GBDT, which is
scale-invariant and can split on negative values.

## Environment (facts measured 2026-09-11)

- One env serves both: python 3.11, tensorflow 2.15.1 (`[and-cuda]`), numpy 1.26.0,
  polars 0.20.8, transformers 4.36.2, CPU torch 2.2.2. The full freeze is written to
  `/share1/$USER/a2/env_freeze.txt`, which closes the unpinned-venv gap noted in
  `docs/ada_cluster_setup.md` §2.
- Storage: the plan was `/share1/$USER/a2` (25 GB quota, empty), but **`/share1` is not
  mounted on compute nodes**. Prep job 2694345 failed instantly with no log. Diagnostic job
  2694346 on gnode002 found `/share1: No such file or directory` and `/ssd_scratch` present
  (node-local, 563 GB free). `$HOME` is shared but has only ~5.8 GB free. The revised design
  is one self-provisioning GPU job: env and data are set up in parallel on `/ssd_scratch`,
  a smoke test of both models runs, then the full controls. Small outputs are persisted to
  `$HOME/a2`. EB-NeRD weights (~1 GB+) stay on scratch.
- Ada account now on **QoS `low`** (cpu=10, gpu=1, mem=32000M per user; was `medium` in A1).
  `u22-cpu` is `AllowAccounts=devalab`, so `research` jobs go to `u22`.
- Downloads: HuggingFace ~9 MB/s from Ada, EB-NeRD S3 ~18 KB/s. So MIND comes from HF,
  and `ebnerd_small.zip` is pushed from the local machine (sha256-checked).
- None of A1's Ada artifacts survive in `$HOME`: no `ebnerd_shared_cache`, no `ebnerd_gbdt`
  venv, and no K 65-feature booster. Any A2 step needing K's scores must retrain K.

## Execution record (2026-09-11)

| Job | What | Outcome |
|---|---|---|
| 2694345 | CPU prep job (env + data on `/share1`) | FAILED at 0s, no log: `/share1` absent on compute nodes |
| 2694346 | 1-CPU diagnostic | Confirmed: `/share1` absent, `/ssd_scratch` present (gnode002) |
| 2694353 | Combined self-provisioning control job, `"mind ebnerd"` | **CANCELLED by Claude at 04:00:55 elapsed** (own job). Data setup failed at once: missing `mkdir` before `unzip -d`, now fixed. The env install never finished: PyPI on the compute node averaged ~0.3 MB/s (3.1 GB of uv cache in ~3 h), then stalled with no log change from 22:23 to 23:19, still downloading cudnn/cublas/nccl/cusparse. **A GPU was held idle for ~4 h.** The design had assumed 10–15 min based on the *login node's* HuggingFace speed; compute-node PyPI speed was never measured. |

| 2694462 | 2-CPU check of `$HOME/venvs/perf_bench` on a compute node | COMPLETED (gnode046): py 3.11.16, torch 2.11.0+cu128, numpy 1.26.4 |
| 2694501 | **Option B, MIND control** (official config, 10 epochs, MINDlarge train → dev) | Running on gnode075; submitted only after the utils sha256 matched local. Loaded 2,232,748 train / 376,471 dev impressions in 129 s, **exactly A1's counts**, which confirms the pushed data is A1's data. |

| 2694505 | **Option B, EB-NeRD control** (official config, 5 epochs, ebnerd_small train → validation) | Submitted after sha256 matched local for the token npz, runner, and data module. Queued behind 2694501 (`QOSMaxCpuPerUserLimit`, 1-GPU cap). |
| 2694506 | **Option B, EB-NeRD treatment** (control + freshness late fusion; everything else fixed) | Same hash-checked runner (`0e04f442…`). Queued behind 2694505. |
| 2694529 | **Option B, MIND treatment** (title 30 + abstract 50; everything else fixed) | Same runner (`0e04f442…`). Submitted after the control's epoch-1 timing made the 10-epoch projection (~37 h) fit the 72 h limit. Queued behind 2694506. **CANCELLED 2 min in** — cancelling 2694506 had started it ahead of the fast EB-NeRD pair, and it carried the pre-fix chunking. |
| 2694962 / 2694963 / 2694964 | **Resubmission after the OOM fixes**: EB-NeRD control, EB-NeRD treatment, MIND treatment (in that order, so the ~30-min arms finish before the ~37-h one) | Runner `a7bb6264…`, sha256-verified on Ada. Smoke-verified numerically neutral before resubmission. |

**MIND control, epoch 1 (measured 2026-09-12):**
- Train time 3,787 s (63 min) on the 2080 Ti for 3,383,656 examples. The same example count
  as J's MINDlarge run, which confirms the sampling semantics.
- Dev monitor AUC **0.6692** after 1 of 10 epochs. That is already above J's best MINDlarge-dev
  0.6579 and in the neighbourhood of published NRMS (67.76, *test*, not directly comparable).
- Epoch 1 is a monitoring number, not a result: the reported number is the final epoch (no
  selection on dev).
- ~70–80 min per epoch including dev eval, so the 10 epochs should finish at ~14:30 Ada time.

**MIND control finished (job 2694501, 2026-09-12, 13 h 13 m for 10 epochs).**

Evaluated by `a2_evaluate_scores.py` over the dumped per-impression scores, all 376,471
MINDlarge-dev impressions / 255,990 users, 0 skipped:

| Metric | Reproduced official NRMS | 95% CI | Candidate J (A1 best) | Published NRMS (*test*) |
|---|---:|---|---:|---:|
| AUC | **0.6831** | 0.6821–0.6839 | 0.6579 | 0.6776 |
| MRR | 0.3803 | 0.3792–0.3814 | 0.3581 | 0.3305 † |
| nDCG@5 | 0.3620 | 0.3608–0.3632 | 0.3411 | 0.3594 |
| nDCG@10 | 0.4279 | 0.4269–0.4290 | 0.4064 | 0.4163 |
| diversity@10 | 0.8343 | 0.8337–0.8348 | — | — |
| novelty@10 | 17.6230 | 17.6139–17.6324 | — | — |
| coverage@10 | 0.0627 | point est. (ADR-007) | — | — |

The AUC agrees to four decimals with the runner's independently computed monitoring AUC,
i.e. two implementations (sklearn per impression vs the harness's vectorised
Mann-Whitney form) on the same scores.

**Two caveats, so these are not over-read:**
- The comparison with J is **marginal, not paired**: J's per-impression scores were never
  saved, so only its published CIs are available. The intervals do not overlap, and the gap
  (+0.0252 AUC) is 25x the width of either, but a paired test would be the stronger claim
  and cannot be run retrospectively.
- **MRR is not comparable to the MIND paper's.** A1 established that the official
  `evaluate.py` sums 1/rank over *all* clicked items, while this project's harness is
  first-hit-only (PROJECT_STATE, Codabench converter validation). † is therefore marked as
  a different statistic, not a worse score. The published figures are also on the blind
  *test* split, not dev.

Per-epoch dev AUC rose from 0.6692 (epoch 1) to 0.6836 (epoch 9), ending at 0.6831. The
reported arm is the **final** epoch, not the best: the official `fit()` selects nothing on
dev, and honouring that is what makes this a reproduction. The published 0.6776 is on the
blind *test* split and so is not directly comparable, but the reproduction landing in that
neighbourhood — and **+0.025 above Candidate J on the identical split** — is the evidence Q3
needs that the control is trustworthy.

These are monitoring numbers from the runner. The reported figures come from
`a2_evaluate_scores.py` over the dumped per-impression scores.

**A scare that was not a defect, recorded because the first reading was wrong.** The
Option B sbatch was briefly believed to have lost the scores on node-local scratch, because
the *Option A* script (`a2_official_nrms_train.sbatch`, never successfully run) copies
results with a `*_scores.parquet` glob that does not match the runner's `scores.parquet`.
The Option B script passes `--out-dir "$P/results/$NAME"`, so the runner wrote directly into
shared `$HOME`: `scores.parquet` (92.6 MB) and `results.json` were there all along, and the
attempted rescue found nothing on `/ssd_scratch` because nothing was ever written there. The
glob was fixed in the unused Option A script anyway, since it would bite if a TF env ever
becomes available.

**EB-NeRD control finished (job 2694962, 26 m 43 s, 5 epochs).** Evaluated over all 244,647
`ebnerd_small` validation impressions / 15,342 users, 0 skipped:

| Metric | Reproduced official NRMS | 95% CI | Candidate K (65-feat, same split) | Published NRMS (EB-NeRD *test*) |
|---|---:|---|---:|---:|
| AUC | **0.5613** | 0.5597–0.5629 | **0.7588** | 0.6103 |
| MRR | 0.3512 | 0.3496–0.3528 | 0.5295 | 0.3975 |
| nDCG@5 | 0.3904 | 0.3884–0.3922 | 0.5939 | 0.4445 |
| nDCG@10 | 0.4680 | 0.4663–0.4696 | 0.6306 | 0.5124 |
| diversity@10 | 0.7894 | 0.7879–0.7910 | — | — |
| novelty@10 | 17.2076 | 17.1927–17.2219 | — | — |
| coverage@10 | 0.2021 | point est. (ADR-007) | — | — |

Early stopping behaved as the official recipe intends: val AUC peaked at **epoch 3**
(0.5966) and declined to 0.5885 by epoch 5, and the epoch-3 weights were restored
(`ModelCheckpoint(save_best_only)` + `load_weights`).

**The headline for EB-NeRD is the opposite of MIND's, and it is a real finding.** The
faithful official baseline scores 0.5613 while Candidate K's GBDT scores **0.7588 on the
identical split** — a ~0.20 AUC gap in A1's favour. That is the RecSys 2024 pattern this
project already documented (ADR-013: every top team used a GBDT over engineered features,
not a neural news encoder). So on MIND the reproduction *beat* our A1 model, and on EB-NeRD
our A1 model beats the reproduction by a wide margin.

Two honest caveats: our 0.5613 is below the paper's 0.6103, which is measured on the blind
*test* split after training on the full `ebnerd_large`, whereas this run used `ebnerd_small`
(193,617 training samples) per `args_nrms.py`'s own default `datasplit`; and the K
comparison is marginal, not paired.

**EB-NeRD treatment finished (job 2694991, 29 m 58 s, 5 epochs).** Training trace, which is
worth reading in full because the freshness weight does not behave monotonically:

| Epoch | loss | early-stop val AUC | lr | freshness weight |
|---|---:|---:|---:|---:|
| **1 (restored)** | 1.4128 | **0.608583** | 1e-4 | **−0.0660** |
| 2 | 1.3467 | 0.593709 | 1e-4 | +0.0036 |
| 3 | 1.3101 | 0.584285 | 1e-4 | +0.0678 |
| 4 | 1.2658 | 0.580050 | **2e-5** | +0.0859 |
| 5 | 1.2538 | 0.575717 | 2e-5 | +0.1022 |

- **The shipped model is epoch 1**, restored by the official `save_best_only` +
  `load_weights` path. Its freshness weight is **negative (−0.066)**, and since the feature
  is `log1p(age in hours)`, negative means **older candidates score lower — fresher wins**.
  That agrees with ADR-013's finding that `article_age_h` dominates EB-NeRD.
- **The weight's sign flips after epoch 1** and grows positive while val AUC falls
  monotonically — drift as the encoders overfit, not a contradictory finding. Any claim
  about freshness must cite the restored epoch, not the final one. (An earlier reading of
  this run's log *tail* recorded the positive values and drew the opposite conclusion; the
  full trace corrects it.)
- `ReduceLROnPlateau` fired as configured at epoch 4 (1e-4 → 2e-5).
- The treatment's best early-stop AUC (0.6086) is **+0.0120 above the control's** (0.5966),
  and its validation monitor AUC is 0.5677 vs 0.5613. The paired bootstrap over identical
  impressions is the test that decides whether that gain is real.
- Age guard on the real data: median candidate age 3.02 h, log-age varies within 99.997% of
  samples — the units bug could not recur silently.

## EB-NeRD A/B result — the first complete Q3 answer (2026-09-12)

**Control** = reproduced official NRMS. **Treatment** = control + freshness late fusion,
one change, everything else held fixed (same seed, data, epochs, hyperparameters).
**Test** = paired bootstrap over the identical 244,647 validation impressions / 15,342
users, resampling users.

| Metric | Control | Treatment | Paired Δ (treat − ctrl) | 95% CI | Verdict |
|---|---:|---:|---:|---|---|
| **AUC** (primary) | 0.5613 | 0.5677 | **+0.0064** | +0.0053, +0.0075 | **CI excludes zero** |
| MRR | 0.3512 | 0.3555 | +0.0043 | +0.0032, +0.0054 | CI excludes zero |
| nDCG@5 | 0.3904 | 0.3957 | +0.0053 | +0.0042, +0.0065 | CI excludes zero |
| nDCG@10 | 0.4680 | 0.4725 | +0.0044 | +0.0035, +0.0054 | CI excludes zero |
| diversity@10 (guardrail) | 0.7894 | 0.7900 | +0.0006 | +0.0004, +0.0008 | improved |
| novelty@10 (guardrail) | 17.2076 | 17.2356 | +0.0280 | +0.0269, +0.0292 | improved |
| coverage@10 (guardrail) | 0.2021 | 0.1987 | −0.0035 | point only (ADR-007) | slight dip |

**Q3 is satisfied on EB-NeRD**: the baseline is reproduced, improved by one principled
change, isolated by ablation, and the claimed gain ships a paired 95% CI that excludes zero.
**No guardrail regressed** — diversity and novelty both improved CI-clear.

Four things the headline must not hide:
1. **The gain is small in absolute terms** (+0.0064 AUC). It is *tightly* estimated, not
   large: the CI is narrow because n is 244,647 impressions, not because the effect is big.
2. **The improved baseline is still far below Candidate K** (0.5677 vs **0.7588** on the
   identical split). Beating the official baseline does not make this the better model, and
   the report must say so. A1's GBDT remains the strongest EB-NeRD system this project has.
3. **Coverage@10 fell** (0.2021 → 0.1987, −1.7% relative). It carries no CI by ADR-007's
   reasoning (a set-union statistic is biased under with-replacement resampling), so it is
   reported as a point difference and flagged, not waved through. A freshness-weighted
   ranker concentrating on recent articles is a plausible mechanism worth stating.
4. **The freshness direction comes from the restored epoch**, where the weight is −0.066
   (fresher wins). Later epochs' positive weights were discarded by early stopping.

**EB-NeRD control OOM'd first, and what that exposed (job 2694505, 2026-09-12).** The control
completed epoch 1 (early-stop val AUC 0.5960, 201 s/epoch, 1,019 samples/s) and then died in
epoch 2's early-stop scoring with `torch.OutOfMemoryError` — 1.38 GiB requested, 1.18 GiB
free on the 11 GB 2080 Ti.

The cause is arithmetic, not a leak. That block scored 1,024 samples at once, and each
sample carries 20 history + 5 candidate articles, so 25,600 rows enter the news encoder
together; their attention scores alone are 25,600 × 20 heads × 30 × 30 × 4 B ≈ 1.8 GB.
Epoch 1 fit; epoch 2 met a fragmented allocator (1.88 GB "reserved but unallocated").

Three fixes, and one scheduling consequence:
- `--es-batch` (default **128**, was a hard-coded 1024) with per-chunk frees: ~0.23 GB.
- `evaluate()`'s news-encoding chunk is now **width-aware**, `2048 × (30/T)²`. This matters
  for the MIND *treatment*, whose 80-token input (title 30 + abstract 50) makes attention
  ~7× the control's per row; a flat 2048 would have allocated ~1 GB per chunk on a card that
  had already OOM'd once.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, the allocator's own advice for that
  fragmentation signature.
- **Queue reordered.** Cancelling the doomed EB-NeRD treatment (2694506) immediately started
  the MIND treatment (2694529, the ~37 h job), which would have put the two ~30-minute
  EB-NeRD arms behind it for two days — and it carried the unfixed width-aware chunk itself.
  It was cancelled 2 minutes in and will be resubmitted behind the EB-NeRD pair, so the
  first complete A/B result arrives today rather than on the 14th.

**Fix verified on the hardware that broke (2026-09-12).** The resubmitted control (2694962)
cleared **epoch 2** — the exact early-stop block that killed 2694505 — with loss 1.3449 and
val AUC 0.5958, and continued into epoch 3. Epoch timings are unchanged (201.9 s vs 201.5 s),
so the smaller chunks cost no measurable throughput.

**GPU determinism, stated because the smoke tests claim bitwise reproducibility.** Epoch 1's
val AUC differs between the failed and resubmitted runs in the ninth decimal
(0.5960318848 vs 0.5960318833) on identical code, data and seed. That is ordinary
GPU non-determinism (reduction order), not an effect of the fix: the CPU smoke runs
reproduce bit-for-bit. **So the reproducibility claim in the report must be "bitwise on CPU,
run-to-run stable to ~8 decimals on GPU", not "bitwise everywhere".**

**A second hardware failure, different cause: an incompatible GPU (2694988, 2694989).**
Excluding gnode033 sent the EB-NeRD treatment to **gnode012**, which carries a **GTX 1080 Ti
(sm_61)**. This torch build supports sm_75+, so the job ran for 90 s and then died with
`CUDA error: no kernel image is available for execution on the device`. The chained MIND
treatment (the ~37 h job) then started on the same node and was cancelled 2 minutes in.

**Why the existing guard missed it.** The script asserted `torch.cuda.is_available()`, which
returns **True** on an sm_61 card — the device is visible, only its kernels are missing. The
guard now (a) prints the device and its compute capability, (b) fails if it is below sm_75,
and (c) executes a real CUDA matmul, so an unusable GPU is caught in seconds with a legible
message rather than mid-training.

**Scheduling fix: constrain, don't blacklist.** Ada's `u22` nodes advertise a **`2080ti`
feature**; the nodes that failed (gnode012's 1080 Ti, gnode033's broken driver) come from the
featureless pool, while every successful run (MIND control, EB-NeRD control, both on
gnode075) used a 2080 Ti. Submissions now use `--constraint=2080ti`, which selects hardware
by capability instead of excluding bad nodes one failure at a time.

**Three jobs died instantly on one broken node (2694963, 2694964, 2694986, 2026-09-12).**
Every failure landed on **gnode033**, whose driver is broken:

```
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 580.178
```

`nvidia-smi` exits 18 there, and `set -euo pipefail` ends the script on that line — hence
00:00:00 elapsed, exit 18, empty stderr, and `SLURM_JOB_GPUS = 0`. Every job that landed on
gnode075 (MIND control, EB-NeRD control) ran fine.

**Correction of record.** The first diagnosis here blamed a CPU-cap race: two `-c 8` jobs
starting together against the QoS `low` cap of 10. That was wrong, and it was written from
`sacct`'s `Reason=QOSMaxCpuPerUserLimit` (a stale *pending* reason) before the job's own log
was read. The log names the real cause. Keeping the wrong version visible here is the point:
the `Reason` column described why the job had waited, not why it died.

**Fixes:**
- `--exclude=gnode033` at submit time, since the node is broken rather than busy.
- `nvidia-smi` made non-fatal in the script, so a bad node reports a clear warning and lets
  the explicit `torch.cuda.is_available()` assert be the thing that fails.
- `--dependency=afterany:<jobid>` chaining is kept, though it was adopted for the wrong
  reason: it still guarantees only one job is eligible at a time, which suits the 1-GPU cap.
  `afterany`, not `afterok`, so a failed arm does not block the next one.

**MIND treatment throughput, measured (job 2694992, 2026-09-12).** The first step-progress
line gives **300 samples/s at width 80**, against the control's **893 samples/s at width 30**
(3,383,656 samples / 3,787 s) — a **2.98× slowdown**, against the 3.2× predicted from the
local CPU smoke, so the projection method held. Training ETA is **179 min/epoch**; the
control's epochs cost 79 min total (63 training + ~16 evaluating), and the treatment's
evaluation is also slower at 80 tokens, giving **~3.6 h/epoch and ~36–37 h for 10 epochs**.
That fits the 72 h wall limit with ~2× margin, and lands early on **2026-09-14**, well inside
the 2026-09-20 deadline. The step logging added after job 2694501 is what made this
checkable 13 minutes in rather than 3 hours in.

## MIND A/B result — Q3 complete on both datasets (2026-09-14)

**Treatment finished cleanly: all 10 epochs, no errors, 34.06h wall (well inside the
72h limit, and ~3.5h faster than the ~37h projection).** Per-epoch monitor AUC
ranged 0.6769 (epoch 1) to a peak of 0.6916 (epoch 5), settling at **0.6868** by
epoch 10 — the official recipe's "final epoch, no selection" number. Checksummed
against Ada (sha256 match) before evaluation.

Evaluated by the harness over all 376,471 MINDlarge-dev impressions, same
methodology as EB-NeRD's A/B (§ above):

| Metric | Control | Treatment | Paired Δ | 95% CI | Verdict |
|---|---:|---:|---:|---|---|
| **AUC** (primary) | 0.6831 | 0.6868 | **+0.0037** | +0.0031, +0.0044 | CI-clear win |
| MRR | 0.3803 | 0.3829 | +0.0026 | +0.0018, +0.0034 | CI-clear win |
| nDCG@5 | 0.3620 | 0.3691 | +0.0071 | +0.0063, +0.0078 | CI-clear win |
| nDCG@10 | 0.4279 | 0.4314 | +0.0034 | +0.0028, +0.0041 | CI-clear win |
| diversity@10 (guardrail) | 0.8343 | 0.8283 | **−0.0060** | −0.0062, −0.0057 | **CI-clear REGRESSION** |
| novelty@10 (guardrail) | 17.623 | 17.643 | +0.0197 | +0.0174, +0.0221 | CI-clear improvement |
| coverage@10 (guardrail) | 0.0627 | 0.0626 | −0.0001 | point only (ADR-007) | negligible |

**The headline is not just the win — it's that a guardrail genuinely regressed
while the primary metric won CI-clear.** This is the first time in this project's
entire history (both assignments) that a candidate's guardrail has regressed
rather than improved or held flat; every earlier win (EB-NeRD's freshness
treatment included) improved every CI-bearing guardrail. Per the professor's
explicit framing, a guardrail regression is not allowed to be waved through by a
primary-metric win — it is reported here as exactly that: a real cost of the
title+abstract change, not smoothed into the accuracy headline. Plausible
mechanism: richer candidate text content (49 additional abstract tokens) likely
sharpens the model's confidence toward specific topical matches with a user's
history, at some expense to the categorical variety of what it ranks highly.

**Head/tail slicing reproduces EB-NeRD's exact directional pattern —
independently, on a different dataset, a different treatment, and a different
mechanism:**

| Slice | $n$ | Control AUC | Paired ΔAUC | 95% CI | Verdict |
|---|---:|---:|---:|---|---|
| Head (top 20% by train clicks) | 143,164 | 0.7411 | **−0.0063** | −0.0071, −0.0053 | CI-clear loss |
| Tail | 233,307 | 0.6474 | **+0.0098** | +0.0090, +0.0108 | CI-clear gain |

Both datasets' treatments help most where the content signal is weakest relative
to prior popularity (tail/unpopular articles) and can mildly hurt where an
article is already popular enough to be identified from title alone. This is
independent replication of the same qualitative finding across EB-NeRD (freshness)
and MIND (richer text), from unrelated mechanisms — the kind of cross-dataset
consistency slicing was built to be able to surface, not designed in advance.

**Warm/cold: the gain is entirely a warm-user effect.**

| Cohort | $n$ | Paired ΔAUC | 95% CI | Verdict |
|---|---:|---:|---|---|
| Warm | 323,747 | +0.0043 | +0.0036, +0.0050 | CI-clear win |
| Cold | 52,724 | +0.0002 | −0.0020, +0.0022 | **not significant** |

Sensible mechanistically: richer candidate text cannot improve a user
representation built from zero history. The reproduced-then-improved MIND
pipeline still does not close cold-start, consistent with every prior finding
in this project (A1's embeddings, this ADR's control-vs-J comparison) that
better modeling raises the whole curve without narrowing the warm/cold gap.

## Addendum (2026-09-14) — J's real deployed diversity@10, for scale on the regression

Before deciding whether the treatment ships to the MIND leaderboard, the engineer asked for
diversity@10 magnitude against what is currently live (Candidate J). It was never computed
for J anywhere in this project — checked directly, not assumed (no diversity/novelty/coverage
field exists in any of J's config/results files or scripts). Computed here directly from J's
actual submitted `MINDlarge_test/prediction.txt` (the literal ranking that scored 0.6462),
joined against the raw candidate order in `MINDlarge_test/behaviors.tsv` and the processed
test catalog's categories — not a re-run, not an estimate.

| | Diversity@10 | 95% CI | Split | Model |
|---|---:|---|---|---|
| Control (reproduced NRMS) | 0.8343 | 0.8337–0.8348 | MINDlarge-**dev** | Option B reproduction |
| Treatment (+title/abstract) | 0.8283 | 0.8277–0.8289 | MINDlarge-**dev** | Option B reproduction |
| **Candidate J, currently live** | **0.7840** | 0.7830–0.7850 | MINDlarge-**test** | NRMS-lite (ADR-012) |

Sampled systematically, 200,000 of 2,370,727 real test impressions, 163,251 users, 0 skipped.

**Not a paired comparison — stated plainly.** J's number is on the blind test split, not dev,
and from an entirely different architecture, not the official reproduction. It is a real
descriptive data point, not a controlled one.

**The magnitude, for scale.** The treatment's regression from the control is −0.0060
(−0.72% relative) — an order of magnitude smaller than the gap between either new candidate
and what is actually live: treatment vs. J is **+0.0443** (+5.65% relative). Whatever the
control-vs-treatment decision, both sit well above J's real deployed diversity.

**Decision: still open, for the engineer.** This addendum supplies evidence, not a
recommendation on ship/no-ship.

**Q3 is now complete on both datasets.** EB-NeRD: reproduced (0.5613), improved
by freshness (0.5677, +0.0064 CI-clear, no guardrail regression). MIND:
reproduced (0.6831), improved by title+abstract (0.6868, +0.0037 CI-clear,
**one guardrail regression, disclosed**). Both satisfy Q3's four requirements —
reproduce, improve by one change, ablate, ship a paired CI — and both now also
carry Q5's slicing and, for EB-NeRD, Q9's ablation.

**MIND treatment timing (projected before submission, for the historical record).** On identical 461-example local runs,
the treatment's 80-token input cost 3.2× the control's 30 tokens (52.1 s vs 16.3 s). Applying
that ratio gives ~3.7 h per epoch and ~37 h for 10 epochs, inside the 72 h limit with ~2×
margin. Submitted behind the EB-NeRD jobs. With the 1-GPU cap, the expected MIND treatment
result is ~2026-09-14/15, against a 2026-09-20 deadline. This is a projection: the treatment's
own step logging will confirm or refute it within its first epoch.

**Earlier hold, now released:** the MIND treatment was held deliberately, not forgotten. Its 80-token input costs more per step
than the control's 30, so its submission waits on the control's measured epoch time: 10
epochs must fit inside the 3-day `--time` limit.

**Runner provenance note.** The local EB-NeRD smoke test found a crash in the runner's
EB-NeRD-only early-stopping split (`DatetimeIndex >= Timestamp` already returns an ndarray,
so the `.to_numpy()` call failed). Fixed as `np.asarray(...)`. Job 2694501 runs the pre-fix
runner. The fix touches only the `--dataset ebnerd` branch, so the MIND control's code path
is byte-identical in behaviour. EB-NeRD jobs are submitted only with the fixed runner.

**Timebox status (2026-09-11 23:19):** no official model has trained. The blocker is Ada env
provisioning, not the official code. Per the engineer's rule, this goes back to the engineer
as a decision rather than being retried silently.

Submission-time corrections, each from a real rejection rather than a guess:
- `--qos=medium` → `low` (`Invalid qos specification`; `sacctmgr` shows only `low`).
- `-p u22-cpu` → `u22` (`AllowAccounts=devalab`).
- `--mem-per-cpu=4G` × 8 → `--mem=31000M` (8 × 4G = 32,768 MB exceeds the 32,000 MB
  `QOSMaxMemoryPerUser`).

**Data provenance.** MIND train/dev are the exact A1 files, pushed from the local machine and
sha256-verified on Ada (train `120dbabb…6840`, dev `a9ce423c…66c8`). HuggingFace's
`Recommenders/MIND` zips are not byte-identical (train 531,360,717 vs 530,197,363 bytes),
and paired comparability with J and the embedding baseline requires the same 376,471 dev
impressions. Only `MINDlarge_utils.zip`, which has no local copy, comes from HuggingFace.

## Addendum (2026-09-11, late): timebox expired → Option B

**Trigger.** This is the engineer's own rule: "If it's not training cleanly by then, stop, fall
back to Option B, and document why in the ADR rather than quietly extending the deadline."
After one Ada session, no official model had trained.

**Why Option A failed. The cause is environment provisioning, not the official code:**

| Measurement (2026-09-11) | Rate | Implication for the ~3 GB TF 2.15 + CUDA wheel set |
|---|---:|---|
| PyPI from an Ada compute node (job 2694353, uv log + cache size) | ~0.3 MB/s, then a ~1 h stall | Never completed in 4 h |
| PyPI from the Ada login node (`files.pythonhosted.org`, 20 s sample) | ~0.09 MB/s | ~9 h |
| PyPI from the local Mac (manylinux x86_64 TF wheel, 20 s sample) | ~0.37 MB/s | ~2.3 h, plus ~25 min upload at a measured 1.5–2.75 MB/s |
| HuggingFace from the Ada login node | ~9 MB/s | Not the bottleneck |

There was no bounded path to a working TF env on Ada.

Routes not taken, recorded so they can be revisited deliberately:
- A locally-built offline wheelhouse (~2.5–3 h before first training).
- Kaggle/Colab, whose GPU images ship TF preinstalled. This needs the engineer to run
  notebooks by hand; no Kaggle API credentials exist on this machine (ADR-014).

**What Option B is.** A faithful PyTorch reimplementation of each official NRMS *config*,
run in the torch env ADR-014 already proved on Ada GPUs (`$HOME/venvs/perf_bench`,
torch cu128):
- **MIND:** the `recommenders` NRMS config. Official `MINDlarge_utils` GloVe `embedding.npy`
  and `word_dict.pkl`, title length 30, 20 heads × 20 dims, attention hidden 200, npratio 4,
  history 50, batch 32, lr 1e-4, dropout 0.2, 5 epochs.
- **EB-NeRD:** the `ebnerd-benchmark` NRMS config. xlm-roberta-large word embeddings over
  title+subtitle+body tokens (max 30), history 20, same attention sizes and optimiser settings.

**What Option B may and may not claim.**
- It may be reported as "NRMS reimplemented to the official configuration", with the
  reproduction checked against the published numbers: MIND 67.76 test AUC; EB-NeRD 61.03
  test AUC, via a Codabench submission if needed.
- It may NOT be called "the official code". The report must name this substitution and cite
  this addendum.
- The control/treatment pairing, paired bootstrap, and guardrails are unaffected. Both arms
  come from the same implementation, so the *relative* claim does not depend on
  implementation fidelity.

**Official details verified from source (2026-09-11), which the port must match.** Files read:
recommenders `layers.py`, `nrms.py`, `mind_iterator.py`, `newsrec_utils.py`; ebnerd-benchmark
`nrms.py`, `_behaviors.py`, `_articles.py`, `_polars.py`, `_nlp.py`.
- **SelfAttention:** bias-free Q/K/V (glorot), scaled by √head_dim, no output projection,
  output = 20×20 = 400-d. NRMS calls it as `[y, y, y]`, so there is **no padding mask**.
- **AttLayer2:** `exp(tanh(xW+b)·q) / (Σ + 1e-7)`, hidden 200. No mask.
- **News encoder:** Embedding (trainable) → Dropout → SelfAttention → Dropout → AttLayer2.
- **User encoder:** SelfAttention → AttLayer2, no dropout.
- **Training:** softmax + categorical cross-entropy, Adam, no gradient clipping. The scorer
  applies a sigmoid, which is monotone in the logit.
- **MIND history:** the **most recent** 50 (`history[-his_size:]`), left-padded with dummy
  news 0. This corrects an earlier working assumption of "first 50".
- **MIND negatives:** `newsample` **pads with news 0** when there are fewer than 4 negatives;
  it does not resample.
- **MIND tokenization:** `[\w]+|[.,!?;|]` on lowercased text; unknown words map to 0.
- **EB-NeRD history:** `list.tail(20)`, left-padded with id 0. Unknown articles take the
  "zeros" token representation.
- **EB-NeRD negatives:** `sampling_strategy_wu2019`, with replacement.
- **EB-NeRD text:** `concat_str([title, subtitle, body], " ")`, then the xlm-roberta
  tokenizer with `add_special_tokens=False`, `padding="max_length"`, `truncation=True`,
  `max_length=30`.
- **EB-NeRD embeddings:** `model.embeddings.word_embeddings.weight`.

**Settings anchored to official config files** (read from `MINDlarge_utils.zip`):
- **MIND epochs: 10**, from `nrms.yaml` `train.epochs`. The notebook's `epochs=5` was a
  MINDdemo override. This supersedes the earlier TF-path script's default of 5.
- MIND vocabulary: `word_dict.pkl` (41,030 words) with GloVe `embedding.npy`
  (41,031×300, float64). Row 0 is padding; 4,704 rows are all-zero (words missing from
  GloVe get zeros, not random init).
- **MIND treatment abstract length: 50**, from `naml.yaml` `data.body_size: 50`, the official
  multi-field MIND config ("body" is the abstract field in the MIND utils). The treatment's
  news input is therefore [title 30 | abstract 50], with the title segment identical to
  the control's.

**Fidelity-preserving engineering choices (exact, not approximations):**
- EB-NeRD tokenization runs locally under the official `transformers==4.36.2` pin. Ada's
  torch env has 5.16.1, and tokenizer ids are not assumed stable across a major version.
- The xlm-roberta embedding matrix is trimmed to the token ids that actually occur. Rows for
  unseen tokens are never looked up, so they receive exactly zero gradient and Adam never
  moves them; outputs are therefore unchanged.

**Bug caught by the paired test, not by review (2026-09-12).** The first local EB-NeRD smoke
test gave treatment metrics *identical* to the control's: paired Δ exactly +0.0000 on all
four metrics, and the same early-stop val AUC to 16 digits. That happened even though the
freshness weight had moved to −0.011.

Root cause: the token-prep script converted `published_time` with `astype("int64") / 1e9`.
pandas 2.3.3 keeps a pyarrow `timestamp[us]` column as `datetime64[us]`, so the int cast
gives microseconds and every publish time came out ~1000× too small. Every candidate then
had an age of ~467,656 h (~53 years). Within-impression age differences (tens of hours)
vanished under `log1p` (std 0.0000 in every sampled impression), which made the freshness
term a constant shift per impression and therefore a no-op under softmax. The weight still
moved only because Adam normalises even a numerically-zero gradient to steps of about lr.

Fix and guards:
- (a) A unit-agnostic `to_epoch_seconds`, with a regression test that pins the `[us]` case.
- (b) A plausibility assertion in the prep script (publish time between 1990 and 2030). Its
  first version used 2000 and fired on real data: the catalog holds 2 genuine pre-2000
  archive articles (1998-12-27 and 1999-11-09), out of 20,738.
- (c) A runner guard that aborts before training if the median candidate age is outside
  0–8,760 h, or if log-age varies within fewer than 50% of samples in the freshness arm.

**EB-NeRD smoke after the fixes** (local CPU, 2 epochs, 3,000 train impressions, 400
validation impressions; a code-path check, not an effect estimate):
- Age guard: median candidate age 3.2 h, and log-age varies in 100% of samples.
- The treatment now differs from the control: early-stop val AUC 0.5358 vs 0.5281; paired AUC
  +0.0025 [−0.0010, +0.0059], not significant at this scale.
- **Reproducibility:** the control arm reproduced itself exactly across two separate runs
  (epoch-1 loss 1.5901988347371419 and val AUC 0.5280855624142662, to every digit), with the
  same seed, code, and data.

**Evaluation (`scripts/a2_evaluate_scores.py`).** The runner writes only
`impression_id, user_id, labels, scores`. All reported numbers come from this separate step,
which uses the project's own harness:
- AUC via `per_impression_auc` (tie-exact).
- MRR and nDCG@5/@10 via ADR-007's deterministic tie-break.
- Per-arm 95% CIs via `ranking_metric_ci`, resampling users.
- The A/B test via `paired_metric_diff_ci`, imported from `run_gated_cohort_experiment.py`
  (ADR-010's statistic, not reimplemented). It reports treatment − control over identical
  impressions, and a gain counts only if the CI excludes zero.

The script refuses to pair arms whose impression sets or candidate labels differ.

Checks run on 2026-09-12:
- On the MINDsmall smoke outputs, its control AUC (0.5290) matches the runner's
  independently computed monitor AUC (0.5290).
- A self-paired run gives exactly +0.0000 [+0.0000, +0.0000].
- Unit tests pin the zero-difference, detection, and refusal cases.

The guardrail metrics (diversity/novelty@10, coverage) need candidate article ids. They are
re-derived per impression from the source zips, because the runner preserves each
impression's original candidate order.

**Guardrails implemented and validated (2026-09-12).** How they are built:
- Ids come from the raw zips (MIND `behaviors.tsv`, EB-NeRD `article_ids_inview`), prefixed
  to A1's format. Categories come from A1's processed `articles.parquet`.
- Novelty popularity comes from the **train** split only, via `build_train_popularity`
  (alpha 1.0, Q9). MINDlarge's 83.5M-row train table is filtered to clicked rows in pyarrow
  first, which keeps it ~250 MB instead of ~5 GB on the 8 GB machine.
- MRR/nDCG/top-10 tie-breaks are seeded with A1's processed impression ids, so they match
  A1 exactly.
- Every impression's raw candidate count and labels must equal the runner's, or the script
  refuses (guards against misaligned ids).

Validated on the smoke outputs:
- The alignment check passed on every real impression for both datasets.
- EB-NeRD control diversity@10 0.7885 / novelty@10 17.28 are consistent with A1's
  ebnerd_small figures (0.789 / 17.19).
- MIND values fall in A1's range.
- The `regressed` flag fires correctly: the MIND smoke pair's diversity@10 CI lay entirely
  below zero. That is smoke scale, not a finding.
- 7 unit tests.

Guardrail regression is reported as "paired CI entirely below zero". Coverage@10 is a point
difference only, per ADR-007.

**Revisit** if a TF env becomes cheaply available (Kaggle, or a faster mirror). The
official-code run would then be a validation of Option B's control, not a replacement for
the ablation.

## Q2: why NRMS is the only re-ranker in the true-retrieval evaluation (decided 2026-09-12)

A2 Q2 asks for top-K candidates from A1's generator (K ~ 100–200), re-ranked. Both
leaderboard submissions instead rank each impression's own in-view list, which is what the
competitions score. The true-retrieval evaluation is therefore reported separately and
honestly, and **only NRMS re-ranks in it**.

Candidate K is excluded by construction, not by preference. Its features are
*impression-conditional*: `position_in_view`, `relative_position_in_view`, the five
`*_rank_in_imp` features, `article_age_minus_imp_min`, `lt_embed_sim_minus_imp_max`,
`n_candidates`, and the three `context_*` features are all defined relative to the other
candidates in the list the publisher actually showed. A candidate that A1's retriever
surfaces but that was never shown has no position in that list, no rank among its peers, and
no in-list extrema to be compared against. Any value for them would be invented, and the
invention would silently drive the ranking.

Rejected alternative: a reduced "retrieval-safe" GBDT trained without those features. It
would answer a different question (a different model's ceiling), add a second model to
maintain and explain, and still not be Candidate K. NRMS scores any article from its text
plus the user's history, so it needs no such surgery.

The ceiling is already measured and bounds this evaluation: recall@200 is 2.62% (BM25) and
2.78% (embeddings) on MINDsmall-dev, 2.77% / 1.21% on ebnerd_small. At least ~97% of clicked
articles never enter a top-200 list, so no re-ranker can recover them.

## Risks

- `/share1` may not be mounted on compute nodes. The prep job fails fast on this.
- recommenders' `MINDIterator` is pure Python and may be slow at MINDlarge scale
  (~3.4M training samples per epoch). This will be measured in the first epoch, not assumed.
- xlm-roberta-large's word-embedding table is ~1 GB, against an 11 GB 2080 Ti.

## Revisit when

- The timebox expires without clean training → Option B addendum.
- The reproduced numbers land far from the published ones (beyond scale/split differences) →
  investigate before any treatment claim.
