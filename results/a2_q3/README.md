# A2 Q3 results — reproduced official NRMS baselines and the EB-NeRD A/B

Evidence behind every number in `decisions/ADR-015-a2-official-baseline-reproduction.md`.
Small JSON only: the per-impression score files are 132 MB and stay out of git (Q8), but
their sha256 are recorded below so the large artifacts remain verifiable.

## Files

| File | What |
|---|---|
| `mind_control_eval.json` | Harness metrics + CIs + guardrails, MIND control, all 376,471 MINDlarge-dev impressions |
| `ebnerd_control_eval.json` | Same, EB-NeRD control, all 244,647 `ebnerd_small` validation impressions |
| `ebnerd_ab_paired_eval.json` | **The A/B verdict**: both arms plus paired bootstrap CIs and guardrails |
| `*_runner_results.json` | Per-epoch training traces from the runner (loss, val AUC, lr, freshness weight, timings, hparams) |

## Provenance

| Run | Ada job | Cost | Headline |
|---|---|---|---|
| MIND control | 2694501 | 13 h 13 m, 10 epochs | AUC 0.6831 (CI 0.6821–0.6839) |
| EB-NeRD control | 2694962 | 26 m 43 s, 5 epochs, best at epoch 3 | AUC 0.5613 (CI 0.5597–0.5629) |
| EB-NeRD treatment | 2694991 | 29 m 58 s, 5 epochs, best at epoch 1 | AUC 0.5677; paired Δ +0.0064 [+0.0053, +0.0075] |

Produced by `scripts/a2_nrms_official_run.py` (training/scoring, on Ada) and
`scripts/a2_evaluate_scores.py` (metrics, CIs, paired test, guardrails, locally).

## Excluded score files (sha256)

Durable copies: `~/a2_model_artifacts/a2_q3_results/` and Ada `$HOME/a2/results/`.

```
6f2bc39a1aed5c6ab006e344ceaa7d7b6de09b44ca9e12d70697263a6f173601  ebnerd_control_scores.parquet
bbeaffaec19c15bd9896fda6ca8d248fd1ebbd22839bbdbf054348e0f256ef05  ebnerd_treatment_scores.parquet
bf0a730fd4495f06f2b4f19c583d0ba7dd4f0a81c3a49403a9e026aad47c8339  mind_control_scores.parquet
```

## Q5 slices (added 2026-09-12)

| File | What |
|---|---|
| `ebnerd_ab_sliced_eval.json` | EB-NeRD A/B with head/tail + warm/cold slices and per-slice paired CIs |
| `mind_control_sliced_eval.json` | MIND control, same slices |

Definition and findings: ADR-007's 2026-09-12 addendum. Headline: EB-NeRD's freshness gain
**reverses on head articles** (head Δ−0.0217 CI-clear loss vs tail Δ+0.0076 gain), so the
+0.0064 aggregate is a tail gain diluted by a head regression.
