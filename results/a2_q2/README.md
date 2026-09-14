# A2 Q2 results — the literal retrieve-then-rank harness

Evidence behind ADR-015's Q2 addendum. Produced by
`scripts/a2_q2_retrieve_rerank_eval.py`.

## Files

| File | What |
|---|---|
| `mind_small_bm25.json` | MINDsmall-dev, BM25 retrieval top-200, re-ranked by a locally-trained control NRMS checkpoint |

## Status: an interim, reduced-scale result — not Q3's leaderboard-scale answer

This session's environment cannot reach Ada (SSH rejected outright — see
ADR-015's addendum), so the checkpoint used here is **not** Q3's official
reproduction (AUC 0.6831 on MINDlarge-dev, 10 epochs, full data). It's a
locally-trained control checkpoint, CPU, MINDsmall, 20,000/156,965 train
impressions, 1 epoch (dev monitor AUC 0.5676 — see `checkpoint_results.json`
in the training run's own output dir), built specifically to exercise this
harness end-to-end with a real (not toy) model while Ada is unreachable.

| | Value | 95% CI | n (of 3,000 sampled) |
|---|---:|---|---|
| Hit rate (= recall@200, measured fresh on this sample) | 3.70% | 3.05–4.37% | 3,000 |
| Before (BM25 order) AUC | 0.5483 | 0.5021–0.5957 | 111 |
| After (NRMS re-rank) AUC | 0.5019 | 0.4500–0.5524 | 111 |
| Paired after − before AUC | **−0.0464** | −0.1154, +0.0246 | 111 |

**Not significant** (CI spans zero) and, at this checkpoint's training
scale, directionally NRMS re-ranking does not beat raw BM25 order within the
retrieved 200 — expected: a 1-epoch/20K-impression checkpoint is far short
of the real 10-epoch/full-data reproduction already shown (elsewhere in this
project) to reach AUC 0.6831 on the impression's-own-candidate-list task.
This result demonstrates the harness is correct and honestly reported, not
that NRMS re-ranking doesn't help — that question needs the real checkpoint.

The hit rate (3.70%, CI 3.05–4.37%) is close to, though not exactly
overlapping, ADR-006's independently-measured MINDsmall-dev BM25 recall@200
(2.62%, CI 2.52–2.72%) — a different systematic sample of impressions, same
order of magnitude, consistent with the same underlying retrieval ceiling.

## Re-run at full scale once Ada is reachable

```bash
poetry run python scripts/a2_q2_retrieve_rerank_eval.py \
    --mind-bundle large --mind-dev-zip data/raw/mind/MINDlarge_dev.zip \
    --checkpoint <mind_treatment_v2>/model_weights.pt \
    --abstract-size <0 control / 50 treatment> \
    --retriever bm25 --k 200 --n-impressions 20000 \
    --out results/a2_q2/mind_large_bm25.json
```

This is the result that should replace the table above once
`mind_treatment_v2` (job 2695890, the only full-scale MIND checkpoint any
run will have actually saved — the original `mind_control`/`mind_treatment`
runs predate the weight-saving fix and their weights cannot be recovered)
is reachable. `--abstract-size 50` matches that checkpoint's arm
(treatment); there is no saved full-scale control checkpoint to compare
against unless `mind_control` is also retrained with weight-saving.
