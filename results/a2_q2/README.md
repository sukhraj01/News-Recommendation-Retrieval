# A2 Q2 results — the literal retrieve-then-rank harness

Evidence behind ADR-015's Q2 addenda. Produced by
`scripts/a2_q2_retrieve_rerank_eval.py`.

## Files

| File | What |
|---|---|
| `mind_large_bm25.json` | **The real answer.** MINDlarge-dev, BM25 retrieval top-200, re-ranked by the actual full-scale treatment checkpoint (`mind_treatment_v2`, job 2695890: 10 epochs, dev monitor AUC 0.6868, matching the already-evaluated Q3 result to 4 decimal places) |
| `mind_small_bm25.json` | Superseded interim check (below) — kept for the record, not the reported answer |

## The real result (2026-09-17)

Full-scale checkpoint, 8,000 sampled MINDlarge-dev impressions, BM25 top-200 retrieval:

| | Value | 95% CI | n |
|---|---:|---|---|
| Hit rate (= recall@200, measured fresh on this sample) | 3.21% | 2.83–3.61% | 8,000 |
| Before (BM25 order) AUC | 0.5428 | 0.5086–0.5753 | 257 (hits only) |
| After (NRMS re-rank) AUC | **0.7224** | 0.6890–0.7542 | 257 (hits only) |
| Paired after − before AUC | **+0.1796** | +0.1375, +0.2213 | CI-clear |
| Paired after − before MRR | +0.0034 | +0.0022, +0.0045 | CI-clear |
| Paired after − before nDCG@5 | +0.1113 | +0.0757, +0.1483 | CI-clear |
| Paired after − before nDCG@10 | +0.1359 | +0.1016, +0.1740 | CI-clear |

**A large, unambiguous, CI-clear win for NRMS re-ranking.** Within the ~3.2% of
impressions where the retriever's top-200 actually contains the real click, the
trained NRMS reorders those 200 candidates dramatically better than raw BM25
lexical-score order — AUC rises from barely-above-random (0.5428) to a strong
0.7224. This directly confirms the interim reduced-scale check's own stated
caveat: that earlier result (paired AUC −0.0464, not significant) was evidence
about a 1-epoch/20,000-impression undertrained checkpoint, not evidence that
re-ranking fails. With the real, fully-trained model, it clearly helps.

**What this does and does not say about the two-stage pipeline overall.** The
~96.8% of impressions where the retriever never surfaces the true click at all
are structurally unrecoverable by any re-ranker (ADR-006/008's own recall@200
ceiling) — re-ranking is a real, large improvement *conditional on* retrieval
succeeding, not a fix for retrieval's own miss rate. Both facts are part of the
honest answer to Q2.

The hit rate (3.21%, CI 2.83–3.61%) is consistent with ADR-006's independently-
measured MINDlarge-scale BM25 recall@200 citation and with the MINDsmall-dev
interim check below (3.70%, CI 3.05–4.37%) — same order of magnitude across two
different samples and two different corpus scales, as expected.

## Superseded: the interim, reduced-scale result (2026-09-14)

Produced while this session's environment could not reach Ada. Checkpoint: a
locally-trained control, CPU, MINDsmall, 20,000/156,965 train impressions, 1
epoch (dev monitor AUC 0.5676 — barely-trained by construction, unlike the real
checkpoint above).

| | Value | 95% CI | n (of 3,000 sampled) |
|---|---:|---|---|
| Hit rate (= recall@200) | 3.70% | 3.05–4.37% | 3,000 |
| Before (BM25 order) AUC | 0.5483 | 0.5021–0.5957 | 111 |
| After (NRMS re-rank) AUC | 0.5019 | 0.4500–0.5524 | 111 |
| Paired after − before AUC | −0.0464 | −0.1154, +0.0246 | not significant |

Kept here as the historical record of what was reported and why (an honest
interim check, not a substitute), per this project's decision-reversal
principle — not overwritten, superseded by the real result above.
