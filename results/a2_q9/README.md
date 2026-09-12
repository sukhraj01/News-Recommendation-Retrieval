# A2 Q9 — metrics with and without features unavailable at serving time

Evidence for ADR-013's 2026-09-12 Q9 addendum. Four-arm `ebnerd_small` run adding
`K_rank_noctx`, which withholds `context_read_time` and `context_scroll_percentage` — the
two features ADR-013 flagged as the weakest serving-time claim (a page's total read time is
only known once the user leaves it).

| Arm | AUC | 95% CI |
|---|---:|---|
| `K_rank` (with) | 0.7581 | 0.7564–0.7600 |
| `K_rank_noctx` (without) | 0.7570 | 0.7552–0.7588 |

**Paired: +0.0011 [+0.0008, +0.0015]** — CI-clear but negligible against the +0.2140 margin
over `embed_sim`. For contrast, withholding position features *improves* AUC by 0.0007, so
neither flagged group is load-bearing; freshness is.

`K_rank` and `K_rank_nopos` reproduced bit-identically against the three-arm run
(0.758123 / 0.758808), confirming the arms are independent and the run is deterministic.

Boosters (19 MB, not tracked) are durable at `~/a2_model_artifacts/candidate_k_q9_noctx_2026-09-12/`.
