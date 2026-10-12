#!/usr/bin/env python3
"""End-to-end causality audit of the trailing-CTR GBDT (ADR-018 addendum), using the TRAINED models.

1. Future-label invariance: permute (inside each impression) the label of every validation impression with
   time >= T0 (start of stream day 4), recompute the trailing features, re-score with the SAME model.
   Scores of every row with t <= T0 + D must be bit-identical; scores after must change (non-vacuity).
2. Placebo: replace ALL labels (train+validation) by within-impression permutations when building the
   trailing features, score the trained model: with no real label signal in the features, precision@1 must
   fall back toward the no-trailing level (a model that exploited a hidden channel would not).
"""
import json, sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "scripts"))
import lightgbm as lgb
import numpy as np
import run_bandit_experiment as rbe
import run_drift_experiment as rde
from src.bandit.trailing import trailing_feature_matrix
from src.retrieval.ebnerd_features import _rank_within_group

CACHE = _REPO / "data" / "processed" / "ebnerd" / "small" / "trailing_cache"
MODELS = _REPO / "experiments" / "trailing_ctr_gbdt_2026-10-12"
tr, va = np.load(CACHE / "tr_meta.npz"), np.load(CACHE / "va_meta.npz")
X_va = np.load(CACHE / "va_X.npy")
sizes_all = np.concatenate([tr["sizes"], va["sizes"]]); starts_all = np.concatenate([[0], np.cumsum(sizes_all)[:-1]])
_, art_idx = np.unique(np.concatenate([tr["article"], va["article"]]), return_inverse=True)
t_imp = np.concatenate([tr["time"], va["time"]]); base = t_imp.min()
t_row = np.repeat(np.round(t_imp - base).astype(np.int64), sizes_all)
y_all = np.concatenate([tr["label"], va["label"]]); n_tr_rows = len(tr["label"]); n_tr_imp = len(tr["sizes"])
T0 = np.round(va["time"][0] - base) + 4 * 86400

def permute_within(y, imp_mask, seed):
    rng = np.random.default_rng(seed); y2 = y.copy()
    for i in np.flatnonzero(imp_mask):
        a, s = starts_all[i], sizes_all[i]
        y2[a:a + s] = 0; y2[a + rng.integers(s)] = 1
    return y2

stream = rbe.load_stream()
out = {}
for D in (0, 900, 3600):
    booster = lgb.Booster(model_file=str(MODELS / f"model_trailing_D{D}.txt"))
    def score(y):
        F = trailing_feature_matrix(art_idx, t_row, y, starts_all, sizes_all, delay_s=D, rank_fn=_rank_within_group)
        return booster.predict(np.concatenate([X_va, F[n_tr_rows:]], axis=1)).astype(np.float32)
    s0 = score(y_all)
    imp_future = np.zeros(len(sizes_all), bool); imp_future[n_tr_imp:] = t_imp[n_tr_imp:] - base >= T0
    s1 = score(permute_within(y_all, imp_future, 1))
    row_t = t_row[n_tr_rows:]
    safe = row_t <= T0 + D
    out[f"D={D}"] = {"rows_checked_unaffected": int(safe.sum()), "max_abs_score_diff_unaffected": float(np.abs(s0 - s1)[safe].max()),
                     "rows_after_cutoff_that_changed": float((np.abs(s0 - s1)[~safe] > 0).mean())}
    plac = permute_within(y_all, np.ones(len(sizes_all), bool), 2)
    s2 = score(plac)
    hit = lambda s: rde.topk_hit(s.astype(np.float64), stream).mean()
    out[f"D={D}"]["placebo_top1_all_validation"] = float(hit(s2)); out[f"D={D}"]["real_top1_all_validation"] = float(hit(s0))
    out[f"D={D}"]["original_ranker_top1_all_validation"] = float(hit(stream.lgb_score))
    print(D, out[f"D={D}"], flush=True)
(rde.OUT / "fair_baseline_audit.json").write_text(json.dumps(out, indent=1))
