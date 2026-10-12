#!/usr/bin/env python3
"""Why does the hybrid bandit beat LightGBM-greedy? Feature-group ablation (ADR-017). TUNING PERIOD ONLY.

Hypothesis: continuous per-article metadata (body_len, title_len, sentiment_score, n_topics,
n_entities, is_premium + category/type/sentiment one-hots) jointly act as a pseudo article-ID, so a
shared-theta linear model can learn which *articles are being clicked right now* from its own
past rewards (real-time popularity -- a signal Candidate K cannot have, because its popularity
feature is history-window only by design). Zeroing those columns removes the pseudo-ID; if the gain
persists, it is not that mechanism.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "scripts"))
import numpy as np
import run_bandit_experiment as rbe
from src.bandit.runner import run_prequential
from src.bandit.stream import ImpressionStream

stream = rbe.load_stream(); tune, _, _ = rbe.split_periods(stream); d0 = tune.base_dim
names = json.loads((rbe.STREAM_DIR / "meta.json").read_text())["feature_names"] + ["lgb_prob", "lgb_centred"]
W = json.loads((rbe.OUT_DIR / "tune.json").read_text())["winners"]
w = W["hybrid_linucb|per_impression"]
fingerprints = ["title_len", "body_len", "sentiment_score", "n_topics", "n_entities", "is_premium"]
onehots = [n for n in names if n.startswith(("category_code=", "article_type=", "sentiment_label="))]
H = np.array(tune.features)  # in-memory copy of the tuning window
groups = {
    "full hybrid (reference)": [],
    "A: zero static-metadata fingerprints": fingerprints,
    "B: A + category/type/sentiment one-hots": fingerprints + onehots,
    "C: LightGBM columns + bias only": [n for n in names[:d0] if n != "bias"],
}
lgb = np.mean(run_prequential(rbe.make_policy("lgbm_greedy", {}, d0, 0), tune).reward)
out = {"lgbm_greedy_tune_ctr": float(lgb), "config": w["params"], "variants": {}}
print("LGB", lgb)
for tag, drop in groups.items():
    X = H.copy()
    idx = [names.index(n) for n in drop]
    X[:, idx] = 0.0
    s = ImpressionStream(features=X, lgb_score=tune.lgb_score, label=tune.label, sizes=tune.sizes, time=tune.time,
                         user_id=tune.user_id, article_id=tune.article_id, history_len=tune.history_len, base_dim=d0)
    r = run_prequential(rbe.make_policy("hybrid_linucb", w["params"], d0, 0), s).reward
    days = ((tune.time - tune.time[0]) // 86400).astype(int)
    out["variants"][tag] = {"tune_ctr": float(r.mean()), "n_zeroed": len(idx), "by_day": [float(r[days == d].mean()) for d in range(3)]}
    print(f"{tag:45s} ctr {r.mean():.4f}  by-day {[round(float(r[days==d].mean()),4) for d in range(3)]}", flush=True)
(rbe.OUT_DIR / "mechanism_ablation.json").write_text(json.dumps(out, indent=1))
