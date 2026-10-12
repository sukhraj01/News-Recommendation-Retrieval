#!/usr/bin/env python3
"""LightGBM retrained with causal trailing article-CTR features, one model per feedback delay D (ADR-018 addendum).

The fair baseline for the forgetting bandit: same booster recipe as Candidate K's shipped arm
(K_rank_nopos: lambdarank, identical LGB_PARAMS / rounds / early-stopping rule, the same 63 leakage-safe
columns, no lifetime aggregates) PLUS six features computed strictly from labels older than D
(src/bandit/trailing.py). Trailing statistics run over the concatenated train+validation timeline, so
the first validation hours are seeded by the end of train, exactly as a live system would be.

Early stopping uses the last 24 h of the TRAIN window only (validation never used for any decision).
Nothing here modifies Candidate K's artifacts: new models go to experiments/trailing_ctr_gbdt_2026-10-12/
and new score arrays next to the ADR-017 stream.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "scripts"))

import lightgbm as lgb
import numpy as np
import pandas as pd

from bandit_prepare_stream import K_DIR
from run_ebnerd_gbdt_experiment import (EARLY_STOPPING_ROUNDS, LGB_PARAMS, NUM_BOOST_ROUND, _embeddings_paths,
                                        load_behaviors)
from src.bandit.trailing import FEATURE_NAMES_TRAILING, trailing_feature_matrix
from src.evaluation.ranking_metrics import per_impression_auc
from src.retrieval.ebnerd_features import (CATEGORICAL_FEATURES, FEATURE_NAMES, _rank_within_group,
                                           build_feature_frame, build_history_popularity, build_user_profiles,
                                           load_article_table)

STREAM = _REPO / "data" / "processed" / "ebnerd" / "small" / "bandit_stream"
CACHE = _REPO / "data" / "processed" / "ebnerd" / "small" / "trailing_cache"
MODELS = _REPO / "experiments" / "trailing_ctr_gbdt_2026-10-12"
CHUNK = 40_000


def sorted_behaviors(zip_path, split):
    beh = load_behaviors(zip_path, split)
    beh["_t"] = pd.to_datetime(beh["impression_time"])
    return beh.sort_values(["_t", "impression_id"], kind="stable").reset_index(drop=True)


def build_split(zip_path, split, art, keep, prof, pop):
    beh = sorted_behaviors(zip_path, split)
    mats, labels, sizes, arts = [], [], [], []
    for lo in range(0, len(beh), CHUNK):
        X, meta = build_feature_frame(beh.iloc[lo: lo + CHUNK], art, prof, pop, with_labels=True)
        mats.append(np.ascontiguousarray(X[:, keep]))
        labels.append(meta["label"]); sizes.append(meta["group_sizes"]); arts.append(meta["candidate_article_id"])
        print(f"   {split}: {min(lo + CHUNK, len(beh)):,}/{len(beh):,}", flush=True)
    t = (beh["_t"] - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
    return (np.concatenate(mats), np.concatenate(labels), np.concatenate(sizes), np.concatenate(arts), t,
            beh["user_id"].to_numpy(dtype=np.int64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", type=int, nargs="+", default=[0, 900, 1800, 3600])
    args = ap.parse_args()
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
    zip_path = _REPO / "data" / "raw" / "ebnerd" / "ebnerd_small.zip"
    emb_npy, emb_json = _embeddings_paths("small")
    cfg = json.loads((K_DIR / "config.json").read_text())
    assert cfg["feature_names"] == FEATURE_NAMES
    withheld = set(cfg["arms"]["K_rank_nopos"]["withheld"])
    keep = [i for i, n in enumerate(FEATURE_NAMES) if n not in withheld]
    names63 = [FEATURE_NAMES[i] for i in keep]
    assert len(names63) == 63

    if not (CACHE / "tr_X.npy").exists():
        art = load_article_table(zip_path, embeddings_npy=emb_npy, embeddings_json=emb_json)
        for split, tag in (("train", "tr"), ("validation", "va")):
            print(f"[build] {split} ({time.time() - t0:.0f}s)", flush=True)
            prof = build_user_profiles(zip_path, split, art); pop = build_history_popularity(zip_path, split, art)
            X, y, sz, aid, t, uid = build_split(zip_path, split, art, keep, prof, pop)
            np.save(CACHE / f"{tag}_X.npy", X); np.savez(CACHE / f"{tag}_meta.npz", label=y, sizes=sz, article=aid, time=t, user=uid)
            del X
    tr, va = np.load(CACHE / "tr_meta.npz"), np.load(CACHE / "va_meta.npz")
    # the validation arrays must be row-for-row the ADR-017 stream
    st = np.load(STREAM / "impressions.npz")
    assert np.array_equal(va["sizes"], st["sizes"]) and np.array_equal(va["time"], st["time"])
    assert np.array_equal(va["article"], np.load(STREAM / "article_id.npy")) and np.array_equal(va["label"], np.load(STREAM / "label.npy"))
    X_tr = np.load(CACHE / "tr_X.npy"); X_va = np.load(CACHE / "va_X.npy")
    print(f"[load] train {X_tr.shape}, validation {X_va.shape} ({time.time() - t0:.0f}s)", flush=True)

    # ---- one timeline: train then validation -----------------------------------------------------
    sizes_all = np.concatenate([tr["sizes"], va["sizes"]])
    starts_all = np.concatenate([[0], np.cumsum(sizes_all)[:-1]])
    art_all = np.concatenate([tr["article"], va["article"]])
    _, art_idx = np.unique(art_all, return_inverse=True)
    base = float(min(tr["time"].min(), va["time"].min()))
    t_imp = np.concatenate([tr["time"], va["time"]])
    assert np.all(np.diff(t_imp) >= 0), "timeline must be chronological (train ends before validation starts)"
    t_row = np.repeat(np.round(t_imp - base).astype(np.int64), sizes_all)
    y_all = np.concatenate([tr["label"], va["label"]])
    n_tr_rows, n_tr_imp = len(tr["label"]), len(tr["sizes"])

    # early-stopping holdout: last 24 h of train
    tt = tr["time"]; is_stop = tt >= tt.max() - 24 * 3600
    row_stop = np.repeat(is_stop, tr["sizes"])
    fit_imp, stop_imp = np.flatnonzero(~is_stop), np.flatnonzero(is_stop)
    cat_names = [c for c in CATEGORICAL_FEATURES if c in names63]
    out = {"features_base": names63, "features_trailing": FEATURE_NAMES_TRAILING, "per_delay": {}}
    sizes_va = va["sizes"]; label_va = va["label"]

    for D in args.deltas:
        td = time.time()
        F = trailing_feature_matrix(art_idx, t_row, y_all, starts_all, sizes_all, delay_s=D, rank_fn=_rank_within_group)
        names = names63 + FEATURE_NAMES_TRAILING
        cat_idx = [names.index(c) for c in cat_names]
        Xf_tr = np.concatenate([X_tr, F[:n_tr_rows]], axis=1)
        fit_rows, stop_rows = np.flatnonzero(~row_stop), np.flatnonzero(row_stop)
        dtrain = lgb.Dataset(Xf_tr[fit_rows], label=tr["label"][fit_rows], categorical_feature=cat_idx, free_raw_data=False,
                             group=tr["sizes"][fit_imp])
        dstop = lgb.Dataset(Xf_tr[stop_rows], label=tr["label"][stop_rows], categorical_feature=cat_idx, reference=dtrain,
                            free_raw_data=False, group=tr["sizes"][stop_imp])
        params = {**LGB_PARAMS, "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5], "label_gain": [0, 1]}
        booster = lgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND, valid_sets=[dstop],
                            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        del Xf_tr, dtrain, dstop
        Xf_va = np.concatenate([X_va, F[n_tr_rows:]], axis=1)
        score = booster.predict(Xf_va, num_iteration=booster.best_iteration).astype(np.float32)
        np.save(STREAM / f"gbdt_ctr_D{D}_score.npy", score)
        booster.save_model(str(MODELS / f"model_trailing_D{D}.txt"))
        auc = per_impression_auc(score.astype(np.float64), label_va, sizes_va)
        off = np.concatenate([[0], np.cumsum(sizes_va)[:-1]])
        mx = np.maximum.reduceat(score, off); eq = (score == np.repeat(mx, sizes_va)).astype(float)
        top1 = np.add.reduceat(eq * label_va, off) / np.add.reduceat(eq, off)
        imp = dict(sorted(zip(names, booster.feature_importance("gain").tolist()), key=lambda kv: -kv[1]))
        out["per_delay"][str(D)] = {"best_iteration": booster.best_iteration, "stream_auc_all_validation": float(np.nanmean(auc)),
                                    "top1_all_validation": float(top1.mean()), "train_s": time.time() - td,
                                    "top10_gain": dict(list(imp.items())[:10]),
                                    "trailing_gain_share": float(sum(imp[n] for n in FEATURE_NAMES_TRAILING) / sum(imp.values()))}
        print(f"[D={D}s] iter {booster.best_iteration}  AUC {np.nanmean(auc):.4f}  top1 {top1.mean():.4f}  "
              f"trailing-gain share {out['per_delay'][str(D)]['trailing_gain_share']:.3f}  ({time.time() - td:.0f}s)", flush=True)
        del Xf_va, F
    out["lgb_params"] = LGB_PARAMS; out["num_boost_round"] = NUM_BOOST_ROUND; out["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    (_REPO / "results" / "bandit_ebnerd_small" / "drift").mkdir(parents=True, exist_ok=True)
    (_REPO / "results" / "bandit_ebnerd_small" / "drift" / "trailing_gbdt_training.json").write_text(json.dumps(out, indent=1))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
