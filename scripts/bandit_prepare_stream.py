#!/usr/bin/env python3
"""Materialise the chronological bandit stream for ebnerd_small validation (ADR-017).

One-off, read-only with respect to Candidate K: loads its frozen boosters, runs the
EXISTING `build_feature_frame`, and writes arrays under data/processed/ (git-ignored).

Outputs (data/processed/ebnerd/small/bandit_stream/):
  features.npy   float32 (rows, d0+2)  linear features + 2 hybrid columns
  lgb_score.npy  float32 (rows,)       K_rank_nopos (the shipped ranker) raw score
  sim_prob.npy   float32 (rows,)       K_cls click prob, normalised inside the impression
  label.npy / article_id.npy / prior_exposure.npy   per candidate row
  impressions.npz   sizes, time, user_id, impression_id, history_len
  meta.json         transformer state, temperature, sigma_c, sanity AUC

Scaling statistics, the softmax temperature and sigma_c are fitted on the last 24 h of
the TRAIN window only. Validation is never used to fit anything here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_ebnerd_gbdt_experiment import _embeddings_paths, load_behaviors
from src.bandit.features import (
    LinearFeatureTransformer,
    assert_leakage_safe,
    fit_softmax_temperature,
    hybrid_columns,
)
from src.evaluation.ranking_metrics import per_impression_auc
from src.retrieval.ebnerd_features import (
    FEATURE_NAMES,
    build_feature_frame,
    build_history_popularity,
    build_user_profiles,
    load_article_table,
)

K_DIR = _REPO / "experiments" / "candidate_k_gbdt_ebnerd_small_2026-09-12"
CHUNK = 40_000


def load_boosters() -> tuple[dict, dict]:
    cfg = json.loads((K_DIR / "config.json").read_text())
    assert cfg["feature_names"] == FEATURE_NAMES, "Candidate K's feature order drifted"
    boosters, cols = {}, {}
    for arm in ("K_rank_nopos", "K_cls"):
        boosters[arm] = lgb.Booster(model_file=str(K_DIR / f"model_{arm}.txt"))
        withheld = set(cfg["arms"][arm]["withheld"])
        cols[arm] = [i for i, n in enumerate(FEATURE_NAMES) if n not in withheld]
    return boosters, cols


def score(boosters, cols, X, arm):
    return boosters[arm].predict(X[:, cols[arm]] if len(cols[arm]) != X.shape[1] else X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="small")
    ap.add_argument("--limit", type=int, default=None, help="dry-run: first N chronological validation impressions")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    zip_path = _REPO / "data" / "raw" / "ebnerd" / f"ebnerd_{args.bundle}.zip"
    out = Path(args.out) if args.out else _REPO / "data" / "processed" / "ebnerd" / args.bundle / "bandit_stream"
    out.mkdir(parents=True, exist_ok=True)
    emb_npy, emb_json = _embeddings_paths(args.bundle)
    boosters, cols = load_boosters()

    t0 = time.time()
    art = load_article_table(zip_path, embeddings_npy=emb_npy, embeddings_json=emb_json)

    # ---- fit-side statistics: last 24h of the TRAIN window ----------------
    tr_beh = load_behaviors(zip_path, "train")
    tr_prof = build_user_profiles(zip_path, "train", art)
    tr_pop = build_history_popularity(zip_path, "train", art)
    tt = pd.to_datetime(tr_beh["impression_time"])
    fit_beh = tr_beh[tt >= tt.max() - pd.Timedelta(hours=24)].reset_index(drop=True)
    X_fit, meta_fit = build_feature_frame(fit_beh, art, tr_prof, tr_pop, with_labels=True)
    transformer = LinearFeatureTransformer({
        "category_code": len(art.category_vocab),
        "article_type": len(art.article_type_vocab),
        "sentiment_label": len(art.sentiment_label_vocab),
    }).fit(X_fit)
    assert_leakage_safe(transformer.source_names)
    s_fit = score(boosters, cols, X_fit, "K_rank_nopos")
    sizes_fit = meta_fit["group_sizes"]
    tau = fit_softmax_temperature(s_fit, meta_fit["label"], sizes_fit)
    starts_fit = np.concatenate([[0], np.cumsum(sizes_fit)[:-1]])
    centred = s_fit - np.repeat(np.add.reduceat(s_fit, starts_fit) / sizes_fit, sizes_fit)
    sigma_c = float(centred.std())
    print(f"[fit] {len(fit_beh):,} train-tail impressions; d0={transformer.dim}  tau={tau:.4f}  sigma_c={sigma_c:.4f}"
          f"  ({time.time()-t0:.0f}s)", flush=True)

    # exposure counts from TRAIN (strictly earlier in time than any validation impression)
    from collections import Counter
    exposure: Counter = Counter()
    for iv in tr_beh["article_ids_inview"]:
        exposure.update(int(a) for a in iv)
    del X_fit, tr_beh, tr_prof, tr_pop

    # ---- validation stream, chronological ---------------------------------
    va_beh = load_behaviors(zip_path, "validation")
    va_prof = build_user_profiles(zip_path, "validation", art)
    va_pop = build_history_popularity(zip_path, "validation", art)
    va_beh["_t"] = pd.to_datetime(va_beh["impression_time"])
    va_beh = va_beh.sort_values(["_t", "impression_id"], kind="stable").reset_index(drop=True)
    if args.limit:
        va_beh = va_beh.iloc[: args.limit].reset_index(drop=True)
    n_imp = len(va_beh)
    total_rows = int(va_beh["article_ids_inview"].map(len).sum())
    d0 = transformer.dim
    H = np.lib.format.open_memmap(out / "features.npy", mode="w+", dtype=np.float32, shape=(total_rows, d0 + 2))
    lgb_score = np.zeros(total_rows, np.float32)
    sim_prob = np.zeros(total_rows, np.float32)
    label = np.zeros(total_rows, np.int8)
    article_id = np.zeros(total_rows, np.int64)
    prior_exposure = np.zeros(total_rows, np.int32)
    sizes_all = np.zeros(n_imp, np.int64)
    hist_len = np.zeros(n_imp, np.float64)

    row = 0
    for lo in range(0, n_imp, CHUNK):
        sub = va_beh.iloc[lo: lo + CHUNK]
        X, meta = build_feature_frame(sub, art, va_prof, va_pop, with_labels=True)
        sizes = meta["group_sizes"]
        r = X.shape[0]
        s_nopos = score(boosters, cols, X, "K_rank_nopos").astype(np.float32)
        p_cls = score(boosters, cols, X, "K_cls").astype(np.float64)
        st = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        p_norm = p_cls / np.repeat(np.add.reduceat(p_cls, st), sizes)
        base = transformer.transform(X)
        H[row: row + r, :d0] = base
        H[row: row + r, d0:] = hybrid_columns(s_nopos, sizes, tau, sigma_c)
        lgb_score[row: row + r] = s_nopos
        sim_prob[row: row + r] = p_norm
        label[row: row + r] = meta["label"]
        article_id[row: row + r] = meta["candidate_article_id"]
        sizes_all[lo: lo + len(sub)] = sizes
        lh = X[st, FEATURE_NAMES.index("log_history_len")].astype(np.float64)
        hist_len[lo: lo + len(sub)] = np.where(np.isfinite(lh), np.expm1(lh), 0.0)
        # prior exposure: train + earlier validation impressions (same-time ties by sorted order)
        pos = row
        for iv in sub["article_ids_inview"]:
            for a in iv:
                prior_exposure[pos] = exposure[int(a)]
                pos += 1
            exposure.update(int(a) for a in iv)
        row += r
        print(f"[stream] {min(lo + CHUNK, n_imp):,}/{n_imp:,} impressions  ({time.time()-t0:.0f}s)", flush=True)
    H.flush()
    assert row == total_rows

    np.save(out / "lgb_score.npy", lgb_score); np.save(out / "sim_prob.npy", sim_prob)
    np.save(out / "label.npy", label); np.save(out / "article_id.npy", article_id)
    np.save(out / "prior_exposure.npy", prior_exposure)
    np.savez(
        out / "impressions.npz",
        sizes=sizes_all,
        time=(va_beh["_t"] - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(),
        user_id=va_beh["user_id"].to_numpy(dtype=np.int64),
        impression_id=va_beh["impression_id"].to_numpy(dtype=np.int64),
        history_len=hist_len,
    )
    auc = per_impression_auc(lgb_score.astype(np.float64), label, sizes_all)
    top1 = np.array([label[o + int(np.argmax(lgb_score[o: o + s]))] for o, s in
                     zip(np.concatenate([[0], np.cumsum(sizes_all)[:-1]]), sizes_all)])
    meta_out = {
        "bundle": args.bundle, "n_impressions": n_imp, "n_rows": total_rows, "base_dim": d0,
        "feature_names": transformer.feature_names, "transformer": transformer.state(),
        "softmax_tau": tau, "sigma_c": sigma_c,
        "ranker": "K_rank_nopos", "simulator_model": "K_cls", "k_dir": str(K_DIR),
        "sanity": {"lgb_mean_per_impression_auc": float(np.nanmean(auc)), "lgb_top1_ctr": float(top1.mean()),
                   "random_ctr": float(np.mean(1.0 / sizes_all))},
        "time_range": [str(va_beh["_t"].iloc[0]), str(va_beh["_t"].iloc[-1])],
        "limit": args.limit,
    }
    (out / "meta.json").write_text(json.dumps(meta_out, indent=2))
    print(json.dumps(meta_out["sanity"], indent=2), f"\ndone in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
