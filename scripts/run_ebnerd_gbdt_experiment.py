#!/usr/bin/env python3
"""Candidate K (ADR-013): LightGBM learning-to-rank over engineered EB-NeRD features.

The question
------------
Every EB-NeRD scorer this project has deployed ranks candidates by content
similarity to the user's history, and the real Codabench leaderboard says that
approach is worth 0.5404 AUC — below the challenge's own naive "most clicks"
baseline (0.5970). RecSys Challenge 2024's "Common Themes" (arXiv:2409.20483)
reports that every top-scoring team instead used a GBDT over engineered
behavioural/temporal features. Does that transfer here, on `ebnerd_small`, with
the leakage discipline ADR-009 established?

Two arms, isolating the objective choice
----------------------------------------
- **K-rank** — `objective="lambdarank"`, one group per impression. This is the
  listwise objective actually suited to the task: EB-NeRD scores each impression
  independently, and exactly one candidate per impression is clicked (verified:
  100% of impressions have >=1 click, mean 1.004). Ranking within the group is
  the whole problem; absolute click probability is irrelevant.
- **K-cls** — `objective="binary"` over the identical feature matrix. This is the
  pointwise framing Candidate H2 used on MIND (`HistGradientBoostingClassifier`).
  Included so the lambdarank result is attributable to the listwise objective
  rather than merely to "a tree model with better features".

Both arms share one feature matrix, so the only difference is the loss.

Baselines measured on the identical validation impressions
----------------------------------------------------------
Marginal CIs across different experiments are not a like-for-like test, so all
three baselines are recomputed here and compared by *paired* bootstrap
(ADR-010's `paired_metric_diff_ci`, imported not reimplemented):

- `embed_sim` — unweighted mean-of-history embedding cosine. This reproduces
  ADR-008's deployed `build_user_embedding_query` semantics, so it stands in for
  the 0.5404 leaderboard submission on local data.
- `popularity` — the leak-safe history-window click count, standing in for the
  challenge's "most clicks" baseline (0.5970 on the real test set).
- `random` — deterministic per-impression pseudo-random scores, the ADR-007
  tie-break substrate. Sanity floor; should land at 0.50.

Early stopping never sees validation
------------------------------------
The stopping set is the last 24 hours of the *training* window, held out
temporally. Using the validation split to choose the tree count would be tuning
against the number being reported — the exact thing ADR-009/ADR-010 rule out for
first attempts, and a milder version of the leak that separated the real
competition's 88.64 from its honest 76.99.

Usage:
    poetry run python scripts/run_ebnerd_gbdt_experiment.py [--bundle small]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_gated_cohort_experiment import paired_metric_diff_ci
from src.evaluation.ranking_metrics import (
    mrr,
    ndcg_at_k,
    per_impression_auc,
    rank_candidates,
    ranking_metric_ci,
)
from src.retrieval.ebnerd_features import (
    BEHAVIOR_COLUMNS,
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    SHORT_TERM_HALF_LIFE_HOURS,
    add_session_position_columns,
    build_feature_frame,
    build_history_popularity,
    build_user_profiles,
    load_article_table,
)

SEED = 0
N_BOOTSTRAP = 2000

# Untuned, a-priori defaults. Chosen from LightGBM's own documented starting
# points for a few-million-row tabular problem, not searched against validation.
LGB_PARAMS: dict = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 8,
    "verbosity": -1,
    "seed": SEED,
}
NUM_BOOST_ROUND = 1500
EARLY_STOPPING_ROUNDS = 75

# `position_in_view` encodes Ekstra Bladet's own in-view ordering. It is
# available at prediction time and is not a temporal leak, but a gain driven by
# it is partly "learned the publisher's existing ranker" rather than "learned
# the user". Measured on ebnerd_small train, clicked candidates sit at mean
# position 5.12 vs 7.92 for non-clicked — far too strong to absorb silently, so
# it gets its own ablation arm rather than a footnote. See ADR-013.
POSITION_FEATURES = ("position_in_view", "relative_position_in_view")

# A2 Q9: "report metrics with and without features unavailable at serving time".
# These two are a DIFFERENT category from ADR-009's `total_inviews` /
# `total_pageviews` / `total_read_time`, which are forbidden outright and never
# read. These are admitted-but-flagged: they describe the page the user was on
# when the in-view list was logged, EB-NeRD ships them in the test set, and they
# are not future information about the candidate — but a page's total read time
# is only fully known once the user leaves it, so a strict production loop would
# not have the final value at request time. ADR-013 called this "the weakest
# such claim in this table" and left it unquantified. This arm quantifies it.
SERVING_UNAVAILABLE_FEATURES = ("context_read_time", "context_scroll_percentage")

# (arm name, LightGBM objective, extra params, features withheld from this arm)
ARMS: tuple[tuple[str, str, dict, tuple[str, ...]], ...] = (
    ("K_rank", "lambdarank", {"metric": "ndcg", "ndcg_eval_at": [5], "label_gain": [0, 1]}, ()),
    ("K_cls", "binary", {"metric": "auc"}, ()),
    ("K_rank_nopos", "lambdarank", {"metric": "ndcg", "ndcg_eval_at": [5], "label_gain": [0, 1]},
     POSITION_FEATURES),
    ("K_rank_noctx", "lambdarank", {"metric": "ndcg", "ndcg_eval_at": [5], "label_gain": [0, 1]},
     SERVING_UNAVAILABLE_FEATURES),
)


def _embeddings_paths(bundle: str) -> tuple[Path, Path]:
    base = _REPO_ROOT / "data" / "processed" / "ebnerd" / bundle / "embeddings"
    stem = "sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2"
    return base / f"{stem}.npy", base / f"{stem}.json"


def load_behaviors(zip_path: Path, split: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        beh = pd.read_parquet(
            io.BytesIO(zf.read(f"{split}/behaviors.parquet")),
            columns=BEHAVIOR_COLUMNS + ["article_ids_clicked"],
        )
    # Computed once here, on the FULL split, before any subsampling/chunking --
    # session_position/session_start_gap_h need every impression in a session
    # to be correct, and both train sampling and validation chunking happen
    # after this point.
    return add_session_position_columns(beh)


def build_user_embedding_matrix(zip_path: Path, split: str, art) -> tuple[np.ndarray, dict]:
    """Per-user mean-of-history embedding, computed once for a whole split.

    Deliberately *unweighted* — this reproduces ADR-008's deployed
    `build_user_embedding_query` semantics (plain mean over the full history, no
    recency or engagement weighting), so the baseline it feeds is comparable to
    the 0.5430 that method scored on this split, rather than to Candidate K's own
    engagement-weighted `lt_embed_sim` feature.

    Split out from the per-chunk scoring because it depends only on the history
    file: recomputing it per chunk would repeat the whole build for every batch.
    """
    hist = _read_history_for_embedding(zip_path, split)
    dim = art.embeddings.shape[1]
    user_vec = np.zeros((len(hist), dim), dtype=np.float32)
    for row, arts in enumerate(hist["article_id_fixed"]):
        pos = [art.id_to_pos.get(int(a), -1) for a in arts]
        pos = [p for p in pos if p >= 0]
        if pos:
            user_vec[row] = art.embeddings[pos].mean(axis=0)
    norms = np.linalg.norm(user_vec, axis=1, keepdims=True)
    user_vec /= np.maximum(norms, 1e-9)
    return user_vec, {int(u): i for i, u in enumerate(hist["user_id"].to_numpy())}


def _read_history_for_embedding(zip_path: Path, split: str) -> pd.DataFrame:
    import io as _io

    from src.utils.io import read_zip_member_bytes

    raw = read_zip_member_bytes(Path(zip_path), f"{split}/history.parquet")
    return pd.read_parquet(_io.BytesIO(raw), columns=["user_id", "article_id_fixed"])


def embed_sim_for_chunk(
    user_vec: np.ndarray, u_index: dict, art, behaviors: pd.DataFrame, meta: dict
) -> np.ndarray:
    """Cosine(candidate, user's mean-history vector) for one chunk of impressions."""
    sizes = meta["group_sizes"]
    imp_row = np.repeat(np.arange(len(behaviors), dtype=np.int64), sizes)
    upos = np.fromiter(
        (u_index.get(int(u), -1) for u in behaviors["user_id"].to_numpy()),
        dtype=np.int64,
        count=len(behaviors),
    )[imp_row]
    cand = meta["candidate_article_id"]
    apos = np.fromiter(
        (art.id_to_pos.get(int(a), -1) for a in cand), dtype=np.int64, count=len(cand)
    )
    out = np.zeros(len(apos), dtype=np.float32)
    sel = np.flatnonzero((upos >= 0) & (apos >= 0))
    for lo in range(0, sel.size, 500_000):
        idx = sel[lo : lo + 500_000]
        out[idx] = np.einsum("ij,ij->i", user_vec[upos[idx]], art.embeddings[apos[idx]])
    return out


def stream_validation_metrics(
    behaviors: pd.DataFrame,
    art,
    prof,
    pop,
    user_vec: np.ndarray,
    u_index: dict,
    *,
    models: dict,
    arm_features: dict,
    chunk_size: int,
    full_metric_cap: int,
) -> pd.DataFrame:
    """Score validation in chunks and return one metrics row per impression.

    The validation feature matrix is built, scored and discarded one chunk at a
    time, so peak memory is bounded by `chunk_size` rather than by the split.
    That is what makes `ebnerd_large` tractable: its validation matrix is
    projected in the tens of GB, which would otherwise have to sit alongside the
    training matrix and LightGBM's binned copy.

    AUC is computed for every impression via the vectorised Mann-Whitney path.
    MRR/nDCG still use ADR-007's exact `rank_candidates` tie-break, which is
    inherently per-impression Python, so they are computed only for the first
    `full_metric_cap` impressions; beyond that they are left NaN and
    `ranking_metric_ci` reports them as skipped rather than averaging them in.
    AUC — the metric Codabench actually scores — is never subsampled.
    """
    rng = np.random.default_rng(SEED)
    pop_col = FEATURE_NAMES.index("log_history_popularity")
    frames: list[pd.DataFrame] = []
    seen = 0

    for start in range(0, len(behaviors), chunk_size):
        sub = behaviors.iloc[start : start + chunk_size]
        X, meta = build_feature_frame(sub, art, prof, pop, with_labels=True)
        labels, sizes = meta["label"], meta["group_sizes"]

        chunk_scores: dict[str, np.ndarray] = {}
        for arm, booster in models.items():
            names = arm_features[arm]
            if len(names) == len(FEATURE_NAMES):
                mat = X
            else:
                mat = X[:, [FEATURE_NAMES.index(n) for n in names]]
            chunk_scores[arm] = booster.predict(mat, num_iteration=booster.best_iteration)
        chunk_scores["popularity"] = X[:, pop_col].astype(np.float64)
        chunk_scores["embed_sim"] = embed_sim_for_chunk(user_vec, u_index, art, sub, meta)
        chunk_scores["random"] = rng.random(X.shape[0])

        exact = seen < full_metric_cap
        frame = pd.DataFrame({
            "impression_id": [str(i) for i in meta["impression_id"]],
            "user_id": sub["user_id"].to_numpy(),
        })
        for label, sc in chunk_scores.items():
            sc = np.asarray(sc, dtype=np.float64)
            frame[f"auc_{label}"] = per_impression_auc(sc, labels, sizes)
            if exact:
                m, n5, n10 = _exact_rank_metrics(sc, labels, meta)
            else:
                nan = np.full(len(sizes), np.nan)
                m, n5, n10 = nan, nan, nan
            frame[f"mrr_{label}"] = m
            frame[f"ndcg5_{label}"] = n5
            frame[f"ndcg10_{label}"] = n10
        frames.append(frame)

        seen += len(sub)
        del X, meta, chunk_scores
        print(
            f"      scored {seen:,}/{len(behaviors):,} validation impressions"
            f"{'' if exact else '  (AUC only past the exact-metric cap)'}",
            flush=True,
        )

    return pd.concat(frames, ignore_index=True)


def _exact_rank_metrics(scores, labels, meta):
    """MRR / nDCG@5 / nDCG@10 using ADR-007's deterministic tie-break, per impression."""
    sizes = meta["group_sizes"]
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    imp_ids = meta["impression_id"]
    m = np.empty(len(sizes)); n5 = np.empty(len(sizes)); n10 = np.empty(len(sizes))
    for i, (start, size) in enumerate(zip(starts, sizes)):
        sl = slice(start, start + size)
        ranked = np.asarray(labels[sl], dtype=bool)[
            rank_candidates(scores[sl], str(imp_ids[i]), seed=SEED)
        ]
        m[i] = mrr(ranked)
        n5[i] = ndcg_at_k(ranked, 5)
        n10[i] = ndcg_at_k(ranked, 10)
    return m, n5, n10


def per_impression_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    meta: dict,
    user_ids: np.ndarray,
    label: str,
) -> pd.DataFrame:
    """One row per impression: AUC / MRR / nDCG@5 / nDCG@10 under `scores`.

    Reuses ADR-007's `rank_candidates` (deterministic per-impression-seeded
    tie-break) and `safe_auc` unchanged, so these numbers sit on the same
    measurement substrate as every other ranking result in this project.
    """
    sizes = meta["group_sizes"]
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    imp_ids = meta["impression_id"]

    rows = []
    for i, (start, size) in enumerate(zip(starts, sizes)):
        s = scores[start : start + size]
        y = labels[start : start + size].astype(bool)
        imp_id = str(imp_ids[i])
        order = rank_candidates(s, imp_id, seed=SEED)
        ranked = y[order]
        rows.append(
            (
                imp_id,
                user_ids[i],
                safe_auc(s, y),
                mrr(ranked),
                ndcg_at_k(ranked, 5),
                ndcg_at_k(ranked, 10),
            )
        )
    return pd.DataFrame(
        rows, columns=["impression_id", "user_id", f"auc_{label}", f"mrr_{label}",
                       f"ndcg5_{label}", f"ndcg10_{label}"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="small", choices=["small", "demo", "large"])
    parser.add_argument("--out", default=None)
    # Explicit path overrides so the identical script runs unchanged on Ada,
    # where the bundle and its embedding cache live under /ssd_scratch rather
    # than in the repo. Keeping one script (rather than an Ada fork) is what
    # stops the cluster run from silently drifting from the validated local one.
    parser.add_argument("--zip-path", default=None)
    parser.add_argument("--embeddings-npy", default=None)
    parser.add_argument("--embeddings-json", default=None)
    parser.add_argument(
        "--score-chunk-size", type=int, default=250_000,
        help="Validation impressions per scoring batch. Bounds peak memory: the "
             "validation feature matrix is built, scored and discarded one batch "
             "at a time rather than materialised whole.",
    )
    parser.add_argument(
        "--full-metric-impressions", type=int, default=1_000_000,
        help="Compute MRR/nDCG exactly for this many validation impressions. AUC "
             "(the Codabench metric) is always computed for ALL impressions; only "
             "the rank metrics, which need ADR-007's per-impression tie-break, are "
             "capped. Impressions beyond the cap report NaN and are counted as "
             "skipped by ranking_metric_ci, never averaged in.",
    )
    parser.add_argument(
        "--train-sample-impressions", type=int, default=None,
        help="Deterministically subsample this many TRAIN impressions before "
             "feature building. Sampling whole impressions (never individual "
             "candidates) keeps every lambdarank group intact.",
    )
    args = parser.parse_args()

    zip_path = Path(args.zip_path) if args.zip_path else (
        _REPO_ROOT / "data" / "raw" / "ebnerd" / f"ebnerd_{args.bundle}.zip"
    )
    if args.embeddings_npy and args.embeddings_json:
        emb_npy, emb_json = Path(args.embeddings_npy), Path(args.embeddings_json)
    else:
        emb_npy, emb_json = _embeddings_paths(args.bundle)
    timings: dict[str, float] = {}

    t0 = time.time()
    art = load_article_table(zip_path, embeddings_npy=emb_npy, embeddings_json=emb_json)
    timings["article_table_s"] = time.time() - t0
    print(f"[1/6] article table: {len(art.category):,} articles ({timings['article_table_s']:.1f}s)", flush=True)

    # Only TRAIN is materialised. Validation is streamed in chunks during
    # evaluation (see `stream_validation_metrics`): at ebnerd_large scale a
    # single validation feature matrix is projected in the tens of GB, and
    # holding it alongside the training matrix and LightGBM's binned copy is
    # what would turn a long, expensive cluster job into an OOM at the last step.
    t0 = time.time()
    tr_prof = build_user_profiles(zip_path, "train", art)
    tr_pop = build_history_popularity(zip_path, "train", art)
    tr_beh = load_behaviors(zip_path, "train")
    if args.train_sample_impressions and args.train_sample_impressions < len(tr_beh):
        rng0 = np.random.default_rng(SEED)
        keep = np.sort(rng0.choice(len(tr_beh), args.train_sample_impressions, replace=False))
        tr_beh = tr_beh.iloc[keep].reset_index(drop=True)
        print(f"      subsampled train to {len(tr_beh):,} impressions (seed {SEED})", flush=True)
    X_tr, meta_tr = build_feature_frame(tr_beh, art, tr_prof, tr_pop, with_labels=True)
    timings["features_train_s"] = time.time() - t0
    print(
        f"[2/6] train: {len(tr_beh):,} impressions -> {X_tr.shape[0]:,} rows x "
        f"{X_tr.shape[1]} feats ({X_tr.nbytes/1e9:.2f} GB, "
        f"{timings['features_train_s']:.1f}s)",
        flush=True,
    )
    tr = {"beh": tr_beh, "X": X_tr, "meta": meta_tr}

    y_tr = tr["meta"]["label"]

    # --- temporal early-stopping holdout: last 24h of the TRAIN window ------
    tr_times = pd.to_datetime(tr["beh"]["impression_time"]).to_numpy()
    cutoff = tr_times.max() - np.timedelta64(24, "h")
    is_stop_imp = tr_times >= cutoff
    sizes_tr = tr["meta"]["group_sizes"]
    row_is_stop = np.repeat(is_stop_imp, sizes_tr)
    print(
        f"[3/6] early-stopping holdout = last 24h of train: "
        f"{int(is_stop_imp.sum()):,}/{len(is_stop_imp):,} impressions "
        f"(cutoff {pd.Timestamp(cutoff)}) — validation split never used for stopping",
        flush=True,
    )

    cat_idx = [FEATURE_NAMES.index(c) for c in CATEGORICAL_FEATURES]
    # Explicit integer index arrays rather than boolean masks. Boolean masking
    # here silently produced mismatched row counts between the feature matrix and
    # the label vector; integer indices make the selection unambiguous, and the
    # assertions below turn any future mismatch into a loud failure instead of a
    # model quietly trained against misaligned labels.
    fit_rows = np.flatnonzero(~row_is_stop)
    stop_rows = np.flatnonzero(row_is_stop)
    n_train_rows = tr["X"].shape[0]
    fit_X, fit_y = tr["X"][fit_rows], y_tr[fit_rows]
    stop_X, stop_y = tr["X"][stop_rows], y_tr[stop_rows]
    fit_groups, stop_groups = sizes_tr[np.flatnonzero(~is_stop_imp)], sizes_tr[np.flatnonzero(is_stop_imp)]

    assert fit_X.shape[0] == fit_y.shape[0] == int(fit_groups.sum()), (
        f"train split misaligned: X={fit_X.shape[0]} y={fit_y.shape[0]} groups={int(fit_groups.sum())}"
    )
    assert stop_X.shape[0] == stop_y.shape[0] == int(stop_groups.sum()), (
        f"stop split misaligned: X={stop_X.shape[0]} y={stop_y.shape[0]} groups={int(stop_groups.sum())}"
    )
    assert fit_X.shape[0] + stop_X.shape[0] == n_train_rows, "split does not partition train rows"

    # `fit_X`/`stop_X` are fancy-indexed COPIES (numpy always copies for
    # integer-array indexing), not views -- so until this point `tr["X"]`,
    # `fit_X` and `stop_X` all coexist, roughly DOUBLING the true cost of this
    # step versus what the matrix's own size suggests (measured directly: at
    # ebnerd_large's real scale, ~10.7GB for tr["X"] alone, another ~10.7GB
    # combined for fit_X+stop_X since together they partition it exactly).
    # `tr["X"]` is never read again past the assertion above (confirmed by
    # grep, not assumed) -- freeing it here is a real, currently-unclaimed
    # ~10GB back before training or validation scoring need the headroom.
    del tr["X"]

    results: dict[str, dict] = {}
    models: dict[str, lgb.Booster] = {}

    arm_features: dict[str, list[str]] = {}
    for arm, objective, extra, dropped in ARMS:
        keep = [i for i, n in enumerate(FEATURE_NAMES) if n not in dropped]
        names = [FEATURE_NAMES[i] for i in keep]
        arm_features[arm] = names
        # Categorical positions are re-derived against this arm's own column
        # order; reusing the full-matrix indices would mislabel columns whenever
        # an arm withholds a feature that sits before a categorical one.
        arm_cat = [names.index(c) for c in CATEGORICAL_FEATURES if c in names]
        take = (lambda M: M[:, keep]) if dropped else (lambda M: M)

        params = {**LGB_PARAMS, "objective": objective, **extra}
        dtrain = lgb.Dataset(
            take(fit_X), label=fit_y, categorical_feature=arm_cat, free_raw_data=False,
            group=fit_groups if objective == "lambdarank" else None,
        )
        dstop = lgb.Dataset(
            take(stop_X), label=stop_y, categorical_feature=arm_cat, reference=dtrain,
            free_raw_data=False,
            group=stop_groups if objective == "lambdarank" else None,
        )
        t0 = time.time()
        booster = lgb.train(
            params, dtrain, num_boost_round=NUM_BOOST_ROUND, valid_sets=[dstop],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        timings[f"train_{arm}_s"] = time.time() - t0
        models[arm] = booster
        note = f" (withheld: {', '.join(dropped)})" if dropped else ""
        print(
            f"[4/6] {arm} ({objective}, {len(names)} feats): "
            f"best_iter={booster.best_iteration} ({timings[f'train_{arm}_s']:.0f}s){note}",
            flush=True,
        )
        del dtrain, dstop

    # `fit_X`/`stop_X`/`fit_y`/`stop_y` are never read again past this point
    # (confirmed by grep, not assumed) -- freeing them here matters because
    # the LAST arm to run (K_rank_nopos, which withholds two features) needs
    # a column-sliced COPY (numpy always copies for fancy/boolean column
    # indexing) that coexists with these originals for the duration of that
    # arm's training. Measured directly on ebnerd_large's real 4M-impression
    # training sample: this combination (~9.9GB fit_X + ~1.6GB stop_X +
    # ~9.6GB sliced copy + ~1.6GB sliced stop copy, roughly 22.7GB just from
    # these four arrays) is what triggered a real oom-kill SLURM logged
    # against the 40G job budget. This alone doesn't eliminate that in-loop
    # peak (freeing happens AFTER the loop, not during the arm that needs
    # the coexistence) -- see the sbatch script's reduced
    # --train-sample-impressions and increased --mem-per-cpu for what
    # actually addresses that peak. This free is what keeps memory down for
    # the validation-scoring phase that follows, which would otherwise stack
    # ~9GB of validation behaviors on top of these ~11.5GB doing nothing.
    del fit_X, stop_X, fit_y, stop_y

    # --- evaluation: stream validation, never materialise it whole ---------
    t0 = time.time()
    va_prof = build_user_profiles(zip_path, "validation", art)
    va_pop = build_history_popularity(zip_path, "validation", art)
    va_beh = load_behaviors(zip_path, "validation")
    user_vec, u_index = build_user_embedding_matrix(zip_path, "validation", art)
    print(
        f"[5/6] scoring {len(va_beh):,} validation impressions in chunks of "
        f"{args.score_chunk_size:,} (arms + embed_sim/popularity/random baselines)",
        flush=True,
    )
    merged = stream_validation_metrics(
        va_beh, art, va_prof, va_pop, user_vec, u_index,
        models=models, arm_features=arm_features,
        chunk_size=args.score_chunk_size,
        full_metric_cap=args.full_metric_impressions,
    )
    timings["scoring_validation_s"] = time.time() - t0
    scores_va = [a[0] for a in ARMS] + ["popularity", "embed_sim", "random"]

    for label in scores_va:
        entry = {}
        for metric in ("auc", "mrr", "ndcg5", "ndcg10"):
            r = ranking_metric_ci(merged, f"{metric}_{label}", N_BOOTSTRAP, SEED)
            entry[metric] = {
                "value": r.metric, "ci_low": r.ci_low, "ci_high": r.ci_high,
                "n_impressions": r.n_impressions, "n_users": r.n_users, "n_skipped": r.n_skipped,
            }
        results[label] = entry

    paired: dict[str, dict] = {}
    for arm in [a[0] for a in ARMS]:
        for base in ("embed_sim", "popularity", "random"):
            d, lo, hi = paired_metric_diff_ci(merged, f"auc_{arm}", f"auc_{base}", N_BOOTSTRAP, SEED)
            paired[f"{arm}_vs_{base}"] = {"auc_diff": d, "ci_low": lo, "ci_high": hi}
    # The position-bias question: how much of K_rank's edge survives without it?
    d, lo, hi = paired_metric_diff_ci(merged, "auc_K_rank", "auc_K_rank_nopos", N_BOOTSTRAP, SEED)
    paired["K_rank_vs_K_rank_nopos"] = {"auc_diff": d, "ci_low": lo, "ci_high": hi}

    # A2 Q9: the cost of dropping the two serving-time-unavailable features,
    # measured on the identical impressions by the same paired test rather than
    # inferred from two marginal CIs. Positive = keeping them helps.
    if "auc_K_rank_noctx" in merged:
        d, lo, hi = paired_metric_diff_ci(
            merged, "auc_K_rank", "auc_K_rank_noctx", N_BOOTSTRAP, SEED)
        paired["K_rank_vs_K_rank_noctx"] = {
            "auc_diff": d, "ci_low": lo, "ci_high": hi,
            "withheld": list(SERVING_UNAVAILABLE_FEATURES),
            "reading": ("positive means the flagged features help; their contribution is "
                        "the price of a claim that is weakest at serving time (ADR-013)"),
        }

    importance = {
        arm: dict(
            sorted(
                zip(arm_features[arm], models[arm].feature_importance("gain").tolist()),
                key=lambda kv: -kv[1],
            )
        )
        for arm in models
    }

    out_dir = Path(args.out) if args.out else (
        _REPO_ROOT / "experiments" / f"candidate_k_gbdt_ebnerd_{args.bundle}_{date.today()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "candidate": "K", "adr": "ADR-013", "bundle": args.bundle,
        "zip_path": str(zip_path), "train_sample_impressions": args.train_sample_impressions,
        "score_chunk_size": args.score_chunk_size,
        "full_metric_impressions": args.full_metric_impressions,
        "lgb_params": LGB_PARAMS, "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "early_stopping_holdout": "last 24h of train window (validation never used)",
        "short_term_half_life_hours": SHORT_TERM_HALF_LIFE_HOURS,
        "n_features": len(FEATURE_NAMES), "feature_names": FEATURE_NAMES,
        "arms": {a[0]: {"objective": a[1], "withheld": list(a[3])} for a in ARMS},
        "categorical_features": CATEGORICAL_FEATURES,
        "seed": SEED, "n_bootstrap": N_BOOTSTRAP,
    }, indent=2))
    (out_dir / "results.json").write_text(json.dumps({
        "metrics": results, "paired_auc_vs_baselines": paired,
        "best_iteration": {a: models[a].best_iteration for a in models},
        "timings_s": timings,
        "n_train_impressions": int(len(tr["beh"])),
        "n_validation_impressions": int(len(va_beh)),
        "reference_points": {
            "deployed_ebnerd_codabench_auc": 0.5404,
            "challenge_most_clicks_baseline": 0.5970,
            "challenge_inview_rate_baseline": 0.5450,
            "challenge_winner_clean_auc": 0.7699,
            "challenge_winner_raw_auc": 0.8924,
            "challenge_2nd_blackpearl_auc": 0.8815,
        },
    }, indent=2))
    (out_dir / "feature_importance.json").write_text(json.dumps(importance, indent=2))
    # Persisted so the test-set run recodes its (different) article corpus onto
    # exactly these codes. Without this the model would be applied to renumbered
    # categoricals and degrade silently. See ADR-013.
    (out_dir / "vocabularies.json").write_text(json.dumps(art.vocabularies(), indent=2))
    for arm, booster in models.items():
        booster.save_model(str(out_dir / f"model_{arm}.txt"))
    merged.to_parquet(out_dir / "per_impression_metrics.parquet", index=False)

    print(f"\n[6/6] === Candidate K — ebnerd_{args.bundle} validation ===")
    print(f"{'arm':<12} {'AUC':>8} {'95% CI':>18} {'MRR':>8} {'nDCG@5':>8} {'nDCG@10':>8}")
    for label in [a[0] for a in ARMS] + ["popularity", "embed_sim", "random"]:
        m = results[label]
        auc = m["auc"]
        ci = f"{auc['ci_low']:.4f}-{auc['ci_high']:.4f}"
        print(
            f"{label:<12} {auc['value']:>8.4f} {ci:>18} "
            f"{m['mrr']['value']:>8.4f} {m['ndcg5']['value']:>8.4f} {m['ndcg10']['value']:>8.4f}"
        )
    print("\nPaired AUC differences (positive = arm beats baseline):")
    for k, v in paired.items():
        sig = "CI-clear" if (v["ci_low"] > 0 or v["ci_high"] < 0) else "not CI-clear"
        print(f"  {k:<28} {v['auc_diff']:+.4f}  [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]  {sig}")
    print(f"\nTop 15 features by gain (K_rank):")
    for name, gain in list(importance["K_rank"].items())[:15]:
        print(f"  {name:<34} {gain:>14,.0f}")
    print(f"\nArtifacts -> {out_dir}")


if __name__ == "__main__":
    main()
