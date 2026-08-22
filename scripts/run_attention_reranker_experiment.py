#!/usr/bin/env python3
"""Candidate I (2026-08-22 session, ADR-011): a small, end-to-end-trained
neural re-ranker — candidate-aware attention pooling over a user's RAW
history embedding sequence (`src/retrieval/rerank.py::AttentionScorer`),
using the already-computed, frozen MiniLM embeddings as the only input
(the text encoder itself is never fine-tuned; only the attention layer is
trained). Motivated directly by ADR-010's Candidate H addendum: two model
classes (linear, nonlinear-tree) over a fixed set of precomputed scalar
similarity scores both lost against the deployed baseline, pointing at the
*feature set* — not the combiner's model class — as the likely ceiling.
This candidate tests that directly by operating on the raw embeddings
themselves instead of a handful of scalars derived from them.

This project's first neural-network *training* (every other `torch` usage
is frozen `sentence-transformers` inference only, see `embed.py`). Trained
on MINDsmall-**train** only, with a 95/5 user-level holdout carved from
train (seed=0) used for epoch-level early stopping — MINDsmall-**dev** is
touched exactly once, at the very end, for the real reported number,
matching every other candidate's train/dev discipline in this project.

Per the objective's explicit instruction, this script's real run stops at
the MINDsmall-dev checkpoint and reports the result — it does not proceed
to MINDlarge regardless of outcome; that decision belongs to the engineer.

Usage:
    # Benchmark a small subset first (per CLAUDE.md's benchmarking
    # philosophy - measure before committing to a long run):
    poetry run python scripts/run_attention_reranker_experiment.py \
        --max-train-impressions 2000 --epochs 1

    # Real run:
    poetry run python scripts/run_attention_reranker_experiment.py
"""
import json
import sys
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from run_bm25_experiment import COLD_THRESHOLD, dataset_paths
from run_gated_cohort_experiment import paired_metric_diff_ci
from run_learned_combiner_experiment import _train_paths
from src.evaluation.ranking_metrics import (
    build_train_popularity, mrr, ndcg_at_k, rank_candidates, ranking_metric_ci, safe_auc,
)
from src.retrieval.embed import DEFAULT_MODEL, build_embedding_index, model_slug
from src.retrieval.rerank import AttentionScorer, build_user_history_vectors
from src.retrieval.score import EmbeddingScorer

N_BOOTSTRAP = 2000
SEED = 0
METRIC_COLUMNS = ["auc", "mrr", "ndcg5", "ndcg10"]
BASELINE_AUC_OVERALL = 0.63399337223849
HOLDOUT_FRACTION = 0.05
D_ATTN = 64
LR = 1e-3
MINIBATCH = 64
MAX_EPOCHS = 5


def _build_side(paths: dict) -> dict:
    """Lean side-builder: only frozen embeddings + raw history sequences.
    Deliberately does NOT reuse `run_learned_combiner_experiment.py::_build_side`
    (which also builds a BM25 index and symbolic-feature lookups this
    candidate doesn't use, per the objective's "frozen embeddings as
    input" scope)."""
    articles = pd.read_parquet(paths["articles"])
    history = pd.read_parquet(paths["history"])
    impressions = pd.read_parquet(paths["impressions"])

    cache_path = paths["articles"].parent / "embeddings" / f"{model_slug(DEFAULT_MODEL)}.npy"
    embed_index = build_embedding_index(articles, model_name=DEFAULT_MODEL, cache_path=cache_path)
    vector_lookup = dict(zip(embed_index.article_ids, embed_index.vectors))
    dim = embed_index.vectors.shape[1]

    history_vectors_by_user: dict[str, np.ndarray] = {}
    history_len_by_user: dict[str, int] = {}
    for row in history.itertuples(index=False):
        history_vectors_by_user[row.user_id] = build_user_history_vectors(row.article_ids, vector_lookup)
        history_len_by_user[row.user_id] = len(row.article_ids)

    return {
        "impressions": impressions,
        "embed_index": embed_index,
        "vector_lookup": vector_lookup,
        "history_vectors_by_user": history_vectors_by_user,
        "history_len_by_user": history_len_by_user,
        "dim": dim,
    }


def _resolve_candidates(candidate_ids: list[str], clicked: np.ndarray, vector_lookup: dict):
    """Gathers candidate vectors, skipping ids absent from `vector_lookup`
    (the "skip what's missing" convention every builder in this project
    uses) - not expected to trigger for MINDsmall train/dev (confirmed
    clean elsewhere in this project; the documented `N89741`-style gap is
    MINDlarge_test-only), but handled defensively rather than assumed.
    Returns `(None, None, None)` if zero candidates resolve, else
    `(vecs, resolved_candidate_ids, clicked_r)` - the resolved id list is
    needed (not just vecs/labels) so callers can look up per-candidate
    side information, e.g. popularity, that stays aligned after filtering."""
    if all(c in vector_lookup for c in candidate_ids):
        return np.stack([vector_lookup[c] for c in candidate_ids]).astype(np.float32), candidate_ids, clicked
    keep = [i for i, c in enumerate(candidate_ids) if c in vector_lookup]
    if not keep:
        return None, None, None
    vecs = np.stack([vector_lookup[candidate_ids[i]] for i in keep]).astype(np.float32)
    resolved_ids = [candidate_ids[i] for i in keep]
    return vecs, resolved_ids, clicked[keep]


def _history_for(side: dict, user_id: str) -> np.ndarray:
    return side["history_vectors_by_user"].get(user_id, np.empty((0, side["dim"]), dtype=np.float32))


def _log_popularity_tensor(candidate_ids: list[str], popularity: dict | None) -> torch.Tensor | None:
    """`popularity` (train-split-only, `build_train_popularity`'s
    Laplace-smoothed dict) -> a `(K,)` log-popularity tensor per
    candidate, same `np.log(popularity[c])` convention Candidates F/H1/H2
    already use, or `None` if popularity isn't enabled for this run."""
    if popularity is None:
        return None
    return torch.from_numpy(
        np.log(np.array([popularity[c] for c in candidate_ids], dtype=np.float32))
    )


def _user_level_holdout_split(impressions: pd.DataFrame, fraction: float, seed: int) -> tuple[set, set]:
    users = impressions["user_id"].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(users)
    n_holdout = max(1, int(fraction * len(shuffled)))
    return set(shuffled[n_holdout:]), set(shuffled[:n_holdout])  # (fit_users, holdout_users)


def _train_one_epoch(
    model: AttentionScorer, groups: list, side: dict, optimizer, bce, minibatch: int,
    rng: np.random.Generator, popularity: dict | None = None,
) -> float:
    order = rng.permutation(len(groups))
    optimizer.zero_grad()
    accum = 0
    loss_sum, loss_count = 0.0, 0
    for idx in order:
        (user_id, _impression_id), group = groups[idx]
        candidate_ids = group["article_id"].tolist()
        clicked = group["clicked"].to_numpy(dtype=bool)
        vecs, resolved_ids, clicked_r = _resolve_candidates(candidate_ids, clicked, side["vector_lookup"])
        if vecs is None:
            continue
        history_vecs = _history_for(side, user_id)
        if history_vecs.shape[0] == 0 and popularity is None:
            # AttentionScorer returns a constant all-zero score for a
            # zero-history user with no popularity term (the project-wide
            # cold-start convention, see rerank.py) - that output doesn't
            # depend on Wq/Wk at all, so there is no gradient to learn from
            # these impressions. Skipping them during training is a no-op
            # for the model's learned behavior, not an approximation;
            # they're still scored normally (as a 0/tie) at eval time.
            # When popularity IS enabled, these impressions DO have a real
            # gradient path (via pop_weight) and must NOT be skipped - see
            # ADR-011's addendum.
            continue
        cand_t = torch.from_numpy(vecs)
        hist_t = torch.from_numpy(history_vecs)
        labels_t = torch.from_numpy(clicked_r.astype(np.float32))
        log_pop_t = _log_popularity_tensor(resolved_ids, popularity)

        logits = model(cand_t, hist_t, log_popularity=log_pop_t)
        loss = bce(logits, labels_t) / minibatch
        loss.backward()
        loss_sum += loss.item() * minibatch
        loss_count += 1
        accum += 1
        if accum == minibatch:
            optimizer.step()
            optimizer.zero_grad()
            accum = 0
    if accum > 0:
        optimizer.step()
        optimizer.zero_grad()
    return loss_sum / max(loss_count, 1)


def _evaluate(
    model: AttentionScorer, side: dict, cold_threshold: int, users_filter: set | None = None,
    popularity: dict | None = None,
) -> pd.DataFrame:
    """Holdout/train-internal evaluation only — no baseline column (early
    stopping only needs this candidate's own AUC). MINDsmall-dev's final
    evaluation uses `_evaluate_dev_with_baseline` instead, which also
    computes the deployed baseline's AUC on the same impressions for the
    paired bootstrap."""
    model.eval()
    impressions = side["impressions"]
    if users_filter is not None:
        impressions = impressions[impressions["user_id"].isin(users_filter)]
    impressions_sorted = impressions.sort_values(["user_id", "impression_id"], kind="stable")

    rows = []
    with torch.no_grad():
        for (user_id, impression_id), group in impressions_sorted.groupby(["user_id", "impression_id"], sort=False):
            candidate_ids = group["article_id"].tolist()
            clicked = group["clicked"].to_numpy(dtype=bool)
            vecs, resolved_ids, clicked_r = _resolve_candidates(candidate_ids, clicked, side["vector_lookup"])
            if vecs is None:
                continue
            log_pop_t = _log_popularity_tensor(resolved_ids, popularity)
            scores = model(
                torch.from_numpy(vecs), torch.from_numpy(_history_for(side, user_id)), log_popularity=log_pop_t
            ).numpy()

            hlen = side["history_len_by_user"].get(user_id, 0)
            cohort = "cold" if hlen < cold_threshold else "warm"
            order = rank_candidates(scores, impression_id, seed=SEED)
            ranked_clicked = clicked_r[order]
            rows.append({
                "impression_id": impression_id, "user_id": user_id, "cohort": cohort,
                "auc": safe_auc(scores, clicked_r),
                "mrr": mrr(ranked_clicked),
                "ndcg5": ndcg_at_k(ranked_clicked, 5),
                "ndcg10": ndcg_at_k(ranked_clicked, 10),
            })
    model.train()
    return pd.DataFrame(rows)


def _evaluate_dev_with_baseline(
    model: AttentionScorer, side: dict, cold_threshold: int, popularity: dict | None = None,
) -> pd.DataFrame:
    """Dev-only evaluation that also computes the deployed baseline's own
    AUC for the SAME impressions (needed for the paired bootstrap), same
    convention Candidate G's `auc`/`auc_baseline` columns established."""
    model.eval()
    vector_lookup = side["vector_lookup"]
    embed_scorer = EmbeddingScorer(side["embed_index"])
    embed_query_by_user: dict[str, np.ndarray | None] = {}

    impressions_sorted = side["impressions"].sort_values(["user_id", "impression_id"], kind="stable")
    rows = []
    with torch.no_grad():
        for (user_id, impression_id), group in impressions_sorted.groupby(["user_id", "impression_id"], sort=False):
            candidate_ids = group["article_id"].tolist()
            clicked = group["clicked"].to_numpy(dtype=bool)
            vecs, resolved_ids, clicked_r = _resolve_candidates(candidate_ids, clicked, vector_lookup)
            if vecs is None:
                continue
            log_pop_t = _log_popularity_tensor(resolved_ids, popularity)
            scores = model(
                torch.from_numpy(vecs), torch.from_numpy(_history_for(side, user_id)), log_popularity=log_pop_t
            ).numpy()

            if user_id not in embed_query_by_user:
                embed_query_by_user[user_id] = _mean_pool(side, user_id, vector_lookup)
            embed_scores = embed_scorer.score(embed_query_by_user[user_id], candidate_ids)

            hlen = side["history_len_by_user"].get(user_id, 0)
            cohort = "cold" if hlen < cold_threshold else "warm"
            order = rank_candidates(scores, impression_id, seed=SEED)
            ranked_clicked = clicked_r[order]
            rows.append({
                "impression_id": impression_id, "user_id": user_id, "cohort": cohort,
                "auc": safe_auc(scores, clicked_r),
                "mrr": mrr(ranked_clicked),
                "ndcg5": ndcg_at_k(ranked_clicked, 5),
                "ndcg10": ndcg_at_k(ranked_clicked, 10),
                "auc_baseline": safe_auc(embed_scores, clicked),
            })
    model.train()
    return pd.DataFrame(rows)


def _mean_pool(side: dict, user_id: str, vector_lookup: dict) -> np.ndarray | None:
    hist_vecs = side["history_vectors_by_user"].get(user_id)
    if hist_vecs is None or hist_vecs.shape[0] == 0:
        return None
    mean = hist_vecs.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm == 0:
        return None
    return (mean / norm).astype(np.float32)


def run(
    bundle: str = "small", max_train_impressions: int | None = None, epochs: int = MAX_EPOCHS,
    use_popularity: bool = False,
) -> tuple[dict, dict]:
    torch.manual_seed(SEED)

    dev_paths = dataset_paths("mind", bundle)
    train_paths = _train_paths(bundle)

    t0 = time.time()
    train_side = _build_side(train_paths)
    dev_side = _build_side(dev_paths)
    build_s = time.time() - t0

    popularity = None
    if use_popularity:
        # Train-split-only, Laplace-smoothed - identical construction to
        # every other candidate in this project (ADR-007/009's anti-gaming
        # requirement), reused unchanged rather than reimplemented.
        popularity = build_train_popularity(
            train_side["impressions"], n_catalog=len(train_side["vector_lookup"]), alpha=1.0
        )

    fit_users, holdout_users = _user_level_holdout_split(train_side["impressions"], HOLDOUT_FRACTION, SEED)

    impressions_fit = train_side["impressions"][train_side["impressions"]["user_id"].isin(fit_users)]
    impressions_fit_sorted = impressions_fit.sort_values(["user_id", "impression_id"], kind="stable")
    groups = list(impressions_fit_sorted.groupby(["user_id", "impression_id"], sort=False))
    if max_train_impressions is not None:
        groups = groups[:max_train_impressions]

    all_clicked = train_side["impressions"]["clicked"].to_numpy(dtype=bool)
    click_rate = float(all_clicked.mean())
    pos_weight = torch.tensor((1 - click_rate) / click_rate)

    model = AttentionScorer(dim=train_side["dim"], d_attn=D_ATTN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    rng = np.random.default_rng(SEED)

    epoch_log = []
    best_state = None
    best_holdout_auc = -1.0
    t0 = time.time()
    for epoch in range(epochs):
        epoch_t0 = time.time()
        train_loss = _train_one_epoch(model, groups, train_side, optimizer, bce, MINIBATCH, rng, popularity=popularity)
        holdout_df = _evaluate(model, train_side, COLD_THRESHOLD, users_filter=holdout_users, popularity=popularity)
        holdout_auc = float(holdout_df["auc"].dropna().mean()) if not holdout_df.empty else float("nan")
        epoch_s = time.time() - epoch_t0
        epoch_log.append({
            "epoch": epoch, "train_loss": train_loss, "holdout_auc": holdout_auc, "seconds": round(epoch_s, 2),
        })
        print(f"epoch {epoch}: train_loss={train_loss:.4f} holdout_auc={holdout_auc:.4f} ({epoch_s:.1f}s)", flush=True)
        if holdout_auc > best_holdout_auc:
            best_holdout_auc = holdout_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    train_s = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    t0 = time.time()
    per_impression = _evaluate_dev_with_baseline(model, dev_side, COLD_THRESHOLD, popularity=popularity)
    eval_s = time.time() - t0

    metric_results: dict[str, dict] = {}
    for metric in METRIC_COLUMNS:
        slices = {
            "overall": per_impression,
            "warm": per_impression[per_impression["cohort"] == "warm"],
            "cold": per_impression[per_impression["cohort"] == "cold"],
        }
        metric_results[metric] = {
            name: asdict(ranking_metric_ci(df, metric, n_bootstrap=N_BOOTSTRAP, seed=SEED))
            for name, df in slices.items()
        }
    point, ci_low, ci_high = paired_metric_diff_ci(per_impression, "auc", "auc_baseline")
    paired_result = {
        "metric": "auc_overall_attention_minus_baseline",
        "point_diff": point, "ci_low": ci_low, "ci_high": ci_high,
        "ci_excludes_zero": bool(ci_low > 0),
    }

    n_holdout_users = len(holdout_users)
    n_cold_dev = sum(1 for v in dev_side["history_len_by_user"].values() if v < COLD_THRESHOLD)
    config = {
        "dataset": "mind", "bundle": bundle, "method": "attention_reranker", "candidate": "I",
        "d_attn": D_ATTN, "lr": LR, "minibatch": MINIBATCH, "epochs": epochs,
        "use_popularity": use_popularity,
        "fitted_pop_weight": float(model.pop_weight.item()) if use_popularity else None,
        "n_train_impressions_used": len(groups),
        "n_fit_users": len(fit_users), "n_holdout_users": n_holdout_users,
        "train_click_rate": click_rate, "bce_pos_weight": float(pos_weight),
        "best_epoch": int(np.argmax([e["holdout_auc"] for e in epoch_log])) if epoch_log else None,
        "best_holdout_auc": best_holdout_auc,
        "epoch_log": epoch_log,
        "cold_threshold": COLD_THRESHOLD,
        "n_bootstrap": N_BOOTSTRAP,
        "n_users": len(dev_side["history_len_by_user"]),
        "n_cold_users": n_cold_dev,
        "n_impressions": len(per_impression),
        "build_seconds": round(build_s, 2),
        "train_seconds": round(train_s, 2),
        "eval_seconds": round(eval_s, 2),
        "baseline_auc_overall_embed": BASELINE_AUC_OVERALL,
        "date": str(date.today()),
    }
    results = {"metrics": metric_results, "paired_vs_baseline": paired_result}
    return config, results, model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="small")
    parser.add_argument("--max-train-impressions", type=int, default=None,
                         help="cap on training impressions per epoch, for a quick benchmark run")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--use-popularity", action="store_true",
                         help="ADR-011 addendum: add a learnable log-popularity term alongside the embeddings")
    args = parser.parse_args()

    config, results, model = run(
        bundle=args.bundle, max_train_impressions=args.max_train_impressions,
        epochs=args.epochs, use_popularity=args.use_popularity,
    )

    suffix = f"_e{args.epochs}" + ("_pop" if args.use_popularity else "")
    out_dir = _REPO_ROOT / "experiments" / f"candidate_i_attention_reranker_mind_small_{date.today().isoformat()}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    if args.max_train_impressions is None:
        torch.save(model.state_dict(), out_dir / "model.pt")

    print(f"\nWrote {out_dir}/config.json and results.json")
    print(json.dumps({k: v for k, v in config.items() if k != "epoch_log"}, indent=2))
    for metric, slices in results["metrics"].items():
        print(f"\n{metric}")
        for name, r in slices.items():
            print(f"  {name}: {r['metric']:.4f} (95% CI: {r['ci_low']:.4f}-{r['ci_high']:.4f}), "
                  f"n_impressions={r['n_impressions']}, n_users={r['n_users']}, n_skipped={r['n_skipped']}")
    p = results["paired_vs_baseline"]
    print(f"\npaired bootstrap (attention - baseline, overall AUC): {p['point_diff']:+.4f} "
          f"(95% CI: {p['ci_low']:+.4f} to {p['ci_high']:+.4f}), CI excludes zero: {p['ci_excludes_zero']}")
