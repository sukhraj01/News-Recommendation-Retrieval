#!/usr/bin/env python3
"""Performance ablations: swap or remove a component, measure the LATENCY delta.

This is a different axis from every ablation this project has run before.
ADR-009's leaky-feature ablation, ADR-010/011/012/013's candidate arms and
ADR-013's `nopos` arm all ask "what does this component do to ACCURACY". These
ask "what does it do to latency and throughput", holding the question of
accuracy separate — and, where an arm trades one for the other (the approximate
index, the reduced feature set), reporting BOTH, because a speedup quoted
without its accuracy cost is not a result, it is half of one.

Arms, all four named in the brief:

  bm25    sparse weight-matrix scorer  vs  `rank_bm25.BM25Okapi.get_scores`
          — the naive baseline the rewrite replaced (ADR-006). Exact-match
            equivalence was already verified; this measures the price.
  ann     brute-force cosine  vs  FAISS IVF (approximate)
          — ADR-008 rejected FAISS on a 0.99ms/query brute-force measurement at
            42,416 docs. This re-tests that at 72,023 and reports the recall
            agreement FAISS gives up to win.
  lgbm    LightGBM 65 features  vs  a reduced top-K set (retrained)
          — plus the directly-measured cost of the short-term feature block,
            which the deployed model's own gain importance ranks near the
            bottom.
  nrms    NRMS with vs. without candidate news-vector caching
          — the deployed `NRMSLiteScorer` re-encodes every candidate title on
            every impression; the cached arm encodes the catalog once.

Usage:
    poetry run python benchmarks/ablations.py --which bm25 ann nrms
    poetry run python benchmarks/ablations.py --which lgbm
    poetry run python benchmarks/ablations.py --which all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from benchmarks.loaders import (  # noqa: E402
    embeddings_cache_path,
    load_articles,
    load_history,
    sample_impressions_by_user,
)
from benchmarks.timing import (  # noqa: E402
    git_dirty,
    git_rev,
    hardware_label,
    hardware_spec,
    peak_rss_gb,
    write_results,
)

SEED = 0


def _summarize(times: list[float]) -> dict:
    a = np.asarray(times, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean_ms": round(float(a.mean() * 1e3), 4),
        "p50_ms": round(float(np.percentile(a, 50) * 1e3), 4),
        "p90_ms": round(float(np.percentile(a, 90) * 1e3), 4),
        "total_s": round(float(a.sum()), 4),
        "throughput_per_s": round(float(a.size / a.sum()), 2) if a.sum() > 0 else None,
    }


def _delta(fast: dict, slow: dict) -> dict:
    """Speedup expressed on both the mean and the median.

    Both, because these distributions are right-skewed on an 8GB machine and a
    mean-only speedup can be dominated by a handful of tail samples — a real
    effect worth seeing, but not one that should be the whole headline.
    """
    return {
        "speedup_mean": round(slow["mean_ms"] / fast["mean_ms"], 2) if fast["mean_ms"] else None,
        "speedup_p50": round(slow["p50_ms"] / fast["p50_ms"], 2) if fast["p50_ms"] else None,
        "delta_mean_ms": round(slow["mean_ms"] - fast["mean_ms"], 4),
    }


# --------------------------------------------------------------------------
# 1. BM25: sparse weight matrix vs. rank_bm25's own get_scores
# --------------------------------------------------------------------------
def ablate_bm25(bundle: str, split: str, n_users: int, n_naive_users: int) -> dict:
    """The naive arm is deliberately run over FEWER users.

    `BM25Okapi.get_scores` does one O(corpus_size) numpy pass per query token,
    and this project's queries concatenate a whole click history — hundreds of
    tokens. ADR-006 recorded the naive path as projecting to 10+ hours at
    MINDsmall scale. Running it over the full sample here would take longer
    than the entire rest of this benchmark suite, so it gets its own smaller,
    systematically-drawn subset and the comparison is made per-query.
    """
    from rank_bm25 import BM25Okapi

    from src.retrieval.index import build_index
    from src.retrieval.query import build_user_query
    from src.retrieval.score import score_all
    from src.retrieval.tokenize import tokenize

    articles = load_articles("mind", bundle, split)
    rows, meta = sample_impressions_by_user("mind", bundle, split, n_users, seed=SEED)
    history = load_history("mind", bundle, split, rows["user_id"].unique())

    t0 = time.perf_counter()
    index = build_index(articles)
    sparse_build_s = time.perf_counter() - t0

    text_lookup = dict(zip(
        articles["article_id"],
        articles["title"].fillna("") + " " + articles["abstract"].fillna(""),
    ))
    queries = [build_user_query(list(a), text_lookup) for a in history["article_ids"]]
    queries = [q for q in queries if q]

    # The naive baseline needs rank_bm25's own fitted object. `build_index`
    # fits one internally and discards it; refitting here keeps the two arms
    # honestly independent (and its cost is reported, since "the naive path is
    # cheaper to set up" is a real part of the trade).
    corpus_text = articles["title"].fillna("") + " " + articles["abstract"].fillna("")
    t0 = time.perf_counter()
    corpus = [tokenize(t) for t in corpus_text]
    naive = BM25Okapi(corpus)
    naive_build_s = time.perf_counter() - t0

    for q in queries[:5]:
        score_all(index, q)
    sparse_times = []
    for q in queries:
        t = time.perf_counter(); score_all(index, q); sparse_times.append(time.perf_counter() - t)

    naive_subset = queries[:: max(1, len(queries) // n_naive_users)][:n_naive_users]
    naive.get_scores(naive_subset[0])  # warmup
    naive_times = []
    for q in naive_subset:
        t = time.perf_counter(); naive.get_scores(q); naive_times.append(time.perf_counter() - t)

    # Equivalence spot-check. ADR-006 already verified exact-match on 30 real
    # users, but a performance ablation that silently compared two DIFFERENT
    # computations would be worthless, so it is re-checked here rather than
    # assumed from the record.
    q = naive_subset[0]
    a = score_all(index, q)
    b = np.asarray(naive.get_scores(q))
    max_abs_diff = float(np.max(np.abs(a - b)))

    sparse, naive_s = _summarize(sparse_times), _summarize(naive_times)
    n_users_pop = meta["population_users"]
    return {
        "ablation": "bm25_scorer",
        "question": "sparse weight-matrix scorer vs. rank_bm25.BM25Okapi.get_scores",
        "corpus_articles": int(len(articles)),
        "mean_query_tokens": round(float(np.mean([len(q) for q in queries])), 1),
        "arms": {
            "sparse_matrix (deployed)": {**sparse, "index_build_s": round(sparse_build_s, 3)},
            "rank_bm25_get_scores (naive)": {**naive_s, "index_build_s": round(naive_build_s, 3)},
        },
        "delta": _delta(sparse, naive_s),
        "equivalence_check": {
            "max_abs_score_diff": max_abs_diff,
            "identical": bool(max_abs_diff < 1e-9),
            "note": "re-verified here, not taken on trust from ADR-006's record",
        },
        "projected_full_split_min": {
            "sparse": round(sparse["mean_ms"] / 1e3 * n_users_pop / 60, 2),
            "naive": round(naive_s["mean_ms"] / 1e3 * n_users_pop / 60, 2),
            "basis_users": n_users_pop,
        },
    }


# --------------------------------------------------------------------------
# 2. Brute-force cosine vs. FAISS IVF
# --------------------------------------------------------------------------
def ablate_ann(bundle: str, split: str, n_users: int, k: int, nlist: int, nprobe: int) -> dict:
    """Re-tests ADR-008's "brute force is enough, FAISS unjustified" decision.

    Reports recall@k of the approximate index against the brute-force ranking
    as ground truth. That is the number that decides whether the speedup is
    usable: an ANN index that is 5x faster but agrees with only 60% of the true
    top-k has not made the system faster, it has made it a different system.
    """
    import faiss

    from src.retrieval.embed import (
        DEFAULT_MODEL,
        build_embedding_index,
        build_user_embedding_query,
        model_slug,
    )

    articles = load_articles("mind", bundle, split)
    rows, _meta = sample_impressions_by_user("mind", bundle, split, n_users, seed=SEED)
    history = load_history("mind", bundle, split, rows["user_id"].unique())

    cache = embeddings_cache_path("mind", bundle, split, model_slug(DEFAULT_MODEL))
    index = build_embedding_index(articles, model_name=DEFAULT_MODEL, cache_path=cache)
    vectors = np.ascontiguousarray(index.vectors.astype(np.float32))
    n_docs, dim = vectors.shape

    vector_lookup = dict(zip(index.article_ids, index.vectors))
    queries = [build_user_embedding_query(list(a), vector_lookup) for a in history["article_ids"]]
    queries = [q for q in queries if q is not None]
    Q = np.ascontiguousarray(np.stack(queries).astype(np.float32))

    # Vectors are already L2-normalised (embed.py), so inner product IS cosine
    # similarity — the same quantity the brute-force path computes. Using an L2
    # index here would silently change the metric.
    t0 = time.perf_counter()
    quantizer = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    faiss.seed_rand = SEED
    ivf.train(vectors)
    ivf.add(vectors)
    ivf.nprobe = nprobe
    ivf_build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    flat = faiss.IndexFlatIP(dim)
    flat.add(vectors)
    flat_build_s = time.perf_counter() - t0

    def brute(q):
        scores = vectors @ q
        kk = min(k, scores.size)
        top = np.argpartition(-scores, kk - 1)[:kk]
        return top[np.argsort(-scores[top])]

    for i in range(min(5, len(Q))):
        brute(Q[i]); ivf.search(Q[i : i + 1], k)

    brute_times, ivf_times, flat_times = [], [], []
    brute_top, ivf_top = [], []
    for i in range(len(Q)):
        q = Q[i]
        t = time.perf_counter(); bt = brute(q); brute_times.append(time.perf_counter() - t)
        t = time.perf_counter(); _, I = ivf.search(q[None, :], k); ivf_times.append(time.perf_counter() - t)
        t = time.perf_counter(); flat.search(q[None, :], k); flat_times.append(time.perf_counter() - t)
        brute_top.append(bt); ivf_top.append(I[0])

    recalls = [
        len(set(b.tolist()) & set(i[i >= 0].tolist())) / len(b)
        for b, i in zip(brute_top, ivf_top)
    ]

    b, iv, fl = _summarize(brute_times), _summarize(ivf_times), _summarize(flat_times)
    return {
        "ablation": "ann_index",
        "question": "brute-force cosine (deployed) vs. FAISS IVF approximate, and FAISS exact",
        "corpus_articles": int(n_docs), "dim": int(dim), "k": k,
        "ivf_params": {"nlist": nlist, "nprobe": nprobe},
        "arms": {
            "brute_force_numpy (deployed)": {**b, "index_build_s": 0.0,
                                             "note": "no index to build — a plain matvec"},
            "faiss_ivf_approximate": {**iv, "index_build_s": round(ivf_build_s, 3)},
            "faiss_flat_exact": {**fl, "index_build_s": round(flat_build_s, 3)},
        },
        "delta_ivf_vs_brute": _delta(iv, b),
        "delta_flat_vs_brute": _delta(fl, b),
        "accuracy_cost": {
            f"recall@{k}_vs_brute_force": round(float(np.mean(recalls)), 4),
            "min_recall": round(float(np.min(recalls)), 4),
            "frac_queries_perfect": round(float(np.mean([r == 1.0 for r in recalls])), 4),
            "note": "brute force is ground truth by construction; IVF's shortfall is the price of its speed",
        },
    }


# --------------------------------------------------------------------------
# 3. NRMS: with vs. without candidate news-vector caching
# --------------------------------------------------------------------------
def ablate_nrms(bundle: str, split: str, n_users: int, device: str) -> dict:
    """The deployed `NRMSLiteScorer` runs the news encoder over every candidate
    title on every impression. Nothing about a candidate's news vector depends
    on the user, so it can be computed once for the whole catalog and looked
    up — the same amortisation `BM25Scorer`/`EmbeddingScorer` already do for
    their own full-corpus score vectors, which NRMS never got.

    History news vectors are cacheable on exactly the same argument and are
    cached in the same arm, since a user's history titles do not change between
    their impressions either.

    This arm is NOT a proposal to change the model — it computes the identical
    function. It is a measurement of what the missing cache costs, which is why
    the arm also checks that the two paths agree numerically.
    """
    import torch

    from src.retrieval.nrms import NRMSLite, build_vocab
    from src.retrieval.nrms_training import NRMSLiteScorer, build_title_matrix
    from benchmarks.profile_mind import _load_nrms_config

    articles = load_articles("mind", bundle, split)
    rows, _meta = sample_impressions_by_user("mind", bundle, split, n_users, seed=SEED)
    history = load_history("mind", bundle, split, rows["user_id"].unique())
    hist_by_user = dict(zip(history["user_id"], history["article_ids"]))

    cfg = _load_nrms_config()
    word2id = build_vocab(articles["title"].fillna("").tolist(), min_freq=2)
    title_matrix, news_id2row, pad_row = build_title_matrix(articles, word2id, cfg["max_title_len"])

    torch.manual_seed(SEED)
    model = NRMSLite(vocab_size=len(word2id), embed_dim=cfg["embed_dim"],
                     num_heads=cfg["num_heads"]).eval()
    dev = torch.device(device)
    model, title_matrix = model.to(dev), title_matrix.to(dev)
    scorer = NRMSLiteScorer(model, news_id2row, pad_row, title_matrix, cfg["max_history_len"], dev)

    # ---- cached arm: encode the whole catalog once -----------------------
    t0 = time.perf_counter()
    with torch.no_grad():
        chunks = []
        for lo in range(0, title_matrix.shape[0], 2048):
            chunks.append(model.news_encoder(title_matrix[lo : lo + 2048]))
        news_vecs = torch.cat(chunks)
    cache_build_s = time.perf_counter() - t0

    @torch.no_grad()
    def cached_score(hist_ids, cand_ids):
        k = len(cand_ids)
        known = np.fromiter((c in news_id2row for c in cand_ids), dtype=bool, count=k)
        out = np.full(k, -np.inf, dtype=np.float64)
        if not known.any():
            return out
        hist = list(hist_ids)[-cfg["max_history_len"]:]
        hrows = [news_id2row.get(a, pad_row) for a in hist]
        hlen = len(hrows)
        hrows = hrows + [pad_row] * (cfg["max_history_len"] - hlen)
        hv = news_vecs[torch.tensor(hrows, device=dev)].unsqueeze(0)
        hmask = torch.tensor([[i >= hlen for i in range(cfg["max_history_len"])]],
                             dtype=torch.bool, device=dev)
        uvec = model.user_encoder(hv, hmask)
        crows = torch.tensor([news_id2row[c] for c, m in zip(cand_ids, known) if m], device=dev)
        cv = news_vecs[crows].unsqueeze(0)
        out[known] = torch.bmm(cv, uvec.unsqueeze(-1)).squeeze(-1).squeeze(0).cpu().numpy()
        return out

    work = []
    for uid, urows in rows.groupby("user_id", sort=False):
        hist = list(hist_by_user.get(uid, []))
        for _iid, imp in urows.groupby("impression_id", sort=False):
            work.append((hist, imp["article_id"].tolist()))

    for h, c in work[:5]:
        scorer.score(h, c); cached_score(h, c)

    uncached_times, cached_times = [], []
    for h, c in work:
        t = time.perf_counter(); scorer.score(h, c); uncached_times.append(time.perf_counter() - t)
        t = time.perf_counter(); cached_score(h, c); cached_times.append(time.perf_counter() - t)

    h, c = work[0]
    a, bvals = scorer.score(h, c), cached_score(h, c)
    finite = np.isfinite(a) & np.isfinite(bvals)
    max_diff = float(np.max(np.abs(a[finite] - bvals[finite]))) if finite.any() else 0.0

    un, ca = _summarize(uncached_times), _summarize(cached_times)
    n_imp = len(work)
    breakeven = cache_build_s / ((un["mean_ms"] - ca["mean_ms"]) / 1e3) if un["mean_ms"] > ca["mean_ms"] else None
    return {
        "ablation": "nrms_candidate_cache",
        "question": "NRMS with vs. without cached candidate/history news vectors",
        "device": str(dev),
        "n_impressions": n_imp,
        "catalog_articles": int(len(articles)),
        "arms": {
            "uncached (deployed NRMSLiteScorer)": {**un, "cache_build_s": 0.0},
            "cached_news_vectors": {**ca, "cache_build_s": round(cache_build_s, 3)},
        },
        "delta": _delta(ca, un),
        "equivalence_check": {
            "max_abs_score_diff": max_diff,
            "note": "same function, computed two ways — a nonzero diff here would invalidate the arm",
        },
        "cache_amortisation": {
            "one_time_encode_s": round(cache_build_s, 3),
            "breakeven_impressions": round(breakeven, 1) if breakeven else None,
            "note": "impressions needed before the one-time catalog encode pays for itself",
        },
    }


# --------------------------------------------------------------------------
# 4. LightGBM: full 65 features vs. a reduced set
# --------------------------------------------------------------------------
def ablate_lgbm(bundle: str, split: str, n_impressions: int, top_k: int,
                n_train_impressions: int) -> dict:
    """Two questions, kept apart because they have different answers.

    (a) INFERENCE cost as a function of feature count — measured by retraining
        a top-K booster (by the deployed model's own recorded gain) and timing
        `predict` on both. Reported with the validation-AUC cost, since a
        reduced model that is faster and worse is a trade, not a win.
    (b) FEATURE-ENGINEERING cost — the part `profile_ebnerd.py` shows dominates.
        The short-term feature block is timed directly as a sub-stage there;
        here its share is reported next to its gain rank, because "expensive
        and low-gain" is the actual investable finding, not "the booster is
        slightly faster with fewer columns".
    """
    import json

    import lightgbm as lgb

    from src.evaluation.ranking_metrics import per_impression_auc
    from src.retrieval.ebnerd_features import (
        CATEGORICAL_FEATURES,
        FEATURE_NAMES,
        add_session_position_columns,
        build_feature_frame,
        build_history_popularity,
        build_user_profiles,
        load_article_table,
    )
    from run_ebnerd_gbdt_experiment import LGB_PARAMS, load_behaviors

    from benchmarks.profile_ebnerd import sample_behaviors

    zip_path = _REPO_ROOT / "data" / "raw" / "ebnerd" / f"ebnerd_{bundle}.zip"
    emb_dir = _REPO_ROOT / "data" / "processed" / "ebnerd" / bundle / "embeddings"
    slug = "sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2"

    art = load_article_table(zip_path, embeddings_npy=emb_dir / f"{slug}.npy",
                             embeddings_json=emb_dir / f"{slug}.json")

    imp = json.loads((_REPO_ROOT / "experiments" / "candidate_k_gbdt_ebnerd_small_2026-08-26"
                      / "feature_importance.json").read_text())["K_rank"]
    ranked = [f for f, _ in sorted(imp.items(), key=lambda kv: -kv[1])]
    # Restrict to features the deployed booster was actually trained on, so the
    # reduced arm is a strict subset of the full arm's inputs and the two are
    # comparable. Ranking over features the full model never saw would make the
    # "reduced vs full" delta partly a different-features effect.
    ranked = [f for f in ranked if f in FEATURE_NAMES]
    keep = ranked[:top_k]
    keep_idx = [FEATURE_NAMES.index(f) for f in keep]
    st_features = [f for f in FEATURE_NAMES if f.startswith("st_")]

    # ---- validation matrix (shared by both arms) ------------------------
    v_prof = build_user_profiles(zip_path, split, art)
    v_pop = build_history_popularity(zip_path, split, art)
    v_beh = add_session_position_columns(load_behaviors(zip_path, split))
    v_beh, v_meta = sample_behaviors(v_beh, n_impressions)
    Xv, mv = build_feature_frame(v_beh, art, v_prof, v_pop, with_labels=True)

    full = lgb.Booster(model_file=str(DEFAULT_LGBM_MODEL))

    # ---- retrain the reduced arm ----------------------------------------
    t_prof = build_user_profiles(zip_path, "train", art)
    t_pop = build_history_popularity(zip_path, "train", art)
    t_beh = add_session_position_columns(load_behaviors(zip_path, "train"))
    t_beh, _ = sample_behaviors(t_beh, n_train_impressions)
    Xt, mt = build_feature_frame(t_beh, art, t_prof, t_pop, with_labels=True)

    params = dict(LGB_PARAMS)
    params.update({"objective": "lambdarank", "metric": "ndcg",
                   "ndcg_eval_at": [5], "label_gain": [0, 1]})
    cat_idx = [keep.index(c) for c in CATEGORICAL_FEATURES if c in keep]
    t0 = time.perf_counter()
    reduced = lgb.train(
        params,
        lgb.Dataset(Xt[:, keep_idx], label=mt["label"], group=mt["group_sizes"],
                    categorical_feature=cat_idx, feature_name=keep, free_raw_data=True),
        num_boost_round=full.num_trees(),
    )
    reduced_train_s = time.perf_counter() - t0
    del Xt, mt

    # Same resolver as profile_ebnerd.py — see its docstring for why
    # booster.feature_name() cannot be trusted for this model.
    from benchmarks.profile_ebnerd import resolve_model_features

    full_names, full_idx, _src = resolve_model_features(full, DEFAULT_LGBM_MODEL)
    Xv_full = Xv if full_idx is None else Xv[:, full_idx]
    Xv_red = Xv[:, keep_idx]

    def bench_predict(booster, mat, iters=5):
        booster.predict(mat[:10_000])
        ts = []
        for _ in range(iters):
            t = time.perf_counter(); s = booster.predict(mat); ts.append(time.perf_counter() - t)
        return ts, s

    ts_full, sc_full = bench_predict(full, Xv_full)
    ts_red, sc_red = bench_predict(reduced, Xv_red)

    labels, sizes = mv["label"], mv["group_sizes"]
    auc_full = float(np.nanmean(per_impression_auc(np.asarray(sc_full, dtype=np.float64), labels, sizes)))
    auc_red = float(np.nanmean(per_impression_auc(np.asarray(sc_red, dtype=np.float64), labels, sizes)))

    n_rows, n_imp = int(Xv.shape[0]), int(len(v_beh))
    f_stats = _summarize([t / n_imp for t in ts_full])
    r_stats = _summarize([t / n_imp for t in ts_red])

    return {
        "ablation": "lgbm_feature_count",
        "question": "LightGBM full 65 features vs. reduced top-K (by the deployed model's own gain)",
        "n_validation_impressions": n_imp,
        "n_validation_candidate_rows": n_rows,
        "n_train_impressions_for_reduced_arm": int(len(t_beh)),
        "arms": {
            f"full_{len(full_names)}_features (deployed)": {
                "n_features": len(full_names), "n_trees": int(full.num_trees()),
                "predict_ms_per_impression": f_stats["mean_ms"],
                "predict_total_s": round(float(np.mean(ts_full)), 4),
                "rows_per_s": round(n_rows / float(np.mean(ts_full)), 0),
                "mean_auc": round(auc_full, 4),
                "trained": "deployed Candidate K model_K_rank.txt",
            },
            f"reduced_{top_k}_features": {
                "n_features": top_k, "n_trees": int(reduced.num_trees()),
                "predict_ms_per_impression": r_stats["mean_ms"],
                "predict_total_s": round(float(np.mean(ts_red)), 4),
                "rows_per_s": round(n_rows / float(np.mean(ts_red)), 0),
                "mean_auc": round(auc_red, 4),
                "retrained_s": round(reduced_train_s, 1),
                "features": keep,
            },
        },
        "delta": {
            "predict_speedup_mean": round(f_stats["mean_ms"] / r_stats["mean_ms"], 2)
            if r_stats["mean_ms"] else None,
            "auc_delta": round(auc_red - auc_full, 4),
            "note": "AUC here is the unweighted mean of per-impression AUC on the sampled "
                    "validation impressions — a like-for-like comparison between the two arms, "
                    "NOT comparable to ADR-013's bootstrapped full-split figure",
        },
        "feature_engineering_context": {
            "short_term_features": st_features,
            "short_term_in_top_k": [f for f in st_features if f in keep],
            "gain_rank_of_best_short_term_feature": (
                min((ranked.index(f) + 1 for f in st_features if f in ranked), default=None)
            ),
            "note": "the per-impression COST of this block is measured directly by "
                    "profile_ebnerd.py's 1a_short_term_features sub-stage",
        },
    }


DEFAULT_LGBM_MODEL = (_REPO_ROOT / "experiments" / "candidate_k_gbdt_ebnerd_small_2026-08-26"
                      / "model_K_rank.txt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", nargs="+", default=["all"],
                    choices=["all", "bm25", "ann", "nrms", "lgbm"])
    ap.add_argument("--bundle", default="large")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n-users", type=int, default=400)
    ap.add_argument("--naive-bm25-users", type=int, default=40)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--nlist", type=int, default=256)
    ap.add_argument("--nprobe", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ebnerd-bundle", default="small")
    ap.add_argument("--ebnerd-split", default="validation")
    ap.add_argument("--lgbm-impressions", type=int, default=25_000)
    ap.add_argument("--lgbm-train-impressions", type=int, default=120_000)
    ap.add_argument("--lgbm-top-k", type=int, default=20)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    which = ["bm25", "ann", "nrms", "lgbm"] if "all" in args.which else args.which
    results: dict = {}

    if "bm25" in which:
        print("[ablation] bm25 sparse vs. naive rank_bm25 …", flush=True)
        results["bm25"] = ablate_bm25(args.bundle, args.split, args.n_users, args.naive_bm25_users)
    if "ann" in which:
        print("[ablation] brute-force cosine vs. FAISS IVF …", flush=True)
        results["ann"] = ablate_ann(args.bundle, args.split, args.n_users,
                                    args.k, args.nlist, args.nprobe)
    if "nrms" in which:
        print("[ablation] NRMS candidate-vector caching …", flush=True)
        results["nrms"] = ablate_nrms(args.bundle, args.split, args.n_users, args.device)
    if "lgbm" in which:
        print("[ablation] LightGBM feature count …", flush=True)
        results["lgbm"] = ablate_lgbm(args.ebnerd_bundle, args.ebnerd_split,
                                      args.lgbm_impressions, args.lgbm_top_k,
                                      args.lgbm_train_impressions)

    payload = {
        "benchmark": "performance_ablations",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_rev": git_rev(), "git_dirty": git_dirty(), "tag": args.tag,
        "hardware_label": hardware_label(), "hardware": hardware_spec(),
        "ablations": results,
        "peak_rss_gb": round(peak_rss_gb(), 2),
    }
    path = write_results("performance_ablations", payload)

    import json as _json
    print(_json.dumps(results, indent=2, default=str))
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
