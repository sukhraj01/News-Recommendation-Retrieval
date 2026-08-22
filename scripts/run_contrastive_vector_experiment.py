#!/usr/bin/env python3
"""Isolated experiment: does EB-NeRD's provided `contrastive_vector` artifact
outperform this project's from-scratch MiniLM embedding on local validation?

ADR-008 rejected using EB-NeRD's provided embedding artifacts in favor of
computing one model ourselves over both datasets — but that decision's cost
(what accuracy, if any, is left on the table by not using the provided
artifact) was never measured, only argued. This script measures it, as an
isolated addendum to ADR-008 (see the ADR's addendum section), not a
reversal of it: the shared MIND/EB-NeRD pipeline, `embed.py`, `score.py`,
`retrieve.py`, and `run_embed_experiment.py`/`run_ranking_eval.py` are all
untouched.

The only new logic here is `load_contrastive_index`, which builds an
`EmbeddingIndex` (the same dataclass `embed.py` uses) from EB-NeRD's
provided `contrastive_vector.parquet` instead of encoding text with
MiniLM. Everything downstream — `build_user_embedding_query`,
`EmbeddingScorer`, `embed_retrieve_top_k`, `recall_at_k`, and every Q4
ranking metric (AUC/MRR/nDCG/diversity/novelty/coverage + bootstrap CI) —
is imported and reused completely unchanged, so the embedding source is
isolated as the only variable, per this experiment's own objective.

Per the artifact's documented packaging (`ebnerd-benchmark`'s
reproducibility scripts, `examples/reproducibility_scripts/
ebnerd_nrms_docvec.py`: `create_article_id_to_value_mapping(df=df_articles,
value_col=df_articles.columns[-1])`), the archive contains a single
`contrastive_vector.parquet` with an article-id column and the embedding
vector as the LAST column. No public documentation describing the
training methodology (base model, objective, which text fields) was found
outside the archive itself — `inspect_artifact` below prints whatever
accompanying docs/README the zip actually ships, rather than assuming.

Usage (once the artifact + local ebnerd_small feature store both exist):
    poetry run python scripts/run_contrastive_vector_experiment.py \\
        --artifact-zip data/raw/ebnerd/Ekstra_Bladet_contrastive_vector.zip
"""
import argparse
import json
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import numpy as np
import pandas as pd

from run_bm25_experiment import COLD_THRESHOLD, K_VALUES, MAX_K, dataset_paths
from src.evaluation.metrics import recall_at_k
from src.evaluation.ranking_metrics import (
    build_train_popularity,
    coverage,
    intra_list_diversity,
    mrr,
    ndcg_at_k,
    novelty,
    rank_candidates,
    ranking_metric_ci,
    safe_auc,
)
from src.retrieval.embed import EmbeddingIndex, build_user_embedding_query
from src.retrieval.retrieve import embed_retrieve_top_k
from src.retrieval.score import EmbeddingScorer
from src.utils.ids import prefix_id

DATASET = "ebnerd"
BUNDLE = "small"
N_BOOTSTRAP = 2000
SEED = 0
K_DIVERSITY_NOVELTY = 10  # matches ADR-007's nDCG@10 anchor, reused unchanged

# MiniLM baseline this experiment compares against — ADR-008's real
# ebnerd_small validation numbers (experiments/embed_ebnerd_small_2026-08-10/,
# experiments/ranking_embed_ebnerd_small_2026-08-10/), hardcoded here rather
# than re-read at runtime so the comparison target is pinned and auditable.
MINILM_BASELINE_RECALL = {
    "50": 0.0014, "100": 0.0043, "200": 0.0121,
}
MINILM_BASELINE_RANKING = {
    "auc": 0.5430, "mrr": 0.3437, "ndcg5": 0.3804, "ndcg10": 0.4591,
    "diversity10": 0.7890, "novelty10": 17.19, "coverage10": 0.2050,
}


def inspect_artifact(zip_path: Path) -> zipfile.ZipFile:
    """Prints the archive's real contents and any accompanying docs before
    anything is assumed about its format — per this experiment's own
    objective ("confirm it's per-article vectors keyed by article ID").
    """
    zf = zipfile.ZipFile(zip_path)
    names = zf.namelist()
    print("=" * 80)
    print(f"{zip_path.name} — {len(names)} entries")
    print("=" * 80)
    for n in names:
        info = zf.getinfo(n)
        print(f"  {n}  ({info.file_size} bytes)")

    doc_names = [n for n in names if n.lower().endswith((".md", ".txt", ".rst", ".json"))]
    for n in doc_names:
        print(f"\n--- {n} ---")
        print(zf.read(n).decode("utf-8", errors="replace")[:5000])
    if not doc_names:
        print("\nNo README/txt/md/json accompanying the parquet was found in the archive.")

    return zf


def load_contrastive_index(
    zip_path: Path, local_articles: pd.DataFrame, member: str | None = None
) -> tuple[EmbeddingIndex, dict]:
    """Builds an `EmbeddingIndex` from EB-NeRD's provided contrastive-vector
    artifact, restricted to the article IDs this project's local
    `ebnerd_small` corpus actually has (so it's a drop-in substitute for
    `build_embedding_index`'s return value, consumable by the unmodified
    `EmbeddingScorer`/`embed_retrieve_top_k`/`build_user_embedding_query`).

    Coverage (how many of the local corpus's articles the artifact
    actually has a vector for) is reported explicitly, not assumed to be
    100% — an article present locally but absent from the artifact is
    excluded from this index's catalog entirely (not zero-filled), so it
    falls through to `score.py`'s existing, unmodified missing-id handling
    (`_lookup_scores`'s -inf fallback) exactly like a genuinely
    out-of-catalog id would for BM25 or MiniLM.
    """
    zf = zipfile.ZipFile(zip_path)
    if member is None:
        candidates = [n for n in zf.namelist() if n.endswith("contrastive_vector.parquet")]
        if not candidates:
            candidates = [n for n in zf.namelist() if n.endswith(".parquet")]
        if not candidates:
            raise FileNotFoundError(
                f"No .parquet member found in {zip_path}; archive contents: {zf.namelist()}"
            )
        member = candidates[0]

    print(f"Loading vectors from archive member: {member}")
    with zf.open(member) as f:
        raw = pd.read_parquet(f)

    print(f"contrastive_vector.parquet: shape={raw.shape}, columns={list(raw.columns)}")
    print(raw.dtypes)
    print(raw.head(3))

    id_col = "article_id" if "article_id" in raw.columns else raw.columns[0]
    # Per ebnerd-benchmark's own loading convention (create_article_id_to_value_mapping
    # uses df.columns[-1]): the embedding vector is the last column.
    vec_col = raw.columns[-1]
    print(f"Using id_col={id_col!r}, vec_col={vec_col!r}")

    raw_vector_by_id = dict(zip(raw[id_col].astype(str), raw[vec_col]))

    article_ids: list[str] = []
    vectors: list[np.ndarray] = []
    n_missing = 0
    for local_id, raw_id in zip(local_articles["article_id"], local_articles["_raw_id"]):
        vec = raw_vector_by_id.get(raw_id)
        if vec is None:
            n_missing += 1
            continue
        v = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm == 0:
            n_missing += 1
            continue
        article_ids.append(local_id)
        vectors.append(v / norm)

    coverage_stats = {
        "n_local_articles": len(local_articles),
        "n_covered": len(article_ids),
        "n_missing": n_missing,
        "coverage_fraction": round(len(article_ids) / len(local_articles), 4) if len(local_articles) else 0.0,
        "artifact_total_rows": len(raw),
        "artifact_vector_dim": int(np.asarray(raw[vec_col].iloc[0]).shape[0]) if len(raw) else None,
    }
    print(f"Coverage: {coverage_stats}")

    index = EmbeddingIndex(
        article_ids=article_ids,
        id_to_col={aid: i for i, aid in enumerate(article_ids)},
        vectors=np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32),
    )
    return index, coverage_stats


def run_recall(index: EmbeddingIndex, history: pd.DataFrame, impressions: pd.DataFrame) -> dict:
    vector_lookup = dict(zip(index.article_ids, index.vectors))

    top_k_by_user: dict[str, list[str]] = {}
    history_len_by_user: dict[str, int] = {}
    for row in history.itertuples(index=False):
        query = build_user_embedding_query(row.article_ids, vector_lookup)
        top_k_by_user[row.user_id] = embed_retrieve_top_k(index, query, k=MAX_K)
        history_len_by_user[row.user_id] = len(row.article_ids)

    positives = impressions.loc[impressions["clicked"], ["user_id", "article_id"]].copy()
    positives["history_len"] = positives["user_id"].map(history_len_by_user).fillna(0).astype(int)
    positives["cohort"] = np.where(positives["history_len"] < COLD_THRESHOLD, "cold", "warm")

    k_results: dict[str, dict] = {}
    for k in K_VALUES:
        positives[f"hit@{k}"] = [
            article_id in top_k_by_user.get(user_id, [])[:k]
            for user_id, article_id in zip(positives["user_id"], positives["article_id"])
        ]
        slices = {"overall": positives, "warm": positives[positives["cohort"] == "warm"],
                  "cold": positives[positives["cohort"] == "cold"]}
        k_results[str(k)] = {
            name: asdict(recall_at_k(df.rename(columns={f"hit@{k}": "hit"}), n_bootstrap=N_BOOTSTRAP))
            for name, df in slices.items()
        }
    return k_results


def _finite_scores_for_auc(scores: np.ndarray) -> np.ndarray:
    """`score.py::_lookup_scores` (shared, unmodified) scores a candidate
    absent from the index's catalog as -inf, by design, so it deterministically
    ranks last (see its docstring). That's fine for `rank_candidates`'
    lexsort and for `mrr`/`ndcg_at_k` (which only see the resulting boolean
    rank order, never raw scores) — but MiniLM's index always covers 100%
    of the local catalog by construction (it encodes every local article
    itself), so this path was never exercised at any real rate before.
    This artifact's coverage is not guaranteed to be 100% (see
    `load_contrastive_index`'s coverage stats), and `sklearn.roc_auc_score`
    (`safe_auc`) rejects non-finite scores outright. Substituting the
    impression's own minimum finite score minus 1 for every -inf preserves
    the exact same relative rank -inf already produced (still ranks below
    every real score) without introducing an actual infinity — a local fix
    for a new caller's new access pattern, not a change to the shared
    scorer's contract.
    """
    if not np.isneginf(scores).any():
        return scores
    finite = scores[np.isfinite(scores)]
    floor = (finite.min() - 1.0) if finite.size else -1.0
    return np.where(np.isneginf(scores), floor, scores)


def run_ranking(
    index: EmbeddingIndex, articles: pd.DataFrame, history: pd.DataFrame,
    impressions: pd.DataFrame, train_impressions: pd.DataFrame,
) -> tuple[dict, dict]:
    scorer = EmbeddingScorer(index)
    vector_lookup = dict(zip(index.article_ids, index.vectors))

    query_by_user: dict[str, np.ndarray | None] = {}
    history_len_by_user: dict[str, int] = {}
    for row in history.itertuples(index=False):
        query_by_user[row.user_id] = build_user_embedding_query(row.article_ids, vector_lookup)
        history_len_by_user[row.user_id] = len(row.article_ids)

    category_lookup = dict(zip(articles["article_id"], articles["category"]))
    popularity = build_train_popularity(train_impressions, n_catalog=len(articles), alpha=1.0)
    cohort_by_user = {
        uid: ("cold" if hlen < COLD_THRESHOLD else "warm")
        for uid, hlen in history_len_by_user.items()
    }

    impressions_sorted = impressions.sort_values(["user_id", "impression_id"], kind="stable")

    rows: list[dict] = []
    top_k_ids_by_cohort: dict[str, list[list[str]]] = {"overall": [], "warm": [], "cold": []}
    for (user_id, impression_id), group in impressions_sorted.groupby(
        ["user_id", "impression_id"], sort=False
    ):
        candidate_ids = group["article_id"].tolist()
        clicked = group["clicked"].to_numpy(dtype=bool)
        query = query_by_user.get(user_id)
        cohort = cohort_by_user.get(user_id, "cold")

        scores = scorer.score(query, candidate_ids)
        order = rank_candidates(scores, impression_id, seed=SEED)
        ranked_ids = [candidate_ids[i] for i in order]
        ranked_clicked = clicked[order]

        k_eff = min(K_DIVERSITY_NOVELTY, len(ranked_ids))
        top_ids = ranked_ids[:k_eff]
        top_categories = [category_lookup.get(a) for a in top_ids]

        rows.append({
            "impression_id": impression_id,
            "user_id": user_id,
            "cohort": cohort,
            "auc": safe_auc(_finite_scores_for_auc(scores), clicked),
            "mrr": mrr(ranked_clicked),
            "ndcg5": ndcg_at_k(ranked_clicked, 5),
            "ndcg10": ndcg_at_k(ranked_clicked, 10),
            "diversity10": intra_list_diversity(top_categories),
            "novelty10": novelty(top_ids, popularity),
        })
        top_k_ids_by_cohort["overall"].append(top_ids)
        top_k_ids_by_cohort[cohort].append(top_ids)

    per_impression = pd.DataFrame(rows)
    metric_results: dict[str, dict] = {}
    for metric in ["auc", "mrr", "ndcg5", "ndcg10", "diversity10", "novelty10"]:
        slices = {
            "overall": per_impression,
            "warm": per_impression[per_impression["cohort"] == "warm"],
            "cold": per_impression[per_impression["cohort"] == "cold"],
        }
        metric_results[metric] = {
            name: asdict(ranking_metric_ci(df, metric, n_bootstrap=N_BOOTSTRAP, seed=SEED))
            for name, df in slices.items()
        }

    coverage_results = {
        name: (coverage(ids, len(articles)) if ids else float("nan"))
        for name, ids in top_k_ids_by_cohort.items()
    }
    return metric_results, coverage_results


def _ci_clear_win(candidate_low: float, baseline: float) -> bool:
    """True if the candidate's own CI lower bound already clears the
    (point-estimate) baseline — a conservative, explicit "CI-clear margin"
    check, not just a point-estimate comparison."""
    return candidate_low > baseline


def run(artifact_zip: Path) -> dict:
    paths = dataset_paths(DATASET, BUNDLE)
    articles = pd.read_parquet(paths["articles"])
    history = pd.read_parquet(paths["history"])
    impressions = pd.read_parquet(paths["impressions"])
    train_impressions = pd.read_parquet(
        paths["articles"].parent / "train" / "impressions.parquet"
    )

    articles = articles.copy()
    articles["_raw_id"] = articles["article_id"].str.replace("ebnerd:", "", regex=False)

    inspect_artifact(artifact_zip)
    index, coverage_stats = load_contrastive_index(artifact_zip, articles)

    t0 = time.time()
    recall_results = run_recall(index, history, impressions)
    recall_s = time.time() - t0

    t0 = time.time()
    ranking_metrics, coverage_results = run_ranking(index, articles, history, impressions, train_impressions)
    ranking_s = time.time() - t0

    comparison = {
        "recall_at_k": {
            k: {
                "contrastive_overall": recall_results[k]["overall"]["recall"],
                "minilm_baseline": MINILM_BASELINE_RECALL.get(k),
                "ci_clear_win": _ci_clear_win(
                    recall_results[k]["overall"]["ci_low"], MINILM_BASELINE_RECALL.get(k, float("inf"))
                ),
            }
            for k in recall_results
        },
        "ranking_overall": {
            metric: {
                "contrastive": ranking_metrics[metric]["overall"]["metric"],
                "contrastive_ci_low": ranking_metrics[metric]["overall"]["ci_low"],
                "contrastive_ci_high": ranking_metrics[metric]["overall"]["ci_high"],
                "minilm_baseline": MINILM_BASELINE_RANKING.get(metric),
                "ci_clear_win": _ci_clear_win(
                    ranking_metrics[metric]["overall"]["ci_low"], MINILM_BASELINE_RANKING.get(metric, float("inf"))
                ) if metric != "novelty10" else None,  # novelty has no "higher is strictly better" framing here
            }
            for metric in ranking_metrics
        },
    }

    config = {
        "dataset": DATASET,
        "bundle": BUNDLE,
        "artifact": "Ekstra_Bladet_contrastive_vector.zip",
        "artifact_source_url": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_contrastive_vector.zip",
        "coverage": coverage_stats,
        "k_values": K_VALUES,
        "cold_threshold": COLD_THRESHOLD,
        "diversity_novelty_k": K_DIVERSITY_NOVELTY,
        "n_bootstrap": N_BOOTSTRAP,
        "recall_seconds": round(recall_s, 2),
        "ranking_seconds": round(ranking_s, 2),
        "date": str(date.today()),
        "note": "Isolated addendum experiment per ADR-008 — does NOT change embed.py/score.py/"
                "retrieve.py/run_embed_experiment.py/run_ranking_eval.py or ADR-008's decision.",
    }

    results = {
        "recall": recall_results,
        "ranking": {"metrics": ranking_metrics, "coverage": coverage_results},
        "comparison_vs_minilm_baseline": comparison,
    }
    return {"config": config, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-zip",
        default=str(_REPO_ROOT / "data" / "raw" / "ebnerd" / "Ekstra_Bladet_contrastive_vector.zip"),
    )
    args = parser.parse_args()

    output = run(Path(args.artifact_zip))
    config, results = output["config"], output["results"]

    out_dir = _REPO_ROOT / "experiments" / f"contrastive_vector_ebnerd_small_{date.today().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\nWrote {out_dir}/config.json and results.json")
    print(json.dumps(config, indent=2))
    print("\n=== recall@K vs MiniLM baseline ===")
    for k, c in results["comparison_vs_minilm_baseline"]["recall_at_k"].items():
        print(f"  k={k}: contrastive={c['contrastive_overall']:.4f} vs minilm={c['minilm_baseline']:.4f} "
              f"ci_clear_win={c['ci_clear_win']}")
    print("\n=== Q4 ranking (overall) vs MiniLM baseline ===")
    for metric, c in results["comparison_vs_minilm_baseline"]["ranking_overall"].items():
        print(f"  {metric}: contrastive={c['contrastive']:.4f} "
              f"(95% CI {c['contrastive_ci_low']:.4f}-{c['contrastive_ci_high']:.4f}) "
              f"vs minilm={c['minilm_baseline']} ci_clear_win={c['ci_clear_win']}")
