#!/usr/bin/env python3
"""Why does this profile's BM25 projection disagree with ADR-006's 9.8 min?

ADR-006's addendum records: "2.31s / 1,000 users -> 590.2s projected (9.8 min)
for all 255,990 users" — i.e. **2.31 ms/user**. `profile_mind.py` measures
`score_all` at ~4.3 ms/user on the same machine class and the same corpus,
projecting to ~21 min. A 2x disagreement between two of this project's own
numbers cannot be left as a footnote; either the old number is wrong, the new
one is, or they measure different things.

The one methodological difference that is documented is the sample: ADR-006's
1,000 users were drawn by an unrecorded method, while `profile_mind.py` draws
systematically across the whole split. BM25 `score_all` cost scales with the
number of DISTINCT query terms, which scales with history length — so if the
old sample was a prefix (`head(1000)`) of a file whose ordering correlates with
history length, the two numbers are measuring different user populations, not
different code.

This script tests exactly that, and nothing else: same corpus, same index, same
`score_all`, two sampling methods. It reports the ratio, which is the part that
transfers across machines even though the absolute ms/user does not.

Run on a compute node (never the login node, never concurrently with another
benchmark).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.loaders import load_articles, split_dir  # noqa: E402
from benchmarks.timing import git_rev, hardware_label, hardware_spec, write_results  # noqa: E402
from src.retrieval.index import build_index  # noqa: E402
from src.retrieval.query import build_user_query  # noqa: E402
from src.retrieval.score import score_all  # noqa: E402

N = 1000
POP_USERS = 255_990


def bench(queries, index, label):
    for q in queries[:20]:
        score_all(index, q)
    ts = []
    for q in queries:
        t = time.perf_counter()
        score_all(index, q)
        ts.append(time.perf_counter() - t)
    a = np.asarray(ts)
    return {
        "sampling": label, "n": len(a),
        "mean_ms": round(float(a.mean() * 1e3), 4),
        "p50_ms": round(float(np.percentile(a, 50) * 1e3), 4),
        "p90_ms": round(float(np.percentile(a, 90) * 1e3), 4),
        "projected_all_users_min_from_mean": round(float(a.mean() * POP_USERS / 60), 2),
        "projected_all_users_min_from_p50": round(float(np.percentile(a, 50) * POP_USERS / 60), 2),
    }


def main() -> None:
    articles = load_articles("mind", "large", "dev")
    index = build_index(articles)
    text_lookup = dict(zip(
        articles["article_id"],
        articles["title"].fillna("") + " " + articles["abstract"].fillna(""),
    ))

    hist = pq.read_table(split_dir("mind", "large", "dev") / "user_history.parquet",
                         columns=["user_id", "article_ids"]).to_pandas()
    total = len(hist)

    prefix = hist.head(N)
    step = total // N
    systematic = hist.iloc[::step].head(N)

    out = {}
    for label, sub in (("prefix_head_1000", prefix), ("systematic_1000", systematic)):
        qs = [build_user_query(list(a), text_lookup) for a in sub["article_ids"]]
        hl = sub["article_ids"].map(len).to_numpy()
        tok = np.array([len(q) for q in qs])
        res = bench(qs, index, label)
        res.update({
            "history_len_mean": round(float(hl.mean()), 2),
            "history_len_p50": float(np.median(hl)),
            "query_tokens_mean": round(float(tok.mean()), 1),
            "query_tokens_p50": float(np.median(tok)),
        })
        out[label] = res

    p, s = out["prefix_head_1000"], out["systematic_1000"]
    payload = {
        "benchmark": "reconcile_adr006_bm25_projection",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_rev": git_rev(),
        "hardware_label": hardware_label(), "hardware": hardware_spec(),
        "corpus_articles": int(len(articles)), "total_users_in_split": int(total),
        "adr006_recorded": {"ms_per_user": 2.31, "projected_min": 9.8,
                            "source": "ADR-006 addendum 2026-08-11, 1,000-user sample"},
        "arms": out,
        "finding": {
            "prefix_vs_systematic_mean_ratio": round(s["mean_ms"] / p["mean_ms"], 3),
            "history_len_ratio": round(s["history_len_mean"] / p["history_len_mean"], 3),
            "query_tokens_ratio": round(s["query_tokens_mean"] / p["query_tokens_mean"], 3),
        },
    }
    path = write_results("reconcile_adr006", payload)
    for k, v in out.items():
        print(f"{k:20s} hist_len mean={v['history_len_mean']:6.1f} p50={v['history_len_p50']:5.0f} | "
              f"tokens mean={v['query_tokens_mean']:7.1f} | score_all mean={v['mean_ms']:7.3f}ms "
              f"p50={v['p50_ms']:7.3f}ms -> proj {v['projected_all_users_min_from_mean']:6.2f}min "
              f"(p50 {v['projected_all_users_min_from_p50']:6.2f}min)")
    print(f"\nprefix/systematic mean ratio: {payload['finding']['prefix_vs_systematic_mean_ratio']}")
    print(f"ADR-006 recorded 2.31 ms/user -> 9.8 min")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
