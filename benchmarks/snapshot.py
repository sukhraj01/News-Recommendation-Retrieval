#!/usr/bin/env python3
"""Lightweight before/after latency+throughput snapshot, for routine use.

The point of this script is that it is CHEAP enough to actually run before
every commit that touches a performance-relevant path. The full
`profile_mind.py` / `profile_ebnerd.py` / `ablations.py` suite is the
authoritative measurement and takes tens of minutes; this is the regression
tripwire, targeted at well under a minute on the local machine.

It deliberately runs against MINDsmall-dev rather than MINDlarge-dev. That
makes it a *relative* instrument, not an absolute one: the numbers here are
not the numbers ADR-014 reports, and the file says so in every record it
writes. What it detects is a CHANGE — a stage that was 3ms/query last commit
and is 30ms now — which is what a pre-commit check needs to catch and does not
require full scale to see.

Two modes:

    snapshot.py                  # measure, append to PERFORMANCE_LOG.md
    snapshot.py --compare        # measure, diff against the last snapshot on
                                 # this same hardware, exit 1 on a regression

`--compare` is what makes the habit enforceable rather than aspirational: it
returns a nonzero exit code when a stage regresses beyond `--threshold`, so it
can sit in a git hook or CI step and actually block, instead of printing a
number nobody reads.

Comparison is only ever made against a prior snapshot with the SAME hardware
label and the SAME sample parameters. Comparing an M3 run against an A100 run,
or 150 users against 1,200, would manufacture regressions out of nothing —
mismatched baselines are skipped with a message, never silently reinterpreted.

Usage:
    poetry run python benchmarks/snapshot.py --note "before: NRMS cache change"
    poetry run python benchmarks/snapshot.py --compare --note "after: NRMS cache"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.loaders import (  # noqa: E402
    embeddings_cache_path,
    load_articles,
    load_history,
    sample_impressions_by_user,
)
from benchmarks.timing import (  # noqa: E402
    RESULTS_DIR,
    StageTimer,
    git_dirty,
    git_rev,
    hardware_label,
    hardware_spec,
    peak_rss_gb,
)

LOG_PATH = Path(__file__).resolve().parent / "PERFORMANCE_LOG.md"
# Minimum absolute per-impression change, in ms, for a stage to be judged at
# all. See `compare` for the measurement that set this.
MIN_ABS_MS = 0.25
SNAPSHOT_DIR = RESULTS_DIR / "snapshots"
SEED = 0

# Paths whose modification should trigger a snapshot. Used by
# `--check-paths` (and the git hook) to decide whether this commit is
# performance-relevant at all — most commits are not, and running the snapshot
# on a docs-only change would train everyone to ignore it.
PERF_RELEVANT = (
    "src/retrieval/", "src/evaluation/ranking_metrics.py", "src/datasets/",
    "benchmarks/", "scripts/run_", "scripts/generate_",
)


def measure(bundle: str, split: str, n_users: int, device: str) -> dict:
    """The cheap subset: query construction, BM25 scoring, embedding scoring,
    candidate lookup, ranking. NRMS is excluded by default — it dominates the
    full profile precisely because it is slow, which would make a "lightweight"
    check anything but."""
    from src.evaluation.ranking_metrics import rank_candidates, safe_auc
    from src.retrieval.embed import (
        DEFAULT_MODEL, build_embedding_index, build_user_embedding_query, model_slug,
    )
    from src.retrieval.index import build_index
    from src.retrieval.query import build_user_query
    from src.retrieval.score import _lookup_scores, score_all

    articles = load_articles("mind", bundle, split)
    rows, sample_meta = sample_impressions_by_user("mind", bundle, split, n_users, seed=SEED)
    history = load_history("mind", bundle, split, rows["user_id"].unique())
    hist_by_user = dict(zip(history["user_id"], history["article_ids"]))

    timer = StageTimer()

    t0 = time.perf_counter()
    bm25 = build_index(articles)
    setup_bm25 = time.perf_counter() - t0

    cache = embeddings_cache_path("mind", bundle, split, model_slug(DEFAULT_MODEL))
    t0 = time.perf_counter()
    emb = build_embedding_index(articles, model_name=DEFAULT_MODEL, cache_path=cache)
    setup_embed = time.perf_counter() - t0

    text_lookup = dict(zip(
        articles["article_id"],
        articles["title"].fillna("") + " " + articles["abstract"].fillna(""),
    ))
    vector_lookup = dict(zip(emb.article_ids, emb.vectors))

    def one_pass(t: StageTimer) -> int:
        n = 0
        for uid, urows in rows.groupby("user_id", sort=False):
            hist = list(hist_by_user.get(uid, []))
            with t("query_tokenize"):
                q = build_user_query(hist, text_lookup)
            with t("bm25_score_all"):
                bfull = score_all(bm25, q)
            with t("embed_query_build"):
                ev = build_user_embedding_query(hist, vector_lookup)
            with t("embed_score_all"):
                efull = np.zeros(len(emb.article_ids)) if ev is None else emb.vectors @ ev
            for iid, imp in urows.groupby("impression_id", sort=False):
                cand = imp["article_id"].tolist()
                clicked = imp["clicked"].to_numpy(dtype=bool)
                with t("bm25_candidate_lookup"):
                    bs = _lookup_scores(bfull, bm25.id_to_col, cand)
                with t("embed_candidate_lookup"):
                    _lookup_scores(efull, emb.id_to_col, cand)
                with t("rank_and_auc"):
                    rank_candidates(bs, iid, seed=SEED)
                    safe_auc(bs, clicked)
                n += 1
        return n

    one_pass(StageTimer())  # warmup, discarded
    t0 = time.perf_counter()
    n_imp = one_pass(timer)
    wall = time.perf_counter() - t0

    stats = timer.summary()
    return {
        "hardware_label": hardware_label(),
        "hardware": hardware_spec(),
        "config": {"dataset": "mind", "bundle": bundle, "split": split,
                   "n_users": n_users, "device": device, "seed": SEED},
        "sample": sample_meta,
        "n_impressions": n_imp,
        "n_articles": int(len(articles)),
        "setup_s": {"bm25_index_build": round(setup_bm25, 4),
                    "embed_index_load": round(setup_embed, 4)},
        "wall_s": round(wall, 4),
        "impressions_per_s": round(n_imp / wall, 1),
        "ms_per_impression": round(wall * 1e3 / n_imp, 4),
        "stages_ms_per_impression": {
            s.stage: round(s.total_s * 1e3 / n_imp, 4) for s in stats
        },
        "stages_throughput_per_s": {s.stage: round(s.throughput_per_s, 1) for s in stats},
        "peak_rss_gb": round(peak_rss_gb(), 2),
    }


def _comparable_previous(current: dict) -> dict | None:
    """Most recent prior snapshot with the same hardware AND the same sample
    config. Anything else is not a baseline — see the module docstring."""
    if not SNAPSHOT_DIR.exists():
        return None
    for path in sorted(SNAPSHOT_DIR.glob("*.json"), reverse=True):
        try:
            prev = json.loads(path.read_text())
        except Exception:
            continue
        if (prev.get("hardware_label") == current["hardware_label"]
                and prev.get("config") == current["config"]):
            prev["_path"] = str(path)
            return prev
    return None


def compare(current: dict, previous: dict, threshold: float,
            min_abs_ms: float = MIN_ABS_MS) -> tuple[list[dict], bool]:
    """A stage counts as regressed only if it is BOTH relatively worse by more
    than `threshold` percent AND absolutely worse by more than `min_abs_ms`
    milliseconds per impression.

    The absolute floor is not padding — without it this gate is unusable.
    Measured directly: two back-to-back runs of this script with identical code
    and identical parameters moved `bm25_candidate_lookup` by -44.5% and
    `embed_query_build` by -20.4%, because those stages cost ~0.02-0.04
    ms/impression and are down in per-call timer noise. A gate that fires on
    that trains people to bypass it, which is worse than having no gate. The
    stages that actually matter here cost 1-35 ms/impression, well clear of the
    floor.
    """
    rows, regressed = [], False
    for stage, now in current["stages_ms_per_impression"].items():
        before = previous["stages_ms_per_impression"].get(stage)
        if before is None:
            rows.append({"stage": stage, "before_ms": None, "after_ms": now,
                         "pct_change": None, "abs_delta_ms": None, "status": "new"})
            continue
        pct = ((now - before) / before * 100.0) if before > 0 else 0.0
        abs_delta = now - before
        significant = abs(abs_delta) >= min_abs_ms
        if pct > threshold and significant:
            status = "REGRESSED"
            regressed = True
        elif pct < -threshold and significant:
            status = "improved"
        elif abs(pct) > threshold:
            status = "noise"  # relatively large, absolutely negligible
        else:
            status = "flat"
        rows.append({"stage": stage, "before_ms": before, "after_ms": now,
                     "pct_change": round(pct, 1), "abs_delta_ms": round(abs_delta, 4),
                     "status": status})
    return rows, regressed


def append_log(current: dict, previous: dict | None, diff: list[dict] | None,
               note: str, regressed: bool) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.write_text(_LOG_HEADER)

    lines = [
        f"\n### {current['timestamp']} — `{current['git_rev']}`"
        f"{' (dirty tree)' if current['git_dirty'] else ''}\n",
        f"**Hardware:** {current['hardware_label']} · "
        f"**Config:** MIND {current['config']['bundle']}/{current['config']['split']}, "
        f"{current['config']['n_users']} users, {current['n_impressions']} impressions, "
        f"{current['n_articles']:,} articles\n",
    ]
    if note:
        lines.append(f"**Note:** {note}\n")
    lines.append(
        f"\n**End-to-end:** {current['ms_per_impression']} ms/impression "
        f"({current['impressions_per_s']} impressions/s) · peak RSS "
        f"{current['peak_rss_gb']} GB\n\n"
    )

    if diff is not None and previous is not None:
        lines.append(f"Compared against `{Path(previous['_path']).name}` "
                     f"(`{previous['git_rev']}`):\n\n")
        lines.append("| Stage | Before (ms/imp) | After (ms/imp) | Change | |\n")
        lines.append("|---|---:|---:|---:|---|\n")
        for r in diff:
            mark = {"REGRESSED": "🔴", "improved": "🟢", "flat": "·",
                    "noise": "~", "new": "＋"}[r["status"]]
            before = f"{r['before_ms']}" if r["before_ms"] is not None else "—"
            pct = f"{r['pct_change']:+.1f}%" if r["pct_change"] is not None else "—"
            lines.append(f"| `{r['stage']}` | {before} | {r['after_ms']} | {pct} | {mark} |\n")
        lines.append(f"\n**Verdict:** {'REGRESSION DETECTED' if regressed else 'no regression'}\n")
    else:
        lines.append("| Stage | ms/impression | throughput (calls/s) |\n|---|---:|---:|\n")
        for stage, ms in current["stages_ms_per_impression"].items():
            lines.append(f"| `{stage}` | {ms} | "
                         f"{current['stages_throughput_per_s'][stage]:,.0f} |\n")
        lines.append("\n_(no comparable prior snapshot — recorded as a baseline)_\n")

    with LOG_PATH.open("a") as fh:
        fh.writelines(lines)


_LOG_HEADER = """# Performance Log

Rolling record of latency/throughput snapshots, one entry per run of
`benchmarks/snapshot.py`. Written automatically — do not hand-edit entries.

**What these numbers are.** A cheap regression tripwire measured on
**MINDsmall-dev**, not the authoritative full-scale profile. They are meaningful
as a *change* against the previous entry with the same hardware and config; they
are NOT the numbers ADR-014 reports, which come from `profile_mind.py` /
`profile_ebnerd.py` / `ablations.py` at real MINDlarge / ebnerd_small scale.

**The habit.** Any commit touching a performance-relevant path
(`src/retrieval/`, `src/evaluation/ranking_metrics.py`, `src/datasets/`,
`benchmarks/`, `scripts/run_*`, `scripts/generate_*`) gets a before/after pair:

```bash
# before the change
poetry run python benchmarks/snapshot.py --note "before: <what you're about to do>"
# after the change
poetry run python benchmarks/snapshot.py --compare --note "after: <what you did>"
```

`--compare` exits nonzero on a regression past the threshold, so it can gate a
commit rather than merely inform one. See ADR-014 for the methodology and for
why a regression here is worth blocking on.

---
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default="small")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n-users", type=int, default=150)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compare", action="store_true",
                    help="diff against the last comparable snapshot; exit 1 on regression")
    ap.add_argument("--threshold", type=float, default=15.0,
                    help="percent slowdown counted as a regression (default 15)")
    ap.add_argument("--min-abs-ms", type=float, default=MIN_ABS_MS,
                    help="minimum absolute ms/impression change to judge a stage at all "
                         f"(default {MIN_ABS_MS}); below this, timer noise dominates")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-log", action="store_true", help="measure only, don't touch the log")
    ap.add_argument("--check-paths", nargs="*", default=None,
                    help="exit 0 immediately if none of these paths are performance-relevant")
    args = ap.parse_args()

    if args.check_paths is not None:
        relevant = [p for p in args.check_paths
                    if any(p.startswith(pref) for pref in PERF_RELEVANT)]
        if not relevant:
            print("[snapshot] no performance-relevant paths changed — skipping")
            return
        print(f"[snapshot] performance-relevant paths changed: {', '.join(relevant)}")

    current = measure(args.bundle, args.split, args.n_users, args.device)
    current["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    current["git_rev"] = git_rev()
    current["git_dirty"] = git_dirty()
    current["note"] = args.note

    previous = _comparable_previous(current) if args.compare else None
    diff, regressed = (compare(current, previous, args.threshold, args.min_abs_ms)
                       if previous else (None, False))

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out = SNAPSHOT_DIR / f"{time.strftime('%Y-%m-%dT%H-%M-%S')}_snapshot.json"
    out.write_text(json.dumps(current, indent=2, default=str))

    if not args.no_log:
        append_log(current, previous, diff, args.note, regressed)

    print(f"[snapshot] {current['hardware_label']} · "
          f"{current['ms_per_impression']} ms/impression "
          f"({current['impressions_per_s']} impressions/s)")
    for stage, ms in current["stages_ms_per_impression"].items():
        print(f"    {stage:26s} {ms:9.4f} ms/imp")
    if diff:
        print()
        for r in diff:
            if r["pct_change"] is not None:
                print(f"    {r['stage']:26s} {r['before_ms']:8.4f} -> {r['after_ms']:8.4f} "
                      f"({r['pct_change']:+6.1f}%, {r['abs_delta_ms']:+.3f}ms)  {r['status']}")
    elif args.compare:
        print("    (no comparable prior snapshot — recorded as a baseline)")
    print(f"[snapshot] written {out.name}"
          f"{'' if args.no_log else f' + appended {LOG_PATH.name}'}")

    if regressed:
        print(f"\n[snapshot] REGRESSION: a stage slowed by more than {args.threshold}%.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
