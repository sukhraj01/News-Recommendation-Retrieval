"""A2 Q3/Q5: metrics, guardrails, and the paired A/B test over runner outputs (ADR-015).

Reads the `scores.parquet` files written by `a2_nrms_official_run.py`, one
per arm (impression_id, user_id, labels, scores), and reports in A/B-test
framing:

  control    = reproduced official-config NRMS
  treatment  = control + one principled change
  primary    = AUC; also MRR, nDCG@5, nDCG@10
  guardrails = diversity@10 and novelty@10 (must not regress), plus
               coverage@10 (a point estimate, no CI, per ADR-007)
  test       = paired bootstrap 95% CI of (treatment - control) over the SAME
               impressions, resampling users (not impressions).
               `run_gated_cohort_experiment.paired_metric_diff_ci`, ADR-010's
               statistic, is imported, not reimplemented. A gain is claimed
               only if the CI excludes zero; a guardrail regresses if its CI is
               entirely below zero.

Every metric is the project's own harness (`src/evaluation/ranking_metrics.py`),
built exactly as `run_ranking_eval.py` builds A1's numbers:
  - tie-exact vectorised AUC
  - MRR/nDCG/top-10 under ADR-007's deterministic tie-break, seeded with A1's
    processed impression ids (`mind:dev:<id>`, `ebnerd:validation:<id>`)
  - categories from A1's processed articles.parquet
  - popularity for novelty from the TRAIN split only (Q9), via
    `build_train_popularity`, Laplace alpha 1.0
  - the coverage denominator is the eval split's catalog size.

The guardrails need each candidate's article id. The runner stores only
labels + scores, in the raw zip's candidate order, so ids are re-read from
the raw zip by impression_id. Every impression's raw candidate count and
labels must equal the runner's, or the script refuses to continue.

Usage:
  python scripts/a2_evaluate_scores.py --control C/scores.parquet \
      [--treatment T/scores.parquet] --out results.json \
      [--dataset mind --source-zip data/raw/mind/MINDlarge_dev.zip \
       --articles data/processed/mind/large/dev/articles.parquet \
       --train-impressions data/processed/mind/large/train/impressions.parquet]
"""
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.evaluation.ranking_metrics import (  # noqa: E402
    build_train_popularity,
    coverage,
    intra_list_diversity,
    mrr,
    ndcg_at_k,
    novelty,
    per_impression_auc,
    rank_candidates,
    ranking_metric_ci,
)
from run_gated_cohort_experiment import paired_metric_diff_ci  # noqa: E402

ACCURACY = ("auc", "mrr", "ndcg5", "ndcg10")
GUARDRAILS = ("diversity10", "novelty10")
K_BEYOND = 10  # ADR-007; same as run_ranking_eval.K_DIVERSITY_NOVELTY
TIEBREAK_PREFIX = {"mind": "mind:dev:", "ebnerd": "ebnerd:validation:"}
ID_PREFIX = {"mind": "mind:", "ebnerd": "ebnerd:"}


def raw_candidates(dataset: str, source_zip: str) -> dict[int, tuple[list[str], list[int]]]:
    """impression_id -> (prefixed candidate ids, labels), in raw order."""
    z = zipfile.ZipFile(source_zip)
    out: dict[int, tuple[list[str], list[int]]] = {}
    if dataset == "mind":
        member = [m for m in z.namelist() if m.endswith("behaviors.tsv")][0]
        with z.open(member) as f:
            for ln in f:
                r = ln.decode("utf-8").strip("\n").split("\t")
                toks = [t.split("-") for t in r[4].split()]
                out[int(r[0])] = ([f"mind:{n}" for n, _ in toks], [int(lab) for _, lab in toks])
        return out
    t = pq.read_table(io.BytesIO(z.read("validation/behaviors.parquet")),
                      columns=["impression_id", "article_ids_inview", "article_ids_clicked"]).to_pandas()
    for imp, inview, clicked in zip(t["impression_id"], t["article_ids_inview"], t["article_ids_clicked"]):
        cl = {int(x) for x in clicked}
        out[int(imp)] = ([f"ebnerd:{int(a)}" for a in inview], [int(int(a) in cl) for a in inview])
    return out


def train_popularity(train_impressions: str, n_catalog: int) -> dict[str, float]:
    """A1's `build_train_popularity`, fed only the clicked rows. It filters
    on `clicked` itself, so the result is identical, and filtering in pyarrow
    first keeps MINDlarge's 83.5M-row train table at ~250MB, not ~5GB."""
    t = pq.read_table(train_impressions, columns=["article_id", "clicked"])
    t = t.filter(pc.field("clicked"))
    df = t.to_pandas()
    df["article_id"] = df["article_id"].astype(str)
    return build_train_popularity(df, n_catalog=n_catalog, alpha=1.0)


def per_impression_metrics(df: pd.DataFrame, tiebreak_prefix: str = "", cands=None,
                           category=None, popularity=None) -> tuple[pd.DataFrame, list]:
    """One row per impression (user_id, accuracy metrics, and guardrails when
    candidate ids are given), plus the list of top-10 id lists for coverage."""
    sizes = df["labels"].map(len).to_numpy()
    flat_scores = np.concatenate(df["scores"].map(np.asarray).to_list()).astype(np.float64)
    flat_labels = np.concatenate(df["labels"].map(np.asarray).to_list()).astype(bool)
    out = pd.DataFrame({
        "impression_id": df["impression_id"].to_numpy(),
        "user_id": df["user_id"].astype(str).to_numpy(),
        "auc": per_impression_auc(flat_scores, flat_labels, sizes),
    })
    cols = {k: [] for k in ("mrr", "ndcg5", "ndcg10", "diversity10", "novelty10")}
    top_lists = []
    for imp, sc, lab in zip(df["impression_id"], df["scores"], df["labels"]):
        order = rank_candidates(np.asarray(sc), f"{tiebreak_prefix}{imp}")
        ranked = np.asarray(lab, dtype=bool)[order]
        cols["mrr"].append(mrr(ranked))
        cols["ndcg5"].append(ndcg_at_k(ranked, 5))
        cols["ndcg10"].append(ndcg_at_k(ranked, 10))
        if cands is not None:
            ids, raw_labels = cands[int(imp)]
            if len(ids) != len(lab) or list(raw_labels) != list(map(int, lab)):
                sys.exit(f"FATAL: impression {imp}: runner candidates/labels do not match the raw zip "
                         f"({len(lab)} vs {len(ids)}); guardrail ids would be misaligned")
            top = [ids[i] for i in order[:min(K_BEYOND, len(ids))]]
            top_lists.append(top)
            cols["diversity10"].append(intra_list_diversity([category.get(a) for a in top]))
            cols["novelty10"].append(novelty(top, popularity))
    for k in ("mrr", "ndcg5", "ndcg10"):
        out[k] = cols[k]
    if cands is not None:
        out["diversity10"], out["novelty10"] = cols["diversity10"], cols["novelty10"]
    return out, top_lists


def arm_summary(pi: pd.DataFrame, top_lists: list, catalog_size: int | None) -> dict:
    res = {}
    for m in ACCURACY + GUARDRAILS:
        if m not in pi:
            continue
        r = ranking_metric_ci(pi, m)
        res[m] = {"value": r.metric, "ci_low": r.ci_low, "ci_high": r.ci_high,
                  "n_impressions": r.n_impressions, "n_users": r.n_users, "n_skipped": r.n_skipped}
    if top_lists and catalog_size:
        res["coverage10"] = {"value": coverage(top_lists, catalog_size), "ci_low": None, "ci_high": None,
                             "note": "point estimate; set-union statistic, no bootstrap CI (ADR-007)"}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--treatment")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", choices=["mind", "ebnerd"],
                    help="enables A1-style tie-break ids and (with the three paths below) guardrails")
    ap.add_argument("--source-zip")
    ap.add_argument("--articles")
    ap.add_argument("--train-impressions")
    a = ap.parse_args()

    tb = TIEBREAK_PREFIX.get(a.dataset, "")
    cands = category = popularity = catalog_size = None
    guardrails_on = bool(a.dataset and a.source_zip and a.articles and a.train_impressions)
    if guardrails_on:
        arts = pd.read_parquet(a.articles, columns=["article_id", "category"])
        category = dict(zip(arts["article_id"], arts["category"]))
        catalog_size = len(arts)
        popularity = train_popularity(a.train_impressions, catalog_size)
        cands = raw_candidates(a.dataset, a.source_zip)

    ctrl = pd.read_parquet(a.control)
    pc_, top_c = per_impression_metrics(ctrl, tb, cands, category, popularity)
    report = {"tiebreak_prefix": tb, "guardrails": guardrails_on,
              "control": {"file": a.control, **arm_summary(pc_, top_c, catalog_size)}}

    if a.treatment:
        treat = pd.read_parquet(a.treatment)
        # Pairing precondition: same impressions, same candidate lists.
        m = ctrl[["impression_id", "labels"]].merge(
            treat[["impression_id", "labels"]], on="impression_id", suffixes=("_c", "_t"))
        if len(m) != len(ctrl) or len(m) != len(treat):
            sys.exit(f"FATAL: arms cover different impressions ({len(ctrl)} vs {len(treat)}, "
                     f"{len(m)} shared); a paired test would be invalid")
        if not all(list(x) == list(y) for x, y in zip(m["labels_c"], m["labels_t"])):
            sys.exit("FATAL: candidate labels differ between arms for the same impression")
        pt, top_t = per_impression_metrics(treat, tb, cands, category, popularity)
        report["treatment"] = {"file": a.treatment, **arm_summary(pt, top_t, catalog_size)}
        both = pc_.merge(pt, on=["impression_id", "user_id"], suffixes=("_control", "_treatment"))
        paired = {}
        for mname in ACCURACY + GUARDRAILS:
            if f"{mname}_control" not in both:
                continue
            point, lo, hi = paired_metric_diff_ci(both, f"{mname}_treatment", f"{mname}_control")
            paired[mname] = {"diff": point, "ci_low": lo, "ci_high": hi,
                             "significant": bool(lo > 0 or hi < 0)}
            if mname in GUARDRAILS:
                paired[mname]["regressed"] = bool(hi < 0)
        if "coverage10" in report["control"]:
            paired["coverage10"] = {"diff": report["treatment"]["coverage10"]["value"]
                                    - report["control"]["coverage10"]["value"],
                                    "note": "point difference only (ADR-007)"}
        report["paired_treatment_minus_control"] = paired

    Path(a.out).write_text(json.dumps(report, indent=1))
    for arm in ("control", "treatment"):
        if arm in report:
            print(arm, {k: round(v["value"], 4) for k, v in report[arm].items()
                        if isinstance(v, dict) and "value" in v})
    for k, v in report.get("paired_treatment_minus_control", {}).items():
        if "ci_low" in v:
            flag = "SIGNIFICANT" if v["significant"] else "n.s."
            if v.get("regressed"):
                flag += " GUARDRAIL REGRESSED"
            print(f"paired {k}: {v['diff']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}] {flag}")
        else:
            print(f"paired {k}: {v['diff']:+.4f} (point only)")


if __name__ == "__main__":
    main()
