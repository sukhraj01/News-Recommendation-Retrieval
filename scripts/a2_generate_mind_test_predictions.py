#!/usr/bin/env python3
"""Generate the official-format MINDlarge_test Codabench prediction file from
a trained `OfficialNRMS` checkpoint (ADR-015, A2 Q3's MIND treatment).

Routed through `src/submission/mind_format.py::write_predictions` unchanged
-- the same module every real submission from this project has used -- via
`OfficialNRMSScorer` (`src/retrieval/nrms_official.py`) implementing the
same `Scorer` protocol Candidate J's `NRMSLiteScorer` already does.

Test's article catalog and per-user history come from
`src/pipeline/orchestrator.py::build_mind_test` -- the same tested,
schema-validated feature-store path Candidate J's equivalent script
(`generate_mind_nrms_predictions.py`) and the Q2 retrieve-rerank harness
(`a2_q2_retrieve_rerank_eval.py`) already use, NOT a second hand-rolled
TSV parse.

This replaces an earlier version of this script that hand-rolled its own
`news.tsv`/`behaviors.tsv` readers instead of going through
`build_mind_test`. That duplication is what caused a real, shipped bug
(ADR-015's 2026-09-17 addendum): the hand-rolled history reader keyed its
dict by the RAW user_id ("U84103"), but `mind_format.py::read_raw_impressions`
looks history up by the PREFIXED user_id ("mind:U84103") -- every lookup
missed, so every one of MINDlarge_test's 2,370,727 impressions was scored
with an empty (all-padding) history regardless of the user's real one.
`OfficialNRMSScorer` was already designed against the orchestrator's output
("query ... as stored in user_history.parquet", per its own docstring); the
hand-rolled reader was the deviation, and the class of risk it introduced
(a second parser that can silently drift from the tested one) does not go
away with a narrower patch. Retiring it removes the duplication instead of
patching around it.

Test-catalog-only scoring is preserved by construction: `build_mind_test`
parses ONLY MINDlarge_test's own news.tsv into `articles.parquet` (never
unioned with train/dev), so a candidate real in train/dev but absent from
test's own news.tsv still falls through to the documented -inf
missing-candidate convention (`OfficialNRMSScorer`), exactly as the SAME
real bug already caught once in Candidate J's script requires.

Usage:
    python scripts/a2_generate_mind_test_predictions.py \
        --checkpoint $HOME/a2/results/mind_treatment_v2/model_weights.pt \
        --mind-test-zip $HOME/a2/raw/MINDlarge_test.zip \
        --mind-utils-dir $HOME/a2/mind_utils \
        --data-dir $HOME/a2/data_large \
        --abstract-size 50 --out-dir $HOME/a2/results/mind_treatment_v2_test
"""
import argparse
import pickle
import sys
import time
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.pipeline.orchestrator import build_mind_test  # noqa: E402
from src.retrieval.nrms_official import OfficialNRMS, OfficialNRMSScorer  # noqa: E402
from src.retrieval.nrms_official_data import build_mind_news_tokens  # noqa: E402
from src.submission.mind_format import read_raw_impressions, write_predictions  # noqa: E402


def history_hit_rate(zip_path: Path, query_by_user: dict) -> tuple[float, int, int]:
    """Fraction of REAL impressions whose user maps to a non-empty history --
    a fast (no model, no GPU), full-scale sanity gate to run before any
    GPU time or Codabench upload. Catches exactly the class of bug this
    script's previous version shipped: a key/format mismatch that makes
    every `query_by_user` lookup silently miss and fall back to empty.

    This is a NECESSARY-not-sufficient check: it would not catch a scorer
    that used history incorrectly once retrieved, only a pipeline that
    fails to retrieve it at all. Paired with `scripts/a2_check_mind_history_coverage.py`
    (which additionally compares against the raw ground-truth non-empty-
    history rate) for the full pre-upload gate."""
    rows = read_raw_impressions(zip_path, has_labels=False)
    n_hit = sum(1 for r in rows if len(query_by_user.get(r["user_id"], [])) > 0)
    return n_hit / len(rows), n_hit, len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="model_weights.pt from the real training run")
    ap.add_argument("--mind-test-zip", type=Path, required=True)
    ap.add_argument("--mind-utils-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="scratch dir for the built parquet feature store "
                         "(src/pipeline/orchestrator.py::build_mind_test writes here)")
    ap.add_argument("--abstract-size", type=int, default=0,
                    help="0 for the control (title-only); 50 for the treatment (ADR-015)")
    ap.add_argument("--title-size", type=int, default=30)  # nrms.yaml, unchanged across arms
    ap.add_argument("--max-history-len", type=int, default=50)  # nrms.yaml his_size
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-history-hit-rate", type=float, default=0.30,
                    help="abort before any GPU scoring if the fraction of real impressions "
                         "with non-empty mapped history falls below this. MINDlarge's real "
                         "non-cold-start rate is well above this threshold in every split "
                         "this project has measured; a value this low means the query_by_user "
                         "join is broken, not that users genuinely lack history.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__}, device {device}", flush=True)

    print("Building MINDlarge_test articles/candidates/user_history via the real loader "
          "(src/pipeline/orchestrator.py::build_mind_test)...", flush=True)
    t0 = time.time()
    build_mind_test(args.mind_test_zip, args.data_dir)
    test_articles = pd.read_parquet(args.data_dir / "test" / "articles.parquet")
    test_history = pd.read_parquet(args.data_dir / "test" / "user_history.parquet")
    print(f"test: {len(test_articles):,} articles, {len(test_history):,} users with history "
          f"({time.time()-t0:.1f}s)", flush=True)

    query_by_user = dict(zip(test_history["user_id"], test_history["article_ids"]))

    # Fast, full-scale, no-GPU preflight: catches a broken join before any
    # GPU time is spent, not just on a hand-sampled handful of impressions.
    hit_rate, n_hit, n_total = history_hit_rate(args.mind_test_zip, query_by_user)
    print(f"history hit-rate preflight: {n_hit:,}/{n_total:,} impressions ({hit_rate:.1%}) "
          f"map to a non-empty history", flush=True)
    if hit_rate < args.min_history_hit_rate:
        sys.exit(f"FATAL: history hit-rate {hit_rate:.1%} is below --min-history-hit-rate "
                  f"{args.min_history_hit_rate:.1%} -- the query_by_user join is almost "
                  f"certainly broken (this is exactly how ADR-015's 2026-09-17 bug presented). "
                  f"Not spending GPU time scoring a run that would need to be redone.")

    word_dict = pickle.load(open(args.mind_utils_dir / "word_dict.pkl", "rb"))
    embedding = np.load(args.mind_utils_dir / "embedding.npy").astype(np.float32)
    print(f"official vocab/embedding: {embedding.shape} (fixed, not reconstructed)", flush=True)

    t0 = time.time()
    # test_articles' article_id is ALREADY prefixed (src/datasets/mind.py's
    # parse_mind_test_candidates), so it is used directly as the token_lookup
    # key -- no extra prefix_id round-trip needed here.
    nid2index, tokens = build_mind_news_tokens(
        test_articles["article_id"].tolist(), test_articles["title"].tolist(),
        test_articles["abstract"].tolist(), word_dict, args.title_size, args.abstract_size,
    )
    token_lookup = {aid: tokens[idx] for aid, idx in nid2index.items()}
    print(f"test catalog: {len(token_lookup):,} articles, width {args.title_size + args.abstract_size}, "
          f"{time.time()-t0:.1f}s", flush=True)

    model = OfficialNRMS(embedding, head_num=20, head_dim=20, attention_hidden_dim=200,
                         dropout=0.2).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)  # raises on any shape mismatch
    model.eval()
    print(f"loaded {args.checkpoint} -- shapes matched exactly", flush=True)

    scorer = OfficialNRMSScorer(model, token_lookup, args.max_history_len, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "prediction.txt"
    t0 = time.time()
    n = write_predictions(out_path, args.mind_test_zip, scorer, query_by_user, [],
                          has_labels=False, seed=args.seed)
    elapsed = time.time() - t0
    print(f"wrote {n:,} prediction lines in {elapsed:.1f}s ({elapsed/60:.1f}min) -> {out_path}", flush=True)

    zip_path = args.out_dir / "prediction.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_path, arcname="prediction.txt")
    print(f"packaged -> {zip_path}", flush=True)


if __name__ == "__main__":
    main()
