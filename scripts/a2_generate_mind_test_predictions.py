#!/usr/bin/env python3
"""Generate the official-format MINDlarge_test Codabench prediction file from
a trained `OfficialNRMS` checkpoint (ADR-015, A2 Q3's MIND treatment).

Routed through `src/submission/mind_format.py::write_predictions` unchanged
-- the same module every real submission from this project has used -- via
`OfficialNRMSScorer` (`src/retrieval/nrms_official.py`) implementing the
same `Scorer` protocol Candidate J's `NRMSLiteScorer` already does.

Unlike Candidate J's equivalent script (`generate_mind_nrms_predictions.py`),
there is no vocabulary to reconstruct: `OfficialNRMS` uses the OFFICIAL,
FIXED `word_dict.pkl`/`embedding.npy` (the same files used at training time,
sha256-verified when staged), not a vocabulary built from scratch. The
correctness check that remains is the one every checkpoint load gets for
free: `model.load_state_dict()` raises on any shape mismatch.

Test-catalog-only scoring, per the SAME real bug this project already hit
once building Candidate J's equivalent script: article tokens are built
from MINDlarge_test's OWN news.tsv only, never unioned with train/dev's
catalog -- a candidate real in train/dev but absent from test's own news.tsv
must fall through to the documented -inf missing-candidate convention, not
resolve to a borrowed (wrong-split) row.

Usage:
    python scripts/a2_generate_mind_test_predictions.py \
        --checkpoint $HOME/a2/results/mind_treatment/model_weights.pt \
        --mind-test-zip $HOME/a2/raw/MINDlarge_test.zip \
        --mind-utils-dir $HOME/a2/mind_utils \
        --abstract-size 50 --out-dir $HOME/a2/results/mind_treatment_test
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

from src.retrieval.nrms_official import OfficialNRMS, OfficialNRMSScorer  # noqa: E402
from src.retrieval.nrms_official_data import build_mind_news_tokens  # noqa: E402
from src.submission.mind_format import write_predictions  # noqa: E402
from src.utils.ids import prefix_id  # noqa: E402


def read_news_tsv_raw(zip_path: Path) -> tuple[list[str], list[str], list[str]]:
    """(ids, titles, abstracts), unprefixed, in file order -- the SAME raw
    parse `a2_nrms_official_run.py::load_mind_split` used at training time
    (MIND's fixed official column order: id, category, subcategory, title,
    abstract, url, title_entities, abstract_entities)."""
    z = zipfile.ZipFile(zip_path)
    member = [m for m in z.namelist() if m.endswith("/news.tsv") or m == "news.tsv"][0]
    with z.open(member) as f:
        rows = [ln.decode("utf-8").strip("\n").split("\t") for ln in f]
    return [r[0] for r in rows], [r[3] for r in rows], [r[4] for r in rows]


def read_history_raw(zip_path: Path) -> dict[str, list[str]]:
    """user_id -> prefixed history article_ids, from the test split's own
    behaviors.tsv (MIND's static-per-user-per-split history convention,
    ADR-002) -- the query side `OfficialNRMSScorer` needs."""
    z = zipfile.ZipFile(zip_path)
    member = [m for m in z.namelist() if m.endswith("/behaviors.tsv") or m == "behaviors.tsv"][0]
    hist_by_user: dict[str, list[str]] = {}
    with z.open(member) as f:
        for ln in f:
            r = ln.decode("utf-8").strip("\n").split("\t")
            user_id, history = r[1], r[3]
            if user_id not in hist_by_user:  # static per user for the split; first sighting suffices
                hist_by_user[user_id] = [prefix_id("mind", a) for a in history.split(" ")] if history else []
    return hist_by_user


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="model_weights.pt from the real training run")
    ap.add_argument("--mind-test-zip", type=Path, required=True)
    ap.add_argument("--mind-utils-dir", type=Path, required=True)
    ap.add_argument("--abstract-size", type=int, default=0,
                    help="0 for the control (title-only); 50 for the treatment (ADR-015)")
    ap.add_argument("--title-size", type=int, default=30)  # nrms.yaml, unchanged across arms
    ap.add_argument("--max-history-len", type=int, default=50)  # nrms.yaml his_size
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__}, device {device}", flush=True)

    word_dict = pickle.load(open(args.mind_utils_dir / "word_dict.pkl", "rb"))
    embedding = np.load(args.mind_utils_dir / "embedding.npy").astype(np.float32)
    print(f"official vocab/embedding: {embedding.shape} (fixed, not reconstructed)", flush=True)

    t0 = time.time()
    news_ids, titles, abstracts = read_news_tsv_raw(args.mind_test_zip)
    nid2index, tokens = build_mind_news_tokens(
        news_ids, titles, abstracts, word_dict, args.title_size, args.abstract_size
    )
    # Test-catalog-only: token_lookup covers ONLY articles real in this
    # split's own news.tsv (nid2index's keys), so a candidate absent here
    # (documented MIND missing-candidate quirk) falls through to
    # OfficialNRMSScorer's -inf convention rather than resolving to a
    # wrong-split row -- the exact bug this project already hit once
    # building Candidate J's equivalent script.
    token_lookup = {prefix_id("mind", aid): tokens[idx] for aid, idx in nid2index.items()}
    print(f"test catalog: {len(token_lookup):,} articles, width {args.title_size + args.abstract_size}, "
          f"{time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    query_by_user = read_history_raw(args.mind_test_zip)
    print(f"test history: {len(query_by_user):,} users, {time.time()-t0:.1f}s", flush=True)

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
