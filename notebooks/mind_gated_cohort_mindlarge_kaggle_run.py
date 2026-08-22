# %% CELL 1 — setup: unpack the bundled project code, then force-write the
# two files that had real cross-platform bugs (flat-root zip packing on
# the HF MIND mirror; hardcoded Mac-only "mps" device) directly with their
# fixed content. This makes Cell 1 self-healing regardless of which
# version of the uploaded bundle is actually attached — no dependency on
# re-uploading correctly.
import os
import shutil
import subprocess
import sys
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rank_bm25", "sentence-transformers"], check=True)

REPO_DIR = "/kaggle/working/repo"
BUNDLE_NAME = "mind_gated_cohort_mindlarge_src_bundle.zip"
REQUIRED_SUBDIRS = {"src", "scripts", "experiments"}

bundle_zip = None
extracted_dir = None
for root, dirs, files in os.walk("/kaggle/input"):
    if BUNDLE_NAME in files:
        bundle_zip = os.path.join(root, BUNDLE_NAME)
        break
    if REQUIRED_SUBDIRS.issubset(set(dirs)):
        extracted_dir = root
        break

if not os.path.exists(REPO_DIR):
    if bundle_zip is not None:
        os.makedirs(REPO_DIR)
        shutil.unpack_archive(bundle_zip, REPO_DIR, "zip")
    elif extracted_dir is not None:
        shutil.copytree(extracted_dir, REPO_DIR)
    else:
        raise FileNotFoundError(
            f"Neither {BUNDLE_NAME} nor an extracted src/scripts/experiments folder "
            "found under /kaggle/input — upload the bundle as a Notebook input first."
        )

(Path(REPO_DIR) / "src/utils/io.py").write_text(r'''
"""Zip-aware raw file readers and a parquet write wrapper.

Raw MIND/EB-NeRD bundles stay zipped on disk (data/raw/, gitignored);
nothing is extracted. EB-NeRD zips also contain __MACOSX/ junk metadata
entries — callers must read the exact named member, never glob the zip.

`read_zip_member_bytes`'s fallback addendum (2026-08-12, Part 2 execution):
`ebnerd_small.zip`/`ebnerd_demo.zip` both pack members at a flat path
(`articles.parquet`, `{split}/behaviors.parquet`) and every EB-NeRD caller
in this project (`src/datasets/ebnerd.py`, `src/submission/ebnerd_format.py`)
was written and tested against that layout. The real `ebnerd_testset.zip`
does not follow it — running Part 2 for real on Kaggle surfaced that its
members are wrapped in an extra top-level directory
(`ebnerd_testset/test/behaviors.parquet`, not `test/behaviors.parquet`),
which raised a `KeyError` on the first real read. Rather than hardcoding
that one prefix (an `articles_large_only.zip`-shaped bundle could easily
use a different one, and guessing would just move the same fragility
elsewhere), the exact-path lookup is tried first (unchanged, fast, and
still what every existing fixture/test exercises) and only falls back to
a suffix search across the real namelist (excluding `__MACOSX/` junk) if
that fails — so every caller's already-correct flat member names keep
working unmodified against either packaging convention.

Second fallback addendum (2026-08-21, MINDlarge Kaggle verification, ADR-010):
the opposite mismatch. `src/datasets/mind.py` always requests
`f"{zip_path.stem}/news.tsv"` (e.g. `MINDlarge_train/news.tsv`) because
every MIND zip downloaded from the official source packs members inside a
folder named after the zip itself — confirmed directly at this project's
very first session and true of every local MIND zip since. The Hugging
Face mirror this project's own `download.py` uses as its download source
does NOT follow that convention for `MINDlarge_train.zip`: real download,
inspected directly, packs `news.tsv`/`behaviors.tsv`/etc. flat at the zip
root with no folder prefix at all — the suffix fallback above can't match
this (it requires a `/` before `member_name`, which a genuinely flat entry
never has), so it raised `KeyError` with zero suffix candidates on a real
Kaggle run. Same fix philosophy as the first addendum: try the caller's
existing exact/suffix-based lookups first (unchanged), and only fall back
further, to an exact basename match among flat (no `/`) entries, if both
fail — never guessing which convention a *new* bundle will use, just
handling the ones actually observed in the wild so far.
"""
import zipfile
from pathlib import Path

import pandas as pd


def read_zip_member_bytes(zip_path: Path, member_name: str) -> bytes:
    with zipfile.ZipFile(zip_path) as z:
        try:
            with z.open(member_name) as f:
                return f.read()
        except KeyError:
            pass

        namelist = [n for n in z.namelist() if not n.startswith("__MACOSX/")]
        suffix_candidates = [n for n in namelist if n.endswith("/" + member_name)]
        basename = member_name.rsplit("/", 1)[-1]
        flat_candidates = [n for n in namelist if n == basename]
        candidates = suffix_candidates or flat_candidates

        if len(candidates) != 1:
            raise KeyError(
                f"{member_name!r} not found in {zip_path} as an exact "
                f"member name, and neither fallback found exactly 1 "
                f"candidate (suffix: {suffix_candidates}, flat: {flat_candidates})"
            ) from None
        with z.open(candidates[0]) as f:
            return f.read()


def read_zip_tsv(zip_path: Path, member_name: str, names: list[str]) -> pd.DataFrame:
    import io

    raw = read_zip_member_bytes(zip_path, member_name)
    return pd.read_csv(
        io.BytesIO(raw),
        sep="\t",
        header=None,
        names=names,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )


def read_zip_parquet(zip_path: Path, member_name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """`columns=None` reads every column (default, unchanged behavior).
    Passing an explicit subset avoids materializing columns a caller never
    reads — real cost at real scale (see `ebnerd_format.py`'s addendum)."""
    import io

    raw = read_zip_member_bytes(zip_path, member_name)
    return pd.read_parquet(io.BytesIO(raw), columns=columns)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame the caller has already deterministically sorted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
''')

(Path(REPO_DIR) / "src/retrieval/embed.py").write_text(r'''
"""Embedding-based (semantic) index construction and query representation.

Phase 4's second `Scorer` consumer, per ADR-008. Mirrors `index.py`/`query.py`'s
BM25 shapes deliberately:

- `EmbeddingIndex` is `BM25Index`'s counterpart — `article_ids`/`id_to_col`
  are identical in spirit, `vectors` (dense, L2-normalized) replaces the
  sparse BM25 weight matrix.
- `build_user_embedding_query` is `build_user_query`'s counterpart —
  mean-pool a history's article vectors instead of concatenating tokens.
  Empty history returns `None` (not an empty array), matching
  `build_user_query`'s empty-list return for the same true-cold-start case
  (ADR-005) — `None` is used here rather than an all-zero vector because a
  zero vector is a valid (if degenerate) point in embedding space, whereas
  "no representation exists" must be structurally distinguishable from it.

Per ADR-008: `title + " " + abstract` is encoded once per article with a
purpose-built multilingual sentence-embedding model
(`paraphrase-multilingual-MiniLM-L12-v2`, chosen over mean-pooled raw
BERT/XLM-R and over `multilingual-e5-small` — see ADR-008's benchmark
evidence), and the result is cached to disk keyed by `(model_name,
article_ids order)`: encoding ~75k articles through a transformer is not
the ~1s BM25 index build was, so unlike `index.py`, rebuilding on every
script invocation would be real, avoidable, repeated cost.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_BATCH_SIZE = 64


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


@dataclass
class EmbeddingIndex:
    article_ids: list[str]
    id_to_col: dict[str, int]
    vectors: np.ndarray  # (n_docs, dim), L2-normalized rows


def _default_device() -> str:
    """Best available backend, auto-detected rather than hardcoded (2026-08-21
    addendum, MINDlarge Kaggle verification): `device` used to default to
    `"mps"` unconditionally — correct for this project's own Mac development
    machine, but `"mps"` (Apple's GPU backend) doesn't exist at all on
    Kaggle's Linux runners, where it raised `RuntimeError: PyTorch is not
    linked with support for mps devices` on first real use. Preference
    order: CUDA (Kaggle GPU runtime) > MPS (this project's Mac) > CPU
    (always available, the safe universal fallback)."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_encoder(model_name: str = DEFAULT_MODEL, device: str | None = None):
    """Isolates the only `sentence_transformers` import in this project,
    mirroring how `index.py` isolates `rank_bm25`. `device=None` (default)
    auto-detects the best available backend — see `_default_device`; pass
    an explicit value to override."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device or _default_device())


def _encode(encoder, texts: list[str], batch_size: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, encoder.get_sentence_embedding_dimension()), dtype=np.float32)
    return encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def build_embedding_index(
    articles: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    cache_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    encoder=None,
) -> EmbeddingIndex:
    """Build (or load-from-cache) an `EmbeddingIndex` over `title + " " +
    abstract` for every article, per ADR-002/ADR-005's mandatory text
    fields.

    `encoder` may be passed directly (used by tests to inject a stub
    encoder and skip a real model load); otherwise `load_encoder(model_name,
    device)` is used.
    """
    article_ids = articles["article_id"].tolist()

    if cache_path is not None:
        cached = _load_cache(cache_path, model_name, article_ids)
        if cached is not None:
            return cached

    text = (articles["title"].fillna("") + " " + articles["abstract"].fillna("")).tolist()
    if encoder is None:
        encoder = load_encoder(model_name, device)
    vectors = _encode(encoder, text, batch_size)

    index = EmbeddingIndex(
        article_ids=article_ids,
        id_to_col={aid: i for i, aid in enumerate(article_ids)},
        vectors=vectors,
    )

    if cache_path is not None:
        _write_cache(cache_path, model_name, index)

    return index


def _load_cache(cache_path: Path, model_name: str, article_ids: list[str]) -> EmbeddingIndex | None:
    sidecar = cache_path.with_suffix(".json")
    if not (cache_path.exists() and sidecar.exists()):
        return None
    meta = json.loads(sidecar.read_text())
    if meta.get("model") != model_name or meta.get("article_ids") != article_ids:
        return None  # stale cache (different model or article set) — recompute, never silently reuse
    vectors = np.load(cache_path)
    return EmbeddingIndex(
        article_ids=article_ids,
        id_to_col={aid: i for i, aid in enumerate(article_ids)},
        vectors=vectors,
    )


def _write_cache(cache_path: Path, model_name: str, index: EmbeddingIndex) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, index.vectors)
    cache_path.with_suffix(".json").write_text(
        json.dumps({"model": model_name, "article_ids": index.article_ids})
    )


def build_user_embedding_query(
    article_ids: list[str], vector_lookup: dict[str, np.ndarray]
) -> np.ndarray | None:
    """`vector_lookup` maps article_id -> its L2-normalized embedding vector.

    Mean-pools a user's history article vectors, then re-normalizes (per
    Q3's "mean-pooled embeddings" spec) so the result is comparable to the
    index's own L2-normalized rows via a plain dot product (cosine
    similarity). Unknown article_ids (not in `vector_lookup`) are skipped,
    same treatment as `build_user_query`'s missing-text-lookup case.

    Returns `None` for an empty/fully-unresolvable history — a true
    cold-start user, per ADR-005/ADR-008 — never a NaN-filled vector.
    """
    vecs = [vector_lookup[a] for a in article_ids if a in vector_lookup]
    if not vecs:
        return None
    mean = np.mean(vecs, axis=0)
    norm = np.linalg.norm(mean)
    if norm == 0:
        return None
    return (mean / norm).astype(np.float32)


def build_user_embedding_query_recency(
    article_ids: list[str], vector_lookup: dict[str, np.ndarray], decay: float = 0.9
) -> np.ndarray | None:
    """Candidate C (2026-08-21 local-validation session): same contract as
    `build_user_embedding_query`, but weights each resolvable history
    vector by exponential decay based on its position in `article_ids`,
    assuming (per MIND's documented convention, but — per ADR-005 — never
    independently verified against real timestamps) that the list is
    ordered oldest-to-most-recent, i.e. `article_ids[-1]` is the most
    recent click. Weight for the i-th *resolvable* entry (0-indexed from
    the end) is `decay ** i`, so the most recent resolvable click gets
    weight 1.0 and older ones decay geometrically — deliberately computed
    over the filtered (resolvable) sequence, not the raw history, so a
    handful of unresolvable ids interspersed in the middle of a history
    don't shift surrounding weights. `decay=0.9` is a single untuned
    starting point (not searched), same "test whether it helps at all"
    framing `run_hybrid_experiment.py`'s untuned 50/50 blend uses for
    Candidate B — a real signal here would justify tuning it later.

    ADR-005 rejected recency weighting for BM25 query construction
    specifically because MIND's history order is unverified — this
    function doesn't resolve that risk, it deliberately re-tests it (this
    is Candidate C's whole point), so any real result from this function
    inherits that same unverified-ordering caveat and must be reported
    with it, not silently.
    """
    resolvable = [vector_lookup[a] for a in article_ids if a in vector_lookup]
    if not resolvable:
        return None
    n = len(resolvable)
    weights = np.array([decay ** (n - 1 - i) for i in range(n)], dtype=np.float64)
    mean = np.average(np.stack(resolvable), axis=0, weights=weights)
    norm = np.linalg.norm(mean)
    if norm == 0:
        return None
    return (mean / norm).astype(np.float32)
''')

for mod in list(sys.modules):
    if mod.startswith("src.") or mod in (
        "run_gated_cohort_experiment", "run_learned_combiner_experiment",
        "run_bm25_experiment", "run_ranking_eval",
    ):
        del sys.modules[mod]

sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "scripts"))
print("repo ready at", REPO_DIR, "-", os.listdir(REPO_DIR))


# %% CELL 2 — download MINDlarge's raw train+dev zips (via this project's
# own `download_mind_bundle`, same Hugging Face mirror URL this project's
# `src/pipeline/download.py` already uses — no new download logic
# introduced here) and build the processed bundle via the project's own,
# already-fixed `build_mind_split` (the naive-loop-hang and memory-scaling
# bugs at this exact row count were fixed in an earlier session — see
# `src/datasets/mind.py`'s `_explode_impressions` docstring — reused
# unmodified here, not re-derived).
from pathlib import Path
from src.pipeline.download import download_mind_bundle
from src.pipeline.orchestrator import build_mind_split

download_mind_bundle("MINDlarge_train.zip")
download_mind_bundle("MINDlarge_dev.zip")

mind_raw = Path(REPO_DIR) / "data" / "raw" / "mind"
mind_out = Path(REPO_DIR) / "data" / "processed" / "mind" / "large"
build_mind_split(mind_raw / "MINDlarge_train.zip", "train", mind_out)
build_mind_split(mind_raw / "MINDlarge_dev.zip", "dev", mind_out)
print("Built:", sorted(str(p) for p in mind_out.rglob("*.parquet")))


# %% CELL 3 — run Candidate G. Reuses Candidate F's MINDsmall-fitted
# coefficients exactly (no retraining) — this run answers "does the SAME
# fitted model, evaluated at MINDlarge's real scale and real warm/cold
# cohort mix, still show the paired win it showed on MINDsmall-dev," per
# ADR-010's condition.
from run_gated_cohort_experiment import run

config, results = run(bundle="large")
print(__import__("json").dumps(config, indent=2))


# %% CELL 4 — print the comparison plainly, same format as every other
# candidate's local run.
for metric, slices in results["metrics"].items():
    print(f"\n{metric}")
    for name, r in slices.items():
        print(f"  {name}: {r['metric']:.4f} (95% CI: {r['ci_low']:.4f}-{r['ci_high']:.4f}), "
              f"n_impressions={r['n_impressions']}, n_users={r['n_users']}, n_skipped={r['n_skipped']}")

p = results["paired_vs_baseline"]
print(f"\npaired bootstrap (gated - baseline, overall AUC): {p['point_diff']:+.4f} "
      f"(95% CI: {p['ci_low']:+.4f} to {p['ci_high']:+.4f}), "
      f"CI excludes zero: {p['ci_excludes_zero']}")


# %% CELL 5 — SUMMARY (paste this whole cell's output back)
import json

print("=" * 80)
print("SUMMARY (paste this whole block back)")
print("=" * 80)
print(json.dumps(config, indent=2))
print(json.dumps(results, indent=2))


# %% CELL 6 — package the two small result JSON files for download (NOT
# the ~1.3GB processed MINDlarge bundle itself — reproducible from the raw
# zips + this project's own unmodified pipeline code, no need to move it).
out_dir = Path("/kaggle/working") / f"candidate_g_gated_cohort_mind_large_{config['date']}"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "config.json").write_text(json.dumps(config, indent=2))
(out_dir / "results.json").write_text(json.dumps(results, indent=2))

shutil.make_archive("/kaggle/working/candidate_g_mindlarge_results", "zip", out_dir)
print(
    "Download /kaggle/working/candidate_g_mindlarge_results.zip and place its contents at "
    f"experiments/{out_dir.name}/ locally."
)
