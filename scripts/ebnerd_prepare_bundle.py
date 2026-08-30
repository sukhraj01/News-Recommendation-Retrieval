#!/usr/bin/env python3
"""Fetch an EB-NeRD bundle and precompute its article embeddings (ADR-013).

Split out from `run_ebnerd_gbdt_experiment.py` because on Ada these two steps
have completely different resource profiles from training: the download is
network-bound and needs resumable retry (Ada's compute nodes have a history of
throttled/dropped transfers — see ADR-012's saga), and the encoding is
GPU-bound. Doing them in the training job would mean re-downloading and
re-encoding on every requeue.

Both steps are idempotent: an existing complete zip is not re-fetched, and
`build_embedding_index`'s own on-disk cache (keyed by model name and article-id
order) short-circuits the encode.

The EB-NeRD bundles are served unauthenticated from S3, so unlike MIND
(consent-gated, see `src/pipeline/download.py`) this can be scripted end to end:

    https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip     2.97 GB
    https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip   1.52 GB

Usage:
    python scripts/ebnerd_prepare_bundle.py --bundle large \
        --raw-dir /ssd_scratch/$USER/ebnerd_raw \
        --embeddings-dir /ssd_scratch/$USER/ebnerd_embeddings
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from src.retrieval.embed import DEFAULT_MODEL, build_embedding_index, model_slug

S3_BASE = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com"
BUNDLES = {
    "demo": "ebnerd_demo.zip",
    "small": "ebnerd_small.zip",
    "large": "ebnerd_large.zip",
    "testset": "ebnerd_testset.zip",
}
# ebnerd_testset ships behaviors/history but no articles table of its own; the
# catalog for it is published separately.
ARTICLES_ONLY = "articles_large_only.zip"


def download_resumable(url: str, dest: Path, max_attempts: int = 12) -> Path:
    """Download `url` to `dest`, resuming a partial transfer across attempts.

    Same shape as `src/pipeline/download.py::_download_with_retry` and for the
    same reason: this project has already lost multi-hour Ada downloads to a
    throttled connection dropping mid-stream, and a non-resumable retry restarts
    from zero. Writes to `<dest>.part` and renames only on completion, so an
    interrupted run never leaves a truncated file that later looks complete.
    """
    import urllib.error
    import urllib.request

    if dest.exists():
        print(f"  {dest.name}: already present ({dest.stat().st_size/2**30:.2f} GB), skipping")
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        request = urllib.request.Request(url)
        if have:
            request.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                total = response.headers.get("Content-Length")
                total = (int(total) + have) if total else None
                mode = "ab" if have and response.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                with open(part, mode) as fh:
                    t0, last = time.time(), have
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        have += len(chunk)
                        if time.time() - t0 > 30:
                            rate = (have - last) / (time.time() - t0) / 2**20
                            pct = f" ({100*have/total:.1f}%)" if total else ""
                            print(f"    {have/2**30:.2f} GB{pct} @ {rate:.1f} MB/s", flush=True)
                            t0, last = time.time(), have
            if total is None or have >= total:
                part.rename(dest)
                print(f"  {dest.name}: done ({have/2**30:.2f} GB)")
                return dest
            print(f"  short read ({have}/{total}), retrying")
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            wait = min(60, 2**attempt)
            print(f"  attempt {attempt}/{max_attempts} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"failed to download {url} after {max_attempts} attempts")


def read_articles_frame(zip_path: Path) -> pd.DataFrame:
    """Read the raw articles table into the shape `build_embedding_index` wants.

    Two adaptations: raw EB-NeRD names the short-text field `subtitle` where
    this project's unified schema (ADR-002) calls it `abstract`, and article ids
    are bare integers where the embedding cache is keyed by this project's
    prefixed string ids. Both mappings must match
    `ebnerd_features._load_aligned_embeddings` exactly or the vectors silently
    align to the wrong articles.
    """
    with zipfile.ZipFile(zip_path) as zf:
        name = next(
            n for n in zf.namelist()
            if n.endswith("articles.parquet") and "__MACOSX" not in n
        )
        art = pd.read_parquet(io.BytesIO(zf.read(name)), columns=["article_id", "title", "subtitle"])
    return pd.DataFrame(
        {
            "article_id": ["ebnerd:" + str(int(a)) for a in art["article_id"]],
            "title": art["title"].fillna(""),
            "abstract": art["subtitle"].fillna(""),
        }
    )


def report_scale(zip_path: Path) -> None:
    """Print real per-split counts and a memory projection before anything big runs.

    CLAUDE.md's Memory Estimation rule: project peak memory from measured
    per-unit numbers before touching a new data scale, rather than discovering
    the ceiling by hitting it. The per-row figure is measured, not guessed —
    240 bytes/row for the 60-feature float32 matrix, from the ebnerd_small run.
    """
    from src.retrieval.ebnerd_features import FEATURE_NAMES

    bytes_per_row = 4 * len(FEATURE_NAMES)
    print(f"\n  scale report for {zip_path.name}:")
    with zipfile.ZipFile(zip_path) as zf:
        splits = sorted({n.split("/")[0] for n in zf.namelist()
                         if "/" in n and "__MACOSX" not in n})
        for split in splits:
            member = f"{split}/behaviors.parquet"
            if member not in zf.namelist():
                continue
            beh = pd.read_parquet(io.BytesIO(zf.read(member)), columns=["article_ids_inview"])
            n_imp = len(beh)
            n_rows = int(sum(len(v) for v in beh["article_ids_inview"]))
            gb = n_rows * bytes_per_row / 2**30
            print(
                f"    {split:<12} {n_imp:>12,} impressions  {n_rows:>14,} candidate rows"
                f"  ~{gb:6.1f} GB feature matrix ({n_rows/n_imp:.1f} cands/imp)"
            )
            del beh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, choices=sorted(BUNDLES))
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--embeddings-dir", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None, help="cuda / cpu / mps; auto-detected if omitted")
    ap.add_argument("--skip-embeddings", action="store_true")
    ap.add_argument("--scale-report", action="store_true", help="print split sizes and exit")
    args = ap.parse_args()

    raw_dir, emb_dir = Path(args.raw_dir), Path(args.embeddings_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)

    filename = BUNDLES[args.bundle]
    zip_path = raw_dir / filename
    print(f"[1/3] fetching {filename}")
    download_resumable(f"{S3_BASE}/{filename}", zip_path)

    # The test set has no articles table of its own.
    articles_zip = zip_path
    if args.bundle == "testset":
        articles_zip = raw_dir / ARTICLES_ONLY
        print(f"[1b/3] fetching {ARTICLES_ONLY} (testset ships no article catalog)")
        download_resumable(f"{S3_BASE}/{ARTICLES_ONLY}", articles_zip)

    print("[2/3] scale report")
    report_scale(zip_path)
    if args.scale_report:
        return

    if args.skip_embeddings:
        print("[3/3] embeddings skipped by request")
        return

    print(f"[3/3] encoding articles with {args.model}")
    frame = read_articles_frame(articles_zip)
    cache = emb_dir / f"{model_slug(args.model)}.npy"
    t0 = time.time()
    index = build_embedding_index(
        frame, model_name=args.model, cache_path=cache,
        batch_size=args.batch_size, device=args.device,
    )
    print(
        f"  {len(index.article_ids):,} articles -> {index.vectors.shape} "
        f"in {time.time()-t0:.0f}s; cached at {cache}"
    )
    norms = np.linalg.norm(index.vectors, axis=1)
    print(f"  L2 norms: min {norms.min():.4f} max {norms.max():.4f} (expect ~1.0)")


if __name__ == "__main__":
    main()
