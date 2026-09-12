"""Build EB-NeRD's official-NRMS article inputs locally (ADR-015, Option B).

Runs under the official `transformers==4.36.2` pin, which Ada's torch env
does not have (it has 5.16.1), so tokenizer ids never depend on an
unverified major-version upgrade. It reproduces ebnerd-benchmark's
reproducibility script exactly:

    concat_str_columns([title, subtitle, body], sep=" ")
    tokenizer(text, add_special_tokens=False, padding="max_length",
              max_length=30, truncation=True)
    word embeddings = model.embeddings.word_embeddings.weight  (xlm-roberta-large)

Two exact transformations shrink what gets shipped to Ada:
  - Vocabulary trimming. Only token ids that actually occur (plus id 0,
    which the official "zeros" unknown-article representation uses) keep
    an embedding row, re-indexed densely. A row that is never looked up
    gets exactly zero gradient, and Adam leaves it unchanged, so every
    output is identical to using the full 250,002-row matrix.
  - The embedding matrix is read straight from `model.safetensors` (no
    torch needed), using the same tensor `AutoModel` exposes.

Output (.npz): article_ids, tokens (n_articles+1, max_len) with the last
row = the unknown-article "zeros" representation, embedding
(n_used, 1024) float32, used_token_ids, published_time_s.

Usage: python scripts/a2_ebnerd_prepare_tokens.py --zip data/raw/ebnerd/ebnerd_small.zip \
           --model-dir <xlm-roberta-large dir> --out <file.npz>
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True)
ap.add_argument("--model-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-length", type=int, default=30)  # args_nrms.py max_title_length
a = ap.parse_args()

import transformers  # noqa: E402

assert transformers.__version__ == "4.36.2", f"official pin is 4.36.2, got {transformers.__version__}"

z = zipfile.ZipFile(a.zip)
arts = pq.read_table(io.BytesIO(z.read("articles.parquet")),
                     columns=["article_id", "title", "subtitle", "body", "published_time"]).to_pandas()
# pl.concat_str(..., separator=" ") -- no nulls in any of the three columns
# (checked 2026-09-11: 0 nulls; empty strings are kept, as polars does).
for c in ("title", "subtitle", "body"):
    assert arts[c].notna().all(), f"{c} has nulls; polars concat_str would null the whole row"
text = (arts["title"] + " " + arts["subtitle"] + " " + arts["body"]).tolist()

tok = AutoTokenizer.from_pretrained(a.model_dir)
enc = tok(text, add_special_tokens=False, padding="max_length",
          max_length=a.max_length, truncation=True)["input_ids"]
enc = np.asarray(enc, dtype=np.int64)
assert enc.shape == (len(arts), a.max_length)

# Official unknown-article representation ("zeros"): token id 0 repeated.
unknown = np.zeros((1, a.max_length), dtype=np.int64)
all_tokens = np.vstack([enc, unknown])
used = np.unique(all_tokens)
remap = {int(t): i for i, t in enumerate(used)}
tokens = np.vectorize(remap.__getitem__)(all_tokens).astype(np.int32)

st_path = Path(a.model_dir) / "model.safetensors"
with safe_open(str(st_path), framework="np") as f:
    key = [k for k in f.keys() if k.endswith("embeddings.word_embeddings.weight")]
    assert len(key) == 1, f"expected one word-embedding tensor, found {key}"
    full = f.get_tensor(key[0])
emb = np.ascontiguousarray(full[used].astype(np.float32))

# Unit-agnostic conversion. The first version did astype(int64)/1e9, but
# pandas keeps timestamp[us] as datetime64[us], so every publish time came
# out 1000x too small and the freshness treatment became a silent no-op
# (ADR-015, 2026-09-12).
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrieval.nrms_official_data import to_epoch_seconds  # noqa: E402

pub_s = to_epoch_seconds(arts["published_time"])
# The bound only has to catch a units error: 1000x too small lands in
# 1970, near 0. It does not police real dates. The catalog genuinely
# contains archive articles from 1998 (first try with a year-2000 bound
# fired on real data), hence 1990.
lo, hi = 631_152_000.0, 1_893_456_000.0  # 1990-01-01 .. 2030-01-01
assert ((pub_s > lo) & (pub_s < hi)).all(), \
    f"implausible publish times: min {pub_s.min():.3e} max {pub_s.max():.3e} (expected {lo:.2e}..{hi:.2e})"
print(f"publish times OK: {int((pub_s < 946_684_800).sum())} of {len(pub_s)} articles predate 2000")
np.savez_compressed(
    a.out, article_ids=arts["article_id"].to_numpy(np.int64), tokens=tokens,
    embedding=emb, used_token_ids=used.astype(np.int64), published_time_s=pub_s,
)
meta = {
    "n_articles": int(len(arts)), "max_length": a.max_length,
    "full_vocab": int(full.shape[0]), "embed_dim": int(full.shape[1]),
    "used_tokens": int(len(used)), "embedding_key": key[0],
    "pad_token_id": tok.pad_token_id, "transformers": transformers.__version__,
    "unknown_row_index": int(len(arts)),
}
Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))
