"""EB-NeRD Codabench submission — Part 2, the real test-set run, FOR KAGGLE.

Paste each `# %% CELL` block into its own cell in a Kaggle notebook and run
top to bottom. Purpose: generate `predictions.txt` for the real, blind
`ebnerd_testset.zip` (13,536,710 impressions — 13,336,710 regular +
200,000 `is_beyond_accuracy`, per Part 0/1's confirmed real numbers) and
package it for upload to https://www.codabench.org/competitions/2469/.

Before running:

1. **Turn the GPU accelerator on** (Notebook Settings -> Accelerator ->
   GPU T4 x2 or P100). Unlike Part 0's investigation, this session does
   real work that benefits from it: encoding all 125,541 articles in
   `articles_large_only.zip` through the sentence-transformer. Everything
   downstream of the encode step (the actual 13.5M-impression scoring
   loop) is plain numpy/CPU work regardless of accelerator — the GPU only
   buys you the one-time corpus encode.
2. Upload `notebooks/ebnerd_part2_src_bundle.zip` as a new private Kaggle
   Dataset (Kaggle -> Datasets -> New Dataset -> upload the zip; Kaggle
   auto-extracts zips uploaded as datasets, so the dataset will contain a
   `src/` directory directly under `/kaggle/input/<your-dataset-slug>/`).
   Add it as a Data source on this notebook. This is this project's own
   validated `src/submission/ebnerd_format.py` (Part 1, cross-checked
   against `evaluation/official/evaluate.py` on `ebnerd_small`) plus its
   dependencies — reusing it byte-for-byte avoids re-deriving/redrifting
   the scoring and format logic on Kaggle. Frozen from commit `fa85e48`
   plus this session's streaming fix (see Cell 1's note).
3. Cell 0 below `wget`s `ebnerd_testset.zip` and `articles_large_only.zip`
   directly into `/kaggle/working/` (same URLs Part 0 used) — no need to
   attach them as a Data source first, though that also works if you
   already have them there.
4. Run cells in order. **Cell 6 is a benchmark-then-decide gate, not an
   automatic green light** — per this project's own discipline (every
   large-scale run so far has been measured and projected before being
   committed to), it prints a real projected full-run time from a real
   sample measured on this exact Kaggle instance, and the actual full run
   (Cell 7) stays behind a `RUN_FULL_JOB = True` flag you set only after
   reading that projection. Do not flip it blind.
5. Copy the printed Cell 6 projection (and Cell 3's discovery output) back
   to the engineer — same relay pattern as Part 0.
"""

# %% CELL 0 — download the two source files directly into the notebook.
# Skip if already attached under /kaggle/input via Data sources.
# ~1.6GB total (1.5GB testset + ~140MB articles), well within a Kaggle
# session's disk quota.
# !wget -q https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip -O /kaggle/working/ebnerd_testset.zip
# !wget -q https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/articles_large_only.zip -O /kaggle/working/articles_large_only.zip
# !ls -lh /kaggle/working/


# %% CELL 1 — discover inputs (test zip, articles zip, this project's src/
# bundle) under both /kaggle/input and /kaggle/working, and put the src/
# bundle on sys.path. Also installs rank_bm25, which Kaggle's base image
# does not ship (sentence-transformers/torch/pandas/pyarrow/scipy/sklearn
# already do).
import os
import sys
import zipfile

TARGETS = ["ebnerd_testset.zip", "articles_large_only.zip"]

found = {}
src_bundle_root = None
for search_root in ["/kaggle/input", "/kaggle/working"]:
    for root, _dirs, files in os.walk(search_root):
        for f in files:
            if f in TARGETS and f not in found:
                found[f] = os.path.join(root, f)
        if os.path.basename(root) == "src" and "submission" in _dirs:
            src_bundle_root = os.path.dirname(root)  # parent of src/

for t in TARGETS:
    print(f"{t}: {found.get(t, 'NOT FOUND')}")
print(f"src/ bundle parent dir: {src_bundle_root or 'NOT FOUND'}")

missing = [t for t in TARGETS if t not in found]
if missing or src_bundle_root is None:
    raise FileNotFoundError(
        f"Missing: {missing if missing else []}"
        f"{' + src/ bundle' if src_bundle_root is None else ''}. "
        "Run Cell 0 for the zips, and attach the uploaded src-bundle "
        "Dataset (see this script's header) for src/."
    )

sys.path.insert(0, src_bundle_root)

get_ipython().system("pip install -q rank_bm25")

import torch
print(f"\ncuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
else:
    print(
        "WARNING: no GPU detected. Notebook Settings -> Accelerator -> "
        "GPU. The encode step (Cell 4) will be much slower on CPU, and "
        "this session's whole point (per the task brief) was to use it."
    )

from src.datasets.ebnerd import _parse_history, parse_ebnerd_articles
from src.retrieval.embed import DEFAULT_MODEL, build_embedding_index, build_user_embedding_query, model_slug
from src.retrieval.index import build_index
from src.retrieval.query import build_user_query
from src.retrieval.score import BM25Scorer, EmbeddingScorer
from src.submission.ebnerd_format import iter_raw_impressions, ranks_for_impression
from src.utils.io import read_zip_parquet

TESTSET_ZIP = found["ebnerd_testset.zip"]
ARTICLES_ZIP = found["articles_large_only.zip"]
print("\nimports OK")


# %% CELL 2 — memory/CPU headroom check. Cheap, informational — per
# CLAUDE.md's Memory Estimation clause, know the ceiling before projecting
# against it in Cell 6.
import psutil

vm = psutil.virtual_memory()
print(f"total RAM: {vm.total / 1024**3:.1f} GB")
print(f"available RAM right now: {vm.available / 1024**3:.1f} GB")
print(f"CPU count: {os.cpu_count()}")


# %% CELL 3 — DISCOVERY. Confirm the real facts this run depends on
# directly against the actual files, rather than trusting Part 0/1's
# relayed numbers unchanged (those were real measurements too, but on
# whatever exact file was live then — cheap to re-confirm, expensive to
# be wrong about downstream). In particular: does ebnerd_testset.zip ship
# `test/history.parquet`? Part 0's investigation never explicitly checked
# this (it only inspected `test/behaviors.parquet`), and query construction
# for every method in this project depends on per-user history existing.
print("=" * 80)
print("ebnerd_testset.zip — full member listing")
print("=" * 80)
test_zip = zipfile.ZipFile(TESTSET_ZIP)
test_names = [n for n in test_zip.namelist() if not n.endswith("/")]
for n in test_names:
    print(f"  {n}  ({test_zip.getinfo(n).file_size} bytes)")

has_history = any(n.endswith("test/history.parquet") for n in test_names)
print(f"\ntest/history.parquet present: {has_history}")
if not has_history:
    raise RuntimeError(
        "STOP — test/history.parquet not found in ebnerd_testset.zip. "
        "Query construction (BM25 and embeddings alike) needs per-user "
        "history; this is a real blocker, not something to route around "
        "silently (CLAUDE.md's Resource Availability clause). Relay this "
        "finding back before writing any workaround."
    )

# NOTE (found running this cell for real): the real ebnerd_testset.zip
# wraps every member in an extra top-level directory
# (ebnerd_testset/test/behaviors.parquet, not test/behaviors.parquet) —
# ebnerd_small.zip/ebnerd_demo.zip don't do this, which is why it wasn't
# caught until Kaggle. read_zip_parquet (used here instead of a raw
# zipfile.open) now has a suffix-match fallback for exactly this, so the
# plain "test/behaviors.parquet" member name below still resolves
# correctly against either packaging convention.
behaviors_meta = read_zip_parquet(
    TESTSET_ZIP, "test/behaviors.parquet",
    columns=["impression_id", "user_id", "is_beyond_accuracy"],
)
n_test_impressions = len(behaviors_meta)
print(f"\ntest impressions (real count): {n_test_impressions}")
EXPECTED_TOTAL = 13_536_710
if n_test_impressions != EXPECTED_TOTAL:
    print(
        f"WARNING: real count ({n_test_impressions}) != Part 1's relayed "
        f"figure ({EXPECTED_TOTAL}). Using the real measured count for "
        "every downstream check in this notebook, not the old number."
    )
    EXPECTED_TOTAL = n_test_impressions

if "is_beyond_accuracy" in behaviors_meta.columns:
    print("\nis_beyond_accuracy value_counts:")
    print(behaviors_meta["is_beyond_accuracy"].value_counts(dropna=False))

n_test_users = behaviors_meta["user_id"].nunique()
print(f"\nunique users referenced in test behaviors: {n_test_users}")
del behaviors_meta

print("\n" + "=" * 80)
print("articles_large_only.zip — real article count")
print("=" * 80)
articles = parse_ebnerd_articles(ARTICLES_ZIP)
print(f"articles: {len(articles)}")
EXPECTED_ARTICLES = 125_541
if len(articles) != EXPECTED_ARTICLES:
    print(
        f"NOTE: real count ({len(articles)}) differs from the task "
        f"brief's stated figure ({EXPECTED_ARTICLES}) — using the real "
        "count, not the stated one, for everything below."
    )

history = _parse_history(TESTSET_ZIP, "test")
print(f"\ntest/history.parquet rows (users with history): {len(history)}")
hist_lens = history["article_ids"].apply(len)
print("history length stats (articles/user):")
print(hist_lens.describe())


# %% CELL 4 — BUILD THE CORPUS INDEXES over the full articles_large_only
# corpus (needed regardless of impression count: both the BM25/embedding
# candidate-scoring side AND the history-query side draw from this same
# corpus). This is the actual GPU-beneficial step — it's also a real,
# not-sampled measurement (there's no "small" version of encoding the
# whole corpus, you need every article's vector anyway), so it doubles as
# hard evidence for Cell 6's projection rather than something benchmarked
# separately.
import time
from pathlib import Path

t0 = time.time()
bm25_index = build_index(articles)
bm25_build_s = time.time() - t0
bm25_mb = bm25_index.weights_t.data.nbytes / 1024**2
print(f"BM25 index: {bm25_build_s:.1f}s, weights_t {bm25_mb:.1f} MB "
      f"({bm25_index.weights_t.nnz} nonzeros)")

# build_embedding_index's disk cache (_load_cache/_write_cache) calls
# cache_path.with_suffix(...), which only exists on Path — a plain string
# here raises AttributeError (hit for real on Kaggle: str has no attribute
# with_suffix). Must be a Path, not str + str concatenation.
embed_cache_path = Path("/kaggle/working/embeddings_cache") / (model_slug(DEFAULT_MODEL) + ".npy")
device = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
embed_index = build_embedding_index(
    articles, model_name=DEFAULT_MODEL, cache_path=embed_cache_path,
    device=device,
)
embed_build_s = time.time() - t0
embed_mb = embed_index.vectors.nbytes / 1024**2
print(f"Embedding index ({device}): {embed_build_s:.1f}s "
      f"({len(articles) / max(embed_build_s, 1e-9):.1f} articles/s), "
      f"vectors {embed_mb:.1f} MB")


# %% CELL 5 — CONFIG + build per-user queries, gated by RUN_METHODS.
#
# Real evidence found this session (measured against real EB-NeRD text and
# the real 807,677-user/144.6-avg-history-length distribution Cell 3
# reported): BM25's query dict concatenates a user's ENTIRE history as an
# unweighted token multiset (ADR-005 — no dedup, repeats are informative),
# so its per-user cost scales with history length. Measured ~15.9KB/user
# -> projects to ~12.25GB for all 807,677 users. Embeddings mean-pool to
# one fixed 384-dim vector/user regardless of history length -> ~1.16GB.
# That's a real ~10.6x gap, not a rounding difference, and it lands on top
# of whatever's already resident (Cell 2 showed available RAM dropping
# from 29.6GB to 15.5GB just from re-running earlier cells in this same
# kernel — if you're seeing something similar, restart the kernel for a
# clean baseline before trusting any of these numbers).
#
# Combined with embeddings already winning on ebnerd_small-validation
# accuracy (AUC 0.5430 vs. BM25's 0.5288, Part 1), BM25 is now opt-in
# here, not built by default — this updates this session's earlier "build
# both, decide at Cell 6" plan with new evidence, per CLAUDE.md's Decision
# Reversal principle. Add "bm25" below only after confirming real headroom
# (check the RAM print at the end of this cell) — everything from here
# through Cell 9 (query build, Cell 6's benchmark, Cell 7's full run) is
# driven by this one list.
RUN_METHODS = ["embed"]  # add "bm25" too only if you've confirmed ~12GB headroom

query_by_user = {}
scorers = {}
empty_queries = {}

if "bm25" in RUN_METHODS:
    text_lookup = dict(zip(
        articles["article_id"],
        articles["title"].fillna("") + " " + articles["abstract"].fillna(""),
    ))
    t0 = time.time()
    query_by_user["bm25"] = {
        row.user_id: build_user_query(row.article_ids, text_lookup)
        for row in history.itertuples(index=False)
    }
    print(f"BM25 query_by_user: {len(query_by_user['bm25'])} users, "
          f"{time.time() - t0:.1f}s")
    scorers["bm25"] = BM25Scorer(bm25_index)
    empty_queries["bm25"] = []
    vm = psutil.virtual_memory()
    print(f"  RAM available after BM25 query build: {vm.available / 1024**3:.1f} GB")

if "embed" in RUN_METHODS:
    vector_lookup = dict(zip(embed_index.article_ids, embed_index.vectors))
    t0 = time.time()
    query_by_user["embed"] = {
        row.user_id: build_user_embedding_query(row.article_ids, vector_lookup)
        for row in history.itertuples(index=False)
    }
    print(f"Embedding query_by_user: {len(query_by_user['embed'])} users, "
          f"{time.time() - t0:.1f}s")
    scorers["embed"] = EmbeddingScorer(embed_index)
    empty_queries["embed"] = None
    vm = psutil.virtual_memory()
    print(f"  RAM available after embedding query build: {vm.available / 1024**3:.1f} GB")

methods = {name: (scorers[name], query_by_user[name], empty_queries[name]) for name in RUN_METHODS}


# %% CELL 6 — BENCHMARK on a real sample of THIS file, on THIS Kaggle
# instance, before committing to the full 13.5M-impression job. Per
# CLAUDE.md's Benchmarking Philosophy: measure with context, project, then
# decide — not a formality, this notebook doesn't know Kaggle's actual
# per-impression throughput until it's measured here (the project's own
# prior benchmark, 2.93ms/impression, was measured on a different machine
# entirely — a real but non-transferable number).
#
# Deterministic systematic sample (every STRIDE-th impression) rather than
# "first N" — a prefix could all land in one narrow user/pool cluster
# (e.g. entirely inside the 200,000-row is_beyond_accuracy block) and bias
# the throughput estimate. This also exercises iter_raw_impressions'
# streaming path end-to-end on the real file before the full run does.
SAMPLE_SIZE = 100_000
STRIDE = max(1, EXPECTED_TOTAL // SAMPLE_SIZE)

def sample_rows(has_labels=False):
    for i, row in enumerate(iter_raw_impressions(TESTSET_ZIP, "test", has_labels=False)):
        if i % STRIDE == 0:
            yield row

# methods (name -> (scorer, query_by_user_dict, empty_query)) is built in
# Cell 5, gated by RUN_METHODS — not redefined here, so this cell only
# ever benchmarks whatever was actually built (avoids referencing a query
# dict that was deliberately skipped for memory reasons).
projections = {}
for name, (scorer, q_by_user, empty_query) in methods.items():
    t0 = time.time()
    n = 0
    for row in sample_rows():
        query = q_by_user.get(row["user_id"], empty_query)
        scores = scorer.score(query, row["article_ids"])
        ranks_for_impression(scores, row["raw_impression_id"], seed=0)
        n += 1
    elapsed = time.time() - t0
    ms_per_impression = 1000 * elapsed / n
    projected_s = ms_per_impression / 1000 * EXPECTED_TOTAL
    projections[name] = projected_s
    print(f"{name}: {n} sampled impressions in {elapsed:.1f}s "
          f"({ms_per_impression:.3f} ms/impression) -> "
          f"projected full run: {projected_s / 3600:.2f} hours "
          f"({projected_s / 60:.0f} min)")

vm = psutil.virtual_memory()
print(f"\nRAM available after benchmark: {vm.available / 1024**3:.1f} GB")

print("\n" + "=" * 80)
print("DECISION GATE — read before touching Cell 7")
print("=" * 80)
print(
    "Kaggle GPU sessions are commonly capped around 9-12h per run and "
    "~30h/week of GPU quota total (check your own account's actual "
    "limit — this notebook can't see your quota). Compare the projected "
    "hours above against that budget for whichever method(s) are in "
    "RUN_METHODS (set in Cell 5).\n"
    "If a projection is comfortably under budget: proceed to Cell 7.\n"
    "If a projection is borderline or over budget, real alternatives "
    "(per CLAUDE.md's Resource Availability clause — pick one "
    "deliberately, don't silently downgrade):\n"
    "  (a) run embeddings only, not both (Cell 5's current default) — "
    "Part 1 already showed embeddings winning on ebnerd_small-validation "
    "(AUC 0.5430 vs. BM25's 0.5288), and this session found BM25's query "
    "construction alone costs ~10.6x more RAM (~12.25GB vs. ~1.16GB "
    "projected for the real 807,677-user test set), so it's the "
    "defensible single choice on both accuracy and resource grounds;\n"
    "  (b) split the run across multiple Kaggle sessions using an "
    "impression-index checkpoint/resume (would need a small code change "
    "to this notebook's Cell 7 loop — not built here since it's only "
    "needed if the real projection actually demands it);\n"
    "  (c) use Kaggle's 'Save & Run All (Commit)' to run in the "
    "background beyond an interactive session's own timeout, if your "
    "account's session/quota limits allow it."
)


# %% CELL 7 — THE FULL RUN. Gated behind RUN_FULL_JOB — set it to True
# only after reading Cell 6's real projection. Uses RUN_METHODS as set in
# Cell 5 (not redefined here — redefining it in this cell would silently
# discard a deliberate choice made back in Cell 5, e.g. adding "bm25").
# Writes directly to /kaggle/working/ (streamed, one line at a time — the
# whole point of this session's src/ fix), with periodic progress printed
# so a long run is inspectable rather than a black box.
RUN_FULL_JOB = False  # <-- set True deliberately, after reading Cell 6

if not RUN_FULL_JOB:
    print("RUN_FULL_JOB is False — not running. Set it True after "
          "reviewing Cell 6's projection.")
else:
    import json

    PROGRESS_EVERY = 500_000
    for name in RUN_METHODS:
        scorer, q_by_user, empty_query = methods[name]
        out_path = f"/kaggle/working/predictions_{name}.txt"
        t0 = time.time()
        n = 0
        with open(out_path, "w") as f:
            for row in iter_raw_impressions(TESTSET_ZIP, "test", has_labels=False):
                query = q_by_user.get(row["user_id"], empty_query)
                scores = scorer.score(query, row["article_ids"])
                ranks = ranks_for_impression(scores, row["raw_impression_id"], seed=0)
                f.write(f"{row['raw_impression_id']} {json.dumps(ranks, separators=(',', ':'))}\n")
                n += 1
                if n % PROGRESS_EVERY == 0:
                    elapsed = time.time() - t0
                    rate = n / elapsed
                    eta_s = (EXPECTED_TOTAL - n) / rate
                    print(f"  [{name}] {n}/{EXPECTED_TOTAL} "
                          f"({100 * n / EXPECTED_TOTAL:.1f}%), "
                          f"{rate:.1f} impressions/s, ETA {eta_s / 60:.1f} min")
        elapsed = time.time() - t0
        print(f"[{name}] DONE: {n} lines in {elapsed:.1f}s -> {out_path}")


# %% CELL 8 — VALIDATE line count and format before packaging/downloading
# anything. Same class of cheap full-file check the MINDlarge_test session
# used before trusting a multi-hour job's output.
import json as _json

for name in RUN_METHODS:
    path = f"/kaggle/working/predictions_{name}.txt"
    if not os.path.exists(path):
        print(f"[{name}] {path} not found — skipping (was RUN_FULL_JOB True?)")
        continue
    n_lines = 0
    n_malformed = 0
    with open(path) as f:
        for line in f:
            n_lines += 1
            try:
                impid, ranks_raw = line.rstrip("\n").split(" ", 1)
                ranks = _json.loads(ranks_raw)
                assert sorted(ranks) == list(range(1, len(ranks) + 1))
            except Exception:
                n_malformed += 1
    ok = (n_lines == EXPECTED_TOTAL) and (n_malformed == 0)
    print(f"[{name}] lines: {n_lines} (expected {EXPECTED_TOTAL}), "
          f"malformed: {n_malformed} -> {'OK' if ok else 'FAILED — do not package/upload'}")


# %% CELL 9 — PACKAGE. predictions_large_random.zip's confirmed real
# structure (Part 0): a single `predictions.txt` at the zip root, no
# nested folder. Pick the method to submit (if both were run, choose the
# one that's actually going to Codabench — only one file per submission).
SUBMIT_METHOD = RUN_METHODS[0]  # change if you ran both and want the other

src_txt = f"/kaggle/working/predictions_{SUBMIT_METHOD}.txt"
out_zip = "/kaggle/working/prediction.zip"
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(src_txt, arcname="predictions.txt")

with zipfile.ZipFile(out_zip) as zf:
    names = zf.namelist()
    print(f"{out_zip} contents: {names}")
    assert names == ["predictions.txt"], (
        f"expected exactly one root-level predictions.txt, got {names}"
    )
print("Packaged OK.")


# %% CELL 10 — final instructions (nothing left to run).
print(
    "Download ONLY /kaggle/working/prediction.zip back to the local repo "
    "(submissions/ebnerd_testset_" + SUBMIT_METHOD + "/prediction.zip) — "
    "not ebnerd_testset.zip or articles_large_only.zip, per the task "
    "brief's step 7.\n\n"
    "Then, manually (your own Codabench account):\n"
    "  1. Go to https://www.codabench.org/competitions/2469/\n"
    "  2. Submit prediction.zip to the leaderboard.\n"
    "  3. Screenshot the result for Q6.\n\n"
    "Relay back: this notebook's full printed output (Cells 3, 6, 8 "
    "especially), and the final leaderboard screenshot/score."
)
