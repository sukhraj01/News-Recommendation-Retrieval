"""EB-NeRD Codabench submission — contrastive-vector method, real test-set
run, FOR KAGGLE.

Paste each `# %% CELL` block into its own cell in a Kaggle notebook and run
top to bottom. Purpose: generate a real `predictions.txt` for the blind
`ebnerd_testset.zip` using EB-NeRD's provided `contrastive_vector` artifact
instead of this project's own MiniLM embeddings, and package it for upload
to https://www.codabench.org/competitions/2469/ as a SECOND submission —
the local validation comparison (ADR-008 Addendum, 2026-08-19) found the
contrastive vector wins recall@K and every Q4 accuracy metric on
`ebnerd_small` by a CI-clear margin, so this measures whether that holds on
the real held-out test set.

This is `notebooks/ebnerd_part2_kaggle_test_run.py` (the exact notebook
that produced the already-submitted, Codabench-validated
`submissions/ebnerd_testset_embed/prediction.zip`, leaderboard score
0.5404, ID 888045) with ONE substitution: Cell 4 builds the embedding index
via `load_contrastive_index` (already built and verified in the local
`ebnerd_small` comparison — the identical function, imported unchanged from
the same `scripts/run_contrastive_vector_experiment.py`) instead of
`build_embedding_index` encoding text with MiniLM. Every other cell —
discovery, the benchmark-then-decide gate, the checkpointed full run,
line-count/format validation, packaging — is unchanged in structure from
that known-working reference. `src/submission/ebnerd_format.py` (the
official-format converter — `write_predictions`'s line/loop shape is
hand-inlined into Cell 7 here, same as the reference, for its
checkpointing) is reused completely unmodified, as is every other shared
pipeline file.

**Output format, pre-verified against the real, already-uploaded reference
file** (`submissions/ebnerd_testset_embed/prediction.zip`, inspected
directly before writing this notebook): a zip containing exactly one
root-level `predictions.txt`, one line per impression,
`{raw_impression_id} [rank_1,...,rank_N]` (1-indexed ranks, JSON array, no
spaces after commas), 13,536,710 lines, in the exact row order of
`test/behaviors.parquet` — Cell 8 here re-derives and checks all of this
against the real file before anything is packaged, exactly like the
reference notebook did (and like CLAUDE.md's "verify before handing back"
discipline requires) — not assumed correct because it worked once before
for a different embedding source.

**No GPU needed this time** — unlike the MiniLM reference run, there is no
encode step here (the contrastive vectors are provided, not computed), so
the one GPU-beneficial step in the original notebook doesn't exist in this
one. Leave the accelerator OFF to preserve Kaggle GPU-hour quota for other
work; everything here (loading vectors, mean-pooling queries, the scoring
loop) is plain CPU/numpy regardless.

**One real cost difference to expect, not assume:** the contrastive
vectors are 768-dim vs. MiniLM's 384-dim — twice the width, so each
`EmbeddingScorer` dot product costs roughly 2x. The reference run's real
projection was ~12.80h (confirmed feasible in a single ~25h Kaggle commit
run). Cell 6 below measures THIS run's real ms/impression on real Kaggle
hardware rather than assuming the same 12.80h still holds — per this
project's own Benchmarking Philosophy, read its printed projection before
touching Cell 7's `RUN_FULL_JOB` flag.

Before running:

1. Leave GPU accelerator OFF (see above).
2. Upload `ebnerd_contrastive_vector_testset_src_bundle.zip` (written
   alongside this script) as a new private Kaggle Dataset and attach it as
   a Data source — same pattern as the reference notebook, just a
   different (leaner — no MIND code, includes `src/submission/`) bundle.
3. Cell 0 `wget`s `ebnerd_testset.zip`, `articles_large_only.zip`, and
   `Ekstra_Bladet_contrastive_vector.zip` directly into `/kaggle/working/`.
   ~1.9GB total (1.5GB testset + ~140MB articles + ~341MB contrastive
   vectors).
4. Run cells in order. Cell 6 is a benchmark-then-decide gate, exactly like
   the reference — do not flip `RUN_FULL_JOB` blind.
5. Copy Cell 3's discovery output and Cell 6's projection back before
   committing to the full run, same relay pattern as every prior Kaggle
   handoff in this project.
6. If Cell 6's projection is close to or over a single session's cap,
   Cell 7's cross-session checkpoint/resume logic (identical to the
   reference notebook's, unchanged) is the fallback — see its own
   docstring for the exact workflow.
"""

# %% CELL 0 — download the three source files directly into the notebook.
# !wget -q https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip -O /kaggle/working/ebnerd_testset.zip
# !wget -q https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/articles_large_only.zip -O /kaggle/working/articles_large_only.zip
# !wget -q https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_contrastive_vector.zip -O /kaggle/working/Ekstra_Bladet_contrastive_vector.zip
# !ls -lh /kaggle/working/


# %% CELL 1 — discover inputs, install rank_bm25 (still needed only because
# src/retrieval/score.py imports src/retrieval/index.py at module level,
# which imports rank_bm25 — BM25 itself is never invoked by this run), put
# the src bundle on sys.path.
import os
import sys
import time
import zipfile

NOTEBOOK_START_TIME = time.time()

TARGETS = ["ebnerd_testset.zip", "articles_large_only.zip", "Ekstra_Bladet_contrastive_vector.zip"]

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
        "Run Cell 0 for the three zips, and attach the uploaded src-bundle "
        "Dataset (see this script's header) for src/."
    )

sys.path.insert(0, src_bundle_root)
sys.path.insert(0, os.path.join(src_bundle_root, "scripts"))

get_ipython().system("pip install -q rank_bm25")

TESTSET_ZIP = found["ebnerd_testset.zip"]
ARTICLES_ZIP = found["articles_large_only.zip"]
CONTRASTIVE_ZIP = found["Ekstra_Bladet_contrastive_vector.zip"]
print("\nimports OK (deferred — see Cell 4 for the retrieval-module imports)")


# %% CELL 2 — memory/CPU headroom check.
import psutil

vm = psutil.virtual_memory()
print(f"total RAM: {vm.total / 1024**3:.1f} GB")
print(f"available RAM right now: {vm.available / 1024**3:.1f} GB")
print(f"CPU count: {os.cpu_count()}")


# %% CELL 3 — DISCOVERY. Re-confirm the real facts this run depends on
# directly against the actual files on THIS session, same as the reference
# notebook's Cell 3 (its own findings — the extra top-level directory
# ebnerd_testset.zip wraps every member in, and test/history.parquet's real
# presence — are already handled by the shared, unmodified
# src/utils/io.py::read_zip_parquet; nothing new to discover about the
# packaging here, but the real counts are re-measured, not assumed to
# match the reference run's numbers unchanged).
import zipfile

from src.datasets.ebnerd import _parse_history, parse_ebnerd_articles
from src.utils.io import read_zip_parquet

print("=" * 80)
print("ebnerd_testset.zip — behaviors.parquet")
print("=" * 80)
behaviors_meta = read_zip_parquet(
    TESTSET_ZIP, "test/behaviors.parquet",
    columns=["impression_id", "user_id", "is_beyond_accuracy"],
)
n_test_impressions = len(behaviors_meta)
print(f"test impressions (real count): {n_test_impressions}")
EXPECTED_TOTAL = 13_536_710
if n_test_impressions != EXPECTED_TOTAL:
    print(f"WARNING: real count ({n_test_impressions}) != reference run's "
          f"figure ({EXPECTED_TOTAL}). Using the real measured count for "
          "every downstream check in this notebook.")
    EXPECTED_TOTAL = n_test_impressions
del behaviors_meta

print("\n" + "=" * 80)
print("articles_large_only.zip — real article count")
print("=" * 80)
articles = parse_ebnerd_articles(ARTICLES_ZIP)
print(f"articles: {len(articles)}")

history = _parse_history(TESTSET_ZIP, "test")
print(f"\ntest/history.parquet rows (users with history): {len(history)}")


# %% CELL 4 — BUILD THE CONTRASTIVE-VECTOR INDEX over the full
# articles_large_only corpus. THE ONE SUBSTITUTION this notebook makes
# relative to the reference: load_contrastive_index (imported unchanged
# from scripts/run_contrastive_vector_experiment.py — the exact function
# already verified in the local ebnerd_small comparison) replaces
# build_embedding_index. It also prints the real archive contents/any
# accompanying docs (inspect_artifact) before loading anything, same
# "confirm before assuming" discipline as the local run.
#
# Real coverage note from the local ebnerd_small run: the contrastive
# artifact's own row count (125,541) already matched ebnerd_small's parent
# corpus size exactly, so full coverage of articles_large_only here is
# expected but re-measured below, not assumed.
from pathlib import Path

from run_contrastive_vector_experiment import inspect_artifact, load_contrastive_index
from src.retrieval.score import EmbeddingScorer

inspect_artifact(Path(CONTRASTIVE_ZIP))

articles = articles.copy()
articles["_raw_id"] = articles["article_id"].str.replace("ebnerd:", "", regex=False)

t0 = time.time()
contrastive_index, coverage_stats = load_contrastive_index(Path(CONTRASTIVE_ZIP), articles)
index_build_s = time.time() - t0
contrastive_mb = contrastive_index.vectors.nbytes / 1024**2
print(f"\nContrastive index: {index_build_s:.1f}s, vectors {contrastive_mb:.1f} MB")
print(f"Coverage: {coverage_stats}")
if coverage_stats["coverage_fraction"] < 0.99:
    print(
        "NOTE: coverage below 99% — a real fraction of articles_large_only "
        "has no contrastive vector and will fall back to score.py's "
        "existing -inf-for-missing-id handling (deterministically ranks "
        "last), same as any genuinely out-of-catalog article. Not an "
        "error, but worth relaying back explicitly rather than assuming "
        "full coverage silently held."
    )


# %% CELL 5 — CONFIG + build per-user queries. Single method here
# (RUN_METHODS kept as a list, matching the reference notebook's shape, in
# case a future session wants to add "bm25" back for comparison — not
# built by default here, this run's whole point is the contrastive vector).
#
# Real memory note: contrastive vectors are 768-dim vs. MiniLM's 384-dim —
# the reference run's own measured ~1.16GB projection for 807,677 users'
# mean-pooled query vectors roughly doubles here (~2.3GB). Still well
# within the ~29.6GB RAM the reference run found available; re-check the
# RAM print below on THIS session rather than assuming that holds.
RUN_METHODS = ["contrastive"]

from src.retrieval.embed import build_user_embedding_query

vector_lookup = dict(zip(contrastive_index.article_ids, contrastive_index.vectors))
t0 = time.time()
query_by_user = {
    "contrastive": {
        row.user_id: build_user_embedding_query(row.article_ids, vector_lookup)
        for row in history.itertuples(index=False)
    }
}
print(f"Contrastive query_by_user: {len(query_by_user['contrastive'])} users, "
      f"{time.time() - t0:.1f}s")

scorers = {"contrastive": EmbeddingScorer(contrastive_index)}
empty_queries = {"contrastive": None}

vm = psutil.virtual_memory()
print(f"RAM available after query build: {vm.available / 1024**3:.1f} GB")

methods = {name: (scorers[name], query_by_user[name], empty_queries[name]) for name in RUN_METHODS}


# %% CELL 6 — BENCHMARK on a real sample of THIS file, on THIS Kaggle
# instance, before committing to the full run. Identical logic to the
# reference notebook's Cell 6 (locality-preserving contiguous chunks, an
# isolated cache-miss diagnostic decomposed from the caching benefit) —
# only the method name changed. Read its printed projection; do not assume
# the reference run's ~12.80h still applies (see this file's docstring —
# 768-dim vectors cost ~2x more per dot product than MiniLM's 384-dim).
import numpy as np

from src.evaluation.ranking_metrics import rank_candidates
from src.submission.ebnerd_format import ranks_for_impression, sample_raw_impressions

N_CHUNKS = 10
CHUNK_LEN = 10_000
rng = np.random.default_rng(0)
chunk_starts = sorted(rng.integers(0, max(1, EXPECTED_TOTAL - CHUNK_LEN), size=N_CHUNKS).tolist())
chunk_row_positions = [p for start in chunk_starts for p in range(start, start + CHUNK_LEN)]
print(f"benchmark sample: {N_CHUNKS} contiguous chunks of {CHUNK_LEN} rows each "
      f"(starts: {chunk_starts}), {len(chunk_row_positions)} rows total")

t0 = time.time()
chunk_sample = sample_raw_impressions(TESTSET_ZIP, "test", has_labels=False, row_positions=chunk_row_positions)
print(f"sample selection + parsing: {time.time() - t0:.1f}s for {len(chunk_sample)} rows")


def _force_cache_miss_copy(query):
    if query is None:
        return None
    if isinstance(query, list):
        return list(query)
    return np.array(query, copy=True)


print()
isolated_ms = {}
for name, (scorer, q_by_user, empty_query) in methods.items():
    real_query = next((q for q in q_by_user.values() if q is not None and len(q) > 0), empty_query)
    real_candidates = chunk_sample[0]["article_ids"]
    N_ISOLATED = 50
    t0 = time.time()
    for _ in range(N_ISOLATED):
        scorer.score(_force_cache_miss_copy(real_query), real_candidates)
    isolated_ms[name] = 1000 * (time.time() - t0) / N_ISOLATED
    print(f"{name}: isolated cache-miss cost = {isolated_ms[name]:.4f} ms/call "
          f"({len(real_candidates)} candidates, {contrastive_index.vectors.shape[1]}-dim vectors)")

print()
projections = {}
for name, (scorer, q_by_user, empty_query) in methods.items():
    t0 = time.time()
    n = 0
    for row in chunk_sample:
        query = q_by_user.get(row["user_id"], empty_query)
        scores = scorer.score(query, row["article_ids"])
        ranks_for_impression(scores, row["raw_impression_id"], seed=0)
        n += 1
    elapsed = time.time() - t0
    ms_per_impression = 1000 * elapsed / n
    projected_s = ms_per_impression / 1000 * EXPECTED_TOTAL
    projections[name] = projected_s
    cache_benefit = isolated_ms[name] / max(ms_per_impression, 1e-9)
    print(f"{name}: {n} sampled impressions in {elapsed:.1f}s "
          f"({ms_per_impression:.3f} ms/impression, {cache_benefit:.1f}x caching benefit) -> "
          f"projected full run: {projected_s / 3600:.2f} hours ({projected_s / 60:.0f} min)")

vm = psutil.virtual_memory()
print(f"\nRAM available after benchmark: {vm.available / 1024**3:.1f} GB")

print("\n" + "=" * 80)
print("DECISION GATE — read before touching Cell 7")
print("=" * 80)
print(
    "Compare the projected hours above against the reference MiniLM run's "
    "real ~12.80h (confirmed feasible in a single ~25h Kaggle commit run). "
    "If this projection is comfortably under that same ~25h commit-run "
    "cap, set RUN_FULL_JOB=True below for a single-pass run, same as the "
    "reference. If it's close to or over the cap, leave RUN_FULL_JOB=False, "
    "relay this cell's output back, and decide on the cross-session "
    "checkpoint workflow (Cell 7's docstring) instead of guessing."
)


# %% CELL 7 — THE FULL RUN, with cross-session checkpointing as a safety
# net — identical logic to the reference notebook's Cell 7 (see its
# docstring there for the full cross-session recovery workflow if this
# ever triggers); only the output filename (predictions_contrastive.txt)
# and RUN_METHODS differ.
#
# Do not flip this True without reading Cell 6's real projection first.
RUN_FULL_JOB = False  # set True only after reading Cell 6's real projection
MAX_RUNTIME_HOURS = 20.0  # same comfortable buffer under the ~25h commit-run limit the reference run confirmed; lower this and re-plan if Cell 6 projects meaningfully over ~20h

if not RUN_FULL_JOB:
    print("RUN_FULL_JOB is False — not running. Set it True after "
          "reviewing Cell 6's projection.")
else:
    import json

    from src.submission.ebnerd_format import iter_raw_impressions_from

    PROGRESS_EVERY = 500_000
    CHECK_DEADLINE_EVERY = 1_000
    deadline = NOTEBOOK_START_TIME + MAX_RUNTIME_HOURS * 3600

    for name in RUN_METHODS:
        scorer, q_by_user, empty_query = methods[name]
        out_path = f"/kaggle/working/predictions_{name}.txt"

        checkpoint_path = None
        for search_root in ["/kaggle/working", "/kaggle/input"]:
            for root, _dirs, files in os.walk(search_root):
                candidate = os.path.join(root, f"predictions_{name}.txt")
                if os.path.exists(candidate):
                    checkpoint_path = candidate
                    break
            if checkpoint_path:
                break

        resume_from = 0
        if checkpoint_path:
            with open(checkpoint_path) as f:
                lines = f.readlines()
            n_valid = len(lines)
            if lines:
                try:
                    impid, ranks_raw = lines[-1].rstrip("\n").split(" ", 1)
                    ranks = json.loads(ranks_raw)
                    assert sorted(ranks) == list(range(1, len(ranks) + 1))
                except Exception:
                    n_valid = len(lines) - 1
                    print(f"  [{name}] last checkpoint line looked truncated — "
                          f"dropping it, resuming from the line before")
            resume_from = n_valid

            if resume_from > 0:
                real_first = sample_raw_impressions(TESTSET_ZIP, "test", has_labels=False, row_positions=[0])[0]
                real_last = sample_raw_impressions(TESTSET_ZIP, "test", has_labels=False, row_positions=[resume_from - 1])[0]
                checkpoint_first_impid = lines[0].split(" ", 1)[0]
                checkpoint_last_impid = lines[resume_from - 1].split(" ", 1)[0]
                if (checkpoint_first_impid != real_first["raw_impression_id"]
                        or checkpoint_last_impid != real_last["raw_impression_id"]):
                    raise RuntimeError(
                        f"STOP — checkpoint at {checkpoint_path} does not match "
                        f"this real ebnerd_testset.zip's row ordering. This "
                        f"looks like a stale or unrelated file — delete or "
                        f"rename it before re-running rather than silently "
                        f"trusting it."
                    )

            if checkpoint_path != out_path or n_valid != len(lines):
                with open(out_path, "w") as f:
                    f.writelines(lines[:n_valid])
            print(f"[{name}] found checkpoint at {checkpoint_path}: verified "
                  f"against the real zip, resuming from row "
                  f"{resume_from}/{EXPECTED_TOTAL} "
                  f"({100 * resume_from / EXPECTED_TOTAL:.1f}%)")

        if resume_from >= EXPECTED_TOTAL:
            print(f"[{name}] already complete ({resume_from} lines) — nothing to do")
            continue

        t0 = time.time()
        n = resume_from
        stopped_early = False
        with open(out_path, "a") as f:
            for row in iter_raw_impressions_from(TESTSET_ZIP, "test", has_labels=False, start_row=resume_from):
                query = q_by_user.get(row["user_id"], empty_query)
                scores = scorer.score(query, row["article_ids"])
                ranks = ranks_for_impression(scores, row["raw_impression_id"], seed=0)
                f.write(f"{row['raw_impression_id']} {json.dumps(ranks, separators=(',', ':'))}\n")
                n += 1
                if n % PROGRESS_EVERY == 0:
                    elapsed = time.time() - t0
                    rate = (n - resume_from) / elapsed
                    eta_s = (EXPECTED_TOTAL - n) / rate
                    print(f"  [{name}] {n}/{EXPECTED_TOTAL} "
                          f"({100 * n / EXPECTED_TOTAL:.1f}%), "
                          f"{rate:.1f} impressions/s, ETA {eta_s / 60:.1f} min")
                if n % CHECK_DEADLINE_EVERY == 0 and time.time() >= deadline:
                    f.flush()
                    stopped_early = True
                    break
        if stopped_early:
            print(f"\n[{name}] CHECKPOINT — stopped cleanly at {n}/{EXPECTED_TOTAL} "
                  f"({100 * n / EXPECTED_TOTAL:.1f}%) after {MAX_RUNTIME_HOURS}h. "
                  f"Download {out_path}, upload it as a new version of your "
                  f"Kaggle checkpoint Dataset, attach it in a fresh session, "
                  f"and re-run this notebook from Cell 1 — Cell 7 will "
                  f"resume from here automatically.")
        else:
            elapsed = time.time() - t0
            print(f"[{name}] DONE: {n} total lines ({n - resume_from} written "
                  f"this session in {elapsed:.1f}s) -> {out_path}")


# %% CELL 8 — VALIDATE line count and format before packaging/downloading
# anything. Identical check to the reference notebook's Cell 8, plus one
# extra structural check specific to this run: re-parses a handful of real
# lines and confirms they match the exact `impid [ranks]` shape verified
# directly against the real, already-uploaded reference prediction.zip
# (submissions/ebnerd_testset_embed/prediction.zip) before this notebook
# was written — not just "valid JSON," the literal shape Codabench's
# evaluate.py parses.
import json as _json

for name in RUN_METHODS:
    path = f"/kaggle/working/predictions_{name}.txt"
    if not os.path.exists(path):
        print(f"[{name}] {path} not found — skipping (was RUN_FULL_JOB True?)")
        continue
    n_lines = 0
    n_malformed = 0
    sample_lines = []
    with open(path) as f:
        for line in f:
            n_lines += 1
            if n_lines <= 3:
                sample_lines.append(line.rstrip("\n"))
            try:
                impid, ranks_raw = line.rstrip("\n").split(" ", 1)
                ranks = _json.loads(ranks_raw)
                assert sorted(ranks) == list(range(1, len(ranks) + 1))
                assert impid.isdigit(), "impression id should be a bare integer string, no prefix"
            except Exception:
                n_malformed += 1
    if n_malformed > 0:
        status = f"FAILED — {n_malformed} malformed line(s), do not package/upload"
    elif n_lines < EXPECTED_TOTAL:
        status = (f"INCOMPLETE — {100 * n_lines / EXPECTED_TOTAL:.1f}% written so "
                  f"far, re-run Cell 7 (resumes automatically) to continue")
    elif n_lines > EXPECTED_TOTAL:
        status = f"FAILED — {n_lines} > expected {EXPECTED_TOTAL}, investigate before packaging"
    else:
        status = "OK"
    print(f"[{name}] lines: {n_lines} (expected {EXPECTED_TOTAL}), "
          f"malformed: {n_malformed} -> {status}")
    print(f"  sample lines: {sample_lines}")


# %% CELL 9 — PACKAGE. Same confirmed real structure as the reference
# submission's prediction.zip (verified by inspecting that actual file
# directly before this notebook was written): a single `predictions.txt`
# at the zip root, no nested folder.
SUBMIT_METHOD = RUN_METHODS[0]

src_txt = f"/kaggle/working/predictions_{SUBMIT_METHOD}.txt"
if not os.path.exists(src_txt):
    print(f"{src_txt} not found — nothing to package. Not an error if "
          f"RUN_FULL_JOB was False or Cell 7 hasn't run yet.")
else:
    out_zip = "/kaggle/working/prediction_contrastive.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(src_txt, arcname="predictions.txt")

    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
        print(f"{out_zip} contents: {names}")
        assert names == ["predictions.txt"], (
            f"expected exactly one root-level predictions.txt, got {names}"
        )
    print("Packaged OK — matches the real reference submission's structure.")


# %% CELL 10 — final instructions.
if not os.path.exists("/kaggle/working/prediction_contrastive.zip"):
    print("No prediction_contrastive.zip yet — Cell 9 had nothing to "
          "package. Nothing to download or submit yet.")
else:
    print(
        "Download ONLY /kaggle/working/prediction_contrastive.zip back to "
        "the local repo (submissions/ebnerd_testset_contrastive/prediction.zip) "
        "— not the raw source zips.\n\n"
        "Then, manually (your own Codabench account):\n"
        "  1. Go to https://www.codabench.org/competitions/2469/\n"
        "  2. Submit prediction_contrastive.zip as a SECOND, separate "
        "entry — this does not replace the existing MiniLM submission "
        "(ID 888045, score 0.5404); both can coexist on the leaderboard.\n"
        "  3. Screenshot the result.\n\n"
        "Relay back: this notebook's full printed output (Cells 3, 6, 8 "
        "especially), and the final leaderboard screenshot/score."
    )
