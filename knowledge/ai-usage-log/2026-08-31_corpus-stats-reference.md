# AI Usage Log — 2026-08-31 — Corpus/Index Stats Reference Table

## Prompts (verbatim, in order)

### Prompt 1

> Current Objective: Compute and report every basic descriptive/corpus statistic for both datasets — the boring "how big is this thing" numbers, not methodology. A written quiz just asked for vocabulary size, total token count, index size, and average tokens/document with zero warning, and the same style of question can recur in the viva. Get real numbers, not estimates.
>
> For MIND (small: train+dev, large: test) and EB-NeRD (demo/small/large as applicable): report total unique articles, total impressions, total unique users, average/median history length, average/median candidates per impression, per split.
> For the BM25 index specifically: total vocabulary size (post-tokenization), vocabulary size after stopword removal, total token count across the corpus, average tokens per document (title+abstract), index size on disk/in memory, average/median document length.
> For the embedding index: embedding dimension, total number of vectors, index size, encoding time (already have some of this — 264.4s for 72,023 MIND articles — extend to EB-NeRD's equivalent).
> For the final ranker/model artifacts (NRMS-lite vocab, LightGBM feature count): NRMS's actual trained vocabulary size (real number from the run, not the 3,000-example benchmark), LightGBM's 65 features by name in one place.
> Output this as one dense reference table — dataset × split × stat — something skimmable in under 2 minutes before a viva, not prose.

## What was AI-generated vs. human-written/edited

- **All computation was AI-run, against real local data/artifacts** — no numbers were estimated or reasoned about in the abstract. Specifically:
  - Dataset/corpus/impression/user/history/candidate stats: computed directly from `data/processed/{mind,ebnerd}/**/*.parquet` via a pyarrow+pandas script written for this session (`compute_split_stats.py`), run once per split to bound peak memory on this 8GB machine (largest run: MINDlarge_train, ~1.2GB RSS).
  - BM25 vocabulary/token/doc-length/index-size stats: computed by importing and running this project's own `src/retrieval/tokenize.py`/`src/retrieval/index.py` (not a reimplementation) over the four corpora an actual BM25 experiment indexed (MINDsmall-dev, MINDlarge-dev, ebnerd_demo, ebnerd_small); disk size = real pickle of the built `BM25Index`, memory size = summed real `nbytes`/`sys.getsizeof` of its components.
  - Embedding index stats: read directly from the real cached `.npy`/`.json` sidecar files already on disk at `data/processed/*/embeddings/` (shape, dtype, file size) — no re-encoding.
  - NRMS-lite vocab sizes (21,319 / 26,293) and LightGBM's 65 feature names: pulled verbatim from `experiments/candidate_j_nrms_lite_ada_{2026-08-24,mindlarge_2026-08-25}/config.json` and `experiments/candidate_k_gbdt_ebnerd_large_ada/config.json`; the 65-feature count was cross-checked live against `len(src.retrieval.ebnerd_features.FEATURE_NAMES)`.
  - `ebnerd_large`/`ebnerd_testset` were confirmed (via `PROJECT_STATE.md`/ADR-013 grep, and absence of the raw zips under `data/raw/ebnerd/`) to be undownloaded locally — only their impression counts (measured on Ada during real remote runs) are reported; article/user/history/candidate stats and any embedding index for `ebnerd_large` are flagged as unavailable rather than estimated.
- **Deliverable**: a single dense HTML reference table published as a Claude Artifact (not committed to the repo as a doc — ephemeral quiz/viva-prep aid, not a project decision record). All table content is AI-authored; no human edits were made to the artifact or the underlying computation scripts before delivery.
