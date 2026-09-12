# Corpus Stats — Anchor Numbers

Recomputed 2026-09-11 with `scripts/compute_corpus_stats.py`, using this project's
own `tokenize()` + `build_index()` on title+abstract. The result matches the
2026-08-31 run exactly, and ADR-006's 23.5 MB sparse-matrix figure.

## The 8 to remember

| # | Anchor | MIND (MINDlarge-dev) | EB-NeRD (ebnerd_small) |
|---|---|---:|---:|
| 1 | Articles indexed | **72,023** | **20,738** |
| 2 | Vocab, pre → post stopword | **63,561 → 63,441** | **43,582 → 43,440** |
| 3 | Total tokens, pre → post stopword | **3.43M → 2.33M** | **521K → 354K** |
| 4 | Avg tokens/doc, post-stopword (median) | **32.4** (26) | **17.1** (17) |
| 5 | BM25 sparse matrix / retained in RAM / pickle | **23.5 / 35.2 / 25.9 MB** | **4.0 / 10.1 / 5.2 MB** |
| 6 | Impressions / users (eval split) | **376,471 / 255,990** | **244,647 / 15,342** |
| 7 | Candidates per impression (mean) | **37.4** | **12.0** |
| 8 | Test set impressions (blind) | **2,370,727** | **13,536,710** |

The one-line story: removing the ~120 English+Danish stopwords drops just **0.2%
of vocabulary types** but **32% of tokens**. That is Zipf's law: a few types
carry a large share of the mass. It also explains why stopword removal moved
EB-NeRD recall@200 from below random to above random (ADR-005).

## Full table (every locally-built corpus)

| Corpus | Docs | Vocab pre | Vocab post | Tokens pre | Tokens post | Avg/doc post | Median | Sparse MB | RAM MB | Pickle MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MINDsmall dev | 42,416 | 49,901 | 49,785 | 1,929,612 | 1,310,688 | 30.9 | 25 | 13.3 | 21.9 | 14.9 |
| MINDlarge dev | 72,023 | 63,561 | 63,441 | 3,428,786 | 2,334,433 | 32.4 | 26 | 23.5 | 35.2 | 25.9 |
| MINDlarge test | 120,959 | 80,337 | 80,215 | 5,886,366 | 4,016,079 | 33.2 | 27 | 40.2 | 57.1 | 44.1 |
| ebnerd_demo | 11,777 | 31,642 | 31,505 | 303,890 | 205,029 | 17.4 | 17 | 2.4 | 7.0 | 3.1 |
| ebnerd_small | 20,738 | 43,582 | 43,440 | 521,089 | 353,632 | 17.1 | 17 | 4.0 | 10.1 | 5.2 |

## Split sizes

| Split | Articles | Impressions | Users | Candidate rows |
|---|---:|---:|---:|---:|
| MINDsmall train / dev | 51,282 / 42,416 | 156,965 / 73,152 | 50,000 / 50,000 | 5.84M / 2.74M |
| MINDlarge train / dev / test | 101,527 / 72,023 / 120,959 | 2,232,748 / 376,471 / 2,370,727 | 711,222 / 255,990 / 702,005 | 83.5M / 14.1M / 93.1M |
| ebnerd_small train / val | 20,738 (shared) | 232,887 / 244,647 | 15,143 / 15,342 | 2.59M / 2.93M |
| ebnerd_large train / val / test | not downloaded locally | 12,063,890 / 12,566,385 / 13,536,710 | — | — |

## Definitions and caveats

- **Pre-stopword**: raw lowercase `\w+` tokens. **Post-stopword**: what the index stores, so
  the index vocab equals the post-stopword vocab exactly in all five corpora.
- **Sparse MB**: exact `nbytes` of the CSR weight matrix. **RAM MB**: the index object's
  `tracemalloc`-retained footprint (matrix plus vocab/id dicts). The 2026-08-31
  artifact's "memory" column (43.9 MB for MINDlarge-dev) used a `sys.getsizeof` estimate,
  a different method. This RAM figure supersedes it.
- MIND has a separate article catalog per split. EB-NeRD shares one catalog within a bundle.
- Impression and user counts come from the 2026-08-31 full scan. They are cross-checked
  against parquet metadata here (user_history rows = users) and against J/K run records.
- ebnerd_large counts were measured on Ada during Candidate K runs (ADR-013). No local
  articles or index exist for it.
