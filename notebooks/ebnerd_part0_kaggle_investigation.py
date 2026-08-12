"""EB-NeRD Codabench submission — Part 0 format investigation, FOR KAGGLE.

Paste each `# %% CELL` block into its own cell in a Kaggle notebook and run
top to bottom. Purpose: ground-truth the real submission format and the real
ebnerd_testset schema by inspecting the actual files, per CLAUDE.md's
Resource Availability clause (predictions_large_random.zip / ebnerd_testset.zip
/ ebnerd_large.zip / articles_large_only.zip can't be pulled to the local
8GB machine, so this step has to run here).

Before running:
1. Add your existing Kaggle Dataset/Competition input containing
   predictions_large_random.zip, ebnerd_testset.zip, articles_large_only.zip
   (ebnerd_large.zip optional, only needed if articles_large_only.zip turns
   out insufficient) to this notebook's Data sources.
2. Upload `ebnerd_small_demo_article_ids.csv` (written alongside this script,
   in the same scratchpad/notebooks location) as a small private Kaggle
   Dataset and add it too — it's the union of ebnerd_demo + ebnerd_small's
   article IDs (21,700 raw int article_ids), computed locally from this
   project's already-built feature store, used to check whether the smaller
   local bundles already cover the test period's articles.
3. Run all cells. Copy the full printed output back — that's what drives
   Part 1's converter design and confirms/refutes the task brief's
   assumptions before any local code gets written against them.
"""

# %% CELL 0 — discover inputs by filename (works regardless of dataset slug)
import os
import zipfile
import io

TARGETS = [
    "predictions_large_random.zip",
    "ebnerd_testset.zip",
    "articles_large_only.zip",
    "ebnerd_large.zip",
    "ebnerd_small_demo_article_ids.csv",
]

found = {}
for root, _dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f in TARGETS:
            found[f] = os.path.join(root, f)

for t in TARGETS:
    print(f"{t}: {found.get(t, 'NOT FOUND')}")

missing_required = [t for t in TARGETS[:3] if t not in found]
if missing_required:
    raise FileNotFoundError(
        f"Missing required inputs: {missing_required}. Add the dataset "
        "containing them under this notebook's Data sources before continuing."
    )


# %% CELL 1 — inspect predictions_large_random.zip's literal structure
print("=" * 80)
print("predictions_large_random.zip")
print("=" * 80)

pred_zip = zipfile.ZipFile(found["predictions_large_random.zip"])
names = pred_zip.namelist()
print(f"{len(names)} entries")
for n in names[:50]:
    info = pred_zip.getinfo(n)
    print(f"  {n}  ({info.file_size} bytes)")
if len(names) > 50:
    print(f"  ... and {len(names) - 50} more")

# Find the actual prediction file(s) — likely a .txt or .csv, not a directory
candidate_files = [n for n in names if not n.endswith("/")]
for n in candidate_files:
    print(f"\n--- first 10 lines of {n} ---")
    with pred_zip.open(n) as f:
        for i, line in enumerate(io.TextIOWrapper(f, encoding="utf-8")):
            if i >= 10:
                break
            print(repr(line))


# %% CELL 2 — inspect ebnerd_testset.zip against Table 7's documented schema
import pandas as pd

print("=" * 80)
print("ebnerd_testset.zip — file layout")
print("=" * 80)

test_zip = zipfile.ZipFile(found["ebnerd_testset.zip"])
test_names = test_zip.namelist()
for n in test_names:
    if not n.endswith("/"):
        info = test_zip.getinfo(n)
        print(f"  {n}  ({info.file_size} bytes)")

# Find the test-split behaviors.parquet member (path may vary — print above
# confirms the exact member name before this assumption is trusted)
behaviors_member = next(
    n for n in test_names if n.endswith("behaviors.parquet") and "test" in n.lower()
)
print(f"\nUsing behaviors member: {behaviors_member}")

with test_zip.open(behaviors_member) as f:
    behaviors = pd.read_parquet(io.BytesIO(f.read()))

print("\ncolumns:", list(behaviors.columns))
print("\ndtypes:\n", behaviors.dtypes)
print("\nrow count:", len(behaviors))

# Table 7 says test set drops: Article ID, Next read-time, Next Scroll
# Percentage, Clicked Article IDs — confirm directly rather than trusting the text
expected_absent = [
    "article_id", "next_read_time", "next_scroll_percentage", "article_ids_clicked",
]
for col in expected_absent:
    present = col in behaviors.columns
    print(f"  '{col}' present in test behaviors.parquet: {present}")

# Table 7 says test adds is_beyond_accuracy for a 200,000-sample subset
if "is_beyond_accuracy" in behaviors.columns:
    print("\nis_beyond_accuracy value_counts:")
    print(behaviors["is_beyond_accuracy"].value_counts(dropna=False))
else:
    print("\n'is_beyond_accuracy' column NOT FOUND — check exact column name above")

print("\nsample rows:")
print(behaviors.head(5).to_string())

# What does the fixed 250-article beyond-accuracy pool actually look like?
if "is_beyond_accuracy" in behaviors.columns:
    ba_rows = behaviors[behaviors["is_beyond_accuracy"] == True]
    print(f"\nbeyond-accuracy rows: {len(ba_rows)}")
    if len(ba_rows) > 0:
        inview_lengths = ba_rows["article_ids_inview"].apply(len)
        print("inview list length stats (should mostly be one fixed pool size):")
        print(inview_lengths.describe())
        pools = ba_rows["article_ids_inview"].apply(lambda x: tuple(sorted(x)))
        print(f"distinct inview pools across all beyond-accuracy rows: {pools.nunique()}")


# %% CELL 3 — does articles_large_only.zip cover the test period, or does
# ebnerd_small/demo already cover it?
print("=" * 80)
print("articles_large_only.zip vs. test in-view coverage")
print("=" * 80)

art_zip = zipfile.ZipFile(found["articles_large_only.zip"])
art_names = art_zip.namelist()
print("members:", [n for n in art_names if not n.endswith("/")])

articles_member = next(n for n in art_names if n.endswith("articles.parquet"))
with art_zip.open(articles_member) as f:
    large_articles = pd.read_parquet(io.BytesIO(f.read()), columns=["article_id", "published_time"])

large_ids = set(large_articles["article_id"].astype("int64"))
print(f"\narticles_large_only: {len(large_ids)} unique article_ids")
print("published_time range:", large_articles["published_time"].min(), "->", large_articles["published_time"].max())

# unique in-view ids referenced by the test set
test_inview_ids = set()
for lst in behaviors["article_ids_inview"]:
    test_inview_ids.update(int(x) for x in lst)
print(f"\ntest set: {len(test_inview_ids)} unique in-view article_ids across all impressions")

covered_by_large = test_inview_ids & large_ids
print(f"covered by articles_large_only: {len(covered_by_large)} / {len(test_inview_ids)} "
      f"({100 * len(covered_by_large) / len(test_inview_ids):.2f}%)")

if "ebnerd_small_demo_article_ids.csv" in found:
    small_demo_ids = set(pd.read_csv(found["ebnerd_small_demo_article_ids.csv"])["article_id"].astype("int64"))
    print(f"\nlocal ebnerd_small + ebnerd_demo union: {len(small_demo_ids)} unique article_ids")
    covered_by_small_demo = test_inview_ids & small_demo_ids
    print(f"covered by small+demo alone: {len(covered_by_small_demo)} / {len(test_inview_ids)} "
          f"({100 * len(covered_by_small_demo) / len(test_inview_ids):.2f}%)")
else:
    print("\nebnerd_small_demo_article_ids.csv not uploaded — skipping the small/demo coverage check")

still_missing = test_inview_ids - large_ids
print(f"\ntest in-view ids covered by NEITHER source: {len(still_missing)}")
if still_missing:
    print("sample of uncovered ids:", list(still_missing)[:20])


# %% CELL 4 — print a compact summary to paste back
print("=" * 80)
print("SUMMARY (paste this whole block back)")
print("=" * 80)
print(f"prediction file entries: {names}")
print(f"test behaviors columns: {list(behaviors.columns)}")
print(f"is_beyond_accuracy present: {'is_beyond_accuracy' in behaviors.columns}")
print(f"test row count: {len(behaviors)}")
print(f"test unique in-view article count: {len(test_inview_ids)}")
print(f"articles_large_only coverage: {len(covered_by_large)}/{len(test_inview_ids)}")
