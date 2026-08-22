import os
os.makedirs("/kaggle/working/raw", exist_ok=True)
# !wget https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip -O /kaggle/working/raw/ebnerd_small.zip
# !wget https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_contrastive_vector.zip -O /kaggle/working/raw/Ekstra_Bladet_contrastive_vector.zip
# !ls -la /kaggle/working/raw/

import os, shutil, subprocess, sys

REAL_TARGETS = ["ebnerd_small.zip", "Ekstra_Bladet_contrastive_vector.zip"]
found = {}
for search_root in ["/kaggle/input", "/kaggle/working"]:
    for root, _dirs, files in os.walk(search_root):
        for f in files:
            if f in REAL_TARGETS and f not in found:
                found[f] = os.path.join(root, f)
for t in REAL_TARGETS:
    print(f"{t}: {found.get(t, 'NOT FOUND')}")
missing = [t for t in REAL_TARGETS if t not in found]
if missing:
    raise FileNotFoundError(f"Missing required inputs: {missing}")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rank_bm25"], check=True)

REPO_DIR = "/kaggle/working/repo"
if not os.path.exists(REPO_DIR):
    shutil.copytree("/kaggle/input/datasets/apollo19/contrastive", REPO_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "scripts"))
print("repo ready at", REPO_DIR)

# %% CELL 2 — build the local ebnerd_small feature store on Kaggle, via the
# project's OWN unmodified pipeline code (src/pipeline/orchestrator.py's
# build_ebnerd_bundle) — guarantees byte-identical schema/splitting logic
# to what produced the MiniLM baseline this experiment compares against,
# rather than re-deriving parsing logic independently in this notebook.
from pathlib import Path

from src.pipeline.orchestrator import build_ebnerd_bundle

processed_dir = Path(REPO_DIR) / "data" / "processed" / "ebnerd" / "small"
build_ebnerd_bundle(Path(found["ebnerd_small.zip"]), processed_dir)
print("Built:", list(processed_dir.rglob("*.parquet")))


# %% CELL 3 — inspect the contrastive-vector archive BEFORE assuming
# anything about its format (this experiment's own objective: "confirm
# it's per-article vectors keyed by article ID... check what it was
# actually trained on").
sys.path.insert(0, os.path.join(REPO_DIR, "scripts"))
from run_contrastive_vector_experiment import inspect_artifact

artifact_zip = Path(found["Ekstra_Bladet_contrastive_vector.zip"])
zf = inspect_artifact(artifact_zip)


# %% CELL 4 — run the full experiment: load the contrastive-vector index
# (the one new function), then reuse recall@K + the Q4 ranking harness
# completely unchanged.
#
# dataset_paths() (from the bundled run_bm25_experiment.py) resolves paths
# via src/utils/config.py's PROCESSED_DIR, which is PROJECT_ROOT/data/
# processed — PROJECT_ROOT here is REPO_DIR (2 parents up from
# src/utils/config.py), matching where Cell 2 wrote the feature store.
from run_contrastive_vector_experiment import run

output = run(artifact_zip)
config, results = output["config"], output["results"]
print(__import__("json").dumps(config, indent=2))


# %% CELL 5 — print the direct comparison against the MiniLM baseline
# (ADR-008's real ebnerd_small validation numbers), with an explicit
# CI-clear-margin check, not just a point-estimate comparison.
print("=== recall@K vs MiniLM baseline ===")
for k, c in results["comparison_vs_minilm_baseline"]["recall_at_k"].items():
    print(f"  k={k}: contrastive={c['contrastive_overall']:.4f} vs minilm={c['minilm_baseline']:.4f} "
          f"ci_clear_win={c['ci_clear_win']}")

print("\n=== Q4 ranking (overall) vs MiniLM baseline ===")
for metric, c in results["comparison_vs_minilm_baseline"]["ranking_overall"].items():
    print(f"  {metric}: contrastive={c['contrastive']:.4f} "
          f"(95% CI {c['contrastive_ci_low']:.4f}-{c['contrastive_ci_high']:.4f}) "
          f"vs minilm={c['minilm_baseline']} ci_clear_win={c['ci_clear_win']}")


# %% CELL 6 — SUMMARY (paste this whole cell's output back)
import json
print("=" * 80)
print("SUMMARY (paste this whole block back)")
print("=" * 80)
print(json.dumps(config, indent=2))
print(json.dumps(results["comparison_vs_minilm_baseline"], indent=2))


# %% CELL 7 — package just the two small JSON result files for download
# (NOT the artifact itself — it's reproducible from the pinned S3 URL in
# config.json, no need to move 341MB back over the same unreliable link).
out_dir = Path("/kaggle/working") / f"contrastive_vector_ebnerd_small_{config['date']}"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "config.json").write_text(json.dumps(config, indent=2))
(out_dir / "results.json").write_text(json.dumps(results, indent=2))

shutil.make_archive("/kaggle/working/contrastive_vector_results", "zip", out_dir)
print("Download /kaggle/working/contrastive_vector_results.zip and place its contents at "
      f"experiments/{out_dir.name}/ locally.")


# %% CELL 2 — build the local ebnerd_small feature store on Kaggle, via the
# project's OWN unmodified pipeline code (src/pipeline/orchestrator.py's
#6- build_ebnerd_bundle) — guarantees byte-identical schema/splitting logic
# to what produced the MiniLM baseline this experiment compares against,
# rather than re-deriving parsing logic independently in this notebook.
from pathlib import Path

from src.pipeline.orchestrator import build_ebnerd_bundle

processed_dir = Path(REPO_DIR) / "data" / "processed" / "ebnerd" / "small"
build_ebnerd_bundle(Path(found["ebnerd_small.zip"]), processed_dir)
print("Built:", list(processed_dir.rglob("*.parquet")))


# %% CELL 3 — inspect the contrastive-vector archive BEFORE assuming
# anything about its format (this experiment's own objective: "confirm
# it's per-article vectors keyed by article ID... check what it was
# actually trained on").
sys.path.insert(0, os.path.join(REPO_DIR, "scripts"))
from run_contrastive_vector_experiment import inspect_artifact

artifact_zip = Path(found["Ekstra_Bladet_contrastive_vector.zip"])
zf = inspect_artifact(artifact_zip)


# %% CELL 4 — run the full experiment: load the contrastive-vector index
# (the one new function), then reuse recall@K + the Q4 ranking harness
# completely unchanged.
#
# dataset_paths() (from the bundled run_bm25_experiment.py) resolves paths
# via src/utils/config.py's PROCESSED_DIR, which is PROJECT_ROOT/data/
# processed — PROJECT_ROOT here is REPO_DIR (2 parents up from
# src/utils/config.py), matching where Cell 2 wrote the feature store.
from run_contrastive_vector_experiment import run

output = run(artifact_zip)
config, results = output["config"], output["results"]
print(__import__("json").dumps(config, indent=2))


# %% CELL 5 — print the direct comparison against the MiniLM baseline
# (ADR-008's real ebnerd_small validation numbers), with an explicit
# CI-clear-margin check, not just a point-estimate comparison.
print("=== recall@K vs MiniLM baseline ===")
for k, c in results["comparison_vs_minilm_baseline"]["recall_at_k"].items():
    print(f"  k={k}: contrastive={c['contrastive_overall']:.4f} vs minilm={c['minilm_baseline']:.4f} "
          f"ci_clear_win={c['ci_clear_win']}")

print("\n=== Q4 ranking (overall) vs MiniLM baseline ===")
for metric, c in results["comparison_vs_minilm_baseline"]["ranking_overall"].items():
    print(f"  {metric}: contrastive={c['contrastive']:.4f} "
          f"(95% CI {c['contrastive_ci_low']:.4f}-{c['contrastive_ci_high']:.4f}) "
          f"vs minilm={c['minilm_baseline']} ci_clear_win={c['ci_clear_win']}")


# %% CELL 6 — SUMMARY (paste this whole cell's output back)
import json
print("=" * 80)
print("SUMMARY (paste this whole block back)")
print("=" * 80)
print(json.dumps(config, indent=2))
print(json.dumps(results["comparison_vs_minilm_baseline"], indent=2))


# %% CELL 7 — package just the two small JSON result files for download
# (NOT the artifact itself — it's reproducible from the pinned S3 URL in
# config.json, no need to move 341MB back over the same unreliable link).
out_dir = Path("/kaggle/working") / f"contrastive_vector_ebnerd_small_{config['date']}"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "config.json").write_text(json.dumps(config, indent=2))
(out_dir / "results.json").write_text(json.dumps(results, indent=2))

shutil.make_archive("/kaggle/working/contrastive_vector_results", "zip", out_dir)
print("Download /kaggle/working/contrastive_vector_results.zip and place its contents at "
      f"experiments/{out_dir.name}/ locally.")


