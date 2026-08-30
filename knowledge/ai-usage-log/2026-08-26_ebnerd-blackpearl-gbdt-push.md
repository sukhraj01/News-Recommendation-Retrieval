# Session Log — 2026-08-26 — EB-NeRD last-ditch push (BlackPearl-style GBDT)

Verbatim prompts from the engineer, in order. AI-vs-human attribution noted per artifact at the end.

---

## Prompt 1 (session opening)

> **Current Objective:** Last-ditch EB-NeRD push, 4 days available. Current state: leaderboard score 0.5404 (submission 888045), below even the official challenge's naive popularity baseline (0.5970). Target: close as much of that gap as possible toward 0.80, grounded in RecSys Challenge 2024's real published results (arXiv:2409.20483) — not the originally-floated 0.85, which only the winning team hit, and only using features later found to leak future information (organizers' own ablation: winner's clean score was 76.99, not 89.24). 2nd place (BlackPearl) is the real target reference: 82.20 AUC, clean, no leakage.
>
> 1. Review what EB-NeRD infra already exists — H2's tree-model code, the feature store, anything from the BM25/embedding retrieval work that's reusable. Don't rebuild what's already there.
>
> 2. Design a feature set that mirrors BlackPearl's core idea (long-term stable interest signals vs. short-term fast-changing signals), since I could only get BlackPearl's abstract/summary, not the full paper (ACM DOI 10.1145/3687151.3687163 and the ResearchGate mirror both failed to fetch) — the specific feature list below is reconstructed from that summary, not verified against their actual methodology, so treat it as a starting hypothesis, not a confirmed spec:
>    - Long-term: aggregated category/subcategory/entity affinity from the user's full 21-day history, click-count-weighted.
>    - Short-term: recency-weighted recent clicks (exponential decay), time-since-last-click, session device/context, last-N-category sequence.
>    If you have a way to actually pull the full paper text (library access, etc.), do that first — it's worth the extra hour before committing to a feature list built on a secondhand summary.
>
> 3. Model: GBDT (LightGBM or CatBoost, whichever H2 already used) as the primary lever, not a neural architecture — this matches what actually won the real competition, per the challenge paper's own "Common Themes" section.
>
> 4. Leakage discipline is not optional here — it's literally what separated the top score from the honest one in the real competition. Every new feature must be checked against real available-at-prediction-time information, same standard as ADR-007's train-only novelty computation. Document this check explicitly per feature, don't just assert it.
>
> 5. Before any full-scale run: estimate memory/time from measured per-unit numbers (37M impressions is EB-NeRD-large scale) per CLAUDE.md's Memory Estimation section, and decide small vs. large EB-NeRD bundle based on what 4 days actually allows.
>
> 6. Checkpoint at roughly day 2: report real intermediate AUC against local validation, then decide — same "checkpoint, report back, then decide" pattern as I → I-long — whether to keep extending features/ensembling or lock in what exists. Don't wait until day 4 to find out it's not working.
>
> 7. Document as a new candidate/ADR: compare against 0.5404 (current), 0.5970 (official popularity baseline we're currently below), 0.7699/0.8220 (honest literature ceiling, winner/2nd-place clean), and whatever this candidate actually achieves. State plainly if 0.80 wasn't reached and why, don't round up.

---

## Prompt 2 (mid-session, unprompted)

> just for info you can run fast local test on y pc fr arger tests i do have ada access wher i have 128gb ram and gpus

## Prompt 3 (answering a blocking decision question)

Asked two questions: (a) how to fix the corrupted local venv, (b) which EB-NeRD bundle to target.

> **Env fix:** "if we need a very bg test we can do it on ada cluster"
> **Bundle:** "Go straight to ebnerd_large on Ada"

Note: (a) did not select any of the offered environment-fix options. The local
venv was provably miscomputing, which blocks local pre-flight verification of
anything before it ships to SLURM, so the conservative offered option (rebuild
on Python 3.11, which `pyproject.toml` already declares) was taken and reported.

---

## Session notes

### Literature grounding (AI-run searches, human-directed)

Two numbers in the opening brief were wrong and were corrected against the
primary source (arXiv:2409.20483) rather than propagated:

- BlackPearl (2nd place) scored **88.15**, not 82.20. No published leakage-free
  BlackPearl figure exists — the organizers ablated only the winner
  (88.64 -> 76.99). The honest clean-performance anchor is therefore 76.99.
- Candidate H2 used `sklearn.ensemble.HistGradientBoostingClassifier`, not
  LightGBM or CatBoost. It has no ranking objective, so it could not have been
  reused for a listwise model.

BlackPearl's full text was NOT obtainable (ACM DOI 403, ResearchGate 403), as
the brief anticipated. A better-grounded substitute was found instead:
FeatureSalad's LightGBM Ranker paper (85.13 AUC, top academic team) plus its
public code repo.

### Environment defect (AI-found, human-notified, AI-fixed)

The project venv ran numpy 1.26.4 on CPython 3.14.0 — a combination with no
released wheel, so it had been source-built against an untested interpreter. It
returned provably wrong results (a stored bool array disagreeing with a fresh
recompute of the identical expression). Full evidence in ADR-013's
"Environment defect" section. Venv rebuilt on Python 3.11; full suite then
passed 295/1 skipped. **Open item:** it is not known whether earlier recorded
results in this project came from the defective build; they have not been
re-verified.

### Real bug found and fixed during development (AI-found, AI-fixed)

EB-NeRD stores timestamps at microsecond resolution. An `int64 / 1e9`
conversion under-scaled by 1000x, silently turning `article_age_h` into a
near-constant (median -467,870 h). Fixed at the root via an explicit
`datetime64[s]` cast; median candidate age is now 3.7 h, with 58% under 6 h,
which is plausible for a news front page. Regression-tested.

### AI vs. human attribution

| Artifact | Status |
|---|---|
| `src/retrieval/ebnerd_features.py` | AI-written, human-directed (feature families specified by the engineer's brief) |
| `scripts/run_ebnerd_gbdt_experiment.py` | AI-written |
| `scripts/ebnerd_prepare_bundle.py` | AI-written |
| `scripts/ebnerd_gbdt_ada.sbatch` | AI-written, modelled on the engineer's existing Candidate J sbatch |
| `scripts/generate_ebnerd_gbdt_predictions.py` | AI-written |
| `tests/unit/test_ebnerd_features.py` | AI-written |
| `tests/integration/test_ebnerd_gbdt_leakage.py` | AI-written |
| `decisions/ADR-013-ebnerd-gbdt-ranker.md` | AI-written; the two literature corrections are AI-found, not in the brief |
| Bundle decision (ebnerd_large on Ada) | Human decision |
| Environment fix choice | AI judgment call after the question went unanswered; reported to the engineer |

### Infra review findings (AI-generated investigation, human-directed)


### Real local result (2026-08-26, ebnerd_small)

K_rank_nopos 0.7524 (95% CI 0.7507-0.7543) vs deployed method's 0.5430 on the
identical split (+0.2084 CI-clear). Reported to the engineer as a checkpoint
per the original brief's instruction #6. Two follow-up decisions asked via
AskUserQuestion and answered:

> **Next step:** "Ada / ebnerd_large first, submit only that"
> **Arm to ship:** "K-rank-nopos (Recommended)"

### Post-checkpoint hardening (AI-run, human-directed by the two decisions above)

- Refactored validation scoring from whole-split materialization to chunked
  streaming (18 min -> 149s, memory-bounded) so `ebnerd_large`'s tens-of-GB
  validation matrix cannot OOM on Ada. Verified bit-identical to the
  pre-refactor numbers (max |AUC delta| 3e-07, one float-noise case; MRR exact).
- Extended `ebnerd_gbdt_ada.sbatch` into one end-to-end job: download ->
  encode -> train -> testset download/encode -> submission generation.
- Dry-ran the submission generator against ebnerd_small validation (treated as
  unlabelled) and scored the output with the bundled official
  `evaluate.py`: **AUC 0.7524, exact match** to the harness's own number.
  Caught and fixed one real defect this way: EB-NeRD's Codabench wants
  `predictions.txt` inside the zip; the bundled evaluator (MIND-flavoured)
  wants `prediction.txt`. Now asserted explicitly in the script.
- A leakage integration test failed on first run (13 articles/5.5M rows carry a
  `published_time` after the impression showing them). Measured rather than
  suppressed: CTR in those rows is *lower* than baseline in both splits, so
  it's a data artifact, not an exploitable leak. Test corrected to bound the
  rate rather than forbid it; left untransformed in the feature (clipping would
  bias toward "maximally fresh", the one direction the model rewards).
- Found and fixed a real defect of my own: importing `lightgbm` inside the same
  pytest process as a torch-based test segfaulted `make test` deterministically
  at `test_nrms.py`. Fixed by moving `per_impression_auc` to
  `src/evaluation/ranking_metrics.py` (its natural home, beside `safe_auc`)
  rather than working around it — this also keeps lightgbm out of the general
  test process entirely. Full suite: 313 passed, 1 skipped, 14 deselected.
- Confirmed no SSH key access to Ada from this session (password auth only) —
  the `ebnerd_large` run and the real Codabench submission are handed to the
  engineer to execute, not run from here.

**No Ada steps have been executed.** Everything in `scripts/ebnerd_gbdt_ada.sbatch`
is verified locally in shape (prep script, training script, submission script,
official-evaluator round trip) but not yet run on the cluster.

---

## Session continuation (2026-08-27 through 2026-08-30, same continuous session)

The system clock advanced across several calendar days during this one
continuous conversation (no session restart) while iterating against a live
Ada SLURM job. Logged as one entry per CLAUDE.md's "at the start of each
session" rule, since it was one session throughout.

### Prompt (Ada resync attempt reported back)

> [pasted terminal: rsync transfer of 300 files completed; `sbatch: error:
> Unable to open file scripts/mind_nrms_predictions.sbatch` from a stray
> earlier command; then `cd ~/assignment-1-news-retrieval` and
> `sbatch scripts/ebnerd_gbdt_ada.sbatch` failing]
>
> sbatch: error: CPU count per node can not be satisfied
> sbatch: error: Batch job submission failed: Requested node configuration is not available
> [sukhraj.singh@ada assignment-1-news-retrieval]$
>
> ealrei iwas runnign like this
> [sukhraj.singh@ada assignment-1-news-retrieval]$ sbatch --export=ALL,OUT_DIR=$HOME/mind_nrms_predictions_corrected scripts/mind_nrms_predictions.sbatch

### Prompt (job submitted successfully after cpus-per-task fix)

> [sukhraj.singh@ada assignment-1-news-retrieval]$ sbatch scripts/ebnerd_gbdt_ada.sbatch
> Submitted batch job 2679035
> [sukhraj.singh@ada assignment-1-news-retrieval]$ squeue -u $USER
>              JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
>            2679035       u22 ebnerd_g sukhraj.  R       0:13      1 gnode033
> [sukhraj.singh@ada assignment-1-news-retrieval]$

### Prompt (job cancelled during slow download)

> [pasted tail -f output: slow 0.1-0.2 MB/s download progress lines, ending in]
> slurmstepd: error: *** JOB 2679035 ON gnode033 CANCELLED AT 2026-08-26T20:36:29 ***

### Prompt

> sbatch: error: CPU count per node can not be satisfied
> [sukhraj.singh@ada assignment-1-news-retrieval]$ sbatch scripts/ebnerd_gbdt_ada_train.sbatch
> Submitted batch job 2679595
> [sukhraj.singh@ada assignment-1-news-retrieval]$ squeue -u $USER
>              JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
>            2679595       u22 ebnerd_g sukhraj. PD       0:00      1 (Priority)
> [sukhraj.singh@ada assignment-1-news-retrieval]$ tail -f ~/assignment-1-news-retrieval/ebnerd_gbdt_k_train_<jobid>.out
> -bash: jobid: No such file or directory
> [sukhraj.singh@ada assignment-1-news-retrieval]$ squeue -u $USER
>              JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
>            2679595       u22 ebnerd_g sukhraj.  R       0:12      1 gnode068
> [sukhraj.singh@ada assignment-1-news-retrieval]$

### Prompt

> [sukhraj.singh@ada assignment-1-news-retrieval]$ tail -f ~/assignment-1-news-retrieval/ebnerd_gbdt_k_train_2679595.out
> ==========================================
> SLURM_JOB_ID = 2679595
> SLURM_NODELIST = gnode068
> SLURM_JOB_GPUS = 1
> ==========================================
> python 3.11.16 | numpy 1.26.4
> ebnerd_large.zip missing on this node -- fetching now (self-healing)
> ebnerd_testset.zip missing on this node -- fetching now (self-healing)
> articles_large_only.zip missing on this node -- fetching now (self-healing)

### Prompt

> if everythign is gettgn dowlaoded agian wth did we dowlaod for so long

### Prompt (real throttled download pasted)

> bro its takign so long
> articles_large_only.zip missing on this node -- fetching now (self-healing)
> [1/3] fetching ebnerd_large.zip
>     0.00 GB (0.0%) @ 0.0 MB/s
>     [... many slow progress lines down to ...]
>     0.16 GB (5.5%) @ 0.0 MB/s

### Prompt

> bro but ealier also i downlaoded glvoe btu such theing not hapended

### Prompt

> Submitted batch job 2679155
> [sukhraj.singh@ada assignment-1-news-retrieval]$ squeue -u $USER
>              JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
>            2679155       u22 ebnerd_g sukhraj.  R       0:27      1 gnode035
> [sukhraj.singh@ada assignment-1-news-retrieval]$

### Prompt

> JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
>            2679155       u22 ebnerd_g sukhraj.  R       0:27      1 gnode035
> [sukhraj.singh@ada assignment-1-news-retrieval]$ scontrol show job 2679155 | grep -E "JobName|Command|Gres"
> JobId=2679155 JobName=ebnerd_gbdt_k_prep
>    Command=/home2/sukhraj.singh/assignment-1-news-retrieval/scripts/ebnerd_gbdt_ada_prepare.sbatch
> [sukhraj.singh@ada assignment-1-news-retrieval]$

### Prompt

> should be compelte now the job

### Prompt (full prepare-stage completion pasted)

> [sukhraj.singh@ada ~]$ tail -f ~/assignment-1-news-retrieval/ebnerd_gbdt_k_prep_2680004.out
>
>   scale report for ebnerd_testset.zip:
> [3/3] embeddings skipped by request
>
> ############ STEP 3/3: fetch articles_large_only (testset's article catalog) ############
> -rw-r--r-- 1 sukhraj.singh research 143M Aug 28 07:47 /ssd_scratch/sukhraj.singh/ebnerd_raw/articles_large_only.zip
>
> === STAGE 1 DONE — no GPU was requested or held ===
> Next: sbatch --dependency=afterok:$SLURM_JOB_ID scripts/ebnerd_gbdt_ada_train.sbatch
> (replace $SLURM_JOB_ID with this job's real ID if running that command outside this job)
> [sukhraj.singh@ada ~]$

### Prompt

> [sukhraj.singh@ada ~]$ ls -lh ~/ebnerd_shared_cache/
> total 4.7G
> -rw-r--r-- 1 sukhraj.singh research 143M Aug 28 07:48 articles_large_only.zip
> -rw-r--r-- 1 sukhraj.singh research 3.0G Aug 28 07:06 ebnerd_large.zip
> -rw-r--r-- 1 sukhraj.singh research 1.6G Aug 28 07:48 ebnerd_testset.zip
> [sukhraj.singh@ada ~]$

### Prompt

> Submitted batch job 2680738
> [sukhraj.singh@ada assignment-1-news-retrieval]$ squeue -u $USER
>              JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
>            2680738       u22 ebnerd_g sukhraj.  R       0:04      1 gnode068
> [sukhraj.singh@ada assignment-1-news-retrieval]$

### Prompt (full run through the OOM traceback)

> [sukhraj.singh@ada assignment-1-news-retrieval]$ tail -f ~/assignment-1-news-retrieval/ebnerd_gbdt_k_train_2680738.out
> ==========================================
> SLURM_JOB_ID = 2680738
> SLURM_NODELIST = gnode068
> SLURM_JOB_GPUS = 1
> ==========================================
> python 3.11.16 | numpy 1.26.4
>   found ebnerd_large.zip in /home2/sukhraj.singh/ebnerd_shared_cache -- copying locally (no S3 fetch needed)
>   found ebnerd_testset.zip in /home2/sukhraj.singh/ebnerd_shared_cache -- copying locally (no S3 fetch needed)
>   found articles_large_only.zip in /home2/sukhraj.singh/ebnerd_shared_cache -- copying locally (no S3 fetch needed)
> [... nvidia-smi output, RTX 2080 Ti ...]
> ############ STEP 1/4: encode ebnerd_large articles (GPU) ############
> [... encode succeeds, 125,541 articles ...]
> ############ STEP 2/4: train + evaluate Candidate K on ebnerd_large ############
> [... K_rank best_iter=340 (1692s), K_cls best_iter=927 (3500s) ...]
> /var/spool/slurmd/job2680914/slurm_script: line 147: 1149071 Killed                  python scripts/run_ebnerd_gbdt_experiment.py [...]
> slurmstepd: error: Detected 1 oom-kill event(s) in StepId=2680914.batch. Some of your processes may have been killed by the cgroup out-of-memory handler.

(Note: job ID in the OOM trace is 2680914, a resubmission during this same
exchange — the engineer pasted output across two related job IDs in sequence
without a separate prompt boundary between them.)

### Prompt

> but the venv can stay active for only 6 hours wht after that

### Prompt (rank_bm25 install saga, verbatim terminal paste)

> (ebnerd_gbdt) [sukhraj.singh@ada assignment-1-news-retrieval]$ uv pip install rank-bm25
> Using Python 3.11.16 environment at: /home2/sukhraj.singh/venvs/ebnerd_gbdt
> Resolved 2 packages in 556ms
> ░░░░░░░░░░░░░░░░░░░░ [0/1] Installing wheels...
> thread 'main2' (34418) panicked at crates/uv-configuration/src/threading.rs:70:14:
> failed to initialize global rayon pool: ThreadPoolBuildError { kind: IOError(Os { code: 11, kind: WouldBlock, message: "Resource temporarily unavailable" }) }
> [... repeated ModuleNotFoundError: No module named 'rank_bm25' on retry ...]
> (ebnerd_gbdt) [sukhraj.singh@ada assignment-1-news-retrieval]$ python -m ensurepip --upgrade
> [... pip 24.0 installed successfully ...]
> (ebnerd_gbdt) [sukhraj.singh@ada assignment-1-news-retrieval]$ python -m pip install rank-bm25
> Successfully installed rank-bm25-0.2.2
> (ebnerd_gbdt) [sukhraj.singh@ada assignment-1-news-retrieval]$ python -c "import rank_bm25; print('rank_bm25 OK')"
> OpenBLAS blas_thread_init: pthread_create failed for thread 11 of 40: Resource temporarily unavailable
> OpenBLAS blas_thread_init: RLIMIT_NPROC 200 current, 256467 max
> [... repeated for threads 12-39 ...]
> ImportError: Error importing numpy: you should not try to import numpy from
>         its source directory [...]

### Prompt

> and can you be repssible this time its ben 2 days ia ma only runnign fiels which have ended into an error

### Prompt (srun schema check result)

> (ebnerd_gbdt) [sukhraj.singh@ada ~]$ srun -A research --partition=u22 --time=00:05:00 --cpus-per-task=2 --mem=4G bash -c "source ~/venvs/ebnerd_gbdt/bin/activate && python ~/check_testset_schema.py"
> srun: Changing TimeLimit to 6 hours for interactive jobs
> srun: job 2681503 queued and waiting for resources
> srun: job 2681503 has been allocated resources
> member: ['ebnerd_testset/test/behaviors.parquet']
>   impression_id: uint32
>   impression_time: timestamp[us]
>   read_time: float
>   scroll_percentage: float
>   device_type: int8
>   article_ids_inview: large_list<item: int32>
>   user_id: uint32
>   is_sso_user: bool
>   gender: int8
>   postcode: int8
>   age: int8
>   is_subscriber: bool
>   session_id: uint32
>   is_beyond_accuracy: bool
> (ebnerd_gbdt) [sukhraj.singh@ada ~]$

### Prompt

> --------------------------------------------------------------------------------
> Disk quotas for user sukhraj.singh (uid 4127):
>      Filesystem   space   quota   limit   grace   files   quota   limit   grace
>           /home  16598M  25600M  26624M           74899    300k    305k
>         /share1      4K  25000M  27000M               1    3000    3200
> ---------------------------------------------------------------------------------
> [sukhraj.singh@ada ~]$ tail -f ~/assignment-1-news-retrieval/ebnerd_gbdt_k_train_2680914.out
>       subsampled train to 4,000,000 impressions (seed 0)
> [2/6] train: 4,000,000 impressions -> 44,366,367 rows x 65 feats (11.54 GB, 2652.5s)
> [3/6] early-stopping holdout = last 24h of train: 556,994/4,000,000 impressions (cutoff 2023-05-24 06:59:59) — validation split never used for stopping
> [LightGBM] [Warning] Met negative value in categorical features, will convert it to NaN
> [4/6] K_rank (lambdarank, 65 feats): best_iter=340 (1692s)
> [LightGBM] [Warning] Met negative value in categorical features, will convert it to NaN
> [4/6] K_cls (binary, 65 feats): best_iter=927 (3500s)
> [LightGBM] [Warning] Met negative value in categorical features, will convert it to NaN
> /var/spool/slurmd/job2680914/slurm_script: line 147: 1149071 Killed                  python scripts/run_ebnerd_gbdt_experiment.py --bundle large --zip-path "$RAW_DIR/ebnerd_large.zip" --embeddings-npy "$EMB_DIR/$MINI.npy" --embeddings-json "$EMB_DIR/$MINI.json" --train-sample-impressions 4000000 --score-chunk-size 250000 --full-metric-impressions 1000000 --out "$MODEL_DIR"
> slurmstepd: error: Detected 1 oom-kill event(s) in StepId=2680914.batch. Some of your processes may have been killed by the cgroup out-of-memory handler.

(This OOM trace was pasted twice across the session at two points; both
instances refer to the same real job 2680914. Logged once here per its actual
occurrence; both prompts are recorded above for completeness since each was a
distinct message from the engineer.)

### Prompt

> why do even need t put a time imit i ahve fuckign 4 days batch days task limit

### Prompt

> havent toched atyign tell me what to run now

### Prompt (final successful score-only job, no time-limit kill)

> [5/5] wrote 13,536,710 lines over 205,925,868 candidate rows in 124.3 min
>       /home2/sukhraj.singh/ebnerd_gbdt_k_results/submission/predictions.txt  (670 MB)
>       /home2/sukhraj.singh/ebnerd_gbdt_k_results/submission/prediction.zip  (219 MB)  <- upload this
>
> === DONE ===
> Validation metrics (from training) : /home2/sukhraj.singh/ebnerd_gbdt_k_results/candidate_k_gbdt_ebnerd_large/results.json
> Upload to Codabench                : /home2/sukhraj.singh/ebnerd_gbdt_k_results/submission/prediction.zip
> total 890M
> -rw-r--r-- 1 sukhraj.singh research 220M Aug 29 20:57 prediction.zip
> -rw-r--r-- 1 sukhraj.singh research 671M Aug 29 20:55 predictions.txt

### Prompt

> pulled

### Prompt (real leaderboard result, two screenshots attached)

> atlast got a good result the screenshots are on the desktop they will be latest ones as i jsut took them place them correclty and see the reuslts i clicked two screen shots so check the latest 2

[Two screenshots: Codabench submission list showing ID 907863 / prediction.zip
/ 2026-08-29 23:01 / Finished / Score 0.7542, and a "Ranking Metrics: Grouped
by Selected Dates" detail table, mean AUC 0.7535 over 2023-06-01..06-08.]

### Prompt

> same upload jsut this one finshed ealry tht sit commit this session adn log evey detial nbeeded as we have to do after every session i need ot clsoe this

---

## Session-continuation AI-vs-human attribution

| Artifact | Status |
|---|---|
| `src/retrieval/ebnerd_features.py` short-term per-impression fix (`UserHistoryRaw`, `compute_short_term_features`) | AI-written; the underlying question ("are we using recency right / not gaming the system") was asked by the engineer, the code fix and its verification (adversarial test, real-data bit-identity check) are AI work |
| `context_category_match`/`context_topic_overlap`/`context_embed_sim`/`session_position`/`session_start_gap_h` features | AI-found via dataset re-audit (engineer asked to "check the dataset again"), AI-implemented |
| `load_test_behaviors` missing-`article_id` fix, relocation to avoid the lightgbm/torch segfault | AI-found (from a real Ada traceback) and AI-fixed |
| The `tr["X"]`/`fit_X`/`stop_X` OOM fix | AI-found (from a real SLURM oom-kill event) and AI-fixed |
| `ebnerd_gbdt_ada_prepare.sbatch` / `_train.sbatch` / `_score_only.sbatch`, the `$HOME` shared-cache pattern, thread-count caps, the `#SBATCH`-ordering self-catch | AI-written, iterated in direct response to real job failures the engineer reported |
| All `--time`/`--mem-per-cpu` resource values | AI-computed from measured job output where possible (score-only job's `--time`), then AI-corrected to a generous proven-safe default after the engineer pointed out the real constraint (4-day project budget, not a tight per-job budget) was being mis-modelled |
| ADR-013's three addenda (2026-08-27, 2026-08-29, 2026-08-30) | AI-written from real measured results at each stage |
| Real Codabench result (submission 907863, Score 0.7542) | Real leaderboard outcome; uploaded by the engineer, not AI-executed |
| PROJECT_STATE.md final banner + Candidate K section | AI-written |
