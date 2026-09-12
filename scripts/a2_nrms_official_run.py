"""A2 Q3 runner (ADR-015, Option B): official-config NRMS on MIND or EB-NeRD.

One runner, two datasets, two arms. Model = `src/retrieval/nrms_official.py`;
input semantics = `src/retrieval/nrms_official_data.py`. The training
procedure per dataset follows that dataset's official code:

  MIND (recommenders nrms.yaml + MINDIterator):
    epochs 10, batch 32, Adam lr 1e-4, npratio 4; negatives RE-SAMPLED every
    epoch and impression order shuffled (load_data_from_file); no early
    stopping, no checkpoint selection. The final epoch is reported, and dev
    is scored each epoch for logging only.
  EB-NeRD (ebnerd-benchmark ebnerd_nrms.py + args_nrms.py):
    epochs 5, batch 32, Adam lr 1e-4, npratio 4, history 20; negatives sampled
    ONCE with replacement; the early-stopping set is the last day of the
    training data; the best val-AUC weights are reloaded at the end (the
    official ModelCheckpoint + load_weights); ReduceLROnPlateau(factor 0.2,
    Keras patience 2, min 1e-6); EarlyStopping(Keras patience 4).

Arms:
  control   = the official config.
  treatment = MIND: title + abstract news input (`--abstract-size`).
              EB-NeRD: + freshness late fusion (`use_freshness`).

Output (--out-dir): scores.parquet (impression_id, user_id, labels, scores),
one row per evaluated impression, plus results.json. Metrics and paired
bootstrap CIs are computed afterwards, locally, by the project's own
harness, never by this script. The AUC printed here is for monitoring.
"""
import argparse
import io
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrieval.nrms_official import OfficialNRMS, freshness_feature, official_loss  # noqa: E402
from src.retrieval.nrms_official_data import (  # noqa: E402
    build_mind_news_tokens,
    ebnerd_sample_negatives,
    left_pad_recent,
    mind_newsample,
)

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
ap.add_argument("--arm", choices=["control", "treatment"], required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=None, help="default: official (MIND 10, EB-NeRD 5)")
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--eval-batch", type=int, default=512)
ap.add_argument("--max-train-impressions", type=int, default=None, help="smoke runs only")
ap.add_argument("--max-eval-impressions", type=int, default=None, help="smoke runs only")
ap.add_argument("--require-gpu", action="store_true")
ap.add_argument("--log-every", type=int, default=5000, help="progress line every N train steps (0 = off)")
ap.add_argument("--es-batch", type=int, default=128,
                help="EB-NeRD early-stop scoring batch. Each sample carries history+candidate "
                     "articles, so attention memory is O(batch * articles * T^2): 1024 OOM'd "
                     "an 11GB 2080 Ti (job 2694505).")
# MIND
ap.add_argument("--mind-train-zip")
ap.add_argument("--mind-dev-zip")
ap.add_argument("--mind-utils-dir")
ap.add_argument("--abstract-size", type=int, default=None,
                help="MIND treatment only: abstract tokens appended to the 30 title tokens")
# EB-NeRD
ap.add_argument("--ebnerd-zip")
ap.add_argument("--ebnerd-tokens", help="npz from scripts/a2_ebnerd_prepare_tokens.py")
a = ap.parse_args()

if a.dataset == "mind" and a.arm == "treatment" and not a.abstract_size:
    sys.exit("MIND treatment requires --abstract-size (ADR-015: title+abstract input)")

random.seed(a.seed)
np.random.seed(a.seed)
torch.manual_seed(a.seed)
torch.cuda.manual_seed_all(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if a.require_gpu and dev.type != "cuda":
    sys.exit("FATAL: --require-gpu set but torch sees no GPU")
OUT = Path(a.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
log = {"args": vars(a), "device": str(dev), "torch": torch.__version__, "epochs": []}
print(f"device={dev} torch={torch.__version__}", flush=True)


def systematic(n: int, k: int | None) -> np.ndarray:
    """Every (n // k)-th index, never a prefix (EB-NeRD clusters rows)."""
    if not k or k >= n:
        return np.arange(n)
    return np.arange(0, n, n // k)[:k]


# ======================================================================== MIND
def read_zip_tsv(zpath: str, name: str) -> list[list[str]]:
    """Split on tab exactly as MINDIterator does (`line.strip("\\n").split(col_spliter)`)."""
    z = zipfile.ZipFile(zpath)
    member = [m for m in z.namelist() if m.endswith("/" + name) or m == name][0]
    with z.open(member) as f:
        return [ln.decode("utf-8").strip("\n").split("\t") for ln in f]


def load_mind_split(zpath: str, word_dict, title_size: int, abstract_size: int):
    news = read_zip_tsv(zpath, "news.tsv")
    nid2index, tokens = build_mind_news_tokens(
        [r[0] for r in news], [r[3] for r in news], [r[4] for r in news],
        word_dict, title_size, abstract_size)
    beh = read_zip_tsv(zpath, "behaviors.tsv")
    imps = []
    for r in beh:
        hist = [nid2index.get(n, 0) for n in r[3].split()] if r[3] else []
        cands, labels = [], []
        for tok in r[4].split():
            nid, lab = tok.split("-")
            cands.append(nid2index.get(nid, 0))
            labels.append(int(lab))
        imps.append((int(r[0]), r[1], hist, cands, labels))
    return tokens, imps


def mind_epoch_examples(imps, his_size: int, npratio: int, rng: random.Random):
    """MINDIterator.load_data_from_file with npratio > 0: shuffled impression
    order; one sample per clicked candidate = [pos] + newsample(negs)."""
    order = list(range(len(imps)))
    rng.shuffle(order)
    hist, cand = [], []
    for i in order:
        _, _, h, cs, ls = imps[i]
        negs = [c for c, lab in zip(cs, ls) if lab == 0]
        hp = left_pad_recent(h, his_size)
        for c, lab in zip(cs, ls):
            if lab == 1:
                hist.append(hp)
                cand.append([c] + mind_newsample(negs, npratio, rng))
    return np.asarray(hist, dtype=np.int64), np.asarray(cand, dtype=np.int64)


# ===================================================================== EB-NeRD
def read_zip_parquet(zpath: str, member: str, columns: list[str]) -> pd.DataFrame:
    z = zipfile.ZipFile(zpath)
    return pq.read_table(io.BytesIO(z.read(member)), columns=columns).to_pandas()


def load_ebnerd_split(zpath: str, split: str, aid2row: dict, unknown_row: int, his_size: int):
    """ebnerd_from_path: behaviors joined to the user's history, keeping the
    most recent `his_size` history ids, left-padded with id 0. Id 0 is not an
    article, so it maps to the "zeros" unknown representation, like any
    other id missing from the article dict."""
    beh = read_zip_parquet(zpath, f"{split}/behaviors.parquet",
                           ["impression_id", "impression_time", "article_ids_inview",
                            "article_ids_clicked", "user_id"])
    hist = read_zip_parquet(zpath, f"{split}/history.parquet", ["user_id", "article_id_fixed"])
    hist_map = dict(zip(hist["user_id"], hist["article_id_fixed"]))
    row = lambda aid: aid2row.get(int(aid), unknown_row)  # noqa: E731
    imps = []
    for r in beh.itertuples(index=False):
        h = left_pad_recent([int(x) for x in hist_map.get(r.user_id, [])], his_size, pad=0)
        inview = [int(x) for x in r.article_ids_inview]
        clicked = set(int(x) for x in r.article_ids_clicked)
        imps.append({
            "impression_id": int(r.impression_id), "user_id": int(r.user_id),
            "t": pd.Timestamp(r.impression_time).timestamp(),
            "hist": [row(x) for x in h],
            "cands": [row(x) for x in inview],
            "labels": [int(x in clicked) for x in inview],
            "clicked_rows": [row(x) for x in inview if x in clicked],
            "neg_rows": [row(x) for x in inview if x not in clicked],
        })
    return imps


# ====================================================================== common
def evaluate(model, news_tokens_t, imps_eval, his_size, pub_s=None, use_fresh=False):
    """Fast eval, as recommenders run_fast_eval does: encode every news item
    once, then per impression, user vector . candidate vectors."""
    model.eval()
    with torch.no_grad():
        # Width-aware chunk: self-attention cost per row is O(T^2), so the
        # MIND treatment's 80 tokens (title 30 + abstract 50) needs ~7x less
        # rows per chunk than the control's 30 to hold the same memory. A flat
        # 2048 would allocate ~1GB of attention scores per chunk at T=80 on an
        # 11GB card that job 2694505 already OOM'd once.
        t_width = news_tokens_t.shape[1]
        chunk = max(128, int(2048 * (30.0 / t_width) ** 2))
        vecs = torch.cat([model.encode_news(news_tokens_t[i:i + chunk])
                          for i in range(0, news_tokens_t.shape[0], chunk)])
        scores_all = []
        for s in range(0, len(imps_eval), a.eval_batch):
            chunk = imps_eval[s:s + a.eval_batch]
            h = torch.as_tensor(np.asarray([c["hist"] for c in chunk]), device=dev)
            users = model.user_encoder(vecs[h])  # (b, d)
            for j, c in enumerate(chunk):
                cand = torch.as_tensor(c["cands"], device=dev)
                sc = vecs[cand] @ users[j]
                if use_fresh:
                    age = torch.as_tensor(freshness_feature(np.full(len(c["cands"]), c["t"]),
                                                            pub_s[c["cands"]]), device=dev)
                    sc = sc + model.freshness_w * age
                scores_all.append(sc.float().cpu().numpy())
    model.train()
    return scores_all


def mean_auc(scores, imps_eval) -> float:
    aucs = [roc_auc_score(c["labels"], s) for s, c in zip(scores, imps_eval)
            if 0 < sum(c["labels"]) < len(c["labels"])]
    return float(np.mean(aucs)) if aucs else float("nan")


def train_batches(model, opt, hist, cand, news_tokens_t, age=None):
    """One epoch. Logs progress every `--log-every` steps, because a MIND epoch
    is ~106k steps and epoch-end logging alone gave no ETA for the first
    long job (2694501)."""
    model.train()
    perm = np.random.permutation(len(hist))
    tot, n = 0.0, 0
    n_steps = (len(perm) + a.batch_size - 1) // a.batch_size
    t_ep = time.time()
    for s in range(0, len(perm), a.batch_size):
        if a.log_every and n and n % a.log_every == 0:
            el = time.time() - t_ep
            print(f"  step {n:,}/{n_steps:,} loss {tot / n:.4f} "
                  f"{n * a.batch_size / el:,.0f} samples/s, epoch ETA {el / n * (n_steps - n) / 60:.1f} min",
                  flush=True)
        idx = perm[s:s + a.batch_size]
        h = news_tokens_t[torch.as_tensor(hist[idx], device=dev)]
        c = news_tokens_t[torch.as_tensor(cand[idx], device=dev)]
        ag = torch.as_tensor(age[idx], device=dev) if age is not None else None
        loss = official_loss(model(h, c, ag))
        opt.zero_grad()
        loss.backward()  # no gradient clipping: the official Keras optimiser has none
        opt.step()
        tot += loss.item()
        n += 1
    return tot / max(n, 1)


# ======================================================================== main
if a.dataset == "mind":
    import pickle
    U = Path(a.mind_utils_dir)
    word_dict = pickle.load(open(U / "word_dict.pkl", "rb"))
    embedding = np.load(U / "embedding.npy").astype(np.float32)
    TITLE, HIS, NP = 30, 50, 4  # nrms.yaml
    ABS = a.abstract_size if a.arm == "treatment" else 0
    EPOCHS = a.epochs or 10     # nrms.yaml train.epochs
    t = time.time()
    tr_tokens, tr_imps = load_mind_split(a.mind_train_zip, word_dict, TITLE, ABS)
    dv_tokens, dv_imps = load_mind_split(a.mind_dev_zip, word_dict, TITLE, ABS)
    tr_imps = [tr_imps[i] for i in systematic(len(tr_imps), a.max_train_impressions)]
    dv_imps = [dv_imps[i] for i in systematic(len(dv_imps), a.max_eval_impressions)]
    dv_eval = [{"impression_id": i, "user_id": u, "hist": left_pad_recent(h, HIS),
                "cands": cs, "labels": ls, "t": 0.0} for i, u, h, cs, ls in dv_imps]
    log["load_s"] = time.time() - t
    log["n_train_impressions"], log["n_eval_impressions"] = len(tr_imps), len(dv_eval)
    print(f"MIND loaded in {log['load_s']:.0f}s: train {len(tr_imps):,} dev {len(dv_eval):,} "
          f"width {TITLE + ABS}", flush=True)

    model = OfficialNRMS(embedding).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    tr_tok_t = torch.as_tensor(tr_tokens, device=dev, dtype=torch.long)
    dv_tok_t = torch.as_tensor(dv_tokens, device=dev, dtype=torch.long)
    rng = random.Random(a.seed)
    for ep in range(1, EPOCHS + 1):
        t = time.time()
        hist, cand = mind_epoch_examples(tr_imps, HIS, NP, rng)  # fresh negatives each epoch
        loss = train_batches(model, opt, hist, cand, tr_tok_t)
        train_s = time.time() - t
        scores = evaluate(model, dv_tok_t, dv_eval, HIS)
        rec = {"epoch": ep, "loss": loss, "n_examples": int(len(hist)), "train_s": train_s,
               "dev_mean_auc_monitor": mean_auc(scores, dv_eval)}
        log["epochs"].append(rec)
        print(json.dumps(rec), flush=True)
    final_scores, eval_rows = scores, dv_eval  # final epoch, no selection (official fit)

else:
    P = np.load(a.ebnerd_tokens)
    aid2row = {int(x): i for i, x in enumerate(P["article_ids"])}
    unknown_row = len(P["article_ids"])  # last tokens row = official "zeros" representation
    pub_s = np.append(P["published_time_s"], np.nan)  # unknown article -> age 0
    HIS, NP = 20, 4             # args_nrms.py
    EPOCHS = a.epochs or 5      # args_nrms.py
    use_fresh = a.arm == "treatment"
    t = time.time()
    tr = load_ebnerd_split(a.ebnerd_zip, "train", aid2row, unknown_row, HIS)
    va = load_ebnerd_split(a.ebnerd_zip, "validation", aid2row, unknown_row, HIS)
    tr = [tr[i] for i in systematic(len(tr), a.max_train_impressions)]
    va = [va[i] for i in systematic(len(va), a.max_eval_impressions)]
    log["load_s"] = time.time() - t

    # sampling_strategy_wu2019, once, with replacement; one sample per click.
    nrng = np.random.default_rng(a.seed)
    samples = []
    for imp in tr:
        for pos in imp["clicked_rows"]:
            negs = ebnerd_sample_negatives(imp["neg_rows"], NP, nrng)
            if negs is not None:
                samples.append((imp["t"], imp["hist"], [pos] + negs))
    ts = np.asarray([s[0] for s in samples])
    days = pd.to_datetime(ts, unit="s").normalize()
    last_dt = days.max() - pd.Timedelta(days=1)  # official: last day of training data
    is_es = np.asarray(days >= last_dt)  # DatetimeIndex comparison already yields ndarray
    hist_all = np.asarray([s[1] for s in samples], dtype=np.int64)
    cand_all = np.asarray([s[2] for s in samples], dtype=np.int64)
    age_all = freshness_feature(np.repeat(ts[:, None], NP + 1, 1), pub_s[cand_all])
    # Guard (ADR-015, 2026-09-12): a units bug once made every age ~53 years,
    # so within-impression age differences vanished under log1p and the
    # freshness arm became a silent no-op. Fail before spending GPU time.
    raw_age_h = (ts[:, None] - pub_s[cand_all]) / 3600.0
    med_age_h = float(np.nanmedian(raw_age_h))
    varies = float(np.mean(age_all.std(axis=1) > 1e-3))
    log["age_check"] = {"median_candidate_age_h": med_age_h, "frac_samples_age_varies": varies}
    print(f"age check: median candidate age {med_age_h:.1f}h, "
          f"{varies:.1%} of samples have varying log-age", flush=True)
    if not (0.0 <= med_age_h <= 24 * 365):
        sys.exit(f"FATAL: median candidate age {med_age_h:.1f}h is implausible (units bug?)")
    if use_fresh and varies < 0.5:
        sys.exit(f"FATAL: log-age varies within only {varies:.1%} of samples; "
                 "freshness would be a near-constant shift (softmax no-op)")
    log.update(n_train_samples=int((~is_es).sum()), n_early_stop_samples=int(is_es.sum()),
               n_eval_impressions=len(va), n_train_impressions=len(tr))
    print(f"EB-NeRD loaded in {log['load_s']:.0f}s: train samples {int((~is_es).sum()):,} "
          f"early-stop samples {int(is_es.sum()):,} val impressions {len(va):,}", flush=True)

    model = OfficialNRMS(P["embedding"], use_freshness=use_fresh).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    tok_t = torch.as_tensor(P["tokens"], device=dev, dtype=torch.long)
    best_auc, best_state, wait_es, wait_lr = -1.0, None, 0, 0
    for ep in range(1, EPOCHS + 1):
        t = time.time()
        loss = train_batches(model, opt, hist_all[~is_es], cand_all[~is_es], tok_t,
                             age_all[~is_es] if use_fresh else None)
        # val_auc as Keras computes it: over the flattened softmax outputs of
        # the sampled early-stop set. sklearn's exact AUC stands in for
        # Keras's 200-threshold approximation; it is used only for
        # model/LR selection, not for any reported number.
        model.eval()
        with torch.no_grad():
            # --es-batch, not 1024: each sample carries 20 history + 5 candidate
            # articles, so B=1024 pushes 25,600 rows through the news encoder at
            # once and its attention scores alone are B*25*20heads*30*30*4B ~
            # 1.8GB. That OOM'd job 2694505 on an 11GB 2080 Ti in epoch 2 (epoch
            # 1 fit, then fragmentation). 128 keeps it ~0.23GB.
            probs = []
            es_idx = np.flatnonzero(is_es)
            for s in range(0, len(es_idx), a.es_batch):
                sl = es_idx[s:s + a.es_batch]
                lg = model(tok_t[torch.as_tensor(hist_all[sl], device=dev)],
                           tok_t[torch.as_tensor(cand_all[sl], device=dev)],
                           torch.as_tensor(age_all[sl], device=dev) if use_fresh else None)
                probs.append(torch.softmax(lg, -1).cpu().numpy())
                del lg
        probs = np.concatenate(probs)
        y = np.zeros_like(probs)
        y[:, 0] = 1
        val_auc = float(roc_auc_score(y.ravel(), probs.ravel()))
        rec = {"epoch": ep, "loss": loss, "val_auc": val_auc, "lr": opt.param_groups[0]["lr"],
               "train_s": time.time() - t,
               "freshness_w": model.freshness_w.item() if use_fresh else None}
        if val_auc > best_auc:
            best_auc, best_state = val_auc, {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait_es = wait_lr = 0
        else:
            wait_es += 1
            wait_lr += 1
            if wait_lr >= 2:  # Keras ReduceLROnPlateau(patience=2, factor=0.2, min_lr=1e-6)
                for g in opt.param_groups:
                    g["lr"] = max(g["lr"] * 0.2, 1e-6)
                wait_lr = 0
        log["epochs"].append(rec)
        print(json.dumps(rec), flush=True)
        if wait_es >= 4:  # Keras EarlyStopping(patience=4)
            break
    model.load_state_dict(best_state)  # official: ModelCheckpoint(save_best_only) + load_weights
    log["best_early_stop_val_auc"] = best_auc
    final_scores = evaluate(model, tok_t, va, HIS, pub_s=pub_s, use_fresh=use_fresh)
    eval_rows = va

# ===================================================================== outputs
pd.DataFrame({
    "impression_id": [r["impression_id"] for r in eval_rows],
    "user_id": [r["user_id"] for r in eval_rows],
    "labels": [list(map(int, r["labels"])) for r in eval_rows],
    "scores": [s.tolist() for s in final_scores],
}).to_parquet(OUT / "scores.parquet", index=False)
log["final_mean_auc_monitor"] = mean_auc(final_scores, eval_rows)
log["n_params"] = int(sum(p.numel() for p in model.parameters()))
log["total_s"] = time.time() - T0
if model.use_freshness:
    log["final_freshness_w"] = model.freshness_w.item()
(OUT / "results.json").write_text(json.dumps(log, indent=1, default=str))
print(f"DONE monitor AUC {log['final_mean_auc_monitor']:.4f} in {log['total_s']:.0f}s", flush=True)
