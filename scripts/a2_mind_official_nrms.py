"""A2 Q3 control, MIND: microsoft/recommenders' official NRMS, run as-is (ADR-015).

Follows `examples/00_quick_start/nrms_MIND.ipynb`: `prepare_hparams` over the
official `nrms.yaml` from `MINDlarge_utils.zip`, with its GloVe-initialised
`embedding.npy` / `word_dict.pkl` / `uid2index.pkl`, the notebook's
`epochs=5, batch_size=32`, and `NRMSModel(hparams, MINDIterator, seed)`.

The one addition, needed for a paired bootstrap (ADR-015), is a per-impression
dump of dev labels + scores. recommenders trains for a fixed number of epochs
with no early stopping and no checkpoint selection on dev, so the dev number
is not tuned on dev. (J's number was: it picked its best epoch on dev,
ADR-012.)

Usage (on Ada, inside the official_nrms venv):
  python scripts/a2_mind_official_nrms.py --data-dir $ROOT/data/mind \
      --out-dir $ROOT/results/mind_nrms_control [--smoke-lines 2000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--epochs", type=int, default=5)       # nrms_MIND.ipynb
ap.add_argument("--batch-size", type=int, default=32)  # nrms_MIND.ipynb
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--smoke-lines", type=int, default=None,
                help="Smoke runs only: truncate train/dev behaviors.tsv to N lines.")
a = ap.parse_args()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from recommenders.models.deeprec.deeprec_utils import cal_metric  # noqa: E402
from recommenders.models.newsrec.io.mind_iterator import MINDIterator  # noqa: E402
from recommenders.models.newsrec.models.nrms import NRMSModel  # noqa: E402
from recommenders.models.newsrec.newsrec_utils import prepare_hparams  # noqa: E402

gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus, flush=True)
if not gpus:
    sys.exit("FATAL: TensorFlow sees no GPU; refusing to train the control on CPU silently.")

D, OUT = Path(a.data_dir), Path(a.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
train_news, train_beh = D / "train/news.tsv", D / "train/behaviors.tsv"
dev_news, dev_beh = D / "dev/news.tsv", D / "dev/behaviors.tsv"
if a.smoke_lines:
    for name, src in (("train", train_beh), ("dev", dev_beh)):
        dst = OUT / f"smoke_{name}_behaviors.tsv"
        with open(src) as f, open(dst, "w") as g:
            for i, line in enumerate(f):
                if i >= a.smoke_lines:
                    break
                g.write(line)
    train_beh, dev_beh = OUT / "smoke_train_behaviors.tsv", OUT / "smoke_dev_behaviors.tsv"

U = D / "utils"
hparams = prepare_hparams(
    str(U / "nrms.yaml"),
    wordEmb_file=str(U / "embedding.npy"),
    wordDict_file=str(U / "word_dict.pkl"),
    userDict_file=str(U / "uid2index.pkl"),
    batch_size=a.batch_size,
    epochs=a.epochs,
    show_step=2000,
)
print(hparams, flush=True)
model = NRMSModel(hparams, MINDIterator, seed=a.seed)

t0 = time.time()
model.fit(str(train_news), str(train_beh), str(dev_news), str(dev_beh))
fit_s = time.time() - t0
model.model.save_weights(str(OUT / "nrms_official.weights.h5"))

t1 = time.time()
imp_idx, labels, preds = model.run_fast_eval(str(dev_news), str(dev_beh))
eval_s = time.time() - t1
official = cal_metric(labels, preds, hparams.metrics)

# Map MINDIterator's positional impression index back to behaviors.tsv's own
# impression ids, and check that every candidate list has the expected length.
beh = pd.read_csv(dev_beh, sep="\t", header=None, usecols=[0, 1, 4],
                  names=["impression_id", "user_id", "impressions"])
n_cand = beh["impressions"].str.count(" ").to_numpy() + 1
rows = []
for i, lab, pr in zip(imp_idx, labels, preds):
    assert len(lab) == len(pr) == n_cand[i], f"candidate count mismatch at impression index {i}"
    rows.append((int(beh.at[i, "impression_id"]), beh.at[i, "user_id"],
                 np.asarray(lab, dtype=np.int8).tolist(), np.asarray(pr, dtype=np.float32).tolist()))
pd.DataFrame(rows, columns=["impression_id", "user_id", "labels", "scores"]).to_parquet(
    OUT / "dev_scores.parquet", index=False)

results = {
    "run": "mind_official_nrms_control",
    "official_metrics_dev": {k: float(v) for k, v in official.items()},
    "n_dev_impressions": len(rows),
    "timings_s": {"fit": fit_s, "eval_dev": eval_s},
    "args": vars(a),
    "hparams": {k: (v if isinstance(v, (int, float, str, bool, list)) else str(v))
                for k, v in hparams.values().items()},
}
(OUT / "results.json").write_text(json.dumps(results, indent=1, default=str))
print(json.dumps(results["official_metrics_dev"], indent=1))
