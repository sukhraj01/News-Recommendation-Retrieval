"""A2 Q3 control, EB-NeRD: the official ebnerd-benchmark NRMS, run as-is (ADR-015).

Follows `examples/reproducibility_scripts/ebnerd_nrms.py` (ebnerd-benchmark,
commit recorded in env_provenance.txt) line for line for the model, hparams,
xlm-roberta-large word embeddings, text columns, and training sampling
(`sampling_strategy_wu2019`, npratio 4). Hyperparameters are NOT retyped here:
they come from the official `args_nrms.get_args()` parser, so its defaults
are the defaults.

The only departures are ADR-015's adaptations, required for a paired
bootstrap against a treatment:
  1. train on `<datasplit>/train` only. The official script concatenates
     train+validation because it targets the unlabeled test set.
  2. score the labeled `<datasplit>/validation` split (not the test set), and
     dump per-impression scores + labels, so our own harness can bootstrap.
The official "last day of training data" early-stopping split is kept.

Usage (on Ada, inside the official_nrms venv):
  python scripts/a2_ebnerd_official_nrms.py \
      --official-examples $ROOT/src/ebnerd-benchmark/examples/reproducibility_scripts \
      --out-dir $ROOT/results/ebnerd_nrms_control \
      -- --data_path $ROOT/data/ebnerd --datasplit ebnerd_small
Everything after `--` goes to the official parser unchanged.
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

ours_ap = argparse.ArgumentParser()
ours_ap.add_argument("--official-examples", required=True)
ours_ap.add_argument("--out-dir", required=True)
ours_ap.add_argument("--val-chunks", type=int, default=20)
ours_ap.add_argument(
    "--max-val-impressions", type=int, default=None,
    help="Smoke runs only: systematic sample of validation impressions.",
)
if "--" in sys.argv:
    split = sys.argv.index("--")
    ours = ours_ap.parse_args(sys.argv[1:split])
    official_argv = sys.argv[split + 1:]
else:
    ours, official_argv = ours_ap.parse_known_args()

sys.path.insert(0, ours.official_examples)
sys.argv = [sys.argv[0]] + official_argv
from args_nrms import get_args  # noqa: E402  (official parser, official defaults)

args = get_args()

import polars as pl  # noqa: E402
import tensorflow as tf  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from ebrec.evaluation import AucScore, MetricEvaluator, MrrScore, NdcgScore  # noqa: E402
from ebrec.models.newsrec import NRMSModel  # noqa: E402
from ebrec.models.newsrec.dataloader import NRMSDataLoader, NRMSDataLoaderPretransform  # noqa: E402
from ebrec.models.newsrec.model_config import hparams_nrms, hparams_to_dict  # noqa: E402
from ebrec.utils._articles import (  # noqa: E402
    convert_text2encoding_with_transformers,
    create_article_id_to_value_mapping,
)
from ebrec.utils._behaviors import (  # noqa: E402
    add_prediction_scores,
    create_binary_labels_column,
    ebnerd_from_path,
    sampling_strategy_wu2019,
)
from ebrec.utils._constants import (  # noqa: E402
    DEFAULT_BODY_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
)
from ebrec.utils._nlp import get_transformers_word_embeddings  # noqa: E402
from ebrec.utils._polars import concat_str_columns, split_df_chunks  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"
OUT = Path(ours.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus, flush=True)
if not gpus:
    sys.exit("FATAL: TensorFlow sees no GPU; refusing to train the control on CPU silently.")

t_start = time.time()
PATH = Path(args.data_path).expanduser()
DATASPLIT = args.datasplit
HISTORY_SIZE = args.history_size
TRAIN_FRACTION = args.train_fraction if not args.debug else 0.0001

# ---- hparams: identical assignments to the official script -----------------
hparams = hparams_nrms
hparams.title_size = args.max_title_length
hparams.history_size = args.history_size
hparams.head_num = args.head_num
hparams.head_dim = args.head_dim
hparams.attention_hidden_dim = args.attention_hidden_dim
hparams.optimizer = args.optimizer
hparams.loss = args.loss
hparams.dropout = args.dropout
hparams.learning_rate = args.learning_rate
hparams.newsencoder_units_per_layer = None

# ---- articles + xlm-roberta word embeddings (official) ----------------------
df_articles = pl.read_parquet(PATH.joinpath("articles.parquet"))
transformer_model = AutoModel.from_pretrained(args.transformer_model_name)
transformer_tokenizer = AutoTokenizer.from_pretrained(args.transformer_model_name)
word2vec_embedding = get_transformers_word_embeddings(transformer_model)
df_articles, cat_cal = concat_str_columns(
    df_articles, columns=[DEFAULT_TITLE_COL, DEFAULT_SUBTITLE_COL, DEFAULT_BODY_COL]
)
df_articles, token_col_title = convert_text2encoding_with_transformers(
    df_articles, transformer_tokenizer, cat_cal, max_length=args.max_title_length
)
article_mapping = create_article_id_to_value_mapping(df=df_articles, value_col=token_col_title)
del transformer_model
gc.collect()

COLUMNS = [
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_USER_COL,
]

# ---- training data: ADR-015 adaptation 1 (train split only) ----------------
df = (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "train"), history_size=HISTORY_SIZE, padding=0)
    .sample(fraction=TRAIN_FRACTION, shuffle=True, seed=args.seed)
    .select(COLUMNS)
    .pipe(sampling_strategy_wu2019, npratio=args.npratio, shuffle=True,
          with_replacement=True, seed=args.seed)
    .pipe(create_binary_labels_column)
)
import datetime as dt  # noqa: E402

last_dt = df[DEFAULT_IMPRESSION_TIMESTAMP_COL].dt.date().max() - dt.timedelta(days=1)
df_train = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() < last_dt)
df_es = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() >= last_dt)
print(f"train rows {df_train.height:,} | early-stop rows {df_es.height:,}", flush=True)

Loader = NRMSDataLoaderPretransform if args.nrms_loader == "NRMSDataLoaderPretransform" else NRMSDataLoader
common = dict(article_dict=article_mapping, unknown_representation="zeros",
              history_column=DEFAULT_HISTORY_ARTICLE_ID_COL, batch_size=args.bs_train)
train_dl = Loader(behaviors=df_train, eval_mode=False, **common)
es_dl = Loader(behaviors=df_es, eval_mode=False, **common)

weights_path = OUT / "weights"
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(filepath=str(weights_path), monitor="val_auc", mode="max",
                                       save_best_only=True, save_weights_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.2, patience=2, min_lr=1e-6),
]
model = NRMSModel(hparams=hparams, word2vec_embedding=word2vec_embedding, seed=42)
model.model.compile(optimizer=model.model.optimizer, loss=model.model.loss, metrics=["AUC"])
t_fit = time.time()
hist = model.model.fit(train_dl, validation_data=es_dl, epochs=args.epochs, callbacks=callbacks)
fit_s = time.time() - t_fit
model.model.load_weights(str(weights_path))

# ---- ADR-015 adaptation 2: score the labeled validation split ---------------
df_val = (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "validation"), history_size=HISTORY_SIZE, padding=0)
    .select(COLUMNS)
    .pipe(create_binary_labels_column, shuffle=False)  # keep in-view order
)
if ours.max_val_impressions:
    step = max(1, df_val.height // ours.max_val_impressions)
    df_val = df_val.gather_every(step).head(ours.max_val_impressions)  # systematic, never a prefix

t_score = time.time()
scored = []
for i, chunk in enumerate(split_df_chunks(df_val, n_chunks=ours.val_chunks), start=1):
    dl = NRMSDataLoader(behaviors=chunk, eval_mode=True, **{**common, "batch_size": args.bs_test})
    scores = model.scorer.predict(dl)
    scored.append(add_prediction_scores(chunk, scores.tolist()))
    print(f"val chunk {i}/{ours.val_chunks} ({chunk.height:,} impressions)", flush=True)
    tf.keras.backend.clear_session()
    del dl, scores
    gc.collect()
score_s = time.time() - t_score
df_val = pl.concat(scored)
df_val.select(
    DEFAULT_IMPRESSION_ID_COL, DEFAULT_USER_COL, DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_LABELS_COL, "scores"
).write_parquet(OUT / "val_scores.parquet")

# Official evaluator — logged as a cross-check against our own harness.
ev = MetricEvaluator(
    labels=df_val[DEFAULT_LABELS_COL].to_list(),
    predictions=df_val["scores"].to_list(),
    metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
)
ev.evaluate()
results = {
    "run": "ebnerd_official_nrms_control",
    "official_metrics_validation": {k: float(v) for k, v in ev.evaluations.items()},
    "n_val_impressions": df_val.height,
    "n_train_rows": df_train.height,
    "n_early_stop_rows": df_es.height,
    "history": {k: [float(x) for x in v] for k, v in hist.history.items()},
    "timings_s": {"fit": fit_s, "score_val": score_s, "total": time.time() - t_start},
    "official_args": vars(args),
    "hparams": hparams_to_dict(hparams),
    "adaptations": ["train split only (not train+validation)", "score labeled validation, not test"],
}
(OUT / "results.json").write_text(json.dumps(results, indent=1, default=str))
print(json.dumps(results["official_metrics_validation"], indent=1))
