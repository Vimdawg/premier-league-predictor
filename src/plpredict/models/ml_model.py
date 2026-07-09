"""LightGBM W/D/L classifier on walk-forward features.

Split protocol (chronological, never random — later matches must not inform
earlier predictions):
  train  : seasons up to 22/23
  stop   : 23/24 (early-stopping only)
  val    : 24/25 + 25/26 (untouched during training; used to pick the
           ensemble blend weight and report honest metrics)
The deployed model is refitted on all seasons at the best iteration.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from plpredict.config import PROCESSED_DIR
from plpredict.features import FEATURE_COLS

MODEL_PATH = PROCESSED_DIR / "lgbm_model.txt"

STOP_SEASONS = ["2324"]
VAL_SEASONS = ["2425", "2526"]

PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 60,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 27,
}


def _split(feat: pd.DataFrame):
    is_val = feat["season"].isin(VAL_SEASONS) & (feat["division"] == "E0")
    is_stop = feat["season"].isin(STOP_SEASONS)
    train = feat[~is_val & ~is_stop]
    stop = feat[is_stop]
    val = feat[is_val]
    return train, stop, val


def train_model(feat: pd.DataFrame) -> tuple[lgb.Booster, pd.DataFrame]:
    """Returns the deployed booster and the validation frame with ML
    probabilities attached (columns ml_H/ml_D/ml_A)."""
    train, stop, val = _split(feat)

    dtrain = lgb.Dataset(train[FEATURE_COLS], label=train["target"])
    dstop = lgb.Dataset(stop[FEATURE_COLS], label=stop["target"], reference=dtrain)
    booster = lgb.train(
        PARAMS, dtrain, num_boost_round=2000,
        valid_sets=[dstop], callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    best_iter = booster.best_iteration

    val = val.copy()
    probs = booster.predict(val[FEATURE_COLS], num_iteration=best_iter)
    val[["ml_H", "ml_D", "ml_A"]] = probs

    # Deployed model: refit on everything at the tuned iteration count.
    dall = lgb.Dataset(feat[FEATURE_COLS], label=feat["target"])
    final = lgb.train(PARAMS, dall, num_boost_round=best_iter)
    final.save_model(str(MODEL_PATH))
    return final, val


def load_model() -> lgb.Booster:
    return lgb.Booster(model_file=str(MODEL_PATH))


def predict_proba(booster: lgb.Booster, features: pd.DataFrame) -> np.ndarray:
    return booster.predict(features[FEATURE_COLS])
