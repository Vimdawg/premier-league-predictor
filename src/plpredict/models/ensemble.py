"""Blend Dixon-Coles and LightGBM probabilities; backtest on 24/25 + 25/26.

The DC side of the validation predictions is produced by refitting the DC
model at the start of each month and predicting only that month's matches, so
every probability is out-of-sample. The blend weight is the log-loss
minimizer on the validation seasons, benchmarked against Bet365 closing odds
and a uniform baseline.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from plpredict.config import PROCESSED_DIR
from plpredict.models import dixon_coles as dc

ENSEMBLE_PATH = PROCESSED_DIR / "ensemble.json"


def log_loss(y: np.ndarray, probs: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1)
    return float(-np.mean(np.log(p)))


def dc_val_probs(matches: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample DC probabilities for every validation match."""
    val = val.copy().sort_values("Date")
    val["month"] = val["Date"].dt.to_period("M")
    out = []
    for month, grp in val.groupby("month", sort=True):
        model = dc.fit(matches, as_of=month.to_timestamp())
        for _, r in grp.iterrows():
            if r["HomeTeam"] in model.attack and r["AwayTeam"] in model.attack:
                ph, pd_, pa = model.outcome_probs(r["HomeTeam"], r["AwayTeam"])
            else:  # brand-new team with no history at all
                ph, pd_, pa = 1 / 3, 1 / 3, 1 / 3
            out.append({**r, "dc_H": ph, "dc_D": pd_, "dc_A": pa})
    return pd.DataFrame(out)


def fit_blend(val: pd.DataFrame) -> dict:
    """Pick blend weight w (p = w*DC + (1-w)*ML) minimizing validation
    log-loss; return weight plus benchmark metrics."""
    y = val["target"].to_numpy()
    p_dc = val[["dc_H", "dc_D", "dc_A"]].to_numpy()
    p_ml = val[["ml_H", "ml_D", "ml_A"]].to_numpy()

    weights = np.linspace(0, 1, 101)
    losses = [log_loss(y, w * p_dc + (1 - w) * p_ml) for w in weights]
    best_w = float(weights[int(np.argmin(losses))])

    metrics = {
        "weight_dc": best_w,
        "n_val": int(len(val)),
        "logloss_dc": log_loss(y, p_dc),
        "logloss_ml": log_loss(y, p_ml),
        "logloss_blend": float(np.min(losses)),
        "logloss_uniform": float(-np.log(1 / 3)),
    }

    odds = val[["B365H", "B365D", "B365A"]].to_numpy(dtype=float)
    ok = ~np.isnan(odds).any(axis=1)
    if ok.sum() > 0:
        implied = 1 / odds[ok]
        implied /= implied.sum(axis=1, keepdims=True)
        metrics["logloss_bookmaker"] = log_loss(y[ok], implied)
        blend = best_w * p_dc + (1 - best_w) * p_ml
        metrics["logloss_blend_on_odds_subset"] = log_loss(y[ok], blend[ok])

    ENSEMBLE_PATH.write_text(json.dumps(metrics, indent=1))
    return metrics


def load_blend() -> dict:
    return json.loads(ENSEMBLE_PATH.read_text())


def blend_probs(p_dc: np.ndarray, p_ml: np.ndarray, weight_dc: float) -> np.ndarray:
    return weight_dc * p_dc + (1 - weight_dc) * p_ml
