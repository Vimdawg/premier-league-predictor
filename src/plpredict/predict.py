"""Per-fixture predictions for the target season.

For every remaining fixture this produces the blended W/D/L probabilities and
a score matrix whose outcome masses are rescaled to match the blend (so the
scoreline detail comes from Dixon-Coles while the win/draw/loss balance
reflects the ensemble).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plpredict.config import PROCESSED_DIR
from plpredict.features import FEATURE_COLS, final_state
from plpredict.models import dixon_coles as dc
from plpredict.models import ensemble, ml_model

PREDICTIONS_PARQUET = PROCESSED_DIR / "match_predictions.parquet"
SCORE_MATRICES_NPZ = PROCESSED_DIR / "score_matrices.npz"


def _rescale_matrix(m: np.ndarray, target: tuple[float, float, float]) -> np.ndarray:
    """Scale the home-win/draw/away-win regions of a score matrix to hit the
    target outcome probabilities."""
    tri_l = np.tril(np.ones_like(m), -1)
    diag = np.eye(m.shape[0])
    tri_u = np.triu(np.ones_like(m), 1)
    out = m.copy()
    for region, t in zip((tri_l, diag, tri_u), target):
        mass = (m * region).sum()
        if mass > 1e-12:
            out += (t / mass - 1) * m * region
    return out / out.sum()


def predict_remaining(
    matches: pd.DataFrame, fixtures: pd.DataFrame
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Blended predictions for every unplayed fixture of the target season.

    Returns (predictions frame, {fixture_index: rescaled score matrix}).
    """
    model = dc.fit(matches)
    model.save()
    booster = ml_model.load_model()
    weight_dc = ensemble.load_blend()["weight_dc"]
    state = final_state(matches)

    remaining = fixtures[~fixtures["finished"]].reset_index(drop=True)

    feat_rows = []
    for _, fx in remaining.iterrows():
        date = pd.Timestamp(fx["kickoff_time"]).tz_localize(None)
        feat_rows.append(state.features_for(fx["home"], fx["away"], date))
    feats = pd.DataFrame(feat_rows)

    p_ml = booster.predict(feats[FEATURE_COLS])

    rows, matrices = [], {}
    for i, fx in remaining.iterrows():
        home, away = fx["home"], fx["away"]
        m = model.score_matrix(home, away)
        p_dc = np.array(model.outcome_probs(home, away))
        blend = ensemble.blend_probs(p_dc, p_ml[i], weight_dc)
        blend = blend / blend.sum()
        m = _rescale_matrix(m, tuple(blend))
        matrices[i] = m

        best = np.unravel_index(np.argmax(m), m.shape)
        lam, mu = model.goal_rates(home, away)
        rows.append({
            "fixture_idx": i,
            "gameweek": fx["gameweek"],
            "kickoff_time": fx["kickoff_time"],
            "home": home, "away": away,
            "p_home": blend[0], "p_draw": blend[1], "p_away": blend[2],
            "xg_home": lam, "xg_away": mu,
            "likely_score": f"{best[0]}-{best[1]}",
        })

    preds = pd.DataFrame(rows)
    preds.to_parquet(PREDICTIONS_PARQUET, index=False)
    np.savez_compressed(
        SCORE_MATRICES_NPZ, **{str(k): v for k, v in matrices.items()}
    )
    return preds, matrices


def load_predictions() -> pd.DataFrame:
    return pd.read_parquet(PREDICTIONS_PARQUET)


def load_score_matrices() -> dict[int, np.ndarray]:
    data = np.load(SCORE_MATRICES_NPZ)
    return {int(k): data[k] for k in data.files}
