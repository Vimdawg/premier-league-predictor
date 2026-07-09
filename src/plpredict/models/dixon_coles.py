"""Time-decayed Dixon-Coles (1997) goal model.

Each team gets a log attack and defense rating; a shared home-advantage term
and the rho correction for low-scoring dependence complete the model. Matches
are weighted by exp(-xi * days_ago), so recent form dominates. PL and
Championship matches are fitted jointly — promoted/relegated teams moving
between divisions anchor the cross-division strength scale, which is how
newly promoted sides get a rating before playing a PL match.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from plpredict.config import PROCESSED_DIR

MODEL_PATH = PROCESSED_DIR / "dixon_coles.json"

# Decay: half-life of ~390 days (a season plus its summer).
DEFAULT_XI = np.log(2) / 390
MAX_GOALS = 10


@dataclass
class DixonColesModel:
    teams: list[str]
    attack: dict[str, float]
    defense: dict[str, float]
    home_adv: float
    rho: float
    base: float
    fitted_at: str = ""
    meta: dict = field(default_factory=dict)

    # ---------- persistence ----------
    def save(self, path: Path = MODEL_PATH) -> None:
        path.write_text(json.dumps(self.__dict__, indent=1, default=str))

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "DixonColesModel":
        return cls(**json.loads(path.read_text()))

    # ---------- prediction ----------
    def goal_rates(self, home: str, away: str) -> tuple[float, float]:
        lam = np.exp(self.base + self.home_adv + self.attack[home] + self.defense[away])
        mu = np.exp(self.base + self.attack[away] + self.defense[home])
        return float(lam), float(mu)

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        """P(home scores i, away scores j) for i, j in [0, MAX_GOALS]."""
        lam, mu = self.goal_rates(home, away)
        p_home = poisson.pmf(np.arange(MAX_GOALS + 1), lam)
        p_away = poisson.pmf(np.arange(MAX_GOALS + 1), mu)
        m = np.outer(p_home, p_away)
        # Dixon-Coles low-score correction.
        m[0, 0] *= 1 - lam * mu * self.rho
        m[0, 1] *= 1 + lam * self.rho
        m[1, 0] *= 1 + mu * self.rho
        m[1, 1] *= 1 - self.rho
        m = np.clip(m, 0, None)
        return m / m.sum()

    def outcome_probs(self, home: str, away: str) -> tuple[float, float, float]:
        """(P_home_win, P_draw, P_away_win)."""
        m = self.score_matrix(home, away)
        return (
            float(np.tril(m, -1).sum()),  # home rows > away cols
            float(np.trace(m)),
            float(np.triu(m, 1).sum()),
        )


def _tau_log(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """log of the Dixon-Coles correction factor per match."""
    tau = np.ones_like(lam)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1 - lam[m00] * mu[m00] * rho
    tau[m01] = 1 + lam[m01] * rho
    tau[m10] = 1 + mu[m10] * rho
    tau[m11] = 1 - rho
    return np.log(np.clip(tau, 1e-10, None))


def fit(
    matches: pd.DataFrame,
    xi: float = DEFAULT_XI,
    as_of: pd.Timestamp | None = None,
) -> DixonColesModel:
    """Fit by weighted MLE on columns Date/HomeTeam/AwayTeam/FTHG/FTAG.

    `as_of` limits training to matches strictly before that date (for
    backtesting); decay weights are measured from it.
    """
    df = matches.dropna(subset=["FTHG", "FTAG"]).copy()
    if as_of is not None:
        df = df[df["Date"] < as_of]
    else:
        as_of = df["Date"].max() + pd.Timedelta(days=1)

    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    h = df["HomeTeam"].map(idx).to_numpy()
    a = df["AwayTeam"].map(idx).to_numpy()
    x = df["FTHG"].to_numpy(dtype=float)
    y = df["FTAG"].to_numpy(dtype=float)
    days_ago = (as_of - df["Date"]).dt.days.to_numpy(dtype=float)
    w = np.exp(-xi * days_ago)

    # Parameter vector: attack (n), defense (n), home_adv, rho, base.
    p0 = np.concatenate([np.zeros(2 * n), [0.25, -0.05, np.log(x.mean() + 1e-9)]])

    def nll(p: np.ndarray) -> float:
        att = p[:n] - p[:n].mean()  # identifiability: attacks sum to 0
        dfn = p[n : 2 * n] - p[n : 2 * n].mean()
        home_adv, rho, base = p[-3], p[-2], p[-1]
        lam = np.exp(base + home_adv + att[h] + dfn[a])
        mu = np.exp(base + att[a] + dfn[h])
        ll = (
            x * np.log(lam) - lam + y * np.log(mu) - mu
            + _tau_log(x, y, lam, mu, rho)
        )
        return -float(np.sum(w * ll))

    res = minimize(nll, p0, method="L-BFGS-B", options={"maxiter": 500})
    p = res.x
    att = p[:n] - p[:n].mean()
    dfn = p[n : 2 * n] - p[n : 2 * n].mean()

    return DixonColesModel(
        teams=teams,
        attack={t: float(att[idx[t]]) for t in teams},
        defense={t: float(dfn[idx[t]]) for t in teams},
        home_adv=float(p[-3]),
        rho=float(p[-2]),
        base=float(p[-1]),
        fitted_at=str(as_of.date()),
        meta={"n_matches": int(len(df)), "xi": xi, "converged": bool(res.success)},
    )
