"""Monte Carlo season simulation.

Plays the remaining fixtures N times by sampling scorelines from each
fixture's (blend-rescaled) score matrix, adds points already banked from
played matches, and tallies final-table distributions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plpredict.config import PROCESSED_DIR

N_SIMS = 10_000
RNG_SEED = 2627

SIM_TABLE_PARQUET = PROCESSED_DIR / "sim_table.parquet"
POSITION_MATRIX_PARQUET = PROCESSED_DIR / "position_matrix.parquet"
TEAM_GOALS_NPY = PROCESSED_DIR / "sim_team_goals.npy"
TEAM_ORDER_JSON = PROCESSED_DIR / "sim_team_order.json"


def simulate_season(
    fixtures: pd.DataFrame,
    matrices: dict[int, np.ndarray],
    n_sims: int = N_SIMS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (summary table, position probability matrix).

    Also persists per-simulation team goal totals for the player model.
    """
    teams = sorted(set(fixtures["home"]) | set(fixtures["away"]))
    t_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    rng = np.random.default_rng(RNG_SEED)

    pts = np.zeros((n_sims, n), dtype=np.int32)
    gf = np.zeros((n_sims, n), dtype=np.int32)
    ga = np.zeros((n_sims, n), dtype=np.int32)

    # Bank played results identically across simulations.
    played = fixtures[fixtures["finished"]]
    for _, fx in played.iterrows():
        hi, ai = t_idx[fx["home"]], t_idx[fx["away"]]
        hg, ag = int(fx["home_goals"]), int(fx["away_goals"])
        gf[:, hi] += hg; ga[:, hi] += ag
        gf[:, ai] += ag; ga[:, ai] += hg
        if hg > ag:
            pts[:, hi] += 3
        elif hg < ag:
            pts[:, ai] += 3
        else:
            pts[:, hi] += 1; pts[:, ai] += 1

    # Sample remaining fixtures.
    remaining = fixtures[~fixtures["finished"]].reset_index(drop=True)
    size = matrices[0].shape[0] if matrices else 11
    for i, fx in remaining.iterrows():
        hi, ai = t_idx[fx["home"]], t_idx[fx["away"]]
        m = matrices[i]
        draws = rng.choice(m.size, size=n_sims, p=m.ravel())
        hg = (draws // size).astype(np.int32)
        ag = (draws % size).astype(np.int32)
        gf[:, hi] += hg; ga[:, hi] += ag
        gf[:, ai] += ag; ga[:, ai] += hg
        home_w = hg > ag
        away_w = ag > hg
        drawn = ~home_w & ~away_w
        pts[:, hi] += 3 * home_w + drawn
        pts[:, ai] += 3 * away_w + drawn

    # Rank teams per simulation: points, goal difference, goals for.
    gd = gf - ga
    rank_key = pts * 10_000_000 + (gd + 500) * 1_000 + gf
    order = np.argsort(-rank_key, axis=1, kind="stable")
    positions = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    positions[rows, order] = np.arange(n)[None, :] + 1  # 1 = champion

    pos_counts = np.zeros((n, n))
    for t in range(n):
        pos_counts[t] = np.bincount(positions[:, t] - 1, minlength=n)
    pos_probs = pos_counts / n_sims

    summary = pd.DataFrame({
        "team": teams,
        "exp_pts": pts.mean(axis=0),
        "exp_gd": gd.mean(axis=0),
        "exp_gf": gf.mean(axis=0),
        "p_title": pos_probs[:, 0],
        "p_top4": pos_probs[:, :4].sum(axis=1),
        "p_top6": pos_probs[:, :6].sum(axis=1),
        "p_relegation": pos_probs[:, -3:].sum(axis=1),
        "median_pos": np.median(positions, axis=0),
    }).sort_values("exp_pts", ascending=False).reset_index(drop=True)
    summary.insert(0, "pos", summary.index + 1)

    pos_matrix = pd.DataFrame(
        pos_probs, index=teams, columns=[str(i + 1) for i in range(n)]
    ).loc[summary["team"]]

    summary.to_parquet(SIM_TABLE_PARQUET, index=False)
    pos_matrix.to_parquet(POSITION_MATRIX_PARQUET)
    np.save(TEAM_GOALS_NPY, gf)
    TEAM_ORDER_JSON.write_text(pd.Series(teams).to_json())
    return summary, pos_matrix


def load_sim_table() -> pd.DataFrame:
    return pd.read_parquet(SIM_TABLE_PARQUET)


def load_position_matrix() -> pd.DataFrame:
    return pd.read_parquet(POSITION_MATRIX_PARQUET)


def load_team_goals() -> tuple[np.ndarray, list[str]]:
    goals = np.load(TEAM_GOALS_NPY)
    teams = list(pd.read_json(TEAM_ORDER_JSON, typ="series"))
    return goals, teams
