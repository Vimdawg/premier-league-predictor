"""Top scorer / top assists prediction.

Each player's share of his team's goals (and assists) is estimated from FPL
data — actual output blended with expected stats, shrunk by projected
minutes — and simulated team goal totals from the season simulation are then
allocated multinomially to players in every simulation run. The share left
unassigned in each squad implicitly covers new signings and squad churn.

Until the FPL API rolls over to the target season, promoted clubs have no
player data and are skipped; `plpredict update` picks them up automatically
after the rollover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plpredict.config import PROCESSED_DIR
from plpredict.sources import fpl
from plpredict.simulate import load_team_goals

RNG_SEED = 1857
MAX_TEAM_MINUTES = 38 * 90.0
ASSISTS_PER_GOAL = 0.72  # league-wide ratio of credited assists to goals

SCORERS_PARQUET = PROCESSED_DIR / "top_scorers.parquet"
ASSISTS_PARQUET = PROCESSED_DIR / "top_assists.parquet"


def build_player_shares(players: pd.DataFrame) -> pd.DataFrame:
    """Per-player expected share of team goals/assists for the new season."""
    df = players.copy()
    for col in ("expected_goals", "expected_assists"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["minutes_share"] = (df["minutes"] / MAX_TEAM_MINUTES).clip(0, 1)

    # Stabilized individual output: blend actual with expected stats.
    df["goal_output"] = 0.6 * df["goals_scored"] + 0.4 * df["expected_goals"]
    df["assist_output"] = 0.6 * df["assists"] + 0.4 * df["expected_assists"]

    # Penalty takers get a bump — penalties are sticky season to season.
    is_pen_taker = df["penalties_order"].fillna(99) == 1
    df.loc[is_pen_taker, "goal_output"] *= 1.10

    # Unavailable players (left club / long-term injury) are zeroed.
    unavailable = df["status"].isin(["u", "n"])
    df.loc[unavailable, ["goal_output", "assist_output"]] = 0.0

    team_go = df.groupby("team")["goal_output"].transform("sum")
    team_ao = df.groupby("team")["assist_output"].transform("sum")
    # Shrink raw shares: players rarely repeat 100% of their output share, and
    # the remainder absorbs incoming transfers.
    df["goal_share"] = 0.85 * (df["goal_output"] / team_go.clip(lower=1e-9))
    df["assist_share"] = 0.85 * (df["assist_output"] / team_ao.clip(lower=1e-9))
    return df


def simulate_players(n_keep: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Allocate simulated team goals to players across every season run.

    Returns (top scorers leaderboard, top assists leaderboard) with expected
    totals and P(top scorer)/P(top assists).
    """
    team_goals, teams = load_team_goals()  # (n_sims, n_teams)
    n_sims = team_goals.shape[0]
    rng = np.random.default_rng(RNG_SEED)

    shares = build_player_shares(fpl.load_players())
    shares = shares[shares["team"].isin(teams)]

    boards = {}
    for kind, share_col, scale in (
        ("goals", "goal_share", 1.0),
        ("assists", "assist_share", ASSISTS_PER_GOAL),
    ):
        player_names, player_teams, totals = [], [], []
        for ti, team in enumerate(teams):
            squad = shares[(shares["team"] == team) & (shares[share_col] > 0.005)]
            if squad.empty:
                continue
            p = squad[share_col].to_numpy()
            p_full = np.append(p, max(1.0 - p.sum(), 0.0))  # slack = rest of squad
            p_full /= p_full.sum()
            events = np.round(team_goals[:, ti] * scale).astype(np.int64)
            # One multinomial draw per simulation, vectorized over sims.
            alloc = rng.multinomial(events, p_full)  # (n_sims, len(squad)+1)
            totals.append(alloc[:, :-1])
            player_names.extend(squad["web_name"])
            player_teams.extend(squad["team"])

        all_totals = np.concatenate(totals, axis=1)  # (n_sims, n_players)
        winner = all_totals.argmax(axis=1)
        p_top = np.bincount(winner, minlength=all_totals.shape[1]) / n_sims

        board = pd.DataFrame({
            "player": player_names,
            "team": player_teams,
            f"exp_{kind}": all_totals.mean(axis=0),
            f"p_top_{kind}": p_top,
            f"{kind}_p90_high": np.percentile(all_totals, 90, axis=0),
        }).sort_values(f"exp_{kind}", ascending=False).head(n_keep).reset_index(drop=True)
        boards[kind] = board

    boards["goals"].to_parquet(SCORERS_PARQUET, index=False)
    boards["assists"].to_parquet(ASSISTS_PARQUET, index=False)
    return boards["goals"], boards["assists"]


def load_scorers() -> pd.DataFrame:
    return pd.read_parquet(SCORERS_PARQUET)


def load_assists() -> pd.DataFrame:
    return pd.read_parquet(ASSISTS_PARQUET)
