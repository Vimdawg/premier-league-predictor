"""Feature engineering: Elo ratings and rolling form, computed strictly
from information available before each match (walk-forward, no leakage).
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 60.0
# Between seasons every team is pulled toward the mean; promoted teams enter
# carrying their Championship Elo, which is fitted on the same scale.
ELO_SEASON_REGRESS = 0.20

FEATURE_COLS = [
    "elo_diff", "elo_home", "elo_away",
    "form5_pts_h", "form5_pts_a", "form10_pts_h", "form10_pts_a",
    "form5_gf_h", "form5_ga_h", "form5_gf_a", "form5_ga_a",
    "rest_days_h", "rest_days_a",
    "home_top_flight", "away_top_flight",
]


class RollingState:
    """Walk-forward Elo + form state; updated one match at a time."""

    def __init__(self) -> None:
        self.elo: dict[str, float] = {}
        self.recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.last_played: dict[str, pd.Timestamp] = {}
        self.last_division: dict[str, str] = {}
        self.current_season: str | None = None

    def _season_rollover(self) -> None:
        if not self.elo:
            return
        mean = float(np.mean(list(self.elo.values())))
        for t in self.elo:
            self.elo[t] += ELO_SEASON_REGRESS * (mean - self.elo[t])

    def features_for(self, home: str, away: str, date: pd.Timestamp) -> dict[str, float]:
        eh = self.elo.get(home, ELO_START)
        ea = self.elo.get(away, ELO_START)

        def form(team: str) -> dict[str, float]:
            rec = list(self.recent[team])
            last5, last10 = rec[-5:], rec
            return {
                "form5_pts": np.mean([r[0] for r in last5]) if last5 else 1.0,
                "form10_pts": np.mean([r[0] for r in last10]) if last10 else 1.0,
                "form5_gf": np.mean([r[1] for r in last5]) if last5 else 1.2,
                "form5_ga": np.mean([r[2] for r in last5]) if last5 else 1.2,
            }

        fh, fa = form(home), form(away)
        rest_h = min((date - self.last_played[home]).days, 30) if home in self.last_played else 30
        rest_a = min((date - self.last_played[away]).days, 30) if away in self.last_played else 30
        return {
            "elo_diff": eh - ea,
            "elo_home": eh,
            "elo_away": ea,
            "form5_pts_h": fh["form5_pts"], "form5_pts_a": fa["form5_pts"],
            "form10_pts_h": fh["form10_pts"], "form10_pts_a": fa["form10_pts"],
            "form5_gf_h": fh["form5_gf"], "form5_ga_h": fh["form5_ga"],
            "form5_gf_a": fa["form5_gf"], "form5_ga_a": fa["form5_ga"],
            "rest_days_h": rest_h, "rest_days_a": rest_a,
            "home_top_flight": 1.0 if self.last_division.get(home, "E0") == "E0" else 0.0,
            "away_top_flight": 1.0 if self.last_division.get(away, "E0") == "E0" else 0.0,
        }

    def update(self, row: pd.Series) -> None:
        home, away = row["HomeTeam"], row["AwayTeam"]
        hg, ag = row["FTHG"], row["FTAG"]

        eh = self.elo.get(home, ELO_START)
        ea = self.elo.get(away, ELO_START)
        exp_home = 1.0 / (1.0 + 10 ** (-((eh + ELO_HOME_ADV) - ea) / 400.0))
        score = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        margin_mult = np.log(abs(hg - ag) + 1) + 1.0
        delta = ELO_K * margin_mult * (score - exp_home)
        self.elo[home] = eh + delta
        self.elo[away] = ea - delta

        pts_h = 3 if hg > ag else (1 if hg == ag else 0)
        pts_a = 3 if ag > hg else (1 if hg == ag else 0)
        self.recent[home].append((pts_h, hg, ag))
        self.recent[away].append((pts_a, ag, hg))
        self.last_played[home] = row["Date"]
        self.last_played[away] = row["Date"]
        self.last_division[home] = row.get("division", "E0")
        self.last_division[away] = row.get("division", "E0")


def build_training_features(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per match with pre-match features and the W/D/L target."""
    matches = matches.sort_values("Date").reset_index(drop=True)
    state = RollingState()
    rows = []
    for _, row in matches.iterrows():
        if row["season"] != state.current_season:
            state._season_rollover()
            state.current_season = row["season"]
        feats = state.features_for(row["HomeTeam"], row["AwayTeam"], row["Date"])
        feats.update(
            Date=row["Date"], HomeTeam=row["HomeTeam"], AwayTeam=row["AwayTeam"],
            season=row["season"], division=row["division"],
            target={"H": 0, "D": 1, "A": 2}["H" if row["FTHG"] > row["FTAG"] else ("D" if row["FTHG"] == row["FTAG"] else "A")],
        )
        for c in ("B365H", "B365D", "B365A"):
            feats[c] = row.get(c, np.nan)
        rows.append(feats)
        state.update(row)
    return pd.DataFrame(rows)


def final_state(matches: pd.DataFrame) -> RollingState:
    """State after every played match — used to featurize future fixtures."""
    state = RollingState()
    for _, row in matches.sort_values("Date").iterrows():
        if row["season"] != state.current_season:
            state._season_rollover()
            state.current_season = row["season"]
        state.update(row)
    return state
