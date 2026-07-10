"""Fantasy Premier League API client: 26/27 fixtures, teams, and player stats.

The FPL API is free and unauthenticated. bootstrap-static carries the 20
current-season teams and every player with cumulative goals/assists/minutes;
/fixtures/ carries the full 380-match schedule with gameweek numbers.
"""

from __future__ import annotations

import pandas as pd
import requests

from plpredict.config import FPL_BASE, PROCESSED_DIR, canonical_team

FIXTURES_PARQUET = PROCESSED_DIR / "fpl_fixtures.parquet"
PLAYERS_PARQUET = PROCESSED_DIR / "fpl_players.parquet"
TEAMS_PARQUET = PROCESSED_DIR / "fpl_teams.parquet"

_HEADERS = {"User-Agent": "plpredict/0.1 (personal research project)"}


def _get(path: str) -> dict | list:
    resp = requests.get(f"{FPL_BASE}/{path}", headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_bootstrap() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (teams, players) for the current FPL season."""
    data = _get("bootstrap-static/")

    teams = pd.DataFrame(data["teams"])[["id", "code", "name", "short_name", "strength"]]
    teams["team"] = teams["name"].map(canonical_team)

    players = pd.DataFrame(data["elements"])
    keep = [
        "id", "code", "photo", "web_name", "first_name", "second_name", "team", "element_type",
        "minutes", "goals_scored", "assists", "starts",
        "expected_goals", "expected_assists",
        "status", "chance_of_playing_next_round", "penalties_order",
    ]
    players = players[[c for c in keep if c in players.columns]].rename(
        columns={"team": "team_id"}
    )
    team_map = dict(zip(teams["id"], teams["team"]))
    players["team"] = players["team_id"].map(team_map)
    players["position"] = players["element_type"].map(
        {1: "GK", 2: "DEF", 3: "MID", 4: "FWD", 5: "MGR"}
    )
    players = players[players["position"] != "MGR"]

    teams.to_parquet(TEAMS_PARQUET, index=False)
    players.to_parquet(PLAYERS_PARQUET, index=False)
    return teams, players


def fetch_fixtures(teams: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return all 380 fixtures with canonical team names and played results."""
    if teams is None:
        teams = pd.read_parquet(TEAMS_PARQUET) if TEAMS_PARQUET.exists() else fetch_bootstrap()[0]
    team_map = dict(zip(teams["id"], teams["team"]))

    fixtures = pd.DataFrame(_get("fixtures/"))
    fixtures = fixtures[
        ["id", "event", "kickoff_time", "team_h", "team_a",
         "team_h_score", "team_a_score", "finished"]
    ].rename(columns={"event": "gameweek"})
    fixtures["home"] = fixtures["team_h"].map(team_map)
    fixtures["away"] = fixtures["team_a"].map(team_map)
    fixtures["kickoff_time"] = pd.to_datetime(fixtures["kickoff_time"])
    fixtures.to_parquet(FIXTURES_PARQUET, index=False)
    return fixtures


def fetch_player_history(player_id: int) -> pd.DataFrame:
    """Per-gameweek log for one player (current FPL season): minutes, goals,
    assists, opponent. Powers the rotation view on player pages."""
    data = _get(f"element-summary/{player_id}/")
    hist = pd.DataFrame(data.get("history", []))
    if hist.empty:
        return hist
    keep = ["round", "kickoff_time", "opponent_team", "was_home",
            "minutes", "goals_scored", "assists", "total_points"]
    hist = hist[[c for c in keep if c in hist.columns]]
    hist["kickoff_time"] = pd.to_datetime(hist["kickoff_time"])
    return hist


def load_players() -> pd.DataFrame:
    if not PLAYERS_PARQUET.exists():
        fetch_bootstrap()
    return pd.read_parquet(PLAYERS_PARQUET)


def load_fixtures() -> pd.DataFrame:
    if not FIXTURES_PARQUET.exists():
        fetch_fixtures()
    return pd.read_parquet(FIXTURES_PARQUET)


def load_teams() -> pd.DataFrame:
    if not TEAMS_PARQUET.exists():
        fetch_bootstrap()
    return pd.read_parquet(TEAMS_PARQUET)
