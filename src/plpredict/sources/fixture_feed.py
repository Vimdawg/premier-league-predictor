"""26/27 fixture list from fixturedownload.com.

The FPL API only exposes the new season after its July rollover, so this feed
is the fixture source of record until then. `load_target_fixtures` prefers the
FPL fixture list once it is serving the target season (it then also carries
live scores), falling back to this feed otherwise.
"""

from __future__ import annotations

import pandas as pd
import requests

from plpredict.config import PROCESSED_DIR, TARGET_SEASON_START_YEAR, canonical_team

FEED_URL = f"https://fixturedownload.com/feed/json/epl-{TARGET_SEASON_START_YEAR}"
FIXTURES_PARQUET = PROCESSED_DIR / "fixtures_2627.parquet"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


def fetch_fixtures(force: bool = False) -> pd.DataFrame:
    if FIXTURES_PARQUET.exists() and not force:
        return pd.read_parquet(FIXTURES_PARQUET)
    resp = requests.get(FEED_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df = df.rename(
        columns={
            "RoundNumber": "gameweek",
            "DateUtc": "kickoff_time",
            "HomeTeam": "home",
            "AwayTeam": "away",
            "HomeTeamScore": "home_goals",
            "AwayTeamScore": "away_goals",
        }
    )[["gameweek", "kickoff_time", "home", "away", "home_goals", "away_goals"]]
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"])
    df["home"] = df["home"].map(canonical_team)
    df["away"] = df["away"].map(canonical_team)
    df["finished"] = df["home_goals"].notna() & df["away_goals"].notna()
    df.to_parquet(FIXTURES_PARQUET, index=False)
    return df


def load_target_fixtures(force: bool = False) -> pd.DataFrame:
    """The 380 fixtures of the target season, from the best available source."""
    from plpredict.sources import fpl

    try:
        fpl_fixtures = fpl.fetch_fixtures() if force else fpl.load_fixtures()
        kickoffs = fpl_fixtures["kickoff_time"].dropna()
        if not kickoffs.empty and kickoffs.min().year == TARGET_SEASON_START_YEAR:
            return fpl_fixtures.rename(
                columns={"team_h_score": "home_goals", "team_a_score": "away_goals"}
            )[["gameweek", "kickoff_time", "home", "away",
               "home_goals", "away_goals", "finished"]]
    except requests.RequestException:
        pass
    return fetch_fixtures(force=force)
