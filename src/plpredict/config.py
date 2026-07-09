"""Central configuration: paths, season constants, team-name normalization."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DB = DATA_DIR / "cache.sqlite"

for _d in (RAW_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
API_FOOTBALL_DAILY_LIMIT = 95  # stay safely under the 100/day free cap

FPL_BASE = "https://fantasy.premierleague.com/api"
FOOTBALL_DATA_BASE = "https://www.football-data.co.uk/mmz4281"

# The season being predicted. "2627" is football-data.co.uk's code for 2026/27.
TARGET_SEASON = "2627"
TARGET_SEASON_START_YEAR = 2026

# Historical seasons used for training (football-data.co.uk season codes).
TRAIN_SEASONS_PL = [
    "1516", "1617", "1718", "1819", "1920",
    "2021", "2122", "2223", "2324", "2425", "2526",
]
# Championship seasons used to build priors for promoted teams.
TRAIN_SEASONS_CHAMP = ["2425", "2526"]

# football-data.co.uk division codes.
DIV_PL = "E0"
DIV_CHAMP = "E1"

# Canonical names keyed by every alias seen in football-data.co.uk and the FPL
# API. Covers clubs that have appeared in the PL or Championship recently so
# whichever three came up from the 25/26 Championship are handled.
TEAM_ALIASES: dict[str, str] = {
    # football-data.co.uk style
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield United": "Sheffield United",
    "Sheffield Utd": "Sheffield United",
    "Sheffield Weds": "Sheffield Wednesday",
    "Tottenham": "Tottenham Hotspur",
    "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
    "Newcastle": "Newcastle United",
    "Leicester": "Leicester City",
    "Leeds": "Leeds United",
    "Norwich": "Norwich City",
    "Cardiff": "Cardiff City",
    "Swansea": "Swansea City",
    "Hull": "Hull City",
    "Stoke": "Stoke City",
    "Birmingham": "Birmingham City",
    "Coventry": "Coventry City",
    "Derby": "Derby County",
    "Ipswich": "Ipswich Town",
    "Luton": "Luton Town",
    "Oxford": "Oxford United",
    "Plymouth": "Plymouth Argyle",
    "Preston": "Preston North End",
    "QPR": "Queens Park Rangers",
    "Blackburn": "Blackburn Rovers",
    "Bristol City": "Bristol City",
    "Charlton": "Charlton Athletic",
    "Wrexham": "Wrexham",
    "Portsmouth": "Portsmouth",
    "Middlesbrough": "Middlesbrough",
    "Millwall": "Millwall",
    "Sunderland": "Sunderland",
    "Watford": "Watford",
    "Burnley": "Burnley",
    "Fulham": "Fulham",
    "Brentford": "Brentford",
    "Brighton": "Brighton & Hove Albion",
    "Bournemouth": "AFC Bournemouth",
    "Southampton": "Southampton",
    "Everton": "Everton",
    "Liverpool": "Liverpool",
    "Arsenal": "Arsenal",
    "Chelsea": "Chelsea",
    "Aston Villa": "Aston Villa",
    "Crystal Palace": "Crystal Palace",
    # FPL style (bootstrap-static "name" field)
    "Manchester City": "Manchester City",
    "Manchester Utd": "Manchester United",
    "Manchester United": "Manchester United",
    "Newcastle United": "Newcastle United",
    "Nottingham Forest": "Nottingham Forest",
    "Spurs": "Tottenham Hotspur",
    "Tottenham Hotspur": "Tottenham Hotspur",
    "West Ham United": "West Ham United",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
    "Brighton & Hove Albion": "Brighton & Hove Albion",
    "Brighton and Hove Albion": "Brighton & Hove Albion",
    "AFC Bournemouth": "AFC Bournemouth",
    "Leicester City": "Leicester City",
    "Leeds United": "Leeds United",
    "Ipswich Town": "Ipswich Town",
    "Luton Town": "Luton Town",
    "Sheffield Wednesday": "Sheffield Wednesday",
    "West Bromwich Albion": "West Bromwich Albion",
    "Queens Park Rangers": "Queens Park Rangers",
    "Norwich City": "Norwich City",
    "Cardiff City": "Cardiff City",
    "Swansea City": "Swansea City",
    "Hull City": "Hull City",
    "Stoke City": "Stoke City",
    "Birmingham City": "Birmingham City",
    "Coventry City": "Coventry City",
    "Derby County": "Derby County",
    "Oxford United": "Oxford United",
    "Plymouth Argyle": "Plymouth Argyle",
    "Preston North End": "Preston North End",
    "Blackburn Rovers": "Blackburn Rovers",
    "Charlton Athletic": "Charlton Athletic",
}


def canonical_team(name: str) -> str:
    """Map any source's team name to its canonical form.

    Unknown names pass through unchanged (stripped) so a brand-new club never
    crashes the pipeline — it just won't merge with other sources until an
    alias is added.
    """
    name = name.strip()
    return TEAM_ALIASES.get(name, name)
