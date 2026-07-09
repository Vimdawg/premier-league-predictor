"""Historical + in-season match results from football-data.co.uk CSVs."""

from __future__ import annotations

import io

import pandas as pd
import requests

from plpredict.config import (
    DIV_CHAMP,
    DIV_PL,
    FOOTBALL_DATA_BASE,
    PROCESSED_DIR,
    RAW_DIR,
    TARGET_SEASON,
    TRAIN_SEASONS_CHAMP,
    TRAIN_SEASONS_PL,
    canonical_team,
)

# Columns kept from the raw CSVs. B365* are Bet365 closing odds, used only as
# a benchmark for model evaluation, never as a model input.
KEEP_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"]

MATCHES_PARQUET = PROCESSED_DIR / "matches.parquet"


def _season_url(season: str, div: str) -> str:
    return f"{FOOTBALL_DATA_BASE}/{season}/{div}.csv"


def download_season(season: str, div: str, force: bool = False) -> pd.DataFrame | None:
    """Download one season CSV, caching the raw file locally.

    Returns None when the file doesn't exist yet (e.g. the 26/27 file before
    any round has been played).
    """
    raw_path = RAW_DIR / f"{div}_{season}.csv"
    is_current = season == TARGET_SEASON
    if raw_path.exists() and not force and not is_current:
        text = raw_path.read_text(encoding="utf-8", errors="replace")
    else:
        resp = requests.get(_season_url(season, div), timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        # An empty or headerless body also means "season not started".
        if len(resp.text.strip().splitlines()) < 2:
            return None
        raw_path.write_text(resp.text, encoding="utf-8")
        text = resp.text

    df = pd.read_csv(io.StringIO(text), on_bad_lines="skip", encoding_errors="replace")
    df = df.dropna(how="all")
    cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[cols].dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    df["HomeTeam"] = df["HomeTeam"].map(canonical_team)
    df["AwayTeam"] = df["AwayTeam"].map(canonical_team)
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["season"] = season
    df["division"] = div
    return df


def fetch_all_matches(force_current: bool = True) -> pd.DataFrame:
    """Download every training season plus any played 26/27 matches.

    Historical seasons are served from the raw-file cache; the current season
    is re-downloaded on each call so weekly updates pick up new results.
    """
    frames: list[pd.DataFrame] = []
    for season in TRAIN_SEASONS_PL:
        df = download_season(season, DIV_PL)
        if df is not None:
            frames.append(df)
    for season in TRAIN_SEASONS_CHAMP:
        df = download_season(season, DIV_CHAMP)
        if df is not None:
            frames.append(df)
    current = download_season(TARGET_SEASON, DIV_PL, force=force_current)
    if current is not None:
        frames.append(current)

    matches = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    matches.to_parquet(MATCHES_PARQUET, index=False)
    return matches


def load_matches() -> pd.DataFrame:
    if not MATCHES_PARQUET.exists():
        return fetch_all_matches()
    return pd.read_parquet(MATCHES_PARQUET)
