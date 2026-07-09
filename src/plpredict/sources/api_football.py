"""API-Football (dashboard.api-football.com) client for the free plan.

Two protections make the 100-requests/day cap a non-issue:
- every response is cached in SQLite keyed by endpoint+params, so a repeated
  call never touches the network;
- a persistent per-day counter hard-stops at API_FOOTBALL_DAILY_LIMIT.

On the free plan only a limited window of seasons is served; use
`plan_seasons_available()` to discover what the key can access instead of
assuming.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3

import requests

from plpredict.config import (
    API_FOOTBALL_BASE,
    API_FOOTBALL_DAILY_LIMIT,
    API_FOOTBALL_KEY,
    CACHE_DB,
)

PREMIER_LEAGUE_ID = 39


class DailyLimitReached(RuntimeError):
    pass


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(CACHE_DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS api_cache "
        "(key TEXT PRIMARY KEY, fetched_at TEXT, response TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS api_requests (day TEXT PRIMARY KEY, count INTEGER)"
    )
    return conn


def requests_used_today() -> int:
    day = dt.date.today().isoformat()
    with _conn() as conn:
        row = conn.execute("SELECT count FROM api_requests WHERE day = ?", (day,)).fetchone()
    return row[0] if row else 0


def _record_request(conn: sqlite3.Connection) -> None:
    day = dt.date.today().isoformat()
    conn.execute(
        "INSERT INTO api_requests (day, count) VALUES (?, 1) "
        "ON CONFLICT(day) DO UPDATE SET count = count + 1",
        (day,),
    )


def get(endpoint: str, force: bool = False, **params) -> dict:
    """GET an API-Football endpoint, serving from cache when possible."""
    key = endpoint + "?" + json.dumps(params, sort_keys=True)
    with _conn() as conn:
        if not force:
            row = conn.execute(
                "SELECT response FROM api_cache WHERE key = ?", (key,)
            ).fetchone()
            if row:
                return json.loads(row[0])

        if requests_used_today() >= API_FOOTBALL_DAILY_LIMIT:
            raise DailyLimitReached(
                f"Refusing to exceed {API_FOOTBALL_DAILY_LIMIT} API-Football requests today"
            )
        if not API_FOOTBALL_KEY:
            raise RuntimeError(
                "API_FOOTBALL_KEY is not set — copy .env.example to .env and add your key"
            )

        resp = requests.get(
            f"{API_FOOTBALL_BASE}/{endpoint}",
            headers={"x-apisports-key": API_FOOTBALL_KEY},
            params=params,
            timeout=30,
        )
        _record_request(conn)
        resp.raise_for_status()
        data = resp.json()
        if data.get("errors"):
            # Plan/season restrictions come back as 200 + errors payload.
            raise RuntimeError(f"API-Football error for {endpoint}: {data['errors']}")

        conn.execute(
            "INSERT OR REPLACE INTO api_cache (key, fetched_at, response) VALUES (?, ?, ?)",
            (key, dt.datetime.now().isoformat(), json.dumps(data)),
        )
        return data


def plan_seasons_available() -> list[int]:
    """Seasons of PL data this API key can actually access."""
    data = get("leagues", id=PREMIER_LEAGUE_ID)
    seasons = data["response"][0]["seasons"] if data["response"] else []
    return sorted(s["year"] for s in seasons)


def top_scorers(season: int) -> dict:
    return get("players/topscorers", league=PREMIER_LEAGUE_ID, season=season)


def top_assists(season: int) -> dict:
    return get("players/topassists", league=PREMIER_LEAGUE_ID, season=season)


def standings(season: int) -> dict:
    return get("standings", league=PREMIER_LEAGUE_ID, season=season)
