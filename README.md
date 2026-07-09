# PL Predict — Premier League 26/27 Prediction Model

Predicts every 26/27 Premier League match, the final table, and the Golden
Boot / most-assists races. Hybrid model: a time-decayed Dixon-Coles goal model
blended with a LightGBM classifier, feeding a 10,000-run Monte Carlo season
simulation. Ships with a Streamlit dashboard and a weekly update loop.

## Setup

```bash
# install uv if needed: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
cp .env.example .env   # optional: add your API-Football key
```

The core pipeline runs entirely on free, keyless sources:

| Source | Provides |
|---|---|
| [football-data.co.uk](https://www.football-data.co.uk) | 11 seasons of PL + Championship results, Bet365 odds benchmark, weekly 26/27 results |
| FPL API (`fantasy.premierleague.com/api`) | squads and per-player goals/assists/minutes (26/27 after its July rollover) |
| [fixturedownload.com](https://fixturedownload.com) | the 380-fixture 26/27 schedule |
| [API-Football](https://dashboard.api-football.com/) *(optional, free plan)* | historical player stats / metadata; client caps itself below 100 req/day with full response caching |

## Usage

```bash
uv run plpredict update       # full pipeline: fetch → train → predict → simulate → players
uv run streamlit run app/dashboard.py
```

Individual steps: `fetch`, `train`, `predict`, `simulate`, `players`, `status`.
Run `update` after each gameweek once the season starts (21 Aug 2026) — played
results are banked, models refit, and all predictions re-simulated.

## How it works

1. **Dixon-Coles** (`models/dixon_coles.py`) — per-team attack/defense ratings
   with home advantage and low-score correlation, exponentially decayed
   (half-life ≈ one season), fitted by MLE jointly over PL + Championship so
   promoted clubs are rated before their first PL match.
2. **LightGBM** (`models/ml_model.py`) — W/D/L classifier over walk-forward
   Elo, rolling form, rest days, and division-experience features. Validated
   chronologically (train ≤22/23, early-stop 23/24, evaluate 24/25+25/26).
3. **Ensemble** (`models/ensemble.py`) — log-loss-optimal blend, backtested
   against Bet365 closing odds. Current: 1.0165 vs bookmaker 0.9947 vs
   uniform 1.0986 on 760 held-out matches.
4. **Season simulation** (`simulate.py`) — samples scorelines from
   blend-rescaled score matrices 10,000×; outputs expected points, full
   position-probability matrix, title/top-4/relegation odds.
5. **Player model** (`players.py`) — allocates each team's simulated goals to
   players multinomially by expected share (FPL output blended with xG/xA,
   minutes- and penalty-adjusted) → Golden Boot / assists distributions.

## Tests

```bash
uv run pytest
```
