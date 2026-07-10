# PL Predict — Complete System Guide

A reference manual for the Premier League 26/27 prediction system: what it does,
how every piece works, how to operate it, and the gotchas discovered while
building it.

---

## Table of Contents

1. [What the System Does](#1-what-the-system-does)
2. [Architecture at a Glance](#2-architecture-at-a-glance)
3. [Data Sources](#3-data-sources)
4. [Project Structure](#4-project-structure)
5. [The Prediction Models](#5-the-prediction-models)
6. [Season Simulation](#6-season-simulation)
7. [Player Predictions (Golden Boot & Assists)](#7-player-predictions-golden-boot--assists)
8. [The CLI](#8-the-cli)
9. [The Dashboard](#9-the-dashboard)
10. [Operating the System (Weekly Workflow)](#10-operating-the-system-weekly-workflow)
11. [Generated Artifacts Reference](#11-generated-artifacts-reference)
12. [Known Gotchas & Design Decisions](#12-known-gotchas--design-decisions)
13. [Extending the System](#13-extending-the-system)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. What the System Does

Predicts, for the 2026/27 Premier League season:

- **Every match** (all 380 fixtures): win/draw/loss probabilities, expected
  goals per side, and the single most likely scoreline.
- **The final table**: expected points, plus the probability of every team
  finishing in every position (title, top 4, top 6, relegation odds).
- **The Golden Boot and top-assists races**: expected goals/assists per player
  and the probability of each player finishing top.

It is a **living system**: after each real gameweek you run one command,
played results are banked, the models refit, and every prediction updates.

Prediction quality (honest out-of-sample backtest on 760 matches from the
24/25 + 25/26 seasons, log-loss, lower is better):

| Model | Log-loss |
|---|---|
| Uniform guess (⅓ / ⅓ / ⅓) | 1.0986 |
| LightGBM alone | 1.0349 |
| Dixon-Coles alone | 1.0182 |
| **Ensemble blend (deployed)** | **1.0165** |
| Bet365 closing odds (practical ceiling) | 0.9947 |

Being within ~0.02 of the bookmaker is strong for a public-data model —
bookmakers price in team news, injuries and betting-market information the
model cannot see.

---

## 2. Architecture at a Glance

```
DATA SOURCES                     PIPELINE                        OUTPUTS
─────────────                    ────────                        ───────
football-data.co.uk ──┐
  (11 PL seasons +    │   ┌──────────────────┐
   2 Championship,    ├──▶│ fetch            │──▶ matches.parquet
   Bet365 odds)       │   └──────────────────┘    fpl_*.parquet
                      │                           fixtures_2627.parquet
FPL API ──────────────┤   ┌──────────────────┐
  (squads, player     │   │ train            │──▶ lgbm_model.txt
   stats, fixtures    ├──▶│  Elo/form feats  │    ensemble.json
   after rollover)    │   │  LightGBM        │    (backtest metrics)
                      │   │  DC backtest     │
fixturedownload.com ──┤   └──────────────────┘
  (26/27 fixture      │   ┌──────────────────┐
   list until FPL     ├──▶│ predict          │──▶ dixon_coles.json
   rolls over)        │   │  fit DC on all   │    match_predictions.parquet
                      │   │  blend w/ ML     │    score_matrices.npz
API-Football ─────────┘   └──────────────────┘
  (optional, free           ┌──────────────────┐
   tier, cached,        ┌──▶│ simulate 10,000× │──▶ sim_table.parquet
   ≤95 req/day)         │   └──────────────────┘    position_matrix.parquet
                        │   ┌──────────────────┐    sim_team_goals.npy
                        └──▶│ players          │──▶ top_scorers.parquet
                            └──────────────────┘    top_assists.parquet
                                     │
                                     ▼
                            Streamlit dashboard (app/dashboard.py)
```

Everything above runs with `uv run plpredict update` (or the individual
subcommands), and the dashboard reads only the generated artifacts — it never
trains anything itself, so it loads instantly.

---

## 3. Data Sources

| Source | What it provides | Auth | Notes |
|---|---|---|---|
| **football-data.co.uk** | Full match results for PL seasons 2015/16→2025/26, Championship 24/25–25/26, Bet365 closing odds | None | Historical CSVs are cached in `data/raw/`; the current-season file is re-downloaded on every fetch so weekly results flow in. Odds are used **only** as an evaluation benchmark, never as a model input. |
| **FPL API** (`fantasy.premierleague.com/api`) | The 20 current teams, every player's minutes/goals/assists/xG/xA, penalty-taker order, injury status, per-gameweek match logs, fixture list with live scores | None | **Rolls over to the new season in mid-July.** Until then it serves last season's data (which is exactly what the player model trains on). |
| **fixturedownload.com** | The 380-fixture 26/27 schedule with gameweek numbers | None | Blocks default user agents — the client sends a browser UA. Used as the fixture source of record until the FPL rollover; `load_target_fixtures()` switches to FPL automatically once FPL's first kickoff year matches the target season. |
| **API-Football** (dashboard.api-football.com) | Historical player stats, standings, metadata | Key in `.env` | **Optional.** Free tier = 100 req/day and limited seasons. The client hard-stops at 95 requests/day (counter persisted in `data/cache.sqlite`) and caches every response, so repeat calls never hit the network. The core pipeline never depends on it. |
| **Premier League CDN** (`resources.premierleague.com`) | Club crests (`badges/70/t{code}@x2.png`) and player photos (`photos/players/110x140/p{code}.png`) | None | Keyed by FPL `code` fields. Clubs absent from FPL (promoted, pre-rollover) get a generated initials-avatar SVG instead. |

### Which team codes matter

- Crest/photo codes come from the FPL bootstrap (`code` column stored in
  `fpl_teams.parquet` / `fpl_players.parquet`).
- Two hardcoded fallbacks exist in the dashboard for pre-rollover gaps:
  Ipswich Town = 40, Hull City = 88. Coventry City has no known code and
  shows an initials avatar until the FPL API rolls over.

---

## 4. Project Structure

```
FunProject/
├── .env / .env.example        # API_FOOTBALL_KEY (optional)
├── .streamlit/config.toml     # dark theme (purple bg, PL-green accent)
├── .claude/launch.json        # dashboard launch config for browser preview
├── pyproject.toml             # uv project; `plpredict` CLI entry point
├── data/                      # ALL generated data — gitignored
│   ├── raw/                   # cached source CSVs (E0_1516.csv, ...)
│   ├── processed/             # model artifacts (see §11)
│   └── cache.sqlite           # API-Football response cache + request counter
├── app/
│   ├── dashboard.py           # the entire Streamlit UI
│   └── assets/
│       ├── logo.png           # transparent purple logo
│       └── logo_white.png     # white version used in the top-left brand bar
├── src/plpredict/
│   ├── config.py              # paths, season constants, TEAM_ALIASES, canonical_team()
│   ├── cli.py                 # fetch / train / predict / simulate / players / update / status
│   ├── features.py            # walk-forward Elo + rolling-form feature builder
│   ├── predict.py             # per-fixture blended predictions + score matrices
│   ├── simulate.py            # 10,000-run Monte Carlo season simulation
│   ├── players.py             # goal/assist share model → Golden Boot sim
│   ├── sources/
│   │   ├── football_data.py   # football-data.co.uk downloader
│   │   ├── fpl.py             # FPL API client (+ per-player match history)
│   │   ├── fixture_feed.py    # fixturedownload.com + source auto-switching
│   │   └── api_football.py    # optional API-Football client (capped + cached)
│   └── models/
│       ├── dixon_coles.py     # the statistical goal model
│       ├── ml_model.py        # LightGBM W/D/L classifier
│       └── ensemble.py        # blend weight fitting + backtest
└── tests/
    ├── test_models.py         # synthetic-data sanity tests (no network)
    └── test_simulation.py     # invariants on the real artifacts
```

### Team-name canonicalization

Every source spells clubs differently ("Man City" / "Manchester City",
"Spurs" / "Tottenham Hotspur", "Nott'm Forest"...). `config.canonical_team()`
maps all known aliases to one canonical name; **unknown names pass through
unchanged** so a brand-new club never crashes the pipeline — it just won't
merge across sources until you add an alias to `TEAM_ALIASES`.

---

## 5. The Prediction Models

Three components are combined:

### 5a. Dixon-Coles goal model (`models/dixon_coles.py`) — the workhorse

The classic Dixon & Coles (1997) model. Each team `i` gets a log **attack**
rating `att_i` and **defense** rating `def_i`; the expected goals in a match
are:

```
home goals  λ = exp(base + home_adv + att_home + def_away)
away goals  μ = exp(base +            att_away + def_home)
```

plus the **rho correction** that adjusts the probabilities of the low-scoring
outcomes (0-0, 1-0, 0-1, 1-1) where independent Poissons are known to be wrong.

Key implementation choices:

- **Time decay**: every match is weighted `exp(-ξ · days_ago)` with a
  half-life of ~390 days, so recent form dominates but a full season of
  history still matters.
- **Joint PL + Championship fit**: promoted clubs (Coventry, Ipswich, Hull)
  get ratings before playing a single PL match, anchored to the PL scale by
  clubs that moved between divisions. This is why the sim places them
  17th–20th rather than treating them as unknowns.
- Fitted by maximum likelihood with `scipy.optimize` (L-BFGS-B), attacks
  constrained to sum to zero for identifiability.
- The fitted model is saved to `dixon_coles.json` during `predict` so the
  dashboard can display attack/defense ranks on club pages.

Typical fitted values: `home_adv ≈ +0.19`, `rho ≈ -0.07` — consistent with
the published literature.

### 5b. Feature engineering + LightGBM (`features.py`, `models/ml_model.py`)

A gradient-boosted W/D/L classifier over **walk-forward features** — every
feature for a match is computed strictly from information available before
that match (no leakage):

- Elo ratings (K=20, +60 home advantage, goal-margin multiplier, 20% regression
  to the mean between seasons)
- Rolling points-per-game and goals for/against over the last 5 and 10 matches
- Rest days since each side's previous match (capped at 30)
- Whether each side's last match was top-flight (division-experience signal)

Split protocol (chronological, never random):

| Purpose | Seasons |
|---|---|
| Train | 2015/16 → 2022/23 |
| Early stopping | 2023/24 |
| Validation (blend fitting + honest metrics) | 2024/25 + 2025/26 |

The deployed model is refit on **all** data at the tuned iteration count.

### 5c. Ensemble (`models/ensemble.py`)

The validation seasons get out-of-sample probabilities from **both** models
(the DC side refits monthly and predicts only the following month's matches).
A grid search finds the blend weight `w` minimizing validation log-loss:

```
p_final = w · p_DixonColes + (1-w) · p_LightGBM        currently w = 0.76
```

Metrics (including the bookmaker benchmark) are saved to `ensemble.json` and
shown on the dashboard's Model tab.

### How the blend meets the scorelines

`predict.py` needs full **score matrices** (P(home i, away j) up to 10 goals),
not just W/D/L. It takes the DC score matrix and rescales its three regions
(home-win triangle, draw diagonal, away-win triangle) so the total masses match
the blended W/D/L probabilities — scoreline detail from Dixon-Coles, outcome
balance from the ensemble.

---

## 6. Season Simulation

`simulate.py` runs the season **10,000 times** (fixed RNG seed 2627 for
reproducibility):

1. Points/goals from already-played fixtures are banked identically in every run.
2. Each remaining fixture's scoreline is sampled from its rescaled score matrix.
3. Teams are ranked per run by points → goal difference → goals for.
4. Aggregation produces: expected points, the full 20×20 position-probability
   matrix, title/top-4/top-6/relegation probabilities, and the most likely
   final table.

It also saves per-run **team goal totals** (`sim_team_goals.npy`) — the input
to the player model, so the Golden Boot simulation is consistent with the
season simulation (a run where City score 95 gives Haaland more goals than a
run where they score 70).

---

## 7. Player Predictions (Golden Boot & Assists)

`players.py` converts team-level simulations into player-level races:

1. **Expected share**: each player's share of his club's goals is estimated
   from FPL data — `0.6 · actual + 0.4 · expected (xG/xA)` for stability,
   ×1.10 for first-choice penalty takers, zeroed for players flagged
   unavailable. Shares are shrunk by 0.85; the leftover slack absorbs new
   signings and squad churn.
2. **Allocation**: in every one of the 10,000 season runs, each team's
   simulated goal total is distributed to its players with one multinomial
   draw (assists use the league-wide 0.72 assists-per-goal ratio).
3. **Aggregation**: expected goals/assists per player, 90th-percentile
   outcomes, and P(top scorer)/P(top assists) = the fraction of runs won.

**Limitation**: clubs missing from the FPL data (promoted clubs before the
July rollover) have no players in the race. They join automatically after the
first `update` following the rollover.

---

## 8. The CLI

All commands: `uv run plpredict <command>`

| Command | What it does | When to run |
|---|---|---|
| `fetch` | Downloads/refreshes all source data | Automatically part of `update` |
| `train` | Builds features, trains LightGBM, runs the DC walk-forward backtest, fits the blend weight | After fetch when history changed |
| `predict` | Fits DC on all data, blends with ML, writes per-fixture predictions + score matrices, saves the DC model | After train |
| `simulate` | 10,000-run Monte Carlo → table artifacts | After predict |
| `players` | Golden Boot / assists simulation | After simulate |
| `update` | **All of the above in order** — the one command you actually need | Weekly, after each gameweek |
| `status` | Shows whether an API-Football key is set and today's request count | Whenever curious |

`update` takes ~2–3 minutes; the DC monthly-refit backtest inside `train`
is the slow part.

Dashboard: `uv run streamlit run app/dashboard.py` → http://localhost:8501
(also available via the `dashboard` entry in `.claude/launch.json`).

Tests: `uv run pytest` (12 tests: model math on synthetic data + invariants
on the real artifacts; artifact tests auto-skip if the pipeline hasn't run).

---

## 9. The Dashboard

Single file: `app/dashboard.py`. Reads artifacts only — safe to restart anytime.

### Layout

- **Brand bar** — the PL 26/27 Predictor logo fixed in the top-left corner of
  every page; clicking it returns home.
- **Hero** — headline, model summary line, and four favorite tiles (title,
  Golden Boot, top assists, relegation).
- **Four tabs**:
  1. **Final table** — most likely table with crests, zone-tinted rows
     (green = Champions League, blue = Europe, orange = relegation) and inline
     probability bars, beside the finishing-position heatmap.
  2. **Match predictions** — fixture cards with filters (see below).
  3. **Golden Boot & assists** — ranked player cards with photos and P(top).
  4. **Model** — the backtest chart and season-tracking notes.

### Match-prediction filters

- **Teams** multiselect (empty = all). Selecting a team defaults the gameweek
  to "All".
- **Predicted result** — context-sensitive:
  - With teams selected: *Win / Draw / Loss* relative to those teams.
  - Without: *Home win / Draw / Away win* league-wide.
  - Win/loss use the **most likely outcome** (argmax of the three
    probabilities). **Draw is special**: a draw is almost never the argmax,
    so the draw filter instead shows fixtures with **≥25% draw probability,
    sorted most draw-likely first**.
- **Gameweek** — "All" or a specific week. Display caps at 60 cards.

### Click-through navigation (query params, deep-linkable)

- Any club name/crest → `?club=<name>`: model ranks (attack/defense #),
  prediction tiles, finishing-position distribution, last 10 real results,
  next 10 predicted fixtures, and a **squad & rotation table** (minutes share,
  starts, G/A, availability, penalty takers).
- Any player name/photo → `?player=<fpl id>`: season stats + xG/xA, projected
  26/27 goals/assists, a **minutes-per-gameweek rotation chart**
  (green = 60+ min start, blue = sub, gap = unused — injuries are instantly
  visible), and the last-10-match log fetched live from the FPL
  element-summary endpoint (cached 1 hour).
- "← Back to dashboard" on every detail page; the brand logo also goes home.

Players outside the top-40 leaderboards still get a projection on their page,
computed live as `goal_share × simulated team goals`.

---

## 10. Operating the System (Weekly Workflow)

### Now (pre-season, July 2026)

Nothing required. Optionally re-run `update` after the FPL API rolls over to
26/27 (mid-July) — that brings in the new squads, summer transfers, and the
promoted clubs' players.

### During the season (from 21 Aug 2026)

After each gameweek completes:

```bash
uv run plpredict update
```

That single command: pulls the new results, refits every model with the extra
information, re-simulates the remaining season, and refreshes the player
races. Then refresh the dashboard tab.

### Occasional maintenance

- **New club appears** (promotion) → add its name variants to `TEAM_ALIASES`
  in `config.py` if the sources spell it differently.
- **New season** → update `TARGET_SEASON` / `TARGET_SEASON_START_YEAR` and
  append the finished season to `TRAIN_SEASONS_PL` in `config.py`.
- **API-Football key** → put it in `.env` (`API_FOOTBALL_KEY=...`); check
  usage with `plpredict status`.

---

## 11. Generated Artifacts Reference

All in `data/processed/` (gitignored — regenerate with `update`):

| File | Producer | Contents |
|---|---|---|
| `matches.parquet` | fetch | Every historical match (Date, teams, goals, division, B365 odds) |
| `fpl_teams.parquet` | fetch | 20 FPL teams + crest `code` |
| `fpl_players.parquet` | fetch | Player stats + photo `code`, penalty order, status |
| `fpl_fixtures.parquet` | fetch | FPL fixture list (current FPL season) |
| `fixtures_2627.parquet` | fetch | 26/27 fixtures from fixturedownload.com |
| `lgbm_model.txt` | train | Deployed LightGBM booster |
| `ensemble.json` | train | Blend weight + all backtest metrics |
| `dixon_coles.json` | predict | Fitted DC ratings (attack/defense/home_adv/rho) |
| `match_predictions.parquet` | predict | Per-fixture probabilities, xG, likely score |
| `score_matrices.npz` | predict | 11×11 scoreline distributions per fixture |
| `sim_table.parquet` | simulate | Expected points + headline probabilities |
| `position_matrix.parquet` | simulate | 20×20 P(team finishes position) |
| `sim_team_goals.npy` + `sim_team_order.json` | simulate | Per-run team goals (player model input) |
| `top_scorers.parquet` / `top_assists.parquet` | players | The leaderboards |

`data/cache.sqlite` holds API-Football response cache + the daily request
counter (survives restarts).

---

## 12. Known Gotchas & Design Decisions

1. **FPL rollover** — the FPL API serves *last* season until mid-July.
   `fixture_feed.load_target_fixtures()` auto-detects (checks the first
   kickoff year) and falls back to fixturedownload.com. Player data for
   promoted clubs simply doesn't exist until rollover.
2. **API-Football free tier** — 100 req/day and no live 26/27 data. That is
   why the core pipeline never depends on it; it's a capped, cached extra.
3. **Draw filter semantics** — a draw is essentially never the single most
   likely result, so "filter by predicted draws" would show nothing. The draw
   filter deliberately means "genuinely draw-likely (≥25%)" instead.
4. **Bookmaker odds are a benchmark, not a feature** — training on B365 odds
   would make the model parasitic on the thing it's compared against.
5. **No random train/test splits** — all evaluation is chronological
   (walk-forward). Random splits leak future form into the past.
6. **Streamlit `img { max-width:100% }`** collapses images to zero width
   inside shrink-to-fit containers (table cells, fixed divs). Any `<img>` in
   injected HTML needs an explicit width or `max-width:none`. This bit us
   twice (table crests, brand bar).
7. **`st.dataframe` canvas blanking** — the glide-data-grid sometimes fails to
   repaint after tab switches. The league table is custom HTML for this
   reason (and it looks better).
8. **Logo assets** — the source image was a JPEG with the transparency
   checkerboard baked in; `app/assets/*.png` are processed copies (alpha
   recovered from ink luminance). The white version is used because the
   purple original vanishes on the dark theme. The PL lion is a Premier League
   trademark — fine for a personal project, replace it if the project ever
   becomes commercial.
9. **fixturedownload.com** rejects default HTTP user agents — the client
   sends a browser UA string.
10. **LightGBM on macOS** requires OpenMP: `brew install libomp`.
11. **Fixed RNG seeds** (2627 for the season sim, 1857 for players) — results
    are reproducible run-to-run; change the seeds to explore sim variance.

---

## 13. Extending the System

Ideas, in rough order of impact:

- **xG-based training data** — replace goals with expected goals from
  Understat/FBref in the DC likelihood; xG is less noisy than goals.
- **Injury/availability awareness** — the FPL `status` field is already
  fetched; down-weight teams missing key players in `predict`.
- **In-season market blend** — pull live odds and blend them in (clearly the
  fastest route below 1.0 log-loss, at the cost of independence).
- **Automatic weekly updates** — wrap `plpredict update` in a scheduled job
  (cron / launchd / a Claude Code scheduled task) every Monday in season.
- **Prediction history tracking** — snapshot `match_predictions.parquet`
  before each update to chart how forecasts evolved (great content for the
  Model tab).
- **Cup/European congestion features** — rest-day effects are captured, but
  explicit European-fixture flags could sharpen midweek-affected predictions.

Where to hook in: new data sources go in `src/plpredict/sources/`, new
features in `features.py` (add the column name to `FEATURE_COLS`), new
outputs get a writer in the pipeline module + a loader used by
`app/dashboard.py`.

---

## 14. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| Dashboard says "No model outputs found" | Run `uv run plpredict update` first. |
| `lib_lightgbm.dylib ... libomp.dylib` error | `brew install libomp`. |
| Player pages show placeholders / promoted clubs have no squad | FPL API hasn't rolled over yet, or the CDN lacks a photo for a new player. Resolves itself; run `update` after mid-July. |
| Fixture feed returns 403 | fixturedownload.com UA blocking — confirm the browser UA header is intact in `fixture_feed.py`. |
| `DailyLimitReached` from API-Football | You hit the self-imposed 95-req cap. Wait for the UTC reset or rely on the cache (`plpredict status` shows the count). |
| A team appears twice / doesn't merge across sources | Missing alias — add it to `TEAM_ALIASES` in `config.py`, then `fetch` again. |
| Predictions look stale after a gameweek | You ran the dashboard but not `update`. The dashboard only displays artifacts. |
| Images invisible in custom HTML | Streamlit's global `img{max-width:100%}` — give the `<img>` explicit dimensions (see §12.6). |
| Want a clean rebuild | Delete `data/processed/` (keep `data/raw/` to avoid re-downloading history) and run `update`. |

---

*Repo: https://github.com/Vimdawg/premier-league-predictor · Built July 2026.*
