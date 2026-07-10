"""Streamlit dashboard for the PL 26/27 prediction model.

Run with: uv run streamlit run app/dashboard.py

Navigation: the main view has four tabs; clicking any club or player anywhere
routes (via URL query params, so pages are deep-linkable) to a detail page —
clubs get recent results, upcoming fixtures, model ratings, and a squad
rotation table; players get season stats, 26/27 projections, and a
minutes-per-gameweek rotation chart.
"""

from __future__ import annotations

import json
from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from plpredict import players as players_mod
from plpredict import predict, simulate
from plpredict.models.dixon_coles import MODEL_PATH as DC_MODEL_PATH
from plpredict.models.dixon_coles import DixonColesModel
from plpredict.models.ensemble import ENSEMBLE_PATH
from plpredict.sources import fpl
from plpredict.sources.football_data import MATCHES_PARQUET
from plpredict.sources.fpl import PLAYERS_PARQUET, TEAMS_PARQUET

st.set_page_config(page_title="PL 26/27 Predictor", page_icon="⚽", layout="wide")

# ---------------------------------------------------------------- palette
SURFACE = "#191427"
C_HOME, C_DRAW, C_AWAY = "#3987e5", "#575170", "#d95926"
ACCENT = "#00ff85"
SEQ_BLUE = [[0.0, SURFACE], [0.25, "#14315e"], [0.55, "#1c5cab"],
            [0.8, "#3987e5"], [1.0, "#8fc0f2"]]

FALLBACK_BADGE = {"Ipswich Town": 40, "Hull City": 88}
FALLBACK_SHORT = {"Ipswich Town": "IPS", "Hull City": "HUL", "Coventry City": "COV"}
STATUS_TEXT = {"a": "Available", "d": "Doubtful", "i": "Injured",
               "s": "Suspended", "u": "Unavailable", "n": "Not in squad"}

st.markdown("""
<style>
#MainMenu, footer, [data-testid="stToolbar"] {visibility: hidden;}
.block-container {padding-top: 1.2rem; max-width: 1200px;}

a.qlink {color: inherit; text-decoration: none;}
a.qlink:hover {color: #00ff85;}
a.back {color: #8d88a3; text-decoration: none; font-size: .85rem;}
a.back:hover {color: #00ff85;}

.hero {
  background: linear-gradient(120deg, #2d0f45 0%, #16102b 55%, #0e2231 100%);
  border: 1px solid rgba(255,255,255,.07); border-radius: 18px;
  padding: 26px 30px 22px; margin-bottom: 6px;
}
.hero h1 {margin: 0; font-size: 1.9rem; letter-spacing: -.02em; color: #fff;}
.hero .sub {color: #b9b3cf; font-size: .92rem; margin-top: 4px;}
.hero .sub b {color: #00ff85; font-weight: 600;}
.hero.compact {padding: 20px 26px; display: flex; align-items: center; gap: 18px;}
.hero.compact img.crest {height: 64px;}
.hero.compact img.face {height: 76px; border-radius: 12px;}
.hero.compact .meta .sub {margin-top: 2px;}

.tiles {display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 18px;}
.tiles.solo {margin-top: 12px;}
.tile {background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.07);
  border-radius: 14px; padding: 12px 16px;}
.tile .k {font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: #8d88a3;}
.tile .v {font-size: 1.12rem; font-weight: 700; color: #fff; margin-top: 3px;
  display: flex; align-items: center; gap: 8px;}
.tile .v img {height: 22px;}
.tile .p {font-size: .8rem; color: #00ff85; margin-top: 1px;}

.mgrid {display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr));
  gap: 12px; margin-top: 10px;}
.mcard {background: #191427; border: 1px solid rgba(255,255,255,.07);
  border-radius: 14px; padding: 14px 16px 12px;}
.mcard .when {font-size: .72rem; color: #8d88a3; text-transform: uppercase;
  letter-spacing: .06em; margin-bottom: 10px;}
.mrow {display: flex; align-items: center; justify-content: space-between; gap: 8px;}
.mteam {display: flex; align-items: center; gap: 8px; flex: 1; min-width: 0;
  font-weight: 600; font-size: .95rem; color: #f5f4fa;}
.mteam.right {justify-content: flex-end;}
.mteam img {height: 26px; flex-shrink: 0;}
.mscore {font-size: 1.05rem; font-weight: 800; color: #fff; background: rgba(255,255,255,.07);
  border-radius: 8px; padding: 3px 10px; white-space: nowrap;}
.pbar {display: flex; height: 8px; border-radius: 4px; overflow: hidden;
  margin-top: 12px; gap: 2px;}
.pbar span {height: 100%;}
.plabels {display: flex; justify-content: space-between; font-size: .74rem;
  color: #b9b3cf; margin-top: 5px;}
.plabels .h {color: #7fb2ee;} .plabels .a {color: #e89a72;}
.xg {font-size: .72rem; color: #8d88a3; margin-top: 6px;}

.prow {display: flex; align-items: center; gap: 12px; background: #191427;
  border: 1px solid rgba(255,255,255,.07); border-radius: 12px;
  padding: 8px 14px 8px 10px; margin-bottom: 8px;}
.prow .rank {width: 20px; text-align: center; color: #8d88a3; font-weight: 700;}
.prow img {height: 44px; width: 35px; object-fit: cover; border-radius: 8px;
  background: rgba(255,255,255,.05);}
.prow .who {flex: 1; min-width: 0;}
.prow .who .nm {font-weight: 650; font-size: .93rem; color: #f5f4fa;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.prow .who .tm {font-size: .74rem; color: #8d88a3;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.prow .stat {text-align: right;}
.prow .stat .xv {font-weight: 750; font-size: 1.0rem; color: #fff;}
.prow .stat .pt {font-size: .72rem; color: #00ff85;}
.prow .track {height: 5px; border-radius: 3px; background: rgba(255,255,255,.06);
  width: 90px;}
.prow .track i {display: block; height: 100%; border-radius: 3px; background: #3987e5;}

.legend {display: flex; gap: 18px; font-size: .8rem; color: #b9b3cf; margin: 6px 0 2px;}
.legend i {display: inline-block; width: 10px; height: 10px; border-radius: 3px;
  margin-right: 6px;}

@media (max-width: 1150px) {
  .prow .track {display: none;}
  .tiles {grid-template-columns: repeat(2, 1fr);}
}

table.ltable {width: 100%; border-collapse: collapse; font-size: .9rem;}
table.ltable th {text-align: left; font-size: .7rem; text-transform: uppercase;
  letter-spacing: .07em; color: #8d88a3; font-weight: 600; padding: 6px 8px;
  border-bottom: 1px solid rgba(255,255,255,.1);}
table.ltable th.num, table.ltable td.num {text-align: right;}
table.ltable td {padding: 5px 8px; border-bottom: 1px solid rgba(255,255,255,.05);
  color: #f5f4fa; white-space: nowrap;}
table.ltable td.pos {color: #8d88a3; font-weight: 700; width: 26px;}
table.ltable img {height: 22px; width: 22px; min-width: 22px; object-fit: contain;
  vertical-align: middle;}
table.ltable img.face {height: 30px; width: 24px; border-radius: 5px; object-fit: cover;}
table.ltable td.tm {font-weight: 600;}
table.ltable .pw {display: inline-block; width: 64px; height: 6px; border-radius: 3px;
  background: rgba(255,255,255,.07); vertical-align: middle; margin-right: 7px;}
table.ltable .pw i {display: block; height: 100%; border-radius: 3px;}
table.ltable .pv {font-size: .8rem; color: #b9b3cf;}
tr.zone-cl td {background: rgba(0,255,133,.05);}
tr.zone-eu td {background: rgba(57,135,229,.07);}
tr.zone-rl td {background: rgba(217,89,38,.08);}

.chip {display: inline-block; min-width: 22px; text-align: center; padding: 1px 6px;
  border-radius: 6px; font-weight: 800; font-size: .74rem;}
.chip.W {background: rgba(0,255,133,.15); color: #7dffc4;}
.chip.D {background: rgba(255,255,255,.1); color: #b9b3cf;}
.chip.L {background: rgba(217,89,38,.18); color: #e89a72;}
.muted {color: #8d88a3; font-size: .8rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_data(ttl=600)
def load_all():
    teams = pd.read_parquet(TEAMS_PARQUET) if TEAMS_PARQUET.exists() else pd.DataFrame()
    players = pd.read_parquet(PLAYERS_PARQUET) if PLAYERS_PARQUET.exists() else pd.DataFrame()
    return (
        simulate.load_sim_table(),
        simulate.load_position_matrix(),
        predict.load_predictions(),
        players_mod.load_scorers(),
        players_mod.load_assists(),
        json.loads(ENSEMBLE_PATH.read_text()),
        teams, players,
    )


@st.cache_data(ttl=3600)
def load_matches() -> pd.DataFrame:
    return pd.read_parquet(MATCHES_PARQUET) if MATCHES_PARQUET.exists() else pd.DataFrame()


@st.cache_data(ttl=3600)
def load_dc_ratings() -> dict:
    if not DC_MODEL_PATH.exists():
        return {}
    m = DixonColesModel.load()
    return {"attack": m.attack, "defense": m.defense}


@st.cache_data(ttl=3600)
def player_history(pid: int) -> pd.DataFrame:
    try:
        return fpl.fetch_player_history(pid)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def player_shares() -> pd.DataFrame:
    if not PLAYERS_PARQUET.exists():
        return pd.DataFrame()
    return players_mod.build_player_shares(pd.read_parquet(PLAYERS_PARQUET))


try:
    sim_table, pos_matrix, preds, scorers, assists, metrics, fpl_teams, fpl_players = load_all()
except FileNotFoundError:
    st.error("No model outputs found — run `uv run plpredict update` first.")
    st.stop()

_team_code = {} if fpl_teams.empty else dict(zip(fpl_teams["team"], fpl_teams["code"]))
_team_short = {} if fpl_teams.empty else dict(zip(fpl_teams["team"], fpl_teams["short_name"]))
_team_code.update({k: v for k, v in FALLBACK_BADGE.items() if k not in _team_code})
_team_short.update({k: v for k, v in FALLBACK_SHORT.items() if k not in _team_short})
_id_short = {} if fpl_teams.empty else dict(zip(fpl_teams["id"], fpl_teams["short_name"]))


def badge(team: str) -> str:
    code = _team_code.get(team)
    if code is not None:
        return f"https://resources.premierleague.com/premierleague/badges/70/t{int(code)}@x2.png"
    initials = _team_short.get(team, team[:3].upper())
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40'>"
           f"<circle cx='20' cy='20' r='19' fill='%23241d3a' stroke='%23575170'/>"
           f"<text x='20' y='25' font-size='12' font-family='sans-serif' font-weight='700'"
           f" fill='%23b9b3cf' text-anchor='middle'>{initials}</text></svg>")
    return "data:image/svg+xml," + quote(svg, safe="'/%:;,= ")


def short(team: str) -> str:
    return _team_short.get(team, team)


_photo_code = {}
_pid = {}
if not fpl_players.empty and "code" in fpl_players.columns:
    for r in fpl_players.itertuples():
        _photo_code[(r.web_name, r.team)] = int(r.code)
        _pid[(r.web_name, r.team)] = int(r.id)


def photo(player: str, team: str) -> str:
    code = _photo_code.get((player, team))
    if code is None:
        return badge(team)
    return f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{code}.png"


def club_href(team: str) -> str:
    return f"?club={quote(team)}"


def club_link(team: str, label: str | None = None) -> str:
    return f'<a class="qlink" href="{club_href(team)}" target="_self">{label or team}</a>'


def player_href(pid: int) -> str:
    return f"?player={pid}"


def player_link(player: str, team: str, label: str | None = None) -> str:
    pid = _pid.get((player, team))
    if pid is None:
        return label or player
    return f'<a class="qlink" href="{player_href(pid)}" target="_self">{label or player}</a>'


def theme_fig(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, sans-serif", color="#b9b3cf", size=13),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.12)")
    return fig


def back_link() -> None:
    st.markdown('<a class="back" href="./" target="_self">← Back to dashboard</a>',
                unsafe_allow_html=True)


# ================================================================ club page
def club_page(team: str) -> None:
    if team not in set(sim_table["team"]):
        st.error(f"Unknown club: {team}")
        back_link()
        return
    back_link()

    row = sim_table[sim_table["team"] == team].iloc[0]
    ratings = load_dc_ratings()
    rank_line = ""
    if ratings:
        teams27 = list(sim_table["team"])
        att_rank = sorted(teams27, key=lambda t: -ratings["attack"].get(t, -9)).index(team) + 1
        def_rank = sorted(teams27, key=lambda t: ratings["defense"].get(t, 9)).index(team) + 1
        rank_line = f"Attack #{att_rank} · Defense #{def_rank} of 20 (model ratings)"

    st.markdown(f"""
<div class="hero compact">
  <img class="crest" src="{badge(team)}"/>
  <div class="meta">
    <h1>{team}</h1>
    <div class="sub">{rank_line}</div>
  </div>
</div>
<div class="tiles solo">
  <div class="tile"><div class="k">Predicted finish</div>
    <div class="v">{int(row['pos'])}{'th' if int(row['pos']) not in (1,2,3) else ['st','nd','rd'][int(row['pos'])-1]}</div>
    <div class="p">{row['exp_pts']:.1f} expected points</div></div>
  <div class="tile"><div class="k">Title</div><div class="v">{row['p_title']:.1%}</div>
    <div class="p">champions in {row['p_title']*10000:.0f} of 10k sims</div></div>
  <div class="tile"><div class="k">Top 4</div><div class="v">{row['p_top4']:.1%}</div>
    <div class="p">Champions League odds</div></div>
  <div class="tile"><div class="k">Relegation</div><div class="v">{row['p_relegation']:.1%}</div>
    <div class="p">drop in {row['p_relegation']*10000:.0f} of 10k sims</div></div>
</div>
""", unsafe_allow_html=True)

    # Finishing-position distribution.
    dist = pos_matrix.loc[team]
    fig = go.Figure(go.Bar(
        x=[str(i) for i in range(1, 21)], y=dist.to_numpy(),
        marker=dict(color=C_HOME, cornerradius=3),
        hovertemplate="P%{x}: %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(yaxis=dict(tickformat=".0%"), xaxis_title="Final position",
                      bargap=0.25)
    st.plotly_chart(theme_fig(fig, 200), use_container_width=True)

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Recent matches")
        matches = load_matches()
        if matches.empty:
            st.info("No match history downloaded yet — run `plpredict fetch`.")
        else:
            mine = matches[(matches["HomeTeam"] == team) | (matches["AwayTeam"] == team)]
            mine = mine.sort_values("Date").tail(10).iloc[::-1]
            trs = []
            for m in mine.itertuples():
                at_home = m.HomeTeam == team
                opp = m.AwayTeam if at_home else m.HomeTeam
                gf, ga = (m.FTHG, m.FTAG) if at_home else (m.FTAG, m.FTHG)
                res = "W" if gf > ga else ("D" if gf == ga else "L")
                comp = "PL" if m.division == "E0" else "Championship"
                trs.append(
                    f'<tr><td class="muted">{m.Date.strftime("%d %b %y")}</td>'
                    f'<td><span class="chip {res}">{res}</span></td>'
                    f'<td class="tm">{"vs" if at_home else "at"} '
                    f'<img src="{badge(opp)}"/> {club_link(opp, short(opp))}</td>'
                    f'<td class="num">{gf}–{ga}</td>'
                    f'<td class="muted">{comp}</td></tr>'
                )
            st.markdown('<table class="ltable"><tbody>' + "".join(trs) + "</tbody></table>",
                        unsafe_allow_html=True)

    with right:
        st.subheader("Next fixtures — predicted")
        mine = preds[(preds["home"] == team) | (preds["away"] == team)]
        mine = mine.sort_values("kickoff_time").head(10)
        trs = []
        for m in mine.itertuples():
            at_home = m.home == team
            opp = m.away if at_home else m.home
            p_win = m.p_home if at_home else m.p_away
            trs.append(
                f'<tr><td class="muted">GW{int(m.gameweek)}</td>'
                f'<td class="tm">{"vs" if at_home else "at"} '
                f'<img src="{badge(opp)}"/> {club_link(opp, short(opp))}</td>'
                f'<td><span class="pw"><i style="width:{p_win:.1%};background:{C_HOME}"></i></span>'
                f'<span class="pv">{p_win:.0%} win</span></td>'
                f'<td class="num muted">{m.likely_score}</td></tr>'
            )
        st.markdown('<table class="ltable"><tbody>' + "".join(trs) + "</tbody></table>",
                    unsafe_allow_html=True)

    st.subheader("Squad & rotation")
    squad = fpl_players[fpl_players["team"] == team] if not fpl_players.empty else pd.DataFrame()
    if squad.empty:
        st.info(
            "No squad data yet for this club — the FPL API still serves last "
            "season and will include newly promoted sides after its July rollover."
        )
        return
    squad = squad.sort_values("minutes", ascending=False)
    squad = squad[squad["minutes"] > 0].head(28)
    max_mins = 38 * 90
    trs = []
    for p in squad.itertuples():
        mshare = min(p.minutes / max_mins, 1.0)
        status = STATUS_TEXT.get(p.status, "")
        pen = " · penalties" if (pd.notna(p.penalties_order) and p.penalties_order == 1) else ""
        trs.append(
            f'<tr><td><img class="face" src="{photo(p.web_name, team)}"/></td>'
            f'<td class="tm">{player_link(p.web_name, team)}</td>'
            f'<td class="muted">{p.position}</td>'
            f'<td><span class="pw"><i style="width:{mshare:.1%};background:{C_HOME}"></i></span>'
            f'<span class="pv">{mshare:.0%} mins</span></td>'
            f'<td class="num">{int(p.starts)} starts</td>'
            f'<td class="num">{int(p.goals_scored)} G</td>'
            f'<td class="num">{int(p.assists)} A</td>'
            f'<td class="muted">{status}{pen}</td></tr>'
        )
    st.markdown(
        '<div class="muted" style="margin-bottom:6px">Last completed season · minutes share '
        'is the rotation signal: starters sit high, rotation options mid, fringe low.</div>'
        '<table class="ltable"><tbody>' + "".join(trs) + "</tbody></table>",
        unsafe_allow_html=True,
    )


# ================================================================ player page
def player_page(pid: int) -> None:
    back_link()
    match = fpl_players[fpl_players["id"] == pid] if not fpl_players.empty else pd.DataFrame()
    if match.empty:
        st.error("Unknown player.")
        return
    p = match.iloc[0]
    team = p["team"]
    full_name = f"{p['first_name']} {p['second_name']}"
    status = STATUS_TEXT.get(p["status"], "")
    chance = p.get("chance_of_playing_next_round")
    if status and status != "Available" and pd.notna(chance):
        status += f" ({int(chance)}% to play)"
    pen = " · takes penalties" if (pd.notna(p["penalties_order"]) and p["penalties_order"] == 1) else ""

    st.markdown(f"""
<div class="hero compact">
  <img class="face" src="{photo(p['web_name'], team)}"/>
  <div class="meta">
    <h1>{full_name}</h1>
    <div class="sub">{p['position']} · <img src="{badge(team)}" style="height:18px;vertical-align:-3px"/>
    {club_link(team)} · {status or 'Available'}{pen}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    xg = pd.to_numeric(p.get("expected_goals"), errors="coerce")
    xa = pd.to_numeric(p.get("expected_assists"), errors="coerce")

    # 26/27 projection: prefer the simulated leaderboards, fall back to
    # share × expected team goals for everyone else.
    in_scorers = scorers[(scorers["player"] == p["web_name"]) & (scorers["team"] == team)]
    in_assists = assists[(assists["player"] == p["web_name"]) & (assists["team"] == team)]
    if not in_scorers.empty:
        proj_g = in_scorers.iloc[0]["exp_goals"]
        proj_note = f"{in_scorers.iloc[0]['p_top_goals']:.0%} Golden Boot"
    else:
        shares = player_shares()
        srow = shares[(shares["web_name"] == p["web_name"]) & (shares["team"] == team)]
        team_row = sim_table[sim_table["team"] == team]
        proj_g = (float(srow.iloc[0]["goal_share"]) * float(team_row.iloc[0]["exp_gf"])
                  if not srow.empty and not team_row.empty else float("nan"))
        proj_note = "share × simulated team goals"
    proj_a = (in_assists.iloc[0]["exp_assists"] if not in_assists.empty else float("nan"))

    mins = int(p["minutes"])
    st.markdown(f"""
<div class="tiles solo">
  <div class="tile"><div class="k">Minutes (25/26)</div><div class="v">{mins:,}</div>
    <div class="p">{int(p['starts'])} starts · {mins / 3420:.0%} of season</div></div>
  <div class="tile"><div class="k">Goals (25/26)</div><div class="v">{int(p['goals_scored'])}</div>
    <div class="p">xG {xg:.1f}</div></div>
  <div class="tile"><div class="k">Assists (25/26)</div><div class="v">{int(p['assists'])}</div>
    <div class="p">xA {xa:.1f}</div></div>
  <div class="tile"><div class="k">Projected 26/27</div>
    <div class="v">{'' if pd.isna(proj_g) else f'{proj_g:.1f} G'}{'' if pd.isna(proj_a) else f' · {proj_a:.1f} A'}</div>
    <div class="p">{proj_note}</div></div>
</div>
""", unsafe_allow_html=True)

    hist = player_history(pid)
    if hist.empty:
        st.info("No per-match history available for this player.")
        return

    st.subheader("Minutes per gameweek — rotation view")
    colors = [ACCENT if m >= 60 else (C_HOME if m > 0 else "rgba(255,255,255,.15)")
              for m in hist["minutes"]]
    fig = go.Figure(go.Bar(
        x=hist["round"], y=hist["minutes"],
        marker=dict(color=colors, cornerradius=2),
        hovertemplate="GW%{x}: %{y} mins<extra></extra>",
    ))
    fig.add_hline(y=60, line_dash="dot", line_color="rgba(255,255,255,.25)")
    fig.update_layout(xaxis_title="Gameweek", yaxis=dict(range=[0, 100], title="Minutes"),
                      bargap=0.2)
    st.plotly_chart(theme_fig(fig, 220), use_container_width=True)
    st.markdown(
        f'<div class="legend"><span><i style="background:{ACCENT}"></i>60+ mins (nailed)</span>'
        f'<span><i style="background:{C_HOME}"></i>Sub / partial</span>'
        '<span><i style="background:rgba(255,255,255,.25)"></i>Unused</span></div>',
        unsafe_allow_html=True,
    )

    st.subheader("Last 10 matches")
    recent = hist.sort_values("kickoff_time").tail(10).iloc[::-1]
    trs = []
    for h in recent.itertuples():
        opp = _id_short.get(h.opponent_team, "?")
        trs.append(
            f'<tr><td class="muted">GW{int(h.round)}</td>'
            f'<td class="tm">{"vs" if h.was_home else "at"} {opp}</td>'
            f'<td class="num">{int(h.minutes)}′</td>'
            f'<td class="num">{int(h.goals_scored)} G</td>'
            f'<td class="num">{int(h.assists)} A</td>'
            f'<td class="num muted">{int(h.total_points)} pts</td></tr>'
        )
    st.markdown('<table class="ltable"><tbody>' + "".join(trs) + "</tbody></table>",
                unsafe_allow_html=True)


# ================================================================ router
qp = st.query_params
if "club" in qp:
    club_page(qp["club"])
    st.stop()
if "player" in qp:
    try:
        player_page(int(qp["player"]))
    except ValueError:
        st.error("Bad player id.")
        back_link()
    st.stop()


# ================================================================ main page
fav = sim_table.iloc[0]
boot = scorers.iloc[0]
playmaker = assists.iloc[0]
releg = sim_table.sort_values("p_relegation", ascending=False).iloc[0]

st.markdown(f"""
<div class="hero">
  <h1>⚽ Premier League 26/27 Predictor</h1>
  <div class="sub">Dixon-Coles + LightGBM hybrid · <b>10,000</b> Monte Carlo season
  simulations · blend <b>{metrics['weight_dc']:.0%}</b> statistical /
  <b>{1 - metrics['weight_dc']:.0%}</b> ML · kicks off <b>21 Aug 2026</b> ·
  click any club or player for details</div>
  <div class="tiles">
    <div class="tile"><div class="k">Title favorite</div>
      <div class="v"><img src="{badge(fav['team'])}"/>{club_link(fav['team'], short(fav['team']))}</div>
      <div class="p">{fav['p_title']:.0%} champion · {fav['exp_pts']:.0f} xPts</div></div>
    <div class="tile"><div class="k">Golden Boot favorite</div>
      <div class="v"><img src="{photo(boot['player'], boot['team'])}" style="border-radius:5px"/>{player_link(boot['player'], boot['team'])}</div>
      <div class="p">{boot['p_top_goals']:.0%} · {boot['exp_goals']:.0f} xG</div></div>
    <div class="tile"><div class="k">Top assists favorite</div>
      <div class="v"><img src="{photo(playmaker['player'], playmaker['team'])}" style="border-radius:5px"/>{player_link(playmaker['player'], playmaker['team'])}</div>
      <div class="p">{playmaker['p_top_assists']:.0%} · {playmaker['exp_assists']:.0f} xA</div></div>
    <div class="tile"><div class="k">Relegation favorite</div>
      <div class="v"><img src="{badge(releg['team'])}"/>{club_link(releg['team'], short(releg['team']))}</div>
      <div class="p">{releg['p_relegation']:.0%} to go down</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

tab_table, tab_matches, tab_players, tab_model = st.tabs(
    ["📋 Final table", "🗓️ Match predictions", "🥇 Golden Boot & assists", "📈 Model"]
)

with tab_table:
    left, right = st.columns([13, 12], gap="large")
    with left:
        st.subheader("Most likely final table")

        def pbar(v: float, color: str) -> str:
            return (f'<span class="pw"><i style="width:{v:.1%};background:{color}"></i></span>'
                    f'<span class="pv">{v:.0%}</span>')

        trs = []
        for r in sim_table.itertuples():
            zone = "zone-cl" if r.pos <= 4 else ("zone-eu" if r.pos <= 6 else ("zone-rl" if r.pos >= 18 else ""))
            trs.append(
                f'<tr class="{zone}"><td class="pos">{r.pos}</td>'
                f'<td><a class="qlink" href="{club_href(r.team)}" target="_self"><img src="{badge(r.team)}"/></a></td>'
                f'<td class="tm">{club_link(r.team)}</td>'
                f'<td class="num">{r.exp_pts:.1f}</td>'
                f'<td>{pbar(r.p_title, ACCENT)}</td>'
                f'<td>{pbar(r.p_top4, C_HOME)}</td>'
                f'<td>{pbar(r.p_relegation, C_AWAY)}</td></tr>'
            )
        st.markdown(
            '<table class="ltable"><thead><tr><th>#</th><th></th><th>Team</th>'
            '<th class="num">xPts</th><th>Title</th><th>Top 4</th><th>Drop</th></tr></thead>'
            "<tbody>" + "".join(trs) + "</tbody></table>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="legend"><span><i style="background:rgba(0,255,133,.5)"></i>Champions League</span>'
            '<span><i style="background:rgba(57,135,229,.6)"></i>Europe</span>'
            '<span><i style="background:rgba(217,89,38,.6)"></i>Relegation</span></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.subheader("Where each team can finish")
        z = pos_matrix.to_numpy()
        fig = go.Figure(go.Heatmap(
            z=z, x=[f"{i}" for i in range(1, 21)],
            y=[short(t) for t in pos_matrix.index],
            colorscale=SEQ_BLUE, zmin=0, zmax=max(0.4, float(z.max())),
            hovertemplate="%{y} finishes P%{x}: %{z:.1%}<extra></extra>",
            colorbar=dict(tickformat=".0%", outlinewidth=0, thickness=12),
        ))
        fig.update_layout(xaxis_title="Final position", yaxis=dict(autorange="reversed"))
        st.plotly_chart(theme_fig(fig, 760), use_container_width=True)

with tab_matches:
    gws = sorted(int(g) for g in preds["gameweek"].dropna().unique())
    gw = st.select_slider("Gameweek", options=gws, value=gws[0])
    sub = preds[preds["gameweek"] == gw].sort_values("kickoff_time")

    st.markdown(
        f'<div class="legend" style="margin-top:8px">'
        f'<span><i style="background:{C_HOME}"></i>Home win</span>'
        f'<span><i style="background:{C_DRAW}"></i>Draw</span>'
        f'<span><i style="background:{C_AWAY}"></i>Away win</span></div>',
        unsafe_allow_html=True,
    )

    cards = []
    for r in sub.itertuples():
        when = pd.Timestamp(r.kickoff_time).strftime("%a %d %b · %H:%M")
        cards.append(f"""
<div class="mcard">
  <div class="when">{when}</div>
  <div class="mrow">
    <div class="mteam"><img src="{badge(r.home)}"/>{club_link(r.home, short(r.home))}</div>
    <div class="mscore">{r.likely_score.replace('-', ' – ')}</div>
    <div class="mteam right">{club_link(r.away, short(r.away))}<img src="{badge(r.away)}"/></div>
  </div>
  <div class="pbar">
    <span style="width:{r.p_home:.1%};background:{C_HOME}"></span>
    <span style="width:{r.p_draw:.1%};background:{C_DRAW}"></span>
    <span style="width:{r.p_away:.1%};background:{C_AWAY}"></span>
  </div>
  <div class="plabels"><span class="h">{r.p_home:.0%}</span><span>{r.p_draw:.0%}</span><span class="a">{r.p_away:.0%}</span></div>
  <div class="xg">xG {r.xg_home:.1f} – {r.xg_away:.1f} · most likely score shown</div>
</div>""")
    st.markdown('<div class="mgrid">' + "".join(cards) + "</div>", unsafe_allow_html=True)

with tab_players:
    st.caption(
        "Each team's simulated goals are shared out to players by expected share. "
        "Newly promoted clubs appear once the FPL API rolls over to 26/27. "
        "Click a player for stats, projections, and rotation history."
    )
    col1, col2 = st.columns(2, gap="large")
    for col, board, kind, title in [
        (col1, scorers, "goals", "🥇 Golden Boot"),
        (col2, assists, "assists", "🅰️ Most assists"),
    ]:
        with col:
            st.subheader(title)
            top = board.head(12)
            vmax = float(top[f"exp_{kind}"].max())
            rows = []
            for i, r in enumerate(top.itertuples(), 1):
                exp_v = getattr(r, f"exp_{kind}")
                p_top = getattr(r, f"p_top_{kind}")
                rows.append(f"""
<div class="prow">
  <div class="rank">{i}</div>
  <img src="{photo(r.player, r.team)}" onerror="this.src='{badge(r.team)}'"/>
  <div class="who"><div class="nm">{player_link(r.player, r.team)}</div>
    <div class="tm">{club_link(r.team)}</div></div>
  <div class="track"><i style="width:{exp_v / vmax:.0%}"></i></div>
  <div class="stat"><div class="xv">{exp_v:.1f}</div><div class="pt">{p_top:.0%} top</div></div>
</div>""")
            st.markdown("".join(rows), unsafe_allow_html=True)

with tab_model:
    st.subheader("Backtest — 760 held-out matches (24/25 + 25/26)")
    st.caption(
        "Log-loss on out-of-sample W/D/L predictions (lower is better). The blend "
        "beats both components; bookmaker closing odds are the practical ceiling "
        "for public-data models."
    )
    rows = [
        ("Uniform (1/3 each)", metrics["logloss_uniform"], C_DRAW),
        ("LightGBM alone", metrics["logloss_ml"], C_HOME),
        ("Dixon-Coles alone", metrics["logloss_dc"], C_HOME),
        ("Ensemble blend", metrics["logloss_blend"], ACCENT),
        ("Bet365 closing odds", metrics.get("logloss_bookmaker", float("nan")), C_AWAY),
    ]
    fig = go.Figure(go.Bar(
        y=[r[0] for r in rows][::-1], x=[r[1] for r in rows][::-1],
        orientation="h",
        marker=dict(color=[r[2] for r in rows][::-1], cornerradius=4),
        text=[f"{r[1]:.4f}" for r in rows][::-1], textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(xaxis=dict(title="Log-loss", range=[0.9, 1.15]))
    st.plotly_chart(theme_fig(fig, 320), use_container_width=True)

    st.subheader("Season tracking")
    st.info(
        "Once the season kicks off (21 Aug 2026), run `uv run plpredict update` "
        "after each gameweek: played results are banked, models refit, and this "
        "page will chart prediction accuracy against actual outcomes."
    )
