"""Command-line pipeline: plpredict <fetch|train|predict|simulate|players|update|status>."""

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")


def cmd_fetch(_args) -> None:
    from plpredict.sources import fixture_feed, football_data, fpl

    matches = football_data.fetch_all_matches()
    print(f"matches: {len(matches)} (through {matches['Date'].max().date()})")
    teams, players_df = fpl.fetch_bootstrap()
    print(f"FPL: {len(teams)} teams, {len(players_df)} players")
    fx = fixture_feed.load_target_fixtures(force=True)
    print(f"target-season fixtures: {len(fx)} ({int(fx['finished'].sum())} played)")


def cmd_train(_args) -> None:
    from plpredict.features import build_training_features
    from plpredict.models import ensemble, ml_model
    from plpredict.sources import football_data

    matches = football_data.load_matches()
    feat = build_training_features(matches)
    _, val = ml_model.train_model(feat)
    print("LightGBM trained; running Dixon-Coles walk-forward backtest…")
    val_dc = ensemble.dc_val_probs(matches, val)
    metrics = ensemble.fit_blend(val_dc)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


def cmd_predict(_args) -> None:
    from plpredict import predict
    from plpredict.sources import fixture_feed, football_data

    preds, _ = predict.predict_remaining(
        football_data.load_matches(), fixture_feed.load_target_fixtures()
    )
    print(f"predicted {len(preds)} fixtures → data/processed/match_predictions.parquet")
    gw1 = preds[preds["gameweek"] == preds["gameweek"].min()]
    cols = ["home", "away", "p_home", "p_draw", "p_away", "likely_score"]
    print(gw1[cols].round(3).to_string(index=False))


def cmd_simulate(_args) -> None:
    from plpredict import predict, simulate
    from plpredict.sources import fixture_feed

    fx = fixture_feed.load_target_fixtures()
    summary, _ = simulate.simulate_season(fx, predict.load_score_matrices())
    cols = ["pos", "team", "exp_pts", "p_title", "p_top4", "p_relegation"]
    print(summary[cols].round(3).to_string(index=False))


def cmd_players(_args) -> None:
    from plpredict import players

    scorers, assists = players.simulate_players()
    print("--- Golden Boot ---")
    print(scorers.head(10).round(3).to_string(index=False))
    print("--- Top assists ---")
    print(assists.head(10).round(3).to_string(index=False))


def cmd_update(args) -> None:
    """Weekly refresh: new results in, refit, re-simulate."""
    cmd_fetch(args)
    cmd_train(args)
    cmd_predict(args)
    cmd_simulate(args)
    cmd_players(args)
    print("update complete")


def cmd_status(_args) -> None:
    from plpredict.config import API_FOOTBALL_KEY
    from plpredict.sources import api_football

    print(f"API-Football key set: {bool(API_FOOTBALL_KEY)}")
    print(f"API-Football requests used today: {api_football.requests_used_today()}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="plpredict")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in [
        ("fetch", cmd_fetch), ("train", cmd_train), ("predict", cmd_predict),
        ("simulate", cmd_simulate), ("players", cmd_players),
        ("update", cmd_update), ("status", cmd_status),
    ]:
        sub.add_parser(name).set_defaults(func=fn)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
