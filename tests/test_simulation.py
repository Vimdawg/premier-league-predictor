"""Season-simulation invariants on the real fitted artifacts.

Skipped automatically when the pipeline hasn't been run yet.
"""

import numpy as np
import pytest

from plpredict.config import PROCESSED_DIR
from plpredict.simulate import SIM_TABLE_PARQUET

pytestmark = pytest.mark.skipif(
    not SIM_TABLE_PARQUET.exists(), reason="run `plpredict update` first"
)


@pytest.fixture(scope="module")
def artifacts():
    from plpredict import predict, simulate
    from plpredict.sources import fixture_feed

    return (
        simulate.load_sim_table(),
        simulate.load_position_matrix(),
        predict.load_predictions(),
        fixture_feed.load_target_fixtures(),
    )


def test_twenty_teams_and_full_schedule(artifacts):
    table, _, preds, fixtures = artifacts
    assert len(table) == 20
    assert len(fixtures) == 380
    # Every unplayed fixture carries a prediction.
    assert len(preds) == (~fixtures["finished"]).sum()


def test_match_probs_are_distributions(artifacts):
    _, _, preds, _ = artifacts
    total = preds[["p_home", "p_draw", "p_away"]].sum(axis=1)
    assert np.allclose(total, 1.0, atol=1e-6)


def test_position_matrix_rows_sum_to_one(artifacts):
    _, pos_matrix, _, _ = artifacts
    assert np.allclose(pos_matrix.sum(axis=1), 1.0, atol=1e-6)
    # Columns too: exactly one team occupies each position per simulation.
    assert np.allclose(pos_matrix.sum(axis=0), 1.0, atol=1e-6)


def test_expected_points_plausible(artifacts):
    table, _, _, _ = artifacts
    # A 20-team season awards between 2 and 3 points per match on average.
    total_pts = table["exp_pts"].sum()
    assert 380 * 2 <= total_pts <= 380 * 3
    assert table["exp_pts"].max() < 110  # nobody beats the all-time record on average
    assert table["exp_pts"].min() > 5


def test_probability_columns_bounded(artifacts):
    table, _, _, _ = artifacts
    for col in ("p_title", "p_top4", "p_relegation"):
        assert table[col].between(0, 1).all()
    assert table["p_title"].sum() == pytest.approx(1.0, abs=1e-6)
    assert table["p_relegation"].sum() == pytest.approx(3.0, abs=1e-6)


def test_promoted_teams_rated_conservatively(artifacts):
    table, _, _, _ = artifacts
    promoted = table[table["team"].isin(["Coventry City", "Ipswich Town", "Hull City"])]
    assert len(promoted) == 3
    # Promoted sides should project in the bottom half, not mid-table or above.
    assert (promoted["pos"] > 10).all()
