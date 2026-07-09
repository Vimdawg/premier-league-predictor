"""Model sanity tests on synthetic data (no network, no fitted artifacts)."""

import numpy as np
import pandas as pd
import pytest

from plpredict.config import canonical_team
from plpredict.models import dixon_coles as dc
from plpredict.predict import _rescale_matrix


@pytest.fixture(scope="module")
def synthetic_model():
    """Fit DC on a synthetic double round-robin where Alpha >> Delta."""
    rng = np.random.default_rng(7)
    strength = {"Alpha": 2.2, "Beta": 1.5, "Gamma": 1.2, "Delta": 0.7}
    teams = list(strength)
    rows = []
    date = pd.Timestamp("2025-08-01")
    for _round in range(6):  # 6 double round-robins for signal
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                rows.append({
                    "Date": date,
                    "HomeTeam": h, "AwayTeam": a,
                    "FTHG": rng.poisson(strength[h] / strength[a] * 1.3),
                    "FTAG": rng.poisson(strength[a] / strength[h]),
                })
                date += pd.Timedelta(hours=8)
    return dc.fit(pd.DataFrame(rows))


def test_score_matrix_is_distribution(synthetic_model):
    m = synthetic_model.score_matrix("Alpha", "Delta")
    assert m.shape == (11, 11)
    assert np.all(m >= 0)
    assert m.sum() == pytest.approx(1.0)


def test_outcome_probs_sum_to_one(synthetic_model):
    probs = synthetic_model.outcome_probs("Beta", "Gamma")
    assert sum(probs) == pytest.approx(1.0)


def test_stronger_team_favored(synthetic_model):
    p_home, _, p_away = synthetic_model.outcome_probs("Alpha", "Delta")
    assert p_home > 0.5 > p_away
    # And the reverse fixture still favors Alpha despite away disadvantage.
    p_home_rev, _, p_away_rev = synthetic_model.outcome_probs("Delta", "Alpha")
    assert p_away_rev > p_home_rev


def test_home_advantage_positive(synthetic_model):
    assert synthetic_model.home_adv > 0


def test_rescale_matrix_hits_target(synthetic_model):
    m = synthetic_model.score_matrix("Beta", "Gamma")
    target = (0.5, 0.3, 0.2)
    out = _rescale_matrix(m, target)
    assert out.sum() == pytest.approx(1.0)
    assert np.tril(out, -1).sum() == pytest.approx(0.5, abs=1e-9)
    assert np.trace(out) == pytest.approx(0.3, abs=1e-9)


def test_canonical_team_aliases():
    assert canonical_team("Man City") == "Manchester City"
    assert canonical_team("Spurs") == "Tottenham Hotspur"
    assert canonical_team("Nott'm Forest") == "Nottingham Forest"
    # Unknown clubs pass through instead of crashing.
    assert canonical_team("Newly Founded FC") == "Newly Founded FC"
