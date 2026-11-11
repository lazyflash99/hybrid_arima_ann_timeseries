import numpy as np

import stationarity as st


def test_decide_rules():
    assert st.decide(0.01, 0.5)[0] == "agree: stationary"
    assert st.decide(0.5, 0.01)[0] == "agree: non-stationary"
    assert st.decide(0.01, 0.01)[0] == "disagree"


def test_seasonal_strength_detects_season():
    t = np.arange(240)
    rng = np.random.RandomState(0)
    assert st.seasonal_strength(np.sin(2 * np.pi * t / 12) + 0.05 * rng.normal(size=240), 12) > 0.9
    assert np.isnan(st.seasonal_strength(rng.normal(size=100), 1))
