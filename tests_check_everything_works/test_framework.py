import numpy as np
import pytest

import hybrid_framework as fw


def _series(n=150):
    rng = np.random.RandomState(0)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + np.sin(y[t - 1]) + 0.3 * rng.normal()
    return y + 10


@pytest.mark.parametrize("name", ["ARIMA", "SARIMA", "ETS", "Theta", "OLS AR", "Naive", "Seasonal naive"])
def test_linear_models_are_causal(name):
    y = _series()
    m = fw.make_linear(name, 12).fit(y[:120])
    base = m.predict_one_step(y)
    z = y.copy(); z[130:] += 50
    assert np.allclose(base[:131], m.predict_one_step(z)[:131], equal_nan=True, atol=1e-6)
    assert np.isfinite(base[120:]).all()


@pytest.mark.parametrize("nl", ["MLP", "SVR", "LightGBM"])
@pytest.mark.parametrize("comb", ["additive", "multiplicative", "joint-input", "validation-weighted"])
def test_combiners_are_causal(nl, comb):
    y = _series()
    m = fw.make_combo("OLS AR", nl, comb, 3, 0, 1).fit(y[:120])
    base = m.predict_one_step(y)
    z = y.copy(); z[130:] += 50
    assert np.allclose(base[:131], m.predict_one_step(z)[:131], equal_nan=True, atol=1e-6)
    assert np.isfinite(base[120:]).all()


def test_gru_runs_or_is_skipped():
    if fw.torch is None:
        assert fw.available_nonlinear(["GRU"]) == []
    else:
        y = _series()
        f = fw.SmallGRU(3, 0).fit(y[:100]).predict_one_step(y)
        assert np.isfinite(f[100:]).all()


def test_additive_equals_linear_plus_network():
    y = _series()
    m = fw.make_combo("Naive", "SVR", "additive", 3, 0, 1).fit(y[:120])
    L, full = m._raw(y)
    assert np.allclose(m.predict_one_step(y)[120:], full[120:])
    assert not np.allclose(full[120:], L[120:])
