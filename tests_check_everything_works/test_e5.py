"""E5 must give valid forecasts whichever linear model AutoARIMA selects."""
import numpy as np
import pytest

import extensions as ex


class _Model:
    def __init__(self, d, ma):
        self.d, self.ma_lags, self.ar_lags, self.phi, self.theta0, self.constant = d, ma, [1], np.array([0.5]), 0.0, True


def _fake_auto(d, ma):
    class Fake:
        def __init__(self, verbose=False):
            pass

        def fit(self, y):
            self.model = _Model(d, ma)
            return self
    return Fake


@pytest.mark.parametrize("d,ma", [(0, [1]), (1, [1, 2]), (2, []), (2, [1])])
def test_e5_runs_when_arima_is_not_pure_ar(monkeypatch, d, ma):
    monkeypatch.setattr(ex, "AutoARIMA", _fake_auto(d, ma))
    y = np.cumsum(np.random.RandomState(0).normal(size=140)) + 50
    m = ex.JointARNN(2, 2, seed=0).fit(y[:120])
    assert m.d == min(d, 1)
    f = m.predict_one_step(y)[120:]
    assert np.isfinite(f).all()


def test_e5_fallback_has_no_look_ahead(monkeypatch):
    monkeypatch.setattr(ex, "AutoARIMA", _fake_auto(1, [1]))
    y = np.cumsum(np.random.RandomState(1).normal(size=140)) + 50
    m = ex.JointARNN(2, 2, seed=0).fit(y[:120])
    base, z = m.predict_one_step(y), y.copy()
    z[130] += 25.0
    assert np.allclose(base[:131], m.predict_one_step(z)[:131], equal_nan=True)


def test_pure_ar_recovers_coefficient():
    rng, y = np.random.RandomState(2), [0.0]
    for _ in range(3000):
        y.append(0.7 * y[-1] + rng.normal())
    lags, const, phi = ex._pure_ar(np.array(y), 0)
    assert lags == [1] and abs(phi[0] - 0.7) < 0.05 and abs(const) < 0.1


def test_pure_ar_is_empty_on_white_noise():
    lags, _, phi = ex._pure_ar(np.random.RandomState(3).normal(size=500), 0)
    assert lags == [] and len(phi) == 0
