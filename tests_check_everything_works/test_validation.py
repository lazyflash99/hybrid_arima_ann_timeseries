import numpy as np
import pytest

import arima_model as am
import validation as v
from ann_model import NeuralNetwork
from data_loader import DATASETS
from hybrid_model import HybridModel


def test_folds_are_chronological_and_inside_training():
    folds = v.validation_folds(200, 2)
    assert folds[0][0] == 160 and folds[-1][1] == 200
    assert all(a < b for a, b in folds) and folds[0][1] == folds[1][0]


def test_refit_origins():
    assert v.refit_origins(100, 130, 3) == [100, 110, 120]
    assert v.refit_origins(100, 130, 1) == [100]


def _series(n=160):
    rng = np.random.RandomState(0)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + rng.normal()
    return y


def _make():
    return HybridModel(am.AutoARIMA(subset=False, verbose=False), 2, 2, seed=0)


def test_walk_forward_has_no_look_ahead():
    y, n = _series(), 120
    base, _ = v.walk_forward(_make, y, n, 2)
    t = n + 25
    z = y.copy()
    z[t:] += 100.0                           # perturb every value from index t onwards
    moved, _ = v.walk_forward(_make, z, n, 2)
    assert np.allclose(base[:t - n + 1][:t - n], moved[:t - n])


def test_refit_only_sees_data_before_block():
    y, n = _series(), 120
    seen = []

    class Spy(NeuralNetwork):
        def fit(self, s):
            seen.append(len(s))
            return super().fit(s)
    v.walk_forward(lambda: Spy(2, 2, seed=0), y, n, 2)
    assert seen == [120, 140]


def test_tuning_never_receives_test_data(monkeypatch):
    """With the test values replaced by NaN, tuning must still return a finite score, which shows it does not read them."""
    d = dict(DATASETS["Sunspot"])
    real = d["load"]
    def load():
        y = real().copy()
        y[d["n_train"]:] = np.nan
        return y
    d["load"] = load
    monkeypatch.setitem(v.DATASETS, "Sunspot", d)
    r = v.tune_job(("Sunspot", "ANN", "quick"))
    assert np.isfinite(r["val_mse"]) and r["params"] is not None


def test_iterated_one_step_matches_predict():
    y = _series()
    m = NeuralNetwork(2, 2, seed=0).fit(y[:120])
    assert np.isclose(v.iterated_forecast(m, y[:130], 1), m.predict_one_step(y[:131])[130])


def test_diebold_mariano_sign_and_equal():
    rng = np.random.RandomState(0)
    bad, good = rng.normal(0, 3, 200), rng.normal(0, 1, 200)
    s, p = v.diebold_mariano(bad, good)
    assert s > 0 and p < 0.01
    assert v.diebold_mariano(good, good) == (0.0, 1.0)


def test_mcs_keeps_best_drops_clearly_worse():
    rng = np.random.RandomState(0)
    res = v.model_confidence_set({"a": rng.normal(1, .1, 200), "b": rng.normal(1, .1, 200), "c": rng.normal(5, .1, 200)}, boot=200)
    assert res["c"][1] is False or res["c"][1] == np.False_
    assert res["a"][1] or res["b"][1]


@pytest.mark.parametrize("model", v.MODELS_FIXED)
def test_every_model_builds_with_registry_params(model):
    d = DATASETS["Sunspot"]
    assert v.build(model, d, v.registry_params(model, d), 0) is not None
