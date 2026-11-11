import numpy as np

import plots_decomposition as pd_mod
import residual_diagnostics as rd


def test_nn_linearity_test_detects_nonlinearity():
    rng = np.random.RandomState(0)
    white = rng.normal(size=400)
    assert rd.teraesvirta_test(white)[1] > 0.01
    x = np.zeros(400)
    for t in range(1, 400):
        x[t] = 0.9 * np.sin(2.5 * x[t - 1]) + 0.1 * rng.normal()
    assert rd.teraesvirta_test(x)[1] < 0.01


def test_tests_return_all_columns():
    rng = np.random.RandomState(1)
    e = rng.normal(size=120)
    out = rd.tests(e, rng.normal(size=120))
    for k in ("ljung_box_p_6", "box_pierce_p_12", "jarque_bera_p", "shapiro_p", "arch_lm_p", "bds_p", "nn_linearity_p"):
        assert 0 <= out[k] <= 1


def test_r_squared():
    e = np.array([1.0, 2.0, 3.0])
    assert pd_mod.r_squared(e, e) == 1.0
