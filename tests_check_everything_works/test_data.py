import numpy as np
import pytest

from data_loader import DATASETS, mackey_glass, synthetic_additive
from run_experiments import EXPERIMENTS


@pytest.mark.parametrize("name", list(DATASETS))
def test_lengths_and_finite(name):
    d = DATASETS[name]
    y = d["load"]()
    assert len(y) == d["n_train"] + d["n_test"] and np.isfinite(y).all()
    assert max(d["horizons"].values()) == d["n_test"]


def test_registry_matches_paper_experiments():
    for name, cfg in EXPERIMENTS.items():
        assert DATASETS[name]["ann"] == cfg["ann"] and DATASETS[name]["ext_kwargs"] == cfg["ext_kwargs"]


def test_synthetic_reproducible():
    assert np.array_equal(synthetic_additive()[0], synthetic_additive()[0])
    assert np.array_equal(mackey_glass(), mackey_glass())
