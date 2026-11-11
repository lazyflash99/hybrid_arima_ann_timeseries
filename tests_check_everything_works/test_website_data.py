"""The website JSON must parse as strict JSON: no NaN and no Infinity anywhere."""
import glob
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = sorted(glob.glob(os.path.join(ROOT, "website", "data", "*.json")) +
               glob.glob(os.path.join(ROOT, "website", "research_site", "data", "**", "*.json"), recursive=True))


def _reject(token):
    raise ValueError(f"non-finite value {token}")


@pytest.mark.skipif(not FILES, reason="website data not generated")
@pytest.mark.parametrize("path", FILES, ids=[os.path.relpath(p, ROOT) for p in FILES])
def test_json_has_no_nan_or_inf(path):
    with open(path, encoding="utf-8") as f:
        json.load(f, parse_constant=_reject)


def test_synthetic_generators_are_reproducible():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "website"))
    import website_experiments as w
    import numpy as np
    assert np.array_equal(w.additive_ar_sin(1.0, 0.5, 0), w.additive_ar_sin(1.0, 0.5, 0))
    for form in ("additive", "multiplicative"):
        y = w.level_series(form, 0.3, 1)
        assert np.isfinite(y).all() and len(y) == w.N_SERIES
