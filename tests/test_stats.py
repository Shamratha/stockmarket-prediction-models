"""Sanity tests for the significance-testing tools."""

import numpy as np
import pytest

from marketpulse.evaluation import metrics_from_returns
from marketpulse.stats import bootstrap_ci, diebold_mariano

rng = np.random.default_rng(7)


def test_dm_identical_errors_not_significant():
    e = rng.normal(0, 1, 100)
    stat, p = diebold_mariano(e, e.copy())
    assert p == pytest.approx(1.0)


def test_dm_detects_clearly_better_model():
    e_good = rng.normal(0, 0.5, 200)
    e_bad = rng.normal(0, 3.0, 200)
    stat, p = diebold_mariano(e_good, e_bad)
    assert stat < 0          # model A has lower loss
    assert p < 0.01


def test_dm_symmetry():
    e1 = rng.normal(0, 1, 150)
    e2 = rng.normal(0, 2, 150)
    s12, p12 = diebold_mariano(e1, e2)
    s21, p21 = diebold_mariano(e2, e1)
    assert s12 == pytest.approx(-s21)
    assert p12 == pytest.approx(p21)


def test_dm_rejects_tiny_samples():
    with pytest.raises(ValueError):
        diebold_mariano(np.ones(5), np.ones(5))


def test_bootstrap_ci_contains_point_estimate():
    true = 100 + rng.normal(0, 5, 120)
    pred = true + rng.normal(0, 2, 120)
    point, lo, hi = bootstrap_ci(true, pred, 'rmse')
    assert lo <= point <= hi
    assert lo > 0


def test_bootstrap_ci_narrows_with_smaller_errors():
    true = np.full(200, 100.0)
    _, lo_a, hi_a = bootstrap_ci(true, true + rng.normal(0, 1, 200), 'rmse')
    _, lo_b, hi_b = bootstrap_ci(true, true + rng.normal(0, 10, 200), 'rmse')
    assert (hi_a - lo_a) < (hi_b - lo_b)


def test_directional_accuracy_perfect_and_inverted():
    true = np.array([0.01, -0.02, 0.03, -0.01])
    last = np.full(4, 100.0)
    m = metrics_from_returns(true, true, last)
    assert m['dir_acc'] == pytest.approx(100.0)
    m = metrics_from_returns(true, -true, last)
    assert m['dir_acc'] == pytest.approx(0.0)
