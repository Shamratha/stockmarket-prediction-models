"""Leakage tests: no forecaster may see the future it is predicting."""

import numpy as np
import pytest

from marketpulse.data import train_test_split_series
from marketpulse.forecasting.base import walk_forward
from marketpulse.forecasting.baselines import DriftForecaster
from marketpulse.forecasting.torch_models import LSTMForecaster, make_windows

import pandas as pd

rng = np.random.default_rng(0)
RETURNS = rng.normal(0, 0.02, 400).astype(np.float32)


def test_split_is_chronological_and_disjoint():
    s = pd.Series(np.arange(100.0))
    train, test = train_test_split_series(s, test_size=30)
    assert len(train) == 70 and len(test) == 30
    assert train.index.max() < test.index.min()
    assert set(train.index).isdisjoint(test.index)


def test_walk_forward_train_excludes_test():
    seen = {}

    class Spy(DriftForecaster):
        def fit(self, train_returns):
            seen['n_train'] = len(train_returns)

    walk_forward(Spy(), RETURNS, test_size=50)
    assert seen['n_train'] == len(RETURNS) - 50


def test_drift_prediction_ignores_future():
    """Mutating future values must not change earlier predictions."""
    f = DriftForecaster()
    preds_a = f.predict_history(RETURNS.copy(), 50)

    corrupted = RETURNS.copy()
    corrupted[-1] = 99.0  # poison the final day
    preds_b = f.predict_history(corrupted, 50)
    # every prediction except (possibly) none may use day -1;
    # prediction FOR the last day uses history up to day -2
    np.testing.assert_allclose(preds_a, preds_b)


def test_lstm_prediction_ignores_future():
    f = LSTMForecaster(epochs=2, window=16)
    f.fit(RETURNS[:-50])
    preds_a = f.predict_history(RETURNS.copy(), 50)
    corrupted = RETURNS.copy()
    corrupted[-1] = 99.0
    preds_b = f.predict_history(corrupted, 50)
    np.testing.assert_allclose(preds_a, preds_b, rtol=1e-5)


def test_make_windows_alignment():
    """Window i must contain returns [i-w, i) and target return i."""
    r = np.arange(20, dtype=np.float32)
    X, y = make_windows(r, window=5)
    assert X.shape == (15, 5) and y.shape == (15,)
    np.testing.assert_allclose(X[0].numpy(), [0, 1, 2, 3, 4])
    assert y[0] == 5.0
    np.testing.assert_allclose(X[-1].numpy(), [14, 15, 16, 17, 18])
    assert y[-1] == 19.0


def test_feature_target_is_strictly_future():
    """add_features' target at row t must equal the return realized at t+1."""
    from marketpulse.data import add_features

    idx = pd.date_range('2024-01-01', periods=120)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 120))), index=idx)
    df = pd.DataFrame({'Open': close, 'High': close * 1.01,
                       'Low': close * 0.99, 'Close': close,
                       'Volume': np.full(120, 1e6)}, index=idx)
    feats = add_features(df)
    t = feats.index[10]
    t_next = df.index[df.index.get_loc(t) + 1]
    expected = np.log(df.loc[t_next, 'Close'] / df.loc[t, 'Close'])
    assert feats.loc[t, 'target_ret'] == pytest.approx(expected)
