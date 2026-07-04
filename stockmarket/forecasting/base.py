"""Common forecaster interface + walk-forward driver.

A forecaster models next-day log returns. During walk-forward evaluation it
always receives the TRUE history up to day t and predicts the return for
day t+1 — one-step-ahead, no leakage, no recursive error hiding.
"""

import numpy as np


class Forecaster:
    name = 'base'

    def fit(self, train_returns: np.ndarray) -> None:
        raise NotImplementedError

    def predict_history(self, full_returns: np.ndarray, test_size: int) -> np.ndarray:
        """Return one-step-ahead predictions for the last `test_size` days.

        full_returns contains train + test; implementations may only use
        values strictly before the day they are predicting.
        """
        raise NotImplementedError

    def forecast_recursive(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Predict `horizon` future returns by feeding predictions back in."""
        hist = list(history)
        out = []
        for _ in range(horizon):
            nxt = self._predict_one(np.asarray(hist))
            out.append(nxt)
            hist.append(nxt)
        return np.array(out)

    def _predict_one(self, history: np.ndarray) -> float:
        raise NotImplementedError


def walk_forward(forecaster, returns, test_size):
    """Fit on the train split, then produce one-step-ahead test predictions."""
    returns = np.asarray(returns, dtype=np.float32)
    train = returns[:-test_size]
    forecaster.fit(train)
    preds = forecaster.predict_history(returns, test_size)
    true = returns[-test_size:]
    return np.asarray(preds, dtype=float), np.asarray(true, dtype=float)
