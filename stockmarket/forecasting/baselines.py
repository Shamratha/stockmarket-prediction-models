"""Baselines every deep model must beat: drift and ARIMA."""

import numpy as np

from .base import Forecaster


class DriftForecaster(Forecaster):
    """Predict tomorrow's return as the mean return of the last `window` days."""

    name = 'Drift'

    def __init__(self, window=21):
        self.window = window

    def fit(self, train_returns):
        pass

    def predict_history(self, full_returns, test_size):
        preds = []
        for t in range(len(full_returns) - test_size, len(full_returns)):
            preds.append(np.mean(full_returns[max(0, t - self.window):t]))
        return np.array(preds)

    def _predict_one(self, history):
        return float(np.mean(history[-self.window:]))


class ARIMAForecaster(Forecaster):
    """ARIMA on log returns via statsmodels; rolling one-step forecasts
    without refit (state updated with each new observation)."""

    name = 'ARIMA'

    def __init__(self, order=(2, 0, 1)):
        self.order = order
        self.res = None

    def fit(self, train_returns):
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(np.asarray(train_returns, dtype=float), order=self.order)
        self.res = model.fit()

    def predict_history(self, full_returns, test_size):
        res = self.res
        preds = []
        test = np.asarray(full_returns[-test_size:], dtype=float)
        for t in range(test_size):
            preds.append(res.forecast(1)[0])
            res = res.append(test[t: t + 1], refit=False)
        return np.array(preds)

    def _predict_one(self, history):
        return float(self.res.forecast(1)[0])
