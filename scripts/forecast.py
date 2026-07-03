"""Forecast one ticker with one model — quick single-run CLI.

Usage: python scripts/forecast.py TICKER MODEL [HORIZON]
  MODEL in: drift, arima, lstm, gru, transformer, nbeats, patchtst
  e.g. python scripts/forecast.py TSLA patchtst 30
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from marketpulse.data import load_prices, log_returns

MODELS = {
    'drift': ('marketpulse.forecasting.baselines', 'DriftForecaster'),
    'arima': ('marketpulse.forecasting.baselines', 'ARIMAForecaster'),
    'lstm': ('marketpulse.forecasting.torch_models', 'LSTMForecaster'),
    'gru': ('marketpulse.forecasting.torch_models', 'GRUForecaster'),
    'transformer': ('marketpulse.forecasting.torch_models', 'TransformerForecaster'),
    'nbeats': ('marketpulse.forecasting.nbeats', 'NBeatsForecaster'),
    'patchtst': ('marketpulse.forecasting.patchtst', 'PatchTSTForecaster'),
}


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'GOOG'
    model_key = (sys.argv[2] if len(sys.argv) > 2 else 'patchtst').lower()
    horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    mod_name, cls_name = MODELS[model_key]
    import importlib

    cls = getattr(importlib.import_module(mod_name), cls_name)
    model = cls()

    df = load_prices(ticker)
    close = df['Close']
    rets = log_returns(close).values.astype(np.float32)

    print(f'training {cls_name} on {len(rets)} days of {ticker}...')
    model.fit(rets)
    fc = model.forecast_recursive(rets, horizon)
    prices = close.values[-1] * np.exp(np.cumsum(fc))

    print(f'\n{ticker} close today: {close.values[-1]:.2f}')
    for i, p in enumerate(prices, 1):
        print(f'  day +{i:2d}: {p:10.2f}  ({(p / close.values[-1] - 1) * 100:+.2f}%)')
    print('\nNote: recursive forecasts compound uncertainty — treat as a scenario, not truth.')


if __name__ == '__main__':
    main()
