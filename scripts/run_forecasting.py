"""Run the full forecasting zoo on one ticker and produce charts + a results table.

Usage: python scripts/run_forecasting.py [TICKER]
Env:   TEST_SIZE (default 100), EPOCHS (default 200) for quick smoke runs.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from marketpulse.data import load_prices, log_returns, add_features, train_test_split_series
from marketpulse.evaluation import metrics_from_returns, plot_one_step, plot_multi_step, results_table
from marketpulse.forecasting.base import walk_forward
from marketpulse.forecasting.baselines import DriftForecaster, ARIMAForecaster
from marketpulse.forecasting.xgb import XGBForecaster
from marketpulse.forecasting.torch_models import LSTMForecaster, GRUForecaster, TransformerForecaster
from marketpulse.forecasting.nbeats import NBeatsForecaster
from marketpulse.forecasting.patchtst import PatchTSTForecaster

TICKER = sys.argv[1] if len(sys.argv) > 1 else 'GOOG'
TEST_SIZE = int(os.environ.get('TEST_SIZE', 100))
EPOCHS = int(os.environ.get('EPOCHS', 200))
HORIZON = 30


def slug(name):
    return name.lower().replace(' ', '-')


def main():
    df = load_prices(TICKER)
    close = df['Close']
    returns = log_returns(close).values.astype(np.float32)
    print(f'{TICKER}: {len(df)} rows {df.index[0].date()} -> {df.index[-1].date()}')

    # aligned so prediction i targets close index position len-TEST_SIZE+i
    test_dates = close.index[-TEST_SIZE:]
    last_prices = close.values[-TEST_SIZE - 1:-1]
    true_prices = close.values[-TEST_SIZE:]

    rows = []

    def score(name, preds, true, train_seconds):
        m = metrics_from_returns(true, preds, last_prices)
        pred_prices = last_prices * np.exp(preds)
        plot_one_step(name, test_dates, true_prices, pred_prices, m, f'forecast-{slug(name)}.png')
        rows.append({'model': name, **m, 'train_s': round(train_seconds, 1)})
        print(f'{name:12s} rmse {m["rmse"]:8.3f}  mape {m["mape"]:5.2f}%  dir {m["dir_acc"]:5.1f}%  ({train_seconds:.0f}s)')

    deep_kw = dict(epochs=EPOCHS)
    models = [
        DriftForecaster(),
        ARIMAForecaster(),
        LSTMForecaster(**deep_kw),
        GRUForecaster(**deep_kw),
        TransformerForecaster(**deep_kw),
        NBeatsForecaster(**deep_kw),
        PatchTSTForecaster(**deep_kw),
    ]

    for model in models:
        t0 = time.time()
        preds, true = walk_forward(model, returns, TEST_SIZE)
        score(model.name, preds, true, time.time() - t0)

        # recursive 30-day fan chart from the start of the test window
        hist = returns[:-TEST_SIZE]
        fc_returns = model.forecast_recursive(hist, HORIZON)
        anchor_price = close.values[-TEST_SIZE - 1]
        path = anchor_price * np.exp(np.cumsum(fc_returns))
        hist_dates = close.index[-TEST_SIZE - 60:-TEST_SIZE]
        hist_price = close.values[-TEST_SIZE - 60:-TEST_SIZE]
        fc_dates = close.index[-TEST_SIZE:-TEST_SIZE + HORIZON]
        plot_multi_step(model.name, hist_dates, hist_price, fc_dates, [path],
                        close.values[-TEST_SIZE:-TEST_SIZE + HORIZON],
                        f'forecast30-{slug(model.name)}.png')

    # XGBoost works on the feature matrix, evaluated on the same days
    t0 = time.time()
    feats = add_features(df)
    feats = feats.iloc[:-1]  # last row has no target
    train_f, test_f = train_test_split_series(feats, TEST_SIZE)
    xgb = XGBForecaster()
    xgb.fit_features(train_f)
    preds = xgb.predict_features(test_f)
    true = test_f['target_ret'].values
    # align prices for XGB rows: target_ret at row t is return of day t+1
    xgb_last = close.loc[test_f.index].values
    m = metrics_from_returns(true, preds, xgb_last)
    pred_prices = xgb_last * np.exp(preds)
    true_p = xgb_last * np.exp(true)
    plot_one_step('XGBoost', test_f.index, true_p, pred_prices, m, 'forecast-xgboost.png')
    rows.append({'model': 'XGBoost', **m, 'train_s': round(time.time() - t0, 1)})
    print(f'{"XGBoost":12s} rmse {m["rmse"]:8.3f}  mape {m["mape"]:5.2f}%  dir {m["dir_acc"]:5.1f}%')

    table = results_table(rows)
    print('\n', table.to_string(index=False))


if __name__ == '__main__':
    main()
