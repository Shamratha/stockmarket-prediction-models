"""Run the full forecasting zoo on one ticker and produce charts + a results
table with significance testing (bootstrap CI on RMSE, Diebold-Mariano test
against the drift baseline).

Usage: python scripts/run_forecasting.py [TICKER]
Env:   TEST_SIZE (default 100), EPOCHS (default 200) for quick smoke runs.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from stockmarket.data import load_prices, log_returns, add_features, train_test_split_series
from stockmarket.evaluation import metrics_from_returns, plot_one_step, plot_multi_step, results_table
from stockmarket.forecasting.base import walk_forward
from stockmarket.forecasting.baselines import DriftForecaster, ARIMAForecaster
from stockmarket.forecasting.xgb import XGBForecaster
from stockmarket.forecasting.torch_models import LSTMForecaster, GRUForecaster, TransformerForecaster
from stockmarket.forecasting.nbeats import NBeatsForecaster
from stockmarket.forecasting.patchtst import PatchTSTForecaster
from stockmarket.stats import diebold_mariano, bootstrap_ci

TICKER = (sys.argv[1] if len(sys.argv) > 1 else 'GOOG').upper()
TK = TICKER.lower()
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

    test_dates = close.index[-TEST_SIZE:]
    last_prices = close.values[-TEST_SIZE - 1:-1]
    true_prices = close.values[-TEST_SIZE:]

    rows = []
    price_errors = {}  # model -> forecast error series (true - pred prices)

    def score(name, preds, true, aligned_last, aligned_true_prices, dates, train_seconds):
        m = metrics_from_returns(true, preds, aligned_last)
        pred_prices = aligned_last * np.exp(np.asarray(preds, dtype=float))
        price_errors[name] = aligned_true_prices - pred_prices
        _, rmse_lo, rmse_hi = bootstrap_ci(aligned_true_prices, pred_prices, 'rmse')
        plot_one_step(name, dates, aligned_true_prices, pred_prices, m,
                      f'forecast-{TK}-{slug(name)}.png')
        rows.append({'model': name, **m, 'rmse_ci_lo': round(rmse_lo, 3),
                     'rmse_ci_hi': round(rmse_hi, 3), 'train_s': round(train_seconds, 1)})
        print(f'{name:12s} rmse {m["rmse"]:8.3f} [{rmse_lo:.2f}, {rmse_hi:.2f}]  '
              f'mape {m["mape"]:5.2f}%  dir {m["dir_acc"]:5.1f}%  ({train_seconds:.0f}s)')

    deep_kw = dict(epochs=EPOCHS)
    models = [
        DriftForecaster(),          # reference model for the DM test — keep first
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
        score(model.name, preds, true, last_prices, true_prices, test_dates,
              time.time() - t0)

        hist = returns[:-TEST_SIZE]
        fc_returns = model.forecast_recursive(hist, HORIZON)
        anchor_price = close.values[-TEST_SIZE - 1]
        path = anchor_price * np.exp(np.cumsum(fc_returns))
        hist_dates = close.index[-TEST_SIZE - 60:-TEST_SIZE]
        hist_price = close.values[-TEST_SIZE - 60:-TEST_SIZE]
        fc_dates = close.index[-TEST_SIZE:-TEST_SIZE + HORIZON]
        plot_multi_step(model.name, hist_dates, hist_price, fc_dates, [path],
                        close.values[-TEST_SIZE:-TEST_SIZE + HORIZON],
                        f'forecast30-{TK}-{slug(model.name)}.png')

    # XGBoost works on the feature matrix, evaluated on the same protocol
    t0 = time.time()
    feats = add_features(df).iloc[:-1]  # last row has no target
    train_f, test_f = train_test_split_series(feats, TEST_SIZE)
    xgb = XGBForecaster()
    xgb.fit_features(train_f)
    preds = xgb.predict_features(test_f)
    true = test_f['target_ret'].values
    xgb_last = close.loc[test_f.index].values
    xgb_true_prices = xgb_last * np.exp(true)
    score('XGBoost', preds, true, xgb_last, xgb_true_prices, test_f.index,
          time.time() - t0)

    # Diebold-Mariano vs the drift baseline (squared-error loss)
    ref = price_errors['Drift']
    for row in rows:
        name = row['model']
        if name == 'Drift':
            row['dm_p_vs_drift'] = None
            continue
        errs = price_errors[name]
        n = min(len(errs), len(ref))
        stat, p = diebold_mariano(errs[-n:], ref[-n:])
        row['dm_p_vs_drift'] = round(p, 3)
        row['dm_stat'] = round(stat, 2)

    table = results_table(rows, csv_name=f'forecasting_results_{TK}.csv')
    print('\n', table.to_string(index=False))
    n_sig = sum(1 for r in rows if r.get('dm_p_vs_drift') is not None and r['dm_p_vs_drift'] < 0.05)
    print(f'\nDM test vs drift: {n_sig}/{len(rows) - 1} models significantly different at p<0.05')


if __name__ == '__main__':
    main()
