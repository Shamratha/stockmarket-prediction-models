"""Honest forecast evaluation: one-step-ahead walk-forward metrics and charts.

Every forecaster predicts the NEXT-day log return given only past data. We
score RMSE / MAPE on reconstructed prices plus directional accuracy on the
returns — the number that actually matters for trading.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style='whitegrid')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, 'output')


def metrics_from_returns(true_ret, pred_ret, last_prices):
    """Score predicted next-day log returns against realized ones.

    last_prices: price at day t (aligned with predictions for day t+1),
    used to reconstruct predicted prices for RMSE/MAPE.
    """
    true_ret = np.asarray(true_ret, dtype=float)
    pred_ret = np.asarray(pred_ret, dtype=float)
    last_prices = np.asarray(last_prices, dtype=float)

    true_price = last_prices * np.exp(true_ret)
    pred_price = last_prices * np.exp(pred_ret)

    rmse = float(np.sqrt(np.mean((true_price - pred_price) ** 2)))
    mape = float(np.mean(np.abs((true_price - pred_price) / true_price)) * 100)
    dir_acc = float(np.mean(np.sign(true_ret) == np.sign(pred_ret)) * 100)
    return {'rmse': rmse, 'mape': mape, 'dir_acc': dir_acc}


def plot_one_step(name, dates, true_price, pred_price, m, fname):
    plt.figure(figsize=(14, 5))
    plt.plot(dates, true_price, label='actual close', color='black', lw=1.6)
    plt.plot(dates, pred_price, label='predicted close (1-step ahead)', color='crimson', lw=1.2, alpha=0.85)
    plt.title(f'{name} — walk-forward next-day prediction | '
              f'RMSE {m["rmse"]:.2f} | MAPE {m["mape"]:.2f}% | direction {m["dir_acc"]:.1f}%')
    plt.legend()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, bbox_inches='tight', dpi=110)
    plt.close()
    return path


def plot_multi_step(name, hist_dates, hist_price, fc_dates, paths, true_future, fname):
    """Recursive multi-step forecast fan chart (clearly labeled as recursive)."""
    plt.figure(figsize=(14, 5))
    plt.plot(hist_dates, hist_price, color='black', lw=1.4, label='history')
    if true_future is not None:
        plt.plot(fc_dates, true_future, color='dimgray', lw=1.6, ls='--', label='actual (held out)')
    for i, p in enumerate(paths):
        plt.plot(fc_dates, p, lw=1.0, alpha=0.8,
                 label='recursive forecast' if i == 0 else None)
    plt.title(f'{name} — 30-day recursive forecast (uncertainty compounds each step)')
    plt.legend()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, bbox_inches='tight', dpi=110)
    plt.close()
    return path


def results_table(rows, csv_name='forecasting_results.csv'):
    df = pd.DataFrame(rows).sort_values('rmse').reset_index(drop=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, csv_name), index=False)
    return df
