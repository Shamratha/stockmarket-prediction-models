"""Run Monte Carlo simulations + portfolio optimization, save charts.

Usage: python scripts/run_simulations.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from marketpulse.data import load_prices, log_returns
from marketpulse.simulation.monte_carlo import gbm_paths, dynamic_vol_paths, correlated_paths
from marketpulse.simulation.portfolio import random_portfolios, optimize, portfolio_stats

sns.set_theme(style='whitegrid')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'output')
os.makedirs(OUT, exist_ok=True)

TICKERS = ['GOOG', 'TSLA', 'AMD', 'MSFT', 'AAPL']


def fan_chart(title, close, paths, fname, hist_days=120):
    plt.figure(figsize=(14, 5))
    hist = close[-hist_days:]
    x_hist = np.arange(len(hist))
    x_fc = np.arange(len(hist), len(hist) + paths.shape[1])
    plt.plot(x_hist, hist, color='black', lw=1.5, label='history')
    for p in paths[:60]:
        plt.plot(x_fc, p, lw=0.5, alpha=0.25, color='crimson')
    for q, style in ((5, ':'), (50, '-'), (95, ':')):
        plt.plot(x_fc, np.percentile(paths, q, axis=0), color='navy', ls=style,
                 lw=1.5, label=f'p{q}')
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(OUT, fname), bbox_inches='tight', dpi=110)
    plt.close()


def main():
    goog = load_prices('GOOG')['Close'].values

    paths = gbm_paths(goog, horizon=252, n_paths=300)
    fan_chart('GBM Monte Carlo — GOOG, 252 trading days ahead (300 paths)',
              goog, paths, 'sim-gbm.png')

    paths = dynamic_vol_paths(goog, horizon=252, n_paths=300)
    fan_chart('Dynamic-volatility (EWMA) Monte Carlo — GOOG, 252 days ahead',
              goog, paths, 'sim-dynamic-vol.png')

    closes = {}
    for t in TICKERS:
        try:
            closes[t] = load_prices(t)['Close'].values
        except Exception as e:
            print(f'[sim] skipping {t}: {e}')
    names, multi = correlated_paths(closes, horizon=252, n_paths=100)
    plt.figure(figsize=(14, 5))
    for i, n in enumerate(names):
        norm = multi[:, :, i] / closes[n][-1]
        plt.plot(np.percentile(norm, 50, axis=0), lw=1.6, label=f'{n} median')
    plt.title('Correlated multi-asset Monte Carlo — median normalized paths')
    plt.legend()
    plt.savefig(os.path.join(OUT, 'sim-multi-asset.png'), bbox_inches='tight', dpi=110)
    plt.close()

    # portfolio optimization on daily log returns
    rets = np.column_stack([
        log_returns(load_prices(t)['Close']).values[-500:] for t in names
    ])
    mean_rets, cov = rets.mean(axis=0), np.cov(rets.T)
    w, r, v, s = random_portfolios(mean_rets, cov)
    w_sharpe = optimize(mean_rets, cov, 'sharpe')
    w_minvol = optimize(mean_rets, cov, 'minvol')

    plt.figure(figsize=(10, 7))
    sc = plt.scatter(v, r, c=s, cmap='viridis', s=4, alpha=0.5)
    plt.colorbar(sc, label='Sharpe')
    for wx, label, color in ((w_sharpe, 'max Sharpe', 'red'), (w_minvol, 'min variance', 'blue')):
        ret, vol, sh = portfolio_stats(wx, mean_rets, cov)
        plt.scatter([vol], [ret], color=color, marker='*', s=400, edgecolors='black',
                    label=f'{label} (Sharpe {sh:.2f})')
    plt.xlabel('annualized volatility')
    plt.ylabel('annualized return')
    plt.title(f'Efficient frontier — {", ".join(names)} (20k random portfolios)')
    plt.legend()
    plt.savefig(os.path.join(OUT, 'sim-efficient-frontier.png'), bbox_inches='tight', dpi=110)
    plt.close()

    print('optimal weights (max Sharpe):')
    for n, wt in zip(names, w_sharpe):
        print(f'  {n}: {wt * 100:.1f}%')
    print('simulation charts saved to output/')


if __name__ == '__main__':
    main()
