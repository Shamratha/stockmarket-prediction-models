"""Single-asset backtester with transaction costs.

Agents emit one action per day: 0 = hold, 1 = buy one unit, 2 = sell one unit
(the original repo's convention, kept because it makes every agent comparable),
charged `fee_bps` per side. Produces the classic buy/sell signal chart plus an
equity curve vs buy & hold, and risk metrics.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style='whitegrid')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(ROOT, 'output')


def run_backtest(prices, actions, initial_money=10000.0, fee_bps=10.0, verbose=False):
    """Execute an action sequence against a price series."""
    prices = np.asarray(prices, dtype=float)
    fee = fee_bps / 1e4
    cash = initial_money
    inventory = 0
    buys, sells = [], []
    equity = np.zeros(len(prices))

    for t, action in enumerate(actions):
        price = prices[t]
        if action == 1 and cash >= price * (1 + fee):
            cash -= price * (1 + fee)
            inventory += 1
            buys.append(t)
            if verbose:
                print(f'day {t}: buy 1 unit at {price:.2f}, balance {cash:.2f}')
        elif action == 2 and inventory > 0:
            cash += price * (1 - fee)
            inventory -= 1
            sells.append(t)
            if verbose:
                print(f'day {t}: sell 1 unit at {price:.2f}, balance {cash:.2f}')
        equity[t] = cash + inventory * price

    total_gains = equity[-1] - initial_money
    roi = total_gains / initial_money * 100

    eq_ret = np.diff(equity) / equity[:-1]
    sharpe = float(np.mean(eq_ret) / (np.std(eq_ret) + 1e-12) * np.sqrt(252))
    peak = np.maximum.accumulate(equity)
    max_dd = float(((equity - peak) / peak).min() * 100)

    bh_units = initial_money / prices[0]
    bh_roi = (bh_units * prices[-1] - initial_money) / initial_money * 100

    return {
        'equity': equity, 'buys': buys, 'sells': sells,
        'total_gains': float(total_gains), 'roi': float(roi),
        'sharpe': sharpe, 'max_drawdown': max_dd, 'buy_hold_roi': float(bh_roi),
        'n_trades': len(buys) + len(sells),
    }


def plot_backtest(name, prices, result, fname, dates=None):
    x = np.arange(len(prices)) if dates is None else dates
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]}
    )
    ax1.plot(x, prices, color='crimson', lw=1.6)
    ax1.plot(x, prices, '^', markersize=8, color='m',
             label='buy', markevery=result['buys'])
    ax1.plot(x, prices, 'v', markersize=8, color='k',
             label='sell', markevery=result['sells'])
    ax1.set_title(
        f'{name} — gains {result["total_gains"]:.2f} ({result["roi"]:.2f}%) vs buy&hold '
        f'{result["buy_hold_roi"]:.2f}% | Sharpe {result["sharpe"]:.2f} | '
        f'maxDD {result["max_drawdown"]:.1f}% | {result["n_trades"]} trades'
    )
    ax1.legend()

    equity = result['equity']
    bh = prices / prices[0] * equity[0]
    ax2.plot(x, equity, label='agent equity', color='navy', lw=1.4)
    ax2.plot(x, bh, label='buy & hold', color='gray', ls='--', lw=1.2)
    ax2.legend()
    ax2.set_ylabel('portfolio value')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, bbox_inches='tight', dpi=110)
    plt.close()
    return path
