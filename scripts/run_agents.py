"""Backtest every trading agent out-of-sample and produce charts + a table.

Train segment: first 80% of the series. Test segment: last 20% — every chart
and number below is from data the learning agents never saw.

Usage: python scripts/run_agents.py [TICKER]
Env:   ES_ITER (default 200), RL_STEPS (default 100000) for quick smoke runs.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from marketpulse.data import load_prices
from marketpulse.agents.backtest import run_backtest, plot_backtest, OUTPUT_DIR
from marketpulse.agents.rules import turtle_actions, sma_cross_actions, rsi_actions
from marketpulse.agents.evolution import EvolutionAgent
from marketpulse.agents.rl import train_rl_agent, rl_actions

TICKER = sys.argv[1] if len(sys.argv) > 1 else 'GOOG'
ES_ITER = int(os.environ.get('ES_ITER', 200))
RL_STEPS = int(os.environ.get('RL_STEPS', 100_000))
FEE_BPS = 10.0
WINDOW = 30


def main():
    df = load_prices(TICKER)
    close = df['Close'].values.astype(float)
    split = int(len(close) * 0.8)
    train_prices, test_prices = close[:split], close[split:]
    test_dates = df.index[split:]
    print(f'{TICKER}: train {split} days, test {len(test_prices)} days')

    rows = []

    def record(name, actions, train_seconds):
        result = run_backtest(test_prices, actions, fee_bps=FEE_BPS)
        slug = name.lower().replace(' ', '-')
        plot_backtest(name, test_prices, result, f'agent-{slug}.png', dates=test_dates)
        rows.append({
            'agent': name, 'roi_pct': round(result['roi'], 2),
            'buy_hold_pct': round(result['buy_hold_roi'], 2),
            'sharpe': round(result['sharpe'], 2),
            'max_dd_pct': round(result['max_drawdown'], 2),
            'trades': result['n_trades'], 'train_s': round(train_seconds, 1),
        })
        print(f'{name:22s} ROI {result["roi"]:7.2f}%  (B&H {result["buy_hold_roi"]:.2f}%)  '
              f'Sharpe {result["sharpe"]:5.2f}  trades {result["n_trades"]}')

    # rule-based (no training)
    record('Turtle breakout', turtle_actions(test_prices), 0)
    record('SMA crossover', sma_cross_actions(test_prices), 0)
    record('RSI mean-reversion', rsi_actions(test_prices), 0)

    # evolution strategy
    t0 = time.time()
    es = EvolutionAgent(window=WINDOW, fee_bps=FEE_BPS)
    es.train(train_prices, iterations=ES_ITER, print_every=max(1, ES_ITER // 4))
    record('Evolution strategy', es.act_series(test_prices), time.time() - t0)

    # RL agents
    for algo in ('DQN', 'PPO'):
        t0 = time.time()
        model = train_rl_agent(algo, train_prices, window=WINDOW,
                               timesteps=RL_STEPS, fee_bps=FEE_BPS)
        record(algo, rl_actions(model, test_prices, window=WINDOW, fee_bps=FEE_BPS),
               time.time() - t0)

    table = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    table.to_csv(os.path.join(OUTPUT_DIR, 'agent_results.csv'), index=False)
    print('\n', table.to_string(index=False))


if __name__ == '__main__':
    main()
