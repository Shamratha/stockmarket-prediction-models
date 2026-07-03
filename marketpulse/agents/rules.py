"""Rule-based baseline agents: turtle breakout, SMA crossover, RSI."""

import numpy as np
import pandas as pd


def turtle_actions(close, lookback=None):
    """Breakout: buy above the rolling max, sell below the rolling min."""
    close = pd.Series(np.asarray(close, dtype=float))
    lookback = lookback or int(np.ceil(len(close) * 0.1))
    rolling_max = close.shift(1).rolling(lookback).max()
    rolling_min = close.shift(1).rolling(lookback).min()
    actions = np.zeros(len(close), dtype=int)
    actions[close > rolling_max] = 1
    actions[close < rolling_min] = 2
    return actions


def sma_cross_actions(close, short=None, long=None):
    """Golden cross: buy when short SMA crosses above long SMA, sell on cross down."""
    close = pd.Series(np.asarray(close, dtype=float))
    short = short or max(5, int(0.025 * len(close)))
    long = long or max(10, int(0.05 * len(close)))
    short_ma = close.rolling(short, min_periods=1).mean()
    long_ma = close.rolling(long, min_periods=1).mean()
    signal = (short_ma > long_ma).astype(int)
    cross = signal.diff().fillna(0)
    actions = np.zeros(len(close), dtype=int)
    actions[cross == 1] = 1
    actions[cross == -1] = 2
    return actions


def rsi_actions(close, period=14, oversold=30, overbought=70):
    """Mean reversion: buy oversold, sell overbought."""
    close = pd.Series(np.asarray(close, dtype=float))
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rsi = 100 - 100 / (1 + gain / (loss + 1e-12))
    actions = np.zeros(len(close), dtype=int)
    actions[rsi < oversold] = 1
    actions[rsi > overbought] = 2
    return actions
