"""Backtester cost-accounting tests: every dollar must be accounted for."""

import numpy as np
import pytest

from stockmarket.agents.backtest import run_backtest


def test_buy_sell_cash_arithmetic_with_fees():
    prices = np.array([100.0, 110.0, 120.0])
    actions = [1, 0, 2]  # buy at 100, sell at 120, 10 bps per side
    r = run_backtest(prices, actions, initial_money=10000, fee_bps=10)
    expected_cash = 10000 - 100 * 1.001 + 120 * 0.999
    assert r['equity'][-1] == pytest.approx(expected_cash)
    assert r['total_gains'] == pytest.approx(expected_cash - 10000)
    assert r['buys'] == [0] and r['sells'] == [2]
    assert r['n_trades'] == 2


def test_zero_fees_roundtrip_is_pure_price_diff():
    prices = np.array([50.0, 60.0])
    r = run_backtest(prices, [1, 2], initial_money=1000, fee_bps=0)
    assert r['total_gains'] == pytest.approx(10.0)


def test_cannot_sell_with_empty_inventory():
    prices = np.array([100.0, 100.0, 100.0])
    r = run_backtest(prices, [2, 2, 2], initial_money=1000)
    assert r['sells'] == [] and r['n_trades'] == 0
    assert r['equity'][-1] == pytest.approx(1000)


def test_cannot_buy_without_cash():
    prices = np.array([100.0, 100.0])
    r = run_backtest(prices, [1, 1], initial_money=100.0, fee_bps=10)
    # 100 cash cannot cover 100 * 1.001
    assert r['buys'] == []
    assert r['equity'][-1] == pytest.approx(100.0)


def test_equity_marks_inventory_to_market():
    prices = np.array([100.0, 150.0, 50.0])
    r = run_backtest(prices, [1, 0, 0], initial_money=1000, fee_bps=0)
    assert r['equity'][1] == pytest.approx(900 + 150)
    assert r['equity'][2] == pytest.approx(900 + 50)


def test_buy_hold_roi():
    prices = np.array([100.0, 200.0])
    r = run_backtest(prices, [0, 0], initial_money=1000)
    assert r['buy_hold_roi'] == pytest.approx(100.0)


def test_max_drawdown_negative_or_zero():
    prices = np.linspace(100, 80, 10)
    r = run_backtest(prices, [1] + [0] * 9, initial_money=1000, fee_bps=0)
    assert r['max_drawdown'] <= 0


def test_trading_env_matches_backtester_fees():
    """One buy step in the gym env must debit exactly price * (1 + fee)."""
    from stockmarket.agents.env import TradingEnv

    prices = np.full(40, 100.0)
    env = TradingEnv(prices, window=5, initial_money=10000, fee_bps=10)
    env.reset()
    env.step(1)
    assert env.cash == pytest.approx(10000 - 100 * 1.001)
    assert env.inventory == 1
    env.step(2)
    assert env.cash == pytest.approx(10000 - 100 * 1.001 + 100 * 0.999)
    assert env.inventory == 0


def test_trading_env_reward_is_log_value_change():
    from stockmarket.agents.env import TradingEnv

    prices = np.concatenate([np.full(30, 100.0), [100.0, 110.0, 121.0]])
    env = TradingEnv(prices, window=5, initial_money=10000, fee_bps=0)
    env.reset()
    env.t = 30
    env.prev_value = 10000
    env.step(1)  # buy at prices[30]=100, next price 110
    expected = np.log((10000 - 100 + 110) / 10000)
    _, reward, _, _, _ = env.step(0)  # hold into 121
    assert reward == pytest.approx(np.log((10000 - 100 + 121) / (10000 - 100 + 110)))
