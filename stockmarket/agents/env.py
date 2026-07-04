"""Gymnasium trading environment for the RL agents.

Observation: the last `window` log returns, plus [inventory_frac, cash_frac].
Actions: 0 hold, 1 buy one unit, 2 sell one unit (same as the backtester).
Reward: change in log portfolio value (fees included).
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TradingEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, prices, window=30, initial_money=10000.0, fee_bps=10.0):
        super().__init__()
        self.prices = np.asarray(prices, dtype=np.float64)
        self.returns = np.diff(np.log(self.prices), prepend=np.log(self.prices[0]))
        self.window = window
        self.initial_money = initial_money
        self.fee = fee_bps / 1e4

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window + 2,), dtype=np.float32
        )
        self.reset()

    def _obs(self):
        r = self.returns[self.t - self.window + 1: self.t + 1]
        port = self.cash + self.inventory * self.prices[self.t]
        extra = np.array([
            self.inventory * self.prices[self.t] / port,
            self.cash / port,
        ])
        return np.concatenate([r * 100.0, extra]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.window
        self.cash = self.initial_money
        self.inventory = 0
        self.prev_value = self.initial_money
        return self._obs(), {}

    def step(self, action):
        price = self.prices[self.t]
        if action == 1 and self.cash >= price * (1 + self.fee):
            self.cash -= price * (1 + self.fee)
            self.inventory += 1
        elif action == 2 and self.inventory > 0:
            self.cash += price * (1 - self.fee)
            self.inventory -= 1

        self.t += 1
        done = self.t >= len(self.prices) - 1
        value = self.cash + self.inventory * self.prices[self.t]
        reward = float(np.log(value / self.prev_value))
        self.prev_value = value
        return self._obs(), reward, done, False, {'portfolio_value': value}
