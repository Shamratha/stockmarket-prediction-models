"""Evolution-strategy trading agent — the original repo's signature idea,
modernized: vectorized NumPy, trained on one data segment and traded on
another, with transaction costs.
"""

import numpy as np

from .backtest import run_backtest


class ESNetwork:
    """1 hidden layer MLP: state -> 3 action scores."""

    def __init__(self, input_size, hidden=64, output=3, seed=42):
        rng = np.random.default_rng(seed)
        self.weights = [
            rng.standard_normal((input_size, hidden)) * 0.1,
            rng.standard_normal((hidden, output)) * 0.1,
            np.zeros((1, hidden)),
        ]

    def predict(self, state):
        h = np.tanh(state @ self.weights[0] + self.weights[2])
        return h @ self.weights[1]


class EvolutionAgent:
    def __init__(self, window=30, hidden=64, population=24, sigma=0.1,
                 lr=0.03, fee_bps=10.0, initial_money=10000.0, seed=42):
        self.window = window
        self.net = ESNetwork(window + 1, hidden, seed=seed)
        self.population = population
        self.sigma = sigma
        self.lr = lr
        self.fee_bps = fee_bps
        self.initial_money = initial_money
        self.rng = np.random.default_rng(seed)

    def _states(self, prices):
        """Precompute normalized return-window states for the whole series."""
        prices = np.asarray(prices, dtype=float)
        rets = np.diff(prices, prepend=prices[0])
        states = np.zeros((len(prices), self.window + 1))
        for t in range(len(prices)):
            lo = max(0, t - self.window + 1)
            w = rets[lo:t + 1]
            states[t, self.window - len(w):self.window] = w
            states[t, -1] = 0.0  # inventory flag, filled during rollout
        scale = np.std(states[:, :self.window]) + 1e-8
        states[:, :self.window] /= scale
        return states

    def _actions(self, prices, weights):
        states = self._states(prices)
        actions = np.zeros(len(prices), dtype=int)
        inventory = 0
        cash = self.initial_money
        net = ESNetwork.__new__(ESNetwork)
        net.weights = weights
        for t in range(len(prices)):
            states[t, -1] = 1.0 if inventory > 0 else 0.0
            a = int(np.argmax(net.predict(states[t:t + 1])[0]))
            if a == 1 and cash >= prices[t]:
                cash -= prices[t]
                inventory += 1
            elif a == 2 and inventory > 0:
                cash += prices[t]
                inventory -= 1
            actions[t] = a
        return actions

    def _reward(self, prices, weights):
        result = run_backtest(prices, self._actions(prices, weights),
                              self.initial_money, self.fee_bps)
        return result['roi']

    def train(self, prices, iterations=300, print_every=50):
        w = self.net.weights
        for it in range(iterations):
            noise = [
                [self.rng.standard_normal(x.shape) for x in w]
                for _ in range(self.population)
            ]
            rewards = np.array([
                self._reward(prices, [x + self.sigma * n for x, n in zip(w, eps)])
                for eps in [n for n in noise]
            ])
            std = rewards.std()
            if std > 1e-9:
                norm = (rewards - rewards.mean()) / std
                for i in range(len(w)):
                    update = np.sum([norm[k] * noise[k][i] for k in range(self.population)], axis=0)
                    w[i] = w[i] + self.lr / (self.population * self.sigma) * update
            if (it + 1) % print_every == 0:
                print(f'iter {it + 1}: train ROI {self._reward(prices, w):.2f}%')
        self.net.weights = w

    def act_series(self, prices):
        return self._actions(prices, self.net.weights)
