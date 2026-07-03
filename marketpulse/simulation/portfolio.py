"""Efficient-frontier portfolio optimization (max Sharpe / min variance) via
random search + SLSQP refinement."""

import numpy as np


def portfolio_stats(weights, mean_rets, cov, rf=0.0):
    ret = float(weights @ mean_rets) * 252
    vol = float(np.sqrt(weights @ cov @ weights)) * np.sqrt(252)
    sharpe = (ret - rf) / (vol + 1e-12)
    return ret, vol, sharpe


def random_portfolios(mean_rets, cov, n=20000, seed=42):
    rng = np.random.default_rng(seed)
    k = len(mean_rets)
    w = rng.random((n, k))
    w /= w.sum(axis=1, keepdims=True)
    rets = w @ mean_rets * 252
    vols = np.sqrt(np.einsum('ij,jk,ik->i', w, cov, w)) * np.sqrt(252)
    sharpes = rets / (vols + 1e-12)
    return w, rets, vols, sharpes


def optimize(mean_rets, cov, objective='sharpe'):
    from scipy.optimize import minimize

    k = len(mean_rets)

    def neg_sharpe(w):
        return -portfolio_stats(w, mean_rets, cov)[2]

    def variance(w):
        return float(w @ cov @ w)

    fun = neg_sharpe if objective == 'sharpe' else variance
    res = minimize(
        fun, np.full(k, 1 / k), method='SLSQP',
        bounds=[(0, 1)] * k,
        constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}],
    )
    return res.x
