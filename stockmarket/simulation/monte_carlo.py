"""Monte Carlo price simulations: GBM, dynamic (EWMA) volatility, and
correlated multi-asset paths."""

import numpy as np


def gbm_paths(close, horizon=252, n_paths=200, seed=42):
    """Geometric Brownian motion calibrated to historical drift/volatility."""
    rng = np.random.default_rng(seed)
    rets = np.diff(np.log(np.asarray(close, dtype=float)))
    mu, sigma = rets.mean(), rets.std()
    drift = mu - 0.5 * sigma ** 2
    shocks = rng.standard_normal((n_paths, horizon))
    log_paths = np.cumsum(drift + sigma * shocks, axis=1)
    return close[-1] * np.exp(log_paths)


def dynamic_vol_paths(close, horizon=252, n_paths=200, lam=0.94, seed=42):
    """GBM with EWMA (RiskMetrics) volatility updated along each path."""
    rng = np.random.default_rng(seed)
    rets = np.diff(np.log(np.asarray(close, dtype=float)))
    mu = rets.mean()
    var0 = rets.var()

    paths = np.zeros((n_paths, horizon))
    for p in range(n_paths):
        var = var0
        price = np.log(close[-1])
        for t in range(horizon):
            sigma = np.sqrt(var)
            shock = rng.standard_normal()
            step = (mu - 0.5 * var) + sigma * shock
            price += step
            paths[p, t] = price
            var = lam * var + (1 - lam) * step ** 2
    return np.exp(paths)


def correlated_paths(closes, horizon=252, n_paths=100, seed=42):
    """Multi-asset GBM preserving the historical correlation matrix
    (Cholesky). `closes` is a dict name -> price array."""
    rng = np.random.default_rng(seed)
    names = list(closes)
    rets = np.column_stack([np.diff(np.log(np.asarray(closes[n], dtype=float)))[-500:]
                            for n in names])
    mu = rets.mean(axis=0)
    cov = np.cov(rets.T)
    chol = np.linalg.cholesky(cov + 1e-12 * np.eye(len(names)))

    last = np.array([closes[n][-1] for n in names])
    out = np.zeros((n_paths, horizon, len(names)))
    for p in range(n_paths):
        z = rng.standard_normal((horizon, len(names))) @ chol.T
        drift = mu - 0.5 * np.diag(cov)
        log_paths = np.cumsum(drift + z, axis=0)
        out[p] = last * np.exp(log_paths)
    return names, out
