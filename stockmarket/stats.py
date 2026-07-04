"""Statistical significance tools for forecast comparison.

- Diebold-Mariano test (with the Harvey-Leybourne-Newbold small-sample
  correction): is model A's forecast loss actually different from model B's,
  or is the gap noise?
- Bootstrap confidence intervals on RMSE / MAPE across the walk-forward
  window.
"""

import numpy as np
from scipy import stats as scipy_stats


def diebold_mariano(errors_a, errors_b, h=1, power=2):
    """DM test on two aligned forecast-error series.

    errors_* are forecast errors (true - pred). Loss defaults to squared
    error (power=2). Returns (dm_stat, p_value) two-sided; a negative stat
    means model A has LOWER loss than model B.
    """
    e_a = np.asarray(errors_a, dtype=float)
    e_b = np.asarray(errors_b, dtype=float)
    if e_a.shape != e_b.shape:
        raise ValueError('error series must be aligned')
    n = len(e_a)
    if n < 10:
        raise ValueError('too few observations for a DM test')

    d = np.abs(e_a) ** power - np.abs(e_b) ** power
    dbar = d.mean()

    # long-run variance of the loss differential (autocovariances to lag h-1)
    gamma = [np.mean((d - dbar) ** 2)]
    for lag in range(1, h):
        gamma.append(np.mean((d[lag:] - dbar) * (d[:-lag] - dbar)))
    lrv = gamma[0] + 2 * sum(gamma[1:])
    if lrv <= 0:
        return 0.0, 1.0

    dm = dbar / np.sqrt(lrv / n)

    # Harvey-Leybourne-Newbold small-sample correction
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * scipy_stats.t.sf(np.abs(dm), df=n - 1)
    return float(dm), float(p)


def bootstrap_ci(true_prices, pred_prices, metric='rmse', n_boot=2000,
                 alpha=0.05, seed=42):
    """Percentile bootstrap CI for RMSE or MAPE over the test window.

    Returns (point_estimate, ci_low, ci_high).
    """
    true_prices = np.asarray(true_prices, dtype=float)
    pred_prices = np.asarray(pred_prices, dtype=float)
    n = len(true_prices)
    rng = np.random.default_rng(seed)

    def compute(t, p):
        if metric == 'rmse':
            return np.sqrt(np.mean((t - p) ** 2))
        if metric == 'mape':
            return np.mean(np.abs((t - p) / t)) * 100
        raise ValueError(metric)

    idx = rng.integers(0, n, size=(n_boot, n))
    vals = np.array([compute(true_prices[i], pred_prices[i]) for i in idx])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(compute(true_prices, pred_prices)), float(lo), float(hi)
