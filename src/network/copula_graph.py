"""
Copula-based graph construction for Network Momentum strategy.

Builds adjacency matrices from pairwise copula dependence,
capturing nonlinear and tail dependence that Euclidean-distance-based
graph learning (Kalofolias) cannot detect.
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata
from scipy.optimize import minimize_scalar

# ──────────────────────────────────────────────
# Empirical CDF transform
# ──────────────────────────────────────────────

def to_pseudo_observations(x: np.ndarray) -> np.ndarray:
    """
    Transform data to pseudo-observations (empirical CDF).

    Uses rank/(n+1) to avoid boundary issues at 0 and 1.

    Parameters
    ----------
    x : np.ndarray
        Raw observations, shape (T,).

    Returns
    -------
    np.ndarray
        Pseudo-observations in (0, 1), shape (T,).
    """
    n = len(x)
    return rankdata(x) / (n + 1)


# ──────────────────────────────────────────────
# Copula families
# ──────────────────────────────────────────────

def _gaussian_nll(rho: float, u: np.ndarray, v: np.ndarray) -> float:
    """Negative log-likelihood for Gaussian copula."""
    from scipy.stats import norm

    x = norm.ppf(u)
    y = norm.ppf(v)
    n = len(u)

    rho = np.clip(rho, -0.99, 0.99)
    r2 = rho ** 2

    nll = (n / 2) * np.log(1 - r2) + (1 / (2 * (1 - r2))) * np.sum(
        r2 * (x ** 2 + y ** 2) - 2 * rho * x * y
    )
    return nll


def _clayton_logpdf(theta: float, u: np.ndarray, v: np.ndarray) -> float:
    """Log-likelihood for Clayton copula (theta > 0)."""
    if theta <= 0:
        return -np.inf
    n = len(u)
    ll = n * np.log(1 + theta)
    ll += -(1 + theta) * np.sum(np.log(u) + np.log(v))
    ll += -(2 + 1 / theta) * np.sum(np.log(u ** (-theta) + v ** (-theta) - 1))
    return ll


def _gumbel_logpdf(theta: float, u: np.ndarray, v: np.ndarray) -> float:
    """Log-likelihood for Gumbel copula (theta >= 1)."""
    if theta < 1:
        return -np.inf

    lu = -np.log(u)
    lv = -np.log(v)

    A = (lu ** theta + lv ** theta) ** (1 / theta)

    ll = -np.sum(A)
    ll += np.sum((theta - 1) * (np.log(lu) + np.log(lv)))
    ll += np.sum(np.log(A * (theta - 1) + 1))  # approximate
    ll += np.sum((1 / theta - 2) * np.log(lu ** theta + lv ** theta))

    if np.isnan(ll) or np.isinf(ll):
        return -np.inf
    return ll


# ──────────────────────────────────────────────
# Copula fitting
# ──────────────────────────────────────────────

def fit_gaussian_copula(u: np.ndarray, v: np.ndarray) -> dict:
    """Fit Gaussian copula, return parameter and AIC."""
    result = minimize_scalar(
        lambda rho: _gaussian_nll(rho, u, v),
        bounds=(-0.99, 0.99), method='bounded'
    )
    nll = result.fun
    aic = 2 * 1 + 2 * nll  # 1 parameter
    tau = (2 / np.pi) * np.arcsin(result.x)  # Kendall's tau
    return {'family': 'gaussian', 'param': result.x, 'tau': tau,
            'aic': aic, 'lower_tail': 0, 'upper_tail': 0}


def fit_clayton_copula(u: np.ndarray, v: np.ndarray) -> dict:
    """Fit Clayton copula, return parameter and AIC."""
    result = minimize_scalar(
        lambda theta: -_clayton_logpdf(theta, u, v),
        bounds=(0.01, 20), method='bounded'
    )
    theta = result.x
    nll = result.fun
    aic = 2 * 1 + 2 * nll
    tau = theta / (theta + 2)
    lower_tail = 2 ** (-1 / theta)  # lower tail dependence
    return {'family': 'clayton', 'param': theta, 'tau': tau,
            'aic': aic, 'lower_tail': lower_tail, 'upper_tail': 0}


def fit_gumbel_copula(u: np.ndarray, v: np.ndarray) -> dict:
    """Fit Gumbel copula, return parameter and AIC."""
    result = minimize_scalar(
        lambda theta: -_gumbel_logpdf(theta, u, v),
        bounds=(1.01, 20), method='bounded'
    )
    theta = result.x
    nll = result.fun
    aic = 2 * 1 + 2 * nll
    tau = 1 - 1 / theta
    upper_tail = 2 - 2 ** (1 / theta)  # upper tail dependence
    return {'family': 'gumbel', 'param': theta, 'tau': tau,
            'aic': aic, 'lower_tail': 0, 'upper_tail': upper_tail}


def fit_best_copula(u: np.ndarray, v: np.ndarray) -> dict:
    """
    Fit Gaussian, Clayton, and Gumbel copulas. Return the best by AIC.

    Parameters
    ----------
    u, v : np.ndarray
        Pseudo-observations in (0, 1), shape (T,).

    Returns
    -------
    dict
        Best copula fit with keys: family, param, tau, aic,
        lower_tail, upper_tail.
    """
    candidates = []

    try:
        candidates.append(fit_gaussian_copula(u, v))
    except Exception:
        pass

    try:
        candidates.append(fit_clayton_copula(u, v))
    except Exception:
        pass

    try:
        candidates.append(fit_gumbel_copula(u, v))
    except Exception:
        pass

    if not candidates:
        return {'family': 'none', 'param': 0, 'tau': 0,
                'aic': np.inf, 'lower_tail': 0, 'upper_tail': 0}

    return min(candidates, key=lambda x: x['aic'])


# ──────────────────────────────────────────────
# Copula adjacency matrix
# ──────────────────────────────────────────────

def copula_adjacency(returns: pd.DataFrame, date: pd.Timestamp,
                     lookback: int = 252) -> tuple:
    """
    Build adjacency matrix from pairwise copula fitting.

    For each pair of assets:
    1. Transform returns to pseudo-observations
    2. Fit Gaussian, Clayton, Gumbel copulas
    3. Select best by AIC
    4. Use Kendall's tau as edge weight (positive only)

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns, shape (T, N).
    date : pd.Timestamp
        End date of lookback window.
    lookback : int
        Number of trading days to look back.

    Returns
    -------
    tuple of (np.ndarray, list[str], pd.DataFrame)
        - Adjacency matrix A, shape (N, N).
        - List of ticker names.
        - DataFrame of pairwise copula details (family, params, tail dep).
    """
    # Get lookback window, drop assets with insufficient data
    window = returns.loc[:date].tail(lookback)
    valid_cols = window.columns[window.notna().sum() >= lookback * 0.95]
    window = window[valid_cols].dropna()

    tickers = list(window.columns)
    n = len(tickers)

    A = np.zeros((n, n))
    details = []

    for i in range(n):
        for j in range(i + 1, n):
            # Pseudo-observations
            u = to_pseudo_observations(window.iloc[:, i].values)
            v = to_pseudo_observations(window.iloc[:, j].values)

            # Fit best copula
            result = fit_best_copula(u, v)

            # Edge weight: positive Kendall's tau only
            weight = max(result['tau'], 0)
            A[i, j] = weight
            A[j, i] = weight

            details.append({
                'asset_i': tickers[i],
                'asset_j': tickers[j],
                'family': result['family'],
                'param': result['param'],
                'tau': result['tau'],
                'lower_tail': result['lower_tail'],
                'upper_tail': result['upper_tail'],
                'aic': result['aic'],
            })

    details_df = pd.DataFrame(details)
    return A, tickers, details_df


def copula_ensemble_adjacency(returns: pd.DataFrame, date: pd.Timestamp,
                              lookbacks: list = None) -> tuple:
    """
    Build ensemble copula adjacency from multiple lookback windows.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns, shape (T, N).
    date : pd.Timestamp
        Current trading date.
    lookbacks : list of int
        Lookback windows. Default [252, 504, 756].

    Returns
    -------
    tuple of (np.ndarray, list[str])
        - Averaged adjacency matrix.
        - List of ticker names.
    """
    if lookbacks is None:
        lookbacks = [252, 504, 756]  # shorter than Kalofolias (copula is slower)

    graphs = []
    common_tickers = None

    # First pass: find common tickers
    for lb in lookbacks:
        try:
            A, tickers, _ = copula_adjacency(returns, date, lookback=lb)
            if common_tickers is None:
                common_tickers = set(tickers)
            else:
                common_tickers = common_tickers & set(tickers)
        except Exception:
            continue

    if common_tickers is None or len(common_tickers) < 5:
        raise ValueError(f"Not enough common assets for copula graph at {date}")

    common_tickers = sorted(common_tickers)

    # Second pass: learn graphs on common tickers
    for lb in lookbacks:
        try:
            # Filter returns to common tickers before fitting
            filtered_returns = returns[common_tickers]
            A, tickers, _ = copula_adjacency(filtered_returns, date, lookback=lb)
            graphs.append(A)
        except Exception:
            continue

    A_bar = np.mean(graphs, axis=0)
    return A_bar, common_tickers
