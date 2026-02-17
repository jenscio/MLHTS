"""
Graph learning for Network Momentum strategy.

Learns sparse adjacency matrices from momentum features using the
Kalofolias (2016) convex optimisation, as applied in Pu et al. (2023).

The key insight: compute pairwise distances on the 8-dimensional
daily feature vectors, then average over the lookback window.
This is equivalent to tr(V^T L V) decomposing into a sum of
daily Laplacian quadratic forms.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

LOOKBACK_WINDOWS = [252, 504, 756, 1008, 1260]   # 1y, 2y, 3y, 4y, 5y

ALPHA_GRID = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
BETA_GRID  = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]


# ──────────────────────────────────────────────
# Averaged distance computation
# ──────────────────────────────────────────────

def compute_avg_distance_matrix(momentum_features: pd.DataFrame,
                                date: pd.Timestamp,
                                lookback: int = 252,
                                tickers: list = None) -> tuple:
    """
    Compute averaged pairwise distance matrix over a lookback window.

    For each day in the window, extracts the N x 8 feature matrix,
    computes pairwise squared Euclidean distances, and averages.

    Parameters
    ----------
    momentum_features : pd.DataFrame
        Output of compute_momentum_features(). MultiIndex columns.
    date : pd.Timestamp
        End date of lookback window.
    lookback : int
        Number of trading days.
    tickers : list, optional
        Restrict to these tickers. If None, uses all with full data.

    Returns
    -------
    tuple of (np.ndarray, list[str])
        - Average distance matrix, shape (N, N).
        - List of tickers used.
    """
    from features.momentum import get_feature_matrix

    all_dates = momentum_features.index[momentum_features.index <= date]
    window_dates = all_dates[-lookback:]

    # Determine tickers: use all that have data on the first day of the window
    if tickers is None:
        U_first = get_feature_matrix(momentum_features, window_dates[0]).dropna()
        tickers = sorted(U_first.index.tolist())

    n = len(tickers)
    Z_sum = np.zeros((n, n))
    count = 0

    for d in window_dates:
        U = get_feature_matrix(momentum_features, d)
        U = U.reindex(tickers)

        # Skip days with any missing data
        if U.isna().any().any():
            continue

        Z_sum += squareform(pdist(U.values, 'sqeuclidean'))
        count += 1

    if count == 0:
        raise ValueError(f"No complete days in lookback window ending {date}")

    Z_avg = Z_sum / count
    return Z_avg, tickers


# ──────────────────────────────────────────────
# Core graph learning (closed-form iterative)
# ──────────────────────────────────────────────

def learn_graph_from_distances(Z: np.ndarray, alpha: float, beta: float,
                               max_iter: int = 200, step_size: float = 0.1,
                               tol: float = 1e-6) -> np.ndarray:
    """
    Learn a sparse graph from a precomputed distance matrix via
    proximal gradient descent on the Kalofolias (2016) objective.

    min_W  sum_{ij} Z_ij * W_ij  -  alpha * sum_i log(d_i)  +  beta * ||W||_F^2
    s.t.   W_ij >= 0, W_ii = 0, W symmetric

    Parameters
    ----------
    Z : np.ndarray, shape (N, N)
        Pairwise distance matrix (symmetric, zero diagonal).
    alpha : float
        Log-barrier weight. Larger = denser graph.
    beta : float
        L2 penalty. Larger = sparser graph.
    max_iter : int
        Maximum iterations.
    step_size : float
        Gradient step size.
    tol : float
        Convergence tolerance.

    Returns
    -------
    np.ndarray, shape (N, N)
        Symmetric adjacency matrix.
    """
    n = Z.shape[0]

    # Normalise distances to mean 1
    z_upper = Z[np.triu_indices(n, k=1)]
    z_mean = z_upper.mean() if z_upper.mean() > 0 else 1.0
    Z_norm = Z / z_mean

    # Initialise: inverse distance heuristic
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            W[i, j] = W[j, i] = max(1.0 / (Z_norm[i, j] + 1e-6) - 1, 0)

    # Normalise initial W
    if W.max() > 0:
        W = W / W.max()

    denom = 1.0 + step_size * 2 * beta

    for iteration in range(max_iter):
        W_old = W.copy()

        d = W.sum(axis=1)
        d = np.maximum(d, 1e-10)
        inv_d = 1.0 / d

        for i in range(n):
            for j in range(i + 1, n):
                grad = Z_norm[i, j] - alpha * (inv_d[i] + inv_d[j])
                w_new = max(0, (W[i, j] - step_size * grad) / denom)
                W[i, j] = w_new
                W[j, i] = w_new

        if np.abs(W - W_old).max() < tol:
            break

    # Threshold small weights
    if W.max() > 0:
        W[W < 1e-4 * W.max()] = 0

    return W


def learn_graph(V: np.ndarray, alpha: float, beta: float,
                solver: str = 'SCS', verbose: bool = False) -> np.ndarray:
    """
    Learn a sparse graph via CVXPY from a feature matrix.

    Computes pairwise distances from V, then solves Kalofolias.
    For the averaged-distance approach, use learn_graph_from_distances() instead.

    Parameters
    ----------
    V : np.ndarray, shape (N, D)
        Feature matrix (e.g. single-day N x 8).
    alpha, beta : float
        Graph learning hyperparameters.

    Returns
    -------
    np.ndarray, shape (N, N)
    """
    Z = squareform(pdist(V, metric='sqeuclidean'))
    return learn_graph_from_distances(Z, alpha, beta)


# ──────────────────────────────────────────────
# Graph normalisation
# ──────────────────────────────────────────────

def normalise_adjacency(A: np.ndarray) -> np.ndarray:
    """Symmetric degree normalisation: A_tilde = D^{-1/2} A D^{-1/2}."""
    degrees = A.sum(axis=1)
    degrees = np.maximum(degrees, 1e-10)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    return D_inv_sqrt @ A @ D_inv_sqrt


# ──────────────────────────────────────────────
# Graph ensemble
# ──────────────────────────────────────────────

def learn_ensemble_graph(momentum_features: pd.DataFrame, date: pd.Timestamp,
                         alpha: float, beta: float,
                         lookbacks: list = None,
                         verbose: bool = False) -> tuple:
    """
    Learn and average graphs from multiple lookback windows.

    For each lookback, computes the averaged daily distance matrix,
    then learns a graph from it. Final graph is the average of all.

    Parameters
    ----------
    momentum_features : pd.DataFrame
        Output of compute_momentum_features().
    date : pd.Timestamp
        Current trading date.
    alpha, beta : float
        Graph learning hyperparameters.
    lookbacks : list of int
        Lookback windows. Default [252, 504, 756, 1008, 1260].

    Returns
    -------
    tuple of (np.ndarray, list[str])
        - Normalised adjacency matrix.
        - List of ticker names.
    """
    if lookbacks is None:
        lookbacks = LOOKBACK_WINDOWS

    # First pass: find common tickers across all windows
    common_tickers = None
    for delta in lookbacks:
        try:
            _, tickers = compute_avg_distance_matrix(
                momentum_features, date, lookback=delta
            )
            if common_tickers is None:
                common_tickers = set(tickers)
            else:
                common_tickers = common_tickers & set(tickers)
        except (ValueError, KeyError):
            continue

    if common_tickers is None or len(common_tickers) < 5:
        raise ValueError(f"Not enough common assets at {date}")

    common_tickers = sorted(common_tickers)

    # Second pass: learn graphs on common tickers
    graphs = []
    for delta in lookbacks:
        try:
            Z, _ = compute_avg_distance_matrix(
                momentum_features, date, lookback=delta, tickers=common_tickers
            )
            A = learn_graph_from_distances(Z, alpha=alpha, beta=beta)
            graphs.append(A)

            if verbose:
                stats = graph_stats(A)
                print(f"  Lookback {delta}: {stats['n_edges']} edges, "
                      f"sparsity {stats['sparsity']:.3f}")
        except (ValueError, KeyError):
            continue

    if not graphs:
        raise ValueError(f"Could not learn any graphs at {date}")

    A_bar = np.mean(graphs, axis=0)

    # Threshold small weights in ensemble
    if A_bar.max() > 0:
        A_bar[A_bar < 1e-4 * A_bar.max()] = 0

    A_tilde = normalise_adjacency(A_bar)

    return A_tilde, common_tickers


# ──────────────────────────────────────────────
# Graph statistics
# ──────────────────────────────────────────────

def graph_stats(A: np.ndarray) -> dict:
    """Compute topological statistics of a graph."""
    n = A.shape[0]
    n_possible = n * (n - 1) / 2

    edges = np.triu(A, k=1)
    n_edges = np.count_nonzero(edges)

    binary_A = (A > 1e-10).astype(float)
    np.fill_diagonal(binary_A, 0)
    degrees = binary_A.sum(axis=1)

    return {
        'n_nodes': n,
        'n_edges': n_edges,
        'sparsity': n_edges / max(n_possible, 1),
        'avg_degree': degrees.mean(),
        'max_degree': degrees.max(),
        'min_degree': degrees.min(),
        'avg_weight': edges[edges > 0].mean() if n_edges > 0 else 0,
    }


# ──────────────────────────────────────────────
# Combined graph (Kalofolias + Copula)
# ──────────────────────────────────────────────

def combine_graphs(A_kalofolias: np.ndarray, tickers_kalo: list,
                   A_copula: np.ndarray, tickers_copula: list,
                   weight_kalo: float = 0.5) -> tuple:
    """
    Combine Kalofolias and copula adjacency matrices.

    Aligns to common tickers, normalises each to [0,1], weighted average.
    """
    common = sorted(set(tickers_kalo) & set(tickers_copula))
    if len(common) < 5:
        raise ValueError("Fewer than 5 common tickers between graphs")

    idx_kalo = [tickers_kalo.index(t) for t in common]
    idx_copa = [tickers_copula.index(t) for t in common]

    Ak = A_kalofolias[np.ix_(idx_kalo, idx_kalo)]
    Ac = A_copula[np.ix_(idx_copa, idx_copa)]

    if Ak.max() > 0:
        Ak = Ak / Ak.max()
    if Ac.max() > 0:
        Ac = Ac / Ac.max()

    A_combined = weight_kalo * Ak + (1 - weight_kalo) * Ac
    A_tilde = normalise_adjacency(A_combined)

    return A_tilde, common