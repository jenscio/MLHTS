"""
Test graph learning with averaged daily distances.
Usage: python src/network/check_graph.py
"""

import pandas as pd
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from features.momentum import compute_momentum_features
from network.graph_learning import (
    compute_avg_distance_matrix, learn_graph_from_distances,
    learn_ensemble_graph, normalise_adjacency, graph_stats
)

ASSET_CLASSES = {
    'GLD': 'Commodity', 'SLV': 'Commodity', 'PPLT': 'Commodity',
    'USO': 'Commodity', 'BNO': 'Commodity', 'UNG': 'Commodity',
    'DBC': 'Commodity', 'CPER': 'Commodity', 'CORN': 'Commodity',
    'WEAT': 'Commodity',
    'SPY': 'Equity', 'QQQ': 'Equity', 'IWM': 'Equity',
    'EFA': 'Equity', 'EEM': 'Equity', 'EWJ': 'Equity',
    'EWL': 'Equity', 'EWZ': 'Equity', 'EWG': 'Equity',
    'EWU': 'Equity', 'FXI': 'Equity', 'CSSMI.SW': 'Equity',
    'SHY': 'Fixed Income', 'IEF': 'Fixed Income',
    'TLT': 'Fixed Income', 'LQD': 'Fixed Income',
    'HYG': 'Fixed Income', 'BNDX': 'Fixed Income',
    'TIP': 'Fixed Income', 'EMB': 'Fixed Income',
    'UUP': 'Currency', 'FXE': 'Currency', 'FXB': 'Currency',
    'FXY': 'Currency', 'FXA': 'Currency', 'FXF': 'Currency',
    'FXC': 'Currency', 'CEW': 'Currency',
}


def print_top_edges(A, tickers, n_top=15):
    edges = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            if A[i, j] > 1e-6:
                c1 = ASSET_CLASSES.get(tickers[i], '?')
                c2 = ASSET_CLASSES.get(tickers[j], '?')
                edges.append((tickers[i], c1, tickers[j], c2, A[i, j]))
    edges.sort(key=lambda x: -x[4])

    print(f"  {'Asset 1':<10} {'Class':<14} {'Asset 2':<10} {'Class':<14} {'Weight':>8}")
    print(f"  {'-'*60}")
    for t1, c1, t2, c2, w in edges[:n_top]:
        cross = '←→' if c1 != c2 else '   '
        print(f"  {t1:<10} {c1:<14} {t2:<10} {c2:<14} {w:>8.4f} {cross}")

    n_intra = sum(1 for _, c1, _, c2, _ in edges if c1 == c2)
    n_inter = sum(1 for _, c1, _, c2, _ in edges if c1 != c2)
    print(f"\n  Total: {len(edges)} edges ({n_intra} intra-class, {n_inter} inter-class)")


def main():
    data_path = os.path.join(os.path.dirname(__file__), '../../data/raw/prices.csv')
    prices = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Loaded {prices.shape[1]} assets\n")

    print("Computing momentum features...")
    features = compute_momentum_features(prices)

    test_date = pd.Timestamp('2020-01-02')
    print(f"Test date: {test_date.date()}\n")

    # ── Single lookback test ──
    print("=" * 60)
    print("SINGLE LOOKBACK (252 days)")
    print("=" * 60)

    Z, tickers = compute_avg_distance_matrix(features, test_date, lookback=252)
    print(f"Assets: {len(tickers)}")
    print(f"Distance stats: min={Z[Z>0].min():.2f}, mean={Z[Z>0].mean():.2f}, "
          f"max={Z.max():.2f}\n")

    test_params = [
        (0.5, 0.5),
        (1.0, 0.5),
        (2.0, 1.0),
    ]

    for alpha, beta in test_params:
        t0 = time.time()
        A = learn_graph_from_distances(Z, alpha=alpha, beta=beta)
        elapsed = time.time() - t0
        stats = graph_stats(A)
        print(f"alpha={alpha}, beta={beta}: "
              f"{stats['n_edges']} edges, sparsity={stats['sparsity']:.3f}, "
              f"time={elapsed:.2f}s")

    # Show best edges
    best_alpha, best_beta = test_params[1]
    A = learn_graph_from_distances(Z, alpha=best_alpha, beta=best_beta)
    print(f"\nTop edges (alpha={best_alpha}, beta={best_beta}):")
    print_top_edges(A, tickers)

    # ── Ensemble test ──
    print("\n" + "=" * 60)
    print("ENSEMBLE (multiple lookbacks)")
    print("=" * 60)

    t0 = time.time()
    A_tilde, ens_tickers = learn_ensemble_graph(
        features, test_date, alpha=best_alpha, beta=best_beta, verbose=True
    )
    elapsed = time.time() - t0

    stats = graph_stats(A_tilde)
    print(f"\nEnsemble: {stats['n_edges']} edges, sparsity={stats['sparsity']:.3f}, "
          f"time={elapsed:.1f}s")
    print(f"\nTop ensemble edges:")
    print_top_edges(A_tilde, ens_tickers)

    print("\nDone!")


if __name__ == '__main__':
    main()