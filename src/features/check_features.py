"""
Quick sanity check for momentum features.

Run this after downloading price data to verify features look reasonable.
Usage: python src/features/check_features.py
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path so we can import momentum module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from features.momentum import (
    compute_momentum_features, get_feature_matrix,
    get_stacked_features, FEATURE_NAMES
)


def main():
    # ── Load data ──
    data_path = os.path.join(os.path.dirname(__file__), '../../data/raw/prices.csv')
    if not os.path.exists(data_path):
        print("ERROR: prices.csv not found. Run download.py first.")
        return

    prices = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Loaded prices: {prices.shape[0]} days, {prices.shape[1]} assets")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Assets: {list(prices.columns)}\n")

    # ── Compute features ──
    print("Computing momentum features...")
    features = compute_momentum_features(prices, winsorise_features=True)
    print(f"Feature matrix shape: {features.shape}")
    print(f"Features: {FEATURE_NAMES}\n")

    # ── Check a single date ──
    # Pick a date roughly in the middle of the dataset
    mid_idx = len(features) // 2
    sample_date = features.index[mid_idx]
    print(f"Sample date: {sample_date.date()}")

    U_t = get_feature_matrix(features, sample_date)
    print(f"Feature matrix U_t shape: {U_t.shape}  (should be N_assets x 8)")
    print(f"Non-NaN assets: {U_t.dropna().shape[0]}\n")

    # Show features for a few assets
    sample_tickers = [t for t in ['SPY', 'GLD', 'TLT', 'FXF', 'EWL'] if t in U_t.index]
    print(f"Features for sample assets on {sample_date.date()}:")
    print(U_t.loc[sample_tickers].round(3).to_string())
    print()

    # ── Basic statistics ──
    print("Feature statistics (across all assets and dates):")
    stats = []
    for feat in FEATURE_NAMES:
        if feat in features.columns.get_level_values('feature'):
            feat_data = features.xs(feat, level='feature', axis=1)
            stats.append({
                'feature': feat,
                'mean': feat_data.stack().mean(),
                'std': feat_data.stack().std(),
                'min': feat_data.stack().min(),
                'max': feat_data.stack().max(),
                'pct_nan': feat_data.isna().mean().mean() * 100,
            })
    stats_df = pd.DataFrame(stats).set_index('feature')
    print(stats_df.round(3).to_string())
    print()

    # ── Check stacked features (input to graph learning) ──
    # Use last available date with enough history
    recent_date = features.dropna(how='all').index[-1]
    try:
        V_t = get_stacked_features(features, recent_date, lookback=252)
        print(f"Stacked feature matrix V_t shape: {V_t.shape}"
              f"  (should be N_assets x {8 * 252})")
        print(f"Assets with complete data: {V_t.shape[0]}")
    except ValueError as e:
        print(f"Could not build stacked features: {e}")

    # ── Sanity checks ──
    print("\n── Sanity Checks ──")

    # Features should be roughly centered around 0
    overall_mean = stats_df['mean'].abs().mean()
    print(f"Average absolute feature mean: {overall_mean:.3f}"
          f"  {'✓ OK' if overall_mean < 1.0 else '⚠ Check centering'}")

    # Std should be reasonable (not exploding)
    max_std = stats_df['std'].max()
    print(f"Max feature std: {max_std:.3f}"
          f"  {'✓ OK' if max_std < 10.0 else '⚠ Check scaling'}")

    # Winsorisation should prevent extreme values
    max_abs = max(stats_df['max'].max(), abs(stats_df['min'].min()))
    print(f"Max absolute value: {max_abs:.3f}"
          f"  {'✓ OK' if max_abs < 50.0 else '⚠ Check winsorisation'}")

    # NaN percentage should be reasonable
    max_nan = stats_df['pct_nan'].max()
    print(f"Max NaN percentage: {max_nan:.1f}%"
          f"  {'✓ OK' if max_nan < 30.0 else '⚠ Check data coverage'}")

    print("\nDone! Features look ready for graph learning.")


if __name__ == '__main__':
    main()
