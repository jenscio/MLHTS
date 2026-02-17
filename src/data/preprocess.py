"""
Data preprocessing pipeline.

Loads raw price data, computes returns and momentum features,
and saves processed data for downstream use.

Usage: python src/data/preprocess.py
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from features.momentum import compute_momentum_features

# Asset class mapping
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


def preprocess(raw_path: str = None, processed_dir: str = None):
    """
    Full preprocessing pipeline: raw prices → processed features.

    Saves:
        - prices.parquet: cleaned daily close prices
        - returns.parquet: daily simple returns
        - momentum_features.parquet: all 8 momentum features
        - asset_classes.csv: ticker → asset class mapping
    """
    # Default paths
    base_dir = os.path.join(os.path.dirname(__file__), '../..')
    if raw_path is None:
        raw_path = os.path.join(base_dir, 'data/raw/prices.csv')
    if processed_dir is None:
        processed_dir = os.path.join(base_dir, 'data/processed')

    os.makedirs(processed_dir, exist_ok=True)

    # ── Load raw prices ──
    print("Loading raw prices...")
    prices = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    prices = prices.sort_index()
    print(f"  {prices.shape[0]} days, {prices.shape[1]} assets")
    print(f"  {prices.index[0].date()} to {prices.index[-1].date()}")

    # ── Compute returns ──
    print("Computing returns...")
    returns = prices.pct_change(fill_method=None)

    # ── Compute momentum features ──
    print("Computing momentum features (this may take a minute)...")
    features = compute_momentum_features(prices, winsorise_features=True)

    # ── Save ──
    print("Saving processed data...")
    prices.to_parquet(os.path.join(processed_dir, 'prices.parquet'))
    returns.to_parquet(os.path.join(processed_dir, 'returns.parquet'))
    features.to_parquet(os.path.join(processed_dir, 'momentum_features.parquet'))

    # Asset class mapping (only for assets in our universe)
    ac = pd.Series(ASSET_CLASSES, name='asset_class')
    ac.index.name = 'ticker'
    ac = ac[ac.index.isin(prices.columns)]
    ac.to_csv(os.path.join(processed_dir, 'asset_classes.csv'))

    print(f"\nSaved to {processed_dir}/:")
    print(f"  prices.parquet            ({prices.shape})")
    print(f"  returns.parquet           ({returns.shape})")
    print(f"  momentum_features.parquet ({features.shape})")
    print(f"  asset_classes.csv         ({len(ac)} assets)")
    print("\nDone!")

    return prices, returns, features


def load_processed(processed_dir: str = None) -> tuple:
    """
    Load processed data from disk.

    Returns
    -------
    tuple of (prices, returns, features, asset_classes)
    """
    if processed_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), '../..')
        processed_dir = os.path.join(base_dir, 'data/processed')

    prices = pd.read_parquet(os.path.join(processed_dir, 'prices.parquet'))
    returns = pd.read_parquet(os.path.join(processed_dir, 'returns.parquet'))
    features = pd.read_parquet(os.path.join(processed_dir, 'momentum_features.parquet'))
    ac = pd.read_csv(os.path.join(processed_dir, 'asset_classes.csv'), index_col=0).squeeze()

    return prices, returns, features, ac


if __name__ == '__main__':
    preprocess()
