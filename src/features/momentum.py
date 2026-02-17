"""
Momentum feature engineering for Network Momentum strategy.

Implements the 8 momentum features from Pu et al. (2023):
- 5 volatility-scaled returns (1d, 1m, 3m, 6m, 1y)
- 3 normalised MACD indicators (8/24, 16/48, 32/96)

References:
    Pu, X., Roberts, S., Dong, X., & Zohren, S. (2023).
    Network Momentum across Asset Classes. arXiv:2308.11294

    Baz, J., Granger, N., Harvey, C.R., Le Roux, N., & Rattray, S. (2015).
    Dissecting Investment Strategies in the Cross Section and Time Series.

    Lim, B., Zohren, S., & Roberts, S. (2019).
    Enhancing Time Series Momentum Strategies Using Deep Neural Networks.
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

VOL_LOOKBACKS = [1, 21, 63, 126, 252]          # 1d, 1m, 3m, 6m, 1y
MACD_PAIRS = [(8, 24), (16, 48), (32, 96)]     # (short, long) EMA scales
EWM_VOL_SPAN = 60                               # span for volatility estimation
WINSOR_HALFLIFE = 252                            # half-life for winsorisation stats
WINSOR_MULTIPLIER = 5                            # cap at ±5 std from mean


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def ewm_std(series: pd.Series, span: int) -> pd.Series:
    """
    Exponentially weighted moving standard deviation.

    Parameters
    ----------
    series : pd.Series
        Input time series (typically daily returns).
    span : int
        Span for the EWM (controls decay, alpha = 2 / (span + 1)).

    Returns
    -------
    pd.Series
        EWM standard deviation.
    """
    return series.ewm(span=span, min_periods=span).std()


def ewm_mean(series: pd.Series, span: int) -> pd.Series:
    """
    Exponentially weighted moving average.

    Parameters
    ----------
    series : pd.Series
        Input time series.
    span : int
        Span for the EWM.

    Returns
    -------
    pd.Series
        EWM mean.
    """
    return series.ewm(span=span, min_periods=span).mean()


def ema_from_scale(series: pd.Series, scale: int) -> pd.Series:
    """
    Exponential moving average with smoothing factor alpha = 1/scale.

    This matches the MACD definition in Baz et al. (2015) and Pu et al. (2023),
    where the scale J gives alpha = 1/J.

    Parameters
    ----------
    series : pd.Series
        Input time series (typically prices).
    scale : int
        Scale parameter J, giving alpha = 1/J.

    Returns
    -------
    pd.Series
        EMA with the specified scale.
    """
    return series.ewm(alpha=1.0/scale, min_periods=scale).mean()


# ──────────────────────────────────────────────
# Volatility-scaled returns
# ──────────────────────────────────────────────

def compute_vol_scaled_returns(prices: pd.DataFrame, daily_returns: pd.DataFrame) -> dict:
    """
    Compute volatility-scaled returns over 5 lookback horizons.

    For each horizon delta, the feature is:
        r_{t-delta:t} / (sigma_t * sqrt(delta))

    where sigma_t is the 60-day EWM standard deviation of daily returns.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices, shape (T, N).
    daily_returns : pd.DataFrame
        Daily simple returns, shape (T, N).

    Returns
    -------
    dict
        Keys are feature names, values are pd.DataFrames of shape (T, N).
    """
    # Estimate daily volatility: 60-day EWM std
    sigma = daily_returns.apply(lambda col: ewm_std(col, span=EWM_VOL_SPAN))

    features = {}
    for delta in VOL_LOOKBACKS:
        # Cumulative return over past delta days
        period_return = prices / prices.shift(delta) - 1

        # Volatility-scaled return
        vol_scaled = period_return / (sigma * np.sqrt(delta))

        label = f'vol_ret_{delta}d'
        features[label] = vol_scaled

    return features


# ──────────────────────────────────────────────
# Normalised MACD indicators
# ──────────────────────────────────────────────

def compute_normalised_macd(prices: pd.DataFrame) -> dict:
    """
    Compute normalised MACD indicators for 3 speed pairs.

    For each (S, L) pair:
        1. MACD = EMA(price, S) - EMA(price, L)
        2. MACD_norm = MACD / rolling_std(price, 63)
        3. y = MACD_norm / rolling_std(MACD_norm, 252)

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices, shape (T, N).

    Returns
    -------
    dict
        Keys are feature names, values are pd.DataFrames of shape (T, N).
    """
    features = {}

    for short_scale, long_scale in MACD_PAIRS:
        # Step 1: raw MACD (fast EMA - slow EMA)
        ema_fast = prices.apply(lambda col: ema_from_scale(col, short_scale))
        ema_slow = prices.apply(lambda col: ema_from_scale(col, long_scale))
        macd_raw = ema_fast - ema_slow

        # Step 2: normalise by 63-day rolling price std
        price_std_63 = prices.rolling(window=63, min_periods=21).std()
        macd_norm = macd_raw / price_std_63

        # Step 3: normalise by 252-day rolling std of MACD_norm itself
        macd_norm_std_252 = macd_norm.rolling(window=252, min_periods=63).std()
        y = macd_norm / macd_norm_std_252

        label = f'macd_{short_scale}_{long_scale}'
        features[label] = y

    return features


# ──────────────────────────────────────────────
# Winsorisation
# ──────────────────────────────────────────────

def winsorise(features: dict, halflife: int = WINSOR_HALFLIFE,
              multiplier: float = WINSOR_MULTIPLIER) -> dict:
    """
    Winsorise features to mitigate outlier influence.

    Each feature is capped and floored at ±multiplier times its
    EWM standard deviation from its EWM mean, using the specified half-life.

    Parameters
    ----------
    features : dict
        Keys are feature names, values are pd.DataFrames.
    halflife : int
        Half-life in days for EWM statistics.
    multiplier : float
        Number of standard deviations for cap/floor.

    Returns
    -------
    dict
        Winsorised features with same structure.
    """
    winsorised = {}

    for name, df in features.items():
        ewm_mu = df.ewm(halflife=halflife, min_periods=halflife).mean()
        ewm_sigma = df.ewm(halflife=halflife, min_periods=halflife).std()

        upper = ewm_mu + multiplier * ewm_sigma
        lower = ewm_mu - multiplier * ewm_sigma

        clipped = df.clip(lower=lower, upper=upper)
        winsorised[name] = clipped

    return winsorised


# ──────────────────────────────────────────────
# Main feature pipeline
# ──────────────────────────────────────────────

def compute_momentum_features(prices: pd.DataFrame, winsorise_features: bool = True) -> pd.DataFrame:
    """
    Compute all 8 momentum features for each asset.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices, shape (T, N). Index is DatetimeIndex,
        columns are asset tickers.
    winsorise_features : bool
        Whether to apply winsorisation. Default True.

    Returns
    -------
    pd.DataFrame
        MultiIndex columns: (feature_name, ticker).
        Shape is (T, 8*N) effectively, but stored as MultiIndex for clarity.
        Use .xs('SPY', level='ticker', axis=1) to get all features for SPY.
        Use .xs('vol_ret_21d', level='feature', axis=1) to get one feature for all assets.
    """
    # Daily simple returns
    daily_returns = prices.pct_change(fill_method=None)

    # Compute features
    vol_features = compute_vol_scaled_returns(prices, daily_returns)
    macd_features = compute_normalised_macd(prices)

    all_features = {**vol_features, **macd_features}

    # Winsorise
    if winsorise_features:
        all_features = winsorise(all_features)

    # Combine into single DataFrame with MultiIndex columns
    panels = {}
    for name, df in all_features.items():
        for ticker in df.columns:
            panels[(name, ticker)] = df[ticker]

    result = pd.DataFrame(panels)
    result.columns = pd.MultiIndex.from_tuples(result.columns, names=['feature', 'ticker'])

    # Sort columns for clean access
    result = result.sort_index(axis=1)

    return result


def get_feature_matrix(momentum_features: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """
    Extract the (N_assets x 8) feature matrix for a single date.

    This is the matrix U_t from Pu et al. Section 2.2, used as input
    to graph learning.

    Parameters
    ----------
    momentum_features : pd.DataFrame
        Output of compute_momentum_features().
    date : pd.Timestamp
        The date to extract features for.

    Returns
    -------
    pd.DataFrame
        Shape (N_assets, 8), index is tickers, columns are feature names.
    """
    row = momentum_features.loc[date]
    return row.unstack(level='feature')


def get_stacked_features(momentum_features: pd.DataFrame, date: pd.Timestamp,
                         lookback: int = 252) -> pd.DataFrame:
    """
    Stack feature matrices over a lookback window to produce V_t.

    V_t has shape (N_assets, 8 * lookback) — each row is an asset,
    columns are the concatenation of 8 features over `lookback` days.
    This is the input to graph learning (Eq. 4 in Pu et al.).

    Only includes assets whose features are fully available for the
    entire lookback window (i.e. the asset started trading early enough).

    Parameters
    ----------
    momentum_features : pd.DataFrame
        Output of compute_momentum_features().
    date : pd.Timestamp
        End date of the lookback window.
    lookback : int
        Number of trading days to look back.

    Returns
    -------
    pd.DataFrame
        Shape (N_assets, 8 * lookback).
    """
    # Get the lookback window
    mask = momentum_features.index <= date
    window = momentum_features.loc[mask].tail(lookback)

    if len(window) < lookback:
        raise ValueError(
            f"Only {len(window)} days available before {date}, need {lookback}"
        )

    tickers = momentum_features.columns.get_level_values('ticker').unique()

    rows = {}
    for ticker in tickers:
        ticker_data = window.xs(ticker, level='ticker', axis=1)

        # Only include if all 8 features are valid on the first day of the window
        # This means the asset was already trading with enough history
        if ticker_data.iloc[0].isna().any():
            continue

        # Flatten: (lookback, 8) -> (8 * lookback,)
        rows[ticker] = ticker_data.values.flatten()

    result = pd.DataFrame.from_dict(rows, orient='index')
    return result


# ──────────────────────────────────────────────
# Feature names for reference
# ──────────────────────────────────────────────

FEATURE_NAMES = [
    'vol_ret_1d',    # 1-day vol-scaled return
    'vol_ret_21d',   # 1-month vol-scaled return
    'vol_ret_63d',   # 3-month vol-scaled return
    'vol_ret_126d',  # 6-month vol-scaled return
    'vol_ret_252d',  # 1-year vol-scaled return
    'macd_8_24',     # MACD (fast)
    'macd_16_48',    # MACD (medium)
    'macd_32_96',    # MACD (slow)
]