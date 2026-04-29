"""
Trend-following feature set for NIFTY intraday options system.

~30 clean, causal features split into four groups:
  1. Price structure (10)   — returns, VWAP, bar shape
  2. Momentum (8)           — real-time, directional
  3. Regime signals (8)     — ADX, swing structure flags
  4. Context (6)            — time, volatility, prior session levels

Design principles:
  • All features computed causally (bar i uses only data 0..i or 0..i-1).
  • No mean-reversion indicators (RSI, MACD, Bollinger are intentionally absent).
  • Lookback periods ≤ 20 bars to stay real-time.
  • Return pct_changes are lagged by 1 bar to avoid label overlap.
"""
import numpy as np
import pandas as pd
import logging

from .indicators import _atr, _dmi

logger = logging.getLogger(__name__)
EPS = 1e-9

# ---------------------------------------------------------------------------
# Ordered feature name list (used for model training feature selection)
# ---------------------------------------------------------------------------
TREND_FEATURE_COLS: list[str] = [
    # 1. Price structure
    'ret_1b', 'ret_3b', 'ret_6b', 'ret_12b',
    'vwap_dist_pct', 'vwap_slope_3b',
    'bar_range_norm', 'bar_body_ratio',
    'gap_from_prev_close', 'day_position',
    # 2. Momentum
    'roc_1b', 'roc_3b',
    'bar_close_vs_open',
    'buying_pressure',
    'vol_ratio',
    'up_vol_frac',
    'consec_up', 'consec_dn',
    # 3. Regime signals
    'adx_20', 'adx_slope_3b',
    'higher_high_flag', 'higher_low_flag',
    'lower_high_flag',  'lower_low_flag',
    'swing_high_dist',  'swing_low_dist',
    # 4. Context
    'time_normalized',
    'dow_encoded',
    'intraday_vol',
    'vix_level',
    'prev_close_dist',
    'prev_hl_dist',
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all ~30 trend-following features on a 5-min OHLCV DataFrame.

    Required columns: open (optional), high, low, close
    Helpful but optional: volume, date, minute_of_day, dow, vwap,
                          vix, prev_close, prev_high, prev_low

    Returns a copy with new feature columns appended.
    All features are NaN-safe: missing upstream columns produce 0.0.
    """
    df = df.copy()
    c  = df['close']
    h  = df['high']
    lo = df['low']
    op = df['open'] if 'open' in df.columns else c.copy()

    has_vol  = 'volume' in df.columns and df['volume'].gt(0).any()
    vol      = df['volume'] if has_vol else pd.Series(1.0, index=df.index)

    has_date = 'date' in df.columns

    # ------------------------------------------------------------------
    # 1. PRICE STRUCTURE
    # ------------------------------------------------------------------

    # Returns: lagged by 1 so the feature window ends at t-1 (no label overlap).
    c_lag1 = c.shift(1)
    for n in [1, 3, 6, 12]:
        col = f'ret_{n}b'
        df[col] = c_lag1.pct_change(n) * 100
        # Zero out values whose lookback window crosses a session boundary.
        if has_date:
            overnight = df['date'] != df['date'].shift(n)
            df.loc[overnight, col] = 0.0

    # VWAP and VWAP-derived features
    if 'vwap' in df.columns:
        vwap = df['vwap']
    else:
        vwap = _build_intraday_vwap(h, lo, c, vol, df.get('date'))

    df['vwap_dist_pct'] = (c - vwap) / (vwap + EPS) * 100
    df['vwap_slope_3b'] = vwap.diff(3) / (vwap.shift(3) + EPS) * 100

    # Bar shape
    atr14, _ = _atr(h, lo, c, 14)
    bar_rng   = (h - lo).replace(0, np.nan)

    df['bar_range_norm'] = (h - lo) / (atr14 + EPS)
    df['bar_body_ratio'] = ((c - op).abs() / bar_rng).fillna(0.5).clip(0.0, 1.0)

    # Gap from prior session close (non-zero only at the first bar of a new day)
    if has_date:
        first_bar = df['date'] != df['date'].shift(1)
        prev_close_ref = c.shift(1)
        df['gap_from_prev_close'] = np.where(
            first_bar,
            (c - prev_close_ref) / (prev_close_ref + EPS) * 100,
            0.0,
        )
    else:
        df['gap_from_prev_close'] = 0.0

    # Position within today's range (0 = at day low, 1 = at day high)
    if has_date:
        day_high = df.groupby('date')['high'].transform('cummax')
        day_low  = df.groupby('date')['low'].transform('cummin')
    else:
        day_high = h.rolling(78, min_periods=1).max()   # ~78 5-min bars/day
        day_low  = lo.rolling(78, min_periods=1).min()
    df['day_position'] = (c - day_low) / (day_high - day_low + EPS)

    # ------------------------------------------------------------------
    # 2. MOMENTUM  (computed from current bar — no lag — intentionally real-time)
    # ------------------------------------------------------------------

    df['roc_1b'] = c.pct_change(1) * 100
    df['roc_3b'] = c.pct_change(3) * 100

    # Body direction normalised to ATR
    df['bar_close_vs_open'] = (c - op) / (atr14 + EPS)

    # Buying pressure: (close - low) / range
    df['buying_pressure'] = ((c - lo) / bar_rng).fillna(0.5).clip(0.0, 1.0)

    # Volume ratio vs 10-bar rolling average
    vol_ma = vol.rolling(10, min_periods=1).mean()
    df['vol_ratio'] = (vol / (vol_ma + EPS)).clip(0.0, 10.0)

    # Up-volume fraction over last 3 bars
    up_bars = (c > op).astype(float)
    df['up_vol_frac'] = up_bars.rolling(3, min_periods=1).mean()

    # Consecutive up / down closes (vectorised)
    df['consec_up'] = _consec_count(c > c.shift(1))
    df['consec_dn'] = _consec_count(c < c.shift(1))

    # ------------------------------------------------------------------
    # 3. REGIME SIGNALS
    # ------------------------------------------------------------------

    # ADX (20-period, using DMI helper)
    pdi, ndi, adx20_raw = _dmi(h, lo, c, 20)
    df['adx_20']       = adx20_raw
    df['adx_slope_3b'] = adx20_raw.diff(3)

    # Swing structure flags
    #   Higher-high: rolling 5-bar high > rolling 5-bar high 15 bars ago
    #   Higher-low : rolling 5-bar low  > rolling 5-bar low  15 bars ago
    #   Lower-high : rolling 5-bar high < rolling 5-bar high 15 bars ago
    #   Lower-low  : rolling 5-bar low  < rolling 5-bar low  15 bars ago
    recent_high = h.rolling(5, min_periods=1).max()
    prior_high  = h.shift(5).rolling(15, min_periods=5).max()
    recent_low  = lo.rolling(5, min_periods=1).min()
    prior_low   = lo.shift(5).rolling(15, min_periods=5).min()

    df['higher_high_flag'] = (recent_high > prior_high).astype(int)
    df['higher_low_flag']  = (recent_low  > prior_low).astype(int)
    df['lower_high_flag']  = (recent_high < prior_high).astype(int)
    df['lower_low_flag']   = (recent_low  < prior_low).astype(int)

    # Swing distance: how far price is from the 20-bar rolling high/low (in ATR units)
    roll_max_20 = h.rolling(20, min_periods=5).max()
    roll_min_20 = lo.rolling(20, min_periods=5).min()
    df['swing_high_dist'] = (roll_max_20 - c) / (atr14 + EPS)
    df['swing_low_dist']  = (c - roll_min_20) / (atr14 + EPS)

    # ------------------------------------------------------------------
    # 4. CONTEXT
    # ------------------------------------------------------------------

    # Time of day normalised to [0, 1]
    if 'minute_of_day' in df.columns:
        df['time_normalized'] = (df['minute_of_day'] / 375.0).clip(0.0, 1.0)
    else:
        df['time_normalized'] = 0.5

    # Day of week encoded to [0, 1]
    if 'dow' in df.columns:
        df['dow_encoded'] = (df['dow'].clip(0, 4) / 4.0)
    else:
        df['dow_encoded'] = 0.0

    # Intraday volatility: rolling 10-bar mean of bar range (absolute points)
    df['intraday_vol'] = (h - lo).rolling(10, min_periods=1).mean()

    # VIX level (daily; held constant per session)
    if 'vix' in df.columns:
        df['vix_level'] = df['vix'].fillna(15.0)
    else:
        df['vix_level'] = 15.0

    # Distance from prior day's close (as % of current price)
    if 'prev_close' in df.columns:
        df['prev_close_dist'] = (c - df['prev_close']) / (df['prev_close'] + EPS) * 100
    else:
        df['prev_close_dist'] = df['gap_from_prev_close']

    # Distance from nearest prior session high or low (whichever is closer)
    if 'prev_high' in df.columns and 'prev_low' in df.columns:
        dist_ph = (c - df['prev_high']).abs() / (c + EPS) * 100
        dist_pl = (c - df['prev_low']).abs()  / (c + EPS) * 100
        df['prev_hl_dist'] = dist_ph.where(dist_ph < dist_pl, dist_pl)
    else:
        df['prev_hl_dist'] = 0.0

    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_intraday_vwap(high: pd.Series, low: pd.Series, close: pd.Series,
                          volume: pd.Series,
                          date_col: 'pd.Series | None' = None) -> pd.Series:
    """Compute intraday VWAP, reset each session."""
    typical = (high + low + close) / 3.0
    tp_vol  = typical * volume

    if date_col is not None:
        cum_tpv = tp_vol.groupby(date_col).cumsum()
        cum_vol = volume.groupby(date_col).cumsum()
    else:
        cum_tpv = tp_vol.cumsum()
        cum_vol = volume.cumsum()

    return cum_tpv / (cum_vol + EPS)


def _consec_count(condition: pd.Series) -> pd.Series:
    """
    Vectorised consecutive-count of True values (resets to 0 on False).
    E.g., [F, T, T, T, F, T] → [0, 1, 2, 3, 0, 1]
    """
    result = np.zeros(len(condition), dtype=float)
    vals   = condition.values.astype(bool)
    for i in range(1, len(vals)):
        result[i] = result[i - 1] + 1 if vals[i] else 0
    return pd.Series(result, index=condition.index)
