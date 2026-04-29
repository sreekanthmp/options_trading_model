"""
Trend Regime Detector — ADX(20) + price swing structure.

Design: simple, causal, no future leakage.

Regime outputs:
  REGIME_TRENDING (0): ADX > 22 AND consistent swing direction
  REGIME_RANGING  (1): any other condition → BLOCKED

No HMM, no hidden states, no complex state machines.

Usage (offline):
    regimes = detect_trend_regime(df)  # returns pd.Series per bar

Usage (live, single bar):
    regime = live_regime_from_row(row)
    direction = live_direction_from_row(row)
"""
import numpy as np
import pandas as pd
import logging

from ..config import REGIME_TRENDING, REGIME_RANGING

logger = logging.getLogger(__name__)

EPS = 1e-9
ADX_TREND_THRESHOLD = 22.0   # ADX must exceed this for trending candidate
ADX_PERIOD          = 20     # ADX smoothing period
SWING_WINDOW        = 5      # local max/min half-window for swing detection
MIN_SWINGS          = 3      # minimum confirmed swings to assess structure


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _adx_series(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = ADX_PERIOD) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute (ADX, +DI, -DI) series causally.
    Returns three pd.Series of the same length as inputs.
    """
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up_move   = high.diff()
    down_move = -low.diff()

    pdm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    ndm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_n = tr.rolling(period).mean()
    pdi   = 100 * pd.Series(pdm, index=close.index).rolling(period).mean() / (atr_n + EPS)
    ndi   = 100 * pd.Series(ndm, index=close.index).rolling(period).mean() / (atr_n + EPS)

    dx  = 100 * (pdi - ndi).abs() / (pdi + ndi + EPS)
    adx = dx.rolling(period).mean()

    return adx, pdi, ndi


def _swing_highs_lows(highs: np.ndarray, lows: np.ndarray,
                      half_window: int = SWING_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """
    Find swing highs and swing lows in a causal price window.

    Swing high: bar i is a local maximum within ±half_window.
    Swing low:  bar i is a local minimum within ±half_window.

    Returns (swing_highs, swing_lows) as 1D arrays.
    """
    n = len(highs)
    s_highs: list[float] = []
    s_lows:  list[float] = []

    # Only iterate up to n - half_window so the right-side window is causal.
    # In offline batch mode (detect_trend_regime) the full window is available;
    # in live mode only left-side bars are checked (see live_regime_from_row).
    for i in range(half_window, n - half_window):
        left  = slice(i - half_window, i)
        right = slice(i + 1, i + half_window + 1)
        if highs[i] >= np.max(highs[left]) and highs[i] >= np.max(highs[right]):
            s_highs.append(highs[i])
        if lows[i] <= np.min(lows[left]) and lows[i] <= np.min(lows[right]):
            s_lows.append(lows[i])

    return np.array(s_highs), np.array(s_lows)


def _structure_direction(s_highs: np.ndarray, s_lows: np.ndarray,
                          min_swings: int = MIN_SWINGS) -> str:
    """
    Determine price structure direction from swing arrays.

    Returns: 'UP', 'DOWN', or 'NEUTRAL'
    """
    if len(s_highs) < min_swings or len(s_lows) < min_swings:
        return 'NEUTRAL'

    last_h = s_highs[-min_swings:]
    last_l = s_lows[-min_swings:]

    higher_highs = all(last_h[j] > last_h[j - 1] for j in range(1, len(last_h)))
    higher_lows  = all(last_l[j] > last_l[j - 1] for j in range(1, len(last_l)))
    lower_highs  = all(last_h[j] < last_h[j - 1] for j in range(1, len(last_h)))
    lower_lows   = all(last_l[j] < last_l[j - 1] for j in range(1, len(last_l)))

    if higher_highs and higher_lows:
        return 'UP'
    if lower_highs and lower_lows:
        return 'DOWN'
    return 'NEUTRAL'


# ---------------------------------------------------------------------------
# Offline batch regime detection
# ---------------------------------------------------------------------------

def detect_trend_regime(df: pd.DataFrame, lookback: int = 100) -> pd.Series:
    """
    Compute causal trend regime for every bar in df.

    Args:
        df:       DataFrame with columns [high, low, close].
        lookback: Rolling window of bars used to assess swing structure.

    Returns:
        pd.Series of REGIME_TRENDING (0) or REGIME_RANGING (1), same index as df.

    Causal guarantee: bar i uses only bars 0..i.
    """
    n = len(df)
    regimes = np.full(n, REGIME_RANGING, dtype=int)

    high  = df['high'].values
    low   = df['low'].values

    adx_s, _, _ = _adx_series(df['high'], df['low'], df['close'])
    adx = adx_s.values

    for i in range(lookback, n):
        adx_val = adx[i]
        if np.isnan(adx_val) or adx_val < ADX_TREND_THRESHOLD:
            continue   # stays RANGING

        win_start = max(0, i - lookback)
        # CAUSAL FIX: end the swing-detection window SWING_WINDOW bars before bar i
        # so the right-side confirmation in _swing_highs_lows never reads future data.
        # Without this, bar i-half_window's right side includes bars up to i — look-ahead.
        win_end = max(win_start + SWING_WINDOW * 2 + 1, i - SWING_WINDOW)
        h_win = high[win_start:win_end]
        l_win = low[win_start:win_end]

        s_highs, s_lows = _swing_highs_lows(h_win, l_win)
        direction = _structure_direction(s_highs, s_lows)

        if direction in ('UP', 'DOWN'):
            regimes[i] = REGIME_TRENDING

    return pd.Series(regimes, index=df.index, name='trend_regime')


def detect_trend_direction_series(df: pd.DataFrame, lookback: int = 100) -> pd.Series:
    """
    Like detect_trend_regime but returns 'UP', 'DOWN', or 'NEUTRAL' per bar.
    Used for label alignment (to match model direction with regime direction).
    """
    n = len(df)
    directions = ['NEUTRAL'] * n

    high = df['high'].values
    low  = df['low'].values

    adx_s, _, _ = _adx_series(df['high'], df['low'], df['close'])
    adx = adx_s.values

    for i in range(lookback, n):
        if np.isnan(adx[i]) or adx[i] < ADX_TREND_THRESHOLD:
            continue
        win_start = max(0, i - lookback)
        # CAUSAL FIX: same as detect_trend_regime — exclude right-side lookahead
        win_end   = max(win_start + SWING_WINDOW * 2 + 1, i - SWING_WINDOW)
        s_h, s_l  = _swing_highs_lows(high[win_start:win_end], low[win_start:win_end])
        directions[i] = _structure_direction(s_h, s_l)

    return pd.Series(directions, index=df.index, name='trend_direction')


# ---------------------------------------------------------------------------
# Live (single-bar) regime detection
# ---------------------------------------------------------------------------

def live_regime_from_row(row: 'pd.Series | dict',
                          adx_threshold: float = ADX_TREND_THRESHOLD) -> int:
    """
    Real-time regime detection from a single bar's pre-computed features.

    Requires features computed by trend_features.add_trend_features():
        adx_20, higher_high_flag, higher_low_flag, lower_high_flag, lower_low_flag,
        minute_of_day.

    Returns REGIME_TRENDING (0) or REGIME_RANGING (1).

    Block rules:
      • First 15 min of session (open noise)
      • Last 30 min of session (close noise)
    """
    minute_of_day = int(row.get('minute_of_day', 30))

    # Session noise blocks
    if minute_of_day < 15 or minute_of_day >= 345:
        return REGIME_RANGING

    # ADX threshold
    adx = float(row.get('adx_20', row.get('adx_14', 0.0)))
    if np.isnan(adx) or adx < adx_threshold:
        return REGIME_RANGING

    # Swing structure flags (computed by trend_features)
    hh = int(row.get('higher_high_flag', 0))
    hl = int(row.get('higher_low_flag',  0))
    lh = int(row.get('lower_high_flag',  0))
    ll = int(row.get('lower_low_flag',   0))

    uptrend   = (hh == 1 and hl == 1)
    downtrend = (lh == 1 and ll == 1)

    if uptrend or downtrend:
        return REGIME_TRENDING

    return REGIME_RANGING


def live_direction_from_row(row: 'pd.Series | dict') -> str:
    """
    Real-time trend direction from pre-computed structure flags.

    Returns: 'UP', 'DOWN', or 'NEUTRAL'.

    Rules (priority order):
      1. higher_high_flag + higher_low_flag → UP
      2. lower_high_flag  + lower_low_flag  → DOWN
      3. vwap_slope_3b > 0.02              → UP  (fallback)
      4. vwap_slope_3b < -0.02             → DOWN (fallback)
      5. NEUTRAL
    """
    hh = int(row.get('higher_high_flag', 0))
    hl = int(row.get('higher_low_flag',  0))
    lh = int(row.get('lower_high_flag',  0))
    ll = int(row.get('lower_low_flag',   0))

    if hh == 1 and hl == 1:
        return 'UP'
    if lh == 1 and ll == 1:
        return 'DOWN'

    vwap_slope = float(row.get('vwap_slope_3b', 0.0))
    if vwap_slope > 0.02:
        return 'UP'
    if vwap_slope < -0.02:
        return 'DOWN'

    return 'NEUTRAL'
