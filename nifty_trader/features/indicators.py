import numpy as np
import pandas as pd

# Use a slightly smaller EPS to avoid interfering with high-precision instruments
EPS = 1e-10

def _atr(h, lo, c, n):
    """Average True Range and raw True Range."""
    tr = pd.concat(
        [h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(n).mean(), tr

def _rsi(c, n):
    """Relative Strength Index."""
    d = c.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / (l + EPS))

def _macd(c, fast=12, slow=26, sig=9):
    """Scale-invariant MACD Histogram percentage."""
    e1 = c.ewm(span=fast, adjust=False).mean()
    e2 = c.ewm(span=slow, adjust=False).mean()
    m = e1 - e2
    s = m.ewm(span=sig, adjust=False).mean()
    # Returns percentage distance from price for stationarity
    return (m - s) / (c + EPS) * 100

def _cci(h, lo, c, n=20):
    """Commodity Channel Index."""
    tp = (h + lo + c) / 3
    ma = tp.rolling(n).mean()
    md = (tp - ma).abs().rolling(n).mean()
    return (tp - ma) / (0.015 * md + EPS)

def _mfi(h, lo, c, vol, n=14):
    """Money Flow Index with volume-zero safety."""
    tp = (h + lo + c) / 3
    rmf = tp * (vol + 1) # Safeguard for index data
    pos = (tp > tp.shift(1)).astype(float) * rmf
    neg = (tp < tp.shift(1)).astype(float) * rmf
    mfr = pos.rolling(n).sum() / (neg.rolling(n).sum() + EPS)
    return 100 - 100 / (1 + mfr)

def _obv(c, vol):
    """Cumulative On-Balance Volume."""
    direction = np.sign(c.diff().fillna(0))
    return (direction * (vol + 1)).cumsum()

def _dmi(h, lo, c, n=14):
    """Directional Movement Index (PDI, NDI, ADX)."""
    tr = pd.concat(
        [h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()],
        axis=1
    ).max(axis=1)

    up_move = h.diff()
    down_move = -lo.diff()

    pdm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    ndm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_n = tr.rolling(n).mean()
    pdi = 100 * pd.Series(pdm, index=c.index).rolling(n).mean() / (atr_n + EPS)
    ndi = 100 * pd.Series(ndm, index=c.index).rolling(n).mean() / (atr_n + EPS)

    dx = 100 * (pdi - ndi).abs() / (pdi + ndi + EPS)
    adx = dx.rolling(n).mean()

    return pdi, ndi, adx

def _supertrend(h, lo, c, n=10, mult=3.0):
    """Supertrend direction: +1 when price is above upper band (bullish), -1 otherwise."""
    atr, _ = _atr(h, lo, c, n)
    hl2 = (h + lo) / 2
    upper_basic = hl2 + mult * atr
    lower_basic = hl2 - mult * atr

    upper = upper_basic.copy()
    lower = lower_basic.copy()
    direction = pd.Series(1, index=c.index, dtype=float)

    for i in range(1, len(c)):
        # Upper band
        if upper_basic.iloc[i] < upper.iloc[i - 1] or c.iloc[i - 1] > upper.iloc[i - 1]:
            upper.iloc[i] = upper_basic.iloc[i]
        else:
            upper.iloc[i] = upper.iloc[i - 1]
        # Lower band
        if lower_basic.iloc[i] > lower.iloc[i - 1] or c.iloc[i - 1] < lower.iloc[i - 1]:
            lower.iloc[i] = lower_basic.iloc[i]
        else:
            lower.iloc[i] = lower.iloc[i - 1]
        # Direction
        if direction.iloc[i - 1] == -1 and c.iloc[i] > upper.iloc[i]:
            direction.iloc[i] = 1
        elif direction.iloc[i - 1] == 1 and c.iloc[i] < lower.iloc[i]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

    return direction

def _keltner(c, atr14, n=20, mult=2.0):
    """Keltner Channel Position (0 to 1)."""
    ma = c.rolling(n).mean()
    upper = ma + mult * atr14
    lower = ma - mult * atr14
    return (c - lower) / (upper - lower + EPS)