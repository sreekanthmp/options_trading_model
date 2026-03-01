import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq

# ==========================
# FRACTIONAL DIFFERENTIATION
# (OFFLINE / TRAINING ONLY)
# ==========================

def causal_fracdiff(current_window: np.ndarray, d: float = 0.35, thresh: float = 1e-4) -> float:
    """
    Efficient causal fractional differentiation for live use.
    Only uses the last W bars (window = len(weights)).
    Args:
        current_window: np.ndarray, most recent W values (oldest first, newest last)
        d: float, fractional differencing order
        thresh: float, weight threshold
    Returns:
        float, fracdiff value for the current bar
    """
    w = _fracdiff_weights(d, thresh)
    if len(current_window) < len(w):
        # Not enough history, return np.nan
        return np.nan
    return np.dot(w, current_window[-len(w):])


def _fracdiff_weights(d: float, thresh: float = 1e-4) -> np.ndarray:
    w = [1.0]
    k = 1
    while True:
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < thresh:
            break
        w.append(wk)
        k += 1
    return np.array(w)


def fracdiff_series(s: pd.Series, d: float = 0.35) -> pd.Series:
    """
    Correct fractional differentiation (offline only).
    """
    w = _fracdiff_weights(d)[::-1]
    width = len(w)

    out = np.convolve(s.values, w, mode="full")
    out = out[width-1 : width-1 + len(s)]
    out[:width-1] = np.nan

    return pd.Series(out, index=s.index)


# ==========================
# FFT REGIME (OPTIONAL)
# ==========================

def extract_fft_regime(series: pd.Series, n_bars: int = 250) -> float:
    """
    Research-only market cycle estimator.
    DO NOT compute per-bar in live trading.
    """
    if len(series) < n_bars:
        return 0.0

    tail = series.tail(n_bars).values
    tail = tail - np.mean(tail)

    fft_vals = np.abs(rfft(tail))
    freqs = rfftfreq(len(tail), d=1.0)

    power = fft_vals[1:]
    if len(power) == 0:
        return 0.0

    idx = np.argmax(power) + 1
    return float(1.0 / freqs[idx]) if freqs[idx] > 0 else 0.0