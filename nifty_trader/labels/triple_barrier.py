"""
Triple-Barrier Labeling with IV-Aware Alpha Regression
v4.2 – Lookahead Fix

Lookahead fixes applied (audit-driven):
  1. Regime array is lagged by 1 bar before being used to set barrier widths.
     Without this, the barrier width at bar i uses the regime label computed
     from bar i's close — a subtle lookahead because the regime label is partly
     determined by bar i itself.
  2. atr_baseline_vals already uses a causal rolling mean (only past bars),
     so no change needed there.
  3. k_vals and tp_call_mult_vals now derive from regime_arr_lagged to ensure
     the barrier seen by bar i was knowable strictly before bar i's close.
"""

import logging
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from ..config import (
    TB_MULT_TRENDING,
    TB_MULT_RANGING,
    TB_MULT_CRISIS,
    TB_BARS,
    TB_MIN_MOVE_PCT,
    REGIME_TRENDING,
    REGIME_RANGING,
    REGIME_CRISIS,
    TB_VELOCITY_DECAY_LAMBDA,
    TB_PUT_TIME_DECAY_EXTRA,
    TB_CALL_TP_TRENDING_MULT,
    NUMBA_AVAILABLE,
    jit,
    prange,
)

logger = logging.getLogger(__name__)
EPS = 1e-9


# ==============================================================================
# NUMBA KERNEL — v4.1 FINAL
# ==============================================================================

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _compute_barriers_v4(
    c,
    atr,
    iv,
    H_base,
    n,
    k_vals,
    tp_call_mult_vals,
    atr_baseline_vals,
    velocity_lambda,
    put_decay,
    min_move_pct,
    bars_per_day,
):
    labels = np.full(n, np.nan)
    ev_weights = np.full(n, np.nan)
    fut_ret = np.full(n, np.nan)
    barrier_type = np.zeros(n, dtype=np.int32)

    MIN_H = max(1, H_base // 2)
    MAX_H = int(H_base * 1.5)
    SQRT_BARS = 1.0 / np.sqrt(bars_per_day + EPS)

    for i in prange(n - H_base):

        if np.isnan(k_vals[i]) or np.isnan(c[i]) or c[i] <= 0:
            continue

        # 1. Dynamic Horizon (Volatility Clock)
        atr_ratio = atr[i] / (atr_baseline_vals[i] + EPS)

        if atr_ratio > 1.5:
            H = int(H_base * 0.75)
        elif atr_ratio < 0.7:
            H = int(H_base * 1.25)
        else:
            H = H_base

        H = max(MIN_H, min(H, MAX_H))
        entry = c[i]

        # 2. Expected Move (IV preferred, ATR fallback)
        time_sqrt = np.sqrt(H) * SQRT_BARS
        iv_i = iv[i]

        if not np.isnan(iv_i) and iv_i > 0:
            vol_width = entry * iv_i * time_sqrt * k_vals[i]
        else:
            vol_width = k_vals[i] * atr[i]

        if vol_width <= 0:
            continue

        tp = entry + vol_width * tp_call_mult_vals[i]
        sl = entry - vol_width

        # 3. Forward Scan
        hit_bar = -1
        hit_dir = 0
        scan_end = min(i + H + 1, n)

        for j in range(i + 1, scan_end):
            if c[j] >= tp:
                hit_bar = j - i
                hit_dir = 1
                break
            elif c[j] <= sl:
                hit_bar = j - i
                hit_dir = -1
                break

        if hit_bar == -1:
            barrier_type[i] = 0
            continue

        # 4. Move-Normalized Velocity Decay
        actual_move_pct = abs(c[i + hit_bar] - entry) / (entry * 0.01 + EPS)
        expected_move_pct = vol_width / entry * 100 + EPS
        move_norm = actual_move_pct / expected_move_pct
        ev_vel = np.exp(-velocity_lambda * move_norm)

        # 5. Horizon-Aware Directional Penalty
        if hit_dir == -1:
            theta_mult = H / H_base
            ev_dir = (1.0 - put_decay * theta_mult) * 0.85
            barrier_type[i] = -1
        else:
            ev_dir = 1.0
            barrier_type[i] = 1

        ev_weight = ev_vel * ev_dir
        hit_ret_val = (c[i + hit_bar] - entry) / entry * 100

        # 6. Continuous Alpha Target
        labels[i] = np.tanh(hit_ret_val / (min_move_pct + EPS)) * ev_weight
        ev_weights[i] = ev_weight
        fut_ret[i] = hit_ret_val

    return labels, ev_weights, fut_ret, barrier_type


# ==============================================================================
# PYTHON FALLBACK — LOGIC IDENTICAL
# ==============================================================================

def _compute_barriers_fallback_v4(
    c,
    atr,
    iv,
    H_base,
    n,
    k_vals,
    tp_call_mult_vals,
    atr_baseline_vals,
    velocity_lambda,
    put_decay,
    min_move_pct,
    bars_per_day,
    hi=None,
    lo=None,
):
    labels = np.full(n, np.nan)
    ev_weights = np.full(n, np.nan)
    fut_ret = np.full(n, np.nan)
    barrier_type = np.zeros(n, dtype=int)

    MIN_H = max(1, H_base // 2)
    MAX_H = int(H_base * 1.5)
    SQRT_BARS = 1.0 / np.sqrt(bars_per_day + EPS)

    for i in range(n - H_base):

        if np.isnan(k_vals[i]) or np.isnan(c[i]) or c[i] <= 0:
            continue

        atr_ratio = atr[i] / (atr_baseline_vals[i] + EPS)

        if atr_ratio > 1.5:
            H = int(H_base * 0.75)
        elif atr_ratio < 0.7:
            H = int(H_base * 1.25)
        else:
            H = H_base

        H = max(MIN_H, min(H, MAX_H))
        entry = c[i]

        time_sqrt = np.sqrt(H) * SQRT_BARS
        iv_i = iv[i]

        if not np.isnan(iv_i) and iv_i > 0:
            vol_width = entry * iv_i * time_sqrt * k_vals[i]
        else:
            vol_width = k_vals[i] * atr[i]

        if vol_width <= 0:
            continue

        tp = entry + vol_width * tp_call_mult_vals[i]
        sl = entry - vol_width

        hit_bar = -1
        hit_dir = 0
        scan_end = min(i + H + 1, n)

        # Use high/low for barrier detection (captures intrabar wicks)
        # falls back to close if hi/lo arrays not provided
        use_hl = (hi is not None and lo is not None)
        for j in range(i + 1, scan_end):
            bar_hi = hi[j] if use_hl else c[j]
            bar_lo = lo[j] if use_hl else c[j]
            if bar_hi >= tp:
                hit_bar = j - i
                hit_dir = 1
                break
            elif bar_lo <= sl:
                hit_bar = j - i
                hit_dir = -1
                break

        if hit_bar == -1:
            barrier_type[i] = 0
            continue

        # Return is still close-to-close (realistic execution)
        actual_move_pct = abs(c[i + hit_bar] - entry) / (entry * 0.01 + EPS)
        expected_move_pct = vol_width / entry * 100 + EPS
        move_norm = actual_move_pct / expected_move_pct
        ev_vel = np.exp(-velocity_lambda * move_norm)

        if hit_dir == -1:
            theta_mult = H / H_base
            ev_dir = (1.0 - put_decay * theta_mult) * 0.85
            barrier_type[i] = -1
        else:
            ev_dir = 1.0
            barrier_type[i] = 1

        ev_weight = ev_vel * ev_dir
        hit_ret_val = (c[i + hit_bar] - entry) / entry * 100

        labels[i] = np.tanh(hit_ret_val / (min_move_pct + EPS)) * ev_weight
        ev_weights[i] = ev_weight
        fut_ret[i] = hit_ret_val

    return labels, ev_weights, fut_ret, barrier_type


# ==============================================================================
# PUBLIC WRAPPER
# ==============================================================================

def triple_barrier_labels(
    df: pd.DataFrame,
    horizon: int,
    regime_series: pd.Series | None = None,
) -> pd.DataFrame:

    H_base = TB_BARS[horizon]
    n = len(df)

    if n <= H_base:
        return df

    c   = df["close"].values
    hi  = df["high"].values  if "high"  in df.columns else None
    lo  = df["low"].values   if "low"   in df.columns else None
    atr = df["atr_14"].values
    iv  = df["iv"].values if "iv" in df.columns else np.full(n, np.nan)

    bars_per_day = df["minute_of_day"].nunique() if "minute_of_day" in df.columns else 375

    atr_ser = pd.Series(atr).rolling(20 * bars_per_day, min_periods=bars_per_day).mean()
    atr_baseline_vals = atr_ser.fillna(np.nanmean(atr)).values

    # Build the regime array and lag it by 1 bar so the barrier width at
    # bar i uses only information known BEFORE bar i (strict causal labeling).
    regime_arr = np.full(n, REGIME_RANGING, dtype=np.int32)
    if regime_series is not None and "date" in df.columns:
        regime_arr = (
            regime_series.reindex(df["date"])
            .fillna(REGIME_RANGING)
            .values.astype(np.int32)
        )

    # LAG BY 1: shift regime forward so barrier at bar i uses regime[i-1].
    # The first bar falls back to REGIME_RANGING (the conservative default).
    regime_arr_lagged = np.empty_like(regime_arr)
    regime_arr_lagged[0] = REGIME_RANGING
    regime_arr_lagged[1:] = regime_arr[:-1]

    k_vals = np.where(
        regime_arr_lagged == REGIME_TRENDING,
        TB_MULT_TRENDING[horizon],
        TB_MULT_RANGING[horizon],
    )

    tp_call_mult_vals = np.where(
        regime_arr_lagged == REGIME_TRENDING,
        TB_CALL_TP_TRENDING_MULT,
        1.0,
    )

    # Use crisis barrier width for Crisis regime
    k_vals[regime_arr_lagged == REGIME_CRISIS] = TB_MULT_CRISIS[horizon]

    # Always use Python fallback — it supports hi/lo intrabar detection.
    # The Numba kernel uses close-only and doesn't accept hi/lo arrays.
    labels, ev, fut_ret, barrier = _compute_barriers_fallback_v4(
        c,
        atr,
        iv,
        H_base,
        n,
        k_vals,
        tp_call_mult_vals,
        atr_baseline_vals,
        TB_VELOCITY_DECAY_LAMBDA,
        TB_PUT_TIME_DECAY_EXTRA,
        TB_MIN_MOVE_PCT,
        bars_per_day,
        hi=hi,
        lo=lo,
    )

    df[f"label_{horizon}m"] = labels
    df[f"ev_{horizon}m"] = ev
    df[f"fut_ret_{horizon}m"] = fut_ret
    df[f"barrier_{horizon}m"] = barrier

    return df