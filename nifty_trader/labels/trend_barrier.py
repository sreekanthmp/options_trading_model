"""
Triple-Barrier Labeling for Trend-Following System.

Horizon: 30 minutes on 5-min bars = 6 bars forward.

Barrier configuration (aligned with 2:1 reward-to-risk):
  Upper barrier: +0.5% from entry close  → label = 1 (win)
  Lower barrier: -0.3% from entry close  → label = 0 (loss)
  Time barrier:  6 bars (30 min)         → label = 0 (no edge)

Asymmetric barriers reflect the R:R target.
No lookahead: labels for bar i use only bars i+1 .. i+6.

Usage:
    from nifty_trader.labels.trend_barrier import make_trend_labels
    df['label'] = make_trend_labels(df)
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default barrier parameters (can be overridden via function arguments)
# ---------------------------------------------------------------------------
DEFAULT_H           = 6      # 6 × 5-min bars = 30-min horizon
DEFAULT_UPPER_PCT   = 0.005  # +0.5% to hit upper barrier
DEFAULT_LOWER_PCT   = 0.003  # -0.3% to hit lower barrier (absolute value)
EPS                 = 1e-9


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_trend_labels(
    df: pd.DataFrame,
    h: int   = DEFAULT_H,
    upper_pct: float = DEFAULT_UPPER_PCT,
    lower_pct: float = DEFAULT_LOWER_PCT,
) -> pd.Series:
    """
    Compute triple-barrier labels for every bar in df.

    Label = 1 if the upper barrier (+upper_pct) is hit within h bars BEFORE
            the lower barrier (-lower_pct) is hit.
    Label = 0 if the lower barrier is hit first OR the time barrier expires.

    Args:
        df:        DataFrame with a 'close' column, sorted by time ascending.
        h:         Forward horizon in bars (default 6 = 30 min on 5-min data).
        upper_pct: Upper barrier as a fraction of entry close (e.g. 0.005 = 0.5%).
        lower_pct: Lower barrier as a fraction of entry close (e.g. 0.003 = 0.3%).

    Returns:
        pd.Series of int (0 or 1), NaN for the last h bars (no full forward window).

    No lookahead guarantee:
        Label at bar i is computed using close[i+1 .. i+h] only.
        close[i] (the entry bar) is not part of the label forward window.
    """
    closes   = df['close'].values.astype(float)
    n        = len(closes)
    labels   = np.full(n, np.nan)

    for i in range(n - h):
        entry = closes[i]
        if entry <= 0 or np.isnan(entry):
            continue

        upper = entry * (1.0 + upper_pct)
        lower = entry * (1.0 - lower_pct)

        label = 0   # default: time barrier or lower hit
        for j in range(i + 1, i + h + 1):
            price = closes[j]
            if price >= upper:
                label = 1
                break
            if price <= lower:
                label = 0
                break

        labels[i] = label

    series = pd.Series(labels, index=df.index, name='label')
    logger.debug(
        f"[TrendBarrier] h={h} upper={upper_pct*100:.2f}% lower={lower_pct*100:.2f}%  "
        f"total={n}  labeled={(~series.isna()).sum()}  "
        f"positive_rate={series.mean():.3f}"
    )
    return series


def make_trend_labels_with_meta(
    df: pd.DataFrame,
    h: int           = DEFAULT_H,
    upper_pct: float = DEFAULT_UPPER_PCT,
    lower_pct: float = DEFAULT_LOWER_PCT,
) -> pd.DataFrame:
    """
    Extended labeling that returns a DataFrame with the label plus metadata:
        label          — 0 or 1 (see make_trend_labels)
        barrier_hit    — 'upper', 'lower', or 'time'
        bars_to_exit   — number of bars until barrier was hit
        max_excursion  — maximum favourable % move before exit
        max_adverse    — maximum adverse % move before exit

    Useful for debugging and model analysis.
    """
    closes = df['close'].values.astype(float)
    n      = len(closes)

    labels       = np.full(n, np.nan)
    barrier_hit  = np.full(n, '', dtype=object)
    bars_to_exit = np.full(n, np.nan)
    max_exc      = np.full(n, np.nan)
    max_adv      = np.full(n, np.nan)

    for i in range(n - h):
        entry = closes[i]
        if entry <= 0 or np.isnan(entry):
            continue

        upper = entry * (1.0 + upper_pct)
        lower = entry * (1.0 - lower_pct)

        lbl    = 0
        hit    = 'time'
        bars   = h
        best   = 0.0
        worst  = 0.0

        for k, j in enumerate(range(i + 1, i + h + 1), start=1):
            price     = closes[j]
            pct_chg   = (price - entry) / (entry + EPS) * 100
            best      = max(best,  pct_chg)
            worst     = min(worst, pct_chg)

            if price >= upper:
                lbl  = 1
                hit  = 'upper'
                bars = k
                break
            if price <= lower:
                lbl  = 0
                hit  = 'lower'
                bars = k
                break

        labels[i]       = lbl
        barrier_hit[i]  = hit
        bars_to_exit[i] = bars
        max_exc[i]      = best
        max_adv[i]      = worst

    return pd.DataFrame({
        'label':          labels,
        'barrier_hit':    barrier_hit,
        'bars_to_exit':   bars_to_exit,
        'max_excursion':  max_exc,
        'max_adverse':    max_adv,
    }, index=df.index)


def label_stats(labels: pd.Series) -> dict:
    """Return a summary statistics dict for a label series."""
    valid = labels.dropna()
    return {
        'total':         len(valid),
        'positive':      int(valid.sum()),
        'negative':      int((valid == 0).sum()),
        'positive_rate': round(float(valid.mean()), 4) if len(valid) > 0 else 0.0,
    }
