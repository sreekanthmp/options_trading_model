"""Technical analysis: build_analysis, detect_micro_regime, score_feature."""
import logging
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from ..config import BAR_WIDTH, REGIME_NAMES, REGIME_ICONS, REGIME_CRISIS

logger = logging.getLogger(__name__)

# ==============================================================================
# 8. SIGNAL GENERATION & OPTIONS TRADE LOGIC
# ==============================================================================

def score_feature(value, low_bad, low_ok, high_ok, high_bad):
    """
    Map a feature value to [-2, +2] score.
    < low_bad -> -2 (very bearish / very bad)
    low_bad..low_ok -> -1
    low_ok..high_ok -> 0 (neutral)
    high_ok..high_bad -> +1
    > high_bad -> +2 (very bullish / very good)
    """
    if value <= low_bad:  return -2
    if value <= low_ok:   return -1
    if value <= high_ok:  return 0
    if value <= high_bad: return 1
    return 2


def build_analysis(row: pd.Series, current_regime: int) -> dict:
    """
    Compute full technical analysis scores and interpretation for dashboard.
    Returns a dict of analysis components.
    """
    # ---- Momentum score (RSI / Stoch / Williams) ----
    rsi   = float(row.get('rsi_14', 50))
    stoch = float(row.get('stoch_k', 50))
    willr = float(row.get('willr', -50))
    rsi_s = score_feature(rsi,   20, 35, 65, 80)   # >80 overbought = bearish
    stk_s = score_feature(stoch, 15, 25, 75, 85)
    wil_s = score_feature(willr, -95, -80, -20, -5)
    momentum_score = round((rsi_s + stk_s + wil_s) / 3, 2)

    # ---- Trend score (EMAs, ADX, Supertrend) ----
    e9_21   = float(row.get('ema9_21',  0))
    e21_50  = float(row.get('ema21_50', 0))
    adx     = float(row.get('adx_14',   0))
    sup     = float(row.get('supertrend', 1))
    dmi_diff= float(row.get('dmi_diff',  0))
    trend_score = round(
        (e9_21 * 0.25 + e21_50 * 0.25 + sup * 0.25 + np.sign(dmi_diff) * 0.25), 2
    )   # -1..+1

    # ---- Volatility / IV analysis ----
    iv      = float(row.get('iv_proxy', 0))
    iv_rank = float(row.get('iv_rank_approx', 50))
    bb_sq   = float(row.get('bb_squeeze', 0))
    bb_pos  = float(row.get('bb_pos', 0.5))
    bb_wid  = float(row.get('bb_width', 0))

    # iv_proxy thresholds (annualised %): 0.5=low, 1.0=normal, 1.5=high, 2.0=crisis
    if iv < 0.5:    iv_regime = "IV LOW"
    elif iv < 1.0:  iv_regime = "IV NORMAL-LOW"
    elif iv < 1.5:  iv_regime = "IV NORMAL-HIGH"
    else:           iv_regime = "IV HIGH (premium)"

    # ---- Volume / Flow ----
    mfi     = float(row.get('mfi_14', 50))
    obv_s   = float(row.get('obv_slope', 0))
    vol_rat = float(row.get('vol_ratio', 1))
    flow_score = score_feature(mfi, 20, 35, 65, 80)

    # ---- VWAP analysis ----
    vwap_d  = float(row.get('vwap_dist', 0))
    ab_vwap = int(row.get('above_vwap', 0))

    # ---- Support / Resistance ----
    dist_hi = float(row.get('dist_hi_60', 0))
    dist_lo = float(row.get('dist_lo_60', 0))
    pivot_hi= float(row.get('dist_pivot_hi', 0))
    pivot_lo= float(row.get('dist_pivot_lo', 0))

    # ---- Opening Range ----
    or_pos  = float(row.get('or_pos', 0.5))
    or_bk_u = int(row.get('or_break_up', 0))
    or_bk_d = int(row.get('or_break_dn', 0))

    # ---- Overall bias ----
    # Combines momentum, trend, VWAP, OR position
    vwap_bias  = 1 if ab_vwap else -1
    or_bias    = 1 if or_bk_u else (-1 if or_bk_d else 0)
    macd_bias  = np.sign(float(row.get('macd_h', 0)))
    overall_score = round(
        momentum_score * 0.25 +
        trend_score * 0.35 +
        vwap_bias  * 0.20 +
        or_bias    * 0.10 +
        macd_bias  * 0.10, 3
    )   # range roughly -1 to +1

    if overall_score > 0.35:   overall_bias = "BULLISH"
    elif overall_score > 0.10: overall_bias = "MILD BULLISH"
    elif overall_score > -0.10:overall_bias = "NEUTRAL"
    elif overall_score > -0.35:overall_bias = "MILD BEARISH"
    else:                      overall_bias = "BEARISH"

    return {
        'rsi': rsi, 'stoch_k': stoch, 'willr': willr,
        'momentum_score': momentum_score,
        'trend_score': trend_score,
        'e9_21': e9_21, 'e21_50': e21_50, 'adx': adx,
        'supertrend': sup, 'dmi_diff': dmi_diff,
        'iv': iv, 'iv_rank': iv_rank, 'iv_regime': iv_regime,
        'bb_squeeze': bb_sq, 'bb_pos': bb_pos, 'bb_width': bb_wid,
        'mfi': mfi, 'obv_slope': obv_s, 'vol_ratio': vol_rat,
        'flow_score': flow_score,
        'vwap_dist': vwap_d, 'above_vwap': ab_vwap,
        'dist_hi_60': dist_hi, 'dist_lo_60': dist_lo,
        'pivot_hi': pivot_hi, 'pivot_lo': pivot_lo,
        'or_pos': or_pos, 'or_break_up': or_bk_u, 'or_break_dn': or_bk_d,
        'overall_score': overall_score,
        'overall_bias': overall_bias,
        'regime_name': REGIME_NAMES[current_regime],
        'regime_icon': REGIME_ICONS[current_regime],
        'macd_h': float(row.get('macd_h', 0)),
        'cci': float(row.get('cci_20', 0)),
        'bear_3bar': int(row.get('bear_3bar', 0)),
        'bull_3bar': int(row.get('bull_3bar', 0)),
        'price_accel': float(row.get('price_accel', 0)),
        'consec_green': int(row.get('consec_green', 0)),
        'consec_red': int(row.get('consec_red', 0)),
        'day_above_ma50': int(row.get('day_above_ma50', 1)),
        'day_above_ma200': int(row.get('day_above_ma200', 1)),
        'gap_pct': float(row.get('gap_pct', 0)),
        'weekly_ret': float(row.get('weekly_ret', 0)),
        'day_trend_strength': float(row.get('day_trend_strength', 0)),
    }


# ==============================================================================
# 8. INTRADAY MICRO-REGIME DETECTION
# ==============================================================================

def detect_micro_regime(df_recent: pd.DataFrame) -> str:
    """
    Lightweight intraday micro-regime detector using the last 60 1-min candles.

    Returns one of:
      'TRENDING_UP'   -- strong upward momentum, above average range
      'TRENDING_DN'   -- strong downward momentum
      'RANGING'       -- low momentum, choppy price action
      'BREAKOUT'      -- range expansion after squeeze (BB squeeze then pop)
      'UNKNOWN'       -- insufficient data

    Logic:
      1. ATR-to-range ratio: if current ATR > 1.3x 60-bar ATR mean -> trending
      2. Direction: last 20-bar net return vs ATR band determines up/down
      3. Range contraction: if ATR < 0.7x mean for last 10 bars -> squeeze/ranging
      4. Breakout: squeeze resolved (was < 0.7x, now > 1.1x)
    """
    if df_recent is None or len(df_recent) < 30:
        return 'UNKNOWN'

    df = df_recent.tail(60).copy()
    if 'atr_14' not in df.columns or df['atr_14'].isna().all():
        return 'UNKNOWN'

    atr_vals = df['atr_14'].dropna()
    if len(atr_vals) < 10:
        return 'UNKNOWN'

    atr_mean   = atr_vals.mean()
    atr_now    = atr_vals.iloc[-1]
    atr_ratio  = atr_now / (atr_mean + 1e-9)

    # Net return over last 20 bars relative to ATR
    close = df['close'].values
    net20 = (close[-1] - close[max(-20, -len(close))]) / (atr_mean + 1e-9)

    # Check if prior 10 bars were in a squeeze (low ATR)
    prior_atr_ratio = atr_vals.iloc[-11:-1].mean() / (atr_mean + 1e-9) if len(atr_vals) > 11 else 1.0

    if prior_atr_ratio < 0.70 and atr_ratio > 1.10:
        return 'BREAKOUT'
    elif atr_ratio > 1.30:
        return 'TRENDING_UP' if net20 > 0.3 else 'TRENDING_DN'
    elif atr_ratio < 0.70:
        return 'RANGING'
    else:
        return 'RANGING' if abs(net20) < 0.5 else ('TRENDING_UP' if net20 > 0 else 'TRENDING_DN')


