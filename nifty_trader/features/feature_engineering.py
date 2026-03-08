"""Feature engineering: 1-min, HTF, and daily features (120+ features)."""
import os, logging
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from ..config import FRACDIFF_D, FRACDIFF_THRESH, REGIME_RANGING, REGIME_TRENDING
from .indicators import _atr, _rsi, _macd, _cci, _mfi, _obv, _dmi, _supertrend, _keltner
from .fractional_diff import fracdiff_series

logger = logging.getLogger(__name__)
EPS = 1e-9


# ==============================================================================
# 1-MIN FEATURES
# ==============================================================================

def add_1min_features_production(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, lo = df['close'], df['high'], df['low']
    op = df['open'] if 'open' in df.columns else c
    if 'volume' in df.columns and df['volume'].gt(1).any():
        vol = df['volume']
    else:
        # No real volume available (index tick-count feed or missing column).
        # Features that use raw volume magnitude (mfi_14, obv_slope,
        # vwap_vol_pressure) will be near-constant — they are excluded from
        # the training feature set via get_feature_cols() so this is safe.
        # pressure_ratio and tick_imbalance use np.sign(c.diff()) only, so
        # they remain valid regardless of volume quality.
        vol = pd.Series(1, index=df.index)
        logger.debug("Volume column missing or constant — using tick-direction proxies only")
    EPS = 1e-9

    bars_per_day = 375  # NIFTY session: 09:15–15:30 = 375 1-min bars, always fixed

    # ------------------------------------------------------------------
    # Returns  (LEAKAGE FIX: use shift(1) so ret_nm ends at bar T-1)
    # ------------------------------------------------------------------
    # Without shift: ret_5m at bar T = (close[T] - close[T-5]) / close[T-5].
    # The label predicts close[T+k], but close[T] is the ENTRY price — using
    # it as a feature and the label endpoint creates temporal overlap.
    # With shift(1): ret_5m = (close[T-1] - close[T-6]) / close[T-6].
    # The feature window [T-6, T-1] is fully disjoint from the label window
    # [T, T+k], so there is zero leakage.
    c_lag1 = c.shift(1)
    for n in [1, 3, 5, 10, 15, 30, 60]:
        df[f'ret_{n}m'] = c_lag1.pct_change(n) * 100

    # ------------------------------------------------------------------
    # ATR & IV
    # ------------------------------------------------------------------
    atr5,  _ = _atr(h, lo, c, 5)
    atr14, _ = _atr(h, lo, c, 14)
    df['atr_5']      = atr5
    df['atr_14']     = atr14
    df['atr_14_pct'] = atr14 / (c + EPS) * 100
    df['atr_ratio']  = atr5 / (atr14 + EPS)

    iv_proxy = df['atr_14_pct'] * np.sqrt(bars_per_day)
    df['iv_proxy'] = iv_proxy
    df['iv_final'] = df['iv'].fillna(iv_proxy) if 'iv' in df.columns else iv_proxy
    df['iv_final'] = df['iv_final'].clip(0.02, 3.0)

    iv_rank_window = min(60 * bars_per_day, len(df))
    iv_rank = df['iv_final'].rolling(iv_rank_window, min_periods=bars_per_day).rank(pct=True) * 100
    df['iv_rank']        = iv_rank
    df['iv_rank_approx'] = iv_rank   # alias used by signal_generator
    df['iv_pct_change']  = df['iv_final'].pct_change(bars_per_day) * 100

    # ------------------------------------------------------------------
    # Volatility (rolling std of returns)
    # ------------------------------------------------------------------
    r1 = c.pct_change(1) * 100
    for n in [5, 10, 20, 60]:
        df[f'vol_{n}m'] = r1.rolling(n).std()

    # ------------------------------------------------------------------
    # Bollinger Bands
    # ------------------------------------------------------------------
    ma20 = c.rolling(20).mean()
    s20  = c.rolling(20).std()
    bb_upper = ma20 + 2 * s20
    bb_lower = ma20 - 2 * s20
    bb_width  = (bb_upper - bb_lower) / (ma20 + EPS) * 100
    df['bb_pos']    = (c - bb_lower) / (bb_upper - bb_lower + EPS)
    df['bb_width']  = bb_width
    df['bb_squeeze'] = (bb_width < bb_width.rolling(125).mean()).astype(int)

    # ------------------------------------------------------------------
    # Keltner Channel
    # ------------------------------------------------------------------
    df['kc_pos'] = _keltner(c, atr14)

    # ------------------------------------------------------------------
    # EMAs
    # ------------------------------------------------------------------
    ema9   = c.ewm(span=9,   adjust=False).mean()
    ema21  = c.ewm(span=21,  adjust=False).mean()
    ema50  = c.ewm(span=50,  adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    df['ema9_dist']  = (c - ema9)   / (c + EPS) * 100
    df['ema21_dist'] = (c - ema21)  / (c + EPS) * 100
    df['ema50_dist'] = (c - ema50)  / (c + EPS) * 100
    df['ema200_dist']= (c - ema200) / (c + EPS) * 100
    df['ema9_21']    = ((ema9 > ema21).astype(int) - (ema9 < ema21).astype(int))
    df['ema21_50']   = ((ema21 > ema50).astype(int) - (ema21 < ema50).astype(int))
    df['ema9_slope'] = ema9.diff(3) / (ema9.shift(3) + EPS) * 100
    df['ema21_slope']= ema21.diff(5) / (ema21.shift(5) + EPS) * 100

    # ------------------------------------------------------------------
    # RSI (multiple periods + slope + divergence)
    # ------------------------------------------------------------------
    rsi7  = _rsi(c, 7)
    rsi14 = _rsi(c, 14)
    rsi21 = _rsi(c, 21)
    df['rsi_7']    = rsi7
    df['rsi_14']   = rsi14
    df['rsi_21']   = rsi21
    df['rsi_slope']= rsi14.diff(5)
    # Divergence: price makes higher high but RSI doesn't (bearish) or vice versa
    df['rsi_div']  = (c.diff(10).fillna(0) * rsi14.diff(10).fillna(0) < 0).astype(int)

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------
    macd_h = _macd(c)
    df['macd_h']    = macd_h
    df['macd_cross']= (macd_h > 0).astype(int).diff().abs().fillna(0).astype(int)

    # ------------------------------------------------------------------
    # Stochastic
    # ------------------------------------------------------------------
    low14  = lo.rolling(14).min()
    high14 = h.rolling(14).max()
    stoch_k = (c - low14) / (high14 - low14 + EPS) * 100
    stoch_d = stoch_k.rolling(3).mean()
    df['stoch_k']    = stoch_k
    df['stoch_d']    = stoch_d
    df['stoch_cross']= ((stoch_k > stoch_d).astype(int).diff().abs().fillna(0)).astype(int)

    # ------------------------------------------------------------------
    # Williams %R
    # ------------------------------------------------------------------
    df['willr'] = -100 * (high14 - c) / (high14 - low14 + EPS)

    # ------------------------------------------------------------------
    # CCI
    # ------------------------------------------------------------------
    df['cci_20'] = _cci(h, lo, c, 20)

    # ------------------------------------------------------------------
    # ROC (Rate of Change)  (LEAKAGE FIX: same shift(1) as ret_nm above)
    # ------------------------------------------------------------------
    for n in [5, 10, 20, 60]:
        df[f'roc_{n}'] = c_lag1.pct_change(n) * 100

    # ------------------------------------------------------------------
    # DMI / ADX
    # ------------------------------------------------------------------
    pdi, ndi, adx = _dmi(h, lo, c, 14)
    df['adx_14']   = adx
    df['dmi_pdi']  = pdi
    df['dmi_ndi']  = ndi
    df['dmi_diff'] = pdi - ndi

    # ------------------------------------------------------------------
    # MFI & OBV
    # ------------------------------------------------------------------
    df['mfi_14']   = _mfi(h, lo, c, vol, 14)
    obv = _obv(c, vol)
    df['obv_slope']= obv.diff(10) / (obv.abs().rolling(10).mean() + EPS)

    # ------------------------------------------------------------------
    # Supertrend
    # ------------------------------------------------------------------
    df['supertrend'] = _supertrend(h, lo, c, n=10, mult=3.0)

    # ------------------------------------------------------------------
    # Candle patterns
    # ------------------------------------------------------------------
    body   = (c - op).abs()
    hi_lo  = (h - lo).clip(lower=EPS)
    df['body']        = body / (hi_lo) * 100
    df['upper_wick']  = (h - c.combine(op, max)) / (hi_lo) * 100
    df['lower_wick']  = (c.combine(op, min) - lo) / (hi_lo) * 100
    df['is_green']    = (c >= op).astype(int)
    df['big_candle']  = (body / (hi_lo) > 0.7).astype(int)
    df['doji']        = (body / (hi_lo) < 0.1).astype(int)
    df['hammer']      = ((df['lower_wick'] > 60) & (df['upper_wick'] < 20) & (body / (hi_lo) < 0.3)).astype(int)

    # ------------------------------------------------------------------
    # Range features
    # ------------------------------------------------------------------
    df['range_1m']  = hi_lo
    df['range_5m']  = hi_lo.rolling(5).sum()
    df['range_15m'] = hi_lo.rolling(15).sum()

    # ------------------------------------------------------------------
    # Price dynamics
    # ------------------------------------------------------------------
    df['price_accel']   = c.diff(1).diff(1)
    df['body_momentum'] = (c - op).rolling(5).sum() / (hi_lo.rolling(5).sum() + EPS) * 100

    # ------------------------------------------------------------------
    # VWAP
    # ------------------------------------------------------------------
    typ = (h + lo + c) / 3
    df['vwap'] = typ.groupby(df['date']).expanding().mean().reset_index(level=0, drop=True)
    df['vwap_dist']  = (c - df['vwap']) / (df['vwap'] + EPS) * 100
    df['above_vwap'] = (c > df['vwap']).astype(int)
    df['vwap_slope'] = df['vwap_dist'].diff(5)

    # VWAP velocity/acceleration for v3.2
    df['vwap_dev_vel']      = df['vwap_dist'].diff(3)
    df['vwap_dev_accel']    = df['vwap_dev_vel'].diff(3)
    df['vwap_vol_pressure'] = df['vwap_dist'] * vol / (vol.rolling(20).mean() + EPS)

    # ------------------------------------------------------------------
    # Opening Range
    # ------------------------------------------------------------------
    or_mask = df['minute_of_day'] < 15
    df['or_high'] = df[or_mask].groupby('date')['high'].transform('max')
    df['or_low']  = df[or_mask].groupby('date')['low'].transform('min')
    df[['or_high', 'or_low']] = df.groupby('date')[['or_high', 'or_low']].ffill()
    df['or_high'] = df['or_high'].fillna(h)
    df['or_low']  = df['or_low'].fillna(lo)
    or_range = (df['or_high'] - df['or_low']).clip(lower=EPS)
    df['or_range']    = or_range / (c + EPS) * 100
    df['or_pos']      = (c - df['or_low']) / (or_range + EPS)
    df['or_break_up'] = (c > df['or_high']).astype(int)
    df['or_break_dn'] = (c < df['or_low']).astype(int)

    # ------------------------------------------------------------------
    # Rolling high / low distances
    # ------------------------------------------------------------------
    for n in [20, 60, 120]:
        df[f'dist_hi_{n}'] = (h.rolling(n).max() - c) / (c + EPS) * 100
        df[f'dist_lo_{n}'] = (c - lo.rolling(n).min()) / (c + EPS) * 100

    # Pivot distances — use PREVIOUS day's high/low (causal: known at bar open).
    # Previously used full-day transform('max/min') which is lookahead (bar at
    # 9:16 AM would "know" the day's 3:29 PM high). Now uses shifted daily hi/lo.
    _daily_hi = h.groupby(df['date']).max()   # Series indexed by date
    _daily_lo = lo.groupby(df['date']).min()
    _prev_hi  = _daily_hi.shift(1)            # previous day's high
    _prev_lo  = _daily_lo.shift(1)            # previous day's low
    df['dist_pivot_hi'] = (_prev_hi[df['date']].values - c) / (c + EPS) * 100
    df['dist_pivot_lo'] = (c - _prev_lo[df['date']].values) / (c + EPS) * 100

    # ------------------------------------------------------------------
    # Session metadata
    # ------------------------------------------------------------------
    df['session_pct']  = df['minute_of_day'] / float(bars_per_day)
    dt_idx = pd.to_datetime(df['date'])
    df['dow']          = dt_idx.dt.dayofweek.astype(float)
    df['is_expiry']    = (dt_idx.dt.dayofweek == 1).astype(int)  # 1 = Tuesday (NIFTY weekly expiry since Sep 2024)
    df['session_open'] = (df['minute_of_day'] < 30).astype(int)
    df['session_pm']   = (df['minute_of_day'] > 270).astype(int)  # after 13:45

    # ------------------------------------------------------------------
    # Consecutive green / red bars
    # ------------------------------------------------------------------
    green = (c >= op).astype(int)
    red   = (c < op).astype(int)
    # Count consecutive streak using groupby-cumsum trick
    grp_g = (green != green.shift(1)).cumsum()
    grp_r = (red   != red.shift(1)).cumsum()
    df['consec_green'] = green.groupby(grp_g).cumsum() * green
    df['consec_red']   = red.groupby(grp_r).cumsum() * red

    # ------------------------------------------------------------------
    # Volume features
    # ------------------------------------------------------------------
    vol_ma = vol.rolling(20).mean().clip(lower=1)
    df['vol_ratio']      = vol / (vol_ma + EPS)
    df['vol_price_corr'] = c.pct_change(1).rolling(20).corr(vol.pct_change(1))

    # ------------------------------------------------------------------
    # Tick imbalance proxies (v3.2)
    # ------------------------------------------------------------------
    tick_dir = np.sign(c.diff().fillna(0))
    tick_imb = tick_dir.rolling(20).mean()
    df['tick_imbalance']  = tick_imb
    df['tick_imbal_slope']= tick_imb.diff(5)
    buy_vol  = (tick_dir > 0).astype(float) * (vol + 1)
    sell_vol = (tick_dir < 0).astype(float) * (vol + 1)
    df['buy_pressure']   = buy_vol.rolling(10).sum()
    df['sell_pressure']  = sell_vol.rolling(10).sum()
    df['pressure_ratio'] = df['buy_pressure'] / (df['sell_pressure'] + EPS)

    # ------------------------------------------------------------------
    # Multi-bar patterns
    # ------------------------------------------------------------------
    df['bear_3bar'] = ((c.shift(2) > c.shift(1)) & (c.shift(1) > c) &
                       (c.shift(2) > op.shift(2)) & (c.shift(1) > op.shift(1)) &
                       (c < op)).astype(int)
    df['bull_3bar'] = ((c.shift(2) < c.shift(1)) & (c.shift(1) < c) &
                       (c.shift(2) < op.shift(2)) & (c.shift(1) < op.shift(1)) &
                       (c > op)).astype(int)

    # ------------------------------------------------------------------
    # Feature interaction crosses (v3.2)
    # ------------------------------------------------------------------
    df['adx_rsi_trend']  = adx * (rsi14 - 50) / 50
    df['squeeze_atr']    = df['bb_squeeze'] * df['atr_14_pct']
    df['vwap_momentum']  = df['vwap_dist'] * df['ret_5m']
    df['vol_dmi_cross']  = df['vol_ratio'] * (pdi - ndi) / 100
    df['session_iv']     = df['session_pct'] * df['iv_proxy']
    df['adx_macd']       = adx * macd_h
    df['range_rsi_extr'] = ((rsi14 < 30) | (rsi14 > 70)).astype(float) * (50 - rsi14).abs()
    df['imbal_accel']    = df['tick_imbal_slope'].diff(3)

    # ------------------------------------------------------------------
    # Time-decay multiplier (v4.0)
    # ------------------------------------------------------------------
    df['time_decay_mult'] = 1.0 - 0.5 * df['session_pct']

    # ------------------------------------------------------------------
    # Fractionally differentiated series (v3.1 / v4.0)
    # ------------------------------------------------------------------
    try:
        df['close_fd']      = fracdiff_series(c,    d=0.35)
        df['frac_diff_close']= df['close_fd']
        df['vwap_fd']       = fracdiff_series(df['vwap'], d=0.35)
        df['ret5m_fd']      = fracdiff_series(df['ret_5m'].fillna(0), d=0.35)
        df['ret15m_fd']     = fracdiff_series(df['ret_15m'].fillna(0), d=0.35)
    except Exception:
        for col in ['close_fd', 'frac_diff_close', 'vwap_fd', 'ret5m_fd', 'ret15m_fd']:
            df[col] = 0.0

    # ------------------------------------------------------------------
    # FFT cycle (v4.0 — used as regime hint, disabled in models via FEATURE_LIVE_OK)
    # ------------------------------------------------------------------
    df['fft_cycle'] = 0.0  # placeholder; computed per-regime in regimes module

    # ------------------------------------------------------------------
    # Technical Analysis Composite Scores (v4.1 — ML/TA integration)
    # ------------------------------------------------------------------
    # WHY: These scores combine multiple indicators into interpretable signals
    # that ML models can learn to trust/ignore based on regime and context.
    # Instead of TA being display-only, models now see when TA strongly 
    # disagrees with their predictions and can learn from those conflicts.
    #
    # PROBLEM SOLVED: Previously, system would predict DOWN with 95% confidence
    # while TA showed strong BULLISH momentum (RSI 88, supertrend bull, etc.)
    # but models couldn't see this conflict. They would fight the tape and lose.
    #
    # SOLUTION: Add TA composite scores as features:
    #   - ta_momentum_score: RSI + Stochastic + Williams %R combined
    #   - ta_trend_score: EMA crossovers + ADX + Supertrend + DMI
    #   - ta_flow_score: Money Flow Index (institutional buying/selling)
    #   - ta_overall_score: Weighted combination of all TA components
    #
    # Now models can:
    #   1. Learn when TA patterns are reliable vs noise
    #   2. Adjust confidence based on ML-TA agreement
    #   3. Avoid counter-trend trades during strong momentum
    #   4. Detect exhaustion (high TA score + overbought = reversal setup)
    #
    # Example patterns models will learn:
    #   - ML=DOWN + ta_overall_score=+1.5 → Reduce confidence (fighting trend)
    #   - ML=DOWN + ta_overall_score=+1.5 + RSI>80 → Keep confidence (exhaustion)
    #   - ML=UP + ta_overall_score=-1.5 → Reduce confidence (weak setup)
    #
    # Training note: Models trained pre-v4.1 must be RETRAINED with these features.
    # Live trading with old models will fail scaler validation and halt safely.
    # ------------------------------------------------------------------
    
    def _score_feature(val, low_bad, low_ok, high_ok, high_bad):
        """Map indicator value to [-2, +2] score."""
        if val <= low_bad:  return -2
        if val <= low_ok:   return -1
        if val <= high_ok:  return 0
        if val <= high_bad: return 1
        return 2
    
    # Momentum score: RSI + Stoch + Williams
    rsi_score   = df['rsi_14'].apply(lambda x: _score_feature(x, 20, 35, 65, 80))
    stoch_score = df['stoch_k'].apply(lambda x: _score_feature(x, 15, 25, 75, 85))
    willr_score = df['willr'].apply(lambda x: _score_feature(x, -95, -80, -20, -5))
    df['ta_momentum_score'] = (rsi_score + stoch_score + willr_score) / 3
    
    # Trend score: EMAs + ADX + Supertrend + DMI
    # ema9_21, ema21_50 already exist and are directional (-1/+1)
    # supertrend is already -1/+1
    # dmi_diff is directional
    df['ta_trend_score'] = (
        df['ema9_21'] * 0.25 +
        df['ema21_50'] * 0.25 +
        df['supertrend'] * 0.25 +
        np.sign(df['dmi_diff']) * 0.25
    )
    
    # Flow score: MFI (money flow index)
    df['ta_flow_score'] = df['mfi_14'].apply(lambda x: _score_feature(x, 20, 35, 65, 80)) / 2
    
    # Overall TA score: weighted combination
    # Combines momentum, trend, VWAP bias, OR breakout, MACD
    vwap_bias  = df['above_vwap'].apply(lambda x: 1 if x else -1)
    or_bias    = df['or_break_up'].apply(lambda x: 1 if x else 0) - df['or_break_dn'].apply(lambda x: 1 if x else 0)
    macd_bias  = np.sign(macd_h)
    
    df['ta_overall_score'] = (
        df['ta_momentum_score'] * 0.25 +
        df['ta_trend_score'] * 0.35 +
        vwap_bias * 0.20 +
        or_bias * 0.10 +
        macd_bias * 0.10
    ).clip(-2, 2)  # Keep in reasonable range
    
    # ------------------------------------------------------------------

    return df

# Alias for backward compatibility
add_1min_features = add_1min_features_production

def add_htf_features(df1m: pd.DataFrame,
                     df_htf: pd.DataFrame | None,
                     prefix: str,
                     ret_periods: list) -> pd.DataFrame:
    """As-of merge of higher-timeframe features into 1-min frame."""
    if df_htf is None or len(df_htf) < 15:
        # Need at least 15 bars for basic RSI calculation
        cols = [f'{prefix}ret_{n}' for n in ret_periods] + [
            f'{prefix}rsi', f'{prefix}macd_h', f'{prefix}bb_pos',
            f'{prefix}atr_pct', f'{prefix}ema9_21', f'{prefix}vol_10',
            f'{prefix}above_vwap', f'{prefix}adx', f'{prefix}cci',
            f'{prefix}stoch_k', f'{prefix}willr']
        for col in cols:
            df1m[col] = 0.0
        return df1m

    d = df_htf.copy()
    c, h, lo = d['close'], d['high'], d['low']

    # LEAKAGE FIX: HTF ret_n must use the PREVIOUS bar's close as reference.
    # Without shift, the 1-min bars at 10:01–10:14 receive the 10:15 bar's
    # pct_change which embeds the 10:15 close — future information they cannot
    # know. Shifting c by 1 makes ret_n describe the move from T-n-1 to T-1,
    # so every 1-min bar only sees completed HTF candles.
    c_lag = c.shift(1)
    for n in ret_periods:
        d[f'{prefix}ret_{n}'] = c_lag.pct_change(n) * 100
    d[f'{prefix}rsi']     = _rsi(c, 14)
    d[f'{prefix}macd_h']  = _macd(c)
    ma20 = c.rolling(20).mean(); s20 = c.rolling(20).std()
    d[f'{prefix}bb_pos']  = (c-(ma20-2*s20)) / (4*s20+1e-9)
    atr14, _              = _atr(h, lo, c, 14)
    d[f'{prefix}atr_pct'] = atr14 / (c + 1e-9) * 100
    e9  = c.ewm(span=9,  adjust=False).mean()
    e21 = c.ewm(span=21, adjust=False).mean()
    d[f'{prefix}ema9_21'] = ((e9>e21).astype(int) - (e9<e21).astype(int))
    d[f'{prefix}vol_10']  = c.pct_change(1).rolling(10).std() * 100
    typ = (h + lo + c)/3
    d[f'{prefix}above_vwap'] = (c > typ.expanding().mean()).astype(int)
    # Extra HTF features
    _, _, adx_h = _dmi(h, lo, c, 14)
    d[f'{prefix}adx']     = adx_h
    d[f'{prefix}cci']     = _cci(h, lo, c, 20)
    low14h  = lo.rolling(14).min(); high14h = h.rolling(14).max()
    k_h = (c - low14h) / (high14h - low14h + 1e-9) * 100
    d[f'{prefix}stoch_k'] = k_h
    d[f'{prefix}willr']   = -100 * (high14h - c) / (high14h - low14h + 1e-9)

    # Forward-fill indicators that need longer warmup (ADX, CCI, MACD) to maximize data retention
    # This is critical for 15-min data where we only get ~26 bars from the API
    d[f'{prefix}adx'] = d[f'{prefix}adx'].ffill()
    d[f'{prefix}cci'] = d[f'{prefix}cci'].ffill()
    d[f'{prefix}macd_h'] = d[f'{prefix}macd_h'].ffill()
    
    feat_cols = [col for col in d.columns if col.startswith(prefix)]
    # For 15-min data with limited bars, only require RSI (needs just 14 bars)
    # For 5-min data with more bars, require both RSI and MACD
    if len(df_htf) < 50:
        # Limited data - only require RSI
        core_cols = ['datetime', f'{prefix}rsi']
    else:
        # Sufficient data - require RSI and MACD
        core_cols = ['datetime', f'{prefix}rsi', f'{prefix}macd_h']
    
    keep = d[['datetime'] + feat_cols].dropna(subset=core_cols).sort_values('datetime')

    # Safety check: if HTF data is empty or has no features, return with zeros
    if keep.empty or len(feat_cols) == 0:
        print(f"  [WARN] [HTF {prefix}] Insufficient valid data: "
              f"input={len(df_htf)} bars, after_indicators={len(keep)} valid bars")
        if len(df_htf) > 0:
            print(f"      Time range: {df_htf['datetime'].min()} to {df_htf['datetime'].max()}")
            print(f"      Hint: Need more bars for indicator warmup (RSI=14, MACD=26, ADX=28)")
        for col in [f'{prefix}ret_{n}' for n in ret_periods] + \
                   [f'{prefix}rsi', f'{prefix}macd_h', f'{prefix}bb_pos', 
                    f'{prefix}atr_pct', f'{prefix}ema9_21', f'{prefix}vol_10',
                    f'{prefix}above_vwap', f'{prefix}adx', f'{prefix}cci',
                    f'{prefix}stoch_k', f'{prefix}willr']:
            if col not in df1m.columns:
                df1m[col] = 0.0
        return df1m

    df1m = df1m.sort_values('datetime')
    merged = pd.merge_asof(df1m, keep, on='datetime', direction='backward')
    
    # Fill NaN values for all expected HTF feature columns
    for col in feat_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)
        else:
            # If column doesn't exist after merge, create it with zeros
            merged[col] = 0.0
    
    # DEBUG: Check if merge produced valid values for recent rows
    if len(merged) > 0 and len(keep) > 0:
        sample_col = f'{prefix}rsi'
        if sample_col in merged.columns:
            non_zero_count = (merged[sample_col] != 0.0).sum()
            non_zero_pct = (non_zero_count / len(merged)) * 100
            latest_val = merged[sample_col].iloc[-1]
            
            if non_zero_pct > 50 and latest_val != 0.0:
                # Success case - show brief confirmation
                print(f"  [OK] [HTF {prefix}] {non_zero_count}/{len(merged)} rows with data, "
                      f"latest {sample_col}={latest_val:.1f}")
            elif latest_val == 0.0:
                # Problem case - show detailed debug
                print(f"  [WARN] [HTF {prefix}] Latest row has 0.0! "
                      f"non_zero={non_zero_pct:.1f}% ({non_zero_count}/{len(merged)}), "
                      f"df1m_time={merged['datetime'].iloc[-1]}, "
                      f"htf_latest={keep['datetime'].max()}")
    
    return merged

def add_daily_features(df1m: pd.DataFrame,
                       df1d: pd.DataFrame | None) -> pd.DataFrame:
    """Prior-day close, gap, daily trend, daily IV proxy."""
    dcols = ['day_ret_1','day_ret_5','day_ret_20','day_rsi','day_atr_pct',
             'day_bb_pos','day_above_ma50','day_above_ma200','day_vol_20',
             'prev_close','prev_high','prev_low',
             'gap_pct','above_prev_close','dist_prev_hi','dist_prev_lo',
             'day_iv_proxy','day_iv_rank',
             'day_adx','day_trend_strength','weekly_ret','monthly_ret',
             'day_cci','day_stoch_k']

    # --- FIX 1: Prevent MergeError by dropping existing daily columns ---
    # This prevents the duplicate _x, _y columns during the loop
    existing_cols = [c for c in dcols if c in df1m.columns]
    if existing_cols:
        df1m = df1m.drop(columns=existing_cols)

    if df1d is None:
        for col in dcols: df1m[col] = 0.0
        return df1m

    d = df1d.copy()
    c, h, lo = d['close'], d['high'], d['low']

    # Technical Calculations
    for n in [1,5,20]:    d[f'day_ret_{n}'] = c.pct_change(n)*100
    d['weekly_ret']     = c.pct_change(5)*100
    d['monthly_ret']    = c.pct_change(21)*100
    d['day_rsi']        = _rsi(c, 14)
    atr14, _            = _atr(h, lo, c, 14)
    d['day_atr_pct']    = atr14 / (c+1e-9) * 100
    ma20 = c.rolling(20).mean(); s20=c.rolling(20).std()
    d['day_bb_pos']     = (c-(ma20-2*s20))/(4*s20+1e-9)
    d['day_above_ma50'] = (c > c.rolling(50).mean()).astype(int)
    d['day_above_ma200']= (c > c.rolling(200).mean()).astype(int)
    d['day_vol_20']     = c.pct_change(1).rolling(20).std()*100
    d['prev_close']     = c.shift(1)
    d['prev_high']      = h.shift(1)
    d['prev_low']       = lo.shift(1)
    d['day_iv_proxy']   = d['day_atr_pct'] * np.sqrt(252)
    d['day_iv_rank']    = d['day_iv_proxy'].rolling(252, min_periods=50).rank(pct=True)*100
    _, _, adx_d         = _dmi(h, lo, c, 14)
    d['day_adx']        = adx_d
    
    ma200d = c.rolling(200).mean()
    d['day_trend_strength'] = (c - ma200d) / (d['day_vol_20'] * ma200d / 100 + 1e-9)
    d['day_cci']        = _cci(h, lo, c, 20)
    low14d = lo.rolling(14).min(); high14d = h.rolling(14).max()
    d['day_stoch_k']    = (c - low14d) / (high14d - low14d + 1e-9) * 100

    d['date'] = d['datetime'].dt.date
    dcols_avail = ['date','day_ret_1','day_ret_5','day_ret_20','day_rsi',
                   'day_atr_pct','day_bb_pos','day_above_ma50','day_above_ma200',
                   'day_vol_20','prev_close','prev_high','prev_low',
                   'day_iv_proxy','day_iv_rank','day_adx','day_trend_strength',
                   'weekly_ret','monthly_ret','day_cci','day_stoch_k']
    
    sel = d[dcols_avail].dropna(subset=['prev_close'])

    # --- FIX 2: Safe Merge ---
    df1m = df1m.merge(sel, on='date', how='left')
    
    # --- FIX 3: Monday/Holiday Morning Bridge ---
    # If today's date doesn't exist in df1d yet, forward fill from Friday's data
    df1m[dcols_avail[1:]] = df1m[dcols_avail[1:]].ffill()

    # Safety: ensure prev_* columns exist
    if 'prev_close' not in df1m.columns:
        df1m['prev_close'] = df1m.groupby('date')['close'].transform('first').shift(1).ffill()
    if 'prev_high' not in df1m.columns:
        df1m['prev_high'] = df1m.groupby('date')['high'].transform('first').shift(1).ffill()
    if 'prev_low' not in df1m.columns:
        df1m['prev_low'] = df1m.groupby('date')['low'].transform('first').shift(1).ffill()

    # Derived calculations
    df1m['gap_pct']          = (df1m['open'] - df1m['prev_close']) / (df1m['prev_close']+1e-9)*100
    df1m['above_prev_close'] = (df1m['close'] > df1m['prev_close']).astype(int).fillna(0)
    df1m['dist_prev_hi']     = (df1m['close'] - df1m['prev_high']) / (df1m['close']+1e-9)*100
    df1m['dist_prev_lo']     = (df1m['close'] - df1m['prev_low'])  / (df1m['close']+1e-9)*100

    # Final Cleanup
    for col in dcols:
        if col in df1m.columns:
            df1m[col] = df1m[col].fillna(0.0)
        else:
            df1m[col] = 0.0
            
    return df1m

# All feature column names
# ==============================================================================
# 1️⃣ FEATURE TIERING & 2️⃣ TRAIN-LIVE MISMATCH GUARD (2026 LIVE SURVIVABILITY)
# ==============================================================================

# WHY: Feature bloat ruins live performance through overfitting, feature drift,
#      and unavailable data sources. Tiering ensures only battle-tested core
#      features are always used, regime-specific features activate conditionally,
#      and experimental features stay OFF until proven.

# FEATURE_LIVE_OK: Marks which features are reliably available in live execution.
# Features marked False are DROPPED during training to prevent train-live mismatch.
FEATURE_LIVE_OK = {
    # Unavailable in live: require external data sources or OI feeds
    'option_chain_ndi': False,        # OI-based delta imbalance (no live feed)
    'hdfc_ret_1m': False,             # Single-stock data (not in base feed)
    'reliance_ret_1m': False,         # Single-stock data (not in base feed)
    'banknifty_spread': False,        # Requires simultaneous BankNifty feed
    'banknifty_divergence': False,    # Requires simultaneous BankNifty feed
    # Experimental features (OFF by default until proven in paper trading)
    'fft_cycle': False,               # FFT now handled separately as regime hint
    'cycle_vol_interaction': False,   # Dependent on fft_cycle
}

# Tier-1: Core features (max 30, always enabled)
# These are the battle-tested backbone: price action, volatility, trend, momentum.
TIER_1_CORE = [
    # Returns (short-term price action)
    'ret_5m', 'ret_15m',
    # Volatility (core risk metric)
    'atr_14_pct', 'atr_ratio',
    # Momentum & Trend
    'rsi_14', 'rsi_slope',
    'adx_14',
    # VWAP (institutional flow)
    'vwap_dist', 'vwap_dev_vel',
    # Bollinger (volatility regime)
    'bb_width', 'bb_squeeze',
    # EMA structure (trend confirmation)
    'ema9_21', 'ema21_50',
    # Opening Range Breakout (high-conviction setup)
    'or_break_up', 'or_break_dn',
    # IV proxy (option-specific)
    'iv_rank_approx',
    # Session context
    'session_pct', 'is_expiry',
]

# Tier-2: Regime-conditional features
# TRENDING regime: use directional indicators (MACD, ADX-MACD, EMA slope)
# RANGING regime: use mean-reversion indicators (RSI extremes, Stochastic, Williams %R)
TIER_2_TRENDING = [
    'macd_h',          # MACD histogram (momentum)
    'adx_macd',        # ADX × MACD interaction (strong trend + momentum)
    'ema9_slope',      # EMA9 slope (short-term trend velocity)
]

TIER_2_RANGING = [
    'range_rsi_extr',  # RSI extremes in range (oversold/overbought)
    'stoch_k',         # Stochastic (mean-reversion timing)
    'willr',           # Williams %R (mean-reversion extremes)
]

# Tier-3: Experimental (OFF by default, manually enable after validation)
TIER_3_EXPERIMENTAL = [
    # Example: 'fft_cycle', 'option_chain_ndi', etc.
    # These are explicitly excluded via FEATURE_LIVE_OK
]

def get_active_feature_cols(regime: int = 1) -> list:
    """
    Returns the active feature list based on regime.
    
    Args:
        regime: 0=TRENDING, 1=RANGING, 2=CRISIS
        
    Returns:
        List of feature column names to use for training/prediction.
        
    WHY: Dynamic feature selection prevents overfitting and ensures
         model uses only relevant indicators for current market regime.
    """
    # Start with Tier-1 core (always enabled)
    active = list(TIER_1_CORE)
    
    # Add regime-conditional Tier-2 features
    if regime == REGIME_TRENDING:
        active.extend(TIER_2_TRENDING)
    elif regime == REGIME_RANGING:
        active.extend(TIER_2_RANGING)
    # CRISIS regime: use only Tier-1 core (no regime-specific features)
    
    # Filter out unavailable features (train-live mismatch guard)
    active = [f for f in active if FEATURE_LIVE_OK.get(f, True)]
    
    return active

def get_feature_cols():
    """
    29-feature balanced set (v4.1 — ML/TA integration).

    GROUP A — Momentum core (9): proven top importance across all horizons.
    GROUP B — Orthogonal signals (9): exhaustion, mean-reversion, structure,
              context. Covers what pure momentum misses (overbought exits,
              squeeze breakouts, IV spikes, session noise).
    GROUP C — Market structure (7): HTF confirmation at 5-min and 15-min.
    GROUP D — Technical Analysis Scores (4): composite TA signals that allow
              models to learn when to trust/ignore technical patterns.

    Disabled (7): require data feeds not available live.
    """
    base = [
        # ----- GROUP A: Momentum core (fracdiff + HTF + raw) -----
        'ret5m_fd',       # fracdiff 5-min return — stationarity-preserving momentum
        'ret15m_fd',      # fracdiff 15-min return
        'tf5_ret_1',      # 5-min HTF last-bar return (HTF confirmation)
        'tf5_ret_3',      # 5-min HTF 3-bar return
        'tf15_ret_1',     # 15-min HTF last-bar return
        'ret_5m',         # raw 5-min return
        'ret_15m',        # raw 15-min return
        'roc_5',          # 5-bar rate of change
        'vwap_dev_accel', # VWAP deviation acceleration (flow turning point)

        # ----- GROUP B: Exhaustion / mean-reversion / structure -----
        'vwap_dev_vel',   # VWAP deviation velocity (institutional flow speed)
        'rsi_14',         # RSI — overbought/oversold exhaustion
        'adx_14',         # ADX — trend strength filter (avoid ranging noise)
        'atr_14_pct',     # ATR% — volatility context for barrier sizing
        'iv_rank_approx', # IV rank — options pricing context
        'pressure_ratio', # buy/sell volume imbalance
        'tick_imbalance', # tick direction imbalance (microstructure)
        'bb_squeeze',     # Bollinger squeeze — breakout setup flag

        # ----- GROUP C: Market structure / context -----
        'gap_pct',        # overnight gap (daily open vs prev close)
        'or_break_up',    # opening range breakout up flag
        'or_break_dn',    # opening range breakout down flag
        'dist_pivot_lo',  # distance from prev-day low (causal pivot)
        'session_pct',    # session progress (0-1 arc)
        'is_expiry',      # Tuesday expiry flag (NIFTY weekly since Sep 2024)

        # Regime metadata (passed through but not a price feature)
        'adx_rsi_trend',  # ADX × (RSI-50)/50 interaction — trend quality

        # ----- GROUP D: Technical Analysis Composite Scores (v4.1) -----
        'ta_momentum_score',  # Combined RSI + Stoch + Williams score (-2 to +2)
        'ta_trend_score',     # Combined EMA + ADX + Supertrend + DMI score (-1 to +1)
        'ta_flow_score',      # MFI-based money flow score (-1 to +1)
        'ta_overall_score',   # Overall TA bias combining all components (-2 to +2)

        # ----- DISABLED (unavailable live — excluded via FEATURE_LIVE_OK) -----
        'option_chain_ndi',
        'hdfc_ret_1m', 'reliance_ret_1m',
        'banknifty_spread', 'banknifty_divergence',
        'fft_cycle', 'cycle_vol_interaction',
    ]
    return base

FEATURE_COLS = get_feature_cols()


