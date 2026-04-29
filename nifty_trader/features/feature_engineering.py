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

    # Zero out returns whose lookback window spans overnight (different date than n bars ago).
    # Without this, ret_5m at 9:20 AM uses yesterday's close as reference — injecting the
    # overnight gap direction as an intraday signal, biasing models toward yesterday's trend.
    if 'date' in df.columns:
        for n in [1, 3, 5, 10, 15, 30, 60]:
            date_n_ago = df['date'].shift(n)
            overnight_mask = df['date'] != date_n_ago
            df.loc[overnight_mask, f'ret_{n}m'] = 0.0

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

    iv_rank_window = min(20 * bars_per_day, len(df))
    iv_rank_min_periods = min(bars_per_day, iv_rank_window)
    iv_rank = df['iv_final'].rolling(iv_rank_window, min_periods=iv_rank_min_periods).rank(pct=True) * 100
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
    df['adx_14']    = adx
    df['dmi_pdi']   = pdi
    df['dmi_ndi']   = ndi
    df['dmi_diff']  = pdi - ndi
    # 3-bar ADX slope: positive = momentum building, negative = exhausting.
    # Used by Gate7c-ADX v4.0 to distinguish rising-weak from falling-strong ADX.
    df['adx_slope'] = adx.diff(3)

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
    # VWAP  (volume-weighted average price, reset each session)
    # ------------------------------------------------------------------
    typ = (h + lo + c) / 3
    _cum_tv  = (typ * vol).groupby(df['date'], group_keys=False).apply(lambda g: g.expanding().sum())
    _cum_vol = vol.groupby(df['date'], group_keys=False).apply(lambda g: g.expanding().sum())
    df['vwap'] = _cum_tv / (_cum_vol + EPS)
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
    df['is_expiry']    = (dt_idx.dt.dayofweek == 1).astype(int)  # 1 = Tuesday (NIFTY weekly expiry since Sep 2024 — was Thursday before)
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
    # Price Action Structures (v4.2)
    # Computed on every bar. Used by signal_generator Gate 7e (pattern
    # alignment) and day_predictor opening-structure scorer.
    # ------------------------------------------------------------------

    # ── 1. Swing HH/HL / LH/LL structure ──────────────────────────────
    # A swing high is a bar whose high is higher than the 3 bars on each
    # side. A swing low is the mirror. We detect HH/HL (bullish structure)
    # and LH/LL (bearish structure) over the last 20-bar window.
    SWING_N = 3   # bars each side for pivot confirmation
    STRUCT_W = 20  # window to count swings
    pivot_hi = ((h == h.rolling(2 * SWING_N + 1, center=True).max())
                .astype(float).shift(SWING_N))   # shift by SWING_N to avoid lookahead
    pivot_lo = ((lo == lo.rolling(2 * SWING_N + 1, center=True).min())
                .astype(float).shift(SWING_N))
    df['pivot_hi'] = pivot_hi.fillna(0.0)
    df['pivot_lo'] = pivot_lo.fillna(0.0)

    # Compare the most recent two swing highs and two swing lows
    # to determine HH/HL vs LH/LL structure.
    def _structure_score(ph_series, pl_series, close_series, win):
        """Return +1 (HH/HL bullish), -1 (LH/LL bearish), 0 (mixed)."""
        scores = []
        for i in range(len(ph_series)):
            if i < win:
                scores.append(0.0)
                continue
            ph_w = ph_series.iloc[i - win:i]
            pl_w = pl_series.iloc[i - win:i]
            ph_idx = ph_w[ph_w > 0].index
            pl_idx = pl_w[pl_w > 0].index
            if len(ph_idx) >= 2 and len(pl_idx) >= 2:
                hi_vals = close_series.loc[ph_idx[-2:]]
                lo_vals = close_series.loc[pl_idx[-2:]]
                hh = float(hi_vals.iloc[-1]) > float(hi_vals.iloc[-2])
                hl = float(lo_vals.iloc[-1]) > float(lo_vals.iloc[-2])
                lh = float(hi_vals.iloc[-1]) < float(hi_vals.iloc[-2])
                ll = float(lo_vals.iloc[-1]) < float(lo_vals.iloc[-2])
                if hh and hl:
                    scores.append(1.0)
                elif lh and ll:
                    scores.append(-1.0)
                else:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return pd.Series(scores, index=ph_series.index)

    df['struct_score'] = _structure_score(df['pivot_hi'], df['pivot_lo'], c, STRUCT_W)
    # Smooth over 3 bars so a single noise bar doesn't flip the structure
    df['struct_score'] = df['struct_score'].rolling(3, min_periods=1).mean()

    # ── 2. Fair Value Gap (FVG) ────────────────────────────────────────
    # Bullish FVG: bar[i-2].high < bar[i].low  (gap up between 2 bars ago and now)
    # Bearish FVG: bar[i-2].low  > bar[i].high (gap down)
    # Flag stays 1 for 5 bars (price tends to retest within ~5 bars or forget it).
    fvg_bull_raw = (h.shift(2) < lo).astype(float)   # gap up
    fvg_bear_raw = (lo.shift(2) > h).astype(float)   # gap down
    df['fvg_bull'] = fvg_bull_raw.rolling(5, min_periods=1).max()
    df['fvg_bear'] = fvg_bear_raw.rolling(5, min_periods=1).max()

    # ── 3. Liquidity Sweep ─────────────────────────────────────────────
    # Equal highs: previous swing high within 0.05% of current high.
    # A sweep UP = price pokes above equal high then closes back inside (bear trap).
    # A sweep DN = price pokes below equal low then closes back inside (bull trap).
    EQ_TOL = 0.0005   # 0.05% tolerance for "equal" pivots
    eq_hi = h.rolling(STRUCT_W).max()
    eq_lo = lo.rolling(STRUCT_W).min()
    # Sweep UP: high touches equal-high zone but close retreats below it
    sweep_up = ((h >= eq_hi * (1 - EQ_TOL)) & (c < eq_hi)).astype(float)
    # Sweep DN: low touches equal-low zone but close recovers above it
    sweep_dn = ((lo <= eq_lo * (1 + EQ_TOL)) & (c > eq_lo)).astype(float)
    df['liq_sweep_up'] = sweep_up   # bearish — long liquidity taken, likely reversal
    df['liq_sweep_dn'] = sweep_dn   # bullish — short liquidity taken, likely reversal

    # ── 4. NR4 / NR7 (Narrow Range compression) ───────────────────────
    # NR4: today's bar range is the smallest of the last 4 bars (compression)
    # NR7: smallest of last 7 bars (tighter squeeze, stronger breakout signal)
    bar_range = (h - lo).clip(lower=EPS)
    df['nr4'] = (bar_range == bar_range.rolling(4, min_periods=4).min()).astype(int)
    df['nr7'] = (bar_range == bar_range.rolling(7, min_periods=7).min()).astype(int)

    # ── 5. ORB strength & direction ───────────────────────────────────
    # or_break_up / or_break_dn already computed above.
    # Add distance above/below the OR boundary as a continuous feature.
    # Positive = above OR high (CE friendly), negative = below OR low (PE friendly).
    df['orb_dist'] = np.where(
        df['or_break_up'] == 1,
         (c - df['or_high']) / (df['or_range'].clip(lower=EPS) / 100 * c + EPS),
        np.where(
            df['or_break_dn'] == 1,
            (c - df['or_low'])  / (df['or_range'].clip(lower=EPS) / 100 * c + EPS),
            0.0
        )
    )
    df['orb_dist'] = df['orb_dist'].clip(-5, 5)

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
            f'{prefix}stoch_k', f'{prefix}willr', f'{prefix}close_chg']
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
    # Raw close-to-close change in points (signed). Used by Gate5e-B momentum filter.
    # Uses c_lag (shifted close) so the value seen by 1m bars is the completed HTF bar's move.
    d[f'{prefix}close_chg'] = c_lag.diff(1)

    # Forward-fill AND back-fill indicators that need longer warmup (ADX, CCI, MACD).
    # ffill: propagates last valid value forward (standard).
    # bfill: fills leading NaNs at the start of the series with the first valid value.
    # Without bfill, ADX is NaN for the first 28 bars → merges as 0.0 into 1m frame
    # → adx_5m=0 in bar_log even when market is clearly trending.
    d[f'{prefix}adx']    = d[f'{prefix}adx'].ffill().bfill()
    d[f'{prefix}cci']    = d[f'{prefix}cci'].ffill().bfill()
    d[f'{prefix}macd_h'] = d[f'{prefix}macd_h'].ffill().bfill()
    
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
                    f'{prefix}stoch_k', f'{prefix}willr', f'{prefix}close_chg']:
            if col not in df1m.columns:
                df1m[col] = 0.0
        return df1m

    # merge_asof requires sorted, no-duplicate, tz-naive keys.
    # Use a temporary _dt_key column so we never modify the original datetime column.
    # Defensive: if datetime became the index (e.g. after a merge), promote it back.
    if 'datetime' not in df1m.columns and df1m.index.name == 'datetime':
        df1m = df1m.reset_index()
    if 'datetime' not in df1m.columns:
        logger.warning(f"[HTF {prefix}] df1m missing 'datetime' column — columns: {list(df1m.columns[:10])}. Returning with zeros.")
        for col in [f'{prefix}ret_{n}' for n in ret_periods] + \
                   [f'{prefix}rsi', f'{prefix}macd_h', f'{prefix}bb_pos',
                    f'{prefix}atr_pct', f'{prefix}ema9_21', f'{prefix}vol_10',
                    f'{prefix}above_vwap', f'{prefix}adx', f'{prefix}cci',
                    f'{prefix}stoch_k', f'{prefix}willr', f'{prefix}close_chg']:
            if col not in df1m.columns:
                df1m[col] = 0.0
        return df1m
    df1m = df1m.sort_values('datetime').reset_index(drop=True).copy()
    # Drop any existing HTF columns with this prefix from a prior merge (cold-start or previous bar)
    # to avoid merge_asof creating tf5_rsi_x/tf5_rsi_y duplicates that all become 0.
    existing_htf = [c for c in df1m.columns if c.startswith(prefix)]
    if existing_htf:
        df1m = df1m.drop(columns=existing_htf)
    df1m['_dt_key'] = pd.to_datetime(df1m['datetime']).dt.floor('min')
    if hasattr(df1m['_dt_key'].dtype, 'tz') and df1m['_dt_key'].dtype.tz is not None:
        df1m['_dt_key'] = df1m['_dt_key'].dt.tz_localize(None)

    keep = keep.copy()
    keep['_dt_key'] = pd.to_datetime(keep['datetime'])
    if hasattr(keep['_dt_key'].dtype, 'tz') and keep['_dt_key'].dtype.tz is not None:
        keep['_dt_key'] = keep['_dt_key'].dt.tz_localize(None)
    keep = keep.sort_values('_dt_key').drop_duplicates(subset=['_dt_key'], keep='last').reset_index(drop=True)
    # Drop 'datetime' from keep before merge_asof to avoid datetime_x/datetime_y collision
    # which would silently rename df1m's 'datetime' and break subsequent HTF merges.
    keep = keep.drop(columns=['datetime'], errors='ignore')

    # Diagnostic: log key ranges if they look misaligned
    if len(df1m) > 0 and len(keep) > 0:
        df1m_min, df1m_max = df1m['_dt_key'].min(), df1m['_dt_key'].max()
        keep_min, keep_max = keep['_dt_key'].min(), keep['_dt_key'].max()
        if df1m_max < keep_min or df1m_min > keep_max:
            logger.warning(f"[HTF {prefix}] No overlap: df1m=[{df1m_min} .. {df1m_max}] keep=[{keep_min} .. {keep_max}]")
        logger.debug(f"[HTF {prefix}] merge_asof: df1m=[{df1m_min} .. {df1m_max}] keep=[{keep_min} .. {keep_max}] df1m_tz={df1m['_dt_key'].dtype} keep_tz={keep['_dt_key'].dtype}")
    merged = pd.merge_asof(df1m, keep, on='_dt_key', direction='backward')
    merged = merged.drop(columns=['_dt_key'])

    # Fill NaN values for all expected HTF feature columns.
    # ffill() first: carries last valid HTF value forward across all 1m bars
    # (merge_asof gaps happen when dropna removed warmup rows from keep).
    # fillna(0.0) as final fallback only for truly missing data at session start.
    for col in feat_cols:
        if col in merged.columns:
            merged[col] = merged[col].ffill().fillna(0.0)
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
                      f"htf_latest={keep['_dt_key'].max() if '_dt_key' in keep.columns else 'n/a'}")
                print(f"  [WARN] [HTF {prefix}] keep rows={len(keep)}, "
                      f"df1m_dt_range={merged['datetime'].min()} to {merged['datetime'].iloc[-1]}, "
                      f"keep_sample_rsi={keep[sample_col].iloc[-3:].tolist() if len(keep)>=3 else keep[sample_col].tolist()}")
    
    return merged

def compute_options_chain_features(options_dir: str) -> pd.DataFrame:
    """
    Compute daily options chain features from NSE bhavcopy CSVs.
    Returns a DataFrame indexed by date with columns:
      pcr_oi       — Put/Call ratio by open interest (>1.2 = bullish floor)
      max_pain     — Strike where total OI loss for writers is minimum
      atm_ce_oi    — CE open interest at nearest ATM strike
      atm_pe_oi    — PE open interest at nearest ATM strike
      iv_skew      — ATM PE close minus ATM CE close (positive = fear of downside)
      oi_buildup   — Net OI change: PE chg_oi - CE chg_oi (positive = bullish)
    """
    import glob
    records = []
    files = sorted(glob.glob(os.path.join(options_dir, 'nifty_*.csv')))
    for fpath in files:
        try:
            df = pd.read_csv(fpath)
            if df.empty or 'option_type' not in df.columns:
                continue
            df = df[df['open_int'] > 0].copy()
            if df.empty:
                continue

            # Parse date
            date_str = os.path.basename(fpath).replace('nifty_', '').replace('.csv', '')
            try:
                trade_date = pd.to_datetime(date_str).date()
            except Exception:
                continue

            # Spot estimate: weighted average of high-OI ATM strikes
            df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
            df = df.dropna(subset=['strike'])
            ce = df[df['option_type'] == 'CE']
            pe = df[df['option_type'] == 'PE']

            # PCR by OI
            total_ce_oi = ce['open_int'].sum()
            total_pe_oi = pe['open_int'].sum()
            pcr_oi = total_pe_oi / (total_ce_oi + EPS)

            # Spot estimate from max OI CE strike (highest gamma = closest to ATM)
            if len(ce) == 0:
                continue
            spot_est = ce.loc[ce['open_int'].idxmax(), 'strike']

            # ATM = strike nearest to spot
            all_strikes = df['strike'].unique()
            atm_strike = all_strikes[np.argmin(np.abs(all_strikes - spot_est))]

            atm_ce = ce[ce['strike'] == atm_strike]
            atm_pe = pe[pe['strike'] == atm_strike]
            atm_ce_oi_val = atm_ce['open_int'].sum()
            atm_pe_oi_val = atm_pe['open_int'].sum()
            atm_ce_close  = atm_ce['close'].mean() if len(atm_ce) > 0 else 0.0
            atm_pe_close  = atm_pe['close'].mean() if len(atm_pe) > 0 else 0.0
            iv_skew       = atm_pe_close - atm_ce_close  # positive = put fear

            # OI buildup: net PE - CE chg_oi (positive = smart money buying puts = bearish view)
            pe_chg = pe['chg_oi'].sum() if 'chg_oi' in pe.columns else 0.0
            ce_chg = ce['chg_oi'].sum() if 'chg_oi' in ce.columns else 0.0
            oi_buildup = pe_chg - ce_chg

            # Max pain: strike where total loss for all option writers is minimum
            # Only check strikes within 1500 pts of spot (avoid illiquid far OTM)
            near_strikes = all_strikes[np.abs(all_strikes - spot_est) <= 1500]
            pain = {}
            for s in near_strikes:
                ce_loss = ((s - ce[ce['strike'] < s]['strike']) *
                           ce[ce['strike'] < s]['open_int']).sum()
                pe_loss = ((pe[pe['strike'] > s]['strike'] - s) *
                           pe[pe['strike'] > s]['open_int']).sum()
                pain[s] = ce_loss + pe_loss
            max_pain_val = min(pain, key=pain.get) if pain else spot_est
            # Express max pain as % distance from spot
            max_pain_dist = (max_pain_val - spot_est) / (spot_est + EPS) * 100

            records.append({
                'date':          trade_date,
                'pcr_oi':        round(float(pcr_oi), 4),
                'max_pain_dist': round(float(max_pain_dist), 4),
                'atm_ce_oi':     round(float(atm_ce_oi_val), 0),
                'atm_pe_oi':     round(float(atm_pe_oi_val), 0),
                'iv_skew':       round(float(iv_skew), 2),
                'oi_buildup':    round(float(oi_buildup), 0),
            })
        except Exception as e:
            logger.debug(f"[OptionsFeatures] Skipped {fpath}: {e}")
            continue

    if not records:
        logger.warning("[OptionsFeatures] No options data loaded — all features will be 0")
        return pd.DataFrame(columns=['date','pcr_oi','max_pain_dist','atm_ce_oi',
                                     'atm_pe_oi','iv_skew','oi_buildup'])
    result = pd.DataFrame(records).sort_values('date').reset_index(drop=True)
    logger.info(f"[OptionsFeatures] Loaded {len(result)} days of options chain features "
                f"({result['date'].min()} to {result['date'].max()})")
    return result


def add_options_chain_features(df1m: pd.DataFrame,
                                options_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Merge daily options chain features into 1-min dataframe.
    Uses PREVIOUS day's options data (avoid lookahead — today's OI isn't
    known until end of day).
    """
    opt_cols = ['pcr_oi', 'max_pain_dist', 'atm_ce_oi', 'atm_pe_oi',
                'iv_skew', 'oi_buildup']

    # Drop existing to prevent _x/_y merge collision
    df1m = df1m.drop(columns=[c for c in opt_cols if c in df1m.columns])

    if options_df is None or options_df.empty:
        for col in opt_cols:
            df1m[col] = 0.0
        return df1m

    opt = options_df.copy()
    opt['date'] = pd.to_datetime(opt['date']).dt.date
    # Shift by 1 day: use yesterday's options data for today's signals
    opt['date'] = opt['date'].apply(lambda d: d)
    opt = opt.rename(columns={'date': 'prev_opt_date'})

    # Build a date → next_trading_date map
    all_dates = sorted(df1m['date'].unique())
    date_to_next = {}
    for i, d in enumerate(all_dates):
        if i + 1 < len(all_dates):
            date_to_next[opt['prev_opt_date'].iloc[0].__class__(d.year, d.month, d.day)] = all_dates[i + 1]

    # Map options date → next trading day (so today's 1m bars use yesterday's OI)
    opt_mapped = opt.copy()
    opt_mapped['date'] = opt_mapped['prev_opt_date'].apply(
        lambda d: date_to_next.get(d, None)
    )
    opt_mapped = opt_mapped.dropna(subset=['date'])
    opt_mapped = opt_mapped[['date'] + opt_cols]

    df1m = df1m.merge(opt_mapped, on='date', how='left')

    # Normalize ATM OI to ratio (CE OI / total) — more stable than raw OI counts
    total_atm_oi = df1m['atm_ce_oi'] + df1m['atm_pe_oi'] + EPS
    df1m['atm_oi_skew'] = (df1m['atm_pe_oi'] - df1m['atm_ce_oi']) / total_atm_oi
    opt_cols.append('atm_oi_skew')

    # Fill missing (days before options data starts)
    for col in opt_cols:
        df1m[col] = df1m[col].ffill().fillna(0.0)

    return df1m


def load_vix_data(vix_path: str = 'india_vix.csv') -> pd.DataFrame:
    """Load and preprocess India VIX CSV once. Returns shifted DataFrame ready to merge."""
    vix_cols = ['day_vix', 'day_vix_regime', 'day_vix_chg']
    if not os.path.exists(vix_path):
        logger.warning(f"[VIX] {vix_path} not found — run india_vix_downloader.py.")
        return pd.DataFrame(columns=['date'] + vix_cols)

    vix = pd.read_csv(vix_path)
    vix['date'] = pd.to_datetime(vix['date']).dt.date
    vix = vix.sort_values('date').drop_duplicates('date')
    vix['day_vix']        = vix['vix_close']
    vix['day_vix_regime'] = pd.cut(vix['vix_close'],
                                    bins=[0, 13, 20, 999],
                                    labels=[0, 1, 2]).astype(float)
    vix5avg = vix['vix_close'].rolling(5, min_periods=1).mean()
    vix['day_vix_chg'] = (vix['vix_close'] - vix5avg) / (vix5avg + EPS) * 100
    vix['date_next'] = vix['date'].shift(-1)
    vix_use = vix[['date_next', 'day_vix', 'day_vix_regime', 'day_vix_chg']].dropna(subset=['date_next'])
    vix_use = vix_use.rename(columns={'date_next': 'date'})
    vix_use['date'] = pd.to_datetime(vix_use['date']).dt.date
    return vix_use, vix['day_vix'].min(), vix['day_vix'].max()


def add_vix_features(df1m: pd.DataFrame, vix_path: str = 'india_vix.csv',
                     vix_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add India VIX as daily features to 1-min dataframe.
    Pass vix_df (from load_vix_data()) to avoid re-reading CSV every bar.
    """
    vix_cols = ['day_vix', 'day_vix_regime', 'day_vix_chg']
    df1m = df1m.drop(columns=[c for c in vix_cols if c in df1m.columns])

    if vix_df is None:
        result = load_vix_data(vix_path)
        if isinstance(result, tuple):
            vix_df, vmin, vmax = result
        else:
            vix_df = result
            vmin = vmax = 0.0
    else:
        vmin = vmax = 0.0

    if vix_df.empty:
        for col in vix_cols:
            df1m[col] = 0.0
        return df1m

    df1m = df1m.merge(vix_df, on='date', how='left')
    for col in vix_cols:
        df1m[col] = df1m[col].ffill().fillna(0.0)

    if vmin != vmax:
        logger.info(f"[VIX] Added VIX features. Range: {vmin:.1f}–{vmax:.1f}")
    else:
        logger.info(f"[VIX] Added VIX features.")
    return df1m


# ==============================================================================
# TIER 1 + TIER 2 CONTEXTUAL FEATURES
# ==============================================================================

def add_calendar_features(df1m: pd.DataFrame) -> pd.DataFrame:
    """
    Add day-of-week and expiry-week flags.

    Features added:
      day_of_week    — 0=Mon, 1=Tue, ..., 4=Fri (Monday opens weak, Friday expiry effect)
      is_expiry_week — 1 if current week contains Tuesday (NIFTY weekly expiry week)
      is_monday      — 1 on Monday (gap-prone, often weak open)
      is_friday      — 1 on Friday (position squaring, end-of-week moves)
    """
    cols = ['day_of_week', 'is_expiry_week', 'is_monday', 'is_friday']
    df1m = df1m.drop(columns=[c for c in cols if c in df1m.columns])

    dt = pd.to_datetime(df1m['datetime'] if 'datetime' in df1m.columns else df1m.index)
    df1m['day_of_week']    = dt.dt.dayofweek.astype(float)
    df1m['is_monday']      = (dt.dt.dayofweek == 0).astype(float)
    df1m['is_friday']      = (dt.dt.dayofweek == 4).astype(float)

    # Expiry week: week that contains Tuesday (NIFTY weekly expiry since Sep 2024)
    # A week contains Tuesday if any day Mon-Fri of that ISO week is a Tuesday.
    # Simplest: day_of_week <= 1 means Tuesday hasn't happened yet this week,
    # day_of_week >= 1 means this week has/had a Tuesday.
    # Since NIFTY expiry is every Tuesday, every trading week is expiry week —
    # so we use a more useful definition: flag the 2 days around expiry (Mon+Tue).
    df1m['is_expiry_week'] = (dt.dt.dayofweek <= 1).astype(float)  # Mon=0, Tue=1

    logger.info("[Calendar] Added day-of-week and expiry-week features.")
    return df1m




def load_fii_dii_data(fii_path: str = 'fii_dii_flow.csv') -> pd.DataFrame:
    """Load and preprocess FII/DII CSV once. Returns shifted DataFrame ready to merge."""
    cols = ['fii_net_buy', 'dii_net_buy', 'fii_dii_net', 'fii_flow_regime', 'fii_5d_cumulative']
    if not os.path.exists(fii_path):
        logger.warning(f"[FII] {fii_path} not found — run fii_dii_downloader.py.")
        return pd.DataFrame(columns=['date'] + cols)

    fii = pd.read_csv(fii_path)
    fii['date'] = pd.to_datetime(fii['date']).dt.date
    fii = fii.sort_values('date').drop_duplicates('date')
    fii['fii_dii_net']       = fii['fii_net_buy'] + fii['dii_net_buy']
    fii['fii_flow_regime']   = pd.cut(fii['fii_net_buy'],
                                       bins=[-999999, -2000, 2000, 999999],
                                       labels=[-1, 0, 1]).astype(float)
    fii['fii_5d_cumulative'] = fii['fii_net_buy'].rolling(5, min_periods=1).sum()
    fii['date_next'] = fii['date'].shift(-1)
    fii_use = fii[['date_next', 'fii_net_buy', 'dii_net_buy', 'fii_dii_net',
                   'fii_flow_regime', 'fii_5d_cumulative']].dropna(subset=['date_next'])
    fii_use = fii_use.rename(columns={'date_next': 'date'})
    fii_use['date'] = pd.to_datetime(fii_use['date']).dt.date
    return fii_use


def add_fii_dii_features(df1m: pd.DataFrame,
                          fii_path: str = 'fii_dii_flow.csv',
                          fii_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add FII/DII daily flow as features.
    Pass fii_df (from load_fii_dii_data()) to avoid re-reading CSV every bar.
    """
    cols = ['fii_net_buy', 'dii_net_buy', 'fii_dii_net', 'fii_flow_regime', 'fii_5d_cumulative']
    df1m = df1m.drop(columns=[c for c in cols if c in df1m.columns])

    if fii_df is None:
        fii_df = load_fii_dii_data(fii_path)

    if fii_df.empty:
        for col in cols:
            df1m[col] = 0.0
        return df1m

    df1m = df1m.merge(fii_df, on='date', how='left')
    for col in cols:
        df1m[col] = df1m[col].ffill().fillna(0.0)

    logger.info(f"[FII] Added FII/DII flow features. {len(fii_df)} days loaded.")
    return df1m


def load_sp500_data(sp500_path: str = 'sp500_daily.csv') -> pd.DataFrame:
    """Load and preprocess S&P 500 CSV once. Returns shifted DataFrame ready to merge."""
    cols = ['sp500_ret_1d', 'sp500_ret_5d', 'global_risk_on']
    if not os.path.exists(sp500_path):
        logger.warning(f"[SP500] {sp500_path} not found — run sp500_downloader.py.")
        return pd.DataFrame(columns=['date'] + cols)

    sp = pd.read_csv(sp500_path)
    sp['date'] = pd.to_datetime(sp['date']).dt.date
    sp = sp.sort_values('date').drop_duplicates('date')
    sp['sp500_ret_1d'] = sp['close'].pct_change(1) * 100
    sp['sp500_ret_5d'] = sp['close'].pct_change(5) * 100
    sp['global_risk_on'] = pd.cut(sp['sp500_ret_1d'],
                                   bins=[-999, -0.5, 0.5, 999],
                                   labels=[-1, 0, 1]).astype(float)
    # S&P 500 closes at ~2:30 AM IST (next calendar day).
    sp['date_next'] = sp['date'].shift(-1)
    sp_use = sp[['date_next', 'sp500_ret_1d', 'sp500_ret_5d', 'global_risk_on']].dropna(subset=['date_next'])
    sp_use = sp_use.rename(columns={'date_next': 'date'})
    sp_use['date'] = pd.to_datetime(sp_use['date']).dt.date
    return sp_use


def add_global_market_features(df1m: pd.DataFrame,
                                 sp500_path: str = 'sp500_daily.csv',
                                 sp500_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add US S&P 500 previous close return as global risk-on/off context.
    Pass sp500_df (from load_sp500_data()) to avoid re-reading CSV every bar.
    """
    cols = ['sp500_ret_1d', 'sp500_ret_5d', 'global_risk_on']
    df1m = df1m.drop(columns=[c for c in cols if c in df1m.columns])

    if sp500_df is None:
        sp500_df = load_sp500_data(sp500_path)

    if sp500_df.empty:
        for col in cols:
            df1m[col] = 0.0
        return df1m

    df1m = df1m.merge(sp500_df, on='date', how='left')
    for col in cols:
        df1m[col] = df1m[col].ffill().fillna(0.0)

    logger.info(f"[SP500] Added global market features. {len(sp500_df)} days loaded.")
    return df1m


def compute_pcr_volume_features(options_dir: str = 'nifty_options_data') -> pd.DataFrame:
    """
    Read all daily options CSVs once and return a DataFrame of PCR volume +
    ATM IV features indexed by date. Call this ONCE at startup, then pass the
    result to add_pcr_volume_features() every bar.
    """
    import glob
    cols = ['pcr_vol', 'pcr_oi_vol_diff', 'atm_iv_ce', 'atm_iv_pe', 'atm_iv_avg']
    records = []
    files = sorted(glob.glob(os.path.join(options_dir, 'nifty_*.csv')))

    for fpath in files:
        try:
            df = pd.read_csv(fpath)
            if df.empty or 'option_type' not in df.columns:
                continue
            date_str = os.path.basename(fpath).replace('nifty_', '').replace('.csv', '')
            try:
                trade_date = pd.to_datetime(date_str).date()
            except Exception:
                continue

            df['strike']    = pd.to_numeric(df['strike'],    errors='coerce')
            df['contracts'] = pd.to_numeric(df['contracts'], errors='coerce').fillna(0)
            df['close']     = pd.to_numeric(df['close'],     errors='coerce').fillna(0)
            df['open_int']  = pd.to_numeric(df['open_int'],  errors='coerce').fillna(0)
            df = df.dropna(subset=['strike'])

            ce = df[df['option_type'] == 'CE']
            pe = df[df['option_type'] == 'PE']
            if len(ce) == 0:
                continue

            total_ce_vol    = ce['contracts'].sum()
            total_pe_vol    = pe['contracts'].sum()
            pcr_vol         = total_pe_vol / (total_ce_vol + EPS)
            total_ce_oi     = ce['open_int'].sum()
            total_pe_oi     = pe['open_int'].sum()
            pcr_oi_val      = total_pe_oi / (total_ce_oi + EPS)
            pcr_oi_vol_diff = pcr_oi_val - pcr_vol

            spot_est    = ce.loc[ce['open_int'].idxmax(), 'strike'] if ce['open_int'].sum() > 0 else ce['strike'].median()
            all_strikes = df['strike'].unique()
            atm_strike  = all_strikes[np.argmin(np.abs(all_strikes - spot_est))]
            atm_ce_row  = ce[ce['strike'] == atm_strike]
            atm_pe_row  = pe[pe['strike'] == atm_strike]
            atm_ce_close = atm_ce_row['close'].mean() if len(atm_ce_row) > 0 else 0.0
            atm_pe_close = atm_pe_row['close'].mean() if len(atm_pe_row) > 0 else 0.0
            atm_iv_ce   = (atm_ce_close / (spot_est + EPS)) * np.sqrt(252) * 100
            atm_iv_pe   = (atm_pe_close / (spot_est + EPS)) * np.sqrt(252) * 100
            atm_iv_avg  = (atm_iv_ce + atm_iv_pe) / 2.0

            records.append({
                'date':            trade_date,
                'pcr_vol':         round(float(pcr_vol), 4),
                'pcr_oi_vol_diff': round(float(pcr_oi_vol_diff), 4),
                'atm_iv_ce':       round(float(atm_iv_ce), 2),
                'atm_iv_pe':       round(float(atm_iv_pe), 2),
                'atm_iv_avg':      round(float(atm_iv_avg), 2),
            })
        except Exception as e:
            logger.debug(f"[PCRVol] Skipped {fpath}: {e}")
            continue

    if not records:
        logger.warning("[PCRVol] No data loaded")
        return pd.DataFrame(columns=['date'] + cols)

    result = pd.DataFrame(records).sort_values('date').reset_index(drop=True)
    logger.info(f"[PCRVol] Computed PCR-volume + ATM IV features. {len(result)} days.")
    return result


def add_pcr_volume_features(df1m: pd.DataFrame,
                              pcr_df: pd.DataFrame | None = None,
                              options_dir: str = 'nifty_options_data') -> pd.DataFrame:
    """
    Merge pre-computed PCR volume + ATM IV features into 1-min dataframe.
    Pass pcr_df from compute_pcr_volume_features() for live use (fast).
    If pcr_df is None, computes from scratch (training use only).
    """
    cols = ['pcr_vol', 'pcr_oi_vol_diff', 'atm_iv_ce', 'atm_iv_pe', 'atm_iv_avg']
    df1m = df1m.drop(columns=[c for c in cols if c in df1m.columns])

    if pcr_df is None:
        pcr_df = compute_pcr_volume_features(options_dir)

    if pcr_df is None or pcr_df.empty:
        for col in cols:
            df1m[col] = 0.0
        return df1m

    result = pcr_df.copy()
    result['date'] = pd.to_datetime(result['date']).dt.date
    # Shift by 1 day (no lookahead)
    result['date_next'] = result['date'].shift(-1)
    result_use = result[['date_next'] + cols].dropna(subset=['date_next'])
    result_use = result_use.rename(columns={'date_next': 'date'})
    result_use['date'] = pd.to_datetime(result_use['date']).dt.date

    df1m = df1m.merge(result_use, on='date', how='left')
    for col in cols:
        df1m[col] = df1m[col].ffill().fillna(0.0)

    logger.info(f"[PCRVol] Added PCR-volume + ATM IV features. {len(pcr_df)} days.")
    return df1m


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
    day_open                 = df1m.groupby('date')['open'].transform('first')
    df1m['gap_pct']          = (day_open - df1m['prev_close']) / (df1m['prev_close']+1e-9)*100
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
    # Calendar: always available (derived from datetime)
    'day_of_week': True,
    'is_expiry_week': True,
    'is_monday': True,
    'is_friday': True,
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
    'adx_14', 'adx_slope',
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

        # ----- GROUP E: Calendar context (derived from datetime, zero cost) -----
        'day_of_week',    # 0=Mon..4=Fri (each day has distinct NIFTY behavior)
        'is_expiry_week', # 1 on Mon/Tue (expiry-week positioning effect)
        'is_monday',      # 1 on Monday (gap-prone, often weak open)
        'is_friday',      # 1 on Friday (position squaring, end-of-week moves)

        # ----- GROUP I: Price Action Structures (v4.2) -----
        'struct_score',   # +1=HH/HL bullish structure, -1=LH/LL bearish, 0=mixed
        'fvg_bull',       # fair value gap bullish (gap up imbalance in last 5 bars)
        'fvg_bear',       # fair value gap bearish (gap down imbalance in last 5 bars)
        'liq_sweep_up',   # liquidity sweep of equal highs (potential bearish reversal)
        'liq_sweep_dn',   # liquidity sweep of equal lows (potential bullish reversal)
        'nr4',            # NR4 inside compression (breakout setup)
        'nr7',            # NR7 tight compression (stronger breakout signal)
        'orb_dist',       # distance from OR boundary (ORB strength)

        # ----- DISABLED (unavailable live — excluded via FEATURE_LIVE_OK) -----
        'option_chain_ndi',
        'hdfc_ret_1m', 'reliance_ret_1m',
        'banknifty_spread', 'banknifty_divergence',
        'fft_cycle', 'cycle_vol_interaction',
    ]
    return base

FEATURE_COLS = get_feature_cols()


