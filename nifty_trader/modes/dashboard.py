"""Live dashboard rendering and training summary display."""
import logging
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from ..config import (
    BAR_WIDTH, HORIZONS, REGIME_NAMES, REGIME_ICONS, REGIME_CRISIS,
    CONF_MODERATE, STOP_LOSS_PCT, TARGET_PCT,
)
from ..features.feature_engineering import FEATURE_COLS, FEATURE_LIVE_OK
from ..execution.orders import _next_expiry_mins, estimate_option_premium
from ..utils.safeguards import safe_value
from ..signals.analysis import score_feature

logger = logging.getLogger(__name__)

# ==============================================================================
# 9. SUMMARY REPORTING (TRAINING)
# ==============================================================================

def print_training_summary(models: dict):
    print(f"\n{'='*78}")
    print("TRAINING COMPLETE -- Walk-Forward Results")
    print(f"{'='*78}")
    hdr = (f"  {'Hz':<5} {'Acc':>6} {'Base':>6} {'Edge':>7} "
           f"{'Trades':>7} {'WR':>6} {'PF':>6} {'Sharpe':>7} "
           f"{'MaxDD':>7} {'Total':>8}")
    print(hdr)
    print(f"  {'-'*76}")

    for label, bt_key in [('ALL REGIMES', 'backtest'),
                           ('TRENDING ONLY', 'backtest_trending')]:
        print(f"\n  --- {label} ---")
        for h, res in sorted(models.items()):
            if res is None: continue
            bt   = res.get(bt_key, {})
            if not bt or bt.get('trades', 0) == 0:
                print(f"  {h}m     (no trades in this filter)")
                continue
            edge = (res['avg_acc'] - res['baseline']) * 100
            print(f"  {h}m{'':<3} {res['avg_acc']:>5.1%} {res['baseline']:>6.1%} "
                  f"{edge:>+6.1f}pp "
                  f"{bt['trades']:>7,} {bt['win_rate']:>6.1%} "
                  f"{bt['profit_factor']:>6.2f} {bt['sharpe']:>7.2f} "
                  f"{bt['max_dd']:>7.1f}% {bt['total']:>+8.1f}%")
    print(f"\n{'='*78}")
    print("Costs: COST_RT_PCT (2.5%) deducted per trade. Options traders add theta + IV-crush on top.")
    print(f"{'='*78}")


# ==============================================================================
# 10. LIVE DASHBOARD
# ==============================================================================

def _bar(value, min_v, max_v, width=BAR_WIDTH, fill='#', empty='.') -> str:
    """Render a horizontal bar for a value in [min_v, max_v]."""
    clipped = max(min_v, min(max_v, value))
    frac    = (clipped - min_v) / (max_v - min_v + 1e-9)
    filled  = int(round(frac * width))
    return fill * filled + empty * (width - filled)


def _signal_bar(proba: float, width=20) -> str:
    """Two-sided bar: left = DOWN prob, right = UP prob."""
    up_frac = proba
    dn_frac = 1 - proba
    up_bars = int(round(up_frac * width / 2))
    dn_bars = int(round(dn_frac * width / 2))
    mid = width // 2
    buf = list('.' * width)
    # UP side (right of centre)
    for i in range(mid, mid + up_bars):
        if i < width: buf[i] = '^'
    # DOWN side (left of centre)
    for i in range(mid - dn_bars, mid):
        if i >= 0: buf[i] = 'v'
    buf[mid] = '|'
    return ''.join(buf)


def _verdict(score: float) -> str:
    """Convert -1..+1 score to coloured text label."""
    if   score >  0.50: return "*** STRONG BUY  ***"
    elif score >  0.20: return "++  BUY          ++"
    elif score >  0.05: return "+   MILD BUY      +"
    elif score > -0.05: return "--- NEUTRAL      ---"
    elif score > -0.20: return "-   MILD SELL     -"
    elif score > -0.50: return "--  SELL          --"
    else:               return "*** STRONG SELL ***"


def print_live_dashboard(row: pd.Series, analysis: dict,
                          signal: dict | None, models: dict,
                          current_regime: int, trade_info: dict | None,
                          now: datetime,
                          micro_regime: str = 'UNKNOWN',
                          ks_blocked: bool = False,
                          ks_reason: str = '',
                          streaming: bool = False):
    """Full per-minute analysis dashboard printed to terminal."""
    W = 72
    SEP  = "=" * W
    sep2 = "-" * W

    spot    = float(row.get('close', 0))
    ret_1m  = float(row.get('ret_1m',  0))
    ret_5m  = float(row.get('ret_5m',  0))
    ret_15m = float(row.get('ret_15m', 0))
    mod     = int(row.get('minute_of_day', 0))
    hhmm    = (mod + 9*60 + 15)
    t_str   = f"{hhmm//60:02d}:{hhmm%60:02d}"

    print(f"\n{SEP}")
    stream_tag = "  [STREAMING]" if streaming else ""
    print(f"  NIFTY LIVE ANALYSIS   {now.strftime('%Y-%m-%d %H:%M:%S')}   "
          f"IST {t_str}{stream_tag}")
    print(SEP)

    # --- PRICE ---
    arrow = "^" if ret_1m >= 0 else "v"
    print(f"  SPOT  : {spot:>9.2f}   1m: {ret_1m:>+6.3f}%  "
          f"5m: {ret_5m:>+6.3f}%  15m: {ret_15m:>+6.3f}%   {arrow}")
    gap   = float(row.get('gap_pct', 0))
    wret  = float(row.get('weekly_ret', 0))
    print(f"  GAP   : {gap:>+6.3f}%   Week: {wret:>+6.2f}%   "
          f"Day ADX: {analysis['adx']:>5.1f}   "
          f"IV~={analysis['iv']:.2f}%ann  [{analysis['iv_regime']}]")

    # --- REGIME + MICRO-REGIME + KILL-SWITCH ---
    r_icon = analysis['regime_icon']
    print(f"  REGIME: {r_icon}  Micro: {micro_regime:<14}  "
          f"Market bias: {analysis['overall_bias']:<16} "
          f"Score: {analysis['overall_score']:>+.3f}")
    if ks_blocked:
        print(f"  *** KILL-SWITCH ACTIVE: {ks_reason} ***")
    print(sep2)

    # --- MULTI-TIMEFRAME INDICATORS TABLE ---
    print(f"  {'INDICATOR':<18} {'1-MIN':>9} {'5-MIN':>9} {'15-MIN':>9}  "
          f"{'STATUS'}")
    print(f"  {'-'*16} {'-'*9} {'-'*9} {'-'*9}  {'-'*12}")

    def _fmt(v, lo_bad, lo_ok, hi_ok, hi_bad, fmt='{:.1f}'):
        s = score_feature(v, lo_bad, lo_ok, hi_ok, hi_bad)
        tag = ['<<', '<', '~', '>', '>>'][s + 2]
        return f"{fmt.format(v)}{tag}"

    r1_rsi  = float(row.get('rsi_14',        50))
    r5_rsi  = float(row.get('tf5_rsi',       50))
    r15_rsi = float(row.get('tf15_rsi',      50))
    print(f"  {'RSI(14)':<18} {_fmt(r1_rsi,30,40,60,70):>9} "
          f"{_fmt(r5_rsi,30,40,60,70):>9} {_fmt(r15_rsi,30,40,60,70):>9}  "
          f"{'OVERBOUGHT' if r1_rsi>70 else 'OVERSOLD' if r1_rsi<30 else 'NORMAL'}")

    r1_stk  = float(row.get('stoch_k',      50))
    r5_stk  = float(row.get('tf5_stoch_k',  50))
    r15_stk = float(row.get('tf15_stoch_k', 50))
    print(f"  {'Stoch K':<18} {_fmt(r1_stk,15,25,75,85):>9} "
          f"{_fmt(r5_stk,15,25,75,85):>9} {_fmt(r15_stk,15,25,75,85):>9}  "
          f"{'OB' if r1_stk>80 else 'OS' if r1_stk<20 else 'NORMAL'}")

    r1_cci  = float(row.get('cci_20',       0))
    r5_cci  = float(row.get('tf5_cci',      0))
    r15_cci = float(row.get('tf15_cci',     0))
    print(f"  {'CCI(20)':<18} {_fmt(r1_cci,-150,-100,100,150):>9} "
          f"{_fmt(r5_cci,-150,-100,100,150):>9} {_fmt(r15_cci,-150,-100,100,150):>9}  "
          f"{'OB' if r1_cci>100 else 'OS' if r1_cci<-100 else 'NORMAL'}")

    r1_adx  = float(row.get('adx_14',       0))
    r5_adx  = float(row.get('tf5_adx',      0))
    r15_adx = float(row.get('tf15_adx',     0))
    adx_txt = "STRONG TREND" if r1_adx > 30 else "WEAK/RANGING" if r1_adx < 20 else "MODERATE"
    print(f"  {'ADX(14)':<18} {r1_adx:>8.1f}~ {r5_adx:>8.1f}~ {r15_adx:>8.1f}~  {adx_txt}")

    r1_wil  = float(row.get('willr',          -50))
    r5_wil  = float(row.get('tf5_willr',      -50))
    r15_wil = float(row.get('tf15_willr',     -50))
    print(f"  {'Williams %R':<18} {_fmt(r1_wil,-95,-80,-20,-5):>9} "
          f"{_fmt(r5_wil,-95,-80,-20,-5):>9} {_fmt(r15_wil,-95,-80,-20,-5):>9}  "
          f"{'OB' if r1_wil>-5 else 'OS' if r1_wil<-90 else 'NORMAL'}")

    r1_macd = float(row.get('macd_h',         0))
    r5_macd = float(row.get('tf5_macd_h',     0))
    r15_macd= float(row.get('tf15_macd_h',    0))
    print(f"  {'MACD Hist':<18} {r1_macd:>+9.4f} {r5_macd:>+9.4f} {r15_macd:>+9.4f}  "
          f"{'BULL' if r1_macd>0 else 'BEAR'} "
          f"{'CROSS' if row.get('macd_cross',0) else ''}")

    r1_bb   = float(row.get('bb_pos',         0.5))
    r5_bb   = float(row.get('tf5_bb_pos',     0.5))
    r15_bb  = float(row.get('tf15_bb_pos',    0.5))
    bb_wid  = float(row.get('bb_width',       0))
    print(f"  {'BB Position':<18} {r1_bb:>9.3f} {r5_bb:>9.3f} {r15_bb:>9.3f}  "
          f"{'UPPER' if r1_bb>0.8 else 'LOWER' if r1_bb<0.2 else 'MID'}  "
          f"Width={bb_wid:.2f}%{'  [SQUEEZE]' if analysis['bb_squeeze'] else ''}")

    print(sep2)

    # --- VWAP + OR + STRUCTURE ---
    vd  = analysis['vwap_dist']
    ab  = "ABOVE" if analysis['above_vwap'] else "BELOW"
    orp = analysis['or_pos']
    print(f"  VWAP: {ab}  dist={vd:>+6.3f}%   "
          f"OR position: {orp:.2f}  "
          f"{'[OR BREAKOUT UP]' if analysis['or_break_up'] else '[OR BREAKDOWN]' if analysis['or_break_dn'] else '[INSIDE OR]'}")

    phi = analysis['pivot_hi']; plo = analysis['pivot_lo']
    print(f"  Nearest resistance: {phi:>+5.2f}% away   "
          f"Nearest support: {plo:>+5.2f}% below   "
          f"MA50: {'above' if analysis['day_above_ma50'] else 'below'}  "
          f"MA200: {'above' if analysis['day_above_ma200'] else 'below'}")

    sup  = analysis['supertrend']
    cg   = analysis['consec_green']
    cr   = analysis['consec_red']
    accel= analysis['price_accel']
    print(f"  Supertrend: {'BULL' if sup>0 else 'BEAR'}   "
          f"Consec green: {cg}   Consec red: {cr}   "
          f"Price accel: {accel:>+.4f}  "
          f"{'[BULL 3BAR]' if analysis['bull_3bar'] else '[BEAR 3BAR]' if analysis['bear_3bar'] else ''}")

    mfi = analysis['mfi']
    print(f"  MFI(14): {mfi:.1f}  "
          f"{'[BUYING PRESSURE]' if mfi>60 else '[SELLING PRESSURE]' if mfi<40 else '[NEUTRAL FLOW]'}  "
          f"Vol ratio: {analysis['vol_ratio']:.2f}x  "
          f"Trend strength: {analysis['day_trend_strength']:>+.2f}")

    print(sep2)

    # --- COMPOSITE SCORES + BARS ---
    ms = analysis['momentum_score'];  ts = analysis['trend_score']
    fs = analysis['flow_score'] / 2;  ov = analysis['overall_score']
    print(f"  MOMENTUM   [{_bar(ms+2,0,4,20)}]  {ms:>+.2f}")
    print(f"  TREND      [{_bar(ts+1,0,2,20)}]  {ts:>+.2f}")
    print(f"  FLOW/MFI   [{_bar(fs+1,0,2,20)}]  {fs:>+.2f}")
    print(f"  OVERALL    [{_bar(ov+1,0,2,20)}]  {ov:>+.3f}  {_verdict(ov)}")

    print(sep2)

    # --- PER-HORIZON PREDICTIONS ---
    print(f"  PREDICTIONS  (each bar: v=DOWN | = NEUTRAL ^=UP)")
    print(f"  {'Horizon':<10} {'Proba UP':>10} {'Signal bar (DN | UP)':>30} "
          f"  {'Confidence':>10}  Model Acc")

    h_order = sorted(models.keys())
    for h in h_order:
        res = models[h]
        try:
            # Extract active features from model (161 features, not 168)
            active_features = res.get('active_features')
            if active_features is None:
                # Fallback: filter out unavailable features
                active_features = [f for f in FEATURE_COLS if FEATURE_LIVE_OK.get(f, True)]
            
            # Handle NaN values - replace with 0 before passing to model
            X = np.array([[0.0 if pd.isna(row.get(fc, 0)) else row.get(fc, 0) for fc in active_features]], dtype=np.float32)
            live_scaler = res.get('live_scaler')
            if live_scaler is not None:
                try:
                    X = live_scaler.transform(X)
                except Exception:
                    pass  # use raw if scaler fails
            pr = res['final_model'].predict_proba(X)[0]
            proba_up = float(pr[1])
            conf     = max(proba_up, 1 - proba_up)
            pred_str = " UP  " if proba_up > 0.5 else " DOWN"
            acc      = res.get('avg_acc', 0)
            sbar     = _signal_bar(proba_up, width=24)
            star     = "*" if conf >= CONF_MODERATE else " "
            print(f"  {str(h)+'min':<10} {proba_up:>9.1%}  [{sbar}]  "
                  f"{pred_str}{star}  conf={conf:.1%}  acc={acc:.1%}")
            # Store for decision summary
            res['pred_proba_up'] = proba_up
            res['pred_conf'] = conf
        except Exception as e:
            print(f"  {str(h)+'min':<10}  error: {e}")

    print(sep2)

    # --- SIGNAL DECISION ---
    if signal is None:
        # Explain why no signal
        sp   = float(row.get('session_pct', 0))
        mod2 = int(row.get('minute_of_day', 0))
        iv_p = row.get('iv_proxy', 0)
        iv   = float(iv_p) if iv_p and iv_p > 0 else float(row.get('atr_14_pct', 0))
        if current_regime == REGIME_CRISIS:
            reason = "CRISIS regime -- gate filtered (check logs for [Gate] BLOCKED)"
        elif sp < 0.10 or sp > 0.92:
            reason = f"Outside trading session window (sp={sp:.2f})"
        elif 180 <= mod2 <= 225:
            reason = "Lunch hour filter (12:15-13:00)"
        elif row.get('is_expiry', 0) == 1 and mod2 > 315:
            reason = "Expiry day -- no new trades after 14:30"
        elif iv < 0.05:
            reason = f"IV too low for premium trades (iv={iv:.3f})"
        else:
            reason = "Models disagree or confidence below threshold"
        print(f"  SIGNAL : NONE  [{reason}]")
        
        # Even when no signal, show dynamic strike monitor for monitoring
        print(sep2)
        print(f"  DYNAMIC STRIKE MONITOR (ATM Strike)")

        # Calculate dynamic strikes for display even without signal
        spot_now = safe_value(row.get('close', 0))
        atr_now = safe_value(row.get('atr_14', 0))
        # iv_proxy is daily IV (%). Annualise by ×sqrt(252) for estimate_option_premium().
        iv_now_raw = row.get('iv_proxy', 0)
        iv_daily = safe_value(iv_now_raw) if iv_now_raw > 0 else 0.06
        iv_now = iv_daily * (252 ** 0.5)   # annualised IV ~14-18% for NIFTY
        dte_now = _next_expiry_mins()
        
        # ATM strike: highest delta (~0.5), most responsive to spot moves
        atm_now = int(round(spot_now / 50) * 50)
        strike_ce = atm_now
        strike_pe = atm_now

        # Use real LTP from Angel One if available, else fall back to estimate
        ce_ltp = float(row.get('atm_ce_ltp', 0))
        pe_ltp = float(row.get('atm_pe_ltp', 0))
        if ce_ltp <= 0:
            ce_ltp = estimate_option_premium(spot_now, iv_now, dte_now, strike=float(strike_ce), option_type='CE')
        if pe_ltp <= 0:
            pe_ltp = estimate_option_premium(spot_now, iv_now, dte_now, strike=float(strike_pe), option_type='PE')

        ce_label_ns = "ATM"
        pe_label_ns = "ATM"
        print(f"  Active Strikes  CE: {strike_ce} ({ce_label_ns})  |  PE: {strike_pe} ({pe_label_ns})")
        print(f"  Current LTP     CE: Rs {ce_ltp:.2f}  |  PE: Rs {pe_ltp:.2f}")
        print()

        # ACTIONABLE RECOMMENDATIONS: Clear CE or PE guidance for 1, 5, 15 min
        print(f"  {'Horizon':<10} {'Signal':<8} {'Option':<10} {'Target':<12} {'Confidence':<12} {'Expected PnL'}")
        print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
        
        # Delta for ATM options (~0.50); adjusted for moneyness
        _dte_days  = max(dte_now, 1.0) / 375.0
        _sharpness = 1.0 + 3.0 * np.exp(-_dte_days)
        _THETA_PER_MIN = 0.0004   # ~0.04%/min time decay
        _SLIP = 0.03

        for h in [1, 5, 15]:
            if h not in models:
                continue

            # Get ML prediction for this horizon
            proba_up = models[h].get('pred_proba_up', 0.5)
            confidence = models[h].get('confidence', abs(proba_up - 0.5) * 2)

            # Determine direction and which option to trade
            if proba_up > 0.5:
                direction = "BUY"
                option_type = "CE"
                signal_icon = "^"
            else:
                direction = "BUY"
                option_type = "PE"
                signal_icon = "v"

            # Project spot movement
            expected_move = (proba_up - 0.5) * 2.0 * atr_now * np.sqrt(h)

            # Delta-based projection anchored on real current LTP
            if option_type == 'CE':
                current_premium = ce_ltp
                strike_used = strike_ce
                _mono = spot_now / float(strike_ce) if strike_ce > 0 else 1.0
                _delta = float(np.clip(0.50 + _sharpness * (_mono - 1.0), 0.10, 0.90))
                _theta = current_premium * _THETA_PER_MIN * h
                proj_premium = max(current_premium + _delta * expected_move - _theta, 1.0)
            else:
                current_premium = pe_ltp
                strike_used = strike_pe
                _mono = spot_now / float(strike_pe) if strike_pe > 0 else 1.0
                _delta = float(np.clip(0.50 + _sharpness * (1.0 - _mono), 0.10, 0.90))
                _theta = current_premium * _THETA_PER_MIN * h
                proj_premium = max(current_premium - _delta * expected_move - _theta, 1.0)

            proj_premium *= (1.0 - _SLIP)   # apply slippage
            pnl = proj_premium - current_premium
            pnl_pct = (pnl / current_premium * 100) if current_premium > 0 else 0
            
            # Format recommendation
            option_display = f"{option_type} {strike_used}"
            target_display = f"Rs{proj_premium:.2f}"
            conf_display = f"{confidence*100:.1f}%"
            pnl_display = f"{'+' if pnl > 0 else ''}{pnl_pct:.1f}%"
            
            print(f"  {str(h)+'min':<10} {signal_icon} {direction:<6} {option_display:<10} "
                  f"{target_display:<12} {conf_display:<12} {pnl_display}")
        
        print()
        print(f"  Note: Recommendations based on ML predictions + Black-Scholes pricing model")
    
    print(sep2)
    
    # ==============================================================================
    # DECISION SUMMARY & ACTIONABLE RECOMMENDATION
    # ==============================================================================
    print("  DECISION SUMMARY")
    print(sep2)
    
    # Aggregate ML predictions (use stored values from prediction loop above)
    ml_up_count = sum(1 for h in h_order if models[h].get('pred_proba_up', 0.5) > 0.5)
    ml_dn_count = len(h_order) - ml_up_count
    ml_avg_conf = np.mean([models[h].get('pred_conf', 0.5) for h in h_order if 'pred_conf' in models[h]])
    
    # ML consensus
    if ml_up_count > ml_dn_count:
        ml_direction = "BULLISH"
        ml_symbol = "^"
    elif ml_dn_count > ml_up_count:
        ml_direction = "BEARISH"
        ml_symbol = "v"
    else:
        ml_direction = "NEUTRAL"
        ml_symbol = "="

    # Technical analysis bias
    tech_score = analysis['overall_score']
    if tech_score > 0.35:
        tech_direction = "BULLISH"
        tech_symbol = "^"
    elif tech_score < -0.35:
        tech_direction = "BEARISH"
        tech_symbol = "v"
    else:
        tech_direction = "NEUTRAL"
        tech_symbol = "="
    
    # Agreement check
    agreement = (ml_direction == tech_direction) or (ml_direction == "NEUTRAL") or (tech_direction == "NEUTRAL")
    
    print(f"  ML Models ({len(h_order)} horizons):  {ml_direction} {ml_symbol}  "
          f"({ml_up_count} UP, {ml_dn_count} DOWN)  Avg Conf: {ml_avg_conf:.1%}")
    print(f"  Technical Analysis:      {tech_direction} {tech_symbol}  Score: {tech_score:+.3f}")
    print(f"  Agreement:               {'YES' if agreement else 'NO'}")
    
    # Trading recommendation
    print(sep2)
    
    if signal is not None:
        # Active signal - ready to trade
        signal_dir = signal['direction']
        signal_conf = signal['avg_conf']
        print(f"  [SIGNAL] TRADE SIGNAL ACTIVE: {signal_dir} | Confidence: {signal_conf:.1%}")
        print(f"  ACTION: Execute {signal_dir} trade (system will auto-execute in paper mode)")
    else:
        # No signal - explain why and give recommendation
        # Get block reason
        sp = float(row.get('session_pct', 0))
        mod = int(row.get('minute_of_day', 0))
        iv_p = row.get('iv_proxy', 0)
        iv = float(iv_p) if iv_p and iv_p > 0 else float(row.get('atr_14_pct', 0))
        
        if current_regime == REGIME_CRISIS:
            block_reason = "Crisis regime (bypass active if agreement>=85% -- see logs)"
        elif sp < 0.10:
            block_reason = "Pre-market session (first 10%)"
        elif sp > 0.92:
            block_reason = "Post-market session (last 8%)"
        elif 180 <= mod <= 225:
            block_reason = "Lunch hour filter (12:15-13:00)"
        elif row.get('is_expiry', 0) == 1 and mod > 315:
            block_reason = "Expiry day late trading (after 14:30)"
        elif iv < 0.05:
            block_reason = f"IV too low for premium trades (iv={iv:.3f})"
        elif ml_avg_conf < 0.58:
            block_reason = f"Insufficient ML confidence ({ml_avg_conf:.1%} < 58%)"
        elif not agreement:
            block_reason = "ML and technical analysis contradicting"
        else:
            block_reason = "Kill-switch or other filter blocked"
        
        print(f"  NO SIGNAL: {block_reason}")

        # Actionable recommendation — only show when the block is not a hard session/regime gate
        _hard_gate = sp < 0.10 or sp > 0.92 or (180 <= mod < 225) or (row.get('is_expiry', 0) == 1 and mod > 315)
        if not _hard_gate:
            if ml_avg_conf >= 0.55:
                if ml_direction == "BULLISH":
                    print(f"  RECOMMENDATION: Cautious LONG bias - Wait for confirming signal")
                elif ml_direction == "BEARISH":
                    print(f"  RECOMMENDATION: Cautious SHORT bias - Wait for confirming signal")
                else:
                    print(f"  RECOMMENDATION: STAY FLAT - No clear directional edge")
            else:
                print(f"  RECOMMENDATION: STAY FLAT - Low confidence ({ml_avg_conf:.1%})")
    
    print(sep2)
    
    # Continue with existing signal details display if signal exists
    if signal is not None:
        dir_arrow = "^" if signal['direction'] == 'UP' else "v"
        print(f"  SIGNAL  : *** {signal['direction']} {dir_arrow} | "
              f"{signal['strength']} | conf={signal['avg_conf']:.1%} | "
              f"agree={signal['agreement']:.0%} | "
              f"w_up={signal['w_up']:.3f} w_dn={signal['w_dn']:.3f} horizons={signal['n_valid']} ***")

        # v3.1/v3.2 EV + MetaLabeler fields
        ev_raw   = signal.get('ev_raw',  float('nan'))
        ev_net   = signal.get('ev_net',  float('nan'))
        cpctile  = signal.get('conf_pctile', float('nan'))
        sbias    = signal.get('season_bias', 0.0)
        stale_p  = signal.get('stale_penalty', 0.0)
        meta_c   = signal.get('meta_conf', 1.0)
        micro_r  = signal.get('micro_regime', 'UNKNOWN')
        in_tz    = signal.get('in_transition_zone', False)
        tz_tag   = " [TRANSITION ZONE]" if in_tz else ""
        moe_tag  = f"  [MoE:{REGIME_NAMES.get(current_regime,'?')}]"
        print(f"  EV      : raw={ev_raw:>+.4f}  net={ev_net:>+.4f}  "
              f"pctile={cpctile:.0f}th  "
              f"season_bias={sbias:>+.3f}  stale_pen={stale_p:.3f}{tz_tag}")
        print(f"  MetaLbl : correctness_prob={meta_c:.1%}{moe_tag}")

        if trade_info:
            t = trade_info
            print(f"  OPTION  : NIFTY {t['strike']} {t['option_type']}")
            print(f"  PREMIUM : ~{t['est_premium']:.0f}   "
                  f"Lots: {t['contracts']}   "
                  f"Notional: {t['notional']:,.0f}   "
                  f"Max risk: {t['max_risk']:,.0f}")
            print(f"  LEVELS  : Stop={t['stop_price']:.0f}  "
                  f"Target={t['target_price']:.0f}  "
                  f"R:R = 1:{TARGET_PCT/STOP_LOSS_PCT:.1f}")
            print(f"  WARNING : Estimated premiums only. Verify price on your platform.")
    
    print(sep2)
    
    # ==============================================================================
    # DYNAMIC STRIKE MONITOR: Deep ITM Projections
    # ==============================================================================
    # Display dynamic strikes (CE = spot - 100, PE = spot + 100) with slippage-adjusted projections
    if signal and 'dynamic_projections' in signal:
        strike_ce = signal.get('strike_ce', 0)
        strike_pe = signal.get('strike_pe', 0)
        ce_ltp_now = signal.get('ce_ltp_current', 0)
        pe_ltp_now = signal.get('pe_ltp_current', 0)
        projections = signal.get('dynamic_projections', {})
        
        ce_label = "ATM"
        pe_label = "ATM"
        print(f"  DYNAMIC STRIKE MONITOR (ATM Strike)")
        print(f"  Active Strikes  CE: {strike_ce} ({ce_label})  |  PE: {strike_pe} ({pe_label})")
        print(f"  Current LTP     CE: Rs {ce_ltp_now:.2f}  |  PE: Rs {pe_ltp_now:.2f}")
        print(f"  {'-'*70}")
        print(f"  {'Horizon':<10} {'Proj Spot':>12} {'CE Net LTP':>14} {'CE PnL%':>12} "
              f"{'PE Net LTP':>14} {'PE PnL%':>12}")
        print(f"  {'-'*70}")
        
        for h in [1, 5, 15]:
            if h in projections:
                p = projections[h]
                proj_spot = p.get('proj_spot', 0)
                ce_net = p.get('ce_net_ltp', 0)
                pe_net = p.get('pe_net_ltp', 0)
                ce_pnl = p.get('ce_pnl_pct', 0)
                pe_pnl = p.get('pe_pnl_pct', 0)
                
                # Color indicators for PnL
                ce_arrow = "^" if ce_pnl > 0 else "v" if ce_pnl < 0 else "-"
                pe_arrow = "^" if pe_pnl > 0 else "v" if pe_pnl < 0 else "-"

                print(f"  {str(h)+'min':<10} {proj_spot:>12,.2f} "
                      f"Rs{ce_net:>12,.2f} {ce_arrow} {ce_pnl:>+10.2f}% "
                      f"Rs{pe_net:>12,.2f} {pe_arrow} {pe_pnl:>+10.2f}%")
            else:
                print(f"  {str(h)+'min':<10} {'N/A':>12} {'N/A':>14} {'N/A':>12} {'N/A':>14} {'N/A':>12}")
        
        print(f"  {'-'*70}")
        print(f"  Note: Net LTP includes 3% slippage adjustment | Theta decay embedded")

    print(SEP)


# ==============================================================================
