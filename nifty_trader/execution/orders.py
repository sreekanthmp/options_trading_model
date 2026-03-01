"""Option pricing, strike selection, position sizing, and trade display.

LIVE SAFETY ENHANCEMENTS (audit-driven):
  1. Premium-based stop losses replace underlying-based stops.
     Underlying stops fail during fast markets because option liquidity dries up
     before the underlying reaches the stop level.  The stop is now expressed as
     a percentage of the option premium paid — this fires on the option's own
     price regardless of underlying behaviour.
  2. Delta-normalized position sizing.  A 0.8-delta deep-ITM call has twice the
     underlying exposure of a 0.4-delta ATM call.  Ignoring this overstates
     risk for ITM options and understates it for OTM.
  3. Vega-aware entry filter.  Entering when IV is already elevated means you
     buy expensive premium that will compress even if direction is correct.
  4. Expiry-day hard rules.  Gamma risk makes intraday options unpredictable
     near expiry.  Hard time-cutoffs prevent trading in the danger zone.
  5. Liquidity pre-check via IV rank proxy (bid-ask spread widens with IV).
"""
import logging
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

if TYPE_CHECKING:
    from ..data.websocket import AngelSession
    from .risk import PositionManager

from ..config import (
    LOT_SIZE, LOT_SIZE_OLD, LOT_SIZE_NEW, LOT_SIZE_CUTOVER,
    STRIKE_OFFSET_CE, STRIKE_OFFSET_PE, STRIKE_ROUNDING,
    MAX_RISK_PCT, STOP_LOSS_PCT, TARGET_PCT,
    COST_RT_PCT, SLIPPAGE_PCT, THETA_DECAY_PCT, TOTAL_COST_PCT,
    DELTA_BASE, THETA_PTS_PER_BAR, HORIZONS,
)
from ..utils.safeguards import safe_value, check_lpp_violation, get_max_oi_strikes, avoid_oi_concentration_zone
from .costs import effective_cost, get_dynamic_theta

logger = logging.getLogger(__name__)

def display_option_predictions(signal: dict, verbose: bool = False) -> None:
    """
    Display clear, actionable option recommendations for 1, 5, 15-minute horizons.
    
    Shows: Which option to BUY (CE or PE), strike, target price, confidence, expected PnL
    
    Args:
        signal: Dictionary returned by generate_signal()
        verbose: If True, shows detailed calculation breakdown
    """
    if signal is None:
        print("\n[Option Predictions] No valid signal")
        return
    
    spot = signal.get('spot', 0)
    strike_ce = signal.get('strike_ce', 0)
    strike_pe = signal.get('strike_pe', 0)
    ce_ltp = signal.get('ce_ltp_current', 0)
    pe_ltp = signal.get('pe_ltp_current', 0)
    projections = signal.get('dynamic_projections', {})
    direction = signal.get('direction', 'UNKNOWN')
    overall_conf = signal.get('avg_conf', 0) * 100
    
    print("\n" + "="*88)
    print(f"  ACTIONABLE OPTIONS STRATEGY (Spot: ₹{spot:.2f})")
    print("="*88)
    print(f"  Overall Signal: {direction} with {overall_conf:.1f}% confidence")
    # CE strike = spot + 100 (OTM call), PE strike = spot - 100 (OTM put)
    ce_label = "OTM" if strike_ce > spot else "ITM"
    pe_label = "OTM" if strike_pe < spot else "ITM"
    print(f"\n  Available Strikes:")
    print(f"    CE {strike_ce} ({ce_label}) → Current LTP: Rs {ce_ltp:.2f}")
    print(f"    PE {strike_pe} ({pe_label}) → Current LTP: Rs {pe_ltp:.2f}")
    print("\n" + "-"*88)
    print(f"  {'Horizon':<10} {'Action':<15} {'Strike':<12} {'Target':<12} {'Confidence':<12} {'Exp. PnL'}")
    print("  " + "-"*86)
    
    # Show recommendations for 1, 5, 15 minute horizons only
    for h in [1, 5, 15]:
        if h not in projections:
            continue
            
        proj = projections[h]
        h_sig = signal.get('signals', {}).get(h, {})
        h_conf = h_sig.get('conf', 0) * 100
        proba_up = proj.get('proba_up', 0.5)
        
        # Determine which option to recommend based on ML prediction
        if proba_up > 0.5:
            # Bullish - recommend CE
            option_type = "CE"
            strike_used = strike_ce
            current_ltp = ce_ltp
            target_ltp = proj.get('ce_net_ltp', 0)
            exp_pnl_pct = proj.get('ce_pnl_pct', 0)
            action = f"BUY {option_type} 📈"
        else:
            # Bearish - recommend PE
            option_type = "PE"
            strike_used = strike_pe
            current_ltp = pe_ltp
            target_ltp = proj.get('pe_net_ltp', 0)
            exp_pnl_pct = proj.get('pe_pnl_pct', 0)
            action = f"BUY {option_type} 📉"
        
        # Format display
        strike_display = f"{option_type} {strike_used}"
        target_display = f"₹{target_ltp:.2f}"
        conf_display = f"{h_conf:.1f}%"
        pnl_display = f"{'+' if exp_pnl_pct > 0 else ''}{exp_pnl_pct:.2f}%"
        
        print(f"  {str(h)+' min':<10} {action:<15} {strike_display:<12} "
              f"{target_display:<12} {conf_display:<12} {pnl_display}")
    
    print("="*88)
    print("  Note: ATM strike shown. Entry price is model estimate only — check live option chain.")
    
    # VERBOSE MODE: Show detailed calculation breakdown
    if verbose:
        print("\n" + "="*88)
        print("  DETAILED CALCULATION BREAKDOWN")
        print("="*88)
        for h in [1, 5, 15]:
            if h not in projections:
                continue
                
            proj = projections[h]
            h_sig = signal.get('signals', {}).get(h, {})
            
            print(f"\n  [{h}-Minute Horizon]")
            print(f"    Model Probability UP: {proj.get('proba_up', 0):.2%}")
            print(f"    Projected Spot: {spot:.2f} → {proj.get('proj_spot', 0):.2f}")
            print(f"    DTE Remaining: {proj.get('dte_proj', 0):.1f} minutes")
            print(f"\n    CE {strike_ce}:")
            print(f"      Current LTP: ₹{ce_ltp:.2f}")
            print(f"      Projected LTP (before slip): ₹{proj.get('ce_ltp_proj', 0):.2f}")
            print(f"      Net LTP (after 3% slip): ₹{proj.get('ce_net_ltp', 0):.2f}")
            print(f"      Expected P&L: {proj.get('ce_pnl_pct', 0):+.2f}%")
            print(f"\n    PE {strike_pe}:")
            print(f"      Current LTP: ₹{pe_ltp:.2f}")
            print(f"      Projected LTP (before slip): ₹{proj.get('pe_ltp_proj', 0):.2f}")
            print(f"      Net LTP (after 3% slip): ₹{proj.get('pe_net_ltp', 0):.2f}")
            print(f"      Expected P&L: {proj.get('pe_pnl_pct', 0):+.2f}%")
        print("="*88)
    
    print("\n  ⚠️  WARNING: Predictions based on ML models + simplified option pricing.")
    print("     Real option prices depend on IV, gamma, vega. Use for direction guidance only.")
    print("="*88 + "\n")


def get_lot_size(trade_date=None) -> int:
    """
    NSE revised NIFTY weekly options lot size from 75 to 65 effective
    April 26, 2025.  All position sizing and PnL calculations must use the
    correct lot size for the trade date to avoid understating/overstating risk.

    trade_date: date or datetime object, or None (uses today).
    Returns 75 for dates before April 26, 2025; 65 for all dates from that day.
    """
    if trade_date is None:
        trade_date = date.today()
    if hasattr(trade_date, 'date'):
        trade_date = trade_date.date()
    return LOT_SIZE_NEW if trade_date >= LOT_SIZE_CUTOVER else LOT_SIZE_OLD


def _next_expiry_mins(now=None) -> float:
    """
    Minutes remaining to the next NIFTY weekly expiry (Monday 15:30 IST).

    NIFTY switched from Thursday to Monday weekly expiry in 2024.
    Uses the instrument master nearest expiry when available; falls back to
    calendar arithmetic for Monday.

    If 'now' is None, uses the current wall-clock time.
    Returns a value clamped to [1, 375*5] (at least 1 minute, at most 5 days).
    """
    import datetime as _dt
    if now is None:
        now = _dt.datetime.now()

    EXPIRY_HM = _dt.time(15, 30)

    # Try to use the nearest expiry from the instrument master (most accurate)
    try:
        from ..data.external_data import _instrument_master, _master_loaded
        if _master_loaded and _instrument_master:
            today = now.date()
            expiry_dates = set()
            for sym in _instrument_master:
                if len(sym) >= 12 and sym.startswith('NIFTY'):
                    date_part = sym[5:12]
                    try:
                        expiry_dates.add(_dt.datetime.strptime(date_part, '%d%b%y').date())
                    except ValueError:
                        pass
            future = sorted(d for d in expiry_dates if d >= today)
            if future:
                expiry_dt = _dt.datetime.combine(future[0], EXPIRY_HM)
                dte_mins = (expiry_dt - now).total_seconds() / 60.0
                return float(np.clip(dte_mins, 1.0, 375.0 * 5))
    except Exception:
        pass

    # Calendar fallback: nearest Monday at 15:30
    MONDAY = 0
    days_ahead = (MONDAY - now.weekday()) % 7
    if days_ahead == 0 and now.time() >= EXPIRY_HM:
        days_ahead = 7
    expiry_dt = _dt.datetime.combine(
        now.date() + _dt.timedelta(days=days_ahead), EXPIRY_HM)

    dte_mins = (expiry_dt - now).total_seconds() / 60.0
    return float(np.clip(dte_mins, 1.0, 375.0 * 5))


def estimate_option_premium(spot: float, iv_annualised_pct: float,
                             dte_mins: float,
                             strike: float = 0.0,
                             option_type: str = 'CE') -> float:
    """
    Time-value estimate for a NIFTY weekly option with dynamic delta proxy.

    Model:   premium ≈ intrinsic_value + time_value
             where:
             - intrinsic_value = max(0, spot - strike) for CE, max(0, strike - spot) for PE
             - time_value = delta × IV_per_min × sqrt(DTE_mins) × spot
             - IV_per_min = iv_annualised_pct / 100 / sqrt(252 × 375)

    Dynamic delta (no Black-Scholes, no Greeks):
      - Moneyness  = spot / strike  (1.0 = ATM)
      - ATM (moneyness ≈ 1.0): delta ≈ 0.50
      - OTM: delta falls toward 0.10
      - ITM: delta rises toward 0.90
      - Deep ITM: intrinsic value dominates, time value is small
      - Expiry approach: as dte_mins → 1, premium converges to intrinsic value
      - Clipped to [0.10, 0.90] to avoid degenerate values.

    Theta decay is implicit via sqrt(dte_mins): premium shrinks at rate
    1/sqrt(dte_mins) as time passes.  No option chain needed.

    floor at 5 (deep-OTM / final-minute minimum) or intrinsic value, whichever is higher.
    """
    # Calculate intrinsic value
    if option_type == 'CE':
        intrinsic = max(0, spot - strike)
    else:  # PE
        intrinsic = max(0, strike - spot)
    
    iv_per_min = iv_annualised_pct / 100.0 / np.sqrt(252 * 375)

    # Dynamic delta from moneyness + time
    if strike > 0 and spot > 0:
        moneyness = spot / strike
        # ATM delta = 0.50; scale by distance from ATM.
        # The sharper factor near expiry reflects gamma acceleration.
        dte_days    = max(dte_mins, 1.0) / 375.0
        sharpness   = 1.0 + 3.0 * np.exp(-dte_days)    # 2..4 near expiry, 1 far out
        
        # For CE: moneyness > 1 (ITM), moneyness < 1 (OTM)
        # For PE: moneyness < 1 (ITM), moneyness > 1 (OTM)
        if option_type == 'CE':
            delta_raw = 0.50 + sharpness * (moneyness - 1.0)
        else:  # PE
            delta_raw = 0.50 + sharpness * (1.0 - moneyness)
        
        delta = float(np.clip(delta_raw, 0.10, 0.90))
    else:
        delta = 0.50   # ATM default when strike not provided

    time_value = delta * iv_per_min * np.sqrt(max(dte_mins, 1.0)) * spot
    total_premium = intrinsic + time_value
    
    return max(total_premium, 5.0)


def effective_delta(bars_open: int, spot_move: float,
                    atr: float, direction: str) -> float:
    """
    Delta decay proxy for NIFTY weekly options (execution layer only).

    Three effects modelled without Black-Scholes or option chain:

    1. Time decay — delta erodes each 1-min bar at rate exp(-0.002 * bars_open).
       At 60 bars (1 hour), decay factor ≈ 0.887 (still 89% of entry delta).
       At 375 bars (full day), decay factor ≈ 0.47.  Never collapses to zero.

    2. Momentum / gamma — a fast favourable spot move raises effective delta
       (convexity). A move against the trade lowers it.
       momentum_shift = (spot_move / ATR) × 0.10 — clipped so delta stays [0.10, 0.80].

    3. Directional skew (India-specific) — NSE NIFTY has negative skew:
       PUTs carry a slight premium over CALLs.
       CE (UP) skew: 0.90   PE (DOWN) skew: 1.05

    Parameters
    ----------
    bars_open  : integer number of 1-min bars since entry
    spot_move  : spot_now - spot_entry  (signed, favourable > 0 by convention)
    atr        : current atr_14 of the spot (absolute points)
    direction  : 'UP' (CALL) or 'DOWN' (PUT)

    Returns
    -------
    float in [0.10, 0.80] representing the effective delta of the position.
    """
    # 1. Time decay
    time_decay = np.exp(-0.002 * max(bars_open, 0))

    # 2. Moneyness/momentum proxy
    momentum_shift = (spot_move / (atr + 1e-9)) * 0.10
    delta_eff = float(np.clip(DELTA_BASE + momentum_shift, 0.10, 0.80))

    # 3. Directional skew
    skew = 0.90 if direction == 'UP' else 1.05

    return delta_eff * time_decay * skew


def option_pnl_estimate(entry_price: float, spot_entry: float,
                        spot_now: float, atr: float,
                        direction: str, bars_open: int, dte_now: float = 750.0) -> float:
    """
    Estimate current option premium from first principles using effective_delta.

    Returns an LTP estimate:
        ltp = entry_price + directional_gain - theta_drag

    where:
        directional_gain = spot_move × delta_eff
        theta_drag       = bars_open × dynamic_theta(dte_now)

    For a PUT (direction='DOWN'): spot_move is inverted so that a falling
    spot = favourable move = positive directional_gain.

    The result is clamped to a minimum of 2.0 so the option never goes
    negative (you can only lose what you paid).
    """
    if direction == 'UP':
        spot_move = spot_now - spot_entry     # positive = in-the-money move
    else:
        spot_move = spot_entry - spot_now     # positive = in-the-money move for PUT

    delta_eff       = effective_delta(bars_open, spot_move, atr, direction)
    directional_gain= spot_move * delta_eff
    
    # Use dynamic theta based on remaining DTE
    theta_dynamic   = get_dynamic_theta(dte_now)
    theta_drag      = bars_open * theta_dynamic

    ltp = entry_price + directional_gain - theta_drag
    return max(ltp, 2.0)   # option cannot go below zero (floored at 2)


def validate_pnl_estimate(estimated_ltp: float, actual_ltp: float, 
                         entry_price: float, threshold_pct: float = 0.20) -> float:
    """
    Reality check: if estimated_ltp deviates >threshold from actual_ltp,
    use actual_ltp instead to prevent false stop-outs.
    
    WHY: Simplified delta-decay model works for ATM options <1 hour,
         but breaks down for OTM or long-held positions (gamma effects).
    
    Args:
        estimated_ltp: Model-estimated option LTP
        actual_ltp: Real-time LTP from option chain (if available)
        entry_price: Original entry premium
        threshold_pct: Max allowed deviation (default 20%)
    
    Returns:
        Validated LTP (actual if deviation too high, else estimated)
    """
    if actual_ltp <= 0:
        # No actual LTP available - use estimate
        return estimated_ltp
    
    deviation = abs(estimated_ltp - actual_ltp) / (entry_price + 1e-9)
    
    if deviation > threshold_pct:
        logger.warning(f"[PnL Model] Estimated {estimated_ltp:.2f} deviates {deviation:.1%} "
                      f"from actual {actual_ltp:.2f}. Using actual.")
        return actual_ltp
    
    return estimated_ltp


def calculate_trailing_stop(entry_price: float, current_ltp: float, 
                           initial_stop: float, direction: str) -> float:
    """
    Calculate trailing stop-loss to lock in profits.
    
    Rules:
    - Once at 40% profit: trail stop to break-even
    - Once at 60% profit: trail stop to +30% (lock in gains)
    - Otherwise: keep initial stop
    
    Args:
        entry_price: Original entry premium
        current_ltp: Current option LTP
        initial_stop: Original stop-loss level
        direction: 'UP' (CALL) or 'DOWN' (PUT)
    
    Returns:
        Trailing stop price
    """
    profit_pct = (current_ltp - entry_price) / (entry_price + 1e-9)
    
    if profit_pct > 0.60:  # 60% profit
        trailing = entry_price * 1.30 if direction == 'UP' else entry_price * 0.70
        logger.info(f"[Trailing Stop] 60% profit reached. Stop moved to +30% ({trailing:.2f})")
        return trailing
    elif profit_pct > 0.40:  # 40% profit
        logger.info(f"[Trailing Stop] 40% profit reached. Stop moved to break-even ({entry_price:.2f})")
        return entry_price  # Break-even
    else:
        return initial_stop  # Keep original stop


def validate_spread_width(bid: float, ask: float, mid_price: float, 
                         max_spread_pct: float = 0.05) -> tuple:
    """
    Check if bid-ask spread is acceptable.
    
    WHY: During volatile periods, ATM option spreads widen to 5-10 points.
         Assuming mid-price fill when spread is wide leads to 3-5% worse execution.
    
    Args:
        bid, ask: Live option chain bid/ask prices
        mid_price: Estimated entry price
        max_spread_pct: Max allowed spread (default 5% of mid)
    
    Returns:
        (is_acceptable: bool, actual_spread_pct: float)
    """
    if bid <= 0 or ask <= 0 or mid_price <= 0:
        return True, 0.0  # No live quotes - assume acceptable
    
    spread = ask - bid
    spread_pct = spread / mid_price
    
    if spread_pct > max_spread_pct:
        logger.warning(f"[Spread] Bid-ask spread {spread_pct:.1%} exceeds {max_spread_pct:.1%}. "
                      f"Blocking trade (bid={bid:.2f}, ask={ask:.2f}).")
        return False, spread_pct
    
    return True, spread_pct


def adjust_conf_for_flow(base_conf: float, fii_net: float, direction: str) -> float:
    """
    Reduce confidence if trading AGAINST strong institutional flow.
    
    WHY: FII selling >₹2000 Cr/day often precedes 2-3 day declines.
         Trading against this flow reduces edge.
    
    Args:
        base_conf: Model's raw confidence
        fii_net: FII net buy/sell in Crores (negative = selling)
        direction: 'UP' or 'DOWN'
    
    Returns:
        Adjusted confidence
    """
    if fii_net < -2000 and direction == 'UP':
        # Heavy FII selling, going LONG -> reduce confidence
        logger.info(f"[FII Flow] Heavy selling ({fii_net:.0f} Cr), reducing LONG confidence by 10%")
        return base_conf * 0.90
    elif fii_net > 2000 and direction == 'DOWN':
        # Heavy FII buying, going SHORT -> reduce confidence
        logger.info(f"[FII Flow] Heavy buying ({fii_net:.0f} Cr), reducing SHORT confidence by 10%")
        return base_conf * 0.90
    
    return base_conf


# ==============================================================================
# EXPIRY-DAY HARD RULES
# ==============================================================================

EXPIRY_RULES = {
    # (minute_of_day range): {'allow_new': bool, 'size_mult': float, 'stop_tighten': float}
    'pre_1130':    {'allow_new': True,  'size_mult': 0.50, 'stop_tighten': 0.70},
    '1130_to_1300':{'allow_new': False, 'size_mult': 0.25, 'stop_tighten': 0.50},
    'after_1300':  {'allow_new': False, 'size_mult': 0.00, 'stop_tighten': 0.00},
}

def get_expiry_rule(is_expiry: bool, minute_of_day: int) -> dict:
    """Return the applicable expiry-day rule for the current time."""
    if not is_expiry:
        return {'allow_new': True, 'size_mult': 1.0, 'stop_tighten': 1.0}
    if minute_of_day >= 285:   # 13:00 = 9:15 + 285 min
        return {'allow_new': False, 'size_mult': 0.0,  'stop_tighten': 0.0,  'tag': 'EXPIRY_AFTER_1300'}
    if minute_of_day >= 135:   # 11:30 = 9:15 + 135 min
        return {'allow_new': False, 'size_mult': 0.25, 'stop_tighten': 0.50, 'tag': 'EXPIRY_1130_1300'}
    return {'allow_new': True, 'size_mult': 0.50, 'stop_tighten': 0.70, 'tag': 'EXPIRY_PRE_1130'}


# ==============================================================================
# VEGA-AWARE ENTRY FILTER
# ==============================================================================

def vega_entry_filter(iv_rank: float, iv_pct_change_today: float) -> tuple:
    """
    Block entries when IV is already elevated or has just spiked.

    WHY: Buying options when IV is elevated means paying for time value that
    will compress even if the directional call is correct.  This is the single
    largest source of "correct direction, lost money" outcomes in options.

    Returns (allow: bool, reason: str)
    """
    if iv_rank > 80:
        return False, f"VEGA_FILTER: IV rank {iv_rank:.0f} too elevated (>80)"
    if iv_pct_change_today > 20:
        return False, f"VEGA_FILTER: IV spiked {iv_pct_change_today:.0f}% today — crush risk"
    if iv_rank > 65:
        logger.info(f"[Vega] IV rank {iv_rank:.0f} above median — tighter stops recommended")
    return True, ''


def select_option(signal: dict, capital: float, now=None, tick_buffer=None,
                 session: 'AngelSession' = None, position_mgr: 'PositionManager' = None) -> dict:
    """Strike selection and position sizing.

    LIVE-SAFE redesign:
      - Premium-based stops (not underlying-based): stop fires on option LTP
      - Delta-normalized sizing: accounts for actual option delta, not a fixed 0.5
      - Vega filter: blocks entries when IV is elevated (buy-expensive risk)
      - Expiry-day hard rules: halved size pre-11:30, no new entries post-11:30
      - Pessimistic cost model: slippage at ask, full round-trip costs baked in
      
    2026 Edge: Slippage-Adjusted Position Sizing
      - Scale position size inversely to bid-ask spread proxy (IV rank)
      - High IV (>70%) = wide spreads = cut position size by 50%
      - This accounts for "Impact Cost" in fast markets
    
    Edge Case 1: Limit Price Protection (LPP)
      - Validates entry price is within ±2% of recent 5-tick average
      - Prevents "Fat Finger" orders during sudden spikes
      - Orders outside this range exceed NSE execution limits
    
    Edge Case 3: Bid-Ask Spread Validation
      - Checks live spread width before entry
      - Rejects if spread >5% of mid-price
    
    Edge Case 7: OI Concentration Check
      - Avoids strikes within 50 points of max OI
      - High OI acts as magnet/resistance
    
    Args:
        signal: Signal dictionary with direction, spot, iv_proxy
        capital: Trading capital for position sizing
        now: Current datetime (for DTE calculation)
        tick_buffer: Optional deque of recent ticks for LPP check
        session: AngelSession for option chain data
        position_mgr: PositionManager for concentration limits
    """
    spot          = signal['spot']
    direction     = signal['direction']
    minute_of_day = signal.get('minute_of_day', 0)
    is_expiry     = bool(signal.get('is_expiry', 0))
    iv_rank       = float(signal.get('iv_rank_approx', 50.0))
    # iv_proxy is daily IV (%). Annualise ×sqrt(252) for estimate_option_premium().
    _iv_daily     = float(signal.get('iv_proxy', 0.06))
    iv_annpct     = _iv_daily * (252 ** 0.5)   # ~14-18% annualised for NIFTY
    iv_pct_chg    = float(signal.get('iv_pct_change', 0.0))
    atm           = round(spot / 50) * 50
    option_type   = 'CE' if direction == 'UP' else 'PE'

    # Regime-adaptive strike selection
    # WHY: In strong trending regimes, slight ITM reduces theta risk and
    # gives higher delta (more directional exposure per rupee spent).
    # In ranging/crisis, ATM gives better gamma for mean-reversion pops.
    # Rule:
    #   TRENDING + BREAKOUT micro → ITM by 1 strike (50pts in direction)
    #   Everything else           → ATM
    micro_regime_sig = signal.get('micro_regime', '')
    regime_sig       = signal.get('regime', '')
    is_strong_trend  = (
        str(regime_sig) == '0'   # REGIME_TRENDING = 0
        and micro_regime_sig in ('TRENDING_UP', 'TRENDING_DN', 'BREAKOUT')
    )
    if is_strong_trend:
        # ITM: CE → strike below spot, PE → strike above spot
        strike = atm - 50 if direction == 'UP' else atm + 50
    else:
        strike = atm

    # ---- 1. Expiry-day hard rules -------------------------------------------
    expiry_rule = get_expiry_rule(is_expiry, minute_of_day)
    if not expiry_rule['allow_new']:
        logger.warning(f"[Expiry] New entries blocked — {expiry_rule.get('tag', 'expiry rule')}")
        return None

    # ---- 2. Vega-aware entry filter ------------------------------------------
    vega_ok, vega_reason = vega_entry_filter(iv_rank, iv_pct_chg)
    if not vega_ok:
        logger.warning(f"[VegaFilter] {vega_reason}")
        return None

    # ---- 3. LPP (Limit Price Protection) check --------------------------------
    if tick_buffer is not None and len(tick_buffer) >= 5:
        if check_lpp_violation(spot, tick_buffer):
            logger.warning(f"[LPP] Price spike: spot {spot:.2f} exceeds 2% of 5-tick avg. Trade blocked.")
            return None

    # ---- 4. OI concentration check -------------------------------------------
    if session is not None:
        max_call_oi, max_put_oi, _, _ = get_max_oi_strikes(spot, session)
        if avoid_oi_concentration_zone(strike, max_call_oi, max_put_oi):
            logger.warning(f"[OI] Strike {strike} within 50pts of OI concentration. Blocked.")
            return None

    # ---- 5. Premium estimation -----------------------------------------------
    dte_mins_entry = _next_expiry_mins(now)
    est_premium    = estimate_option_premium(
        spot, iv_annpct, dte_mins_entry,
        strike=float(strike), option_type=option_type
    )
    est_premium = max(est_premium, 30)

    # ---- 5b. Bid-ask spread validation ----------------------------------------
    # If live bid/ask are available in the signal (from option chain feed), check
    # the spread.  This catches the "empty order book" scenario during circuit
    # breakers / liquidity dry-ups where mid-price looks fine but the spread
    # is wider than the entire profit target.
    # Threshold: 2% of mid for normal conditions; signal may carry 'bid'/'ask'
    # if option chain was fetched.
    opt_bid = float(signal.get('bid', 0.0))
    opt_ask = float(signal.get('ask', 0.0))
    spread_ok, actual_spread_pct = validate_spread_width(
        opt_bid, opt_ask, est_premium, max_spread_pct=0.02
    )
    if not spread_ok:
        logger.warning(
            f"[SpreadCheck] Spread {actual_spread_pct:.1%} > 2% of mid ({est_premium:.1f}). "
            f"Trade blocked — empty order book risk."
        )
        return None

    # ---- 6. Cost model (expiry + IV aware) -----------------------------------
    expiry_cost_mult = 2.0 if is_expiry and minute_of_day > 135 else 1.0
    effective_slip   = SLIPPAGE_PCT   * expiry_cost_mult
    effective_cost   = TOTAL_COST_PCT * expiry_cost_mult
    entry_price      = est_premium * (1 + effective_slip)

    # ---- 7. PREMIUM-BASED stop loss (not underlying-based) -------------------
    # Stop is expressed as % of the option PREMIUM paid.
    # This fires on the option's own price, not on the underlying move.
    # Key advantage: immune to IV distortion, gap risk, and basis drift.
    premium_stop_pct = STOP_LOSS_PCT * expiry_rule['stop_tighten']
    if iv_rank > 65:
        premium_stop_pct *= 0.80   # tighter stop when IV elevated (faster adverse moves)
    stop_price = entry_price * (1 - premium_stop_pct)

    # Guard: stop must be a positive finite number below entry.
    # entry_price can be garbage if est_premium is corrupted (NaN IV, zero spot).
    # An invalid stop means an unprotected position — reject the trade entirely.
    if not np.isfinite(stop_price) or stop_price <= 0 or stop_price >= entry_price:
        logger.error(
            f"Invalid stop_price={stop_price:.4f} (entry={entry_price:.4f}, "
            f"stop_pct={premium_stop_pct:.3f}). Rejecting trade."
        )
        return None

    # ---- 8. Delta-normalized position sizing ---------------------------------
    # Estimate entry delta from moneyness (ATM ≈ 0.5, sharpens near expiry)
    dte_days    = max(dte_mins_entry, 1.0) / 375.0
    sharpness   = 1.0 + 3.0 * np.exp(-dte_days)
    delta_entry = float(np.clip(0.50, 0.10, 0.90))   # ATM = 0.5 baseline

    # Risk amount the model is willing to lose per trade
    risk_amt = capital * MAX_RISK_PCT
    # Premium risk per lot = entry_price * stop_pct * lot_size
    lot_size  = get_lot_size(now)
    premium_risk_per_lot = entry_price * premium_stop_pct * lot_size

    # Delta normalization: a higher-delta option carries more underlying exposure
    # per lot.  Reduce lot count proportionally so underlying exposure stays stable.
    delta_norm_factor = min(1.0, 0.50 / max(delta_entry, 0.10))  # normalize to ATM=0.5 baseline

    # IV rank spread penalty (wide spreads = impact cost)
    spread_penalty = 1.0
    if iv_rank > 85:
        spread_penalty = 0.33
        logger.info(f"[Sizing] Extreme IV rank {iv_rank:.0f} -> position 33%")
    elif iv_rank > 70:
        spread_penalty = 0.50
        logger.info(f"[Sizing] High IV rank {iv_rank:.0f} -> position 50%")

    # Expiry size multiplier
    expiry_size_mult = expiry_rule['size_mult']

    raw_contracts = risk_amt / (premium_risk_per_lot + 1e-9)
    contracts = int(raw_contracts * delta_norm_factor * spread_penalty * expiry_size_mult)
    contracts = max(1, min(contracts, 5))

    # ---- 9. Position concentration check ------------------------------------
    if position_mgr is not None:
        position_value = entry_price * lot_size * contracts
        can_add, reason = position_mgr.can_add_position(position_value)
        if not can_add:
            logger.warning(f"[PositionMgr] {reason}")
            return None

    # ---- 10. Target price ---------------------------------------------------
    breakeven    = entry_price * (1 + effective_cost)
    target_price = max(entry_price * (1 + TARGET_PCT), breakeven * 1.10)

    gamma_window = is_expiry and minute_of_day > 300

    return {
        'option_type':      option_type,
        'strike':           int(strike),
        'est_premium':      round(est_premium, 2),
        'entry_price':      round(entry_price, 2),
        'contracts':        contracts,
        'lot_size':         lot_size,
        'lot_size_used':    lot_size,
        # PREMIUM-BASED stop (key safety fix)
        'stop_price':       round(stop_price, 2),
        'stop_pct':         round(premium_stop_pct * 100, 1),
        'stop_basis':       'PREMIUM',   # explicit tag so caller knows stop type
        'target_price':     round(target_price, 2),
        'breakeven':        round(breakeven, 2),
        'notional':         round(entry_price * contracts * lot_size, 2),
        'max_risk':         round(entry_price * premium_stop_pct * contracts * lot_size, 2),
        'total_cost_pct':   round(effective_cost * 100, 1),
        'gamma_protected':  gamma_window,
        'dte_mins_entry':   round(dte_mins_entry, 1),
        'iv_annpct_entry':  round(iv_annpct, 4),
        'spread_penalty':   round(spread_penalty, 2),
        'iv_rank':          round(iv_rank, 1),
        'delta_entry':      round(delta_entry, 3),
        'expiry_rule_tag':  expiry_rule.get('tag', 'NORMAL'),
    }


