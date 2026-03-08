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
    LIMIT_BUY_BUFFER_PCT, LIMIT_SELL_BUFFER_PCT,
    LIMIT_FILL_PROB_ATM, LIMIT_FILL_PROB_EXPIRY, LIMIT_FILL_PROB_HIGH_IV,
    LIMIT_SPREAD_NORMAL_PCT, LIMIT_SPREAD_HIGH_IV_PCT,
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
    Trading minutes remaining to the next NIFTY weekly expiry (Tuesday 15:30 IST).

    Returns TRADING minutes (not calendar minutes): counts only market hours
    (09:15-15:30, Mon-Fri), 375 mins/day. This is what option pricing models
    need — theta decays only during market hours, not overnight or on weekends.

    NIFTY switched from Thursday to Tuesday weekly expiry in Sep 2024.
    Uses the instrument master nearest expiry when available; falls back to
    calendar arithmetic for Tuesday.

    If 'now' is None, uses the current wall-clock time.
    Returns a value clamped to [1, 375*5] (at least 1 minute, at most 5 days).
    """
    import datetime as _dt

    if now is None:
        now = _dt.datetime.now()

    # Strip timezone if present
    if hasattr(now, 'tzinfo') and now.tzinfo is not None:
        now = now.replace(tzinfo=None)

    OPEN_HM  = _dt.time(9, 15)
    CLOSE_HM = _dt.time(15, 30)

    def _trading_mins_remaining_today(dt):
        """Minutes left in today's session from dt (0 if outside session)."""
        t = dt.time()
        if t >= CLOSE_HM:
            return 0.0
        open_mins  = OPEN_HM.hour * 60 + OPEN_HM.minute
        close_mins = CLOSE_HM.hour * 60 + CLOSE_HM.minute
        cur_mins   = t.hour * 60 + t.minute
        return max(0.0, min(close_mins, close_mins) - max(open_mins, cur_mins))

    # Determine expiry date
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
                expiry_date = future[0]
                # Fall through to trading-mins calculation below
                pass
            else:
                expiry_date = None
        else:
            expiry_date = None
    except Exception:
        expiry_date = None

    if expiry_date is None:
        # Calendar fallback: nearest Tuesday
        TUESDAY = 1
        days_ahead = (TUESDAY - now.weekday()) % 7
        if days_ahead == 0 and now.time() >= CLOSE_HM:
            days_ahead = 7
        expiry_date = now.date() + _dt.timedelta(days=days_ahead)

    # Count trading minutes from now to expiry close
    # = remaining mins today + 375 × full trading days between now and expiry
    total_trading_mins = 0.0
    current_date = now.date()

    if current_date == expiry_date:
        # Same day as expiry
        total_trading_mins = _trading_mins_remaining_today(now)
    elif current_date < expiry_date:
        # Remaining minutes today
        total_trading_mins += _trading_mins_remaining_today(now)
        # Full trading days between tomorrow and expiry (Mon-Fri only)
        d = current_date + _dt.timedelta(days=1)
        while d < expiry_date:
            if d.weekday() < 5:   # Mon-Fri
                total_trading_mins += 375.0
            d += _dt.timedelta(days=1)
        # Full session on expiry day (09:15 to 15:30)
        if expiry_date.weekday() < 5:
            total_trading_mins += 375.0

    return float(np.clip(total_trading_mins, 1.0, 375.0 * 5))


def estimate_option_premium(spot: float, iv_annualised_pct: float,
                             dte_mins: float,
                             strike: float = 0.0,
                             option_type: str = 'CE') -> float:
    """
    Black-Scholes option pricing for NIFTY weekly options.

    Uses proper BS formula (scipy.stats.norm) which gives the correct
    ATM coefficient (~0.40) vs the old delta-proxy formula that used
    0.50, causing a systematic 20-25% overestimate on all option prices.

    Parameters
    ----------
    spot              : current NIFTY spot price
    iv_annualised_pct : implied volatility in % (e.g. 14.5 for 14.5%)
    dte_mins          : minutes to expiry (clamped to [1, 375*5])
    strike            : option strike (0 → ATM = round(spot/50)*50)
    option_type       : 'CE' (call) or 'PE' (put)

    Returns
    -------
    float : estimated option premium, floored at 2.0
    """
    from scipy.stats import norm as _norm

    if strike <= 0:
        strike = round(spot / 50) * 50

    T = max(dte_mins, 1.0) / (252.0 * 375.0)   # fraction of a trading year
    sigma = max(iv_annualised_pct, 0.1) / 100.0  # decimal annual vol
    r = 0.065                                     # India risk-free rate (~repo rate)

    # Intrinsic value at expiry (fallback for near-zero T)
    if T < 1e-6:
        if option_type == 'CE':
            return max(spot - strike, 2.0)
        else:
            return max(strike - spot, 2.0)

    d1 = (np.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'CE':
        price = spot * _norm.cdf(d1) - strike * np.exp(-r * T) * _norm.cdf(d2)
    else:
        price = strike * np.exp(-r * T) * _norm.cdf(-d2) - spot * _norm.cdf(-d1)

    return max(float(price), 2.0)


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
    if minute_of_day >= 225:   # 13:00 = 9:15 + 225 min  → no new entries after 13:00
        return {'allow_new': False, 'size_mult': 0.0,  'stop_tighten': 0.0,  'tag': 'EXPIRY_AFTER_1300'}
    if minute_of_day >= 135:   # 11:30 = 9:15 + 135 min  → block 11:30–13:00
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


def simulate_limit_order(mid_price: float, side: str,
                         is_expiry: bool = False,
                         iv_rank: float = 50.0,
                         rng: np.random.Generator = None) -> dict:
    """
    Simulate a LIMIT order fill — mirrors SEBI April 2026 rules (no MARKET orders).

    In live trading you place a LIMIT order at a buffer above/below LTP.
    This function simulates:
      1. The limit price you'd submit
      2. Whether the order fills (fill probability based on market conditions)
      3. The actual fill price (limit price = worst case; often fills at mid)
      4. The bid-ask spread cost paid

    Parameters
    ----------
    mid_price : float
        Current mid-price estimate of the option (from estimate_option_premium)
    side : str
        'BUY' (entry) or 'SELL' (exit)
    is_expiry : bool
        Whether today is expiry day (lower fill prob, wider spreads)
    iv_rank : float
        Current IV rank 0-100 (higher = wider spreads, lower fill prob)
    rng : np.random.Generator
        Optional seeded RNG for reproducibility in backtests

    Returns
    -------
    dict with keys:
        filled        : bool   — whether the order filled
        fill_price    : float  — actual execution price (0 if not filled)
        limit_price   : float  — the limit price submitted
        spread_cost   : float  — extra cost vs mid due to spread (points)
        fill_prob     : float  — probability used for this fill attempt
        slip_pct      : float  — total slippage as % of mid_price
        order_type    : str    — always 'LIMIT' (never MARKET)
    """
    if rng is None:
        rng = np.random.default_rng()

    high_iv = iv_rank > 70

    # 1. Determine fill probability
    if is_expiry:
        fill_prob = LIMIT_FILL_PROB_EXPIRY
    elif high_iv:
        fill_prob = LIMIT_FILL_PROB_HIGH_IV
    else:
        fill_prob = LIMIT_FILL_PROB_ATM

    # 2. Spread cost component (extra above/below mid due to bid-ask)
    spread_pct = LIMIT_SPREAD_HIGH_IV_PCT if high_iv else LIMIT_SPREAD_NORMAL_PCT

    # 3. Compute limit price
    if side == 'BUY':
        # Submit limit at mid + buy_buffer (willing to pay up to this)
        limit_price = mid_price * (1.0 + LIMIT_BUY_BUFFER_PCT + spread_pct)
    else:  # SELL
        # Submit limit at mid - sell_buffer (willing to accept down to this)
        limit_price = mid_price * (1.0 - LIMIT_SELL_BUFFER_PCT - spread_pct)

    limit_price = round(max(limit_price, 1.0), 2)

    # 4. Simulate fill
    filled = rng.random() < fill_prob

    if not filled:
        return {
            'filled':     False,
            'fill_price': 0.0,
            'limit_price': limit_price,
            'spread_cost': 0.0,
            'fill_prob':  fill_prob,
            'slip_pct':   0.0,
            'order_type': 'LIMIT',
        }

    # 5. Fill price: typically between mid and limit (random within that band)
    # In liquid ATM options, fills often come at mid or very close to it.
    # We sample uniformly between mid and limit to avoid over-pessimism.
    if side == 'BUY':
        fill_price = round(rng.uniform(mid_price, limit_price), 2)
    else:
        fill_price = round(rng.uniform(limit_price, mid_price), 2)

    spread_cost = abs(fill_price - mid_price)
    slip_pct    = spread_cost / (mid_price + 1e-9)

    return {
        'filled':      True,
        'fill_price':  fill_price,
        'limit_price': limit_price,
        'spread_cost': round(spread_cost, 2),
        'fill_prob':   fill_prob,
        'slip_pct':    round(slip_pct, 4),
        'order_type':  'LIMIT',
    }


def select_option(signal: dict, capital: float, now=None, tick_buffer=None,
                 session: 'AngelSession' = None, position_mgr: 'PositionManager' = None,
                 rng: np.random.Generator = None) -> dict:
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
    # In live mode, Angel One injects the real market LTP into the signal under
    # the key 'ce_ltp_api' / 'pe_ltp_api' (set only when fetched from the live
    # option chain — NOT the BS estimate injected by replay/signal_generator).
    # 'ce_ltp_current' / 'pe_ltp_current' in the signal are rolling-ATM BS
    # estimates and may be at a different strike than the one selected here.
    # Always recompute at the exact selected strike using BS.
    # Live mode overrides this by passing 'ce_ltp_api' / 'pe_ltp_api'.
    dte_mins_entry = _next_expiry_mins(now)
    api_ltp_key = 'ce_ltp_api' if option_type == 'CE' else 'pe_ltp_api'
    api_ltp     = float(signal.get(api_ltp_key, 0.0))
    if api_ltp > 1.0:
        est_premium = api_ltp   # real Angel One market price — use as-is
    else:
        est_premium = estimate_option_premium(
            spot, iv_annpct, dte_mins_entry,
            strike=float(strike), option_type=option_type
        )
    est_premium = max(est_premium, 5)

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

    # ---- 6. LIMIT order simulation (replaces fixed slippage model) -----------
    # SEBI April 2026: MARKET orders banned for algos. Paper trading now
    # simulates a LIMIT BUY at LTP + buffer, with realistic fill probability.
    expiry_cost_mult = 2.0 if is_expiry and minute_of_day > 135 else 1.0
    limit_sim = simulate_limit_order(
        mid_price  = est_premium,
        side       = 'BUY',
        is_expiry  = is_expiry,
        iv_rank    = iv_rank,
        rng        = rng,
    )
    if not limit_sim['filled']:
        logger.info(
            f"[LimitOrder] ENTRY unfilled — fill_prob={limit_sim['fill_prob']:.0%} "
            f"limit={limit_sim['limit_price']:.2f}  iv_rank={iv_rank:.0f}"
        )
        return None   # Missed fill — same behaviour as a real unexecuted LIMIT order

    entry_price   = limit_sim['fill_price']
    effective_cost = COST_RT_PCT * expiry_cost_mult   # brokerage + taxes only (no slippage — captured above)

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
    moneyness   = spot / float(strike) if strike > 0 else 1.0
    if direction == 'UP':   # CE: deeper ITM = higher delta
        delta_raw = 0.50 + sharpness * (moneyness - 1.0)
    else:                   # PE: deeper ITM (strike > spot) = higher delta
        delta_raw = 0.50 + sharpness * (1.0 - moneyness)
    delta_entry = float(np.clip(delta_raw, 0.10, 0.90))

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
        'stop_basis':       'PREMIUM',
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
        # LIMIT order execution details (for paper trade analysis)
        'limit_price':      round(limit_sim['limit_price'], 2),
        'limit_slip_pct':   round(limit_sim['slip_pct'] * 100, 3),
        'limit_spread_cost':round(limit_sim['spread_cost'], 2),
        'limit_fill_prob':  round(limit_sim['fill_prob'], 2),
        'order_type':       'LIMIT',
    }


