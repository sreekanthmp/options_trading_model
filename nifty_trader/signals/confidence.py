"""Signal confidence gates: directional agreement, EV calculation, IV crush check."""
import time, logging
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from ..config import (
    HORIZONS, HORIZON_WEIGHTS,
    CONF_BY_HORIZON, EV_SAFETY_NORMAL, EV_SAFETY_ELEVATED,
    TOTAL_COST_PCT, TB_PUT_TIME_DECAY_EXTRA,
)
from ..execution.costs import effective_cost

logger = logging.getLogger(__name__)

# ==============================================================================
# 3. DIRECTIONAL AGREEMENT GATE (2026 LIVE SURVIVABILITY)
# ==============================================================================

def check_directional_agreement(signals: dict) -> tuple[bool, str]:
    """
    WHY: Multi-horizon models can conflict (5m says UP, 15m says DOWN).
         Trading on conflicting signals leads to whipsaws and losses.
         This gate ensures horizons agree before entry.
    
    Rules:
        1. 5-minute horizon is PRIMARY direction
        2. 15-minute horizon must AGREE or be NEUTRAL
        3. 1-minute horizon is TIMING only (ignored for direction)
        4. 30-minute horizon affects POSITION SIZE only (ignored for direction)
        5. Reject if |conf_5m - conf_15m| > 0.25 (confidence gap too wide)
    
    Args:
        signals: dict of {horizon: {'pred': 0/1, 'conf': float}}
        
    Returns:
        (pass: bool, reason: str)
    """
    # Extract horizon signals
    sig_5m = signals.get(5)
    sig_15m = signals.get(15)
    
    if not sig_5m:
        return False, "5m_missing"
    
    # 5m is primary direction
    primary_dir = sig_5m['pred']  # 1=UP, 0=DOWN
    primary_conf = sig_5m['conf']
    
    # If 15m exists, check agreement
    if sig_15m:
        # 15m must agree OR be neutral (conf < 0.55)
        if sig_15m['conf'] >= 0.55:  # 15m has conviction
            if sig_15m['pred'] != primary_dir:
                return False, "5m_15m_conflict"
        
        # Confidence gap check.
        # Threshold 0.30 (relaxed from 0.25) because in fast NIFTY breakouts
        # the 5m model reacts 1-2 bars before the 15m model catches up.
        # A gap of 0.25-0.30 during a genuine breakout is normal lag, not
        # disagreement — hard-rejecting it filters the most profitable moves.
        # Above 0.30 the disagreement is wide enough to be genuine conflict.
        conf_gap = abs(primary_conf - sig_15m['conf'])
        if conf_gap > 0.30:
            return False, "confidence_gap_too_wide"
    
    return True, "directional_agreement_ok"


# ==============================================================================
# 5. FFT CYCLE SAFETY (2026 LIVE SURVIVABILITY)
# ==============================================================================

def compute_fft_regime_hint(fft_cycle_raw: float, smoothed_history: list) -> str:
    """
    WHY: FFT cycle detection is noisy and unreliable for direct confidence
         adjustments. Convert to simple binary regime hint instead.
    
    **IMPORTANT: This function is INTENTIONALLY NOT INTEGRATED.**
    
    The HMM-based RegimeDetector (using vol, atr, trend, rsi, adx) is the
    AUTHORITATIVE regime source. FFT is too noisy to influence regime directly.
    
    This function is provided for future experimentation ONLY:
      - Could be used as SECONDARY confirmation in detect_micro_regime()
      - Could be logged for post-trade analysis
      - Should NEVER override HMM regime
      - Should NEVER be added back to model features
    
    Current status: DISABLED and UNUSED (safe dead code)
    
    Args:
        fft_cycle_raw: Current FFT cycle length (bars)
        smoothed_history: Last 10 FFT cycle values for EWMA smoothing
        
    Returns:
        'CHOP' (cycle < 35), 'TREND' (cycle > 60), or 'NEUTRAL'
    """
    # EWMA smoothing (span=10) to reduce noise
    smoothed_history.append(fft_cycle_raw)
    if len(smoothed_history) > 10:
        smoothed_history.pop(0)
    
    # Simple exponential moving average
    if len(smoothed_history) >= 3:
        alpha = 2.0 / (10 + 1)
        smoothed = smoothed_history[0]
        for val in smoothed_history[1:]:
            smoothed = alpha * val + (1 - alpha) * smoothed
    else:
        smoothed = fft_cycle_raw
    
    # Binary classification
    if smoothed < 35:
        return 'CHOP'      # Choppy, ranging market
    elif smoothed > 60:
        return 'TREND'     # Trending market
    else:
        return 'NEUTRAL'   # Ambiguous, ignore


# ==============================================================================
# 7. ENTRY MICRO-CONFIRMATION (2026 LIVE SURVIVABILITY)
# ==============================================================================

def check_entry_micro_confirmation(row: pd.Series, direction: str, 
                                    vwap_history: list) -> tuple[bool, str]:
    """
    WHY: Models can signal too early (before price confirms direction).
         This final filter prevents premature entries by requiring price
         to demonstrate commitment to the predicted direction.
    
    Rules (need ONE of the following):
        1. Price holds above/below VWAP for 2 consecutive bars
        2. Candle body > 0.3 x ATR(14) (strong directional candle)
    
    Args:
        row: Current bar data
        direction: 'UP' or 'DOWN'
        vwap_history: Last 2 price positions relative to VWAP [older, newer]
        
    Returns:
        (pass: bool, reason: str)
    """
    close = row.get('close', 0)
    open_price = row.get('open', 0)
    vwap = row.get('vwap_proxy', close)
    atr14 = row.get('atr_14', 0)
    
    # Update VWAP history (1 = above, 0 = below)
    current_vwap_pos = 1 if close > vwap else 0
    vwap_history.append(current_vwap_pos)
    # Use popleft() for deque (pop(0) raises TypeError on deque objects)
    while len(vwap_history) > 2:
        vwap_history.popleft()
    
    # Rule 1: Price holds above/below VWAP for 2 consecutive bars
    if len(vwap_history) >= 2:
        if direction == 'UP' and all(pos == 1 for pos in vwap_history):
            return True, "vwap_hold_above"
        elif direction == 'DOWN' and all(pos == 0 for pos in vwap_history):
            return True, "vwap_hold_below"
    
    # Rule 2: Strong directional candle (body > 0.3 x ATR)
    body = abs(close - open_price)
    if atr14 > 0 and body > 0.3 * atr14:
        # Check candle direction matches signal direction
        candle_up = close > open_price
        if (direction == 'UP' and candle_up) or (direction == 'DOWN' and not candle_up):
            return True, "strong_directional_candle"
    
    return False, "no_micro_confirmation"


def _ev_net(ev_predicted: float, costs: float, minute_of_day: int, iv_rank: float = 50.0) -> float:
    """
    EV_net = EV_predicted - (Costs x SafetyFactor x SpreadPenalty) (Req 7).
    
    Edge Case 5 Mitigation: Low Liquidity/High Spread ("Impact Cost" Case)
    When IV rank >90, bid-ask spreads on NIFTY options widen to 5-10% of premium.
    Even correct direction predictions lose money to execution slippage.
    
    SafetyFactor:
      - 1.8x during open (09:15-09:45) and close (15:00-15:30) sessions
      - 1.0x during normal trading hours
    
    SpreadPenalty:
      - 1.0x when IV rank < 70 (normal spreads)
      - 1.5x when IV rank 70-85 (elevated spreads)
      - 2.0x when IV rank 85-90 (wide spreads)
      - 3.0x when IV rank > 90 (extreme spreads - likely no trade)
    
    This triple-penalty (base cost x safety x spread) ensures EV_net only
    passes when the edge is large enough to overcome real-world friction.
    """
    # Time-based safety factor (open/close volatility)
    if minute_of_day <= 30 or minute_of_day >= 345:
        sf = EV_SAFETY_ELEVATED
    else:
        sf = EV_SAFETY_NORMAL
    
    # Spread penalty based on IV rank (proxy for bid-ask spread width)
    if iv_rank > 90:
        spread_penalty = 3.0  # Extreme: likely unprofitable
    elif iv_rank > 85:
        spread_penalty = 2.0  # Wide spreads
    elif iv_rank > 70:
        spread_penalty = 1.5  # Elevated spreads
    else:
        spread_penalty = 1.0  # Normal
    
    return ev_predicted - (costs * sf * spread_penalty)


def check_iv_crush(row) -> float:
    """Enhancement 4 — Regime-Specific IV Crush Protector.

    When IV rank is very high (>85) AND IV is contracting (iv_pct_change < 0),
    the implied-volatility premium is collapsing.  Options are losing extrinsic
    value faster than normal — direction alone won't save the trade.

    Returns a confidence penalty to subtract from avg_conf.
    """
    iv_rank = float(row.get('iv_rank_approx', 50))
    iv_chg  = float(row.get('iv_pct_change',  0))
    if iv_rank > 85 and iv_chg < 0:
        return 0.10   # deduct 10% confidence when IV is crushing
    return 0.0
