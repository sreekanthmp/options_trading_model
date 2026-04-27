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

    Rules (v2 — aligned with direction-voting logic):
        1. 15-minute horizon is PRIMARY direction (matches voting: 15m weight=0.45)
        2. 30-minute horizon must AGREE with 15m
        3. 5-minute horizon is OPTIONAL — disagreement is ignored (no veto)
        4. 1-minute horizon is TIMING only (ignored for direction)

    WHY v2: In normal mode only 15m+30m vote on direction. The old function used
    5m as primary, which could penalize valid 15m/30m UP signals when 5m said DOWN —
    the exact opposite of the actual direction decision.

    Args:
        signals: dict of {horizon: {'pred': 0/1, 'conf': float}}

    Returns:
        (pass: bool, reason: str)
    """
    sig_15m = signals.get(15)
    sig_30m = signals.get(30)

    # Insufficient data from the two voting horizons — cannot assess agreement.
    # Return True (no penalty) so the vote result stands on its own merits.
    if not sig_15m or not sig_30m:
        return True, "insufficient_data"

    primary_dir = sig_15m['pred']  # 1=UP, 0=DOWN

    # Core check: 15m and 30m must agree
    if sig_30m['pred'] != primary_dir:
        return False, "15m_30m_conflict"

    # 5m is informational only — disagreement is logged at call site but NOT a veto
    sig_5m = signals.get(5)
    if sig_5m and sig_5m['pred'] != primary_dir:
        return True, "5m_disagree_ignored"

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
                                    vwap_history,   # deque(maxlen=N) from SignalState
                                    micro_regime: str = 'UNKNOWN',
                                    avg_conf: float = 0.0) -> tuple[bool, str]:
    """
    WHY: Models can signal too early (before price confirms direction).
         This final filter prevents premature entries by requiring price
         to demonstrate commitment to the predicted direction.

    Rules (need ONE of the following):
        1. Price holds above/below VWAP for 2 consecutive bars
           Exception: RANGING regime + conf>=0.88 → 1 bar sufficient
           (In RANGING, price oscillates around VWAP by design — 2 consecutive
            bars above/below never happens before the move, defeating the purpose)
        2. Candle body > 0.3 x ATR(14) (strong directional candle)

    Args:
        row: Current bar data
        direction: 'UP' or 'DOWN'
        vwap_history: Last 2 price positions relative to VWAP [older, newer]
        micro_regime: Current micro-regime string (e.g. 'RANGING', 'TRENDING_UP')
        avg_conf: Weighted average model confidence (0-1)

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
    # Check only the last 2 positions for consecutive confirmation
    last_2 = list(vwap_history)[-2:]

    # Rule 1: Price holds above/below VWAP for last 2 bars
    if len(last_2) >= 2:
        if direction == 'UP' and all(pos == 1 for pos in last_2):
            return True, "vwap_hold_above"
        elif direction == 'DOWN' and all(pos == 0 for pos in last_2):
            return True, "vwap_hold_below"

    # Rule 1b: RANGING + very high confidence — relax to 1 bar above/below VWAP.
    # In RANGING, price oscillates around VWAP continuously; requiring 2 consecutive
    # bars means entry only happens after the move is half-done. At conf>=0.88 (4/4
    # horizons strongly agreeing), allow 1-bar VWAP cross as sufficient confirmation.
    if micro_regime == 'RANGING' and avg_conf >= 0.88:
        if direction == 'UP' and current_vwap_pos == 1:
            return True, "vwap_cross_ranging_highconf"
        if direction == 'DOWN' and current_vwap_pos == 0:
            return True, "vwap_cross_ranging_highconf"

    # Rule 1c: TRENDING + very high confidence — relax to 1 bar above/below VWAP.
    # In a strong trend (TRENDING_UP/TRENDING_DOWN), price is already far from VWAP.
    # Waiting for 2 consecutive bars means missing the move entirely — price may
    # never pull back to VWAP in a genuine breakout. At conf>=0.85, allow 1 bar.
    is_trending_micro = micro_regime in ('TRENDING_UP', 'TRENDING_DOWN', 'BREAKOUT')
    if is_trending_micro and avg_conf >= 0.85:
        if direction == 'UP' and current_vwap_pos == 1:
            return True, "vwap_cross_trending_highconf"
        if direction == 'DOWN' and current_vwap_pos == 0:
            return True, "vwap_cross_trending_highconf"

    # Rule 2: Strong directional candle (body > 0.3 x ATR)
    body = abs(close - open_price)
    if atr14 > 0 and body > 0.3 * atr14:
        candle_up = close > open_price
        if (direction == 'UP' and candle_up) or (direction == 'DOWN' and not candle_up):
            # Rule 2b: candle is strong BUT check microstructure isn't dead
            # tick_imbalance and pressure_ratio must not both strongly oppose.
            # This catches "big candle into zero momentum" entries (MFE=0 pattern).
            tick_imb      = float(row.get('tick_imbalance', 0.0))
            pressure_ratio = float(row.get('pressure_ratio', 1.0))
            # For UP: need tick_imb > -0.4 (not strongly selling) OR pressure_ratio > 0.7
            # For DOWN: need tick_imb < +0.4 (not strongly buying) OR pressure_ratio < 1.43
            micro_ok = True
            if direction == 'UP' and tick_imb < -0.4 and pressure_ratio < 0.7:
                micro_ok = False
            elif direction == 'DOWN' and tick_imb > 0.4 and pressure_ratio > 1.43:
                micro_ok = False
            if micro_ok:
                return True, "strong_directional_candle"

    # Rule 3: Pattern structure confirmation (v4.2)
    # struct_score > 0 = HH/HL (bullish), < 0 = LH/LL (bearish).
    # If structure strongly aligns with direction AND price is on correct
    # side of VWAP (even for 1 bar), that is sufficient confirmation.
    # This catches setups where VWAP history is too short but structure is clear.
    struct_score = float(row.get('struct_score', 0.0))
    orb_dist     = float(row.get('orb_dist', 0.0))
    struct_aligned = (direction == 'UP'   and struct_score > 0.4) or \
                     (direction == 'DOWN' and struct_score < -0.4)
    # ORB also counts as structure confirmation if price broke out the right way
    orb_aligned    = (direction == 'UP'   and orb_dist > 0.3) or \
                     (direction == 'DOWN' and orb_dist < -0.3)
    if (struct_aligned or orb_aligned) and avg_conf >= 0.80:
        if direction == 'UP' and current_vwap_pos == 1:
            return True, "structure_aligned_vwap"
        if direction == 'DOWN' and current_vwap_pos == 0:
            return True, "structure_aligned_vwap"

    return False, "no_micro_confirmation"


def _ev_net(ev_predicted: float, costs: float, minute_of_day: int, iv_rank: float = 50.0, iv_proxy: float = 0.0) -> float:
    """
    EV_net = EV_predicted - (Costs x SafetyFactor x SpreadPenalty) (Req 7).

    SpreadPenalty uses iv_proxy (absolute annualised vol %) instead of iv_rank
    percentile — iv_rank breaks after crash periods (calm days rank at 5th
    percentile for months). iv_proxy is always meaningful.

    SafetyFactor:
      - 1.8x during open (09:15-09:45) and close (15:00-15:30) sessions
      - 1.0x during normal trading hours

    SpreadPenalty based on iv_proxy (annualised %):
      - iv_proxy > 2.0% (crisis/extreme): 1.8x — very wide spreads
      - iv_proxy > 1.5% (high vol):       1.5x — elevated spreads
      - iv_proxy > 1.0% (above normal):   1.2x — slightly wide
      - iv_proxy <= 1.0% (normal):        1.0x — normal spreads
    """
    # Time-based safety factor (open/close volatility)
    if minute_of_day <= 30 or minute_of_day >= 345:
        sf = EV_SAFETY_ELEVATED
    else:
        sf = EV_SAFETY_NORMAL

    # Spread penalty based on absolute iv_proxy (not relative iv_rank percentile)
    if iv_proxy > 2.0:
        spread_penalty = 1.8
    elif iv_proxy > 1.5:
        spread_penalty = 1.5
    elif iv_proxy > 1.2:
        spread_penalty = 1.2
    else:
        spread_penalty = 1.0

    return ev_predicted - (costs * sf * spread_penalty)


def check_iv_crush(row) -> float:
    """Enhancement 4 — Regime-Specific IV Crush Protector.

    When IV rank is very high (>85) AND IV is contracting (iv_pct_change < 0),
    the implied-volatility premium is collapsing.  Options are losing extrinsic
    value faster than normal — direction alone won't save the trade.

    Returns a confidence penalty to subtract from avg_conf.
    """
    iv_proxy = float(row.get('iv_proxy', 0.0))
    iv_chg   = float(row.get('iv_pct_change', 0))
    # High realized vol (iv_proxy > 1.5% annualised) AND contracting → IV crush risk
    if iv_proxy > 1.5 and iv_chg < 0:
        return 0.10   # deduct 10% confidence when IV is crushing
    return 0.0
