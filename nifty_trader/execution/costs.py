"""Execution cost model: dynamic cost scaling, slippage, theta decay, brokerage."""
import logging
import numpy as np

from ..config import (
    TOTAL_COST_PCT, COST_RT_PCT, SLIPPAGE_PCT, THETA_DECAY_PCT,
    DELTA_BASE, THETA_PTS_PER_BAR,
    BROKERAGE_PER_ORDER, GST_ON_BROKERAGE,
    STT_SELL_PCT, NSE_TXN_PCT, STAMP_DUTY_PCT, STT_EXPIRY_PCT,
)

logger = logging.getLogger(__name__)


def effective_cost(row) -> float:
    """
    Dynamic cost model (v3.3 EV-Shield).
    Scales TOTAL_COST_PCT up during high-friction conditions:
      - Open / close session windows: wider spreads, more slippage (+40%)
      - High IV rank (>80): elevated option premium decay risk (+30%)
      - Active regime transition: uncertainty raises execution cost (+50%)
      - Expiry day: wider spreads, higher STT, gamma slippage (+60%)
    Multipliers are applied sequentially (not additive).
    """
    cost = TOTAL_COST_PCT

    multiplier = 1.0
    if row.get('session_open', 0) or row.get('session_pm', 0):
        multiplier *= 1.4
    if float(row.get('iv_proxy', 0)) > 1.5:  # high realized vol → wider spreads
        multiplier *= 1.3
    if row.get('regime_transition', 0) == 1:
        multiplier *= 1.5
    # Expiry day: wider bid-ask spread (2-4x normal), higher STT (0.125% vs 0.1%),
    # and gamma slippage risk on exits — apply 60% extra cost
    if row.get('is_expiry', 0) == 1:
        multiplier *= 1.6
    # Cap the multiplier at 3.0x (expiry + high-IV + transition could stack)
    multiplier = min(multiplier, 3.0)
    cost *= multiplier
    return cost


def calculate_brokerage(entry_price: float, exit_price: float,
                        qty: int, is_expiry: bool = False) -> dict:
    """
    Calculate exact Angel One brokerage + statutory charges for one round-trip
    NIFTY options trade. Deducted from paper PnL to match live net P&L.

    Parameters
    ----------
    entry_price : float  — premium paid per unit at entry
    exit_price  : float  — premium received per unit at exit
    qty         : int    — total quantity (contracts × lot_size)
    is_expiry   : bool   — True on expiry day (higher STT on sell side)

    Returns
    -------
    dict with:
        total_charges  : float  — total Rs to deduct from gross PnL
        brokerage      : float  — Rs 20 × 2 orders
        gst            : float  — 18% GST on brokerage
        stt            : float  — STT on sell side
        txn_charges    : float  — NSE transaction charges
        stamp_duty     : float  — stamp duty on buy side
    """
    entry_turnover = entry_price * qty
    exit_turnover  = exit_price  * qty

    brokerage   = BROKERAGE_PER_ORDER * 2          # entry + exit order
    gst         = brokerage * GST_ON_BROKERAGE
    stt_pct     = STT_EXPIRY_PCT if is_expiry else STT_SELL_PCT
    stt         = exit_turnover * stt_pct           # STT on sell (exit) side
    txn         = (entry_turnover + exit_turnover) * NSE_TXN_PCT
    stamp       = entry_turnover * STAMP_DUTY_PCT   # stamp duty on buy (entry) side
    total       = round(brokerage + gst + stt + txn + stamp, 2)

    return {
        'total_charges': total,
        'brokerage':     round(brokerage, 2),
        'gst':           round(gst, 2),
        'stt':           round(stt, 2),
        'txn_charges':   round(txn, 2),
        'stamp_duty':    round(stamp, 2),
    }


def get_dynamic_theta(dte_mins: float) -> float:
    """Edge Case 3: DTE-Weighted Theta Scaling.

    Scales theta based on time-decay acceleration (1/sqrt(t)).
    On Friday mornings (high DTE), theta is lower.
    On Wednesday afternoons (low DTE), theta accelerates.

    Args:
        dte_mins: Days to expiry in minutes

    Returns:
        Dynamic theta penalty per 1-min bar
    """
    scale = np.sqrt(750 / max(dte_mins, 30))
    return 0.15 * scale

