"""Execution cost model: dynamic cost scaling, slippage, theta decay."""
import logging
import numpy as np

from ..config import (
    TOTAL_COST_PCT, COST_RT_PCT, SLIPPAGE_PCT, THETA_DECAY_PCT,
    DELTA_BASE, THETA_PTS_PER_BAR,
)

logger = logging.getLogger(__name__)


def effective_cost(row) -> float:
    """
    Dynamic cost model (v3.3 EV-Shield).
    Scales TOTAL_COST_PCT up during high-friction conditions:
      - Open / close session windows: wider spreads, more slippage (+40%)
      - High IV rank (>80): elevated option premium decay risk (+30%)
      - Active regime transition: uncertainty raises execution cost (+50%)
    Multipliers are applied sequentially (not additive).
    """
    cost = TOTAL_COST_PCT


    multiplier = 1.0
    if row.get('session_open', 0) or row.get('session_pm', 0):
        multiplier *= 1.4
    if row.get('iv_rank_approx', 0) > 80:
        multiplier *= 1.3
    if row.get('regime_transition', 0) == 1:
        multiplier *= 1.5
    # Cap the multiplier at 2.0x
    multiplier = min(multiplier, 2.0)
    cost *= multiplier
    return cost


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

