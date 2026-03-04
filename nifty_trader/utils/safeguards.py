"""Edge-case safeguards: NaN shield, Limit Price Protection, dynamic stops."""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def safe_value(val, default=0.0):
    """Edge Case 6: The NaN Shield.
    
    Clean NaN and Inf values before they enter the model to prevent
    "NaN Cascade" where a single API failure poisons rolling indicators.
    
    Args:
        val: Value to check
        default: Default value to return if val is NaN or Inf
    
    Returns:
        Cleaned value safe for feature calculations
    """
    if pd.isna(val) or np.isinf(val):
        return default
    return val


def get_dynamic_theta(dte_mins: float) -> float:
    """Edge Case 3: DTE-Weighted Theta Scaling.
    
    Scales theta based on time-decay acceleration (1/sqrt(t)).
    On Friday mornings (high DTE), theta is lower.
    On Wednesday afternoons (low DTE), theta accelerates.
    
    Args:
        dte_mins: Days to expiry in minutes
    
    Returns:
        Dynamic theta penalty per 1-min bar
    
    Example:
        - Monday 9:30 (DTE ~2100 min): theta = 0.15 * sqrt(750/2100) = 0.09
        - Thursday 14:00 (DTE ~90 min): theta = 0.15 * sqrt(750/90) = 0.43
    """
    # Baseline: 0.15 at 2 days (750 mins) = mid-week reference
    # Scale inversely with sqrt(DTE) to reflect accelerating time decay
    scale = np.sqrt(750 / max(dte_mins, 30))  # floor at 30 min to prevent explosion
    return 0.15 * scale


def check_lpp_violation(proposed_price: float, last_5_ticks: list) -> bool:
    """Edge Case 1: Limit Price Protection (LPP).
    
    Returns True if the proposed price is >2% away from the recent mean.
    This prevents "Fat Finger" errors where the model generates a signal
    during a sudden 1-second spike caused by large institutional orders.
    
    Such prices often exceed the exchange's execution range and result
    in order rejections or pending orders while the market moves away.
    
    Args:
        proposed_price: Entry price calculated by select_option
        last_5_ticks: Recent tick buffer (list of dicts with 'price' key)
    
    Returns:
        True if price violates 2% sanity check
    """
    if len(last_5_ticks) < 5:
        return False  # Not enough history, assume safe

    # Convert to list first — deque doesn't support slice indexing
    recent = list(last_5_ticks)[-5:]
    mean_price = sum(t['price'] for t in recent) / 5
    deviation = abs(proposed_price - mean_price) / mean_price
    return deviation > 0.02  # 2% threshold


def get_max_oi_strikes(spot: float, session=None) -> tuple:
    """
    Fetch strikes with max Call OI and max Put OI within ±500 points.
    High OI acts as magnet/resistance — avoid trading INTO those zones.

    Returns:
        (max_call_oi_strike, max_put_oi_strike, call_oi, put_oi)
    Placeholder: returns (0, 0, 0, 0) until option chain API is integrated.
    """
    try:
        from ..config import _api_limiter
        _api_limiter.wait_and_acquire(tokens=3)
        return (0.0, 0.0, 0, 0)
    except Exception as e:
        logger.error(f"get_max_oi_strikes failed: {e}")
        return (0.0, 0.0, 0, 0)


def avoid_oi_concentration_zone(entry_strike: float,
                                max_call_strike: float,
                                max_put_strike: float,
                                threshold: float = 50.0) -> bool:
    """
    Returns True if entry_strike is within threshold points of max OI strike.
    High OI acts as resistance — avoid trading INTO that zone.
    """
    if max_call_strike > 0 and abs(entry_strike - max_call_strike) < threshold:
        logger.warning(f"[OI] Entry strike {entry_strike} within {threshold} pts of "
                       f"max Call OI {max_call_strike}. Blocking.")
        return True
    if max_put_strike > 0 and abs(entry_strike - max_put_strike) < threshold:
        logger.warning(f"[OI] Entry strike {entry_strike} within {threshold} pts of "
                       f"max Put OI {max_put_strike}. Blocking.")
        return True
    return False
