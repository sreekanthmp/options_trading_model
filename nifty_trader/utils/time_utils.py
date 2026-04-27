"""Session and time helpers."""
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_time_decay_confidence(minute_of_day: int) -> float:
    """
    Time-based decay function for signal confidence (Scarcity Logic).
    
    Late-day signals (after 2:30 PM) require +10% higher confidence
    due to lower liquidity and increased risk of overnight gaps.
    
    Returns multiplier: 1.0 at 10 AM, 1.1 at 2:30 PM+
    """
    # Market opens at 9:15 AM (minute 0), closes at 3:30 PM (minute 375)
    # 2:30 PM = minute 315
    if minute_of_day < 0:
        return 1.0
    
    if minute_of_day >= 315:  # After 2:30 PM
        return 1.10  # Require 10% higher confidence
    elif minute_of_day >= 270:  # After 1:45 PM
        return 1.05  # Require 5% higher confidence
    else:
        return 1.0  # Normal confidence requirements


def calculate_dynamic_stops(entry_price: float, current_atr: float,
                           direction: str, delta: float = 0.5) -> tuple:
    """
    Calculate dynamic stop-loss and take-profit using ATR multiples.

    current_atr is the SPOT ATR (in Nifty index points).  Options move at
    roughly delta × spot_move, so we scale before applying to the option premium.
    With ATM options delta ≈ 0.50; caller can override for deeper/shallower strikes.

    Stop: 2.5x option-ATR below/above entry, CAPPED AT 5% MAX LOSS
    Take-Profit: 4.0x option-ATR (2:1 reward-risk ratio minimum)

    Replaces static 40% stops which were too tight during volatile periods.
    Also fixes the original bug where spot ATR was used directly against option
    premium — for a Rs 12 option with spot ATR=20 the stop became negative.
    """
    option_atr = current_atr * delta
    # Clamp option_atr so stops are always meaningful (at least 1 Rs, at most 50% of entry)
    option_atr = max(1.0, min(option_atr, entry_price * 0.50))

    if direction.upper() in ['UP', 'CALL', 'BULLISH']:
        # ATR-based stop, but never more than 5% loss
        stop_loss   = max(entry_price * 0.95, entry_price - (2.5 * option_atr))
        stop_loss   = max(0.05, stop_loss)  # Min absolute value
        take_profit = entry_price + (4.0 * option_atr)
    else:  # DOWN, PUT, BEARISH
        # ATR-based stop, but never more than 5% loss
        stop_loss   = min(entry_price * 1.05, entry_price + (2.5 * option_atr))
        take_profit = max(0.05, entry_price - (4.0 * option_atr))

    return stop_loss, take_profit

