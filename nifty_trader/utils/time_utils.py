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
                           direction: str) -> tuple:
    """
    Calculate dynamic stop-loss and take-profit using ATR multiples.
    
    Stop: 2.5x ATR (volatility-adjusted safety net)
    Take-Profit: 4.0x ATR (2:1 reward-risk ratio minimum)
    
    Replaces static 40% stops which were too tight during volatile periods.
    """
    if direction.upper() in ['UP', 'CALL', 'BULLISH']:
        stop_loss = entry_price - (2.5 * current_atr)
        take_profit = entry_price + (4.0 * current_atr)
    else:  # DOWN, PUT, BEARISH
        stop_loss = entry_price + (2.5 * current_atr)
        take_profit = entry_price - (4.0 * current_atr)
    
    return stop_loss, take_profit

