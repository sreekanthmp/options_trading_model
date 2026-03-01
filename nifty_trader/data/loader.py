"""CSV OHLCV loading utilities."""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import logging

from ..config import DATA_1MIN, DATA_5MIN, DATA_15MIN, DATA_1DAY

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. DATA LOADING
# ==============================================================================

def to_ist_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    if s.dt.tz is not None:
        s = s.dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    return s


def load_ohlcv(path: str, label: str, min_candles: int = 0) -> pd.DataFrame | None:
    """Load and validate OHLCV data from CSV.
    
    Performs comprehensive validation:
    1. File existence check
    2. DateTime parsing and timezone normalization
    3. OHLC column validation (numeric, non-negative, logical order)
    4. Duplicate timestamp detection
    5. Missing data gaps detection
    """
    if not os.path.isfile(path):
        print(f"  MISSING: {path} ({label} features will be zero)")
        return None
    
    try:
        df = pd.read_csv(path, parse_dates=['datetime'])
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        return None
    
    df['datetime'] = to_ist_naive(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Convert OHLC to numeric, coerce errors
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Validate OHLC logical constraints
    # High >= max(open, close), Low <= min(open, close)
    invalid_bars = (
        (df['high'] < df['open']) | (df['high'] < df['close']) |
        (df['low'] > df['open']) | (df['low'] > df['close']) |
        (df['high'] < df['low']) |
        (df['high'] <= 0) | (df['low'] <= 0)
    )
    if invalid_bars.any():
        n_invalid = invalid_bars.sum()
        logger.warning(f"{label}: {n_invalid} invalid OHLC bars detected, dropping")
        df = df[~invalid_bars]
    
    # Drop rows with any missing OHLC
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    
    # Check for duplicate timestamps
    duplicates = df['datetime'].duplicated()
    if duplicates.any():
        n_dup = duplicates.sum()
        logger.warning(f"{label}: {n_dup} duplicate timestamps detected, keeping last")
        df = df.drop_duplicates(subset='datetime', keep='last')
    
    # Minimum candle check
    if min_candles and len(df) < min_candles:
        print(f"  WARNING: {label} has only {len(df)} rows (need {min_candles})")
        return None
    
    print(f"  {label}: {len(df):,} rows  [{df['datetime'].min().date()} .. {df['datetime'].max().date()}]")
    return df


def load_all_data():
    print("Loading data...")
    df1m  = load_ohlcv(DATA_1MIN,  "1-min",  min_candles=10000)
    df5m  = load_ohlcv(DATA_5MIN,  "5-min")
    df15m = load_ohlcv(DATA_15MIN, "15-min")
    df1d  = load_ohlcv(DATA_1DAY,  "daily")
    if df1m is None:
        raise FileNotFoundError(f"{DATA_1MIN} not found. Run angelone_1min_downloader.py first.")

    # Drop incomplete 1-min days (< 300 candles = partial session)
    day_counts = df1m.groupby(df1m['datetime'].dt.date)['datetime'].transform('count')
    df1m = df1m[day_counts >= 300].copy()
    df1m['date']          = df1m['datetime'].dt.date
    df1m['minute_of_day'] = (df1m['datetime'].dt.hour * 60 +
                              df1m['datetime'].dt.minute) - (9 * 60 + 15)
    print(f"  1-min after filtering: {len(df1m):,} candles | {df1m['date'].nunique()} days")
    return df1m, df5m, df15m, df1d


# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================
