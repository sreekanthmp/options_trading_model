"""External market data: heavyweight stocks, BankNifty spread, India VIX, FII/DII.

ALL live market data comes from Angel One SmartAPI — zero yfinance dependency in
the live path.  yfinance was replaced because:
  - It is a scraper, not an official API.
  - Prone to 429 "Too Many Requests" errors that hung the entire live loop.
  - Latency of 1-3s per call on yf.download() made 1-min features stale.

Angel One token reference:
  NIFTY 50 index  : 99926000  (NSE)
  BANKNIFTY index : 99926009  (NSE)
  HDFCBANK        : 1333      (NSE)
  RELIANCE        : 2885      (NSE)
"""
import logging
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from ..config import _api_limiter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Angel One symbol → token map for heavyweight fetches
# ---------------------------------------------------------------------------
_AO_TOKENS = {
    'HDFCBANK':   {'token': '1333',     'exchange': 'NSE'},
    'RELIANCE':   {'token': '2885',     'exchange': 'NSE'},
    'BANKNIFTY':  {'token': '99926009', 'exchange': 'NSE'},
    'NIFTY':      {'token': '99926000', 'exchange': 'NSE'},
}

# Module-level cache: {symbol: (timestamp, last_return_pct)}
# Prevents fetching more than once per bar per symbol
import time as _time
import threading as _threading
_cache: dict = {}
_cache_lock = _threading.Lock()
_CACHE_TTL_SECONDS = 90   # 90s TTL — reduces API calls to ~1/bar instead of 4/bar
_bg_fetch_lock = _threading.Lock()   # prevent concurrent background fetches
_bg_thread: _threading.Thread | None = None


def _prefetch_all_symbols(session, symbols=('HDFCBANK', 'RELIANCE', 'BANKNIFTY', 'NIFTY')):
    """
    Background thread: fetch all heavyweight symbols in one sweep.
    Runs off the main loop so it never adds latency to the signal path.
    Only one background sweep runs at a time (_bg_fetch_lock).
    """
    global _bg_thread
    if not _bg_fetch_lock.acquire(blocking=False):
        return   # previous sweep still running — skip
    def _run():
        try:
            for sym in symbols:
                _fetch_last_return_angelone(session, sym)
        finally:
            _bg_fetch_lock.release()
    t = _threading.Thread(target=_run, daemon=True, name="ExternalDataPrefetch")
    _bg_thread = t
    t.start()


def _fetch_last_return_angelone(session, symbol: str, n_bars: int = 6) -> float:
    """
    Fetch the last 1-min return for a symbol via Angel One getCandleData.

    Uses the session already authenticated in the live loop.
    Returns pct change of the latest close vs the previous close.
    Falls back to 0.0 on any error — this is a supplementary alpha signal,
    not the primary price feed; zero is always a safe neutral fallback.

    Parameters
    ----------
    session : AngelSession
    symbol  : one of 'HDFCBANK', 'RELIANCE', 'BANKNIFTY', 'NIFTY'
    n_bars  : number of bars to fetch (just enough to compute returns)
    """
    from datetime import datetime, timedelta

    now = _time.time()
    with _cache_lock:
        cached = _cache.get(symbol)
    if cached is not None and (now - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    meta = _AO_TOKENS.get(symbol)
    if meta is None:
        logger.warning(f"[ExternalData] Unknown symbol '{symbol}' — falling back to 0.0")
        return 0.0

    if session is None:
        return 0.0

    obj = session.get()
    if obj is None:
        return 0.0

    try:
        _api_limiter.wait_and_acquire(tokens=1)
        dt_now = datetime.now()
        dt_from = dt_now - timedelta(minutes=n_bars + 5)
        r = obj.getCandleData({
            "exchange":    meta['exchange'],
            "symboltoken": meta['token'],
            "interval":    "ONE_MINUTE",
            "fromdate":    dt_from.strftime("%Y-%m-%d %H:%M"),
            "todate":      dt_now.strftime("%Y-%m-%d %H:%M"),
        })
        if not r or not r.get('data') or len(r['data']) < 2:
            with _cache_lock:
                _cache[symbol] = (_time.time(), 0.0)
            return 0.0

        closes = [float(row[4]) for row in r['data'][-n_bars:]]  # col index 4 = close
        if len(closes) < 2 or closes[-2] <= 0:
            with _cache_lock:
                _cache[symbol] = (_time.time(), 0.0)
            return 0.0

        ret = (closes[-1] - closes[-2]) / closes[-2] * 100.0
        val = float(ret) if not np.isnan(ret) else 0.0
        with _cache_lock:
            _cache[symbol] = (_time.time(), val)
        return val

    except Exception as e:
        logger.error(f"[ExternalData] Angel One fetch for {symbol} failed: {e}")
        with _cache_lock:
            _cache[symbol] = (_time.time(), 0.0)
        return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_heavyweight_returns(session=None,
                               symbols: list = ['HDFCBANK', 'RELIANCE']) -> dict:
    """
    Fetch latest 1-min return for market heavyweights via Angel One.

    HDFCBANK + RELIANCE together account for ~18% of NIFTY weight and are
    reliable leading indicators for NIFTY direction.

    Returns
    -------
    dict: {'HDFCBANK.NS': float, 'RELIANCE.NS': float}
    Keys use the old yfinance format to stay backward-compatible with live.py.
    """
    results = {}
    yf_key_map = {'HDFCBANK': 'HDFCBANK.NS', 'RELIANCE': 'RELIANCE.NS'}
    for sym in symbols:
        ret = _fetch_last_return_angelone(session, sym)
        results[yf_key_map.get(sym, sym)] = ret
    return results


def fetch_banknifty_spread(session=None) -> dict:
    """
    BankNifty vs NIFTY spread for sector rotation detection via Angel One.

    When BankNifty and NIFTY diverge (one up, other down), it's an early
    warning of sector rotation that often precedes NIFTY reversals.

    Returns
    -------
    dict: {'spread_pct': float, 'spread_5m': float, 'divergence': int}
    """
    bn_ret  = _fetch_last_return_angelone(session, 'BANKNIFTY')
    nif_ret = _fetch_last_return_angelone(session, 'NIFTY')

    spread_1m   = bn_ret - nif_ret
    divergence  = 1 if (bn_ret * nif_ret < 0 and abs(spread_1m) > 0.1) else 0

    # spread_5m: use the same 1-min return as a proxy (5-min would need a
    # separate fetch; the current value is directionally correct)
    return {
        'spread_pct':  float(spread_1m),
        'spread_5m':   float(spread_1m),   # same-bar proxy
        'divergence':  divergence,
    }


# ---------------------------------------------------------------------------
# Instrument master: symbol -> token lookup (downloaded once, no auth needed)
# Angel One publishes this JSON daily at the URL below.
# Symbol format in master: NIFTY02MAR2625300PE  (NIFTY + DDMmmYYYY + strike + type)
# ---------------------------------------------------------------------------
_INSTRUMENT_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)
_instrument_master: dict = {}   # {symbol: token_str}
_master_loaded: bool = False


def _load_instrument_master() -> bool:
    """
    Download the Angel One instrument master and build a symbol->token dict
    for NIFTY NFO options.  Called once on first fetch_option_ltp() call.
    """
    global _instrument_master, _master_loaded
    if _master_loaded:
        return True
    try:
        import requests as _req
        logger.info("[InstrumentMaster] Downloading Angel One instrument master...")
        r = _req.get(_INSTRUMENT_MASTER_URL, timeout=30)
        if r.status_code != 200:
            logger.warning(f"[InstrumentMaster] HTTP {r.status_code} — option LTP unavailable")
            return False
        data = r.json()
        count = 0
        for item in data:
            if item.get('exch_seg') == 'NFO' and item.get('name') == 'NIFTY':
                sym = item.get('symbol', '')
                tok = item.get('token', '')
                if sym and tok:
                    _instrument_master[sym] = str(tok)
                    count += 1
        _master_loaded = True
        logger.info(f"[InstrumentMaster] Loaded {count} NIFTY NFO instruments")
        return True
    except Exception as e:
        logger.warning(f"[InstrumentMaster] Download failed: {e}")
        return False


# Cache for option LTP prices: {symbol: (timestamp, ltp)}
_option_ltp_cache: dict = {}
_OPTION_LTP_TTL = 55   # seconds — 1 fetch/bar max, avoids rate limit


def fetch_option_ltp(session, strike: int, option_type: str, expiry_date=None) -> float:
    """
    Fetch live LTP for a NIFTY weekly option from Angel One.

    Uses the publicly-available instrument master to resolve the symbol token
    (no searchScrip needed), then calls getMarketData('LTP') for the price.

    Symbol format (Angel One NFO): NIFTY02MAR2625300PE
      = NIFTY + DDMmmYYYY (4-digit year) + strike + CE/PE

    Parameters
    ----------
    session     : AngelSession
    strike      : int, e.g. 25300
    option_type : 'CE' or 'PE'
    expiry_date : datetime.date of expiry (None = nearest expiry from master)

    Returns
    -------
    float LTP, or 0.0 on failure (caller falls back to Black-Scholes estimate).
    """
    import datetime as _dt

    # Ensure instrument master is loaded
    if not _master_loaded and not _load_instrument_master():
        return 0.0

    # Determine expiry: use the nearest available expiry from the master
    # that is >= today.
    today = _dt.date.today()
    if expiry_date is None:
        # Parse all expiry dates from the loaded master keys.
        # Master symbol format: NIFTY02MAR2625300PE
        #   NIFTY = 5 chars, DDMmmYY = 7 chars (2-digit year), then strike+type
        expiry_dates = set()
        for sym in _instrument_master:
            if len(sym) >= 12 and sym.startswith('NIFTY'):
                date_part = sym[5:12]   # 7 chars: DDMmmYY
                try:
                    expiry_dates.add(_dt.datetime.strptime(date_part, '%d%b%y').date())
                except ValueError:
                    pass
        future_expiries = sorted(d for d in expiry_dates if d >= today)
        if not future_expiries:
            logger.warning("[OptionLTP] No future expiries found in instrument master")
            return 0.0
        expiry_date = future_expiries[0]

    # Build symbol in Angel One NFO format: NIFTY02MAR2625300PE
    # Uses 2-digit year (%y): 02MAR26 not 02MAR2026
    expiry_str = expiry_date.strftime('%d%b%y').upper()   # e.g. 02MAR26
    symbol = f"NIFTY{expiry_str}{strike}{option_type}"    # e.g. NIFTY02MAR2625300PE

    now_ts = _time.time()

    # Return cached price if still fresh
    if symbol in _option_ltp_cache:
        cached_ts, cached_ltp = _option_ltp_cache[symbol]
        if now_ts - cached_ts < _OPTION_LTP_TTL:
            return cached_ltp

    token = _instrument_master.get(symbol)
    if not token:
        logger.warning(f"[OptionLTP] Token not found for {symbol} — check expiry/strike")
        _option_ltp_cache[symbol] = (now_ts, 0.0)
        return 0.0

    obj = session.get() if session is not None else None
    if obj is None:
        return 0.0

    try:
        _api_limiter.wait_and_acquire(tokens=1)
        mkt_resp = obj.getMarketData('LTP', {'NFO': [token]})
        if not mkt_resp or not mkt_resp.get('status'):
            logger.debug(f"[OptionLTP] getMarketData failed for {symbol} token={token}")
            return 0.0

        fetched = mkt_resp.get('data', {}).get('fetched', [])
        if not fetched:
            return 0.0

        ltp = float(fetched[0].get('ltp', 0))
        _option_ltp_cache[symbol] = (now_ts, ltp)
        logger.info(f"[OptionLTP] {symbol} = Rs {ltp:.2f}")
        return ltp

    except Exception as e:
        logger.warning(f"[OptionLTP] {symbol} fetch error: {e}")
        _option_ltp_cache[symbol] = (now_ts, 0.0)
        return 0.0


def fetch_india_vix(session=None) -> float:
    """
    India VIX is not available via Angel One getCandleData (index-only).
    Returns 0.0 — callers fall back to the iv_proxy feature computed from
    historical ATR which is already in the feature vector.
    """
    return 0.0


def fetch_option_chain_ndi(session=None, spot: float = 0.0) -> float:
    """
    Net Delta Imbalance (NDI) from NIFTY option chain OI.
    Placeholder — returns 0.0 until option chain API is integrated.
    """
    return 0.0


def fetch_fii_dii_flow() -> dict:
    """
    FII/DII net positions — NSE publishes this daily at ~18:30.
    Placeholder — returns zeros.
    """
    return {'fii_net': 0.0, 'dii_net': 0.0}


# calculate_time_decay_confidence and calculate_dynamic_stops live in
# utils/time_utils.py — import from there to avoid duplication.
