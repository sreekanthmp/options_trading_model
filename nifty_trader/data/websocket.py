"""Angel One WebSocket streamer, session management, and live candle fetching."""
import os, sys, json, time, threading, logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
from typing import Optional
import warnings

from nifty_trader.data.loader import to_ist_naive
warnings.filterwarnings('ignore')

from ..config import (
    CONFIG_FILE, NIFTY_TOKEN, USE_WEBSOCKET, WEBSOCKET_AVAILABLE,
    TICK_BUFFER_SIZE, _api_limiter,
)
from ..features.feature_engineering import add_1min_features, add_htf_features

try:
    from SmartApi import SmartConnect
    import pyotp
    SMARTAPI_AVAILABLE = True
except ImportError:
    SMARTAPI_AVAILABLE = False

try:
    from SmartApi.smartWebSocketV2 import SmartWebSocketV2
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ==============================================================================
# BAR VALIDATION & DATA INTEGRITY (LIVE-SAFE)
# ==============================================================================

class BarValidator:
    """
    Validates every incoming OHLCV bar before it enters the indicator pipeline.

    Failures here are NOT raised as exceptions — they are flagged so the live
    loop can decide whether to skip the bar, use the previous bar, or halt.

    Checks performed:
      1. OHLC sanity: H >= max(O,C) >= min(O,C) >= L >= 0
      2. Price continuity: gap from prev_close must not exceed MAX_GAP_PCT
      3. Timestamp continuity: gap from prev_bar must not exceed MAX_TS_GAP_SECONDS
      4. Staleness: bar close time must be within MAX_STALE_SECONDS of now()
      5. Non-zero close price

    All thresholds are conservative to reduce false positives but strict enough
    to catch the failure modes documented in the audit.
    """
    MAX_GAP_PCT         = 3.0    # % — gaps beyond this are suspect
    MAX_TS_GAP_SECONDS  = 150    # 2.5 min — more than 2 bars missing
    MAX_STALE_SECONDS   = 300    # 5 min — bar older than this is stale
    FAST_MARKET_PCT     = 0.8    # % move in one 1-min bar = fast market alert

    def __init__(self):
        self._prev_close     : float = 0.0
        self._prev_timestamp : object = None
        self._consecutive_errors : int = 0
        self.HALT_ON_ERRORS  : int = 3   # halt trading after this many consecutive errors

    def validate(self, bar: pd.Series) -> tuple:
        """
        Returns (is_valid: bool, errors: list[str])
        errors is empty when valid.
        """
        errors = []
        now    = datetime.now()

        # 1. Non-zero close
        close = float(bar.get('close', 0))
        if close <= 0:
            errors.append(f"INVALID_CLOSE: close={close}")

        # 2. OHLC sanity
        try:
            o, h, lo, c = (float(bar['open']), float(bar['high']),
                           float(bar['low']),  float(bar['close']))
            if not (h >= max(o, c) and min(o, c) >= lo and lo >= 0):
                errors.append(f"OHLC_VIOLATION: O={o} H={h} L={lo} C={c}")
        except (KeyError, TypeError, ValueError) as e:
            errors.append(f"OHLC_PARSE_ERROR: {e}")

        # 3. Price continuity
        if self._prev_close > 0 and close > 0:
            gap_pct = abs(close - self._prev_close) / self._prev_close * 100
            if gap_pct > self.MAX_GAP_PCT:
                errors.append(f"PRICE_GAP: {gap_pct:.2f}% gap (prev={self._prev_close:.2f} now={close:.2f})")
            if gap_pct > self.FAST_MARKET_PCT:
                logger.warning(f"[BarValidator] FAST_MARKET: {gap_pct:.2f}% move in 1 bar")

        # 4. Timestamp continuity
        try:
            ts = bar.get('datetime', None)
            if ts is not None and self._prev_timestamp is not None:
                ts_dt   = pd.Timestamp(ts)
                prev_dt = pd.Timestamp(self._prev_timestamp)
                gap_sec = (ts_dt - prev_dt).total_seconds()
                if gap_sec > self.MAX_TS_GAP_SECONDS:
                    errors.append(f"TIMESTAMP_GAP: {gap_sec:.0f}s between bars")
        except Exception:
            pass

        # 5. Staleness
        try:
            ts = bar.get('datetime', None)
            if ts is not None:
                ts_dt   = pd.Timestamp(ts)
                age_sec = (now - ts_dt).total_seconds()
                if age_sec > self.MAX_STALE_SECONDS:
                    errors.append(f"STALE_BAR: {age_sec:.0f}s old")
        except Exception:
            pass

        is_valid = len(errors) == 0

        # Track consecutive errors for trading-halt decision
        if is_valid:
            self._consecutive_errors = 0
            # Update state on good bars only
            if close > 0:
                self._prev_close     = close
            ts = bar.get('datetime', None)
            if ts is not None:
                self._prev_timestamp = ts
        else:
            self._consecutive_errors += 1
            for err in errors:
                logger.warning(f"[BarValidator] {err}")

        return is_valid, errors

    def should_halt(self) -> bool:
        """Returns True when consecutive error count exceeds threshold."""
        return self._consecutive_errors >= self.HALT_ON_ERRORS
    
    def get_last_valid_timestamp(self) -> Optional[datetime]:
        """Returns timestamp of last validated bar, or None if none validated yet."""
        return self._prev_timestamp

    def reset(self):
        self._prev_close         = 0.0
        self._prev_timestamp     = None
        self._consecutive_errors = 0


class LatencyMonitor:
    """
    Measures the lag between bar-close timestamp and signal-ready timestamp.

    For 1-min signals to be actionable the lag must be < 500ms.
    If median lag > 1000ms for the last 10 bars, signals on the 1-min timeframe
    are effectively trading into the next bar and should be suppressed.
    """
    WARN_MS  = 500
    HALT_MS  = 2000
    WINDOW   = 10

    def __init__(self):
        self._lags : deque = deque(maxlen=self.WINDOW)

    def record(self, bar_close_dt, signal_ready_dt=None):
        """Call when a signal is ready to be acted on."""
        if signal_ready_dt is None:
            signal_ready_dt = datetime.now()
        try:
            lag_ms = (signal_ready_dt - pd.Timestamp(bar_close_dt)).total_seconds() * 1000
            self._lags.append(lag_ms)
            if lag_ms > self.HALT_MS:
                logger.warning(f"[Latency] CRITICAL lag {lag_ms:.0f}ms — 1m signals unreliable")
            elif lag_ms > self.WARN_MS:
                logger.warning(f"[Latency] HIGH lag {lag_ms:.0f}ms")
        except Exception:
            pass

    def median_lag_ms(self) -> float:
        if not self._lags:
            return 0.0
        return float(np.median(list(self._lags)))

    def is_acceptable(self) -> bool:
        return self.median_lag_ms() < self.HALT_MS


# 11. ANGEL ONE SESSION & LIVE DATA
# ==============================================================================

# ==============================================================================
# WEBSOCKET STREAMER (Real-time tick data - zero lag)
# ==============================================================================

class MarketStreamer:
    """Real-time market data streamer using Angel One WebSocket.
    
    Edge Case 4: Producer-Consumer Pattern for Data Aliasing Prevention
    
    In high-frequency bursts, WebSocket can send multiple ticks per millisecond.
    If on_data() is too heavy (FFT, OFI calculations), it creates \"processing lag\"
    where the bot thinks NIFTY is at 22,100 but the real market is at 22,120.
    
    Solution: Use threaded producer-consumer pattern:
      - WebSocket on_data() = Producer: lightweight, just push to queue
      - Separate _process_engine() = Consumer: heavy lifting every 500ms
    
    This prevents stop-loss triggers from being \"late\" relative to real market.
    """
    
    def __init__(self, auth_token=None, api_key=None, client_id=None, feed_token=None):
        self.last_ltp = 0.0
        self.tick_count = 0
        self.tick_buffer = deque(maxlen=TICK_BUFFER_SIZE)
        self.connected = False
        self.sws = None
        self.lock = threading.Lock()
        self.enabled = False
        
        # v4.0: Order Flow Imbalance (OFI) tracking
        self.buy_volume = 0
        self.sell_volume = 0
        self.last_price = 0.0
        self.ofi_buffer = deque(maxlen=100)  # Last 100 ticks for OFI
        
        # 2026 Edge: Heartbeat monitor for zombie WebSocket detection
        self.last_tick_time = None  # None until first tick received
        self.last_tick_count = 0
        self.reconnect_attempts = 0
        
        # Edge Case 4: Producer-Consumer pattern
        import queue
        self.raw_queue = queue.Queue(maxsize=1000)  # Buffer up to 1000 ticks
        self.worker_running = False
        self.worker = None
        
        if not WEBSOCKET_AVAILABLE:
            return
            
        try:
            self.sws = SmartWebSocketV2(
                auth_token=auth_token,
                api_key=api_key,
                client_code=client_id,
                feed_token=feed_token
            )
            self.enabled = True
        except Exception as e:
            print(f"  WebSocket init error: {e}")
    
    def on_open(self, wsapp):
        self.connected = True
        print("  [WebSocket] Connected")
        
        # Subscribe to NIFTY 50 LTP after connection is established
        try:
            correlation_id = "nifty_live"
            mode = 1  # LTP only (fastest)
            token_list = [{"exchangeType": 1, "tokens": [NIFTY_TOKEN]}]
            
            self.sws.subscribe(correlation_id, mode, token_list)
            print(f"  [WebSocket] Subscribed to NIFTY (token: {NIFTY_TOKEN})")
        except Exception as e:
            print(f"  [WebSocket] Subscribe error: {e}")
    
    def on_error(self, wsapp, error):
        self.connected = False
        print(f"  [WebSocket] Error: {error}")
    
    def on_close(self, wsapp):
        self.connected = False
        print("  [WebSocket] Disconnected")
    
    def on_data(self, wsapp, message):
        """Producer: Lightweight tick ingestion (Edge Case 4).
        
        Only parses JSON and pushes to queue. All heavy processing
        (OFI, heartbeat, buffer management) happens in consumer thread.
        This prevents \"logic lag\" during high-frequency bursts.
        """
        try:
            # Just push raw message to queue - consumer will process it
            if not self.raw_queue.full():
                self.raw_queue.put(message)
        except Exception:
            pass
    
    def _process_engine(self):
        """Consumer: Heavy lifting done here (Edge Case 4).
        
        Runs in separate thread processing ticks every 500ms batch.
        This decouples I/O (WebSocket) from computation (OFI, FFT, etc.).
        """
        while self.worker_running:
            try:
                # Process all queued messages
                messages_batch = []
                try:
                    # Get up to 100 messages at once (non-blocking)
                    while len(messages_batch) < 100:
                        msg = self.raw_queue.get(timeout=0.5)
                        messages_batch.append(msg)
                        self.raw_queue.task_done()
                except:
                    pass  # Queue empty or timeout
                
                # Process batch
                for message in messages_batch:
                    try:
                        if isinstance(message, str):
                            import json
                            message = json.loads(message)
                        
                        # Angel One sends LTP in paise
                        if 'last_traded_price' in message:
                            ltp = float(message['last_traded_price']) / 100.0
                        elif 'ltp' in message:
                            ltp = float(message['ltp']) / 100.0
                        else:
                            continue
                        
                        with self.lock:
                            # 2026 Edge: Update heartbeat timestamp
                            self.last_tick_time = time.time()
                            
                            # v4.0: OFI calculation
                            # If price ticked up, classify as buy; if down, as sell
                            if self.last_price > 0:
                                if ltp > self.last_price:
                                    self.buy_volume += 1
                                    side = 'buy'
                                elif ltp < self.last_price:
                                    self.sell_volume += 1
                                    side = 'sell'
                                else:
                                    side = 'neutral'
                                
                                self.ofi_buffer.append({
                                    'price': ltp,
                                    'side': side,
                                    'time': datetime.now()
                                })
                            
                            self.last_price = ltp
                            self.last_ltp = ltp
                            self.tick_count += 1
                            self.tick_buffer.append({
                                'price': ltp,
                                'time': datetime.now(),
                                'tick': self.tick_count
                            })
                    
                    except Exception:
                        pass
                
                # Brief sleep to prevent CPU spin
                time.sleep(0.01)
                
            except Exception:
                pass
    
    def connect(self):
        """Start WebSocket connection in background thread.
        
        Edge Case 4: Also starts consumer worker thread for processing.
        """
        if not self.enabled or self.sws is None:
            return False
        
        try:
            # Start consumer worker thread (Edge Case 4)
            if not self.worker_running:
                self.worker_running = True
                self.worker = threading.Thread(target=self._process_engine, daemon=True)
                self.worker.start()
                print("  [WebSocket] Consumer worker started")
            
            # Set up callbacks
            self.sws.on_open = self.on_open
            self.sws.on_data = self.on_data
            self.sws.on_error = self.on_error
            self.sws.on_close = self.on_close
            
            # Start connection (subscription happens in on_open callback)
            ws_thread = threading.Thread(target=self.sws.connect, daemon=True)
            ws_thread.start()
            
            print(f"  [WebSocket] Connecting to Angel One...")
            return True
            
        except Exception as e:
            print(f"  [WebSocket] Connect error: {e}")
            return False
    
    def get_latest_price(self):
        """Thread-safe access to latest price."""
        with self.lock:
            return self.last_ltp if self.last_ltp > 0 else None
    
    def get_tick_stats(self):
        """Get statistics from tick buffer."""
        with self.lock:
            if not self.tick_buffer:
                return None
            prices = [t['price'] for t in self.tick_buffer]
            return {
                'count': len(prices),
                'latest': prices[-1],
                'high': max(prices),
                'low': min(prices),
                'range': max(prices) - min(prices)
            }
    
    def get_ofi(self) -> float:
        """Get Order Flow Imbalance ratio.
        
        Returns: Buy imbalance ratio (0.0 to 1.0)
                 >0.80 = strong buying pressure (boost prediction by +0.15)
                 <0.20 = strong selling pressure
        """
        with self.lock:
            total = self.buy_volume + self.sell_volume
            if total == 0:
                return 0.5  # Neutral
            return self.buy_volume / total
    
    def reset_ofi(self):
        """Reset OFI counters (called every minute)."""
        with self.lock:
            self.buy_volume = 0
            self.sell_volume = 0
    
    def check_heartbeat(self, market_open: bool = True,
                        minute_of_day: int = 0) -> bool:
        """Check if WebSocket is actually receiving data (zombie detection).

        Returns: True if healthy, False if zombie (needs reconnect)

        Threshold is adaptive:
          - Normal market hours      : 10s  (NIFTY ticks every 1-3s normally)
          - Opening (< 9:30, < 15m)  : 20s  (first-minute order book settling)
          - Lunch (12:15–13:00)      : 45s  (tick frequency drops to 1 per 20-30s;
                                             a flat 10s threshold triggers false
                                             reconnects during valid quiet periods)
          - Pre-close (> 14:50)      : 20s  (thinner order book, wider ticks)
        """
        if not self.connected or not market_open:
            return True  # Can't check if not connected or market closed

        with self.lock:
            last_tick_time = self.last_tick_time

        # Skip check if no ticks received yet (initial connection phase)
        if last_tick_time is None:
            return True

        time_since_last_tick = time.time() - last_tick_time

        # Adaptive threshold — avoid false reconnects during low-volume windows
        if 60 <= minute_of_day <= 105:    # 10:15–11:00 (opening settled, max volume)
            threshold = 10.0
        elif minute_of_day < 15:          # first 15 min: order book settling
            threshold = 20.0
        elif 180 <= minute_of_day <= 225: # 12:15–13:00 lunch
            threshold = 45.0
        elif minute_of_day >= 335:        # after 14:50 pre-close (335 = 14:50 - 09:15)
            threshold = 20.0
        else:
            threshold = 10.0

        if time_since_last_tick > threshold:
            print(f"  [WARN] [Heartbeat] Zombie WebSocket detected! "
                  f"Last tick: {time_since_last_tick:.1f}s ago "
                  f"(threshold {threshold:.0f}s, min_of_day={minute_of_day})")
            return False

        return True
    
    def force_reconnect(self):
        """Force disconnect and reconnect to fix zombie WebSocket."""
        print("  [Heartbeat] Force reconnecting...")
        self.disconnect()
        time.sleep(2)
        self.reconnect_attempts += 1
        success = self.connect()
        if success:
            print(f"  [OK] [Heartbeat] Reconnected successfully (attempt {self.reconnect_attempts})")
        else:
            print(f"  [FAIL] [Heartbeat] Reconnect failed (attempt {self.reconnect_attempts})")
        return success
    
    def disconnect(self):
        """Close WebSocket connection and stop worker thread."""
        # Stop consumer worker (Edge Case 4)
        self.worker_running = False
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=2.0)
        
        if self.sws:
            try:
                self.sws.close()
            except:
                pass
        self.connected = False


class AngelSession:
    def __init__(self):
        self._obj = None
        self._login_time = None
        self._jwt_token = None
        self._feed_token = None
        self._api_key = None
        self._client_id = None

    def _login(self):
        try:
            import pyotp
            from SmartApi import SmartConnect
            with open(CONFIG_FILE) as f: cfg = json.load(f)
            
            self._api_key = cfg['api_key']
            self._client_id = cfg['client_id']
            
            obj  = SmartConnect(api_key=self._api_key)
            totp = pyotp.TOTP(cfg['totp_secret']).now()
            resp = obj.generateSession(self._client_id, cfg['password'], totp)
            if not resp.get('status'):
                print(f"  Login failed: {resp.get('message')}")
                return None
            
            self._jwt_token = resp['data'].get('jwtToken')
            self._feed_token = obj.getfeedToken()
            self._obj = obj
            self._login_time = datetime.now()
            
            print(f"  Angel One login OK [{self._login_time.strftime('%H:%M:%S')}]")
            return obj
        except Exception as e:
            print(f"  Login error: {e}"); return None

    def get(self):
        if (self._obj is None or
            (self._login_time and
             (datetime.now()-self._login_time).seconds > 14400)):
            return self._login()
        return self._obj
    
    def create_streamer(self) -> MarketStreamer:
        """Create a WebSocket streamer with current session tokens."""
        if not USE_WEBSOCKET or not WEBSOCKET_AVAILABLE:
            return MarketStreamer()  # Dummy streamer
        
        # Ensure we're logged in
        if self._obj is None:
            self.get()
        
        if self._jwt_token and self._feed_token:
            return MarketStreamer(
                auth_token=self._jwt_token,
                api_key=self._api_key,
                client_id=self._client_id,
                feed_token=self._feed_token
            )
        return MarketStreamer()  # Dummy streamer


def fetch_live_candles(session: 'AngelSession', n=250) -> pd.DataFrame | None:
    """Fetch last n 1-min candles from Angel One with rate limiting."""
    # 2026 Edge: Rate limit to prevent 429 errors
    _api_limiter.wait_and_acquire(tokens=2)
    
    obj = session.get()
    if obj is None: return None
    try:
        now = datetime.now()
        fdt = now - timedelta(minutes=n + 30)
        r   = obj.getCandleData({
            "exchange": "NSE", "symboltoken": "99926000",
            "interval": "ONE_MINUTE",
            "fromdate": fdt.strftime("%Y-%m-%d %H:%M"),
            "todate":   now.strftime("%Y-%m-%d %H:%M"),
        })
        if not r or not r.get('data'):
            session._obj = None; return None
        df = pd.DataFrame(r['data'],
             columns=['datetime','open','high','low','close','volume'])
        df['datetime'] = to_ist_naive(pd.to_datetime(df['datetime']))
        for col in ['open','high','low','close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['volume'] = pd.to_numeric(df.get('volume', 0), errors='coerce').fillna(0)
        
        # 2026 Edge: Ghost Candles Forward-Fill
        # On high-volatility days, exchange might miss a tick. Forward-fill missing minutes
        # to keep RSI/EMA calculations consistent (prevents NaN cascades)
        df = df.sort_values('datetime').reset_index(drop=True)
        if len(df) > 1:
            df = df.set_index('datetime').asfreq('1min', method='ffill').reset_index()
        
        return df.dropna(subset=['close']).sort_values('datetime').reset_index(drop=True)
    except Exception as e:
        print(f"  Candle fetch error: {e}"); session._obj = None; return None


def fetch_live_candles_multiday(session: 'AngelSession', days: int = 3) -> pd.DataFrame | None:
    """
    Fetch 1-min candles across multiple trading days by issuing one request
    per calendar day and concatenating.

    Angel One's ONE_MINUTE endpoint returns at most ~500 bars per request
    (~one trading day = 375 bars).  A single request with fromdate N*375
    minutes ago silently truncates to the current session only, returning
    far fewer bars than requested.

    Strategy: request each of the last `days` calendar days separately,
    sleep 2.5s between calls (AB1004 guard), then concatenate and dedup.

    Returns deduplicated, sorted DataFrame or None if all fetches fail.
    """
    obj = session.get()
    if obj is None:
        return None

    frames = []
    today = datetime.now().date()

    for offset in range(days - 1, -1, -1):   # oldest day first
        day = today - timedelta(days=offset)
        if day.weekday() >= 5:                 # skip weekends
            continue
        fdt = datetime(day.year, day.month, day.day, 9, 0)
        tdt = datetime(day.year, day.month, day.day, 15, 31)
        if offset == 0:                        # today: todate = now
            tdt = datetime.now()
        try:
            _api_limiter.wait_and_acquire(tokens=2)
            r = obj.getCandleData({
                "exchange": "NSE", "symboltoken": "99926000",
                "interval": "ONE_MINUTE",
                "fromdate": fdt.strftime("%Y-%m-%d %H:%M"),
                "todate":   tdt.strftime("%Y-%m-%d %H:%M"),
            })
            if not r or not r.get('data'):
                continue
            df = pd.DataFrame(r['data'],
                 columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = to_ist_naive(pd.to_datetime(df['datetime']))
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['volume'] = pd.to_numeric(df.get('volume', 0), errors='coerce').fillna(0)
            df = df.dropna(subset=['close'])
            if not df.empty:
                frames.append(df)
                print(f"    Day {day}: {len(df)} bars fetched")
        except Exception as e:
            print(f"    Day {day}: fetch error — {e}")
        if offset > 0:
            time.sleep(2.5)   # AB1004 guard between day-requests

    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset='datetime', keep='last')
    out = out.sort_values('datetime').reset_index(drop=True)

    # Forward-fill missing 1-min slots within each session (ghost candle fix)
    if len(out) > 1:
        out = out.set_index('datetime').asfreq('1min', method='ffill').reset_index()

    return out.dropna(subset=['close']).reset_index(drop=True)


def fetch_live_htf(session: 'AngelSession', interval: str, n: int) -> pd.DataFrame | None:
    """Fetch higher-TF candles for live context with rate limiting."""
    # 2026 Edge: Rate limit to prevent 429 errors
    _api_limiter.wait_and_acquire(tokens=2)

    obj = session.get()
    if obj is None: return None
    try:
        now = datetime.now()
        # Go back N calendar days (not minutes) to avoid Angel One snapping
        # fromdate to current day when it falls on a weekend/holiday.
        # 5 calendar days guarantees at least 3 trading days of HTF bars.
        fdt = now - timedelta(days=5)
        fdt = fdt.replace(hour=9, minute=0, second=0, microsecond=0)
        r   = obj.getCandleData({
            "exchange": "NSE", "symboltoken": "99926000",
            "interval": interval,
            "fromdate": fdt.strftime("%Y-%m-%d %H:%M"),
            "todate":   now.strftime("%Y-%m-%d %H:%M"),
        })
        if not r or not r.get('data'):
            return None
        df = pd.DataFrame(r['data'],
             columns=['datetime','open','high','low','close','volume'])
        df['datetime'] = to_ist_naive(pd.to_datetime(df['datetime']))
        for col in ['open','high','low','close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna(subset=['close']).sort_values('datetime').reset_index(drop=True)
    except Exception:
        return None


# ==============================================================================
# v4.0: COLD-START SYNC (Pre-flight Data Buffer)
# ==============================================================================

def _fetch_with_retry(fetch_fn, *args, retries: int = 3, delay: float = 3.0, min_rows: int = 10, **kwargs):
    """
    Retry wrapper for Angel One candle fetches during cold-start.

    Angel One occasionally returns None or empty data on the first call due to
    transient session issues. Retrying with a short delay resolves this in
    nearly all cases.

    Parameters
    ----------
    fetch_fn  : fetch_live_candles or fetch_live_htf
    retries   : maximum attempts
    delay     : seconds between attempts
    min_rows  : minimum acceptable DataFrame length
    """
    for attempt in range(1, retries + 1):
        try:
            result = fetch_fn(*args, **kwargs)
            if result is not None and len(result) >= min_rows:
                return result
            row_count = len(result) if result is not None else 0
            # AB1004 rate-limit: result is empty (0 rows) — apply long delay
            # The API returns AB1004 as a response body (not an exception),
            # so we detect it by 0 rows right after a cold-start with many fetches.
            retry_delay = 20.0 if row_count == 0 else delay
            print(f"    Attempt {attempt}/{retries}: got {row_count} rows "
                  f"(need {min_rows}). Retrying in {retry_delay}s...")
            if attempt < retries:
                time.sleep(retry_delay)
        except Exception as e:
            err_str = str(e)
            # AB1004 = rate-limited (old code), AB1019 = "Too many requests" (seen live)
            # Match both error codes and the human-readable message variant
            is_rate_limit = ('AB1004' in err_str or 'AB1019' in err_str or
                             'Too many' in err_str or 'TooMany' in err_str or
                             'too many' in err_str)
            retry_delay = 20.0 if is_rate_limit else delay
            print(f"    Attempt {attempt}/{retries}: exception — {e}. Retrying in {retry_delay}s...")
            if attempt < retries:
                time.sleep(retry_delay)
    return None


def sync_historical_buffer(session: 'AngelSession') -> tuple:
    """
    Cold-start sync: Pre-load historical data before live loop starts.

    1-min fetch uses fetch_live_candles_multiday() — one API call per calendar
    day — to work around Angel One's ~375-bar-per-request cap on ONE_MINUTE.
    A single request for 1000 bars silently truncates to the current session
    only (~200 bars early in the day), causing cold-start failure.

    Returns: (df_1m, df_5m, df_15m) — Pre-processed DataFrames ready for use
    """
    print("\n[Cold-Start Sync] Loading historical buffers...")

    # Pre-sleep before first API call: Angel One enforces a burst window across
    # sessions. If the script was restarted within the same rate-limit window
    # (< 30s since last session's burst), the first fetch hits AB1019 immediately.
    # A 12s wait clears the burst window and avoids wasting a retry slot.
    print("  Waiting 12s to clear API rate-limit window...")
    time.sleep(12)

    # 1. Fetch 2+ days of 15-min candles (with retry)
    print("  Fetching 15-min data (2 days)...")
    df_15m = _fetch_with_retry(fetch_live_htf, session, 'FIFTEEN_MINUTE', 200, min_rows=10)
    if df_15m is None:
        print("  WARNING: 15-min data fetch failed after retries — HTF features will be zeroed")
        df_15m = pd.DataFrame()
    else:
        print(f"  [OK] Loaded {len(df_15m)} bars of 15-min data")

    time.sleep(5)  # AB1004 guard: ~2.5s minimum between API calls; 5s is safe

    # 2. Fetch 2+ days of 5-min candles (with retry)
    print("  Fetching 5-min data (2 days)...")
    df_5m = _fetch_with_retry(fetch_live_htf, session, 'FIVE_MINUTE', 300, min_rows=10)
    if df_5m is None:
        print("  WARNING: 5-min data fetch failed after retries — HTF features will be zeroed")
        df_5m = pd.DataFrame()
    else:
        print(f"  [OK] Loaded {len(df_5m)} bars of 5-min data")

    time.sleep(5)  # AB1004 guard

    # 3. Fetch 1-min candles across multiple days.
    # Angel One's ONE_MINUTE endpoint caps at ~375 bars per request (one session).
    # fetch_live_candles_multiday() issues one request per calendar day and
    # concatenates, giving up to 3 × 375 = 1125 bars without hitting the cap.
    # min_rows is set to 50 (not 300) so a mid-session restart on a short day
    # (e.g. Diwali muhurat trading or first bar after lunch restart) still succeeds.
    print("  Fetching 1-min data (multi-day)...")
    df_1m = None
    for attempt in range(1, 4):
        df_1m = fetch_live_candles_multiday(session, days=3)
        if df_1m is not None and len(df_1m) >= 50:
            break
        print(f"    Attempt {attempt}/3: got {len(df_1m) if df_1m is not None else 0} bars. "
              f"Retrying in 5s...")
        time.sleep(5)
    if df_1m is None or len(df_1m) < 50:
        print("  ERROR: 1-min data fetch failed after retries — cannot start live loop safely")
        return None, None, None
    else:
        print(f"  [OK] Loaded {len(df_1m)} bars of 1-min data ({df_1m['datetime'].iloc[0].date()} "
              f"to {df_1m['datetime'].iloc[-1].date()})")
    
    # 4. Pre-calculate all indicators so they're ready for live loop
    print("  Pre-calculating indicators...")
    try:
        # Add date/minute columns
        df_1m['date'] = df_1m['datetime'].dt.date
        df_1m['minute_of_day'] = (df_1m['datetime'].dt.hour * 60 +
                                  df_1m['datetime'].dt.minute) - (9*60 + 15)

        # Calculate 1-min features
        df_1m = add_1min_features(df_1m)

        # Add HTF features if available
        if not df_5m.empty:
            df_1m = add_htf_features(df_1m, df_5m, 'tf5_', [1, 3, 6])

        if not df_15m.empty:
            df_1m = add_htf_features(df_1m, df_15m, 'tf15_', [1, 4])

        print(f"  [OK] Indicators calculated. Buffer ready with {len(df_1m)} rows")
        print("[Cold-Start Sync] Complete. Live loop can start with full context.\n")

        return df_1m, df_5m, df_15m

    except Exception as e:
        import traceback
        print(f"  ERROR during indicator calculation: {e}")
        traceback.print_exc()
        return None, None, None


def prefetch_instrument_master(session: 'AngelSession') -> bool:
    """
    Download the Angel One instrument master JSON at cold-start.

    Without this, the first real order triggers a 10-15 second HTTP fetch
    inside _get_option_token() while the market is moving.  Doing it here
    at startup means token lookup at trade time is instant (cache hit).

    Called from live_loop() after sync_historical_buffer() completes.
    Returns True if successful, False if the download failed (non-fatal —
    the broker will attempt a fresh download at trade time as fallback).
    """
    try:
        # Import here to avoid circular import at module level
        from ..execution.broker import BrokerOrderManager
        import requests

        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        print("  [Cold-Start] Downloading instrument master (for instant token lookup)...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        instruments = resp.json()

        # Store on module-level singleton so BrokerOrderManager can reuse it
        # without re-downloading.  BrokerOrderManager._get_instrument_data()
        # checks self.instrument_list first — we pre-populate it here.
        # We stash it in a module-level dict that broker.py checks at startup.
        import pandas as pd
        _instrument_cache['data']    = pd.DataFrame(instruments)
        _instrument_cache['updated'] = __import__('datetime').date.today()
        print(f"  [OK] Instrument master loaded: {len(instruments):,} instruments")
        return True
    except Exception as e:
        print(f"  [WARN] Instrument master prefetch failed: {e} — will retry at trade time")
        return False


# Module-level instrument cache shared with broker.py
_instrument_cache: dict = {'data': None, 'updated': None}


