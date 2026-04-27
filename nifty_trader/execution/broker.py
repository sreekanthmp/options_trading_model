"""Real broker order execution for AngelOne (SmartAPI).

CRITICAL: This module places REAL ORDERS with REAL MONEY.
Only called when --mode live (paper_mode=False).

Safety Features:
  1. Order validation before submission
  2. Symbol token lookup with validation
  3. Order confirmation logging
  4. Position tracking to prevent duplicate orders
  5. Emergency flatten capability
"""
import logging
import time
from datetime import date, datetime
from typing import Optional
import pandas as pd
import requests

logger = logging.getLogger(__name__)


def _tick(price: float) -> str:
    """Round price to nearest 0.05 (NSE F&O tick size) and return as string."""
    return str(round(round(price / 0.05) * 0.05, 2))


class BrokerOrderManager:
    """
    Manages real broker order placement for live trading.
    
    Wraps AngelOne SmartAPI with safety checks and logging.
    Tracks open positions to prevent duplicate orders.
    """
    
    def __init__(self, session: 'AngelSession', capital: float):
        """
        Args:
            session: AngelSession instance (logged in)
            capital: Trading capital for position sizing validation
        """
        self.session = session
        self.capital = capital
        self._open_position = None  # Tracks current open position
        self._orders_today = []     # All orders submitted today
        self._symbol_cache = {}     # Cache for option symbol tokens
        
        # Instrument master for token lookup
        self.instrument_list = None
        self.instrument_last_updated = None
        self.instrument_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        
    def place_entry_order(self, trade_info: dict, signal: dict, now: datetime,
                         is_expiry: bool = False) -> Optional[dict]:
        """
        Place a REAL BUY order for an option contract.
        
        Args:
            trade_info: Position details from select_option()
            signal: Signal dict with direction, spot, etc.
            now: Current datetime
            
        Returns:
            dict with order details if successful, None if failed
        """
        if self._open_position is not None:
            logger.warning("[Broker] Already in position - rejecting new entry order")
            return None
        
        # Extract order parameters
        option_type = trade_info.get('option_type', 'CE')
        strike = int(trade_info.get('strike', 0))
        contracts = int(trade_info.get('contracts', 1))
        lot_size = int(trade_info.get('lot_size_used', 65))
        limit_price = float(trade_info.get('entry_price', 0))
        
        # Validate parameters
        if strike <= 0 or contracts <= 0 or limit_price <= 0:
            logger.error(f"[Broker] Invalid order params: strike={strike}, contracts={contracts}, price={limit_price}")
            return None
        
        # Get option symbol token
        symbol_token = self._get_option_token(strike, option_type, now)
        if symbol_token is None:
            logger.error(f"[Broker] Failed to get symbol token for {strike} {option_type}")
            return None
        
        # Build trading symbol (e.g., "NIFTY10MAR2624600CE")
        trading_symbol = self._build_trading_symbol(strike, option_type, now)
        
        # Calculate quantity
        quantity = contracts * lot_size
        
        # Build order params (Angel One SmartAPI format)
        order_params = {
            "variety": "NORMAL",
            "tradingsymbol": trading_symbol,
            "symboltoken": symbol_token,
            "transactiontype": "BUY",
            "exchange": "NFO",  # NSE Futures & Options
            "ordertype": "LIMIT",
            "producttype": "INTRADAY",
            "duration": "DAY",
            "price": _tick(limit_price),
            "squareoff": "0",
            "stoploss": "0",
            "quantity": str(quantity)
        }
        
        # CRITICAL: Final confirmation before placing real order
        logger.critical(
            f"\n{'='*70}\n"
            f"  [BROKER] PLACING REAL ORDER\n"
            f"  Symbol: {trading_symbol}\n"
            f"  Direction: {signal.get('direction')} ({option_type})\n"
            f"  Strike: {strike}\n"
            f"  Quantity: {quantity} ({contracts} lots × {lot_size})\n"
            f"  Type: LIMIT BUY @ Rs {limit_price:.2f}\n"
            f"  Notional: Rs {limit_price * quantity:,.2f}\n"
            f"  Confidence: {signal.get('avg_conf', 0):.1%}\n"
            f"{'='*70}"
        )
        
        # Place order via SmartAPI (with one session re-login retry on failure)
        try:
            api = self.session.get()
            if api is None:
                logger.error("[Broker] SmartAPI session not available")
                return None

            response = api.placeOrder(order_params)

            # Angel One SmartAPI returns the order ID as a plain string on success
            # e.g. '260310000365813' — this IS the order ID, not an error.
            # Only treat as error if it's clearly an error message (non-numeric string).
            if isinstance(response, str):
                if response.strip().isdigit():
                    # Valid order ID returned directly as string
                    order_id = response.strip()
                    logger.critical(f"[Broker] ENTRY ORDER PLACED - Order ID: {order_id} - polling for fill...")
                else:
                    # Actual error string — attempt re-login once
                    logger.error(f"[Broker] placeOrder returned error string: {response!r} — re-logging in")
                    self.session._obj = None
                    api = self.session.get()
                    if api is None:
                        logger.error("[Broker] Re-login failed — cannot place order")
                        return None
                    response = api.placeOrder(order_params)
                    if isinstance(response, str) and response.strip().isdigit():
                        order_id = response.strip()
                        logger.critical(f"[Broker] ENTRY ORDER PLACED after re-login - Order ID: {order_id} - polling for fill...")
                    elif isinstance(response, str):
                        logger.error(f"[Broker] Still error after re-login: {response!r}")
                        return None
                    elif isinstance(response, dict) and response.get('status') and response.get('data'):
                        order_id = response['data'].get('orderid')
                        logger.critical(f"[Broker] ENTRY ORDER PLACED after re-login - Order ID: {order_id} - polling for fill...")
                    else:
                        logger.error(f"[Broker] Order failed after re-login: {response}")
                        return None
            elif isinstance(response, dict) and response.get('status') and response.get('data'):
                order_id = response['data'].get('orderid')
                logger.critical(f"[Broker] ENTRY ORDER PLACED - Order ID: {order_id} - polling for fill...")

            if order_id:

                # Poll orderBook to confirm fill before recording position (max 30s / 10 polls)
                actual_fill_price = limit_price  # default if poll times out
                fill_confirmed = False
                for _attempt in range(10):
                    time.sleep(3)
                    try:
                        book = api.orderBook()
                        if book and book.get('status') and book.get('data'):
                            for o in book['data']:
                                if str(o.get('orderid')) == str(order_id):
                                    status = o.get('orderstatus', '').upper()
                                    if status == 'COMPLETE':
                                        actual_fill_price = float(o.get('averageprice', limit_price) or limit_price)
                                        fill_confirmed = True
                                        logger.critical(f"[Broker] ENTRY FILLED @ Rs {actual_fill_price:.2f}")
                                    elif status in ('REJECTED', 'CANCELLED'):
                                        logger.error(f"[Broker] ENTRY ORDER {status} — no position taken")
                                        return None
                                    break
                    except Exception as _pe:
                        logger.warning(f"[Broker] Entry poll attempt {_attempt+1} failed: {_pe}")
                    if fill_confirmed:
                        break

                if not fill_confirmed:
                    logger.warning("[Broker] Entry fill not confirmed after 30s — assuming filled at limit price")

                limit_price = actual_fill_price  # use confirmed fill price for stop/target offsets

                # Track open position
                # iv_annpct_entry: store at entry so MTM uses stable IV (not bar-by-bar iv_proxy)
                _iv_entry = float(trade_info.get('iv_annpct_entry',
                                  signal.get('iv_annpct_entry', 0.0)))
                self._open_position = {
                    'order_id': order_id,
                    'symbol': trading_symbol,
                    'token': symbol_token,
                    'option_type': option_type,
                    'strike': strike,
                    'quantity': quantity,
                    'contracts': contracts,
                    'entry_price': limit_price,
                    'entry_time': now,
                    'direction': signal.get('direction'),
                    'stop_price': float(trade_info.get('stop_price', 0)),
                    'target_price': float(trade_info.get('target_price', 0)),
                    'is_expiry': is_expiry,           # Fix 4: expiry rules in exit path
                    'iv_annpct_entry': _iv_entry,     # Fix 3: stable IV for MTM BS fallback
                    'peak_ltp': limit_price,           # initialise for trailing stop
                }
                
                # Log order
                self._orders_today.append({
                    'time': now,
                    'order_id': order_id,
                    'action': 'BUY',
                    'symbol': trading_symbol,
                    'quantity': quantity,
                    'price': limit_price,
                    'response': response
                })
                
                return self._open_position

            return None  # order_id was None — already logged above

        except Exception as e:
            logger.exception(f"[Broker] Exception placing order: {e}")
            return None

    def place_exit_order(self, exit_reason: str, now: datetime, 
                        current_ltp: float = 0) -> Optional[dict]:
        """
        Place a REAL SELL order to close the position.
        
        Args:
            exit_reason: Why exiting (STOP_LOSS, TARGET, TIME_EXIT, etc.)
            now: Current datetime
            current_ltp: Current option LTP (for limit price)
            
        Returns:
            dict with exit details if successful, None if failed
        """
        if self._open_position is None:
            logger.warning("[Broker] No open position to exit")
            return None
        
        pos = self._open_position
        
        # Determine limit price for exit
        # Use current LTP as limit — placing AT LTP is a marketable LIMIT order
        # (hits the best bid immediately). Discounting below LTP risks rejection
        # when the bid is at LTP, especially during stop exits in fast-moving markets.
        if current_ltp > 0:
            limit_price = current_ltp   # marketable limit at LTP
        else:
            limit_price = pos['entry_price'] * 0.95  # 5% below entry if no LTP
        
        limit_price = max(limit_price, 1.0)  # Floor at Re 1
        
        # Build order params
        order_params = {
            "variety": "NORMAL",
            "tradingsymbol": pos['symbol'],
            "symboltoken": pos['token'],
            "transactiontype": "SELL",
            "exchange": "NFO",
            "ordertype": "LIMIT",
            "producttype": "INTRADAY",
            "duration": "DAY",
            "price": _tick(limit_price),
            "squareoff": "0",
            "stoploss": "0",
            "quantity": str(pos['quantity'])
        }
        
        logger.critical(
            f"\n{'='*70}\n"
            f"  [BROKER] CLOSING POSITION\n"
            f"  Symbol: {pos['symbol']}\n"
            f"  Reason: {exit_reason}\n"
            f"  Quantity: {pos['quantity']}\n"
            f"  Type: LIMIT SELL @ Rs {limit_price:.2f}\n"
            f"  Entry: Rs {pos['entry_price']:.2f}\n"
            f"  Hold Time: {(now - pos['entry_time']).seconds // 60} min\n"
            f"{'='*70}"
        )
        
        try:
            api = self.session.get()
            if api is None:
                logger.error("[Broker] SmartAPI session not available for exit")
                return None

            response = api.placeOrder(order_params)

            # Angel One returns order ID as plain string on success
            if isinstance(response, str) and response.strip().isdigit():
                order_id = response.strip()
            elif isinstance(response, dict) and response.get('status') and response.get('data'):
                order_id = response['data'].get('orderid')
            else:
                error_msg = response.get('message', str(response)) if isinstance(response, dict) else str(response)
                logger.error(f"[Broker] EXIT ORDER REJECTED: {error_msg}")
                return None

            logger.critical(f"[Broker] EXIT ORDER PLACED - Order ID: {order_id} - polling for fill...")

            # Poll orderBook until COMPLETE or REJECTED (max 30s / 10 polls)
            fill_price = limit_price  # default to limit price if poll fails
            fill_confirmed = False
            for _attempt in range(10):
                time.sleep(3)
                try:
                    book = api.orderBook()
                    if book and book.get('status') and book.get('data'):
                        for o in book['data']:
                            if str(o.get('orderid')) == str(order_id):
                                status = o.get('orderstatus', '').upper()
                                if status == 'COMPLETE':
                                    fill_price = float(o.get('averageprice', limit_price) or limit_price)
                                    fill_confirmed = True
                                    logger.critical(f"[Broker] EXIT FILLED @ Rs {fill_price:.2f} (Order {order_id})")
                                elif status in ('REJECTED', 'CANCELLED'):
                                    logger.error(f"[Broker] EXIT ORDER {status} - position still open!")
                                    # Position NOT cleared — return None so caller escalates
                                    return None
                                break
                except Exception as _pe:
                    logger.warning(f"[Broker] Poll attempt {_attempt+1} failed: {_pe}")
                if fill_confirmed:
                    break

            if not fill_confirmed:
                logger.warning(f"[Broker] Exit fill not confirmed after 30s — assuming filled at limit price")

            # Calculate actual P&L using fill price
            estimated_pnl = (fill_price - pos['entry_price']) * pos['quantity']

            exit_data = {
                'order_id': order_id,
                'exit_price': fill_price,
                'exit_time': now,
                'exit_reason': exit_reason,
                'estimated_pnl': estimated_pnl,
                'entry_price': pos['entry_price'],
                'contracts': pos['contracts'],
                'fill_confirmed': fill_confirmed,
            }

            # Log exit order
            self._orders_today.append({
                'time': now,
                'order_id': order_id,
                'action': 'SELL',
                'symbol': pos['symbol'],
                'quantity': pos['quantity'],
                'price': fill_price,
                'response': response,
                'exit_reason': exit_reason
            })

            # Clear position only after fill confirmed (or timeout fallback)
            self._open_position = None

            return exit_data

        except Exception as e:
            logger.exception(f"[Broker] Exception placing exit order: {e}")
            return None
    
    def emergency_flatten(self, reason: str) -> bool:
        """
        Emergency position exit using an aggressive LIMIT order (not MARKET).

        SEBI April 2026 bans MARKET orders on all segments including NFO.
        We use LIMIT at LTP * 0.97 (3% below last known LTP) — for ATM NIFTY
        options this is 1-4 pts below mid, which fills within seconds in normal
        liquidity. If rejected, retries once at LTP * 0.94 (wider buffer).

        Polls orderBook for up to 15s to confirm fill before clearing local state.
        Returns True if flatten order was placed (fill may be in-flight), False
        if both attempts were rejected by the broker.
        """
        if self._open_position is None:
            return True  # Already flat

        pos = self._open_position

        # Fetch real LTP for limit price — using entry price as base is wrong
        # if market has moved significantly (e.g. stop exit after -30% move).
        # Try to get real LTP first; fall back to entry price if API fails.
        entry_price = pos.get('entry_price', 0.0)
        real_ltp_now = 0.0
        try:
            from ..data.external_data import fetch_option_ltp as _fol
            real_ltp_now = _fol(self.session, int(pos.get('strike', 0)),
                                pos.get('option_type', 'CE'), force_fresh=True)
        except Exception:
            pass
        base_price = real_ltp_now if real_ltp_now > 1.0 else entry_price

        logger.critical(
            f"\n{'='*70}\n"
            f"  [BROKER] EMERGENCY FLATTEN\n"
            f"  Reason: {reason}\n"
            f"  Symbol: {pos['symbol']}\n"
            f"  Quantity: {pos['quantity']}\n"
            f"  Base LTP: Rs {base_price:.2f} ({'real' if real_ltp_now > 1.0 else 'entry fallback'})\n"
            f"{'='*70}"
        )

        try:
            api = self.session.get()
            if api is None:
                logger.error("[Broker] SmartAPI not available for emergency flatten!")
                return False

            order_id = None
            for attempt, discount in enumerate([1.00, 0.97]):
                lp = max(round(base_price * discount, 2), 1.0)
                order_params = {
                    "variety": "NORMAL",
                    "tradingsymbol": pos['symbol'],
                    "symboltoken": pos['token'],
                    "transactiontype": "SELL",
                    "exchange": "NFO",
                    "ordertype": "LIMIT",
                    "producttype": "INTRADAY",
                    "duration": "DAY",
                    "price": _tick(lp),
                    "squareoff": "0",
                    "stoploss": "0",
                    "quantity": str(pos['quantity'])
                }
                response = api.placeOrder(order_params)
                # Angel One returns order ID as plain string on success
                if isinstance(response, str) and response.strip().isdigit():
                    order_id = response.strip()
                    logger.critical(f"[Broker] Emergency LIMIT order placed @ Rs {lp:.2f} - ID: {order_id}")
                    break
                elif isinstance(response, dict) and response.get('status') and response.get('data'):
                    order_id = response['data'].get('orderid')
                    logger.critical(f"[Broker] Emergency LIMIT order placed @ Rs {lp:.2f} - ID: {order_id}")
                    break
                else:
                    err = response.get('message') if isinstance(response, dict) else str(response)
                    logger.error(f"[Broker] Emergency attempt {attempt+1} REJECTED: {err}")
                    if attempt == 0:
                        time.sleep(1)

            if order_id is None:
                logger.error("[Broker] Both emergency flatten attempts rejected - position may still be open!")
                return False

            # Poll for fill confirmation (max 15s / 5 polls — LIMIT fills fast for ATM options)
            for _ in range(5):
                time.sleep(3)
                try:
                    book = api.orderBook()
                    if book and book.get('status') and book.get('data'):
                        for o in book['data']:
                            if str(o.get('orderid')) == str(order_id):
                                status = o.get('orderstatus', '').upper()
                                if status == 'COMPLETE':
                                    logger.critical(f"[Broker] Emergency flatten CONFIRMED @ Rs {o.get('averageprice')}")
                                    self._open_position = None
                                    return True
                                elif status in ('REJECTED', 'CANCELLED'):
                                    logger.error(f"[Broker] Emergency flatten order {status} - position still open!")
                                    return False
                                break
                except Exception:
                    pass

            # Timeout — order placed but fill unconfirmed; assume filled (LIMIT near bid for ATM)
            logger.warning("[Broker] Emergency flatten: fill unconfirmed after 15s — assuming filled")
            self._open_position = None
            return True

        except Exception as e:
            logger.exception(f"[Broker] Exception during emergency flatten: {e}")
            return False
    
    def is_in_position(self) -> bool:
        """Check if currently holding a position."""
        return self._open_position is not None
    
    def get_position(self) -> Optional[dict]:
        """Get current open position details."""
        return self._open_position
    
    def _get_instrument_data(self):
        """
        Downloads and caches the instrument master from Angel One.

        Checks cold-start prefetch cache first (populated by prefetch_instrument_master()
        called at live.py startup), then falls back to downloading if needed.
        Refreshes automatically if more than 24 hours old.
        """
        from datetime import timedelta

        # Check cold-start prefetch cache first (avoids 10-15s HTTP fetch at trade time)
        if self.instrument_list is None:
            try:
                from ..data.websocket import _instrument_cache
                if _instrument_cache.get('data') is not None:
                    self.instrument_list = _instrument_cache['data']
                    self.instrument_last_updated = _instrument_cache['updated']
                    logger.info("[Broker] Using pre-fetched instrument master from cold-start cache")
            except Exception:
                pass  # Fall through to download below

        # Check if instance cache is fresh (less than 24 hours old)
        if self.instrument_list is not None and self.instrument_last_updated is not None:
            age = datetime.now().date() - self.instrument_last_updated
            if age < timedelta(days=1):
                return self.instrument_list

        # Download fresh data
        logger.info("[Broker] Downloading instrument master from Angel One...")
        try:
            response = requests.get(self.instrument_url, timeout=30)
            response.raise_for_status()
            
            # Parse JSON and create DataFrame for fast searching
            instruments = response.json()
            self.instrument_list = pd.DataFrame(instruments)
            self.instrument_last_updated = datetime.now().date()
            
            logger.info(f"[Broker] Loaded {len(self.instrument_list)} instruments")
            return self.instrument_list
            
        except Exception as e:
            logger.exception(f"[Broker] Failed to download instrument master: {e}")
            # Return stale cache if available
            if self.instrument_list is not None:
                logger.warning("[Broker] Using stale instrument cache")
                return self.instrument_list
            return None
    
    def _get_option_token(self, strike: int, option_type: str, now: datetime) -> Optional[str]:
        """
        Fetch symbol token for an option contract from AngelOne.
        
        Uses cached instrument master for fast O(1) lookup.
        ENHANCED: Retries with fresh instrument download on first failure.
        
        Args:
            strike: Strike price (e.g., 24600)
            option_type: 'CE' or 'PE'
            now: Current datetime for expiry calculation
            
        Returns:
            Symbol token string (e.g., '41234') or None if not found
        """
        # Build trading symbol first
        trading_symbol = self._build_trading_symbol(strike, option_type, now)
        
        # Check cache first
        if trading_symbol in self._symbol_cache:
            return self._symbol_cache[trading_symbol]
        
        # Try up to 2 times: once with cache, once with fresh download
        for attempt in range(2):
            # Get instrument data
            df = self._get_instrument_data()
            if df is None or df.empty:
                logger.error(f"[Broker] Instrument master not available (attempt {attempt+1}/2)")
                if attempt == 0:
                    # Force refresh on first failure
                    logger.warning("[Broker] Forcing instrument master refresh...")
                    self.instrument_list = None
                    self.instrument_last_updated = None
                    continue
                return None
            
            # Filter for NFO (Futures & Options segment) and exact symbol match
            try:
                result = df[(df['symbol'] == trading_symbol) & (df['exch_seg'] == 'NFO')]
                
                if not result.empty:
                    token = str(result.iloc[0]['token'])
                    self._symbol_cache[trading_symbol] = token  # Cache for future use
                    logger.info(f"[Broker] Found token for {trading_symbol}: {token}")
                    return token
                else:
                    if attempt == 0:
                        logger.warning(f"[Broker] Token not found for {trading_symbol} in cache, refreshing...")
                        self.instrument_list = None
                        self.instrument_last_updated = None
                        continue
                    else:
                        logger.error(f"[Broker] Token not found for {trading_symbol} after refresh")
                        logger.error(f"[Broker] Ensure option exists and trading symbol is correct")
                        return None
                    
            except KeyError as e:
                logger.exception(f"[Broker] Instrument data missing expected column: {e}")
                if attempt == 0:
                    self.instrument_list = None
                    continue
                return None
        
        return None
    
    def _build_trading_symbol(self, strike: int, option_type: str, now: datetime) -> str:
        """
        Build trading symbol string for NIFTY options.
        
        Format: NIFTY[DDMMMYY][STRIKE][CE/PE]
        Example: NIFTY10MAR2624600CE
        
        Note: NIFTY weekly options expire on Tuesdays (changed from Thursday in Sep 2024)
        """
        # Find next Tuesday expiry
        from datetime import timedelta
        days_ahead = (1 - now.weekday()) % 7  # 1 = Tuesday
        # Market closes at 15:30 — options expire at 15:30 on Tuesday.
        # After 15:30 on Tuesday, the weekly contract has expired; next contract = next week.
        if days_ahead == 0 and (now.hour > 15 or (now.hour == 15 and now.minute >= 30)):
            days_ahead = 7
        expiry_date = (now + timedelta(days=days_ahead)).date()
        
        # Format: NIFTY10MAR2624600CE
        day = expiry_date.strftime('%d')
        month = expiry_date.strftime('%b').upper()
        year = expiry_date.strftime('%y')
        
        symbol = f"NIFTY{day}{month}{year}{strike}{option_type}"
        return symbol
    
    def reset_day(self):
        """Reset daily counters at start of new trading day."""
        self._orders_today = []
        # _open_position NOT reset - positions carry across days if held
