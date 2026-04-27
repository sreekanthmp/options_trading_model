"""
TradeLogger — Structured post-mortem logging for live trading.

Writes a JSON-Lines file (one record per event) to logs/trades_YYYY-MM-DD.jsonl.

Captured fields per event:
  Entry:  timestamp_signal, timestamp_order_placed, symbol, direction, spot,
          entry_price, contracts, regime, regime_conf, model_confidence,
          drift_penalty, stop_price, stop_basis, take_profit, expiry_rule,
          features_snapshot (all active features serialised), delta_entry,
          latency_ms, session_trade_number

  Exit:   timestamp_exit, exit_price, exit_reason, pnl_pts, pnl_pct,
          max_adverse_excursion, max_favorable_excursion, bars_held

  Bar:    timestamp, spot, ltp_est, mae_running, mfe_running (written every bar
          while a position is open — for post-mortem equity curve reconstruction)

Usage:
    tl = TradeLogger()              # creates file for today
    tl.log_entry(signal, trade_info, row, regime, regime_conf, latency_ms)
    tl.log_bar(now, spot, ltp_est, mae, mfe)
    tl.log_exit(now, exit_price, exit_reason, entry_price, contracts)
    tl.log_signal_blocked(now, reason, row)   # every skipped signal bar
"""

import json
import logging
import math
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
LOT_SIZE = 65   # NIFTY lot size for PnL estimation (updated April 26, 2025)


def _safe(v: Any) -> Any:
    """Convert numpy types and NaN to JSON-serialisable Python primitives."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if math.isnan(v) else float(v)
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.ndarray,)):
        return [_safe(x) for x in v.tolist()]
    return v


def _now_iso() -> str:
    return datetime.now().isoformat(timespec='milliseconds')


class TradeLogger:
    """
    One instance per live session.  Writes a rotated JSONL file per day.
    Thread-safe for single-process use (all writes are atomic line appends).

    Supports multiple simultaneously open positions (e.g. trade #1 open while
    trade #3 also opens). Each trade's state is stored in _open_trades[trade_num].
    The single-variable approach (_open_entry_price etc.) was overwritten when a
    second trade opened, zeroing trade #1's MAE/MFE and causing missing BAR events.
    """

    def __init__(self, log_dir: str = LOG_DIR):
        os.makedirs(log_dir, exist_ok=True)
        today = datetime.now().strftime('%Y-%m-%d')
        self._path = os.path.join(log_dir, f'trades_{today}.jsonl')
        self._trade_num = 0
        # Per-trade open state: trade_num -> {entry_time, entry_price, direction, contracts, mae, mfe}
        self._open_trades: dict = {}
        logger.info(f"[TradeLogger] Writing to {self._path}")

    # -----------------------------------------------------------------------
    # Core write helper
    # -----------------------------------------------------------------------
    def _write(self, record: dict):
        try:
            with open(self._path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, default=_safe) + '\n')
        except Exception as e:
            logger.error(f"[TradeLogger] Write failed: {e}")

    # -----------------------------------------------------------------------
    # Trade entry
    # -----------------------------------------------------------------------
    def log_entry(
        self,
        signal: dict,
        trade_info: dict,
        row: pd.Series,
        regime: int,
        regime_conf: float,
        latency_ms: float,
        active_features: list | None = None,
        now: datetime | None = None,
        session_regime: str = '',
        session_regime_score: float = 0.0,
    ):
        self._trade_num += 1
        entry_ts    = now.isoformat(timespec='milliseconds') if now is not None else _now_iso()
        entry_price = float(trade_info.get('entry_price', row.get('close', 0)))
        direction   = signal.get('direction', '?')
        contracts   = int(trade_info.get('contracts', 0))

        # Store per-trade state so concurrent open positions don't overwrite each other.
        self._open_trades[self._trade_num] = {
            'entry_time':   entry_ts,
            'entry_price':  entry_price,
            'direction':    direction,
            'contracts':    contracts,
            'mae':          0.0,
            'mfe':          0.0,
        }

        features_snap = {}
        if active_features:
            for f in active_features:
                v = row.get(f)
                features_snap[f] = _safe(v)

        record = {
            'event':                   'ENTRY',
            'timestamp_signal':        entry_ts,
            'timestamp_order_placed':  _now_iso(),
            'session_trade_number':    self._trade_num,
            'symbol':                  trade_info.get('symbol', 'NIFTY'),
            'option_type':             trade_info.get('option_type', '?'),
            'strike':                  _safe(trade_info.get('strike')),
            'direction':               direction,
            'spot_at_entry':           _safe(row.get('close')),
            'entry_price':             _safe(entry_price),
            'contracts':               contracts,
            'regime':                  regime,
            'regime_conf':             _safe(regime_conf),
            'model_confidence':        _safe(signal.get('avg_conf')),
            'drift_penalty':           _safe(signal.get('drift_penalty', 1.0)),
            'stop_price':              _safe(trade_info.get('stop_price')),
            'stop_basis':              trade_info.get('stop_basis', 'UNKNOWN'),
            'take_profit':             _safe(trade_info.get('take_profit')),
            'delta_entry':             _safe(trade_info.get('delta_entry')),
            'expiry_rule_tag':         trade_info.get('expiry_rule_tag', ''),
            'latency_ms':              _safe(latency_ms),
            'minute_of_day':           _safe(row.get('minute_of_day')),
            'atr_14':                  _safe(row.get('atr_14')),
            'session_regime':          session_regime,
            'session_regime_score':    _safe(session_regime_score),
            'features_snapshot':       features_snap,
        }
        self._write(record)
        logger.info(f"[TradeLogger] ENTRY #{self._trade_num}: "
                    f"{direction} {contracts}x @ {entry_price:.2f}")

    # -----------------------------------------------------------------------
    # Per-bar mark-to-market (while position is open)
    # -----------------------------------------------------------------------
    def log_bar(self, now: datetime, spot: float, ltp_est: float,
                mae: float, mfe: float, trade_num: int | None = None):
        # Update MAE/MFE for the specified trade (or the most recently opened one).
        # Using trade_num avoids updating a different open trade's running stats.
        _tnum = trade_num if trade_num is not None else self._trade_num
        if _tnum in self._open_trades:
            self._open_trades[_tnum]['mae'] = mae
            self._open_trades[_tnum]['mfe'] = mfe
        record = {
            'event':             'BAR',
            'timestamp':         now.isoformat(timespec='milliseconds'),
            'session_trade_number': _tnum,
            'spot':              _safe(spot),
            'ltp_est':           _safe(ltp_est),
            'mae_running':       _safe(mae),
            'mfe_running':       _safe(mfe),
        }
        self._write(record)

    # -----------------------------------------------------------------------
    # Trade exit
    # -----------------------------------------------------------------------
    def log_exit(
        self,
        now: datetime,
        exit_price: float,
        exit_reason: str,
        entry_price: float | None = None,
        contracts: int | None = None,
        trade_num: int | None = None,
    ):
        # Identify which open trade this exit belongs to.
        # Caller can pass trade_num explicitly; otherwise fall back to the most recently opened.
        _tnum = trade_num if trade_num is not None else self._trade_num
        _trade_state = self._open_trades.get(_tnum, {})

        ep        = entry_price if entry_price is not None else _trade_state.get('entry_price', 0.0)
        qty       = contracts   if contracts   is not None else _trade_state.get('contracts', 0)
        direction = _trade_state.get('direction', self._open_trades.get(self._trade_num, {}).get('direction', '?'))
        entry_ts  = _trade_state.get('entry_time', None)
        # Carry running MAE/MFE from per-trade state (not a shared variable that gets overwritten)
        mae       = _trade_state.get('mae', 0.0)
        mfe       = _trade_state.get('mfe', 0.0)

        pnl_pts = exit_price - ep
        pnl_pct = pnl_pts / (ep + 1e-9) * 100

        record = {
            'event':                  'EXIT',
            'timestamp_exit':         now.isoformat(timespec='milliseconds'),
            'trade_entry_timestamp':  entry_ts,
            'session_trade_number':   _tnum,
            'direction':              direction,
            'entry_price':            _safe(ep),
            'exit_price':             _safe(exit_price),
            'exit_reason':            exit_reason,
            'contracts':              qty,
            'pnl_pts':                _safe(pnl_pts),
            'pnl_pct':                _safe(pnl_pct),
            'max_adverse_excursion':  _safe(mae),
            'max_favorable_excursion':_safe(mfe),
        }
        self._write(record)
        logger.info(f"[TradeLogger] EXIT #{_tnum}: {exit_reason} | "
                    f"PnL {pnl_pts:+.2f}pts ({pnl_pct:+.1f}%)")
        # Remove this trade's state from open dict
        self._open_trades.pop(_tnum, None)

    # -----------------------------------------------------------------------
    # Blocked signal (for analysis of gate hit rates)
    # -----------------------------------------------------------------------
    def log_signal_blocked(self, now: datetime, reason: str, row: pd.Series):
        record = {
            'event':         'BLOCKED',
            'timestamp':     now.isoformat(timespec='milliseconds'),
            'reason':        reason,
            'minute_of_day': _safe(row.get('minute_of_day')),
            'close':         _safe(row.get('close')),
        }
        self._write(record)

    # -----------------------------------------------------------------------
    # Safety events (bar errors, shutdown requests, etc.)
    # -----------------------------------------------------------------------
    def log_safety_event(self, event_type: str, detail: str):
        record = {
            'event':      f'SAFETY_{event_type.upper()}',
            'timestamp':  _now_iso(),
            'detail':     detail,
        }
        self._write(record)
        logger.warning(f"[TradeLogger] Safety event {event_type}: {detail}")

    # -----------------------------------------------------------------------
    # End-of-day summary
    # -----------------------------------------------------------------------
    def log_eod(self, equity: float, day_pnl: float, trades: int,
                safety_status: dict | None = None):
        record = {
            'event':         'EOD_SUMMARY',
            'timestamp':     _now_iso(),
            'equity':        _safe(equity),
            'day_pnl':       _safe(day_pnl),
            'trades_today':  trades,
            'safety_status': safety_status or {},
        }
        self._write(record)
        logger.info(f"[TradeLogger] EOD: {trades} trades, PnL {day_pnl:+.2f}, equity {equity:.2f}")
