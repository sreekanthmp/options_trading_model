"""Paper trading engine with anti-wick stops and delta-decay tracking."""
import os, time, logging
import numpy as np
import pandas as pd
from datetime import datetime, date
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from ..config import (STOP_LOSS_PCT, TARGET_PCT, DELTA_BASE, THETA_PTS_PER_BAR,
                      LIMIT_BUY_BUFFER_PCT, LIMIT_SELL_BUFFER_PCT)
from .orders import get_lot_size, simulate_limit_order
from ..utils.safeguards import safe_value
from .costs import get_dynamic_theta, calculate_brokerage
from .orders import effective_delta, option_pnl_estimate
from .risk import KillSwitch

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Paper trading engine integrated with the v3.1 signal pipeline.

    Tracks a single option position at a time (mirrors real intraday practice).
    Records every trade with full signal metadata for post-session analysis.
    Exports a CSV at end of day for offline review.

    Features beyond the basic entry/exit template:
      - Slippage-adjusted PnL (uses entry_price from select_option, not mid)
      - Signal metadata stored per trade (confidence, EV, regime, micro-regime)
      - Time-forced exit at 15:15 to avoid expiry risk
      - Rolling intraday stats: running PnL, win rate, drawdown
      - KillSwitch feedback: calls ks.record_trade() on each exit
      - CSV export: one row per trade, written at end of day
    """

    def __init__(self, capital: float):
        self.capital       = capital
        self._position     = {}          # active position dict, empty if flat
        self._trades: list = []          # completed trade log
        self._peak_equity  = capital     # for intraday drawdown tracking
        self._equity       = capital
        
        # 2026 Edge: Anti-Wick Two-Factor Stop
        self._stop_touch_time = None     # When stop was first touched
        self._stop_touch_prices = deque(maxlen=10)  # Last 10 ticks after stop touch
        
        # Paper Trading Validation Tracking
        self._validation_log = []        # Detailed per-trade validation metrics

    # ------------------------------------------------------------------
    # ENTRY
    # ------------------------------------------------------------------
    def enter(self, signal: dict, trade_info: dict, now):
        """Open a paper position. Silently ignored if already in a trade."""
        if self._position:
            return
        
        # Validate trade_info before entering position
        t = trade_info
        required_keys = ['strike', 'option_type', 'entry_price', 'stop_price', 
                         'target_price', 'contracts']
        missing = [k for k in required_keys if k not in t]
        if missing:
            logger.error(f"Cannot enter trade: missing {missing}")
            return
        
        # Validate price values are positive and logical
        if t['entry_price'] <= 0 or t['stop_price'] < 0 or t['target_price'] <= 0:
            logger.error(f"Invalid prices: entry={t['entry_price']}, stop={t['stop_price']}, target={t['target_price']}")
            return
        
        # Validate option type
        if t['option_type'] not in ['CE', 'PE']:
            logger.error(f"Invalid option_type: {t['option_type']}")
            return
        
        # Validate contracts > 0
        if t['contracts'] <= 0:
            logger.error(f"Invalid contracts: {t['contracts']}")
            return

        t = trade_info
        lot_sz = t.get('lot_size_used', t.get('lot_size', get_lot_size(now)))
        self._position = {
            'symbol':       f"NIFTY {t['strike']} {t['option_type']}",
            'option_type':  t['option_type'],
            'strike':       t['strike'],
            'direction':    signal['direction'],
            'entry_price':  t['entry_price'],
            'stop':         t['stop_price'],
            'target':       t['target_price'],
            'peak_ltp':     t['entry_price'],   # tracks highest LTP seen (for trailing stop)
            'lot_size_used': lot_sz,
            'qty':          t['contracts'] * lot_sz,
            'contracts':    t['contracts'],
            'entry_time':   now,
            # Signal metadata
            'avg_conf':     signal.get('avg_conf',    0.0),
            'ev_net':       signal.get('ev_net',      0.0),
            'strength':     signal.get('strength',    ''),
            'regime':       signal.get('regime',      ''),
            'micro_regime': signal.get('micro_regime',''),
            'agreement':    signal.get('agreement',   0.0),
            # Delta-decay tracking
            'dte_mins_entry':  t.get('dte_mins_entry', 0.0),
            'iv_annpct_entry': t.get('iv_annpct_entry', 0.0),
            'spot_entry':      float(signal.get('spot', 0.0)),
            # LIMIT order execution metadata
            'entry_order_type':   t.get('order_type', 'LIMIT'),
            'entry_limit_price':  t.get('limit_price', t['entry_price']),
            'entry_slip_pct':     t.get('limit_slip_pct', 0.0),
            'entry_spread_cost':  t.get('limit_spread_cost', 0.0),
            'entry_fill_prob':    t.get('limit_fill_prob', 1.0),
            'iv_rank_entry':      t.get('iv_rank', 50.0),
            # Expiry flag — used by exit LIMIT simulation for wider spread penalty
            'is_expiry':          bool(signal.get('is_expiry', 0)),
            # LTP source tracking — True = real API price, False = model estimate
            '_using_real_ltp':    False,
        }

        print(f"\n  [PAPER ENTRY]  {self._position['symbol']}  "
              f"{signal['direction']}  "
              f"mid={t.get('est_premium', t['entry_price']):.2f}  "
              f"limit={t.get('limit_price', t['entry_price']):.2f}  "
              f"fill={t['entry_price']:.2f}  "
              f"slip={t.get('limit_slip_pct', 0.0):.2f}%  "
              f"stop={t['stop_price']:.2f}  target={t['target_price']:.2f}  "
              f"contracts={t['contracts']}  conf={signal.get('avg_conf',0):.1%}  "
              f"ev_net={signal.get('ev_net',0):+.4f}")
        
        # Reset anti-wick tracker for new position
        self._stop_touch_time = None
        self._stop_touch_prices = deque(maxlen=10)

    # ------------------------------------------------------------------
    # MARK-TO-MARKET TRACKING (called every bar while in position)
    # ------------------------------------------------------------------
    def track(self, option_ltp: float, now, ks: 'KillSwitch',
              current_row=None, signal_state=None, rng=None):
        """
        Check live LTP against stop/target/time-exit rules.
        option_ltp: estimated current premium (from select_option logic or live feed)
        current_row: optional pd.Series for the current bar (used for anti-wick stop)
        signal_state: optional SignalState object for tracking cooldown periods
        """
        if not self._position:
            return

        p     = self._position
        entry = p['entry_price']
        qty   = p['qty']
        ltp   = option_ltp
        pnl   = (ltp - entry) * qty

        # Time-forced exit: close all positions by 15:15
        import datetime as _dt
        hm = now.hour * 60 + now.minute
        if hm >= 15 * 60 + 15:
            self._exit('TIME_EXIT', ltp, now, ks, signal_state=signal_state, rng=rng)
            return

        # Update peak LTP for trailing stop calculation
        p['peak_ltp'] = max(p.get('peak_ltp', entry), ltp)
        peak_ltp = p['peak_ltp']

        # Time-aware trailing stop tightening
        # WHY: Late-day theta decays faster (gamma risk near expiry).
        # Trail from PEAK LTP (not entry) so profitable positions are protected,
        # not stopped out by normal volatility on a good trade.
        # Example: entry=47, peak=71 → stop trails 25% below peak=53.25, not 25% of 47=35.25
        minute_of_day = hm - (9 * 60 + 15)
        if minute_of_day >= 270:      # after 1:45 PM — trail 25% below peak
            time_adjusted_stop = peak_ltp * (1 - 0.25)
        elif minute_of_day >= 210:    # after 12:45 PM — trail 32% below peak
            time_adjusted_stop = peak_ltp * (1 - 0.32)
        else:
            time_adjusted_stop = p['stop']   # original stop (40% of entry)
        # Only tighten, never loosen vs original stop
        effective_stop = max(p['stop'], time_adjusted_stop)
        if ltp <= effective_stop and effective_stop > p['stop']:
            self._exit('TIME_TIGHTENED_STOP', ltp, now, ks, signal_state=signal_state, rng=rng)
            return

        # 2026 Edge: Anti-Wick Two-Factor Stop Logic
        # Instead of exiting on first tick that touches stop, require:
        #   (a) Stop touched for 2 consecutive 1-min bars (persistent breach), OR
        #   (b) VWAP of last 10 bars below stop (sustained breakdown)
        #   (c) Flash crash: close below stop by 2x option-ATR
        # Using bar count (not wall-clock seconds) makes this identical in live and replay.
        # In live: each bar = ~60 real seconds; 2 bars = ~2 minutes confirmation.
        # In replay: each bar processed instantly; bar count is the right unit.
        close_price = float(current_row.get('close', ltp)) if current_row is not None else ltp
        current_atr = float(current_row.get('atr_14', 0)) if current_row is not None else 0.0
        # Store current spot on position for use by _exit() decomposition
        p['_spot_now']  = close_price
        p['_atr_now']   = current_atr if current_atr > 0 else (entry * 0.005)
        # Convert spot ATR to approximate option ATR using a 0.5 delta proxy
        option_atr  = current_atr * 0.5 if current_atr > 0 else 0.0

        # Check if stop is touched
        stop_touched = ltp <= p['stop']

        if stop_touched:
            # Record touch bar count if first time
            if self._stop_touch_time is None:
                self._stop_touch_time = 0   # repurposed: counts bars since first touch
                print(f"  [Anti-Wick] Stop touched at {ltp:.2f}. Waiting for bar confirmation...")
            else:
                self._stop_touch_time += 1  # increment bar counter each subsequent bar

            # Track prices after stop touch for VWAP calculation
            self._stop_touch_prices.append(ltp)

            bars_since_touch = self._stop_touch_time

            # Calculate VWAP of recent bars after stop touch
            if len(self._stop_touch_prices) >= 3:
                vwap_after_touch = sum(self._stop_touch_prices) / len(self._stop_touch_prices)
            else:
                vwap_after_touch = ltp

            # Two-factor exit conditions:
            # 1. Bar-based: Stop touched for 2+ consecutive bars (persistent)
            # 2. Price-based: VWAP of last 10 bars is below stop (sustained breakdown)
            #    Requires ≥3 bars of data — prevents instant confirmation on first touch
            # 3. Flash crash: Close price below stop by 2x ATR (genuine breakdown)
            bar_confirmed  = bars_since_touch >= 2
            vwap_confirmed = (len(self._stop_touch_prices) >= 3) and (vwap_after_touch <= p['stop'])
            flash_crash    = (option_atr > 0) and (close_price <= p['stop'] - 2.0 * option_atr)

            if bar_confirmed or vwap_confirmed or flash_crash:
                reason = 'TWO_FACTOR_STOP'
                if flash_crash:
                    reason = 'FLASH_CRASH_STOP'
                elif bar_confirmed:
                    reason = 'BAR_CONFIRMED_STOP'
                elif vwap_confirmed:
                    reason = 'VWAP_CONFIRMED_STOP'

                print(f"  [Anti-Wick] Stop confirmed: {reason} (bars={bars_since_touch}, VWAP={vwap_after_touch:.2f})")
                self._exit(reason, ltp, now, ks, signal_state=signal_state, rng=rng)
                return
        else:
            # Stop not touched — reset tracker if it was previously touched (wick recovery)
            if self._stop_touch_time is not None:
                print(f"  [Anti-Wick] Stop recovered! Price bounced from {min(self._stop_touch_prices):.2f} to {ltp:.2f}")
                self._stop_touch_time = None
                self._stop_touch_prices = deque(maxlen=10)

        # Target hit
        if ltp >= p['target']:
            self._exit('TARGET', ltp, now, ks, signal_state=signal_state, rng=rng)
            return

        # Expansion failure exit
        # WHY: Options only pay when volatility is expanding. If ATR contracts
        # after entry, theta will eat a losing position even more. BUT only exit
        # on genuine sustained squeeze while in a loss — never kill a profitable trade.
        # Conditions:
        #   1. Trade open >= 15 bars (ATR5 needs time to stabilize; 5 bars was too noisy)
        #   2. atr_ratio < 0.60 (tighter threshold — genuine squeeze, not a brief quiet bar)
        #   3. Sustained: atr_ratio was also below 0.70 last bar (not a single-bar dip)
        #   4. Position is in a LOSS (pnl_pct < 0) — never exit profitable trades on squeeze
        if current_row is not None:
            bars_open_now = int((now - p['entry_time']).total_seconds() / 60)
            atr_ratio_now = float(current_row.get('atr_ratio', 1.0))
            pnl_pct = (ltp / entry - 1.0)
            prev_atr_ratio = float(p.get('_prev_atr_ratio', 1.0))
            p['_prev_atr_ratio'] = atr_ratio_now   # store for next bar
            squeeze_sustained = (atr_ratio_now < 0.60) and (prev_atr_ratio < 0.70)
            if bars_open_now >= 15 and squeeze_sustained and pnl_pct < 0.0:
                self._exit('EXPANSION_FAILURE', ltp, now, ks, signal_state=signal_state, rng=rng)
                return

        # Running display: show effective_delta and cumulative theta drag.
        bars_so_far  = int((now - p['entry_time']).total_seconds() / 60)
        spot_entry   = float(p.get('spot_entry', close_price))
        spot_move    = ((close_price - spot_entry)
                        if p.get('direction', 'UP') == 'UP'
                        else (spot_entry - close_price))
        delta_now    = effective_delta(bars_so_far, spot_move, current_atr or entry * 0.005,
                                       p.get('direction', 'UP'))
        # Edge Case 3: DTE-Weighted Theta
        dte_entry = p.get('dte_mins_entry', 750.0)
        dte_now   = max(30.0, dte_entry - bars_so_far)  # DTE decreases as time passes
        theta_so_far = bars_so_far * get_dynamic_theta(dte_now)
        print(f"  [PAPER POS]  {p['symbol']}  LTP={ltp:.2f}  "
              f"PnL={pnl:+.2f}  ({(ltp/entry - 1)*100:+.1f}%)  "
              f"delta={delta_now:.3f}  theta={theta_so_far:.1f}pts  "
              f"bars={bars_so_far}")

    # ------------------------------------------------------------------
    # EXIT
    # ------------------------------------------------------------------
    def _exit(self, reason: str, exit_price: float, now, ks: 'KillSwitch', signal_state=None,
              rng: 'np.random.Generator' = None):
        p     = self._position
        entry = p['entry_price']
        qty   = p['qty']

        # Simulate LIMIT SELL order on exit (SEBI: no MARKET orders)
        # For stop exits: market is moving against us — fill prob is high but
        # we may get slight extra slippage as book thins. For target/time exits:
        # market is stable, fill at limit is near-certain.
        is_expiry   = bool(p.get('is_expiry', False))   # stored at entry time
        iv_rank_pos = float(p.get('iv_rank_entry', 50.0))
        using_real  = bool(p.get('_using_real_ltp', False))   # was exit_price from real API?
        exit_limit  = simulate_limit_order(
            mid_price = exit_price,
            side      = 'SELL',
            is_expiry = is_expiry,
            iv_rank   = iv_rank_pos,
            rng       = rng,
        )
        # On stop exits, if LIMIT sell doesn't fill immediately (rare), accept
        # a slightly worse price rather than missing the exit entirely — use limit_price.
        actual_exit = exit_limit['fill_price'] if exit_limit['filled'] else exit_limit['limit_price']
        actual_exit = max(actual_exit, 1.0)  # option can't go negative

        gross_pnl = (actual_exit - entry) * qty

        # Deduct real brokerage + statutory charges (Angel One NSE F&O)
        # This makes paper PnL = live net PnL (what actually hits your account)
        charges = calculate_brokerage(
            entry_price = entry,
            exit_price  = actual_exit,
            qty         = qty,
            is_expiry   = is_expiry,
        )
        pnl = gross_pnl - charges['total_charges']

        # Edge Case 2: Record exit time for post-trade cooldown
        if signal_state is not None:
            signal_state.last_exit_time = now

        # Update equity + peak for drawdown tracking
        self._equity       += pnl
        self._peak_equity   = max(self._peak_equity, self._equity)
        intraday_dd         = (self._equity - self._peak_equity) / (self._peak_equity + 1e-9)

        hold_mins = int((now - p['entry_time']).total_seconds() / 60)

        # Delta-decay decomposition using effective_delta() model:
        #   theta_drag_pts  = bars_open × get_dynamic_theta(dte_now)  (pure time cost, DTE-adjusted)
        #   directional_pts = spot_move × delta_eff                    (favourable move)
        #   These sum to the theoretical PnL per unit at exit.
        spot_entry = float(p.get('spot_entry', 0.0))
        direction  = p.get('direction', 'UP')
        bars_open  = hold_mins   # 1 bar ≈ 1 min on 1-min feed
        atr_now    = float(p.get('_atr_now', entry * 0.005))
        spot_now   = float(p.get('_spot_now', spot_entry))
        
        # Edge Case 3: DTE-Weighted Theta
        dte_entry = p.get('dte_mins_entry', 750.0)
        dte_now   = max(30.0, dte_entry - bars_open)  # Current DTE = entry DTE - elapsed time
        
        if spot_entry > 0 and spot_now > 0:
            spot_move_at_exit = (spot_now - spot_entry
                                 if direction == 'UP'
                                 else spot_entry - spot_now)
        else:
            spot_move_at_exit = 0.0
        delta_at_exit  = effective_delta(bars_open, spot_move_at_exit, atr_now, direction)
        theta_drag_pts = round(bars_open * get_dynamic_theta(dte_now), 2)  # Edge Case 3
        theta_drag_pnl = round(theta_drag_pts * qty, 2)
        dir_pts        = round(spot_move_at_exit * delta_at_exit, 2)
        dir_pnl        = round(dir_pts * qty, 2)

        trade = {
            'symbol':       p['symbol'],
            'option_type':  p['option_type'],
            'strike':       p['strike'],
            'direction':    p['direction'],
            'entry_price':  entry,
            'exit_price':   round(actual_exit, 2),
            'exit_mid':     round(exit_price, 2),   # LTP before limit sim
            'qty':          qty,
            'contracts':    p['contracts'],
            'lot_size_used': p.get('lot_size_used', get_lot_size()),
            'pnl':          round(pnl, 2),
            'pnl_pct':      round((actual_exit / entry - 1) * 100, 2),
            'entry_time':   p['entry_time'].strftime('%H:%M'),
            'exit_time':    now.strftime('%H:%M'),
            'hold_mins':    hold_mins,
            'exit_reason':  reason,
            # Signal metadata
            'avg_conf':     p['avg_conf'],
            'ev_net':       p['ev_net'],
            'strength':     p['strength'],
            'regime':       p['regime'],
            'micro_regime': p['micro_regime'],
            'agreement':    round(p['agreement'], 3),
            'equity_after': round(self._equity, 2),
            'intraday_dd':  round(intraday_dd * 100, 2),
            # Delta-decay fields
            'theta_drag_pts': theta_drag_pts,
            'theta_drag_pnl': theta_drag_pnl,
            'dir_pnl':        dir_pnl,
            'delta_at_exit':  round(delta_at_exit, 4),
            'bars_open':      bars_open,
            # LIMIT order execution details — entry + exit
            'entry_order_type':    p.get('entry_order_type', 'LIMIT'),
            'entry_limit_price':   round(p.get('entry_limit_price', entry), 2),
            'entry_slip_pct':      round(p.get('entry_slip_pct', 0.0), 3),
            'entry_spread_cost':   round(p.get('entry_spread_cost', 0.0), 2),
            'exit_order_type':     'LIMIT',
            'exit_limit_price':    round(exit_limit['limit_price'], 2),
            'exit_slip_pct':       round(exit_limit['slip_pct'] * 100, 3),
            'exit_spread_cost':    round(exit_limit['spread_cost'], 2),
            'total_spread_cost':   round(p.get('entry_spread_cost', 0.0) + exit_limit['spread_cost'], 2),
            # Brokerage + statutory charges (Angel One, NSE F&O)
            'gross_pnl':           round(gross_pnl, 2),
            'brokerage':           charges['brokerage'],
            'gst':                 charges['gst'],
            'stt':                 charges['stt'],
            'txn_charges':         charges['txn_charges'],
            'stamp_duty':          charges['stamp_duty'],
            'total_charges':       charges['total_charges'],
            # LTP source — tells you if this trade used real market prices or model estimates
            # 'REAL' means both entry and exit used actual option LTP from Angel One API
            # 'MODEL' means fallback delta-decay estimate was used (API unavailable)
            'ltp_source': 'REAL' if using_real else 'MODEL',
        }
        self._trades.append(trade)

        # Feed result back to KillSwitch for consecutive-loss gate
        ks.record_trade(pnl)

        won = pnl > 0
        total_spread = p.get('entry_spread_cost', 0.0) + exit_limit['spread_cost']
        ltp_src_tag = '[REAL]' if using_real else '[MODEL]'
        print(f"\n  [PAPER EXIT]  {p['symbol']}  {reason}  "
              f"mid={exit_price:.2f}  fill={actual_exit:.2f}  {ltp_src_tag}  "
              f"spread={total_spread:.2f}pts  charges={charges['total_charges']:.2f}  "
              f"gross={gross_pnl:+.2f}  net={pnl:+.2f}  "
              f"hold={hold_mins}min  "
              f"delta={delta_at_exit:.3f}  theta={theta_drag_pnl:+.2f}  "
              f"equity={self._equity:,.2f}  "
              f"{'WIN' if won else 'LOSS'}")

        self._position = {}

    # ------------------------------------------------------------------
    # FORCE EXIT (e.g. end of day if still open)
    # ------------------------------------------------------------------
    def force_exit(self, exit_price: float, now, ks: 'KillSwitch', signal_state=None, rng=None):
        if self._position:
            self._exit('EOD_FORCE', exit_price, now, ks, signal_state=signal_state, rng=rng)

    # ------------------------------------------------------------------
    # END-OF-DAY SUMMARY + CSV EXPORT
    # ------------------------------------------------------------------
    def end_of_day(self, trade_date):
        if not self._trades:
            print("\n  [PAPER] No trades executed today.")
            return

        trades    = self._trades
        n         = len(trades)
        wins      = [t for t in trades if t['pnl'] > 0]
        losses    = [t for t in trades if t['pnl'] <= 0]
        total_pnl = sum(t['pnl'] for t in trades)
        ret_pct   = total_pnl / self.capital * 100
        wr        = len(wins) / n
        avg_hold  = sum(t['hold_mins'] for t in trades) / n
        worst_dd  = min(t['intraday_dd'] for t in trades)

        # Total theta drag across all trades today
        total_theta = sum(t.get('theta_drag_pnl', 0.0) for t in trades)

        pnls = [t['pnl'] for t in trades]
        import numpy as _np
        sharpe = (_np.mean(pnls) / (_np.std(pnls) + 1e-9)) * _np.sqrt(n) if n > 1 else 0.0

        # LIMIT order execution quality stats
        total_spread_cost  = sum(t.get('total_spread_cost', 0.0) for t in trades)
        total_charges      = sum(t.get('total_charges', 0.0) for t in trades)
        total_gross_pnl    = sum(t.get('gross_pnl', t['pnl']) for t in trades)
        avg_entry_slip     = _np.mean([t.get('entry_slip_pct', 0.0) for t in trades])
        avg_exit_slip      = _np.mean([t.get('exit_slip_pct',  0.0) for t in trades])
        real_ltp_count     = sum(1 for t in trades if t.get('ltp_source') == 'REAL')

        print(f"\n{'='*62}")
        print(f"  PAPER TRADING SUMMARY  --  {trade_date}")
        print(f"{'='*62}")
        print(f"  Trades      : {n}  ({real_ltp_count} used real API LTP, {n-real_ltp_count} model)")
        print(f"  Win rate    : {wr:.1%}  ({len(wins)}W / {len(losses)}L)")
        print(f"  Gross PnL   : {total_gross_pnl:+,.2f}")
        print(f"  Brokerage   : {-total_charges:+,.2f}  (STT+txn+brokerage)")
        print(f"  Spread cost : {-total_spread_cost:+,.2f}  (bid-ask, entry+exit)")
        print(f"  Theta drag  : {total_theta:+,.2f}  (time decay)")
        print(f"  Net PnL     : {total_pnl:+,.2f}  ({ret_pct:+.2f}% of capital)  <- live equivalent")
        print(f"  Avg entry slip: {avg_entry_slip:.3f}%  |  Avg exit slip: {avg_exit_slip:.3f}%")
        print(f"  Avg win     : {(sum(t['pnl'] for t in wins)/len(wins)) if wins else 0:+.2f}")
        print(f"  Avg loss    : {(sum(t['pnl'] for t in losses)/len(losses)) if losses else 0:+.2f}")
        print(f"  Avg hold    : {avg_hold:.0f} min")
        print(f"  Intraday DD : {worst_dd:.2f}%")
        print(f"  Sharpe      : {sharpe:.2f}")
        print(f"  Order type  : LIMIT (SEBI Apr 2026 compliant)")
        print(f"{'='*62}")

        # Per-trade breakdown — entry/exit fill prices + spread cost
        print(f"\n  {'#':<3} {'Time':<12} {'Symbol':<22} {'Dir':<5} "
              f"{'Mid':>7} {'Fill':>7} {'PnL':>8} {'Spread':>7} {'Theta':>7} {'Reason':<12} {'Conf':>6}")
        print(f"  {'-'*116}")
        for i, t in enumerate(trades, 1):
            print(f"  {i:<3} {t['entry_time']}-{t['exit_time']:<7} "
                  f"{t['symbol']:<22} {t['direction']:<5} "
                  f"{t.get('exit_mid', t['exit_price']):>7.2f} "
                  f"{t['exit_price']:>7.2f} "
                  f"{t['pnl']:>+8.2f} "
                  f"{t.get('total_spread_cost', 0.0):>+7.2f} "
                  f"{t.get('theta_drag_pnl', 0.0):>+7.2f} "
                  f"{t['exit_reason']:<12} "
                  f"{t['avg_conf']:>5.1%}")

        # CSV export
        os.makedirs('paper_trades', exist_ok=True)
        csv_path = os.path.join('paper_trades', f"paper_{trade_date}.csv")
        try:
            import csv
            fieldnames = list(trades[0].keys())
            with open(csv_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(trades)
            print(f"\n  Trades saved to {csv_path}")
        except Exception as e:
            print(f"  CSV export failed: {e}")

        # Reset for next day
        self._trades      = []
        self._peak_equity = self._equity

    @property
    def in_position(self) -> bool:
        return bool(self._position)

    @property
    def position_entry_price(self) -> float:
        return self._position.get('entry_price', 0.0)


