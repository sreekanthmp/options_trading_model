"""Paper trading engine with anti-wick stops and delta-decay tracking."""
import os, time, logging
import numpy as np
import pandas as pd
from datetime import datetime, date
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from ..config import STOP_LOSS_PCT, TARGET_PCT, DELTA_BASE, THETA_PTS_PER_BAR
from .orders import get_lot_size
from ..utils.safeguards import safe_value
from .costs import get_dynamic_theta
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
        self._position = {
            'symbol':       f"NIFTY {t['strike']} {t['option_type']}",
            'option_type':  t['option_type'],
            'strike':       t['strike'],
            'direction':    signal['direction'],
            'entry_price':  t['entry_price'],
            'stop':         t['stop_price'],
            'target':       t['target_price'],
            'lot_size_used': t.get('lot_size_used', t.get('lot_size', get_lot_size(now))),
            'qty':          t['contracts'] * t.get('lot_size_used', t.get('lot_size', get_lot_size(now))),
            'contracts':    t['contracts'],
            'entry_time':   now,
            # Signal metadata
            'avg_conf':     signal.get('avg_conf',    0.0),
            'ev_net':       signal.get('ev_net',      0.0),
            'strength':     signal.get('strength',    ''),
            'regime':       signal.get('regime',      ''),
            'micro_regime': signal.get('micro_regime',''),
            'agreement':    signal.get('agreement',   0.0),
            # Delta-decay tracking: store entry DTE and entry spot so
            # we can decompose exit PnL into directional vs theta drag.
            'dte_mins_entry':  t.get('dte_mins_entry', 0.0),
            'iv_annpct_entry': t.get('iv_annpct_entry', 0.0),
            'spot_entry':      float(signal.get('spot', 0.0)),
        }

        print(f"\n  [PAPER ENTRY]  {self._position['symbol']}  "
              f"{signal['direction']}  entry={t['entry_price']:.2f}  "
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
              current_row=None, signal_state=None):
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
            self._exit('TIME_EXIT', ltp, now, ks, signal_state=signal_state)
            return

        # Time-aware stop tightening
        # WHY: Late-day theta decays faster (gamma risk near expiry).
        # Same 40% stop that's fair at 10am is too loose at 2:30pm.
        # Tighten stop progressively so afternoon trades exit quicker on failure.
        minute_of_day = hm - (9 * 60 + 15)
        if minute_of_day >= 270:      # after 1:45 PM — tighten to 25%
            time_adjusted_stop = entry * (1 - 0.25)
        elif minute_of_day >= 210:    # after 12:45 PM — tighten to 32%
            time_adjusted_stop = entry * (1 - 0.32)
        else:
            time_adjusted_stop = p['stop']   # original stop (40%)
        # Only tighten, never loosen
        effective_stop = max(p['stop'], time_adjusted_stop)
        if ltp <= effective_stop and effective_stop > p['stop']:
            self._exit('TIME_TIGHTENED_STOP', ltp, now, ks, signal_state=signal_state)
            return

        # 2026 Edge: Anti-Wick Two-Factor Stop Logic
        # Instead of exiting on first tick that touches stop, require:
        #   (a) Stop touched for 2 seconds (persistent breach), OR
        #   (b) VWAP of last 10 ticks below stop (sustained breakdown)
        # This prevents single-wick stop-outs that immediately reverse.
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
            # Record touch time if first time
            if self._stop_touch_time is None:
                self._stop_touch_time = time.time()
                print(f"  ⚠️  [Anti-Wick] Stop touched at {ltp:.2f}. Waiting 2s for confirmation...")
            
            # Track prices after stop touch for VWAP calculation
            self._stop_touch_prices.append(ltp)
            
            # Calculate time since stop touch
            time_since_touch = time.time() - self._stop_touch_time
            
            # Calculate VWAP of recent ticks after stop touch
            if len(self._stop_touch_prices) >= 3:
                vwap_after_touch = sum(self._stop_touch_prices) / len(self._stop_touch_prices)
            else:
                vwap_after_touch = ltp
            
            # Two-factor exit conditions:
            # 1. Time-based: Stop touched for 2+ seconds (persistent)
            # 2. Price-based: VWAP of last 10 ticks is below stop (sustained breakdown)
            # 3. Flash crash: Close price below stop by 2x ATR (genuine breakdown)
            time_confirmed = time_since_touch >= 2.0
            vwap_confirmed = vwap_after_touch <= p['stop']
            flash_crash = (option_atr > 0) and (close_price <= p['stop'] - 2.0 * option_atr)
            
            if time_confirmed or vwap_confirmed or flash_crash:
                reason = 'TWO_FACTOR_STOP'
                if flash_crash:
                    reason = 'FLASH_CRASH_STOP'
                elif time_confirmed:
                    reason = 'TIME_CONFIRMED_STOP'
                elif vwap_confirmed:
                    reason = 'VWAP_CONFIRMED_STOP'
                
                print(f"  🛑 [Anti-Wick] Stop confirmed: {reason} (waited {time_since_touch:.1f}s, VWAP={vwap_after_touch:.2f})")
                self._exit(reason, ltp, now, ks, signal_state=signal_state)
                return
        else:
            # Stop not touched - reset tracker if it was previously touched (wick recovery)
            if self._stop_touch_time is not None:
                print(f"  ✅ [Anti-Wick] Stop recovered! Price bounced from {min(self._stop_touch_prices):.2f} to {ltp:.2f}")
                self._stop_touch_time = None
                self._stop_touch_prices = deque(maxlen=10)

        # Target hit
        if ltp >= p['target']:
            self._exit('TARGET', ltp, now, ks, signal_state=signal_state)
            return

        # Expansion failure exit
        # WHY: Options only pay when volatility is expanding. If ATR contracts
        # after entry, theta will eat the position even if direction is correct.
        # Exit early rather than waiting for stop or time exit.
        # Conditions: trade open >5 bars AND ATR ratio collapsed AND position not yet profitable
        if current_row is not None:
            bars_open_now = int((now - p['entry_time']).total_seconds() / 60)
            atr_ratio_now = float(current_row.get('atr_ratio', 1.0))
            pnl_pct = (ltp / entry - 1.0)
            if bars_open_now >= 5 and atr_ratio_now < 0.65 and pnl_pct < 0.05:
                self._exit('EXPANSION_FAILURE', ltp, now, ks, signal_state=signal_state)
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
    def _exit(self, reason: str, exit_price: float, now, ks: 'KillSwitch', signal_state=None):
        p     = self._position
        entry = p['entry_price']
        qty   = p['qty']
        pnl   = (exit_price - entry) * qty
        
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
            'exit_price':   round(exit_price, 2),
            'qty':          qty,
            'contracts':    p['contracts'],
            'lot_size_used': p.get('lot_size_used', get_lot_size()),
            'pnl':          round(pnl, 2),
            'pnl_pct':      round((exit_price / entry - 1) * 100, 2),
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
            'theta_drag_pts': theta_drag_pts,   # Rs/unit eroded by time decay
            'theta_drag_pnl': theta_drag_pnl,   # Rs lost to theta on this trade
            'dir_pnl':        dir_pnl,           # Rs from directional move × delta
            'delta_at_exit':  round(delta_at_exit, 4),
            'bars_open':      bars_open,
        }
        self._trades.append(trade)

        # Feed result back to KillSwitch for consecutive-loss gate
        ks.record_trade(pnl)

        won = pnl > 0
        print(f"\n  [PAPER EXIT]  {p['symbol']}  {reason}  "
              f"exit={exit_price:.2f}  PnL={pnl:+.2f}  "
              f"hold={hold_mins}min  "
              f"delta={delta_at_exit:.3f}  theta={theta_drag_pnl:+.2f}  dir={dir_pnl:+.2f}  "
              f"equity={self._equity:,.2f}  "
              f"{'WIN' if won else 'LOSS'}")

        self._position = {}

    # ------------------------------------------------------------------
    # FORCE EXIT (e.g. end of day if still open)
    # ------------------------------------------------------------------
    def force_exit(self, exit_price: float, now, ks: 'KillSwitch', signal_state=None):
        if self._position:
            self._exit('EOD_FORCE', exit_price, now, ks, signal_state=signal_state)

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

        print(f"\n{'='*62}")
        print(f"  PAPER TRADING SUMMARY  --  {trade_date}")
        print(f"{'='*62}")
        print(f"  Trades      : {n}")
        print(f"  Win rate    : {wr:.1%}  ({len(wins)}W / {len(losses)}L)")
        print(f"  Total PnL   : {total_pnl:+,.2f}  ({ret_pct:+.2f}% of capital)")
        print(f"  Theta drag  : {total_theta:+,.2f}  (Rs lost to time decay across all trades)")
        print(f"  Avg win     : {(sum(t['pnl'] for t in wins)/len(wins)) if wins else 0:+.2f}")
        print(f"  Avg loss    : {(sum(t['pnl'] for t in losses)/len(losses)) if losses else 0:+.2f}")
        print(f"  Avg hold    : {avg_hold:.0f} min")
        print(f"  Intraday DD : {worst_dd:.2f}%")
        print(f"  Sharpe      : {sharpe:.2f}")
        print(f"{'='*62}")

        # Per-trade breakdown — add theta_drag column
        print(f"\n  {'#':<3} {'Time':<12} {'Symbol':<22} {'Dir':<5} "
              f"{'Entry':>7} {'Exit':>7} {'PnL':>8} {'Theta':>7} {'Reason':<12} {'Conf':>6} {'EVnet':>7}")
        print(f"  {'-'*108}")
        for i, t in enumerate(trades, 1):
            print(f"  {i:<3} {t['entry_time']}-{t['exit_time']:<7} "
                  f"{t['symbol']:<22} {t['direction']:<5} "
                  f"{t['entry_price']:>7.2f} {t['exit_price']:>7.2f} "
                  f"{t['pnl']:>+8.2f} "
                  f"{t.get('theta_drag_pnl', 0.0):>+7.2f} "
                  f"{t['exit_reason']:<12} "
                  f"{t['avg_conf']:>5.1%} {t['ev_net']:>+7.4f}")

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


