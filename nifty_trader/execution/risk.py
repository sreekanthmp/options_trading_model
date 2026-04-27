"""Capital protection: KillSwitch, SetupFatigueTracker, PositionManager.

LIVE SAFETY ENHANCEMENTS (audit-driven):
  1. Cool-down timer after consecutive losses — prevents "revenge trading" by
     locking the system for COOL_DOWN_AFTER_CONSEC_LOSS minutes.
  2. Max trades per day hard cap — once hit, no new entries regardless of signal.
  3. Approaching-limit position throttle — reduces size as daily DD approaches limit.
  4. Auto-flatten interface — all callers can trigger emergency flatten via
     KillSwitch.request_flatten().  The live loop polls this each bar.
  5. Duplicate-order guard via OrderDeduplicator.
"""
import time, logging, hashlib
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


from ..config import (
    KILL_DAILY_DD_PCT, KILL_DAILY_DD_PCT_PAPER, KILL_CONSEC_LOSSES, KILL_CONSEC_LOSSES_PAPER, KILL_VOL_SHOCK_MULT,
    KILL_REGIME_FLIP_MINS, KILL_VRECOVERY_AGREE,
    KILL_WEEKLY_DD_HALVE_PCT, KILL_MONTHLY_DD_HALT_PCT, CONF_SIZE_BANDS,
    TRANSITION_SHOCK_MINS, TRANSITION_CONF_BOOST,
    REGIME_CRISIS, REGIME_TRENDING, REGIME_RANGING, REGIME_NAMES,
    _training_feature_stats,
)

logger = logging.getLogger(__name__)

# Additional live-safe constants (not in config to keep backward-compat)
MAX_TRADES_PER_DAY          = 999   # DISABLED - no daily trade limit (only loss-based restrictions)
COOL_DOWN_AFTER_CONSEC_LOSS = 30    # minutes to pause after consecutive-loss gate fires
DAILY_DD_WARN_PCT           = 0.70  # at 70% of daily DD limit, reduce position size
ORDER_DEDUP_EXPIRY_SECONDS  = 30    # duplicate guard window

# ==============================================================================
# 9. KILL-SWITCH (CAPITAL PROTECTION)
# ==============================================================================

class KillSwitch:
    """
    Hard stop-gates for live trading.  All checks are stateless-friendly:
    call .check() every tick -- returns (blocked: bool, reason: str).

    Gates:
      1. Daily drawdown:    day P&L < -KILL_DAILY_DD_PCT of capital  -> halt day
      2. Consecutive losses: >= KILL_CONSEC_LOSSES in a row           -> halt day
      3. Vol shock:         recent ATR > KILL_VOL_SHOCK_MULT * avg    -> halt 30min
      4. Regime flip:       regime changed < KILL_REGIME_FLIP_MINS ago -> wait
      5. Black Swan gap:    opening gap > 2.5x daily ATR              -> halt 45min
    
    Edge Case Mitigations:
      - Edge Case 1: "Ghost Tick" - Zombie WebSocket handled by MarketStreamer.check_heartbeat()
      - Edge Case 2: "Gamma Flip" - Handled by select_option() gamma_window logic
      - Edge Case 3: "Stationarity vs Trend Memory" - Black Swan Gap Gate (below)
      - Edge Case 4: "Regime Whipsaw" - Regime Flip Cooldown with V-Recovery Bypass (below)
      - Edge Case 5: "Low Liquidity/High Spread" - Handled by _ev_net() spread penalty
      - Edge Case 6: "FracDiff Warmup" - Handled by sync_historical_buffer() cold-start
    """

    def __init__(self, capital: float, paper_mode: bool = False):
        self.capital          = capital
        self.paper_mode       = paper_mode
        self._daily_dd_limit  = KILL_DAILY_DD_PCT_PAPER if paper_mode else KILL_DAILY_DD_PCT
        self.day_start_equity = capital
        self.current_equity   = capital
        self.consec_losses    = 0
        self.last_regime      = -1
        self.regime_flip_time = None   # datetime when last flip occurred
        self._day_halted      = False
        self._halt_reason     = ''
        self._vrecovery_bypass = False
        self.trading_halted   = False

        # Enhanced: cool-down after consecutive losses
        self._cool_down_until : datetime | None = None

        # Enhanced: daily trade counter
        self._trades_today : int = 0

        # Enhanced: auto-flatten request flag
        self._flatten_requested : bool  = False
        self._flatten_reason    : str   = ''

        # CRISIS bypass: consecutive TRENDING micro-regime counter
        # Bypass only fires after 2 consecutive TRENDING bars to avoid whipsaws.
        self._consec_trending_bars: int = 0

        # CRISIS bypass: consecutive bars where ML direction is stable (same direction)
        # WHY: Trade 2 (Apr-20) fired after 4 bars of ML=DOWN then 1-bar flip to UP.
        # Model instability on feature noise caused a false signal at the flip point.
        # Require ML direction stable for 2+ bars before bypass fires.
        self._consec_ml_direction: str = ''   # last seen ML direction in CRISIS
        self._consec_ml_bars: int = 0         # consecutive bars with same ML direction

        # Edge Case 3: Black Swan Gap tracking
        self._black_swan_time = None  # When gap shock detected
        self._black_swan_gap = 0.0    # Gap magnitude

        # Regime history for regime_conf_boost() — must be in __init__ so
        # reset_day() can safely call .clear() without AttributeError
        self._regime_history: list = []

        # Weekly / monthly drawdown tracking.
        # Anchored to initial capital so a losing week doesn't permanently
        # shrink the next week's budget (ratchet-effect avoidance).
        self._week_start_equity  : float = capital
        self._month_start_equity : float = capital
        self._current_week       : int   = -1   # ISO week number
        self._current_month      : int   = -1

    def reset_day(self):
        """Call at the start of each trading day."""
        self.day_start_equity  = self.current_equity
        self._day_halted       = False
        self._halt_reason      = ''
        self._vrecovery_bypass = False
        self._black_swan_time  = None
        self._black_swan_gap   = 0.0
        self._trades_today     = 0
        self._flatten_requested = False
        self._flatten_reason    = ''
        # Clear cross-day regime bias — yesterday's ranging bars must not
        # inflate today's confidence floor via regime_conf_boost()
        self._regime_history.clear()
        # Only reset consecutive-loss cooldown if it has already expired.
        # If the cooldown is still active (e.g. late-session losses and restart
        # before the 30-min window expires), keep it — don't give a free reset
        # just because the calendar day rolled over.
        if self._cool_down_until is None or datetime.now() >= self._cool_down_until:
            self.consec_losses    = 0
            self._cool_down_until = None
        else:
            remaining = (self._cool_down_until - datetime.now()).seconds // 60
            logger.info(
                f"[KillSwitch] reset_day: consecutive-loss cooldown still active "
                f"({remaining}min remaining). Carrying over to new session."
            )

    def record_trade(self, pnl: float):
        """Update state after a completed trade."""
        self.current_equity += pnl
        self._trades_today  += 1

        # Consecutive loss tracking
        if pnl < 0:
            self.consec_losses += 1
        else:
            self.consec_losses = 0

        # Kill-switch + cool-down on consecutive losses
        # Paper mode gets a higher limit (3) to gather more statistical samples
        _consec_limit = KILL_CONSEC_LOSSES_PAPER if self.paper_mode else KILL_CONSEC_LOSSES
        if self.consec_losses >= _consec_limit:
            self.trading_halted   = True
            self._cool_down_until = datetime.now() + timedelta(minutes=COOL_DOWN_AFTER_CONSEC_LOSS)
            logger.warning(
                f"[KillSwitch] {self.consec_losses} consecutive losses. "
                f"Trading halted + {COOL_DOWN_AFTER_CONSEC_LOSS}min cool-down."
            )

        # Daily loss limit — trigger auto-flatten immediately
        day_dd = (self.current_equity - self.day_start_equity) / (self.day_start_equity + 1e-9)
        if day_dd <= -self._daily_dd_limit and not self._day_halted:
            self._day_halted    = True
            self._halt_reason   = f"Daily DD {day_dd:.1%} hit limit {-self._daily_dd_limit:.1%}"
            self.request_flatten(f"DAILY_LOSS_LIMIT: {day_dd:.1%}")

    def request_flatten(self, reason: str):
        """Signal the live loop to flatten all positions immediately."""
        self._flatten_requested = True
        self._flatten_reason    = reason
        logger.critical(f"[KillSwitch] AUTO-FLATTEN REQUESTED: {reason}")

    def consume_flatten_request(self) -> tuple:
        """
        Poll this every bar in the live loop.
        Returns (should_flatten: bool, reason: str).
        Resets the flag after returning True so it only fires once.
        """
        if self._flatten_requested:
            self._flatten_requested = False
            return True, self._flatten_reason
        return False, ''

    def get_position_size_multiplier(self) -> float:
        """
        Scale down position size as daily drawdown approaches the daily limit.
        Returns a multiplier in (0, 1].
        At 70% of daily loss limit → 0.5x.
        At 100% → 0 (blocked by gate, but returns 0 for safety).
        """
        day_dd = (self.current_equity - self.day_start_equity) / (self.day_start_equity + 1e-9)
        if day_dd >= 0:
            return 1.0   # In profit — full size
        used_pct = abs(day_dd) / (self._daily_dd_limit + 1e-9)
        if used_pct >= 1.0:
            return 0.0
        if used_pct >= DAILY_DD_WARN_PCT:
            # Linear ramp: from 1.0 at warn threshold to 0.25 at limit
            ratio = (used_pct - DAILY_DD_WARN_PCT) / (1.0 - DAILY_DD_WARN_PCT)
            return max(0.25, 1.0 - 0.75 * ratio)
        return 1.0


    def notify_regime(self, new_regime: int):
        """Call when regime changes. Records flip time for shock zone (Req 10)."""
        if self.last_regime != -1 and new_regime != self.last_regime:
            self.regime_flip_time = datetime.now()
            print(f"  [Regime Flip] {REGIME_NAMES.get(self.last_regime, 'UNKNOWN')} -> "
                  f"{REGIME_NAMES.get(new_regime, 'UNKNOWN')}. Cooldown: {KILL_REGIME_FLIP_MINS} min")
        self.last_regime = new_regime
        # Track regime history for frequency gate (Issue 5)
        now = datetime.now()
        self._regime_history.append((now, new_regime))
        # Time-based purge: keep only entries from the last 4 hours
        # WHY: count-based kept 20 entries across days; intraday regime mix
        # from yesterday would unfairly boost today's conf floor.
        cutoff = now - timedelta(hours=4)
        self._regime_history = [(t, r) for t, r in self._regime_history if t >= cutoff]

    def regime_conf_boost(self) -> float:
        """
        Issue 5 — Regime Frequency Gate.
        If recent regime readings are heavily RANGING (>70%), return an extra
        confidence floor boost (+0.05) so only high-conviction signals fire.
        Returns 0.0 when regime mix is normal.
        """
        if len(self._regime_history) < 5:
            return 0.0
        recent = [r for _, r in self._regime_history[-10:]]
        ranging_pct = sum(1 for r in recent if r == REGIME_RANGING) / len(recent)
        if ranging_pct >= 0.70:
            return 0.05  # require 5pp more confidence in chop-heavy periods
        return 0.0

    def notify_black_swan(self, gap_pct: float, day_atr: float):
        """Call when extreme gap detected at market open."""
        self._black_swan_time = datetime.now()
        self._black_swan_gap = gap_pct
        print(f"  [Black Swan] Opening gap {gap_pct:.2%} > 2.5x daily ATR ({day_atr:.2%}). "
              f"Blocking trades for 45 minutes.")

    def in_transition_zone(self) -> bool:
        """
        Regime Transition Shock Zone (Req 10).
        Returns True during TRANSITION_SHOCK_MINS after a regime change.
        Caller should require STRONG confidence (conf + TRANSITION_CONF_BOOST).
        This is a soft gate: does not fully block but raises the bar.
        """
        if self.regime_flip_time is None:
            return False
        import datetime as _dt
        mins_since = (_dt.datetime.now() - self.regime_flip_time).total_seconds() / 60
        return mins_since < TRANSITION_SHOCK_MINS

    def check(self, current_atr: float = 0, avg_atr: float = 0,
              current_regime: int = REGIME_RANGING,
              micro_regime: str = 'UNKNOWN',
              agreement: float = 0.0,
              minute_of_day: int = 0,
              ml_direction: str = '') -> tuple:
        """
        Returns (blocked: bool, reason: str).
        blocked=True means do NOT enter any new trade this tick.

        V-Recovery Bypass:
          Gate 4 (regime flip cooldown) is skipped if micro_regime == 'BREAKOUT'
          and agreement >= KILL_VRECOVERY_AGREE (default 85%). This allows
          catching V-bottom reversals that follow crisis/regime-flip events.
          Gates 1, 2, 3, 5 (hard daily DD, consec losses, vol shock, black swan)
          are NEVER bypassed.
        
        Edge Case Summary:
          1. Ghost Tick: Handled by MarketStreamer.check_heartbeat() (not here)
          2. Gamma Flip: Handled by select_option() gamma_window (not here)
          3. Black Swan: Gate 5 below (45-min cooldown after extreme gap)
          4. Regime Whipsaw: Gate 4 below (30-min cooldown with V-Recovery bypass)
          5. High Spread: Handled by _ev_net() spread penalty (not here)
          6. FracDiff Warmup: Handled by sync_historical_buffer() (not here)
        """
        import datetime as _dt

        # Gate 1: daily drawdown (hard -- never bypassed)
        day_dd = (self.current_equity - self.day_start_equity) / (self.day_start_equity + 1e-9)
        if day_dd <= -self._daily_dd_limit:
            self._day_halted  = True
            self._halt_reason = f"Daily DD {day_dd:.1%} breached limit {-self._daily_dd_limit:.1%}"

        if self._day_halted:
            return True, self._halt_reason

        # Gate 1b: max trades per day (hard -- never bypassed)
        if self._trades_today >= MAX_TRADES_PER_DAY:
            return True, f"Max trades/day reached: {self._trades_today}/{MAX_TRADES_PER_DAY}"

        # Gate 2: consecutive losses + cool-down (hard -- never bypassed)
        # FIX: check cooldown expiry BEFORE blocking, so the reset fires on bar N
        # and bar N can proceed (not blocked one extra bar by the off-by-one).
        _consec_limit = KILL_CONSEC_LOSSES_PAPER if self.paper_mode else KILL_CONSEC_LOSSES
        if self._cool_down_until and _dt.datetime.now() >= self._cool_down_until:
            # Cooldown expired — reset counter before checking limit
            self.consec_losses    = 0
            self._cool_down_until = None
            logger.info("[KillSwitch] Cool-down expired. Consecutive-loss counter reset.")
        if self.consec_losses >= _consec_limit:
            if self._cool_down_until and _dt.datetime.now() < self._cool_down_until:
                remaining = int((self._cool_down_until - _dt.datetime.now()).total_seconds() // 60)
                return True, (f"Consecutive losses cool-down: {remaining}min remaining "
                              f"({self.consec_losses} losses)")
            else:
                return True, f"Consecutive losses: {self.consec_losses} (limit {_consec_limit})"

        # Gate 3: vol shock (hard -- never bypassed)
        if avg_atr > 0 and current_atr > 0:
            vol_mult = current_atr / (avg_atr + 1e-9)
            if vol_mult > KILL_VOL_SHOCK_MULT:
                return True, f"Vol shock: ATR {vol_mult:.1f}x above average"
        
        # Gate 5: Black Swan gap shock (hard -- never bypassed)
        # Edge Case 3: Block for 45 min after extreme opening gap
        if self._black_swan_time is not None:
            mins_since_gap = (_dt.datetime.now() - self._black_swan_time).total_seconds() / 60
            if mins_since_gap < 45:
                return True, (f"Black Swan cooldown: {mins_since_gap:.0f}/45 min "
                             f"(gap {self._black_swan_gap:.2%})")
            else:
                # Cooldown expired - reset tracker
                self._black_swan_time = None
                self._black_swan_gap = 0.0
                print(f"  [Black Swan] Cooldown expired. Trading re-enabled.")

        # Gate 4: regime flip cooldown -- bypassable on confirmed BREAKOUT
        # Edge Case 4: Regime Whipsaw mitigation
        if self.regime_flip_time is not None:
            mins_since = (_dt.datetime.now() - self.regime_flip_time).total_seconds() / 60
            if mins_since < KILL_REGIME_FLIP_MINS:
                # V-Recovery bypass: BREAKOUT micro-regime with strong agreement
                # allows trading through cooldown to catch post-crisis reversals
                if (micro_regime == 'BREAKOUT' and
                        agreement >= KILL_VRECOVERY_AGREE):
                    # V-Recovery bypass: cooldown waived, store reason for dashboard
                    self._vrecovery_bypass = True
                    print(f"  [V-Recovery Bypass] Breakout confirmed (agreement {agreement:.1%}). "
                          f"Trading allowed despite regime flip.")
                else:
                    self._vrecovery_bypass = False
                    return True, (f"Regime flip {mins_since:.0f}min ago "
                                  f"(cooldown {KILL_REGIME_FLIP_MINS}min)")
            else:
                self._vrecovery_bypass = False
                if self.regime_flip_time is not None:
                    print(f"  [Regime Flip] Cooldown expired. Trading re-enabled.")
                    self.regime_flip_time = None

        # Gate 6: crisis regime
        # Hard block unless confidence is very high AND intraday is calm/trending.
        # This prevents missing strong directional moves when daily HMM lags.
        # Conditions to bypass (all must be true):
        #   - ML agreement >= 85% (avg_conf proxy via `agreement` arg)
        #   - micro_regime is TRENDING for 2 consecutive bars (not just 1 bar)
        #     WHY: both Apr-20 losses entered on single-bar TRENDING_UP that
        #     immediately reversed. Requiring 2 consecutive bars confirms the
        #     trend is established, not a 1-bar spike.
        #   - Not in first/last 30 min (session edges are noisy)
        if current_regime == REGIME_CRISIS:
            # Track consecutive TRENDING micro-regime bars
            if micro_regime in ('TRENDING_UP', 'TRENDING_DN'):
                self._consec_trending_bars += 1
            else:
                self._consec_trending_bars = 0

            # Track consecutive bars with same ML direction
            # WHY: Trade 2 Apr-20 — 4 bars ML=DOWN then 1-bar flip to UP fired entry.
            # Model instability at the flip point = worst possible entry timing.
            if ml_direction and ml_direction == self._consec_ml_direction:
                self._consec_ml_bars += 1
            elif ml_direction:
                self._consec_ml_direction = ml_direction
                self._consec_ml_bars = 1
            else:
                self._consec_ml_bars = 0

            crisis_bypass = (
                agreement >= KILL_VRECOVERY_AGREE - 1e-6   # float tolerance
                and micro_regime in ('TRENDING_UP', 'TRENDING_DN')
                and self._consec_trending_bars >= 2         # trending 2 bars in a row
                and self._consec_ml_bars >= 2               # ML direction stable 2 bars
                and 60 <= minute_of_day <= 345
            )
            if not crisis_bypass:
                return True, "CRISIS regime -- no new trades"
            else:
                logger.info(
                    f"[KillSwitch] CRISIS bypass: agreement={agreement:.1%} "
                    f"micro={micro_regime} consec_trending={self._consec_trending_bars} "
                    f"consec_ml={self._consec_ml_bars} ml_dir={self._consec_ml_direction} "
                    f"min={minute_of_day}"
                )

        return False, ''

    def update_period_equity(self):
        """
        Roll week/month anchors when the calendar period changes.
        Call once per bar (cheap — only updates on period boundary).
        """
        now = datetime.now()
        iso_week  = now.isocalendar()[1]
        month     = now.month
        if self._current_week != iso_week:
            self._week_start_equity = self.current_equity
            self._current_week      = iso_week
        if self._current_month != month:
            self._month_start_equity = self.current_equity
            self._current_month      = month

    def check_drawdown_protection(self) -> tuple:
        """
        Returns (halt: bool, reason: str, size_mult: float).

        halt=True  → monthly DD limit breached; no new entries (manual review required).
        halt=False → size_mult < 1.0 when weekly DD exceeds halve threshold.

        Designed to be called before every new entry attempt.
        Does NOT modify trading_halted — that is reserved for the daily gate
        so the two circuits are independent and don't interfere.
        """
        # Monthly gate (hard — requires manual reset)
        monthly_dd = (self.current_equity - self._month_start_equity) / (self._month_start_equity + 1e-9)
        if monthly_dd <= -KILL_MONTHLY_DD_HALT_PCT:
            return True, (f"Monthly DD {monthly_dd:.1%} exceeded limit "
                          f"{-KILL_MONTHLY_DD_HALT_PCT:.0%} — manual review required"), 0.0

        # Weekly gate (soft — halve size)
        weekly_dd   = (self.current_equity - self._week_start_equity) / (self._week_start_equity + 1e-9)
        size_mult   = 0.5 if weekly_dd <= -KILL_WEEKLY_DD_HALVE_PCT else 1.0
        if size_mult < 1.0:
            logger.info(f"[KillSwitch] Weekly DD {weekly_dd:.1%} > {KILL_WEEKLY_DD_HALVE_PCT:.0%} "
                        f"threshold — position size halved")
        return False, '', size_mult

    def get_conf_size_multiplier(self, conf: float) -> float:
        """
        Confidence-based capital scaling multiplier (0.5–1.0).

        Maps final_conf → fraction of available capital to deploy.
        Applied to the capital argument passed to select_option() so that
        low-confidence signals trade half size while strong signals trade full.

        Bands are defined in config.CONF_SIZE_BANDS and reviewed quarterly.
        """
        for threshold, mult in sorted(CONF_SIZE_BANDS, reverse=True):
            if conf >= threshold:
                return mult
        return 0.5   # fallback for conf below lowest band (should not reach here after floor check)

    def transition_conf_requirement(self) -> float:
        """
        Extra confidence required during transition shock zone (Req 10).
        Returns TRANSITION_CONF_BOOST if in transition, else 0.
        """
        return TRANSITION_CONF_BOOST if self.in_transition_zone() else 0.0


# ==============================================================================
# 4. SETUP FATIGUE KILL-SWITCH (2026 LIVE SURVIVABILITY)
# ==============================================================================

class SetupFatigueTracker:
    """
    WHY: Some setups (e.g., OR breakout, VWAP break) stop working intraday
         due to market conditions. Continuing to trade a failing setup burns
         capital. This tracker disables setups after 2 failures in same regime
         on same day to prevent "insanity" (repeating failed strategies).
    
    How it works:
        1. Track each entry setup type (vwap_break, or_break, trend_pullback, etc.)
        2. On loss, increment failure count for that (setup, regime) pair
        3. After 2 failures, disable setup for rest of day
        4. Reset counters daily
    """
    
    def __init__(self):
        # Key: (setup_type, regime) -> failure_count
        self._failures: dict = {}
        self._disabled_setups: set = set()  # (setup_type, regime) tuples
        self._current_date = None
        
    def reset_day(self, today):
        """Reset all counters at start of new trading day."""
        if self._current_date != today:
            self._failures.clear()
            self._disabled_setups.clear()
            self._current_date = today
    
    def record_loss(self, setup_type: str, regime: int):
        """
        Record a losing trade for this setup-regime pair.
        Disable setup if it reaches 2 failures.
        
        Args:
            setup_type: 'vwap_break', 'or_break', 'trend_pullback', etc.
            regime: REGIME_TRENDING or REGIME_RANGING
        """
        key = (setup_type, regime)
        self._failures[key] = self._failures.get(key, 0) + 1
        
        if self._failures[key] >= 2:
            self._disabled_setups.add(key)
            print(f"  [Setup Fatigue] {setup_type} disabled in regime {regime} "
                  f"after {self._failures[key]} losses today")
    
    def is_disabled(self, setup_type: str, regime: int) -> bool:
        """Check if setup is disabled for this regime."""
        return (setup_type, regime) in self._disabled_setups


# ==============================================================================
# POSITION CONCENTRATION LIMITS (2026 Risk Management Enhancement)
# ==============================================================================

class PositionManager:
    """
    Track open positions and enforce concentration limits.
    
    Prevents over-leveraging by limiting:
    - Maximum number of simultaneous positions
    - Total capital exposure in options
    
    WHY: In volatile trending periods, model might generate 5+ signals.
         Taking all of them violates Kelly Criterion and risks ruin.
    """
    
    def __init__(self, capital: float, max_positions: int = 2, max_exposure_pct: float = 0.05):
        """
        Args:
            capital: Trading capital
            max_positions: Max simultaneous positions (default 2)
            max_exposure_pct: Max total exposure as % of capital (default 5%)
        """
        self.capital = capital
        self.max_positions = max_positions
        self.max_total_exposure = capital * max_exposure_pct
        self.open_positions = []
    
    def can_add_position(self, proposed_value: float) -> tuple:
        """
        Check if new position is allowed under risk limits.
        
        Args:
            proposed_value: Entry price * lot_size * contracts
        
        Returns:
            (allowed: bool, reason: str)
        """
        if len(self.open_positions) >= self.max_positions:
            return False, f"Max positions ({self.max_positions}) reached"
        
        current_exposure = sum(p['value'] for p in self.open_positions)
        if current_exposure + proposed_value > self.max_total_exposure:
            return False, f"Total exposure limit ({self.max_total_exposure/self.capital:.1%}) exceeded"
        
        return True, "ok"
    
    def add_position(self, value: float, info: dict):
        """Record a new open position."""
        self.open_positions.append({'value': value, 'info': info})
    
    def remove_position(self, index: int = 0):
        """Remove a position by index (default: first position)."""
        if 0 <= index < len(self.open_positions):
            self.open_positions.pop(index)
    
    def clear_positions(self):
        """Clear all positions (e.g., at end of day)."""
        self.open_positions.clear()


# ==============================================================================
# ORDER DEDUPLICATOR (prevents double-orders from reconnects / retry loops)
# ==============================================================================

class OrderDeduplicator:
    """
    Prevents the same signal from generating two broker orders within a short
    window.  This guards against:
      - WebSocket reconnects delivering the same candle twice
      - Live-loop iteration re-running on a clock tick before the order ack'd
      - Manual retry after an apparent timeout that actually succeeded

    Usage:
        dedup = OrderDeduplicator()
        if dedup.can_place(signal):
            broker.place(order)
            dedup.register(signal, order_id)
    """

    def __init__(self):
        self._pending: dict = {}   # signal_hash -> {'order_id', 'ts'}

    def _hash(self, signal: dict) -> str:
        key = (f"{signal.get('direction','?')}"
               f"_{signal.get('minute_of_day', 0)}"
               f"_{round(signal.get('spot', 0), -1)}")   # round spot to nearest 10
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def can_place(self, signal: dict) -> bool:
        h = self._hash(signal)
        if h in self._pending:
            age = (datetime.now() - self._pending[h]['ts']).total_seconds()
            if age < ORDER_DEDUP_EXPIRY_SECONDS:
                logger.warning(f"[OrderDedup] Duplicate order blocked (hash={h}, age={age:.0f}s)")
                return False
            else:
                del self._pending[h]   # expired — allow
        return True

    def register(self, signal: dict, order_id: str = ''):
        h = self._hash(signal)
        self._pending[h] = {'order_id': order_id, 'ts': datetime.now()}

    def cleanup(self):
        """Remove expired entries (call periodically)."""
        now = datetime.now()
        expired = [h for h, v in self._pending.items()
                   if (now - v['ts']).total_seconds() > ORDER_DEDUP_EXPIRY_SECONDS]
        for h in expired:
            del self._pending[h]


# ==============================================================================
# MODEL DRIFT DETECTION (2026 Live Accuracy Enhancement)
# ==============================================================================

