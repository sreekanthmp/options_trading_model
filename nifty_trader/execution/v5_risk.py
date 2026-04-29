"""
V5 Session-Level Risk Management.

Implements exactly the risk rules from the system design:

  1. Max 3 trades per session  (V5_MAX_TRADES_DAY)
  2. Daily loss limit Rs 1,500  (V5_DAILY_LOSS_RS)
  3. 15-min cooldown after 2 consecutive losses  (V5_CONSEC_LOSS_TRIGGER/COOLDOWN)
  4. Rolling WR guard — pause if last 5 trades WR < 35%  (V5_WR_WINDOW/MIN)
  5. Gap-day single-trade cap — max 1 trade when opening gap > 1.5%  (V5_GAP_DAY_PCT)
  6. Position sizing: floor(capital × 0.75 / (premium × 65)), halve if VIX > 22

Usage:
    from nifty_trader.execution.v5_risk import V5RiskState, v5_lot_size

    risk = V5RiskState(capital=100_000)
    risk.set_gap_day(gap_pct=row['gap_pct'])      # call once at session open

    # Before generate_signal_v5() enters order execution:
    allowed, reason = risk.check_entry(now=datetime.now())
    if not allowed:
        return _block(reason)

    # After trade closes:
    risk.record_trade(pnl_rs=-350.0, won=False)

    # Sizing:
    lots = v5_lot_size(capital=100_000, premium=230.0, vix=18.5)
"""
import logging
from collections import deque
from datetime import datetime, timedelta

from ..config import (
    V5_MAX_TRADES_DAY,
    V5_DAILY_LOSS_RS,
    V5_CONSEC_LOSS_TRIGGER,
    V5_CONSEC_LOSS_COOLDOWN,
    V5_WR_WINDOW,
    V5_WR_MIN,
    V5_GAP_DAY_PCT,
    V5_GAP_DAY_MAX_TRADES,
    V5_DEPLOY_PCT,
    V5_MAX_LOTS,
    V5_LOT_SIZE,
    V5_VIX_HALVE_THRESHOLD,
)

logger = logging.getLogger(__name__)


class V5RiskState:
    """
    Per-session risk state for the v5 trend-following system.

    One instance per live/paper session; reset at session open via reset_day().

    State tracked:
        trades_today       — count of entries taken this session
        daily_loss_rs      — cumulative session loss in Rs (negative = loss)
        consec_losses      — consecutive losses since last win
        cooldown_until     — datetime when cooldown expires (None = not in cooldown)
        recent_outcomes    — deque of last V5_WR_WINDOW (True=win, False=loss)
        is_gap_day         — True if opening gap exceeded V5_GAP_DAY_PCT
        _halted            — True when day is halted (daily loss limit hit)

    All state is mutable. Callers must call reset_day() at the start of each session.
    """

    def __init__(self, capital: float = 100_000.0):
        self.capital = capital

        # Session state — reset every day
        self.trades_today:    int          = 0
        self.daily_loss_rs:   float        = 0.0
        self.consec_losses:   int          = 0
        self.cooldown_until:  datetime | None = None
        self.recent_outcomes: deque        = deque(maxlen=V5_WR_WINDOW)
        self.is_gap_day:      bool         = False
        self._halted:         bool         = False
        self._halt_reason:    str          = ''

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def reset_day(self) -> None:
        """Reset all intraday counters. Call at session open (09:15 IST)."""
        self.trades_today   = 0
        self.daily_loss_rs  = 0.0
        self.consec_losses  = 0
        self.cooldown_until = None
        self.is_gap_day     = False
        self._halted        = False
        self._halt_reason   = ''
        # recent_outcomes is NOT reset: rolling WR persists across sessions
        # so the 5-trade guard can span multiple days (e.g. 2 Mon + 3 Tue).
        logger.info("[V5Risk] Session reset.")

    def set_gap_day(self, gap_pct: float) -> None:
        """
        Evaluate the opening gap and flag a gap day if |gap_pct| > V5_GAP_DAY_PCT.

        Call once per session with the signed gap percentage
        (positive = gap up, negative = gap down).
        """
        self.is_gap_day = abs(gap_pct) > V5_GAP_DAY_PCT
        if self.is_gap_day:
            logger.info(
                f"[V5Risk] GAP DAY: gap={gap_pct:+.2f}% > {V5_GAP_DAY_PCT}% — "
                f"max {V5_GAP_DAY_MAX_TRADES} trade(s) today."
            )

    # ------------------------------------------------------------------
    # Pre-entry gate
    # ------------------------------------------------------------------

    def check_entry(self, now: datetime) -> tuple[bool, str]:
        """
        Run all session-level risk checks.

        Call immediately before executing an entry order.
        Returns (allowed: bool, reason: str).

        Checks (in priority order):
          1. Day halted (daily loss limit)
          2. Daily loss limit
          3. Max trades per session
          4. Gap-day single-trade cap
          5. Consecutive-loss cooldown
          6. Rolling win-rate guard
        """
        # 1. Day already halted
        if self._halted:
            return False, f"[V5Risk] Day halted: {self._halt_reason}"

        # 2. Daily loss limit
        if self.daily_loss_rs <= -V5_DAILY_LOSS_RS:
            self._halted     = True
            self._halt_reason = (
                f"daily_loss_rs={self.daily_loss_rs:.0f} <= -{V5_DAILY_LOSS_RS:.0f}"
            )
            logger.warning(f"[V5Risk] HALTED: {self._halt_reason}")
            return False, f"[V5Risk] HALTED: {self._halt_reason}"

        # 3. Max trades per session
        max_trades = V5_GAP_DAY_MAX_TRADES if self.is_gap_day else V5_MAX_TRADES_DAY
        if self.trades_today >= max_trades:
            cap_label = "gap-day cap" if self.is_gap_day else "daily trade cap"
            return False, (
                f"[V5Risk] BLOCKED: {cap_label} reached "
                f"({self.trades_today}/{max_trades})"
            )

        # 4. Consecutive-loss cooldown
        if self.cooldown_until is not None and now < self.cooldown_until:
            remaining = int((self.cooldown_until - now).total_seconds() / 60) + 1
            return False, (
                f"[V5Risk] BLOCKED: consecutive-loss cooldown active "
                f"({remaining} min remaining)"
            )
        elif self.cooldown_until is not None and now >= self.cooldown_until:
            # Cooldown expired
            self.cooldown_until = None
            logger.info("[V5Risk] Consecutive-loss cooldown expired.")

        # 5. Rolling WR guard (only fires when window is full)
        if len(self.recent_outcomes) >= V5_WR_WINDOW:
            rolling_wr = sum(self.recent_outcomes) / len(self.recent_outcomes)
            if rolling_wr < V5_WR_MIN:
                return False, (
                    f"[V5Risk] BLOCKED: rolling WR={rolling_wr:.1%} < {V5_WR_MIN:.0%} "
                    f"over last {V5_WR_WINDOW} trades — observation mode"
                )

        return True, ""

    # ------------------------------------------------------------------
    # Post-trade recording
    # ------------------------------------------------------------------

    def record_trade(self, pnl_rs: float, won: bool) -> None:
        """
        Record a completed trade outcome.

        Call immediately after each trade closes (stop/target/time exit).

        Args:
            pnl_rs: Net P&L in Rs (negative = loss, positive = win).
            won:    True if the trade was profitable.
        """
        self.trades_today   += 1
        self.daily_loss_rs  += pnl_rs
        self.recent_outcomes.append(won)

        if won:
            self.consec_losses = 0
        else:
            self.consec_losses += 1
            if self.consec_losses >= V5_CONSEC_LOSS_TRIGGER:
                self.cooldown_until = datetime.now() + timedelta(
                    minutes=V5_CONSEC_LOSS_COOLDOWN
                )
                logger.warning(
                    f"[V5Risk] {self.consec_losses} consecutive losses — "
                    f"cooldown until {self.cooldown_until.strftime('%H:%M')}"
                )

        rolling_wr = (
            sum(self.recent_outcomes) / len(self.recent_outcomes)
            if self.recent_outcomes else 1.0
        )
        logger.info(
            f"[V5Risk] Trade recorded: pnl={pnl_rs:+.0f}Rs won={won} "
            f"trades_today={self.trades_today} "
            f"daily_loss={self.daily_loss_rs:+.0f}Rs "
            f"consec_losses={self.consec_losses} "
            f"rolling_wr={rolling_wr:.1%} "
            f"(n={len(self.recent_outcomes)})"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a snapshot of current risk state (for dashboard / logging)."""
        rolling_wr = (
            sum(self.recent_outcomes) / len(self.recent_outcomes)
            if self.recent_outcomes else None
        )
        return {
            'trades_today':    self.trades_today,
            'daily_loss_rs':   round(self.daily_loss_rs, 2),
            'consec_losses':   self.consec_losses,
            'cooldown_active': self.cooldown_until is not None,
            'rolling_wr':      round(rolling_wr, 4) if rolling_wr is not None else None,
            'wr_window_full':  len(self.recent_outcomes) >= V5_WR_WINDOW,
            'is_gap_day':      self.is_gap_day,
            'halted':          self._halted,
            'halt_reason':     self._halt_reason,
        }


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def v5_lot_size(capital: float, premium: float, vix: float = 15.0) -> int:
    """
    Compute the number of lots for a v5 trade.

    Formula:
        base_lots = floor(capital × V5_DEPLOY_PCT / (premium × V5_LOT_SIZE))
        if VIX > V5_VIX_HALVE_THRESHOLD: lots = floor(base_lots / 2)
        lots = max(1, min(lots, V5_MAX_LOTS))

    Args:
        capital:  Available trading capital in Rs.
        premium:  ATM option LTP in Rs (e.g. 230.0).
        vix:      Current India VIX level (e.g. 18.5).

    Returns:
        Number of lots (integer, minimum 1, maximum V5_MAX_LOTS).

    Examples:
        v5_lot_size(100_000, 230.0, 15.0)  → 3 → capped to 2
        v5_lot_size(100_000, 230.0, 25.0)  → 3 → halved to 1 → max(1,1) = 1
        v5_lot_size(30_000,  150.0, 15.0)  → 2 → capped to 2
        v5_lot_size(30_000,  150.0, 25.0)  → 2 → halved to 1
    """
    if premium <= 0 or capital <= 0:
        return 0

    budget    = capital * V5_DEPLOY_PCT
    cost_per_lot = premium * V5_LOT_SIZE
    base_lots = int(budget / cost_per_lot)

    if vix > V5_VIX_HALVE_THRESHOLD:
        base_lots = base_lots // 2
        logger.debug(
            f"[v5_lot_size] VIX={vix:.1f} > {V5_VIX_HALVE_THRESHOLD} — halving lots"
        )

    lots = max(1, min(base_lots, V5_MAX_LOTS))
    logger.debug(
        f"[v5_lot_size] capital={capital:.0f} premium={premium:.0f} "
        f"vix={vix:.1f} → lots={lots}"
    )
    return lots


# ---------------------------------------------------------------------------
# Module-level singleton (mirrors _signal_state in signal_generator.py)
# ---------------------------------------------------------------------------
_v5_risk_state = V5RiskState()
