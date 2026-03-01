"""
LiveSafetyManager — Single orchestrator for all live-trading safety checks.

Centralises:
  1. Bar validation (delegates to BarValidator)
  2. Latency monitoring (delegates to LatencyMonitor)
  3. Auto-flatten polling (from KillSwitch)
  4. Order deduplication (delegates to OrderDeduplicator)
  5. Regime state machine (delegates to RegimeStateMachine)
  6. Market-open warmup gate (hard bar count before any signals allowed)
  7. Feature drift kill-switch (blocks inference when features are OOD)
  8. Emergency shutdown criteria

Usage in live_loop (each bar):
    safety = LiveSafetyManager(kill_switch)
    ...
    bar_ok, bar_errors = safety.validate_bar(current_bar_series)
    if not bar_ok:
        continue

    should_flatten, reason = safety.check_flatten()
    if should_flatten:
        flatten_all(reason); break

    warmup_ok, warmup_reason = safety.check_warmup(bars_since_open)
    if not warmup_ok:
        continue  # indicators not yet stable

    drift_ok, drift_reason = safety.check_feature_drift(row, active_features, training_stats)
    # drift_ok False means degrade confidence, not block entirely
"""

import logging
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd

from ..data.websocket import BarValidator, LatencyMonitor
from ..execution.risk import KillSwitch, OrderDeduplicator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Warm-up constants: minimum bars after market open before signals are trusted
# ---------------------------------------------------------------------------
WARMUP_BARS = {
    '1m':  20,   # 20 bars = 20 min: RSI(14), EMA(9) need this to stabilise
    '5m':  10,
    '15m': 6,
    '30m': 4,
}
HARD_OPEN_EXCLUDE_MINS = 15   # absolute minimum: never trade first 15 min

# ---------------------------------------------------------------------------
# Feature drift constants
# ---------------------------------------------------------------------------
DRIFT_WARN_SIGMA  = 2.5   # warn when feature drifts beyond this many sigma
DRIFT_BLOCK_SIGMA = 5.0   # degrade confidence when beyond this
DRIFT_KILL_SIGMA  = 8.0   # kill signal entirely when beyond this


class LiveSafetyManager:
    """
    Single point of truth for all live-trading safety decisions.

    Instantiated once at the start of live_loop and consulted each bar.
    """

    def __init__(self, kill_switch: KillSwitch):
        self.ks           = kill_switch
        self.bar_validator = BarValidator()
        self.latency_mon   = LatencyMonitor()
        self.order_dedup   = OrderDeduplicator()
        self._shutdown_requested = False
        self._shutdown_reason    = ''
        self._bars_since_open    = 0
        self._feature_kill_log   = set()   # tracks which features were logged as killed

    # -----------------------------------------------------------------------
    # 1. Bar validation
    # -----------------------------------------------------------------------
    def validate_bar(self, bar: pd.Series) -> tuple:
        """Returns (is_valid: bool, errors: list[str])"""
        is_valid, errors = self.bar_validator.validate(bar)
        if self.bar_validator.should_halt():
            self.request_shutdown(f"DATA: {self.bar_validator._consecutive_errors} consecutive bad bars")
        return is_valid, errors

    # -----------------------------------------------------------------------
    # 2. Latency
    # -----------------------------------------------------------------------
    def record_latency(self, bar_close_dt, signal_ready_dt=None):
        self.latency_mon.record(bar_close_dt, signal_ready_dt)

    def latency_ok(self) -> bool:
        ok = self.latency_mon.is_acceptable()
        if not ok:
            logger.warning(f"[Safety] Median latency {self.latency_mon.median_lag_ms():.0f}ms — 1m signals degraded")
        return ok

    # -----------------------------------------------------------------------
    # 3. Auto-flatten polling
    # -----------------------------------------------------------------------
    def check_flatten(self) -> tuple:
        """Returns (should_flatten: bool, reason: str)"""
        return self.ks.consume_flatten_request()

    # -----------------------------------------------------------------------
    # 4. Order deduplication
    # -----------------------------------------------------------------------
    def can_place_order(self, signal: dict) -> bool:
        return self.order_dedup.can_place(signal)

    def register_order(self, signal: dict, order_id: str = ''):
        self.order_dedup.register(signal, order_id)

    # -----------------------------------------------------------------------
    # 5. Market-open warmup gate
    # -----------------------------------------------------------------------
    def tick_bar(self):
        """Call once per bar to advance the warmup counter."""
        self._bars_since_open += 1

    def seed_warmup(self, bars_already_elapsed: int):
        """
        Pre-seed the warmup counter when starting mid-session.
        Called once after cold-start if we already have live data for today.
        Prevents waiting 20 minutes for warmup when the market has been open for hours.
        """
        self._bars_since_open = max(self._bars_since_open, bars_already_elapsed)

    def reset_day(self):
        """Call at start of each trading session."""
        self._bars_since_open = 0
        self.bar_validator.reset()
        self._feature_kill_log.clear()
        self._shutdown_requested = False
        self._shutdown_reason    = ''

    def check_warmup(self, minute_of_day: int) -> tuple:
        """
        Returns (ready: bool, reason: str).
        Blocks signals if not enough bars have formed since open.

        Two independent checks:
          a) Absolute minute-of-day gate: never trade the first 15 min.
          b) Bar count gate: enough 1-min bars for indicators to be stable.
        """
        if minute_of_day < HARD_OPEN_EXCLUDE_MINS:
            return False, f"WARMUP: market open exclude window ({minute_of_day}/{HARD_OPEN_EXCLUDE_MINS} min)"
        if self._bars_since_open < WARMUP_BARS['1m']:
            return False, f"WARMUP: only {self._bars_since_open}/{WARMUP_BARS['1m']} 1-min bars formed"
        return True, ''

    # -----------------------------------------------------------------------
    # 6. Feature drift kill-switch
    # -----------------------------------------------------------------------
    def check_feature_drift(self, row: pd.Series,
                             active_features: list,
                             training_stats: dict) -> tuple:
        """
        Compares live feature values against their training distribution.

        training_stats: {feature_name: {'mean': float, 'std': float}}
        (Populated during training and saved to meta.json.)

        Returns (confidence_multiplier: float, killed_features: list)
          confidence_multiplier: 1.0 = normal, 0.5 = degraded, 0.0 = kill
          killed_features: list of feature names that are out-of-distribution
        """
        if not training_stats:
            return 1.0, []

        max_z       = 0.0
        killed      = []
        warn_count  = 0

        for feat in active_features:
            stats = training_stats.get(feat)
            if not stats:
                continue
            val = row.get(feat, np.nan)
            if pd.isna(val):
                continue
            std = stats.get('std', 1.0)
            if std < 1e-9:
                continue
            z = abs((float(val) - stats['mean']) / std)
            max_z = max(max_z, z)

            if z > DRIFT_KILL_SIGMA:
                killed.append(feat)
                if feat not in self._feature_kill_log:
                    logger.critical(f"[FeatureDrift] KILLED {feat}: z={z:.1f} (>{DRIFT_KILL_SIGMA})")
                    self._feature_kill_log.add(feat)
            elif z > DRIFT_WARN_SIGMA:
                warn_count += 1

        if warn_count > 5:
            logger.warning(f"[FeatureDrift] {warn_count} features drifting beyond {DRIFT_WARN_SIGMA}σ")

        if max_z >= DRIFT_KILL_SIGMA:
            conf_mult = 0.0   # kill signal entirely
        elif max_z >= DRIFT_BLOCK_SIGMA:
            conf_mult = 0.5   # halve confidence
        elif max_z >= DRIFT_WARN_SIGMA:
            conf_mult = 0.75  # 25% penalty
        else:
            conf_mult = 1.0

        return conf_mult, killed

    # -----------------------------------------------------------------------
    # 7. Emergency shutdown
    # -----------------------------------------------------------------------
    def request_shutdown(self, reason: str):
        self._shutdown_requested = True
        self._shutdown_reason    = reason
        logger.critical(f"[Safety] EMERGENCY SHUTDOWN REQUESTED: {reason}")

    def is_shutdown_requested(self) -> bool:
        return self._shutdown_requested

    def shutdown_reason(self) -> str:
        return self._shutdown_reason

    # -----------------------------------------------------------------------
    # 8. Summary status (for dashboard)
    # -----------------------------------------------------------------------
    def status_dict(self) -> dict:
        return {
            'warmup_bars':       self._bars_since_open,
            'warmup_ready':      self._bars_since_open >= WARMUP_BARS['1m'],
            'latency_ms':        round(self.latency_mon.median_lag_ms(), 0),
            'bar_errors':        self.bar_validator._consecutive_errors,
            'daily_pnl':         round(self.ks.current_equity - self.ks.day_start_equity, 2),
            'size_multiplier':   round(self.ks.get_position_size_multiplier(), 2),
            'trades_today':      self.ks._trades_today,
            'shutdown':          self._shutdown_requested,
        }
