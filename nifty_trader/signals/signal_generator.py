"""
Signal generation: SignalState tracking + generate_signal() (13-gate filter).
"""
import time, logging
from datetime import datetime, timedelta, date
from collections import deque
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from ..config import (
    HORIZONS, HORIZON_WEIGHTS,
    CONF_STRONG, CONF_MODERATE, CONF_MIN, CONF_BY_HORIZON,
    CONF_FLOOR_TRENDING, CONF_FLOOR_RANGING,
    NO_TRADE_PCTILE_LOW, NO_TRADE_PCTILE_HIGH,
    EV_SAFETY_NORMAL, EV_SAFETY_ELEVATED,
    EV_AVG_WIN_MULT, EV_AVG_LOSS_MULT, OPTION_SLIPPAGE_PCT,
    TEMPORAL_FLIP_LOCK_BARS,
    SEASONALITY_MAX_BIAS,
    MAX_TRADES_PER_DIR_RANGING, MAX_TRADES_PER_DIR_TRENDING,
    REGIME_TRENDING, REGIME_RANGING, REGIME_CRISIS, REGIME_NAMES,
    TB_PUT_TIME_DECAY_EXTRA, TOTAL_COST_PCT,
    STRIKE_OFFSET_CE, STRIKE_OFFSET_PE, STRIKE_ROUNDING,
    MANDATORY_FEATURES,
    EXPIRY_CONF_FLOOR, EXPIRY_ZONES, EXPIRY_FORCE_EXIT_MOD,
    DOWN_ONLY_MODE, ENTRY_MOD_MIN, ENTRY_MOD_MAX,
    BLOCK_RANGING_REGIME, BLOCK_TRENDING_REGIME,
    ADX_ENTRY_MAX, ADX_CRISIS_MAX,
    PRESSURE_RATIO_DOWN_MIN,
)
from ..execution.costs import effective_cost
from ..utils.time_utils import calculate_time_decay_confidence
from ..utils.safeguards import safe_value
from ..execution.risk import KillSwitch
from .confidence import (
    check_directional_agreement,
    check_entry_micro_confirmation,
    _ev_net,
    check_iv_crush,
)
from ..features.feature_engineering import FEATURE_COLS, FEATURE_LIVE_OK
from ..execution.orders import _next_expiry_mins, estimate_option_premium, get_expiry_rule
from ..models.ensemble import MetaLabeler

logger = logging.getLogger(__name__)

# Module-level last block reason — set before every `return None` in generate_signal.
# Callers (dashboard, live) can read this immediately after a None return.
last_block_reason: str = ""


def get_last_block_reason() -> str:
    """Return the reason the last generate_signal() call returned None."""
    return last_block_reason


def _block(reason: str):
    """Set last_block_reason and return None (use as: return _block('...')  )."""
    global last_block_reason
    last_block_reason = reason
    return None


def calibrate_conf(conf: float) -> float:
    """
    Fallback confidence mapping for use when Platt-calibrated models are not
    yet available (e.g. before first retrain after deploying calibration code).

    Maps overconfident raw probabilities (0.85-0.95 range) to realistic
    values consistent with observed 55-65% accuracy on NIFTY intraday signals.

    This is a one-time bridge: once the model is retrained with
    _CalibratedWrapper (trainer.py), this function is bypassed because
    predict_proba already returns calibrated probabilities in [0.45, 0.82].

    Usage: only call this on raw model output BEFORE penalty deductions.
    Never apply it to an already-calibrated output (double-calibration degrades quality).
    """
    if conf > 0.90:
        return 0.62
    elif conf > 0.80:
        return 0.60
    elif conf > 0.70:
        return 0.58
    else:
        return 0.55


def validate_model_inputs(row: pd.Series, active_features: list) -> tuple:
    """
    Validate model inputs before prediction.
    Returns (is_valid: bool, error_msg: str, nan_count: int)
    
    Critical checks:
    1. All required features present
    2. No excessive NaN/Inf values (>20%)
    3. Value ranges are reasonable (not 1e+308 outliers)
    """
    # Check mandatory features first
    missing_mandatory = [f for f in MANDATORY_FEATURES if f not in row.index or pd.isna(row.get(f))]
    if missing_mandatory:
        return False, f"Missing mandatory: {missing_mandatory}", 0
    
    # Count NaNs and outliers
    nan_count = 0
    outlier_count = 0
    
    for feat in active_features:
        val = row.get(feat, 0.0)
        if pd.isna(val) or np.isinf(val):
            nan_count += 1
        elif abs(val) > 1e6:
            outlier_count += 1
    
    # Reject if too many NaNs (>20% of features)
    if nan_count > len(active_features) * 0.2:
        return False, f"Too many NaN features: {nan_count}/{len(active_features)}", nan_count
    
    # Warn on outliers but allow
    if outlier_count > 5:
        logger.warning(f"Input validation: {outlier_count} outliers detected")
    
    return True, "", nan_count


class SignalState:
    """
    Per-session stateful object carrying all v3.1 signal quality controls:
      - Confidence history for percentile ranking (Req 6)
      - Performance weights for horizons (Req 5)
      - Temporal consistency gate: lock after 1m direction flips (Req 5)
      - Signal scarcity counter per direction (Req 13)
      - Intraday seasonality EV prior (Req 11) -- learned from training data
      - Staleness tracker: decay confidence when micro-regime flips (Req 14)
    """

    def __init__(self):
        # Rolling confidence history per regime for percentile ranking
        self._conf_history: dict = {r: [] for r in range(3)}
        self._conf_max_hist = 500   # keep last N signals per regime

        # Performance-based horizon weights (Req 5): updated from recent trades
        self.perf_weights: dict = dict(HORIZON_WEIGHTS)   # mutable copy

        # Temporal consistency gate state (Req 5)
        self._last_1m_dirs: list = []   # last 3 predicted 1m directions
        self._lock_bars_remaining: int = 0
        self._lock_cooldown_bars: int = 0  # bars before re-lock is allowed after unlock

        # Signal scarcity: {direction: count_today}
        self.trades_today: dict = {'UP': 0, 'DOWN': 0}
        self._scarcity_date = None

        # Intraday seasonality prior: {minute_of_day: ev_bias}
        # Populated by learn_seasonality(); falls back to zeros.
        self.seasonality_prior: dict = {}

        # Staleness: last micro-regime seen per direction
        self._last_micro_regime: str = 'UNKNOWN'
        self._stale_conf_penalty: float = 0.0

        # Last signal agreement (used by KillSwitch V-Recovery bypass)
        self.last_agreement: float = 0.0
        
        # Edge Case 2: Post-Trade Cooldown (Anti-Churn)
        # Tracks last exit time to prevent immediate re-entry
        self.last_exit_time = None
        
        # VWAP history for micro-confirmation (moved from function attribute)
        # FIX: Use deque with maxlen to prevent unbounded growth in long-running VPS process
        self.vwap_history = deque(maxlen=100)  # Only keep last 100 positions

    def reset_day(self):
        """Reset intraday counters at start of new session."""
        self.trades_today = {'UP': 0, 'DOWN': 0}
        self._lock_bars_remaining = 0
        self._lock_cooldown_bars = 0
        self._last_1m_dirs = []
        self._stale_conf_penalty = 0.0
        self.last_exit_time = None  # Edge Case 2: Reset cooldown
        self.vwap_history.clear()  # Clear VWAP history (deque supports clear())
        # Clear confidence history to prevent memory leak
        for regime in range(3):
            self._conf_history[regime] = []
        # Reset horizon performance weights daily — yesterday's win/loss record
        # must not bias today's horizon weighting (different market context each day).
        self.perf_weights = dict(HORIZON_WEIGHTS)
    
    def in_cooldown(self) -> bool:
        """Edge Case 2: Post-Trade Cooldown (Anti-Churn).
        
        Returns True if currently in post-trade cooldown period.
        Prevents "Double-Dipping" where the system re-enters the same
        trade immediately after hitting target because the model still
        sees a favorable context (e.g., 15-min still bullish after
        5-min trade closed).
        
        Cooldown Duration: 15 minutes (900 seconds)
        
        Returns:
            True if in cooldown, False otherwise
        """
        if self.last_exit_time is None:
            return False
        # 10-minute cooldown. Paper trading showed 4 trades in 75 min on bad days
        # (Mar 12, Apr 20) — 3 min was insufficient to prevent churning on the same
        # failed setup. 10 min forces the micro-structure to change before re-entry.
        elapsed_seconds = (datetime.now() - self.last_exit_time).total_seconds()
        return elapsed_seconds < 600

    def record_signal(self, direction: str, conf: float, regime: int,
                      agreement: float = 0.0):
        """Add confidence to rolling history for percentile ranking."""
        hist = self._conf_history[regime]
        hist.append(conf)
        if len(hist) > self._conf_max_hist:
            hist.pop(0)
        self.last_agreement = agreement

    def conf_percentile(self, conf: float, regime: int) -> float:
        """Return percentile rank of conf vs recent history in this regime.
        
        Returns None if insufficient history (<20 samples), allowing caller
        to skip percentile gate during bootstrap period.
        """
        hist = self._conf_history[regime]
        if len(hist) < 20:
            return None   # insufficient history -> skip percentile check
        arr = np.array(hist)
        return float((arr <= conf).mean() * 100)

    def update_temporal_gate(self, pred_1m: int):
        """
        Track 5m direction flips (Req 5).
        If 5m flips twice in last 3 bars, lock for TEMPORAL_FLIP_LOCK_BARS bars.

        NOTE: Lock counter is now decremented at Gate 6 (before models run) so the
        lock expires correctly even on bars where voting never executes. This method
        only SETS a new lock when a flip pattern is detected — it does NOT decrement.
        """
        self._last_1m_dirs.append(pred_1m)
        if len(self._last_1m_dirs) > 3:
            self._last_1m_dirs.pop(0)

        # Decrement re-lock cooldown
        if self._lock_cooldown_bars > 0:
            self._lock_cooldown_bars -= 1

        # Only set a new lock if not already locked AND cooldown has expired.
        # Without cooldown: lock expires → 3 bars of choppy pred → re-locks immediately
        # → effectively permanent lock all afternoon on volatile days.
        # Cooldown of 10 bars = ~10 min gap between lock episodes.
        if self._lock_bars_remaining == 0 and self._lock_cooldown_bars == 0:
            dirs = self._last_1m_dirs
            if len(dirs) >= 3:
                flips = sum(dirs[i] != dirs[i-1] for i in range(1, len(dirs)))
                if flips >= 2:
                    self._lock_bars_remaining = TEMPORAL_FLIP_LOCK_BARS

    @property
    def temporal_locked(self) -> bool:
        return self._lock_bars_remaining > 0

    def update_perf_weights(self, horizon: int, won: bool):
        """
        Update performance-based horizon weights (Req 5).
        Simple exponential moving average of win/loss per horizon.
        """
        alpha = 0.05
        current = self.perf_weights.get(horizon, HORIZON_WEIGHTS.get(horizon, 0.25))
        reward  = 1.0 if won else 0.0
        self.perf_weights[horizon] = (1 - alpha) * current + alpha * reward
        # Re-normalise so weights sum to 1
        total = sum(self.perf_weights.values()) + 1e-9
        for h in self.perf_weights:
            self.perf_weights[h] /= total

    def check_scarcity(self, direction: str, regime: int, today) -> bool:
        """
        Return True if trade is BLOCKED by scarcity limit (Req 13).
        TRENDING allows more trades; RANGING is strict.
        """
        # Validate today is not None and convert to date for comparison
        if today is None:
            return False  # No date info, allow trade
        # Ensure consistent date comparison
        if isinstance(today, datetime):
            today = today.date()
        elif not isinstance(today, date):
            try:
                today = pd.to_datetime(today).date()
            except:
                return False  # Invalid date, allow trade
        
        if self._scarcity_date != today:
            self.trades_today = {'UP': 0, 'DOWN': 0}
            self._scarcity_date = today
        limit = (MAX_TRADES_PER_DIR_TRENDING if regime == REGIME_TRENDING
                 else MAX_TRADES_PER_DIR_RANGING)
        # Allow extra trades on high-VIX days (set externally by pre-market bias)
        limit += getattr(self, '_extra_trades_today', 0)
        return self.trades_today.get(direction, 0) >= limit

    def record_trade_taken(self, direction: str):
        self.trades_today[direction] = self.trades_today.get(direction, 0) + 1

    def seasonality_bias(self, minute_of_day: int) -> float:
        """Small additive EV bias from intraday seasonality (Req 11)."""
        if not self.seasonality_prior:
            return 0.0
        # Interpolate: find nearest minute bucket
        bucket = round(minute_of_day / 5) * 5
        return float(self.seasonality_prior.get(bucket, 0.0))

    def learn_seasonality(self, df_train: pd.DataFrame,
                          label_col: str, regime: int):
        """
        Learn intraday EV bias per minute bucket (Req 11).
        For each 5-min bucket, compute mean label deviation from 0.5,
        then smooth and clip to SEASONALITY_MAX_BIAS.
        This runs once during training for each regime.
        """
        if df_train.empty or label_col not in df_train.columns:
            return
        df_r = df_train[df_train.get('regime', regime) == regime].copy()
        if df_r.empty or 'minute_of_day' not in df_r.columns:
            return
        df_r['bucket'] = (df_r['minute_of_day'] / 5).round() * 5
        ev_by_bucket   = (df_r.groupby('bucket')[label_col].mean() - 0.5)
        # Smooth with 3-bucket rolling mean to avoid overfitting
        smoothed = ev_by_bucket.rolling(3, center=True, min_periods=1).mean()
        # Clip to max bias
        smoothed = smoothed.clip(-SEASONALITY_MAX_BIAS, SEASONALITY_MAX_BIAS)
        self.seasonality_prior = smoothed.to_dict()

    def update_staleness(self, micro_regime: str, direction: str) -> float:
        """
        Staleness penalty (Req 14): apply confidence decay when micro-regime
        flips against the trade or momentum confirmation disappears.
        Returns penalty to subtract from confidence.
        """
        is_against = (
            (direction == 'UP'   and micro_regime == 'TRENDING_DN') or
            (direction == 'DOWN' and micro_regime == 'TRENDING_UP')
        )
        momentum_gone = micro_regime == 'RANGING'

        if is_against:
            self._stale_conf_penalty = min(0.15, self._stale_conf_penalty + 0.05)
        elif momentum_gone:
            self._stale_conf_penalty = min(0.08, self._stale_conf_penalty + 0.02)
        else:
            # Decay penalty when conditions normalise
            self._stale_conf_penalty = max(0.0, self._stale_conf_penalty - 0.01)

        self._last_micro_regime = micro_regime
        return self._stale_conf_penalty


# Module-level singleton — created here so live_loop can import and use it directly
_signal_state = SignalState()


def generate_signal(row: pd.Series, models: dict, current_regime: int,
                    micro_regime: str = 'UNKNOWN',
                    signal_state: 'SignalState | None' = None,
                    extra_conf_floor: float = 0.0,
                    crisis_bypass: bool = False,
                    regime_conf: float = 0.5) -> dict | None:
    """
    v3.3 EV-First Signal Generation.

    Confidence floors (CONF_BY_HORIZON):
      1m=0.70, 5m=0.65, 15m=0.65, 30m=0.68
      1m/5m floors are intentionally high so 15m/30m dominate the weighted vote.

    Signal gates (ALL must pass in order):
      0.  Post-trade cooldown:  15-min anti-churn lock after each exit
                                [CRISIS bypass: skipped]
      1.  Regime gate:          not CRISIS (unless crisis_bypass=True from kill-switch)
      2.  Session gate:         mod < 335 (14:50 cutoff) AND sp <= 0.92
      3.  Lunch penalty:        12:15-13:00 → -0.05 conf penalty (not hard veto)
      4.  Expiry zone gate:     zone2 (11:30-12:59) and zone4 (14:50+) blocked on Tuesday expiry
      5.  IV floor:             iv_proxy >= 0.05 (market must have enough vol for options)
      5b. Black swan gap gate:  gap > 2.5x day ATR AND mod < 45 → block 45 min
      6.  Temporal gate:        5m prediction flipped twice in last 3 bars → lock 3 bars
                                [was 1m before v3.3 — 1m is too noisy]
                                [CRISIS bypass: skipped]
      3b. TA/ML agreement:      |ta_score| > 1.0 opposing ML → hard veto;
                                mild opposition → graded confidence penalty (max -0.04)
      7.  Micro-regime conf floor: TRENDING>=0.52, RANGING>=0.52
                                + extra_conf_floor (tuesday/lunch now reduce conf, not floor)
      7a. RANGING horizon gate: RANGING regime requires 15m or 30m conviction
                                (conf >= CONF_BY_HORIZON[h]); 1m/5m-only signals blocked
                                [CRISIS bypass: skipped]
      7b. Micro-regime direction: TRENDING_UP → only UP signals; TRENDING_DN → only DOWN
                                [CRISIS bypass: skipped]
      7c-ADX. Flat market:      adx < 20 → hard block (no directional structure)
                                [CRISIS bypass: skipped]
      7c-SQ-ADX. Squeeze+weak: bb_squeeze=1 AND adx < 30 → hard block (theta wins)
                                [CRISIS bypass: skipped; BREAKOUT micro: skipped]
      7c. Volatility expansion: bb_squeeze=1 AND atr_ratio < floor → hard block
      7d. Staleness penalty:    micro-regime opposes direction → up to -0.15 conf decay
                                [CRISIS bypass: skipped]
      7e. IV crush protector:   iv_rank>85 AND iv contracting → -0.10 conf penalty
      8.  Percentile gate:      conf_pctile < 40 → graded penalty (max -0.05);
                                40-60 ambiguous band → hard veto
                                [bootstrap: skipped until 20 samples in history]
      8b. MetaLabeler gate:     P(primary_correct) >= 0.55 required
                                [CRISIS bypass: skipped]
      9.  EV_net gate:          EV_raw - costs*safety*spread_penalty > 0
      10. Signal scarcity:      daily direction limit (disabled: 999/direction)
      11. Time-decay floor:     late-day (mod>=315) requires 10% higher conf floor
      12. Micro-confirmation:   2 consecutive bars same side of VWAP OR strong candle body
                                [CRISIS bypass: skipped]
    """
    ss = signal_state if signal_state is not None else _signal_state

    # Clear stale block reason from previous bar so dashboard never shows old reasons
    global last_block_reason
    last_block_reason = ""

    # Gate 0: Post-trade cooldown removed — signal gates already filter re-entries.
    # Conf floor, agreement, regime, micro-confirmation prevent bad re-entries.

    # Gate 1: Crisis — blocked unless kill-switch bypass conditions were already met
    if current_regime == REGIME_CRISIS and not crisis_bypass:
        return _block("[Gate1] BLOCKED: CRISIS regime")

    # Gate 1b: RANGING regime hard block.
    # Paper data: 5 trades in RANGING, WR=0%, -Rs 2,100 gross loss. Zero evidence of
    # edge in ranging. CRISIS bypass exempt — V-recovery fires out of CRISIS into any regime.
    if BLOCK_RANGING_REGIME and current_regime == REGIME_RANGING and not crisis_bypass:
        return _block("[Gate1b] BLOCKED: RANGING regime — no edge in range-bound market (WR=0% in 31-trade sample)")

    # Gate 1c: TRENDING regime hard block.
    # Paper data: 5 trades in TRENDING, WR=0%, all losses, -Rs 1,600 gross.
    # 6 of 7 all-time winners were CRISIS regime. TRENDING has shown zero live edge.
    # CRISIS bypass exempt — V-recovery fires out of CRISIS into any regime label.
    if BLOCK_TRENDING_REGIME and current_regime == REGIME_TRENDING and not crisis_bypass:
        return _block("[Gate1c] BLOCKED: TRENDING regime — no live edge (WR=0% in 5 trades, all losses)")

    sp  = row.get('session_pct', 0)
    mod = int(row.get('minute_of_day', 0))

    # Gate 2: Session window + late-day cutoff
    # sp > 0.92 blocks signals after ~15:02 (375 × 0.92 = 345 min from 9:15 = 14:50+).
    # Also block signals after minute_of_day=335 (14:50) explicitly:
    # the 1-min bar lag means a 14:50 signal executes at 14:51, leaving 24 min.
    # Any later and theta decay destroys the premium before the trade can develop.
    if sp > 0.92 or mod >= 335:
        return _block(f"[Gate2] BLOCKED: session end (sp={sp:.2f} mod={mod})")

    # Gate 2b: Regime confidence floor
    # regime_conf < 0.60 means the HMM is 40%+ uncertain about which regime we're in.
    # Raised from 0.30 → 0.60: Mar 13 Trade 8 had regime_conf=0.45 and was a loss.
    # At 0.30 the gate only catches near-random regime calls; 0.60 requires genuine
    # regime clarity before entering a position.
    if not crisis_bypass:
        if regime_conf < 0.45:
            return _block(f"[Gate2b] BLOCKED: regime_conf={regime_conf:.2f} < 0.45 — regime too uncertain")
        # 0.45–0.55: HMM uncertain but not random — apply small penalty instead of blocking
        _regime_conf_penalty = 0.04 if regime_conf < 0.55 else 0.0
    else:
        _regime_conf_penalty = 0.0

    # Gate 2d: Entry time-window preference (soft penalty outside 09:45-14:45).
    # Evidence: paper winners concentrated in midday window, but sample is small (n=6).
    # Hard block removed: converts to -0.05 conf penalty outside the window so that
    # strong early-trend or late continuation signals can still pass.
    # CRISIS bypass still skips entirely — V-recovery fires regardless of time.
    _outside_window = not crisis_bypass and (mod < ENTRY_MOD_MIN or mod > ENTRY_MOD_MAX)
    # Penalty applied in confidence flow below (stored as _time_window_penalty)
    _time_window_penalty = 0.05 if _outside_window else 0.0

    # Gate 2e: DOWN-only mode — block CE (UP) entries until live UP edge is validated.
    # Paper data: UP trades 7W/9L net negative after charges. PE trades are profitable.
    # Set DOWN_ONLY_MODE=False in config.py once 30+ CE trades show positive expectancy.
    # CRISIS bypass: skip — crisis V-recovery can be either direction.
    if DOWN_ONLY_MODE and not crisis_bypass:
        # direction not yet known (computed after voting below) — checked again post-vote
        pass   # pre-vote placeholder; actual UP block is applied post-direction below

    # Gate 2c: Stale fractional-differentiation feature guard
    # ret5m_fd and ret15m_fd are first-differenced momentum features computed from
    # a rolling window. If BOTH are exactly 0.0 after the session open (mod > 15),
    # the feature pipeline failed silently (data gap, warmup failure, zero-fill).
    # April 20 had all trades with ret5m_fd=0 AND ret15m_fd=0 — the model ran on
    # incomplete data and the HTF Gate5e (tf5_ret skips when 0) was also bypassed.
    # Skip check for the first 15 minutes (mod <= 15) — FracDiff needs warmup bars.
    if mod > 15 and not crisis_bypass:
        _r5fd  = float(row.get('ret5m_fd',  0.0))
        _r15fd = float(row.get('ret15m_fd', 0.0))
        if _r5fd == 0.0 or _r15fd == 0.0:
            return _block("[Gate2c] BLOCKED: ret5m_fd=0 OR ret15m_fd=0 — feature pipeline stale (partial failure also caught)")

    # Gate 3: Lunch chop penalty (12:15-13:00)
    # WHY: NIFTY often goes flat during European pre-open, BUT some best breakouts
    #      happen at 1:00 PM (Bar 225). Use confidence penalty instead of hard veto.
    lunch_penalty = 0.05 if 180 <= mod <= 225 else 0.0

    # Gate 4: Expiry-day zone gating
    # On expiry day (Tuesday), the session is divided into gamma risk zones.
    # Zone 2 (11:30-12:59) and Zone 4 (14:50+) block entries entirely.
    # Zone 1 and Zone 3 allow entries at reduced size/tighter stops (handled in select_option).
    # This gate only decides whether to generate a signal at all — sizing is in orders.py.
    _is_expiry = row.get('is_expiry', 0) == 1
    if _is_expiry:
        _expiry_rule = get_expiry_rule(True, mod)
        if not _expiry_rule['allow_new']:
            # Zone 2 (11:30-12:59): normally blocked for MM pinning/whipsaw.
            # EXCEPTION: if market is in a strong confirmed trend (adx_1m >= 30
            # AND regime=TRENDING), MMs are being overrun — allow entry at 50% size.
            # adx_1m is used (not adx_5m) because adx_5m can be 0 early session
            # due to insufficient HTF warmup bars.
            _zone_tag = _expiry_rule.get('tag', '')
            if 'ZONE2' in _zone_tag:
                _adx_now   = float(row.get('adx_14', 0))       # 1m ADX
                _adx5_now  = float(row.get('tf5_adx', 0))      # 5m ADX (may be 0)
                _adx_best  = max(_adx_now, _adx5_now)
                _reg_now   = current_regime  # passed in from live.py
                if _adx_best >= 30 and _reg_now == REGIME_TRENDING:
                    # Strong trend overrides Zone 2 block — use 50% size
                    logger.info(f"[Gate4] Zone2 trend-override: adx={_adx_best:.1f}>=30 + TRENDING "
                                f"-> allowing entry at 50% size (mod={mod})")
                    # Don't block — fall through to rest of gates
                    # Size reduction handled by injecting size_mult into signal later
                    pass
                else:
                    return _block(f"[Gate4] BLOCKED: expiry {_zone_tag} (mod={mod})")
            else:
                return _block(f"[Gate4] BLOCKED: expiry {_expiry_rule['tag']} (mod={mod})")
        # Hard cutoff: no new signals in last 40 minutes on expiry day
        if mod >= EXPIRY_FORCE_EXIT_MOD:
            return _block(f"[Gate4] BLOCKED: expiry force-exit zone (mod={mod} >= {EXPIRY_FORCE_EXIT_MOD})")

    # iv_proxy is atr_14_pct * sqrt(bars_per_day) — annualised volatility estimate (~0.5-2%)
    # Use iv_proxy for option pricing; fall back to atr_14_pct only if iv_proxy is missing.
    iv_proxy_val = row.get('iv_proxy', 0)
    iv = iv_proxy_val if iv_proxy_val > 0 else row.get('atr_14_pct', 0)
    # Gate 5: IV floor — require minimum 10% annualised vol.
    # 5% was too lenient: at <10% IV, options have tiny premiums, wide bid-ask,
    # and slow delta response. Theta decay dominates and even correct direction
    # trades lose money. 10% is the minimum for viable options trading in NIFTY.
    if iv < 0.10:
        return _block(f"[Gate5] BLOCKED: iv={iv:.3f} < 0.10 — insufficient volatility for options")

    # Gate 5d: ATM premium floor — block when option premium is too thin for
    # bid-ask spread to be survivable. Uses actual estimated ATM premium (Rs)
    # rather than iv_rank percentile, which becomes meaningless after high-vol
    # periods (post-crash calm days falsely rank at 5th percentile for months).
    # Threshold: Rs 30 minimum. Below this, a 1-2 Rs spread = 3-7% slippage
    # before the trade even moves. Real LTP from Angel One used when available.
    _iv_daily_g5d = safe_value(iv) if iv > 0 else 0.06
    _iv_ann_g5d   = _iv_daily_g5d * np.sqrt(252)
    _spot_g5d     = safe_value(row.get('close', 0))
    _atm_g5d      = int(round(_spot_g5d / STRIKE_ROUNDING) * STRIKE_ROUNDING)
    _dte_g5d      = _next_expiry_mins()
    _ce_ltp_g5d   = safe_value(row.get('atm_ce_ltp', 0))
    _pe_ltp_g5d   = safe_value(row.get('atm_pe_ltp', 0))
    if _ce_ltp_g5d <= 0:
        _ce_ltp_g5d = estimate_option_premium(_spot_g5d, _iv_ann_g5d, _dte_g5d, strike=float(_atm_g5d), option_type='CE')
    if _pe_ltp_g5d <= 0:
        _pe_ltp_g5d = estimate_option_premium(_spot_g5d, _iv_ann_g5d, _dte_g5d, strike=float(_atm_g5d), option_type='PE')
    _atm_premium_g5d = max(_ce_ltp_g5d, _pe_ltp_g5d)
    _min_premium = 20.0 if _is_expiry else 30.0
    if _atm_premium_g5d < _min_premium:
        return _block(f"[Gate5d] BLOCKED: ATM premium={_atm_premium_g5d:.1f} < {_min_premium:.0f} — spread destroys edge")

    # Gate 5b — Black Swan / Gap-Down Shock Filter:
    # Edge Case 3 Mitigation: Stationarity vs Trend Memory
    # If the opening gap is extreme (> 2.5x the day's ATR), the market is in
    # shock-discovery mode for the first 45 minutes.  IV is unreliable, spreads
    # are wide, and direction has low predictability.  
    # 
    # More critically: Fractional differentiation (d=0.35) retains "memory" of
    # pre-gap prices, causing features to remain skewed until FracDiff normalizes.
    # This leads to premature "buy the dip" signals that get crushed.
    # 
    # Solution: Block all entries during gap shock cooldown period.
    gap_pct_raw = float(row.get('gap_pct', 0))  # signed: negative = gap-down
    gap_pct = abs(gap_pct_raw)
    day_atr = float(row.get('day_atr_pct', 0.5))
    if gap_pct > 2.5 * day_atr and mod < 45:
        return _block(f"[Gate5b] BLOCKED: black swan gap ({gap_pct:.2f}% > 2.5x ATR={day_atr:.2f}%)")

    # Gate 5c — Directional Gap Filter:
    # On significant gap-down days (gap < -1.0%), the market has established a
    # bearish context. Buying CE (UP) against this context during the first
    # 90 minutes is a "buy the dip" bet that statistically fails — the model
    # sees oversold RSI and fires UP signals but the trend continues down.
    # Similarly, on gap-up days block PE buys in the first 90 mins.
    # CRISIS bypass does NOT skip this — it's a price-reality filter.
    # Threshold -1.0% / +1.0% catches meaningful gaps without over-filtering.
    # Extended to mod<=150 (11:45 AM) — Mar 19 showed UP losses at mod=122-133
    # (11:17-11:28 AM), just past the old 10:45 cutoff. Gap-down mornings don't
    # reverse until midday; 2.5 hours allows market to find actual direction.
    # NOTE: Gate 5c check moved below voting block (direction must be known first).

    # Gate 6: Temporal consistency gate (Req 5)
    # SOFT PENALTY: 1m direction flip twice in last 3 bars now applies a confidence
    # penalty instead of hard-blocking. A real trend will still clear the final threshold;
    # only marginal signals (which would have been bad entries anyway) are filtered.
    # Counter still decrements correctly so the lock expires after TEMPORAL_FLIP_LOCK_BARS.
    _temporal_penalty = 0.0
    if ss.temporal_locked:
        ss._lock_bars_remaining = max(0, ss._lock_bars_remaining - 1)
        if ss._lock_bars_remaining == 0:
            ss._last_1m_dirs.clear()
            ss._lock_cooldown_bars = 10
        else:
            _temporal_penalty = 0.08   # -0.08 confidence for direction chop
            logger.debug(f"[Gate6] temporal lock penalty={_temporal_penalty:.2f} "
                         f"(bars_remaining={ss._lock_bars_remaining})")

    # 2️⃣ Get active features from first model (all models use same features)
    # This ensures live execution uses the exact feature set from training.
    active_features = None
    if models:
        first_model = next(iter(models.values()))
        active_features = first_model.get('active_features')
    # Fallback to filtered FEATURE_COLS if model doesn't have active_features stored
    if active_features is None:
        active_features = [f for f in FEATURE_COLS if FEATURE_LIVE_OK.get(f, True)]

    # CRITICAL: Validate inputs before prediction
    is_valid, error_msg, nan_count = validate_model_inputs(row, active_features)
    if not is_valid:
        logger.error(f"Model input validation failed: {error_msg}")
        return _block(f"[Gate] BLOCKED: model input invalid — {error_msg}")

    # Build raw feature vector (NaN → 0)
    X_raw = np.array(
        [[0.0 if pd.isna(row.get(c, 0)) else row.get(c, 0) for c in active_features]],
        dtype=np.float32
    )

    # Apply walk-forward live scaler — trained on data up to last fold boundary only.
    # This prevents distribution-shift between training and live inference.
    first_model_for_scaler = models.get(5) or (next(iter(models.values())) if models else None)
    live_scaler = first_model_for_scaler.get('live_scaler') if first_model_for_scaler else None
    if live_scaler is not None:
        try:
            # CRITICAL: Validate scaler shape matches feature count
            # FIX: Raise exception instead of falling back to raw features.
            # WHY: Trading with unscaled data on scaled model = 100% trash signals.
            if hasattr(live_scaler, 'n_features_in_') and live_scaler.n_features_in_ != X_raw.shape[1]:
                error_msg = (f"CRITICAL: Scaler feature mismatch! "
                            f"Expected {live_scaler.n_features_in_}, got {X_raw.shape[1]}. "
                            f"Models were trained on scaled data. Using raw features will cause losses. "
                            f"HALTING TRADING. Please RETRAIN MODELS with current feature set.")
                logger.critical(error_msg)
                raise RuntimeError(error_msg)
            else:
                X = live_scaler.transform(X_raw)
        except RuntimeError:
            raise  # Re-raise the critical error to halt the bot
        except Exception as e:
            logger.critical(f"Scaler transform failed: {e}. HALTING TRADING.")
            raise RuntimeError(f"Scaler failure: {e}")
    else:
        # No scaler found — models were trained on scaled features.
        # Trading on raw features produces garbage predictions → capital loss.
        # HALT immediately. This is a deployment error, not a runtime error.
        error_msg = (
            "CRITICAL: live_scaler is None. Models require scaled input. "
            "Running on raw features would cause systematic prediction errors. "
            "HALTING TRADING. Retrain or restore the model artifact with live_scaler."
        )
        logger.critical(error_msg)
        raise RuntimeError(error_msg)

    # Performance-weighted aggregation (Req 5): use perf_weights instead of fixed
    # HORIZON_WEIGHTS so horizons with better recent track record get more vote.
    weighted_up = 0.0; weighted_dn = 0.0; total_weight = 0.0
    n_valid = 0
    signals = {}
    avg_win_up = 0.0; avg_win_dn = 0.0   # for EV calculation
    # Separate accumulators for true weighted-average confidence.
    # vote_val = w * conf^2 (used for direction voting), so dividing
    # (weighted_up + weighted_dn) / total_weight always equals 1.0.
    # These track sum(w * conf) / sum(w) — the actual mean confidence.
    _conf_weight_sum = 0.0   # sum(w * conf) for participating horizons
    _conf_w_denom    = 0.0   # sum(w)        for participating horizons

    for h, res in sorted(models.items()):
        try:
            # Validate model exists and is fitted
            if res.get('final_model') is None:
                logger.warning(f"[{h}m] Missing final_model, skipping")
                continue
            
            # Mixture of Experts: prefer regime-specific model if available
           # --- Soft regime blending (Patch 4) ---

            # Global model
            global_model = res.get('final_model')
            if global_model is None:
                logger.warning(f"[{h}m] Missing global model, skipping")
                continue

            proba_global = global_model.predict_proba(X)[0]

            if len(proba_global) != 2 or np.any(np.isnan(proba_global)) or np.any(np.isinf(proba_global)):
                logger.error(f"[{h}m] Invalid global model output: {proba_global}")
                continue

            p_global = proba_global[1]

            # Regime model (if available)
            regime_mdl = res.get('regime_models', {}).get(current_regime)

            if regime_mdl is not None:
                proba_regime = regime_mdl.predict_proba(X)[0]

                if len(proba_regime) != 2 or np.any(np.isnan(proba_regime)) or np.any(np.isinf(proba_regime)):
                    logger.warning(f"[{h}m] Invalid regime model output → fallback to global")
                    prob = p_global
                else:
                    p_regime = proba_regime[1]
                    weight = regime_conf  # from HMM
                    prob = weight * p_regime + (1 - weight) * p_global
            else:
                prob = p_global

            # Reconstruct proba array for downstream compatibility
            proba = np.array([1 - prob, prob])

            # Validate probability vector
            if len(proba) != 2 or np.any(np.isnan(proba)) or np.any(np.isinf(proba)):
                logger.error(f"[{h}m] Invalid model output: {proba}")
                continue

            prob = proba[1]  # class-1 probability


            # Preserve raw primary proba BEFORE meta-model overwrites it.
            # Voting and direction use raw proba. Meta-model output is used
            # only for Gate 8b (meta-labeler filter), not for vote weighting.
            # WHY: meta-model predicts P(primary_correct) on a different scale —
            # using it for votes corrupts direction (DOWN 87% → meta 0.48 → silenced,
            # leaving only weak UP votes to win the aggregation).
            proba_raw = proba.copy()

            if res.get('meta_model') is not None:
                atr_pct = row.get('atr_14_pct', 0)
                adx_val = row.get('adx_14', 0)
                dmi_val = row.get('dmi_diff', 0)
                Xmeta = np.array([[proba[1], current_regime,
                                   sp, iv,
                                   0.0 if pd.isna(atr_pct) else atr_pct,
                                   0.0 if pd.isna(adx_val) else adx_val,
                                   0.0 if pd.isna(dmi_val) else dmi_val,
                                   0.0]])   # FSI placeholder (live: no drift avail)
                try:
                    proba_meta = res['meta_model'].predict_proba(Xmeta)[0]
                    # Store meta output for Gate 8b but do NOT overwrite proba_raw
                    signals[h] = signals.get(h, {})
                    signals[h]['meta_proba'] = float(proba_meta[1])
                except Exception:
                    pass

            # Direction and confidence from RAW primary model output only
            pred = 1 if proba_raw[1] > 0.5 else 0
            conf = proba_raw[1] if pred == 1 else proba_raw[0]
            meta_proba_stored = signals.get(h, {}).get('meta_proba')
            signals[h] = {'pred': pred, 'conf': float(conf), 'proba': float(proba_raw[1])}
            # proba_raw is already Platt-calibrated (output of _CalibratedWrapper).
            # We log it as 'calibrated_conf' to distinguish from the raw model score
            # (not separately accessible post-wrapper) and from final_conf (post-blend).
            logger.info(f"[RawProba] h={h}m pred={'UP' if pred==1 else 'DN'} "
                        f"calibrated_conf={conf:.4f} proba_up={proba_raw[1]:.4f} "
                        f"floor={CONF_BY_HORIZON.get(h, CONF_MIN):.2f}")
            if meta_proba_stored is not None:
                signals[h]['meta_proba'] = meta_proba_stored

            # Update temporal gate with 5m prediction (not 1m).
            # 1m flips every few bars by design (54% model, high noise).
            # Locking on 1m flips blocks entries even when 15m trend is clear.
            # 5m has enough bars to confirm a real flip (56% acc, lower noise).
            if h == 5:
                ss.update_temporal_gate(pred)

            # Step 9 (v3.3): EV-weighted aggregation.
            # Direction voting restricted to 15m and 30m only (in normal mode).
            # WHY: 1m (54% acc) and 5m (56% acc) are below break-even after costs.
            # Letting them vote on direction contaminates the high-quality 15m/30m
            # signal and introduces noise. They still run above for temporal gate
            # (ss.update_temporal_gate) and meta-labeler features.
            # CRISIS bypass: all horizons vote — KillSwitch already validated direction.
            # Normal mode: only 15m/30m vote on direction (1m=54% acc, 5m=56% — below
            # break-even after costs; letting them vote contaminates the 15m/30m signal).
            # 1m/5m still run for temporal gate and meta-labeler features.
            _direction_vote_horizons = (1, 5, 15, 30) if crisis_bypass else (15, 30)
            h_conf_floor = CONF_MIN if crisis_bypass else CONF_BY_HORIZON.get(h, CONF_MIN)
            dir_label = 'UP' if pred == 1 else 'DN'
            if conf >= h_conf_floor and h in _direction_vote_horizons:
                w        = ss.perf_weights.get(h, HORIZON_WEIGHTS.get(h, 0.25))
                ev_score = conf   # ev proxy: P(correct) ≈ confidence
                vote_val = w * conf * ev_score
                if pred == 1:
                    weighted_up += vote_val
                    avg_win_up  += conf
                else:
                    weighted_dn += vote_val
                    avg_win_dn  += conf
                total_weight     += vote_val
                _conf_weight_sum += w * conf   # for true weighted-avg conf
                _conf_w_denom    += w
                n_valid          += 1
                logger.debug(f"[Vote] h={h}m pred={dir_label} conf={conf:.3f} floor={h_conf_floor:.3f} vote={vote_val:.4f} COUNTED")
            elif conf >= h_conf_floor:
                logger.debug(f"[Vote] h={h}m pred={dir_label} conf={conf:.3f} TEMPORAL_ONLY (no direction vote for 1m/5m)")
            else:
                logger.debug(f"[Vote] h={h}m pred={dir_label} conf={conf:.3f} floor={h_conf_floor:.3f} BELOW_FLOOR")
        except Exception as e:
            logger.error(f"[{h}m] Prediction failed: {str(e)[:100]}")
            continue  # Skip this horizon, try others

    if n_valid == 0 or total_weight == 0:
        return _block("[Gate] BLOCKED: no valid horizon votes")

    # Directional decision
    if weighted_up > weighted_dn:
        direction = 'UP';   agreement = weighted_up / (weighted_up + weighted_dn)
    else:
        direction = 'DOWN'; agreement = weighted_dn / (weighted_up + weighted_dn)

    logger.debug(f"[Vote] w_up={weighted_up:.4f} w_dn={weighted_dn:.4f} dir={direction} agree={agreement:.3f} n={n_valid}")

    # Agreement threshold: two-tier.
    # < 0.52: genuine tie or inverted consensus — hard block (coin-flip has negative EV after costs)
    # 0.52–0.55: weak majority — early trend or noisy bar; -0.05 penalty, not a block.
    #            Early trends often start with mixed horizon signals; blocking all of them
    #            eliminates the highest-value entries (momentum not yet confirmed on all TFs).
    if agreement < 0.52:
        return _block(f"[Gate] BLOCKED: agreement={agreement:.2f} < 0.52 (no majority)")
    # Store weak-agreement penalty for application in confidence flow below
    _agreement_weak_penalty = 0.03 if agreement < 0.55 else 0.0

    # Gate 2e (post-vote): DOWN-only mode — block CE (UP) direction.
    if DOWN_ONLY_MODE and not crisis_bypass and direction == 'UP':
        return _block("[Gate2e] BLOCKED: DOWN_ONLY_MODE=True — UP (CE) entries disabled until live CE edge validated")

    # Gate 2d-CRISIS-UP (post-vote): confidence penalty for CRISIS UP entries outside window.
    # CRISIS UP outside 11:45-14:15 had 0% WR (6 trades, 0 wins, -32 pts) — strong penalty.
    # Converted from hard block to -0.10 penalty so n=6 doesn't permanently kill the edge.
    # CRISIS DOWN outside window: 2/2 wins — no penalty for DOWN.
    _crisis_up_outside_penalty = 0.0
    if crisis_bypass and direction == 'UP' and (mod < ENTRY_MOD_MIN or mod > ENTRY_MOD_MAX):
        _crisis_up_outside_penalty = 0.10
        logger.debug(f"[Gate2d-CRISIS-UP] CRISIS UP outside window penalty=-0.10 (mod={mod})")

    # Confidence ceiling: post-calibration-fix, real edge never exceeds ~72%.
    # Values above 0.82 after Platt scaling are overfit artefacts — hard cap.
    # This also prevents the OFI boost (live.py) from pushing conf to 0.99.
    # Applied before any downstream gate that checks avg_conf against floors.
    if agreement > 0.82:
        logger.debug(f"[ConfCap] Clipping agreement {agreement:.3f} -> 0.82 (overfit guard)")
        agreement = 0.82

    # ---- Early meta_conf computation (for confidence blend below) -----------
    # Computed here — after direction/signals are determined — so the value is
    # available when we blend before the penalty stack.
    # Gate 8b (hard block on low meta_conf) still lives in its original location
    # and uses this same variable; no double-computation.
    meta_conf = 1.0
    if not crisis_bypass:
        _ref_res_meta = models.get(5) or (next(iter(models.values())) if models else None)
        if _ref_res_meta is not None:
            _ml_early = _ref_res_meta.get('meta_labeler')
            if _ml_early is not None and _ml_early._fitted:
                _primary_proba_early = float(signals.get(5, {}).get('proba',
                    _conf_weight_sum / _conf_w_denom if _conf_w_denom > 0 else 0.5))
                meta_conf = _ml_early.predict_proba(_primary_proba_early, row, current_regime)

    # Gate 5c — Directional Gap Filter: block counter-trend trades in first 60 min on large gaps (>2%)
    if mod <= 60:
        if gap_pct_raw < -2.0 and direction == 'UP':
            return _block(f"[Gate5c] BLOCKED: large gap-down ({gap_pct_raw:.2f}%) — no CE buys before 10:15")
        if gap_pct_raw > 2.0 and direction == 'DOWN':
            return _block(f"[Gate5c] BLOCKED: large gap-up ({gap_pct_raw:.2f}%) — no PE buys before 10:15")

    # Gate 5e — HTF trend alignment + momentum magnitude filter.
    # Two sub-checks (both required, both skipped in crisis_bypass):
    #
    # 5e-A: 3-bar consecutive direction check (existing logic).
    #   tf5_ret_1 and tf5_ret_3 both same sign = 3-bar confirmed trend.
    #   Signal must align with trend — "buy the dip" UP signals in downtrend blocked.
    #
    # 5e-B: 25-point momentum filter (new — from cluster analysis).
    #   Winners had momentum already established: NIFTY moved 25+ pts in signal
    #   direction in the prior 5 minutes before entry. Losers entered into flat moves.
    #   tf5_close_chg = 5m bar close - previous 5m bar close (points, signed).
    #   Require abs(tf5_close_chg) >= 25 pts AND same sign as direction.
    #   Fallback: if tf5_close_chg not available, skip sub-check (avoid false block).
    # Gate 5e — HTF momentum gate: disabled

    # 3️⃣ Directional Agreement Gate — SOFT PENALTY (trend-following mode)
    # Multi-horizon conflict now subtracts confidence rather than hard-blocking.
    # A strong 15m/30m trend signal will still clear the floor even with a conflicting 5m.
    # CRISIS bypass: skip entirely — KillSwitch already validated multi-horizon agreement.
    _dir_agree_penalty = 0.0
    if not crisis_bypass:
        dir_pass, dir_reason = check_directional_agreement(signals)
        if not dir_pass:
            _dir_agree_penalty = 0.08   # -0.08 for horizon conflict
            logger.debug(f"[Gate3-DA] direction agreement penalty={_dir_agree_penalty:.2f} "
                         f"reason={dir_reason}")

    # -------------------------------------------------------------------------
    # CONFIDENCE FLOW — single variable `conf` from here to final floor check.
    #
    # Design:
    #   conf  = raw weighted-average model output (snapshot kept as raw_conf)
    #   Every soft penalty subtracts from conf in-place.
    #   Hard blocks (safety/reality gates) still return _block() before we reach here.
    #   One penalty cap at the end limits total reduction to 0.15.
    #   One final floor check at the end is the only conf vs floor comparison.
    #
    # The old code had two variables (avg_conf, adj_conf), three intermediate floor
    # checks (lines 847, 922, 1130), and the penalty cap was applied after some checks
    # that bypassed it. All of that is collapsed here.
    # -------------------------------------------------------------------------

    conf = _conf_weight_sum / _conf_w_denom if _conf_w_denom > 0 else 0.0

    # ---- Task 4: meta-conf blend -------------------------------------------
    # Blend calibrated model confidence with meta-labeler correctness probability.
    # final_conf = 0.6 * calibrated_conf + 0.4 * meta_conf
    # WHY: meta_conf = P(primary model is correct | market context).
    # It captures regime/session/IV context the primary model cannot see at inference.
    # Both are in [0,1]; the blend shrinks overconfident primary outputs toward
    # the meta estimate, which is trained to predict actual correctness.
    # This is applied BEFORE penalties so that the penalty cap is relative to
    # the blended starting point, not the raw primary output.
    #
    # Fallback: if meta-labeler is not fitted, meta_conf defaults to 0.55 (neutral),
    # so the blend becomes 0.7*conf + 0.3*0.55 = 0.7*conf + 0.165 — mild shrinkage.
    # Weight shifted from 0.6/0.4 to 0.7/0.3: primary model is Platt-calibrated and
    # should dominate; meta-labeler assists without overriding strong model signals.
    _calibrated_conf = conf  # snapshot of Platt-calibrated model output for logging
    _meta_conf_for_blend = meta_conf if meta_conf > 0.0 else 0.55
    conf = 0.7 * conf + 0.3 * _meta_conf_for_blend
    logger.info(
        f"[ConfBlend] calibrated_conf={_calibrated_conf:.4f} "
        f"meta_conf={_meta_conf_for_blend:.4f} "
        f"final_conf(pre-penalty)={conf:.4f} dir={direction}"
    )

    raw_conf = conf   # immutable snapshot — used only for penalty cap calculation

    # [ConfFlow] diagnostic: snapshot all four confidence values at the penalty boundary.
    # raw_model_conf = Platt-calibrated primary model output (pre-blend)
    # meta_conf      = meta-labeler P(primary_correct | context)
    # blended_conf   = 0.7*raw_model_conf + 0.3*meta_conf  (= raw_conf here)
    # final_conf     = blended_conf - sum(all penalties)   [logged again at FinalConf]
    logger.info(
        f"[ConfFlow] raw_model_conf={_calibrated_conf:.4f} "
        f"meta_conf={_meta_conf_for_blend:.4f} "
        f"blended_conf={raw_conf:.4f} "
        f"final_conf=pending_penalties dir={direction}"
    )

    # --- Gate 2b regime_conf penalty (pre-computed above) --------------------
    if _regime_conf_penalty > 0:
        conf -= _regime_conf_penalty
        logger.debug(f"[Gate2b] uncertain regime penalty=-{_regime_conf_penalty:.2f} "
                     f"(regime_conf={regime_conf:.2f}) conf={conf:.3f}")

    # --- Gate 2d time-window penalty + Gate 2d-CRISIS-UP penalty ---------------
    if _time_window_penalty > 0:
        conf -= _time_window_penalty
        logger.debug(f"[Gate2d] outside time window penalty=-{_time_window_penalty:.2f} (mod={mod}) conf={conf:.3f}")
    if _crisis_up_outside_penalty > 0:
        conf -= _crisis_up_outside_penalty
        logger.debug(f"[Gate2d-CRISIS-UP] CRISIS UP outside window penalty=-{_crisis_up_outside_penalty:.2f} conf={conf:.3f}")

    # --- Weak agreement penalty (pre-computed above) -------------------------
    if _agreement_weak_penalty > 0:
        conf -= _agreement_weak_penalty
        logger.debug(f"[Gate3-AGR] weak agreement penalty=-{_agreement_weak_penalty:.2f} (agree={agreement:.3f}) conf={conf:.3f}")

    # --- Gate 3-DA: directional agreement penalty (pre-computed above) -------
    if _temporal_penalty > 0 or _dir_agree_penalty > 0:
        conf -= (_temporal_penalty + _dir_agree_penalty)
        logger.debug(f"[Penalty] temporal={_temporal_penalty:.3f} dir_agree={_dir_agree_penalty:.3f} "
                     f"conf={conf:.3f}")

    # --- Gate 3b: TA/ML opposition -------------------------------------------
    # Hard veto only at abs_ta > 0.85 (near-certain directional contradiction).
    # Everything below that is a graded penalty.
    ta_score  = float(row.get('ta_overall_score', 0.0))
    bb_sq_now = int(row.get('bb_squeeze', 0))
    ta_opposes = (direction == 'DOWN' and ta_score > 0) or (direction == 'UP' and ta_score < 0)
    if ta_opposes:
        abs_ta = abs(ta_score)
        if abs_ta > 0.85:
            return _block(f"[Gate3b] BLOCKED: TA strong opposition — ML={direction} but TA={ta_score:+.3f}")
        ta_penalty = 0.08 if bb_sq_now == 1 else abs_ta * 0.06
        conf -= ta_penalty
        logger.debug(f"[Gate3b] TA penalty={ta_penalty:.3f} (ta={ta_score:+.3f} squeeze={bb_sq_now}) conf={conf:.3f}")

    # --- Tuesday / expiry confidence penalty (context adjustment) ------------
    # Applied as a penalty to conf rather than inflation of the floor.
    # This keeps the scissor effect (floor rises AND conf falls) from compounding.
    dow = int(row.get('dow', 0))
    if dow < 0 or dow > 4:
        dow = 0
    if _is_expiry:
        _expiry_rule_now = get_expiry_rule(True, mod)
        tuesday_penalty = 0.10 if 'ZONE3' in _expiry_rule_now.get('tag', '') else 0.07
    else:
        tuesday_penalty = 0.05 if dow == 1 else 0.0
    if tuesday_penalty > 0:
        conf -= tuesday_penalty
        logger.debug(f"[Penalty] tuesday/expiry={tuesday_penalty:.3f} conf={conf:.3f}")
    if lunch_penalty > 0:
        conf -= lunch_penalty
        logger.debug(f"[Penalty] lunch={lunch_penalty:.3f} conf={conf:.3f}")

    # conf_floor is now the regime/expiry base floor only — no stacked additions.
    is_trending = micro_regime in ('TRENDING_UP', 'TRENDING_DN', 'BREAKOUT')
    is_ranging  = micro_regime == 'RANGING'
    conf_floor  = (CONF_FLOOR_TRENDING if is_trending else
                   CONF_FLOOR_RANGING  if is_ranging  else CONF_MIN)
    conf_floor += extra_conf_floor   # regime-frequency boost from live.py
    if _is_expiry:
        conf_floor = max(conf_floor, EXPIRY_CONF_FLOOR)

    # --- Gate 7a: long-horizon agreement (soft penalty) ----------------------
    if not crisis_bypass:
        long_horizon_agrees = any(
            h in signals
            and signals[h].get('conf', 0) >= CONF_BY_HORIZON.get(h, CONF_MIN)
            and signals[h].get('pred') == (1 if direction == 'UP' else 0)
            for h in (15, 30)
        )
        if not long_horizon_agrees:
            conf -= 0.05
            logger.debug(f"[Gate7a] neither 15m/30m individually above floor — penalty=-0.05 conf={conf:.3f}")

    # --- Gate 7b: micro-regime direction alignment (soft penalty) ------------
    # Was a hard block; converted so early trend reversals are not excluded.
    # Micro-regime (1m lagging) opposing direction = start of a new trend move.
    if is_trending and not crisis_bypass:
        if (direction == 'UP' and micro_regime == 'TRENDING_DN') or \
           (direction == 'DOWN' and micro_regime == 'TRENDING_UP'):
            conf -= 0.06
            logger.debug(f"[Gate7b] micro opposes direction penalty=-0.06 "
                         f"(micro={micro_regime} dir={direction}) conf={conf:.3f}")

    # --- Gate 7b2: trend exhaustion (hard block — price reality) -------------
    adx_val = float(row.get('adx_14', 0.0))
    rsi_val = float(row.get('rsi_14', 50.0))
    if adx_val > 45:
        if direction == 'UP' and rsi_val > 75:
            return _block(f"[Gate7b2] BLOCKED: trend exhausted ADX={adx_val:.1f} RSI={rsi_val:.1f} — chasing UP")
        if direction == 'DOWN' and rsi_val < 25:
            return _block(f"[Gate7b2] BLOCKED: trend exhausted ADX={adx_val:.1f} RSI={rsi_val:.1f} — chasing DOWN")

    # --- Gate 7b3: RSI extreme penalty ---------------------------------------
    # Capped at 0.06 (was 0.08) to reduce overlap with staleness penalty below.
    # RSI extreme + micro-regime opposing = same information (lagging price reversal).
    # Combined RSI+stale is capped after stale computation — see cap block below.
    rsi_penalty = 0.0
    if not crisis_bypass:
        if direction == 'DOWN' and rsi_val < 35:
            rsi_penalty = min(0.06, (35 - rsi_val) / 35 * 0.10)
            conf -= rsi_penalty
            logger.debug(f"[Gate7b3] RSI oversold penalty={rsi_penalty:.3f} (rsi={rsi_val:.1f}) conf={conf:.3f}")
        elif direction == 'UP' and rsi_val > 65:
            rsi_penalty = min(0.06, (rsi_val - 65) / 35 * 0.10)
            conf -= rsi_penalty
            logger.debug(f"[Gate7b3] RSI overbought penalty={rsi_penalty:.3f} (rsi={rsi_val:.1f}) conf={conf:.3f}")

    # --- Gate 7d-PR: pressure ratio for DOWN (hard block when enabled) -------
    if direction == 'DOWN':
        _pr = float(row.get('pressure_ratio', 0.0) or 0.0)
        if _pr > 0 and _pr < PRESSURE_RATIO_DOWN_MIN:
            return _block(f"[Gate7d-PR] BLOCKED: pressure_ratio={_pr:.2f} < {PRESSURE_RATIO_DOWN_MIN} for DOWN")

    # --- Gate 7e: price structure alignment (soft penalties) -----------------
    # ORB opposition in first 30 min retained as hard block (strongest early edge).
    if not crisis_bypass:
        _struct  = float(row.get('struct_score', 0.0))
        _fvg_b   = int(row.get('fvg_bull', 0))
        _fvg_r   = int(row.get('fvg_bear', 0))
        _lsw_up  = int(row.get('liq_sweep_up', 0))
        _lsw_dn  = int(row.get('liq_sweep_dn', 0))
        _nr7     = int(row.get('nr7', 0))
        _orb_up  = int(row.get('or_break_up', 0))
        _orb_dn  = int(row.get('or_break_dn', 0))
        _struct_penalties = []

        if direction == 'UP' and _lsw_up == 1:
            conf -= 0.07; _struct_penalties.append("lsw_up(-0.07)")
        if direction == 'DOWN' and _lsw_dn == 1:
            conf -= 0.07; _struct_penalties.append("lsw_dn(-0.07)")
        if direction == 'UP' and _struct < -0.5 and _fvg_b == 0:
            conf -= 0.06; _struct_penalties.append(f"struct_bear({_struct:.2f})(-0.06)")
        if direction == 'DOWN' and _struct > 0.5 and _fvg_r == 0:
            conf -= 0.06; _struct_penalties.append(f"struct_bull({_struct:.2f})(-0.06)")
        if _nr7 == 1 and _orb_up == 0 and _orb_dn == 0 and abs(_struct) < 0.5:
            conf -= 0.05; _struct_penalties.append("nr7(-0.05)")
        if _orb_dn == 1 and direction == 'UP':
            if mod < 30:
                return _block(f"[Gate7e] BLOCKED: ORB=DN, UP in first 30 min (mod={mod})")
            elif mod < 135:
                conf -= 0.06; _struct_penalties.append("orb_dn_vs_up(-0.06)")
        if _orb_up == 1 and direction == 'DOWN':
            if mod < 30:
                return _block(f"[Gate7e] BLOCKED: ORB=UP, DOWN in first 30 min (mod={mod})")
            elif mod < 135:
                conf -= 0.06; _struct_penalties.append("orb_up_vs_dn(-0.06)")

        if _struct_penalties:
            logger.debug(f"[Gate7e] penalties: {', '.join(_struct_penalties)} conf={conf:.3f}")
        else:
            logger.debug(f"[Gate7e] no penalty (struct={_struct:.2f} nr7={_nr7} orb_up={_orb_up} orb_dn={_orb_dn})")

    # --- Gate 7c-ADX: flat/dead market (hard block <12, soft penalty 12-20) --
    atr_ratio  = float(row.get('atr_ratio', 1.0))
    bb_squeeze = int(row.get('bb_squeeze', 0))
    vol_floor  = 0.70 if is_ranging else 0.85
    if not crisis_bypass:
        if adx_val < 10:
            return _block(f"[Gate7c-ADX] BLOCKED: adx={adx_val:.1f} < 10 — no directional structure")
        elif adx_val < 18:
            adx_penalty = (18 - adx_val) / 18 * 0.10
            conf -= adx_penalty
            logger.debug(f"[Gate7c-ADX] weak ADX penalty={adx_penalty:.3f} (adx={adx_val:.1f}) conf={conf:.3f}")

    # --- Gate 7c-ADX-MAX / CRISIS-ADX: overextension (hard blocks) -----------
    if not crisis_bypass and adx_val > ADX_ENTRY_MAX:
        return _block(f"[Gate7c-ADX-MAX] BLOCKED: adx={adx_val:.1f} > {ADX_ENTRY_MAX} — overextended")
    if crisis_bypass and adx_val > ADX_CRISIS_MAX:
        return _block(f"[Gate7c-CRISIS-ADX] BLOCKED: adx={adx_val:.1f} > {ADX_CRISIS_MAX} in CRISIS")

    # --- Gate 7c-SQ-ADX / VOL: squeeze penalties -----------------------------
    # Both sub-conditions (low ADX + low ATR) fire when bb_squeeze=1 — they are
    # the same underlying condition (contraction). Cap combined at 0.10.
    squeeze_penalty = 0.0
    if not crisis_bypass and bb_squeeze == 1 and micro_regime != 'BREAKOUT':
        if adx_val < 30:
            squeeze_penalty += 0.07
            logger.debug(f"[Gate7c-SQ-ADX] squeeze+weak ADX penalty=0.07 (adx={adx_val:.1f})")
        if atr_ratio < vol_floor:
            squeeze_penalty += 0.05
            logger.debug(f"[Gate7c-VOL] squeeze+low ATR penalty=0.05 (atr_ratio={atr_ratio:.2f})")
        squeeze_penalty = min(squeeze_penalty, 0.10)   # cap: same theme
    elif bb_squeeze == 1 and atr_ratio < vol_floor and micro_regime != 'BREAKOUT':
        squeeze_penalty = 0.04  # small penalty in crisis bypass for theta drag
    if squeeze_penalty > 0:
        conf -= squeeze_penalty
        logger.debug(f"[Gate7c-SQ] total squeeze penalty={squeeze_penalty:.3f} conf={conf:.3f}")

    # --- Staleness penalty ---------------------------------------------------
    # RSI extreme and staleness measure overlapping information (price has been
    # moving against signal direction for several bars). Cap their combined
    # contribution at 0.10 so a single theme can't deduct 0.14+ simultaneously.
    stale_penalty = 0.0 if crisis_bypass else ss.update_staleness(micro_regime, direction)
    if stale_penalty > 0:
        # Apply only the excess above what RSI already deducted, up to the 0.10 cap
        _rsi_stale_combined = rsi_penalty + stale_penalty
        if _rsi_stale_combined > 0.10:
            stale_penalty = max(0.0, 0.10 - rsi_penalty)
            logger.debug(f"[Staleness] RSI+stale combined cap: rsi={rsi_penalty:.3f} stale capped to {stale_penalty:.3f}")
        conf -= stale_penalty
        logger.debug(f"[Staleness] penalty={stale_penalty:.3f} conf={conf:.3f}")

    # --- Gate 7c IV crush ----------------------------------------------------
    iv_crush_penalty = check_iv_crush(row)
    if iv_crush_penalty > 0:
        conf -= iv_crush_penalty
        logger.debug(f"[IVCrush] penalty={iv_crush_penalty:.3f} conf={conf:.3f}")

    # --- Gate 8b: MetaLabeler — two-tier (hard block only at extreme low) ------
    # meta_conf was already incorporated into the confidence blend at the start.
    # Here we apply an additional graduated response:
    #   < 0.45: hard block — meta-labeler strongly predicts primary model is wrong;
    #           at this level the signal has historically been anti-predictive.
    #   0.45–0.50: soft penalty -0.08 — meta-labeler uncertain/mildly negative;
    #              let the penalty cap and final floor decide rather than hard-blocking.
    # WHY: meta-labeler trained on n≈200-500 OOF samples; hard block at 0.50 is
    # too sensitive to meta-model overfit on small calibration sets.
    conf_pctile = ss.conf_percentile(conf, current_regime)   # logging only
    _META_HARD_BLOCK = 0.45
    if not crisis_bypass:
        if meta_conf < _META_HARD_BLOCK:
            return _block(f"[Gate8b] BLOCKED: meta_labeler conf={meta_conf:.3f} < {_META_HARD_BLOCK} (anti-predictive)")
        elif meta_conf < MetaLabeler.META_CONF_THRESH:
            _meta_penalty = 0.08
            conf -= _meta_penalty
            logger.debug(f"[Gate8b] meta_labeler low conf penalty=-{_meta_penalty:.2f} "
                         f"(meta={meta_conf:.3f}) conf={conf:.3f}")

    # --- Momentum: flat 5m return penalty ------------------------------------
    # ADX penalty lives exclusively in Gate 7c-ADX above (no double-penalization).
    _ret5m = float(row.get('ret_5m', 0.0))
    if abs(_ret5m) < 0.001:
        conf -= 0.05
        logger.debug(f"[MomPenalty] flat 5m return penalty=-0.05 (ret5m={_ret5m:.4f}) conf={conf:.3f}")

    # --- Gate 9: EV (deeply negative = hard block; marginal = soft penalty) --
    p_win    = conf
    p_loss   = 1.0 - conf
    ev_raw   = p_win * EV_AVG_WIN_MULT - p_loss * EV_AVG_LOSS_MULT
    iv_rank_current  = float(row.get('iv_rank_approx', 50.0))
    iv_proxy_current = float(row.get('iv_proxy', 0.0))
    ev_net = _ev_net(ev_raw, TOTAL_COST_PCT, mod, iv_rank=iv_rank_current, iv_proxy=iv_proxy_current)
    _ev_min = 0.0015
    if ev_net <= -0.05:
        return _block(f"[Gate9] BLOCKED: ev_net={ev_net:.4f} deeply negative")
    elif ev_net <= _ev_min:
        ev_penalty = min(0.06, abs(ev_net - _ev_min) * 10)
        conf -= ev_penalty
        logger.debug(f"[Gate9] marginal EV penalty={ev_penalty:.3f} (ev_net={ev_net:.4f}) conf={conf:.3f}")

    # --- Seasonality bias (additive, never a veto) ---------------------------
    season_bias = ss.seasonality_bias(mod)
    ev_net_adj  = ev_net + season_bias

    # --- Gate 10: scarcity ---------------------------------------------------
    today = row.get('date', None)
    if ss.check_scarcity(direction, current_regime, today):
        return _block(f"[Gate10] BLOCKED: daily trade limit reached for direction={direction}")

    ss.record_signal(direction, conf, current_regime, agreement=agreement)

    # --- Gate 11b: dead-air penalty ------------------------------------------
    if not crisis_bypass:
        _vwap_vel   = float(row.get('vwap_dev_vel', 0.0))
        _struct_11b = float(row.get('struct_score', 0.0))
        _fvg_bull   = int(row.get('fvg_bull', 0))
        _fvg_bear   = int(row.get('fvg_bear', 0))
        if direction == 'UP':
            _dead_air = _vwap_vel < -0.20 and _struct_11b <= 0.0 and _fvg_bull == 0
        else:
            _dead_air = _vwap_vel > 0.20 and _struct_11b >= 0.0 and _fvg_bear == 0
        if _dead_air:
            conf -= 0.10
            logger.debug(f"[Gate11b] dead-air penalty=-0.10 "
                         f"(vwap_vel={_vwap_vel:.3f} struct={_struct_11b:.2f}) conf={conf:.3f}")

    # -------------------------------------------------------------------------
    # PENALTY CAP: total soft-penalty reduction capped at 0.18.
    # Raised from 0.15: with ~8 independent penalty categories the 0.15 cap was
    # firing on legitimate multi-factor weak setups. 0.18 still prevents stacking
    # from replicating a hard block while allowing meaningful differentiation.
    # Cap applied AFTER all penalties and BEFORE the single final floor check.
    # -------------------------------------------------------------------------
    _total_reduction = raw_conf - conf
    if _total_reduction > 0.18:
        conf = raw_conf - 0.18
        logger.debug(f"[PenaltyCap] {_total_reduction:.3f} reduction capped at 0.18 -> conf={conf:.3f}")

    # -------------------------------------------------------------------------
    # FINAL FLOOR — single check, single variable.
    # conf_floor = regime base (CONF_MIN/TRENDING/RANGING) + extra_conf_floor + expiry.
    # tuesday_penalty and lunch_penalty were subtracted from conf above, NOT added
    # to conf_floor, eliminating the scissor effect.
    # -------------------------------------------------------------------------
    _effective_floor = max(conf_floor, CONF_MIN)
    logger.info(
        f"[FinalConf] raw_model_conf={_calibrated_conf:.4f} "
        f"meta_conf={_meta_conf_for_blend:.4f} "
        f"blended={raw_conf:.4f} "
        f"final_conf={conf:.4f} "
        f"total_penalty={_total_reduction:.4f} floor={_effective_floor:.3f} "
        f"dir={direction}"
    )
    if conf < _effective_floor:
        # Soft buffer: if conf is within 0.02 below floor, allow through when
        # agreement or meta-labeler is strong (genuine signal narrowly penalised).
        _in_buffer = (_effective_floor - conf) <= 0.02
        _buffer_pass = _in_buffer and (agreement > 0.60 or _meta_conf_for_blend > 0.60)
        if not _buffer_pass:
            return _block(f"[FinalFloor] BLOCKED: conf={conf:.3f} < floor={_effective_floor:.3f} "
                          f"(raw={raw_conf:.3f} reduction={_total_reduction:.3f})")
        logger.info(f"[FinalFloor] buffer pass: conf={conf:.3f} within 0.02 of floor={_effective_floor:.3f} "
                    f"agreement={agreement:.3f} meta={_meta_conf_for_blend:.3f}")

    # --- Final selective-entry filters ---------------------------------------
    # These run after final confidence is known and before execution-level
    # micro-confirmation/return dict construction.
    final_conf = conf
    _final_conf_min = max(CONF_MIN, 0.55)
    if final_conf < _final_conf_min:
        logger.info(f"[SKIP] [FinalSelect] final_conf={final_conf:.3f} < min_conf={_final_conf_min:.3f}")
        return None

    _adx_final = row.get('adx_14', None)
    if _adx_final is not None and not pd.isna(_adx_final):
        _adx_final = float(_adx_final)
        if _adx_final < 12:
            logger.info(f"[SKIP] [FinalSelect] adx={_adx_final:.1f} < 12")
            return None

    if 'ret_5m' in row.index:
        _ret5m_final = row.get('ret_5m', None)
        if _ret5m_final is not None and not pd.isna(_ret5m_final):
            _ret5m_final = float(_ret5m_final)
            if abs(_ret5m_final) < 0.001:
                logger.info(f"[SKIP] [FinalSelect] abs(ret_5m)={abs(_ret5m_final):.4f} < 0.001")
                return None

    if final_conf > 0.60 and agreement < 0.55:
        logger.info(f"[SKIP] [FinalSelect] final_conf={final_conf:.3f} > 0.60 but agreement={agreement:.3f} < 0.55")
        return None

    _trades_today = getattr(ss, 'trades_today', {})
    if isinstance(_trades_today, dict):
        _trades_today_total = sum(int(v) for v in _trades_today.values())
    else:
        _trades_today_total = int(_trades_today or 0)
    if _trades_today_total >= 10:
        logger.info(f"[SKIP] [FinalSelect] trades_today={_trades_today_total} >= 10")
        return None

    # Alias for downstream code that still references adj_conf (return dict, micro-confirm)
    adj_conf = conf

    # 7️⃣ Entry Micro-Confirmation (FINAL EXECUTION FILTER)
    # WHY: Models can signal too early. Require price to demonstrate commitment.
    # Use SignalState's vwap_history (reset daily, no memory leak).
    # Rule: 2 consecutive bars above/below VWAP OR strong directional candle.
    # CRISIS bypass: still run for VWAP history update, but don't block on result.
    micro_pass, micro_reason = check_entry_micro_confirmation(
        row, direction, ss.vwap_history,
        micro_regime=micro_regime, avg_conf=adj_conf
    )
    if not micro_pass and not crisis_bypass:
        return _block(f"[Gate] BLOCKED: micro_confirmation={micro_reason}")

    strength = ('STRONG'   if adj_conf >= CONF_STRONG else
                'MODERATE' if adj_conf >= CONF_MODERATE else
                'WEAK')
    time_decay_mult = calculate_time_decay_confidence(mod)
    
    # ==============================================================================
    # OPTION STRIKE MONITOR: Projection Engine for CE/PE
    # ==============================================================================
    # CALCULATION FLOW TRACE:
    # 1. Get current spot price from row['close']
    # 2. Calculate dynamic strikes: CE = spot + 100, PE = spot - 100 (rounded to 50)
    # 3. Estimate current LTP using estimate_option_premium() with current DTE
    # 4. For each horizon (1, 5, 15, 30 min):
    #    a) Calculate projected spot based on model probability
    #    b) Re-price CE/PE at projected spot with reduced DTE
    #    c) Apply 3% slippage penalty
    #    d) Calculate P&L percentage
    # ==============================================================================
    spot = safe_value(row.get('close', 0))
    atr = safe_value(row.get('atr_14', 0))
    # iv here is iv_proxy = atr_14_pct × sqrt(bars_per_day) — this is DAILY IV (%).
    # estimate_option_premium() expects ANNUALISED IV (%).
    # Annualise: daily_iv × sqrt(252) gives real-world IV comparable to NSE VIX (~14-18%).
    # Without this, time_value is 1/sqrt(252) ≈ 1/16th of correct value → Rs 5 instead of Rs 100.
    iv_daily = safe_value(iv) if iv > 0 else 0.06   # typical daily IV fallback ~0.06%
    iv_annpct = iv_daily * np.sqrt(252)              # annualise: ~0.96% × 15.87 ≈ 15.2%
    dte_mins = _next_expiry_mins()

    # Strike selection: ATM for maximum delta sensitivity on short-horizon trades.
    atm = int(round(spot / STRIKE_ROUNDING) * STRIKE_ROUNDING)
    strike_ce = atm
    strike_pe = atm

    # Use real LTP from Angel One if available (injected into row by live_loop).
    # Fall back to formula estimate only when real data is missing.
    ce_ltp_current = safe_value(row.get('atm_ce_ltp', 0))
    pe_ltp_current = safe_value(row.get('atm_pe_ltp', 0))
    if ce_ltp_current <= 0:
        ce_ltp_current = estimate_option_premium(spot, iv_annpct, dte_mins, strike=float(strike_ce), option_type='CE')
    if pe_ltp_current <= 0:
        pe_ltp_current = estimate_option_premium(spot, iv_annpct, dte_mins, strike=float(strike_pe), option_type='PE')
    
    # Projection engine for ALL 4 horizons
    # Uses delta-based projection anchored on real current LTP to avoid
    # Black-Scholes formula divergence when formula DTE differs from reality.
    #
    # projected_ltp = current_ltp + delta × spot_move - theta_decay
    #   delta_ce ≈ +0.50 for ATM CE  (rises when spot rises)
    #   delta_pe ≈ -0.50 for ATM PE  (rises when spot falls)
    #   theta ≈ current_ltp × 0.0004 per minute (empirical for weekly options)
    dynamic_projections = {}
    THETA_PER_MIN = 0.0004   # ~0.04%/min time decay for weekly ATM option
    for h in HORIZONS:
        if h not in signals:
            continue

        proba_up = safe_value(signals[h].get('proba', 0.5))

        # Expected spot move: directional + ATR-scaled
        spot_move = (proba_up - 0.5) * 2.0 * atr * np.sqrt(h)
        proj_spot = safe_value(spot + spot_move)

        # ATM delta ≈ 0.50; adjust for moneyness using same sharpness model
        dte_days  = max(dte_mins, 1.0) / 375.0
        sharpness = 1.0 + 3.0 * np.exp(-dte_days)
        ce_mono   = spot / float(strike_ce) if strike_ce > 0 else 1.0
        pe_mono   = spot / float(strike_pe) if strike_pe > 0 else 1.0
        delta_ce  = float(np.clip(0.50 + sharpness * (ce_mono - 1.0), 0.10, 0.90))
        delta_pe  = float(np.clip(0.50 + sharpness * (1.0 - pe_mono), 0.10, 0.90))

        # Theta decay over horizon minutes
        theta_ce = ce_ltp_current * THETA_PER_MIN * h
        theta_pe = pe_ltp_current * THETA_PER_MIN * h

        # Project LTP: anchor on real current LTP + delta × move - theta
        ce_ltp_proj = max(ce_ltp_current + delta_ce * spot_move - theta_ce, 1.0)
        pe_ltp_proj = max(pe_ltp_current - delta_pe * spot_move - theta_pe, 1.0)

        # Apply slippage
        slippage_factor = 1.0 - OPTION_SLIPPAGE_PCT
        ce_net_ltp = safe_value(ce_ltp_proj * slippage_factor)
        pe_net_ltp = safe_value(pe_ltp_proj * slippage_factor)

        # P&L% vs real current LTP
        ce_pnl_pct = round((ce_net_ltp / ce_ltp_current - 1) * 100, 2) if ce_ltp_current > 0 else 0.0
        pe_pnl_pct = round((pe_net_ltp / pe_ltp_current - 1) * 100, 2) if pe_ltp_current > 0 else 0.0

        dynamic_projections[h] = {
            'proj_spot':   round(proj_spot, 2),
            'ce_ltp_proj': round(ce_ltp_proj, 2),
            'pe_ltp_proj': round(pe_ltp_proj, 2),
            'ce_net_ltp':  round(ce_net_ltp, 2),
            'pe_net_ltp':  round(pe_net_ltp, 2),
            'ce_pnl_pct':  ce_pnl_pct,
            'pe_pnl_pct':  pe_pnl_pct,
            'proba_up':    round(proba_up, 4),
            'dte_proj':    round(max(30.0, dte_mins - h), 1),
        }
    
    return {
        'direction':    direction,
        'strength':     strength,
        'avg_conf':     adj_conf,
        'raw_conf':     raw_conf,
        'stale_penalty':    stale_penalty,
        'iv_crush_penalty': round(iv_crush_penalty, 4),
        'agreement':    agreement,
        'regime':       REGIME_NAMES[current_regime],
        'micro_regime': micro_regime,
        'signals':      signals,
        'spot':         float(row.get('close', 0)),
        'iv_proxy':     float(iv),
        'session':      float(sp),
        'w_up':         round(weighted_up, 3),
        'w_dn':         round(weighted_dn, 3),
        'n_valid':      n_valid,
        'ev_raw':             round(ev_raw,  4),
        'ev_net':             round(ev_net_adj, 4),
        'conf_pctile':        round(conf_pctile, 1) if conf_pctile is not None else 0.0,
        'season_bias':        round(season_bias, 4),
        'conf_floor':         conf_floor,
        'meta_conf':          round(meta_conf, 4),
        'in_transition_zone': False,
        'minute_of_day':      mod,
        'is_expiry':          int(row.get('is_expiry', 0)),
        # v4.0 fields
        'time_decay_mult':    round(time_decay_mult, 3),
        'ofi_boost':          0.0,  # Will be set by live_loop if OFI > 0.80
        'atr_current':        float(row.get('atr_14', 0)),  # For dynamic stops
        'fft_cycle':          float(row.get('fft_cycle', 0)),
        'frac_diff':          float(row.get('frac_diff_close', 0)),
        # Dynamic Strike Monitor fields
        'strike_ce':          strike_ce,
        'strike_pe':          strike_pe,
        'ce_ltp_current':     round(ce_ltp_current, 2),
        'pe_ltp_current':     round(pe_ltp_current, 2),
        'dynamic_projections': dynamic_projections,
    }


