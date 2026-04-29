"""
Signal generation: v5 clean trend-following system.

Strategy: TRENDING regime only → wait → pullback to VWAP → momentum resumes → enter.

Gate structure (3 hard filters + ML + entry):
  1. Regime:    TRENDING only  (ADX-based)
  2. Session:   09:30–14:15 window
  3. IV filter: iv_rank < 80th percentile (avoid overpriced options)
  ML.           P(direction) ≥ 0.60 from 15m/30m models; direction must match regime
  Entry:        Pullback-resume (price retraces to VWAP zone, momentum resumes)
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
    # V5 clean system constants
    V5_HORIZONS, V5_HORIZON_WEIGHTS,
    V5_CONF_ENTRY, V5_CONF_STRONG,
    V5_ENTRY_MOD_MIN, V5_ENTRY_MOD_MAX,
    V5_IV_RANK_MAX,
    V5_PB_VWAP_BAND_PCT, V5_PB_MIN_IMPULSE_PCT,
    V5_PB_DEPTH_MIN, V5_PB_DEPTH_MAX, V5_PB_MAX_WAIT_BARS,
    V5_MOM_VWAP_VEL_MIN, V5_MOM_TICK_IMB_MIN,
    V5_STOP_PCT, V5_TARGET_PCT,
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
from ..execution.v5_risk import V5RiskState, v5_lot_size, _v5_risk_state

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

        # Pullback-resume entry state (v4.0)
        # After a "trend identified" signal fires, we wait for price to pull back
        # into the VWAP zone before taking the actual entry on resumption.
        # _pb_direction: direction of the pending setup ('UP', 'DOWN', or None)
        # _pb_bars_since_signal: bars elapsed since trend was identified
        # _pb_touched_vwap: True once price entered the VWAP zone (within 1 ATR)
        # _pb_max_bars: discard setup if pullback never arrives within this many bars
        self._pb_direction: str | None = None
        self._pb_bars_since_signal: int = 0
        self._pb_touched_vwap: bool = False
        self._pb_max_bars: int = 12   # ~12 min; after that the setup expires

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
        # Reset pullback-resume entry state
        self._pb_direction = None
        self._pb_bars_since_signal = 0
        self._pb_touched_vwap = False
    
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
        Staleness penalty: decay confidence when micro-regime flips against the trade.
        Trend-only: only penalises when micro_regime directly opposes direction.
        The old 'momentum_gone' (RANGING) branch has been removed — RANGING
        micro_regime cannot be reached in trend-only mode (Gate1 hard-blocks it).
        """
        is_against = (
            (direction == 'UP'   and micro_regime == 'TRENDING_DN') or
            (direction == 'DOWN' and micro_regime == 'TRENDING_UP')
        )
        if is_against:
            self._stale_conf_penalty = min(0.15, self._stale_conf_penalty + 0.05)
        else:
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
    v4.1 Pure Trend-Following Signal Generation.

    Strategy: TRENDING regime only → wait → pullback to VWAP → momentum resumes → enter.

    Direction source: 15m/30m ML weighted vote (authoritative).
    TA role: veto only at extreme opposition (|ta_score| > 0.85); soft penalty below that.

    Removed in v4.1 (non-trend logic):
      - RANGING conf_floor (range-trading)
      - Gate7b opposing micro-regime penalty (anti-trend fade)
      - staleness momentum_gone branch (RANGING-specific)
      - liq_sweep penalties (reversal/fade instinct)
      - nr7 + flat struct penalties (range contraction)
      - MomPenalty flat ret_5m (superseded by EntryV4)
      - Gate11b dead-air penalty (superseded by EntryV4)
      - FinalSelect ret_5m redundant check

    Signal gates (ALL must pass in order):
      1.  Regime:          TRENDING only (RANGING/UNCERTAIN/CRISIS hard-blocked unless bypass)
      2.  Session:         mod < 335 AND sp <= 0.92
      2b. Regime conf:     HMM certainty floor (hard block < 0.10, penalty 0.10–0.55)
      2c. Feature guard:   ret5m_fd/ret15m_fd != 0 (pipeline sanity)
      2d. Time window:     soft −0.05 outside 09:45–14:45
      2e. DOWN-only mode:  optional CE block (config flag)
      3.  Lunch penalty:   12:15–13:00 → −0.05 conf
      4.  Expiry zones:    zone2/4 blocked; zone2 trend-override at ADX≥30
      5.  IV floor:        iv_proxy ≥ 0.10
      5b. Gap shock:       gap > 2.5×ATR AND mod < 45 → block
      5c. Gap direction:   counter-gap entries blocked until mod 150
      5d. Premium floor:   ATM premium ≥ Rs 30 (Rs 20 on expiry)
      6.  Temporal gate:   5m flip twice in 3 bars → −0.08 conf penalty
      ML. Voting:          15m/30m weighted vote → direction (1m/5m temporal only)
      3b. TA opposition:   |ta_score| > 0.85 → hard veto; lower → graded penalty
      7.  Conf floor:      TRENDING micro ≥ CONF_FLOOR_TRENDING else CONF_MIN
      7a. Long-horizon:    neither 15m/30m above floor → −0.05
      7c-ADX. Structure:   adx < 12 → hard block; adx < 20 + contracting slope → penalty
      7c-SQ. Squeeze:      bb_squeeze + weak ADX/ATR → up to −0.10
      7d. Staleness:       micro opposes direction → up to −0.15 (trend flip scenario only)
      7e. ORB:             ORB opposite direction in first 30 min → hard block
      7f. IV crush:        iv_rank > 85 + contracting → −0.10
      8b. MetaLabeler:     < 0.45 → −0.05; 0.45–0.55 → −0.08
      9.  EV gate:         ev_net ≤ −0.05 → hard block; marginal → penalty
      10. Scarcity:        daily direction cap
      12. EntryV4:         pullback-resume — Phase 1 trend identified, Phase 2 VWAP
                           retest + vwap_dev_vel/tick_imbalance resumption → ENTER
                           [CRISIS bypass: skipped]
    """
    ss = signal_state if signal_state is not None else _signal_state

    # Clear stale block reason from previous bar so dashboard never shows old reasons
    global last_block_reason
    last_block_reason = ""

    # Gate 0: Post-trade cooldown removed — signal gates already filter re-entries.
    # Conf floor, agreement, regime, micro-confirmation prevent bad re-entries.

    # Gate 1: Pure trend-following — only trade TRENDING or CRISIS (bypass) regime.
    # RANGING, UNCERTAIN and all other regimes are hard-blocked: no edge in non-trending markets.
    if not crisis_bypass and current_regime != REGIME_TRENDING:
        return _block(f"[Gate1] BLOCKED: regime={REGIME_NAMES.get(current_regime,'?')} — only TRENDING regime allowed")

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
        if regime_conf < 0.10:
            return _block(f"[Gate2b] BLOCKED: regime_conf={regime_conf:.2f} < 0.10 — regime too uncertain")
        # 0.10–0.55: HMM uncertain but not random — apply small penalty instead of blocking
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

    # Gate 5c — Directional Gap Filter (trend-aligned, v4.1):
    # A large gap establishes the day's dominant trend direction.
    # Trading AGAINST the gap in the first 150 min is counter-trend — blocked.
    # Trading WITH the gap is allowed (trend continuation in gap direction).
    # This is NOT mean-reversion logic; it enforces trend alignment.
    # CRISIS bypass does NOT skip — it's a price-reality filter.
    # NOTE: Gate 5c check is below the voting block (direction must be known first).

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

    # v4.0: ML sets direction; TA is a soft confirmation, NOT the gatekeeper.
    #
    # WHY: ta_overall_score is built from RSI(14), DMI(14), MACD(12,26) — all
    # lagging by 8–14 bars. Using TA to SET direction guarantees the direction
    # call is stale before entry. The 15m/30m ML models are the authoritative
    # direction source; TA's role is to flag extreme opposition only.
    #
    # Hard veto (Gate 3b, later in pipeline) still fires at |ta_score| > 0.85.
    # Mild TA opposition (0 < |ta_score| <= 0.85) becomes a soft penalty.
    # ta_overall_score == 0 is no longer a block — it means TA is neutral,
    # which is fine for early-trend entries where lagging indicators haven't
    # flipped yet.
    _ta_dir_score = float(row.get('ta_overall_score', 0.0))
    _ml_dir = 'UP' if weighted_up > weighted_dn else 'DOWN'
    direction = _ml_dir
    _ml_disagrees = False   # TA/ML mismatch penalty applied in Gate 3b below

    # VWAP alignment gate removed (v4.0).
    # The pullback-resume entry (Gate 12 / EntryV4) now handles VWAP positioning:
    # it waits for price to retest the VWAP zone and enter on resumption.
    # Keeping a hard "price must be on correct side" gate here would block Phase 2
    # entries where price legitimately touches VWAP during the pullback.

    # Agreement = ML vote share in direction (confidence proxy)
    if direction == 'UP':
        agreement = weighted_up / (weighted_up + weighted_dn) if (weighted_up + weighted_dn) > 0 else 0.5
    else:
        agreement = weighted_dn / (weighted_up + weighted_dn) if (weighted_up + weighted_dn) > 0 else 0.5

    logger.debug(f"[Vote] w_up={weighted_up:.4f} w_dn={weighted_dn:.4f} ta_score={_ta_dir_score:+.3f} ml_dir={_ml_dir} agree={agreement:.3f} n={n_valid}")

    # Agreement threshold: two-tier.
    # < 0.52: genuine tie or inverted consensus — hard block (coin-flip has negative EV after costs)
    # 0.52–0.55: weak majority — early trend or noisy bar; -0.05 penalty, not a block.
    #            Early trends often start with mixed horizon signals; blocking all of them
    #            eliminates the highest-value entries (momentum not yet confirmed on all TFs).
    if agreement < 0.52:
        return _block(f"[Gate] BLOCKED: agreement={agreement:.2f} < 0.52 (no majority)")
    # Store weak-agreement penalty for application in confidence flow below
    _agreement_weak_penalty = 0.03 if agreement < 0.55 else 0.0

    # Gate 2f removed — pure trend-following: ML direction is authoritative.
    # TA score used only as a soft penalty (Gate 3b) not a direction override.

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
    _ml_align_penalty = 0.06 if _ml_disagrees else 0.0
    if _temporal_penalty > 0 or _dir_agree_penalty > 0 or _ml_align_penalty > 0:
        conf -= (_temporal_penalty + _dir_agree_penalty + _ml_align_penalty)
        logger.debug(f"[Penalty] temporal={_temporal_penalty:.3f} dir_agree={_dir_agree_penalty:.3f} "
                     f"ml_align={_ml_align_penalty:.3f} conf={conf:.3f}")

    # --- Pullback phase detection -------------------------------------------
    # When EntryV4 has registered a direction and is actively tracking a pullback
    # (ss._pb_direction == direction), TA weakening, ADX contraction, and staleness
    # are EXPECTED artifacts of the pullback itself — not evidence of a bad setup.
    # Applying these penalties during pullback bars pushes conf below floor on the
    # ideal entry bar (at VWAP zone low), forcing late entry at the resumption spike.
    # Fix: suppress these three soft penalties during tracked pullback bars.
    # Hard vetoes (TA > 0.85, ADX < 12) are preserved regardless.
    _in_tracked_pullback = (ss._pb_direction is not None and ss._pb_direction == direction)
    if _in_tracked_pullback:
        logger.debug(f"[PullbackPhase] active pullback tracking for {direction} — "
                     f"suppressing TA/ADX/staleness soft penalties (bar {ss._pb_bars_since_signal})")

    # --- Gate 3b: TA/ML opposition -------------------------------------------
    # Hard veto only at abs_ta > 0.85 (near-certain directional contradiction).
    # Everything below that is a graded penalty.
    # During tracked pullback: soft penalty suppressed — TA naturally opposes during
    # pullback (RSI/MACD weaken as price retraces) and is expected, not a bad signal.
    ta_score  = float(row.get('ta_overall_score', 0.0))
    bb_sq_now = int(row.get('bb_squeeze', 0))
    ta_opposes = (direction == 'DOWN' and ta_score > 0) or (direction == 'UP' and ta_score < 0)
    if ta_opposes:
        abs_ta = abs(ta_score)
        if abs_ta > 0.85:
            return _block(f"[Gate3b] BLOCKED: TA strong opposition — ML={direction} but TA={ta_score:+.3f}")
        if not _in_tracked_pullback:
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

    # Trend-only: a single conf_floor regardless of micro_regime sub-state.
    # RANGING micro_regime cannot reach here (Gate1 already hard-blocks RANGING regime;
    # UNCERTAIN is also blocked). CONF_FLOOR_RANGING is pure range-trading logic — removed.
    is_trending = micro_regime in ('TRENDING_UP', 'TRENDING_DN', 'BREAKOUT')
    conf_floor  = CONF_FLOOR_TRENDING if is_trending else CONF_MIN
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

    # Gate 7b removed (trend-only, v4.1):
    # micro_regime opposing direction = start of a NEW trend — this is exactly the
    # pullback-resume scenario we want to trade. Penalising it was mean-reversion logic.
    # The pullback-resume entry (EntryV4) handles this correctly via vwap_dev_vel + tick_imb.
    adx_val = float(row.get('adx_14', 0.0))

    # --- Gate 7d-PR: pressure ratio for DOWN (hard block when enabled) -------
    if direction == 'DOWN':
        _pr = float(row.get('pressure_ratio', 0.0) or 0.0)
        if _pr > 0 and _pr < PRESSURE_RATIO_DOWN_MIN:
            return _block(f"[Gate7d-PR] BLOCKED: pressure_ratio={_pr:.2f} < {PRESSURE_RATIO_DOWN_MIN} for DOWN")

    # --- Gate 7e: ORB alignment (trend-only, v4.1) ----------------------------
    # Removed (trend-following rationale):
    #   liq_sweep_up/dn penalties: a sweep of a prior high in an UP trend IS the
    #     continuation move — penalising it is a reversal/fade instinct, not trend-following.
    #   struct_bear/struct_bull penalties: bearish structure in an UP trend is the
    #     pullback zone EntryV4 is designed to enter FROM. Penalising it is wrong.
    #   nr7 (narrow-range-7): range contraction before a breakout is a SETUP, not a penalty.
    #
    # Retained: ORB hard-block in the first 30 min — price has already broken out
    # in the opposite direction; the opening structure is the dominant trade.
    if not crisis_bypass:
        _orb_up = int(row.get('or_break_up', 0))
        _orb_dn = int(row.get('or_break_dn', 0))
        if _orb_dn == 1 and direction == 'UP' and mod < 30:
            return _block(f"[Gate7e] BLOCKED: ORB=DN, UP in first 30 min (mod={mod})")
        if _orb_up == 1 and direction == 'DOWN' and mod < 30:
            return _block(f"[Gate7e] BLOCKED: ORB=UP, DOWN in first 30 min (mod={mod})")
        logger.debug(f"[Gate7e] ORB check: orb_up={_orb_up} orb_dn={_orb_dn} dir={direction} mod={mod}")

    # --- Gate 7c-ADX: flat/dead market — slope-based (v4.0) -------------------
    # v4.0 change: replace ADX LEVEL floor with ADX SLOPE check.
    #
    # WHY: ADX(14) lags momentum bursts by ~7 bars. Requiring adx >= 20 means
    # the trend has been running for 7+ bars before entry is allowed — exactly
    # the late-entry problem. A RISING ADX of 16 is a far better entry than a
    # FALLING ADX of 24 (which signals momentum exhaustion).
    #
    # New rule:
    #   Hard block: adx < 12 (truly dead, no structure at all) — unchanged.
    #   Slope check: adx_slope_3b = adx_14[t] - adx_14[t-3]. If slope <= 0
    #                AND adx < 20, momentum is contracting → soft penalty.
    #   Rising ADX (slope > 0): no penalty regardless of level above 12.
    #   5m strong trend override: unchanged (adx_5m >= 60 skips 1m checks).
    atr_ratio  = float(row.get('atr_ratio', 1.0))
    bb_squeeze = int(row.get('bb_squeeze', 0))
    vol_floor  = 0.85   # trend-only: no RANGING regime, single ATR floor
    adx_5m_val = float(row.get('tf5_adx', 0.0))
    _5m_strong_trend = adx_5m_val >= 60 and current_regime == REGIME_TRENDING
    # adx_slope_3b: change in 1m ADX over the last 3 bars (positive = rising).
    # Stored as a feature 'adx_slope' if available; compute on-the-fly otherwise.
    _adx_slope_3b = float(row.get('adx_slope', float('nan')))
    if np.isnan(_adx_slope_3b):
        # Fallback: feature not in row (old model) — treat as neutral (0)
        _adx_slope_3b = 0.0
    if not crisis_bypass:
        adx_for_filter = adx_5m_val if adx_5m_val > 0 else adx_val
        if adx_for_filter < 12 and not _5m_strong_trend:
            return _block(f"[Gate7c-ADX] BLOCKED: adx_5m={adx_5m_val:.1f} adx_1m={adx_val:.1f} < 12 — no structure")
        elif adx_for_filter < 20 and _adx_slope_3b <= 0 and not _5m_strong_trend:
            # Contracting momentum AND still weak — penalise; rising ADX gets a pass.
            # During tracked pullback: ADX naturally contracts as price retraces — suppress
            # this penalty so the ideal entry at VWAP zone low is not blocked by expected
            # pullback dynamics. Hard block (ADX < 12) still applies.
            if not _in_tracked_pullback:
                adx_penalty = (20 - adx_for_filter) / 20 * 0.10
                conf -= adx_penalty
                logger.debug(f"[Gate7c-ADX] contracting ADX penalty={adx_penalty:.3f} "
                             f"(adx={adx_for_filter:.1f} slope={_adx_slope_3b:+.2f}) conf={conf:.3f}")
            else:
                logger.debug(f"[Gate7c-ADX] pullback phase — suppressing ADX contraction penalty "
                             f"(adx={adx_for_filter:.1f} slope={_adx_slope_3b:+.2f})")
        elif _5m_strong_trend and adx_val < 18:
            logger.debug(f"[Gate7c-ADX] 1m ADX weak ({adx_val:.1f}) but 5m ADX strong ({adx_5m_val:.1f}) — skipping penalty")

    # --- Gate 7c-ADX-MAX removed — strong trends should not be blocked -----------
    # ADX_ENTRY_MAX=40 was blocking entries at peak trend strength (ADX=46 with TRENDING_DN).
    # if not crisis_bypass and adx_val > ADX_ENTRY_MAX: return _block(...)
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
    # During tracked pullback: micro_regime may read TRENDING_DN during a valid UP
    # pullback, accumulating staleness each bar (+0.05/bar). This is an expected
    # read during the retracement — suppressing the penalty prevents the staleness
    # state machine from blocking the ideal entry at the VWAP zone low.
    # State is still updated (ss.update_staleness called) so it reflects reality
    # when the pullback ends — only the conf deduction is skipped during tracking.
    stale_penalty = 0.0 if crisis_bypass else ss.update_staleness(micro_regime, direction)
    if stale_penalty > 0 and not _in_tracked_pullback:
        stale_penalty = min(stale_penalty, 0.10)   # cap at 0.10
        conf -= stale_penalty
        logger.debug(f"[Staleness] penalty={stale_penalty:.3f} conf={conf:.3f}")
    elif stale_penalty > 0 and _in_tracked_pullback:
        logger.debug(f"[Staleness] pullback phase — suppressing stale_penalty={stale_penalty:.3f} "
                     f"(micro_regime={micro_regime} direction={direction})")

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
            conf -= 0.05  # soft penalty — meta-labeler trained on <30 trades, hard block not justified
        elif meta_conf < MetaLabeler.META_CONF_THRESH:
            _meta_penalty = 0.08
            conf -= _meta_penalty
            logger.debug(f"[Gate8b] meta_labeler low conf penalty=-{_meta_penalty:.2f} "
                         f"(meta={meta_conf:.3f}) conf={conf:.3f}")

    # MomPenalty removed (trend-only, v4.1):
    # ret_5m < 0.001 (flat 5m return) is already handled by EntryV4 — the pullback-resume
    # gate requires vwap_dev_vel + tick_imbalance to confirm momentum before entry.
    # A flat ret_5m during the pullback phase is expected and correct; penalising it
    # here would block entries on valid pullback bars.

    # --- Gate 9: EV (deeply negative = hard block; marginal = soft penalty) --
    # Use raw_conf (pre-penalty blended confidence) for EV calculation.
    # Post-penalty conf can go below 0.5 from stacked soft penalties on weak-but-valid
    # market conditions, making EV negative even when the model has genuine edge.
    # EV measures model prediction quality; penalties adjust entry confidence, not EV.
    p_win    = raw_conf
    p_loss   = 1.0 - raw_conf
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

    # Gate 11b removed (trend-only, v4.1):
    # Dead-air detection (vwap_dev_vel opposing + flat struct + no FVG) is superseded
    # by EntryV4 which requires vwap_dev_vel + tick_imbalance to confirm RESUMPTION
    # before the entry is allowed. If momentum is dead, EntryV4 blocks at the entry
    # gate — no need to also penalise confidence here. Double-counting this signal
    # was compressing conf artificially on valid pullback bars.

    # -------------------------------------------------------------------------
    # PENALTY CAP: total soft-penalty reduction capped at 0.10.
    # FIX: lowered from 0.18 — the 0.18 cap allowed signals with 5-8 simultaneous
    # negative conditions to survive. Three genuine penalty categories (e.g. ADX
    # contraction + TA opposition + squeeze) already represent a bad setup and
    # should fail the floor check. 0.10 restores multiplicative rejection power.
    # Cap applied AFTER all penalties and BEFORE the single final floor check.
    # -------------------------------------------------------------------------
    _total_reduction = raw_conf - conf
    if _total_reduction > 0.20:
        return _block(f"[PenaltyBudget] BLOCKED: total_penalty={_total_reduction:.3f} > 0.20 "
                      f"(raw={raw_conf:.3f} dir={direction})")
    if _total_reduction > 0.10:
        conf = raw_conf - 0.10
        logger.debug(f"[PenaltyCap] {_total_reduction:.3f} reduction capped at 0.10 -> conf={conf:.3f}")

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
        # FIX: removed soft buffer — the 0.02-below-floor pass with agreement>0.60
        # was allowing signals that failed the floor to re-enter on a weak majority.
        # The floor is the floor. A conf of 0.50 when floor is 0.52 does not have edge.
        return _block(f"[FinalFloor] BLOCKED: conf={conf:.3f} < floor={_effective_floor:.3f} "
                      f"(raw={raw_conf:.3f} reduction={_total_reduction:.3f})")

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

    # FinalSelect ret_5m check removed (trend-only, v4.1):
    # Duplicate of EntryV4 momentum check — see MomPenalty removal note above.

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

    # Alias for downstream code that still references adj_conf (return dict)
    adj_conf = conf

    # ==========================================================================
    # ENTRY GATE v4.0: PULLBACK-RESUME  (replaces MomGate + 2-bar VWAP confirm)
    #
    # Design rationale (from quant review):
    #   Old entry: required ret_1m > 0.35×ATR AND 2 consecutive VWAP bars.
    #   Both conditions confirm an EXISTING move → systematic entry 3–6 bars late.
    #   The 10% premium stop fires on the natural pullback that follows any burst.
    #
    # New entry: two phases per signal.
    #   Phase 1 — TREND IDENTIFIED (current bar):
    #     All upstream gates cleared → trend is confirmed.
    #     If price is already in the VWAP zone (within 1×ATR), skip to Phase 2
    #     immediately (price pulled back before model fired — ideal setup).
    #     Otherwise record the direction and wait.
    #
    #   Phase 2 — PULLBACK COMPLETE → ENTRY:
    #     On subsequent bars (tracked via ss._pb_*), wait for price to retest
    #     the VWAP zone, then enter on the first bar that resumes the direction.
    #     Resumption signal: vwap_dev_vel moving in direction + tick_imbalance aligned.
    #     Expires after _pb_max_bars (12 bars / ~12 min) to avoid stale setups.
    #
    # CRISIS bypass: skip entirely — V-recovery entries should not wait for pullback.
    # ==========================================================================

    _close_now  = float(row.get('close', 0.0))
    _vwap_now   = float(row.get('vwap', 0.0))
    _atr14_now  = float(row.get('atr_14', 1.0)) or 1.0
    _vwap_vel   = float(row.get('vwap_dev_vel', 0.0))   # 3-bar diff of vwap_dist (%)
    _tick_imb   = float(row.get('tick_imbalance', 0.0))  # 20-bar rolling tick direction

    # "In VWAP zone" = price within 1×ATR of VWAP (either side)
    _in_vwap_zone = abs(_close_now - _vwap_now) <= _atr14_now if _vwap_now > 0 else False

    # Real-time momentum: vwap_dev_vel positive = price moving AWAY from VWAP upward.
    # tick_imbalance > 0 = buy-side pressure dominant.
    # These are bar-level (not 14-bar averages) — they measure what is happening NOW.
    # FIX: raised from 0.02/0.05 — old thresholds were noise-level (~4pt VWAP move, 1 net uptick).
    # 0.08 vwap_dev_vel ≈ 17+ points directional move in 3 bars (real resumption candle).
    # 0.15 tick_imbalance = net 3 more upticks than downticks in 20 bars (filters dead-air).
    _resuming_up   = _vwap_vel > 0.08 and _tick_imb > 0.15
    _resuming_down = _vwap_vel < -0.08 and _tick_imb < -0.15

    if not crisis_bypass:
        # ---- Update VWAP history (still maintained for any downstream consumer) ----
        _vwap_pos = 1 if _close_now > _vwap_now else 0
        ss.vwap_history.append(_vwap_pos)

        # ---- Phase 1: is there an active pending setup? -------------------------
        if ss._pb_direction is not None:
            # Advance the bar counter — expire stale setups
            ss._pb_bars_since_signal += 1
            if ss._pb_bars_since_signal > ss._pb_max_bars:
                logger.debug(f"[EntryV4] pullback setup expired after {ss._pb_bars_since_signal} bars "
                             f"(direction={ss._pb_direction})")
                ss._pb_direction = None
                ss._pb_touched_vwap = False
                ss._pb_bars_since_signal = 0

        # ---- New setup registered on this bar -----------------------------------
        # If no pending setup, register the trend identification produced by the
        # upstream gates on this bar.
        if ss._pb_direction is None:
            if _in_vwap_zone:
                # Price is already at VWAP — no need to wait; treat as immediate
                # pullback-complete. Fall through to resumption check below.
                ss._pb_direction = direction
                ss._pb_touched_vwap = True
                ss._pb_bars_since_signal = 0
                logger.debug(f"[EntryV4] trend={direction} price already in VWAP zone "
                             f"(close={_close_now:.1f} vwap={_vwap_now:.1f} atr={_atr14_now:.1f}) "
                             f"→ immediate pullback-complete")
            else:
                # Price extended away from VWAP — register and wait for retest
                ss._pb_direction = direction
                ss._pb_touched_vwap = False
                ss._pb_bars_since_signal = 0
                logger.debug(f"[EntryV4] trend={direction} registered, waiting for VWAP retest "
                             f"(close={_close_now:.1f} vwap={_vwap_now:.1f} dist={abs(_close_now-_vwap_now):.1f}pt)")
                return _block(f"[EntryV4] WAIT: {direction} trend identified, awaiting VWAP pullback "
                              f"(dist={abs(_close_now-_vwap_now):.1f}pt > 1×ATR={_atr14_now:.1f}pt)")

        # ---- Phase 2: active setup — check for pullback then resumption --------
        if ss._pb_direction == direction:
            # Mark if we've touched the VWAP zone since the setup was registered
            if _in_vwap_zone:
                ss._pb_touched_vwap = True
                logger.debug(f"[EntryV4] {direction} setup touched VWAP zone "
                             f"(bar {ss._pb_bars_since_signal})")

            if not ss._pb_touched_vwap:
                # Still waiting for the pullback — do not enter yet
                return _block(f"[EntryV4] WAIT: {direction} pullback not yet reached VWAP zone "
                              f"(bar {ss._pb_bars_since_signal}/{ss._pb_max_bars})")

            # Pullback has occurred — require resumption momentum to enter
            _resuming = _resuming_up if direction == 'UP' else _resuming_down
            if not _resuming:
                # Reached VWAP but momentum not yet resuming — brief wait
                # Allow entry after 2 bars at VWAP regardless (avoids infinite wait)
                if ss._pb_bars_since_signal < ss._pb_max_bars - 2:
                    return _block(f"[EntryV4] WAIT: {direction} at VWAP, awaiting resumption "
                                  f"(vwap_vel={_vwap_vel:+.3f} tick={_tick_imb:+.3f})")
                else:
                    # Near expiry — relax to single VWAP-side check to avoid missing the entry
                    _on_correct_side = (_close_now > _vwap_now) if direction == 'UP' else (_close_now < _vwap_now)
                    if not _on_correct_side:
                        return _block(f"[EntryV4] BLOCKED: {direction} at VWAP zone expiry, "
                                      f"price on wrong side (close={_close_now:.1f} vwap={_vwap_now:.1f})")
                    logger.debug(f"[EntryV4] {direction} expiry-allow: bar {ss._pb_bars_since_signal}, "
                                 f"price on correct VWAP side")

            # ENTRY ALLOWED — clear the pending setup so next bar starts fresh
            logger.info(f"[EntryV4] ENTRY: {direction} pullback-resume confirmed "
                        f"(bar {ss._pb_bars_since_signal} vwap_vel={_vwap_vel:+.3f} "
                        f"tick={_tick_imb:+.3f} close={_close_now:.1f} vwap={_vwap_now:.1f})")
            ss._pb_direction = None
            ss._pb_touched_vwap = False
            ss._pb_bars_since_signal = 0

        else:
            # Direction of pending setup flipped (e.g. was UP, now DOWN) — discard
            logger.debug(f"[EntryV4] direction flipped ({ss._pb_direction}→{direction}), resetting setup")
            ss._pb_direction = None
            ss._pb_touched_vwap = False
            ss._pb_bars_since_signal = 0
            # Register new direction and wait (same as new-setup branch above)
            if _in_vwap_zone:
                ss._pb_direction = direction
                ss._pb_touched_vwap = True
            else:
                ss._pb_direction = direction
                return _block(f"[EntryV4] WAIT: direction flipped to {direction}, awaiting VWAP retest")

    else:
        # CRISIS bypass: maintain VWAP history but skip pullback logic entirely
        _vwap_pos = 1 if _close_now > _vwap_now else 0
        ss.vwap_history.append(_vwap_pos)
        logger.debug(f"[EntryV4] CRISIS bypass — skipping pullback gate")

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


# ===========================================================================
# V5 CLEAN SIGNAL GENERATOR
# ===========================================================================
# Implements the design exactly:
#   Gate 1: TRENDING regime only (ADX-based, from live_regime_from_row)
#   Gate 2: Session window 09:30–14:15
#   Gate 3: IV rank < 80th percentile
#   ML:     15m + 30m weighted vote  →  P(direction) ≥ V5_CONF_ENTRY (0.60)
#   Align:  ML direction must match regime direction
#   Entry:  Pullback-resume (EntryV5)
#
# Returns the same dict structure as generate_signal() so callers are interchangeable.
# ===========================================================================

def generate_signal_v5(
    row: pd.Series,
    models: dict,
    current_regime: int,
    micro_regime: str = 'UNKNOWN',
    signal_state: 'SignalState | None' = None,
    extra_conf_floor: float = 0.0,
    crisis_bypass: bool = False,
    regime_conf: float = 0.5,
    v5_risk_state: 'V5RiskState | None' = None,
    capital: float = 100_000.0,
) -> 'dict | None':
    """
    v5 clean trend-following signal generator.

    Strategy: TRENDING only → pullback to VWAP → momentum resumes → enter CE/PE.

    Three hard filters (in order):
      1. Regime gate:  TRENDING required (live_regime_from_row or current_regime arg)
      2. Session gate: 09:30–14:15 window
      3. IV rank gate: iv_rank_approx < V5_IV_RANK_MAX (80th percentile)

    ML direction:
      - 15m and 30m models vote (weighted 60/40)
      - P(direction) must exceed V5_CONF_ENTRY (0.60) to proceed
      - ML direction must agree with regime swing structure

    Entry gate (EntryV5 — two-phase pullback-resume):
      Phase 1: Trend confirmed → register direction, wait for VWAP retest
               OR enter immediately if already in VWAP zone
      Phase 2: Price touches VWAP zone (+prior impulse still intact)
               → wait for momentum resumption bar (vwap_dev_vel + tick_imbalance)
               → ENTER on bar close

    Session risk gates (V5RiskState — called before entry execution):
      • Max 3 trades per day (1 on gap days)
      • Daily loss limit Rs 1,500
      • 15-min cooldown after 2 consecutive losses
      • Rolling WR guard: pause if last 5 trades WR < 35%

    Args:
        row:            Current bar feature vector (pd.Series).
        models:         Dict of {horizon: model_dict} from trainer.
        current_regime: Integer regime code (REGIME_TRENDING / REGIME_RANGING).
        micro_regime:   String micro-regime label (unused for hard gates in v5).
        signal_state:   Per-session SignalState; uses module singleton if None.
        extra_conf_floor: Additional confidence floor.
        crisis_bypass:  If True, bypass regime/session/IV gates (V-recovery mode).
        regime_conf:    HMM certainty [0,1]; used to blend regime/global model.
        v5_risk_state:  V5RiskState instance; uses module singleton if None.
        capital:        Trading capital in Rs (for lot sizing).

    Returns:
        dict with signal info (includes 'lots' key) if entry is allowed, else None.
        Sets last_block_reason for callers that want to know why.
    """
    ss  = signal_state if signal_state is not None else _signal_state
    vrs = v5_risk_state if v5_risk_state is not None else _v5_risk_state

    global last_block_reason
    last_block_reason = ""

    mod = int(row.get('minute_of_day', 0))
    sp  = float(row.get('session_pct', 0.0))

    # -----------------------------------------------------------------------
    # GATE 1: Regime — TRENDING only
    # -----------------------------------------------------------------------
    if not crisis_bypass and current_regime != REGIME_TRENDING:
        return _block(
            f"[V5-Gate1] BLOCKED: regime={REGIME_NAMES.get(current_regime, '?')} "
            f"— only TRENDING allowed"
        )

    # -----------------------------------------------------------------------
    # GATE 2: Session window — 09:30–14:15 (mod 15–300)
    # Block first 15 min (open noise) and last 75 min (close theta decay).
    # -----------------------------------------------------------------------
    if not crisis_bypass:
        if mod < V5_ENTRY_MOD_MIN or mod > V5_ENTRY_MOD_MAX or sp > 0.92:
            return _block(
                f"[V5-Gate2] BLOCKED: outside session window "
                f"(mod={mod}, sp={sp:.2f})"
            )

    # -----------------------------------------------------------------------
    # GATE 3: IV rank filter — avoid overpriced options
    # iv_rank_approx = percentile of current IV vs 20-day rolling history.
    # Above 80th percentile = premium is expensive → theta drag dominates.
    # -----------------------------------------------------------------------
    if not crisis_bypass:
        iv_rank = float(row.get('iv_rank_approx', row.get('iv_rank', 50.0)))
        if iv_rank > V5_IV_RANK_MAX:
            return _block(
                f"[V5-Gate3] BLOCKED: iv_rank={iv_rank:.1f} > {V5_IV_RANK_MAX} "
                f"— options overpriced"
            )

    # -----------------------------------------------------------------------
    # GATE 4: Event-day filter
    # Scheduled macro events (RBI policy, Fed, index rebalancing) create
    # vol spikes that invalidate the trend-following model's assumptions.
    # Caller sets row['event_day'] = 1 for known event days.
    # -----------------------------------------------------------------------
    if not crisis_bypass and int(row.get('event_day', 0)) == 1:
        return _block("[V5-Gate4] BLOCKED: event_day=1 — scheduled macro event")

    # -----------------------------------------------------------------------
    # GATE 5: Session-level risk (V5RiskState)
    # Checked here (before ML inference) to avoid wasting compute.
    # -----------------------------------------------------------------------
    if not crisis_bypass:
        _risk_ok, _risk_reason = vrs.check_entry(now=datetime.now())
        if not _risk_ok:
            return _block(_risk_reason)

    # -----------------------------------------------------------------------
    # INPUT VALIDATION
    # -----------------------------------------------------------------------
    active_features = None
    if models:
        first_model = next(iter(models.values()))
        active_features = first_model.get('active_features')
    if active_features is None:
        from ..features.feature_engineering import FEATURE_COLS, FEATURE_LIVE_OK
        active_features = [f for f in FEATURE_COLS if FEATURE_LIVE_OK.get(f, True)]

    is_valid, error_msg, _ = validate_model_inputs(row, active_features)
    if not is_valid:
        return _block(f"[V5-InputCheck] BLOCKED: {error_msg}")

    X_raw = np.array(
        [[0.0 if pd.isna(row.get(c, 0)) else float(row.get(c, 0))
          for c in active_features]],
        dtype=np.float32,
    )

    # Apply walk-forward scaler (mandatory — raw features on scaled model = garbage)
    first_model_for_scaler = models.get(15) or models.get(30) or (
        next(iter(models.values())) if models else None
    )
    live_scaler = first_model_for_scaler.get('live_scaler') if first_model_for_scaler else None
    if live_scaler is None:
        raise RuntimeError(
            "[V5] CRITICAL: live_scaler is None. Models need scaled input. "
            "Retrain or restore model artifact with live_scaler."
        )
    if hasattr(live_scaler, 'n_features_in_') and live_scaler.n_features_in_ != X_raw.shape[1]:
        raise RuntimeError(
            f"[V5] CRITICAL: scaler expects {live_scaler.n_features_in_} features, "
            f"got {X_raw.shape[1]}. Retrain models."
        )
    X = live_scaler.transform(X_raw)

    # -----------------------------------------------------------------------
    # ML VOTING — 15m + 30m only (V5_HORIZONS)
    # -----------------------------------------------------------------------
    vote_up   = 0.0
    vote_dn   = 0.0
    total_w   = 0.0
    conf_wsum = 0.0
    conf_wden = 0.0
    signals   = {}
    n_valid   = 0

    for h in V5_HORIZONS:
        res = models.get(h)
        if res is None or res.get('final_model') is None:
            logger.warning(f"[V5-ML] h={h}m model missing — skipping")
            continue
        try:
            global_model = res['final_model']
            proba_g = global_model.predict_proba(X)[0]
            if len(proba_g) != 2 or np.any(np.isnan(proba_g)):
                logger.warning(f"[V5-ML] h={h}m invalid output {proba_g}")
                continue

            # Blend with regime-specific model when available
            regime_mdl = res.get('regime_models', {}).get(current_regime)
            if regime_mdl is not None:
                proba_r = regime_mdl.predict_proba(X)[0]
                if len(proba_r) == 2 and not np.any(np.isnan(proba_r)):
                    p = regime_conf * proba_r[1] + (1.0 - regime_conf) * proba_g[1]
                else:
                    p = proba_g[1]
            else:
                p = proba_g[1]

            p = float(np.clip(p, 0.45, 0.82))   # Platt-calibration range
            pred = 1 if p > 0.5 else 0
            conf = p if pred == 1 else (1.0 - p)

            signals[h] = {'pred': pred, 'conf': conf, 'proba': p}
            n_valid += 1

            w = V5_HORIZON_WEIGHTS.get(h, 0.5)
            if pred == 1:
                vote_up += w * conf
            else:
                vote_dn += w * conf
            total_w   += w * conf
            # conf_wsum / conf_wden: only include models that agree with dominant direction.
            # Bug: previously accumulated ALL models' confidences regardless of direction,
            # causing avg_conf to be polluted by opposing-direction predictions and
            # reporting artificially low confidence on high-agreement signals.
            # Deferred: direction is known only after all models run; we record per-model
            # data now and compute avg_conf after determining the dominant direction below.
            signals[h]['_w']    = w
            signals[h]['_pred'] = pred
            signals[h]['_conf'] = conf

            logger.info(
                f"[V5-ML] h={h}m pred={'UP' if pred==1 else 'DN'} "
                f"conf={conf:.3f} p_up={p:.3f}"
            )
        except Exception as exc:
            logger.error(f"[V5-ML] h={h}m prediction failed: {exc}")
            continue

    if n_valid == 0 or total_w == 0:
        return _block("[V5-ML] BLOCKED: no valid model outputs")

    # Direction from weighted vote
    direction = 'UP' if vote_up >= vote_dn else 'DOWN'
    agreement = (vote_up if direction == 'UP' else vote_dn) / total_w

    # Require clear majority (≥ 60% of weighted vote)
    if agreement < V5_CONF_ENTRY:
        return _block(
            f"[V5-ML] BLOCKED: ML confidence={agreement:.3f} < {V5_CONF_ENTRY} "
            f"— insufficient directional conviction"
        )

    # Weighted-average confidence: only models that agree with the winning direction.
    # Avoids inflation from opposing models (e.g. 15m=UP 0.65 + 30m=DN 0.60 → avg was 0.625
    # even though 30m disagreed — now avg_conf reflects only the UP-agreeing models).
    _win_pred = 1 if direction == 'UP' else 0
    conf_wsum = sum(
        signals[h]['_w'] * signals[h]['_conf']
        for h in signals
        if signals[h].get('_pred') == _win_pred
    )
    conf_wden = sum(
        signals[h]['_w']
        for h in signals
        if signals[h].get('_pred') == _win_pred
    )
    avg_conf = conf_wsum / conf_wden if conf_wden > 0 else 0.0

    # -----------------------------------------------------------------------
    # DIRECTION-REGIME ALIGNMENT
    # ML direction must match the regime's swing structure.
    # Regime says UPTREND → only CE (UP) signals allowed and vice versa.
    # -----------------------------------------------------------------------
    if not crisis_bypass:
        regime_direction = _regime_swing_direction(row)
        if regime_direction != 'NEUTRAL' and regime_direction != direction:
            return _block(
                f"[V5-Align] BLOCKED: ML={direction} but regime structure={regime_direction} "
                f"— direction conflict"
            )

    # -----------------------------------------------------------------------
    # ENTRY GATE V5: PULLBACK → RESUME
    #
    # Two phases per signal:
    #   Phase 1: Trend confirmed (all gates cleared on this bar)
    #            → if price is already in VWAP zone, go directly to Phase 2
    #            → otherwise register direction and wait for VWAP retest
    #
    #   Phase 2: VWAP zone reached AND impulse structure still intact
    #            → wait for momentum resumption bar
    #            → ENTER on that bar close
    #
    # Expiry logic: setup expires after V5_PB_MAX_WAIT_BARS bars.
    # Crisis bypass: skip pullback gate entirely.
    # -----------------------------------------------------------------------
    if not crisis_bypass:
        entry_allowed, entry_block_msg = _entry_v5(row, direction, ss)
        if not entry_allowed:
            return _block(entry_block_msg)

    # -----------------------------------------------------------------------
    # STRIKE AND PREMIUM CALCULATION (unchanged from v4)
    # -----------------------------------------------------------------------
    iv_proxy_val = float(row.get('iv_proxy', 0.0))
    iv = iv_proxy_val if iv_proxy_val > 0 else float(row.get('atr_14_pct', 0.06))
    iv_daily  = safe_value(iv) if iv > 0 else 0.06
    iv_annpct = iv_daily * np.sqrt(252)

    spot      = safe_value(row.get('close', 0))
    atr_now   = safe_value(row.get('atr_14', 0))
    dte_mins  = _next_expiry_mins()

    atm       = int(round(spot / STRIKE_ROUNDING) * STRIKE_ROUNDING)
    strike_ce = atm
    strike_pe = atm

    ce_ltp = safe_value(row.get('atm_ce_ltp', 0))
    pe_ltp = safe_value(row.get('atm_pe_ltp', 0))
    if ce_ltp <= 0:
        ce_ltp = estimate_option_premium(spot, iv_annpct, dte_mins,
                                          strike=float(strike_ce), option_type='CE')
    if pe_ltp <= 0:
        pe_ltp = estimate_option_premium(spot, iv_annpct, dte_mins,
                                          strike=float(strike_pe), option_type='PE')

    # -----------------------------------------------------------------------
    # POSITION SIZING
    # Compute lots now (after entry gate) so the signal dict carries the
    # confirmed size. Callers (live.py / paper.py) use signal['lots'] directly.
    # -----------------------------------------------------------------------
    _entry_premium = ce_ltp if direction == 'UP' else pe_ltp
    _vix_now       = float(row.get('vix_level', row.get('vix', 15.0)))
    lots = v5_lot_size(capital=capital, premium=_entry_premium, vix=_vix_now)

    # -----------------------------------------------------------------------
    # RECORD SIGNAL
    # -----------------------------------------------------------------------
    ss.record_signal(direction, avg_conf, current_regime, agreement=agreement)

    strength = ('STRONG'   if avg_conf >= V5_CONF_STRONG else
                'MODERATE' if avg_conf >= V5_CONF_ENTRY  else
                'WEAK')

    time_decay_mult = calculate_time_decay_confidence(mod)

    logger.info(
        f"[V5-SIGNAL] {direction} {strength} conf={avg_conf:.3f} "
        f"agree={agreement:.3f} lots={lots} mod={mod} "
        f"regime={REGIME_NAMES.get(current_regime,'?')}"
    )

    return {
        'direction':           direction,
        'strength':            strength,
        'avg_conf':            round(avg_conf, 4),
        'raw_conf':            round(avg_conf, 4),
        'stale_penalty':       0.0,
        'iv_crush_penalty':    0.0,
        'agreement':           round(agreement, 4),
        'regime':              REGIME_NAMES.get(current_regime, '?'),
        'micro_regime':        micro_regime,
        'signals':             signals,
        'spot':                float(row.get('close', 0)),
        'iv_proxy':            float(iv),
        'session':             float(sp),
        'w_up':                round(vote_up, 4),
        'w_dn':                round(vote_dn, 4),
        'n_valid':             n_valid,
        'ev_raw':              0.0,
        'ev_net':              0.0,
        'conf_pctile':         0.0,
        'season_bias':         0.0,
        'conf_floor':          V5_CONF_ENTRY,
        'meta_conf':           1.0,
        'in_transition_zone':  False,
        'minute_of_day':       mod,
        'is_expiry':           int(row.get('is_expiry', 0)),
        'time_decay_mult':     round(time_decay_mult, 3),
        'ofi_boost':           0.0,
        'atr_current':         float(row.get('atr_14', 0)),
        'fft_cycle':           float(row.get('fft_cycle', 0)),
        'frac_diff':           float(row.get('frac_diff_close', 0)),
        'strike_ce':           strike_ce,
        'strike_pe':           strike_pe,
        'ce_ltp_current':      round(ce_ltp, 2),
        'pe_ltp_current':      round(pe_ltp, 2),
        'dynamic_projections': {},
        # V5-specific fields
        'generator_version':   'v5',
        'stop_pct':            V5_STOP_PCT,
        'target_pct':          V5_TARGET_PCT,
        'lots':                lots,
        'risk_state':          vrs.summary(),
    }


# ---------------------------------------------------------------------------
# V5 helpers
# ---------------------------------------------------------------------------

def _regime_swing_direction(row: pd.Series) -> str:
    """
    Extract regime swing direction from pre-computed structure flags.
    Returns 'UP', 'DOWN', or 'NEUTRAL'.
    """
    hh = int(row.get('higher_high_flag', 0))
    hl = int(row.get('higher_low_flag',  0))
    lh = int(row.get('lower_high_flag',  0))
    ll = int(row.get('lower_low_flag',   0))

    if hh == 1 and hl == 1:
        return 'UP'
    if lh == 1 and ll == 1:
        return 'DOWN'
    return 'NEUTRAL'


def _entry_v5(row: pd.Series, direction: str,
               ss: 'SignalState') -> 'tuple[bool, str]':
    """
    EntryV5: two-phase pullback-resume gate.

    Returns (entry_allowed: bool, reason: str).
    True = proceed to entry; False = block with reason.

    State stored in SignalState._pb_* attributes (shared with EntryV4).
    """
    close_now = float(row.get('close', 0.0))
    vwap_now  = float(row.get('vwap', 0.0))
    atr14_now = float(row.get('atr_14', 1.0)) or 1.0
    vwap_vel  = float(row.get('vwap_dev_vel', 0.0))
    tick_imb  = float(row.get('tick_imbalance', 0.0))

    # "In VWAP zone" = price within V5_PB_VWAP_BAND_PCT of VWAP
    # Use ATR as a fallback if VWAP band would be too tight
    vwap_band = max(vwap_now * V5_PB_VWAP_BAND_PCT, atr14_now * 0.5) if vwap_now > 0 else atr14_now
    in_vwap_zone = (abs(close_now - vwap_now) <= vwap_band) if vwap_now > 0 else False

    # Momentum resumption signals (real-time, not lagging)
    resuming_up   = vwap_vel > V5_MOM_VWAP_VEL_MIN  and tick_imb > V5_MOM_TICK_IMB_MIN
    resuming_down = vwap_vel < -V5_MOM_VWAP_VEL_MIN and tick_imb < -V5_MOM_TICK_IMB_MIN
    resuming      = resuming_up if direction == 'UP' else resuming_down

    # Maintain VWAP position history
    ss.vwap_history.append(1 if close_now > vwap_now else 0)

    # --- Advance or expire any existing pending setup ---
    if ss._pb_direction is not None:
        ss._pb_bars_since_signal += 1
        if ss._pb_bars_since_signal > V5_PB_MAX_WAIT_BARS:
            logger.debug(
                f"[EntryV5] setup expired after {ss._pb_bars_since_signal} bars "
                f"(direction={ss._pb_direction})"
            )
            ss._pb_direction       = None
            ss._pb_touched_vwap    = False
            ss._pb_bars_since_signal = 0

    # --- Direction flip: discard old setup ---
    if ss._pb_direction is not None and ss._pb_direction != direction:
        logger.debug(
            f"[EntryV5] direction flipped {ss._pb_direction}→{direction}, resetting"
        )
        ss._pb_direction       = None
        ss._pb_touched_vwap    = False
        ss._pb_bars_since_signal = 0

    # --- No pending setup: register this bar's trend identification ---
    if ss._pb_direction is None:
        ss._pb_direction       = direction
        ss._pb_bars_since_signal = 0

        if in_vwap_zone:
            # Already at VWAP — immediate pullback-complete
            ss._pb_touched_vwap = True
            logger.debug(
                f"[EntryV5] {direction}: price already in VWAP zone "
                f"(close={close_now:.1f} vwap={vwap_now:.1f}) → immediate"
            )
            # Fall through to Phase 2 resumption check below
        else:
            ss._pb_touched_vwap = False
            logger.debug(
                f"[EntryV5] {direction}: registered, waiting for VWAP retest "
                f"(close={close_now:.1f} vwap={vwap_now:.1f} "
                f"dist={abs(close_now - vwap_now):.1f}pt)"
            )
            return False, (
                f"[EntryV5] WAIT: {direction} trend identified, "
                f"awaiting VWAP pullback (dist={abs(close_now-vwap_now):.1f}pt)"
            )

    # --- Phase 2: pending setup active — track pullback and resumption ---
    if ss._pb_direction == direction:

        # Mark when price reaches VWAP zone
        if in_vwap_zone:
            ss._pb_touched_vwap = True
            logger.debug(
                f"[EntryV5] {direction}: touched VWAP zone "
                f"(bar {ss._pb_bars_since_signal})"
            )

        # Still waiting for the pullback
        if not ss._pb_touched_vwap:
            return False, (
                f"[EntryV5] WAIT: {direction} pullback not yet at VWAP "
                f"(bar {ss._pb_bars_since_signal}/{V5_PB_MAX_WAIT_BARS})"
            )

        # Pullback reached — require momentum resumption
        if not resuming:
            # Allow up to 2 extra bars at VWAP for resumption to develop
            bars_at_vwap = ss._pb_bars_since_signal
            if bars_at_vwap < V5_PB_MAX_WAIT_BARS - 2:
                return False, (
                    f"[EntryV5] WAIT: {direction} at VWAP, awaiting resumption "
                    f"(vel={vwap_vel:+.3f} tick={tick_imb:+.3f})"
                )
            # Near expiry of setup — use simple side check
            on_correct_side = (
                (close_now > vwap_now) if direction == 'UP' else (close_now < vwap_now)
            )
            if not on_correct_side:
                ss._pb_direction       = None
                ss._pb_touched_vwap    = False
                ss._pb_bars_since_signal = 0
                return False, (
                    f"[EntryV5] EXPIRED: {direction} price crossed VWAP "
                    f"to wrong side (close={close_now:.1f} vwap={vwap_now:.1f})"
                )
            logger.debug(
                f"[EntryV5] {direction}: setup near-expiry allow "
                f"(bar {ss._pb_bars_since_signal}, price on correct side)"
            )

        # ENTRY ALLOWED — clear setup state
        logger.info(
            f"[EntryV5] ENTRY: {direction} pullback-resume confirmed "
            f"(bar {ss._pb_bars_since_signal} "
            f"vel={vwap_vel:+.3f} tick={tick_imb:+.3f} "
            f"close={close_now:.1f} vwap={vwap_now:.1f})"
        )
        ss._pb_direction       = None
        ss._pb_touched_vwap    = False
        ss._pb_bars_since_signal = 0

    return True, ""
