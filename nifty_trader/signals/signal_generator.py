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
from ..execution.orders import _next_expiry_mins, estimate_option_premium
from ..models.ensemble import MetaLabeler

logger = logging.getLogger(__name__)


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
        self._last_1m_dirs = []
        self._stale_conf_penalty = 0.0
        self.last_exit_time = None  # Edge Case 2: Reset cooldown
        self.vwap_history.clear()  # Clear VWAP history (deque supports clear())
        # Clear confidence history to prevent memory leak
        for regime in range(3):
            self._conf_history[regime] = []
    
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
        # 15-minute cooldown to let market micro-structure reset
        elapsed_seconds = (datetime.now() - self.last_exit_time).total_seconds()
        return elapsed_seconds < 900

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
        Track 1m direction flips (Req 5).
        If 1m flips twice in last 2 bars, lock for TEMPORAL_FLIP_LOCK_BARS bars.
        """
        self._last_1m_dirs.append(pred_1m)
        if len(self._last_1m_dirs) > 3:
            self._last_1m_dirs.pop(0)

        # Count direction flips BEFORE decrementing lock counter
        dirs = self._last_1m_dirs
        if len(dirs) >= 3:
            flips = sum(dirs[i] != dirs[i-1] for i in range(1, len(dirs)))
            if flips >= 2:
                self._lock_bars_remaining = TEMPORAL_FLIP_LOCK_BARS
        
        # Decrement lock counter AFTER checking for new flips
        if self._lock_bars_remaining > 0:
            self._lock_bars_remaining -= 1

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
                    crisis_bypass: bool = False) -> dict | None:
    """
    v3.2 EV-First Signal Generation.

    Signal gates (ALL must pass):
      1.  Regime gate:          not CRISIS (unless crisis_bypass=True from kill-switch)
      2.  Session gate:         9:53-15:12
      3.  Lunch filter:         12:15-13:00 blocked
      4.  Expiry filter:        no new trades after 14:30 on Thursday
      5.  IV gate:              iv_proxy in acceptable range
      5b. Black Swan gap gate:  block for 45 min if gap > 2.5x day ATR
      6.  Temporal gate:        no entry if 1m flipped direction twice in 2 bars
      7.  Micro-regime conf floor: TRENDING>=0.58, RANGING>=0.72
      7b. Micro-regime direction: TRENDING trades must align with trend direction
      7c. IV Crush gate:        deduct 0.10 conf when iv_rank>85 and IV contracting
      8.  Percentile no-trade zone: skip if 40th-60th percentile confidence
      8b. MetaLabeler gate:     meta-labeler must predict primary model is correct
      9.  EV_net gate:          net EV after execution penalty must be positive
      10. Signal scarcity:      max trades per direction per day
      11. Staleness penalty:    decay confidence if micro-regime contradicts trade
    """
    ss = signal_state if signal_state is not None else _signal_state

    # Gate 0: Edge Case 2 - Post-Trade Cooldown (Anti-Churn)
    # Prevent re-entering immediately after exit to avoid "Double-Dipping"
    # CRISIS bypass: skip cooldown — strong trending CRISIS moves can sustain multiple entries.
    if ss.in_cooldown() and not crisis_bypass:
        logger.info("[Gate] BLOCKED: post-trade cooldown (15 min anti-churn)")
        return None

    # Gate 1: Crisis — blocked unless kill-switch bypass conditions were already met
    if current_regime == REGIME_CRISIS and not crisis_bypass:
        return None

    sp  = row.get('session_pct', 0)
    mod = int(row.get('minute_of_day', 0))

    # Gate 2: Session window + late-day cutoff
    # sp > 0.92 blocks signals after ~15:02 (375 × 0.92 = 345 min from 9:15 = 14:50+).
    # Also block signals after minute_of_day=335 (14:50) explicitly:
    # the 1-min bar lag means a 14:50 signal executes at 14:51, leaving 24 min.
    # Any later and theta decay destroys the premium before the trade can develop.
    if sp < 0.10 or sp > 0.92 or mod >= 335:
        return None

    # Gate 3: Lunch chop penalty (12:15-13:00)
    # WHY: NIFTY often goes flat during European pre-open, BUT some best breakouts
    #      happen at 1:00 PM (Bar 225). Use confidence penalty instead of hard veto.
    lunch_penalty = 0.05 if 180 <= mod <= 225 else 0.0

    # Gate 4: Expiry cutoff
    if row.get('is_expiry', 0) == 1 and mod > 315:
        return None

    # iv_proxy is atr_14_pct * sqrt(bars_per_day) — annualised volatility estimate (~0.5-2%)
    # Use iv_proxy for option pricing; fall back to atr_14_pct only if iv_proxy is missing.
    iv_proxy_val = row.get('iv_proxy', 0)
    iv = iv_proxy_val if iv_proxy_val > 0 else row.get('atr_14_pct', 0)
    # Gate 5: IV floor
    if iv < 0.05:
        return None

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
    gap_pct = abs(float(row.get('gap_pct', 0)))
    day_atr = float(row.get('day_atr_pct', 0.5))
    if gap_pct > 2.5 * day_atr and mod < 45:
        return None   # black swan shock gate: wait for market to stabilise

    # Gate 6: Temporal consistency gate (Req 5)
    # Lock new entries if 1m prediction flipped direction twice in last 2 bars.
    # CRISIS bypass: skip — 1m flips are expected noise in a dominant CRISIS trend.
    if ss.temporal_locked and not crisis_bypass:
        return None

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
        return None

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

    for h, res in sorted(models.items()):
        try:
            # Validate model exists and is fitted
            if res.get('final_model') is None:
                logger.warning(f"[{h}m] Missing final_model, skipping")
                continue
            
            # Mixture of Experts: prefer regime-specific model if available
            regime_mdl = res.get('regime_models', {}).get(current_regime)
            primary_mdl = regime_mdl if regime_mdl is not None else res['final_model']
            
            # Validate model has predict_proba method
            if not hasattr(primary_mdl, 'predict_proba'):
                logger.error(f"[{h}m] Model missing predict_proba method")
                continue
            
            proba = primary_mdl.predict_proba(X)[0]

            # Validate probability output
            if len(proba) != 2 or np.any(np.isnan(proba)) or np.any(np.isinf(proba)):
                logger.error(f"[{h}m] Invalid model output: {proba}")
                continue

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
            if meta_proba_stored is not None:
                signals[h]['meta_proba'] = meta_proba_stored

            # Update temporal gate with 1m prediction
            if h == 1:
                ss.update_temporal_gate(pred)

            # Step 9 (v3.3): EV-weighted aggregation — horizon only votes if
            # its confidence clears its own per-horizon floor (CONF_BY_HORIZON).
            # Vote = conf × ev_score × perf_weight, where ev_score = conf
            # (higher probability = higher expected EV, natural proxy).
            # CRISIS bypass: relax per-horizon floors to CONF_MIN so all horizons
            # get a vote. The KillSwitch already validated direction; strict floors
            # here can exclude the very DOWN votes that triggered the bypass.
            h_conf_floor = CONF_MIN if crisis_bypass else CONF_BY_HORIZON.get(h, CONF_MIN)
            dir_label = 'UP' if pred == 1 else 'DN'
            if conf >= h_conf_floor:
                w        = ss.perf_weights.get(h, HORIZON_WEIGHTS.get(h, 0.25))
                ev_score = conf   # ev proxy: P(correct) ≈ confidence
                vote_val = w * conf * ev_score
                if pred == 1:
                    weighted_up += vote_val
                    avg_win_up  += conf
                else:
                    weighted_dn += vote_val
                    avg_win_dn  += conf
                total_weight += vote_val
                n_valid      += 1
                logger.debug(f"[Vote] h={h}m pred={dir_label} conf={conf:.3f} floor={h_conf_floor:.3f} vote={vote_val:.4f} COUNTED")
            else:
                logger.debug(f"[Vote] h={h}m pred={dir_label} conf={conf:.3f} floor={h_conf_floor:.3f} BELOW_FLOOR")
        except Exception as e:
            logger.error(f"[{h}m] Prediction failed: {str(e)[:100]}")
            continue  # Skip this horizon, try others

    if n_valid == 0 or total_weight == 0:
        logger.info("[Gate] BLOCKED: no valid horizon votes")
        return None

    # Directional decision
    if weighted_up > weighted_dn:
        direction = 'UP';   agreement = weighted_up / (weighted_up + weighted_dn)
    else:
        direction = 'DOWN'; agreement = weighted_dn / (weighted_up + weighted_dn)

    logger.debug(f"[Vote] w_up={weighted_up:.4f} w_dn={weighted_dn:.4f} dir={direction} agree={agreement:.3f} n={n_valid}")

    if agreement < 0.60:
        logger.info(f"[Gate] BLOCKED: agreement={agreement:.2f} < 0.60")
        return None

    # 3️⃣ Directional Agreement Gate (CRITICAL FOR LIVE SURVIVABILITY)
    # WHY: Multi-horizon conflicts cause whipsaws. 5m and 15m must agree.
    # CRISIS bypass: skip — KillSwitch already validated multi-horizon agreement.
    if not crisis_bypass:
        dir_pass, dir_reason = check_directional_agreement(signals)
        if not dir_pass:
            logger.info(f"[Gate] BLOCKED: directional_agreement={dir_reason}")
            return None

    avg_conf = (weighted_up + weighted_dn) / total_weight
    if avg_conf < CONF_MIN:
        logger.info(f"[Gate] BLOCKED: avg_conf={avg_conf:.3f} < CONF_MIN={CONF_MIN}")
        return None
    
    # Edge Case 5: Tuesday FinNifty Arbitrage Filter
    # On Tuesdays (FinNifty expiry), NIFTY exhibits "Shadow Volatility"
    # driven by arbitrage rebalancing rather than fundamental price discovery.
    # Require 10% higher confidence to clear this noise.
    dow = int(row.get('dow', 0))  # 0=Mon, 1=Tue, ..., 4=Fri
    # Validate dow is in valid range
    if dow < 0 or dow > 4:
        dow = 0  # Default to Monday if invalid
    
    tuesday_penalty = 0.10 if dow == 1 else 0.0  # Tuesday = +10% conf boost required

    # Primary horizon (5m) must agree with weighted direction
    # CRISIS bypass: skip — 5m may show short-term noise; KillSwitch validates overall direction.
    if not crisis_bypass and 5 in signals and signals[5]['conf'] >= CONF_MIN:
        if signals[5]['pred'] != (1 if direction == 'UP' else 0):
            logger.info(f"[Gate] BLOCKED: 5m pred={'UP' if signals[5]['pred']==1 else 'DN'} disagrees with direction={direction}")
            return None

    # Gate 7: Micro-regime adaptive confidence floor (Req 3)
    is_trending = micro_regime in ('TRENDING_UP', 'TRENDING_DN', 'BREAKOUT')
    is_ranging  = micro_regime == 'RANGING'
    conf_floor  = (CONF_FLOOR_TRENDING if is_trending else
                   CONF_FLOOR_RANGING  if is_ranging  else CONF_MIN)
    conf_floor += extra_conf_floor  # regime-frequency boost (Issue 5)
    conf_floor += tuesday_penalty    # Tuesday requires higher confidence
    conf_floor += lunch_penalty      # Lunch hour requires higher confidence
    if avg_conf < conf_floor:
        logger.info(f"[Gate] BLOCKED: avg_conf={avg_conf:.3f} < conf_floor={conf_floor:.3f} (micro={micro_regime})")
        return None

    # Gate 7b: Micro-regime must be directionally consistent in TRENDING
    # CRISIS bypass: skip — 1-min micro can show short bounces within a strong trend;
    # KillSwitch already validated ML agreement >= 85% across all horizons.
    if is_trending and not crisis_bypass:
        if direction == 'UP'   and micro_regime == 'TRENDING_DN':
            logger.info(f"[Gate] BLOCKED: UP signal but micro=TRENDING_DN")
            return None
        if direction == 'DOWN' and micro_regime == 'TRENDING_UP':
            logger.info(f"[Gate] BLOCKED: DOWN signal but micro=TRENDING_UP")
            return None

    # Gate 7c-VOL: Volatility expansion gate
    # WHY: Trend following in options only works when volatility is expanding.
    # In a squeeze (BB contracted, ATR < avg), theta decay destroys P&L even
    # when direction is correct. Require either:
    #   a) ATR ratio > 0.85 (at least normal volatility, not contracting), OR
    #   b) in BREAKOUT micro-regime (squeeze already resolved)
    # Exception: RANGING regime — lower bar (0.70) since mean-reversion works
    # even in low-vol environments.
    atr_ratio   = float(row.get('atr_ratio', 1.0))
    bb_squeeze  = int(row.get('bb_squeeze', 0))
    vol_floor   = 0.70 if is_ranging else 0.85
    # Squeeze: apply confidence penalty instead of hard veto.
    # A squeeze resolves with a breakout — if model confidence is very high,
    # the breakout may already be starting. Hard-veto misses these entries.
    squeeze_penalty = 0.05 if (bb_squeeze == 1 and atr_ratio < vol_floor and micro_regime != 'BREAKOUT') else 0.0

    # Staleness penalty (Req 14): decay confidence when micro-regime flips
    # CRISIS bypass: skip — TRENDING_UP micro in a CRISIS DOWN move is normal (dead-cat bounce).
    # KillSwitch already validated 85%+ agreement; penalising for micro-bounce would suppress
    # the strongest signals.
    stale_penalty = 0.0 if crisis_bypass else ss.update_staleness(micro_regime, direction)
    adj_conf = avg_conf - stale_penalty - squeeze_penalty
    if adj_conf < conf_floor:
        return None

    # Gate 7c — IV Crush Protector (Enhancement 4):
    # High IV rank + negative IV momentum means extrinsic value is collapsing.
    # Subtract penalty from adj_conf and reject if it falls below the floor.
    iv_crush_penalty = check_iv_crush(row)
    if iv_crush_penalty > 0:
        adj_conf -= iv_crush_penalty
        if adj_conf < conf_floor:
            return None   # IV crush eroded confidence below threshold

    # Gate 8: Percentile no-trade zone (Req 6)
    # 6️⃣ ENHANCED: Require percentile > 65 for trade entry (not just avoiding 40-60).
    # WHY: Only trade high-conviction signals relative to recent history.
    conf_pctile = ss.conf_percentile(adj_conf, current_regime)
    # Skip percentile check if insufficient history (bootstrap period)
    if conf_pctile is not None:
        if conf_pctile < 55:
            logger.info(f"[Gate] BLOCKED: conf_pctile={conf_pctile:.1f} < 55")
            return None  # Signal below 55th percentile - insufficient historical conviction
        if NO_TRADE_PCTILE_LOW <= conf_pctile <= NO_TRADE_PCTILE_HIGH:
            logger.info(f"[Gate] BLOCKED: conf_pctile={conf_pctile:.1f} in no-trade zone {NO_TRADE_PCTILE_LOW}-{NO_TRADE_PCTILE_HIGH}")
            return None  # ambiguous confidence zone (40-60) - absolute no-trade

    # Gate 8b: MetaLabeler filter (v3.2 -- Lopez de Prado meta-labeling)
    # The primary model predicts direction; the meta-labeler predicts whether the
    # primary model is likely to be CORRECT on this specific bar's context.
    # Only trade when meta-confidence >= META_CONF_THRESH (default 0.55).
    # Uses the 5m model's meta-labeler as the canonical filter.
    # CRISIS bypass: skip — meta-labeler trained on TRENDING/RANGING may underrate
    # valid crisis trend signals; KillSwitch agreement already validates quality.
    meta_pass = True
    meta_conf = 1.0
    if not crisis_bypass:
        ref_res   = models.get(5) or (next(iter(models.values())) if models else None)
        if ref_res is not None:
            ml = ref_res.get('meta_labeler')
            if ml is not None and ml._fitted:
                primary_proba = float(signals.get(5, {}).get('proba', adj_conf))
                meta_conf = ml.predict_proba(primary_proba, row, current_regime)
                if meta_conf < MetaLabeler.META_CONF_THRESH:
                    meta_pass = False
        if not meta_pass:
            logger.info(f"[Gate] BLOCKED: meta_labeler conf={meta_conf:.3f} < {MetaLabeler.META_CONF_THRESH}")
            return None   # meta-labeler says primary model likely wrong here

    # EV calculation for EV-first ranking (Req 12)
    # EV = P(win) x AvgWin - P(loss) x AvgLoss - Costs
    # Asymmetric payoff: options have fat right-tail wins (configurable in config.py)
    p_win    = adj_conf
    p_loss   = 1.0 - adj_conf
    avg_win  = EV_AVG_WIN_MULT   # from config: typical 1.2x (winners > 1:1 return)
    avg_loss = EV_AVG_LOSS_MULT  # from config: typical 1.0x (losers capped at premium)
    ev_raw   = p_win * avg_win - p_loss * avg_loss

    # Gate 9: EV_net penalty -- execution costs + safety factor + spread penalty (Req 7)
    # Edge Case 5: Low Liquidity/High Spread mitigation
    iv_rank_current = float(row.get('iv_rank_approx', 50.0))
    ev_net = _ev_net(ev_raw, TOTAL_COST_PCT, mod, iv_rank=iv_rank_current)
    if ev_net <= 0:
        logger.info(f"[Gate] BLOCKED: ev_net={ev_net:.4f} <= 0 (ev_raw={ev_raw:.4f})")
        return None   # trade is not EV-positive after realistic costs

    # Seasonality additive bias (Req 11): small nudge, never a veto
    season_bias = ss.seasonality_bias(mod)
    ev_net_adj  = ev_net + season_bias

    # Gate 10: Signal scarcity (Req 13)
    today = row.get('date', None)
    if ss.check_scarcity(direction, current_regime, today):
        return None   # daily trade limit reached for this direction

    # Record to confidence history for future percentile ranking
    ss.record_signal(direction, adj_conf, current_regime, agreement=agreement)

    # v4.0: Time-based confidence decay (Scarcity Logic)
    # Signals in late-day hours require higher confidence floor
    time_decay_mult = calculate_time_decay_confidence(mod)
    final_conf_floor = conf_floor * time_decay_mult
    if adj_conf < final_conf_floor:
        return None  # Failed late-day confidence requirement

    # 7️⃣ Entry Micro-Confirmation (FINAL EXECUTION FILTER)
    # WHY: Models can signal too early. Require price to demonstrate commitment.
    # Use SignalState's vwap_history (reset daily, no memory leak)
    # CRISIS bypass: still run for VWAP history update, but don't block on result.
    # 5+ consecutive red bars below VWAP already confirms direction commitment.
    micro_pass, micro_reason = check_entry_micro_confirmation(
        row, direction, ss.vwap_history
    )
    if not micro_pass and not crisis_bypass:
        logger.info(f"[Gate] BLOCKED: micro_confirmation={micro_reason}")
        return None  # No micro-confirmation (price not committed to direction)

    strength = ('STRONG'   if adj_conf >= CONF_STRONG else
                'MODERATE' if adj_conf >= CONF_MODERATE else
                'WEAK')
    
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
        'raw_conf':     avg_conf,
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


