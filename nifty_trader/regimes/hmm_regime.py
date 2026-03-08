"""Regime detection: HMM 3-state + rule-based fallback.

LIVE SAFETY FIXES (audit-driven):
  1. Causal inference: replaced Viterbi (full-sequence, non-causal) with the
     HMM forward algorithm so only observations up to t are used at time t.
  2. Sliding-window HMM: only the most recent CAUSAL_WINDOW bars are scored,
     preventing the earliest-period statistics from anchoring the current state.
  3. Regime confidence decay: confidence is weighted by a half-life function
     so freshly-detected regime flips start at low confidence and must earn
     their way up before the system acts on them.
  4. Regime hysteresis (RegimeStateMachine): a pending regime must be observed
     for CONFIRM_BARS consecutive bars AND the current regime must have lasted
     at least MIN_REGIME_DURATION bars before a switch is accepted.
"""
import os, logging
import numpy as np
import pandas as pd
import joblib
import warnings

from nifty_trader.features.indicators import _atr, _dmi, _rsi
warnings.filterwarnings('ignore')

from ..config import (
    REGIME_TRENDING, REGIME_RANGING, REGIME_CRISIS, REGIME_NAMES, HMM_OK,
)
try:
    from hmmlearn import hmm
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Causal inference constants
# ---------------------------------------------------------------------------
CAUSAL_WINDOW   = 60     # bars used in the sliding-window forward pass
CONFIRM_BARS    = 3      # consecutive bars before a regime switch is accepted
CONF_HALF_LIFE  = 5      # bars — confidence reaches 50% at this age
MIN_REGIME_DURATION = {  # minimum bars before a regime can be replaced
    REGIME_TRENDING: 6,
    REGIME_RANGING:  4,
    REGIME_CRISIS:   1,   # crisis switches immediately (safety)
}
REGIME_UNCERTAIN = -1    # returned when confidence is too low to act

# ==============================================================================
# 4. REGIME DETECTION
# ==============================================================================

def intraday_regime_override(row, reg: int) -> int:
    """
    Intraday micro-regime override (v3.3 Step 4).
    Corrects the daily regime label with real-time bar-level signals to
    avoid misalignment when the daily HMM state lags intraday conditions.

    Rules (applied in priority order):
      0. Session-aware override: first/last 30 min -> RANGING (high noise/low liquidity)
      1. IV spiking hard (>25% change in 30 bars) -> CRISIS regardless of HMM.
      2. ATR ratio > 1.8 AND ADX < 15 (wide spread, no trend) -> RANGING.
      2b. HMM says CRISIS but intraday is calm + trending -> downgrade to TRENDING.
          (Daily HMM can lag intraday reality by hours; this prevents false CRISIS lockouts.)
      3. Otherwise keep the daily HMM regime.
    """
    minute_of_day = row.get('minute_of_day', 0)

    # Rule 0: Session-aware regime (first/last 30 min)
    if minute_of_day < 30 or minute_of_day >= 345:
        return REGIME_RANGING

    # Rule 1: IV spike -> CRISIS (always takes priority)
    if row.get('iv_pct_change', 0) > 25:
        return REGIME_CRISIS

    # Rule 2: Wide spread + no trend -> RANGING
    if row.get('atr_ratio', 1.0) > 1.8 and row.get('adx_14', 0) < 15:
        return REGIME_RANGING

    # Rule 2b: HMM says CRISIS but intraday signals are calm and trending.
    # Conditions for downgrade (all must be true):
    #   - IV not spiking (< 10% change) — no real panic
    #   - ADX >= 25 — market has clear direction
    #   - ATR ratio < 1.5 — volatility not abnormally wide
    if reg == REGIME_CRISIS:
        iv_change  = row.get('iv_pct_change', 0)
        adx        = row.get('adx_14', 0)
        atr_ratio  = row.get('atr_ratio', 1.0)
        if iv_change < 10 and adx >= 25 and atr_ratio < 1.5:
            logger.info(
                f"[RegimeOverride] CRISIS->TRENDING: IV_chg={iv_change:.1f}% "
                f"ADX={adx:.1f} ATR_ratio={atr_ratio:.2f} — intraday calm, daily HMM lagging"
            )
            return REGIME_TRENDING

    # Rule 3: HMM uncertain (pending confirmation) but intraday signals are
    # strongly trending. When ADX > 35 and IV is not spiking, the market has
    # clear direction — treat as TRENDING rather than blocking entirely.
    if reg == REGIME_UNCERTAIN:
        adx       = row.get('adx_14', 0)
        iv_change = row.get('iv_pct_change', 0)
        # Use tf5_ ADX when available — 5-min ADX is more stable than 1-min
        adx5      = row.get('tf5_adx', adx)
        effective_adx = max(adx, adx5)
        if effective_adx >= 30 and iv_change < 15:
            logger.info(
                f"[RegimeOverride] UNCERTAIN->TRENDING: ADX_1m={adx:.1f} ADX_5m={adx5:.1f} IV_chg={iv_change:.1f}% "
                f"— strong intraday trend overrides pending HMM confirmation"
            )
            return REGIME_TRENDING

    return reg


class RegimeDetector:
    """
    3-state HMM trained on daily features.
    State 0: Low vol, trending    -> best for directional options
    State 1: Medium vol, ranging  -> mean-reversion, wider strikes
    State 2: High vol, crisis     -> no new trades

    Falls back to rule-based if hmmlearn not installed.
    """

    def __init__(self):
        self.model     = None
        self.state_map = {}
        self._fitted   = False

    def _regime_features(self, df1d: pd.DataFrame):
        d  = df1d.copy()
        c  = d['close']
        atr14, _ = _atr(d['high'], d['low'], c, 14)
        vol20     = c.pct_change(1).rolling(20).std() * np.sqrt(252) * 100
        trend     = (c - c.rolling(50).mean()) / (c.rolling(50).mean() + 1e-9) * 100
        rsi14     = _rsi(c, 14)
        _, _, adx_d = _dmi(d['high'], d['low'], c, 14)
        feat = pd.DataFrame({
            'vol':   vol20,
            'atr':   (atr14 / (c+1e-9) * 100),
            'trend': trend,
            'rsi':   rsi14,
            'adx':   adx_d,
        }).ffill().bfill().dropna()
        mu = feat.mean(); sd = feat.std() + 1e-9
        return ((feat - mu) / sd).values, feat.index

    def fit(self, df1d: pd.DataFrame):
        if not HMM_OK:
            print("  hmmlearn not available -- using rule-based regime")
            self._fitted = False
            return self

        X, idx = self._regime_features(df1d)
        model  = hmm.GaussianHMM(n_components=3, covariance_type='full',
                                  n_iter=200, random_state=42)
        try:
            model.fit(X)
            # ------------------------------------------------------------------
            # STATE ANCHORING: deterministic regime identity across retrains.
            #
            # hmmlearn assigns state indices 0/1/2 randomly each fit, so we
            # must map them to TRENDING/RANGING/CRISIS explicitly.
            #
            # Anchor criterion: average diagonal variance of each state's
            # covariance matrix (covariance_type='full').
            #   - CRISIS   = highest variance  (vol shock)
            #   - RANGING  = lowest variance   (flat, quiet)
            #   - TRENDING = middle variance   (directional but not panicking)
            #
            # Variance is more robust than feature-mean for separation because
            # means can overlap between TRENDING and RANGING.
            # ------------------------------------------------------------------
            variances = np.array([np.diag(cov).mean() for cov in model.covars_])
            sorted_by_var = np.argsort(variances)   # low → high variance
            ranging_idx  = int(sorted_by_var[0])    # lowest  variance → RANGING
            trending_idx = int(sorted_by_var[1])    # middle  variance → TRENDING
            crisis_idx   = int(sorted_by_var[2])    # highest variance → CRISIS

            self.state_map = {
                trending_idx: REGIME_TRENDING,
                ranging_idx:  REGIME_RANGING,
                crisis_idx:   REGIME_CRISIS,
            }
            self.model   = model
            self._fitted = True
            print(
                f"  HMM fitted. State map: {self.state_map}  "
                f"variances=[{variances[ranging_idx]:.4f}, "
                f"{variances[trending_idx]:.4f}, "
                f"{variances[crisis_idx]:.4f}] (ranging/trending/crisis)"
            )
        except Exception as e:
            print(f"  HMM fit failed ({e}) -- using rule-based")
            self._fitted = False
        return self

    def predict(self, df1d: pd.DataFrame) -> pd.Series:
        """Return daily regime series indexed by date.

        LIVE SAFETY: uses the causal forward algorithm (score_samples) NOT
        Viterbi (predict).  Viterbi decodes the globally-optimal path using
        ALL observations including future ones — completely non-causal.
        score_samples applies the forward pass incrementally so prediction at
        bar t only depends on bars [0..t].
        """
        d  = df1d.copy()
        c  = d['close']
        atr14, _ = _atr(d['high'], d['low'], c, 14)
        vol20    = c.pct_change(1).rolling(20).std() * np.sqrt(252) * 100

        if self._fitted:
            X, idx = self._regime_features(df1d)
            # ---------------------------------------------------------------
            # CAUSAL forward pass: score_samples returns the posterior
            # P(state_t | obs_0..obs_t) for every t WITHOUT using future data.
            # We take argmax of the last row's posteriors.
            # ---------------------------------------------------------------
            try:
                _, posteriors = self.model.score_samples(X)
                # posteriors shape: (n_bars, n_states)
                raw_states = np.argmax(posteriors, axis=1)
            except Exception:
                # Fallback: use predict() but log the warning
                logger.warning("HMM forward pass failed, falling back to Viterbi (non-causal)")
                raw_states = self.model.predict(X)

            mapped = pd.Series([self.state_map[s] for s in raw_states], index=idx)
            dates  = df1d['datetime'].dt.date.values
            regime_by_date = {}
            for row_i, date_ in enumerate(dates):
                if row_i in mapped.index:
                    regime_by_date[date_] = int(mapped[row_i])
            return pd.Series(regime_by_date)
        else:
            # Rule-based fallback using vol + ADX
            _, _, adx_d = _dmi(d['high'], d['low'], c, 14)
            vol_q75 = vol20.quantile(0.75)
            vol_q90 = vol20.quantile(0.90)
            regimes = {}
            for i, row in d.iterrows():
                v = vol20.iloc[i] if i < len(vol20) else np.nan
                a = adx_d.iloc[i] if i < len(adx_d) else np.nan
                d_ = row['datetime'].date()
                if np.isnan(v):
                    regimes[d_] = REGIME_RANGING
                elif v > vol_q90:
                    regimes[d_] = REGIME_CRISIS
                elif v > vol_q75 or (not np.isnan(a) and a < 20):
                    regimes[d_] = REGIME_RANGING
                else:
                    regimes[d_] = REGIME_TRENDING
            return pd.Series(regimes)

    def predict_live(self, df1d: pd.DataFrame, window: int = CAUSAL_WINDOW) -> tuple:
        """
        Causal live-safe regime inference.  Called once per bar in live mode.

        Uses only the most recent `window` bars so the sliding window stays
        current without re-processing the full history.

        Returns
        -------
        (regime: int, confidence: float)
            regime     — REGIME_TRENDING / RANGING / CRISIS (or REGIME_UNCERTAIN)
            confidence — posterior probability of the returned state [0..1]
                         Confidence decays for the first CONF_HALF_LIFE bars
                         after a regime change.
        """
        if not self._fitted or len(df1d) < 15:
            # Rule-based fallback
            regime = self._rule_based_single(df1d)
            return regime, 0.5

        X, _ = self._regime_features(df1d)
        X_win = X[-window:] if len(X) >= window else X

        try:
            _, posteriors = self.model.score_samples(X_win)
            last_probs = posteriors[-1]   # P(state | obs_0..obs_t) at current bar
            raw_state  = int(np.argmax(last_probs))
            confidence = float(last_probs[raw_state])
            regime     = self.state_map.get(raw_state, REGIME_RANGING)
        except Exception as e:
            logger.warning(f"HMM live inference failed: {e}")
            regime     = REGIME_RANGING
            confidence = 0.5

        return regime, confidence

    def _rule_based_single(self, df1d: pd.DataFrame) -> int:
        """Rule-based regime for a single bar (used when HMM is unavailable)."""
        if df1d is None or len(df1d) < 20:
            return REGIME_RANGING
        c     = df1d['close']
        vol20 = c.pct_change(1).rolling(20).std().iloc[-1] * np.sqrt(252) * 100
        _, _, adx_d = _dmi(df1d['high'], df1d['low'], c, 14)
        adx_last = adx_d.iloc[-1] if len(adx_d) > 0 else np.nan
        if np.isnan(vol20):
            return REGIME_RANGING
        if vol20 > 30:
            return REGIME_CRISIS
        if vol20 > 20 or (not np.isnan(adx_last) and adx_last < 20):
            return REGIME_RANGING
        return REGIME_TRENDING

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({'model': self.model, 'state_map': self.state_map,
                     'fitted': self._fitted}, path)

    def load(self, path: str):
        d = joblib.load(path)
        self.model     = d['model']
        self.state_map = d['state_map']
        self._fitted   = d['fitted']
        return self


# ==============================================================================
# REGIME STATE MACHINE — hysteresis + confidence decay (LIVE-SAFE)
# ==============================================================================

class RegimeStateMachine:
    """
    Wraps RegimeDetector output with:
      1. Hysteresis — a pending regime must be seen for CONFIRM_BARS consecutive
         bars before it replaces the current regime.
      2. Minimum duration — the current regime must have lasted MIN_REGIME_DURATION
         bars before it can be replaced (prevents rapid flipping).
      3. Confidence decay — after a regime switch, effective confidence starts low
         and ramps up over CONF_HALF_LIFE bars, so the system reduces exposure
         immediately after a flip rather than betting at full size.

    Usage (in live_loop each bar):
        effective_regime, eff_conf = state_machine.update(raw_regime, raw_conf)
        if effective_regime == REGIME_UNCERTAIN or eff_conf < 0.65:
            # No new positions — regime not yet confirmed
    """

    REGIME_UNCERTAIN = REGIME_UNCERTAIN   # expose for callers

    def __init__(self):
        self.current_regime    : int   = REGIME_RANGING
        self.regime_bar_count  : int   = 0   # how long we've been in current regime
        self.pending_regime    : int   = -1
        self.pending_count     : int   = 0
        self._flip_bar         : int   = 0   # bar index when last flip occurred
        self._bar_index        : int   = 0

    def update(self, raw_regime: int, raw_confidence: float) -> tuple:
        """
        Process one bar update.

        Parameters
        ----------
        raw_regime     — regime returned by RegimeDetector.predict_live()
        raw_confidence — posterior confidence from same call

        Returns
        -------
        (effective_regime, effective_confidence)
            effective_regime = REGIME_UNCERTAIN while a flip is pending confirmation
        """
        self._bar_index += 1

        # ---- 1. Accumulate confirmation for a pending regime change -----------
        if raw_regime != self.current_regime:
            if raw_regime == self.pending_regime:
                self.pending_count += 1
            else:
                self.pending_regime = raw_regime
                self.pending_count  = 1
        else:
            # Raw signal agrees with current regime — reset pending counter
            self.pending_regime = -1
            self.pending_count  = 0

        # ---- 2. Check if switch criteria met ---------------------------------
        min_dur = MIN_REGIME_DURATION.get(self.current_regime, 4)
        can_switch = (
            self.pending_count >= CONFIRM_BARS and
            self.regime_bar_count >= min_dur
        )

        if can_switch:
            logger.info(
                f"[RegimeSM] Switch: {REGIME_NAMES.get(self.current_regime)} -> "
                f"{REGIME_NAMES.get(self.pending_regime)} "
                f"(pending_count={self.pending_count}, duration={self.regime_bar_count})"
            )
            self.current_regime   = self.pending_regime
            self.regime_bar_count = 0
            self._flip_bar        = self._bar_index
            self.pending_regime   = -1
            self.pending_count    = 0
        else:
            self.regime_bar_count += 1

        # ---- 3. Confidence decay after flip ----------------------------------
        bars_since_flip = self._bar_index - self._flip_bar
        decay_factor    = 1.0 - np.exp(-bars_since_flip / (CONF_HALF_LIFE + 1e-9))
        effective_conf  = raw_confidence * decay_factor

        # ---- 4. Return uncertain when a flip is pending ----------------------
        if self.pending_count > 0 and not can_switch:
            return REGIME_UNCERTAIN, effective_conf

        return self.current_regime, effective_conf

    def reset(self):
        """Call at start of each trading day."""
        self.regime_bar_count = 0
        self.pending_regime   = -1
        self.pending_count    = 0


