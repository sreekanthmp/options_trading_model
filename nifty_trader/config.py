"""
NIFTY Intraday Options Trading System — Configuration
All constants, thresholds, file paths, and global state live here.
"""
import os, warnings, logging, threading, time
from datetime import datetime, date
warnings.filterwarnings('ignore')

# Numba optional import
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range

# Optional ML libraries
try:
    from sklearn.ensemble import (GradientBoostingClassifier,
                                   RandomForestClassifier, VotingClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    LGB_OK = False

try:
    from hmmlearn import hmm
    HMM_OK = True
except ImportError:
    HMM_OK = False

try:
    from SmartApi.smartWebSocketV2 import SmartWebSocketV2
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'nifty_trader_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
class TokenBucket:
    """Token bucket rate limiter for API call throttling."""
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_time = time.time()
        self.lock = threading.Lock()

    def acquire(self, tokens=1) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_time = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def wait_and_acquire(self, tokens=1):
        while not self.acquire(tokens):
            time.sleep(0.05)

_api_limiter = TokenBucket(rate=0.4, capacity=3)   # ~1 req/2.5s; Angel One AB1004 safe threshold
_training_feature_stats = {}

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
DATA_1MIN   = "NIFTY_1min_history.csv"
DATA_5MIN   = "NIFTY_5min_history.csv"
DATA_15MIN  = "NIFTY_15min_history.csv"
DATA_1DAY   = "NIFTY_1day_history.csv"
MODEL_DIR   = "options_trader_models"
CONFIG_FILE = "config.json"

# ---------------------------------------------------------------------------
# WebSocket settings
# ---------------------------------------------------------------------------
USE_WEBSOCKET    = True
TICK_BUFFER_SIZE = 100
NIFTY_TOKEN      = "99926000"

# ---------------------------------------------------------------------------
# Prediction horizons
# ---------------------------------------------------------------------------
HORIZONS        = [1, 5, 15, 30]
# Weights rebalanced: 5m had 45% weight but only 56% accuracy (was dominating votes).
# 15m/30m have 61%/67% accuracy — they should dominate. 5m demoted to tiebreaker role.
HORIZON_WEIGHTS = {1: 0.05, 5: 0.20, 15: 0.45, 30: 0.30}

# ---------------------------------------------------------------------------
# Triple-barrier parameters
# ---------------------------------------------------------------------------
TB_MULT_TRENDING = {1: 1.2, 5: 1.8, 15: 2.2, 30: 2.8}
TB_MULT_RANGING  = {1: 0.9, 5: 1.2, 15: 1.5, 30: 2.0}
# Fallback for Crisis regime: 2x Ranging width
TB_MULT_CRISIS   = {k: 2.0 * v for k, v in TB_MULT_RANGING.items()}
TB_BARS          = {1: 10,  5: 30,  15: 12,  30: 12}  # Increased 30-min horizon bars to reduce saturation
TB_MIN_MOVE_PCT  = 0.20

# ---------------------------------------------------------------------------
# Regime codes
# ---------------------------------------------------------------------------
REGIME_TRENDING = 0
REGIME_RANGING  = 1
REGIME_CRISIS   = 2
REGIME_NAMES    = {-1: "UNCERTAIN", 0: "TRENDING", 1: "RANGING", 2: "CRISIS"}
REGIME_ICONS    = {-1: "[WAIT]",    0: "[TREND]",  1: "[RANGE]", 2: "[CRISIS]"}

# ---------------------------------------------------------------------------
# Signal confidence thresholds
# ---------------------------------------------------------------------------
CONF_STRONG   = 0.96   # above this = suspect overfit; hard-capped in generator
CONF_MODERATE = 0.92
CONF_MIN      = 0.52   # relaxed: allow signals with modest model confidence
CONF_BY_HORIZON = {1: 0.52, 5: 0.52, 15: 0.52, 30: 0.52}
CONF_FLOOR_TRENDING = 0.52
CONF_FLOOR_RANGING  = 0.52
NO_TRADE_PCTILE_LOW  = 40
NO_TRADE_PCTILE_HIGH = 60

# ---------------------------------------------------------------------------
# Options trade parameters
# ---------------------------------------------------------------------------
LOT_SIZE         = 65
LOT_SIZE_OLD     = 75
LOT_SIZE_NEW     = 65
LOT_SIZE_CUTOVER = date(2025, 4, 26)
MAX_RISK_PCT        = 0.01   # 1% risk per trade (kept for reference, not used for sizing)
CAPITAL_DEPLOY_PCT  = 0.75   # 75% of capital as base premium budget per trade
#   0.75 = leaves headroom for confidence/regime/vol multipliers to scale up to full
#          capital without requiring multipliers > 1.0 (no hidden leverage).
#   1.00 = deploys full capital before multipliers — any mult > 1.0 becomes leverage.
#   Lot count = floor(capital * CAPITAL_DEPLOY_PCT / (premium * lot_size))
#   Example: Rs 30,000 × 75% / (217 × 65) = 1 lot (vs 2 at 100%)
MAX_CONTRACTS       = 2      # 2 lots per trade — enables partial exit at target
# Stop 10% (was 5% — the 5% stop never triggered; 30-min time exit fired first).
# 10% = Rs 20 on Rs 200 ATM premium = 13-pt adverse NIFTY move. Real adverse move, not noise.
# Target 22% (was 25% — slightly reduced to increase target-hit frequency).
# Reward:risk = 10% stop / 22% target = 1:2.2 — requires only 31% WR to break even.
STOP_LOSS_PCT    = 0.10
TARGET_PCT       = 0.22
# Daily trade cap: max 2 entries per session.
# Paper data: 55% of all brokerage cost came from the 3rd+ trade each day.
# 3rd trade only permitted when first 2 were both profitable (see live.py).
MAX_TRADES_PER_SESSION = 10
# Daily loss limit in Rs (hard stop for the day).
# Rs 1500 ≈ two full 10%-stop losses on 1 lot at Rs 230 ATM premium × 65 qty (~Rs 1495).
DAILY_LOSS_LIMIT_RS = 1500.0
# After 2 consecutive losses (any exit reason), block new entries for this many minutes.
# Prevents revenge trading and loss-spiral on choppy days.
# Evidence: Apr 20-24 losing streak — 3-4 consecutive same-day losses compounded losses.
CONSEC_LOSS_COOLDOWN_MINS = 45
# Gap-day single-trade cap: when abs(gap_pct) > GAP_DAY_PCT, limit to 1 trade per day.
# Apr 20: gap_pct=-2.14%, 3 trades taken, all 3 lost (-Rs 4,158). The model is directionally
# confused on large gap days — the single best trade is allowed, then stop.
GAP_DAY_SINGLE_TRADE_PCT = 1.5   # 1.5% gap triggers 1-trade-only cap
# Rolling win-rate guard: if the last N closed trades have WR below this threshold,
# switch to observation mode (no new entries) until market improves.
# Prevents the system from trading into a regime it has no edge in.
ROLLING_WR_WINDOW = 5    # look at last 5 completed trades
ROLLING_WR_MIN    = 0.30  # if WR < 30% → observation mode (1.5 wins in last 5)
# Pre-market skip-day thresholds (D1 checks — evaluated at session start)
VIX_HALVE_THRESHOLD   = 22.0   # VIX > 22 → halve position size (use 0.5 lot multiplier)
VIX_SKIP_THRESHOLD    = 999.0  # disabled — never skip day on VIX
NIFTY_MA20_SKIP_PCT   = 999.0  # disabled — never skip on MA distance
DAILY_LOSS_SKIP_DAYS  = 999    # disabled — never skip on loss streak
DAILY_LOSS_WINDOW     = 5      # rolling window of trading days for loss-day count
# Mid-session double-confirm: after first intraday loss, require near-perfect trend
# score (3.5/4 = 87.5%) before the next entry. Normal threshold is 3/4 (75%).
MID_SESSION_SCORE_FLOOR = 0.875  # session_regime_score must exceed this after 1st loss
# DOWN-only mode: only take PE (DOWN) signals until UP side shows positive live WR.
# UP (CE) trades: 7W/9L net negative after charges in live data. Disable until validated.
# Set to False to re-enable UP trades (requires 30+ CE trades with positive expectancy).
DOWN_ONLY_MODE = False
COST_RT_PCT      = 0.004   # Realistic Angel One round-trip cost (~0.4% on Rs 200-400 ATM premium)
SLIPPAGE_PCT     = 0.008   # Realistic limit-order slippage (0.8% = ~Rs 2 on Rs 250 premium)
THETA_DECAY_PCT  = 0.003   # Theta drag per bar (small — real theta handled in option_pnl_estimate)
TOTAL_COST_PCT   = COST_RT_PCT + SLIPPAGE_PCT + THETA_DECAY_PCT  # ~1.5% total (was 7% — 5x too high, blocked valid EV trades)

# ---------------------------------------------------------------------------
# Real brokerage + statutory costs (Angel One, NSE F&O options — per SIDE)
# These are deducted from paper trade PnL so paper matches live net P&L exactly.
#
# Cost breakdown per TRADE (one round-trip = entry + exit):
#   Brokerage:        Rs 20 flat per order (Angel One flat fee model)
#   STT (buy side):   0% on option BUY (no STT on buyer for non-expiry)
#   STT (sell side):  0.1% of premium × qty  (on sell/exit leg)
#   NSE transaction:  0.053% of premium turnover (both sides)
#   SEBI charges:     Rs 10 per crore of turnover (~0.0001%)
#   GST on brokerage: 18% of brokerage
#   Stamp duty:       0.003% of BUY side premium (Maharashtra)
#
# For a typical ATM option trade:
#   Entry: Rs 400 premium, 1 lot (65 qty) = Rs 26,000 turnover
#   Brokerage both sides: Rs 20 × 2 = Rs 40
#   GST on brokerage: Rs 40 × 0.18 = Rs 7.20
#   STT on sell side: 26,000 × 0.001 = Rs 26
#   NSE transaction (both sides): 26,000 × 2 × 0.00053 = Rs 27.56
#   Stamp duty (buy side): 26,000 × 0.00003 = Rs 0.78
#   Total ≈ Rs 102 per round-trip on Rs 26,000 = ~0.39% round-trip cost
#
# Note: STT on expiry is MUCH higher (0.125% of intrinsic value) — handled separately
BROKERAGE_PER_ORDER  = 20.0     # Rs 20 flat per order (Angel One)
GST_ON_BROKERAGE     = 0.18     # 18% GST on brokerage
STT_SELL_PCT         = 0.001    # 0.1% of premium on SELL side
NSE_TXN_PCT          = 0.00053  # 0.053% of turnover each side
STAMP_DUTY_PCT       = 0.00003  # 0.003% of BUY side turnover
STT_EXPIRY_PCT       = 0.00125  # 0.125% on expiry day (on intrinsic, approx premium)

# ---------------------------------------------------------------------------
# Expiry-day trading rules
# ---------------------------------------------------------------------------
# NIFTY weekly expiry is every TUESDAY (changed from Thursday in Sep 2024).
#
# Expiry day is split into three risk zones based on gamma behaviour:
#
#   Zone 1  — 09:15-11:29  (minute_of_day   0-134)
#     Moderate gamma. Market finds direction from yesterday's OI/PCR.
#     Allow trading but at half position size and tighter 3.5% stops.
#
#   Zone 2  — 11:30-12:59  (minute_of_day 135-224)
#     "Max-pain pinning" zone. NIFTY often grinds sideways as MMs hedge.
#     Low signal quality — whipsaw risk high. Block new entries entirely.
#
#   Zone 3  — 13:00-14:49  (minute_of_day 225-334)
#     "Pin-break" zone. Large directional moves as OTM options go worthless.
#     Highest gamma, fastest theta. Allow at quarter size, ultra-tight 2.5% stop.
#
#   Zone 4  — 14:50-15:30  (minute_of_day 335+)
#     Last 40 minutes. Gamma extreme, bid-ask wide. Block all new entries.
#     (This is also caught by the normal session gate at mod>=335.)
#
# stop_tighten: multiplier applied to STOP_LOSS_PCT (e.g. 0.70 → 3.5% if STOP=5%)
# size_mult:    multiplier applied to contract count from select_option()
# allow_new:    whether new entries are allowed in this zone
EXPIRY_ZONES = {
    # Zone 1 (09:15-11:29): Full capital.
    'zone1': {'allow_new': True,  'size_mult': 1.0, 'stop_tighten': 0.875, 'mod_start':   0, 'mod_end': 134},
    # Zone 2 (11:30-12:59): Max-pain pinning — block new entries.
    'zone2': {'allow_new': False, 'size_mult': 1.0, 'stop_tighten': 0.625, 'mod_start': 135, 'mod_end': 224},
    # Zone 3 (13:00-14:49): Pin-break zone — full capital.
    'zone3': {'allow_new': True,  'size_mult': 1.0, 'stop_tighten': 0.5,   'mod_start': 225, 'mod_end': 334},
    # Zone 4 (14:50+): Extreme gamma + wide bid-ask. Block all new entries.
    'zone4': {'allow_new': False, 'size_mult': 1.0, 'stop_tighten': 0.5,   'mod_start': 335, 'mod_end': 999},
}

EXPIRY_MAX_HOLD_MINS  = 10     # expiry day: exit if losing after 10 min (was 3 — too short for options to develop)
EXPIRY_FORCE_EXIT_MOD = 310    # minute_of_day 310 = 14:40 IST — flatten all positions
EXPIRY_CONF_FLOOR     = 0.65   # need stronger signal on expiry day
EXPIRY_IV_BLOCK_PCT   = 20.0   # annualised IV%; expiry-day IV can spike 2-3x intraday

# ---------------------------------------------------------------------------
# Strike configuration
# ---------------------------------------------------------------------------
# 1-strike ITM for higher delta (~0.60 vs ATM ~0.50) and lower theta drag.
# ITM CE = spot - 50 (call ITM when spot above strike); ITM PE = spot + 50.
# Bid-ask spread is similar to ATM at NIFTY liquidity levels.
# Negative offset = ITM; positive offset = OTM.
STRIKE_OFFSET_CE = 0     # ATM call
STRIKE_OFFSET_PE = 0     # ATM put
STRIKE_ROUNDING  = 50

# ---------------------------------------------------------------------------
# Delta-decay constants
# ---------------------------------------------------------------------------
DELTA_BASE        = 0.45
THETA_PTS_PER_BAR = 0.15

# ---------------------------------------------------------------------------
# LIMIT Order Simulation (paper trading realism — mirrors April 2026 SEBI rules)
# ---------------------------------------------------------------------------
# In live trading, MARKET orders are banned. All orders must be LIMIT orders.
# These constants simulate realistic LIMIT order behaviour in paper trading.
#
# How it works:
#   Entry: place LIMIT BUY at LTP + LIMIT_BUY_BUFFER_PCT above current LTP
#          Fill probability = LIMIT_FILL_PROB_ATM for ATM options
#          If unfilled (rare for ATM), trade is skipped — same as live
#   Exit:  place LIMIT SELL at LTP - LIMIT_SELL_BUFFER_PCT below current LTP
#          Simulates giving up the spread on exit
#
# Bid-ask spread proxy (NSE ATM NIFTY weekly options, normal conditions):
#   ATM spread  ≈ 0.5–2 pts  (tight, high liquidity)
#   OTM spread  ≈ 2–8 pts    (wider, lower OI)
#   High-IV day spread ≈ 3–10 pts
#
# Fill probability (ATM LIMIT near LTP):
#   Normal session : ~97%  (almost always fills for ATM)
#   High-vol/expiry: ~88%  (more slippage, some misses)
LIMIT_BUY_BUFFER_PCT  = 0.003   # 0.3% above LTP for entry — fallback for low-priced options
LIMIT_BUY_BUFFER_PTS  = 1.5     # Rs 1.5 fixed buffer — dominates for ATM options ≥ Rs 100
# Entry limit = max(ltp + LIMIT_BUY_BUFFER_PTS, ltp * (1 + LIMIT_BUY_BUFFER_PCT))
# Real ATM bid-ask spread is 1-3 Rs; 0.3% = 0.6-0.75 Rs on Rs 200-250 — too tight to fill.
LIMIT_SELL_BUFFER_PCT = 0.003   # 0.3% below LTP for exit  (≈1–2 pts on ₹400 ATM)
LIMIT_FILL_PROB_ATM   = 0.97    # 97% fill rate for ATM options in normal session
LIMIT_FILL_PROB_EXPIRY= 0.88    # 88% fill rate on expiry day (wider spreads)
LIMIT_FILL_PROB_HIGH_IV= 0.90   # 90% fill rate when IV rank > 70
# Spread penalty on entry: extra cost when spread is wider than normal
# Applied as additional % of premium above the buffer
LIMIT_SPREAD_NORMAL_PCT = 0.002  # 0.2% extra cost — normal spread (ATM)
LIMIT_SPREAD_HIGH_IV_PCT= 0.006  # 0.6% extra cost — high-IV / OTM spread

# ---------------------------------------------------------------------------
# Kill-switch thresholds
# ---------------------------------------------------------------------------
KILL_DAILY_DD_PCT         = 0.10   # Live: 10% daily loss limit
KILL_DAILY_DD_PCT_PAPER   = 0.12   # Paper: 12% daily loss limit (was 1.00 — paper now has a real circuit breaker)
KILL_CONSEC_LOSSES        = 2      # 2 consecutive losses → 30 min cooldown (live)
KILL_CONSEC_LOSSES_PAPER  = 3      # Paper: allow 3 losses before cooldown
KILL_VOL_SHOCK_MULT       = 3.0    # Raised from 2.5 — normal breakout ATR spikes 2-2.5x; only block flash crashes
KILL_REGIME_FLIP_MINS     = 30
KILL_VRECOVERY_AGREE      = 0.85
# Weekly/monthly drawdown protection
# Weekly: >5% drawdown from the week's starting equity → halve position size.
# Monthly: >10% drawdown from the month's starting equity → halt all new entries.
# These operate on INITIAL capital (not rolling peak) to avoid the ratchet effect
# where a single bad week permanently reduces next week's size budget.
KILL_WEEKLY_DD_HALVE_PCT  = 0.05   # 5% weekly DD → 0.5x size multiplier
KILL_MONTHLY_DD_HALT_PCT  = 0.10   # 10% monthly DD → halt trading, manual review required
# Confidence-based position sizing multipliers.
# Applied to the capital budget passed to select_option().
# Keeps full-size entries only for high-confidence signals; scales down borderline ones.
# Bands chosen so that a 0.52 signal (barely above CONF_MIN) gets 50% capital,
# while a 0.65+ signal (strong) gets full capital.
CONF_SIZE_BANDS = [
    (0.65, 1.00),   # conf >= 0.65 → 100% capital
    (0.60, 0.80),   # conf >= 0.60 → 80%
    (0.55, 0.65),   # conf >= 0.55 → 65%
    (0.00, 0.50),   # conf >= 0.52 (CONF_MIN) → 50%
]

# ---------------------------------------------------------------------------
# EV-Harvesting / v3.1 constants
# ---------------------------------------------------------------------------
FRACDIFF_D      = 0.35
FRACDIFF_THRESH = 1e-4
FSI_SIGMA_THRESH = 2.0
FSI_WINDOW_DAYS  = 5
FSI_BASELINE_YRS = 3
DYN_H_MIN_MULT  = 0.5
DYN_H_MAX_MULT  = 2.0
TB_VELOCITY_DECAY_LAMBDA = 0.5
TB_PUT_TIME_DECAY_EXTRA  = 0.20
TB_CALL_TP_TRENDING_MULT = 1.07
EV_SAFETY_NORMAL   = 1.0
EV_SAFETY_ELEVATED = 1.8
EV_AVG_WIN_MULT    = 1.2   # Options winners: asymmetric upside (can return >1:1)
EV_AVG_LOSS_MULT   = 1.0   # Options losers: capped at premium paid
OPTION_SLIPPAGE_PCT = 0.03  # 3% slippage for option execution (bid-ask + impact)
TEMPORAL_FLIP_LOCK_BARS = 3
TRANSITION_SHOCK_MINS = 15
TRANSITION_CONF_BOOST = 0.07
SEASONALITY_MAX_BIAS  = 0.03
MAX_TRADES_PER_DIR_RANGING  = 10    # relaxed
MAX_TRADES_PER_DIR_TRENDING = 10    # relaxed
# Session time-window for new entries (minute_of_day).
# All 6 paper winners fell in 12:00-14:15 (mod 165-300). Morning session is a loss center.
ENTRY_MOD_MIN = 30    # 09:45 AM — skip first 30 min price discovery noise
ENTRY_MOD_MAX = 330   # 14:45 PM — close to session end
DD_CLUSTER_THRESH = -2.0   # 2pp of cumulated % returns (was -0.02 which flagged 83% of samples)
DD_SAMPLE_WEIGHT  = 0.3
TRAIN_TIME_DECAY  = 1.5

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
BAR_WIDTH = 30

# ---------------------------------------------------------------------------
# Mandatory features (model must always receive these)
# ---------------------------------------------------------------------------
MANDATORY_FEATURES = [
    'close', 'atr_14', 'atr_14_pct',
    'rsi_14', 'adx_14', 'macd_h',
    'vwap_dist', 'above_vwap',
]

# ---------------------------------------------------------------------------
# Quant-rebuild signal quality filters (2026-04-26)
# ---------------------------------------------------------------------------
# ADX valid range for entries: 20-40.
# Paper data: losers had ADX avg=34.5, winners ADX avg=28.7.
# ADX<20 = no directional structure (Gate7c-ADX already blocks).
# ADX>40 = exhausted/overextended trend, high reversal risk.
ADX_ENTRY_MIN = 20   # already enforced in Gate7c-ADX; kept here as config constant
ADX_ENTRY_MAX = 40   # new upper cap: ADX>40 = chasing exhausted move
# Hard-block RANGING regime entirely.
# Paper data: 5 trades in RANGING, WR=0%, all losses, -Rs 2,100 gross.
# There is zero evidence of edge in RANGING; block unconditionally.
BLOCK_RANGING_REGIME = False
# Hard-block TRENDING regime.
# Paper data: 5 trades in TRENDING, WR=0%, all losses, -Rs 1,600 gross.
# 6 of 7 all-time winners were CRISIS regime. TRENDING shows no live edge.
# Set to False only after 30+ TRENDING trades with positive expectancy.
BLOCK_TRENDING_REGIME = False
# ADX hard cap for CRISIS bypass entries.
# CRISIS bypass normally skips the ADX_ENTRY_MAX (40) gate.
# But ADX>50 even in CRISIS = panic exhaustion, not directional momentum.
# Paper data: CRISIS ADX>50 trades were all losses (4/4 = 0% WR).
# This cap is NOT skipped — exhaustion is a price-reality filter.
ADX_CRISIS_MAX = 50
# Pressure ratio floor for DOWN (PE) entries.
# pressure_ratio = put pressure / call pressure from tick-level order flow.
# Paper data (28 trades): signal is inverted at low PR — the two biggest winners had
# PR=0.25 and PR=0.43. Mean "winners=1.59" was skewed by one outlier (PR=4.0).
# Gate disabled (0.0 = never fires) until 50+ DOWN trades provide reliable calibration.
# Set to 1.20+ once calibrated from live data.
PRESSURE_RATIO_DOWN_MIN = 0.0
# MFE confirmation gate: exit early if option has not moved +MFE_CONFIRM_PTS
# in our favour within MFE_CONFIRM_BARS bars of entry.
# Paper data: every single losing trade had MFE=0.00 at exit.
# No winner ever had MFE=0. This is the cleanest loss discriminator.
MFE_CONFIRM_BARS = 2    # exit if option hasn't moved +3pts in our favour within 2 bars
MFE_CONFIRM_PTS  = 3.0  # paper data: every loser had MFE=0, every winner had MFE>=3.07
