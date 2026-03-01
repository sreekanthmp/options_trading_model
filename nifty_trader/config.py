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

_api_limiter = TokenBucket(rate=1.5, capacity=5)   # ~1.5 req/s max; prevents "Access denied"
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
HORIZON_WEIGHTS = {1: 0.10, 5: 0.45, 15: 0.30, 30: 0.15}

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
CONF_STRONG   = 0.68
CONF_MODERATE = 0.60
CONF_MIN      = 0.55
CONF_BY_HORIZON = {1: 0.52, 5: 0.60, 15: 0.65, 30: 0.68}
CONF_FLOOR_TRENDING = 0.58
CONF_FLOOR_RANGING  = 0.72
NO_TRADE_PCTILE_LOW  = 40
NO_TRADE_PCTILE_HIGH = 60

# ---------------------------------------------------------------------------
# Options trade parameters
# ---------------------------------------------------------------------------
LOT_SIZE         = 65
LOT_SIZE_OLD     = 75
LOT_SIZE_NEW     = 65
LOT_SIZE_CUTOVER = date(2025, 4, 26)
MAX_RISK_PCT     = 0.01
STOP_LOSS_PCT    = 0.40
TARGET_PCT       = 0.80
COST_RT_PCT      = 0.025
SLIPPAGE_PCT     = 0.03
THETA_DECAY_PCT  = 0.015
TOTAL_COST_PCT   = COST_RT_PCT + SLIPPAGE_PCT + THETA_DECAY_PCT

# ---------------------------------------------------------------------------
# Strike configuration
# ---------------------------------------------------------------------------
STRIKE_OFFSET_CE = 100
STRIKE_OFFSET_PE = 100
STRIKE_ROUNDING  = 50

# ---------------------------------------------------------------------------
# Delta-decay constants
# ---------------------------------------------------------------------------
DELTA_BASE        = 0.45
THETA_PTS_PER_BAR = 0.15

# ---------------------------------------------------------------------------
# Kill-switch thresholds
# ---------------------------------------------------------------------------
KILL_DAILY_DD_PCT      = 0.02
KILL_CONSEC_LOSSES     = 4
KILL_VOL_SHOCK_MULT    = 2.5
KILL_REGIME_FLIP_MINS  = 30
KILL_VRECOVERY_AGREE   = 0.85

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
MAX_TRADES_PER_DIR_RANGING  = 2
MAX_TRADES_PER_DIR_TRENDING = 4
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
