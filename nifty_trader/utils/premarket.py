"""
Pre-Market Bias Analysis
========================
Runs before 9:15 market open to compute a directional bias score.
Uses already-downloaded CSV files — no extra API calls needed except GIFT Nifty.

Bias score: -2 (strong bearish) to +2 (strong bullish), 0 = neutral.

Components:
  1. India VIX level     — high VIX = volatile day (not directional, but affects sizing)
  2. S&P 500 last close  — US risk-on/off
  3. FII/DII net flow    — institutional positioning
  4. GIFT Nifty gap      — pre-market futures vs yesterday's close

Effect on trading:
  - Bullish bias  (+1/+2): UP signals get conf boost, DOWN signals need higher floor
  - Bearish bias  (-1/-2): DOWN signals get conf boost, UP signals need higher floor
  - High VIX day         : scarcity limit relaxed (allow more trades)
  - Neutral (0)          : no change to normal gates
"""
import os, logging
import pandas as pd
import numpy as np
from datetime import date, datetime

logger = logging.getLogger(__name__)

EPS = 1e-9

# Conf adjustment applied to opposite-direction signals
# e.g. bearish bias → UP signals need +CONF_PENALTY more confidence
BIAS_CONF_PENALTY  = 0.04   # per bias unit (4% conf penalty per bias point)
BIAS_CONF_BOOST    = 0.02   # per bias unit (2% conf boost for aligned signals)

# VIX thresholds
VIX_HIGH_THRESHOLD    = 18.0   # >18 = volatile day → allow 1 extra trade
VIX_HALVE_THRESHOLD   = 22.0   # >22 = high-vol regime → halve position size
VIX_SKIP_THRESHOLD    = 30.0   # >30 = extreme panic → skip day entirely
VIX_EXTREME_THRESHOLD = 25.0   # kept for compatibility

# Max extra trades allowed on high-VIX day
HIGH_VIX_EXTRA_TRADES = 1      # allow 1 extra trade per day when VIX > 18

# NIFTY / MA skip threshold
NIFTY_MA20_SKIP_PCT   = 0.005  # skip if NIFTY is >0.5% away from 20-day MA

# Rolling daily loss guard
DAILY_LOSS_SKIP_DAYS  = 3      # skip day if 3+ losing days in last 5
DAILY_LOSS_WINDOW     = 5


class PreMarketBias:
    """
    Computes and holds the pre-market bias for the current trading day.
    Instantiated once at startup, read during signal generation.
    """

    def __init__(self):
        self.bias_score      = 0        # -2 to +2
        self.vix_level       = 0.0      # yesterday's VIX close
        self.vix_high_day    = False    # True if VIX > 18
        self.vix_extreme_day = False    # True if VIX > 25
        self.vix_halve       = False    # True if VIX > 22 → halve position size
        self.skip_day        = False    # True if any hard D1 rule fires → no trading today
        self.skip_reason     = ''       # human-readable reason for skip_day
        self.daily_loss_skip = False    # True if 3+ losing days in last 5
        self.nifty_ma20_skip = False    # True if NIFTY too far from 20-day MA
        self.sp500_ret       = 0.0      # S&P 500 last return %
        self.fii_net         = 0.0      # FII net buy (crores)
        self.gift_gap_pct    = 0.0      # GIFT Nifty gap % vs prev close
        self.components      = {}       # individual component scores
        self.computed        = False
        self.trade_date      = None

    def compute(self, session=None):
        """
        Compute pre-market bias from available CSV files.
        Call this once before market open.
        session: AngelSession (optional, for GIFT Nifty fetch)
        """
        today = date.today()
        self.trade_date = today
        score = 0
        components = {}

        # ── 1. India VIX ──────────────────────────────────────────────────────
        vix_path = 'india_vix.csv'
        if os.path.exists(vix_path):
            try:
                vix = pd.read_csv(vix_path)
                vix['date'] = pd.to_datetime(vix['date']).dt.date
                vix = vix.sort_values('date')
                # Use last available day (yesterday)
                past = vix[vix['date'] < today]
                if not past.empty:
                    self.vix_level = float(past['vix_close'].iloc[-1])
                    self.vix_high_day    = self.vix_level > VIX_HIGH_THRESHOLD
                    self.vix_extreme_day = self.vix_level > VIX_EXTREME_THRESHOLD
                    # VIX direction: if VIX rising sharply, market is getting fearful
                    if len(past) >= 2:
                        vix_prev = float(past['vix_close'].iloc[-2])
                        vix_chg = (self.vix_level - vix_prev) / (vix_prev + EPS) * 100
                        if vix_chg > 10:
                            # VIX spiking → bearish bias
                            vix_score = -1
                        elif vix_chg < -10:
                            # VIX falling → fear easing → bullish bias
                            vix_score = +1
                        else:
                            vix_score = 0
                    else:
                        vix_score = 0
                    components['vix'] = vix_score
                    score += vix_score
                    logger.info(f"[PreMarket] VIX: {self.vix_level:.1f} "
                                f"({'HIGH' if self.vix_high_day else 'normal'}) "
                                f"score={vix_score:+d}")
            except Exception as e:
                logger.warning(f"[PreMarket] VIX read failed: {e}")
                components['vix'] = 0
        else:
            components['vix'] = 0

        # ── 2. S&P 500 ────────────────────────────────────────────────────────
        sp_path = 'sp500_daily.csv'
        if os.path.exists(sp_path):
            try:
                sp = pd.read_csv(sp_path)
                sp['date'] = pd.to_datetime(sp['date']).dt.date
                sp = sp.sort_values('date')
                past_sp = sp[sp['date'] < today]
                if not past_sp.empty:
                    sp_ret = float(past_sp['close'].pct_change().iloc[-1] * 100)
                    self.sp500_ret = sp_ret
                    if sp_ret > 1.0:
                        sp_score = +1
                    elif sp_ret < -1.0:
                        sp_score = -1
                    else:
                        sp_score = 0
                    components['sp500'] = sp_score
                    score += sp_score
                    logger.info(f"[PreMarket] S&P500: {sp_ret:+.2f}%  score={sp_score:+d}")
            except Exception as e:
                logger.warning(f"[PreMarket] S&P500 read failed: {e}")
                components['sp500'] = 0
        else:
            components['sp500'] = 0

        # ── 3. FII/DII Flow ───────────────────────────────────────────────────
        fii_path = 'fii_dii_flow.csv'
        if os.path.exists(fii_path):
            try:
                fii = pd.read_csv(fii_path)
                fii['date'] = pd.to_datetime(fii['date']).dt.date
                fii = fii.sort_values('date')
                past_fii = fii[fii['date'] < today]
                if not past_fii.empty:
                    self.fii_net = float(past_fii['fii_net_buy'].iloc[-1])
                    if self.fii_net > 2000:
                        fii_score = +1
                    elif self.fii_net < -2000:
                        fii_score = -1
                    else:
                        fii_score = 0
                    components['fii'] = fii_score
                    score += fii_score
                    logger.info(f"[PreMarket] FII net: {self.fii_net:+,.0f} cr  score={fii_score:+d}")
            except Exception as e:
                logger.warning(f"[PreMarket] FII read failed: {e}")
                components['fii'] = 0
        else:
            components['fii'] = 0

        # ── 4. GIFT Nifty gap ─────────────────────────────────────────────────
        # Try fetching GIFT Nifty pre-market price from Angel One
        gift_score = 0
        if session is not None:
            try:
                gift_score = self._fetch_gift_nifty_gap(session)
                self.gift_gap_pct = gift_score * 0.5  # approximate
            except Exception as e:
                logger.warning(f"[PreMarket] GIFT Nifty fetch failed: {e}")
        components['gift_nifty'] = gift_score
        score += gift_score
        if gift_score != 0:
            logger.info(f"[PreMarket] GIFT Nifty gap score={gift_score:+d}")

        # ── D1 Skip-day rules ─────────────────────────────────────────────────
        # These override the bias score — if skip_day=True, no trading at all.

        # Rule D1-A: VIX extreme → skip day
        if self.vix_level > VIX_SKIP_THRESHOLD:
            self.skip_day    = True
            self.skip_reason = f'VIX={self.vix_level:.1f} > {VIX_SKIP_THRESHOLD} (extreme panic)'
        # Rule D1-B: VIX > 22 → halve position size (not a skip, but a constraint)
        if self.vix_level > VIX_HALVE_THRESHOLD:
            self.vix_halve = True
            logger.info(f"[PreMarket] VIX={self.vix_level:.1f} > {VIX_HALVE_THRESHOLD} — position size HALVED")

        # Rule D1-C: NIFTY too far from 20-day MA → skip
        daily_path = 'NIFTY_1day_history.csv'
        if os.path.exists(daily_path):
            try:
                _d = pd.read_csv(daily_path)
                _d['datetime'] = pd.to_datetime(_d['datetime'])
                _d = _d.sort_values('datetime')
                if len(_d) >= 20:
                    _ma20    = float(_d['close'].iloc[-20:].mean())
                    _last_cl = float(_d['close'].iloc[-1])
                    _ma_dev  = abs(_last_cl - _ma20) / (_ma20 + EPS)
                    if _ma_dev > NIFTY_MA20_SKIP_PCT:
                        self.nifty_ma20_skip = True
                        if not self.skip_day:
                            self.skip_day    = True
                            self.skip_reason = (f'NIFTY {_last_cl:.0f} is {_ma_dev*100:.2f}% '
                                                f'from MA20={_ma20:.0f} (>{NIFTY_MA20_SKIP_PCT*100:.1f}% threshold)')
                        logger.info(f"[PreMarket] MA20 skip: NIFTY={_last_cl:.0f} MA20={_ma20:.0f} dev={_ma_dev*100:.2f}%")
            except Exception as _e:
                logger.warning(f"[PreMarket] MA20 check failed: {_e}")

        # Rule D1-D: Rolling daily loss guard — read recent daily P&L from trade logs
        try:
            import glob as _glob
            _log_files = sorted(_glob.glob('logs/trades_*.jsonl'))
            _recent_days_pnl = []
            for _f in _log_files[-DAILY_LOSS_WINDOW - 1:]:  # last N+1 files, skip today
                _f_date = os.path.basename(_f).replace('trades_', '').replace('.jsonl', '')
                if _f_date == str(today):
                    continue  # skip today's file (incomplete)
                _day_pnl = 0.0
                try:
                    with open(_f, encoding='utf-8') as _fh:
                        for _line in _fh:
                            _line = _line.strip()
                            if not _line:
                                continue
                            import json as _json
                            _ev = _json.loads(_line)
                            if _ev.get('event') == 'EXIT':
                                _day_pnl += float(_ev.get('pnl', _ev.get('pnl_rs', 0)))
                except Exception:
                    pass
                if _day_pnl != 0.0 or True:  # include days with zero pnl (no-trade days)
                    _recent_days_pnl.append(_day_pnl)
            _recent_days_pnl = _recent_days_pnl[-DAILY_LOSS_WINDOW:]
            if len(_recent_days_pnl) >= DAILY_LOSS_WINDOW:
                _losing_days = sum(1 for p in _recent_days_pnl if p < 0)
                if _losing_days >= DAILY_LOSS_SKIP_DAYS:
                    self.daily_loss_skip = True
                    if not self.skip_day:
                        self.skip_day    = True
                        self.skip_reason = (f'{_losing_days} losing days in last {DAILY_LOSS_WINDOW} '
                                            f'— regime unfavourable')
                    logger.info(f"[PreMarket] Daily loss skip: {_losing_days}/{DAILY_LOSS_WINDOW} losing days")
        except Exception as _e:
            logger.warning(f"[PreMarket] Daily loss check failed: {_e}")

        # ── Final score ───────────────────────────────────────────────────────
        # Clamp to [-2, +2]
        self.bias_score = max(-2, min(2, score))
        self.components = components
        self.computed   = True

        direction = "BULLISH" if self.bias_score > 0 else ("BEARISH" if self.bias_score < 0 else "NEUTRAL")
        logger.info(f"[PreMarket] Final bias: {self.bias_score:+d} ({direction}) "
                    f"| VIX={self.vix_level:.1f} SP500={self.sp500_ret:+.2f}% "
                    f"FII={self.fii_net:+,.0f}cr"
                    f"{' | SKIP DAY: ' + self.skip_reason if self.skip_day else ''}")
        return self.bias_score

    def _fetch_gift_nifty_gap(self, session) -> int:
        """
        Fetch GIFT Nifty (SGX Nifty) pre-market price and compute gap vs prev close.
        Returns: +1 (bullish gap), -1 (bearish gap), 0 (flat)
        """
        try:
            # GIFT Nifty token on Angel One — NSE_INDEX segment
            # Token 99926000 = NIFTY 50 spot; GIFT is on a different exchange
            # Try fetching last available NIFTY futures price as proxy
            from ..data.websocket import fetch_live_candles
            # This will get the most recent NIFTY candle available pre-market
            # which reflects futures gap direction
            df = fetch_live_candles(session, n=5)
            if df is not None and not df.empty:
                last_close = float(df['close'].iloc[-1])
                # Compare with yesterday's close from daily CSV
                daily_path = 'NIFTY_1day_history.csv'
                if os.path.exists(daily_path):
                    d = pd.read_csv(daily_path)
                    d['datetime'] = pd.to_datetime(d['datetime'])
                    d = d.sort_values('datetime')
                    prev_close = float(d['close'].iloc[-1])
                    gap_pct = (last_close - prev_close) / (prev_close + EPS) * 100
                    self.gift_gap_pct = gap_pct
                    if gap_pct > 0.3:
                        return +1
                    elif gap_pct < -0.3:
                        return -1
        except Exception as e:
            logger.debug(f"[PreMarket] GIFT gap calc failed: {e}")
        return 0

    def get_conf_adjustment(self, signal_direction: str) -> float:
        """
        Returns confidence adjustment for a signal given current bias.
        Positive = boost (aligned with bias)
        Negative = penalty (against bias)

        e.g. bias=+2, signal=UP  → +0.04 boost
             bias=+2, signal=DOWN → -0.08 penalty
        """
        if not self.computed or self.bias_score == 0:
            return 0.0

        abs_bias = abs(self.bias_score)
        bias_dir = 'UP' if self.bias_score > 0 else 'DOWN'

        if signal_direction == bias_dir:
            return abs_bias * BIAS_CONF_BOOST      # aligned: small boost
        else:
            return -(abs_bias * BIAS_CONF_PENALTY)  # opposed: penalty

    def get_extra_trades_allowed(self) -> int:
        """Returns extra trades allowed today based on VIX level."""
        if self.vix_extreme_day:
            return 0   # extreme VIX → no extra trades, be cautious
        if self.vix_high_day:
            return HIGH_VIX_EXTRA_TRADES
        return 0

    def print_summary(self):
        """Print pre-market summary to console."""
        if not self.computed:
            print("  [PreMarket] Not yet computed.")
            return

        bias_str = f"{self.bias_score:+d}"
        direction = "BULLISH" if self.bias_score > 0 else ("BEARISH" if self.bias_score < 0 else "NEUTRAL")
        vix_str   = f"{self.vix_level:.1f} ({'HIGH-VIX DAY' if self.vix_high_day else 'normal'})"

        print(f"\n{'='*60}")
        print(f"  PRE-MARKET BIAS ANALYSIS  [{self.trade_date}]")
        print(f"{'='*60}")
        print(f"  Overall bias  : {bias_str} ({direction})")
        print(f"  India VIX     : {vix_str}")
        print(f"  S&P 500 ret   : {self.sp500_ret:+.2f}%")
        print(f"  FII net flow  : {self.fii_net:+,.0f} crores")
        print(f"  Components    : {self.components}")
        if self.skip_day:
            print(f"  *** SKIP DAY: {self.skip_reason} ***")
        elif self.vix_halve:
            print(f"  [VIX HALVE]  VIX={self.vix_level:.1f} > {VIX_HALVE_THRESHOLD} — position size halved to 0.5x")
        if self.vix_high_day and not self.vix_halve:
            print(f"  [VIX ALERT]  High volatility day — +{self.get_extra_trades_allowed()} extra trade allowed")
        if abs(self.bias_score) >= 2:
            opp = 'DOWN' if self.bias_score > 0 else 'UP'
            print(f"  [BIAS ALERT] Strong {direction} bias — {opp} signals need +{abs(self.bias_score)*BIAS_CONF_PENALTY:.0%} extra conf")
        print(f"{'='*60}\n")


# Module-level singleton — created once at startup, used throughout session
_premarket_bias = PreMarketBias()


def get_premarket_bias() -> PreMarketBias:
    return _premarket_bias
