"""Live trading loop (paper, dashboard, and real modes)."""
import os, sys, time, logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
import warnings
warnings.filterwarnings('ignore')

from ..config import (
    HORIZONS, REGIME_NAMES, REGIME_CRISIS, REGIME_RANGING,
    USE_WEBSOCKET, WEBSOCKET_AVAILABLE, DATA_1DAY, DATA_1MIN,
    _training_feature_stats, EXPIRY_CONF_FLOOR, EXPIRY_FORCE_EXIT_MOD, EXPIRY_MAX_HOLD_MINS,
    LOT_SIZE, CONF_MIN,
    MAX_TRADES_PER_SESSION, DAILY_LOSS_LIMIT_RS, DOWN_ONLY_MODE,
    CONSEC_LOSS_COOLDOWN_MINS, ROLLING_WR_WINDOW, ROLLING_WR_MIN,
    VIX_HALVE_THRESHOLD, MID_SESSION_SCORE_FLOOR,
    GAP_DAY_SINGLE_TRADE_PCT,
)
from ..data.loader import load_ohlcv
from ..features.feature_engineering import (
    add_1min_features, add_htf_features, add_daily_features, get_feature_cols,
    add_vix_features, load_vix_data,
    add_options_chain_features, compute_options_chain_features,
    add_calendar_features,
    add_futures_basis_features, load_futures_basis_data,
    add_fii_dii_features, load_fii_dii_data,
    add_global_market_features, load_sp500_data,
    add_pcr_volume_features, compute_pcr_volume_features,
    FEATURE_COLS, FEATURE_LIVE_OK,
)
from ..regimes.hmm_regime import RegimeDetector, RegimeStateMachine, REGIME_UNCERTAIN, intraday_regime_override, compute_session_regime
from ..signals.analysis import build_analysis, detect_micro_regime
from ..execution.risk import KillSwitch, SetupFatigueTracker
from ..signals.signal_generator import generate_signal, SignalState, _signal_state, get_last_block_reason
from ..execution.orders import select_option, display_option_predictions, option_pnl_estimate
from .dashboard import print_live_dashboard
from ..data.websocket import (
    AngelSession, MarketStreamer, fetch_live_candles, fetch_live_htf, sync_historical_buffer,
)
from ..execution.position_manager import PaperTrader
from ..utils.bar_logger import BarLogger
from ..execution.broker import BrokerOrderManager  # Real broker order execution
from ..data.external_data import (
    fetch_heavyweight_returns, fetch_banknifty_spread, fetch_option_chain_ndi,
    fetch_option_ltp, _prefetch_all_symbols,
)
from ..utils.time_utils import calculate_dynamic_stops
from ..utils.live_safety import LiveSafetyManager
from ..utils.trade_logger import TradeLogger
from ..utils.premarket import get_premarket_bias

logger = logging.getLogger(__name__)


def _quick_ml_agreement(row: pd.Series, models: dict, active_features: list) -> tuple:
    """
    Fast pre-check: poll all horizon models and return (agreement, ml_direction).

    agreement    — fraction of weighted votes for dominant direction (0.0–1.0)
    ml_direction — 'UP' or 'DOWN' (dominant direction), '' on error

    Used by KillSwitch CRISIS bypass — runs before generate_signal() so the
    bypass has a fresh value even when no signal was generated the previous bar.
    Returns (0.0, '') on any error (safe fallback = no bypass).
    """
    import numpy as np
    try:
        X_raw = np.array([[float(row.get(f, 0.0)) for f in active_features]])
        first = next(iter(models.values()))
        scaler = first.get('live_scaler')
        X = scaler.transform(X_raw) if scaler is not None else X_raw

        weighted_up = 0.0; weighted_dn = 0.0
        from ..config import HORIZON_WEIGHTS
        pw = dict(_signal_state.perf_weights) if _signal_state.perf_weights else HORIZON_WEIGHTS
        for h, res in models.items():
            mdl = res.get('final_model')
            if mdl is None or not hasattr(mdl, 'predict_proba'):
                continue
            proba = mdl.predict_proba(X)[0]
            if len(proba) != 2:
                continue
            conf = proba[1] if proba[1] > 0.5 else proba[0]
            w = pw.get(h, HORIZON_WEIGHTS.get(h, 0.25))
            if proba[1] > 0.5:
                weighted_up += w * conf
            else:
                weighted_dn += w * conf
        total = weighted_up + weighted_dn
        if total < 1e-9:
            return 0.0, ''
        agreement = max(weighted_up, weighted_dn) / total
        ml_direction = 'UP' if weighted_up >= weighted_dn else 'DOWN'
        return agreement, ml_direction
    except Exception:
        return 0.0, ''


def _live_loop_cleanup(streamer, paper):
    """Called on normal exit AND Ctrl+C to disconnect and flush paper trades."""
    if streamer:
        print("\nDisconnecting WebSocket...")
        streamer.disconnect()
    # Flush in-memory paper trades to CSV immediately so paper_report.py can
    # read them without waiting for next-day end_of_day() reset.
    if paper is not None and paper._trades:
        from datetime import date as _date
        print(f"  Saving {len(paper._trades)} trade(s) to paper_trades CSV...")
        paper.end_of_day(_date.today().strftime('%Y-%m-%d'))


def _assess_day_quality(row, regime_conf: float, vix_level: float, paper_mode: bool) -> str:
    """
    Assess whether today is a good or bad trading day based on live market conditions.
    Called once at bar 20 (~20 min after startup) when indicators have stabilized.

    Returns 'GOOD' or 'BAD' and prints a summary to the console.
    """
    import datetime as dt

    adx_1m   = float(row.get('adx_14',    0))
    adx_5m   = float(row.get('tf5_adx',   adx_1m))
    rsi_1m   = float(row.get('rsi_14',    50))
    rsi_5m   = float(row.get('tf5_rsi',   50))
    bb_width = float(row.get('bb_width',  0))

    issues = []
    good_signals = []

    # 1. Regime confidence
    if regime_conf >= 0.60:
        good_signals.append(f"regime_conf={regime_conf:.2f} (stable)")
    else:
        issues.append(f"regime_conf={regime_conf:.2f} < 0.60 (unstable)")

    # 2. ADX — trend strength
    avg_adx = (adx_1m + adx_5m) / 2
    if avg_adx >= 30:
        good_signals.append(f"ADX avg={avg_adx:.1f} (trending)")
    else:
        issues.append(f"ADX avg={avg_adx:.1f} < 30 (weak/ranging)")

    # 3. RSI — not exhausted
    rsi_exhausted = (rsi_1m > 80 or rsi_1m < 20) and (rsi_5m > 75 or rsi_5m < 25)
    if rsi_exhausted:
        issues.append(f"RSI exhausted: 1m={rsi_1m:.0f} 5m={rsi_5m:.0f} (move likely over)")
    else:
        good_signals.append(f"RSI healthy: 1m={rsi_1m:.0f} 5m={rsi_5m:.0f}")

    # 4. VIX — not extreme
    if vix_level > 25:
        issues.append(f"VIX={vix_level:.1f} EXTREME (chaotic whipsaws)")
    elif vix_level > 20:
        issues.append(f"VIX={vix_level:.1f} HIGH (elevated volatility risk)")
    else:
        good_signals.append(f"VIX={vix_level:.1f} (manageable)")

    # 5. BB squeeze — low volatility, breakout pending
    if bb_width < 0.15:
        issues.append(f"BB squeeze={bb_width:.2f}% (low volatility, no momentum)")

    verdict = 'BAD' if len(issues) >= 2 else 'GOOD'
    mode_advice = "STAY IN PAPER MODE" if verdict == 'BAD' else ("READY FOR LIVE" if not paper_mode else "GOOD DAY — signals likely")

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  DAY QUALITY ASSESSMENT  [{dt.date.today()}]")
    print(f"{sep}")
    print(f"  Verdict : *** {verdict} DAY ***  →  {mode_advice}")
    print(f"  Good    : {', '.join(good_signals) if good_signals else 'none'}")
    print(f"  Issues  : {', '.join(issues) if issues else 'none'}")
    print(f"{sep}\n")

    import logging
    logging.getLogger(__name__).info(
        f"[DayQuality] {verdict}: good=[{'; '.join(good_signals)}] issues=[{'; '.join(issues)}]"
    )
    return verdict


def live_loop(models: dict, regime_det: RegimeDetector, capital: float = 10000,
              paper_mode: bool = False, dashboard_mode: bool = False, verbose: bool = False):
    import datetime as dt
    import numpy as np
    import time

    # Mode labeling
    if dashboard_mode:
        mode_label = "DASHBOARD"
        print(f"\n{mode_label} mode - Signal monitoring only (NO TRADES)")
    elif paper_mode:
        mode_label = "PAPER"
        print(f"\n{mode_label} mode - Paper trading (no real orders)")
    else:
        mode_label = "LIVE"
        print(f"\n{mode_label} mode - REAL TRADING")
    
    print(f"Capital: Rs {capital:,.0f}. Ctrl+C to stop.")
    if verbose:
        print("Verbose mode: Detailed CE/PE calculations will be shown")
    
    # Initialize AngelOne session early (needed for both broker and data fetching)
    session = AngelSession()
    
    # Initialize Core Components
    paper = PaperTrader(capital) if (paper_mode and not dashboard_mode) else None
    broker = None
    if not paper_mode and not dashboard_mode:
        # LIVE MODE - Real broker order execution
        broker = BrokerOrderManager(session, capital)

        # --- Instrument master pre-fetch (avoids 10-15s HTTP fetch at first trade) ---
        try:
            from ..data.websocket import prefetch_instrument_master
            prefetch_instrument_master(session)
        except Exception as _e:
            logger.warning(f"[Live] Instrument master prefetch failed: {_e}")

        # --- Broker position reconciliation: halt if open positions found that
        #     don't match local state (e.g., manual trade, prior crash) ---
        try:
            _api = session.get()
            if _api is not None:
                _pos_resp = _api.position()
                if _pos_resp and _pos_resp.get('status') and _pos_resp.get('data'):
                    _open_broker_pos = [
                        p for p in _pos_resp['data']
                        if int(p.get('netqty', 0)) != 0
                    ]
                    if _open_broker_pos:
                        logger.critical(
                            f"[Live] STARTUP HALT: {len(_open_broker_pos)} open position(s) found "
                            f"on broker that are unknown to local state. "
                            f"Please manually flatten before starting: "
                            f"{[p.get('tradingsymbol') for p in _open_broker_pos]}"
                        )
                        print(f"\n{'='*70}")
                        print("  STARTUP HALT: Open positions found on broker!")
                        print("  Please manually close all positions before starting.")
                        for p in _open_broker_pos:
                            print(f"    {p.get('tradingsymbol')}: netqty={p.get('netqty')}")
                        print(f"{'='*70}\n")
                        sys.exit(1)
                    else:
                        logger.info("[Live] Position check: flat (no open positions) - OK to trade")
                else:
                    logger.warning("[Live] Could not fetch broker positions for reconciliation — proceeding with caution")
        except Exception as _e:
            logger.warning(f"[Live] Broker position reconciliation check failed: {_e} — proceeding")

        print(f"\n{'='*70}")
        print("  [WARN] LIVE TRADING MODE ACTIVE")
        print("  Real orders will be placed with AngelOne")
        print("  Real money at risk")
        print("  Press Ctrl+C to stop")
        print(f"{'='*70}\n")
        time.sleep(3)  # Give user time to abort
    
    df1d_static = load_ohlcv(DATA_1DAY, "daily (static)")

    # Load options chain features (daily, shift-1 merged — no lookahead)
    # NOTE: reads 1300+ CSV files — do this AFTER cold-start to avoid blocking startup
    _options_df_static = None  # populated after cold-start below

    # Pre-load daily-static external data once (avoids CSV re-read every bar)
    _vix_result        = load_vix_data()
    _vix_df_static     = _vix_result[0] if isinstance(_vix_result, tuple) else _vix_result
    _futures_df_static = load_futures_basis_data()
    _fii_df_static     = load_fii_dii_data()
    _sp500_df_static   = load_sp500_data()

    # Pre-check external data files at startup (warn if missing)
    import os as _os
    for _fname, _label in [('nifty_futures_daily.csv', 'Futures basis'),
                            ('fii_dii_flow.csv',        'FII/DII flow'),
                            ('sp500_daily.csv',         'S&P 500')]:
        if not _os.path.exists(_fname):
            logger.warning(f"[Startup] {_label}: {_fname} not found — feature will be 0 (run downloader)")

    # 1. Initial Regime Detection
    if df1d_static is not None:
        regime_series = regime_det.predict(df1d_static)
        current_regime = int(regime_series.iloc[-1] if len(regime_series) else REGIME_RANGING)
    else:
        current_regime = REGIME_RANGING

    print(f"   Current regime: {REGIME_NAMES[current_regime]}")
    ks = KillSwitch(capital, paper_mode=paper_mode)
    ks.notify_regime(current_regime)
    _post_crisis_cooldown = 0   # bars remaining after CRISIS exit
    _crisis_entry_cooldown = 0  # bars to wait before CRISIS bypass can fire
                                # WHY: CRISIS bypass was entering on 1st bar of
                                # TRENDING_UP after CRISIS onset — immediate whipsaws.
                                # Require 3 bars of stable CRISIS before any bypass entry.
    _non_crisis_streak = 0      # consecutive non-CRISIS bars after last CRISIS bar
                                # WHY: intraday_regime_override can flip CRISIS->TRENDING
                                # for a single bar (ADX momentarily crosses threshold),
                                # then HMM reverts to CRISIS the next bar.  Without this
                                # streak counter, each single-bar flip resets
                                # _post_crisis_cooldown to 3 and produces an endless
                                # POST_CRISIS_COOLDOWN cycle that blocks trading all day.
                                # Require 3 consecutive non-CRISIS bars before starting
                                # the cooldown so transient overrides don't interfere.
    _prev_regime = current_regime
    # Per-day entry filters (same as replay)
    _day_open           = None   # first bar close of today
    _fast_losses_today  = 0      # MAX_HOLD_EXIT count today
    _last_exit_bar_time = None   # bar time of last exit (cooldown)
    COOLDOWN_MINS       = 0     # no time-based cooldown — signal gates already filter bad re-entries
    CONF_ESCALATION     = [0.90, 0.92, 0.95]   # CONF_MIN=0.90 baseline; escalates after losses
    _gap_down_day       = False  # True when today opened with gap >= 0.5% vs prev close
    # Daily session controls
    _session_trades_today = 0    # total trades taken this session (cap = MAX_TRADES_PER_SESSION)
    _session_net_pnl      = 0.0  # cumulative net P&L for today (Rs)
    _session_halted       = False  # True after daily Rs loss limit or trade cap hit
    # Consecutive-loss cooldown: block new entries for CONSEC_LOSS_COOLDOWN_MINS after 2 losses.
    _consec_losses        = 0        # consecutive closed losses since last win
    _consec_loss_until    = None     # datetime until which new entries are blocked
    # Rolling win-rate guard: track last ROLLING_WR_WINDOW closed trades.
    # If WR < ROLLING_WR_MIN, switch to observation mode (no new entries).
    _rolling_results      = []       # list of bool (True=win) for last N closed trades
    _observation_mode     = False    # True when rolling WR too low
    # Mid-session double-confirm: after first intraday loss, require higher trend score
    _session_had_loss     = False    # True once any loss is booked today
    # Gap-day single-trade cap: limit to 1 trade when abs(gap_pct) >= GAP_DAY_SINGLE_TRADE_PCT.
    # Apr 20: 3 trades on -2.14% gap day, all 3 lost (-Rs 4,158). Hard cap at 1.
    _is_gap_day           = False    # set on first bar if gap >= threshold
    # 3rd-trade win gate: 3rd entry only allowed when first 2 trades of the day were profitable.
    # Brokerage analysis: 55% of all charges came from 3rd+ trades on losing days.
    _session_trade_results = []      # list of bool (True=win) for each trade taken today

    # Recover cooldown state from today's JSONL so mid-session restarts
    # don't silently bypass the 30-min post-trade cooldown.
    try:
        import json as _json
        from datetime import date as _date
        _jsonl_path = os.path.join(os.path.dirname(__file__), '..', '..', 'logs',
                                   f'trades_{_date.today()}.jsonl')
        if os.path.exists(_jsonl_path):
            _last_exit_ts = None
            _max_hold_count = 0
            with open(_jsonl_path, encoding='utf-8') as _f:
                for _line in _f:
                    try:
                        _rec = _json.loads(_line)
                        if _rec.get('event') == 'EXIT':
                            _last_exit_ts = _rec.get('timestamp_exit')
                            if _rec.get('exit_reason') == 'MAX_HOLD_EXIT':
                                _max_hold_count += 1
                    except Exception:
                        pass
            if _last_exit_ts:
                _last_exit_bar_time = pd.Timestamp(_last_exit_ts)
                logger.info(f"[Restart] Restored last_exit_bar_time={_last_exit_bar_time} from JSONL")
            _fast_losses_today = _max_hold_count
            if _max_hold_count:
                logger.info(f"[Restart] Restored fast_losses_today={_fast_losses_today} from JSONL")
    except Exception as _e:
        logger.warning(f"[Restart] Could not restore cooldown state: {_e}")

    # Safety orchestrator, regime state machine, and trade logger
    safety        = LiveSafetyManager(ks)
    regime_sm     = RegimeStateMachine()
    setup_fatigue = SetupFatigueTracker()
    tl            = TradeLogger()    # rotates automatically at midnight (new instance per day below)
    bar_log       = BarLogger()      # per-minute CSV: logs/bar_log_YYYY-MM-DD.csv
    
    # Initialize WebSocket Streamer
    streamer = None
    if USE_WEBSOCKET and WEBSOCKET_AVAILABLE:
        print("Initializing WebSocket streamer...")
        streamer = session.create_streamer()
        if streamer and streamer.connect():
            print("  [OK] Real-time streaming ACTIVE - Zero lag mode")
            time.sleep(2)  # Let connection stabilize
        else:
            print("  [WARN] WebSocket unavailable - Using polling mode")
            streamer = None
    else:
        print("  [INFO] Running in polling mode (60-second updates)")

    # State Tracking
    htf5_refresh = 0
    htf15_refresh = 0
    live_df5m = None
    live_df15m = None
    _black_swan_checked_dates: set = set()  # replaces per-day function attributes
    import datetime as _dt_init
    last_trade_day = _dt_init.date.today()   # avoid false "new day" reset on restart
    # 1-bar confirmation: store pending signal, execute only when next bar confirms direction
    _pending_signal    = None   # signal waiting for next-bar confirmation
    _pending_signal_close = None  # close price of bar that generated the signal
    # Same-direction loss block: after a loss, block same direction for 10 bars
    _last_loss_direction = None  # 'UP' or 'DOWN'
    _last_loss_bar       = 0     # bar counter when last loss occurred
    _bar_counter         = 0     # increments each bar
    last_htf_recalc = time.time()  # Track when HTF was last fully recalculated
    htf_recalc_needed = False  # Flag to trigger HTF feature recalculation
    _htf_refresh_running = False   # background HTF refresh in progress
    last_candle_fetch = 0.0    # Throttle: only fetch 1-min candles every 5 min
    # Background option LTP state: updated by a daemon thread, read by main loop
    import threading as _threading_ltp
    _ltp_state = {'ce': 0.0, 'pe': 0.0, 'atm': 0, 'running': False, 'ts': 0.0}
    _ltp_lock  = _threading_ltp.Lock()
    
    # Live buffer: keep enough for feature warmup (EMA200 needs 200 bars) plus
    # the 1000-bar feature hydration window, with headroom for HTF merges.
    # 2000 rows ≈ 5.3 trading days — sufficient for all indicators.
    # The original 20000 (2 months) caused unnecessary concat/drop_duplicates/ffill
    # overhead on a 20k-row frame every second; no indicator needs that history.
    MIN_PRELOAD_BARS = 2000

    # v4.0: Cold-Start Sync (Pre-flight data buffer)
    print("v4.0: Executing cold-start sync...")
    df1m_cold, df5m_cold, df15m_cold = sync_historical_buffer(session)
    # One trading day = 375 1-min bars (9:15 AM - 3:30 PM), so threshold is 300
    # Post-cold-start rate-limit guard: cold-start fires 3 API calls (15m + 5m + 1m).
    # The first per-bar fetch_option_ltp() fires on the very next tick.
    # Without a gap, this 4th call hits AB1019. Sleep 10s to clear the burst window.
    print("  [Rate-limit guard] Sleeping 10s before live loop starts...")
    time.sleep(10)

    if df1m_cold is not None and len(df1m_cold) >= 50:
        print("  [OK] Cold-start complete. Using synced buffer.")
        df1m_live = df1m_cold
        live_df5m = df5m_cold if df5m_cold is not None else None
        live_df15m = df15m_cold if df15m_cold is not None else None
    else:
        # Cold-start failed (API rate-limited). Load legacy buffer for indicator
        # warmup ONLY — mark it as stale so the first live fetch replaces it.
        # CRITICAL: Do NOT trade on stale data. The live loop's first candle
        # fetch will replace the stale tail with real today's data.
        print("  [WARN] Cold-start failed. Loading legacy buffer (stale — will sync on first bar)...")
        df1m_live = load_ohlcv(DATA_1MIN, "1-min (live preload)")
        if df1m_live is None or len(df1m_live) < 300:  # EMA200 warmup minimum
            raise RuntimeError("Not enough 1-min history")
        df1m_live = df1m_live.tail(MIN_PRELOAD_BARS).reset_index(drop=True)
        # Force immediate candle fetch on first bar to replace stale data
        last_candle_fetch = 0.0   # already initialised to 0.0, but be explicit

    # Load options chain + PCR volume features after cold-start (reads 1300+ CSVs each)
    print("  Loading options chain features...")
    try:
        _options_df_static = compute_options_chain_features('nifty_options_data')
        print(f"  [OK] Options chain features: {len(_options_df_static)} days")
    except Exception as _e:
        logger.warning(f"[Startup] Options chain features failed: {_e}")
        _options_df_static = None

    print("  Loading PCR volume + ATM IV features...")
    try:
        _pcr_df_static = compute_pcr_volume_features('nifty_options_data')
        print(f"  [OK] PCR volume features: {len(_pcr_df_static)} days")
    except Exception as _e:
        logger.warning(f"[Startup] PCR volume features failed: {_e}")
        _pcr_df_static = None

    # Seed warmup counter so we don't wait 20 minutes when starting mid-session.
    # If the cold-start loaded today's bars, calculate how many minutes have
    # already elapsed since market open (9:15) and pre-fill the warmup counter.
    import datetime as _dt_seed
    _now_seed = _dt_seed.datetime.now()
    _market_open_seed = _now_seed.replace(hour=9, minute=15, second=0, microsecond=0)
    if _now_seed > _market_open_seed:
        _mins_elapsed = int((_now_seed - _market_open_seed).total_seconds() / 60)
        safety.seed_warmup(_mins_elapsed)
        print(f"  [Warmup] Seeded with {_mins_elapsed} min elapsed since market open — warmup skipped.")

    # Pre-market bias analysis (uses VIX, S&P500, FII/DII CSVs — no extra API calls)
    _pm_bias = get_premarket_bias()
    _pm_bias.compute(session=session)
    _pm_bias.print_summary()
    # Apply high-VIX extra trades to scarcity counter
    _signal_state._extra_trades_today = _pm_bias.get_extra_trades_allowed()
    # D1 skip-day: if any hard pre-market rule fires, block all entries today
    _premarket_skip_day = _pm_bias.skip_day
    if _premarket_skip_day:
        logger.warning(f"[PreMarket] SKIP DAY: {_pm_bias.skip_reason} — no entries today")
    # D1 VIX halve: apply 0.5 multiplier to lot sizing when VIX > 22
    _vix_halve_active = _pm_bias.vix_halve

    # Per-bar state — initialised here so they are always defined even when a
    # bar is skipped via `continue` before the assignment point.
    _warmup_blocked  = True    # conservative default until warmup passes
    drift_conf_mult  = 1.0
    drift_killed     = []
    regime_conf      = 0.5
    row              = pd.Series(dtype=float)  # empty sentinel


    while True:
        now = dt.datetime.now()
        hm = now.hour*60 + now.minute
        today_d = now.date()
        _bar_counter += 1
        _bar_trade_event = {}   # reset each bar; filled on ENTRY or EXIT this bar
        _crisis_bypass   = False  # safe default; overwritten below if signal path runs

        # Market Hours Check (India: 09:15 to 15:30)
        if now.weekday() >= 5 or not (9*60+15 <= hm <= 15*60+30):
            print(f"[{now.strftime('%H:%M:%S')}] Market closed. Waiting...")
            time.sleep(60); continue

        # New Day State Reset
        if last_trade_day != today_d:
            if paper is not None and last_trade_day is not None:
                paper.end_of_day(last_trade_day)
            # Log EOD summary before reset
            if last_trade_day is not None:
                tl.log_eod(
                    equity=ks.current_equity,
                    day_pnl=ks.current_equity - ks.day_start_equity,
                    trades=ks._trades_today,
                    safety_status=safety.status_dict(),
                )
            ks.reset_day()
            safety.reset_day()
            regime_sm.reset()
            setup_fatigue.reset_day(today_d)  # Reset setup fatigue tracker
            tl = TradeLogger()   # new file for the new day
            _signal_state.reset_day()
            last_trade_day = today_d
            # Recompute pre-market bias for new day and update extra trades
            _pm_bias.compute(session=session)
            _pm_bias.print_summary()
            _signal_state._extra_trades_today = _pm_bias.get_extra_trades_allowed()
            _premarket_skip_day = _pm_bias.skip_day
            _vix_halve_active   = _pm_bias.vix_halve
            if _premarket_skip_day:
                logger.warning(f"[PreMarket] SKIP DAY: {_pm_bias.skip_reason} — no entries today")
            # Reset per-day entry filters
            _day_open            = None
            _gap_down_day        = False
            _fast_losses_today   = 0
            _last_exit_bar_time  = None
            _pending_signal      = None
            _pending_signal_close = None
            _last_loss_direction = None
            _last_loss_bar       = 0
            _bar_counter         = 0
            _session_trades_today = 0
            _session_net_pnl      = 0.0
            _session_halted       = False
            _consec_losses        = 0
            _consec_loss_until    = None
            # Rolling WR guard resets daily — new day gets a fresh window.
            # Observation mode carries over from prior day only if prior day had < 3 trades;
            # with 3+ trades the window is meaningful and should reset each morning.
            _rolling_results      = []
            _observation_mode     = False
            _session_had_loss     = False
            _is_gap_day           = False
            _session_trade_results = []
            # Print expiry-day banner so operator knows what rules apply today
            _is_expiry_today = (today_d.weekday() == 1)   # Tuesday
            if _is_expiry_today:
                print(f"\n{'='*70}")
                print(f"  [EXPIRY DAY] Today is Tuesday — NIFTY Weekly Expiry")
                print(f"  All zones: full-size, normal stops, entries allowed all day")
                if EXPIRY_FORCE_EXIT_MOD < 999:
                    print(f"  Force-exit all positions by mod={EXPIRY_FORCE_EXIT_MOD} ({9*60+15+EXPIRY_FORCE_EXIT_MOD} min from midnight)")
                print(f"  Confidence floor: {EXPIRY_CONF_FLOOR:.0%}")
                print(f"{'='*70}\n")
                tl.log_safety_event('EXPIRY_DAY', f'Expiry-day rules active: {today_d}')
            else:
                print(f"[{now.strftime('%H:%M:%S')}] New day -- kill-switch + safety + regime SM + signal state reset.")

            # Re-seed warmup after reset_day() so mid-session restarts don't
            # wait 20 min.  reset_day() zeros the counter; re-seed immediately.
            import datetime as _dt_reseed
            _now_rs = _dt_reseed.datetime.now()
            _open_rs = _now_rs.replace(hour=9, minute=15, second=0, microsecond=0)
            if _now_rs > _open_rs:
                _elapsed_rs = int((_now_rs - _open_rs).total_seconds() / 60)
                safety.seed_warmup(_elapsed_rs)
            
            # Edge Case 3: Black Swan Gap Detection
            # Check for extreme opening gap on new day (after first bar is loaded)
            # This will be checked after features are calculated below

        # 2. Refresh HTF (High Timeframe) Data with Indicator Recalculation
        # Run in a background thread so the live loop doesn't block for 4-5s.
        # HTF indicators (5m/15m RSI/ADX) change slowly — 1-bar staleness is fine.
        minutes_since_htf_recalc = (time.time() - last_htf_recalc) / 60.0
        _htf_due = (minutes_since_htf_recalc >= 3.0 or live_df5m is None or live_df15m is None) \
                   and not _htf_refresh_running

        if _htf_due:
            import threading as _threading

            def _htf_refresh_worker():
                nonlocal live_df5m, live_df15m, htf_recalc_needed, last_htf_recalc, _htf_refresh_running
                try:
                    print(f"[{now.strftime('%H:%M:%S')}] Refreshing HTF indicators (background)...")
                    time.sleep(2.5)  # rate-limit guard
                    new5  = fetch_live_htf(session, 'FIVE_MINUTE', 300)
                    new15 = fetch_live_htf(session, 'FIFTEEN_MINUTE', 200)
                    if new5 is not None and len(new5) >= 15:
                        live_df5m = new5
                        htf_recalc_needed = True
                    if new15 is not None and len(new15) >= 15:
                        live_df15m = new15
                        htf_recalc_needed = True
                    last_htf_recalc = time.time()
                    if htf_recalc_needed:
                        print(f"  [OK] HTF refresh complete (background)")
                except Exception as _e:
                    logger.warning(f"[HTF] Background refresh failed: {_e}")
                finally:
                    _htf_refresh_running = False

            _htf_refresh_running = True
            _t = _threading.Thread(target=_htf_refresh_worker, daemon=True)
            _t.start()

        # 3. BRIDGE THE GAP: Fetch 1-min candles periodically (not every bar).
        # WebSocket keeps the latest candle updated in real time (step 4b below).
        # Historical fetch every 5 min normally; force-fetch if last bar is stale.
        seconds_since_fetch = time.time() - last_candle_fetch
        last_bar_age = 0
        if len(df1m_live) > 0 and 'datetime' in df1m_live.columns:
            last_bar_dt = df1m_live['datetime'].iloc[-1]
            if hasattr(last_bar_dt, 'timestamp'):
                last_bar_age = time.time() - last_bar_dt.timestamp()
        force_fetch = last_bar_age > 90  # force if last bar > 90s old (WebSocket lagging)
        if seconds_since_fetch >= 300 or last_candle_fetch == 0.0 or force_fetch:
            new = fetch_live_candles(session, n=500)
            if new is None or new.empty:
                if last_candle_fetch == 0.0:
                    # First fetch ever — cannot proceed without data
                    print(f"[{now.strftime('%H:%M:%S')}] Waiting for API data...")
                    time.sleep(10); continue
                # Subsequent failure — keep running on existing buffer
                logger.warning("1-min fetch failed, continuing on existing buffer")
                new = pd.DataFrame()
            last_candle_fetch = time.time()
        else:
            new = pd.DataFrame()

        # 4. Merge Live with History
        df1m_live = pd.concat([df1m_live, new], ignore_index=True)
        df1m_live = df1m_live.drop_duplicates(subset='datetime', keep='last')
        
        # 4a. Forward-fill HTF features to new rows (if HTF columns exist from cold-start)
        htf_all_cols = [c for c in df1m_live.columns if c.startswith('tf5_') or c.startswith('tf15_')]
        if len(htf_all_cols) > 0:
            for col in htf_all_cols:
                df1m_live[col] = df1m_live[col].ffill().fillna(0.0)
        
        df1m_live = df1m_live.tail(MIN_PRELOAD_BARS).reset_index(drop=True)
        
        # 4b. WebSocket: Update latest candle with real-time price
        if streamer and streamer.connected:
            live_price = streamer.get_latest_price()
            if live_price and live_price > 0 and len(df1m_live) > 0:
                # Update the last candle's close price with live data
                df1m_live.loc[df1m_live.index[-1], 'close'] = live_price
                # Also update high/low if live price exceeds them
                last_high = df1m_live.loc[df1m_live.index[-1], 'high']
                last_low = df1m_live.loc[df1m_live.index[-1], 'low']
                if live_price > last_high:
                    df1m_live.loc[df1m_live.index[-1], 'high'] = live_price
                if live_price < last_low:
                    df1m_live.loc[df1m_live.index[-1], 'low'] = live_price
                # Update datetime to now so latency monitor sees a fresh bar,
                # not a 5-min-old timestamp from the last API fetch.
                df1m_live.loc[df1m_live.index[-1], 'datetime'] = pd.Timestamp(now)

        # 5. FEATURE HYDRATION (The "Waiting" Fix)
        # We process a large enough tail to ensure indicators like RSI(14) and EMA(200) have data
        TAIL_SIZE = 1000 
        recent = df1m_live.tail(TAIL_SIZE).copy()
        
        recent['date'] = recent['datetime'].dt.date
        recent['minute_of_day'] = (recent['datetime'].dt.hour * 60 + recent['datetime'].dt.minute) - (9*60 + 15)
        
        # Recalculate 1-min indicators
        recent = add_1min_features(recent)
        
        # Add Daily Features (includes the Fix for Monday gaps and MergeErrors)
        recent = add_daily_features(recent, df1d_static)
        
        # Add HTF features — always merge from cached HTF data into fresh 'recent'.
        # 'recent' is rebuilt from scratch each bar (add_1min_features returns a new
        # DataFrame), so HTF columns are absent until explicitly merged here.
        # forward-fill in step 4a only helps df1m_live; 'recent' is a new copy.
        if live_df5m is not None and not live_df5m.empty and len(live_df5m) >= 15:
            recent = add_htf_features(recent, live_df5m, 'tf5_', [1,3,6])
        if live_df15m is not None and not live_df15m.empty and len(live_df15m) >= 15:
            recent = add_htf_features(recent, live_df15m, 'tf15_', [1,4])
        if htf_recalc_needed:
            htf_recalc_needed = False
            print(f"  [OK] HTF features recalculated with fresh API data")

        # Add India VIX features (pre-loaded at startup — no CSV read per bar)
        try:
            recent = add_vix_features(recent, vix_df=_vix_df_static)
        except Exception as _e:
            logger.debug(f"[VIX] Feature add failed: {_e}")
        for _vc in ['day_vix', 'day_vix_regime', 'day_vix_chg']:
            if _vc not in recent.columns:
                recent[_vc] = 0.0
        # Fallback: if merge produced 0.0 (date mismatch), inject last known value directly
        if _vix_df_static is not None and not _vix_df_static.empty and recent['day_vix'].iloc[-1] == 0.0:
            _vix_last = _vix_df_static.iloc[-1]
            recent['day_vix']        = recent['day_vix'].replace(0.0, _vix_last.get('day_vix', 0.0))
            recent['day_vix_regime'] = recent['day_vix_regime'].replace(0.0, _vix_last.get('day_vix_regime', 0.0))
            recent['day_vix_chg']    = recent['day_vix_chg'].replace(0.0, _vix_last.get('day_vix_chg', 0.0))

        # Add options chain features (PCR OI, max pain, IV skew — previous day)
        if _options_df_static is not None and not _options_df_static.empty:
            try:
                recent = add_options_chain_features(recent, _options_df_static)
            except Exception as _e:
                logger.debug(f"[OptChain] Feature add failed: {_e}")
        for _oc in ['pcr_oi', 'max_pain_dist', 'iv_skew', 'oi_buildup', 'atm_oi_skew']:
            if _oc not in recent.columns:
                recent[_oc] = 0.0

        # Calendar features (derived from datetime — always available)
        try:
            recent = add_calendar_features(recent)
        except Exception as _e:
            logger.debug(f"[Calendar] Feature add failed: {_e}")
        for _cc in ['day_of_week', 'is_expiry_week', 'is_monday', 'is_friday']:
            if _cc not in recent.columns:
                recent[_cc] = 0.0

        # PCR volume + ATM IV absolute level (pre-loaded at startup)
        try:
            recent = add_pcr_volume_features(recent, pcr_df=_pcr_df_static)
        except Exception as _e:
            logger.debug(f"[PCRVol] Feature add failed: {_e}")
        for _pv in ['pcr_vol', 'pcr_oi_vol_diff', 'atm_iv_ce', 'atm_iv_pe', 'atm_iv_avg']:
            if _pv not in recent.columns:
                recent[_pv] = 0.0

        # Futures basis (pre-loaded at startup — no CSV read per bar)
        try:
            recent = add_futures_basis_features(recent, futures_df=_futures_df_static)
        except Exception as _e:
            logger.debug(f"[Futures] Feature add failed: {_e}")
        for _fb in ['futures_basis', 'futures_basis_chg']:
            if _fb not in recent.columns:
                recent[_fb] = 0.0

        # FII/DII flow (pre-loaded at startup — no CSV read per bar)
        try:
            recent = add_fii_dii_features(recent, fii_df=_fii_df_static)
        except Exception as _e:
            logger.debug(f"[FII] Feature add failed: {_e}")
        for _fi in ['fii_net_buy', 'dii_net_buy', 'fii_dii_net', 'fii_flow_regime', 'fii_5d_cumulative']:
            if _fi not in recent.columns:
                recent[_fi] = 0.0
        # Fallback: inject last known FII values if merge produced 0 (date mismatch)
        if _fii_df_static is not None and not _fii_df_static.empty and recent['fii_net_buy'].iloc[-1] == 0.0:
            _fii_last = _fii_df_static.iloc[-1]
            for _fc in ['fii_net_buy', 'dii_net_buy', 'fii_dii_net', 'fii_flow_regime', 'fii_5d_cumulative']:
                if _fc in _fii_last.index:
                    recent[_fc] = recent[_fc].replace(0.0, float(_fii_last[_fc]))

        # Global market context — S&P 500 (pre-loaded at startup — no CSV read per bar)
        try:
            recent = add_global_market_features(recent, sp500_df=_sp500_df_static)
        except Exception as _e:
            logger.debug(f"[SP500] Feature add failed: {_e}")
        for _gm in ['sp500_ret_1d', 'sp500_ret_5d', 'global_risk_on']:
            if _gm not in recent.columns:
                recent[_gm] = 0.0
        # Fallback: inject last known SP500 values if merge produced 0 (date mismatch)
        if _sp500_df_static is not None and not _sp500_df_static.empty and recent['sp500_ret_1d'].iloc[-1] == 0.0:
            _sp_last = _sp500_df_static.iloc[-1]
            for _sc in ['sp500_ret_1d', 'sp500_ret_5d', 'global_risk_on']:
                if _sc in _sp_last.index:
                    recent[_sc] = recent[_sc].replace(0.0, float(_sp_last[_sc]))

        # Ensure all expected HTF columns exist (initialize with 0.0 if missing)
        htf_cols_all = ([f'tf5_ret_{n}' for n in [1,3,6]] + 
                        ['tf5_rsi', 'tf5_macd_h', 'tf5_bb_pos', 'tf5_atr_pct',
                         'tf5_ema9_21', 'tf5_vol_10', 'tf5_above_vwap',
                         'tf5_adx', 'tf5_cci', 'tf5_stoch_k', 'tf5_willr'] +
                        [f'tf15_ret_{n}' for n in [1,4]] +
                        ['tf15_rsi', 'tf15_macd_h', 'tf15_bb_pos', 'tf15_atr_pct',
                         'tf15_ema9_21', 'tf15_vol_10', 'tf15_above_vwap',
                         'tf15_adx', 'tf15_cci', 'tf15_stoch_k', 'tf15_willr'])
        
        for col in htf_cols_all:
            if col not in recent.columns:
                recent[col] = 0.0

        # Final Feature alignment (Ensure no columns are missing)
        # Use active_features (filtered by FEATURE_LIVE_OK) to match trained models
        active_features = [f for f in FEATURE_COLS if FEATURE_LIVE_OK.get(f, True)]
        for col in active_features:
            if col not in recent.columns:
                recent[col] = 0.0

        # .copy() makes row a standalone Series so subsequent assignments
        # (hdfc_ret_1m, banknifty_spread, etc.) actually persist on the object
        # and are visible to generate_signal() and select_option() downstream.
        row = recent.iloc[-1].copy()

        # -----------------------------------------------------------------------
        # SAFETY GATE A0: Stale data guard — block signals if last bar is from
        # a previous trading day. This prevents trading on legacy buffer data
        # when cold-start failed and today's live candles haven't arrived yet.
        # -----------------------------------------------------------------------
        row_date = row.get('date', None)
        if row_date is not None and str(row_date) != str(today_d):
            logger.warning(f"[Safety] Stale data: last bar date={row_date}, today={today_d}. Waiting for live data...")
            time.sleep(10)
            continue

        # -----------------------------------------------------------------------
        # SAFETY GATE A: Tick warmup counter (must happen every bar, unconditionally)
        # -----------------------------------------------------------------------
        safety.tick_bar()

        # -----------------------------------------------------------------------
        # SAFETY GATE B: Bar validation — drop corrupt candles immediately
        # -----------------------------------------------------------------------
        bar_ok, bar_errors = safety.validate_bar(row)
        if not bar_ok:
            err_str = "; ".join(bar_errors)
            logger.warning(f"[Safety] Bad bar dropped: {err_str}")
            tl.log_safety_event('BAD_BAR', err_str)
            if safety.is_shutdown_requested():
                reason = safety.shutdown_reason()
                tl.log_safety_event('EMERGENCY_SHUTDOWN', reason)
                logger.critical(f"[Safety] EMERGENCY SHUTDOWN: {reason}")
                # Force-exit any open position before recovery attempt
                _last_close = float(row.get('close', 0)) if 'row' in dir() else 0.0
                if paper is not None and paper.in_position:
                    logger.critical("[Safety] Force-exiting paper position before reconnect")
                    paper.force_exit(_last_close, now, ks, signal_state=_signal_state)
                if broker is not None and broker.is_in_position():
                    logger.critical("[Safety] Force-exiting broker position before reconnect")
                    broker.emergency_flatten(reason)

                # ----------------------------------------------------------------
                # Persistent dropout detection: if we've reconnected >= 4 times
                # in the last 30 minutes and it's still failing, the Angel One API
                # itself is lagging (not our network). Switch to degraded mode:
                # sleep 5 minutes, stop trading, keep the process alive.
                # WHY: On Apr 23, 12 consecutive reconnects fired from 13:13-14:26
                # (73 min). Each reconnect fetched fresh data that also had 5-min
                # gaps, so the validator immediately fired again — infinite loop.
                # ----------------------------------------------------------------
                if not hasattr(live_loop, '_reconnect_times'):
                    live_loop._reconnect_times = []
                live_loop._reconnect_times.append(time.time())
                # Prune entries older than 30 min
                _cutoff = time.time() - 1800
                live_loop._reconnect_times = [t for t in live_loop._reconnect_times if t > _cutoff]
                _recent_reconnects = len(live_loop._reconnect_times)

                if _recent_reconnects >= 4:
                    logger.critical(
                        f"[Safety] PERSISTENT DROPOUT: {_recent_reconnects} reconnects in 30min. "
                        f"Angel One API likely lagging. Entering degraded mode — no trades, "
                        f"sleeping 5 min then retrying."
                    )
                    tl.log_safety_event('DEGRADED_MODE', f'{_recent_reconnects} reconnects in 30min — API lag suspected')
                    time.sleep(300)   # 5-min wait — give API time to recover
                    # Reset reconnect counter so we get 4 fresh attempts after recovery
                    live_loop._reconnect_times = []
                    safety.bar_validator.reset()
                    safety._shutdown_requested = False
                    safety._shutdown_reason    = ''
                    last_candle_fetch = 0.0
                    continue

                # Normal reconnect: wait 30s, resync data, resume
                logger.critical("[Safety] Auto-reconnect: waiting 30s then resyncing data...")
                time.sleep(30)
                try:
                    _r1m, _r5m, _r15m = sync_historical_buffer(session)
                    if _r1m is not None and len(_r1m) >= 50:
                        df1m_live  = _r1m
                        live_df5m  = _r5m  if _r5m  is not None else live_df5m
                        live_df15m = _r15m if _r15m is not None else live_df15m
                        logger.critical(f"[Safety] Reconnect OK — {len(df1m_live)} bars reloaded. Resuming.")
                    else:
                        logger.critical("[Safety] Reconnect data fetch returned too few bars — will retry next bar")
                except Exception as _re:
                    logger.critical(f"[Safety] Reconnect resync failed: {_re} — will retry next bar")
                # Reset the bar validator and shutdown flag so the loop can continue
                safety.bar_validator.reset()
                safety._shutdown_requested = False
                safety._shutdown_reason    = ''
                last_candle_fetch = 0.0   # force immediate fresh fetch on next bar
            else:
                time.sleep(10)
            continue

        # -----------------------------------------------------------------------
        # SAFETY GATE C: Auto-flatten poll (KillSwitch may have triggered externally)
        # -----------------------------------------------------------------------
        should_flatten, flatten_reason = safety.check_flatten()
        if should_flatten:
            tl.log_safety_event('AUTO_FLATTEN', flatten_reason)
            logger.critical(f"[Safety] AUTO-FLATTEN triggered: {flatten_reason}")
            if paper is not None and paper.in_position:
                # force_exit needs a numeric exit price — use last known close
                _flat_price = float(row.get('close', 0)) if 'row' in dir() else 0.0
                paper.force_exit(_flat_price, now, ks, signal_state=_signal_state)
                _last_exit_bar_time = now   # Fix 8: activate cooldown after gate-C flatten
            if broker is not None and broker.is_in_position():
                logger.critical("[Safety] Emergency flatten - attempting to close broker position")
                broker.emergency_flatten(flatten_reason)  # Fix: was wrongly passing (close, now)
                _last_exit_bar_time = now   # Fix 8: activate cooldown after gate-C flatten
            break

        # -----------------------------------------------------------------------
        # SAFETY GATE C1: WebSocket staleness check
        # -----------------------------------------------------------------------
        is_stale, stale_reason = safety.check_websocket_staleness(now)
        if is_stale:
            tl.log_safety_event('WEBSOCKET_STALE', stale_reason)
            logger.critical(f"[Safety] {stale_reason}")
            # Emergency flatten all positions
            if paper is not None and paper.in_position:
                _flat_price = float(row.get('close', 0))
                paper.force_exit(_flat_price, now, ks, signal_state=_signal_state)
            if broker is not None and broker.is_in_position():
                logger.critical("[Safety] WebSocket stale - emergency flatten broker position")
                broker.emergency_flatten(stale_reason)
                _last_exit_bar_time = now
            # Reconnect WebSocket instead of exiting — the feed may recover
            logger.critical("[Safety] WebSocket stale — attempting reconnect, not exiting")
            if streamer and streamer.enabled:
                streamer.force_reconnect()
            time.sleep(30)
            safety.bar_validator.reset()
            last_candle_fetch = 0.0
            continue

        # -----------------------------------------------------------------------
        # SAFETY GATE D: Emergency shutdown
        # -----------------------------------------------------------------------
        if safety.is_shutdown_requested():
            reason = safety.shutdown_reason()
            tl.log_safety_event('EMERGENCY_SHUTDOWN', reason)
            logger.critical(f"[Safety] EMERGENCY SHUTDOWN: {reason}")
            _last_close = float(row.get('close', 0)) if row is not None else 0.0
            if paper is not None and paper.in_position:
                logger.critical("[Safety] Force-exiting paper position before shutdown")
                paper.force_exit(_last_close, now, ks, signal_state=_signal_state)
            if broker is not None and broker.is_in_position():
                logger.critical("[Safety] Force-exiting broker position before shutdown")
                broker.emergency_flatten(reason)
            break

        # -----------------------------------------------------------------------
        # SAFETY GATE E: Record latency for this bar
        # -----------------------------------------------------------------------
        bar_close_dt = row.get('datetime', now)
        safety.record_latency(bar_close_dt, dt.datetime.now())
        latency_ok = safety.latency_ok()   # warns if >2000ms; doesn't block by itself

        # 6. Fill NaN values in features before inference
        # This ensures all features are valid numbers before passing to models
        for col in active_features:
            if pd.isna(row.get(col)):
                row[col] = 0.0

        # 6b. Show WebSocket tick stats if streaming
        if streamer and streamer.connected:
            tick_stats = streamer.get_tick_stats()
            if tick_stats and tick_stats['count'] > 10:
                # Only show every 60 seconds to avoid spam
                if int(now.strftime('%S')) == 0:
                    print(f"  [WebSocket] Live: Rs {tick_stats['latest']:,.2f} | "
                          f"Ticks: {tick_stats['count']} | "
                          f"Range: Rs {tick_stats['range']:.2f}")
            
            # Reset OFI counters every minute
            streamer.reset_ofi()
            
            # 2026 Edge: Heartbeat Monitor - Check for zombie WebSocket
            is_market_open = (9*60+15 <= hm <= 15*60+30)
            _hb_mod = hm - (9*60 + 15)  # minutes since market open (same scale as minute_of_day)
            if not streamer.check_heartbeat(market_open=is_market_open,
                                            minute_of_day=_hb_mod):
                # Zombie detected - force reconnect
                streamer.force_reconnect()
        
        # Heavyweight Returns + BankNifty Spread via Angel One (not yfinance)
        # Background prefetch: kicks off a daemon thread so it never blocks the
        # signal path. Results are read from the 120s TTL cache (always instant).
        # Skip prefetch on HTF-refresh bars to avoid concurrent AB1019 burst.
        if not _htf_due:
            _prefetch_all_symbols(session)
        hw_returns = fetch_heavyweight_returns(session=session)
        row['hdfc_ret_1m']    = hw_returns.get('HDFCBANK.NS', 0.0)
        row['reliance_ret_1m']= hw_returns.get('RELIANCE.NS', 0.0)

        bn_spread = fetch_banknifty_spread(session=session)
        row['banknifty_spread']     = bn_spread['spread_pct']
        row['banknifty_divergence'] = bn_spread['divergence']
        if bn_spread['divergence'] == 1:
            print(f"  [Sector Rotation] BankNifty diverging from NIFTY! "
                  f"Spread: {bn_spread['spread_pct']:+.2f}%")
        
        # Option LTP: fire background fetch (non-blocking), use last known value.
        # The API call takes 3-5s — running it async means zero main-loop delay.
        # Stop-loss/exit paths use force_fresh=True so they always get real prices.
        _spot_now = float(row.get('close', 0))
        _atm_now  = int(round(_spot_now / 50) * 50)

        def _ltp_bg_worker(atm):
            try:
                ce = fetch_option_ltp(session, atm, 'CE')
                pe = fetch_option_ltp(session, atm, 'PE')
                with _ltp_lock:
                    _ltp_state['ce'] = ce if ce > 0 else _ltp_state['ce']
                    _ltp_state['pe'] = pe if pe > 0 else _ltp_state['pe']
                    _ltp_state['atm'] = atm
                    _ltp_state['ts']  = time.time()
            except Exception as _e:
                logger.debug(f"[LTP-BG] fetch failed: {_e}")
            finally:
                with _ltp_lock:
                    _ltp_state['running'] = False

        # Kick off a new fetch if: ATM changed, or last fetch > 55s ago, and not already running
        _ltp_age = time.time() - _ltp_state['ts']
        _atm_changed = (_atm_now != _ltp_state['atm'])
        if not _ltp_state['running'] and (_ltp_age > 55 or _atm_changed or _ltp_state['ts'] == 0.0):
            with _ltp_lock:
                _ltp_state['running'] = True
            import threading as _ltp_th
            _ltp_th.Thread(target=_ltp_bg_worker, args=(_atm_now,), daemon=True).start()

        # Inject last known LTP into row immediately (no wait)
        with _ltp_lock:
            _ce_ltp = _ltp_state['ce']
            _pe_ltp = _ltp_state['pe']
        if _ce_ltp > 0:
            row['atm_ce_ltp'] = _ce_ltp
        if _pe_ltp > 0:
            row['atm_pe_ltp'] = _pe_ltp

        # 2026 Alpha: Net Delta Imbalance (NDI) from Option Chain
        # Fetch every 5 minutes (expensive API call)
        # Track last fetch time to avoid rate limit
        if not hasattr(live_loop, '_last_ndi_fetch'):
            live_loop._last_ndi_fetch = 0
            live_loop._last_ndi_value = 0.0
        
        minutes_since_ndi = (time.time() - live_loop._last_ndi_fetch) / 60.0
        if minutes_since_ndi >= 5.0:
            try:
                spot_price = float(row.get('close', 0))
                ndi = fetch_option_chain_ndi(session, spot_price)
                live_loop._last_ndi_value = ndi
                live_loop._last_ndi_fetch = time.time()
                if abs(ndi) > 0.3:
                    signal_type = "Bullish Floor (Put Writing)" if ndi > 0 else "Bearish Ceiling (Call Writing)"
                    print(f"  [NDI] {signal_type}: {ndi:+.2f}")
            except:
                pass
        
        row['option_chain_ndi'] = live_loop._last_ndi_value
        
        # Edge Case 3: Black Swan Gap Detection (on first bar of day)
        # Check if this is first bar of day and gap is extreme
        minute_of_day_current = int(row.get('minute_of_day', 0))
        if minute_of_day_current <= 5 and today_d not in _black_swan_checked_dates:
            gap_pct = abs(float(row.get('gap_pct', 0)))
            day_atr = float(row.get('day_atr_pct', 0.5))
            if gap_pct > 2.5 * day_atr:
                ks.notify_black_swan(gap_pct, day_atr)
            # Gap-down bad-day flag: signed gap < -0.5% marks a bearish-bias open.
            # After 2 losses on such a day we stop trading (checked at entry time).
            signed_gap = float(row.get('gap_pct', 0))
            if signed_gap <= -0.005:
                _gap_down_day = True
                logger.info(f"[Safety] Gap-down day flagged: gap={signed_gap:.3%}")
            # Gap-day single-trade cap: any gap (up or down) >= GAP_DAY_SINGLE_TRADE_PCT
            # limits the session to 1 trade only.
            # Apr 20: gap=-2.14%, 3 trades taken, WR=0%, -Rs 4,158. Model is directionally
            # confused on large gap days — allow the first signal then stop.
            if gap_pct >= GAP_DAY_SINGLE_TRADE_PCT:
                _is_gap_day = True
                logger.info(f"[Safety] Gap day flagged: |gap|={gap_pct:.2f}% >= {GAP_DAY_SINGLE_TRADE_PCT}% — max 1 trade today")
            _black_swan_checked_dates.add(today_d)

        # Expiry-day: log current zone every bar so operator can track it
        _is_expiry_bar = (row.get('is_expiry', 0) == 1)
        if _is_expiry_bar:
            from ..execution.orders import get_expiry_rule as _get_rule
            _rule = _get_rule(True, minute_of_day_current)
            _zone_tag = _rule.get('tag', 'UNKNOWN')
            _allow = _rule['allow_new']
            if not _allow:
                logger.info(f"[Expiry] mod={minute_of_day_current} zone={_zone_tag} "
                            f"— entries BLOCKED this bar")
            else:
                logger.debug(f"[Expiry] mod={minute_of_day_current} zone={_zone_tag} "
                             f"size={_rule['size_mult']:.2f}x stop={_rule['stop_tighten']:.2f}x")

        # -----------------------------------------------------------------------
        # SAFETY GATE F: Market-open warmup (blocks signals in first 15 min / 20 bars)
        # -----------------------------------------------------------------------
        warmup_ok, warmup_reason = safety.check_warmup(minute_of_day_current)
        if not warmup_ok:
            logger.debug(f"[Safety] {warmup_reason}")
            # Still update dashboard but do not trade
            _warmup_blocked = True
        else:
            _warmup_blocked = False

        # -----------------------------------------------------------------------
        # SAFETY GATE G: Feature drift check (OOD detection)
        # Returns confidence multiplier: 1.0=normal, 0.5=degraded, 0.0=kill
        # -----------------------------------------------------------------------
        drift_conf_mult, drift_killed = safety.check_feature_drift(
            row, active_features, _training_feature_stats
        )

        # -----------------------------------------------------------------------
        # REGIME: Update regime through state machine (hysteresis + confidence decay)
        # Use predict_live() (causal sliding-window) NOT predict() (batch Viterbi).
        # -----------------------------------------------------------------------
        if df1d_static is not None:
            raw_regime, raw_conf = regime_det.predict_live(df1d_static.tail(120))
            current_regime, regime_conf = regime_sm.update(raw_regime, raw_conf)
        else:
            current_regime, regime_conf = REGIME_RANGING, 0.5
        
        # -----------------------------------------------------------------------
        # CRITICAL: Apply intraday regime override to correct HMM lag
        # WHY: Daily HMM can lag intraday reality by hours. Override with real-time signals.
        # Rules: IV spike >25% → CRISIS, High ATR + Low ADX → RANGING, First/last 30min → RANGING
        # -----------------------------------------------------------------------
        current_regime = intraday_regime_override(row, current_regime)

        # Post-CRISIS cooldown: block 3 bars after CRISIS clears (whipsaw zone)
        # FIX: only reset counter if it is already 0 — prevents CRISIS flickering
        # (on/off every few bars) from resetting the counter repeatedly and
        # producing a never-ending cooldown that blocks trading for 60+ bars.
        if _prev_regime != REGIME_CRISIS and current_regime == REGIME_CRISIS:
            # CRISIS just started — reset entry cooldown so bypass must wait 3 bars
            _non_crisis_streak = 0
            _crisis_entry_cooldown = 3
            logger.info(f"CRISIS entered -> {_crisis_entry_cooldown}-bar bypass cooldown")
        if current_regime == REGIME_CRISIS and _crisis_entry_cooldown > 0:
            _crisis_entry_cooldown -= 1

        # Track consecutive non-CRISIS bars to detect genuine CRISIS exit.
        # intraday_regime_override can flip a single bar CRISIS->TRENDING (when ADX
        # momentarily crosses the threshold) then HMM reverts the next bar.
        # Each single-bar flip was resetting _post_crisis_cooldown to 3, causing an
        # endless POST_CRISIS_COOLDOWN cycle that blocked trading all day.
        # Fix: require 3 consecutive non-CRISIS bars before starting the cooldown.
        if current_regime == REGIME_CRISIS:
            _non_crisis_streak = 0
        else:
            _non_crisis_streak += 1

        if _prev_regime == REGIME_CRISIS and _non_crisis_streak == 3:
            # Regime has been genuinely non-CRISIS for 3 consecutive bars.
            # Cooldown reduced from 3->1 bar: the 3-consecutive detection already
            # provides a natural buffer. A 3-bar cooldown on top of that blocks
            # 6 bars total (3 detection + 3 cooldown = 6 min) — missing the
            # recovery entry which is often the best trade after a CRISIS spike.
            if _post_crisis_cooldown == 0:
                _post_crisis_cooldown = 1
                logger.info(f"CRISIS genuinely cleared (3-bar streak) -> 1-bar post-CRISIS cooldown")
            ks._consec_trending_bars = 0  # reset bypass counter on CRISIS exit
            _crisis_entry_cooldown = 0
        if _post_crisis_cooldown > 0:
            _post_crisis_cooldown -= 1
        _prev_regime = current_regime

        # -----------------------------------------------------------------------
        # Detect Micro-Regime (Breakout vs Ranging)
        # -----------------------------------------------------------------------
        micro_regime = detect_micro_regime(recent)

        # Kill-switch Check
        # For CRISIS bypass: compute fresh ML agreement directly from models
        # (last_agreement is 0.0 in CRISIS because no signal ever passes gates).
        current_atr = float(row.get('atr_14', 0))
        avg_atr = float(recent['atr_14'].dropna().mean()) if not recent['atr_14'].dropna().empty else 0
        if current_regime == REGIME_CRISIS:
            _ks_agreement, _ks_ml_direction = _quick_ml_agreement(row, models, active_features)
        else:
            _ks_agreement = _signal_state.last_agreement
            _ks_ml_direction = ''
        ks_blocked, ks_reason = ks.check(current_atr, avg_atr, current_regime,
                                        micro_regime=micro_regime,
                                        agreement=_ks_agreement,
                                        minute_of_day=minute_of_day_current,
                                        ml_direction=_ks_ml_direction)

        analysis = build_analysis(row, current_regime)
        signal = None
        trade_info = None

        # Block signals while regime is uncertain (pending confirmation bars).
        # Do NOT block on regime_conf==0.0 alone: conf is always 0 immediately after
        # a regime flip (decay_factor=1-exp(0)=0) which would spuriously block the
        # first bar of every valid new regime including CRISIS->TRENDING transitions.
        regime_uncertain = (current_regime == REGIME_UNCERTAIN)

        # Log non-ks block reasons for gate analysis
        if _warmup_blocked:
            tl.log_signal_blocked(now, warmup_reason, row)
        elif _crisis_entry_cooldown > 0:
            tl.log_signal_blocked(now, f'CRISIS_ENTRY_COOLDOWN: {_crisis_entry_cooldown} bars remaining', row)
        elif _post_crisis_cooldown > 0:
            tl.log_signal_blocked(now, f'POST_CRISIS_COOLDOWN: {_post_crisis_cooldown} bars', row)
        elif regime_uncertain:
            tl.log_signal_blocked(now, 'REGIME_UNCERTAIN', row)
        elif drift_conf_mult == 0.0 and not _crisis_bypass:
            tl.log_signal_blocked(now, f'FEATURE_DRIFT_KILLED: {drift_killed[:3]}', row)

        # In CRISIS bypass, skip drift kill — large moves are inherently OOD vs training
        # but KillSwitch has already validated 85%+ agreement. Drift kill on a 300pt
        # trending day blocks the exact signals the system should be taking.
        _drift_ok = (drift_conf_mult > 0.0) or _crisis_bypass
        # -----------------------------------------------------------------------
        # DAILY SESSION CONTROLS (hard stops before signal generation)
        # -----------------------------------------------------------------------
        # Gap-day single-trade cap: if today is a gap day, max 1 trade.
        # Apr 20: gap=-2.14%, 3 trades, WR=0%, -Rs 4,158 net.
        _effective_session_cap = 1 if _is_gap_day else MAX_TRADES_PER_SESSION
        # 3rd-trade win gate: 3rd trade only allowed when first 2 were both profitable.
        # Brokerage haircust: 55% of total charges came from 3rd+ trades on losing days.
        if (not _is_gap_day and _session_trades_today == MAX_TRADES_PER_SESSION
                and len(_session_trade_results) >= 2
                and all(_session_trade_results[-2:])):
            # First 2 were wins — allow one more trade beyond the cap
            _effective_session_cap = MAX_TRADES_PER_SESSION + 1
            logger.info("[SessionCap] First 2 trades both profitable — allowing 1 bonus trade")

        # Daily trade cap.
        if _session_trades_today >= _effective_session_cap and not _session_halted:
            _session_halted = True
            _cap_reason = f'gap-day cap=1' if _is_gap_day else f'max {_effective_session_cap} trades'
            logger.info(f"[SessionCap] {_session_trades_today} trades taken today — {_cap_reason}. No more entries.")
            tl.log_safety_event('SESSION_CAP', _cap_reason)

        # Daily Rs loss limit: DAILY_LOSS_LIMIT_RS (default Rs 1500).
        # Stops all new entries when cumulative net loss exceeds the limit.
        if _session_net_pnl <= -DAILY_LOSS_LIMIT_RS and not _session_halted:
            _session_halted = True
            logger.info(f"[DailyLossLimit] Net P&L Rs{_session_net_pnl:+.0f} <= -Rs{DAILY_LOSS_LIMIT_RS:.0f} — halting entries for today.")
            tl.log_safety_event('DAILY_LOSS_LIMIT', f'net_pnl={_session_net_pnl:+.0f}')

        # Consecutive-loss cooldown: after 2 straight losses, block new entries for
        # CONSEC_LOSS_COOLDOWN_MINS. Prevents loss-spiral on choppy/trending-against days.
        if _consec_loss_until is not None and now < _consec_loss_until:
            _consec_loss_remaining = int((_consec_loss_until - now).total_seconds() / 60)
            logger.debug(f"[ConseqLoss] Cooldown active — {_consec_loss_remaining} min remaining")

        # Rolling win-rate guard: if last ROLLING_WR_WINDOW closed trades have WR < ROLLING_WR_MIN,
        # enter observation mode. Clear once WR recovers above threshold.
        if len(_rolling_results) >= ROLLING_WR_WINDOW:
            _recent_wr = sum(_rolling_results[-ROLLING_WR_WINDOW:]) / ROLLING_WR_WINDOW
            if _recent_wr < ROLLING_WR_MIN and not _observation_mode:
                _observation_mode = True
                logger.info(f"[RollingWR] WR={_recent_wr:.0%} < {ROLLING_WR_MIN:.0%} over last {ROLLING_WR_WINDOW} trades — observation mode ON")
                tl.log_safety_event('OBSERVATION_MODE', f'rolling_wr={_recent_wr:.2f}')
            elif _recent_wr >= ROLLING_WR_MIN and _observation_mode:
                _observation_mode = False
                logger.info(f"[RollingWR] WR recovered to {_recent_wr:.0%} — observation mode OFF")

        # Session regime gate: require TRENDING_CONFIRMED before generating any signal.
        # This is the single most important filter — all 6 paper winners were in
        # TRENDING_CONFIRMED sessions; most losses were in RANGING/TRENDING_WEAK.
        # Computed every bar; cached in _session_regime for dashboard display.
        _session_regime_label, _session_regime_score = compute_session_regime(row)
        if _session_regime_label != 'TRENDING_CONFIRMED' and not _crisis_bypass:
            logger.debug(f"[SessionRegime] {_session_regime_label} (score={_session_regime_score:.2f}) — waiting for TRENDING_CONFIRMED")

        _consec_loss_blocked = (_consec_loss_until is not None and now < _consec_loss_until)
        if not ks_blocked and not _warmup_blocked and _post_crisis_cooldown == 0 and not regime_uncertain and _drift_ok \
                and not _session_halted and not _consec_loss_blocked and not _observation_mode \
                and not _premarket_skip_day:
            # Issue 5: regime-frequency boost — raises conf floor in range-heavy periods
            regime_boost = ks.regime_conf_boost()
            # CRISIS bypass: active when KillSwitch allowed trading despite CRISIS regime.
            # ks.check() already validated: micro_regime==BREAKOUT + agreement>=0.85.
            # Passing crisis_bypass=True to generate_signal() unlocks Gate 1 (CRISIS block),
            # Gate 0 (cooldown), Gate 6 (temporal lock), staleness penalty, meta-labeler,
            # CRISIS vote floors, and micro-confirmation — all of which would otherwise
            # block the most profitable V-recovery entries.
            _crisis_bypass = (current_regime == REGIME_CRISIS and not ks_blocked
                              and _crisis_entry_cooldown == 0)
            # Escalate confidence floor after consecutive MAX_HOLD_EXIT losses.
            # CONF_ESCALATION = [0.62, 0.75, 0.90]: 0 losses → 0.62 floor,
            # 1 loss → 0.75 floor, 2+ losses → 0.90 floor.
            # extra_conf_floor is the delta above CONF_MIN (0.55).
            _conf_target    = CONF_ESCALATION[min(_fast_losses_today, len(CONF_ESCALATION) - 1)]
            _loss_extra     = max(0.0, _conf_target - CONF_MIN)
            _combined_extra = regime_boost + _loss_extra
            signal = generate_signal(row, models, current_regime,
                                     micro_regime=micro_regime,
                                     signal_state=_signal_state,
                                     extra_conf_floor=_combined_extra,
                                     crisis_bypass=_crisis_bypass,
                                     regime_conf=regime_conf)

            # Day quality assessment — at bar 1 (if started mid-session),
            # bar 20, and every 30 bars thereafter (~every 30 min)
            if _bar_counter == 1 or _bar_counter == 20 or (_bar_counter > 20 and _bar_counter % 30 == 0):
                _assess_day_quality(row, regime_conf, _pm_bias.vix_level, paper_mode)

            # Apply feature-drift confidence penalty to the generated signal.
            # CRISIS bypass: skip drift penalty — CRISIS conditions are inherently OOD
            # vs. training data (trained on TRENDING/RANGING). With 85%+ ML agreement
            # already validated by KillSwitch, the drift penalty would only suppress
            # legitimate crisis-trend signals.
            if signal and drift_conf_mult < 1.0 and not _crisis_bypass:
                original_conf = signal.get('avg_conf', 0.0)
                signal['avg_conf'] = original_conf * drift_conf_mult
                signal['drift_penalty'] = drift_conf_mult
                logger.warning(f"[Safety] Drift penalty applied: conf {original_conf:.2f} -> {signal['avg_conf']:.2f}")
            
            # v4.0: Order Flow Imbalance (OFI) Boost
            # If >80% of recent ticks are buys (price ticking up), boost prediction.
            # Fix 10: OFI boost must not bypass expiry conf floor or apply on expiry/CRISIS day.
            _ofi_allowed = (
                signal is not None
                and streamer and streamer.connected
                and current_regime != REGIME_CRISIS
                and not bool(row.get('is_expiry', 0) == 1)
            )
            if _ofi_allowed:
                ofi_ratio = streamer.get_ofi()
                if ofi_ratio > 0.80 and signal['direction'] == 'UP':
                    # Strong buying pressure confirms upward signal
                    signal['ofi_boost'] = 0.15
                    signal['avg_conf'] = min(0.99, signal['avg_conf'] + 0.15)
                    print(f"  [OFI Boost] Buy pressure {ofi_ratio:.1%} -> "
                          f"Confidence boosted to {signal['avg_conf']:.1%}")
                elif ofi_ratio < 0.20 and signal['direction'] == 'DOWN':
                    # Strong selling pressure confirms downward signal
                    signal['ofi_boost'] = 0.15
                    signal['avg_conf'] = min(0.99, signal['avg_conf'] + 0.15)
                    print(f"  [OFI Boost] Sell pressure {1-ofi_ratio:.1%} -> "
                          f"Confidence boosted to {signal['avg_conf']:.1%}")
            
            # Pre-market bias adjustment
            # Aligned signals get small boost; opposing signals need extra conf
            if signal and not _crisis_bypass:
                _bias_adj = _pm_bias.get_conf_adjustment(signal['direction'])
                if _bias_adj != 0.0:
                    _orig_conf = signal.get('avg_conf', 0.0)
                    signal['avg_conf'] = max(0.0, min(0.99, _orig_conf + _bias_adj))
                    signal['premarket_bias_adj'] = _bias_adj
                    if _bias_adj < 0:
                        logger.info(f"[PreMarket] Bias={_pm_bias.bias_score:+d} penalises "
                                    f"{signal['direction']} signal: conf {_orig_conf:.2f}->{signal['avg_conf']:.2f}")
                    else:
                        logger.debug(f"[PreMarket] Bias={_pm_bias.bias_score:+d} boosts "
                                     f"{signal['direction']} signal: conf {_orig_conf:.2f}->{signal['avg_conf']:.2f}")

            # Filter signal if in regime transition zone
            if signal and ks.in_transition_zone():
                extra_conf = ks.transition_conf_requirement()
                if signal.get('avg_conf', 0) < (signal.get('conf_floor', 0.5) + extra_conf):
                    signal = None # Suppress signal due to low confidence in shock zone

            if signal:
                # v4.0: Calculate Dynamic ATR-based Stops
                entry_price = float(row.get('close', 0))
                atr_current = signal.get('atr_current', current_atr)
                stop_loss, take_profit = calculate_dynamic_stops(
                    entry_price, atr_current, signal['direction']
                )
                signal['dynamic_stop_loss'] = stop_loss
                signal['dynamic_take_profit'] = take_profit
                signal['stop_type'] = 'DYNAMIC_ATR_2.5X'
                
                # Display clean option predictions for all 4 horizons
                display_option_predictions(signal, verbose=verbose)

                # Dashboard mode: Show predictions only, skip trade execution
                if dashboard_mode:
                    print("\n[Dashboard Mode] Signal displayed - No trade execution")
                    continue

                # Lock in iv_proxy from signal bar so select_option uses correct IV
                signal['iv_proxy'] = float(row.get('iv_proxy', signal.get('iv_proxy', 0.06)))

                # Inject real Angel One LTPs into signal so select_option() uses
                # live market price instead of BS estimate for entry price calculation.
                # atm_ce_ltp / atm_pe_ltp are fetched from the live option chain each bar.
                _real_ce = float(row.get('atm_ce_ltp', 0.0))
                _real_pe = float(row.get('atm_pe_ltp', 0.0))
                if _real_ce > 0:
                    signal['ce_ltp_api'] = _real_ce
                if _real_pe > 0:
                    signal['pe_ltp_api'] = _real_pe

                tick_buffer = streamer.tick_buffer if (streamer and streamer.connected) else None
                signal['minute_of_day'] = now.hour * 60 + now.minute - (9 * 60 + 15)
                signal['spot'] = float(row.get('close', 0))
                _mod_now = signal['minute_of_day']
                _sp_now  = float(row.get('session_pct', 0))

                # Capture day open price — use the actual 09:15 bar, not the restart bar.
                # Mid-session restarts would otherwise anchor _day_open to ~12:00 price,
                # making the intraday trend filter useless on gap-down days.
                if _day_open is None:
                    today_open_bars = df1m_live[df1m_live['datetime'].dt.date == today_d]
                    if len(today_open_bars) > 0:
                        _day_open = float(today_open_bars.iloc[0]['close'])
                    else:
                        _day_open = float(row.get('close', 0))

                # No time-based cooldown — signal gates (conf, agreement, regime,
                # micro-confirmation) already filter bad re-entries. A time block
                # only prevents taking valid signals after a good exit.
                _entry_blocked = False
                _mod_now = int(row.get('minute_of_day', 0))

                # Session regime gate: require TRENDING_CONFIRMED before entry.
                # compute_session_regime() was called above and result cached.
                # Crisis bypass skips this — V-recovery fires regardless of session regime.
                if not _entry_blocked and not _crisis_bypass:
                    if _session_regime_label != 'TRENDING_CONFIRMED':
                        logger.info(f"[SessionRegime] ENTRY BLOCKED: {_session_regime_label} "
                                    f"score={_session_regime_score:.2f} — need TRENDING_CONFIRMED")
                        tl.log_signal_blocked(now,
                            f'SESSION_REGIME: {_session_regime_label} score={_session_regime_score:.2f}',
                            row)
                        _entry_blocked = True
                    # Mid-session double-confirm (D3): after first intraday loss,
                    # require near-perfect session regime score (3.5/4 = 87.5%).
                    # Normal threshold is TRENDING_CONFIRMED (3/4 = 75%).
                    # Evidence: Apr 20-24 second/third trades after a loss compounded losses.
                    elif _session_had_loss and _session_regime_score < MID_SESSION_SCORE_FLOOR:
                        logger.info(f"[MidSession] ENTRY BLOCKED: had loss today, score={_session_regime_score:.2f} "
                                    f"< {MID_SESSION_SCORE_FLOOR:.3f} — need near-perfect trend for re-entry")
                        tl.log_signal_blocked(now,
                            f'MID_SESSION_DOUBLE_CONFIRM: score={_session_regime_score:.2f} < {MID_SESSION_SCORE_FLOOR:.3f}',
                            row)
                        _entry_blocked = True

                # Spot vs day-open filter (first 2 hours only, non-CRISIS-bypass).
                # On Mar 23: spot opened -0.56% down, models signalled UP all morning —
                # all 13 UP signals would have lost. The open price is the single best
                # intraday reference for "has the market accepted direction yet?".
                # After 11:15 (mod>120) the open is stale — reversal trades get blocked.
                # CRISIS bypass exempt: V-recovery needs to fire against gap direction.
                if not _entry_blocked and not _crisis_bypass and _day_open and _mod_now <= 120:
                    sig_dir      = signal.get('direction', 'UP')
                    spot_now_val = float(row.get('close', _day_open))
                    vs_open      = (spot_now_val - _day_open) / _day_open
                    if sig_dir == 'UP' and vs_open < -0.002:
                        logger.info(f"[Order] BLOCKED: spot {vs_open:.3%} below open — no CE buy while market below open")
                        _entry_blocked = True
                    elif sig_dir == 'DOWN' and vs_open > 0.002:
                        logger.info(f"[Order] BLOCKED: spot {vs_open:.3%} above open — no PE buy while market above open")
                        _entry_blocked = True

                # Affordability check: ATM premium > 70% of per-unit capital means
                # we'd need 4+ OTM strikes to afford a lot — delta < 0.25, not worth trading.
                # This is new information (real-time capital state) not available to signal_generator.
                if not _entry_blocked:
                    _atm_ce = float(row.get('atm_ce_ltp', 0))
                    _atm_pe = float(row.get('atm_pe_ltp', 0))
                    _atm_prem = max(_atm_ce, _atm_pe)
                    _max_affordable = capital / LOT_SIZE
                    if _atm_prem > 0 and _atm_prem > _max_affordable * 0.70:
                        logger.info(f"[Order] BLOCKED: ATM premium Rs{_atm_prem:.0f} > 70% of affordable Rs{_max_affordable:.0f} — OTM delta too low")
                        _entry_blocked = True

                # Same-direction loss block: after a loss, block same direction for 10 bars.
                # ~10 min pause lets the 1m model stop thrashing after a wrong-direction loss.
                # Applies to all modes including CRISIS bypass.
                if not _entry_blocked:
                    sig_dir = signal.get('direction', 'UP')
                    if _last_loss_direction == sig_dir and (_bar_counter - _last_loss_bar) < 10:
                        bars_left = 10 - (_bar_counter - _last_loss_bar)
                        logger.info(f"[Order] BLOCKED: same-direction loss block dir={sig_dir} ({bars_left} bars remaining)")
                        _entry_blocked = True

                # CRISIS directional tape filter: in CRISIS regime, require that
                # the 15m momentum and tick_imbalance agree with the signal direction.
                # WHY: CRISIS UP entries on Apr 23 had ret15m=-0.09% and tick_imbalance=-0.2
                # (both bearish) yet model fired UP with conf=0.92 — immediate loss.
                # In CRISIS, model is inherently in OOD territory; tape confirmation is essential.
                if not _entry_blocked and current_regime == REGIME_CRISIS and not _crisis_bypass:
                    sig_dir = signal.get('direction', 'UP')
                    _tf15_ret = float(row.get('tf15_ret_1', 0.0))
                    _tick_imb = float(row.get('tick_imbalance', 0.0))
                    _tape_bear = _tf15_ret < -0.03 and _tick_imb < 0.0
                    _tape_bull = _tf15_ret > 0.03 and _tick_imb > 0.0
                    if sig_dir == 'UP' and _tape_bear:
                        logger.info(f"[Order] CRISIS UP blocked: tape bearish tf15_ret={_tf15_ret:.3f} tick_imb={_tick_imb:.2f}")
                        _entry_blocked = True
                    elif sig_dir == 'DOWN' and _tape_bull:
                        logger.info(f"[Order] CRISIS DOWN blocked: tape bullish tf15_ret={_tf15_ret:.3f} tick_imb={_tick_imb:.2f}")
                        _entry_blocked = True

                # Gap-down bad-day detector: on a day that opened with gap >= -0.5%
                # and we've already taken 2+ losses, stop all new entries for the day.
                # WHY: gap-down days have persistent bearish order-flow; models trained on
                # balanced data overestimate UP probability in such sessions.
                # CRISIS bypass exempt: V-recovery specifically fires against gap direction.
                if not _entry_blocked and not _crisis_bypass:
                    if _gap_down_day and _fast_losses_today >= 2:
                        logger.info(f"[Order] BLOCKED: gap-down bad-day + {_fast_losses_today} losses — halting new entries")
                        _entry_blocked = True

                # DD multiplier: scale down entry confidence requirement as daily loss grows.
                # Uses KillSwitch.get_position_size_multiplier() — at 70% of daily DD limit
                # it returns 0.5, which here raises the required confidence floor proportionally.
                # At multiplier < 1.0, only very high-confidence signals (>= 0.80) are taken.
                if not _entry_blocked:
                    _dd_mult = ks.get_position_size_multiplier()
                    if _dd_mult < 1.0:
                        _dd_conf_floor = 0.62 + (1.0 - _dd_mult) * 0.28   # 0.62 → 0.90 as DD worsens
                        if signal.get('avg_conf', 0.0) < _dd_conf_floor:
                            logger.info(f"[Order] BLOCKED: DD multiplier {_dd_mult:.2f} requires conf>={_dd_conf_floor:.2f}, got {signal.get('avg_conf',0):.2f}")
                            _entry_blocked = True

                if _sp_now > 0.92 or _mod_now >= 340:
                    logger.info(f"[Order] Entry blocked: session_pct={_sp_now:.3f}, minute={_mod_now}")
                    _pending_signal = None  # discard pending — session closing
                elif _entry_blocked:
                    pass  # already logged above
                elif not safety.can_place_order(signal):
                    logger.warning(f"[Safety] Duplicate order suppressed for {signal.get('direction')}")
                else:
                    # 1-bar confirmation: store signal, execute on next bar only if price
                    # moved in signal direction. Filters "entry at local top/bottom" problem
                    # where the option immediately moves against entry in first 3 minutes.
                    # CRISIS bypass + high confidence (>=0.90): skip delay — 4/4 models
                    # agreeing in CRISIS is the strongest possible signal; 1-bar delay only
                    # risks hitting the expiry zone boundary or missing the move.
                    current_close = float(row.get('close', 0))
                    _skip_confirmation = _crisis_bypass and signal.get('avg_conf', 0.0) >= 0.90
                    if _skip_confirmation:
                        logger.info(f"[Confirm] CRISIS bypass + conf={signal.get('avg_conf',0):.2f} >= 0.90 — executing immediately (no 1-bar delay)")
                        _pending_signal = None
                        _pending_signal_close = None
                    elif _pending_signal is None:
                        # New signal: store and wait for next bar to confirm
                        _pending_signal = signal
                        _pending_signal_close = current_close
                        logger.info(f"[Confirm] Signal stored ({signal['direction']}) at {current_close:.2f} — waiting next bar")
                        trade_info = None  # don't execute yet
                    else:
                        # Check if this bar confirms the pending signal direction.
                        # Three conditions ALL required (analysis E6):
                        #   1. Price moved in signal direction (bar-to-bar close)
                        #   2. NIFTY still on correct side of VWAP
                        #   3. Option premium hasn't gone stale (>2% adverse move since trigger)
                        pend_dir  = _pending_signal.get('direction', 'UP')
                        vwap_dist = float(row.get('vwap_dist', 0))
                        # Condition 1: price direction
                        price_ok = ((pend_dir == 'UP'   and current_close > _pending_signal_close) or
                                    (pend_dir == 'DOWN'  and current_close < _pending_signal_close))
                        # Condition 2: VWAP side (DOWN = below VWAP, UP = above VWAP)
                        # vwap_dist = (close - vwap) / vwap; positive means above VWAP
                        if pend_dir == 'DOWN':
                            vwap_ok = vwap_dist <= 0.001   # below or at VWAP
                        else:
                            vwap_ok = vwap_dist >= -0.001  # above or at VWAP
                        # Condition 3: option premium not stale (>2% adverse from signal bar)
                        _pend_opt   = 'CE' if pend_dir == 'UP' else 'PE'
                        _pend_key   = f'{"ce" if pend_dir=="UP" else "pe"}_ltp_current'
                        _signal_ltp = float(_pending_signal.get(_pend_key, 0))
                        _curr_ltp   = float(row.get(f'atm_{"ce" if pend_dir=="UP" else "pe"}_ltp', 0))
                        if _signal_ltp > 0 and _curr_ltp > 0:
                            _prem_move = (_curr_ltp - _signal_ltp) / _signal_ltp
                            prem_ok = _prem_move > -0.02   # premium fell <2% — still enterable
                        else:
                            prem_ok = True   # no LTP data → optimistic default

                        confirmed = price_ok and vwap_ok and prem_ok
                        if confirmed:
                            logger.info(f"[Confirm] CONFIRMED {pend_dir}: close {_pending_signal_close:.2f}->{current_close:.2f} "
                                        f"vwap_dist={vwap_dist:+.4f} prem_move={_prem_move if _signal_ltp > 0 else 0:+.2%}")
                            signal = _pending_signal  # use pending signal for entry
                            _pending_signal = None
                            _pending_signal_close = None
                        else:
                            _reasons = []
                            if not price_ok: _reasons.append(f"price {_pending_signal_close:.2f}->{current_close:.2f}")
                            if not vwap_ok:  _reasons.append(f"VWAP side vwap_dist={vwap_dist:+.4f}")
                            if not prem_ok:  _reasons.append(f"premium stale {_prem_move:+.2%}")
                            logger.info(f"[Confirm] REJECTED {pend_dir}: {'; '.join(_reasons)} — discarded")
                            _pending_signal = _pending_signal_close = None
                            trade_info = None  # don't execute

                    if _pending_signal is not None:
                        trade_info = None  # still waiting for confirmation
                    else:
                        # Fresh LTP fetch at entry moment — bypass 55s cache so
                        # select_option() uses the actual current market price.
                        if signal:
                            _entry_atm = int(round(float(row.get('close', 0)) / 50) * 50)
                            _entry_dir = signal.get('direction', 'UP')
                            _entry_opt = 'CE' if _entry_dir == 'UP' else 'PE'
                            _fresh_ltp = fetch_option_ltp(session, _entry_atm, _entry_opt, force_fresh=True)
                            if _fresh_ltp > 0:
                                signal['ce_ltp_api' if _entry_opt == 'CE' else 'pe_ltp_api'] = _fresh_ltp
                                logger.info(f"[Entry] Fresh LTP {_entry_opt} {_entry_atm}: Rs {_fresh_ltp:.2f}")
                        # Risk-based position sizing:
                        # 1. Gap day (|gap|>=1.5%): force 1L — model confused on gap days
                        #    Apr 20: 3L on -2.14% gap day → -Rs 4,158.
                        # 2. VIX halve (VIX>22): cap at 50% capital → at most 1L.
                        # 3. Weekly DD >5%: halve capital (from check_drawdown_protection).
                        # 4. Confidence scaling: multiplier from get_conf_size_multiplier().
                        # 5. Normal day: full capital, capped by MAX_CONTRACTS in config.
                        # These rules are multiplicative and independent — the most conservative
                        # condition dominates (e.g. gap-day + low-conf = 1L × 0.5 = still 1L).
                        ks.update_period_equity()
                        _monthly_halt, _monthly_reason, _weekly_size_mult = ks.check_drawdown_protection()
                        if _monthly_halt:
                            logger.critical(f"[KillSwitch] {_monthly_reason}")
                            _entry_blocked = True

                        if not _entry_blocked:
                            _conf_val       = signal.get('avg_conf', 0.55)
                            _conf_mult      = ks.get_conf_size_multiplier(_conf_val)
                            _dd_mult        = ks.get_position_size_multiplier()   # daily DD ramp
                            _size_mult_net  = _weekly_size_mult

                            if _is_gap_day:
                                _atm_prem_now = max(float(row.get('atm_ce_ltp', 0)), float(row.get('atm_pe_ltp', 0)))
                                if _atm_prem_now <= 0:
                                    _atm_prem_now = 200.0
                                _trade_capital = _atm_prem_now * 65 * 1.05  # 1 lot + 5% buffer
                                logger.info(f"[GapDay] Forcing 1L sizing: capital={_trade_capital:.0f}")
                            elif _vix_halve_active:
                                _trade_capital = capital * 0.5 * _size_mult_net
                                logger.info(f"[VIXHalve] VIX={_pm_bias.vix_level:.1f} > {VIX_HALVE_THRESHOLD} "
                                            f"— capital={_trade_capital:.0f} (halved × conf={_conf_mult:.2f} × weekly={_weekly_size_mult:.2f})")
                            else:
                                _trade_capital = capital * _size_mult_net

                            logger.info(f"[Sizing] conf={_conf_val:.3f} conf_mult={_conf_mult:.2f} "
                                        f"weekly_mult={_weekly_size_mult:.2f} dd_mult={_dd_mult:.2f} "
                                        f"→ capital=Rs{_trade_capital:.0f} ({_size_mult_net:.0%} of Rs{capital:.0f})")

                        trade_info = select_option(signal, _trade_capital, now=now, tick_buffer=tick_buffer, session=session, position_mgr=ks, crisis_bypass=_crisis_bypass) if (signal and not _entry_blocked) else None
                    if trade_info:
                        # Execute order based on mode
                        if paper is not None and not paper.in_position:
                            # Paper mode - simulated trading
                            paper.enter(signal, trade_info, now)
                            _session_trades_today += 1
                            safety.register_order(signal, order_id=str(now.timestamp()))
                            _signal_state.record_trade_taken(signal.get('direction', 'UP'))
                            tl.log_entry(
                                signal=signal, trade_info=trade_info,
                                row=row, regime=current_regime, regime_conf=regime_conf,
                                latency_ms=safety.latency_mon.median_lag_ms(),
                                active_features=active_features,
                                now=now,
                                session_regime=_session_regime_label,
                                session_regime_score=_session_regime_score,
                            )
                            # Store trade number in position dict so log_bar/log_exit
                            # can attribute BAR events to the correct trade even when
                            # multiple positions are open simultaneously.
                            paper._position['_tl_trade_num'] = tl._trade_num
                            _bar_trade_event = {'event': 'ENTRY'}
                        elif broker is not None and not broker.is_in_position():
                            # LIVE mode - real broker order
                            order_result = broker.place_entry_order(
                                trade_info, signal, now,
                                is_expiry=bool(row.get('is_expiry', 0) == 1),
                            )
                            if order_result:
                                logger.critical(f"[LIVE] Real order placed: {order_result['symbol']}")
                                safety.register_order(signal, order_id=order_result['order_id'])
                                _signal_state.record_trade_taken(signal.get('direction', 'UP'))
                                tl.log_entry(
                                    signal=signal, trade_info=trade_info,
                                    row=row, regime=current_regime, regime_conf=regime_conf,
                                    latency_ms=safety.latency_mon.median_lag_ms(),
                                    active_features=active_features,
                                    now=now,
                                    session_regime=_session_regime_label,
                                    session_regime_score=_session_regime_score,
                                )
                            else:
                                logger.error("[LIVE] Real order FAILED - continuing without position")

        # 8. Mark-to-Market (Paper Tracking or Live Position Monitoring)
        # Skip MTM on the same bar as entry — the position was just opened and
        # the bar's ltp_est was computed before entry. Running track() immediately
        # would compare an already-computed (possibly stale/high) ltp against a
        # freshly set target, causing instant TARGET exits with 0-min hold time.
        if paper is not None and paper.in_position:
            pos = paper._position
            if pos.get('entry_time') == now:
                # Entry bar — skip MTM, let next bar do the first real check.
                # The bar's ltp_est was computed before entry; running track() here
                # would trigger instant TARGET/STOP exits with 0-min hold time.
                spot_now = float(row.get('close', 0))
            else:
                spot_now = float(row.get('close', 0))
                # Use REAL option LTP from Angel One API — fetched at the POSITION STRIKE,
                # not the rolling ATM. As spot drifts, the ATM changes; we must track
                # the specific contract we own (e.g. 24600 PE, not rolling ATM PE).
                opt_type     = pos.get('option_type', 'CE')
                pos_strike   = int(pos.get('strike', 0))
                real_ltp = 0.0
                if pos_strike > 0:
                    real_ltp = fetch_option_ltp(session, pos_strike, opt_type, force_fresh=True)

                # BS fallback: use entry IV (stable intraday) + position strike + trading DTE.
                # Bar-by-bar iv_proxy (ATR-derived) collapses 30-40% in quiet windows, causing
                # BS to severely underprice the option and trigger false stop-outs.
                # option_pnl_estimate (delta-proxy) is also unreliable for >20min holds.
                from nifty_trader.execution.orders import estimate_option_premium as _eop, _next_expiry_mins as _nem
                iv_ann_entry = float(pos.get('iv_annpct_entry', 0.0))
                if iv_ann_entry <= 0:
                    iv_proxy_val = float(row.get('iv_proxy', 0.06))
                    iv_ann_entry = iv_proxy_val * (252 ** 0.5) if iv_proxy_val > 0 else 15.0
                iv_ann_entry = max(iv_ann_entry, 15.0)   # floor at 15% — NIFTY IV rarely below ~12%
                dte_now_trading = _nem(now)
                ltp_model = _eop(spot_now, iv_ann_entry, dte_now_trading,
                                 strike=float(pos_strike) if pos_strike > 0 else 0.0,
                                 option_type=opt_type)

                if real_ltp > 0:
                    ltp_est = real_ltp   # Live: use real market price
                    pos['_using_real_ltp'] = True
                else:
                    ltp_est = ltp_model  # Fallback: BS at entry IV + position strike
                    pos['_using_real_ltp'] = False

                # Running MAE/MFE tracking for the trade logger.
                # We always BUY the option (CE for UP, PE for DOWN), so PnL is
                # always ltp_est - entry_price regardless of trade direction.
                _ep    = pos['entry_price']
                _move  = ltp_est - _ep   # positive = premium has risen = favorable
                _mae   = min(0.0, _move)   # negative = adverse (premium fell)
                _mfe   = max(0.0, _move)   # positive = favorable (premium rose)
                _pos_tl_num = pos.get('_tl_trade_num', tl._trade_num)
                tl.log_bar(now, spot_now, ltp_est, _mae, _mfe, trade_num=_pos_tl_num)

                _was_in_position = paper.in_position
                # Pass signal_state for exit time tracking (Edge Case 2)
                paper.track(ltp_est, now, ks, current_row=row, signal_state=_signal_state)

                # Detect exit: position was open, now it's closed
                if _was_in_position and not paper.in_position:
                    # Retrieve exit info from paper's last completed trade
                    last_t = paper._trades[-1] if paper._trades else {}
                    tl.log_exit(
                        now=now,
                        exit_price=float(last_t.get('exit_price', ltp_est)),
                        exit_reason=last_t.get('exit_reason', 'UNKNOWN'),
                        entry_price=float(last_t.get('entry_price', _ep)),
                        contracts=int(last_t.get('contracts', 0)),
                        trade_num=_pos_tl_num,
                    )
                    _exit_px  = float(last_t.get('exit_price', ltp_est))
                    _entr_px  = float(last_t.get('entry_price', _ep))
                    _exit_pnl = (_exit_px - _entr_px) * int(last_t.get('contracts', 1)) * 65
                    _exit_pct = (_exit_px - _entr_px) / _entr_px * 100 if _entr_px > 0 else 0
                    _bar_trade_event = {
                        'event': 'EXIT',
                        'exit_reason': last_t.get('exit_reason', 'UNKNOWN'),
                        'pnl_rs':  round(_exit_pnl, 2),
                        'pnl_pct': round(_exit_pct, 4),
                    }
                    # Update per-day entry filter counters
                    _last_exit_bar_time = now
                    if last_t.get('exit_reason') == 'MAX_HOLD_EXIT':
                        _fast_losses_today += 1
                    # Track loss direction for same-direction re-entry block
                    # NOTE: trade dict key is 'pnl' (not 'pnl_pts' which doesn't exist)
                    _exit_pnl = float(last_t.get('pnl', last_t.get('pnl_rs', 0)))
                    _session_net_pnl += _exit_pnl
                    # Rolling WR tracking — append result after each closed trade
                    _is_win = _exit_pnl > 0
                    _rolling_results.append(_is_win)
                    if len(_rolling_results) > ROLLING_WR_WINDOW * 2:
                        _rolling_results = _rolling_results[-ROLLING_WR_WINDOW * 2:]
                    # 3rd-trade win gate: track per-session trade results
                    _session_trade_results.append(_is_win)
                    if _exit_pnl < 0:
                        _last_loss_direction = last_t.get('direction', None)
                        _last_loss_bar = _bar_counter
                        logger.info(f"[LossBlock] Loss recorded dir={_last_loss_direction} bar={_bar_counter}")
                        # Consecutive-loss cooldown
                        _consec_losses += 1
                        if _consec_losses >= 2:
                            _consec_loss_until = now + pd.Timedelta(minutes=CONSEC_LOSS_COOLDOWN_MINS)
                            logger.info(f"[ConseqLoss] {_consec_losses} consecutive losses — blocking entries until {_consec_loss_until.strftime('%H:%M')}")
                            tl.log_safety_event('CONSEC_LOSS_COOLDOWN', f'losses={_consec_losses} until={_consec_loss_until.isoformat()}')
                    else:
                        # Win resets the consecutive-loss counter
                        _consec_losses = 0
                    # Mid-session double-confirm tracking
                    if _exit_pnl < 0:
                        _session_had_loss = True

        # LIVE MODE: Monitor broker position and handle exits
        elif broker is not None and broker.is_in_position():
            pos = broker.get_position()
            spot_now = float(row.get('close', 0))

            # Skip MTM on the same bar as entry — mirrors paper mode fix.
            # Entry order takes ~30s to confirm fill; comparing the fresh real_ltp
            # to the just-set target on the same bar can cause instant spurious exits.
            if pos.get('entry_time') == now:
                continue  # entry bar — let next bar do the first real MTM check

            # Fetch real option LTP from market
            opt_type = pos['option_type']
            pos_strike = int(pos['strike'])
            real_ltp = fetch_option_ltp(session, pos_strike, opt_type, force_fresh=True)
            
            if real_ltp <= 0:
                # Fallback to BS model if API fails.
                # Use entry IV (stable) not bar-by-bar iv_proxy which collapses 30-40%
                # in quiet windows and causes false stop-outs (same fix as paper path).
                from nifty_trader.execution.orders import estimate_option_premium as _eop, _next_expiry_mins as _nem
                iv_ann_entry = float(pos.get('iv_annpct_entry', 0.0))
                if iv_ann_entry <= 0:
                    iv_proxy_val = float(row.get('iv_proxy', 0.06))
                    iv_ann_entry = iv_proxy_val * (252 ** 0.5) if iv_proxy_val > 0 else 15.0
                iv_ann_entry = max(iv_ann_entry, 15.0)  # floor at 15% — NIFTY IV rarely below ~12%
                dte_now = _nem(now)
                real_ltp = _eop(spot_now, iv_ann_entry, dte_now, float(pos_strike), opt_type)
            
            # Update peak LTP for trailing stop calculation
            pos['peak_ltp'] = max(pos.get('peak_ltp', pos['entry_price']), real_ltp)
            peak_ltp = pos['peak_ltp']
            
            _ep_live = pos['entry_price']
            _move_live = real_ltp - _ep_live
            tl.log_bar(now, spot_now, real_ltp, min(0.0, _move_live), max(0.0, _move_live))
            
            # Check exit conditions
            should_exit = False
            exit_reason = ''
            
            # Calculate time-based metrics
            hm = now.hour * 60 + now.minute
            minute_of_day = hm - (9 * 60 + 15)
            is_expiry_live = bool(pos.get('is_expiry', False))

            # Expiry-day: force-exit all positions by 14:40 (mod=310)
            # Same logic as paper trader — gamma extreme after this time
            if is_expiry_live and minute_of_day >= EXPIRY_FORCE_EXIT_MOD:
                should_exit = True
                exit_reason = 'EXPIRY_FORCE_EXIT'
                logger.info(f"[LIVE] Expiry force-exit: mod={minute_of_day} >= {EXPIRY_FORCE_EXIT_MOD}")

            # Hard time exit
            # Normal day: exit at 3 min ONLY if trade is at a loss (5% stop already hit
            # or flat/losing). If profitable, let it run up to 30 min.
            # Expiry day: always exit at EXPIRY_MAX_HOLD_MINS regardless.
            entry_time = pos.get('entry_time')
            if not should_exit and entry_time is not None:
                hold_mins = (now - entry_time).total_seconds() / 60
                pnl_pct = (real_ltp - pos['entry_price']) / (pos['entry_price'] + 1e-9)
                if is_expiry_live:
                    if hold_mins >= EXPIRY_MAX_HOLD_MINS:
                        should_exit = True
                        exit_reason = 'MAX_HOLD_EXIT'
                        logger.info(f"[LIVE] Expiry max-hold exit. hold={hold_mins:.0f}min pnl={pnl_pct:+.1%}")
                else:
                    # At 3 min: exit if losing. For high-premium ATM options (>Rs 150),
                    # require -2% loss before cutting — natural tick noise is ±1.5% on
                    # a Rs 200+ option and should not trigger early exit.
                    early_cut_thresh = -0.02 if pos['entry_price'] > 150 else 0.0
                    if hold_mins >= 3 and pnl_pct < early_cut_thresh:
                        should_exit = True
                        exit_reason = 'MAX_HOLD_EXIT'
                        logger.info(f"[LIVE] 3-min loss exit. hold={hold_mins:.0f}min pnl={pnl_pct:+.1%}")
                    elif hold_mins >= 30:
                        should_exit = True
                        exit_reason = 'MAX_HOLD_EXIT'
                        logger.info(f"[LIVE] 30-min max-hold exit. hold={hold_mins:.0f}min pnl={pnl_pct:+.1%}")
            
            # Trailing stop — mirrors paper trader logic exactly
            if not should_exit:
                peak_gain_pct = (peak_ltp - pos['entry_price']) / (pos['entry_price'] + 1e-9)
                if is_expiry_live:
                    # Expiry: ultra-tight 1.5% trail — gamma can reverse in seconds
                    if peak_gain_pct > 0.0:
                        time_adjusted_stop = peak_ltp * (1 - 0.015)
                    elif minute_of_day >= 225:   # pin-break zone
                        time_adjusted_stop = peak_ltp * (1 - 0.15)
                    else:
                        time_adjusted_stop = pos['stop_price']
                else:
                    # Normal day trailing stop — tiered, mirrors position_manager.py exactly
                    if peak_gain_pct > 0.10:
                        time_adjusted_stop = peak_ltp * (1 - 0.08)   # big winner: trail 8%
                    elif peak_gain_pct > 0.05:
                        time_adjusted_stop = peak_ltp * (1 - 0.12)   # decent winner: trail 12%
                    elif peak_gain_pct > 0.0:
                        time_adjusted_stop = peak_ltp * (1 - 0.20)   # small gain: loose trail
                    elif minute_of_day >= 270:   # After 1:45 PM
                        time_adjusted_stop = peak_ltp * (1 - 0.25)
                    elif minute_of_day >= 210:   # After 12:45 PM
                        time_adjusted_stop = peak_ltp * (1 - 0.32)
                    else:
                        time_adjusted_stop = pos['stop_price']

                # Only tighten, never loosen
                effective_stop = max(pos['stop_price'], time_adjusted_stop)
                if real_ltp <= effective_stop and effective_stop > pos['stop_price']:
                    should_exit = True
                    exit_reason = 'TRAIL_STOP'
            
            # Original stop loss
            if not should_exit and real_ltp <= pos['stop_price']:
                should_exit = True
                exit_reason = 'STOP_LOSS'
            # Target
            elif not should_exit and real_ltp >= pos['target_price']:
                should_exit = True
                exit_reason = 'TARGET'
            # Time exit (EOD)
            elif not should_exit and (now.hour == 15 and now.minute >= 15):
                should_exit = True
                exit_reason = 'TIME_EXIT_EOD'
            
            if should_exit:
                exit_result = broker.place_exit_order(exit_reason, now, real_ltp)
                if exit_result:
                    logger.critical(f"[LIVE] Position closed: {exit_reason}")
                    ks.record_trade(exit_result['estimated_pnl'])
                    tl.log_exit(
                        now=now,
                        exit_price=exit_result['exit_price'],
                        exit_reason=exit_reason,
                        entry_price=exit_result['entry_price'],
                        contracts=exit_result['contracts'],
                    )
                    _bx = float(exit_result['exit_price']); _be = float(exit_result['entry_price'])
                    _bar_trade_event = {
                        'event': 'EXIT', 'exit_reason': exit_reason,
                        'pnl_rs':  round((_bx - _be) * int(exit_result['contracts']) * 65, 2),
                        'pnl_pct': round((_bx - _be) / _be * 100 if _be > 0 else 0, 4),
                    }
                    _last_exit_bar_time = now
                    # Fix 9: mirror paper path — increment fast-loss counter for live exits
                    if exit_reason == 'MAX_HOLD_EXIT':
                        _fast_losses_today += 1
                else:
                    # Limit exit rejected — escalate immediately to MARKET order.
                    # Do NOT wait for the next bar: every second in an unwanted
                    # position is additional uncontrolled risk.
                    logger.critical(
                        f"[LIVE] LIMIT EXIT REJECTED for {exit_reason} — "
                        f"escalating to emergency MARKET flatten immediately"
                    )
                    flatten_ok = broker.emergency_flatten(f"LIMIT_REJECTED_{exit_reason}")
                    if flatten_ok:
                        logger.critical("[LIVE] Emergency flatten succeeded")
                        _last_exit_bar_time = now
                        # Estimate PnL from last known prices (actual fill unknown for MARKET)
                        _est_pnl = (real_ltp - pos['entry_price']) * pos['quantity']
                        ks.record_trade(_est_pnl)
                        tl.log_exit(
                            now=now,
                            exit_price=real_ltp,
                            exit_reason=f"EMERGENCY_{exit_reason}",
                            entry_price=pos['entry_price'],
                            contracts=pos['contracts'],
                        )
                        _be2 = float(pos['entry_price'])
                        _bar_trade_event = {
                            'event': 'EXIT', 'exit_reason': f"EMERGENCY_{exit_reason}",
                            'pnl_rs':  round((real_ltp - _be2) * int(pos['contracts']) * 65, 2),
                            'pnl_pct': round((real_ltp - _be2) / _be2 * 100 if _be2 > 0 else 0, 4),
                        }
                    else:
                        logger.critical(
                            "[LIVE] EMERGENCY FLATTEN ALSO FAILED — "
                            "CLOSE POSITION MANUALLY IN ANGEL ONE APP NOW"
                        )

        # KILL-SWITCH EMERGENCY FLATTEN INTEGRATION
        # Check if kill-switch has requested an emergency position flatten
        should_flatten, flatten_reason = ks.consume_flatten_request()
        if should_flatten:
            logger.critical(f"\n{'='*70}\n  [KILL-SWITCH] EMERGENCY FLATTEN TRIGGERED\n  Reason: {flatten_reason}\n{'='*70}")
            
            # Flatten paper position
            if paper is not None and paper.in_position:
                spot_now = float(row.get('close', 0))
                pos = paper._position
                opt_type = pos.get('option_type', 'CE')
                pos_strike = int(pos.get('strike', 0))
                real_ltp = fetch_option_ltp(session, pos_strike, opt_type, force_fresh=True)
                
                if real_ltp <= 0:
                    from nifty_trader.execution.orders import estimate_option_premium as _eop, _next_expiry_mins as _nem
                    iv_ann = float(row.get('iv_proxy', 0.06)) * (252 ** 0.5)
                    dte_now = _nem(now)
                    real_ltp = _eop(spot_now, iv_ann, dte_now, float(pos_strike), opt_type)
                
                # Force immediate exit in paper mode
                paper.track(real_ltp, now, ks, current_row=row, signal_state=_signal_state, force_exit=True)
                logger.critical("[KILL-SWITCH] Paper position flattened")
                
                # Log the forced exit
                if paper._trades:
                    last_t = paper._trades[-1]
                    tl.log_exit(
                        now=now,
                        exit_price=float(last_t.get('exit_price', real_ltp)),
                        exit_reason=f"KILL_SWITCH_{flatten_reason}",
                        entry_price=float(last_t.get('entry_price', pos['entry_price'])),
                        contracts=int(last_t.get('contracts', 0)),
                    )
                    _bks_x = float(last_t.get('exit_price', real_ltp))
                    _bks_e = float(last_t.get('entry_price', pos['entry_price']))
                    _bar_trade_event = {
                        'event': 'EXIT', 'exit_reason': f"KILL_SWITCH_{flatten_reason}",
                        'pnl_rs':  round((_bks_x - _bks_e) * int(last_t.get('contracts', 1)) * 65, 2),
                        'pnl_pct': round((_bks_x - _bks_e) / _bks_e * 100 if _bks_e > 0 else 0, 4),
                    }
            
            # Flatten real broker position
            elif broker is not None and broker.is_in_position():
                # Save position BEFORE emergency_flatten() clears _open_position
                pos = broker.get_position()
                # Fetch real LTP now — emergency_flatten() uses it for limit price,
                # and we log it as exit_price (best estimate of actual fill).
                _ks_opt_type   = pos.get('option_type', 'CE')
                _ks_pos_strike = int(pos.get('strike', 0))
                _ks_real_ltp   = fetch_option_ltp(session, _ks_pos_strike, _ks_opt_type, force_fresh=True)
                if _ks_real_ltp <= 0:
                    _ks_real_ltp = pos['entry_price']  # last resort fallback
                success = broker.emergency_flatten(flatten_reason)
                if success:
                    logger.critical("[KILL-SWITCH] Real broker position flattened successfully")
                    _est_pnl = (_ks_real_ltp - pos['entry_price']) * pos['quantity']
                    ks.record_trade(_est_pnl)
                    tl.log_exit(
                        now=now,
                        exit_price=_ks_real_ltp,
                        exit_reason=f"EMERGENCY_FLATTEN_{flatten_reason}",
                        entry_price=pos['entry_price'],
                        contracts=pos['contracts'],
                    )
                    _bef = float(pos['entry_price'])
                    _bar_trade_event = {
                        'event': 'EXIT', 'exit_reason': f"EMERGENCY_FLATTEN_{flatten_reason}",
                        'pnl_rs':  round((_ks_real_ltp - _bef) * int(pos['contracts']) * 65, 2),
                        'pnl_pct': round((_ks_real_ltp - _bef) / _bef * 100 if _bef > 0 else 0, 4),
                    }
                else:
                    logger.critical("[KILL-SWITCH] EMERGENCY FLATTEN FAILED - MANUAL INTERVENTION REQUIRED")
                    logger.critical("    >> IMMEDIATELY CLOSE POSITION IN ANGEL ONE APP <<")
                    logger.critical("    >> THEN PRESS CTRL+C TO STOP SYSTEM <<")

        # 9. Output Dashboard
        is_streaming = bool(streamer and streamer.connected)
        # Build a composite block reason that incorporates all safety gates
        if _warmup_blocked:
            effective_blocked = True
            effective_reason = warmup_reason
        elif regime_uncertain:
            effective_blocked = True
            effective_reason = f"REGIME_UNCERTAIN: awaiting confirmation bars"
        elif drift_conf_mult == 0.0:
            effective_blocked = True
            effective_reason = f"FEATURE_DRIFT: confidence killed (OOD features: {drift_killed[:3]})"
        else:
            effective_blocked = ks_blocked
            effective_reason = ks_reason

        safety_st = safety.status_dict()
        print_live_dashboard(row, analysis, signal, models, current_regime,
                             trade_info, now, micro_regime=micro_regime,
                             ks_blocked=effective_blocked, ks_reason=effective_reason,
                             streaming=is_streaming,
                             block_reason=get_last_block_reason() if signal is None else '')

        # --- Per-minute bar CSV log ---
        _sig_horizons = signal.get('signals', {}) if signal else {}
        _avg_conf     = float(signal.get('avg_conf', 0.0)) if signal else 0.0
        _agreement    = float(signal.get('agreement', 0.0)) if signal else 0.0
        _meta_conf    = float(signal.get('meta_conf', 0.0)) if signal else 0.0
        bar_log.log(
            row             = row,
            signal          = signal,
            current_regime  = current_regime,
            micro_regime    = micro_regime,
            block_reason    = get_last_block_reason() if signal is None else '',
            ks_blocked      = effective_blocked,
            ks_reason       = effective_reason,
            trade_info      = trade_info,
            analysis        = analysis,
            regime_conf     = regime_conf,
            avg_conf        = _avg_conf,
            agreement       = _agreement,
            meta_conf       = _meta_conf,
            crisis_bypass   = _crisis_bypass,
            consec_losses   = ks.consec_losses,
            trade_event     = _bar_trade_event.get('event'),
            trade_exit_reason = _bar_trade_event.get('exit_reason'),
            trade_pnl_rs    = _bar_trade_event.get('pnl_rs'),
            trade_pnl_pct   = _bar_trade_event.get('pnl_pct'),
            signals_by_horizon = _sig_horizons,
            now             = now,
        )

        # Sleep until the next minute starts.
        # When in a live position, check stop-loss every 10s instead of sleeping the
        # full minute — this ensures a 5% loss triggers an exit within 10 seconds.
        elapsed = (dt.datetime.now() - now).total_seconds()
        sleep_s = max(1, min(60, 60 - elapsed))
        try:
            if paper is not None and paper.in_position:
                # Paper mode intra-bar SL polling — check every 10s so 5% SL
                # fires within 10s instead of waiting the full 60s bar boundary.
                slept = 0
                while slept < sleep_s and paper.in_position:
                    chunk = min(10, sleep_s - slept)
                    time.sleep(chunk)
                    slept += chunk
                    _intra_ltp = fetch_option_ltp(
                        session,
                        int(paper._position['strike']),
                        paper._position['option_type'],
                        force_fresh=True
                    )
                    if _intra_ltp > 0:
                        _intra_stop = float(paper._position['stop'])
                        if _intra_ltp <= _intra_stop:
                            _intra_now = dt.datetime.now()
                            logger.info(
                                f"[Paper] Intra-bar SL hit: LTP={_intra_ltp:.2f} <= stop={_intra_stop:.2f} — exiting"
                            )
                            paper.track(_intra_ltp, _intra_now, ks,
                                        current_row=row, signal_state=_signal_state)
                            break
            elif broker is not None and broker.is_in_position():
                # Intra-bar stop-loss polling — check every 10s
                slept = 0
                while slept < sleep_s:
                    chunk = min(10, sleep_s - slept)
                    time.sleep(chunk)
                    slept += chunk
                    # Re-fetch live LTP and check stop
                    _pos = broker.get_position()
                    if _pos is None:
                        break  # position already closed
                    _opt_type  = _pos['option_type']
                    _strike    = int(_pos['strike'])
                    _entry     = float(_pos['entry_price'])
                    _stop      = float(_pos['stop_price'])
                    _ltp_now   = fetch_option_ltp(session, _strike, _opt_type, force_fresh=True)
                    if _ltp_now > 0 and _ltp_now <= _stop:
                        _pnl_pct = (_ltp_now - _entry) / _entry * 100
                        logger.critical(
                            f"[LIVE] INTRA-BAR STOP HIT: LTP={_ltp_now:.2f} <= stop={_stop:.2f} "
                            f"({_pnl_pct:+.1f}%) — exiting NOW"
                        )
                        _intra_now = dt.datetime.now()
                        _exit_result = broker.place_exit_order('STOP_LOSS', _intra_now, _ltp_now)
                        if _exit_result:
                            logger.critical(f"[LIVE] Intra-bar stop exit filled @ Rs {_exit_result['exit_price']:.2f}")
                            ks.record_trade(_exit_result['estimated_pnl'])
                            tl.log_exit(
                                now=_intra_now,
                                exit_price=_exit_result['exit_price'],
                                exit_reason='STOP_LOSS',
                                entry_price=_exit_result['entry_price'],
                                contracts=_exit_result['contracts'],
                            )
                            _bib_x = float(_exit_result['exit_price']); _bib_e = float(_exit_result['entry_price'])
                            _bar_trade_event = {
                                'event': 'EXIT', 'exit_reason': 'STOP_LOSS',
                                'pnl_rs':  round((_bib_x - _bib_e) * int(_exit_result['contracts']) * 65, 2),
                                'pnl_pct': round((_bib_x - _bib_e) / _bib_e * 100 if _bib_e > 0 else 0, 4),
                            }
                            _last_exit_bar_time = _intra_now
                        else:
                            logger.error("[LIVE] Intra-bar stop exit FAILED — escalating to emergency flatten")
                            broker.emergency_flatten("INTRA_BAR_STOP_LOSS")
                        break
            else:
                time.sleep(sleep_s)
        except KeyboardInterrupt:
            break   # fall through to cleanup below

    # Cleanup on exit (reached on break, normal return, or Ctrl+C during sleep)
    _live_loop_cleanup(streamer, paper)

    





def run_live(models, regime_det, capital=10000, verbose=False):
    live_loop(models, regime_det, capital=capital, verbose=verbose)


def run_paper(models, regime_det, capital=10000, verbose=False):
    try:
        live_loop(models, regime_det, capital=capital, paper_mode=True, verbose=verbose)
    except KeyboardInterrupt:
        print("\n  [Paper] Ctrl+C received — saving trades and exiting.")
    # Note: end_of_day() is called inside live_loop's cleanup block on exit.


def run_dashboard(models, regime_det, capital=10000, verbose=False):
    live_loop(models, regime_det, capital=capital,
              paper_mode=False, dashboard_mode=True, verbose=verbose)
