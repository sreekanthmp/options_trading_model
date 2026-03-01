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
    _training_feature_stats,
)
from ..data.loader import load_ohlcv
from ..features.feature_engineering import (
    add_1min_features, add_htf_features, add_daily_features, get_feature_cols,
    FEATURE_COLS, FEATURE_LIVE_OK,
)
from ..regimes.hmm_regime import RegimeDetector, RegimeStateMachine, REGIME_UNCERTAIN, intraday_regime_override
from ..signals.analysis import build_analysis, detect_micro_regime
from ..execution.risk import KillSwitch, SetupFatigueTracker
from ..signals.signal_generator import generate_signal, SignalState, _signal_state
from ..execution.orders import select_option, display_option_predictions, option_pnl_estimate
from .dashboard import print_live_dashboard
from ..data.websocket import (
    AngelSession, MarketStreamer, fetch_live_candles, fetch_live_htf, sync_historical_buffer,
)
from ..execution.position_manager import PaperTrader
from ..data.external_data import (
    fetch_heavyweight_returns, fetch_banknifty_spread, fetch_option_chain_ndi,
    fetch_option_ltp, _prefetch_all_symbols,
)
from ..utils.time_utils import calculate_dynamic_stops
from ..utils.live_safety import LiveSafetyManager
from ..utils.trade_logger import TradeLogger

logger = logging.getLogger(__name__)


def _quick_ml_agreement(row: pd.Series, models: dict, active_features: list) -> float:
    """
    Fast pre-check: poll all horizon models and return the directional agreement
    (fraction of weighted votes for the dominant direction).

    Used by KillSwitch CRISIS bypass — runs before generate_signal() so the
    bypass has a fresh value even when no signal was generated the previous bar.
    Returns 0.0 on any error (safe fallback = no bypass).
    """
    import numpy as np
    try:
        X_raw = np.array([[float(row.get(f, 0.0)) for f in active_features]])
        # Use first model's scaler (all horizons share the same live_scaler)
        first = next(iter(models.values()))
        scaler = first.get('live_scaler')
        X = scaler.transform(X_raw) if scaler is not None else X_raw

        weighted_up = 0.0; weighted_dn = 0.0
        from ..config import HORIZON_WEIGHTS
        for h, res in models.items():
            mdl = res.get('final_model')
            if mdl is None or not hasattr(mdl, 'predict_proba'):
                continue
            proba = mdl.predict_proba(X)[0]
            if len(proba) != 2:
                continue
            conf = proba[1] if proba[1] > 0.5 else proba[0]
            w = HORIZON_WEIGHTS.get(h, 0.25)
            if proba[1] > 0.5:
                weighted_up += w * conf
            else:
                weighted_dn += w * conf
        total = weighted_up + weighted_dn
        if total < 1e-9:
            return 0.0
        return max(weighted_up, weighted_dn) / total
    except Exception:
        return 0.0


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
    
    # Initialize Core Components
    paper = PaperTrader(capital) if (paper_mode and not dashboard_mode) else None
    df1d_static = load_ohlcv(DATA_1DAY, "daily (static)")

    # 1. Initial Regime Detection
    if df1d_static is not None:
        regime_series = regime_det.predict(df1d_static)
        current_regime = int(regime_series.iloc[-1] if len(regime_series) else REGIME_RANGING)
    else:
        current_regime = REGIME_RANGING

    print(f"   Current regime: {REGIME_NAMES[current_regime]}")
    session = AngelSession()
    ks = KillSwitch(capital)
    ks.notify_regime(current_regime)

    # Safety orchestrator, regime state machine, and trade logger
    safety        = LiveSafetyManager(ks)
    regime_sm     = RegimeStateMachine()
    setup_fatigue = SetupFatigueTracker()
    tl            = TradeLogger()    # rotates automatically at midnight (new instance per day below)
    
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
    last_trade_day = None
    last_htf_recalc = time.time()  # Track when HTF was last fully recalculated
    htf_recalc_needed = False  # Flag to trigger HTF feature recalculation
    
    # Keep a manageable buffer (20k rows is ~2 months of 1-min data)
    MIN_PRELOAD_BARS = 20000

    # v4.0: Cold-Start Sync (Pre-flight data buffer)
    print("v4.0: Executing cold-start sync...")
    df1m_cold, df5m_cold, df15m_cold = sync_historical_buffer(session)
    # One trading day = 375 1-min bars (9:15 AM - 3:30 PM), so threshold is 300
    if df1m_cold is not None and len(df1m_cold) > 300:
        print("  [OK] Cold-start complete. Using synced buffer.")
        df1m_live = df1m_cold
        live_df5m = df5m_cold if df5m_cold is not None else None
        live_df15m = df15m_cold if df15m_cold is not None else None
    else:
        print("  [WARN] Cold-start failed. Loading legacy buffer...")
        df1m_live = load_ohlcv(DATA_1MIN, "1-min (live preload)")
        if df1m_live is None or len(df1m_live) < 2000:
            raise RuntimeError("Not enough 1-min history")
        df1m_live = df1m_live.tail(MIN_PRELOAD_BARS).reset_index(drop=True)

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
        # Only refresh every 30 minutes to avoid insufficient data issues
        # and because HTF indicators change slowly
        minutes_since_htf_recalc = (time.time() - last_htf_recalc) / 60.0
        if minutes_since_htf_recalc >= 30.0 or live_df5m is None or live_df15m is None:
            print(f"[{now.strftime('%H:%M:%S')}] Refreshing HTF indicators...")
            # Fetch fresh HTF data with more bars for indicator calculation
            new5 = fetch_live_htf(session, 'FIVE_MINUTE', 300)
            new15 = fetch_live_htf(session, 'FIFTEEN_MINUTE', 200)
            
            # Only update if we got valid data
            if new5 is not None and len(new5) >= 50:
                live_df5m = new5
                htf_recalc_needed = True
            if new15 is not None and len(new15) >= 30:
                live_df15m = new15
                htf_recalc_needed = True
            
            last_htf_recalc = time.time()

        # 3. BRIDGE THE GAP: Fetch enough candles to cover today's session (09:15 - Now)
        # Fetching 500 candles ensures we have all of today plus the tail of Friday
        new = fetch_live_candles(session, n=500)
        if new is None or new.empty:
            print(f"[{now.strftime('%H:%M:%S')}] Waiting for API data...")
            time.sleep(10); continue

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
        
        # Add HTF features - only recalculate when HTF data is refreshed
        # Otherwise, HTF features already exist in 'recent' via forward-fill from df1m_live (step 4a)
        if htf_recalc_needed:
            # Recalculate HTF features with fresh data every 30 minutes
            if live_df5m is not None and not live_df5m.empty and len(live_df5m) >= 50:
                recent = add_htf_features(recent, live_df5m, 'tf5_', [1,3,6])
            
            if live_df15m is not None and not live_df15m.empty and len(live_df15m) >= 30:
                recent = add_htf_features(recent, live_df15m, 'tf15_', [1,4])
            
            htf_recalc_needed = False
            print(f"  [OK] HTF features recalculated with fresh API data")
        
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
                break
            time.sleep(10); continue

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
            break

        # -----------------------------------------------------------------------
        # SAFETY GATE D: Emergency shutdown
        # -----------------------------------------------------------------------
        if safety.is_shutdown_requested():
            reason = safety.shutdown_reason()
            tl.log_safety_event('EMERGENCY_SHUTDOWN', reason)
            logger.critical(f"[Safety] EMERGENCY SHUTDOWN: {reason}")
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
                    print(f"  [WebSocket] Live: ₹{tick_stats['latest']:,.2f} | "
                          f"Ticks: {tick_stats['count']} | "
                          f"Range: ₹{tick_stats['range']:.2f}")
            
            # Reset OFI counters every minute
            streamer.reset_ofi()
            
            # 2026 Edge: Heartbeat Monitor - Check for zombie WebSocket
            is_market_open = (9*60+15 <= hm <= 15*60+30)
            if not streamer.check_heartbeat(market_open=is_market_open):
                # Zombie detected - force reconnect
                streamer.force_reconnect()
        
        # Heavyweight Returns + BankNifty Spread via Angel One (not yfinance)
        # Background prefetch: kicks off a daemon thread so it never blocks the
        # signal path. Results are read from the 90s TTL cache (always instant).
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
        
        # Fetch real ATM option LTP from Angel One (cached 30s)
        # Inject into row so signal_generator and dashboard use real prices.
        _spot_now = float(row.get('close', 0))
        _atm = int(round(_spot_now / 50) * 50)
        _ce_ltp = fetch_option_ltp(session, _atm, 'CE')
        _pe_ltp = fetch_option_ltp(session, _atm, 'PE')
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
        if minute_of_day_current <= 5 and not hasattr(live_loop, f'_black_swan_checked_{today_d}'):
            gap_pct = abs(float(row.get('gap_pct', 0)))
            day_atr = float(row.get('day_atr_pct', 0.5))
            if gap_pct > 2.5 * day_atr:
                ks.notify_black_swan(gap_pct, day_atr)
            setattr(live_loop, f'_black_swan_checked_{today_d}', True)

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
            _ks_agreement = _quick_ml_agreement(row, models, active_features)
        else:
            _ks_agreement = _signal_state.last_agreement
        ks_blocked, ks_reason = ks.check(current_atr, avg_atr, current_regime,
                                        micro_regime=micro_regime,
                                        agreement=_ks_agreement,
                                        minute_of_day=minute_of_day_current)

        analysis = build_analysis(row, current_regime)
        signal = None
        trade_info = None

        # Block signals while regime is uncertain (pending confirmation bars)
        regime_uncertain = (current_regime == REGIME_UNCERTAIN)

        # Log non-ks block reasons for gate analysis
        if _warmup_blocked:
            tl.log_signal_blocked(now, warmup_reason, row)
        elif regime_uncertain:
            tl.log_signal_blocked(now, 'REGIME_UNCERTAIN', row)
        elif drift_conf_mult == 0.0:
            tl.log_signal_blocked(now, f'FEATURE_DRIFT_KILLED: {drift_killed[:3]}', row)

        if not ks_blocked and not _warmup_blocked and not regime_uncertain and drift_conf_mult > 0.0:
            # Issue 5: regime-frequency boost — raises conf floor in range-heavy periods
            regime_boost = ks.regime_conf_boost()
            signal = generate_signal(row, models, current_regime,
                                     micro_regime=micro_regime,
                                     signal_state=_signal_state,
                                     extra_conf_floor=regime_boost)

            # Apply feature-drift confidence penalty to the generated signal
            if signal and drift_conf_mult < 1.0:
                original_conf = signal.get('avg_conf', 0.0)
                signal['avg_conf'] = original_conf * drift_conf_mult
                signal['drift_penalty'] = drift_conf_mult
                logger.warning(f"[Safety] Drift penalty applied: conf {original_conf:.2f} -> {signal['avg_conf']:.2f}")
            
            # v4.0: Order Flow Imbalance (OFI) Boost
            # If >80% of recent ticks are buys (price ticking up), boost prediction
            if signal and streamer and streamer.connected:
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
                    continue  # Skip to next iteration
                
                # Select Option Strike and Type
                # Edge Case 1: Pass tick buffer for Limit Price Protection (LPP)
                tick_buffer = streamer.tick_buffer if (streamer and streamer.connected) else None

                # Order dedup: prevent duplicate orders for the same signal
                signal['minute_of_day'] = minute_of_day_current
                signal['spot'] = float(row.get('close', 0))
                if not safety.can_place_order(signal):
                    logger.warning(f"[Safety] Duplicate order suppressed for {signal.get('direction')} @ minute {minute_of_day_current}")
                    trade_info = None
                else:
                    trade_info = select_option(signal, capital, now=now, tick_buffer=tick_buffer)
                    if trade_info is None:
                        logger.info(
                            f"[Order] select_option() rejected signal: "
                            f"dir={signal.get('direction')} conf={signal.get('avg_conf', 0):.3f} "
                            f"regime={current_regime} min={minute_of_day_current}"
                        )

                # Execute Paper Trade if not already in position
                if trade_info and paper is not None and not paper.in_position:
                    paper.enter(signal, trade_info, now)
                    safety.register_order(signal, order_id=str(now.timestamp()))
                    _signal_state.record_trade_taken(signal.get('direction', 'UP'))
                    # Structured trade entry log
                    tl.log_entry(
                        signal=signal,
                        trade_info=trade_info,
                        row=row,
                        regime=current_regime,
                        regime_conf=regime_conf,
                        latency_ms=safety.latency_mon.median_lag_ms(),
                        active_features=active_features,
                    )

        # 8. Mark-to-Market (Paper Tracking)
        if paper is not None and paper.in_position:
            spot_now = float(row.get('close', 0))
            atr_now = float(row.get('atr_14', 1.0))
            pos = paper._position
            bars_open = int((now - pos['entry_time']).total_seconds() / 60)
            dte_entry = pos.get('dte_mins_entry', 750.0)
            dte_now = max(30.0, dte_entry - bars_open)

            # Use REAL option LTP from Angel One API when available.
            # This is the single most important realism fix: in live trading,
            # your P&L and stop triggers are based on the actual market price
            # of the option, not a model estimate.
            # row['atm_ce_ltp'] / row['atm_pe_ltp'] are fetched every bar above.
            opt_type  = pos.get('option_type', 'CE')
            real_ltp_key = 'atm_ce_ltp' if opt_type == 'CE' else 'atm_pe_ltp'
            real_ltp  = float(row.get(real_ltp_key, 0.0))

            # Model estimate as fallback when real LTP is unavailable (API miss)
            ltp_model = option_pnl_estimate(pos['entry_price'], pos['spot_entry'],
                                            spot_now, atr_now, pos['direction'],
                                            bars_open, dte_now)

            if real_ltp > 0:
                ltp_est = real_ltp   # Live: use real market price
                pos['_using_real_ltp'] = True
            else:
                ltp_est = ltp_model  # Fallback: model estimate
                pos['_using_real_ltp'] = False

            # Running MAE/MFE tracking for the trade logger
            _ep    = pos['entry_price']
            _dir   = pos.get('direction', 'UP')
            _move  = (ltp_est - _ep) if _dir == 'UP' else (_ep - ltp_est)
            _mae   = min(0.0, _move)   # negative = adverse
            _mfe   = max(0.0, _move)   # positive = favorable
            tl.log_bar(now, spot_now, ltp_est, _mae, _mfe)

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
                )
                
                # TODO: Setup Fatigue Tracking Integration Point
                # When setup types are added to signal dict (e.g., 'vwap_break', 'or_break'),
                # track failures here to auto-disable failing setups:
                # 
                # setup_type = last_t.get('setup_type', 'unknown')
                # if last_t.get('pnl', 0) < 0:  # Loss
                #     setup_fatigue.record_loss(setup_type, current_regime)
                #
                # Before signal generation, check:
                # if setup_fatigue.is_disabled(signal_setup_type, current_regime):
                #     continue  # Skip this signal

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
                             streaming=is_streaming)

        # Sleep until the next minute starts
        elapsed = (dt.datetime.now() - now).seconds
        sleep_s = max(1, 60 - elapsed)
        time.sleep(sleep_s)
    
    # Cleanup on exit
    if streamer:
        print("\nDisconnecting WebSocket...")
        streamer.disconnect()

    





def run_live(models, regime_det, capital=10000, verbose=False):
    live_loop(models, regime_det, capital=capital, verbose=verbose)


def run_paper(models, regime_det, capital=10000, verbose=False):
    live_loop(models, regime_det, capital=capital, paper_mode=True, verbose=verbose)


def run_dashboard(models, regime_det, capital=10000, verbose=False):
    live_loop(models, regime_det, capital=capital,
              paper_mode=False, dashboard_mode=True, verbose=verbose)
