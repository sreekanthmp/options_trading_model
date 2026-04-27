"""
BarLogger — per-minute CSV analytics logger.

Writes one row per bar to  logs/bar_log_YYYY-MM-DD.csv
Columns cover everything needed to analyse gate behaviour, model signals,
market context, and trade outcomes offline.

Usage (in live.py):
    from ..utils.bar_logger import BarLogger
    bar_log = BarLogger()
    ...
    bar_log.log(row, signal, current_regime, micro_regime, block_reason,
                ks_blocked, ks_reason, trade_info, analysis, regime_conf,
                avg_conf=..., agreement=...)
"""
import csv
import os
import logging
from datetime import date, datetime

logger = logging.getLogger(__name__)

# All columns written to the CSV — order matters (header = same order)
COLUMNS = [
    # --- Time ---
    'timestamp', 'date', 'time', 'minute_of_day', 'session_pct', 'day_of_week',

    # --- Price / market context ---
    'spot_close', 'spot_open', 'gap_pct', 'spot_vs_open_pct',
    'vwap', 'vwap_dist_pct', 'vwap_position',
    'ret_1m', 'ret_5m', 'ret_15m',
    'atr_14', 'atr_14_pct', 'iv_proxy', 'iv_rank_approx',
    'bb_position', 'bb_squeeze', 'bb_width',
    'vol_ratio', 'day_atr_pct',

    # --- Indicators ---
    'rsi_1m', 'rsi_5m', 'rsi_15m',
    'adx_1m', 'adx_5m', 'adx_15m',
    'macd_hist_1m', 'macd_hist_5m',
    'stoch_k_1m', 'stoch_k_5m',
    'cci_1m', 'cci_5m',
    'mfi_14', 'pressure_ratio',
    'tick_imbalance', 'vwap_dev_vel',

    # --- TA score ---
    'ta_overall_score', 'ta_momentum', 'ta_trend', 'ta_flow',

    # --- Regime ---
    'regime', 'regime_name', 'regime_conf', 'micro_regime',
    'is_crisis', 'crisis_bypass',

    # --- Model signals ---
    'model_1m_pred', 'model_1m_conf',
    'model_5m_pred', 'model_5m_conf',
    'model_15m_pred', 'model_15m_conf',
    'model_30m_pred', 'model_30m_conf',
    'model_avg_conf', 'model_agreement', 'model_direction',
    'meta_labeler_conf',

    # --- Gate result ---
    'signal_generated',       # 1 = signal passed all gates, 0 = blocked
    'signal_direction',       # UP / DOWN / None
    'signal_strength',        # STRONG / MODERATE / WEAK / None
    'signal_conf',            # final adj_conf if signal generated
    'block_reason',           # gate message if blocked
    'ks_blocked',             # kill-switch blocked?
    'ks_reason',              # kill-switch reason

    # --- Active position (if any) ---
    'in_position',
    'pos_direction', 'pos_strike', 'pos_option_type',
    'pos_entry_price', 'pos_current_ltp', 'pos_pnl_pct', 'pos_hold_mins',
    'pos_peak_ltp', 'pos_stop_price',

    # --- Trade outcome (filled only on EXIT bar) ---
    'trade_event',       # ENTRY / EXIT / None
    'trade_exit_reason', # STOP_LOSS / TARGET / TRAIL_STOP / MAX_HOLD_EXIT etc.
    'trade_pnl_rs',      # net Rs P&L on exit
    'trade_pnl_pct',     # % of premium

    # --- Session totals (running) ---
    'session_trades', 'session_wins', 'session_losses',
    'session_net_pnl', 'session_win_rate',
    'consec_losses',
]


class BarLogger:
    """Writes one CSV row per bar. Rotates to a new file each day."""

    def __init__(self, log_dir: str = 'logs'):
        self._log_dir  = log_dir
        self._date     = None
        self._fh       = None   # file handle
        self._writer   = None
        # running session stats
        self._trades   = 0
        self._wins     = 0
        self._losses   = 0
        self._net_pnl  = 0.0
        os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _rotate(self, today: date):
        """Open a new CSV file for today if the date changed."""
        if self._date == today:
            return
        if self._fh:
            self._fh.close()
        self._date   = today
        self._trades = self._wins = self._losses = 0
        self._net_pnl = 0.0
        path = os.path.join(self._log_dir, f'bar_log_{today}.csv')
        new_file = not os.path.exists(path)
        self._fh     = open(path, 'a', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._fh, fieldnames=COLUMNS, extrasaction='ignore')
        if new_file:
            self._writer.writeheader()
            self._fh.flush()

    # ------------------------------------------------------------------
    def log(self,
            row,             # pd.Series — current bar features
            signal,          # dict | None from generate_signal
            current_regime,  # int
            micro_regime,    # str
            block_reason,    # str — from get_last_block_reason()
            ks_blocked,      # bool
            ks_reason,       # str
            trade_info,      # dict | None — active position from paper trader
            analysis,        # dict from build_analysis
            regime_conf      = 0.5,
            avg_conf         = 0.0,
            agreement        = 0.0,
            meta_conf        = 0.0,
            crisis_bypass    = False,
            consec_losses    = 0,
            trade_event      = None,   # 'ENTRY' / 'EXIT' / None
            trade_exit_reason= None,
            trade_pnl_rs     = None,
            trade_pnl_pct    = None,
            signals_by_horizon = None,  # dict h -> {pred, conf}
            now              = None,
            ):

        if now is None:
            now = datetime.now()
        today = now.date()
        self._rotate(today)

        # --- Update running session stats on EXIT ---
        if trade_event == 'EXIT' and trade_pnl_rs is not None:
            self._trades  += 1
            self._net_pnl += trade_pnl_rs
            if trade_pnl_rs > 0:
                self._wins   += 1
            else:
                self._losses += 1

        def _f(key, default=0.0):
            try:
                v = row.get(key, default)
                return float(v) if v is not None else default
            except Exception:
                return default

        def _s(key, default=''):
            try:
                v = row.get(key, default)
                return str(v) if v is not None else default
            except Exception:
                return default

        regime_names = {0: 'TRENDING', 1: 'RANGING', 2: 'CRISIS', -1: 'UNCERTAIN'}

        # Model signals per horizon
        h_data = signals_by_horizon or {}
        def _hpred(h): return 'UP' if h_data.get(h, {}).get('pred', -1) == 1 else ('DOWN' if h_data.get(h, {}).get('pred', -1) == 0 else '')
        def _hconf(h): return round(float(h_data.get(h, {}).get('conf', 0.0)), 4)

        # Position info
        pos = trade_info or {}
        in_pos = bool(pos)
        entry_px  = float(pos.get('entry_price', 0) or 0)
        cur_ltp   = float(pos.get('current_ltp', 0) or pos.get('ltp', 0) or 0)
        peak_ltp  = float(pos.get('peak_ltp', 0) or 0)
        stop_px   = float(pos.get('stop_price', 0) or 0)
        hold_mins = int(pos.get('hold_mins', 0) or 0)
        pos_pnl   = ((cur_ltp - entry_px) / entry_px * 100) if entry_px > 0 else 0.0

        win_rate  = (self._wins / self._trades * 100) if self._trades > 0 else 0.0

        rec = {
            # Time
            'timestamp':       now.strftime('%Y-%m-%d %H:%M:%S'),
            'date':            str(today),
            'time':            now.strftime('%H:%M'),
            'minute_of_day':   int(_f('minute_of_day')),
            'session_pct':     round(_f('session_pct'), 4),
            'day_of_week':     now.strftime('%A'),

            # Price / context
            'spot_close':      round(_f('close'), 2),
            'spot_open':       round(_f('day_open', _f('close')), 2),
            'gap_pct':         round(_f('gap_pct'), 4),
            'spot_vs_open_pct':round(_f('spot_vs_open_pct', _f('above_prev_close')), 4),
            'vwap':            round(_f('vwap_proxy', _f('close')), 2),
            'vwap_dist_pct':   round(_f('vwap_dev', 0.0), 4),
            'vwap_position':   'ABOVE' if _f('close') > _f('vwap_proxy', _f('close')) else 'BELOW',
            'ret_1m':          round(_f('ret_1m_fd', _f('ret_1m', 0)), 6),
            'ret_5m':          round(_f('ret_5m_fd', _f('ret_5m', 0)), 6),
            'ret_15m':         round(_f('ret_15m_fd', _f('ret_15m', 0)), 6),
            'atr_14':          round(_f('atr_14'), 4),
            'atr_14_pct':      round(_f('atr_14_pct'), 6),
            'iv_proxy':        round(_f('iv_proxy'), 6),
            'iv_rank_approx':  round(_f('iv_rank_approx'), 2),
            'bb_position':     round(_f('bb_position'), 4),
            'bb_squeeze':      int(_f('bb_squeeze')),
            'bb_width':        round(_f('bb_width'), 6),
            'vol_ratio':       round(_f('vol_ratio'), 4),
            'day_atr_pct':     round(_f('day_atr_pct'), 6),

            # Indicators
            'rsi_1m':          round(_f('rsi_14'), 2),
            'rsi_5m':          round(_f('tf5_rsi', 0), 2),
            'rsi_15m':         round(_f('tf15_rsi', 0), 2),
            'adx_1m':          round(_f('adx_14'), 2),
            'adx_5m':          round(_f('tf5_adx', 0), 2),
            'adx_15m':         round(_f('tf15_adx', 0), 2),
            'macd_hist_1m':    round(_f('macd_hist'), 6),
            'macd_hist_5m':    round(_f('tf5_macd_hist', 0), 6),
            'stoch_k_1m':      round(_f('stoch_k'), 2),
            'stoch_k_5m':      round(_f('tf5_stoch_k', 0), 2),
            'cci_1m':          round(_f('cci_20'), 2),
            'cci_5m':          round(_f('tf5_cci', 0), 2),
            'mfi_14':          round(_f('mfi_14'), 2),
            'pressure_ratio':  round(_f('pressure_ratio'), 4),
            'tick_imbalance':  round(_f('tick_imbalance'), 4),
            'vwap_dev_vel':    round(_f('vwap_dev_vel'), 6),

            # TA score
            'ta_overall_score':round(_f('ta_overall_score'), 4),
            'ta_momentum':     round(float(analysis.get('momentum_score', 0) or 0), 4),
            'ta_trend':        round(float(analysis.get('trend_score', 0) or 0), 4),
            'ta_flow':         round(float(analysis.get('flow_score', 0) or 0), 4),

            # Regime
            'regime':          int(current_regime),
            'regime_name':     regime_names.get(int(current_regime), 'UNKNOWN'),
            'regime_conf':     round(float(regime_conf), 4),
            'micro_regime':    str(micro_regime),
            'is_crisis':       1 if int(current_regime) == 2 else 0,
            'crisis_bypass':   1 if crisis_bypass else 0,

            # Model
            'model_1m_pred':   _hpred(1),  'model_1m_conf':  _hconf(1),
            'model_5m_pred':   _hpred(5),  'model_5m_conf':  _hconf(5),
            'model_15m_pred':  _hpred(15), 'model_15m_conf': _hconf(15),
            'model_30m_pred':  _hpred(30), 'model_30m_conf': _hconf(30),
            'model_avg_conf':  round(float(avg_conf), 4),
            'model_agreement': round(float(agreement), 4),
            'model_direction': signal['direction'] if signal else '',
            'meta_labeler_conf': round(float(meta_conf), 4),

            # Gate result
            'signal_generated': 1 if signal else 0,
            'signal_direction': signal['direction'] if signal else '',
            'signal_strength':  signal.get('strength', '') if signal else '',
            'signal_conf':      round(float(signal.get('avg_conf', 0)), 4) if signal else 0.0,
            'block_reason':     block_reason if not signal else '',
            'ks_blocked':       1 if ks_blocked else 0,
            'ks_reason':        ks_reason if ks_blocked else '',

            # Position
            'in_position':      1 if in_pos else 0,
            'pos_direction':    str(pos.get('direction', '')),
            'pos_strike':       int(pos.get('strike', 0) or 0),
            'pos_option_type':  str(pos.get('option_type', '')),
            'pos_entry_price':  round(entry_px, 2),
            'pos_current_ltp':  round(cur_ltp, 2),
            'pos_pnl_pct':      round(pos_pnl, 4),
            'pos_hold_mins':    hold_mins,
            'pos_peak_ltp':     round(peak_ltp, 2),
            'pos_stop_price':   round(stop_px, 2),

            # Trade outcome
            'trade_event':        trade_event or '',
            'trade_exit_reason':  trade_exit_reason or '',
            'trade_pnl_rs':       round(float(trade_pnl_rs), 2) if trade_pnl_rs is not None else '',
            'trade_pnl_pct':      round(float(trade_pnl_pct), 4) if trade_pnl_pct is not None else '',

            # Session totals
            'session_trades':   self._trades,
            'session_wins':     self._wins,
            'session_losses':   self._losses,
            'session_net_pnl':  round(self._net_pnl, 2),
            'session_win_rate': round(win_rate, 1),
            'consec_losses':    int(consec_losses),
        }

        try:
            self._writer.writerow(rec)
            self._fh.flush()
        except Exception as e:
            logger.warning(f"[BarLogger] write failed: {e}")

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None
