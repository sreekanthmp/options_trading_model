"""
Paper Trading Report Generator
================================
Reads logs/trades_YYYY-MM-DD.jsonl files (written by TradeLogger) and
produces a clean human-readable summary to reports/paper_report_YYYY-MM-DD.txt

The JSONL source is the authoritative record — written in real time, survives
Ctrl+C and mid-day restarts.  The old paper_trades/*.csv files are no longer
needed.

Usage:
    python paper_report.py                  # today's report
    python paper_report.py 2026-03-01       # specific date
    python paper_report.py --all            # all available dates combined
"""

import os
import sys
import json
import glob
from datetime import datetime, date

LOG_DIR    = 'logs'
REPORT_DIR = 'reports'
LOT_SIZE   = 65   # NIFTY lot size (since April 26, 2025)

# Default capital: read from config.json (same file the live trader uses).
# Falls back to 30000 if config.json is missing or has no capital key.
def _default_capital() -> float:
    try:
        cfg_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(cfg_path, encoding='utf-8') as _f:
            cfg = json.load(_f)
        return float(cfg.get('capital', 30000.0))
    except Exception:
        return 30000.0

CAPITAL = _default_capital()


# ---------------------------------------------------------------------------
# JSONL reader — builds trade list from ENTRY+EXIT event pairs
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list:
    """Parse a trades_YYYY-MM-DD.jsonl into a list of completed trade dicts."""
    if not os.path.exists(path):
        return []

    events = []
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []

    # Index entries and exits by session_trade_number so overlapping trades
    # (where a second ENTRY fires before the first EXIT) are handled correctly.
    open_entries: dict = {}   # trade_num -> entry event
    trades = []

    for ev in events:
        etype = ev.get('event', '')

        if etype == 'ENTRY':
            t_num = ev.get('session_trade_number')
            open_entries[t_num] = ev

        elif etype == 'EXIT':
            t_num = ev.get('session_trade_number')
            open_entry = open_entries.pop(t_num, None)
            if open_entry is None:
                continue  # orphaned exit — skip

            entry_px  = float(open_entry.get('entry_price', 0))
            exit_px   = float(ev.get('exit_price', 0))
            contracts = int(open_entry.get('contracts', 1))
            qty       = contracts * LOT_SIZE

            # PnL: we BUY the option, so profit = exit - entry regardless of direction
            pnl_pts  = exit_px - entry_px
            pnl_rs   = pnl_pts * qty

            # Approximate brokerage (Angel One flat-fee model)
            # Rs20x2 orders + 18% GST + STT(sell) + NSE txn + stamp duty
            brokerage  = 20.0 * 2
            gst        = brokerage * 0.18
            stt        = exit_px * qty * 0.001
            txn        = (entry_px + exit_px) * qty * 0.00053
            stamp      = entry_px * qty * 0.00003
            total_chrg = brokerage + gst + stt + txn + stamp
            net_pnl_rs = pnl_rs - total_chrg

            entry_ts = open_entry.get('timestamp_signal', '')
            exit_ts  = ev.get('timestamp_exit', '')
            try:
                et = datetime.fromisoformat(entry_ts)
                xt = datetime.fromisoformat(exit_ts)
                hold_mins  = max(0, int((xt - et).total_seconds() / 60))
                entry_time = et.strftime('%H:%M')
                exit_time  = xt.strftime('%H:%M')
            except Exception:
                hold_mins  = 0
                entry_time = entry_ts[11:16] if len(entry_ts) > 15 else ''
                exit_time  = exit_ts[11:16]  if len(exit_ts)  > 15 else ''

            trades.append({
                'direction':            open_entry.get('direction', '?'),
                'option_type':          open_entry.get('option_type', '?'),
                'strike':               open_entry.get('strike', 0),
                'symbol':               f"NIFTY {open_entry.get('strike','')} {open_entry.get('option_type','')}",
                'entry_price':          entry_px,
                'exit_price':           exit_px,
                'pnl_pts':              round(pnl_pts, 2),
                'pnl':                  round(net_pnl_rs, 2),
                'gross_pnl':            round(pnl_rs, 2),
                'total_charges':        round(total_chrg, 2),
                'contracts':            contracts,
                'qty':                  qty,
                'hold_mins':            hold_mins,
                'entry_time':           entry_time,
                'exit_time':            exit_time,
                'exit_reason':          ev.get('exit_reason', 'UNKNOWN'),
                'avg_conf':             float(open_entry.get('model_confidence', 0)),
                'regime':               open_entry.get('regime', -1),
                'minute_of_day':        int(open_entry.get('minute_of_day', 0)),
                'session_regime':       open_entry.get('session_regime', ''),
                'session_regime_score': float(open_entry.get('session_regime_score', 0)),
            })

    # Append still-open trades with OPEN status so they appear in the report
    for t_num, open_entry in sorted(open_entries.items()):
        entry_px  = float(open_entry.get('entry_price', 0))
        contracts = int(open_entry.get('contracts', 1))
        entry_ts  = open_entry.get('timestamp_signal', '')
        try:
            et = datetime.fromisoformat(entry_ts)
            entry_time = et.strftime('%H:%M')
        except Exception:
            entry_time = entry_ts[11:16] if len(entry_ts) > 15 else ''

        trades.append({
            'direction':            open_entry.get('direction', '?'),
            'option_type':          open_entry.get('option_type', '?'),
            'strike':               open_entry.get('strike', 0),
            'symbol':               f"NIFTY {open_entry.get('strike','')} {open_entry.get('option_type','')}",
            'entry_price':          entry_px,
            'exit_price':           None,   # still open
            'pnl_pts':              None,
            'pnl':                  None,
            'gross_pnl':            None,
            'total_charges':        None,
            'contracts':            contracts,
            'qty':                  contracts * LOT_SIZE,
            'hold_mins':            None,
            'entry_time':           entry_time,
            'exit_time':            'OPEN',
            'exit_reason':          'OPEN',
            'avg_conf':             float(open_entry.get('model_confidence', 0)),
            'regime':               open_entry.get('regime', -1),
            'minute_of_day':        int(open_entry.get('minute_of_day', 0)),
            'session_regime':       open_entry.get('session_regime', ''),
            'session_regime_score': float(open_entry.get('session_regime_score', 0)),
        })

    return trades


def float_val(row: dict, key: str, default=0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (ValueError, TypeError):
        return default

def int_val(row: dict, key: str, default=0) -> int:
    try:
        return int(float(row.get(key, default) or default))
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(trades: list, report_date: str) -> str:
    lines = []
    sep   = '=' * 68

    if not trades:
        lines.append(sep)
        lines.append(f"  PAPER TRADE REPORT  --  {report_date}")
        lines.append(sep)
        lines.append("  No trades executed on this date.")
        lines.append(sep)
        return '\n'.join(lines)

    # Separate closed and open trades — open trades have pnl=None
    closed = [t for t in trades if t.get('pnl') is not None]
    open_t = [t for t in trades if t.get('pnl') is None]

    n          = len(trades)
    n_closed   = len(closed)
    wins       = [t for t in closed if float_val(t, 'pnl') > 0]
    losses     = [t for t in closed if float_val(t, 'pnl') <= 0]
    total_pnl  = sum(float_val(t, 'pnl')          for t in closed)
    total_gross= sum(float_val(t, 'gross_pnl')     for t in closed)
    total_chrg = sum(float_val(t, 'total_charges') for t in closed)
    win_pnl    = sum(float_val(t, 'pnl') for t in wins)
    loss_pnl   = sum(float_val(t, 'pnl') for t in losses)
    win_rate   = len(wins) / n_closed if n_closed > 0 else 0
    pf         = abs(win_pnl / loss_pnl) if loss_pnl != 0 else float('inf')
    avg_win    = win_pnl  / len(wins)   if wins   else 0
    avg_loss   = loss_pnl / len(losses) if losses else 0
    avg_hold   = sum(int_val(t, 'hold_mins') for t in closed) / n_closed if n_closed else 0
    ret_pct    = total_pnl / CAPITAL * 100
    equity_end = CAPITAL + total_pnl

    buys  = [t for t in trades if t.get('direction', '') == 'UP']
    sells = [t for t in trades if t.get('direction', '') == 'DOWN']
    ce_tr = [t for t in trades if t.get('option_type', '') == 'CE']
    pe_tr = [t for t in trades if t.get('option_type', '') == 'PE']

    reasons: dict = {}
    for t in closed:
        r = t.get('exit_reason', 'UNKNOWN')
        reasons[r] = reasons.get(r, 0) + 1

    # Header
    lines.append(sep)
    lines.append(f"  PAPER TRADE REPORT  --  {report_date}")
    lines.append(sep)

    # Summary
    lines.append("")
    lines.append("  SUMMARY")
    lines.append("  " + "-" * 40)
    lines.append(f"  Starting capital  : Rs {CAPITAL:>12,.2f}")
    lines.append(f"  Ending equity     : Rs {equity_end:>12,.2f}")
    lines.append(f"  Gross P&L         : Rs {total_gross:>+12,.2f}  (before charges)")
    lines.append(f"  Brokerage + STT   : Rs {-total_chrg:>+12,.2f}  (Angel One charges)")
    lines.append(f"  Net P&L (live eq) : Rs {total_pnl:>+12,.2f}  ({ret_pct:+.2f}%)")
    lines.append("")

    # Trade stats
    lines.append("  TRADE STATISTICS")
    lines.append("  " + "-" * 40)
    open_str = f"  ({len(open_t)} still OPEN)" if open_t else ""
    lines.append(f"  Total trades      : {n}  ({n_closed} closed{open_str})")
    lines.append(f"  UP  (CE buys)     : {len(buys)}  ({len(ce_tr)} CE lots)")
    lines.append(f"  DOWN (PE buys)    : {len(sells)}  ({len(pe_tr)} PE lots)")
    lines.append(f"  Winners           : {len(wins)}")
    lines.append(f"  Losers            : {len(losses)}")
    lines.append(f"  Win rate          : {win_rate:.1%}  (closed trades only)")
    lines.append(f"  Profit factor     : {pf:.2f}")
    lines.append(f"  Avg win           : Rs {avg_win:>+10,.2f}")
    lines.append(f"  Avg loss          : Rs {avg_loss:>+10,.2f}")
    avg_hold_str = '<1 min' if avg_hold < 1 else f'{avg_hold:.0f} min'
    lines.append(f"  Avg hold time     : {avg_hold_str}")
    lines.append("")

    # Exit reasons
    lines.append("  EXIT REASONS")
    lines.append("  " + "-" * 40)
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        lines.append(f"  {reason:<28} : {count:>3}  ({count/n:.0%})")
    lines.append("")

    # Rolling win-rate (last 5 closed trades) — mirrors live observation-mode guard
    ROLLING_N = 5
    if n_closed >= ROLLING_N:
        recent_closed = closed[-ROLLING_N:]
        r_wr = sum(1 for t in recent_closed if float_val(t, 'pnl') > 0) / ROLLING_N
        r_flag = "  ** OBSERVATION MODE WOULD TRIGGER **" if r_wr < 0.30 else ""
        lines.append("  ROLLING WIN RATE (last 5 trades)")
        lines.append("  " + "-" * 40)
        lines.append(f"  Last {ROLLING_N} trades WR: {r_wr:.0%}  (live guard threshold: 30%){r_flag}")
        lines.append("")

    # Per-trade log
    lines.append("  TRADE LOG")
    lines.append("  " + "-" * 115)
    header = (f"  {'#':<3} {'Time':<13} {'Symbol':<22} {'Dir':<5} "
              f"{'Entry':>7} {'Exit':>7} {'Pts':>7} {'Net Rs':>9} "
              f"{'Hold':>5} {'Reason':<22} {'Conf':>6} {'SessReg':<20} {'Result'}")
    lines.append(header)
    lines.append("  " + "-" * 115)

    running = 0.0
    for i, t in enumerate(trades, 1):
        is_open = t.get('pnl') is None
        pnl     = float_val(t, 'pnl')
        pts     = float_val(t, 'pnl_pts')
        entry   = float_val(t, 'entry_price')
        exit_p  = float_val(t, 'exit_price') if not is_open else 0.0
        hold    = int_val(t, 'hold_mins')
        conf    = float_val(t, 'avg_conf')
        symbol  = t.get('symbol', '')[:22]
        dirn    = t.get('direction', '')
        reason  = t.get('exit_reason', '')[:22]
        etime   = t.get('entry_time', '')
        xtime   = t.get('exit_time', '')
        contracts = int_val(t, 'contracts', 1)
        sess_reg  = t.get('session_regime', '')
        sess_sc   = float_val(t, 'session_regime_score')
        sess_str  = f"{sess_reg}({sess_sc:.2f})" if sess_reg else 'N/A'

        if is_open:
            result   = 'OPEN'
            pnl_str  = '     OPEN'
            pts_str  = '    ---'
            exit_str = '   ---'
            hold_str = ' ---'
        else:
            result   = 'WIN ' if pnl > 0 else 'LOSS'
            pnl_str  = f"{pnl:>+9.2f}"
            pts_str  = f"{pts:>+7.2f}"
            exit_str = f"{exit_p:>7.2f}"
            hold_str = '<1m' if hold == 0 else f'{hold:>3}m'
            running += pnl

        lines.append(
            f"  {i:<3} {etime}-{xtime:<7} {symbol:<22} {dirn:<5} {contracts}L "
            f"{entry:>7.2f} {exit_str} {pts_str} {pnl_str} "
            f" {hold_str} {reason:<22} {conf:>5.1%} {sess_str:<20} {result}  "
            f"(running: Rs {running:+,.0f})"
        )

    lines.append("  " + "-" * 115)
    lines.append(f"  {'TOTAL closed (net)':<95} {total_pnl:>+9.2f}")
    lines.append("")

    # Verdict
    # Go-live threshold: WR>=55% AND P&L>0 AND profit factor>=1.5 over 60+ CLOSED trades.
    # At 25-30 trades, 95% CI for true WR is ±15pp — statistically too noisy.
    # Only declare edge after 60+ closed trades.
    GO_LIVE_TRADES = 60
    lines.append("  VERDICT")
    lines.append("  " + "-" * 40)
    if win_rate >= 0.55 and total_pnl > 0 and pf >= 1.5:
        if n_closed >= GO_LIVE_TRADES:
            lines.append(f"  GO-LIVE READY. WR>=55%, P&L positive, PF>=1.5 over {n_closed} trades.")
        else:
            lines.append(f"  TRENDING POSITIVE but only {n_closed} closed trades — need {GO_LIVE_TRADES} for go-live decision.")
    elif win_rate >= 0.55 and total_pnl > 0:
        lines.append("  ACCEPTABLE. WR and P&L positive but profit factor below 1.5 — winners too small.")
    elif total_pnl > 0 and win_rate < 0.50:
        lines.append("  PROFITABLE but low win rate — large wins offsetting losses.")
    elif total_pnl < 0 and win_rate >= 0.50:
        lines.append("  LOSING despite decent win rate — losers bigger than winners.")
    elif total_pnl < 0:
        lines.append("  LOSING. Review entry regime filter (TRENDING_CONFIRMED) and stop placement.")
    else:
        lines.append("  BREAKEVEN.")
    lines.append(f"  Go-live criteria (60+ trades): WR>=55% ({win_rate:.1%}) | P&L>0 ({total_pnl:+.0f}) | PF>=1.5 ({pf:.2f}) | Trades={n_closed}/{GO_LIVE_TRADES}")

    lines.append(sep)
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Source: logs/trades_{report_date}.jsonl  (live JSONL — no restart loss)")
    lines.append(sep)

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(target_date: str = None, all_dates: bool = False):
    os.makedirs(REPORT_DIR, exist_ok=True)

    if all_dates:
        pattern = os.path.join(LOG_DIR, 'trades_*.jsonl')
        files   = sorted(glob.glob(pattern))
        if not files:
            print(f"No trade log files found in {LOG_DIR}/")
            return

        all_trades  = []
        date_range  = []
        daily_pnls  = {}

        for f in files:
            fname = os.path.basename(f)               # trades_2026-03-02.jsonl
            d     = fname.replace('trades_', '').replace('.jsonl', '')
            rows  = load_jsonl(f)
            all_trades.extend(rows)
            date_range.append(d)
            daily_pnls[d] = sum(float_val(t, 'pnl') for t in rows)

        label = f"{date_range[0]} to {date_range[-1]}"
        report_text = build_report(all_trades, label)

        # Per-day P&L table
        extra = ["\n  DAILY P&L BREAKDOWN", "  " + "-" * 40]
        running = CAPITAL
        for d in date_range:
            pnl = daily_pnls.get(d, 0)
            nt  = sum(1 for t in all_trades if True)  # count handled above
            running += pnl
            extra.append(f"  {d}  {pnl:>+10,.2f}  equity={running:>12,.2f}")
        extra.append("  " + "=" * 40)
        extra.append(f"  TOTAL  {sum(daily_pnls.values()):>+10,.2f}")
        report_text += '\n' + '\n'.join(extra)

        out_path = os.path.join(REPORT_DIR, 'paper_report_ALL.txt')

    else:
        if target_date is None:
            target_date = date.today().strftime('%Y-%m-%d')
        jsonl_path  = os.path.join(LOG_DIR, f'trades_{target_date}.jsonl')
        trades      = load_jsonl(jsonl_path)
        report_text = build_report(trades, target_date)
        out_path    = os.path.join(REPORT_DIR, f'paper_report_{target_date}.txt')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {out_path}")


if __name__ == '__main__':
    args = sys.argv[1:]
    # Parse --capital argument
    if '--capital' in args:
        idx = args.index('--capital')
        if idx + 1 < len(args):
            try:
                CAPITAL = float(args[idx + 1])
                args = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]
            except ValueError:
                pass

    if '--all' in args:
        run(all_dates=True)
    elif args:
        run(target_date=args[0])
    else:
        run()
