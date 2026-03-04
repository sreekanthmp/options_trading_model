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
CAPITAL    = 10000.0


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

    trades = []
    open_entry = None

    for ev in events:
        etype = ev.get('event', '')

        if etype == 'ENTRY':
            open_entry = ev

        elif etype == 'EXIT' and open_entry is not None:
            entry_px = float(open_entry.get('entry_price', 0))
            exit_px  = float(ev.get('exit_price', 0))
            contracts = int(open_entry.get('contracts', 1))
            qty       = contracts * LOT_SIZE

            # PnL: we BUY the option, so profit = exit - entry regardless of direction
            pnl_pts   = exit_px - entry_px
            pnl_rs    = pnl_pts * qty

            # Approximate brokerage (Angel One flat-fee model)
            # Rs20×2 orders + 18% GST + STT(sell) + NSE txn + stamp duty
            brokerage   = 20.0 * 2
            gst         = brokerage * 0.18
            stt         = exit_px * qty * 0.001
            txn         = (entry_px + exit_px) * qty * 0.00053
            stamp       = entry_px * qty * 0.00003
            total_chrg  = brokerage + gst + stt + txn + stamp
            net_pnl_rs  = pnl_rs - total_chrg

            entry_ts = open_entry.get('timestamp_signal', '')
            exit_ts  = ev.get('timestamp_exit', '')
            try:
                et = datetime.fromisoformat(entry_ts)
                xt = datetime.fromisoformat(exit_ts)
                hold_mins = max(0, int((xt - et).total_seconds() / 60))
                entry_time = et.strftime('%H:%M')
                exit_time  = xt.strftime('%H:%M')
            except Exception:
                hold_mins  = 0
                entry_time = entry_ts[11:16] if len(entry_ts) > 15 else ''
                exit_time  = exit_ts[11:16]  if len(exit_ts)  > 15 else ''

            trades.append({
                'direction':    open_entry.get('direction', '?'),
                'option_type':  open_entry.get('option_type', '?'),
                'strike':       open_entry.get('strike', 0),
                'symbol':       f"NIFTY {open_entry.get('strike','')} {open_entry.get('option_type','')}",
                'entry_price':  entry_px,
                'exit_price':   exit_px,
                'pnl_pts':      round(pnl_pts, 2),
                'pnl':          round(net_pnl_rs, 2),
                'gross_pnl':    round(pnl_rs, 2),
                'total_charges':round(total_chrg, 2),
                'contracts':    contracts,
                'qty':          qty,
                'hold_mins':    hold_mins,
                'entry_time':   entry_time,
                'exit_time':    exit_time,
                'exit_reason':  ev.get('exit_reason', 'UNKNOWN'),
                'avg_conf':     float(open_entry.get('model_confidence', 0)),
                'regime':       open_entry.get('regime', -1),
                'minute_of_day':int(open_entry.get('minute_of_day', 0)),
            })
            open_entry = None

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

    n          = len(trades)
    wins       = [t for t in trades if float_val(t, 'pnl') > 0]
    losses     = [t for t in trades if float_val(t, 'pnl') <= 0]
    total_pnl  = sum(float_val(t, 'pnl')       for t in trades)
    total_gross= sum(float_val(t, 'gross_pnl')  for t in trades)
    total_chrg = sum(float_val(t, 'total_charges') for t in trades)
    win_pnl    = sum(float_val(t, 'pnl') for t in wins)
    loss_pnl   = sum(float_val(t, 'pnl') for t in losses)
    win_rate   = len(wins) / n if n > 0 else 0
    pf         = abs(win_pnl / loss_pnl) if loss_pnl != 0 else float('inf')
    avg_win    = win_pnl  / len(wins)   if wins   else 0
    avg_loss   = loss_pnl / len(losses) if losses else 0
    avg_hold   = sum(int_val(t, 'hold_mins') for t in trades) / n
    ret_pct    = total_pnl / CAPITAL * 100
    equity_end = CAPITAL + total_pnl

    buys  = [t for t in trades if t.get('direction', '') == 'UP']
    sells = [t for t in trades if t.get('direction', '') == 'DOWN']
    ce_tr = [t for t in trades if t.get('option_type', '') == 'CE']
    pe_tr = [t for t in trades if t.get('option_type', '') == 'PE']

    reasons: dict = {}
    for t in trades:
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
    lines.append(f"  Total trades      : {n}")
    lines.append(f"  UP  (CE buys)     : {len(buys)}  ({len(ce_tr)} CE lots)")
    lines.append(f"  DOWN (PE buys)    : {len(sells)}  ({len(pe_tr)} PE lots)")
    lines.append(f"  Winners           : {len(wins)}")
    lines.append(f"  Losers            : {len(losses)}")
    lines.append(f"  Win rate          : {win_rate:.1%}")
    lines.append(f"  Profit factor     : {pf:.2f}")
    lines.append(f"  Avg win           : Rs {avg_win:>+10,.2f}")
    lines.append(f"  Avg loss          : Rs {avg_loss:>+10,.2f}")
    lines.append(f"  Avg hold time     : {avg_hold:.0f} min")
    lines.append("")

    # Exit reasons
    lines.append("  EXIT REASONS")
    lines.append("  " + "-" * 40)
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        lines.append(f"  {reason:<28} : {count:>3}  ({count/n:.0%})")
    lines.append("")

    # Per-trade log
    lines.append("  TRADE LOG")
    lines.append("  " + "-" * 100)
    header = (f"  {'#':<3} {'Time':<13} {'Symbol':<22} {'Dir':<5} "
              f"{'Entry':>7} {'Exit':>7} {'Pts':>7} {'Net Rs':>9} "
              f"{'Hold':>5} {'Reason':<22} {'Conf':>6} {'Result'}")
    lines.append(header)
    lines.append("  " + "-" * 100)

    running = 0.0
    for i, t in enumerate(trades, 1):
        pnl    = float_val(t, 'pnl')
        pts    = float_val(t, 'pnl_pts')
        entry  = float_val(t, 'entry_price')
        exit_p = float_val(t, 'exit_price')
        hold   = int_val(t, 'hold_mins')
        conf   = float_val(t, 'avg_conf')
        symbol = t.get('symbol', '')[:22]
        dirn   = t.get('direction', '')
        reason = t.get('exit_reason', '')[:22]
        etime  = t.get('entry_time', '')
        xtime  = t.get('exit_time', '')
        result = 'WIN ' if pnl > 0 else 'LOSS'
        running += pnl

        lines.append(
            f"  {i:<3} {etime}-{xtime:<7} {symbol:<22} {dirn:<5} "
            f"{entry:>7.2f} {exit_p:>7.2f} {pts:>+7.2f} {pnl:>+9.2f} "
            f"{hold:>4}m {reason:<22} {conf:>5.1%} {result}  "
            f"(running: Rs {running:+,.0f})"
        )

    lines.append("  " + "-" * 100)
    lines.append(f"  {'TOTAL (net)':<80} {total_pnl:>+9.2f}")
    lines.append("")

    # Verdict
    lines.append("  VERDICT")
    lines.append("  " + "-" * 40)
    if win_rate >= 0.60 and total_pnl > 0:
        lines.append("  GOOD DAY. Win rate and P&L both positive.")
    elif total_pnl > 0 and win_rate < 0.50:
        lines.append("  PROFITABLE but low win rate — big wins offsetting losses.")
    elif total_pnl < 0 and win_rate >= 0.50:
        lines.append("  LOSING DAY despite decent win rate — losers bigger than winners.")
    elif total_pnl < 0:
        lines.append("  LOSING DAY. Review low-confidence signals and stop placement.")
    else:
        lines.append("  BREAKEVEN DAY.")

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
    if '--all' in args:
        run(all_dates=True)
    elif args:
        run(target_date=args[0])
    else:
        run()
