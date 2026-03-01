"""
Paper Trading Report Generator
================================
Reads paper_trades/paper_YYYY-MM-DD.csv files and writes a clean
human-readable summary to reports/paper_report_YYYY-MM-DD.txt

Usage:
    python paper_report.py                  # today's report
    python paper_report.py 2026-03-01       # specific date
    python paper_report.py --all            # all available dates combined
"""

import os
import sys
import csv
import glob
from datetime import datetime, date


PAPER_DIR  = 'paper_trades'
REPORT_DIR = 'reports'


def load_csv(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


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


def build_report(trades: list, report_date: str, capital: float = 10000.0) -> str:
    lines = []
    sep   = '=' * 68

    if not trades:
        lines.append(sep)
        lines.append(f"  PAPER TRADE REPORT  --  {report_date}")
        lines.append(sep)
        lines.append("  No trades executed on this date.")
        lines.append(sep)
        return '\n'.join(lines)

    # ---------------------------------------------------------------
    # Basic counts
    # ---------------------------------------------------------------
    n          = len(trades)
    buys       = [t for t in trades if t.get('direction', '').upper() == 'UP']
    sells      = [t for t in trades if t.get('direction', '').upper() == 'DOWN']
    wins       = [t for t in trades if float_val(t, 'pnl') > 0]
    losses     = [t for t in trades if float_val(t, 'pnl') <= 0]
    total_pnl  = sum(float_val(t, 'pnl') for t in trades)
    win_pnl    = sum(float_val(t, 'pnl') for t in wins)
    loss_pnl   = sum(float_val(t, 'pnl') for t in losses)
    win_rate   = len(wins) / n if n > 0 else 0
    pf         = abs(win_pnl / loss_pnl) if loss_pnl != 0 else float('inf')
    avg_win    = win_pnl  / len(wins)   if wins   else 0
    avg_loss   = loss_pnl / len(losses) if losses else 0
    avg_hold   = sum(int_val(t, 'hold_mins') for t in trades) / n
    ret_pct    = total_pnl / capital * 100
    equity_end = capital + total_pnl

    # cost components
    total_theta    = sum(float_val(t, 'theta_drag_pnl')    for t in trades)
    total_gross    = sum(float_val(t, 'gross_pnl', float_val(t, 'pnl')) for t in trades)
    total_charges  = sum(float_val(t, 'total_charges')     for t in trades)
    total_spread   = sum(float_val(t, 'total_spread_cost') for t in trades)
    total_brok     = sum(float_val(t, 'brokerage')         for t in trades)
    total_stt      = sum(float_val(t, 'stt')               for t in trades)
    real_ltp_count = sum(1 for t in trades if t.get('ltp_source') == 'REAL')
    avg_e_slip     = (sum(float_val(t, 'entry_slip_pct') for t in trades) / n) if n else 0
    avg_x_slip     = (sum(float_val(t, 'exit_slip_pct')  for t in trades) / n) if n else 0

    # CE vs PE
    ce_trades  = [t for t in trades if t.get('option_type', '') == 'CE']
    pe_trades  = [t for t in trades if t.get('option_type', '') == 'PE']

    # exit reasons
    reasons: dict = {}
    for t in trades:
        r = t.get('exit_reason', 'UNKNOWN')
        reasons[r] = reasons.get(r, 0) + 1

    # ---------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------
    lines.append(sep)
    lines.append(f"  PAPER TRADE REPORT  --  {report_date}")
    lines.append(sep)

    # ---------------------------------------------------------------
    # Summary block
    # ---------------------------------------------------------------
    lines.append("")
    lines.append("  SUMMARY")
    lines.append("  " + "-" * 40)
    lines.append(f"  Starting capital  : Rs {capital:>12,.2f}")
    lines.append(f"  Ending equity     : Rs {equity_end:>12,.2f}")
    lines.append(f"  Gross P&L         : Rs {total_gross:>+12,.2f}  (before all costs)")
    lines.append(f"  Brokerage + STT   : Rs {-total_charges:>+12,.2f}  (Angel One real charges)")
    lines.append(f"  Bid-ask spread    : Rs {-total_spread:>+12,.2f}  (execution friction)")
    lines.append(f"  Theta drag        : Rs {total_theta:>+12,.2f}  (time decay)")
    lines.append(f"  Net P&L (live eq) : Rs {total_pnl:>+12,.2f}  ({ret_pct:+.2f}%)  <- live equivalent")
    lines.append(f"  LTP source        : {real_ltp_count}/{n} trades used real API price")
    lines.append(f"  Avg slippage      : entry {avg_e_slip:.3f}%  exit {avg_x_slip:.3f}%")
    lines.append("")

    # ---------------------------------------------------------------
    # Trade stats
    # ---------------------------------------------------------------
    lines.append("  TRADE STATISTICS")
    lines.append("  " + "-" * 40)
    lines.append(f"  Total trades      : {n}")
    lines.append(f"  BUY  signals (CE) : {len(buys)}  ({len(ce_trades)} CE lots traded)")
    lines.append(f"  SELL signals (PE) : {len(sells)}  ({len(pe_trades)} PE lots traded)")
    lines.append(f"  Winners           : {len(wins)}")
    lines.append(f"  Losers            : {len(losses)}")
    lines.append(f"  Win rate          : {win_rate:.1%}")
    lines.append(f"  Profit factor     : {pf:.2f}  (total wins / total losses)")
    lines.append(f"  Avg win           : Rs {avg_win:>+10,.2f}")
    lines.append(f"  Avg loss          : Rs {avg_loss:>+10,.2f}")
    lines.append(f"  Avg hold time     : {avg_hold:.0f} min")
    lines.append("")

    # ---------------------------------------------------------------
    # Exit reason breakdown
    # ---------------------------------------------------------------
    lines.append("  EXIT REASONS")
    lines.append("  " + "-" * 40)
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = count / n * 100
        lines.append(f"  {reason:<28} : {count:>3}  ({pct:.0f}%)")
    lines.append("")

    # ---------------------------------------------------------------
    # Per-trade detail
    # ---------------------------------------------------------------
    lines.append("  TRADE LOG  (Net PnL = Gross - Brokerage - Spread)")
    lines.append("  " + "-" * 116)
    header = (f"  {'#':<3} {'Time':<13} {'Symbol':<22} {'Dir':<5} "
              f"{'Entry':>7} {'Fill':>7} {'Gross':>9} {'Chrg':>7} {'Net':>9} "
              f"{'Hold':>5} {'LTP':>5} {'Reason':<18} {'Conf':>6} {'Result'}")
    lines.append(header)
    lines.append("  " + "-" * 116)

    running_pnl = 0.0
    for i, t in enumerate(trades, 1):
        pnl      = float_val(t, 'pnl')
        gross    = float_val(t, 'gross_pnl', pnl)
        charges  = float_val(t, 'total_charges')
        entry    = float_val(t, 'entry_price')
        exit_p   = float_val(t, 'exit_price')
        hold     = int_val(t, 'hold_mins')
        conf     = float_val(t, 'avg_conf')
        symbol   = t.get('symbol', '')[:22]
        direction= t.get('direction', '')
        reason   = t.get('exit_reason', '')[:18]
        etime    = t.get('entry_time', '')
        xtime    = t.get('exit_time', '')
        ltp_src  = 'REAL' if t.get('ltp_source') == 'REAL' else 'MODL'
        result   = 'WIN' if pnl > 0 else 'LOSS'
        running_pnl += pnl

        lines.append(
            f"  {i:<3} {etime}-{xtime:<7} {symbol:<22} {direction:<5} "
            f"{entry:>7.2f} {exit_p:>7.2f} {gross:>+9.2f} {-charges:>+7.2f} {pnl:>+9.2f} "
            f"{hold:>4}m {ltp_src:>5} {reason:<18} {conf:>5.1%} {result}  "
            f"(running: Rs {running_pnl:+,.0f})"
        )

    lines.append("  " + "-" * 116)
    lines.append(f"  {'TOTAL (net)':<84} {total_pnl:>+9.2f}")
    lines.append("")

    # ---------------------------------------------------------------
    # Quick verdict
    # ---------------------------------------------------------------
    lines.append("  VERDICT")
    lines.append("  " + "-" * 40)
    if n == 0:
        lines.append("  No trades — market may have been ranging all day.")
    elif win_rate >= 0.60 and total_pnl > 0:
        lines.append("  GOOD DAY. Win rate and P&L both positive.")
    elif total_pnl > 0 and win_rate < 0.50:
        lines.append("  PROFITABLE but low win rate — big wins saving small losses.")
        lines.append("  Watch position sizing carefully.")
    elif total_pnl < 0 and win_rate >= 0.50:
        lines.append("  LOSING DAY despite decent win rate — losers bigger than winners.")
        lines.append("  Check stop loss placement and theta drag.")
    elif total_pnl < 0:
        lines.append("  LOSING DAY. Review signals with low confidence (<60%).")
    else:
        lines.append("  BREAKEVEN DAY.")

    lines.append(sep)
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)

    return '\n'.join(lines)


def run(target_date: str = None, all_dates: bool = False):
    os.makedirs(REPORT_DIR, exist_ok=True)

    if all_dates:
        # Load all available CSV files
        pattern = os.path.join(PAPER_DIR, 'paper_*.csv')
        files   = sorted(glob.glob(pattern))
        if not files:
            print(f"No paper trade files found in {PAPER_DIR}/")
            return

        all_trades   = []
        date_range   = []
        daily_pnls   = {}

        for f in files:
            fname = os.path.basename(f)                 # paper_2026-02-27.csv
            d     = fname.replace('paper_', '').replace('.csv', '')
            rows  = load_csv(f)
            all_trades.extend(rows)
            date_range.append(d)
            daily_pnls[d] = sum(float_val(t, 'pnl') for t in rows)

        label = f"{date_range[0]} to {date_range[-1]}"
        report_text = build_report(all_trades, label)

        # Append per-day P&L table
        extra = ["\n  DAILY P&L BREAKDOWN", "  " + "-" * 40]
        running = 10000.0
        for d in date_range:
            pnl = daily_pnls.get(d, 0)
            running += pnl
            extra.append(f"  {d}  {pnl:>+10,.2f}  equity={running:>12,.2f}")
        extra.append("  " + "=" * 40)
        extra.append(f"  TOTAL  {sum(daily_pnls.values()):>+10,.2f}")
        report_text += '\n' + '\n'.join(extra)

        out_path = os.path.join(REPORT_DIR, f"paper_report_ALL.txt")

    else:
        if target_date is None:
            target_date = date.today().strftime('%Y-%m-%d')
        csv_path = os.path.join(PAPER_DIR, f"paper_{target_date}.csv")
        trades   = load_csv(csv_path)
        report_text = build_report(trades, target_date)
        out_path = os.path.join(REPORT_DIR, f"paper_report_{target_date}.txt")

    # Write to file
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    # Also print to console
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
