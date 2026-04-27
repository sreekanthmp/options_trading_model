"""
NIFTY Intraday Options Trading System (Quant Pro v4.0)
Entry point — argument parsing and mode dispatch.

Usage:
    python nifty_options_trader.py --mode train
    python nifty_options_trader.py --mode backtest
    python nifty_options_trader.py --mode live
    python nifty_options_trader.py --mode paper
    python nifty_options_trader.py --mode dashboard
    python nifty_options_trader.py --mode train_live
    
    # Yesterday only (default)
    python angelone/daily_update.py

    # Custom range (e.g. after a long weekend)
    python angelone/daily_update.py --start 2026-02-21 --end 2026-02-26
"""
import argparse
import json
import logging
import os

from .config import SKLEARN_OK

def _read_capital_from_config() -> float:
    try:
        cfg_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
        with open(cfg_path, encoding='utf-8') as f:
            return float(json.load(f).get('capital', 30000.0))
    except Exception:
        return 30000.0

logger = logging.getLogger(__name__)


def main():
    if not SKLEARN_OK:
        print("Install scikit-learn first: pip install scikit-learn")
        return

    parser = argparse.ArgumentParser(description="NIFTY Intraday Options Trading System")
    parser.add_argument("--mode", default="train",
                        choices=["train", "backtest", "live", "train_live", "paper", "dashboard"],
                        help="Operating mode")
    parser.add_argument("--capital", type=float, default=_read_capital_from_config(),
                        help="Starting capital in INR (default: from config.json)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed CE/PE breakdown in dashboard")
    args = parser.parse_args()

    if args.mode in ("train", "train_live"):
        from .modes.train import run_train
        models, regime_det = run_train(capital=args.capital)
        if args.mode == "train_live":
            from .modes.live import run_live
            run_live(models, regime_det, capital=args.capital, verbose=args.verbose)

    elif args.mode == "backtest":
        from .modes.backtest import run_backtest
        run_backtest()

    elif args.mode == "live":
        from .models.trainer import load_all
        from .modes.live import run_live
        models, regime_det = load_all()
        run_live(models, regime_det, capital=args.capital, verbose=args.verbose)

    elif args.mode == "paper":
        from .models.trainer import load_all
        from .modes.live import run_paper
        models, regime_det = load_all()
        run_paper(models, regime_det, capital=args.capital, verbose=args.verbose)

    elif args.mode == "dashboard":
        from .models.trainer import load_all
        from .modes.live import run_dashboard
        models, regime_det = load_all()
        run_dashboard(models, regime_det, capital=args.capital, verbose=args.verbose)


if __name__ == "__main__":
    main()
