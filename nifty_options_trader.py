"""
NIFTY Intraday Options Trading System  (Quant Pro v4.0 — Zero Lag + Alpha Boost)
==================================================================================
Entry point — delegates to the nifty_trader package.

Usage:
    python nifty_options_trader.py --mode train
    python nifty_options_trader.py --mode backtest
    python nifty_options_trader.py --mode live
    python nifty_options_trader.py --mode paper
    python nifty_options_trader.py --mode dashboard
    python nifty_options_trader.py --mode train_live
    python nifty_options_trader.py --mode dashboard --verbose

All logic lives in nifty_trader/:
    config.py               Constants, thresholds, paths
    data/                   CSV loader, WebSocket streamer, external data
    features/               Indicators, feature engineering, fracdiff, FSI
    labels/                 Triple-barrier labeling
    regimes/                HMM regime detection
    models/                 Base models, ensemble, walk-forward trainer
    signals/                Signal generator, confidence gates, analysis
    execution/              Costs, risk/kill-switch, orders, paper trader
    modes/                  train / backtest / live / paper / dashboard runners
    main.py                 CLI argument parsing
"""

from nifty_trader import main

if __name__ == "__main__":
    main()
