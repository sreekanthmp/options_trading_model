"""Paper trading mode — thin wrapper around live_loop."""
from .live import live_loop


def run_paper(models, regime_det, capital: float = 100_000.0, verbose: bool = False):
    """Run the live loop in paper-trading mode (no real orders)."""
    live_loop(models, regime_det, capital=capital, paper_mode=True, verbose=verbose)
