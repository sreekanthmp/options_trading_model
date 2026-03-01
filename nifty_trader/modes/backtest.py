"""Backtest mode: load saved models and display walk-forward metrics."""
import logging

from ..models.trainer import load_all
from .dashboard import print_training_summary

logger = logging.getLogger(__name__)


def run_backtest():
    """Load saved models and show backtest summary."""
    models, regime_det = load_all()
    if models:
        print_training_summary(models)
    else:
        print("No saved models found. Run --mode train first.")
