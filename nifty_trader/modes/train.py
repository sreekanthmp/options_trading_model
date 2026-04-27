"""Training mode: build dataset, train all horizons, save models."""
import logging

from ..config import HORIZONS
from ..data.loader import load_all_data
from ..features.feature_engineering import (add_1min_features, add_htf_features, add_daily_features,
                                             add_vix_features, add_options_chain_features,
                                             compute_options_chain_features,
                                             add_calendar_features, add_futures_basis_features,
                                             add_fii_dii_features, add_global_market_features,
                                             add_pcr_volume_features, compute_pcr_volume_features)
from ..labels.triple_barrier import triple_barrier_labels
from ..regimes.hmm_regime import RegimeDetector
from ..models.trainer import train_horizon, save_all
from .dashboard import print_training_summary

logger = logging.getLogger(__name__)


def build_dataset():
    """Load data and engineer all features + labels. Returns (df, regime_det)."""
    df1m, df5m, df15m, df1d = load_all_data()

    print("Building 1-min features...")
    df = add_1min_features(df1m)

    print("Adding HTF features...")
    df = add_htf_features(df, df5m, 'tf5_', [1, 3, 6])
    df = add_htf_features(df, df15m, 'tf15_', [1, 4])

    print("Adding daily features...")
    df = add_daily_features(df, df1d)

    print("Adding India VIX features...")
    df = add_vix_features(df)

    print("Adding options chain features (PCR OI, max pain, IV skew)...")
    options_df = compute_options_chain_features('nifty_options_data')
    df = add_options_chain_features(df, options_df)

    print("Adding PCR volume + ATM IV features...")
    pcr_df = compute_pcr_volume_features('nifty_options_data')
    df = add_pcr_volume_features(df, pcr_df=pcr_df)

    print("Adding calendar features (day-of-week, expiry-week)...")
    df = add_calendar_features(df)

    print("Adding Nifty futures basis features...")
    df = add_futures_basis_features(df)

    print("Adding FII/DII flow features...")
    df = add_fii_dii_features(df)

    print("Adding global market features (S&P 500)...")
    df = add_global_market_features(df)

    print("Fitting regime detector...")
    regime_det = RegimeDetector()
    regime_det.fit(df1d)

    return df, df1d, regime_det


def run_train(capital=10000):
    """Train all horizon models and save."""
    df, df1d, regime_det = build_dataset()

    regime_series = regime_det.predict(df1d)

    print("\nGenerating triple-barrier labels...")
    for h in HORIZONS:
        print(f"  Labeling {h}-min horizon...")
        df = triple_barrier_labels(df, h, regime_series)
        lc, bc = f'label_{h}m', f'barrier_{h}m'
        print(f"    barrier=+1:{(df[bc]==1).sum():,}  barrier=-1:{(df[bc]==-1).sum():,}  "
              f"barrier=0:{(df[bc]==0).sum():,}  labeled:{df[lc].notna().sum():,}")

    models = {}
    for h in HORIZONS:
        print(f"\nTraining {h}-min horizon...")
        result = train_horizon(df, h, regime_series)
        models[h] = result

    save_all(models, regime_det, df_train_full=df)
    print_training_summary(models)
    return models, regime_det
