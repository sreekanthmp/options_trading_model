"""Feature stability: Feature Stability Index (FSI) and live drift detection."""
import logging
from datetime import datetime
from collections import deque
import numpy as np
import pandas as pd

from ..config import FSI_SIGMA_THRESH, _training_feature_stats

logger = logging.getLogger(__name__)

def compute_fsi(df_features: pd.DataFrame,
                feature_cols: list,
                baseline_rows: int,
                window_rows: int) -> dict:
    """
    Feature Stability Index (FSI) (Req 4).
    Compare rolling window distribution vs long-term baseline.
    Returns {feature: drift_sigma} for each feature.
    A value > FSI_SIGMA_THRESH means the feature has drifted and should be
    down-weighted 50% in the meta-learner.
    """
    fsi = {}
    if len(df_features) < baseline_rows + window_rows:
        return {f: 0.0 for f in feature_cols}

    baseline = df_features[feature_cols].iloc[:baseline_rows]
    recent   = df_features[feature_cols].iloc[-window_rows:]

    bl_mean = baseline.mean()
    bl_std  = baseline.std().replace(0, np.nan)

    rec_mean = recent.mean()
    # Drift = |delta_mean| / baseline_std (z-score of mean shift)
    drift = ((rec_mean - bl_mean) / bl_std).abs()
    for f in feature_cols:
        fsi[f] = float(drift.get(f, 0.0)) if not np.isnan(drift.get(f, 0.0)) else 0.0
    return fsi



class LiveFeatureDriftDetector:
    """
    Monitor feature drift in real-time during live trading.
    
    WHY: If live feature distributions deviate >2.5σ from training baseline,
         model predictions become unreliable (out-of-sample breakdown).
    
    Example: ATR suddenly 3x training mean due to event risk -> signals invalid.
    """
    
    def __init__(self, training_stats: dict = None):
        """
        Args:
            training_stats: {feature_name: {'mean': float, 'std': float}}
        """
        self.baseline = training_stats or {}
        self.drift_history = deque(maxlen=100)  # Keep last 100 drift checks
    
    def set_baseline(self, df_training: pd.DataFrame, feature_cols: list):
        """Compute baseline statistics from training data."""
        self.baseline = {}
        for feat in feature_cols:
            if feat in df_training.columns:
                vals = df_training[feat].dropna()
                if len(vals) > 0:
                    self.baseline[feat] = {
                        'mean': float(vals.mean()),
                        'std': float(vals.std()) + 1e-9
                    }
    
    def check_drift(self, live_row: pd.Series, threshold: float = 2.5) -> dict:
        """
        Check for feature drift in current live bar.
        
        Args:
            live_row: Current 1-min bar features
            threshold: Z-score threshold for drift alert (default 2.5σ)
        
        Returns:
            dict of {feature: z_score} for features that drifted >threshold
        """
        drifted = {}
        
        for feat, stats in self.baseline.items():
            if feat not in live_row:
                continue
            
            val = live_row[feat]
            if pd.isna(val):
                continue
            
            z_score = (val - stats['mean']) / stats['std']
            
            if abs(z_score) > threshold:
                drifted[feat] = float(z_score)
        
        # Log drift events
        if drifted:
            self.drift_history.append({
                'time': datetime.now(),
                'drifted_features': list(drifted.keys()),
                'max_z': max(abs(z) for z in drifted.values())
            })
            logger.warning(f"[Drift] {len(drifted)} features drifted >{threshold}σ: {list(drifted.keys())[:5]}")
        
        return drifted
    
    def get_drift_severity(self) -> float:
        """
        Return drift severity score (0-1) based on recent history.
        
        Returns:
            0.0 = no drift, 1.0 = severe persistent drift
        """
        if len(self.drift_history) == 0:
            return 0.0
        
        # Count drift events in last 20 bars
        recent = list(self.drift_history)[-20:]
        drift_pct = len(recent) / 20.0
        
        # Weighted by max z-score severity
        avg_severity = np.mean([d['max_z'] for d in recent]) / 5.0  # normalize to ~1.0
        
        return min(1.0, drift_pct * avg_severity)

