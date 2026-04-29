"""Meta model builder and MetaLabeler (Lopez de Prado meta-labeling)."""
import logging
import numpy as np
import pandas as pd
import joblib
import warnings

from nifty_trader.models.base_models import _ScaledLR
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from ..config import SKLEARN_OK, LGB_OK
try:
    import lightgbm as lgb
except ImportError:
    pass

logger = logging.getLogger(__name__)

def make_meta_model():
    """Level-1 stacker: LightGBM on [base_proba, regime, session, iv, adx]."""
    if LGB_OK:
        return lgb.LGBMClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=30, random_state=42, verbose=-1)
    else:
        return _ScaledLR(C=1.0, max_iter=500)


# ==============================================================================
# 5b. META-LABELER (Lopez de Prado Meta-Labeling)
# ==============================================================================

class MetaLabeler:
    """
    Meta-labeling (Lopez de Prado, AFML Chapter 10).

    Two-stage prediction:
      Stage 1 (Primary model): Predicts direction (UP=1 / DOWN=0).
                               This is the existing `final_model` per horizon.
      Stage 2 (Meta model):    Predicts whether Stage 1 is CORRECT (1) or WRONG (0).

    In live use:
      - Get primary direction from Stage 1.
      - Get correctness probability from Stage 2.
      - Only take the trade if Stage 2 probability >= META_CONF_THRESH.
      - This mathematically boosts accuracy because we only trade when
        the meta-learner agrees the primary signal is reliable.

    Features for Stage 2 include:
      - Stage 1 output probability (the primary model's raw confidence)
      - Market-context features: regime, session, iv, adx, tick_imbalance,
        vwap_dev_vel, pressure_ratio, bb_squeeze
    """

    META_CONF_THRESH = 0.62   # raised from 0.50: filter requires genuine meta-labeler conviction
    META_FEATURES = [
        'proba_primary', 'regime', 'session_pct', 'iv_proxy',
        'atr_14_pct', 'adx_14', 'dmi_diff',
        'tick_imbalance', 'vwap_dev_vel', 'pressure_ratio',
        'bb_squeeze', 'bb_width', 'adx_rsi_trend', 'vol_ratio',
    ]

    def __init__(self):
        self.model = None
        self._fitted = False

    def fit(self, X_primary: np.ndarray, y_primary: np.ndarray,
            oof_probas: np.ndarray, context_df: pd.DataFrame,
            regimes: np.ndarray, sample_weight: np.ndarray = None):
        """
        Train meta-labeler on OOF predictions from primary model.

        y_meta = 1 if primary model was correct (oof_proba > 0.5 == y_primary)
               = 0 otherwise
        """
        # Step 10 (v3.3): MetaLabeler leakage guard.
        # context_df must be strictly forward-ordered (integer index, no shuffle).
        # Only OOF rows (rows where oof_proba is not NaN) are used; these are
        # already past signal bars from the walk-forward folds so no future
        # data is available to the meta-learner.  Enforce the ordering contract.
        context_df = context_df.reset_index(drop=True)
        if len(context_df) > 0 and 'datetime' in context_df.columns:
            assert context_df['datetime'].is_monotonic_increasing, (
                "MetaLabeler.fit(): context_df is not time-ordered — "
                "possible future-data leakage!"
            )

        valid = ~np.isnan(oof_probas)
        if valid.sum() < 200:
            return

        oof_preds = (oof_probas[valid] > 0.5).astype(int)
        y_meta    = (oof_preds == y_primary[valid]).astype(int)

        if len(np.unique(y_meta)) < 2:
            return   # all correct or all wrong -- degenerate

        # Build meta-feature matrix
        ctx = context_df.iloc[np.where(valid)[0]].reset_index(drop=True)
        Xmeta_parts = [oof_probas[valid].reshape(-1, 1),
                       regimes[valid].reshape(-1, 1)]
        for feat in self.META_FEATURES[2:]:   # skip proba_primary, regime (already added)
            col_vals = ctx[feat].values.reshape(-1, 1) if feat in ctx.columns \
                       else np.zeros((valid.sum(), 1))
            Xmeta_parts.append(col_vals)
        Xmeta = np.hstack(Xmeta_parts).astype(np.float32)

        wmeta = sample_weight[valid] if sample_weight is not None else None

        if LGB_OK:
            meta_clf = lgb.LGBMClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=20, random_state=42, verbose=-1)
        else:
            meta_clf = _ScaledLR(C=0.5, max_iter=500, random_state=42)

        try:
            meta_clf.fit(Xmeta, y_meta, sample_weight=wmeta)
        except TypeError:
            meta_clf.fit(Xmeta, y_meta)

        self.model   = meta_clf
        self._fitted = True

        # Report meta-label statistics
        wr_meta = y_meta.mean()
        print(f"  MetaLabeler: {valid.sum():,} OOF samples  "
              f"primary_correct={wr_meta:.1%}  "
              f"meta features={len(self.META_FEATURES)}")

    def predict_proba(self, proba_primary: float, context_row: pd.Series,
                      regime: int) -> float:
        """
        Returns probability that primary model is correct for this signal.
        Returns 0.55 (neutral pass) if not fitted or on error.
        """
        if not self._fitted or self.model is None:
            return 0.55  # exactly at threshold: meta-labeler not fitted, don't block but don't boost

        parts = [np.array([[proba_primary]]),
                 np.array([[float(regime)]])]
        for feat in self.META_FEATURES[2:]:
            val = context_row.get(feat, 0.0)
            # Handle NaN values
            val = 0.0 if pd.isna(val) else float(val)
            parts.append(np.array([[val]]))
        Xmeta = np.hstack(parts).astype(np.float32)

        try:
            p = self.model.predict_proba(Xmeta)[0]
            return float(p[1])   # probability that primary is correct
        except Exception:
            return 0.55  # exactly at threshold: prediction error, don't block but don't boost

    def save(self, path: str):
        joblib.dump({'model': self.model, 'fitted': self._fitted}, path)

    def load(self, path: str):
        d = joblib.load(path)
        self.model   = d['model']
        self._fitted = d['fitted']
        return self


