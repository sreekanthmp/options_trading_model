"""Base model builders: _ScaledLR wrapper and voting ensemble."""
import logging
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from ..config import SKLEARN_OK, LGB_OK
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

logger = logging.getLogger(__name__)


class _ScaledLR(BaseEstimator, ClassifierMixin):
    """LogisticRegression with internal StandardScaler.
    Inherits BaseEstimator + ClassifierMixin and overrides __sklearn_tags__
    for sklearn >= 1.6 compatibility (Tags API replaced _estimator_type check).
    Accepts sample_weight in fit() so VotingClassifier can forward it correctly.
    """
    def __init__(self, C=1.0, max_iter=500, random_state=None):
        self.C           = C
        self.max_iter    = max_iter
        self.random_state= random_state

    def __sklearn_tags__(self):
        # sklearn 1.6+ uses get_tags() which calls this method.
        # Setting estimator_type = 'classifier' makes is_classifier() return True.
        tags = super().__sklearn_tags__()
        tags.estimator_type = 'classifier'
        return tags

    def fit(self, X, y, sample_weight=None):
        self._sc = StandardScaler()
        self._cl = LogisticRegression(C=self.C, max_iter=self.max_iter,
                                      random_state=self.random_state)
        Xs = self._sc.fit_transform(X)
        self._cl.fit(Xs, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self._cl.predict(self._sc.transform(X))

    def predict_proba(self, X):
        return self._cl.predict_proba(self._sc.transform(X))


def make_base_model():
    """Level-0 ensemble: GBM + RF + LR with soft voting."""
    gbm = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, min_samples_leaf=50, random_state=42)
    rf  = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=50,
        n_jobs=-1, random_state=42)
    lr  = _ScaledLR(C=0.1, max_iter=500, random_state=42)
    return VotingClassifier(
        estimators=[('gbm', gbm), ('rf', rf), ('lr', lr)],
        voting='soft', n_jobs=1)


