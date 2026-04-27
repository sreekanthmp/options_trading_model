"""Walk-forward training pipeline, model persistence (save/load)."""
import os, json, logging
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss

from ..config import (
    CONF_MODERATE, HORIZONS, HORIZON_WEIGHTS,
    REGIME_TRENDING, REGIME_RANGING, REGIME_CRISIS, REGIME_NAMES,
    TB_MULT_TRENDING, TB_MULT_RANGING, TB_BARS, TB_MIN_MOVE_PCT,
    CONF_BY_HORIZON, FSI_SIGMA_THRESH, FSI_WINDOW_DAYS, FSI_BASELINE_YRS,
    DD_CLUSTER_THRESH, DD_SAMPLE_WEIGHT, TRAIN_TIME_DECAY,
    SKLEARN_OK, LGB_OK, MODEL_DIR, COST_RT_PCT,
)
from ..features.feature_engineering import FEATURE_COLS, FEATURE_LIVE_OK, get_feature_cols, get_active_feature_cols
from ..labels.triple_barrier import triple_barrier_labels
from ..regimes.hmm_regime import RegimeDetector
from .base_models import make_base_model
from .ensemble import make_meta_model, MetaLabeler
from ..features.stability import compute_fsi
from ..signals.signal_generator import _signal_state

logger = logging.getLogger(__name__)


class _CalibratedWrapper:
    """
    Platt-scaling (sigmoid) calibrated wrapper for a base classifier.

    Switched from isotonic to sigmoid calibration.
    WHY: isotonic regression interpolates through calibration points directly —
    with only 150-200 samples per fold it memorises the training set, collapsing
    predictions to 0.0/1.0 at feature extremes (RSI=14, ADX=55) even when the
    underlying model is uncertain. Platt scaling fits a 2-parameter sigmoid; it
    cannot memorise the calibration set and produces smooth, distributed probabilities.

    Confidence clip [0.45, 0.82]:
    - Below 0.45: pure noise (model has negative conviction — don't trade)
    - Above 0.82: post-calibration overfit artefact; real directional edge at
      1-min resolution never exceeds ~72% (paper data confirms 100% conf = wrong)
    """
    def __init__(self, base, platt):
        self._base  = base
        self._platt = platt   # sklearn LogisticRegression fit on calibration set
        self.classes_ = base.classes_ if hasattr(base, 'classes_') else np.array([0, 1])

    def predict_proba(self, X):
        raw = self._base.predict_proba(X)[:, 1].reshape(-1, 1)
        # Platt scaling: sigmoid applied via LogisticRegression on raw scores
        cal = self._platt.predict_proba(raw)[:, 1]
        # Hard clip: eliminate overconfidence artefacts and pure-noise signals
        cal = np.clip(cal, 0.45, 0.82)
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# ==============================================================================
# 6. WALK-FORWARD TRAINING PIPELINE
# ==============================================================================

def purge_embargo(train_idx, test_idx, embargo_bars=5):
    """Remove last embargo_bars from train and first embargo_bars from test."""
    train_idx = train_idx[train_idx < test_idx[0] - embargo_bars]
    test_idx  = test_idx[test_idx  > test_idx[0] + embargo_bars]
    return train_idx, test_idx


def _walk_forward_scale(X: np.ndarray, tscv) -> tuple:
    """
    Walk-forward normalization — the ONLY correct approach for time series.

    PROBLEM: fitting a StandardScaler on the full dataset before walk-forward
    CV leaks future statistics (mean, std computed on future folds) into every
    training fold.  This inflates model confidence on normalised features
    (EMA distances, rolling returns, ATR ratios) even when the raw model is
    not lookahead-biased.

    SOLUTION: fit one scaler per fold on the TRAINING portion only, then
    apply to both train and test.  The FINAL scaler (fit on the full dataset
    up to fold 5) is returned for use during live inference — it only ever
    sees data up to the last training sample, matching what will be available
    at deployment time.

    Returns
    -------
    X_scaled   : np.ndarray — same shape as X, scaled per fold
    live_scaler: sklearn StandardScaler fit on data up to last fold boundary
    fold_scalers: list of per-fold scalers (for audit / debugging)
    """
    from sklearn.preprocessing import StandardScaler

    X_scaled     = X.copy().astype(np.float32)
    fold_scalers = []
    live_scaler  = None

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        scaler = StandardScaler()
        scaler.fit(X[tr_idx])          # fit ONLY on training portion of this fold
        X_scaled[tr_idx] = scaler.transform(X[tr_idx])
        X_scaled[te_idx] = scaler.transform(X[te_idx])
        fold_scalers.append(scaler)
        live_scaler = scaler            # keep the last fold's scaler for live use

    return X_scaled, live_scaler, fold_scalers


def _time_decay_weights(n: int) -> np.ndarray:
    """
    Exponential time-decay sample weights (Req 8).
    Recent samples receive exponentially higher weight so the model
    prioritises the most recent 3 years of market behaviour.
    w_i = exp(TRAIN_TIME_DECAY * (i - n) / n)  -> normalised to sum=n.
    """
    idx = np.arange(n, dtype=float)
    w   = np.exp(TRAIN_TIME_DECAY * (idx - n) / n)
    return w / w.mean()   # scale so mean weight = 1


def _drawdown_aware_weights(oof_probas: np.ndarray, y: np.ndarray,
                             rets: np.ndarray) -> np.ndarray:
    """
    Drawdown-aware sample weighting (Req 9).
    Build OOF equity curve, identify drawdown clusters, and down-weight
    samples that occur inside those clusters to reduce serial loss risk.
    """
    n  = len(y)
    sw = np.ones(n)
    valid = ~np.isnan(oof_probas)
    if valid.sum() < 100:
        return sw

    # Reconstruct simplified OOF equity curve
    eq = np.zeros(n)
    for i in range(n):
        if valid[i]:
            pred = 1 if oof_probas[i] > 0.5 else 0
            pnl  = rets[i] if pred == y[i] else -abs(rets[i])
            eq[i] = pnl if np.isfinite(pnl) else 0.0

    cum   = np.cumsum(eq)
    peak  = np.maximum.accumulate(cum)
    dd    = cum - peak   # rolling drawdown

    # Down-weight samples where drawdown exceeds threshold
    in_cluster = dd < DD_CLUSTER_THRESH
    sw[in_cluster] = DD_SAMPLE_WEIGHT
    n_cluster = in_cluster.sum()
    if n_cluster > 0:
        print(f"  DD-aware: {n_cluster:,} samples in DD clusters "
              f"({n_cluster/n:.1%}) -> weight={DD_SAMPLE_WEIGHT}")
    return sw


def train_horizon(df: pd.DataFrame, horizon: int,
                  regime_series: pd.Series) -> dict | None:
    """
    v3.1 training pipeline with:
    - EV weights from triple_barrier_labels as base sample weights (Req 1)
    - Time-decay weights: recent samples weighted higher (Req 8)
    - Drawdown-aware weights: down-weight DD cluster samples (Req 9)
    - FSI feature stability check for meta-learner (Req 4)
    - 3-year recency filter for ensemble (Req 8)
    """
    barrier_col = f'barrier_{horizon}m'   # classification target: 1=UP, -1=DOWN, 0=time
    label_col   = f'label_{horizon}m'     # continuous EV weight (0..1) — used as sample weight
    ret_col     = f'fut_ret_{horizon}m'
    ev_col      = f'ev_{horizon}m'

    print(f"\n{'='*62}")
    print(f"Training {horizon}-min model  (v3.1 EV-Harvesting)")
    print(f"{'='*62}")

    # Step 7 (v3.3): Drop rows where fractionally-differentiated series
    # produced NaN during warm-up (fracdiff width ~600 bars).  These rows
    # would otherwise bleed into the feature matrix as 0-filled noise and
    # corrupt sample weights for the earliest labels.
    fracdiff_cols = ['close_fd', 'vwap_fd', 'ret5m_fd', 'ret15m_fd']
    existing_fd   = [c for c in fracdiff_cols if c in df.columns]
    if existing_fd:
        df = df.dropna(subset=existing_fd).copy()
    print(f"  After fracdiff drop: {len(df):,} rows")

    # Filter: valid labels, session window, all features present
    # Only check columns that actually exist in df (some optional features
    # like fft_cycle, hdfc_ret_1m etc. may not be computed in train mode)
    existing_feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    # Drop all-NaN feature columns from the NaN completeness check to avoid
    # silently zeroing out all rows (e.g. vol_price_corr=NaN for index instruments)
    existing_feature_cols = [c for c in existing_feature_cols if df[c].notna().any()]
    min_mod = 5 if horizon == 1 else 30
    # Filter on barrier_col (direction) being non-zero: skip time-exit bars (barrier=0)
    barrier_valid = df[barrier_col].notna() & (df[barrier_col] != 0) if barrier_col in df.columns \
                    else df[label_col].notna()
    print(f"  barrier_valid rows: {barrier_valid.sum():,}")
    feat_nan_mask = df[existing_feature_cols].notna().all(axis=1)
    print(f"  feat_nan_mask rows: {feat_nan_mask.sum():,}  (checking {len(existing_feature_cols)} features)")
    session_mask = (df['minute_of_day'] >= min_mod) & (df['minute_of_day'] <= 375 - horizon - 5)
    print(f"  session_mask rows: {session_mask.sum():,}")
    mask = (
        barrier_valid &
        feat_nan_mask &
        session_mask
    )
    df_h = df[mask].reset_index(drop=True)
    print(f"  After full mask: {len(df_h):,} rows")

    # Req 8: Ensemble trains on most recent 2 years only
    # (HMM regime detector uses full history, trained separately)
    # Use max date in data (not today) so the window is reproducible and
    # doesn't silently shift each time training is re-run.
    dates_ts   = pd.to_datetime(df_h['date'])
    cutoff_2yr = dates_ts.max() - pd.DateOffset(years=2)
    recent_mask = dates_ts >= cutoff_2yr
    if recent_mask.sum() > 500:
        df_train = df_h[recent_mask].reset_index(drop=True)
        print(f"  2yr recency filter: {recent_mask.sum():,}/{len(df_h):,} rows used")
        # --- Noise filtering: remove weak / random moves ---
        if ret_col in df_train.columns:
            min_move = 0.0005  # 0.05% (tune later)

            before = len(df_train)
            move_mask = np.abs(df_train[ret_col]) > min_move
            df_train = df_train[move_mask].reset_index(drop=True)

            print(f"  Noise filter: {before:,} → {len(df_train):,} rows")
            # --- Patch 5: Remove dead markets (low ADX) ---
        if 'adx_14' in df_train.columns:
            before = len(df_train)
            df_train = df_train[df_train['adx_14'] > 10]
            print(f"  ADX filter: {before:,} → {len(df_train):,}")
    else:
        df_train = df_h   # fallback to full if not enough recent data

    # 2️⃣ Train-Live Feature Mismatch Guard: Filter out unavailable features
    # Remove features that won't be available in live trading to prevent
    # model from learning dependencies on unavailable data sources.
    active_features = [f for f in FEATURE_COLS if FEATURE_LIVE_OK.get(f, True) and f in df_train.columns]
    print(f"  Feature filtering: {len(active_features)}/{len(FEATURE_COLS)} features "
          f"available in live ({len(FEATURE_COLS) - len(active_features)} excluded)")
    
    X_raw = np.nan_to_num(df_train[active_features].values.astype(np.float32), nan=0.0)
    # Classification target: barrier=1 (TP hit) → class 1 (UP), barrier=-1 (SL hit) → class 0 (DOWN)
    y    = (df_train[barrier_col].values > 0).astype(int)
    rets = df_train[ret_col].values.astype(float)
    dates= df_train['date'].values

    # EV weights: use cost-adjusted label (0..1) as sample weight; fall back to raw ev_col
    ev_w = df_train[label_col].values.astype(float) if label_col in df_train.columns \
           else (df_train[ev_col].values.astype(float) if ev_col in df_train.columns else np.ones(len(y)))
    ev_w = np.where(np.isnan(ev_w) | (ev_w <= 0), 1.0, ev_w)
    # Step 8 (v3.3): Clip EV weights to [0.2, 1.0] so no single sample
    # dominates training and outlier velocity-decay extremes are bounded.
    ev_w = np.clip(ev_w, 0.1, 2.0)
    # Time-decay weights (Req 8)
    td_w = _time_decay_weights(len(y))

    # Combined base weight = EV x time_decay (normalised)
    base_w = ev_w * td_w
    base_w = base_w / base_w.mean()

    # Attach regime to each row
    regimes = np.array([regime_series.get(d, REGIME_RANGING) for d in dates])

    # ---- Walk-forward scaling (LIVE-SAFE: no future data in normalization) ----
    tscv_for_scale = TimeSeriesSplit(n_splits=5)
    X, live_scaler, _ = _walk_forward_scale(X_raw, tscv_for_scale)
    print(f"  Walk-forward scaling applied: live_scaler fit on last fold train split")

    print(f"  Rows: {len(X):,}  UP%: {y.mean():.1%}  "
          f"Trending: {(regimes==0).mean():.1%}  "
          f"Ranging: {(regimes==1).mean():.1%}  "
          f"Crisis: {(regimes==2).mean():.1%}")

    if len(np.unique(y)) < 2:
        print("  SKIP: only one class")
        return None

    tscv = TimeSeriesSplit(n_splits=5)
    fold_results = []
    oof_probas   = np.full(len(X), np.nan)

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        tr_idx, te_idx = purge_embargo(tr_idx, te_idx, embargo_bars=5)
        if len(tr_idx) < 500 or len(te_idx) < 100:
            continue
        if len(np.unique(y[tr_idx])) < 2:
            continue

        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        wtr      = base_w[tr_idx]

        print(f"  Fold {fold+1}/5  train={len(tr_idx):,}  test={len(te_idx):,}  ...",
              end='', flush=True)

        base = make_base_model()
        # Pass sample weights; VotingClassifier propagates them to sub-estimators
        try:
            base.fit(Xtr, ytr, sample_weight=wtr)
        except TypeError:
            base.fit(Xtr, ytr)   # fallback if estimator doesn't support weights

        p_te   = base.predict_proba(Xte)[:, 1]
        preds  = (p_te > 0.5).astype(int)
        acc    = accuracy_score(yte, preds)
        oof_probas[te_idx] = p_te

        fold_results.append({
            'model': base, 'test_idx': te_idx,
            'probas': p_te, 'preds': preds,
            'acc': acc, 'rets': rets[te_idx], 'regimes': regimes[te_idx]
        })
        print(f"  acc={acc:.1%}")

    if not fold_results:
        print("  SKIP: no valid folds")
        return None

    avg_acc = float(np.mean([r['acc'] for r in fold_results]))
    print(f"\n  OOF accuracy: {avg_acc:.1%}  Baseline: {y.mean():.1%}  "
          f"Edge: {(avg_acc-y.mean())*100:+.2f}pp")

    # Drawdown-aware sample weights from OOF equity curve (Req 9)
    dd_w = _drawdown_aware_weights(oof_probas, y, rets)

    # --- Feature Stability Index check (Req 4) ---
    baseline_rows = min(len(df_train), int(FSI_BASELINE_YRS * 252 * 375))
    window_rows   = int(FSI_WINDOW_DAYS * 375)
    fsi_scores    = compute_fsi(df_train, active_features, baseline_rows, window_rows)
    drifted_feats = [f for f, v in fsi_scores.items() if v > FSI_SIGMA_THRESH]
    if drifted_feats:
        print(f"  FSI: {len(drifted_feats)} drifted features (>{FSI_SIGMA_THRESH}sigma): "
              f"{drifted_feats[:5]}{'...' if len(drifted_feats)>5 else ''}")

    # --- Meta-learner with FSI-adjusted features (Req 4) ---
    print(f"  Training meta-learner on OOF probas...")
    valid_oof = ~np.isnan(oof_probas)
    if valid_oof.sum() > 200:
        # Build meta features; drifted features receive 50% weight via masking
        fsi_mask = np.array([0.5 if f in drifted_feats else 1.0 for f in active_features])
        X_fsi_adj = X * fsi_mask   # elementwise scaling
        Xmeta = np.column_stack([
            oof_probas[valid_oof],
            regimes[valid_oof],
            df_train['session_pct'].values[valid_oof],
            df_train['iv_proxy'].values[valid_oof],
            df_train['atr_14_pct'].values[valid_oof],
            df_train['adx_14'].values[valid_oof],
            df_train['dmi_diff'].values[valid_oof],
            # FSI drift indicator as explicit meta-feature
            np.array([fsi_scores.get(f, 0) for f in active_features]).mean() *
            np.ones(valid_oof.sum()),
        ])
        ymeta  = y[valid_oof]
        # Combine DD weights with EV weights for meta training
        wmeta  = (base_w[valid_oof] * dd_w[valid_oof])
        wmeta  = wmeta / wmeta.mean()

        if len(np.unique(ymeta)) >= 2:
            meta = make_meta_model()
            try:
                meta.fit(Xmeta, ymeta, sample_weight=wmeta)
            except TypeError:
                meta.fit(Xmeta, ymeta)
        else:
            meta = None
    else:
        meta = None
        X_fsi_adj = X

    # --- MetaLabeler: predicts if primary model is correct (LdP meta-labeling) ---
    print(f"  Training meta-labeler (correctness predictor)...")
    # Ensure required meta features exist (safe fallback)
    for col in ['adx_14', 'atr_14_pct', 'ret_1m']:
        if col not in df_train.columns:
            df_train[col] = 0.0
    meta_labeler = MetaLabeler()
    meta_labeler.fit(
        X_primary     = X,
        y_primary     = y,
        oof_probas    = oof_probas,
        context_df    = df_train,
        regimes       = regimes,
        sample_weight = base_w,
    )

    # --- Calibrated final model on full recent data with combined weights ---
    # Platt scaling (sigmoid / LogisticRegression) fitted on OOF probabilities.
    # WHY switched from isotonic: isotonic memorises the calibration set with
    # only 150-200 samples per fold, producing saturated 0/1 outputs at feature
    # extremes (RSI=14, ADX=55) — this caused the "100% confidence" problem in
    # paper trading. Platt scaling (2-parameter sigmoid) cannot overfit a small
    # calibration set and produces smooth, distributed probabilities in [0.45, 0.82].
    print(f"  Training calibrated final model...")
    final_w    = base_w * dd_w
    final_w    = final_w / final_w.mean()
    final_base = make_base_model()
    try:
        final_base.fit(X, y, sample_weight=final_w)
    except TypeError:
        final_base.fit(X, y)

    # Platt calibration using OOF probas (leak-free: only held-out predictions)
    try:
        from sklearn.linear_model import LogisticRegression as _LR
        from sklearn.metrics import brier_score_loss as _brier
        valid_oof_cal = ~np.isnan(oof_probas)
        if valid_oof_cal.sum() > 100:
            # Fit sigmoid on raw OOF scores → calibrated probability
            platt = _LR(C=1.0, solver='lbfgs', max_iter=1000)
            platt.fit(oof_probas[valid_oof_cal].reshape(-1, 1), y[valid_oof_cal])
            final_cal = _CalibratedWrapper(final_base, platt)
            # Brier score: perfect calibration = acc*(1-acc); ratio < 0.5 = overfit
            raw_brier = _brier(y[valid_oof_cal], oof_probas[valid_oof_cal])
            cal_preds  = final_cal.predict_proba(X[valid_oof_cal])[:, 1]
            cal_brier  = _brier(y[valid_oof_cal], cal_preds)
            acc_base   = float((y[valid_oof_cal] == (oof_probas[valid_oof_cal] > 0.5).astype(int)).mean())
            theoretical_brier = acc_base * (1 - acc_base)
            reliability = cal_brier / (theoretical_brier + 1e-9)
            print(f"  Calibration (Platt): {valid_oof_cal.sum():,} OOF samples — "
                  f"Brier={cal_brier:.4f} (raw={raw_brier:.4f}) "
                  f"reliability={reliability:.2f} (1.0=perfect, <0.5=overfit)")
            if reliability < 0.5:
                print(f"  [CalibWarn] reliability={reliability:.2f} < 0.5 — model still overconfident; "
                      f"consider more training data or stronger regularisation")
        else:
            final_cal = final_base
            print(f"  Calibration: skipped (insufficient OOF samples: {valid_oof_cal.sum()})")
    except Exception as _e:
        logger.warning(f"Platt calibration failed ({_e}) — using uncalibrated model")
        final_cal = final_base

    # --- Mixture of Experts: regime-specific sub-models (v3.2) ---
    # Train one calibrated model per regime. In live mode, generate_signal()
    # routes to the regime-specific expert rather than the global model.
    # Patterns that predict direction in TRENDING markets often anti-predict
    # in RANGING markets; separating them removes this label dilution.
    print(f"  Training Mixture of Experts (regime-specific models)...")
    regime_models: dict = {}
    for rg in [REGIME_TRENDING, REGIME_RANGING]:
        rg_mask = regimes == rg
        if rg_mask.sum() < 300:
            print(f"    {REGIME_NAMES[rg]}: only {rg_mask.sum()} samples -- skip")
            continue
        if len(np.unique(y[rg_mask])) < 2:
            print(f"    {REGIME_NAMES[rg]}: only one class -- skip")
            continue
        Xrg  = X[rg_mask]; yrg = y[rg_mask]
        wrg  = final_w[rg_mask]; wrg = wrg / wrg.mean()
        rg_base = make_base_model()
        try:
            rg_base.fit(Xrg, yrg, sample_weight=wrg)
        except TypeError:
            rg_base.fit(Xrg, yrg)
        # Calibrate regime model with Platt scaling on its OOF slice (leak-free)
        try:
            from sklearn.linear_model import LogisticRegression as _LR
            rg_oof = oof_probas[rg_mask]
            rg_y   = y[rg_mask]
            valid_rg = ~np.isnan(rg_oof)
            if valid_rg.sum() > 100:
                platt_rg = _LR(C=1.0, solver='lbfgs', max_iter=1000)
                platt_rg.fit(rg_oof[valid_rg].reshape(-1, 1), rg_y[valid_rg])
                rg_cal = _CalibratedWrapper(rg_base, platt_rg)
            else:
                rg_cal = rg_base
        except Exception:
            rg_cal = rg_base
        regime_models[rg] = rg_cal
        # OOF accuracy on regime subset (honest: uses held-out predictions, not training set)
        rg_oof_probas = oof_probas[rg_mask]
        rg_oof_valid  = ~np.isnan(rg_oof_probas)
        if rg_oof_valid.sum() > 50:
            rg_acc = accuracy_score(yrg[rg_oof_valid],
                                    (rg_oof_probas[rg_oof_valid] > 0.5).astype(int))
            acc_label = "oof_acc"
        else:
            rg_acc = accuracy_score(yrg, (rg_base.predict_proba(Xrg)[:,1] > 0.5).astype(int))
            acc_label = "fit_acc"
        print(f"    {REGIME_NAMES[rg]:<10} n={rg_mask.sum():,}  "
              f"UP%={yrg.mean():.1%}  {acc_label}={rg_acc:.1%}")
    print(f"  MoE: {len(regime_models)} regime experts trained")

    # Feature importance (from GBM inside ensemble)
    try:
        imp  = final_base.estimators_[0].feature_importances_
        fimp = sorted(zip(active_features, imp), key=lambda x: -x[1])
    except Exception:
        fimp = []

    if fimp:
        print(f"  Top features:")
        for f, v in fimp[:8]:
            print(f"    {f:<30} {v:.4f}  {'#'*int(v*300)}")

    # --- Regime-filtered backtest ---
    bt = backtest_folds(fold_results, rets, regimes, dates_all=dates)
    print(f"  Backtest (all regimes):  {_bt_str(bt)}")
    bt_trend = backtest_folds(fold_results, rets, regimes,
                               regime_filter=REGIME_TRENDING, dates_all=dates)
    print(f"  Backtest (trending only): {_bt_str(bt_trend)}")
    backtest_breakdown(fold_results, y, rets, regimes, df_train, conf_thresh=CONF_MODERATE)

    return {
        'final_model':       final_cal,
        'meta_model':        meta,
        'meta_labeler':      meta_labeler,
        'regime_models':     regime_models,   # MoE: regime-specific experts
        'avg_acc':           avg_acc,
        'baseline':          float(y.mean()),
        'feat_imp':          fimp,
        'fsi_scores':        fsi_scores,
        'drifted_feats':     drifted_feats,
        'backtest':          bt,
        'backtest_trending': bt_trend,
        'horizon':           horizon,
        'active_features':   active_features,  # feature list used (train-live match)
        'live_scaler':       live_scaler,       # walk-forward scaler for live inference
    }


def _bt_str(bt: dict) -> str:
    return (f"trades={bt['trades']:,}  wr={bt['win_rate']:.1%}  "
            f"pf={bt['profit_factor']:.2f}  sharpe={bt['sharpe']:.2f}  "
            f"maxdd={bt['max_dd']:.1f}%  total={bt['total']:+.1f}%")


def backtest_breakdown(fold_results, y_all, rets_all, regimes_all,
                       df_train, conf_thresh=CONF_MODERATE) -> None:
    """
    Print win-rate breakdown by regime × ADX bucket × confidence bin.
    Identifies which conditions produce losers so they can be gated out.
    Called once after training — output goes to stdout only, no return value.
    """
    records = []
    for r in fold_results:
        te = r['test_idx']
        for i, (pred, proba) in enumerate(zip(r['preds'], r['probas'])):
            idx = te[i]
            conf = proba if pred == 1 else (1 - proba)
            if conf < conf_thresh:
                continue
            ret = float(rets_all[idx])
            if not np.isfinite(ret):
                continue
            win = int((ret > 0) == (pred == 1))
            adx = float(df_train['adx_14'].iloc[idx]) if 'adx_14' in df_train.columns else np.nan
            mod = float(df_train['minute_of_day'].iloc[idx]) if 'minute_of_day' in df_train.columns else np.nan
            records.append({
                'win': win,
                'conf': conf,
                'regime': regimes_all[idx],
                'adx': adx,
                'mod': mod,
            })

    if not records:
        return

    rows = pd.DataFrame(records)

    # Regime × ADX bucket
    adx_bins   = [0, 15, 25, 100]
    adx_labels = ['ADX<15', 'ADX15-25', 'ADX>25']
    rows['adx_bucket'] = pd.cut(rows['adx'], bins=adx_bins, labels=adx_labels, right=False)

    regime_map = {0: 'TRENDING', 1: 'RANGING', 2: 'CRISIS', -1: 'UNCERT'}
    rows['regime_name'] = rows['regime'].map(regime_map).fillna('?')

    print(f"\n  {'─'*54}")
    print(f"  Conditional Win Rate Breakdown (conf >= {conf_thresh:.2f})")
    print(f"  {'─'*54}")
    print(f"  {'Condition':<28} {'n':>5}  {'WR':>6}  {'Edge':>6}")
    print(f"  {'─'*54}")

    # By regime
    for rg_name, grp in rows.groupby('regime_name', sort=False):
        wr = grp['win'].mean()
        edge = wr - 0.50
        flag = ' <<' if wr < 0.48 else ''
        print(f"  {rg_name:<28} {len(grp):>5}  {wr:>5.1%}  {edge:>+5.1%}{flag}")

    print()

    # By regime × ADX
    for (rg_name, adx_b), grp in rows.groupby(['regime_name', 'adx_bucket'], sort=False):
        if len(grp) < 20:
            continue
        wr = grp['win'].mean()
        edge = wr - 0.50
        label = f"{rg_name} + {adx_b}"
        flag = ' <<' if wr < 0.47 else ''
        print(f"  {label:<28} {len(grp):>5}  {wr:>5.1%}  {edge:>+5.1%}{flag}")

    print()

    # By confidence bin
    conf_bins   = [0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
    conf_labels = ['0.50-0.55','0.55-0.60','0.60-0.65','0.65-0.70','0.70+']
    rows['conf_bin'] = pd.cut(rows['conf'], bins=conf_bins, labels=conf_labels, right=False)
    for cb, grp in rows.groupby('conf_bin', sort=True):
        if len(grp) < 10:
            continue
        wr = grp['win'].mean()
        edge = wr - 0.50
        flag = ' <<' if wr < 0.48 else ''
        print(f"  conf {cb:<23} {len(grp):>5}  {wr:>5.1%}  {edge:>+5.1%}{flag}")

    print(f"  {'─'*54}")


def backtest_folds(fold_results, rets_all, regimes_all,
                   conf_thresh=CONF_MODERATE,
                   regime_filter=None,
                   dates_all=None) -> dict:
    """Walk-forward OOF backtest with optional regime filter."""
    trade_rets  = []
    trade_dates = []

    for r in fold_results:
        te   = r['test_idx']
        for i, (pred, proba) in enumerate(zip(r['preds'], r['probas'])):
            idx = te[i]
            if regime_filter is not None and regimes_all[idx] != regime_filter:
                continue
            conf = proba if pred == 1 else (1 - proba)
            if conf < conf_thresh:
                continue
            ret = float(rets_all[idx])
            if not np.isfinite(ret):
                continue
            pnl = ret if pred == 1 else -ret
            # NOTE: fut_ret is raw NIFTY spot % move (e.g. 0.2%).
            # COST_RT_PCT (2.5%) is an options cost — too large to deduct here.
            # The training summary footer already warns about costs separately.
            # Cost realism is handled by the EV gate in signal_generator, not here.
            trade_rets.append(pnl)
            if dates_all is not None:
                trade_dates.append(dates_all[idx])

    if len(trade_rets) < 5:
        return {'trades':0,'win_rate':0,'profit_factor':0,
                'sharpe':0,'calmar':0,'max_dd':0,'total':0}

    arr = np.array(trade_rets)
    n   = len(arr)
    wins= (arr > 0).sum()
    wsum= arr[arr>0].sum()
    lsum= abs(arr[arr<0].sum())
    cum = np.cumsum(arr)
    peak= np.maximum.accumulate(cum)
    dd  = cum - peak
    max_dd = float(dd.min())

    # Sharpe: aggregate to daily P&L, then annualise by sqrt(252).
    # This matches industry convention and avoids the per-trade inflation
    # that occurs when multiplying by sqrt(trades/year) on high-frequency series.
    if dates_all is not None and len(trade_dates) == len(trade_rets):
        daily = {}
        for d, p in zip(trade_dates, trade_rets):
            daily[d] = daily.get(d, 0.0) + p
        daily_arr = np.array(list(daily.values()))
        sharpe = (daily_arr.mean() / (daily_arr.std() + 1e-9)) * np.sqrt(252)
    else:
        # Fallback: no dates available — use trade-count annualisation
        years_in_sample = 2.0
        sharpe = (arr.mean() / (arr.std() + 1e-9)) * np.sqrt(n / years_in_sample)

    calmar = (cum[-1] / abs(max_dd + 1e-9)) if max_dd < 0 else 99.0

    return {
        'trades':        int(n),
        'win_rate':      float(wins/n),
        'avg_ret':       float(arr.mean()),
        'profit_factor': float(wsum/lsum) if lsum > 0 else 99.0,
        'sharpe':        float(sharpe),
        'calmar':        float(calmar),
        'max_dd':        float(max_dd),
        'total':         float(arr.sum()),
    }



# ==============================================================================
# 7. SAVE / LOAD
# ==============================================================================

def save_all(models: dict, regime_det: RegimeDetector, df_train_full: pd.DataFrame = None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    regime_det.save(os.path.join(MODEL_DIR, 'regime.pkl'))
    meta = {}
    for h, res in models.items():
        if res is None: continue
        joblib.dump(res['final_model'],
                    os.path.join(MODEL_DIR, f'model_{h}m.pkl'))
        if res['meta_model']:
            joblib.dump(res['meta_model'],
                        os.path.join(MODEL_DIR, f'meta_{h}m.pkl'))
        # MetaLabeler (correctness predictor)
        ml = res.get('meta_labeler')
        if ml is not None and ml._fitted:
            ml.save(os.path.join(MODEL_DIR, f'metalabeler_{h}m.pkl'))
        # Mixture of Experts (regime-specific sub-models)
        rm = res.get('regime_models', {})
        if rm:
            joblib.dump(rm, os.path.join(MODEL_DIR, f'regime_experts_{h}m.pkl'))
        # Walk-forward live scaler (LIVE-SAFE: no future data in normalization)
        sc = res.get('live_scaler')
        if sc is not None:
            joblib.dump(sc, os.path.join(MODEL_DIR, f'scaler_{h}m.pkl'))
        meta[str(h)] = {
            'avg_acc':          res['avg_acc'],
            'baseline':         res['baseline'],
            'backtest':         res['backtest'],
            'backtest_trending':res['backtest_trending'],
            'horizon':          res['horizon'],
            'top_features':     [f for f,_ in res['feat_imp'][:10]],
            'active_features':  res.get('active_features', []),
        }

    # Compute and persist feature distribution stats for live OOD detection.
    # Use the active_features from the 5m model (canonical feature set).
    # Stats are mean/std computed on the full training window — the same data
    # the live_scaler was fit on — so z-scores in LiveSafetyManager are
    # calibrated to the real training distribution, not a future-contaminated one.
    feature_stats = {}
    ref_res = models.get(5) or (next((v for v in models.values() if v), None))
    active_feats = ref_res.get('active_features', []) if ref_res else []
    if df_train_full is not None and active_feats:
        for feat in active_feats:
            if feat not in df_train_full.columns:
                continue
            col = df_train_full[feat].dropna()
            if len(col) < 10:
                continue
            feature_stats[feat] = {
                'mean': float(col.mean()),
                'std':  float(col.std()),
            }
        print(f"  Feature stats computed: {len(feature_stats)} features saved for drift detection")
    else:
        logger.warning("save_all: df_train_full not provided — feature_stats not saved. "
                       "Re-run train with updated run_train() to enable OOD detection.")
    meta['_feature_stats'] = feature_stats

    # Persist seasonality prior so live mode can load it without re-training
    meta['_seasonality'] = _signal_state.seasonality_prior if _signal_state is not None else {}

    with open(os.path.join(MODEL_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved to {MODEL_DIR}/")


def load_all() -> tuple:
    import nifty_trader.config as _cfg

    meta_path = os.path.join(MODEL_DIR, 'meta.json')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"No models in {MODEL_DIR}/  ->  run --mode train first")
    with open(meta_path) as f:
        meta = json.load(f)

    models = {}
    for h in HORIZONS:
        p = os.path.join(MODEL_DIR, f'model_{h}m.pkl')
        if not os.path.isfile(p): continue
        pm  = os.path.join(MODEL_DIR, f'meta_{h}m.pkl')
        pml = os.path.join(MODEL_DIR, f'metalabeler_{h}m.pkl')
        prm = os.path.join(MODEL_DIR, f'regime_experts_{h}m.pkl')

        # Load MetaLabeler
        ml = MetaLabeler()
        if os.path.isfile(pml):
            ml.load(pml)

        # Load regime experts (MoE)
        regime_models = {}
        if os.path.isfile(prm):
            regime_models = joblib.load(prm)

        # Load walk-forward live scaler
        psc = os.path.join(MODEL_DIR, f'scaler_{h}m.pkl')
        live_scaler = joblib.load(psc) if os.path.isfile(psc) else None
        if live_scaler is None:
            logger.warning(f"No live_scaler found for horizon {h}m — raw features used (risk of distribution shift)")

        models[h] = {
            'final_model':   joblib.load(p),
            'meta_model':    joblib.load(pm) if os.path.isfile(pm) else None,
            'meta_labeler':  ml,
            'regime_models': regime_models,
            'live_scaler':   live_scaler,   # walk-forward scaler for live inference
            'avg_acc':       meta[str(h)]['avg_acc'],
            'baseline':      meta[str(h)]['baseline'],
            'backtest':      meta[str(h)]['backtest'],
            'backtest_trending': meta[str(h)].get('backtest_trending', {}),
            'active_features':   meta[str(h)].get('active_features', []),
        }

    regime_det = RegimeDetector()
    rp = os.path.join(MODEL_DIR, 'regime.pkl')
    if os.path.isfile(rp):
        regime_det.load(rp)

    # Restore seasonality prior into global signal state
    if '_seasonality' in meta and meta['_seasonality']:
        _signal_state.seasonality_prior = {
            float(k): float(v) for k, v in meta['_seasonality'].items()
        }
        print(f"  Seasonality prior: {len(_signal_state.seasonality_prior)} buckets loaded")

    # Restore feature distribution stats for live OOD/drift detection.
    # Writes directly into config._training_feature_stats (the dict that
    # LiveSafetyManager.check_feature_drift() reads every bar).
    # Without this the drift check silently returns (1.0, []) on every bar.
    fs = meta.get('_feature_stats', {})
    if fs:
        _cfg._training_feature_stats.update(fs)
        print(f"  Feature drift stats: {len(fs)} features loaded for OOD detection")
    else:
        logger.warning("load_all: no _feature_stats in meta.json — "
                       "feature drift detection is disabled. Re-run --mode train to fix.")

    print(f"Loaded models: {list(models.keys())}")
    return models, regime_det


