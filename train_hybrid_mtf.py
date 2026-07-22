#!/usr/bin/env python3
"""
Hybrid MTF (HMM + LightGBM) Training Script.
Trains LightGBM classifiers for each HMM regime using multi-timeframe features.
[FIX] Corrected Config attribute names and improved data alignment logic.
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import sys
import os
import joblib
from datetime import datetime
from pathlib import Path

from core.hmm_regime_detector import HMMRegimeDetector
from core.mtf_feature_engineer import MTFFeatureEngineer
from config import config


def setup_logging():
    """Configure logging."""
    Path(config.log_directory).mkdir(exist_ok=True)
    log_file = os.path.join(config.log_directory, f"train_hybrid_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    # Suppress noisy third-party loggers
    logging.getLogger('lightgbm').setLevel(logging.WARNING)


def fetch_mtf_data(symbol: str, count: int = 5000):
    """Fetch multi-timeframe data from MT5."""
    timeframes = {
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1
    }
    
    data = {}
    for tf_name, tf_enum in timeframes.items():
        rates = mt5.copy_rates_from_pos(symbol, tf_enum, 0, count)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC')
            data[tf_name] = df
            logging.info(f"[OK] Fetched {len(df)} bars of {tf_name} data")
        else:
            logging.warning(f"[WARN] Failed to fetch {tf_name} data")
    
    return data


def extract_features_and_labels(data: dict, hmm_detector: HMMRegimeDetector):
    """Extract MTF features and create labels for training."""
    df_m5 = data.get('M5')
    df_m15 = data.get('M15')
    df_h1 = data.get('H1')
    
    if df_m15 is None or len(df_m15) < 200:
        logging.error("[FAIL] Insufficient M15 data for regime labeling")
        return pd.DataFrame()

    # 1. Label M15 data with HMM regimes
    logging.info("[INFO] Labeling M15 data with HMM regimes...")
    regimes_m15 = []
    
    # We predict regime for each M15 bar using a rolling window
    for i in range(len(df_m15)):
        if i < 100:
            regimes_m15.append(2)  # Default to SIDEWAY during warmup
            continue
            
        df_window = df_m15.iloc[i-100:i+1].copy()
        regime_id, _, _ = hmm_detector.predict(df_window)
        regimes_m15.append(regime_id)
        
    df_m15['regime'] = regimes_m15
    
    # 2. Extract MTF features (Base timeframe is M5)
    logging.info("[INFO] Extracting multi-timeframe features (Base: M5)...")
    engineer = MTFFeatureEngineer()
    features = engineer.extract_all_features(df_m5, df_m15, df_h1)
    
    if features.empty:
        logging.error("[FAIL] Feature extraction returned empty DataFrame")
        return pd.DataFrame()
        
    # 3. Align M15 regimes to M5 features using merge_asof
    # This ensures each M5 bar gets the regime of the most recent completed M15 bar
    logging.info("[INFO] Aligning M15 regimes to M5 features...")
    df_m15_regimes = df_m15[['time', 'regime']].copy()
    
    features = pd.merge_asof(
        features.sort_values('time'),
        df_m15_regimes.sort_values('time'),
        on='time',
        direction='backward'
    )
    
    # Drop rows where regime is NaN (happens at the very beginning before first M15 close)
    features = features.dropna(subset=['regime'])
    features['regime'] = features['regime'].astype(int)
    
    # 4. Create target variable (Future 10-bar return on M5)
    # UP: return > 0.1%, DOWN: return < -0.1%, SIDEWAY: otherwise
    logging.info("[INFO] Creating target variables (Future returns)...")
    future_returns = df_m5['close'].pct_change(10).shift(-10)
    
    # Align target to features index
    features['target'] = future_returns.loc[features.index]
    
    # Categorize returns
    features['target_class'] = pd.cut(
        features['target'],
        bins=[-np.inf, -0.001, 0.001, np.inf],
        labels=['DOWN', 'SIDEWAY', 'UP']
    )
    
    # Drop rows with NaN in target (last 10 bars) or features
    features = features.dropna()
    
    logging.info(f"[INFO] Total samples after cleaning: {len(features)}")
    logging.info(f"[INFO] Regime distribution: {features['regime'].value_counts().to_dict()}")
    logging.info(f"[INFO] Target distribution: {features['target_class'].value_counts().to_dict()}")
    
    return features


def train_regime_models(features: pd.DataFrame):
    """Train LightGBM models for each regime."""
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    regime_models = {}
    
    # Improved LightGBM parameters for small/medium datasets
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    }
    
    feature_cols = [c for c in features.columns if c not in ['regime', 'target', 'target_class', 'time']]
    label_map = {'DOWN': 0, 'SIDEWAY': 1, 'UP': 2}
    
    for regime_id in [0, 1, 2]:
        regime_name = {0: 'BULL_TREND', 1: 'BEAR_TREND', 2: 'SIDEWAY'}[regime_id]
        logging.info(f"\n[INFO] Training LightGBM for {regime_name} (Regime {regime_id})...")
        
        regime_data = features[features['regime'] == regime_id].copy()
        
        if len(regime_data) < 300:
            logging.warning(f"[SKIP] Insufficient data for {regime_name}: {len(regime_data)} samples (min: 300)")
            continue
        
        X = regime_data[feature_cols]
        y = regime_data['target_class'].map(label_map).astype(int)  # [FIX] Convert to int
        
        class_counts = y.value_counts()
        logging.info(f"[INFO] Class distribution: {class_counts.to_dict()}")
        
        if class_counts.min() < 30:
            logging.warning(f"[SKIP] Severe class imbalance for {regime_name}: min class has {class_counts.min()} samples")
            continue
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, free_raw_data=False)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[test_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=10, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred_class)
        
        logging.info(f"[OK] {regime_name} model trained | Accuracy: {accuracy:.2%} | Iterations: {model.best_iteration}")
        
        regime_models[regime_id] = model
    
    return regime_models, feature_cols


def main():
    """Main training function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("HYBRID MTF MODEL TRAINING")
    logger.info("=" * 70)
    
    if not mt5.initialize():
        logger.error(f"[FAIL] MT5 initialization failed: {mt5.last_error()}")
        sys.exit(1)
        
    try:
        # 1. Load Pre-trained HMM
        # [FIX] Use correct config attribute: config.regime_model_path
        hmm = HMMRegimeDetector(model_path=config.regime_model_path)
        if not hmm.is_trained:
            logger.error(f"[FAIL] HMM model not found at {config.regime_model_path}. Run train_hmm_regime.py first.")
            sys.exit(1)
            
        logger.info(f"[OK] Loaded HMM model from {config.regime_model_path}")
        
        # 2. Fetch Data
        # [FIX] Use correct config attribute: config.symbol
        data = fetch_mtf_data(config.symbol, count=10000)
        
        if len(data) < 3:
            logger.error("[FAIL] Insufficient timeframe data fetched")
            sys.exit(1)
            
        # 3. Extract Features & Labels
        features = extract_features_and_labels(data, hmm)
        
        if features.empty or len(features) < 500:
            logger.error(f"[FAIL] Insufficient total samples for training: {len(features)}")
            sys.exit(1)
            
        # 4. Train LightGBM Models
        regime_models, feature_names = train_regime_models(features)
        
        if not regime_models:
            logger.error("[FAIL] No LightGBM models were trained successfully")
            sys.exit(1)
            
        # 5. Save Models
        # [FIX] Use correct config attribute: config.lightgbm_models_path
        output_data = {
            'models': regime_models,
            'feature_names': feature_names
        }
        joblib.dump(output_data, config.lightgbm_models_path)
        logger.info(f"\n[SUCCESS] Saved {len(regime_models)} LightGBM models to {config.lightgbm_models_path}")
        
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.critical(f"[FAIL] Unexpected error during training: {e}", exc_info=True)
    finally:
        mt5.shutdown()
        logger.info("[OK] MT5 connection closed")


if __name__ == "__main__":
    main()