"""
Hybrid Multi-Timeframe LightGBM Model Training Script
Trains LightGBM models for regime prediction using MTF features.

Usage:
    python scripts/train_hybrid_mtf.py --symbol XAUUSDm --bars 5000
    python scripts/train_hybrid_mtf.py --symbol XAUUSDm --bars 10000 --target-horizon 10

Output:
    Saves trained models to config.lightgbm_models_path (default: lightgbm_regime_models.pkl)
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pickle
import argparse
import logging
import sys
from datetime import datetime
from typing import Dict, Tuple, Optional
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(f"{config.log_directory}/train_lightgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def initialize_mt5() -> bool:
    """Initialize MT5 connection."""
    init_params = {
        "login": config.mt5_login,
        "password": config.mt5_password,
        "server": config.mt5_server,
        "path": config.mt5_path,
        "timeout": 60000
    }
    init_params = {k: v for k, v in init_params.items() if v not in (0, "", None)}
    
    if not mt5.initialize(**init_params):
        logger.error(f"[FAIL] MT5 init failed: {mt5.last_error()}")
        return False
    
    logger.info(f"[OK] MT5 initialized | Account: {mt5.account_info().login}")
    return True


def fetch_multi_timeframe_data(symbol: str, bars: int = 5000) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV data for multiple timeframes."""
    tf_map = {
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1
    }
    
    data = {}
    
    for tf_name, tf_enum in tf_map.items():
        logger.info(f"[DATA] Fetching {bars} bars of {symbol} {tf_name}...")
        
        rates = mt5.copy_rates_from_pos(symbol, tf_enum, 0, bars)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"[DATA] No data for {tf_name}, skipping...")
            continue
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        
        data[tf_name] = df
        logger.info(f"[OK] Fetched {len(df)} bars for {tf_name}")
    
    return data


def calculate_features(df: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
    """Calculate technical features for a single timeframe."""
    df = df.copy()
    
    # Price features
    df[f'{prefix}return_1'] = df['close'].pct_change(1)
    df[f'{prefix}return_5'] = df['close'].pct_change(5)
    df[f'{prefix}return_10'] = df['close'].pct_change(10)
    
    # Moving averages
    df[f'{prefix}ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df[f'{prefix}ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df[f'{prefix}ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # EMA slopes
    df[f'{prefix}ema_10_slope'] = (df[f'{prefix}ema_10'] - df[f'{prefix}ema_10'].shift(5)) / df[f'{prefix}ema_10'].shift(5)
    df[f'{prefix}ema_20_slope'] = (df[f'{prefix}ema_20'] - df[f'{prefix}ema_20'].shift(5)) / df[f'{prefix}ema_20'].shift(5)
    
    # Volatility
    df[f'{prefix}volatility_14'] = df['close'].pct_change().rolling(14).std()
    df[f'{prefix}atr_14'] = calculate_atr(df, 14)
    
    # RSI
    df[f'{prefix}rsi_14'] = calculate_rsi(df['close'], 14)
    
    # Volume features
    if 'tick_volume' in df.columns:
        df[f'{prefix}volume_ma_20'] = df['tick_volume'].rolling(20).mean()
        df[f'{prefix}volume_ratio'] = df['tick_volume'] / (df[f'{prefix}volume_ma_20'] + 1e-10)
    
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def create_labels(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.005) -> pd.Series:
    """
    Create classification labels based on forward returns.
    
    Labels:
    - UP (0): Forward return > threshold
    - DOWN (1): Forward return < -threshold
    - NEUTRAL (2): Otherwise
    """
    forward_return = df['close'].shift(-horizon) / df['close'] - 1
    
    labels = pd.Series(2, index=df.index)  # Default: NEUTRAL
    labels[forward_return > threshold] = 0  # UP
    labels[forward_return < -threshold] = 1  # DOWN
    
    return labels


def align_timeframes(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align multiple timeframes by timestamp."""
    logger.info("[ALIGN] Aligning timeframes...")
    
    # Use M15 as base timeframe
    if 'M15' not in data:
        logger.error("[FAIL] M15 data not available")
        return pd.DataFrame()
    
    base_df = data['M15'].copy()
    
    # Calculate features for each timeframe
    for tf_name, df in data.items():
        df_features = calculate_features(df, prefix=f'{tf_name}_')
        
        # Merge with base dataframe
        if tf_name != 'M15':
            # Resample to M15 frequency
            df_features = df_features.set_index('time').resample('15min').last().reset_index()
        
        base_df = base_df.merge(
            df_features, 
            on='time', 
            how='left',
            suffixes=('', f'_{tf_name}')
        )
    
    # Drop rows with NaN
    base_df = base_df.dropna()
    
    logger.info(f"[OK] Aligned data shape: {base_df.shape}")
    
    return base_df


def train_lightgbm_models(df: pd.DataFrame, target_horizon: int = 5) -> Dict:
    """Train LightGBM models for each target."""
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("[FAIL] lightgbm not installed. Install with: pip install lightgbm")
        sys.exit(1)
    
    logger.info(f"[TRAIN] Training LightGBM models with target horizon: {target_horizon} bars")
    
    # Create labels
    labels = create_labels(df, horizon=target_horizon)
    
    # Feature columns (exclude time, OHLCV, and target)
    feature_cols = [col for col in df.columns if col not in ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]
    
    X = df[feature_cols].values
    y = labels.values
    
    logger.info(f"[TRAIN] Feature matrix shape: {X.shape}")
    logger.info(f"[TRAIN] Label distribution: UP={sum(y==0)}, DOWN={sum(y==1)}, NEUTRAL={sum(y==2)}")
    
    # Split data (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # LightGBM parameters
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
        'verbose': -1,
        'random_state': 42
    }
    
    # Train model
    logger.info("[TRAIN] Training LightGBM model...")
    
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    test_data = lgb.Dataset(X_test, label=y_test, feature_name=feature_cols, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    
    # Evaluate model
    logger.info("[EVAL] Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    
    accuracy = (y_pred_class == y_test).mean()
    logger.info(f"[EVAL] Test Accuracy: {accuracy:.2%}")
    
    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_importance = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    
    logger.info("[EVAL] Top 10 Features:")
    for i, (feat, imp) in enumerate(feature_importance[:10], 1):
        logger.info(f"  {i}. {feat}: {imp:.2f}")
    
    models = {
        'model': model,
        'feature_cols': feature_cols,
        'accuracy': accuracy,
        'feature_importance': feature_importance,
        'target_horizon': target_horizon
    }
    
    return models


def save_models(models: Dict, output_path: str):
    """Save trained models to pickle file."""
    try:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(models, f)
        
        file_size = os.path.getsize(output_path)
        logger.info(f"[OK] Models saved to {output_path} ({file_size:,} bytes)")
        
    except Exception as e:
        logger.error(f"[FAIL] Could not save models: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid Multi-Timeframe LightGBM Models")
    parser.add_argument('--symbol', type=str, default=config.symbol, help='Symbol to train on')
    parser.add_argument('--bars', type=int, default=5000, help='Number of historical bars per timeframe')
    parser.add_argument('--target-horizon', type=int, default=5, help='Forward bars for label creation (default: 5)')
    parser.add_argument('--threshold', type=float, default=0.005, help='Return threshold for UP/DOWN labels (default: 0.5%)')
    parser.add_argument('--output', type=str, default=config.lightgbm_models_path, help='Output models path')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("HYBRID MULTI-TIMEFRAME LIGHTGBM TRAINING")
    logger.info("=" * 80)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Bars per TF: {args.bars}")
    logger.info(f"Target Horizon: {args.target_horizon} bars")
    logger.info(f"Threshold: {args.threshold:.2%}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 80)
    
    # Step 1: Initialize MT5
    if not initialize_mt5():
        sys.exit(1)
    
    try:
        # Step 2: Fetch multi-timeframe data
        data = fetch_multi_timeframe_data(args.symbol, args.bars)
        
        if not data:
            logger.error("[FAIL] No data fetched")
            sys.exit(1)
        
        # Step 3: Align timeframes
        df = align_timeframes(data)
        
        if df.empty:
            logger.error("[FAIL] Timeframe alignment failed")
            sys.exit(1)
        
        # Step 4: Train LightGBM models
        models = train_lightgbm_models(df, target_horizon=args.target_horizon)
        
        # Step 5: Save models
        save_models(models, args.output)
        
        logger.info("=" * 80)
        logger.info("[SUCCESS] LightGBM training complete!")
        logger.info(f"[SUCCESS] Models saved to: {args.output}")
        logger.info(f"[SUCCESS] Test Accuracy: {models['accuracy']:.2%}")
        logger.info("=" * 80)
        
    finally:
        # Cleanup
        mt5.shutdown()
        logger.info("[OK] MT5 shutdown complete")


if __name__ == "__main__":
    main()