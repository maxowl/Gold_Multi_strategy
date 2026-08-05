"""
HMM Regime Model Training Script
Trains a Gaussian Hidden Markov Model for market regime detection.

Usage:
    python scripts/train_hmm_regime.py --symbol XAUUSDm --tf M15 --bars 10000
    python scripts/train_hmm_regime.py --symbol XAUUSDm --tf H1 --bars 5000 --n-states 4

Output:
    Saves trained model to config.regime_model_path (default: hmm_regime_model.pkl)
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pickle
import argparse
import logging
import sys
from datetime import datetime, timedelta
from typing import Tuple, Optional
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(f"{config.log_directory}/train_hmm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
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


def fetch_historical_data(symbol: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
    """Fetch historical OHLCV data from MT5."""
    tf_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1
    }
    
    if timeframe not in tf_map:
        logger.error(f"[FAIL] Unknown timeframe: {timeframe}")
        return None
    
    logger.info(f"[DATA] Fetching {bars} bars of {symbol} {timeframe}...")
    
    rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, bars)
    
    if rates is None or len(rates) == 0:
        logger.error(f"[FAIL] No data fetched for {symbol} {timeframe}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    logger.info(f"[OK] Fetched {len(df)} bars from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for HMM training.
    
    Features:
    1. Log Returns (price momentum)
    2. Realized Volatility (rolling std of returns)
    3. Volume Momentum (rolling mean of tick_volume)
    4. Price Range (high-low normalized by close)
    """
    logger.info("[FEATURE] Engineering features for HMM...")
    
    df = df.copy()
    
    # 1. Log Returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. Realized Volatility (14-period rolling std)
    df['realized_vol'] = df['log_return'].rolling(window=14, min_periods=14).std()
    
    # 3. Volume Momentum (14-period rolling mean, normalized)
    if 'tick_volume' in df.columns:
        df['volume_ma'] = df['tick_volume'].rolling(window=14, min_periods=14).mean()
        df['volume_momentum'] = df['tick_volume'] / df['volume_ma']
    else:
        df['volume_momentum'] = 1.0
    
    # 4. Price Range (normalized)
    df['price_range'] = (df['high'] - df['low']) / df['close']
    
    # 5. Trend Strength (ADX proxy - simplified)
    df['trend_strength'] = abs(df['close'] - df['close'].shift(14)) / (df['close'].rolling(14).std() + 1e-10)
    
    # Drop rows with NaN
    df = df.dropna()
    
    logger.info(f"[OK] Feature engineering complete | Remaining samples: {len(df)}")
    
    return df


def prepare_hmm_features(df: pd.DataFrame) -> np.ndarray:
    """
    Prepare feature matrix for HMM.
    
    HMM works best with 2-3 features to avoid overfitting.
    We use: [log_return, realized_vol]
    """
    features = df[['log_return', 'realized_vol']].values
    
    # Standardize features (zero mean, unit variance)
    features_mean = features.mean(axis=0)
    features_std = features.std(axis=0)
    features_standardized = (features - features_mean) / (features_std + 1e-10)
    
    logger.info(f"[FEATURE] Prepared HMM features | Shape: {features_standardized.shape}")
    logger.info(f"[FEATURE] Feature means: {features_mean}")
    logger.info(f"[FEATURE] Feature stds: {features_std}")
    
    return features_standardized


def train_hmm(features: np.ndarray, n_states: int = 3, n_iter: int = 100) -> object:
    """
    Train Gaussian HMM using hmmlearn library.
    Robust version: Handles API changes across different hmmlearn versions.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        n_states: Number of hidden states (default: 3 for BULL/BEAR/SIDEWAY)
        n_iter: Maximum iterations for EM algorithm
    
    Returns:
        Trained GaussianHMM model
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        logger.error("[FAIL] hmmlearn not installed. Install with: pip install hmmlearn")
        sys.exit(1)
    
    logger.info(f"[TRAIN] Training GaussianHMM with {n_states} states, {n_iter} iterations...")
    logger.info(f"[TRAIN] Training data shape: {features.shape}")
    
    # Initialize HMM
    model = GaussianHMM(
        n_components=n_states,
        covariance_type='full',  # Full covariance matrix
        n_iter=n_iter,
        random_state=42,
        verbose=False  # Disable verbose to prevent console spam
    )
    
    # Train model
    try:
        model.fit(features)
        
        # =========================================================================
        # SAFE ITERATION COUNT EXTRACTION (Handles API Deprecation)
        # =========================================================================
        n_iterations = getattr(model, 'n_iter_', None)
        if n_iterations is None and hasattr(model, 'monitor_'):
            n_iterations = getattr(model.monitor_, 'n_iter', 'unknown')
        if n_iterations is None:
            n_iterations = 'completed'
            
        logger.info(f"[OK] HMM training converged successfully ({n_iterations} iterations)")
        
    except Exception as e:
        logger.error(f"[FAIL] HMM training failed: {e}")
        sys.exit(1)
    
    # Analyze trained states
    hidden_states = model.predict(features)
    
    logger.info("=" * 80)
    logger.info("[ANALYSIS] Trained HMM State Distribution:")
    logger.info("=" * 80)
    
    for state in range(n_states):
        state_mask = hidden_states == state
        state_count = state_mask.sum()
        state_pct = (state_count / len(hidden_states)) * 100
        
        # Calculate mean returns and volatility for this state
        state_returns = features[state_mask, 0]  # log_return
        state_vol = features[state_mask, 1]  # realized_vol
        
        mean_return = state_returns.mean()
        mean_vol = state_vol.mean()
        
        # Interpret state
        if mean_return > 0.001 and mean_vol < 0.01:
            interpretation = "BULL (Low Vol Uptrend)"
        elif mean_return < -0.001 and mean_vol < 0.01:
            interpretation = "BEAR (Low Vol Downtrend)"
        elif mean_vol > 0.015:
            interpretation = "HIGH_VOL (Volatile)"
        else:
            interpretation = "SIDEWAY (Consolidation)"
        
        logger.info(
            f"State {state}: {state_count} samples ({state_pct:.1f}%) | "
            f"Mean Return: {mean_return:.4f} | Mean Vol: {mean_vol:.4f} | "
            f"Interpretation: {interpretation}"
        )
    
    logger.info("=" * 80)
    
    return model


def save_model(model: object, output_path: str):
    """Save trained model to pickle file."""
    try:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        
        file_size = os.path.getsize(output_path)
        logger.info(f"[OK] Model saved to {output_path} ({file_size:,} bytes)")
        
    except Exception as e:
        logger.error(f"[FAIL] Could not save model: {e}")
        sys.exit(1)


def validate_model(model: object, features: np.ndarray) -> dict:
    """Validate trained model with basic metrics."""
    logger.info("[VALIDATE] Running model validation...")
    
    try:
        # Predict hidden states
        hidden_states = model.predict(features)
        
        # Calculate log likelihood
        log_likelihood = model.score(features)
        
        # Calculate state distribution
        unique, counts = np.unique(hidden_states, return_counts=True)
        state_distribution = dict(zip(unique.tolist(), counts.tolist()))
        
        # Calculate transition matrix
        transition_matrix = model.transmat_
        
        validation_results = {
            'log_likelihood': log_likelihood,
            'state_distribution': state_distribution,
            'transition_matrix': transition_matrix,
            'n_samples': len(features),
            'n_states': model.n_components
        }
        
        logger.info(f"[VALIDATE] Log Likelihood: {log_likelihood:.2f}")
        logger.info(f"[VALIDATE] State Distribution: {state_distribution}")
        logger.info(f"[VALIDATE] Transition Matrix:\n{transition_matrix}")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"[FAIL] Validation failed: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Train HMM Regime Model")
    parser.add_argument('--symbol', type=str, default=config.symbol, help='Symbol to train on')
    parser.add_argument('--tf', type=str, default='M15', help='Timeframe (M1, M5, M15, M30, H1, H4, D1)')
    parser.add_argument('--bars', type=int, default=10000, help='Number of historical bars to use')
    parser.add_argument('--n-states', type=int, default=3, help='Number of hidden states (default: 3)')
    parser.add_argument('--n-iter', type=int, default=100, help='Maximum EM iterations (default: 100)')
    parser.add_argument('--output', type=str, default=config.regime_model_path, help='Output model path')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("HMM REGIME MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Timeframe: {args.tf}")
    logger.info(f"Bars: {args.bars}")
    logger.info(f"States: {args.n_states}")
    logger.info(f"Iterations: {args.n_iter}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 80)
    
    # Step 1: Initialize MT5
    if not initialize_mt5():
        sys.exit(1)
    
    try:
        # Step 2: Fetch data
        df = fetch_historical_data(args.symbol, args.tf, args.bars)
        if df is None:
            sys.exit(1)
        
        # Step 3: Engineer features
        df = engineer_features(df)
        
        # Step 4: Prepare HMM features
        features = prepare_hmm_features(df)
        
        # Step 5: Train HMM
        model = train_hmm(features, n_states=args.n_states, n_iter=args.n_iter)
        
        # Step 6: Validate model
        validation_results = validate_model(model, features)
        
        # Step 7: Save model
        save_model(model, args.output)
        
        logger.info("=" * 80)
        logger.info("[SUCCESS] HMM training complete!")
        logger.info(f"[SUCCESS] Model saved to: {args.output}")
        logger.info("=" * 80)
        
    finally:
        # Cleanup
        mt5.shutdown()
        logger.info("[OK] MT5 shutdown complete")


if __name__ == "__main__":
    main()