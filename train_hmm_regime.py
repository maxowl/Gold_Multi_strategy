#!/usr/bin/env python3
"""
HMM Regime Model Training Script.
Fetches historical data from MT5 and trains the GaussianHMM model.
"""
import MetaTrader5 as mt5
import pandas as pd
import logging
import sys
import os
from config import config
from core.hmm_regime_detector import HMMRegimeDetector

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def main():
    setup_logging()
    logger = logging.getLogger("TrainHMM")
    
    logger.info(f"[START] Training HMM Regime Model for {config.symbol}")
    
    # Initialize MT5
    if not mt5.initialize():
        logger.error(f"[FAIL] MT5 initialization failed: {mt5.last_error()}")
        sys.exit(1)
        
    try:
        # Fetch historical data (M15, 5000 bars approx 2-3 months)
        logger.info("[INFO] Fetching 5000 bars of M15 data...")
        rates = mt5.copy_rates_from_pos(config.symbol, mt5.TIMEFRAME_M15, 0, 10000)
        
        if rates is None or len(rates) == 0:
            logger.error("[FAIL] Could not fetch data from MT5. Check symbol and connection.")
            sys.exit(1)
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        logger.info(f"[OK] Fetched {len(df)} bars from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
        
        # Initialize and Train HMM
        detector = HMMRegimeDetector(model_path=config.regime_model_path, n_states=3)
        
        logger.info("[INFO] Starting HMM training...")
        detector.train(df, n_iter=100)
        
        if detector.is_trained:
            logger.info(f"[SUCCESS] Model trained and saved to {config.regime_model_path}")
            
            # Quick validation predict on recent data
            state, conf, details = detector.predict(df.tail(100))
            logger.info(f"[VALIDATION] Current Regime: {details.get('regime_name')} (Conf: {conf:.2f})")
        else:
            logger.error("[FAIL] Training failed. Check logs for details.")
            
    except Exception as e:
        logger.critical(f"[FAIL] Unexpected error during training: {e}", exc_info=True)
    finally:
        # [FIX] Ensure MT5 is always shutdown to prevent zombie processes
        mt5.shutdown()
        logger.info("[OK] MT5 connection closed.")

if __name__ == "__main__":
    main()