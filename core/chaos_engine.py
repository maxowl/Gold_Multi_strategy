"""
Chaos Theory Engine.
Provides Gaussian Squeeze Detection and Bill Williams indicators.
Used by S17 (Chaos Squeeze) strategy.
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple


class ChaosEngine:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_gaussian_squeeze_band(
        self, df: pd.DataFrame, period: int = 30, std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Gaussian Squeeze: Bollinger Bands vs Keltner Channels.
        Squeeze ON: BB inside KC (low volatility, compression)
        Squeeze OFF: BB outside KC (high volatility, expansion)
        
        Returns: (upper_band, lower_band, width)
        """
        empty = pd.Series(dtype=float)
        if df is None or len(df) < period:
            return empty, empty, empty
        
        try:
            close = df['close'].to_numpy().astype(float)
            
            # Calculate ATR for Keltner
            from core.atr_cache import ATRCache
            atr = ATRCache.get_atr(df, 14).to_numpy()
            atr = np.nan_to_num(atr, nan=1.0)
            
            # Bollinger Bands
            sma = pd.Series(close).rolling(period).mean().to_numpy()
            std = pd.Series(close).rolling(period).std().to_numpy()
            
            # Handle NaNs in SMA/STD
            sma = np.nan_to_num(sma, nan=0.0)
            std = np.nan_to_num(std, nan=0.0)
            
            bb_upper = sma + std_dev * std
            bb_lower = sma - std_dev * std
            
            # Keltner Channels
            kc_upper = sma + 1.5 * atr
            kc_lower = sma - 1.5 * atr
            
            # Squeeze detection: BB inside KC
            squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)
            
            # Band width (normalized)
            width = (bb_upper - bb_lower) / (sma + 1e-10)
            
            return (
                pd.Series(bb_upper, index=df.index),
                pd.Series(bb_lower, index=df.index),
                pd.Series(width, index=df.index)
            )
            
        except Exception as e:
            self.logger.error(f"[FAIL] Gaussian squeeze error: {e}")
            return empty, empty, empty