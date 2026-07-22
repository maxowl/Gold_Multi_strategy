"""
Void Structural Engine.
Detects Liquidity Voids (inefficient price movements with little to no trading).
Used by S19_VoidReversal strategy.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict
from core.atr_cache import ATRCache


class VoidStructuralEngine:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def detect_liquidity_voids(self, df: pd.DataFrame, atr_multiplier: float = 3.0) -> List[Dict]:
        """
        Detect Liquidity Voids: 3-candle sequences with abnormally large ranges and minimal overlap.
        Indicates institutional injection and lack of two-way trading.
        """
        if df is None or len(df) < 5:
            return []
            
        try:
            high = df['high'].to_numpy().astype(float)
            low = df['low'].to_numpy().astype(float)
            close = df['close'].to_numpy().astype(float)
            
            atr_series = ATRCache.get_atr(df, 14).to_numpy()
            atr = np.nan_to_num(atr_series, nan=1.0)
            
            voids = []
            
            for i in range(2, len(df)):
                if atr[i] <= 0:
                    continue
                    
                # Calculate the total range of the 3-candle sequence
                range_3 = np.max(high[i-2:i+1]) - np.min(low[i-2:i+1])
                
                # A void must be significantly larger than normal volatility
                if range_3 < (atr_multiplier * atr[i]):
                    continue
                    
                # Determine direction and boundaries
                if close[i] > close[i-2]:
                    direction = 'BULLISH'
                    upper = float(np.max(high[i-2:i+1]))
                    # [FIX] Use the absolute minimum of the 3-candle sequence
                    lower = float(np.min(low[i-2:i+1]))
                else:
                    direction = 'BEARISH'
                    # [FIX] Use the absolute maximum of the 3-candle sequence
                    upper = float(np.max(high[i-2:i+1]))
                    lower = float(np.min(low[i-2:i+1]))
                
                # Check for minimal overlap between the 3 candles (inefficiency)
                # Overlap is defined as the intersection of all 3 candle bodies
                body_highs = np.maximum(close[i-2:i+1], df['open'].to_numpy().astype(float)[i-2:i+1])
                body_lows = np.minimum(close[i-2:i+1], df['open'].to_numpy().astype(float)[i-2:i+1])
                
                overlap_high = np.min(body_highs)
                overlap_low = np.max(body_lows)
                
                # If overlap is very small relative to the total range, it's a void
                if overlap_high <= overlap_low or (overlap_high - overlap_low) < (range_3 * 0.2):
                    voids.append({
                        'direction': direction,
                        'upper': upper,
                        'lower': lower,
                        'bar_index': int(i),
                        'range': float(range_3)
                    })
                    
            return voids
            
        except Exception as e:
            self.logger.error(f"[FAIL] Liquidity void detection error: {e}")
            return []