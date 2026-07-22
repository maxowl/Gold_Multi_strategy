"""
Propulsion Engine.
Detects Propulsion Blocks (high momentum institutional candles).
Used by S14_Propulsion strategy.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict
from core.atr_cache import ATRCache


class PropulsionEngine:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def detect_propulsion_blocks(self, df: pd.DataFrame, lookback: int = 100, min_atr_mult: float = 2.0) -> List[Dict]:
        """
        Detect Propulsion Blocks: Candles with body size > min_atr_mult * ATR.
        Indicates strong institutional momentum and breakout potential.
        """
        if df is None or len(df) < lookback:
            return []
            
        try:
            high = df['high'].to_numpy().astype(float)
            low = df['low'].to_numpy().astype(float)
            close = df['close'].to_numpy().astype(float)
            open_ = df['open'].to_numpy().astype(float)
            
            atr_series = ATRCache.get_atr(df, 14).to_numpy()
            atr = np.nan_to_num(atr_series, nan=1.0)
            
            blocks = []
            start_idx = max(1, len(df) - lookback)
            
            for i in range(start_idx, len(df)):
                if atr[i] <= 0:
                    continue
                    
                body = abs(close[i] - open_[i])
                
                if body >= (min_atr_mult * atr[i]):
                    # True Range calculation for additional context
                    tr1 = high[i] - low[i]
                    tr2 = np.abs(high[i] - close[i-1]) if i > 0 else 0.0
                    tr3 = np.abs(low[i] - close[i-1]) if i > 0 else 0.0
                    
                    # [FIX] Correct the first element which is corrupted by np.roll wrap-around in some implementations
                    # Here we calculate it directly safely
                    true_range = max(tr1, tr2, tr3)
                    
                    if close[i] > open_[i]:
                        direction = 'BULLISH'
                    else:
                        direction = 'BEARISH'
                        
                    blocks.append({
                        'direction': direction,
                        'high': float(high[i]),
                        'low': float(low[i]),
                        'body_size': float(body),
                        'atr_multiple': float(body / atr[i]),
                        'bar_index': int(i)
                    })
                    
            return blocks
            
        except Exception as e:
            self.logger.error(f"[FAIL] Propulsion block detection error: {e}")
            return []