"""
Smart Money Concepts (SMC) Structural Engine.
Detects Swing Highs/Lows, Order Blocks, and Fair Value Gaps.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple


class SMCStructuralEngine:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def detect_swings(self, df: pd.DataFrame, order: int = 5) -> Tuple[List[int], List[int]]:
        """
        Detect Swing Highs and Swing Lows using pivot point logic.
        Returns lists of integer indices.
        """
        if df is None or len(df) < (2 * order + 1):
            return [], []
        
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        
        swing_highs = []
        swing_lows = []
        
        for i in range(order, len(df) - order):
            # Check Swing High: must be strictly higher than all neighbors
            is_swing_high = True
            for j in range(1, order + 1):
                if high[i] < high[i - j] or high[i] < high[i + j]:
                    is_swing_high = False
                    break
            if is_swing_high:
                swing_highs.append(i)
            
            # Check Swing Low: must be strictly lower than all neighbors
            is_swing_low = True
            for j in range(1, order + 1):
                if low[i] > low[i - j] or low[i] > low[i + j]:
                    is_swing_low = False
                    break
            if is_swing_low:
                swing_lows.append(i)
        
        return swing_highs, swing_lows

    def detect_order_blocks(self, df: pd.DataFrame, lookback: int = 100) -> List[Dict]:
        """
        Detect Order Blocks (last opposite candle before strong move).
        """
        if df is None or len(df) < lookback:
            return []
        
        # Preserve original indices
        start_idx = max(0, len(df) - lookback)
        df_slice = df.iloc[start_idx:].copy()
        order_blocks = []
        
        close = df_slice['close'].to_numpy()
        open_ = df_slice['open'].to_numpy()
        high = df_slice['high'].to_numpy()
        low = df_slice['low'].to_numpy()
        original_indices = df_slice.index.to_numpy()
        
        for i in range(2, len(df_slice) - 2):
            original_idx = int(original_indices[i])
            
            # Bullish Order Block: Last bearish candle before bullish impulse
            if close[i] < open_[i]:
                if (close[i+1] > open_[i+1] and close[i+2] > open_[i+2] and
                    close[i+2] > high[i]):
                    order_blocks.append({
                        'type': 'BULLISH',
                        'high': float(high[i]),
                        'low': float(low[i]),
                        'index': original_idx,
                        'time': df_slice['time'].iloc[i] if 'time' in df_slice.columns else None
                    })
            
            # Bearish Order Block: Last bullish candle before bearish impulse
            if close[i] > open_[i]:
                if (close[i+1] < open_[i+1] and close[i+2] < open_[i+2] and
                    close[i+2] < low[i]):
                    order_blocks.append({
                        'type': 'BEARISH',
                        'high': float(high[i]),
                        'low': float(low[i]),
                        'index': original_idx,
                        'time': df_slice['time'].iloc[i] if 'time' in df_slice.columns else None
                    })
        
        return order_blocks

    def detect_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Fair Value Gaps (3-candle imbalance).
        Returns DataFrame with boolean columns 'bullish_fvg' and 'bearish_fvg'.
        """
        if df is None or len(df) < 3:
            return pd.DataFrame()
        
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        
        bullish_fvg = np.zeros(len(df), dtype=bool)
        bearish_fvg = np.zeros(len(df), dtype=bool)
        
        for i in range(2, len(df)):
            if low[i] > high[i-2]:
                bullish_fvg[i-1] = True
            
            if high[i] < low[i-2]:
                bearish_fvg[i-1] = True
        
        result = df.copy()
        result['bullish_fvg'] = bullish_fvg
        result['bearish_fvg'] = bearish_fvg
        return result