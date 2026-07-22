"""
Order Flow Engine.
Analyzes volume imbalances and cumulative delta for institutional footprint detection.
Used by S2_VI_Sweep and S11_LiquidityDelta strategies.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional


class OrderFlowEngine:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def detect_volume_imbalance_zones(self, df: pd.DataFrame, min_imbalance_ratio: float = 2.0, lookback: int = 50) -> List[Dict]:
        """
        Detect Volume Imbalance Zones (gaps between consecutive candles with high volume).
        A bullish imbalance occurs when current low > previous high.
        """
        if df is None or len(df) < lookback:
            return []
        
        try:
            # [FIX] Sanitize OHLC arrays to prevent NaN propagation
            high = np.nan_to_num(df['high'].to_numpy().astype(float), nan=0.0)
            low = np.nan_to_num(df['low'].to_numpy().astype(float), nan=0.0)
            close = np.nan_to_num(df['close'].to_numpy().astype(float), nan=0.0)
            open_ = np.nan_to_num(df['open'].to_numpy().astype(float), nan=0.0)
            
            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].to_numpy().astype(float)
            elif 'volume' in df.columns:
                volume = df['volume'].to_numpy().astype(float)
            else:
                volume = np.ones(len(df))
                
            volume = np.nan_to_num(volume, nan=0.0)
            
            zones = []
            start_idx = max(1, len(df) - lookback)
            
            for i in range(start_idx, len(df)):
                # [FIX] Skip if any price data is invalid
                if high[i] == 0 or low[i] == 0 or high[i-1] == 0 or low[i-1] == 0:
                    continue
                    
                prev_vol = volume[i - 1]
                curr_vol = volume[i]
                if prev_vol <= 0: 
                    continue
                
                vol_ratio = curr_vol / prev_vol
                
                # Bullish Imbalance: Current Low > Previous High
                if low[i] > high[i - 1] and close[i] > open_[i] and vol_ratio >= min_imbalance_ratio:
                    zones.append({
                        'type': 'BULLISH',
                        'upper': float(low[i]),
                        'lower': float(high[i - 1]),
                        'bar_index': int(i),
                        'vol_ratio': float(vol_ratio)
                    })
                
                # Bearish Imbalance: Current High < Previous Low
                elif high[i] < low[i - 1] and close[i] < open_[i] and vol_ratio >= min_imbalance_ratio:
                    zones.append({
                        'type': 'BEARISH',
                        'upper': float(low[i - 1]),
                        'lower': float(high[i]),
                        'bar_index': int(i),
                        'vol_ratio': float(vol_ratio)
                    })
                    
            return zones
            
        except Exception as e:
            self.logger.error(f"[FAIL] Volume imbalance detection error: {e}")
            return []

    def calculate_liquidity_delta(self, df: pd.DataFrame, lookback: int = 20) -> Optional[pd.Series]:
        """
        Calculate Cumulative Volume Delta (CVD) approximation using tick volume and candle direction.
        Positive delta = buying pressure, Negative delta = selling pressure.
        """
        if df is None or len(df) < lookback:
            return None
            
        try:
            close = df['close'].to_numpy().astype(float)
            open_ = df['open'].to_numpy().astype(float)
            
            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].to_numpy().astype(float)
            elif 'volume' in df.columns:
                volume = df['volume'].to_numpy().astype(float)
            else:
                return None
                
            volume = np.nan_to_num(volume, nan=0.0)
            
            # Approximate delta: +volume if bullish, -volume if bearish
            direction = np.where(close >= open_, 1.0, -1.0)
            delta = direction * volume
            
            # Cumulative sum over the lookback window
            cum_delta = pd.Series(delta).rolling(window=lookback, min_periods=1).sum()
            
            return cum_delta
            
        except Exception as e:
            self.logger.error(f"[FAIL] Liquidity delta calculation error: {e}")
            return None