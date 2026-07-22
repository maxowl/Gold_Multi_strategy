"""
Wyckoff VSA (Volume Spread Analysis) Engine.
Detects Wyckoff patterns: Spring, Upthrust, Accumulation/Distribution.
Used by S22_WyckoffSpring strategy.
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict


class WyckoffVSAEngine:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def detect_spring(self, df: pd.DataFrame, lookback: int = 50) -> Optional[Dict]:
        """
        Detect Wyckoff Spring pattern (accumulation reversal).
        [FIX] Corrected loop range to scan the entire DataFrame for valid swing lows,
        preventing the strategy from missing recent Springs.
        """
        if df is None or len(df) < lookback:
            return None
        
        try:
            high = df['high'].to_numpy().astype(float)
            low = df['low'].to_numpy().astype(float)
            close = df['close'].to_numpy().astype(float)
            open_ = df['open'].to_numpy().astype(float)
            
            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].to_numpy().astype(float)
            elif 'volume' in df.columns:
                volume = df['volume'].to_numpy().astype(float)
            else:
                volume = np.ones(len(df))
            
            volume = np.nan_to_num(volume, nan=1.0)
            
            support_levels = []
            # [FIX] Scan the entire DataFrame (up to the last 5 bars) to find all valid swing lows
            for i in range(5, len(df) - 5):
                if i + 5 >= len(low):
                    continue
                window = low[max(0, i-5):min(len(low), i+6)]
                if low[i] == window.min() and len(set(window)) > 1:
                    support_levels.append({'index': i, 'price': float(low[i])})
            
            if not support_levels:
                return None
            
            start_idx = max(1, len(df) - 10)
            
            for i in range(start_idx, len(df)):
                current_low = low[i]
                current_close = close[i]
                current_open = open_[i]
                current_high = high[i]
                
                for support in reversed(support_levels):
                    support_price = support['price']
                    
                    if current_low < support_price:
                        if current_close > support_price:
                            candle_range = current_high - current_low
                            if candle_range > 0:
                                body = abs(current_close - current_open)
                                lower_wick = min(current_close, current_open) - current_low
                                wick_ratio = lower_wick / candle_range
                                
                                if wick_ratio >= 0.60:
                                    # [FIX] Use np.nanmean and validate slice to prevent NaN errors
                                    vol_slice = volume[max(0, i-20):i]
                                    if len(vol_slice) == 0: 
                                        continue
                                    avg_volume = np.nanmean(vol_slice)
                                    
                                    if not np.isnan(avg_volume) and avg_volume > 0 and volume[i] < avg_volume * 1.5:
                                        return {
                                            'type': 'SPRING',
                                            'low': float(current_low),
                                            'support_price': support_price,
                                            'bar_index': int(i),
                                            'wick_ratio': float(wick_ratio),
                                            'volume_ratio': float(volume[i] / avg_volume)
                                        }
            return None
            
        except Exception as e:
            self.logger.error(f"[FAIL] Spring detection error: {e}")
            return None

    def detect_upthrust(self, df: pd.DataFrame, lookback: int = 50) -> Optional[Dict]:
        """
        Detect Wyckoff Upthrust pattern (distribution reversal).
        [FIX] Corrected loop range to scan the entire DataFrame for valid swing highs.
        """
        if df is None or len(df) < lookback:
            return None
        
        try:
            high = df['high'].to_numpy().astype(float)
            low = df['low'].to_numpy().astype(float)
            close = df['close'].to_numpy().astype(float)
            open_ = df['open'].to_numpy().astype(float)
            
            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].to_numpy().astype(float)
            elif 'volume' in df.columns:
                volume = df['volume'].to_numpy().astype(float)
            else:
                volume = np.ones(len(df))
            
            volume = np.nan_to_num(volume, nan=1.0)
            
            resistance_levels = []
            # [FIX] Scan the entire DataFrame (up to the last 5 bars) to find all valid swing highs
            for i in range(5, len(df) - 5):
                if i + 5 >= len(high):
                    continue
                window = high[max(0, i-5):min(len(high), i+6)]
                if high[i] == window.max() and len(set(window)) > 1:
                    resistance_levels.append({'index': i, 'price': float(high[i])})
            
            if not resistance_levels:
                return None
            
            start_idx = max(1, len(df) - 10)
            
            for i in range(start_idx, len(df)):
                current_high = high[i]
                current_close = close[i]
                current_open = open_[i]
                current_low = low[i]
                
                for resistance in reversed(resistance_levels):
                    resistance_price = resistance['price']
                    
                    if current_high > resistance_price:
                        if current_close < resistance_price:
                            candle_range = current_high - current_low
                            if candle_range > 0:
                                body = abs(current_close - current_open)
                                upper_wick = current_high - max(current_close, current_open)
                                wick_ratio = upper_wick / candle_range
                                
                                if wick_ratio >= 0.60:
                                    vol_slice = volume[max(0, i-20):i]
                                    if len(vol_slice) == 0: 
                                        continue
                                    avg_volume = np.nanmean(vol_slice)
                                    
                                    if not np.isnan(avg_volume) and avg_volume > 0 and volume[i] < avg_volume * 1.5:
                                        return {
                                            'type': 'UPTHRUST',
                                            'high': float(current_high),
                                            'resistance_price': resistance_price,
                                            'bar_index': int(i),
                                            'wick_ratio': float(wick_ratio),
                                            'volume_ratio': float(volume[i] / avg_volume)
                                        }
            return None
            
        except Exception as e:
            self.logger.error(f"[FAIL] Upthrust detection error: {e}")
            return None

    def detect_no_supply(self, df: pd.DataFrame, lookback: int = 20) -> bool:
        """
        Detect No Supply condition (bearish candle with low volume).
        [FIX] Added guard against empty slices and zero average volume.
        """
        if df is None or len(df) < lookback:
            return False
        
        try:
            close = df['close'].to_numpy().astype(float)
            open_ = df['open'].to_numpy().astype(float)
            
            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].to_numpy().astype(float)
            elif 'volume' in df.columns:
                volume = df['volume'].to_numpy().astype(float)
            else:
                return False
            
            volume = np.nan_to_num(volume, nan=0.0)
            last_idx = len(df) - 1
            
            if close[last_idx] >= open_[last_idx]:
                return False
            
            # [FIX] Prevent empty slice mean calculation
            start_idx = max(0, last_idx - lookback)
            if start_idx >= last_idx:
                return False
                
            avg_volume = np.nanmean(volume[start_idx:last_idx])
            if np.isnan(avg_volume) or avg_volume <= 0:
                return False
            
            return volume[last_idx] < avg_volume * 0.7
            
        except Exception as e:
            self.logger.error(f"[FAIL] No supply detection error: {e}")
            return False

    def detect_no_demand(self, df: pd.DataFrame, lookback: int = 20) -> bool:
        """
        Detect No Demand condition (bullish candle with low volume).
        [FIX] Added guard against empty slices and zero average volume.
        """
        if df is None or len(df) < lookback:
            return False
        
        try:
            close = df['close'].to_numpy().astype(float)
            open_ = df['open'].to_numpy().astype(float)
            
            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].to_numpy().astype(float)
            elif 'volume' in df.columns:
                volume = df['volume'].to_numpy().astype(float)
            else:
                return False
            
            volume = np.nan_to_num(volume, nan=0.0)
            last_idx = len(df) - 1
            
            if close[last_idx] <= open_[last_idx]:
                return False
            
            # [FIX] Prevent empty slice mean calculation
            start_idx = max(0, last_idx - lookback)
            if start_idx >= last_idx:
                return False
                
            avg_volume = np.nanmean(volume[start_idx:last_idx])
            if np.isnan(avg_volume) or avg_volume <= 0:
                return False
            
            return volume[last_idx] < avg_volume * 0.7
            
        except Exception as e:
            self.logger.error(f"[FAIL] No demand detection error: {e}")
            return False