"""
core/volume_divergence_detector.py
Detects RSI-Volume and Price-Volume divergences for early reversal signals.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging


class VolumeDivergenceDetector:
    """
    Detects volume divergences as leading reversal signals.
    
    Types:
    1. Bullish RSI-Volume Divergence:
       - Price makes lower low
       - RSI makes higher low
       - Volume makes higher low (accumulation)
       
    2. Bearish RSI-Volume Divergence:
       - Price makes higher high
       - RSI makes lower high
       - Volume makes lower high (distribution)
       
    3. Price-Volume Divergence (Classic):
       - Price makes new extreme
       - Volume declining (exhaustion)
    """
    
    def __init__(self, rsi_period: int = 14, lookback: int = 20):
        self.rsi_period = rsi_period
        self.lookback = lookback
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_rsi(self, close: pd.Series) -> pd.Series:
        """Calculate RSI indicator."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def detect_swing_points(self, series: pd.Series, order: int = 3) -> Tuple[list, list]:
        """Detect swing highs and lows in a series."""
        highs = []
        lows = []
        
        for i in range(order, len(series) - order):
            # Swing High
            is_high = True
            for j in range(1, order + 1):
                if series.iloc[i] <= series.iloc[i - j] or series.iloc[i] <= series.iloc[i + j]:
                    is_high = False
                    break
            if is_high:
                highs.append((i, series.iloc[i]))
            
            # Swing Low
            is_low = True
            for j in range(1, order + 1):
                if series.iloc[i] >= series.iloc[i - j] or series.iloc[i] >= series.iloc[i + j]:
                    is_low = False
                    break
            if is_low:
                lows.append((i, series.iloc[i]))
        
        return highs, lows
    
    def detect_divergences(self, df: pd.DataFrame) -> Dict:
        """
        Detect all types of volume divergences.
        
        Returns:
            {
                'bullish_rsi_vol': bool,
                'bearish_rsi_vol': bool,
                'bullish_price_vol': bool,
                'bearish_price_vol': bool,
                'strength': int (0-100),
                'details': str
            }
        """
        if df is None or len(df) < self.lookback + self.rsi_period:
            return {
                'bullish_rsi_vol': False,
                'bearish_rsi_vol': False,
                'bullish_price_vol': False,
                'bearish_price_vol': False,
                'strength': 0,
                'details': 'Insufficient data'
            }
        
        try:
            close = df['close']
            volume = df['tick_volume'] if 'tick_volume' in df.columns else df['volume']
            rsi = self.calculate_rsi(close)
            
            # Detect swing points
            price_highs, price_lows = self.detect_swing_points(close, order=3)
            rsi_highs, rsi_lows = self.detect_swing_points(rsi, order=3)
            vol_highs, vol_lows = self.detect_swing_points(volume, order=3)
            
            result = {
                'bullish_rsi_vol': False,
                'bearish_rsi_vol': False,
                'bullish_price_vol': False,
                'bearish_price_vol': False,
                'strength': 0,
                'details': ''
            }
            
            strength_score = 0
            details = []
            
            # =========================================================================
            # 1. Bullish RSI-Volume Divergence
            # =========================================================================
            if len(price_lows) >= 2 and len(rsi_lows) >= 2 and len(vol_lows) >= 2:
                p1_idx, p1_val = price_lows[-2]
                p2_idx, p2_val = price_lows[-1]
                
                # Find corresponding RSI lows
                r1_val = rsi.iloc[p1_idx]
                r2_val = rsi.iloc[p2_idx]
                
                # Find corresponding Volume lows
                v1_val = volume.iloc[p1_idx]
                v2_val = volume.iloc[p2_idx]
                
                # Check divergence conditions
                if (p2_val < p1_val and  # Price: lower low
                    r2_val > r1_val and  # RSI: higher low
                    v2_val > v1_val):    # Volume: higher low (accumulation)
                    
                    result['bullish_rsi_vol'] = True
                    strength_score += 40
                    details.append(f"Bullish RSI-Vol Div: Price LL, RSI HL, Vol HL")
            
            # =========================================================================
            # 2. Bearish RSI-Volume Divergence
            # =========================================================================
            if len(price_highs) >= 2 and len(rsi_highs) >= 2 and len(vol_highs) >= 2:
                p1_idx, p1_val = price_highs[-2]
                p2_idx, p2_val = price_highs[-1]
                
                r1_val = rsi.iloc[p1_idx]
                r2_val = rsi.iloc[p2_idx]
                
                v1_val = volume.iloc[p1_idx]
                v2_val = volume.iloc[p2_idx]
                
                if (p2_val > p1_val and  # Price: higher high
                    r2_val < r1_val and  # RSI: lower high
                    v2_val < v1_val):    # Volume: lower high (distribution)
                    
                    result['bearish_rsi_vol'] = True
                    strength_score += 40
                    details.append(f"Bearish RSI-Vol Div: Price HH, RSI LH, Vol LH")
            
            # =========================================================================
            # 3. Bullish Price-Volume Divergence (Classic)
            # =========================================================================
            if len(price_lows) >= 2 and len(vol_lows) >= 2:
                p1_idx, p1_val = price_lows[-2]
                p2_idx, p2_val = price_lows[-1]
                
                v1_val = volume.iloc[p1_idx]
                v2_val = volume.iloc[p2_idx]
                
                if (p2_val < p1_val and  # Price: lower low
                    v2_val > v1_val):    # Volume: higher (accumulation at lows)
                    
                    result['bullish_price_vol'] = True
                    strength_score += 20
                    details.append(f"Bullish Price-Vol: Price LL, Volume increasing")
            
            # =========================================================================
            # 4. Bearish Price-Volume Divergence (Classic)
            # =========================================================================
            if len(price_highs) >= 2 and len(vol_highs) >= 2:
                p1_idx, p1_val = price_highs[-2]
                p2_idx, p2_val = price_highs[-1]
                
                v1_val = volume.iloc[p1_idx]
                v2_val = volume.iloc[p2_idx]
                
                if (p2_val > p1_val and  # Price: higher high
                    v2_val < v1_val):    # Volume: lower (exhaustion at highs)
                    
                    result['bearish_price_vol'] = True
                    strength_score += 20
                    details.append(f"Bearish Price-Vol: Price HH, Volume declining")
            
            result['strength'] = min(100, strength_score)
            result['details'] = ' | '.join(details) if details else 'No divergence detected'
            
            return result
            
        except Exception as e:
            self.logger.error(f"[VOL_DIV] Error: {e}")
            return {
                'bullish_rsi_vol': False,
                'bearish_rsi_vol': False,
                'bullish_price_vol': False,
                'bearish_price_vol': False,
                'strength': 0,
                'details': f'Error: {e}'
            }
    
    def get_signal_modifier(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        Get signal modifier based on volume divergences.
        
        Returns:
            - modifier: float (-1.0 to +1.0)
            - reason: str
        """
        div = self.detect_divergences(df)
        
        modifier = 0.0
        reasons = []
        
        # Bullish signals
        if div['bullish_rsi_vol']:
            modifier += 0.5
            reasons.append("Bullish RSI-Vol Divergence")
        if div['bullish_price_vol']:
            modifier += 0.3
            reasons.append("Bullish Price-Vol Divergence")
        
        # Bearish signals
        if div['bearish_rsi_vol']:
            modifier -= 0.5
            reasons.append("Bearish RSI-Vol Divergence")
        if div['bearish_price_vol']:
            modifier -= 0.3
            reasons.append("Bearish Price-Vol Divergence")
        
        # Clamp to [-1.0, 1.0]
        modifier = max(-1.0, min(1.0, modifier))
        
        reason = ' + '.join(reasons) if reasons else 'No divergence'
        
        return modifier, reason