"""
Fibonacci Engine.
Calculates Fibonacci retracement and extension levels for TP calculation.
Used by S1, S4, S7, S14, S22 strategies.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

from core.smc_engine import SMCStructuralEngine


class FibonacciEngine:
    """
    Engine for Fibonacci-based price level calculations.
    """
    
    # Standard Fibonacci levels
    RETRACEMENT_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786]
    EXTENSION_LEVELS = [1.000, 1.272, 1.618, 2.000, 2.618]
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.smc = SMCStructuralEngine()
    
    def calculate_retracement_levels(
        self, swing_high: float, swing_low: float, is_uptrend: bool = True
    ) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels.
        
        Args:
            swing_high: High price of the swing
            swing_low: Low price of the swing
            is_uptrend: If True, retracement is from high (looking for support)
                       If False, retracement is from low (looking for resistance)
        
        Returns:
            Dict with level names as keys and prices as values
        """
        if swing_high <= swing_low:
            self.logger.warning(f"[FIB] Invalid swing: high ({swing_high}) <= low ({swing_low})")
            return {}
        
        swing_range = swing_high - swing_low
        levels = {}
        
        for fib_level in self.RETRACEMENT_LEVELS:
            if is_uptrend:
                # Retracement from high (looking for support in uptrend)
                price = swing_high - (swing_range * fib_level)
            else:
                # Retracement from low (looking for resistance in downtrend)
                price = swing_low + (swing_range * fib_level)
            
            level_name = f"fib_{fib_level:.3f}".replace('.', '_')
            levels[level_name] = round(price, 2)
        
        # Add swing high and low
        levels['swing_high'] = round(swing_high, 2)
        levels['swing_low'] = round(swing_low, 2)
        
        return levels
    
    def calculate_extension_levels(
        self, a_price: float, b_price: float, c_price: float, is_buy: bool = True
    ) -> Dict[str, float]:
        """
        Calculate Fibonacci extension levels using A-B-C projection.
        
        A-B-C Projection:
        - A: Start of first move
        - B: End of first move (first swing)
        - C: End of retracement (pullback point)
        
        Extension = C + (B - A) * fib_level
        
        Args:
            a_price: Price at point A (start)
            b_price: Price at point B (first swing)
            c_price: Price at point C (pullback)
            is_buy: True for bullish projection, False for bearish
        
        Returns:
            Dict with level names as keys and prices as values
        """
        ab_range = abs(b_price - a_price)  # Use absolute value for both directions
        
        if ab_range == 0:
            self.logger.warning("[FIB] A-B range is zero, cannot calculate extensions")
            return {}
        
        levels = {}
        
        for fib_level in self.EXTENSION_LEVELS:
            if is_buy:
                # Bullish extension: C + (B - A) * level
                price = c_price + (ab_range * fib_level)
            else:
                # Bearish extension: C - (A - B) * level
                price = c_price - (ab_range * fib_level)
            
            level_name = f"ext_{fib_level:.3f}".replace('.', '_')
            levels[level_name] = round(price, 2)
        
        # Add reference points
        levels['point_a'] = round(a_price, 2)
        levels['point_b'] = round(b_price, 2)
        levels['point_c'] = round(c_price, 2)
        
        return levels
    
    def detect_swing_points(
        self, df: pd.DataFrame, order: int = 5, lookback: int = 50
    ) -> Optional[Dict]:
        """
        Detect recent swing high and low for Fibonacci calculations.
        
        Returns:
            Dict with 'swing_high', 'swing_low', 'high_idx', 'low_idx'
            or None if insufficient data
        """
        if df is None or len(df) < (2 * order + 1):
            return None
        
        # Use SMC engine to detect swings
        swing_highs, swing_lows = self.smc.detect_swings(df, order=order)
        
        if not swing_highs or not swing_lows:
            return None
        
        # Get most recent swings within lookback
        current_idx = len(df) - 1
        min_idx = max(0, current_idx - lookback)
        
        recent_highs = [idx for idx in swing_highs if idx >= min_idx]
        recent_lows = [idx for idx in swing_lows if idx >= min_idx]
        
        if not recent_highs or not recent_lows:
            return None
        
        # Get the most recent swing high and low
        high_idx = max(recent_highs)
        low_idx = max(recent_lows)
        
        swing_high = float(df['high'].iloc[high_idx])
        swing_low = float(df['low'].iloc[low_idx])
        
        return {
            'swing_high': swing_high,
            'swing_low': swing_low,
            'high_idx': high_idx,
            'low_idx': low_idx,
            'is_uptrend': high_idx > low_idx  # If high came after low, likely uptrend
        }
    
    def find_nearest_fib_level(
        self, current_price: float, fib_levels: Dict[str, float], tolerance: float = 0.5
    ) -> Optional[Dict]:
        """
        Find the nearest Fibonacci level to current price.
        
        Args:
            current_price: Current market price
            fib_levels: Dict of Fibonacci levels
            tolerance: Maximum distance to consider as "near"
        
        Returns:
            Dict with 'level_name', 'level_price', 'distance', 'is_support'
            or None if no level within tolerance
        """
        nearest = None
        min_distance = float('inf')
        
        for level_name, level_price in fib_levels.items():
            if level_name.startswith('swing_') or level_name.startswith('point_'):
                continue  # Skip reference points
            
            distance = abs(current_price - level_price)
            
            if distance < min_distance and distance <= tolerance:
                min_distance = distance
                nearest = {
                    'level_name': level_name,
                    'level_price': level_price,
                    'distance': distance,
                    'is_support': current_price > level_price  # If price above, it's support
                }
        
        return nearest
    
    def calculate_tp_from_fib(
        self, entry_price: float, sl_price: float, df: pd.DataFrame, 
        is_buy: bool, min_rr: float = 2.0
    ) -> Dict:
        """
        Calculate Take Profit using Fibonacci extensions.
        Wrapper method for backward compatibility.
        
        Returns:
            Dict with 'valid', 'tp_price', 'reason', 'risk_reward'
        """
        swings = self.detect_swing_points(df, order=3, lookback=50)
        if swings is None:
            return {'valid': False, 'tp_price': 0, 'reason': 'no swings detected'}
        
        risk = abs(entry_price - sl_price)
        if risk == 0:
            return {'valid': False, 'tp_price': 0, 'reason': 'zero risk'}
        
        # Determine A, B, C points
        if is_buy:
            a_price = swings['swing_low']
            b_price = swings['swing_high']
            c_price = entry_price
        else:
            a_price = swings['swing_high']
            b_price = swings['swing_low']
            c_price = entry_price
        
        # Calculate extensions
        extensions = self.calculate_extension_levels(a_price, b_price, c_price, is_buy)
        
        # Find first extension level that meets min_rr
        for level_name, level_price in extensions.items():
            if level_name.startswith('point_'):
                continue
            
            reward = abs(level_price - entry_price)
            rr = reward / risk
            
            if rr >= min_rr:
                # Extract fib level from name (e.g., 'ext_1_272' -> 1.272)
                fib_level = float(level_name.replace('ext_', '').replace('_', '.'))
                
                return {
                    'valid': True,
                    'tp_price': round(level_price, 2),
                    'reason': f'Fibonacci Extension ({fib_level:.3f})',
                    'risk_reward': round(rr, 2),
                    'fib_level': fib_level
                }
        
        return {'valid': False, 'tp_price': 0, 'reason': 'no suitable extension'}