"""
Chandelier Exit Engine.
Calculates trailing stop based on highest high/lowest low minus ATR multiplier.
Used by TREND category strategies.
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional


class ChandelierEngine:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_trailing_stop(
        self, df: pd.DataFrame, is_buy: bool, current_sl: float,
        entry_price: float, current_price: float,
        lookback: int = 22, multiplier: float = 3.0, min_distance: float = 0.0
    ) -> Optional[float]:
        """
        Calculate Chandelier Exit trailing stop.
        
        Formula:
        BUY:  New SL = Max(Highest_High[lookback] - multiplier*ATR, Current_SL)
        SELL: New SL = Min(Lowest_Low[lookback] + multiplier*ATR, Current_SL)
        
        Guards:
        1. Ratchet Guard: Never widen SL
        2. Price Boundary Guard: SL must not cross current price
        3. Min Distance Guard: SL must respect minimum distance
        """
        if df is None or len(df) < lookback:
            return None
        
        try:
            high = df['high'].to_numpy()
            low = df['low'].to_numpy()
            
            # Calculate ATR if not in df
            if 'atr' in df.columns:
                atr = df['atr'].to_numpy()
            else:
                from core.atr_cache import ATRCache
                atr = ATRCache.get_atr(df, 14).to_numpy()
            
            current_atr = atr[-1]
            if pd.isna(current_atr) or current_atr <= 0:
                return None
            
            # Use nanmax/nanmin to handle NaN values safely
            recent_highs = high[-lookback:]
            recent_lows = low[-lookback:]
            
            if is_buy:
                # BUY: Hang stop below highest high
                highest_high = np.nanmax(recent_highs)
                if np.isnan(highest_high):
                    return current_sl
                
                raw_stop = highest_high - (multiplier * current_atr)
                
                # Ratchet Guard: Never move SL down (widen)
                if current_sl > 0 and raw_stop <= current_sl:
                    return current_sl
                
                # Price Boundary Guard: SL must be below current price
                if raw_stop >= current_price - min_distance:
                    return current_sl
                
                return float(raw_stop)
            
            else:
                # SELL: Hang stop above lowest low
                lowest_low = np.nanmin(recent_lows)
                if np.isnan(lowest_low):
                    return current_sl
                
                raw_stop = lowest_low + (multiplier * current_atr)
                
                # Ratchet Guard: Never move SL up (widen)
                if current_sl > 0 and raw_stop >= current_sl:
                    return current_sl
                
                # Price Boundary Guard: SL must be above current price
                if raw_stop <= current_price + min_distance:
                    return current_sl
                
                return float(raw_stop)
                
        except Exception as e:
            self.logger.error(f"[FAIL] Chandelier calculation error: {e}")
            return None