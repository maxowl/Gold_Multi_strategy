"""
Time-Price Engine.
Analyzes time-based price action, VWAP, and session boundaries.
Provides robust timezone handling for global markets.
"""
import pandas as pd
import numpy as np
import pytz
import logging
from typing import Optional


class TimePriceEngine:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.utc_tz = pytz.utc
        self.ny_tz = pytz.timezone('America/New_York')

    def _to_ny_time(self, timestamp) -> Optional[int]:
        """
        Convert timestamp to NY hour with DST safety.
        Prevents crashes during Daylight Saving Time transitions.
        """
        if timestamp is None:
            return None
        
        try:
            if not isinstance(timestamp, pd.Timestamp):
                timestamp = pd.to_datetime(timestamp)
            
            if timestamp.tzinfo is None:
                # [FIX] Handle non-existent times during DST spring forward
                timestamp = timestamp.tz_localize(self.utc_tz, nonexistent='shift_forward')
            else:
                timestamp = timestamp.tz_convert(self.utc_tz)
            
            # [FIX] Handle ambiguous times during DST fall back
            ny_time = timestamp.tz_convert(self.ny_tz)
            return ny_time.hour
            
        except Exception as e:
            self.logger.error(f"[FAIL] Time conversion error: {e}")
            return None

    def calculate_vwap(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Calculate Volume Weighted Average Price (VWAP) for the current session.
        Resets daily based on NY midnight.
        """
        if df is None or df.empty or 'time' not in df.columns:
            return None
            
        try:
            high = df['high'].to_numpy().astype(float)
            low = df['low'].to_numpy().astype(float)
            close = df['close'].to_numpy().astype(float)
            
            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].to_numpy().astype(float)
            elif 'volume' in df.columns:
                volume = df['volume'].to_numpy().astype(float)
            else:
                return None
                
            volume = np.nan_to_num(volume, nan=0.0)
            
            # Typical price
            typical_price = (high + low + close) / 3.0
            
            # Calculate NY hours to detect session resets (midnight NY)
            ny_hours = []
            for t in df['time']:
                h = self._to_ny_time(t)
                ny_hours.append(h if h is not None else -1)
                
            ny_hours = np.array(ny_hours)
            
            # Find session boundaries (where hour drops or crosses midnight)
            # Simplified: reset when hour goes from 23 to 0, or just use cumulative sum with reset
            vwap = np.zeros(len(df))
            cum_tp_vol = 0.0
            cum_vol = 0.0
            
            prev_hour = ny_hours[0]
            
            for i in range(len(df)):
                curr_hour = ny_hours[i]
                
                # Reset at NY Midnight (hour 0) or if data gap detected
                if curr_hour == 0 and prev_hour != 0 and i > 0:
                    cum_tp_vol = 0.0
                    cum_vol = 0.0
                    
                cum_tp_vol += typical_price[i] * volume[i]
                cum_vol += volume[i]
                
                if cum_vol > 0:
                    vwap[i] = cum_tp_vol / cum_vol
                else:
                    vwap[i] = typical_price[i]
                    
                prev_hour = curr_hour
                
            return pd.Series(vwap, index=df.index)
            
        except Exception as e:
            self.logger.error(f"[FAIL] VWAP calculation error: {e}")
            return None