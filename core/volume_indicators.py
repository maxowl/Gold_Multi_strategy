"""
Volume Indicators Engine.
Provides Twiggs Money Flow, Ease of Movement, and Chaikin Money Flow indicators.
Used by S13_TMF_EOM and other volume-based strategies.
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional


class VolumeIndicatorsEngine:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_twiggs_money_flow(self, df: pd.DataFrame, period: int = 21) -> Optional[pd.Series]:
        """
        Calculate Twiggs Money Flow (TMF).
        Measures the flow of money into and out of a security, adjusting for volatility.
        """
        if df is None or len(df) < period:
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
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            
            tr_high = np.maximum(high, prev_close)
            tr_low = np.minimum(low, prev_close)
            
            tr_range = tr_high - tr_low
            tr_range = np.where(tr_range == 0, 1e-10, tr_range)
            
            # Money Flow Multiplier
            mfm = ((close - tr_low) - (tr_high - close)) / tr_range
            mfv = mfm * volume
            
            mfv_series = pd.Series(mfv)
            vol_series = pd.Series(volume)
            
            sum_mfv = mfv_series.rolling(period, min_periods=period).sum()
            sum_vol = vol_series.rolling(period, min_periods=period).sum()
            
            tmf = sum_mfv / (sum_vol + 1e-10)
            return tmf
            
        except Exception as e:
            self.logger.error(f"[FAIL] TMF calculation error: {e}")
            return None

    def calculate_ease_of_movement(self, df: pd.DataFrame, period: int = 14) -> Optional[pd.Series]:
        """
        Calculate Ease of Movement (EOM).
        Measures the relationship between price change and volume.
        [FIX] Prevents extreme infinity spikes on Doji bars (where high == low).
        """
        if df is None or len(df) < period:
            return None
        
        try:
            high = df['high'].to_numpy().astype(float)
            low = df['low'].to_numpy().astype(float)
            
            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].to_numpy().astype(float)
            elif 'volume' in df.columns:
                volume = df['volume'].to_numpy().astype(float)
            else:
                return None
            
            volume = np.nan_to_num(volume, nan=0.0)
            
            midpoint = (high + low) / 2.0
            prev_midpoint = np.roll(midpoint, 1)
            prev_midpoint[0] = midpoint[0]
            distance = midpoint - prev_midpoint
            
            range_ = high - low
            
            # [FIX] Identify valid ranges to prevent division by near-zero which causes infinity spikes
            valid_range_mask = range_ > 1e-10
            range_safe = np.where(valid_range_mask, range_, 1.0)
            
            box_ratio = (volume / 100000000.0) / range_safe
            box_ratio = np.where(box_ratio == 0, 1e-10, box_ratio)
            
            one_period_eom = distance / box_ratio
            
            # [FIX] Zero out EOM for bars with zero/negligible range to prevent math explosion
            one_period_eom = np.where(valid_range_mask, one_period_eom, 0.0)
            
            eom = pd.Series(one_period_eom).rolling(period, min_periods=period).mean()
            return eom
            
        except Exception as e:
            self.logger.error(f"[FAIL] EOM calculation error: {e}")
            return None

    def calculate_chaikin_money_flow(self, df: pd.DataFrame, period: int = 20) -> Optional[pd.Series]:
        """
        Calculate Chaikin Money Flow (CMF).
        Measures the amount of Money Flow Volume over a specific period.
        """
        if df is None or len(df) < period:
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
            
            range_ = high - low
            range_ = np.where(range_ == 0, 1e-10, range_)
            
            mfm = ((close - low) - (high - close)) / range_
            mfv = mfm * volume
            
            mfv_series = pd.Series(mfv)
            vol_series = pd.Series(volume)
            
            sum_mfv = mfv_series.rolling(period, min_periods=period).sum()
            sum_vol = vol_series.rolling(period, min_periods=period).sum()
            
            cmf = sum_mfv / (sum_vol + 1e-10)
            return cmf
            
        except Exception as e:
            self.logger.error(f"[FAIL] CMF calculation error: {e}")
            return None