"""
Volume Flow Engine.
Calculates Volume Flow Index (VFI) to detect institutional accumulation/distribution.
Used by S20_VFIAccumulation strategy.
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional


class VolumeFlowEngine:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_volume_flow_index(self, df: pd.DataFrame, period: int = 21, coef: float = 0.2, vcoef: float = 2.5) -> Optional[pd.Series]:
        """
        Calculate Volume Flow Index (VFI).
        VFI > 0 indicates accumulation (uptrend), VFI < 0 indicates distribution (downtrend).
        """
        if df is None or len(df) < period:
            return None
            
        try:
            close = df['close'].to_numpy().astype(float)
            
            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].to_numpy().astype(float)
            elif 'volume' in df.columns:
                volume = df['volume'].to_numpy().astype(float)
            else:
                return None
                
            volume = np.nan_to_num(volume, nan=0.0)
            
            # Price change
            price_change = np.diff(close)
            price_change = np.insert(price_change, 0, 0.0)
            
            # Cutoff: coef * StdDev of price changes
            std_price = pd.Series(price_change).rolling(period, min_periods=period).std().to_numpy()
            # [FIX] Use 0.0 so cutoff becomes 0 instead of NaN, preventing invalid comparisons
            std_price = np.nan_to_num(std_price, nan=0.0)
            cutoff = coef * std_price
            
            # Volume cutoff: vcoef * average volume
            vol_avg = pd.Series(volume).rolling(period, min_periods=period).mean().to_numpy()
            vol_avg = np.nan_to_num(vol_avg, nan=0.0)
            vol_cutoff = vcoef * vol_avg
            
            # [FIX] Replace 0 or NaN in vol_cutoff with infinity to safely skip abnormal/invalid bars
            vol_cutoff = np.where(vol_cutoff <= 0, np.inf, vol_cutoff)
            
            # Calculate VFI components
            vfi_components = np.zeros(len(df))
            
            for i in range(1, len(df)):
                # Skip if volume is below the volume cutoff
                if volume[i] > vol_cutoff[i]:
                    continue
                    
                if price_change[i] > cutoff[i]:
                    vfi_components[i] = volume[i]
                elif price_change[i] < -cutoff[i]:
                    vfi_components[i] = -volume[i]
                else:
                    vfi_components[i] = 0.0
                    
            # VFI is the sum of components over the period, normalized by average volume
            vfi_sum = pd.Series(vfi_components).rolling(period, min_periods=period).sum().to_numpy()
            
            # Prevent division by zero
            vol_avg_safe = np.where(vol_avg == 0, 1e-10, vol_avg)
            
            vfi = vfi_sum / vol_avg_safe
            
            return pd.Series(vfi, index=df.index)
            
        except Exception as e:
            self.logger.error(f"[FAIL] VFI calculation error: {e}")
            return None