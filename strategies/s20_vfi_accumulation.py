"""
Strategy 20: Volume Flow Index Accumulation (Trend).
Enters on VFI zero-line crossovers indicating institutional accumulation or distribution.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.volume_flow_engine import VolumeFlowEngine
from core.atr_cache import ATRCache


class Strategy20_VFIAccumulation(BaseStrategy):
    def __init__(self):
        super().__init__(name="S20_VFIAccumulation", strategy_category="TREND", min_risk_reward=2.0)
        self.vf_engine = VolumeFlowEngine()

    def evaluate(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m15, 50):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        vfi = self.vf_engine.calculate_volume_flow_index(df_m15, period=21, coef=0.2, vcoef=2.5)
        
        # [FIX] Guard against None, empty series, and NaN values
        if vfi is None or vfi.empty:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "vfi_calc_failed"}}
            
        vfi_clean = vfi.dropna()
        if len(vfi_clean) < 2:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_vfi_data"}}

        v_curr = float(vfi_clean.iloc[-1])
        v_prev = float(vfi_clean.iloc[-2])
        
        # [FIX] Explicit NaN check to prevent ValueError in comparisons
        if np.isnan(v_curr) or np.isnan(v_prev):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "vfi_nan"}}

        close = float(df_m15['close'].iloc[-1])
        atr_m15 = ATRCache.get_atr(df_m15, 14).iloc[-1]
        if pd.isna(atr_m15) or atr_m15 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        # Bullish Zero Cross: VFI crosses from negative to positive (Accumulation)
        if v_prev <= 0 and v_curr > 0:
            entry_price = close
            recent_low = float(df_m15['low'].iloc[-10:].min())
            sl_info = self.calculate_session_sl(entry_price, recent_low, df_m15, is_buy=True)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = entry_price + (2.5 * risk)
            
            signal = self.build_signal(
                "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.80,
                extra_meta={'vfi_value': v_curr}
            )
            self.log_signal_summary(signal)
            return signal

        # Bearish Zero Cross: VFI crosses from positive to negative (Distribution)
        elif v_prev >= 0 and v_curr < 0:
            entry_price = close
            recent_high = float(df_m15['high'].iloc[-10:].max())
            sl_info = self.calculate_session_sl(entry_price, recent_high, df_m15, is_buy=False)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = entry_price - (2.5 * risk)
            
            signal = self.build_signal(
                "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.80,
                extra_meta={'vfi_value': v_curr}
            )
            self.log_signal_summary(signal)
            return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_cross"}}