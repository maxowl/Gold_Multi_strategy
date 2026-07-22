"""
Strategy 16: Roofing Filter + EMD Mean Reversion.
Uses Ehlers Roofing Filter to isolate the dominant cycle and enters on cycle extremes.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.dsp_engine import DSPEngine
from core.atr_cache import ATRCache


class Strategy16_RoofingEMD(BaseStrategy):
    def __init__(self):
        super().__init__(name="S16_RoofingEMD", strategy_category="MEAN_REVERSION", min_risk_reward=1.5)
        self.dsp = DSPEngine()

    def evaluate(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m15, 50):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        # Apply Roofing Filter to extract cycle
        roofing = self.dsp.roofing_filter(df_m15['close'])
        
        # [FIX] Guard against None, empty series, and NaN values from warmup
        if roofing is None or roofing.empty:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "roofing_calc_failed"}}
            
        roofing_clean = roofing.dropna()
        if len(roofing_clean) < 3:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_cycle_data"}}

        r_curr = float(roofing_clean.iloc[-1])
        r_prev = float(roofing_clean.iloc[-2])
        r_prev2 = float(roofing_clean.iloc[-3])
        
        close = float(df_m15['close'].iloc[-1])
        atr_m15 = ATRCache.get_atr(df_m15, 14).iloc[-1]
        if pd.isna(atr_m15) or atr_m15 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        # Bullish Cycle Turn: Cycle was falling, hits bottom, and starts rising
        if r_prev2 > r_prev and r_curr > r_prev and r_curr < 0:
            entry_price = close
            recent_low = float(df_m15['low'].iloc[-10:].min())
            sl_info = self.calculate_session_sl(entry_price, recent_low, df_m15, is_buy=True, atr_multiplier=2.0)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = entry_price + (2.5 * risk)
            
            signal = self.build_signal(
                "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.75,
                extra_meta={'cycle_value': r_curr}
            )
            self.log_signal_summary(signal)
            return signal

        # Bearish Cycle Turn: Cycle was rising, hits top, and starts falling
        elif r_prev2 < r_prev and r_curr < r_prev and r_curr > 0:
            entry_price = close
            recent_high = float(df_m15['high'].iloc[-10:].max())
            sl_info = self.calculate_session_sl(entry_price, recent_high, df_m15, is_buy=False, atr_multiplier=2.0)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = entry_price - (2.5 * risk)
            
            signal = self.build_signal(
                "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.75,
                extra_meta={'cycle_value': r_curr}
            )
            self.log_signal_summary(signal)
            return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_turn"}}