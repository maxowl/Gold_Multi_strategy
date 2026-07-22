"""
Strategy 18: Ehlers Digital Vector Oscillator (Mean Reversion).
Enters on momentum exhaustion detected by the vector oscillator crossing its signal line at extremes.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.dsp_ehlers_engine import EhlersDSPEngine
from core.atr_cache import ATRCache


class Strategy18_EhlersVector(BaseStrategy):
    def __init__(self):
        super().__init__(name="S18_EhlersVector", strategy_category="MEAN_REVERSION", min_risk_reward=1.5)
        self.dsp = EhlersDSPEngine()

    def evaluate(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m15, 50):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        # Calculate dominant period
        period = self.dsp.homodyne_discriminator(df_m15['high'], df_m15['low'])
        if period is None or period.empty:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "homodyne_failed"}}

        # Calculate vector oscillator using the dominant period
        vector, signal = self.dsp.digital_vector_oscillator(df_m15['close'], period)
        
        # [FIX] Guard against empty series and NaN values
        if vector.empty or signal.empty:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "vector_calc_failed"}}
            
        v_clean = vector.dropna()
        s_clean = signal.dropna()
        if len(v_clean) < 2 or len(s_clean) < 2:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_vector_data"}}

        v_curr = float(v_clean.iloc[-1])
        v_prev = float(v_clean.iloc[-2])
        s_curr = float(s_clean.iloc[-1])
        s_prev = float(s_clean.iloc[-2])
        
        close = float(df_m15['close'].iloc[-1])
        atr_m15 = ATRCache.get_atr(df_m15, 14).iloc[-1]
        if pd.isna(atr_m15) or atr_m15 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        # Mean Reversion Logic at Extremes
        # Bullish Reversal: Vector was below -50 (extreme oversold) and crosses back above signal
        if v_prev < -50 and v_curr > s_curr and v_prev <= s_prev:
            entry_price = close
            recent_low = float(df_m15['low'].iloc[-10:].min())
            sl_info = self.calculate_session_sl(entry_price, recent_low, df_m15, is_buy=True, atr_multiplier=1.5)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = entry_price + (2.0 * risk)
            
            signal_dict = self.build_signal(
                "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.75,
                extra_meta={'vector_value': v_curr}
            )
            self.log_signal_summary(signal_dict)
            return signal_dict

        # Bearish Reversal: Vector was above +50 (extreme overbought) and crosses back below signal
        elif v_prev > 50 and v_curr < s_curr and v_prev >= s_prev:
            entry_price = close
            recent_high = float(df_m15['high'].iloc[-10:].max())
            sl_info = self.calculate_session_sl(entry_price, recent_high, df_m15, is_buy=False, atr_multiplier=1.5)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = entry_price - (2.0 * risk)
            
            signal_dict = self.build_signal(
                "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.75,
                extra_meta={'vector_value': v_curr}
            )
            self.log_signal_summary(signal_dict)
            return signal_dict

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_extreme_cross"}}