"""
Strategy 10: Ehlers MESA MAMA/FAMA Crossover (Trend).
Enters on adaptive moving average crossovers indicating trend shifts.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.dsp_ehlers_engine import EhlersDSPEngine
from core.atr_cache import ATRCache


class Strategy10_EhlersMESA(BaseStrategy):
    def __init__(self):
        super().__init__(name="S10_EhlersMESA", strategy_category="TREND", min_risk_reward=2.0)
        self.dsp = EhlersDSPEngine()

    def evaluate(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m15, 50):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        mama, fama = self.dsp.ehlers_mesa(df_m15['close'])
        
        # [FIX] Guard against empty series to prevent IndexError
        if mama.empty or fama.empty or len(mama) < 2:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "mesa_calc_failed"}}

        mama_curr = float(mama.iloc[-1])
        mama_prev = float(mama.iloc[-2])
        fama_curr = float(fama.iloc[-1])
        fama_prev = float(fama.iloc[-2])
        
        close = float(df_m15['close'].iloc[-1])
        atr_m15 = ATRCache.get_atr(df_m15, 14).iloc[-1]
        if pd.isna(atr_m15) or atr_m15 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        # Bullish Crossover: MAMA crosses above FAMA
        if mama_prev <= fama_prev and mama_curr > fama_curr:
            entry_price = close
            recent_low = float(df_m15['low'].iloc[-10:].min())
            sl_info = self.calculate_session_sl(entry_price, recent_low, df_m15, is_buy=True)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = entry_price + (3.0 * risk) # Fallback TP, Dynamic Exit will handle actual exit
            
            signal = self.build_signal(
                "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, 
                "M15", 0.80, requires_dynamic_exit=True, dynamic_exit_threshold="mama_fama_cross"
            )
            self.log_signal_summary(signal)
            return signal

        # Bearish Crossover: MAMA crosses below FAMA
        elif mama_prev >= fama_prev and mama_curr < fama_curr:
            entry_price = close
            recent_high = float(df_m15['high'].iloc[-10:].max())
            sl_info = self.calculate_session_sl(entry_price, recent_high, df_m15, is_buy=False)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = entry_price - (3.0 * risk)
            
            signal = self.build_signal(
                "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, 
                "M15", 0.80, requires_dynamic_exit=True, dynamic_exit_threshold="mama_fama_cross"
            )
            self.log_signal_summary(signal)
            return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_cross"}}