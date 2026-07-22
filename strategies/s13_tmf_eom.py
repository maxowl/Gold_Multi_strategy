"""
Strategy 13: Twiggs Money Flow + Ease of Movement (Trend).
Enters when both volume-weighted momentum indicators confirm a strong directional move.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
# [FIX] Corrected import to match Batch 7 filename (volume_indicators.py)
from core.volume_indicators import VolumeIndicatorsEngine
from core.atr_cache import ATRCache


class Strategy13_TMF_EOM(BaseStrategy):
    def __init__(self):
        super().__init__(name="S13_TMF_EOM", strategy_category="TREND", min_risk_reward=2.0)
        self.vol_engine = VolumeIndicatorsEngine()

    def evaluate(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m15, 50):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        tmf = self.vol_engine.calculate_twiggs_money_flow(df_m15, period=21)
        eom = self.vol_engine.calculate_ease_of_movement(df_m15, period=14)
        
        if tmf is None or eom is None or len(tmf) < 2 or len(eom) < 2:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "indicator_calc_failed"}}

        tmf_curr = float(tmf.iloc[-1])
        eom_curr = float(eom.iloc[-1])
        
        # Guard against NaN values from indicator warmup
        if np.isnan(tmf_curr) or np.isnan(eom_curr):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "indicator_nan"}}

        close = float(df_m15['close'].iloc[-1])
        
        atr_m15 = ATRCache.get_atr(df_m15, 14).iloc[-1]
        if pd.isna(atr_m15) or atr_m15 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        # Bullish Confirmation: TMF > 0 (Buying pressure) and EOM > 0 (Upward ease)
        if tmf_curr > 0.05 and eom_curr > 0:
            entry_price = close
            recent_low = float(df_m15['low'].iloc[-10:].min())
            sl_info = self.calculate_session_sl(entry_price, recent_low, df_m15, is_buy=True)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = entry_price + (2.5 * risk)
            
            signal = self.build_signal(
                "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.80,
                extra_meta={'tmf': tmf_curr, 'eom': eom_curr}
            )
            self.log_signal_summary(signal)
            return signal

        # Bearish Confirmation: TMF < 0 (Selling pressure) and EOM < 0 (Downward ease)
        elif tmf_curr < -0.05 and eom_curr < 0:
            entry_price = close
            recent_high = float(df_m15['high'].iloc[-10:].max())
            sl_info = self.calculate_session_sl(entry_price, recent_high, df_m15, is_buy=False)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = entry_price - (2.5 * risk)
            
            signal = self.build_signal(
                "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.80,
                extra_meta={'tmf': tmf_curr, 'eom': eom_curr}
            )
            self.log_signal_summary(signal)
            return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_confirmation"}}