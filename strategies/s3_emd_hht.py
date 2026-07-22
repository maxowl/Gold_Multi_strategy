"""
Strategy 3: EMD + Hilbert Phase Reversal (Trend).
Enters when the dominant cycle phase crosses zero, indicating a trend shift.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.dsp_engine import DSPEngine
from core.atr_cache import ATRCache


class Strategy3_EMD_HHT(BaseStrategy):
    def __init__(self):
        super().__init__(name="S3_EMD_HHT", strategy_category="TREND", min_risk_reward=2.0)
        self.dsp = DSPEngine()

    def evaluate(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m15, 50):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        # Extract first Intrinsic Mode Function (IMF)
        imf = self.dsp.empirical_mode_decomposition(df_m15['close'], max_imfs=1)
        if imf is None or len(imf) < 20:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "emd_failed"}}

        # Calculate instantaneous phase
        phase = self.dsp.hilbert_phase(imf)
        if phase is None or len(phase) < 2:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "hilbert_failed"}}

        phase_curr = float(phase.iloc[-1])
        phase_prev = float(phase.iloc[-2])
        
        close = float(df_m15['close'].iloc[-1])
        atr_m15 = ATRCache.get_atr(df_m15, 14).iloc[-1]
        if pd.isna(atr_m15) or atr_m15 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        # Bullish Phase Cross (Negative to Positive)
        if phase_prev < 0 and phase_curr >= 0:
            entry_price = close
            recent_low = float(df_m15['low'].iloc[-10:].min())
            sl_info = self.calculate_session_sl(entry_price, recent_low, df_m15, is_buy=True)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = entry_price + (3.0 * risk)
            
            signal = self.build_signal(
                "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, 
                "M15", 0.80, requires_dynamic_exit=True, dynamic_exit_threshold="phase_reversal"
            )
            self.log_signal_summary(signal)
            return signal

        # Bearish Phase Cross (Positive to Negative)
        elif phase_prev > 0 and phase_curr <= 0:
            entry_price = close
            recent_high = float(df_m15['high'].iloc[-10:].max())
            sl_info = self.calculate_session_sl(entry_price, recent_high, df_m15, is_buy=False)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = entry_price - (3.0 * risk)
            
            signal = self.build_signal(
                "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, 
                "M15", 0.80, requires_dynamic_exit=True, dynamic_exit_threshold="phase_reversal"
            )
            self.log_signal_summary(signal)
            return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_cross"}}