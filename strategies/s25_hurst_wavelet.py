"""
Strategy 25: Hurst Exponent + Wavelet Denoise (Trend).
Enters when Hurst > 0.55 (trending) and wavelet-denoised slope confirms direction.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.hurst_wavelet_engine import HurstWaveletEngine
from core.atr_cache import ATRCache


class Strategy25_HurstWavelet(BaseStrategy):
    def __init__(self):
        super().__init__(name="S25_HurstWavelet", strategy_category="TREND", min_risk_reward=2.0)
        self.hw = HurstWaveletEngine()

    def evaluate(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m15, 100):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        hurst = self.hw.calculate_hurst_exponent(df_m15['close'], max_lag=50)
        denoised = self.hw.wavelet_denoise(df_m15['close'], wavelet='db4', level=3)
        
        # [FIX] Guard against None or empty denoised series
        if denoised is None or denoised.empty:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "wavelet_failed"}}

        slope = self.hw.calculate_wavelet_slope(denoised, window=5)
        
        if np.isnan(hurst) or np.isnan(slope):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "nan_values"}}

        close = float(df_m15['close'].iloc[-1])
        atr_m15 = ATRCache.get_atr(df_m15, 14).iloc[-1]
        if pd.isna(atr_m15) or atr_m15 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        # Trending Market Condition: Hurst > 0.55
        if hurst > 0.55:
            # Bullish Trend: Positive slope
            if slope > 0:
                entry_price = close
                recent_low = float(df_m15['low'].iloc[-10:].min())
                sl_info = self.calculate_session_sl(entry_price, recent_low, df_m15, is_buy=True)
                if not sl_info['valid']: 
                    return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
                
                risk = abs(entry_price - sl_info['sl_price'])
                tp_price = entry_price + (3.0 * risk)
                
                signal = self.build_signal(
                    "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, 
                    "M15", 0.85, requires_dynamic_exit=True, dynamic_exit_threshold="hurst_below_0.5",
                    extra_meta={'hurst': hurst, 'slope': slope}
                )
                self.log_signal_summary(signal)
                return signal

            # Bearish Trend: Negative slope
            elif slope < 0:
                entry_price = close
                recent_high = float(df_m15['high'].iloc[-10:].max())
                sl_info = self.calculate_session_sl(entry_price, recent_high, df_m15, is_buy=False)
                if not sl_info['valid']: 
                    return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
                
                risk = abs(entry_price - sl_info['sl_price'])
                tp_price = entry_price - (3.0 * risk)
                
                signal = self.build_signal(
                    "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, 
                    "M15", 0.85, requires_dynamic_exit=True, dynamic_exit_threshold="hurst_below_0.5",
                    extra_meta={'hurst': hurst, 'slope': slope}
                )
                self.log_signal_summary(signal)
                return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_trend_condition"}}