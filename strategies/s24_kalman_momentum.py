"""
Strategy 24: Kalman Filter Momentum (Trend).
Enters when the fast Kalman filter crosses the slow filter, indicating a trend shift.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.kalman_squeeze_engine import KalmanSqueezeEngine
from core.atr_cache import ATRCache


class Strategy24_KalmanMomentum(BaseStrategy):
    def __init__(self):
        super().__init__(name="S24_KalmanMomentum", strategy_category="TREND", min_risk_reward=2.0)
        self.kalman = KalmanSqueezeEngine()

    def evaluate(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m15, 50):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        # Apply dual Kalman filters
        squeeze_result = self.kalman.apply_kalman_squeeze(df_m15['close'], fast_process_noise=0.1, slow_process_noise=0.001)
        
        # [FIX] Guard against None return from Kalman Engine
        if squeeze_result is None or 'fast_filter' not in squeeze_result or 'slow_filter' not in squeeze_result:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "kalman_calc_failed"}}

        fast = squeeze_result['fast_filter']
        slow = squeeze_result['slow_filter']
        
        if fast.empty or slow.empty or len(fast) < 2:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_filter_data"}}

        f_curr = float(fast.iloc[-1])
        f_prev = float(fast.iloc[-2])
        s_curr = float(slow.iloc[-1])
        s_prev = float(slow.iloc[-2])
        
        if np.isnan(f_curr) or np.isnan(s_curr):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "filter_nan"}}

        close = float(df_m15['close'].iloc[-1])
        atr_m15 = ATRCache.get_atr(df_m15, 14).iloc[-1]
        if pd.isna(atr_m15) or atr_m15 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        # Bullish Cross: Fast crosses above Slow
        if f_prev <= s_prev and f_curr > s_curr:
            entry_price = close
            recent_low = float(df_m15['low'].iloc[-10:].min())
            sl_info = self.calculate_session_sl(entry_price, recent_low, df_m15, is_buy=True)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = entry_price + (3.0 * risk) # Fallback, dynamic exit will handle
            
            signal = self.build_signal(
                "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, 
                "M15", 0.80, requires_dynamic_exit=True, dynamic_exit_threshold="kalman_cross"
            )
            self.log_signal_summary(signal)
            return signal

        # Bearish Cross: Fast crosses below Slow
        elif f_prev >= s_prev and f_curr < s_curr:
            entry_price = close
            recent_high = float(df_m15['high'].iloc[-10:].max())
            sl_info = self.calculate_session_sl(entry_price, recent_high, df_m15, is_buy=False)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = entry_price - (3.0 * risk)
            
            signal = self.build_signal(
                "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, 
                "M15", 0.80, requires_dynamic_exit=True, dynamic_exit_threshold="kalman_cross"
            )
            self.log_signal_summary(signal)
            return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_cross"}}