"""
Strategy 17: Chaos Gaussian Squeeze Breakout (Trend).
Enters on volatility expansion after a period of compression (squeeze).
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.chaos_engine import ChaosEngine
from core.atr_cache import ATRCache


class Strategy17_ChaosSqueeze(BaseStrategy):
    def __init__(self):
        super().__init__(name="S17_ChaosSqueeze", strategy_category="TREND", min_risk_reward=2.0)
        self.chaos = ChaosEngine()

    def evaluate(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m15, 50):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        upper, lower, width = self.chaos.calculate_gaussian_squeeze_band(df_m15, period=30, std_dev=2.0)
        
        # [FIX] Guard against empty series returned from Chaos Engine
        if upper.empty or lower.empty or width.empty:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "squeeze_calc_failed"}}

        # Clean NaNs from width series
        width_clean = width.dropna()
        if len(width_clean) < 2:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_width_data"}}

        w_curr = float(width_clean.iloc[-1])
        w_prev = float(width_clean.iloc[-2])
        
        close = float(df_m15['close'].iloc[-1])
        atr_m15 = ATRCache.get_atr(df_m15, 14).iloc[-1]
        if pd.isna(atr_m15) or atr_m15 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        # Calculate historical average width to define "Squeeze ON" vs "Squeeze OFF"
        avg_width = float(width_clean.iloc[-30:].mean()) if len(width_clean) >= 30 else float(width_clean.mean())
        
        # Squeeze Breakout Logic
        # Previous bar was in squeeze (narrow width), current bar is expanding (width > avg)
        if w_prev < avg_width * 0.8 and w_curr > avg_width:
            # Bullish Breakout: Close above upper band
            if close > float(upper.iloc[-1]):
                entry_price = close
                recent_low = float(df_m15['low'].iloc[-10:].min())
                sl_info = self.calculate_session_sl(entry_price, recent_low, df_m15, is_buy=True)
                if not sl_info['valid']: 
                    return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
                
                risk = abs(entry_price - sl_info['sl_price'])
                tp_price = entry_price + (3.0 * risk)
                
                signal = self.build_signal(
                    "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.80,
                    extra_meta={'squeeze_breakout': 'BULLISH', 'band_width': w_curr}
                )
                self.log_signal_summary(signal)
                return signal

            # Bearish Breakout: Close below lower band
            elif close < float(lower.iloc[-1]):
                entry_price = close
                recent_high = float(df_m15['high'].iloc[-10:].max())
                sl_info = self.calculate_session_sl(entry_price, recent_high, df_m15, is_buy=False)
                if not sl_info['valid']: 
                    return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
                
                risk = abs(entry_price - sl_info['sl_price'])
                tp_price = entry_price - (3.0 * risk)
                
                signal = self.build_signal(
                    "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.80,
                    extra_meta={'squeeze_breakout': 'BEARISH', 'band_width': w_curr}
                )
                self.log_signal_summary(signal)
                return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_breakout"}}