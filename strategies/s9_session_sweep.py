"""
Strategy 9: Session Liquidity Sweep (Scalp).
Enters when price sweeps previous session high/low and rejects.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.session_volatility import SessionVolatilityManager
from core.atr_cache import ATRCache


class Strategy9_SessionSweep(BaseStrategy):
    def __init__(self):
        super().__init__(name="S9_SessionSweep", strategy_category="SCALP", min_risk_reward=1.5)
        self.session_mgr = SessionVolatilityManager()

    def evaluate(self, df_m5: pd.DataFrame, df_m15: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m5, 50) or 'time' not in df_m5.columns:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        current_session = self.session_mgr.get_current_session(df_m5['time'].iloc[-1])
        
        # Only trade during high liquidity sessions
        if current_session not in ['LONDON_OPEN', 'NY_OPEN', 'LONDON']:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": f"wrong_session_{current_session}"}}

        # Find the previous session's high and low
        # Simplified: Look back 24 bars (approx 2 hours on M5) to find session boundaries
        lookback = 24
        if len(df_m5) < lookback:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_history"}}
            
        prev_high = float(df_m5['high'].iloc[-lookback:-1].max())
        prev_low = float(df_m5['low'].iloc[-lookback:-1].min())
        
        close = float(df_m5['close'].iloc[-1])
        low = float(df_m5['low'].iloc[-1])
        high = float(df_m5['high'].iloc[-1])
        
        atr_m5 = ATRCache.get_atr(df_m5, 14).iloc[-1]
        if pd.isna(atr_m5) or atr_m5 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        # Bullish Sweep: Price swept previous low but closed back above it
        if low < prev_low and close > prev_low:
            entry_price = close
            sl_info = self.calculate_session_sl(entry_price, low, df_m5, is_buy=True, atr_multiplier=1.0)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = prev_high # Target the opposite side of the range
            
            # Fallback if range is too tight
            if abs(tp_price - entry_price) < risk * self.min_risk_reward:
                tp_price = entry_price + (2.0 * risk)
                
            signal = self.build_signal(
                "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, "M5", 0.75,
                extra_meta={
                    'sweep_level': prev_low, 
                    'friction_sensitive': True, 
                    'trailing_method': 'fixed_dollar'
                }
            )
            self.log_signal_summary(signal)
            return signal

        # Bearish Sweep: Price swept previous high but closed back below it
        elif high > prev_high and close < prev_high:
            entry_price = close
            sl_info = self.calculate_session_sl(entry_price, high, df_m5, is_buy=False, atr_multiplier=1.0)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = prev_low
            
            if abs(entry_price - tp_price) < risk * self.min_risk_reward:
                tp_price = entry_price - (2.0 * risk)
                
            signal = self.build_signal(
                "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, "M5", 0.75,
                extra_meta={
                    'sweep_level': prev_high, 
                    'friction_sensitive': True, 
                    'trailing_method': 'fixed_dollar'
                }
            )
            self.log_signal_summary(signal)
            return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_sweep"}}