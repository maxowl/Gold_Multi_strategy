"""
Strategy 23: Midnight Judas Swing (Scalp).
Enters on false breakouts of the Asian session range during the NY Open.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.session_volatility import SessionVolatilityManager
from core.atr_cache import ATRCache


class Strategy23_MidnightJudas(BaseStrategy):
    def __init__(self):
        super().__init__(name="S23_MidnightJudas", strategy_category="SCALP", min_risk_reward=2.0)
        self.session_mgr = SessionVolatilityManager()

    def evaluate(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m15, 50) or 'time' not in df_m15.columns:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        current_session = self.session_mgr.get_current_session(df_m15['time'].iloc[-1])
        
        if current_session != 'NY_OPEN':
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": f"wrong_session_{current_session}"}}

        # Identify Asian Session Range (approx last 12-16 hours before NY Open)
        # On M15, 12 hours = 48 bars. Let's look back 50 to 10 bars ago.
        lookback_start = 50
        lookback_end = 10
        
        if len(df_m15) < lookback_start:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_history"}}
            
        asian_slice = df_m15.iloc[-lookback_start:-lookback_end]
        asian_high = float(asian_slice['high'].max())
        asian_low = float(asian_slice['low'].min())
        
        close = float(df_m15['close'].iloc[-1])
        low = float(df_m15['low'].iloc[-1])
        high = float(df_m15['high'].iloc[-1])
        
        atr_m15 = ATRCache.get_atr(df_m15, 14).iloc[-1]
        if pd.isna(atr_m15) or atr_m15 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        # Bullish Judas: Swept Asian Low, then reversed and closed back inside/above
        if low < asian_low and close > asian_low:
            entry_price = close
            sl_info = self.calculate_session_sl(entry_price, low, df_m15, is_buy=True, atr_multiplier=1.0)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = asian_high # Target opposite side of Asian range
            
            if abs(tp_price - entry_price) < risk * self.min_risk_reward:
                tp_price = entry_price + (2.0 * risk)
                
            signal = self.build_signal(
                "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.80,
                extra_meta={
                    'judas_type': 'BULLISH', 
                    'asian_range': f"{asian_low:.2f}-{asian_high:.2f}",
                    'friction_sensitive': True,
                    'trailing_method': 'fixed_dollar'
                }
            )
            self.log_signal_summary(signal)
            return signal

        # Bearish Judas: Swept Asian High, then reversed and closed back inside/below
        elif high > asian_high and close < asian_high:
            entry_price = close
            sl_info = self.calculate_session_sl(entry_price, high, df_m15, is_buy=False, atr_multiplier=1.0)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = asian_low
            
            if abs(entry_price - tp_price) < risk * self.min_risk_reward:
                tp_price = entry_price - (2.0 * risk)
                
            signal = self.build_signal(
                "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.80,
                extra_meta={
                    'judas_type': 'BEARISH', 
                    'asian_range': f"{asian_low:.2f}-{asian_high:.2f}",
                    'friction_sensitive': True,
                    'trailing_method': 'fixed_dollar'
                }
            )
            self.log_signal_summary(signal)
            return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_judas_swing"}}