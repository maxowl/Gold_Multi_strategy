"""
Strategy 22: Wyckoff Spring Accumulation (Mean Reversion).
Enters on Wyckoff Spring patterns confirmed by No Supply volume condition.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.wyckoff_vsa_engine import WyckoffVSAEngine
from core.atr_cache import ATRCache
from core.fibonacci_engine import FibonacciEngine


class Strategy22_WyckoffSpring(BaseStrategy):
    def __init__(self):
        super().__init__(name="S22_WyckoffSpring", strategy_category="MEAN_REVERSION", min_risk_reward=2.0)
        self.wyckoff = WyckoffVSAEngine()
        self.fib = FibonacciEngine()

    def evaluate(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m15, 50):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        spring = self.wyckoff.detect_spring(df_m15, lookback=50)
        
        if spring is None:
            upthrust = self.wyckoff.detect_upthrust(df_m15, lookback=50)
            if upthrust is None:
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_pattern"}}
            return self._evaluate_upthrust(df_m15, upthrust)
        
        support_price = spring.get('support_price', 0.0)
        if support_price == 0.0 or np.isnan(support_price):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "invalid_support"}}

        no_supply = self.wyckoff.detect_no_supply(df_m15, lookback=20)
        
        close = float(df_m15['close'].iloc[-1])
        atr_m15 = ATRCache.get_atr(df_m15, 14).iloc[-1]
        if pd.isna(atr_m15) or atr_m15 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        if close > support_price:
            entry_price = close
            spring_low = float(spring.get('low', support_price))
            
            sl_info = self.calculate_session_sl(entry_price, spring_low, df_m15, is_buy=True, atr_multiplier=1.0)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            fib_tp = self.fib.calculate_tp_from_fib(entry_price, sl_info['sl_price'], df_m15, is_buy=True, min_rr=self.min_risk_reward)
            if fib_tp['valid']:
                tp_price = fib_tp['tp_price']
            else:
                risk = abs(entry_price - sl_info['sl_price'])
                tp_price = entry_price + (2.5 * risk)
            
            confidence = 0.85 if no_supply else 0.65
            
            signal = self.build_signal(
                "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", confidence,
                extra_meta={'pattern': 'SPRING', 'no_supply': no_supply}
            )
            self.log_signal_summary(signal)
            return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_recovery"}}

    def _evaluate_upthrust(self, df_m15: pd.DataFrame, upthrust: dict) -> dict:
        resistance_price = upthrust.get('resistance_price', 0.0)
        if resistance_price == 0.0 or np.isnan(resistance_price):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "invalid_resistance"}}

        no_demand = self.wyckoff.detect_no_demand(df_m15, lookback=20)
        
        close = float(df_m15['close'].iloc[-1])
        atr_m15 = ATRCache.get_atr(df_m15, 14).iloc[-1]
        if pd.isna(atr_m15) or atr_m15 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        if close < resistance_price:
            entry_price = close
            upthrust_high = float(upthrust.get('high', resistance_price))
            
            sl_info = self.calculate_session_sl(entry_price, upthrust_high, df_m15, is_buy=False, atr_multiplier=1.0)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            fib_tp = self.fib.calculate_tp_from_fib(entry_price, sl_info['sl_price'], df_m15, is_buy=False, min_rr=self.min_risk_reward)
            if fib_tp['valid']:
                tp_price = fib_tp['tp_price']
            else:
                risk = abs(entry_price - sl_info['sl_price'])
                tp_price = entry_price - (2.5 * risk)
            
            confidence = 0.85 if no_demand else 0.65
            
            signal = self.build_signal(
                "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", confidence,
                extra_meta={'pattern': 'UPTHRUST', 'no_demand': no_demand}
            )
            self.log_signal_summary(signal)
            return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_recovery"}}