"""
Strategy 4: Change of Character + Inducement (SMC).
Enters on market structure shifts (CHOCH) after liquidity sweeps.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.smc_engine import SMCStructuralEngine
from core.atr_cache import ATRCache
from core.fibonacci_engine import FibonacciEngine


class Strategy4_CHOCH_IDM(BaseStrategy):
    def __init__(self):
        super().__init__(name="S4_CHOCH_IDM", strategy_category="SMC", min_risk_reward=2.5)
        self.smc = SMCStructuralEngine()
        self.fib = FibonacciEngine()

    def evaluate(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m15, 50):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        swings_high, swings_low = self.smc.detect_swings(df_m15, order=3)
        
        # [FIX] Guard against insufficient swings to prevent IndexError
        if len(swings_high) < 2 or len(swings_low) < 2:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_swings"}}

        close = float(df_m15['close'].iloc[-1])
        prev_close = float(df_m15['close'].iloc[-2])
        
        atr_m15 = ATRCache.get_atr(df_m15, 14).iloc[-1]
        if pd.isna(atr_m15) or atr_m15 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        last_sh = float(df_m15['high'].iloc[swings_high[-1]])
        last_sl = float(df_m15['low'].iloc[swings_low[-1]])
        prev_sh = float(df_m15['high'].iloc[swings_high[-2]])
        prev_sl = float(df_m15['low'].iloc[swings_low[-2]])

        # Bullish CHOCH: Higher Low + Break above last Swing High
        if last_sl > prev_sl and close > last_sh and prev_close <= last_sh:
            entry_price = close
            sl_info = self.calculate_session_sl(entry_price, last_sl, df_m15, is_buy=True)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            fib_tp = self.fib.calculate_tp_from_fib(entry_price, sl_info['sl_price'], df_m15, is_buy=True, min_rr=self.min_risk_reward)
            if fib_tp['valid']:
                tp_price = fib_tp['tp_price']
            else:
                risk = abs(entry_price - sl_info['sl_price'])
                tp_price = entry_price + (3.0 * risk)
                
            signal = self.build_signal(
                "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.85, 
                extra_meta={'choch_level': last_sh}
            )
            self.log_signal_summary(signal)
            return signal

        # Bearish CHOCH: Lower High + Break below last Swing Low
        elif last_sh < prev_sh and close < last_sl and prev_close >= last_sl:
            entry_price = close
            sl_info = self.calculate_session_sl(entry_price, last_sh, df_m15, is_buy=False)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            fib_tp = self.fib.calculate_tp_from_fib(entry_price, sl_info['sl_price'], df_m15, is_buy=False, min_rr=self.min_risk_reward)
            if fib_tp['valid']:
                tp_price = fib_tp['tp_price']
            else:
                risk = abs(entry_price - sl_info['sl_price'])
                tp_price = entry_price - (3.0 * risk)
                
            signal = self.build_signal(
                "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.85, 
                extra_meta={'choch_level': last_sl}
            )
            self.log_signal_summary(signal)
            return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_choch"}}