"""
Strategy 14: Propulsion Block Breakout (Trend).
Enters on strong momentum candles (Propulsion Blocks) indicating institutional activity.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.propulsion_engine import PropulsionEngine
from core.atr_cache import ATRCache
from core.fibonacci_engine import FibonacciEngine


class Strategy14_Propulsion(BaseStrategy):
    def __init__(self):
        super().__init__(name="S14_Propulsion", strategy_category="TREND", min_risk_reward=2.5)
        self.prop_engine = PropulsionEngine()
        self.fib = FibonacciEngine()

    def evaluate(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m15, 100):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        atr_m15 = ATRCache.get_atr(df_m15, 14).iloc[-1]
        if pd.isna(atr_m15) or atr_m15 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        # Detect Propulsion Blocks (candles > 2.0 ATR)
        propulsion_blocks = self.prop_engine.detect_propulsion_blocks(df_m15, lookback=100, min_atr_mult=2.0)
        
        if not propulsion_blocks:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_propulsion"}}

        close = float(df_m15['close'].iloc[-1])
        
        # Check if current price is tapping or breaking the most recent propulsion block
        for pb in reversed(propulsion_blocks):
            if pb['bar_index'] < len(df_m15) - 5: 
                continue # Only consider very recent blocks
                
            # Bullish Propulsion Breakout
            if pb['direction'] == 'BULLISH' and close > pb['high']:
                entry_price = close
                sl_info = self.calculate_session_sl(entry_price, float(pb['low']), df_m15, is_buy=True)
                if not sl_info['valid']: 
                    continue
                
                fib_tp = self.fib.calculate_tp_from_fib(entry_price, sl_info['sl_price'], df_m15, is_buy=True, min_rr=self.min_risk_reward)
                tp_price = fib_tp['tp_price'] if fib_tp['valid'] else entry_price + (3.5 * abs(entry_price - sl_info['sl_price']))
                
                signal = self.build_signal(
                    "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.85,
                    extra_meta={'propulsion_high': pb['high']}
                )
                self.log_signal_summary(signal)
                return signal

            # Bearish Propulsion Breakout
            elif pb['direction'] == 'BEARISH' and close < pb['low']:
                entry_price = close
                sl_info = self.calculate_session_sl(entry_price, float(pb['high']), df_m15, is_buy=False)
                if not sl_info['valid']: 
                    continue
                
                fib_tp = self.fib.calculate_tp_from_fib(entry_price, sl_info['sl_price'], df_m15, is_buy=False, min_rr=self.min_risk_reward)
                tp_price = fib_tp['tp_price'] if fib_tp['valid'] else entry_price - (3.5 * abs(entry_price - sl_info['sl_price']))
                
                signal = self.build_signal(
                    "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.85,
                    extra_meta={'propulsion_low': pb['low']}
                )
                self.log_signal_summary(signal)
                return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_breakout"}}