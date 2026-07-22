"""
Strategy 11: Liquidity Delta Divergence (Scalp).
Enters on M1 when price moves opposite to cumulative volume delta, indicating absorption.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.orderflow_engine import OrderFlowEngine
from core.atr_cache import ATRCache


class Strategy11_LiquidityDelta(BaseStrategy):
    def __init__(self):
        super().__init__(name="S11_LiquidityDelta", strategy_category="SCALP", min_risk_reward=1.5)
        self.of_engine = OrderFlowEngine()

    def evaluate(self, df_m1: pd.DataFrame, df_m5: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m1, 50):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        # Calculate Liquidity Delta
        delta_series = self.of_engine.calculate_liquidity_delta(df_m1, lookback=20)
        
        # [FIX] Guard against None or empty series to prevent IndexError
        if delta_series is None or len(delta_series) < 5:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "delta_calc_failed"}}

        close = float(df_m1['close'].iloc[-1])
        prev_close = float(df_m1['close'].iloc[-2])
        delta_curr = float(delta_series.iloc[-1])
        delta_prev = float(delta_series.iloc[-2])
        
        atr_m1 = ATRCache.get_atr(df_m1, 14).iloc[-1]
        if pd.isna(atr_m1) or atr_m1 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        # Bullish Divergence: Price makes lower low, but Delta makes higher low (Absorption of selling)
        if close < prev_close and delta_curr > delta_prev and delta_curr > 0:
            entry_price = close
            recent_low = float(df_m1['low'].iloc[-10:].min())
            sl_info = self.calculate_session_sl(entry_price, recent_low, df_m1, is_buy=True, atr_multiplier=1.0)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = entry_price + (2.0 * risk)
            
            signal = self.build_signal(
                "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, "M1", 0.70,
                extra_meta={
                    'delta_divergence': 'BULLISH', 
                    'friction_sensitive': True, 
                    'trailing_method': 'fixed_dollar'
                }
            )
            self.log_signal_summary(signal)
            return signal

        # Bearish Divergence: Price makes higher high, but Delta makes lower high (Absorption of buying)
        elif close > prev_close and delta_curr < delta_prev and delta_curr < 0:
            entry_price = close
            recent_high = float(df_m1['high'].iloc[-10:].max())
            sl_info = self.calculate_session_sl(entry_price, recent_high, df_m1, is_buy=False, atr_multiplier=1.0)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = entry_price - (2.0 * risk)
            
            signal = self.build_signal(
                "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, "M1", 0.70,
                extra_meta={
                    'delta_divergence': 'BEARISH', 
                    'friction_sensitive': True, 
                    'trailing_method': 'fixed_dollar'
                }
            )
            self.log_signal_summary(signal)
            return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_divergence"}}