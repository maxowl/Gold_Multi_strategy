"""
Strategy 1: Inverse Order Block Rejection (SMC).
Enters on M15 Order Block taps confirmed by H1 trend direction.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.smc_engine import SMCStructuralEngine
from core.atr_cache import ATRCache
from core.fibonacci_engine import FibonacciEngine


class Strategy1_IOB_Rejection(BaseStrategy):
    def __init__(self):
        super().__init__(name="S1_IOB_Rejection", strategy_category="SMC", min_risk_reward=2.0)
        self.smc = SMCStructuralEngine()
        self.fib = FibonacciEngine()

    def evaluate(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m15, 50) or not self._validate_data(df_h1, 50):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        # Check H1 trend using EMA 50
        h1_close = df_h1['close'].to_numpy()
        h1_ema50 = pd.Series(h1_close).ewm(span=50, adjust=False).mean().to_numpy()
        h1_trend_bull = h1_close[-1] > h1_ema50[-1]
        h1_trend_bear = h1_close[-1] < h1_ema50[-1]

        # Detect Order Blocks on M15
        obs = self.smc.detect_order_blocks(df_m15, lookback=100)
        if not obs:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_obs"}}

        close = float(df_m15['close'].iloc[-1])
        low = float(df_m15['low'].iloc[-1])
        high = float(df_m15['high'].iloc[-1])
        
        atr_m15 = ATRCache.get_atr(df_m15, 14).iloc[-1]
        if pd.isna(atr_m15) or atr_m15 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        # Iterate from most recent OB backwards
        for ob in reversed(obs):
            # Bullish OB Rejection
            if ob['type'] == 'BULLISH' and h1_trend_bull:
                if low <= ob['high'] and close > ob['high']:
                    entry_price = close
                    sl_info = self.calculate_session_sl(entry_price, float(ob['low']), df_m15, is_buy=True)
                    if not sl_info['valid']: 
                        continue
                    
                    # Calculate TP using Fibonacci with Fallback
                    fib_tp = self.fib.calculate_tp_from_fib(entry_price, sl_info['sl_price'], df_m15, is_buy=True, min_rr=self.min_risk_reward)
                    if fib_tp['valid']:
                        tp_price = fib_tp['tp_price']
                        tp_reason = fib_tp['reason']
                    else:
                        risk = abs(entry_price - sl_info['sl_price'])
                        tp_price = entry_price + (2.5 * risk)
                        tp_reason = 'Fixed 2.5R (Fib Fallback)'
                        
                    signal = self.build_signal(
                        "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.75,
                        extra_meta={'ob_level': ob['low'], 'tp_reason': tp_reason}
                    )
                    self.log_signal_summary(signal)
                    return signal

            # Bearish OB Rejection
            elif ob['type'] == 'BEARISH' and h1_trend_bear:
                if high >= ob['low'] and close < ob['low']:
                    entry_price = close
                    sl_info = self.calculate_session_sl(entry_price, float(ob['high']), df_m15, is_buy=False)
                    if not sl_info['valid']: 
                        continue
                    
                    fib_tp = self.fib.calculate_tp_from_fib(entry_price, sl_info['sl_price'], df_m15, is_buy=False, min_rr=self.min_risk_reward)
                    if fib_tp['valid']:
                        tp_price = fib_tp['tp_price']
                        tp_reason = fib_tp['reason']
                    else:
                        risk = abs(entry_price - sl_info['sl_price'])
                        tp_price = entry_price - (2.5 * risk)
                        tp_reason = 'Fixed 2.5R (Fib Fallback)'
                        
                    signal = self.build_signal(
                        "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.75,
                        extra_meta={'ob_level': ob['high'], 'tp_reason': tp_reason}
                    )
                    self.log_signal_summary(signal)
                    return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_setup"}}