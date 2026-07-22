"""
Strategy 7: Macro Fair Value Gap (SMC).
Detects large FVGs on H1 and enters on M15 pullbacks into the gap.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.smc_engine import SMCStructuralEngine
from core.atr_cache import ATRCache
from core.fibonacci_engine import FibonacciEngine


class Strategy7_MacroFVG(BaseStrategy):
    def __init__(self):
        super().__init__(name="S7_MacroFVG", strategy_category="SMC", min_risk_reward=2.5)
        self.smc = SMCStructuralEngine()
        self.fib = FibonacciEngine()

    def evaluate(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m15, 50) or not self._validate_data(df_h1, 20):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        # Detect FVGs on H1 (Macro level)
        h1_fvg_df = self.smc.detect_fvg(df_h1)
        if h1_fvg_df.empty:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "fvg_detection_failed"}}

        # Filter for recent bullish and bearish FVGs
        bullish_fvgs = h1_fvg_df[h1_fvg_df['bullish_fvg'] == True]
        bearish_fvgs = h1_fvg_df[h1_fvg_df['bearish_fvg'] == True]

        close = float(df_m15['close'].iloc[-1])
        low = float(df_m15['low'].iloc[-1])
        high = float(df_m15['high'].iloc[-1])
        
        atr_m15 = ATRCache.get_atr(df_m15, 14).iloc[-1]
        if pd.isna(atr_m15) or atr_m15 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        # Check Bullish FVG Tap on M15
        if not bullish_fvgs.empty:
            # Get the most recent H1 bullish FVG
            last_bull_idx = bullish_fvgs.index[-1]
            # Calculate actual gap boundaries from H1 OHLC
            fvg_top = float(df_h1['low'].iloc[last_bull_idx])
            fvg_bottom = float(df_h1['high'].iloc[last_bull_idx - 2])
            
            if fvg_top > fvg_bottom and low <= fvg_top and close > fvg_bottom:
                entry_price = close
                sl_info = self.calculate_session_sl(entry_price, fvg_bottom, df_m15, is_buy=True, atr_multiplier=1.0)
                if sl_info['valid']:
                    fib_tp = self.fib.calculate_tp_from_fib(entry_price, sl_info['sl_price'], df_h1, is_buy=True, min_rr=self.min_risk_reward)
                    tp_price = fib_tp['tp_price'] if fib_tp['valid'] else entry_price + (3.0 * abs(entry_price - sl_info['sl_price']))
                    
                    signal = self.build_signal(
                        "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.85,
                        extra_meta={'macro_fvg': 'BULLISH', 'fvg_bottom': fvg_bottom}
                    )
                    self.log_signal_summary(signal)
                    return signal

        # Check Bearish FVG Tap on M15
        if not bearish_fvgs.empty:
            last_bear_idx = bearish_fvgs.index[-1]
            fvg_bottom = float(df_h1['high'].iloc[last_bear_idx])
            fvg_top = float(df_h1['low'].iloc[last_bear_idx - 2])
            
            if fvg_top > fvg_bottom and high >= fvg_bottom and close < fvg_top:
                entry_price = close
                sl_info = self.calculate_session_sl(entry_price, fvg_top, df_m15, is_buy=False, atr_multiplier=1.0)
                if sl_info['valid']:
                    fib_tp = self.fib.calculate_tp_from_fib(entry_price, sl_info['sl_price'], df_h1, is_buy=False, min_rr=self.min_risk_reward)
                    tp_price = fib_tp['tp_price'] if fib_tp['valid'] else entry_price - (3.0 * abs(entry_price - sl_info['sl_price']))
                    
                    signal = self.build_signal(
                        "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.85,
                        extra_meta={'macro_fvg': 'BEARISH', 'fvg_top': fvg_top}
                    )
                    self.log_signal_summary(signal)
                    return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_tap"}}