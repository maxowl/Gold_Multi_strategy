"""
Strategy 21: Breaker Block + FVG + Volume POC Triple Confluence (SMC).
Enters only when three structural elements align at the same price zone.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.breaker_vp_engine import BreakerVPEngine
from core.atr_cache import ATRCache


class Strategy21_BreakerFVGPOC(BaseStrategy):
    def __init__(self):
        super().__init__(name="S21_BreakerFVGPOC", strategy_category="SMC", min_risk_reward=2.5)
        self.bvp = BreakerVPEngine()

    def evaluate(self, df_m5: pd.DataFrame, df_h1: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m5, 100):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        breakers = self.bvp.detect_breaker_blocks(df_m5, lookback=100, order=3)
        fvgs = self.bvp.detect_fvg_boxes(df_m5, atr_multiplier=0.3)
        
        if not breakers or not fvgs:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_structures"}}

        ticks_df = self.bvp.create_synthetic_ticks(df_m5.tail(50))
        if ticks_df.empty:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "tick_gen_failed"}}
            
        poc = self.bvp.calculate_session_vp_poc(ticks_df, bins=100)
        if poc == 0.0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "poc_zero"}}

        atr_m5 = ATRCache.get_atr(df_m5, 14).iloc[-1]
        if pd.isna(atr_m5) or atr_m5 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        close = float(df_m5['close'].iloc[-1])
        
        tolerance = atr_m5 * 0.5
        confluence = self.bvp.validate_triple_confluence(breakers, fvgs, poc, tolerance)
        
        if confluence is None:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_triple_confluence"}}

        if confluence['type'] == 'BULLISH' and close <= confluence['entry_level'] + tolerance:
            entry_price = close
            structural_low = min(confluence['breaker_lower'], confluence['fvg_lower'])
            sl_info = self.calculate_session_sl(entry_price, structural_low, df_m5, is_buy=True)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = max(confluence['breaker_upper'], confluence['fvg_upper']) + (0.5 * atr_m5)
            
            if abs(tp_price - entry_price) < risk * self.min_risk_reward:
                tp_price = entry_price + (3.4 * risk)
            
            signal = self.build_signal(
                "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, "M5", 0.90,
                extra_meta={'confluence_type': 'TRIPLE_BULLISH', 'poc': poc}
            )
            self.log_signal_summary(signal)
            return signal

        elif confluence['type'] == 'BEARISH' and close >= confluence['entry_level'] - tolerance:
            entry_price = close
            structural_high = max(confluence['breaker_upper'], confluence['fvg_upper'])
            sl_info = self.calculate_session_sl(entry_price, structural_high, df_m5, is_buy=False)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = min(confluence['breaker_lower'], confluence['fvg_lower']) - (0.5 * atr_m5)
            
            if abs(entry_price - tp_price) < risk * self.min_risk_reward:
                tp_price = entry_price - (3.4 * risk)
            
            signal = self.build_signal(
                "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, "M5", 0.90,
                extra_meta={'confluence_type': 'TRIPLE_BEARISH', 'poc': poc}
            )
            self.log_signal_summary(signal)
            return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_tap"}}