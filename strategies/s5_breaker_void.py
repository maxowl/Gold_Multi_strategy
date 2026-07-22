"""
Strategy 5: Breaker Block + Liquidity Void + Volume Profile (SMC/VP).
Enters on high-probability confluence zones with VP-based SL/TP.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.breaker_vp_engine import BreakerVPEngine
from core.atr_cache import ATRCache


class Strategy5_Breaker_Void(BaseStrategy):
    def __init__(self):
        super().__init__(name="S5_Breaker_Void", strategy_category="SMC", min_risk_reward=2.5)
        self.bvp = BreakerVPEngine()

    def evaluate(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m15, 100):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        breakers = self.bvp.detect_breaker_blocks(df_m15, lookback=100)
        fvgs = self.bvp.detect_fvg_boxes(df_m15, atr_multiplier=0.5)
        
        if not breakers or not fvgs:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_structures"}}

        atr_m15 = ATRCache.get_atr(df_m15, 14).iloc[-1]
        if pd.isna(atr_m15) or atr_m15 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        close = float(df_m15['close'].iloc[-1])
        
        # Calculate Volume Profile for POC confluence
        ticks_df = self.bvp.create_synthetic_ticks(df_m15.tail(50))
        vp_levels = self.bvp.calculate_volume_profile_levels(ticks_df, bins=100)
        poc = vp_levels.get('poc', 0.0)

        for b in reversed(breakers):
            for f in reversed(fvgs):
                if b['type'] != f['type']: 
                    continue
                
                # Calculate overlap between Breaker and FVG
                overlap_lower = max(b['lower'], f['lower'])
                overlap_upper = min(b['upper'], f['upper'])
                
                if overlap_lower < overlap_upper:
                    # Bullish Confluence Tap
                    if b['type'] == 'BULLISH' and close <= overlap_upper and close >= overlap_lower:
                        entry_price = close
                        sl_info = self.calculate_session_sl(entry_price, overlap_lower, df_m15, is_buy=True)
                        if not sl_info['valid']: 
                            continue
                        
                        # Use VP-based TP if valid, else fallback
                        if poc > 0:
                            vp_sl_tp = self.bvp.calculate_vp_based_sl_tp(entry_price, True, vp_levels, atr_m15, strategy_type='TREND')
                            tp_price = vp_sl_tp['tp']
                        else:
                            tp_price = 0.0
                            
                        # Fallback if VP TP is too close or invalid
                        risk = abs(entry_price - sl_info['sl_price'])
                        if tp_price == 0.0 or abs(tp_price - entry_price) < risk * self.min_risk_reward:
                            tp_price = entry_price + (3.0 * risk)
                            
                        signal = self.build_signal(
                            "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.90, 
                            extra_meta={'confluence': 'Breaker+FVG', 'poc': poc}
                        )
                        self.log_signal_summary(signal)
                        return signal

                    # Bearish Confluence Tap
                    elif b['type'] == 'BEARISH' and close >= overlap_lower and close <= overlap_upper:
                        entry_price = close
                        sl_info = self.calculate_session_sl(entry_price, overlap_upper, df_m15, is_buy=False)
                        if not sl_info['valid']: 
                            continue
                        
                        if poc > 0:
                            vp_sl_tp = self.bvp.calculate_vp_based_sl_tp(entry_price, False, vp_levels, atr_m15, strategy_type='TREND')
                            tp_price = vp_sl_tp['tp']
                        else:
                            tp_price = 0.0
                            
                        risk = abs(entry_price - sl_info['sl_price'])
                        if tp_price == 0.0 or abs(entry_price - tp_price) < risk * self.min_risk_reward:
                            tp_price = entry_price - (3.0 * risk)
                            
                        signal = self.build_signal(
                            "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.90, 
                            extra_meta={'confluence': 'Breaker+FVG', 'poc': poc}
                        )
                        self.log_signal_summary(signal)
                        return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_tap"}}