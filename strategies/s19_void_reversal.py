"""
Strategy 19: Liquidity Void Reversal (Scalp).
Enters on M5 when price taps into a liquidity void and shows immediate rejection.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.void_structural_engine import VoidStructuralEngine
from core.atr_cache import ATRCache


class Strategy19_VoidReversal(BaseStrategy):
    def __init__(self):
        super().__init__(name="S19_VoidReversal", strategy_category="SCALP", min_risk_reward=1.5)
        self.void_engine = VoidStructuralEngine()

    def evaluate(self, df_m5: pd.DataFrame, df_m15: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m5, 50):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        voids = self.void_engine.detect_liquidity_voids(df_m5, atr_multiplier=3.0)
        if not voids:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_voids"}}

        close = float(df_m5['close'].iloc[-1])
        low = float(df_m5['low'].iloc[-1])
        high = float(df_m5['high'].iloc[-1])
        
        atr_m5 = ATRCache.get_atr(df_m5, 14).iloc[-1]
        if pd.isna(atr_m5) or atr_m5 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        # Iterate through recent voids to find a tap and rejection
        for void in reversed(voids):
            if void['bar_index'] < len(df_m5) - 10: 
                continue # Only consider recent voids
                
            # Bullish Void Reversal: Price dipped into bullish void and closed back above lower boundary
            if void['direction'] == 'BULLISH' and low <= void['lower'] and close > void['lower']:
                entry_price = close
                sl_info = self.calculate_session_sl(entry_price, float(low), df_m5, is_buy=True, atr_multiplier=1.0)
                if not sl_info['valid']: 
                    continue
                
                risk = abs(entry_price - sl_info['sl_price'])
                tp_price = void['upper'] # Target the top of the void
                
                # Fallback if void is too small
                if abs(tp_price - entry_price) < risk * self.min_risk_reward:
                    tp_price = entry_price + (2.0 * risk)
                    
                signal = self.build_signal(
                    "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, "M5", 0.70,
                    extra_meta={
                        'void_type': 'BULLISH', 
                        'friction_sensitive': True, 
                        'trailing_method': 'fixed_dollar'
                    }
                )
                self.log_signal_summary(signal)
                return signal

            # Bearish Void Reversal: Price spiked into bearish void and closed back below upper boundary
            elif void['direction'] == 'BEARISH' and high >= void['upper'] and close < void['upper']:
                entry_price = close
                sl_info = self.calculate_session_sl(entry_price, float(high), df_m5, is_buy=False, atr_multiplier=1.0)
                if not sl_info['valid']: 
                    continue
                
                risk = abs(entry_price - sl_info['sl_price'])
                tp_price = void['lower']
                
                if abs(entry_price - tp_price) < risk * self.min_risk_reward:
                    tp_price = entry_price - (2.0 * risk)
                    
                signal = self.build_signal(
                    "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, "M5", 0.70,
                    extra_meta={
                        'void_type': 'BEARISH', 
                        'friction_sensitive': True, 
                        'trailing_method': 'fixed_dollar'
                    }
                )
                self.log_signal_summary(signal)
                return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_rejection"}}