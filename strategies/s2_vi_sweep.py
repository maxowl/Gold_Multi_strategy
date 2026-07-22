"""
Strategy 2: Volume Imbalance Sweep (Scalp).
Enters on M1 Volume Imbalance zone sweeps with tight risk management.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.orderflow_engine import OrderFlowEngine
from core.atr_cache import ATRCache


class Strategy2_VI_Sweep(BaseStrategy):
    def __init__(self):
        super().__init__(name="S2_VI_Sweep", strategy_category="SCALP", min_risk_reward=1.5)
        self.of_engine = OrderFlowEngine()

    def evaluate(self, df_m1: pd.DataFrame, df_m15: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m1, 50):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        zones = self.of_engine.detect_volume_imbalance_zones(df_m1, min_imbalance_ratio=2.0, lookback=50)
        if not zones:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_zones"}}

        close = float(df_m1['close'].iloc[-1])
        low = float(df_m1['low'].iloc[-1])
        high = float(df_m1['high'].iloc[-1])
        
        atr_m1 = ATRCache.get_atr(df_m1, 14).iloc[-1]
        if pd.isna(atr_m1) or atr_m1 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        for zone in reversed(zones):
            # Bullish Sweep
            if zone['type'] == 'BULLISH':
                if low <= zone['upper'] and close > zone['upper']:
                    entry_price = close
                    sl_info = self.calculate_session_sl(entry_price, float(zone['lower']), df_m1, is_buy=True, atr_multiplier=1.0)
                    if not sl_info['valid']: 
                        continue
                    
                    risk = abs(entry_price - sl_info['sl_price'])
                    tp_price = entry_price + (2.0 * risk)
                    
                    signal = self.build_signal(
                        "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, "M1", 0.70, 
                        extra_meta={
                            'vi_zone': zone, 
                            'friction_sensitive': True, 
                            'trailing_method': 'fixed_dollar'
                        }
                    )
                    self.log_signal_summary(signal)
                    return signal

            # Bearish Sweep
            elif zone['type'] == 'BEARISH':
                if high >= zone['lower'] and close < zone['lower']:
                    entry_price = close
                    sl_info = self.calculate_session_sl(entry_price, float(zone['upper']), df_m1, is_buy=False, atr_multiplier=1.0)
                    if not sl_info['valid']: 
                        continue
                    
                    risk = abs(entry_price - sl_info['sl_price'])
                    tp_price = entry_price - (2.0 * risk)
                    
                    signal = self.build_signal(
                        "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, "M1", 0.70, 
                        extra_meta={
                            'vi_zone': zone, 
                            'friction_sensitive': True, 
                            'trailing_method': 'fixed_dollar'
                        }
                    )
                    self.log_signal_summary(signal)
                    return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_sweep"}}