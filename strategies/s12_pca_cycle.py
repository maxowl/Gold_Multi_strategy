"""
Strategy 12: PCA Cycle Turning Point (Trend).
Uses PCA to extract the dominant cyclical component and enters on cycle reversals.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.pca_engine import PCAEngine
from core.atr_cache import ATRCache


class Strategy12_PCA_Cycle(BaseStrategy):
    def __init__(self):
        super().__init__(name="S12_PCA_Cycle", strategy_category="TREND", min_risk_reward=2.0)
        self.pca = PCAEngine()

    def evaluate(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m15, 100):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        # Extract principal components
        pca_result = self.pca.extract_principal_components(df_m15, n_components=3, lookback=100)
        
        # [FIX] Guard against None or missing keys
        if pca_result is None or 'components' not in pca_result or len(pca_result['components']) == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "pca_calc_failed"}}

        # Get the first principal component (dominant cycle)
        cycle_component = pca_result['components'][0]
        if len(cycle_component) < 3:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_cycle_data"}}

        c_curr = float(cycle_component.iloc[-1])
        c_prev = float(cycle_component.iloc[-2])
        c_prev2 = float(cycle_component.iloc[-3])
        
        close = float(df_m15['close'].iloc[-1])
        atr_m15 = ATRCache.get_atr(df_m15, 14).iloc[-1]
        if pd.isna(atr_m15) or atr_m15 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        # Bullish Cycle Turn: Component was falling, hits bottom, and starts rising
        if c_prev2 > c_prev and c_curr > c_prev:
            entry_price = close
            recent_low = float(df_m15['low'].iloc[-10:].min())
            sl_info = self.calculate_session_sl(entry_price, recent_low, df_m15, is_buy=True)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = entry_price + (3.0 * risk)
            
            signal = self.build_signal(
                "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.75,
                extra_meta={'pca_explained_var': pca_result['explained_variance'][0]}
            )
            self.log_signal_summary(signal)
            return signal

        # Bearish Cycle Turn: Component was rising, hits top, and starts falling
        elif c_prev2 < c_prev and c_curr < c_prev:
            entry_price = close
            recent_high = float(df_m15['high'].iloc[-10:].max())
            sl_info = self.calculate_session_sl(entry_price, recent_high, df_m15, is_buy=False)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = entry_price - (3.0 * risk)
            
            signal = self.build_signal(
                "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.75,
                extra_meta={'pca_explained_var': pca_result['explained_variance'][0]}
            )
            self.log_signal_summary(signal)
            return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_turn"}}