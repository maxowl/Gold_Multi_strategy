"""
Strategy 15: HFT Statistical Arbitrage (Mean Reversion).
Enters on extreme Z-Score deviations and exits when price reverts to the mean.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.stat_arb_engine import StatArbEngine
from core.atr_cache import ATRCache


class Strategy15_HFT_StatArb(BaseStrategy):
    def __init__(self):
        super().__init__(name="S15_HFT_StatArb", strategy_category="MEAN_REVERSION", min_risk_reward=1.5)
        self.stat_arb = StatArbEngine()

    def evaluate(self, df_m5: pd.DataFrame, df_m15: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m5, 100):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        z_score_series = self.stat_arb.calculate_z_score(df_m5['close'], lookback=100)
        
        # [FIX] Guard against None or empty series
        if z_score_series is None or len(z_score_series) == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "z_score_calc_failed"}}

        z_curr = float(z_score_series.iloc[-1])
        
        if np.isnan(z_curr):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "z_score_nan"}}

        close = float(df_m5['close'].iloc[-1])
        atr_m5 = ATRCache.get_atr(df_m5, 14).iloc[-1]
        if pd.isna(atr_m5) or atr_m5 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        # Mean Reversion Logic
        # BUY if Z-Score is extremely negative (Oversold)
        if z_curr < -2.0:
            entry_price = close
            sl_info = self.calculate_session_sl(entry_price, float(df_m5['low'].iloc[-20:].min()), df_m5, is_buy=True, atr_multiplier=2.0)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            # TP is the mean (Z=0), estimated as current price + (Z * StdDev)
            # Fallback to 2.0R if dynamic exit fails
            tp_price = entry_price + (2.0 * risk)
            
            signal = self.build_signal(
                "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, "M5", 0.80,
                requires_dynamic_exit=True, dynamic_exit_threshold="z_score_cross_zero",
                extra_meta={'z_score': z_curr, 'friction_sensitive': True}
            )
            self.log_signal_summary(signal)
            return signal

        # SELL if Z-Score is extremely positive (Overbought)
        elif z_curr > 2.0:
            entry_price = close
            sl_info = self.calculate_session_sl(entry_price, float(df_m5['high'].iloc[-20:].max()), df_m5, is_buy=False, atr_multiplier=2.0)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            risk = abs(entry_price - sl_info['sl_price'])
            tp_price = entry_price - (2.0 * risk)
            
            signal = self.build_signal(
                "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, "M5", 0.80,
                requires_dynamic_exit=True, dynamic_exit_threshold="z_score_cross_zero",
                extra_meta={'z_score': z_curr, 'friction_sensitive': True}
            )
            self.log_signal_summary(signal)
            return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "z_score_normal"}}