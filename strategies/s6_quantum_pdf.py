"""
Strategy 6: Quantum Probability Density Function (Mean Reversion).
Enters when price reaches extreme low/high probability zones expecting a revert to the mean.
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.quant_math_engine import QuantMathEngine
from core.atr_cache import ATRCache


class Strategy6_QuantumPDF(BaseStrategy):
    def __init__(self):
        super().__init__(name="S6_QuantumPDF", strategy_category="MEAN_REVERSION", min_risk_reward=1.5)
        self.quant = QuantMathEngine()

    def evaluate(self, df_m5: pd.DataFrame, df_m15: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m5, 100):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        # [FIX] Handle the new dict return type from calculate_quantum_pdf
        pdf_data = self.quant.calculate_quantum_pdf(df_m5['close'], bins=50, lookback=100)
        if pdf_data is None or len(pdf_data['pdf']) == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "pdf_calc_failed"}}

        high_prob_zones = self.quant.find_pdf_peaks(pdf_data, threshold=0.70)
        if not high_prob_zones:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_zones"}}

        close = float(df_m5['close'].iloc[-1])
        atr_m5 = ATRCache.get_atr(df_m5, 14).iloc[-1]
        if pd.isna(atr_m5) or atr_m5 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        # Find the nearest high probability zone (Mean)
        nearest_zone = min(high_prob_zones, key=lambda z: abs(close - z['center']))
        mean_price = float(nearest_zone['center'])
        
        # Calculate current price percentile relative to recent range
        recent_high = float(df_m5['high'].iloc[-50:].max())
        recent_low = float(df_m5['low'].iloc[-50:].min())
        price_range = recent_high - recent_low
        
        if price_range == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "zero_range"}}
            
        percentile = (close - recent_low) / price_range

        # Mean Reversion Logic
        # BUY if price is at extreme low (below 15th percentile) and below the mean zone
        if percentile < 0.15 and close < mean_price:
            entry_price = close
            sl_info = self.calculate_session_sl(entry_price, recent_low, df_m5, is_buy=True, atr_multiplier=1.5)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            # TP is the mean price (nearest high prob zone)
            tp_price = mean_price
            
            # Ensure minimum R:R
            risk = abs(entry_price - sl_info['sl_price'])
            reward = abs(tp_price - entry_price)
            if reward < risk * self.min_risk_reward:
                tp_price = entry_price + (risk * self.min_risk_reward)
                
            signal = self.build_signal(
                "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, "M5", 0.75,
                extra_meta={'mean_target': mean_price, 'percentile': percentile}
            )
            self.log_signal_summary(signal)
            return signal

        # SELL if price is at extreme high (above 85th percentile) and above the mean zone
        elif percentile > 0.85 and close > mean_price:
            entry_price = close
            sl_info = self.calculate_session_sl(entry_price, recent_high, df_m5, is_buy=False, atr_multiplier=1.5)
            if not sl_info['valid']: 
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
            
            tp_price = mean_price
            
            risk = abs(entry_price - sl_info['sl_price'])
            reward = abs(entry_price - tp_price)
            if reward < risk * self.min_risk_reward:
                tp_price = entry_price - (risk * self.min_risk_reward)
                
            signal = self.build_signal(
                "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, "M5", 0.75,
                extra_meta={'mean_target': mean_price, 'percentile': percentile}
            )
            self.log_signal_summary(signal)
            return signal

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "no_extreme"}}