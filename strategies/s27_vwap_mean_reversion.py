"""
Strategy 27: VWAP Mean Reversion (Scalp).

Enters on price reversion to VWAP with tight risk management.
Uses TimePriceEngine for VWAP calculation.

Category: SCALP
Optimal Regimes: CLASSIC_RANGE, TIGHT_RANGE, QUIET_RALLY, SLOW_BLEED
Timeframe: M5 (Primary), M15 (Confirmation)
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.time_price_engine import TimePriceEngine
from core.atr_cache import ATRCache


class S27_VWAP_MeanReversion(BaseStrategy):
    """
    VWAP Mean Reversion Strategy.
    
    Logic:
    - Calculate VWAP for current session
    - Enter when price reverts to VWAP from extreme
    - Use ATR for stop loss
    - Target 1.5-2.0R
    
    Advantages:
    - Institutional reference point
    - Works well in range-bound markets
    - Clear risk/reward
    """
    
    def __init__(self):
        super().__init__(
            name='S27_VWAP_MeanReversion',
            category='SCALP',
            description='VWAP Mean Reversion with Session Awareness'
        )
        self.time_price_engine = TimePriceEngine()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def evaluate(self, df_primary: pd.DataFrame, df_htf: pd.DataFrame = None) -> dict:
        """
        Evaluate VWAP mean reversion signal.
        
        Args:
            df_primary: M5 data
            df_htf: M15 data (optional confirmation)
            
        Returns:
            Signal dict with BUY/SELL or NEUTRAL
        """
        if df_primary is None or df_primary.empty or len(df_primary) < 50:
            return {'signal': 'NEUTRAL', 'meta': {'reason': 'Insufficient data'}}
        
        # Calculate VWAP
        vwap_series = self.time_price_engine.calculate_vwap(df_primary)
        
        if vwap_series is None or vwap_series.isna().all():
            return {'signal': 'NEUTRAL', 'meta': {'reason': 'VWAP calculation failed'}}
        
        current_price = df_primary['close'].iloc[-1]
        current_vwap = vwap_series.iloc[-1]
        
        # Calculate deviation from VWAP
        deviation = (current_price - current_vwap) / current_vwap
        
        # Get ATR for volatility context
        atr_series = ATRCache.get_atr(df_primary, 14)
        if atr_series.empty or atr_series.isna().all():
            return {'signal': 'NEUTRAL', 'meta': {'reason': 'ATR calculation failed'}}
        
        current_atr = atr_series.iloc[-1]
        
        # Check for mean reversion opportunities
        # BUY when price is below VWAP by > 1 ATR (oversold)
        # SELL when price is above VWAP by > 1 ATR (overbought)
        
        atr_deviation = abs(current_price - current_vwap) / current_atr
        
        signal_type = 'NEUTRAL'
        entry_price = current_price
        sl_price = 0.0
        tp_price = 0.0
        confidence = 0.0
        reason = ''
        
        if atr_deviation > 1.5:  # Significant deviation
            if current_price < current_vwap:
                # BUY signal - price below VWAP
                signal_type = 'BUY_MARKET'
                sl_price = current_price - (current_atr * 1.5)
                tp_price = current_vwap  # Target: reversion to VWAP
                confidence = min(0.85, 0.5 + (atr_deviation - 1.5) * 0.2)
                reason = f'Price {atr_deviation:.2f} ATR below VWAP - Mean Reversion Buy'
                
            else:
                # SELL signal - price above VWAP
                signal_type = 'SELL_MARKET'
                sl_price = current_price + (current_atr * 1.5)
                tp_price = current_vwap  # Target: reversion to VWAP
                confidence = min(0.85, 0.5 + (atr_deviation - 1.5) * 0.2)
                reason = f'Price {atr_deviation:.2f} ATR above VWAP - Mean Reversion Sell'
        
        if signal_type == 'NEUTRAL':
            return {'signal': 'NEUTRAL', 'meta': {'reason': 'No significant VWAP deviation'}}
        
        # Validate R:R
        risk = abs(entry_price - sl_price)
        reward = abs(tp_price - entry_price)
        rr = reward / risk if risk > 0 else 0
        
        if rr < 1.3:  # Minimum R:R for micro-account
            return {'signal': 'NEUTRAL', 'meta': {'reason': f'R:R too low ({rr:.2f})'}}
        
        # HTF confirmation (optional)
        htf_confirmed = False
        if df_htf is not None and not df_htf.empty and len(df_htf) >= 20:
            htf_vwap = self.time_price_engine.calculate_vwap(df_htf)
            if htf_vwap is not None and not htf_vwap.isna().all():
                htf_price = df_htf['close'].iloc[-1]
                htf_current_vwap = htf_vwap.iloc[-1]
                
                # Check if HTF supports the signal
                if 'BUY' in signal_type and htf_price < htf_current_vwap:
                    htf_confirmed = True
                    confidence = min(0.95, confidence * 1.1)
                elif 'SELL' in signal_type and htf_price > htf_current_vwap:
                    htf_confirmed = True
                    confidence = min(0.95, confidence * 1.1)
        
        # Build signal
        meta = {
            'strategy': self.name,
            'strategy_category': self.category,
            'entry_price': round(entry_price, 2),
            'sl_price': round(sl_price, 2),
            'tp_price': round(tp_price, 2),
            'risk_reward': round(rr, 2),
            'confidence': confidence,
            'timeframe': 'M5',
            'expiration_bars': 12,  # 1 hour
            'requires_dynamic_exit': True,
            'dynamic_exit_threshold': 'vwap_cross',
            'position_multiplier': 1.0,
            'trailing_enabled': True,
            'partial_close_enabled': True,
            'trailing_method': 'fixed_dollar',
            'vwap_value': round(current_vwap, 2),
            'atr_deviation': round(atr_deviation, 2),
            'htf_confirmed': htf_confirmed,
            'reason': reason
        }
        
        signal = {
            'signal': signal_type,
            'meta': meta
        }
        
        self.log_signal_summary(signal)
        return signal