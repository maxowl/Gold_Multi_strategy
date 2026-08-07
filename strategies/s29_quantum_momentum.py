"""
Strategy 29: Quantum Momentum (Mean Reversion).

Uses Quantum PDF to identify probability density peaks and trade reversals.
Uses QuantMathEngine for PDF and Fractal Dimension analysis.

Category: MEAN_REVERSION
Optimal Regimes: CLASSIC_RANGE, TIGHT_RANGE, EXHAUSTED_BULL, EXHAUSTED_BEAR
Timeframe: M15 (Primary), M5 (Confirmation)
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.quant_math_engine import QuantMathEngine
from core.atr_cache import ATRCache


class S29_QuantumMomentum(BaseStrategy):
    """
    Quantum Momentum Strategy.
    
    Logic:
    - Calculate Quantum PDF to find high-probability price zones
    - Identify PDF peaks (price levels with high probability)
    - Enter when price approaches peak and shows reversal signs
    - Use Fractal Dimension to filter choppy markets
    
    Advantages:
    - Statistical edge (probability-based)
    - Works in range-bound markets
    - Identifies institutional accumulation zones
    """
    
    def __init__(self):
        super().__init__(
            name='S29_QuantumMomentum',
            category='MEAN_REVERSION',
            description='Quantum PDF Probability-Based Trading'
        )
        self.quant_engine = QuantMathEngine()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def evaluate(self, df_primary: pd.DataFrame, df_htf: pd.DataFrame = None) -> dict:
        """
        Evaluate quantum momentum signal.
        
        Args:
            df_primary: M15 data
            df_htf: M5 data (optional confirmation)
            
        Returns:
            Signal dict with BUY/SELL or NEUTRAL
        """
        if df_primary is None or df_primary.empty or len(df_primary) < 100:
            return {'signal': 'NEUTRAL', 'meta': {'reason': 'Insufficient data'}}
        
        # Calculate Fractal Dimension first (filter choppy markets)
        fdi = self.quant_engine.calculate_fractal_dimension(df_primary['close'], max_lag=20)
        
        # Skip if market is too choppy (FDI > 1.6) or too trending (FDI < 1.4)
        # We want mean-reverting markets (FDI 1.4-1.6)
        if fdi < 1.4 or fdi > 1.6:
            return {'signal': 'NEUTRAL', 'meta': {'reason': f'FDI {fdi:.2f} not in optimal range (1.4-1.6)'}}
        
        # Calculate Quantum PDF
        pdf_data = self.quant_engine.calculate_quantum_pdf(df_primary['close'], bins=50, lookback=100)
        
        if pdf_data is None:
            return {'signal': 'NEUTRAL', 'meta': {'reason': 'PDF calculation failed'}}
        
        # Find PDF peaks
        peaks = self.quant_engine.find_pdf_peaks(pdf_data, threshold=0.7)
        
        if not peaks:
            return {'signal': 'NEUTRAL', 'meta': {'reason': 'No significant PDF peaks found'}}
        
        current_price = df_primary['close'].iloc[-1]
        
        # Find nearest peak
        nearest_peak = None
        min_distance = float('inf')
        
        for peak in peaks:
            peak_price = peak['center']
            distance = abs(current_price - peak_price)
            if distance < min_distance:
                min_distance = distance
                nearest_peak = peak
        
        if nearest_peak is None:
            return {'signal': 'NEUTRAL', 'meta': {'reason': 'No nearest peak identified'}}
        
        peak_price = nearest_peak['center']
        peak_probability = nearest_peak['probability']
        
        # Get ATR for context
        atr_series = ATRCache.get_atr(df_primary, 14)
        if atr_series.empty or atr_series.isna().all():
            return {'signal': 'NEUTRAL', 'meta': {'reason': 'ATR calculation failed'}}
        
        current_atr = atr_series.iloc[-1]
        
        # Check if price is near peak (within 0.5 ATR)
        distance_atr = min_distance / current_atr
        
        if distance_atr > 1.0:  # Price too far from peak
            return {'signal': 'NEUTRAL', 'meta': {'reason': f'Price too far from peak ({distance_atr:.2f} ATR)'}}
        
        # Determine signal based on price position relative to peak
        signal_type = 'NEUTRAL'
        entry_price = current_price
        sl_price = 0.0
        tp_price = 0.0
        confidence = 0.0
        reason = ''
        
        # Check recent price action (last 3 bars)
        recent_closes = df_primary['close'].iloc[-3:].values
        price_trend = recent_closes[-1] - recent_closes[0]
        
        if current_price < peak_price:
            # Price below peak - look for BUY if showing reversal
            if price_trend > 0:  # Price starting to rise
                signal_type = 'BUY_MARKET'
                sl_price = current_price - (current_atr * 1.5)
                tp_price = peak_price + (current_atr * 0.5)  # Target: slightly above peak
                confidence = 0.60 + (peak_probability * 0.3)
                reason = f'Price approaching PDF peak from below (P={peak_probability:.3f})'
        
        elif current_price > peak_price:
            # Price above peak - look for SELL if showing reversal
            if price_trend < 0:  # Price starting to fall
                signal_type = 'SELL_MARKET'
                sl_price = current_price + (current_atr * 1.5)
                tp_price = peak_price - (current_atr * 0.5)  # Target: slightly below peak
                confidence = 0.60 + (peak_probability * 0.3)
                reason = f'Price approaching PDF peak from above (P={peak_probability:.3f})'
        
        if signal_type == 'NEUTRAL':
            return {'signal': 'NEUTRAL', 'meta': {'reason': 'No reversal signal near peak'}}
        
        # Validate R:R
        risk = abs(entry_price - sl_price)
        reward = abs(tp_price - entry_price)
        rr = reward / risk if risk > 0 else 0
        
        if rr < 1.5:  # Minimum R:R
            return {'signal': 'NEUTRAL', 'meta': {'reason': f'R:R too low ({rr:.2f})'}}
        
        # Build signal
        meta = {
            'strategy': self.name,
            'strategy_category': self.category,
            'entry_price': round(entry_price, 2),
            'sl_price': round(sl_price, 2),
            'tp_price': round(tp_price, 2),
            'risk_reward': round(rr, 2),
            'confidence': confidence,
            'timeframe': 'M15',
            'expiration_bars': 16,  # ~4 hours
            'requires_dynamic_exit': True,
            'dynamic_exit_threshold': 'peak_cross',
            'position_multiplier': 1.0,
            'trailing_enabled': True,
            'partial_close_enabled': True,
            'trailing_method': 'fixed_dollar',
            'fractal_dimension': round(fdi, 3),
            'peak_price': round(peak_price, 2),
            'peak_probability': round(peak_probability, 4),
            'distance_atr': round(distance_atr, 2),
            'reason': reason
        }
        
        signal = {
            'signal': signal_type,
            'meta': meta
        }
        
        self.log_signal_summary(signal)
        return signal