"""
Strategy 28: Multi-Timeframe Confluence (Trend).

Enters when multiple timeframes align with momentum confirmation.
Uses MTFFeatureEngineer for feature extraction.

Category: TREND
Optimal Regimes: HEALTHY_UPTREND, HEALTHY_DOWNTREND, QUIET_RALLY, SLOW_BLEED
Timeframe: M15 (Primary), H1 (Confirmation)
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.mtf_feature_engineer import MTFFeatureEngineer
from core.atr_cache import ATRCache


class S28_MTF_Confluence(BaseStrategy):
    """
    Multi-Timeframe Confluence Strategy.
    
    Logic:
    - Extract features from M5, M15, H1
    - Enter when trend aligns across timeframes
    - Confirm with momentum (MACD, RSI)
    - Use ATR for risk management
    
    Advantages:
    - High win rate (trend following)
    - Multiple confirmation layers
    - Reduced false signals
    """
    
    def __init__(self):
        super().__init__(
            name='S28_MTF_Confluence',
            category='TREND',
            description='Multi-Timeframe Trend Confluence'
        )
        self.mtf_engine = MTFFeatureEngineer()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def evaluate(self, df_primary: pd.DataFrame, df_htf: pd.DataFrame = None) -> dict:
        """
        Evaluate MTF confluence signal.
        
        Args:
            df_primary: M15 data
            df_htf: H1 data
            
        Returns:
            Signal dict with BUY/SELL or NEUTRAL
        """
        if df_primary is None or df_primary.empty or len(df_primary) < 50:
            return {'signal': 'NEUTRAL', 'meta': {'reason': 'Insufficient data'}}
        
        # Extract features from both timeframes
        df_m15 = df_primary
        df_h1 = df_htf if df_htf is not None else None
        
        if df_h1 is None or df_h1.empty or len(df_h1) < 50:
            return {'signal': 'NEUTRAL', 'meta': {'reason': 'H1 data insufficient'}}
        
        # Get features
        features = self.mtf_engine.extract_all_features(df_m15, df_h1)
        
        if features.empty:
            return {'signal': 'NEUTRAL', 'meta': {'reason': 'Feature extraction failed'}}
        
        # Get latest features
        latest = features.iloc[-1]
        
        # Check trend alignment
        m15_trend = latest.get('m15_trend', 0)
        h1_trend = latest.get('h1_trend', 0)
        
        # Check momentum alignment
        m15_macd_hist = latest.get('m15_macd_hist', 0)
        h1_macd_hist = latest.get('h1_macd_hist', 0)
        
        m15_rsi = latest.get('m15_rsi_14', 50)
        h1_rsi = latest.get('h1_rsi_14', 50)
        
        # Determine signal
        signal_type = 'NEUTRAL'
        confidence = 0.0
        reason = ''
        
        # BUY conditions
        if (m15_trend > 0 and h1_trend > 0 and
            m15_macd_hist > 0 and h1_macd_hist > 0 and
            m15_rsi > 50 and h1_rsi > 50 and
            m15_rsi < 70 and h1_rsi < 70):
            
            signal_type = 'BUY_MARKET'
            confidence = 0.80
            reason = 'MTF Bullish Confluence: All timeframes aligned upward'
            
            # Boost confidence for stronger signals
            if m15_macd_hist > 0.5 and h1_macd_hist > 0.5:
                confidence += 0.05
            if m15_trend > 0.5 and h1_trend > 0.5:
                confidence += 0.05
        
        # SELL conditions
        elif (m15_trend < 0 and h1_trend < 0 and
              m15_macd_hist < 0 and h1_macd_hist < 0 and
              m15_rsi < 50 and h1_rsi < 50 and
              m15_rsi > 30 and h1_rsi > 30):
            
            signal_type = 'SELL_MARKET'
            confidence = 0.80
            reason = 'MTF Bearish Confluence: All timeframes aligned downward'
            
            # Boost confidence for stronger signals
            if m15_macd_hist < -0.5 and h1_macd_hist < -0.5:
                confidence += 0.05
            if m15_trend < -0.5 and h1_trend < -0.5:
                confidence += 0.05
        
        if signal_type == 'NEUTRAL':
            return {'signal': 'NEUTRAL', 'meta': {'reason': 'No MTF confluence detected'}}
        
        # Calculate entry, SL, TP
        current_price = df_m15['close'].iloc[-1]
        entry_price = current_price
        
        # Get ATR for SL calculation
        atr_series = ATRCache.get_atr(df_m15, 14)
        if atr_series.empty or atr_series.isna().all():
            return {'signal': 'NEUTRAL', 'meta': {'reason': 'ATR calculation failed'}}
        
        current_atr = atr_series.iloc[-1]
        
        if 'BUY' in signal_type:
            sl_price = current_price - (current_atr * 2.0)
            tp_price = current_price + (current_atr * 4.0)  # 2R target
        else:
            sl_price = current_price + (current_atr * 2.0)
            tp_price = current_price - (current_atr * 4.0)  # 2R target
        
        # Validate R:R
        risk = abs(entry_price - sl_price)
        reward = abs(tp_price - entry_price)
        rr = reward / risk if risk > 0 else 0
        
        if rr < 1.8:  # Minimum R:R for trend strategy
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
            'expiration_bars': 20,  # ~5 hours
            'requires_dynamic_exit': False,
            'position_multiplier': 1.0,
            'trailing_enabled': True,
            'partial_close_enabled': True,
            'trailing_method': 'atr_based',
            'm15_trend': round(m15_trend, 3),
            'h1_trend': round(h1_trend, 3),
            'm15_macd_hist': round(m15_macd_hist, 4),
            'h1_macd_hist': round(h1_macd_hist, 4),
            'm15_rsi': round(m15_rsi, 2),
            'h1_rsi': round(h1_rsi, 2),
            'reason': reason
        }
        
        signal = {
            'signal': signal_type,
            'meta': meta
        }
        
        self.log_signal_summary(signal)
        return signal