"""
Rule-Based Regime Classifier.
Provides fallback regime classification using ADX-like trend strength.
Used when HMM/LightGBM models are not available.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict
from core.atr_cache import ATRCache


class RegimeClassifier:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def classify(self, df_m5: pd.DataFrame, df_m15: pd.DataFrame = None,
                 df_h1: pd.DataFrame = None) -> Dict:
        """
        Classify current regime using rule-based features.
        """
        features = self.extract_features(df_m5, df_m15, df_h1)
        
        if not features:
            return {
                'regime_name': 'UNKNOWN',
                'trend': 'UNKNOWN',
                'volatility': 'NORMAL',
                'trend_strength': 0.5
            }
        
        trend_strength = features.get('trend_strength', 0.5)
        
        # Classify based on trend strength
        if trend_strength > 25:
            trend = features.get('trend_direction', 'UNKNOWN')
            regime_name = f"STRONG_{trend}"
        elif trend_strength > 15:
            trend = features.get('trend_direction', 'UNKNOWN')
            regime_name = f"MILD_{trend}"
        else:
            trend = 'SIDEWAY'
            regime_name = 'SIDEWAY'
        
        return {
            'regime_name': regime_name,
            'trend': trend,
            'volatility': 'NORMAL',
            'trend_strength': trend_strength
        }
    
    def extract_features(self, df_m5: pd.DataFrame, df_m15: pd.DataFrame = None,
                         df_h1: pd.DataFrame = None) -> Dict:
        """
        Extract rule-based features for regime classification.
        [FIX] Added comprehensive NaN handling for ADX calculation.
        """
        features = {}
        
        if df_m5 is None or len(df_m5) < 50:
            return features
        
        # ADX-like trend strength (simplified)
        close = df_m5['close'].to_numpy()
        high = df_m5['high'].to_numpy()
        low = df_m5['low'].to_numpy()
        
        # +DM and -DM
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # ATR
        atr_series = ATRCache.get_atr(df_m5, 14)
        if atr_series.isna().all():
            features['trend_strength'] = 0.5
        else:
            atr = atr_series.to_numpy()
            
            period = 14
            if len(plus_dm) >= period:
                plus_di = 100 * pd.Series(plus_dm).rolling(period).mean().to_numpy() / (atr[1:] + 1e-10)
                minus_di = 100 * pd.Series(minus_dm).rolling(period).mean().to_numpy() / (atr[1:] + 1e-10)
                
                # [FIX] Replace NaN with 0 before calculating DX
                plus_di = np.nan_to_num(plus_di, nan=0.0)
                minus_di = np.nan_to_num(minus_di, nan=0.0)
                
                # DX and ADX
                dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
                dx = np.nan_to_num(dx, nan=0.0)
                
                adx = pd.Series(dx).rolling(period).mean().to_numpy()
                
                # [FIX] Get last non-NaN value safely
                valid_adx = adx[~np.isnan(adx)]
                if len(valid_adx) > 0:
                    features['trend_strength'] = float(valid_adx[-1])
                else:
                    features['trend_strength'] = 0.5
            else:
                features['trend_strength'] = 0.5
        
        # Determine trend direction
        if len(close) >= 50:
            sma_50 = np.mean(close[-50:])
            if close[-1] > sma_50:
                features['trend_direction'] = 'BULL_TREND'
            else:
                features['trend_direction'] = 'BEAR_TREND'
        
        return features