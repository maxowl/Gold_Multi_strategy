"""
core/choppy_detector.py
Multi-Indicator Ensemble for Choppy Market Detection
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging


class ChoppyDetector:
    """
    Detects choppy market conditions using 5 indicators:
    1. ADX (Trend Strength)
    2. Hurst Exponent (Fractal Dimension)
    3. Bollinger Band Width (Volatility Compression)
    4. Price Efficiency Ratio (Noise vs Signal)
    5. Consecutive Direction Changes (Whipsaw Count)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX (Average Directional Index)."""
        if df is None or len(df) < period + 1:
            return 0.0
        
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Calculate +DM and -DM
            plus_dm = np.zeros(len(df))
            minus_dm = np.zeros(len(df))
            
            for i in range(1, len(df)):
                up_move = high[i] - high[i-1]
                down_move = low[i-1] - low[i]
                
                if up_move > down_move and up_move > 0:
                    plus_dm[i] = up_move
                if down_move > up_move and down_move > 0:
                    minus_dm[i] = down_move
            
            # Calculate TR (True Range)
            tr = np.zeros(len(df))
            for i in range(1, len(df)):
                tr[i] = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                )
            
            # Smooth with Wilder's method
            atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
            plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values / (atr + 1e-10)
            minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values / (atr + 1e-10)
            
            # Calculate DX and ADX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
            
            return float(adx[-1]) if not np.isnan(adx[-1]) else 0.0
            
        except Exception as e:
            self.logger.error(f"[ADX] Error: {e}")
            return 0.0
    
    def calculate_efficiency_ratio(self, df: pd.DataFrame, period: int = 10) -> float:
        """
        Calculate Price Efficiency Ratio (Kaufman's ER).
        ER = Direction / Volatility
        
        ER ≈ 1.0: Strong trend (efficient movement)
        ER ≈ 0.0: Choppy (high noise, low direction)
        """
        if df is None or len(df) < period + 1:
            return 0.5
        
        try:
            close = df['close'].values
            
            # Direction: Net price change over period
            direction = abs(close[-1] - close[-period-1])
            
            # Volatility: Sum of absolute bar-to-bar changes
            volatility = sum(abs(close[i] - close[i-1]) for i in range(-period, 0))
            
            if volatility == 0:
                return 0.5
            
            er = direction / volatility
            return float(np.clip(er, 0, 1))
            
        except Exception as e:
            self.logger.error(f"[ER] Error: {e}")
            return 0.5
    
    def count_direction_changes(self, df: pd.DataFrame, lookback: int = 20) -> int:
        """
        Count consecutive direction changes (whipsaw indicator).
        More changes = more choppy.
        """
        if df is None or len(df) < lookback + 1:
            return 0
        
        try:
            close = df['close'].values[-lookback-1:]
            
            # Calculate bar-to-bar direction
            directions = np.sign(np.diff(close))
            
            # Count direction changes
            changes = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i-1])
            
            return int(changes)
            
        except Exception as e:
            self.logger.error(f"[DIR_CHANGES] Error: {e}")
            return 0
    
    def calculate_bb_width_percentile(self, df: pd.DataFrame, period: int = 20, lookback: int = 100) -> float:
        """
        Calculate Bollinger Band Width Percentile.
        Low percentile = Compression (potential breakout or choppy)
        """
        if df is None or len(df) < lookback:
            return 0.5
        
        try:
            close = df['close'].values
            
            # Calculate BB Width
            sma = pd.Series(close).rolling(window=period, min_periods=period).mean()
            std = pd.Series(close).rolling(window=period, min_periods=period).std()
            bb_width = (4 * std / (sma + 1e-10)) * 100
            
            # Percentile rank
            recent_widths = bb_width.iloc[-lookback:]
            current_width = bb_width.iloc[-1]
            
            if pd.isna(current_width):
                return 0.5
            
            percentile = (recent_widths < current_width).sum() / len(recent_widths)
            return float(percentile)
            
        except Exception as e:
            self.logger.error(f"[BB_WIDTH] Error: {e}")
            return 0.5
    
    def detect_choppy(self, df: pd.DataFrame, hurst_value: float = None) -> Dict:
        """
        Comprehensive choppy detection using 5 indicators.
        
        Returns:
            {
                'choppy_score': 0-100,
                'is_choppy': bool,
                'severity': 'LOW' | 'MEDIUM' | 'HIGH' | 'EXTREME',
                'components': dict,
                'recommendation': str
            }
        """
        if df is None or len(df) < 30:
            return {
                'choppy_score': 50,
                'is_choppy': False,
                'severity': 'UNKNOWN',
                'components': {},
                'recommendation': 'Insufficient data'
            }
        
        try:
            # =========================================================================
            # Component 1: ADX (Weight: 25%)
            # =========================================================================
            adx = self.calculate_adx(df, period=14)
            # ADX < 20 = Choppy (score 100), ADX > 40 = Strong Trend (score 0)
            adx_score = max(0, min(100, (25 - adx) * 4))
            
            # =========================================================================
            # Component 2: Hurst Exponent (Weight: 25%)
            # =========================================================================
            if hurst_value is None:
                from core.hurst_wavelet_engine import HurstWaveletEngine
                hurst_engine = HurstWaveletEngine()
                hurst_value = hurst_engine.calculate_hurst_exponent(df['close'], max_lag=50)
            
            # Hurst ≈ 0.5 = Choppy (score 100), Hurst > 0.6 or < 0.4 = Not Choppy (score 0)
            hurst_distance = abs(hurst_value - 0.5)
            hurst_score = max(0, min(100, (0.1 - hurst_distance) * 1000))
            
            # =========================================================================
            # Component 3: Efficiency Ratio (Weight: 20%)
            # =========================================================================
            er = self.calculate_efficiency_ratio(df, period=10)
            # ER < 0.3 = Choppy (score 100), ER > 0.7 = Trending (score 0)
            er_score = max(0, min(100, (0.5 - er) * 250))
            
            # =========================================================================
            # Component 4: Direction Changes (Weight: 15%)
            # =========================================================================
            dir_changes = self.count_direction_changes(df, lookback=20)
            # >10 changes = Choppy (score 100), <5 changes = Trending (score 0)
            dir_score = max(0, min(100, (dir_changes - 5) * 20))
            
            # =========================================================================
            # Component 5: BB Width Percentile (Weight: 15%)
            # =========================================================================
            bb_percentile = self.calculate_bb_width_percentile(df, period=20, lookback=100)
            # Percentile < 20% = Compression/Choppy (score 100), > 80% = Volatile (score 0)
            bb_score = max(0, min(100, (0.3 - bb_percentile) * 333))
            
            # =========================================================================
            # Weighted Ensemble Score
            # =========================================================================
            choppy_score = (
                adx_score * 0.25 +
                hurst_score * 0.25 +
                er_score * 0.20 +
                dir_score * 0.15 +
                bb_score * 0.15
            )
            
            # Determine severity
            if choppy_score >= 80:
                severity = 'EXTREME'
                is_choppy = True
            elif choppy_score >= 65:
                severity = 'HIGH'
                is_choppy = True
            elif choppy_score >= 50:
                severity = 'MEDIUM'
                is_choppy = True
            elif choppy_score >= 35:
                severity = 'LOW'
                is_choppy = False
            else:
                severity = 'NONE'
                is_choppy = False
            
            # Recommendation
            if severity == 'EXTREME':
                recommendation = 'STOP TRADING - Extreme choppy conditions'
            elif severity == 'HIGH':
                recommendation = 'Reduce position size 50%, use only mean-reversion strategies'
            elif severity == 'MEDIUM':
                recommendation = 'Reduce position size 25%, avoid trend-following'
            elif severity == 'LOW':
                recommendation = 'Normal trading, monitor closely'
            else:
                recommendation = 'Strong trending market, full position size'
            
            return {
                'choppy_score': choppy_score,
                'is_choppy': is_choppy,
                'severity': severity,
                'components': {
                    'adx': {'value': adx, 'score': adx_score},
                    'hurst': {'value': hurst_value, 'score': hurst_score},
                    'efficiency_ratio': {'value': er, 'score': er_score},
                    'direction_changes': {'value': dir_changes, 'score': dir_score},
                    'bb_width_percentile': {'value': bb_percentile, 'score': bb_score}
                },
                'recommendation': recommendation
            }
            
        except Exception as e:
            self.logger.error(f"[CHOPPY] Error: {e}")
            return {
                'choppy_score': 50,
                'is_choppy': False,
                'severity': 'UNKNOWN',
                'components': {},
                'recommendation': f'Error: {e}'
            }
    
    def get_position_size_multiplier(self, choppy_result: Dict) -> float:
        """
        Get position size multiplier based on choppy severity.
        
        Returns:
            multiplier: 0.0 to 1.0
        """
        severity = choppy_result.get('severity', 'NONE')
        
        multipliers = {
            'EXTREME': 0.0,    # Stop trading
            'HIGH': 0.5,       # 50% size
            'MEDIUM': 0.75,    # 75% size
            'LOW': 1.0,        # Full size
            'NONE': 1.0        # Full size
        }
        
        return multipliers.get(severity, 1.0)