"""
Choppy Market Detector - Multi-Factor Analysis.

Detects choppy (whipsaw) market conditions using multiple factors.
Critical for trend-following strategies which perform poorly in choppy markets.

Choppy Score (0-100):
  0-20:   Trending market (optimal for trend strategies)
  21-40:  Mildly choppy (acceptable)
  41-60:  Moderately choppy (caution)
  61-80:  Very choppy (avoid trend strategies)
  81-100: Extremely choppy (avoid all trading)

Detection Factors:
  1. Direction Changes: Number of rapid reversals
  2. ATR Ratio: Current volatility vs historical
  3. Range Efficiency: Net move vs total range traveled
  4. Candle Patterns: Doji, small body, overlapping candles
  5. Timeframe Consistency: Choppy on multiple timeframes

Severity Levels:
  NONE:      0-20  - Trending, all strategies OK
  LOW:       21-40 - Mild choppy, trend strategies OK
  MEDIUM:    41-60 - Moderate choppy, reduce position size
  HIGH:      61-80 - Very choppy, avoid trend strategies
  EXTREME:   81-100 - Extreme choppy, avoid all trading
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from config import config


class ChoppyDetector:
    """
    Detects choppy (whipsaw) market conditions using multiple factors.
    
    Provides:
      - Choppy score (0-100)
      - Severity level (NONE, LOW, MEDIUM, HIGH, EXTREME)
      - Detailed factor breakdown
      - Whipsaw pattern detection
      - Timeframe-specific analysis
    """

    # Severity thresholds
    SEVERITY_NONE = 'NONE'
    SEVERITY_LOW = 'LOW'
    SEVERITY_MEDIUM = 'MEDIUM'
    SEVERITY_HIGH = 'HIGH'
    SEVERITY_EXTREME = 'EXTREME'

    def __init__(self):
        """Initialize ChoppyDetector with configuration."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Direction change detection parameters
        self.direction_change_threshold = 3  # Minimum changes to be considered choppy

        # ATR ratio thresholds
        self.atr_ratio_high = 1.5  # Current ATR / historical ATR
        self.atr_ratio_low = 0.5

        # Range efficiency threshold
        self.range_efficiency_threshold = 0.3  # Below this = choppy

        # Candle pattern thresholds
        self.doji_body_threshold = 0.2  # Body / range ratio for doji
        self.small_body_threshold = 0.3  # Body / range ratio for small body

        # Factor weights (total = 1.0)
        self.factor_weights = {
            'direction_changes': 0.30,
            'atr_ratio': 0.20,
            'range_efficiency': 0.25,
            'candle_patterns': 0.15,
            'timeframe_consistency': 0.10
        }

        self.logger.info("[CHOPPY] Initialized with multi-factor analysis")

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def calculate_choppy_score(self, df_m15: pd.DataFrame,
                                 df_m5: pd.DataFrame = None,
                                 df_h1: pd.DataFrame = None) -> Dict:
        """
        Calculate comprehensive choppy score (0-100).
        
        Args:
            df_m15: Primary timeframe data (M15)
            df_m5: Secondary timeframe data (M5, optional)
            df_h1: Higher timeframe data (H1, optional)
            
        Returns:
            Dict with:
              - score: Choppy score (0-100)
              - severity: Severity level
              - factors: Individual factor scores
              - is_choppy: Boolean flag (score > 50)
              - details: Detailed breakdown
        """
        # Validate input
        if df_m15 is None or df_m15.empty or len(df_m15) < 30:
            return self._build_default_result("Insufficient data")

        try:
            # Extract price data
            close = df_m15['close'].values.astype(float)
            high = df_m15['high'].values.astype(float)
            low = df_m15['low'].values.astype(float)
            open_ = df_m15['open'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=close[0])
            high = np.nan_to_num(high, nan=high[0])
            low = np.nan_to_num(low, nan=low[0])
            open_ = np.nan_to_num(open_, nan=open_[0])

            # =========================================================================
            # FACTOR 1: Direction Changes
            # =========================================================================
            direction_score = self._score_direction_changes(close)

            # =========================================================================
            # FACTOR 2: ATR Ratio
            # =========================================================================
            atr_score = self._score_atr_ratio(close, high, low)

            # =========================================================================
            # FACTOR 3: Range Efficiency
            # =========================================================================
            efficiency_score = self._score_range_efficiency(close, high, low)

            # =========================================================================
            # FACTOR 4: Candle Patterns
            # =========================================================================
            candle_score = self._score_candle_patterns(open_, high, low, close)

            # =========================================================================
            # FACTOR 5: Timeframe Consistency
            # =========================================================================
            tf_score = self._score_timeframe_consistency(
                close, df_m5, df_h1
            )

            # =========================================================================
            # WEIGHTED COMBINATION
            # =========================================================================
            factors = {
                'direction_changes': direction_score,
                'atr_ratio': atr_score,
                'range_efficiency': efficiency_score,
                'candle_patterns': candle_score,
                'timeframe_consistency': tf_score
            }

            # Calculate weighted score
            total_score = 0.0
            for factor_name, factor_score in factors.items():
                weight = self.factor_weights.get(factor_name, 0.2)
                total_score += factor_score * weight

            # Clamp to 0-100
            total_score = max(0, min(100, total_score))

            # Determine severity
            severity = self.get_severity(total_score)

            # Detect whipsaw patterns
            whipsaw_detected = self.detect_whipsaw(close, high, low)

            # Build result
            result = {
                'score': round(total_score, 1),
                'severity': severity,
                'is_choppy': total_score > 50,
                'is_extreme': total_score > 80,
                'factors': {k: round(v, 1) for k, v in factors.items()},
                'whipsaw_detected': whipsaw_detected,
                'details': {
                    'direction_changes': direction_score,
                    'atr_ratio': atr_score,
                    'range_efficiency': efficiency_score,
                    'candle_patterns': candle_score,
                    'timeframe_consistency': tf_score
                }
            }

            # Log if choppy detected
            if total_score > 50:
                self.logger.info(
                    f"[CHOPPY] Detected: Score {total_score:.1f} | "
                    f"Severity: {severity} | "
                    f"Whipsaw: {'YES' if whipsaw_detected else 'NO'}"
                )

            return result

        except Exception as e:
            self.logger.error(f"[CHOPPY] Calculation error: {e}")
            return self._build_default_result(f"Error: {str(e)}")

    # =========================================================================
    # FACTOR 1: DIRECTION CHANGES
    # =========================================================================

    def _score_direction_changes(self, close: np.ndarray, lookback: int = 30) -> float:
        """
        Score based on number of direction changes.
        
        More direction changes = more choppy.
        
        Returns:
            Score 0-100
        """
        try:
            if len(close) < lookback:
                return 50.0

            # Calculate price changes
            changes = np.diff(close[-lookback:])

            # Detect sign changes (direction reversals)
            signs = np.sign(changes)
            direction_changes = np.sum(np.abs(np.diff(signs)) > 0)

            # Normalize by lookback
            max_changes = lookback - 2
            if max_changes == 0:
                return 50.0

            change_ratio = direction_changes / max_changes

            # Score: 0 changes = 0 (trending), all changes = 100 (choppy)
            score = change_ratio * 100

            return max(0, min(100, score))

        except Exception:
            return 50.0

    # =========================================================================
    # FACTOR 2: ATR RATIO
    # =========================================================================

    def _score_atr_ratio(self, close: np.ndarray, high: np.ndarray,
                          low: np.ndarray, period: int = 14) -> float:
        """
        Score based on ATR ratio (current vs historical).
        
        High ATR ratio indicates increased volatility which often
        correlates with choppy conditions.
        
        Returns:
            Score 0-100
        """
        try:
            if len(close) < period * 2:
                return 50.0

            # Calculate True Range
            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(np.abs(high[1:] - close[:-1]),
                                       np.abs(low[1:] - close[:-1])))

            # Current ATR (last period)
            current_atr = np.mean(tr[-period:])

            # Historical ATR (period before)
            historical_atr = np.mean(tr[-period*2:-period])

            if historical_atr == 0:
                return 50.0

            # Calculate ratio
            atr_ratio = current_atr / historical_atr

            # Score based on ratio
            # Ratio 1.0 = normal (50 score)
            # Ratio 1.5+ = high volatility (80+ score)
            # Ratio 0.5- = low volatility (20- score)
            if atr_ratio >= self.atr_ratio_high:
                score = 80 + (atr_ratio - self.atr_ratio_high) * 20
            elif atr_ratio <= self.atr_ratio_low:
                score = 20 - (self.atr_ratio_low - atr_ratio) * 20
            else:
                # Linear interpolation between low and high
                score = 20 + (atr_ratio - self.atr_ratio_low) / \
                        (self.atr_ratio_high - self.atr_ratio_low) * 60

            return max(0, min(100, score))

        except Exception:
            return 50.0

    # =========================================================================
    # FACTOR 3: RANGE EFFICIENCY
    # =========================================================================

    def _score_range_efficiency(self, close: np.ndarray, high: np.ndarray,
                                  low: np.ndarray, lookback: int = 30) -> float:
        """
        Score based on range efficiency.
        
        Range Efficiency = Net Move / Total Range Traveled
        
        Low efficiency (price moves back and forth) = choppy.
        High efficiency (price moves in one direction) = trending.
        
        Returns:
            Score 0-100
        """
        try:
            if len(close) < lookback:
                return 50.0

            recent_close = close[-lookback:]
            recent_high = high[-lookback:]
            recent_low = low[-lookback:]

            # Net move (start to end)
            net_move = abs(recent_close[-1] - recent_close[0])

            # Total range traveled (sum of all bar ranges)
            total_range = np.sum(recent_high - recent_low)

            if total_range == 0:
                return 50.0

            # Calculate efficiency
            efficiency = net_move / total_range

            # Score: Low efficiency = high choppy score
            # Efficiency 1.0 = perfect trend (0 score)
            # Efficiency 0.0 = pure choppy (100 score)
            score = (1 - efficiency) * 100

            return max(0, min(100, score))

        except Exception:
            return 50.0

    # =========================================================================
    # FACTOR 4: CANDLE PATTERNS
    # =========================================================================

    def _score_candle_patterns(self, open_: np.ndarray, high: np.ndarray,
                                low: np.ndarray, close: np.ndarray,
                                lookback: int = 20) -> float:
        """
        Score based on candle patterns.
        
        Detects:
          - Doji candles (body very small)
          - Small body candles
          - Overlapping candles
          - Long wick candles
        
        Returns:
            Score 0-100
        """
        try:
            if len(close) < lookback:
                return 50.0

            recent_open = open_[-lookback:]
            recent_high = high[-lookback:]
            recent_low = low[-lookback:]
            recent_close = close[-lookback:]

            # Calculate body and range for each candle
            bodies = np.abs(recent_close - recent_open)
            ranges = recent_high - recent_low

            # Avoid division by zero
            ranges_safe = np.where(ranges > 0, ranges, 1e-10)
            body_ratios = bodies / ranges_safe

            # Count doji candles (body < 20% of range)
            doji_count = np.sum(body_ratios < self.doji_body_threshold)

            # Count small body candles (body < 30% of range)
            small_body_count = np.sum(body_ratios < self.small_body_threshold)

            # Count overlapping candles (candle completely inside previous)
            overlap_count = 0
            for i in range(1, len(recent_close)):
                prev_high = recent_high[i-1]
                prev_low = recent_low[i-1]
                curr_high = recent_high[i]
                curr_low = recent_low[i]

                if curr_high < prev_high and curr_low > prev_low:
                    overlap_count += 1

            # Count long wick candles
            upper_wicks = recent_high - np.maximum(recent_open, recent_close)
            lower_wicks = np.minimum(recent_open, recent_close) - recent_low
            long_wick_count = np.sum((upper_wicks / ranges_safe) > 0.6) + \
                              np.sum((lower_wicks / ranges_safe) > 0.6)

            # Calculate total choppy indicators
            total_indicators = doji_count + small_body_count + overlap_count + long_wick_count
            max_indicators = lookback * 4  # Max 4 indicators per candle

            if max_indicators == 0:
                return 50.0

            # Score based on indicator frequency
            indicator_ratio = total_indicators / max_indicators
            score = indicator_ratio * 100

            return max(0, min(100, score))

        except Exception:
            return 50.0

    # =========================================================================
    # FACTOR 5: TIMEFRAME CONSISTENCY
    # =========================================================================

    def _score_timeframe_consistency(self, close_m15: np.ndarray,
                                       df_m5: pd.DataFrame = None,
                                       df_h1: pd.DataFrame = None) -> float:
        """
        Score based on choppy consistency across timeframes.
        
        If multiple timeframes are choppy, score is higher.
        
        Returns:
            Score 0-100
        """
        try:
            scores = []

            # M15 score (already calculated from main data)
            m15_score = self._simple_choppy_from_close(close_m15)
            scores.append(('M15', m15_score))

            # M5 score (if available)
            if df_m5 is not None and len(df_m5) >= 30:
                m5_close = df_m5['close'].values.astype(float)
                m5_score = self._simple_choppy_from_close(m5_close)
                scores.append(('M5', m5_score))

            # H1 score (if available)
            if df_h1 is not None and len(df_h1) >= 30:
                h1_close = df_h1['close'].values.astype(float)
                h1_score = self._simple_choppy_from_close(h1_close)
                scores.append(('H1', h1_score))

            # If only one timeframe, return its score
            if len(scores) == 1:
                return scores[0][1]

            # Check consistency
            choppy_count = sum(1 for _, score in scores if score > 50)

            if choppy_count == len(scores):
                # All timeframes choppy - high score
                return 90.0
            elif choppy_count == len(scores) - 1:
                # Most timeframes choppy - medium-high score
                return 70.0
            elif choppy_count == 1:
                # One timeframe choppy - medium-low score
                return 40.0
            else:
                # No timeframes choppy - low score
                return 10.0

        except Exception:
            return 50.0

    def _simple_choppy_from_close(self, close: np.ndarray, lookback: int = 30) -> float:
        """Simple choppy score from close prices only."""
        try:
            if len(close) < lookback:
                return 50.0

            recent = close[-lookback:]
            changes = np.diff(recent)
            signs = np.sign(changes)
            direction_changes = np.sum(np.abs(np.diff(signs)) > 0)

            max_changes = len(changes) - 1
            if max_changes == 0:
                return 50.0

            return (direction_changes / max_changes) * 100

        except Exception:
            return 50.0

    # =========================================================================
    # SEVERITY MAPPING
    # =========================================================================

    def get_severity(self, score: float) -> str:
        """
        Map choppy score to severity level.
        
        Args:
            score: Choppy score (0-100)
            
        Returns:
            Severity level string
        """
        if score >= 81:
            return self.SEVERITY_EXTREME
        elif score >= 61:
            return self.SEVERITY_HIGH
        elif score >= 41:
            return self.SEVERITY_MEDIUM
        elif score >= 21:
            return self.SEVERITY_LOW
        else:
            return self.SEVERITY_NONE

    # =========================================================================
    # WHIPSAW DETECTION
    # =========================================================================

    def detect_whipsaw(self, close: np.ndarray, high: np.ndarray,
                        low: np.ndarray, lookback: int = 10) -> bool:
        """
        Detect specific whipsaw pattern.
        
        Whipsaw: Price breaks a level then reverses quickly,
        trapping traders who entered on the breakout.
        
        Returns:
            True if whipsaw detected
        """
        try:
            if len(close) < lookback + 5:
                return False

            recent_close = close[-lookback:]
            recent_high = high[-lookback:]
            recent_low = low[-lookback:]

            # Find recent swing high/low
            swing_high_idx = np.argmax(recent_high[:-3])
            swing_low_idx = np.argmin(recent_low[:-3])

            swing_high = recent_high[swing_high_idx]
            swing_low = recent_low[swing_low_idx]

            # Check if price broke and reversed
            last_close = recent_close[-1]

            # Whipsaw: Price broke swing high but closed below it
            if last_close < swing_high and np.max(recent_close[-5:]) > swing_high:
                return True

            # Whipsaw: Price broke swing low but closed above it
            if last_close > swing_low and np.min(recent_close[-5:]) < swing_low:
                return True

            return False

        except Exception:
            return False

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _build_default_result(self, reason: str) -> Dict:
        """Build default result when calculation fails."""
        return {
            'score': 50.0,
            'severity': self.SEVERITY_MEDIUM,
            'is_choppy': False,
            'is_extreme': False,
            'factors': {},
            'whipsaw_detected': False,
            'details': {},
            'error': reason
        }

    def format_choppy_log(self, result: Dict) -> str:
        """
        Format a concise log string for choppy detection.
        
        Args:
            result: Result from calculate_choppy_score
            
        Returns:
            Formatted log string
        """
        score = result.get('score', 0)
        severity = result.get('severity', 'UNKNOWN')
        whipsaw = result.get('whipsaw_detected', False)

        factors = result.get('factors', {})
        top_factor = max(factors.items(), key=lambda x: x[1])[0] if factors else 'None'

        whipsaw_str = " | WHIPSAW" if whipsaw else ""

        return (
            f"[CHOPPY] Score: {score:.0f} | "
            f"Severity: {severity}{whipsaw_str} | "
            f"Top Factor: {top_factor}"
        )

    def get_choppy_recommendation(self, result: Dict) -> Dict:
        """
        Get trading recommendations based on choppy detection.
        
        Args:
            result: Result from calculate_choppy_score
            
        Returns:
            Dict with recommendations
        """
        score = result.get('score', 0)
        severity = result.get('severity', self.SEVERITY_NONE)

        recommendations = {
            'should_trade': True,
            'avoid_strategies': [],
            'position_multiplier': 1.0,
            'reason': 'Normal conditions'
        }

        if severity == self.SEVERITY_EXTREME:
            recommendations['should_trade'] = False
            recommendations['position_multiplier'] = 0.0
            recommendations['reason'] = 'Extreme choppy - avoid all trading'
            recommendations['avoid_strategies'] = ['ALL']

        elif severity == self.SEVERITY_HIGH:
            recommendations['position_multiplier'] = 0.5
            recommendations['reason'] = 'High choppy - reduce position size'
            recommendations['avoid_strategies'] = ['TREND']

        elif severity == self.SEVERITY_MEDIUM:
            recommendations['position_multiplier'] = 0.75
            recommendations['reason'] = 'Moderate choppy - caution advised'
            recommendations['avoid_strategies'] = ['TREND', 'SMC']

        elif severity == self.SEVERITY_LOW:
            recommendations['reason'] = 'Mild choppy - normal trading'

        return recommendations