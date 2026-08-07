"""
Propulsion Engine.

Provides momentum and propulsion analysis for trend-following strategies.
Measures the "force" behind price movements to identify strong trends.

Propulsion Concepts:
  - Momentum: Rate of price change
  - Acceleration: Rate of momentum change
  - Thrust: Directional force
  - Trend Strength: Persistence of movement

Used by:
  - S14_Propulsion (Propulsion-based trend strategy)
  - Trend strength analysis
  - Momentum-based entry/exit signals
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple


class PropulsionEngine:
    """
    Propulsion and Momentum Analysis engine.
    
    Features:
      - Propulsion score calculation
      - Momentum analysis
      - Acceleration detection
      - Directional thrust calculation
      - Trend strength measurement
    """

    def __init__(self):
        """Initialize PropulsionEngine."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Propulsion parameters
        self.momentum_period = 14  # Momentum lookback
        self.acceleration_period = 7  # Acceleration lookback
        self.smoothing_period = 5  # Smoothing for noise reduction
        self.trend_period = 20  # Trend strength period

    # =========================================================================
    # PROPULSION SCORE
    # =========================================================================

    def calculate_propulsion_score(
        self, df: pd.DataFrame, lookback: int = None
    ) -> Dict:
        """
        Calculate overall propulsion score.
        
        Combines momentum, acceleration, and thrust into a single score.
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of bars to analyze
            
        Returns:
            Dict with propulsion analysis
        """
        if lookback is None:
            lookback = self.momentum_period * 2

        if df is None or df.empty or len(df) < lookback:
            return {
                'propulsion_score': 0.0,
                'momentum': 0.0,
                'acceleration': 0.0,
                'thrust': 0.0,
                'trend_strength': 0.0,
                'direction': 'NEUTRAL'
            }

        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            # Calculate components
            momentum = self.calculate_momentum(close)
            acceleration = self.calculate_acceleration(momentum)
            thrust = self.calculate_thrust(close, high, low)
            trend_strength = self.calculate_trend_strength(close, high, low)

            # Calculate propulsion score (0-100)
            propulsion_score = self._combine_propulsion_components(
                momentum, acceleration, thrust, trend_strength
            )

            # Determine direction
            direction = self._determine_direction(momentum, thrust)

            return {
                'propulsion_score': float(propulsion_score),
                'momentum': float(momentum),
                'acceleration': float(acceleration),
                'thrust': float(thrust),
                'trend_strength': float(trend_strength),
                'direction': direction,
                'momentum_series': momentum,
                'acceleration_series': acceleration
            }

        except Exception as e:
            self.logger.error(f"[PROPULSION] Score calculation error: {e}")
            return {
                'propulsion_score': 0.0,
                'momentum': 0.0,
                'acceleration': 0.0,
                'thrust': 0.0,
                'trend_strength': 0.0,
                'direction': 'NEUTRAL'
            }

    # =========================================================================
    # MOMENTUM CALCULATION
    # =========================================================================

    def calculate_momentum(self, close: np.ndarray, period: int = None) -> np.ndarray:
        """
        Calculate price momentum.
        
        Momentum = Current Price - Price N periods ago
        
        Args:
            close: Close price array
            period: Momentum lookback period
            
        Returns:
            Momentum array
        """
        if period is None:
            period = self.momentum_period

        if close is None or len(close) < period + 1:
            return np.zeros_like(close) if close is not None else np.array([])

        try:
            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))

            n = len(close)
            momentum = np.zeros(n)

            for i in range(period, n):
                momentum[i] = close[i] - close[i - period]

            # Smooth momentum
            momentum = self._smooth(momentum, self.smoothing_period)

            return momentum

        except Exception as e:
            self.logger.debug(f"[PROPULSION] Momentum calculation error: {e}")
            return np.zeros_like(close)

    # =========================================================================
    # ACCELERATION CALCULATION
    # =========================================================================

    def calculate_acceleration(self, momentum: np.ndarray, period: int = None) -> np.ndarray:
        """
        Calculate momentum acceleration.
        
        Acceleration = Change in momentum over time
        
        Args:
            momentum: Momentum array
            period: Acceleration lookback period
            
        Returns:
            Acceleration array
        """
        if period is None:
            period = self.acceleration_period

        if momentum is None or len(momentum) < period + 1:
            return np.zeros_like(momentum) if momentum is not None else np.array([])

        try:
            n = len(momentum)
            acceleration = np.zeros(n)

            for i in range(period, n):
                acceleration[i] = momentum[i] - momentum[i - period]

            # Smooth acceleration
            acceleration = self._smooth(acceleration, self.smoothing_period)

            return acceleration

        except Exception as e:
            self.logger.debug(f"[PROPULSION] Acceleration calculation error: {e}")
            return np.zeros_like(momentum)

    # =========================================================================
    # THRUST CALCULATION
    # =========================================================================

    def calculate_thrust(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """
        Calculate directional thrust.
        
        Thrust measures the directional force of price movement.
        Positive thrust = upward force
        Negative thrust = downward force
        
        Args:
            close: Close price array
            high: High price array
            low: Low price array
            
        Returns:
            Thrust array
        """
        if close is None or len(close) < 10:
            return np.zeros_like(close) if close is not None else np.array([])

        try:
            n = len(close)
            thrust = np.zeros(n)

            for i in range(1, n):
                # Calculate candle range
                candle_range = high[i] - low[i]
                if candle_range <= 0:
                    continue

                # Calculate body position (close relative to range)
                body_position = (close[i] - low[i]) / candle_range

                # Calculate price change
                price_change = close[i] - close[i-1]

                # Thrust combines direction and strength
                # Positive thrust: close near high (bullish)
                # Negative thrust: close near low (bearish)
                direction_factor = (body_position - 0.5) * 2  # -1 to 1

                # Scale by price change magnitude
                magnitude = abs(price_change) / candle_range

                thrust[i] = direction_factor * magnitude

            # Smooth thrust
            thrust = self._smooth(thrust, self.smoothing_period)

            return thrust

        except Exception as e:
            self.logger.debug(f"[PROPULSION] Thrust calculation error: {e}")
            return np.zeros_like(close)

    # =========================================================================
    # TREND STRENGTH CALCULATION
    # =========================================================================

    def calculate_trend_strength(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int = None
    ) -> np.ndarray:
        """
        Calculate trend strength.
        
        Trend strength measures the persistence of price movement.
        High strength = strong trend
        Low strength = weak or ranging market
        
        Args:
            close: Close price array
            high: High price array
            low: Low price array
            period: Trend strength period
            
        Returns:
            Trend strength array (0-1)
        """
        if period is None:
            period = self.trend_period

        if close is None or len(close) < period + 10:
            return np.zeros_like(close) if close is not None else np.array([])

        try:
            n = len(close)
            trend_strength = np.zeros(n)

            for i in range(period, n):
                # Calculate directional movement
                window_close = close[i-period:i+1]
                window_high = high[i-period:i+1]
                window_low = low[i-period:i+1]

                # Price change over period
                price_change = abs(window_close[-1] - window_close[0])

                # Total range over period
                total_range = np.sum(window_high - window_low)

                if total_range <= 0:
                    trend_strength[i] = 0
                    continue

                # Trend strength = directional movement / total movement
                # High ratio = strong trend (price moves in one direction)
                # Low ratio = weak trend (price moves back and forth)
                trend_strength[i] = price_change / total_range

            # Clamp to [0, 1]
            trend_strength = np.clip(trend_strength, 0, 1)

            return trend_strength

        except Exception as e:
            self.logger.debug(f"[PROPULSION] Trend strength calculation error: {e}")
            return np.zeros_like(close)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """Apply simple moving average smoothing."""
        try:
            if len(data) < period:
                return data

            smoothed = pd.Series(data).rolling(period, min_periods=1).mean().values
            return np.nan_to_num(smoothed, nan=0.0)

        except Exception:
            return data

    def _combine_propulsion_components(
        self, momentum: np.ndarray, acceleration: np.ndarray,
        thrust: np.ndarray, trend_strength: np.ndarray
    ) -> float:
        """
        Combine propulsion components into a single score (0-100).
        
        Returns:
            Propulsion score
        """
        try:
            # Get current values
            current_momentum = momentum[-1] if len(momentum) > 0 else 0
            current_acceleration = acceleration[-1] if len(acceleration) > 0 else 0
            current_thrust = thrust[-1] if len(thrust) > 0 else 0
            current_trend = trend_strength[-1] if len(trend_strength) > 0 else 0

            # Normalize momentum (scale to 0-1)
            momentum_abs = abs(current_momentum)
            momentum_normalized = min(1.0, momentum_abs / 10.0)  # Assume 10 is max typical momentum

            # Normalize acceleration
            acceleration_normalized = min(1.0, abs(current_acceleration) / 5.0)

            # Thrust is already -1 to 1, convert to 0-1
            thrust_normalized = (current_thrust + 1) / 2

            # Trend strength is already 0-1
            trend_normalized = current_trend

            # Weighted combination
            # Momentum: 35%, Acceleration: 20%, Thrust: 25%, Trend: 20%
            score = (
                momentum_normalized * 0.35 +
                acceleration_normalized * 0.20 +
                thrust_normalized * 0.25 +
                trend_normalized * 0.20
            )

            return score * 100  # Convert to 0-100 scale

        except Exception:
            return 0.0

    def _determine_direction(self, momentum: np.ndarray, thrust: np.ndarray) -> str:
        """Determine propulsion direction."""
        try:
            current_momentum = momentum[-1] if len(momentum) > 0 else 0
            current_thrust = thrust[-1] if len(thrust) > 0 else 0

            # Combine momentum and thrust for direction
            direction_score = current_momentum * 0.6 + current_thrust * 0.4

            if direction_score > 0.5:
                return 'STRONG_UP'
            elif direction_score > 0.1:
                return 'UP'
            elif direction_score < -0.5:
                return 'STRONG_DOWN'
            elif direction_score < -0.1:
                return 'DOWN'
            else:
                return 'NEUTRAL'

        except Exception:
            return 'NEUTRAL'

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_propulsion_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive propulsion summary.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with complete propulsion analysis
        """
        result = {
            'propulsion_score': 0.0,
            'momentum': 0.0,
            'acceleration': 0.0,
            'thrust': 0.0,
            'trend_strength': 0.0,
            'direction': 'NEUTRAL',
            'signal': 'NONE'
        }

        if df is None or df.empty or len(df) < 30:
            return result

        try:
            # Calculate propulsion score
            propulsion_result = self.calculate_propulsion_score(df)

            result['propulsion_score'] = propulsion_result['propulsion_score']
            result['momentum'] = propulsion_result['momentum']
            result['acceleration'] = propulsion_result['acceleration']
            result['thrust'] = propulsion_result['thrust']
            result['trend_strength'] = propulsion_result['trend_strength']
            result['direction'] = propulsion_result['direction']

            # Determine signal
            score = propulsion_result['propulsion_score']
            direction = propulsion_result['direction']

            if score >= 70:
                result['signal'] = 'STRONG_PROPULSION'
            elif score >= 50:
                result['signal'] = 'MODERATE_PROPULSION'
            elif score >= 30:
                result['signal'] = 'WEAK_PROPULSION'
            else:
                result['signal'] = 'NO_PROPULSION'

            return result

        except Exception as e:
            self.logger.error(f"[PROPULSION] Summary error: {e}")
            return result

    def format_propulsion_log(self, propulsion_result: Dict) -> str:
        """
        Format propulsion result as concise log string.
        
        Args:
            propulsion_result: Result from calculate_propulsion_score
            
        Returns:
            Formatted log string
        """
        if propulsion_result is None:
            return "[PROPULSION] Analysis failed"

        score = propulsion_result.get('propulsion_score', 0)
        momentum = propulsion_result.get('momentum', 0)
        acceleration = propulsion_result.get('acceleration', 0)
        thrust = propulsion_result.get('thrust', 0)
        direction = propulsion_result.get('direction', 'NEUTRAL')

        return (
            f"[PROPULSION] Score: {score:.1f} | "
            f"Momentum: {momentum:.2f} | "
            f"Accel: {acceleration:.2f} | "
            f"Thrust: {thrust:.2f} | "
            f"Direction: {direction}"
        )

    def is_strong_propulsion(self, propulsion_result: Dict, threshold: float = 60.0) -> bool:
        """
        Check if propulsion is strong enough for trend-following.
        
        Args:
            propulsion_result: Result from calculate_propulsion_score
            threshold: Minimum score for strong propulsion
            
        Returns:
            True if propulsion is strong
        """
        if propulsion_result is None:
            return False

        score = propulsion_result.get('propulsion_score', 0)
        return score >= threshold