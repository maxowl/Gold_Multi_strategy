"""
Kalman Filter & Squeeze Detection Engine.

Provides Kalman filter-based signal processing and volatility squeeze detection.
Combines the power of optimal state estimation with volatility compression analysis.

Kalman Filter:
  Optimal recursive filter that estimates the true state of a system
  from noisy measurements. Used for trend extraction and noise reduction.

Squeeze Detection:
  Detects when Bollinger Bands are inside Keltner Channels,
  indicating volatility compression and potential breakout.

Used by:
  - S24_KalmanMomentum (Kalman-based momentum strategy)
  - Volatility squeeze detection
  - Trend filtering
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional


class KalmanSqueezeEngine:
    """
    Kalman Filter and Squeeze Detection engine.
    
    Features:
      - Kalman filter for price smoothing
      - Kalman-based trend calculation
      - Bollinger Band + Keltner Channel squeeze detection
      - Squeeze momentum calculation
      - Breakout direction prediction
    """

    def __init__(self):
        """Initialize KalmanSqueezeEngine."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Kalman filter parameters
        self.process_noise = 0.01  # Q - process noise covariance
        self.measurement_noise = 0.1  # R - measurement noise covariance

        # Squeeze parameters
        self.bb_period = 20  # Bollinger Bands period
        self.bb_std = 2.0  # Bollinger Bands standard deviation
        self.kc_period = 20  # Keltner Channel period
        self.kc_atr_mult = 1.5  # Keltner Channel ATR multiplier

        # Momentum parameters
        self.momentum_period = 12  # Momentum lookback period

    # =========================================================================
    # KALMAN FILTER
    # =========================================================================

    def apply_kalman_filter(
        self, signal: np.ndarray, process_noise: float = None,
        measurement_noise: float = None
    ) -> Optional[np.ndarray]:
        """
        Apply Kalman filter to smooth a signal.
        
        Kalman filter equations:
          Predict: x_pred = x_prev, P_pred = P_prev + Q
          Update: K = P_pred / (P_pred + R), x = x_pred + K*(z - x_pred)
        
        Args:
            signal: Input signal (price series)
            process_noise: Process noise covariance Q
            measurement_noise: Measurement noise covariance R
            
        Returns:
            Filtered signal, or None on failure
        """
        if signal is None or len(signal) < 5:
            return None

        try:
            # Handle NaN
            signal = np.nan_to_num(signal, nan=np.nanmean(signal))

            if process_noise is None:
                process_noise = self.process_noise
            if measurement_noise is None:
                measurement_noise = self.measurement_noise

            n = len(signal)
            filtered = np.zeros(n)

            # Initialize state
            x = signal[0]  # State estimate
            P = 1.0  # Error covariance

            for i in range(n):
                # Predict step
                x_pred = x
                P_pred = P + process_noise

                # Update step
                K = P_pred / (P_pred + measurement_noise)  # Kalman gain
                x = x_pred + K * (signal[i] - x_pred)
                P = (1 - K) * P_pred

                filtered[i] = x

            return filtered

        except Exception as e:
            self.logger.error(f"[KALMAN] Filter application error: {e}")
            return None

    def calculate_kalman_trend(
        self, close: np.ndarray, trend_period: int = 20
    ) -> Optional[np.ndarray]:
        """
        Calculate trend using Kalman filter.
        
        Uses Kalman filter to extract trend component from price.
        
        Args:
            close: Close price array
            trend_period: Period for trend calculation
            
        Returns:
            Trend array, or None on failure
        """
        if close is None or len(close) < trend_period + 10:
            return None

        try:
            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))

            # Apply Kalman filter with adjusted parameters
            kalman_signal = self.apply_kalman_filter(
                close,
                process_noise=0.005,  # Lower noise for trend
                measurement_noise=0.2  # Higher measurement noise
            )

            if kalman_signal is None:
                return None

            # Calculate trend as difference between Kalman signal and SMA
            sma = pd.Series(close).rolling(trend_period).mean().values
            sma = np.nan_to_num(sma, nan=np.nanmean(sma))

            trend = kalman_signal - sma

            return trend

        except Exception as e:
            self.logger.error(f"[KALMAN] Trend calculation error: {e}")
            return None

    # =========================================================================
    # SQUEEZE DETECTION
    # =========================================================================

    def detect_squeeze(
        self, df: pd.DataFrame, bb_period: int = None, kc_period: int = None
    ) -> Dict:
        """
        Detect volatility squeeze (Bollinger Bands inside Keltner Channels).
        
        Squeeze ON: BB inside KC (low volatility, compression)
        Squeeze OFF: BB outside KC (high volatility, expansion)
        
        Args:
            df: DataFrame with OHLCV data
            bb_period: Bollinger Bands period
            kc_period: Keltner Channel period
            
        Returns:
            Dict with squeeze status and bands
        """
        default_result = {
            'squeeze_on': False,
            'squeeze_count': 0,
            'bb_upper': None,
            'bb_lower': None,
            'kc_upper': None,
            'kc_lower': None,
            'band_width': None,
            'momentum': 0.0,
            'breakout_direction': 'UNKNOWN'
        }

        if df is None or df.empty or len(df) < 30:
            return default_result

        try:
            if bb_period is None:
                bb_period = self.bb_period
            if kc_period is None:
                kc_period = self.kc_period

            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            n = len(close)

            # Calculate Bollinger Bands
            sma = pd.Series(close).rolling(bb_period).mean().values
            std = pd.Series(close).rolling(bb_period).std().values

            bb_upper = sma + self.bb_std * std
            bb_lower = sma - self.bb_std * std

            # Calculate Keltner Channels
            atr = self._calculate_atr(high, low, close, kc_period)

            kc_upper = sma + self.kc_atr_mult * atr
            kc_lower = sma - self.kc_atr_mult * atr

            # Handle NaN
            bb_upper = np.nan_to_num(bb_upper, nan=np.nanmax(bb_upper))
            bb_lower = np.nan_to_num(bb_lower, nan=np.nanmin(bb_lower))
            kc_upper = np.nan_to_num(kc_upper, nan=np.nanmax(kc_upper))
            kc_lower = np.nan_to_num(kc_lower, nan=np.nanmin(kc_lower))

            # Detect squeeze: BB inside KC
            squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)

            # Count consecutive squeeze bars
            squeeze_count = 0
            for i in range(n - 1, -1, -1):
                if squeeze_on[i]:
                    squeeze_count += 1
                else:
                    break

            # Calculate band width (normalized)
            band_width = (bb_upper - bb_lower) / (sma + 1e-10)

            # Calculate momentum
            momentum = self._calculate_momentum(close, bb_period)

            # Determine breakout direction
            current_momentum = momentum[-1] if len(momentum) > 0 else 0
            if squeeze_count == 0 and len(squeeze_on) > 1:
                # Squeeze just ended - determine direction
                if current_momentum > 0:
                    breakout_direction = 'UP'
                elif current_momentum < 0:
                    breakout_direction = 'DOWN'
                else:
                    breakout_direction = 'UNKNOWN'
            elif squeeze_count > 0:
                breakout_direction = 'PENDING'
            else:
                breakout_direction = 'NONE'

            return {
                'squeeze_on': bool(squeeze_on[-1]),
                'squeeze_count': squeeze_count,
                'bb_upper': float(bb_upper[-1]),
                'bb_lower': float(bb_lower[-1]),
                'kc_upper': float(kc_upper[-1]),
                'kc_lower': float(kc_lower[-1]),
                'band_width': float(band_width[-1]),
                'momentum': float(current_momentum),
                'breakout_direction': breakout_direction,
                'bb_upper_series': bb_upper,
                'bb_lower_series': bb_lower,
                'kc_upper_series': kc_upper,
                'kc_lower_series': kc_lower
            }

        except Exception as e:
            self.logger.error(f"[KALMAN] Squeeze detection error: {e}")
            return default_result

    # =========================================================================
    # SQUEEZE STATUS
    # =========================================================================

    def get_squeeze_status(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive squeeze status.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with detailed squeeze analysis
        """
        squeeze_result = self.detect_squeeze(df)

        if squeeze_result is None:
            return {
                'status': 'ERROR',
                'squeeze_on': False,
                'squeeze_count': 0,
                'momentum': 0.0,
                'breakout_direction': 'UNKNOWN'
            }

        # Determine status
        squeeze_on = squeeze_result.get('squeeze_on', False)
        squeeze_count = squeeze_result.get('squeeze_count', 0)
        momentum = squeeze_result.get('momentum', 0)
        breakout_direction = squeeze_result.get('breakout_direction', 'UNKNOWN')

        if squeeze_on:
            if squeeze_count >= 15:
                status = 'LONG_SQUEEZE'
            elif squeeze_count >= 8:
                status = 'MEDIUM_SQUEEZE'
            else:
                status = 'SHORT_SQUEEZE'
        elif breakout_direction in ['UP', 'DOWN']:
            status = 'BREAKOUT'
        else:
            status = 'NORMAL'

        return {
            'status': status,
            'squeeze_on': squeeze_on,
            'squeeze_count': squeeze_count,
            'momentum': momentum,
            'breakout_direction': breakout_direction,
            'band_width': squeeze_result.get('band_width', 0),
            'details': squeeze_result
        }

    # =========================================================================
    # MOMENTUM CALCULATION
    # =========================================================================

    def calculate_momentum(self, df: pd.DataFrame, period: int = None) -> Optional[np.ndarray]:
        """
        Calculate momentum using linear regression.
        
        Args:
            df: DataFrame with OHLCV data
            period: Momentum lookback period
            
        Returns:
            Momentum array, or None on failure
        """
        if df is None or df.empty or len(df) < 20:
            return None

        try:
            if period is None:
                period = self.momentum_period

            close = df['close'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))

            momentum = self._calculate_momentum(close, period)

            return momentum

        except Exception as e:
            self.logger.error(f"[KALMAN] Momentum calculation error: {e}")
            return None

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _calculate_atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> np.ndarray:
        """Calculate Average True Range."""
        try:
            n = len(high)
            tr = np.zeros(n)

            for i in range(1, n):
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr[i] = max(tr1, tr2, tr3)

            # Handle first element
            tr[0] = high[0] - low[0]

            # Calculate ATR using EMA
            atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values

            return np.nan_to_num(atr, nan=np.nanmean(atr))

        except Exception:
            return np.zeros(len(high))

    def _calculate_momentum(self, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate momentum using linear regression slope."""
        try:
            n = len(close)
            momentum = np.zeros(n)

            for i in range(period, n):
                window = close[i-period:i+1]

                # Linear regression slope
                x = np.arange(len(window))
                slope, _ = np.polyfit(x, window, 1)

                momentum[i] = slope

            return momentum

        except Exception:
            return np.zeros(len(close))

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_kalman_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive Kalman analysis.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with complete analysis
        """
        if df is None or df.empty or len(df) < 30:
            return {
                'kalman_signal': None,
                'kalman_trend': None,
                'squeeze': None,
                'momentum': None
            }

        close = df['close'].values.astype(float)

        # Apply Kalman filter
        kalman_signal = self.apply_kalman_filter(close)

        # Calculate trend
        kalman_trend = self.calculate_kalman_trend(close)

        # Detect squeeze
        squeeze = self.detect_squeeze(df)

        # Calculate momentum
        momentum = self.calculate_momentum(df)

        return {
            'kalman_signal': kalman_signal,
            'kalman_trend': kalman_trend,
            'squeeze': squeeze,
            'momentum': momentum,
            'current_price': float(close[-1]) if len(close) > 0 else 0
        }

    def format_kalman_log(self, squeeze_status: Dict, momentum: float = 0) -> str:
        """
        Format Kalman analysis result as concise log string.
        
        Args:
            squeeze_status: Result from get_squeeze_status
            momentum: Current momentum value
            
        Returns:
            Formatted log string
        """
        if squeeze_status is None:
            return "[KALMAN] Analysis failed"

        status = squeeze_status.get('status', 'UNKNOWN')
        squeeze_count = squeeze_status.get('squeeze_count', 0)
        direction = squeeze_status.get('breakout_direction', 'UNKNOWN')

        return (
            f"[KALMAN] Status: {status} | "
            f"Squeeze: {squeeze_count} bars | "
            f"Direction: {direction} | "
            f"Momentum: {momentum:.2f}"
        )

    def is_squeeze_breakout(self, df: pd.DataFrame) -> Dict:
        """
        Check if there's a squeeze breakout signal.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with breakout signal
        """
        squeeze_result = self.detect_squeeze(df)

        if squeeze_result is None:
            return {
                'is_breakout': False,
                'direction': 'UNKNOWN',
                'strength': 0.0
            }

        squeeze_on = squeeze_result.get('squeeze_on', False)
        breakout_direction = squeeze_result.get('breakout_direction', 'UNKNOWN')
        momentum = squeeze_result.get('momentum', 0)

        # Breakout if squeeze just ended and momentum is strong
        is_breakout = (not squeeze_on and breakout_direction in ['UP', 'DOWN'])

        # Calculate breakout strength
        strength = min(1.0, abs(momentum) / 10.0)  # Normalize momentum

        return {
            'is_breakout': is_breakout,
            'direction': breakout_direction,
            'strength': strength,
            'momentum': momentum
        }