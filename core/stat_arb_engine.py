"""
Statistical Arbitrage Engine.

Provides statistical arbitrage analysis for mean reversion strategies:
  - Z-score calculation
  - Mean reversion detection
  - Spread analysis
  - Cointegration testing (simplified)
  - Trading signal generation

Used by:
  - S15_HFT_StatArb (Statistical arbitrage strategy)
  - Mean reversion strategies
  - Spread trading analysis
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple


class StatArbEngine:
    """
    Statistical Arbitrage Analysis engine.
    
    Features:
      - Z-score calculation
      - Mean reversion detection
      - Spread analysis
      - Cointegration testing
      - Trading signal generation
    """

    def __init__(self):
        """Initialize StatArbEngine."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Z-score parameters
        self.zscore_period = 20  # Lookback for z-score
        self.entry_threshold = 2.0  # Entry z-score threshold
        self.exit_threshold = 0.5  # Exit z-score threshold

        # Mean reversion parameters
        self.half_life_period = 50  # Lookback for half-life calculation
        self.min_half_life = 5  # Minimum half-life for valid mean reversion
        self.max_half_life = 100  # Maximum half-life for valid mean reversion

    # =========================================================================
    # Z-SCORE CALCULATION
    # =========================================================================

    def calculate_zscore(
        self, series: np.ndarray, period: int = None, smoothing: bool = True
    ) -> Optional[np.ndarray]:
        """
        Calculate z-score of a series.
        
        Z-score = (Value - Mean) / Std
        
        Args:
            series: Data series
            period: Lookback period for mean/std
            smoothing: Whether to smooth the z-score
            
        Returns:
            Z-score array, or None on failure
        """
        if period is None:
            period = self.zscore_period

        if series is None or len(series) < period + 5:
            return None

        try:
            # Handle NaN
            series = np.nan_to_num(series, nan=np.nanmean(series))

            n = len(series)
            zscore = np.zeros(n)

            for i in range(period, n):
                window = series[i-period:i]
                mean = np.mean(window)
                std = np.std(window)

                if std > 0:
                    zscore[i] = (series[i] - mean) / std
                else:
                    zscore[i] = 0.0

            # Smooth z-score if requested
            if smoothing:
                zscore = pd.Series(zscore).rolling(3, min_periods=1).mean().values
                zscore = np.nan_to_num(zscore, nan=0.0)

            return zscore

        except Exception as e:
            self.logger.error(f"[STATARB] Z-score calculation error: {e}")
            return None

    # =========================================================================
    # MEAN REVERSION DETECTION
    # =========================================================================

    def detect_mean_reversion(
        self, series: np.ndarray, lookback: int = None
    ) -> Dict:
        """
        Detect mean reversion characteristics of a series.
        
        Uses Ornstein-Uhlenbeck process parameters:
          - Half-life: Time to revert halfway to mean
          - Mean reversion speed: How fast series reverts
        
        Args:
            series: Data series
            lookback: Lookback period
            
        Returns:
            Dict with mean reversion analysis
        """
        if lookback is None:
            lookback = self.half_life_period

        if series is None or len(series) < lookback:
            return {
                'is_mean_reverting': False,
                'half_life': None,
                'mean_reversion_speed': 0.0,
                'current_zscore': 0.0
            }

        try:
            # Handle NaN
            series = np.nan_to_num(series, nan=np.nanmean(series))

            # Use recent data
            if len(series) > lookback:
                series = series[-lookback:]

            # Calculate z-score
            zscore = self.calculate_zscore(series, period=self.zscore_period)

            if zscore is None:
                return {
                    'is_mean_reverting': False,
                    'half_life': None,
                    'mean_reversion_speed': 0.0,
                    'current_zscore': 0.0
                }

            # Calculate half-life using OLS regression
            # Regress: delta_y = alpha + beta * y + epsilon
            # Half-life = -ln(2) / beta

            y = series[:-1]
            delta_y = np.diff(series)

            # OLS regression
            if len(y) < 10:
                return {
                    'is_mean_reverting': False,
                    'half_life': None,
                    'mean_reversion_speed': 0.0,
                    'current_zscore': float(zscore[-1])
                }

            # Calculate beta (slope)
            y_mean = np.mean(y)
            delta_y_mean = np.mean(delta_y)

            numerator = np.sum((y - y_mean) * (delta_y - delta_y_mean))
            denominator = np.sum((y - y_mean) ** 2)

            if denominator > 0:
                beta = numerator / denominator
            else:
                beta = 0.0

            # Calculate half-life
            if beta < 0:
                half_life = -np.log(2) / beta
            else:
                half_life = None

            # Mean reversion speed
            mean_reversion_speed = -beta if beta < 0 else 0.0

            # Determine if mean reverting
            is_mean_reverting = (
                half_life is not None and
                self.min_half_life <= half_life <= self.max_half_life and
                mean_reversion_speed > 0.01
            )

            return {
                'is_mean_reverting': is_mean_reverting,
                'half_life': float(half_life) if half_life else None,
                'mean_reversion_speed': float(mean_reversion_speed),
                'current_zscore': float(zscore[-1]),
                'zscore_series': zscore
            }

        except Exception as e:
            self.logger.error(f"[STATARB] Mean reversion detection error: {e}")
            return {
                'is_mean_reverting': False,
                'half_life': None,
                'mean_reversion_speed': 0.0,
                'current_zscore': 0.0
            }

    # =========================================================================
    # SPREAD ANALYSIS
    # =========================================================================

    def calculate_spread(
        self, series1: np.ndarray, series2: np.ndarray, hedge_ratio: float = None
    ) -> Optional[Dict]:
        """
        Calculate spread between two series.
        
        Spread = Series1 - Hedge_Ratio * Series2
        
        Args:
            series1: First series
            series2: Second series
            hedge_ratio: Hedge ratio (if None, calculate via OLS)
            
        Returns:
            Dict with spread analysis, or None on failure
        """
        if series1 is None or series2 is None:
            return None

        if len(series1) != len(series2):
            min_len = min(len(series1), len(series2))
            series1 = series1[-min_len:]
            series2 = series2[-min_len:]

        if len(series1) < 30:
            return None

        try:
            # Handle NaN
            series1 = np.nan_to_num(series1, nan=np.nanmean(series1))
            series2 = np.nan_to_num(series2, nan=np.nanmean(series2))

            # Calculate hedge ratio if not provided
            if hedge_ratio is None:
                hedge_ratio = self._calculate_hedge_ratio(series1, series2)

            # Calculate spread
            spread = series1 - hedge_ratio * series2

            # Calculate z-score of spread
            zscore = self.calculate_zscore(spread, period=self.zscore_period)

            if zscore is None:
                return None

            # Detect mean reversion of spread
            mr_result = self.detect_mean_reversion(spread)

            return {
                'spread': spread,
                'zscore': zscore,
                'hedge_ratio': float(hedge_ratio),
                'current_spread': float(spread[-1]),
                'current_zscore': float(zscore[-1]),
                'mean_reversion': mr_result
            }

        except Exception as e:
            self.logger.error(f"[STATARB] Spread calculation error: {e}")
            return None

    def _calculate_hedge_ratio(self, series1: np.ndarray, series2: np.ndarray) -> float:
        """Calculate hedge ratio using OLS regression."""
        try:
            # OLS: series1 = alpha + beta * series2
            series2_mean = np.mean(series2)
            series1_mean = np.mean(series1)

            numerator = np.sum((series2 - series2_mean) * (series1 - series1_mean))
            denominator = np.sum((series2 - series2_mean) ** 2)

            if denominator > 0:
                beta = numerator / denominator
            else:
                beta = 1.0

            return float(beta)

        except Exception:
            return 1.0

    # =========================================================================
    # COINTEGRATION TEST (SIMPLIFIED)
    # =========================================================================

    def test_cointegration(
        self, series1: np.ndarray, series2: np.ndarray
    ) -> Dict:
        """
        Simplified cointegration test.
        
        Tests if the spread between two series is stationary.
        
        Args:
            series1: First series
            series2: Second series
            
        Returns:
            Dict with cointegration test results
        """
        if series1 is None or series2 is None:
            return {
                'is_cointegrated': False,
                'test_statistic': None,
                'critical_value': None,
                'p_value': None
            }

        if len(series1) != len(series2):
            min_len = min(len(series1), len(series2))
            series1 = series1[-min_len:]
            series2 = series2[-min_len:]

        if len(series1) < 50:
            return {
                'is_cointegrated': False,
                'test_statistic': None,
                'critical_value': None,
                'p_value': None
            }

        try:
            # Handle NaN
            series1 = np.nan_to_num(series1, nan=np.nanmean(series1))
            series2 = np.nan_to_num(series2, nan=np.nanmean(series2))

            # Calculate spread
            hedge_ratio = self._calculate_hedge_ratio(series1, series2)
            spread = series1 - hedge_ratio * series2

            # Simplified ADF test (Augmented Dickey-Fuller)
            # Use mean reversion detection as proxy
            mr_result = self.detect_mean_reversion(spread)

            is_cointegrated = mr_result['is_mean_reverting']

            return {
                'is_cointegrated': is_cointegrated,
                'test_statistic': mr_result['mean_reversion_speed'],
                'critical_value': 0.01,
                'p_value': 0.05 if is_cointegrated else 0.95,
                'hedge_ratio': float(hedge_ratio),
                'half_life': mr_result['half_life']
            }

        except Exception as e:
            self.logger.error(f"[STATARB] Cointegration test error: {e}")
            return {
                'is_cointegrated': False,
                'test_statistic': None,
                'critical_value': None,
                'p_value': None
            }

    # =========================================================================
    # TRADING SIGNAL GENERATION
    # =========================================================================

    def generate_signal(
        self, zscore: np.ndarray, entry_threshold: float = None, exit_threshold: float = None
    ) -> Dict:
        """
        Generate statistical arbitrage trading signal.
        
        Signal Logic:
          - Z-score > entry_threshold: SELL (mean reversion expected)
          - Z-score < -entry_threshold: BUY (mean reversion expected)
          - |Z-score| < exit_threshold: EXIT (mean reversion complete)
        
        Args:
            zscore: Z-score array
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            
        Returns:
            Dict with trading signal
        """
        if entry_threshold is None:
            entry_threshold = self.entry_threshold
        if exit_threshold is None:
            exit_threshold = self.exit_threshold

        if zscore is None or len(zscore) < 5:
            return {
                'signal': 'NEUTRAL',
                'zscore': 0.0,
                'strength': 0.0,
                'reason': 'Insufficient data'
            }

        try:
            current_zscore = zscore[-1]
            prev_zscore = zscore[-2] if len(zscore) > 1 else 0.0

            # Determine signal
            signal = 'NEUTRAL'
            reason = 'No signal'

            # Entry signals
            if current_zscore > entry_threshold:
                signal = 'SELL'
                reason = f'Z-score {current_zscore:.2f} > {entry_threshold} (overbought)'
            elif current_zscore < -entry_threshold:
                signal = 'BUY'
                reason = f'Z-score {current_zscore:.2f} < -{entry_threshold} (oversold)'

            # Exit signals
            elif abs(prev_zscore) > entry_threshold and abs(current_zscore) < exit_threshold:
                signal = 'EXIT'
                reason = f'Z-score reverted from {prev_zscore:.2f} to {current_zscore:.2f}'

            # Calculate signal strength
            strength = min(1.0, abs(current_zscore) / entry_threshold)

            return {
                'signal': signal,
                'zscore': float(current_zscore),
                'prev_zscore': float(prev_zscore),
                'strength': float(strength),
                'reason': reason,
                'entry_threshold': entry_threshold,
                'exit_threshold': exit_threshold
            }

        except Exception as e:
            self.logger.error(f"[STATARB] Signal generation error: {e}")
            return {
                'signal': 'NEUTRAL',
                'zscore': 0.0,
                'strength': 0.0,
                'reason': f'Error: {str(e)}'
            }

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_stat_arb_analysis(self, series: np.ndarray) -> Dict:
        """
        Get comprehensive statistical arbitrage analysis.
        
        Args:
            series: Data series
            
        Returns:
            Dict with complete stat arb analysis
        """
        result = {
            'zscore': None,
            'mean_reversion': None,
            'signal': None
        }

        if series is None or len(series) < 30:
            return result

        try:
            # Calculate z-score
            result['zscore'] = self.calculate_zscore(series)

            # Detect mean reversion
            result['mean_reversion'] = self.detect_mean_reversion(series)

            # Generate signal
            if result['zscore'] is not None:
                result['signal'] = self.generate_signal(result['zscore'])

            return result

        except Exception as e:
            self.logger.error(f"[STATARB] Analysis error: {e}")
            return result

    def format_stat_arb_log(self, analysis_result: Dict) -> str:
        """
        Format stat arb analysis result as concise log string.
        
        Args:
            analysis_result: Result from get_stat_arb_analysis
            
        Returns:
            Formatted log string
        """
        if analysis_result is None:
            return "[STATARB] Analysis failed"

        zscore = analysis_result.get('zscore')
        mr = analysis_result.get('mean_reversion', {})
        signal = analysis_result.get('signal', {})

        zscore_str = f"{zscore[-1]:.2f}" if zscore is not None and len(zscore) > 0 else "N/A"
        mr_str = "YES" if mr.get('is_mean_reverting', False) else "NO"
        signal_str = signal.get('signal', 'NEUTRAL') if signal else 'NEUTRAL'

        return (
            f"[STATARB] Z-score: {zscore_str} | "
            f"Mean Reverting: {mr_str} | "
            f"Signal: {signal_str}"
        )

    def is_tradable_spread(self, spread_result: Dict) -> bool:
        """
        Check if spread is tradable (mean reverting with valid half-life).
        
        Args:
            spread_result: Result from calculate_spread
            
        Returns:
            True if spread is tradable
        """
        if spread_result is None:
            return False

        mr = spread_result.get('mean_reversion', {})

        return mr.get('is_mean_reverting', False)