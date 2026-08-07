"""
John Ehlers DSP Engine.

Provides advanced DSP indicators developed by John Ehlers:
  - MESA (Maximum Entropy Spectral Analysis) Adaptive Moving Average
  - Instantaneous Trendline
  - SuperSmoother and Roofing filters
  - Cycle Period detection
  - Instantaneous Phase calculation

Used by:
  - S10_EhlersMESA (MESA Adaptive MA)
  - S18_EhlersVector (Vector indicators)
  - Market cycle analysis
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict


class EhlersDSPEngine:
    """
    John Ehlers DSP indicators engine.
    
    Features:
      - MESA Adaptive Moving Average
      - Instantaneous Trendline
      - SuperSmoother filter
      - Roofing filter
      - Cycle period detection
      - Phase calculation
    """

    def __init__(self):
        """Initialize EhlersDSPEngine."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Default parameters
        self.mesa_fast_limit = 0.5
        self.mesa_slow_limit = 0.05

    # =========================================================================
    # MESA (MAXIMUM ENTROPY SPECTRAL ANALYSIS)
    # =========================================================================

    def ehlers_mesa(
        self, close: np.ndarray, fast_limit: float = None, slow_limit: float = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Calculate MESA Adaptive Moving Average.
        
        MESA uses Hilbert Transform to calculate instantaneous phase
        and adapts the moving average period based on the dominant cycle.
        
        Args:
            close: Close price array
            fast_limit: Fast limit for MAMA (default 0.5)
            slow_limit: Slow limit for MAMA (default 0.05)
            
        Returns:
            Tuple of (mama, fama) arrays, or (None, None) on failure
        """
        if fast_limit is None:
            fast_limit = self.mesa_fast_limit
        if slow_limit is None:
            slow_limit = self.mesa_slow_limit

        if close is None or len(close) < 50:
            return None, None

        try:
            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))

            n = len(close)

            # Initialize arrays
            mama = np.zeros(n)
            fama = np.zeros(n)
            i1 = np.zeros(n)  # In-phase component
            q1 = np.zeros(n)  # Quadrature component
            jI = np.zeros(n)
            jQ = np.zeros(n)
            i2 = np.zeros(n)
            q2 = np.zeros(n)
            re = np.zeros(n)
            im = np.zeros(n)
            period = np.zeros(n)
            smooth_period = np.zeros(n)
            phase = np.zeros(n)

            # Smooth price
            smooth = np.zeros(n)
            for i in range(6, n):
                smooth[i] = (4 * close[i] + 3 * close[i-1] + 2 * close[i-2] + close[i-3]) / 10

            # Detrender
            detrender = np.zeros(n)
            for i in range(12, n):
                detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i-2] -
                                0.5769 * smooth[i-4] - 0.0962 * smooth[i-6]) * \
                               (0.075 * period[i-1] + 0.54)

            # Compute InPhase and Quadrature components
            for i in range(12, n):
                q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i-2] -
                         0.5769 * detrender[i-4] - 0.0962 * detrender[i-6]) * \
                        (0.075 * period[i-1] + 0.54)
                i1[i] = detrender[i-3]

                # Advance phase of i1 and q1 by 90 degrees
                jI[i] = (0.0962 * i1[i] + 0.5769 * i1[i-2] -
                         0.5769 * i1[i-4] - 0.0962 * i1[i-6]) * \
                        (0.075 * period[i-1] + 0.54)
                jQ[i] = (0.0962 * q1[i] + 0.5769 * q1[i-2] -
                         0.5769 * q1[i-4] - 0.0962 * q1[i-6]) * \
                        (0.075 * period[i-1] + 0.54)

                # Phasor addition for 3-bar averaging
                i2[i] = i1[i] - jQ[i]
                q2[i] = q1[i] + jI[i]

                # Smooth the I and Q components
                i2[i] = 0.2 * i2[i] + 0.8 * i2[i-1]
                q2[i] = 0.2 * q2[i] + 0.8 * q2[i-1]

                # Homodyne Discriminator
                re[i] = i2[i] * i2[i-1] + q2[i] * q2[i-1]
                im[i] = i2[i] * q2[i-1] - q2[i] * i2[i-1]
                re[i] = 0.2 * re[i] + 0.8 * re[i-1]
                im[i] = 0.2 * im[i] + 0.8 * im[i-1]

                # Calculate period
                if im[i] != 0 and re[i] != 0:
                    period[i] = 2 * np.pi / np.arctan(im[i] / re[i])

                # Limit period
                if period[i] > 1.5 * period[i-1]:
                    period[i] = 1.5 * period[i-1]
                if period[i] < 0.67 * period[i-1]:
                    period[i] = 0.67 * period[i-1]
                if period[i] < 6:
                    period[i] = 6
                if period[i] > 50:
                    period[i] = 50

                period[i] = 0.2 * period[i] + 0.8 * period[i-1]
                smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i-1]

                # Calculate phase
                if i1[i] != 0:
                    phase[i] = np.degrees(np.arctan(q1[i] / i1[i]))

                delta_phase = phase[i-1] - phase[i]
                if delta_phase < 1:
                    delta_phase = 1

                # Calculate adaptive alpha
                alpha = fast_limit / delta_phase
                if alpha < slow_limit:
                    alpha = slow_limit
                if alpha > fast_limit:
                    alpha = fast_limit

                # MAMA and FAMA
                mama[i] = alpha * close[i] + (1 - alpha) * mama[i-1]
                fama[i] = 0.5 * alpha * mama[i] + (1 - 0.5 * alpha) * fama[i-1]

            return mama, fama

        except Exception as e:
            self.logger.error(f"[EHLERS] MESA calculation error: {e}")
            return None, None

    # =========================================================================
    # INSTANTANEOUS TRENDLINE
    # =========================================================================

    def instantaneous_trendline(self, close: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate Instantaneous Trendline using Hilbert Transform.
        
        Args:
            close: Close price array
            
        Returns:
            Trendline array, or None on failure
        """
        if close is None or len(close) < 30:
            return None

        try:
            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))

            n = len(close)
            trendline = np.zeros(n)

            # Simple approach: use MAMA as trendline
            mama, _ = self.ehlers_mesa(close)

            if mama is not None:
                return mama

            # Fallback: simple moving average
            period = 20
            for i in range(period, n):
                trendline[i] = np.mean(close[i-period:i])

            return trendline

        except Exception as e:
            self.logger.error(f"[EHLERS] Trendline error: {e}")
            return None

    # =========================================================================
    # SUPERSMOOTHER FILTER
    # =========================================================================

    def supersmoother(self, close: np.ndarray, period: int = 10) -> Optional[np.ndarray]:
        """
        Calculate SuperSmoother filter (Ehlers' improved low-pass filter).
        
        Args:
            close: Close price array
            period: Filter period
            
        Returns:
            Filtered signal, or None on failure
        """
        if close is None or len(close) < period + 10:
            return None

        try:
            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))

            n = len(close)
            filtered = np.zeros(n)

            # Calculate coefficients
            a = np.exp(-1.414 * np.pi / period)
            b = 2 * a * np.cos(1.414 * np.pi / period)
            c2 = b
            c3 = -a * a
            c1 = 1 - c2 - c3

            # Apply filter
            for i in range(2, n):
                filtered[i] = c1 * (close[i] + close[i-1]) / 2 + \
                              c2 * filtered[i-1] + c3 * filtered[i-2]

            return filtered

        except Exception as e:
            self.logger.error(f"[EHLERS] SuperSmoother error: {e}")
            return None

    # =========================================================================
    # ROOFING FILTER
    # =========================================================================

    def roofing_filter(
        self, close: np.ndarray, hp_period: int = 48, lp_period: int = 10
    ) -> Optional[np.ndarray]:
        """
        Calculate Roofing filter (High-pass + Low-pass).
        
        The Roofing filter removes both trend and noise,
        leaving only the cycle component.
        
        Args:
            close: Close price array
            hp_period: High-pass filter period
            lp_period: Low-pass filter period
            
        Returns:
            Filtered signal, or None on failure
        """
        if close is None or len(close) < hp_period + lp_period:
            return None

        try:
            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))

            n = len(close)
            hp = np.zeros(n)  # High-pass component
            lp = np.zeros(n)  # Low-pass component
            roofing = np.zeros(n)

            # High-pass filter coefficients
            alpha_hp = (1 - np.sin(2 * np.pi / hp_period)) / np.cos(2 * np.pi / hp_period)

            # Low-pass filter coefficients (SuperSmoother)
            a = np.exp(-1.414 * np.pi / lp_period)
            b = 2 * a * np.cos(1.414 * np.pi / lp_period)
            c2 = b
            c3 = -a * a
            c1 = 1 - c2 - c3

            # Apply high-pass filter
            for i in range(2, n):
                hp[i] = (1 - alpha_hp / 2) ** 2 * (close[i] - 2 * close[i-1] + close[i-2]) + \
                        2 * (1 - alpha_hp) * hp[i-1] - (1 - alpha_hp) ** 2 * hp[i-2]

            # Apply low-pass filter (SuperSmoother) to high-pass result
            for i in range(2, n):
                lp[i] = c1 * (hp[i] + hp[i-1]) / 2 + c2 * lp[i-1] + c3 * lp[i-2]

            roofing = lp

            return roofing

        except Exception as e:
            self.logger.error(f"[EHLERS] Roofing filter error: {e}")
            return None

    # =========================================================================
    # CYCLE PERIOD DETECTION
    # =========================================================================

    def calculate_cycle_period(self, close: np.ndarray, lookback: int = 100) -> Optional[float]:
        """
        Calculate dominant cycle period using autocorrelation.
        
        Args:
            close: Close price array
            lookback: Number of bars to analyze
            
        Returns:
            Dominant cycle period (in bars), or None on failure
        """
        if close is None or len(close) < lookback:
            return None

        try:
            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))

            # Detrend the signal
            detrended = close - np.mean(close[-lookback:])

            # Calculate autocorrelation
            n = len(detrended[-lookback:])
            autocorr = np.correlate(detrended[-lookback:], detrended[-lookback:], mode='full')
            autocorr = autocorr[n-1:]  # Keep positive lags only
            autocorr = autocorr / autocorr[0]  # Normalize

            # Find first peak after zero (skip lag 0)
            min_period = 5  # Minimum cycle period
            max_period = min(50, lookback // 2)  # Maximum cycle period

            best_period = None
            best_corr = 0

            for lag in range(min_period, max_period):
                if autocorr[lag] > best_corr and autocorr[lag] > autocorr[lag-1] and autocorr[lag] > autocorr[lag+1]:
                    best_corr = autocorr[lag]
                    best_period = lag

            return float(best_period) if best_period else None

        except Exception as e:
            self.logger.error(f"[EHLERS] Cycle period error: {e}")
            return None

    # =========================================================================
    # PHASE CALCULATION
    # =========================================================================

    def calculate_phase(self, close: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate instantaneous phase using Hilbert Transform.
        
        Args:
            close: Close price array
            
        Returns:
            Phase array (degrees), or None on failure
        """
        if close is None or len(close) < 30:
            return None

        try:
            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))

            # Use MESA to get phase
            n = len(close)
            phase = np.zeros(n)

            # Simplified phase calculation
            mama, fama = self.ehlers_mesa(close)

            if mama is None or fama is None:
                return None

            # Phase is based on the relationship between MAMA and FAMA
            for i in range(1, n):
                if mama[i] != mama[i-1]:
                    phase[i] = np.degrees(np.arctan2(fama[i] - fama[i-1], mama[i] - mama[i-1]))
                else:
                    phase[i] = phase[i-1]

            return phase

        except Exception as e:
            self.logger.error(f"[EHLERS] Phase calculation error: {e}")
            return None

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_cycle_analysis(self, close: np.ndarray) -> Dict:
        """
        Get comprehensive cycle analysis.
        
        Args:
            close: Close price array
            
        Returns:
            Dict with cycle analysis results
        """
        result = {
            'dominant_period': None,
            'phase': None,
            'trendline': None,
            'mama': None,
            'fama': None
        }

        if close is None or len(close) < 50:
            return result

        # Calculate dominant period
        result['dominant_period'] = self.calculate_cycle_period(close)

        # Calculate MESA
        mama, fama = self.ehlers_mesa(close)
        result['mama'] = mama
        result['fama'] = fama

        # Calculate trendline
        result['trendline'] = self.instantaneous_trendline(close)

        # Calculate phase
        result['phase'] = self.calculate_phase(close)

        return result

    def format_ehlers_log(self, mama: np.ndarray, fama: np.ndarray,
                           period: float = None) -> str:
        """
        Format Ehlers analysis result as concise log string.
        
        Args:
            mama: MAMA array
            fama: FAMA array
            period: Dominant cycle period
            
        Returns:
            Formatted log string
        """
        if mama is None or fama is None:
            return "[EHLERS] MESA calculation failed"

        last_mama = mama[-1] if len(mama) > 0 else 0
        last_fama = fama[-1] if len(fama) > 0 else 0
        period_str = f"{period:.1f} bars" if period else "N/A"

        return (
            f"[EHLERS] MAMA: {last_mama:.2f} | "
            f"FAMA: {last_fama:.2f} | "
            f"Cycle: {period_str}"
        )