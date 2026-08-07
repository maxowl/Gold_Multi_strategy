"""
Digital Signal Processing Engine.

Provides EMD (Empirical Mode Decomposition) and Hilbert Transform
for price signal analysis.

Used by:
  - S3_EMD_HHT (EMD + Hilbert-Huang Transform)
  - S16_RoofingEMD (Roofing filter with EMD)
  - Market cycle analysis
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Optional


class DSPEngine:
    """
    Digital Signal Processing engine for financial time series.
    
    Features:
      - Empirical Mode Decomposition (EMD)
      - Hilbert Transform for instantaneous phase/frequency
      - Signal envelope calculation
      - Trend extraction
    """

    def __init__(self):
        """Initialize DSPEngine."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # EMD parameters
        self.max_imfs = 5  # Maximum number of Intrinsic Mode Functions
        self.sifting_iterations = 10  # Max iterations for sifting process
        self.stop_threshold = 0.1  # Stop criterion for sifting

    # =========================================================================
    # EMPIRICAL MODE DECOMPOSITION
    # =========================================================================

    def empirical_mode_decomposition(
        self, signal: np.ndarray, max_imfs: int = None
    ) -> Optional[List[np.ndarray]]:
        """
        Decompose signal into Intrinsic Mode Functions (IMFs) using EMD.
        
        EMD separates a signal into oscillatory components (IMFs) and a trend.
        Each IMF represents a different frequency scale.
        
        Args:
            signal: Input signal (price series)
            max_imfs: Maximum number of IMFs to extract
            
        Returns:
            List of IMF arrays, or None on failure
        """
        if max_imfs is None:
            max_imfs = self.max_imfs

        if signal is None or len(signal) < 10:
            return None

        try:
            # Handle NaN
            signal = np.nan_to_num(signal, nan=np.nanmean(signal))

            imfs = []
            residual = signal.copy()

            for _ in range(max_imfs):
                # Extract one IMF through sifting
                imf = self._sift(residual)

                if imf is None:
                    break

                imfs.append(imf)
                residual = residual - imf

                # Check if residual is monotonic (stop condition)
                if self._is_monotonic(residual):
                    break

            # Add residual as the last component (trend)
            if len(imfs) > 0 and not self._is_monotonic(residual):
                imfs.append(residual)

            return imfs if len(imfs) > 0 else None

        except Exception as e:
            self.logger.error(f"[DSP] EMD error: {e}")
            return None

    def _sift(self, signal: np.ndarray) -> Optional[np.ndarray]:
        """
        Sifting process to extract one IMF.
        
        Returns:
            Extracted IMF, or None on failure
        """
        try:
            imf = signal.copy()

            for _ in range(self.sifting_iterations):
                # Find local maxima and minima
                maxima_idx = self._find_extrema(imf, 'max')
                minima_idx = self._find_extrema(imf, 'min')

                if len(maxima_idx) < 2 or len(minima_idx) < 2:
                    break

                # Interpolate envelopes
                upper_env = self._interpolate_envelope(imf, maxima_idx)
                lower_env = self._interpolate_envelope(imf, minima_idx)

                # Calculate mean envelope
                mean_env = (upper_env + lower_env) / 2

                # Subtract mean from signal
                imf_new = imf - mean_env

                # Check stop criterion
                diff = np.sum(np.abs(imf_new - imf)) / (np.sum(np.abs(imf)) + 1e-10)
                imf = imf_new

                if diff < self.stop_threshold:
                    break

            return imf

        except Exception as e:
            self.logger.debug(f"[DSP] Sifting error: {e}")
            return None

    def _find_extrema(self, signal: np.ndarray, extrema_type: str) -> np.ndarray:
        """Find indices of local maxima or minima."""
        try:
            if extrema_type == 'max':
                # Local maxima: signal[i] > signal[i-1] and signal[i] > signal[i+1]
                extrema_idx = []
                for i in range(1, len(signal) - 1):
                    if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                        extrema_idx.append(i)
                return np.array(extrema_idx)
            else:
                # Local minima: signal[i] < signal[i-1] and signal[i] < signal[i+1]
                extrema_idx = []
                for i in range(1, len(signal) - 1):
                    if signal[i] < signal[i-1] and signal[i] < signal[i+1]:
                        extrema_idx.append(i)
                return np.array(extrema_idx)

        except Exception:
            return np.array([])

    def _interpolate_envelope(self, signal: np.ndarray, extrema_idx: np.ndarray) -> np.ndarray:
        """Interpolate envelope through extrema points."""
        try:
            if len(extrema_idx) < 2:
                return np.zeros_like(signal)

            x = extrema_idx
            y = signal[extrema_idx]

            # Linear interpolation
            envelope = np.interp(np.arange(len(signal)), x, y)

            return envelope

        except Exception:
            return np.zeros_like(signal)

    def _is_monotonic(self, signal: np.ndarray) -> bool:
        """Check if signal is monotonic (increasing or decreasing)."""
        try:
            if len(signal) < 2:
                return True

            # Check if all differences have the same sign
            diffs = np.diff(signal)
            return np.all(diffs >= 0) or np.all(diffs <= 0)

        except Exception:
            return True

    # =========================================================================
    # HILBERT TRANSFORM
    # =========================================================================

    def hilbert_phase(self, signal: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate instantaneous phase using Hilbert Transform.
        
        Args:
            signal: Input signal (typically an IMF)
            
        Returns:
            Instantaneous phase array (radians), or None on failure
        """
        if signal is None or len(signal) < 5:
            return None

        try:
            # Handle NaN
            signal = np.nan_to_num(signal, nan=0.0)

            # Calculate analytic signal using Hilbert Transform
            analytic = self._hilbert_transform(signal)

            if analytic is None:
                return None

            # Extract phase
            phase = np.angle(analytic)

            # Unwrap phase to remove discontinuities
            phase = np.unwrap(phase)

            return phase

        except Exception as e:
            self.logger.error(f"[DSP] Hilbert phase error: {e}")
            return None

    def hilbert_frequency(self, signal: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate instantaneous frequency using Hilbert Transform.
        
        Args:
            signal: Input signal (typically an IMF)
            
        Returns:
            Instantaneous frequency array, or None on failure
        """
        if signal is None or len(signal) < 5:
            return None

        try:
            # Handle NaN
            signal = np.nan_to_num(signal, nan=0.0)

            # Calculate analytic signal
            analytic = self._hilbert_transform(signal)

            if analytic is None:
                return None

            # Calculate instantaneous phase
            phase = np.unwrap(np.angle(analytic))

            # Calculate instantaneous frequency (derivative of phase)
            frequency = np.gradient(phase)

            # Normalize to [0, pi] range
            frequency = np.abs(frequency)
            frequency = np.mod(frequency, np.pi)

            return frequency

        except Exception as e:
            self.logger.error(f"[DSP] Hilbert frequency error: {e}")
            return None

    def _hilbert_transform(self, signal: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate analytic signal using Hilbert Transform.
        
        Uses FFT-based implementation.
        
        Returns:
            Analytic signal (complex), or None on failure
        """
        try:
            n = len(signal)

            # FFT of signal
            fft_signal = np.fft.fft(signal)

            # Create Hilbert transform multiplier
            h = np.zeros(n)
            if n > 0:
                h[0] = 1
                if n % 2 == 0:
                    h[n // 2] = 1
                    h[1:n // 2] = 2
                else:
                    h[1:(n + 1) // 2] = 2

            # Apply Hilbert transform
            analytic_fft = fft_signal * h

            # Inverse FFT to get analytic signal
            analytic = np.fft.ifft(analytic_fft)

            return analytic

        except Exception as e:
            self.logger.error(f"[DSP] Hilbert transform error: {e}")
            return None

    # =========================================================================
    # ENVELOPE CALCULATION
    # =========================================================================

    def calculate_envelope(self, signal: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate signal envelope (upper and lower).
        
        Args:
            signal: Input signal
            
        Returns:
            Tuple of (upper_envelope, lower_envelope), or None on failure
        """
        if signal is None or len(signal) < 10:
            return None

        try:
            # Handle NaN
            signal = np.nan_to_num(signal, nan=np.nanmean(signal))

            # Find extrema
            maxima_idx = self._find_extrema(signal, 'max')
            minima_idx = self._find_extrema(signal, 'min')

            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                # Fallback: use rolling max/min
                window = min(10, len(signal) // 2)
                upper = pd.Series(signal).rolling(window, center=True).max().values
                lower = pd.Series(signal).rolling(window, center=True).min().values
                return upper, lower

            # Interpolate envelopes
            upper_env = self._interpolate_envelope(signal, maxima_idx)
            lower_env = self._interpolate_envelope(signal, minima_idx)

            return upper_env, lower_env

        except Exception as e:
            self.logger.error(f"[DSP] Envelope calculation error: {e}")
            return None

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def extract_trend(self, signal: np.ndarray, max_imfs: int = 3) -> Optional[np.ndarray]:
        """
        Extract trend from signal by removing high-frequency IMFs.
        
        Args:
            signal: Input signal
            max_imfs: Number of IMFs to remove (high frequency)
            
        Returns:
            Trend signal, or None on failure
        """
        imfs = self.empirical_mode_decomposition(signal, max_imfs)

        if imfs is None or len(imfs) == 0:
            return None

        # Trend is the sum of remaining IMFs (excluding high-frequency ones)
        trend = np.sum(imfs[max_imfs:], axis=0) if len(imfs) > max_imfs else imfs[-1]

        return trend

    def get_dominant_frequency(self, signal: np.ndarray) -> Optional[float]:
        """
        Get dominant frequency from signal using FFT.
        
        Args:
            signal: Input signal
            
        Returns:
            Dominant frequency (cycles per sample), or None on failure
        """
        if signal is None or len(signal) < 10:
            return None

        try:
            # Handle NaN
            signal = np.nan_to_num(signal, nan=np.nanmean(signal))

            # Remove trend
            signal = signal - np.mean(signal)

            # FFT
            fft_result = np.fft.rfft(signal)
            fft_magnitude = np.abs(fft_result)

            # Find dominant frequency (excluding DC component)
            if len(fft_magnitude) > 1:
                dominant_idx = np.argmax(fft_magnitude[1:]) + 1
                dominant_freq = dominant_idx / len(signal)
                return dominant_freq

            return None

        except Exception as e:
            self.logger.error(f"[DSP] Dominant frequency error: {e}")
            return None

    def format_dsp_log(self, imfs: List[np.ndarray], trend: np.ndarray = None) -> str:
        """
        Format DSP analysis result as concise log string.
        
        Args:
            imfs: List of IMFs
            trend: Trend component
            
        Returns:
            Formatted log string
        """
        if imfs is None:
            return "[DSP] No IMFs extracted"

        imf_count = len(imfs)
        trend_str = "with trend" if trend is not None else "no trend"

        return f"[DSP] {imf_count} IMFs extracted {trend_str}"