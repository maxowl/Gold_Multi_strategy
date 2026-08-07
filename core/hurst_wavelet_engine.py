"""
Hurst Exponent & Wavelet Analysis Engine.

Provides fractal analysis and multi-resolution wavelet decomposition
for market regime classification.

Hurst Exponent Interpretation:
  H > 0.5: Trending market (persistent)
  H = 0.5: Random walk
  H < 0.5: Mean-reverting market (anti-persistent)

Wavelet Analysis:
  Decomposes signal into multiple frequency bands
  for multi-resolution analysis.

Used by:
  - S25_HurstWavelet (Hurst + Wavelet strategy)
  - Regime classification
  - Market structure analysis
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple


class HurstWaveletEngine:
    """
    Hurst Exponent and Wavelet Analysis engine.
    
    Features:
      - Hurst exponent calculation (R/S analysis)
      - Fractal dimension calculation
      - Wavelet decomposition (Haar wavelet)
      - Wavelet energy analysis
      - Regime classification
    """

    def __init__(self):
        """Initialize HurstWaveletEngine."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Default parameters
        self.hurst_window = 100  # Window size for Hurst calculation
        self.wavelet_levels = 4  # Number of wavelet decomposition levels

    # =========================================================================
    # HURST EXPONENT CALCULATION
    # =========================================================================

    def calculate_hurst_exponent(
        self, series: np.ndarray, max_lag: int = None
    ) -> Optional[float]:
        """
        Calculate Hurst exponent using Rescaled Range (R/S) analysis.
        
        H > 0.5: Trending (persistent)
        H = 0.5: Random walk
        H < 0.5: Mean-reverting (anti-persistent)
        
        Args:
            series: Time series data
            max_lag: Maximum lag for R/S calculation
            
        Returns:
            Hurst exponent (0-1), or None on failure
        """
        if series is None or len(series) < 20:
            return None

        try:
            # Handle NaN
            series = np.nan_to_num(series, nan=np.nanmean(series))

            n = len(series)
            if max_lag is None:
                max_lag = min(n // 2, 50)

            if max_lag < 10:
                return None

            # Calculate log returns for better stationarity
            returns = np.diff(np.log(series + 1e-10))

            if len(returns) < 20:
                return None

            # R/S analysis for multiple lags
            lags = range(10, max_lag)
            rs_values = []

            for lag in lags:
                rs = self._calculate_rs(returns, lag)
                if rs is not None and rs > 0:
                    rs_values.append((lag, rs))

            if len(rs_values) < 3:
                return None

            # Linear regression: log(R/S) = H * log(lag) + c
            log_lags = np.log([x[0] for x in rs_values])
            log_rs = np.log([x[1] for x in rs_values])

            # Remove NaN values
            valid_mask = ~np.isnan(log_lags) & ~np.isnan(log_rs)
            if np.sum(valid_mask) < 3:
                return None

            log_lags = log_lags[valid_mask]
            log_rs = log_rs[valid_mask]

            # Linear regression
            slope, _ = np.polyfit(log_lags, log_rs, 1)

            # Clamp to [0, 1]
            hurst = max(0.0, min(1.0, slope))

            return float(hurst)

        except Exception as e:
            self.logger.error(f"[HURST] Hurst calculation error: {e}")
            return None

    def _calculate_rs(self, series: np.ndarray, lag: int) -> Optional[float]:
        """
        Calculate Rescaled Range (R/S) for a given lag.
        
        Args:
            series: Time series data
            lag: Lag for calculation
            
        Returns:
            R/S value, or None on failure
        """
        try:
            n = len(series)
            if n < lag:
                return None

            # Split into chunks
            num_chunks = n // lag
            if num_chunks == 0:
                return None

            rs_values = []

            for i in range(num_chunks):
                chunk = series[i * lag:(i + 1) * lag]

                if len(chunk) < lag:
                    continue

                # Mean
                mean = np.mean(chunk)

                # Deviations from mean
                deviations = chunk - mean

                # Cumulative sum
                cumsum = np.cumsum(deviations)

                # Range
                range_val = np.max(cumsum) - np.min(cumsum)

                # Standard deviation
                std_val = np.std(chunk)

                if std_val > 0:
                    rs_values.append(range_val / std_val)

            if not rs_values:
                return None

            return float(np.mean(rs_values))

        except Exception:
            return None

    # =========================================================================
    # FRACTAL DIMENSION
    # =========================================================================

    def calculate_fractal_dimension(
        self, series: np.ndarray, box_sizes: List[int] = None
    ) -> Optional[float]:
        """
        Calculate fractal dimension using box-counting method.
        
        Fractal dimension D:
          D ≈ 1: Smooth, trending
          D ≈ 1.5: Random walk
          D ≈ 2: Rough, mean-reverting
        
        Args:
            series: Time series data
            box_sizes: List of box sizes for counting
            
        Returns:
            Fractal dimension (1-2), or None on failure
        """
        if series is None or len(series) < 20:
            return None

        try:
            # Handle NaN
            series = np.nan_to_num(series, nan=np.nanmean(series))

            # Normalize series to [0, 1]
            series_min = np.min(series)
            series_max = np.max(series)
            if series_max == series_min:
                return None

            normalized = (series - series_min) / (series_max - series_min)

            if box_sizes is None:
                box_sizes = [2, 4, 8, 16, 32]

            # Filter box sizes that are too large
            box_sizes = [bs for bs in box_sizes if bs < len(normalized)]

            if len(box_sizes) < 3:
                return None

            # Count boxes for each size
            box_counts = []
            for box_size in box_sizes:
                count = self._count_boxes(normalized, box_size)
                if count > 0:
                    box_counts.append((box_size, count))

            if len(box_counts) < 3:
                return None

            # Linear regression: log(N) = -D * log(1/box_size) + c
            log_sizes = np.log([1.0 / x[0] for x in box_counts])
            log_counts = np.log([x[1] for x in box_counts])

            # Remove NaN values
            valid_mask = ~np.isnan(log_sizes) & ~np.isnan(log_counts)
            if np.sum(valid_mask) < 3:
                return None

            log_sizes = log_sizes[valid_mask]
            log_counts = log_counts[valid_mask]

            # Linear regression
            slope, _ = np.polyfit(log_sizes, log_counts, 1)

            # Fractal dimension is the slope
            fractal_dim = max(1.0, min(2.0, slope))

            return float(fractal_dim)

        except Exception as e:
            self.logger.error(f"[HURST] Fractal dimension error: {e}")
            return None

    def _count_boxes(self, series: np.ndarray, box_size: int) -> int:
        """Count number of boxes needed to cover the series."""
        try:
            n = len(series)
            num_boxes_x = n // box_size

            if num_boxes_x == 0:
                return 0

            # Divide y-axis into boxes
            series_range = np.max(series) - np.min(series)
            num_boxes_y = int(series_range * num_boxes_x) + 1

            count = 0
            for i in range(num_boxes_x):
                start_idx = i * box_size
                end_idx = min((i + 1) * box_size, n)

                chunk = series[start_idx:end_idx]
                if len(chunk) == 0:
                    continue

                chunk_min = np.min(chunk)
                chunk_max = np.max(chunk)

                # Count boxes needed for this chunk
                boxes_in_chunk = int((chunk_max - chunk_min) * num_boxes_x) + 1
                count += boxes_in_chunk

            return count

        except Exception:
            return 0

    # =========================================================================
    # WAVELET DECOMPOSITION
    # =========================================================================

    def wavelet_decomposition(
        self, signal: np.ndarray, levels: int = None, wavelet: str = 'haar'
    ) -> Optional[Dict]:
        """
        Decompose signal using wavelet transform.
        
        Args:
            signal: Input signal
            levels: Number of decomposition levels
            wavelet: Wavelet type ('haar', 'db2', 'db4')
            
        Returns:
            Dict with 'approximations' and 'details' lists, or None on failure
        """
        if signal is None or len(signal) < 8:
            return None

        try:
            # Handle NaN
            signal = np.nan_to_num(signal, nan=np.nanmean(signal))

            if levels is None:
                levels = self.wavelet_levels

            # Limit levels based on signal length
            max_levels = int(np.log2(len(signal))) - 1
            levels = min(levels, max_levels)

            if levels < 1:
                return None

            approximations = []
            details = []

            current_signal = signal.copy()

            for level in range(levels):
                if len(current_signal) < 4:
                    break

                # Apply wavelet transform
                approx, detail = self._wavelet_transform(current_signal, wavelet)

                approximations.append(approx)
                details.append(detail)

                # Continue with approximation
                current_signal = approx

            return {
                'approximations': approximations,
                'details': details,
                'levels': len(approximations),
                'wavelet': wavelet
            }

        except Exception as e:
            self.logger.error(f"[WAVELET] Decomposition error: {e}")
            return None

    def _wavelet_transform(
        self, signal: np.ndarray, wavelet: str = 'haar'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply one level of wavelet transform.
        
        Args:
            signal: Input signal
            wavelet: Wavelet type
            
        Returns:
            Tuple of (approximation, detail)
        """
        n = len(signal)

        if n < 2:
            return signal, np.zeros_like(signal)

        if wavelet == 'haar':
            # Haar wavelet: simple averaging and differencing
            approx = np.zeros(n // 2)
            detail = np.zeros(n // 2)

            for i in range(n // 2):
                approx[i] = (signal[2 * i] + signal[2 * i + 1]) / np.sqrt(2)
                detail[i] = (signal[2 * i] - signal[2 * i + 1]) / np.sqrt(2)

            return approx, detail

        else:
            # Default to Haar for simplicity
            return self._wavelet_transform(signal, 'haar')

    # =========================================================================
    # WAVELET ENERGY
    # =========================================================================

    def calculate_wavelet_energy(
        self, signal: np.ndarray, levels: int = None
    ) -> Optional[Dict]:
        """
        Calculate energy distribution across wavelet levels.
        
        Args:
            signal: Input signal
            levels: Number of decomposition levels
            
        Returns:
            Dict with energy per level, or None on failure
        """
        decomposition = self.wavelet_decomposition(signal, levels)

        if decomposition is None:
            return None

        try:
            energies = []

            # Energy in approximation
            if decomposition['approximations']:
                last_approx = decomposition['approximations'][-1]
                approx_energy = np.sum(last_approx ** 2)
                energies.append(('approximation', float(approx_energy)))

            # Energy in details
            for i, detail in enumerate(decomposition['details']):
                detail_energy = np.sum(detail ** 2)
                energies.append((f'detail_{i+1}', float(detail_energy)))

            # Calculate total energy
            total_energy = sum(e[1] for e in energies)

            # Calculate percentages
            energy_percentages = {}
            for name, energy in energies:
                if total_energy > 0:
                    energy_percentages[name] = energy / total_energy * 100
                else:
                    energy_percentages[name] = 0.0

            return {
                'energies': dict(energies),
                'total_energy': float(total_energy),
                'percentages': energy_percentages,
                'dominant_level': max(energy_percentages.keys(), key=lambda k: energy_percentages[k])
            }

        except Exception as e:
            self.logger.error(f"[WAVELET] Energy calculation error: {e}")
            return None

    # =========================================================================
    # REGIME CLASSIFICATION
    # =========================================================================

    def classify_regime(self, series: np.ndarray) -> Dict:
        """
        Classify market regime using Hurst exponent and wavelet analysis.
        
        Args:
            series: Price series
            
        Returns:
            Dict with regime classification
        """
        result = {
            'hurst_exponent': None,
            'fractal_dimension': None,
            'regime': 'UNKNOWN',
            'confidence': 0.0,
            'wavelet_energy': None
        }

        if series is None or len(series) < 50:
            return result

        try:
            # Calculate Hurst exponent
            hurst = self.calculate_hurst_exponent(series)
            result['hurst_exponent'] = hurst

            # Calculate fractal dimension
            fractal_dim = self.calculate_fractal_dimension(series)
            result['fractal_dimension'] = fractal_dim

            # Calculate wavelet energy
            energy = self.calculate_wavelet_energy(series)
            result['wavelet_energy'] = energy

            # Classify regime based on Hurst
            if hurst is not None:
                if hurst > 0.65:
                    result['regime'] = 'STRONG_TREND'
                    result['confidence'] = min(1.0, (hurst - 0.5) * 4)
                elif hurst > 0.55:
                    result['regime'] = 'MILD_TREND'
                    result['confidence'] = min(1.0, (hurst - 0.5) * 4)
                elif hurst > 0.45:
                    result['regime'] = 'RANDOM_WALK'
                    result['confidence'] = 0.5
                elif hurst > 0.35:
                    result['regime'] = 'MILD_MEAN_REVERSION'
                    result['confidence'] = min(1.0, (0.5 - hurst) * 4)
                else:
                    result['regime'] = 'STRONG_MEAN_REVERSION'
                    result['confidence'] = min(1.0, (0.5 - hurst) * 4)

            return result

        except Exception as e:
            self.logger.error(f"[HURST] Regime classification error: {e}")
            return result

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_analysis_summary(self, series: np.ndarray) -> Dict:
        """
        Get comprehensive analysis summary.
        
        Args:
            series: Price series
            
        Returns:
            Dict with complete analysis
        """
        result = {
            'hurst': self.calculate_hurst_exponent(series),
            'fractal_dimension': self.calculate_fractal_dimension(series),
            'regime': self.classify_regime(series),
            'wavelet_energy': self.calculate_wavelet_energy(series)
        }

        return result

    def format_hurst_log(self, hurst: float, fractal_dim: float = None,
                          regime: str = None) -> str:
        """
        Format Hurst analysis result as concise log string.
        
        Args:
            hurst: Hurst exponent
            fractal_dim: Fractal dimension
            regime: Regime classification
            
        Returns:
            Formatted log string
        """
        if hurst is None:
            return "[HURST] Calculation failed"

        # Interpret Hurst
        if hurst > 0.6:
            interpretation = "STRONG TREND"
        elif hurst > 0.5:
            interpretation = "TRENDING"
        elif hurst > 0.4:
            interpretation = "MEAN-REVERTING"
        else:
            interpretation = "STRONG MEAN-REVERSION"

        fractal_str = f" | FD: {fractal_dim:.2f}" if fractal_dim else ""
        regime_str = f" | Regime: {regime}" if regime else ""

        return (
            f"[HURST] H: {hurst:.3f} ({interpretation}){fractal_str}{regime_str}"
        )