"""
Quantitative Math Engine.

Provides advanced mathematical analysis for quantitative trading:
  - Quantum PDF (Probability Density Function) calculation
  - Fractal dimension analysis
  - Statistical measures
  - Entropy calculation
  - Peak detection

Used by:
  - S6_QuantumPDF (Quantum PDF-based strategy)
  - S29_QuantumMomentum (Quantum momentum strategy)
  - Statistical analysis
  - Market structure analysis
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple


class QuantMathEngine:
    """
    Quantitative Math Analysis engine.
    
    Features:
      - Quantum PDF calculation
      - PDF peak detection
      - Fractal dimension calculation
      - Statistical measures
      - Entropy calculation
    """

    def __init__(self):
        """Initialize QuantMathEngine."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Quantum PDF parameters
        self.pdf_bins = 50  # Number of bins for PDF
        self.pdf_lookback = 100  # Lookback for PDF calculation
        self.peak_threshold = 0.7  # Threshold for peak detection

        # Fractal dimension parameters
        self.fractal_lags = [2, 4, 8, 16]  # Lags for fractal calculation

    # =========================================================================
    # QUANTUM PDF CALCULATION
    # =========================================================================

    def calculate_quantum_pdf(
        self, data: np.ndarray, bins: int = None, lookback: int = None
    ) -> Optional[Dict]:
        """
        Calculate Quantum Probability Density Function.
        
        The Quantum PDF represents the probability distribution of price
        over a lookback period, identifying high-probability zones.
        
        Args:
            data: Price series
            bins: Number of bins for PDF
            lookback: Number of bars for calculation
            
        Returns:
            Dict with PDF data, or None on failure
        """
        if bins is None:
            bins = self.pdf_bins
        if lookback is None:
            lookback = self.pdf_lookback

        if data is None or len(data) < 20:
            return None

        try:
            # Handle NaN
            data = np.nan_to_num(data, nan=np.nanmean(data))

            # Use lookback window
            if len(data) > lookback:
                data = data[-lookback:]

            n = len(data)

            # Calculate PDF using histogram
            hist, bin_edges = np.histogram(data, bins=bins, density=True)

            # Calculate bin centers
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Normalize PDF
            pdf_sum = np.sum(hist)
            if pdf_sum > 0:
                pdf_normalized = hist / pdf_sum
            else:
                pdf_normalized = hist

            # Calculate cumulative distribution
            cdf = np.cumsum(pdf_normalized)

            # Find peaks
            peaks = self.find_pdf_peaks(pdf_normalized, bin_centers, self.peak_threshold)

            # Calculate statistics
            mean_price = np.mean(data)
            std_price = np.std(data)
            median_price = np.median(data)

            # Calculate quantiles
            q25 = np.percentile(data, 25)
            q50 = np.percentile(data, 50)
            q75 = np.percentile(data, 75)

            return {
                'pdf': pdf_normalized,
                'bin_edges': bin_edges,
                'bin_centers': bin_centers,
                'cdf': cdf,
                'peaks': peaks,
                'mean': float(mean_price),
                'std': float(std_price),
                'median': float(median_price),
                'quantiles': {
                    'q25': float(q25),
                    'q50': float(q50),
                    'q75': float(q75)
                },
                'bins': bins,
                'lookback': lookback
            }

        except Exception as e:
            self.logger.error(f"[QUANT] PDF calculation error: {e}")
            return None

    def find_pdf_peaks(
        self, pdf: np.ndarray, bin_centers: np.ndarray = None, threshold: float = None
    ) -> List[Dict]:
        """
        Find peaks in PDF.
        
        Peaks represent high-probability price zones.
        
        Args:
            pdf: Probability density values
            bin_centers: Bin center prices
            threshold: Minimum height threshold for peaks
            
        Returns:
            List of peak dicts
        """
        if threshold is None:
            threshold = self.peak_threshold

        if pdf is None or len(pdf) < 5:
            return []

        try:
            peaks = []
            n = len(pdf)

            # Normalize PDF for threshold comparison
            max_pdf = np.max(pdf)
            if max_pdf > 0:
                pdf_normalized = pdf / max_pdf
            else:
                pdf_normalized = pdf

            # Find local maxima above threshold
            for i in range(1, n - 1):
                if (pdf_normalized[i] >= threshold and
                    pdf[i] > pdf[i-1] and
                    pdf[i] > pdf[i+1]):

                    peak = {
                        'index': i,
                        'pdf_value': float(pdf[i]),
                        'normalized_value': float(pdf_normalized[i]),
                        'strength': float(pdf_normalized[i])
                    }

                    # Add price if bin centers available
                    if bin_centers is not None and i < len(bin_centers):
                        peak['price'] = float(bin_centers[i])

                    peaks.append(peak)

            # Sort by strength (descending)
            peaks.sort(key=lambda x: x['strength'], reverse=True)

            return peaks

        except Exception as e:
            self.logger.debug(f"[QUANT] Peak detection error: {e}")
            return []

    # =========================================================================
    # FRACTAL DIMENSION CALCULATION
    # =========================================================================

    def calculate_fractal_dimension(
        self, data: np.ndarray, max_lag: int = None
    ) -> Optional[float]:
        """
        Calculate fractal dimension using box-counting method.
        
        Fractal dimension D:
          D ≈ 1: Smooth, trending
          D ≈ 1.5: Random walk
          D ≈ 2: Rough, mean-reverting
        
        Args:
            data: Price series
            max_lag: Maximum lag for calculation
            
        Returns:
            Fractal dimension (1-2), or None on failure
        """
        if data is None or len(data) < 20:
            return None

        try:
            # Handle NaN
            data = np.nan_to_num(data, nan=np.nanmean(data))

            # Normalize data to [0, 1]
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max == data_min:
                return None

            normalized = (data - data_min) / (data_max - data_min)

            if max_lag is None:
                max_lag = len(normalized) // 2

            # Use R/S analysis for fractal dimension
            rs_values = []
            for lag in self.fractal_lags:
                if lag < max_lag:
                    rs = self._calculate_rs_for_fractal(normalized, lag)
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

            # Fractal dimension = 2 - Hurst exponent
            # Hurst exponent is the slope
            hurst = max(0.0, min(1.0, slope))
            fractal_dim = 2.0 - hurst

            # Clamp to [1, 2]
            fractal_dim = max(1.0, min(2.0, fractal_dim))

            return float(fractal_dim)

        except Exception as e:
            self.logger.error(f"[QUANT] Fractal dimension error: {e}")
            return None

    def _calculate_rs_for_fractal(self, data: np.ndarray, lag: int) -> Optional[float]:
        """Calculate R/S value for fractal dimension."""
        try:
            n = len(data)
            if n < lag:
                return None

            # Split into chunks
            num_chunks = n // lag
            if num_chunks == 0:
                return None

            rs_values = []

            for i in range(num_chunks):
                chunk = data[i * lag:(i + 1) * lag]

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
    # STATISTICAL ANALYSIS
    # =========================================================================

    def calculate_statistics(self, data: np.ndarray) -> Dict:
        """
        Calculate comprehensive statistical measures.
        
        Args:
            data: Data series
            
        Returns:
            Dict with statistical measures
        """
        if data is None or len(data) < 5:
            return {
                'mean': 0.0,
                'std': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'range': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'entropy': 0.0
            }

        try:
            # Handle NaN
            data = np.nan_to_num(data, nan=np.nanmean(data))

            # Basic statistics
            mean = np.mean(data)
            std = np.std(data)
            median = np.median(data)
            min_val = np.min(data)
            max_val = np.max(data)
            range_val = max_val - min_val

            # Skewness
            if std > 0:
                skewness = np.mean(((data - mean) / std) ** 3)
            else:
                skewness = 0.0

            # Kurtosis
            if std > 0:
                kurtosis = np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
            else:
                kurtosis = 0.0

            # Entropy
            entropy = self.calculate_entropy(data)

            return {
                'mean': float(mean),
                'std': float(std),
                'median': float(median),
                'min': float(min_val),
                'max': float(max_val),
                'range': float(range_val),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'entropy': float(entropy)
            }

        except Exception as e:
            self.logger.error(f"[QUANT] Statistics calculation error: {e}")
            return {
                'mean': 0.0,
                'std': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'range': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'entropy': 0.0
            }

    # =========================================================================
    # ENTROPY CALCULATION
    # =========================================================================

    def calculate_entropy(self, data: np.ndarray, bins: int = 20) -> float:
        """
        Calculate Shannon entropy of data.
        
        Entropy measures the uncertainty/randomness in the data.
        High entropy = more random
        Low entropy = more predictable
        
        Args:
            data: Data series
            bins: Number of bins for entropy calculation
            
        Returns:
            Shannon entropy value
        """
        if data is None or len(data) < 10:
            return 0.0

        try:
            # Handle NaN
            data = np.nan_to_num(data, nan=np.nanmean(data))

            # Calculate histogram
            hist, _ = np.histogram(data, bins=bins, density=True)

            # Normalize to probability
            hist_sum = np.sum(hist)
            if hist_sum > 0:
                prob = hist / hist_sum
            else:
                return 0.0

            # Remove zero probabilities
            prob = prob[prob > 0]

            # Calculate entropy
            entropy = -np.sum(prob * np.log2(prob))

            return float(entropy)

        except Exception as e:
            self.logger.debug(f"[QUANT] Entropy calculation error: {e}")
            return 0.0

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_quant_analysis(self, data: np.ndarray) -> Dict:
        """
        Get comprehensive quantitative analysis.
        
        Args:
            data: Data series
            
        Returns:
            Dict with complete quantitative analysis
        """
        result = {
            'pdf': None,
            'fractal_dimension': None,
            'statistics': None,
            'entropy': 0.0
        }

        if data is None or len(data) < 20:
            return result

        try:
            # Calculate PDF
            result['pdf'] = self.calculate_quantum_pdf(data)

            # Calculate fractal dimension
            result['fractal_dimension'] = self.calculate_fractal_dimension(data)

            # Calculate statistics
            result['statistics'] = self.calculate_statistics(data)

            # Calculate entropy
            result['entropy'] = self.calculate_entropy(data)

            return result

        except Exception as e:
            self.logger.error(f"[QUANT] Analysis error: {e}")
            return result

    def format_quant_log(self, quant_result: Dict) -> str:
        """
        Format quantitative analysis result as concise log string.
        
        Args:
            quant_result: Result from get_quant_analysis
            
        Returns:
            Formatted log string
        """
        if quant_result is None:
            return "[QUANT] Analysis failed"

        pdf = quant_result.get('pdf')
        fractal = quant_result.get('fractal_dimension')
        stats = quant_result.get('statistics', {})
        entropy = quant_result.get('entropy', 0)

        pdf_str = "N/A"
        if pdf:
            peaks = pdf.get('peaks', [])
            pdf_str = f"{len(peaks)} peaks"

        fractal_str = f"{fractal:.2f}" if fractal else "N/A"

        return (
            f"[QUANT] PDF: {pdf_str} | "
            f"Fractal: {fractal_str} | "
            f"Entropy: {entropy:.2f} | "
            f"Mean: {stats.get('mean', 0):.2f}"
        )

    def is_high_probability_zone(
        self, price: float, pdf_result: Dict, tolerance: float = 0.5
    ) -> bool:
        """
        Check if price is in a high-probability zone.
        
        Args:
            price: Price to check
            pdf_result: Result from calculate_quantum_pdf
            tolerance: Tolerance for zone check
            
        Returns:
            True if price is in high-probability zone
        """
        if pdf_result is None or price <= 0:
            return False

        try:
            peaks = pdf_result.get('peaks', [])

            for peak in peaks:
                peak_price = peak.get('price', 0)
                if peak_price > 0:
                    distance = abs(price - peak_price) / price
                    if distance <= tolerance / 100:  # Convert to percentage
                        return True

            return False

        except Exception:
            return False