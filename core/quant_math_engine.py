"""
Quantum Mathematics Engine.
Provides Quantum Probability Density Function (PDF) and Fractal Dimension analysis.
Used by S6_QuantumPDF strategy.
"""
import pandas as pd
import numpy as np
import logging
from scipy import stats
from typing import Optional, Dict


class QuantMathEngine:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_quantum_pdf(self, series: pd.Series, bins: int = 50,
                              lookback: int = 100) -> Optional[Dict]:
        """
        Calculate Quantum Probability Density Function using Kernel Density Estimation.
        
        [FIX] Returns a dict containing both the PDF array and the actual price eval_points
        to prevent index-to-price mapping errors in strategies.
        
        Returns:
            Dict with:
            - 'pdf': numpy array of probability densities
            - 'eval_points': numpy array of actual price levels corresponding to pdf bins
        """
        if series is None or len(series) < lookback:
            return None
        
        x = series.to_numpy()[-lookback:]
        x = x[~np.isnan(x)]
        if len(x) < 20:
            return None
        
        try:
            # Calculate optimal bandwidth using Silverman's rule
            std = np.std(x)
            iqr = stats.iqr(x)
            n = len(x)
            
            if std == 0 or iqr == 0:
                return None
            
            # Bandwidth (h) calculation
            h = 0.9 * min(std, iqr / 1.34) * (n ** (-0.2))
            
            # Evaluation points across the price range
            eval_points = np.linspace(x.min(), x.max(), bins)
            
            # Gaussian KDE calculation
            pdf = np.zeros(bins)
            for i, ep in enumerate(eval_points):
                kernel_vals = np.exp(-0.5 * ((x - ep) / h) ** 2) / (h * np.sqrt(2 * np.pi))
                pdf[i] = np.mean(kernel_vals)
            
            # Normalize PDF so sum = 1
            pdf = pdf / (np.sum(pdf) + 1e-10)
            
            return {'pdf': pdf, 'eval_points': eval_points}
            
        except Exception as e:
            self.logger.error(f"[FAIL] Quantum PDF calculation error: {e}")
            return None

    def find_pdf_peaks(self, pdf_data: Dict, threshold: float = 0.7) -> list:
        """
        Find peaks in PDF above threshold.
        
        [FIX] Accepts the new dict format and uses actual price levels for 'center'
        instead of array indices, preventing price-distance calculation errors.
        
        Returns list of dicts with:
        - 'index': array index of the peak
        - 'center': actual price level of the peak
        - 'probability': probability density value
        """
        # Guard against invalid input
        if pdf_data is None or 'pdf' not in pdf_data or 'eval_points' not in pdf_data:
            return []
            
        pdf = pdf_data['pdf']
        eval_points = pdf_data['eval_points']
        
        if len(pdf) == 0 or len(pdf) != len(eval_points):
            return []
        
        peaks = []
        max_val = np.max(pdf)
        threshold_val = max_val * threshold
        
        # Find local maxima above threshold
        for i in range(1, len(pdf) - 1):
            if pdf[i] > threshold_val and pdf[i] > pdf[i-1] and pdf[i] > pdf[i+1]:
                peaks.append({
                    'index': i, 
                    'center': float(eval_points[i]),  # [FIX] Use actual price level
                    'probability': float(pdf[i])
                })
        
        return peaks

    def calculate_fractal_dimension(self, series: pd.Series, max_lag: int = 20) -> float:
        """
        Calculate Fractal Dimension Index (FDI).
        FDI > 1.5: Mean-reverting (complex, choppy)
        FDI = 1.5: Random walk
        FDI < 1.5: Trending (smooth, persistent)
        """
        if series is None or len(series) < max_lag * 2:
            return 1.5
        
        try:
            x = series.to_numpy().astype(float)
            x = x[~np.isnan(x)]
            
            if len(x) < max_lag * 2:
                return 1.5
            
            # Calculate log length vs log scale
            lags = range(2, min(max_lag, len(x) // 2))
            lengths = []
            
            for lag in lags:
                # Path length at this scale
                diffs = np.abs(x[lag:] - x[:-lag])
                path_length = np.sum(diffs)
                lengths.append((lag, path_length))
            
            if len(lengths) < 3:
                return 1.5
            
            log_lags = np.log([l[0] for l in lengths])
            log_lengths = np.log([l[1] for l in lengths])
            
            # Linear regression slope
            slope, _ = np.polyfit(log_lags, log_lengths, 1)
            
            # FDI = 2 - slope (for 1D time series)
            fdi = 2.0 - slope
            
            return max(1.0, min(2.0, fdi))
            
        except Exception as e:
            self.logger.error(f"[FAIL] Fractal dimension calculation error: {e}")
            return 1.5