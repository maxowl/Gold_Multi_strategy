"""
Digital Signal Processing Engine.
Provides EMD (Empirical Mode Decomposition), Hilbert Transform, and Roofing Filter.
Used by S3 (EMD-HHT), S16 (Roofing EMD), and other DSP-based strategies.
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, List


class DSPEngine:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def empirical_mode_decomposition(self, series: pd.Series, max_imfs: int = 3) -> pd.Series:
        """
        Simplified EMD: Extract first IMF using iterative sifting.
        Returns first IMF (highest frequency component).
        """
        if series is None or len(series) < 20:
            return pd.Series(dtype=float)
        
        x = series.to_numpy().astype(float)
        x = x[~np.isnan(x)]
        
        if len(x) < 20:
            return pd.Series(dtype=float)
        
        try:
            # Iterative sifting to extract first IMF
            residue = x.copy()
            
            for _ in range(10):  # Max 10 sifting iterations
                # Find local extrema
                maxima = []
                minima = []
                
                for i in range(1, len(residue) - 1):
                    if residue[i] > residue[i-1] and residue[i] > residue[i+1]:
                        maxima.append(i)
                    elif residue[i] < residue[i-1] and residue[i] < residue[i+1]:
                        minima.append(i)
                
                if len(maxima) < 2 or len(minima) < 2:
                    break
                
                # Interpolate upper envelope
                upper_env = np.interp(np.arange(len(residue)), maxima, residue[maxima])
                
                # Interpolate lower envelope
                lower_env = np.interp(np.arange(len(residue)), minima, residue[minima])
                
                # Mean envelope
                mean_env = (upper_env + lower_env) / 2.0
                
                # Subtract mean
                h = residue - mean_env
                
                # Check stopping criterion (SD < 0.3)
                sd = np.sum((h - residue) ** 2) / (np.sum(residue ** 2) + 1e-10)
                
                if sd < 0.3:
                    residue = h
                    break
                
                residue = h
            
            # Validate length before creating Series to prevent ValueError
            if len(residue) > len(series):
                residue = residue[:len(series)]
            elif len(residue) < len(series):
                # Pad with NaN if residue is shorter
                residue = np.pad(residue, (0, len(series) - len(residue)), constant_values=np.nan)
            
            return pd.Series(residue, index=series.index)
            
        except Exception as e:
            self.logger.error(f"[FAIL] EMD calculation error: {e}")
            return pd.Series(dtype=float)

    def hilbert_phase(self, series: pd.Series) -> pd.Series:
        """
        Calculate instantaneous phase using Hilbert Transform.
        Phase ranges from -π to +π.
        """
        if series is None or len(series) < 20:
            return pd.Series(dtype=float)
        
        x = series.to_numpy().astype(float)
        x = x[~np.isnan(x)]
        
        if len(x) < 20:
            return pd.Series(dtype=float)
        
        try:
            # Hilbert Transform via FFT
            n = len(x)
            fft_x = np.fft.fft(x)
            
            # Create analytic signal
            h = np.zeros(n)
            if n % 2 == 0:
                h[0] = h[n // 2] = 1
                h[1:n // 2] = 2
            else:
                h[0] = 1
                h[1:(n + 1) // 2] = 2
            
            analytic = np.fft.ifft(fft_x * h)
            
            # Instantaneous phase
            phase = np.angle(analytic)
            
            # Pad if necessary
            if len(phase) < len(series):
                phase = np.pad(phase, (0, len(series) - len(phase)), constant_values=np.nan)
            elif len(phase) > len(series):
                phase = phase[:len(series)]
                
            return pd.Series(phase, index=series.index)
            
        except Exception as e:
            self.logger.error(f"[FAIL] Hilbert phase calculation error: {e}")
            return pd.Series(dtype=float)

    def roofing_filter(self, series: pd.Series) -> pd.Series:
        """
        Ehlers Roofing Filter: Removes trend and extracts cycles.
        High-pass filter that removes periods > 48 bars.
        """
        if series is None or len(series) < 20:
            return pd.Series(dtype=float)
        
        x = series.to_numpy().astype(float)
        
        try:
            # Roofing filter coefficients
            # High-pass filter: removes trend (period > 48 bars)
            alpha = 0.045  # Cutoff frequency parameter
            
            filtered = np.zeros(len(x))
            
            for i in range(2, len(x)):
                if np.isnan(x[i]) or np.isnan(x[i-1]) or np.isnan(x[i-2]):
                    filtered[i] = filtered[i-1] if i > 0 else 0.0
                    continue
                    
                filtered[i] = (
                    (1 - alpha / 2) ** 2 * (x[i] - 2 * x[i-1] + x[i-2]) +
                    2 * (1 - alpha) * filtered[i-1] -
                    (1 - alpha) ** 2 * filtered[i-2]
                )
            
            return pd.Series(filtered, index=series.index)
            
        except Exception as e:
            self.logger.error(f"[FAIL] Roofing filter error: {e}")
            return pd.Series(dtype=float)