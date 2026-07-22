"""
Hurst Exponent & Wavelet Engine.
Provides Hurst exponent calculation and wavelet denoising.
Used by S25 (Hurst Wavelet) strategy.
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional


class HurstWaveletEngine:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_hurst_exponent(self, series: pd.Series, max_lag: int = 50) -> float:
        """
        Calculate Hurst Exponent using Rescaled Range (R/S) Analysis.
        H > 0.5: Trending (persistent)
        H = 0.5: Random walk
        H < 0.5: Mean-reverting (anti-persistent)
        """
        if series is None or len(series) < max_lag * 2:
            return 0.5
        
        try:
            x = series.to_numpy().astype(float)
            x = x[~np.isnan(x)]
            
            if len(x) < max_lag * 2:
                return 0.5
            
            lags = range(2, min(max_lag, len(x) // 2))
            tau = []
            
            for lag in lags:
                diffs = x[lag:] - x[:-lag]
                std = np.std(diffs)
                if std > 0:
                    tau.append((lag, std))
            
            if len(tau) < 3:
                return 0.5
            
            log_lags = np.log([t[0] for t in tau])
            log_tau = np.log([t[1] for t in tau])
            
            # Linear regression
            slope, _ = np.polyfit(log_lags, log_tau, 1)
            
            return max(0.0, min(1.0, slope))
            
        except Exception as e:
            self.logger.error(f"[FAIL] Hurst calculation error: {e}")
            return 0.5

    def wavelet_denoise(self, series: pd.Series, wavelet: str = 'db4', level: int = 3) -> pd.Series:
        """
        Apply wavelet denoising using PyWavelets.
        Removes high-frequency noise while preserving trend.
        """
        if series is None or len(series) < 20:
            return pd.Series(dtype=float)
        
        try:
            import pywt
            
            x = series.to_numpy().astype(float)
            x = x[~np.isnan(x)]
            
            if len(x) < 20:
                return pd.Series(dtype=float)
            
            # Wavelet decomposition
            coeffs = pywt.wavedec(x, wavelet, level=level)
            
            # Threshold detail coefficients (soft thresholding)
            threshold = np.median(np.abs(coeffs[-1])) / 0.6745 * np.sqrt(2 * np.log(len(x)))
            
            # Apply threshold to detail coefficients
            denoised_coeffs = [coeffs[0]]  # Keep approximation
            for i in range(1, len(coeffs)):
                denoised_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
            
            # Reconstruct signal
            denoised = pywt.waverec(denoised_coeffs, wavelet)
            
            # Match original length
            if len(denoised) > len(series):
                denoised = denoised[:len(series)]
            elif len(denoised) < len(series):
                denoised = np.pad(denoised, (0, len(series) - len(denoised)), constant_values=np.nan)
            
            return pd.Series(denoised, index=series.index)
            
        except ImportError:
            self.logger.warning("[WARN] PyWavelets not installed, returning original series")
            return series
        except Exception as e:
            self.logger.error(f"[FAIL] Wavelet denoise error: {e}")
            return pd.Series(dtype=float)

    def calculate_wavelet_slope(self, series: pd.Series, window: int = 5) -> float:
        """
        Calculate slope of wavelet-denoised series.
        Positive slope = uptrend, Negative = downtrend.
        """
        if series is None or len(series) < window:
            return 0.0
        
        try:
            x = series.to_numpy().astype(float)
            recent = x[-window:]
            
            # Drop NaNs for polyfit
            recent = recent[~np.isnan(recent)]
            if len(recent) < 2:
                return 0.0
            
            # Linear regression slope
            x_vals = np.arange(len(recent))
            slope, _ = np.polyfit(x_vals, recent, 1)
            
            return float(slope)
            
        except Exception as e:
            self.logger.error(f"[FAIL] Wavelet slope error: {e}")
            return 0.0