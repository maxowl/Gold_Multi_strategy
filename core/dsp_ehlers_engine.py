"""
Ehlers DSP Engine.
Provides MESA (MAMA/FAMA), Homodyne Discriminator, and Digital Vector Oscillator.
Used by S10 (MESA), S18 (Vector), and other Ehlers-based strategies.
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple


class EhlersDSPEngine:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def ehlers_mesa(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Ehlers MESA: MAMA (Mesa Adaptive Moving Average) and FAMA (Following Adaptive MA).
        Used for trend detection and crossover signals.
        """
        if series is None or len(series) < 30:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        
        x = series.to_numpy().astype(float)
        
        try:
            # Smoothed price (weighted average)
            smooth = np.zeros(len(x))
            for i in range(6, len(x)):
                smooth[i] = (
                    4 * x[i] + 3 * x[i-1] + 2 * x[i-2] + x[i-3]
                ) / 10.0
            
            # In-phase and quadrature components
            in_phase = np.zeros(len(x))
            quadrature = np.zeros(len(x))
            
            for i in range(7, len(x)):
                in_phase[i] = (smooth[i] - smooth[i-7]) / 2.0
                quadrature[i] = smooth[i-3] - smooth[i-4]
            
            # Period measurement
            period = np.zeros(len(x))
            for i in range(9, len(x)):
                real_part = in_phase[i] * in_phase[i-1] + quadrature[i] * quadrature[i-1]
                imag_part = in_phase[i] * quadrature[i-1] - in_phase[i-1] * quadrature[i]
                
                # Add NaN/infinity check to prevent arctan errors
                if abs(real_part) > 0.001 and np.isfinite(real_part) and np.isfinite(imag_part):
                    ratio = imag_part / real_part
                    if np.isfinite(ratio):
                        period[i] = 2 * np.pi / np.arctan(ratio)
                    else:
                        period[i] = period[i-1]
                else:
                    period[i] = period[i-1]
                
                # Smooth period
                if i >= 10:
                    period[i] = 0.33 * period[i] + 0.67 * period[i-1]
                
                # Clamp period to reasonable range
                period[i] = max(6, min(50, period[i]))
            
            # Alpha calculation
            alpha = np.zeros(len(x))
            for i in range(len(x)):
                if period[i] > 0:
                    alpha[i] = 2.0 / (period[i] + 1)
                else:
                    alpha[i] = 0.05
            
            # MAMA (fast adaptive MA)
            mama = np.zeros(len(x))
            for i in range(1, len(x)):
                mama[i] = alpha[i] * x[i] + (1 - alpha[i]) * mama[i-1]
            
            # FAMA (slow adaptive MA)
            fama = np.zeros(len(x))
            for i in range(1, len(x)):
                fama[i] = 0.5 * alpha[i] * mama[i] + (1 - 0.5 * alpha[i]) * fama[i-1]
            
            return pd.Series(mama, index=series.index), pd.Series(fama, index=series.index)
            
        except Exception as e:
            self.logger.error(f"[FAIL] MESA calculation error: {e}")
            return pd.Series(dtype=float), pd.Series(dtype=float)

    def homodyne_discriminator(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """
        Ehlers Homodyne Discriminator: Estimates dominant cycle period.
        Returns period in bars.
        """
        if high is None or low is None or len(high) < 30:
            return pd.Series(dtype=float)
        
        h = high.to_numpy().astype(float)
        l = low.to_numpy().astype(float)
        
        try:
            # Price midpoint
            mid = (h + l) / 2.0
            
            # Smoothed price
            smooth = np.zeros(len(mid))
            for i in range(6, len(mid)):
                smooth[i] = (
                    4 * mid[i] + 3 * mid[i-1] + 2 * mid[i-2] + mid[i-3]
                ) / 10.0
            
            # In-phase and quadrature
            in_phase = np.zeros(len(smooth))
            quadrature = np.zeros(len(smooth))
            
            for i in range(7, len(smooth)):
                in_phase[i] = (smooth[i] - smooth[i-7]) / 2.0
                quadrature[i] = smooth[i-3] - smooth[i-4]
            
            # Period calculation
            period = np.zeros(len(smooth))
            for i in range(9, len(smooth)):
                real_part = in_phase[i] * in_phase[i-1] + quadrature[i] * quadrature[i-1]
                imag_part = in_phase[i] * quadrature[i-1] - in_phase[i-1] * quadrature[i]
                
                if abs(real_part) > 0.001 and np.isfinite(real_part) and np.isfinite(imag_part):
                    ratio = imag_part / real_part
                    if np.isfinite(ratio):
                        period[i] = 2 * np.pi / np.arctan(ratio)
                    else:
                        period[i] = period[i-1]
                else:
                    period[i] = period[i-1]
                
                if i >= 10:
                    period[i] = 0.33 * period[i] + 0.67 * period[i-1]
                
                period[i] = max(6, min(50, period[i]))
            
            return pd.Series(period, index=high.index)
            
        except Exception as e:
            self.logger.error(f"[FAIL] Homodyne calculation error: {e}")
            return pd.Series(dtype=float)

    def digital_vector_oscillator(
        self, series: pd.Series, period: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Digital Vector Oscillator: Measures momentum strength.
        Returns (vector, signal_line).
        """
        if series is None or period is None or len(series) < 20:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        
        x = series.to_numpy().astype(float)
        p = period.to_numpy().astype(float)
        
        try:
            # Calculate vector based on dominant period
            vector = np.zeros(len(x))
            
            for i in range(1, len(x)):
                lookback = int(p[i]) if not np.isnan(p[i]) and p[i] > 0 else 14
                lookback = max(2, min(50, lookback))
                
                if i >= lookback:
                    # Price change over dominant period
                    price_change = x[i] - x[i - lookback]
                    
                    # Normalize by ATR-like measure
                    range_sum = 0
                    for j in range(i - lookback + 1, i + 1):
                        range_sum += abs(x[j] - x[j-1])
                    
                    if range_sum > 0:
                        vector[i] = price_change / range_sum * 100
            
            # Signal line (EMA of vector)
            signal = np.zeros(len(vector))
            signal[0] = vector[0]
            
            for i in range(1, len(vector)):
                signal[i] = 0.2 * vector[i] + 0.8 * signal[i-1]
            
            return pd.Series(vector, index=series.index), pd.Series(signal, index=series.index)
            
        except Exception as e:
            self.logger.error(f"[FAIL] Vector oscillator error: {e}")
            return pd.Series(dtype=float), pd.Series(dtype=float)