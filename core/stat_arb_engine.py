"""
Statistical Arbitrage Engine.
Provides Z-Score mean reversion analysis and cointegration tests.
Used by S15_HFT_StatArb strategy.
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple


class StatArbEngine:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_z_score(self, series: pd.Series, lookback: int = 100) -> Optional[pd.Series]:
        """
        Calculate rolling Z-Score of a price series.
        Z = (Price - Mean) / StdDev
        
        Used for mean reversion entry signals (Z > 2 or Z < -2).
        """
        if series is None or len(series) < lookback:
            return None
        
        try:
            x = series.to_numpy().astype(float)
            
            mean = pd.Series(x).rolling(window=lookback, min_periods=lookback).mean()
            std = pd.Series(x).rolling(window=lookback, min_periods=lookback).std()
            
            # Prevent division by zero
            std_safe = std.replace(0, 1e-10)
            
            z_score = (pd.Series(x) - mean) / std_safe
            
            return z_score
            
        except Exception as e:
            self.logger.error(f"[FAIL] Z-Score calculation error: {e}")
            return None

    def detect_mean_reversion_opportunity(self, series: pd.Series,
                                          lookback: int = 100,
                                          z_threshold: float = 2.0) -> dict:
        """
        Detect mean reversion opportunity based on Z-Score extremes.
        
        Returns dict with:
        - 'opportunity': bool
        - 'direction': 'BUY', 'SELL', or None
        - 'z_score': current Z-Score value
        """
        z_score = self.calculate_z_score(series, lookback)
        
        # [FIX] Check for None AND empty series to prevent IndexError
        if z_score is None or len(z_score) == 0:
            return {'opportunity': False, 'direction': None, 'z_score': 0.0}
        
        current_z = z_score.iloc[-1]
        
        # Handle NaN in current Z-Score
        if np.isnan(current_z):
            return {'opportunity': False, 'direction': None, 'z_score': 0.0}
        
        if current_z < -z_threshold:
            # Oversold - BUY signal
            return {'opportunity': True, 'direction': 'BUY', 'z_score': float(current_z)}
        elif current_z > z_threshold:
            # Overbought - SELL signal
            return {'opportunity': True, 'direction': 'SELL', 'z_score': float(current_z)}
        else:
            # No opportunity
            return {'opportunity': False, 'direction': None, 'z_score': float(current_z)}

    def calculate_half_life(self, series: pd.Series) -> float:
        """
        Calculate the half-life of mean reversion using Ornstein-Uhlenbeck process.
        
        Used to determine how quickly a series reverts to its mean.
        Shorter half-life = faster mean reversion.
        
        Returns half-life in bars.
        """
        if series is None or len(series) < 30:
            return 0.0
        
        try:
            x = series.to_numpy().astype(float)
            x = x[~np.isnan(x)]
            
            if len(x) < 30:
                return 0.0
            
            # Calculate lagged series and differences
            lag = pd.Series(x).shift(1).dropna()
            diff = pd.Series(x).diff().dropna()
            
            # Align indices
            common_idx = lag.index.intersection(diff.index)
            lag_aligned = lag.loc[common_idx]
            diff_aligned = diff.loc[common_idx]
            
            if len(lag_aligned) < 10:
                return 0.0
            
            # Linear regression: diff = beta * lag + epsilon
            # Using numpy for speed
            lag_arr = lag_aligned.to_numpy()
            diff_arr = diff_aligned.to_numpy()
            
            # beta = cov(diff, lag) / var(lag)
            cov_matrix = np.cov(diff_arr, lag_arr)
            var_lag = cov_matrix[1, 1]
            
            if var_lag == 0:
                return 0.0
            
            beta = cov_matrix[0, 1] / var_lag
            
            # If beta >= 0, series is not mean reverting
            if beta >= 0:
                return 0.0
            
            # Half-life = -ln(2) / beta
            half_life = -np.log(2) / beta
            
            return float(half_life)
            
        except Exception as e:
            self.logger.error(f"[FAIL] Half-life calculation error: {e}")
            return 0.0

    def calculate_hurst_exponent(self, series: pd.Series, max_lag: int = 50) -> float:
        """
        Calculate Hurst Exponent using Rescaled Range (R/S) Analysis.
        H > 0.5: Trending (persistent)
        H = 0.5: Random walk
        H < 0.5: Mean-reverting (anti-persistent)
        
        Duplicate of HurstWaveletEngine method, kept here for stat arb context.
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
            
            slope, _ = np.polyfit(log_lags, log_tau, 1)
            
            return max(0.0, min(1.0, slope))
            
        except Exception as e:
            self.logger.error(f"[FAIL] Hurst calculation error: {e}")
            return 0.5