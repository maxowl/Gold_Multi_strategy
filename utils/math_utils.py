"""
Mathematical Utilities.
Provides optimized mathematical functions for quantitative analysis.
Designed to be lightweight and avoid heavy external dependencies.
"""
import numpy as np
import pandas as pd
from typing import Union

def fast_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average efficiently."""
    if series is None or series.empty:
        return pd.Series(dtype=float)
    return series.ewm(span=span, adjust=False).mean()

def calculate_rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling Z-Score with NaN and Zero-Division protection."""
    if series is None or len(series) < window:
        return pd.Series(dtype=float)
    
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    
    # Prevent division by zero
    std_safe = np.where(std == 0, 1e-10, std)
    zscore = (series - mean) / std_safe
    
    return pd.Series(zscore, index=series.index)

def normalize_array(arr: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """Min-Max normalize an array to [0, 1] range with NaN handling."""
    if isinstance(arr, pd.Series):
        arr = arr.to_numpy()
        
    arr = arr.astype(float)
    valid_mask = ~np.isnan(arr)
    
    if not np.any(valid_mask):
        return np.zeros_like(arr)
        
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    
    if max_val == min_val:
        return np.zeros_like(arr)
        
    normalized = np.zeros_like(arr)
    normalized[valid_mask] = (arr[valid_mask] - min_val) / (max_val - min_val)
    return normalized

def calculate_half_life(series: pd.Series) -> float:
    """
    Calculate the half-life of a mean-reverting time series.
    Used in Statistical Arbitrage and Ornstein-Uhlenbeck processes.
    """
    if series is None or len(series) < 20:
        return 0.0
        
    lag = series.shift(1).dropna()
    diff = series.diff().dropna()
    
    # Align indices
    common_idx = lag.index.intersection(diff.index)
    lag = lag.loc[common_idx]
    diff = diff.loc[common_idx]
    
    if len(lag) == 0:
        return 0.0
        
    # Linear regression: diff = beta * lag + epsilon
    # beta = cov(diff, lag) / var(lag)
    cov_matrix = np.cov(diff.to_numpy(), lag.to_numpy())
    var_lag = cov_matrix[1, 1]
    
    if var_lag == 0:
        return 0.0
        
    beta = cov_matrix[0, 1] / var_lag
    
    if beta >= 0:
        return 0.0 # Not mean reverting
        
    half_life = -np.log(2) / beta
    return float(half_life)