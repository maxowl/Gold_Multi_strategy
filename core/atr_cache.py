"""
ATR Cache Manager.
Provides class-level caching for ATR calculations to prevent redundant computations.
"""
import pandas as pd
import numpy as np
from typing import ClassVar, Dict, Tuple


class ATRCache:
    """Class-level cache for ATR values."""
    # Cache key structure: (dataframe_id, period, length)
    _cache: ClassVar[Dict[Tuple[int, int, int], pd.Series]] = {}
    
    @classmethod
    def get_atr(cls, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate ATR with caching.
        Cache key is based on DataFrame id, period, and length.
        """
        if df is None or len(df) < period:
            return pd.Series(dtype=float)
        
        # Create cache key based on data fingerprint
        cache_key = (id(df), period, len(df))
        
        if cache_key in cls._cache:
            cached = cls._cache[cache_key]
            # Verify cached data matches current DataFrame
            if len(cached) == len(df):
                return cached
        
        # Calculate ATR
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        close = df['close'].to_numpy()
        
        # True Range calculation
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First element doesn't have previous close
        
        # ATR = SMA of TR
        atr = pd.Series(tr).rolling(window=period, min_periods=period).mean()
        
        # Store in cache (limit cache size to prevent memory leak)
        if len(cls._cache) > 100:
            keys_to_remove = list(cls._cache.keys())[:50]
            for key in keys_to_remove:
                del cls._cache[key]
        
        cls._cache[cache_key] = atr
        return atr
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached ATR values."""
        cls._cache.clear()