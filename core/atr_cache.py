"""
ATR Cache - Thread-Safe Average True Range Caching.

Provides cached ATR (Average True Range) calculations to improve performance.
Used by strategies and engines throughout the system.

Features:
  - Thread-safe with locking
  - TTL-based cache invalidation (5 minutes)
  - Cache size limit (LRU eviction)
  - Vectorized ATR calculation with numpy
  - NaN and edge case handling
  - Static method interface for easy access

ATR Formula:
  True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
  ATR = SMA(True Range, period)
"""
import pandas as pd
import numpy as np
import logging
import time
from typing import Optional, Dict, Tuple
from threading import Lock
from collections import OrderedDict


class ATRCache:
    """
    Thread-safe ATR cache with TTL and LRU eviction.

    Cache key: (symbol_id, timeframe, period, data_hash)
    Cache value: (atr_series, timestamp)
    """

    _cache: OrderedDict = OrderedDict()
    _lock = Lock()
    _logger = logging.getLogger(__name__)

    # Cache configuration
    MAX_CACHE_SIZE = 1000  # Maximum number of cached ATR series
    CACHE_TTL_SECONDS = 300  # 5 minutes TTL

    @staticmethod
    def get_atr(df: pd.DataFrame, period: int = 14, symbol: str = "UNKNOWN") -> pd.Series:
        """
        Get ATR from cache or calculate if not cached.

        This is the main entry point for ATR calculations.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: ATR period (default 14)
            symbol: Symbol name for logging

        Returns:
            pd.Series with ATR values, or empty Series on error
        """
        try:
            # Validate input
            if df is None or df.empty:
                return pd.Series(dtype=float)

            required_cols = ['high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                ATRCache._logger.warning(f"[ATR_CACHE] Missing columns: {required_cols}")
                return pd.Series(dtype=float)

            if len(df) < period:
                return pd.Series(dtype=float)

            # Generate cache key
            cache_key = ATRCache._generate_cache_key(df, period, symbol)

            # Check cache with lock
            with ATRCache._lock:
                if cache_key in ATRCache._cache:
                    atr_series, timestamp = ATRCache._cache[cache_key]
                    
                    # Check TTL
                    if time.time() - timestamp < ATRCache.CACHE_TTL_SECONDS:
                        # Move to end (LRU)
                        ATRCache._cache.move_to_end(cache_key)
                        return atr_series
                    else:
                        # Expired, remove from cache
                        del ATRCache._cache[cache_key]

            # Calculate ATR (outside lock to avoid blocking)
            atr_series = ATRCache.calculate_atr(df, period)

            # Update cache with lock
            with ATRCache._lock:
                # LRU eviction if cache is full
                if len(ATRCache._cache) >= ATRCache.MAX_CACHE_SIZE:
                    # Remove oldest item
                    ATRCache._cache.popitem(last=False)

                # Add to cache
                ATRCache._cache[cache_key] = (atr_series, time.time())

            return atr_series

        except Exception as e:
            ATRCache._logger.error(f"[ATR_CACHE] Error getting ATR: {e}")
            return pd.Series(dtype=float)

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate ATR using vectorized numpy operations.

        Formula:
          True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
          ATR = SMA(True Range, period)

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: ATR period (default 14)

        Returns:
            pd.Series with ATR values
        """
        try:
            if df is None or df.empty or len(df) < period:
                return pd.Series(dtype=float)

            # Extract OHLC data
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            # Handle NaN values
            high = np.nan_to_num(high, nan=0.0)
            low = np.nan_to_num(low, nan=0.0)
            close = np.nan_to_num(close, nan=0.0)

            # Calculate True Range (vectorized)
            true_range = ATRCache._true_range(high, low, close)

            # Calculate ATR (Simple Moving Average of True Range)
            atr = pd.Series(true_range).rolling(window=period, min_periods=period).mean()

            # Fill NaN values at the beginning with first valid ATR
            first_valid = atr.first_valid_index()
            if first_valid is not None and first_valid > 0:
                atr.iloc[:first_valid] = atr.iloc[first_valid]

            return atr

        except Exception as e:
            ATRCache._logger.error(f"[ATR_CACHE] Error calculating ATR: {e}")
            return pd.Series(dtype=float)

    @staticmethod
    def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Calculate True Range using vectorized operations.

        True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices

        Returns:
            Array of true range values
        """
        try:
            # Shift close by 1 to get previous close
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]  # First element has no previous

            # Calculate three components
            high_low = high - low
            high_prev_close = np.abs(high - prev_close)
            low_prev_close = np.abs(low - prev_close)

            # True Range is the maximum of the three
            true_range = np.maximum(high_low, np.maximum(high_prev_close, low_prev_close))

            return true_range

        except Exception as e:
            ATRCache._logger.error(f"[ATR_CACHE] Error calculating true range: {e}")
            return np.zeros(len(high))

    @staticmethod
    def _generate_cache_key(df: pd.DataFrame, period: int, symbol: str) -> Tuple:
        """
        Generate unique cache key based on data characteristics.

        Args:
            df: DataFrame with OHLC data
            period: ATR period
            symbol: Symbol name

        Returns:
            Tuple representing cache key
        """
        try:
            # Use last close price and length as data identifier
            last_close = float(df['close'].iloc[-1]) if not df.empty else 0.0
            data_length = len(df)
            
            # Use timestamp of last row if available
            if 'time' in df.columns and not df.empty:
                last_time = df['time'].iloc[-1]
                if isinstance(last_time, pd.Timestamp):
                    last_time = last_time.timestamp()
                else:
                    last_time = 0.0
            else:
                last_time = 0.0

            return (symbol, period, data_length, last_close, last_time)

        except Exception as e:
            ATRCache._logger.error(f"[ATR_CACHE] Error generating cache key: {e}")
            return (symbol, period, 0, 0.0, 0.0)

    @staticmethod
    def invalidate_cache():
        """
        Clear all cached ATR values.

        Call this when:
          - Data source changes
          - Symbol changes
          - Need to force recalculation
        """
        with ATRCache._lock:
            cache_size = len(ATRCache._cache)
            ATRCache._cache.clear()
            ATRCache._logger.info(f"[ATR_CACHE] Cache invalidated, cleared {cache_size} entries")

    @staticmethod
    def get_cache_stats() -> Dict:
        """
        Get cache statistics for monitoring.

        Returns:
            Dict with cache statistics
        """
        with ATRCache._lock:
            cache_size = len(ATRCache._cache)
            current_time = time.time()
            
            # Count expired entries
            expired_count = sum(
                1 for _, (_, timestamp) in ATRCache._cache.items()
                if current_time - timestamp >= ATRCache.CACHE_TTL_SECONDS
            )

            return {
                'cache_size': cache_size,
                'max_size': ATRCache.MAX_CACHE_SIZE,
                'ttl_seconds': ATRCache.CACHE_TTL_SECONDS,
                'expired_entries': expired_count,
                'active_entries': cache_size - expired_count,
                'utilization_pct': (cache_size / ATRCache.MAX_CACHE_SIZE) * 100 if ATRCache.MAX_CACHE_SIZE > 0 else 0
            }

    @staticmethod
    def cleanup_expired():
        """
        Remove expired entries from cache.

        Call this periodically to prevent memory buildup.
        """
        with ATRCache._lock:
            current_time = time.time()
            keys_to_remove = [
                key for key, (_, timestamp) in ATRCache._cache.items()
                if current_time - timestamp >= ATRCache.CACHE_TTL_SECONDS
            ]

            for key in keys_to_remove:
                del ATRCache._cache[key]

            if keys_to_remove:
                ATRCache._logger.info(f"[ATR_CACHE] Cleaned up {len(keys_to_remove)} expired entries")