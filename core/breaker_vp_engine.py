"""
Breaker & Volume Profile Engine.

Provides volume-based market analysis:
  - Volume Profile calculation
  - Point of Control (POC) detection
  - Value Area calculation (VAH/VAL)
  - High Volume Node (HVN) detection
  - Low Volume Node (LVN) detection
  - Volume Delta analysis
  - On Balance Volume (OBV)

Used by:
  - AdaptiveTPEngine (Layer 2: Volume Profile TP)
  - S21_BreakerFVGPOC
  - S22_WyckoffSpring
  - Volume-based regime detection
"""
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional
from collections import deque


class BreakerVPEngine:
    """
    Volume Profile and Breaker analysis engine.
    
    Features:
      - Volume Profile histogram calculation
      - POC (Point of Control) detection
      - Value Area (VAH/VAL) calculation
      - HVN/LVN detection
      - Volume Delta analysis
      - OBV calculation
      - Result caching for performance
    """

    def __init__(self):
        """Initialize BreakerVPEngine."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Default parameters
        self.default_bins = 120
        self.value_area_pct = 0.70  # 70% value area
        self.hvn_threshold = 1.5  # HVN if volume > 1.5x average
        self.lvn_threshold = 0.3  # LVN if volume < 0.3x average

        # Cache for performance
        self._cache = {}
        self._cache_ttl = 60  # 60 seconds
        self._cache_max_size = 20

        # Statistics
        self._calculation_count = 0
        self._cache_hits = 0

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def calculate_volume_profile_levels(
        self,
        df: pd.DataFrame,
        bins: int = None,
        value_area_pct: float = None
    ) -> Optional[Dict]:
        """
        Main entry point: Calculate all volume profile levels.
        
        Args:
            df: DataFrame with OHLCV data
            bins: Number of price bins (default: 120)
            value_area_pct: Value area percentage (default: 0.70)
            
        Returns:
            Dict with:
              - poc: Point of Control price
              - vah: Value Area High
              - val: Value Area Low
              - hvns: List of High Volume Nodes
              - lvns: List of Low Volume Nodes
              - profile: Complete volume profile histogram
        """
        if bins is None:
            bins = self.default_bins
        if value_area_pct is None:
            value_area_pct = self.value_area_pct

        if df is None or df.empty or len(df) < 20:
            return None

        # Check cache
        cache_key = self._generate_cache_key(df, bins, value_area_pct)
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        try:
            # Extract price and volume data
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            close = df['close'].values.astype(float)

            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].values.astype(float)
            elif 'volume' in df.columns:
                volume = df['volume'].values.astype(float)
            else:
                self.logger.warning("[VP] No volume column found")
                return None

            # Handle NaN
            high = np.nan_to_num(high, nan=high[0])
            low = np.nan_to_num(low, nan=low[0])
            close = np.nan_to_num(close, nan=close[0])
            volume = np.nan_to_num(volume, nan=0.0)

            # Calculate volume profile
            profile = self._calculate_profile(high, low, close, volume, bins)

            if profile is None or len(profile) == 0:
                return None

            # Calculate POC
            poc = self._find_poc(profile)

            # Calculate Value Area
            vah, val = self._calculate_value_area(profile, poc, value_area_pct)

            # Calculate HVN and LVN
            hvns = self._find_hvn(profile)
            lvns = self._find_lvn(profile)

            result = {
                'poc': poc,
                'vah': vah,
                'val': val,
                'hvns': hvns,
                'lvns': lvns,
                'profile': profile,
                'bins': bins,
                'value_area_pct': value_area_pct,
                'total_volume': float(np.sum(volume)),
                'price_range': {
                    'high': float(np.max(high)),
                    'low': float(np.min(low))
                }
            }

            # Store in cache
            self._store_in_cache(cache_key, result)
            self._calculation_count += 1

            return result

        except Exception as e:
            self.logger.error(f"[VP] Volume profile calculation error: {e}")
            return None

    # =========================================================================
    # PROFILE CALCULATION
    # =========================================================================

    def _calculate_profile(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        bins: int
    ) -> Optional[Dict]:
        """
        Calculate volume profile histogram.
        
        Distributes volume across price bins based on candle ranges.
        
        Returns:
            Dict with bin_edges, bin_centers, bin_volumes
        """
        try:
            price_min = float(np.min(low))
            price_max = float(np.max(high))

            if price_max <= price_min:
                return None

            # Create price bins
            bin_edges = np.linspace(price_min, price_max, bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Initialize bin volumes
            bin_volumes = np.zeros(bins)

            # Distribute volume across bins
            for i in range(len(close)):
                candle_low = low[i]
                candle_high = high[i]
                candle_volume = volume[i]

                # Skip if no volume
                if candle_volume <= 0:
                    continue

                # Find bins that overlap with this candle's range
                for j in range(bins):
                    bin_low = bin_edges[j]
                    bin_high = bin_edges[j + 1]

                    # Check if bin overlaps with candle range
                    if bin_high >= candle_low and bin_low <= candle_high:
                        # Calculate overlap percentage
                        overlap_low = max(bin_low, candle_low)
                        overlap_high = min(bin_high, candle_high)

                        if overlap_high > overlap_low:
                            candle_range = candle_high - candle_low
                            if candle_range > 0:
                                overlap_pct = (overlap_high - overlap_low) / candle_range
                                bin_volumes[j] += candle_volume * overlap_pct

            return {
                'bin_edges': bin_edges,
                'bin_centers': bin_centers,
                'bin_volumes': bin_volumes
            }

        except Exception as e:
            self.logger.error(f"[VP] Profile calculation error: {e}")
            return None

    # =========================================================================
    # POC (POINT OF CONTROL)
    # =========================================================================

    def _find_poc(self, profile: Dict) -> float:
        """
        Find Point of Control (price with highest volume).
        
        Returns:
            POC price
        """
        try:
            bin_centers = profile['bin_centers']
            bin_volumes = profile['bin_volumes']

            # Find bin with maximum volume
            poc_idx = np.argmax(bin_volumes)
            poc_price = float(bin_centers[poc_idx])

            return poc_price

        except Exception:
            return 0.0

    # =========================================================================
    # VALUE AREA (VAH/VAL)
    # =========================================================================

    def _calculate_value_area(
        self, profile: Dict, poc: float, value_area_pct: float
    ) -> Tuple[float, float]:
        """
        Calculate Value Area High (VAH) and Value Area Low (VAL).
        
        Value Area contains the specified percentage of total volume,
        centered around POC.
        
        Returns:
            Tuple of (vah, val)
        """
        try:
            bin_centers = profile['bin_centers']
            bin_volumes = profile['bin_volumes']
            bin_edges = profile['bin_edges']

            total_volume = np.sum(bin_volumes)
            target_volume = total_volume * value_area_pct

            # Find POC bin index
            poc_idx = np.argmin(np.abs(bin_centers - poc))

            # Start from POC and expand outward
            current_volume = bin_volumes[poc_idx]
            low_idx = poc_idx
            high_idx = poc_idx

            while current_volume < target_volume:
                # Check if we can expand
                can_expand_low = low_idx > 0
                can_expand_high = high_idx < len(bin_volumes) - 1

                if not can_expand_low and not can_expand_high:
                    break

                # Choose direction with higher adjacent volume
                low_vol = bin_volumes[low_idx - 1] if can_expand_low else 0
                high_vol = bin_volumes[high_idx + 1] if can_expand_high else 0

                if low_vol >= high_vol and can_expand_low:
                    low_idx -= 1
                    current_volume += bin_volumes[low_idx]
                elif can_expand_high:
                    high_idx += 1
                    current_volume += bin_volumes[high_idx]
                else:
                    break

            vah = float(bin_edges[high_idx + 1])
            val = float(bin_edges[low_idx])

            return vah, val

        except Exception:
            return 0.0, 0.0

    # =========================================================================
    # HVN (HIGH VOLUME NODES)
    # =========================================================================

    def _find_hvn(self, profile: Dict) -> List[float]:
        """
        Find High Volume Nodes (prices with significantly high volume).
        
        Returns:
            List of HVN prices
        """
        try:
            bin_centers = profile['bin_centers']
            bin_volumes = profile['bin_volumes']

            if len(bin_volumes) == 0:
                return []

            avg_volume = np.mean(bin_volumes)
            threshold = avg_volume * self.hvn_threshold

            hvns = []
            for i in range(len(bin_volumes)):
                if bin_volumes[i] > threshold:
                    hvns.append(float(bin_centers[i]))

            # Merge nearby HVNs (within 1% of price range)
            merged_hvns = self._merge_nearby_levels(hvns, bin_centers)

            return merged_hvns

        except Exception:
            return []

    # =========================================================================
    # LVN (LOW VOLUME NODES)
    # =========================================================================

    def _find_lvn(self, profile: Dict) -> List[float]:
        """
        Find Low Volume Nodes (prices with significantly low volume).
        
        These are often breakout points.
        
        Returns:
            List of LVN prices
        """
        try:
            bin_centers = profile['bin_centers']
            bin_volumes = profile['bin_volumes']

            if len(bin_volumes) == 0:
                return []

            avg_volume = np.mean(bin_volumes)
            threshold = avg_volume * self.lvn_threshold

            lvns = []
            for i in range(len(bin_volumes)):
                if bin_volumes[i] < threshold:
                    lvns.append(float(bin_centers[i]))

            return lvns

        except Exception:
            return []

    # =========================================================================
    # VOLUME DELTA
    # =========================================================================

    def calculate_volume_delta(self, df: pd.DataFrame, lookback: int = 20) -> Optional[Dict]:
        """
        Calculate volume delta (buy vs sell pressure).
        
        Approximation using candle direction:
          - Bullish candle: volume adds to buy delta
          - Bearish candle: volume adds to sell delta
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of bars to analyze
            
        Returns:
            Dict with delta metrics
        """
        if df is None or df.empty or len(df) < lookback:
            return None

        try:
            open_ = df['open'].values.astype(float)
            close = df['close'].values.astype(float)

            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].values.astype(float)
            elif 'volume' in df.columns:
                volume = df['volume'].values.astype(float)
            else:
                return None

            # Handle NaN
            volume = np.nan_to_num(volume, nan=0.0)

            # Calculate direction
            direction = np.where(close >= open_, 1.0, -1.0)

            # Calculate signed volume
            signed_volume = direction * volume

            # Recent analysis
            recent_delta = np.sum(signed_volume[-lookback:])
            total_volume = np.sum(volume[-lookback:])

            # Cumulative delta (rolling)
            cum_delta = np.cumsum(signed_volume)

            # Delta divergence detection
            price_change = close[-1] - close[-lookback]
            delta_direction = 1 if recent_delta > 0 else -1
            price_direction = 1 if price_change > 0 else -1

            # Divergence: price up but delta down, or price down but delta up
            divergence = (price_direction != delta_direction)

            return {
                'recent_delta': float(recent_delta),
                'total_volume': float(total_volume),
                'delta_pct': float(recent_delta / total_volume) if total_volume > 0 else 0.0,
                'cum_delta': float(cum_delta[-1]),
                'divergence': divergence,
                'direction': 'BULLISH' if recent_delta > 0 else 'BEARISH',
                'strength': abs(float(recent_delta / total_volume)) if total_volume > 0 else 0.0
            }

        except Exception as e:
            self.logger.error(f"[VP] Volume delta calculation error: {e}")
            return None

    # =========================================================================
    # OBV (ON BALANCE VOLUME)
    # =========================================================================

    def calculate_obv(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Calculate On Balance Volume (OBV).
        
        OBV adds volume on up days, subtracts on down days.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            OBV series
        """
        if df is None or df.empty or len(df) < 5:
            return None

        try:
            close = df['close'].values.astype(float)

            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].values.astype(float)
            elif 'volume' in df.columns:
                volume = df['volume'].values.astype(float)
            else:
                return None

            # Handle NaN
            volume = np.nan_to_num(volume, nan=0.0)

            # Calculate OBV
            obv = np.zeros(len(close))
            obv[0] = volume[0]

            for i in range(1, len(close)):
                if close[i] > close[i - 1]:
                    obv[i] = obv[i - 1] + volume[i]
                elif close[i] < close[i - 1]:
                    obv[i] = obv[i - 1] - volume[i]
                else:
                    obv[i] = obv[i - 1]

            return pd.Series(obv, index=df.index)

        except Exception as e:
            self.logger.error(f"[VP] OBV calculation error: {e}")
            return None

    # =========================================================================
    # CACHING
    # =========================================================================

    def _generate_cache_key(self, df: pd.DataFrame, bins: int, value_area_pct: float) -> str:
        """Generate cache key from DataFrame characteristics."""
        try:
            last_close = float(df['close'].iloc[-1])
            data_len = len(df)
            return f"{last_close:.2f}_{data_len}_{bins}_{value_area_pct}"
        except Exception:
            return ""

    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """Get result from cache if valid."""
        if key in self._cache:
            cached_time, result = self._cache[key]
            if time.time() - cached_time < self._cache_ttl:
                return result
            else:
                del self._cache[key]
        return None

    def _store_in_cache(self, key: str, result: Dict):
        """Store result in cache."""
        # Remove oldest entries if cache is full
        if len(self._cache) >= self._cache_max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]

        self._cache[key] = (time.time(), result)

    def clear_cache(self):
        """Clear all cached results."""
        self._cache.clear()
        self.logger.debug("[VP] Cache cleared")

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _merge_nearby_levels(self, levels: List[float], bin_centers: np.ndarray) -> List[float]:
        """Merge nearby levels that are within 1% of price range."""
        if not levels or len(levels) < 2:
            return levels

        price_range = float(np.max(bin_centers) - np.min(bin_centers))
        if price_range == 0:
            return levels

        tolerance = price_range * 0.01  # 1% tolerance

        levels_sorted = sorted(levels)
        merged = [levels_sorted[0]]

        for level in levels_sorted[1:]:
            if abs(level - merged[-1]) > tolerance:
                merged.append(level)

        return merged

    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache stats
        """
        return {
            'cache_size': len(self._cache),
            'cache_max_size': self._cache_max_size,
            'cache_ttl': self._cache_ttl,
            'total_calculations': self._calculation_count,
            'cache_hits': self._cache_hits,
            'cache_hit_rate': round(
                self._cache_hits / max(1, self._calculation_count + self._cache_hits) * 100, 1
            )
        }

    def format_vp_log(self, result: Dict) -> str:
        """
        Format volume profile result as concise log string.
        
        Args:
            result: Result from calculate_volume_profile_levels
            
        Returns:
            Formatted log string
        """
        if result is None:
            return "[VP] No volume profile data"

        poc = result.get('poc', 0)
        vah = result.get('vah', 0)
        val = result.get('val', 0)
        hvn_count = len(result.get('hvns', []))
        lvn_count = len(result.get('lvns', []))

        return (
            f"[VP] POC: {poc:.2f} | "
            f"VA: {val:.2f}-{vah:.2f} | "
            f"HVN: {hvn_count} | LVN: {lvn_count}"
        )