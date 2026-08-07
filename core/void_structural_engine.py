"""
Void Structural Engine.

Provides liquidity void and structural analysis:
  - Liquidity void detection
  - Void fill analysis
  - Structural level identification
  - Gap analysis
  - Reversal zone detection

Used by:
  - S5_Breaker_Void (Breaker + Void strategy)
  - S19_VoidReversal (Void reversal strategy)
  - Liquidity analysis
  - Structural trading
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple


class VoidStructuralEngine:
    """
    Void Structural Analysis engine.
    
    Features:
      - Liquidity void detection
      - Void fill tracking
      - Structural level identification
      - Price gap analysis
      - Reversal zone detection
    """

    def __init__(self):
        """Initialize VoidStructuralEngine."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Void detection parameters
        self.void_threshold_atr = 1.5  # Minimum void size in ATR units
        self.void_lookback = 50  # Lookback for void detection
        self.fill_threshold = 0.5  # Void fill threshold (50%)

        # Structural level parameters
        self.swing_order = 5  # Bars on each side for swing detection
        self.level_tolerance = 0.002  # 0.2% tolerance for level matching

    # =========================================================================
    # LIQUIDITY VOID DETECTION
    # =========================================================================

    def detect_liquidity_void(
        self, df: pd.DataFrame, lookback: int = None
    ) -> List[Dict]:
        """
        Detect liquidity voids (gaps where price moved quickly without trading).
        
        Liquidity voids are areas where:
          - Price moved rapidly (large candle)
          - Low volume in the gap area
          - Price tends to return to fill the void
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of bars to look back
            
        Returns:
            List of liquidity void dicts
        """
        if lookback is None:
            lookback = self.void_lookback

        if df is None or df.empty or len(df) < lookback:
            return []

        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            # Get volume
            volume = self._get_volume(df)

            # Calculate ATR for threshold
            atr = self._calculate_atr(high, low, close, 14)
            if atr is None or len(atr) == 0:
                return []

            current_atr = atr[-1]
            void_threshold = current_atr * self.void_threshold_atr

            voids = []

            # Detect bullish voids (gap up)
            bullish_voids = self._detect_bullish_voids(
                close, high, low, volume, void_threshold, lookback
            )
            voids.extend(bullish_voids)

            # Detect bearish voids (gap down)
            bearish_voids = self._detect_bearish_voids(
                close, high, low, volume, void_threshold, lookback
            )
            voids.extend(bearish_voids)

            return voids

        except Exception as e:
            self.logger.error(f"[VOID] Void detection error: {e}")
            return []

    def _detect_bullish_voids(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray,
        volume: np.ndarray, threshold: float, lookback: int
    ) -> List[Dict]:
        """Detect bullish liquidity voids (gap up)."""
        voids = []

        try:
            n = len(close)
            start_idx = max(1, n - lookback)

            for i in range(start_idx, n):
                # Bullish void: large gap up with low volume
                gap = low[i] - high[i-1]

                if gap > threshold:
                    # Check volume (should be lower than average)
                    if volume is not None and i > 10:
                        avg_volume = np.mean(volume[max(0, i-20):i])
                        volume_ratio = volume[i] / avg_volume if avg_volume > 0 else 1.0
                    else:
                        volume_ratio = 1.0

                    voids.append({
                        'type': 'BULLISH_VOID',
                        'index': i,
                        'gap_top': float(low[i]),
                        'gap_bottom': float(high[i-1]),
                        'gap_size': float(gap),
                        'volume_ratio': float(volume_ratio),
                        'filled': False,
                        'fill_percentage': 0.0
                    })

        except Exception as e:
            self.logger.debug(f"[VOID] Bullish void detection error: {e}")

        return voids

    def _detect_bearish_voids(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray,
        volume: np.ndarray, threshold: float, lookback: int
    ) -> List[Dict]:
        """Detect bearish liquidity voids (gap down)."""
        voids = []

        try:
            n = len(close)
            start_idx = max(1, n - lookback)

            for i in range(start_idx, n):
                # Bearish void: large gap down with low volume
                gap = low[i-1] - high[i]

                if gap > threshold:
                    # Check volume
                    if volume is not None and i > 10:
                        avg_volume = np.mean(volume[max(0, i-20):i])
                        volume_ratio = volume[i] / avg_volume if avg_volume > 0 else 1.0
                    else:
                        volume_ratio = 1.0

                    voids.append({
                        'type': 'BEARISH_VOID',
                        'index': i,
                        'gap_top': float(low[i-1]),
                        'gap_bottom': float(high[i]),
                        'gap_size': float(gap),
                        'volume_ratio': float(volume_ratio),
                        'filled': False,
                        'fill_percentage': 0.0
                    })

        except Exception as e:
            self.logger.debug(f"[VOID] Bearish void detection error: {e}")

        return voids

    # =========================================================================
    # VOID FILL ANALYSIS
    # =========================================================================

    def check_void_fill(
        self, df: pd.DataFrame, voids: List[Dict]
    ) -> List[Dict]:
        """
        Check if detected voids are being filled.
        
        Args:
            df: DataFrame with OHLCV data
            voids: List of detected voids
            
        Returns:
            Updated void list with fill status
        """
        if df is None or df.empty or not voids:
            return voids

        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            current_price = close[-1]

            for void in voids:
                void_top = void['gap_top']
                void_bottom = void['gap_bottom']
                void_size = void['gap_size']

                if void_size <= 0:
                    continue

                # Calculate fill percentage
                if void['type'] == 'BULLISH_VOID':
                    # Bullish void fills when price returns down into the gap
                    if current_price < void_top:
                        fill_depth = void_top - current_price
                        fill_percentage = min(1.0, fill_depth / void_size)
                        void['fill_percentage'] = float(fill_percentage)
                        void['filled'] = fill_percentage >= self.fill_threshold
                    else:
                        void['fill_percentage'] = 0.0
                        void['filled'] = False

                else:  # BEARISH_VOID
                    # Bearish void fills when price returns up into the gap
                    if current_price > void_bottom:
                        fill_depth = current_price - void_bottom
                        fill_percentage = min(1.0, fill_depth / void_size)
                        void['fill_percentage'] = float(fill_percentage)
                        void['filled'] = fill_percentage >= self.fill_threshold
                    else:
                        void['fill_percentage'] = 0.0
                        void['filled'] = False

            return voids

        except Exception as e:
            self.logger.error(f"[VOID] Void fill check error: {e}")
            return voids

    # =========================================================================
    # STRUCTURAL LEVELS
    # =========================================================================

    def find_structural_levels(
        self, df: pd.DataFrame, lookback: int = 100
    ) -> Dict:
        """
        Find key structural levels (swing highs/lows, support/resistance).
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of bars to analyze
            
        Returns:
            Dict with structural levels
        """
        if df is None or df.empty or len(df) < 30:
            return {
                'swing_highs': [],
                'swing_lows': [],
                'resistance': None,
                'support': None
            }

        try:
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Handle NaN
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            # Use lookback window
            if len(high) > lookback:
                high = high[-lookback:]
                low = low[-lookback:]

            # Find swing highs
            swing_highs = self._find_swing_points(high, 'high')

            # Find swing lows
            swing_lows = self._find_swing_points(low, 'low')

            # Determine current resistance and support
            current_price = df['close'].values[-1]

            resistance = None
            support = None

            # Find nearest resistance above current price
            for sh in swing_highs:
                if sh['price'] > current_price:
                    if resistance is None or sh['price'] < resistance['price']:
                        resistance = sh

            # Find nearest support below current price
            for sl in swing_lows:
                if sl['price'] < current_price:
                    if support is None or sl['price'] > support['price']:
                        support = sl

            return {
                'swing_highs': swing_highs,
                'swing_lows': swing_lows,
                'resistance': resistance,
                'support': support,
                'current_price': float(current_price)
            }

        except Exception as e:
            self.logger.error(f"[VOID] Structural levels error: {e}")
            return {
                'swing_highs': [],
                'swing_lows': [],
                'resistance': None,
                'support': None
            }

    def _find_swing_points(self, data: np.ndarray, point_type: str) -> List[Dict]:
        """Find swing points in data."""
        swings = []

        try:
            n = len(data)
            order = self.swing_order

            for i in range(order, n - order):
                if point_type == 'high':
                    if data[i] == np.max(data[i-order:i+order+1]):
                        swings.append({
                            'index': i,
                            'price': float(data[i]),
                            'type': 'HIGH'
                        })
                else:
                    if data[i] == np.min(data[i-order:i+order+1]):
                        swings.append({
                            'index': i,
                            'price': float(data[i]),
                            'type': 'LOW'
                        })

            # Keep only recent swings
            swings = swings[-10:]

        except Exception:
            pass

        return swings

    # =========================================================================
    # GAP ANALYSIS
    # =========================================================================

    def analyze_gaps(self, df: pd.DataFrame, lookback: int = 50) -> List[Dict]:
        """
        Analyze price gaps.
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of bars to analyze
            
        Returns:
            List of gap analysis results
        """
        if df is None or df.empty or len(df) < 20:
            return []

        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            open_ = df['open'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))
            open_ = np.nan_to_num(open_, nan=np.nanmean(open_))

            gaps = []
            n = len(close)
            start_idx = max(1, n - lookback)

            for i in range(start_idx, n):
                # Calculate gap
                gap = open_[i] - close[i-1]
                gap_pct = gap / close[i-1] * 100 if close[i-1] > 0 else 0

                # Significant gap threshold (0.1%)
                if abs(gap_pct) > 0.1:
                    gap_type = 'GAP_UP' if gap > 0 else 'GAP_DOWN'

                    # Check if gap was filled
                    filled = self._check_gap_filled(gap_type, open_[i], close[i-1], close[i:])

                    gaps.append({
                        'type': gap_type,
                        'index': i,
                        'gap_size': float(gap),
                        'gap_pct': float(gap_pct),
                        'filled': filled,
                        'open_price': float(open_[i]),
                        'prev_close': float(close[i-1])
                    })

            return gaps

        except Exception as e:
            self.logger.error(f"[VOID] Gap analysis error: {e}")
            return []

    def _check_gap_filled(self, gap_type: str, open_price: float,
                           prev_close: float, future_close: np.ndarray) -> bool:
        """Check if a gap was filled."""
        try:
            if gap_type == 'GAP_UP':
                # Gap up is filled if price returns below previous close
                return any(future_close < prev_close)
            else:
                # Gap down is filled if price returns above previous close
                return any(future_close > prev_close)
        except Exception:
            return False

    # =========================================================================
    # REVERSAL ZONES
    # =========================================================================

    def detect_reversal_zones(
        self, df: pd.DataFrame, voids: List[Dict]
    ) -> List[Dict]:
        """
        Detect reversal zones at liquidity voids.
        
        Reversal zones are areas where price is likely to reverse
        after filling a liquidity void.
        
        Args:
            df: DataFrame with OHLCV data
            voids: List of detected voids
            
        Returns:
            List of reversal zone dicts
        """
        if df is None or df.empty or not voids:
            return []

        try:
            close = df['close'].values.astype(float)
            current_price = close[-1]

            reversal_zones = []

            for void in voids:
                if void['filled'] or void['fill_percentage'] > 0.7:
                    # Void is mostly filled - potential reversal zone
                    void_top = void['gap_top']
                    void_bottom = void['gap_bottom']

                    if void['type'] == 'BULLISH_VOID':
                        # Bullish void filled = potential bearish reversal
                        reversal_zones.append({
                            'type': 'BEARISH_REVERSAL',
                            'zone_top': float(void_top),
                            'zone_bottom': float(void_bottom),
                            'fill_percentage': float(void['fill_percentage']),
                            'strength': float(void['fill_percentage'])
                        })
                    else:
                        # Bearish void filled = potential bullish reversal
                        reversal_zones.append({
                            'type': 'BULLISH_REVERSAL',
                            'zone_top': float(void_top),
                            'zone_bottom': float(void_bottom),
                            'fill_percentage': float(void['fill_percentage']),
                            'strength': float(void['fill_percentage'])
                        })

            return reversal_zones

        except Exception as e:
            self.logger.error(f"[VOID] Reversal zone detection error: {e}")
            return []

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_volume(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Get volume array from DataFrame."""
        try:
            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].values.astype(float)
            elif 'volume' in df.columns:
                volume = df['volume'].values.astype(float)
            else:
                return None

            return np.nan_to_num(volume, nan=1.0)
        except Exception:
            return None

    def _calculate_atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> Optional[np.ndarray]:
        """Calculate Average True Range."""
        try:
            n = len(high)
            if n < period + 1:
                return None

            tr = np.zeros(n)

            for i in range(1, n):
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr[i] = max(tr1, tr2, tr3)

            tr[0] = high[0] - low[0]

            # Calculate ATR using EMA
            atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values

            return np.nan_to_num(atr, nan=np.nanmean(tr))

        except Exception:
            return None

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_void_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive void structural analysis.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with complete void analysis
        """
        result = {
            'voids': [],
            'structural_levels': None,
            'gaps': [],
            'reversal_zones': []
        }

        if df is None or df.empty or len(df) < 30:
            return result

        try:
            # Detect liquidity voids
            result['voids'] = self.detect_liquidity_void(df)

            # Check void fill status
            result['voids'] = self.check_void_fill(df, result['voids'])

            # Find structural levels
            result['structural_levels'] = self.find_structural_levels(df)

            # Analyze gaps
            result['gaps'] = self.analyze_gaps(df)

            # Detect reversal zones
            result['reversal_zones'] = self.detect_reversal_zones(df, result['voids'])

            return result

        except Exception as e:
            self.logger.error(f"[VOID] Analysis error: {e}")
            return result

    def format_void_log(self, analysis_result: Dict) -> str:
        """
        Format void analysis result as concise log string.
        
        Args:
            analysis_result: Result from get_void_analysis
            
        Returns:
            Formatted log string
        """
        if analysis_result is None:
            return "[VOID] Analysis failed"

        voids = analysis_result.get('voids', [])
        gaps = analysis_result.get('gaps', [])
        reversal_zones = analysis_result.get('reversal_zones', [])

        filled_voids = sum(1 for v in voids if v.get('filled', False))
        unfilled_voids = len(voids) - filled_voids

        return (
            f"[VOID] Voids: {len(voids)} (filled: {filled_voids}) | "
            f"Gaps: {len(gaps)} | "
            f"Reversal Zones: {len(reversal_zones)}"
        )

    def is_void_reversal_signal(self, analysis_result: Dict) -> Dict:
        """
        Check for void reversal signal.
        
        Args:
            analysis_result: Result from get_void_analysis
            
        Returns:
            Dict with reversal signal
        """
        if analysis_result is None:
            return {'signal': 'NEUTRAL', 'reason': 'No data'}

        reversal_zones = analysis_result.get('reversal_zones', [])

        if not reversal_zones:
            return {'signal': 'NEUTRAL', 'reason': 'No reversal zones'}

        # Find strongest reversal zone
        strongest = max(reversal_zones, key=lambda x: x.get('strength', 0))

        if strongest['type'] == 'BULLISH_REVERSAL':
            return {
                'signal': 'BUY_REVERSAL',
                'reason': f"Bullish reversal zone at {strongest['zone_bottom']:.2f}",
                'strength': strongest['strength'],
                'zone': strongest
            }
        elif strongest['type'] == 'BEARISH_REVERSAL':
            return {
                'signal': 'SELL_REVERSAL',
                'reason': f"Bearish reversal zone at {strongest['zone_top']:.2f}",
                'strength': strongest['strength'],
                'zone': strongest
            }
        else:
            return {'signal': 'NEUTRAL', 'reason': 'Unknown zone type'}