"""
Order Flow Analysis Engine.

Provides order flow analysis for volume-based trading:
  - Volume Imbalance detection
  - Cumulative Volume Delta (CVD)
  - Liquidity pool detection
  - Order absorption detection
  - Delta-Price divergence

Used by:
  - S2_VI_Sweep (Volume Imbalance Sweep)
  - S11_LiquidityDelta (Liquidity + Delta analysis)
  - S30_VolumeProfileReversal (Volume-based reversal)
  - Microstructure analysis
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple


class OrderFlowEngine:
    """
    Order Flow Analysis engine.
    
    Features:
      - Volume imbalance detection
      - Cumulative Volume Delta (CVD) calculation
      - Liquidity pool identification
      - Order absorption detection
      - Delta-price divergence analysis
    """

    def __init__(self):
        """Initialize OrderFlowEngine."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Detection parameters
        self.imbalance_threshold = 2.0  # Imbalance ratio threshold
        self.cvd_lookback = 50  # CVD calculation window
        self.absorption_threshold = 0.7  # Absorption detection threshold
        self.divergence_lookback = 20  # Divergence detection window

    # =========================================================================
    # VOLUME IMBALANCE DETECTION
    # =========================================================================

    def detect_volume_imbalance(
        self, df: pd.DataFrame, min_imbalance_ratio: float = None, lookback: int = None
    ) -> List[Dict]:
        """
        Detect volume imbalance zones.
        
        Volume Imbalance: Price moves through a zone with significantly
        more volume on one side (buy or sell), indicating institutional activity.
        
        Args:
            df: DataFrame with OHLCV data
            min_imbalance_ratio: Minimum ratio to qualify as imbalance
            lookback: Number of bars to look back
            
        Returns:
            List of imbalance zone dicts
        """
        if min_imbalance_ratio is None:
            min_imbalance_ratio = self.imbalance_threshold
        if lookback is None:
            lookback = self.cvd_lookback

        if df is None or df.empty or len(df) < 20:
            return []

        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            # Get volume if available
            volume = self._get_volume(df)
            if volume is None:
                return []

            imbalances = []

            # Detect bullish imbalance zones
            bullish_imbalances = self._detect_bullish_imbalance(
                close, high, low, volume, min_imbalance_ratio, lookback
            )
            imbalances.extend(bullish_imbalances)

            # Detect bearish imbalance zones
            bearish_imbalances = self._detect_bearish_imbalance(
                close, high, low, volume, min_imbalance_ratio, lookback
            )
            imbalances.extend(bearish_imbalances)

            return imbalances

        except Exception as e:
            self.logger.error(f"[ORDERFLOW] Imbalance detection error: {e}")
            return []

    def _detect_bullish_imbalance(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray,
        volume: np.ndarray, min_ratio: float, lookback: int
    ) -> List[Dict]:
        """Detect bullish volume imbalance zones."""
        imbalances = []

        try:
            n = len(close)

            for i in range(lookback, n):
                # Bullish imbalance: strong close near high with high volume
                candle_range = high[i] - low[i]
                if candle_range <= 0:
                    continue

                # Body position (close relative to range)
                body_position = (close[i] - low[i]) / candle_range

                # Volume ratio vs average
                avg_volume = np.mean(volume[max(0, i-20):i])
                if avg_volume <= 0:
                    continue
                volume_ratio = volume[i] / avg_volume

                # Bullish imbalance: close in top 30% of range with high volume
                if body_position > 0.7 and volume_ratio >= min_ratio:
                    imbalances.append({
                        'type': 'BULLISH',
                        'index': i,
                        'price': float(close[i]),
                        'high': float(high[i]),
                        'low': float(low[i]),
                        'volume_ratio': float(volume_ratio),
                        'body_position': float(body_position),
                        'strength': float(volume_ratio / min_ratio)
                    })

        except Exception as e:
            self.logger.debug(f"[ORDERFLOW] Bullish imbalance error: {e}")

        return imbalances

    def _detect_bearish_imbalance(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray,
        volume: np.ndarray, min_ratio: float, lookback: int
    ) -> List[Dict]:
        """Detect bearish volume imbalance zones."""
        imbalances = []

        try:
            n = len(close)

            for i in range(lookback, n):
                candle_range = high[i] - low[i]
                if candle_range <= 0:
                    continue

                # Body position (close relative to range)
                body_position = (close[i] - low[i]) / candle_range

                # Volume ratio vs average
                avg_volume = np.mean(volume[max(0, i-20):i])
                if avg_volume <= 0:
                    continue
                volume_ratio = volume[i] / avg_volume

                # Bearish imbalance: close in bottom 30% of range with high volume
                if body_position < 0.3 and volume_ratio >= min_ratio:
                    imbalances.append({
                        'type': 'BEARISH',
                        'index': i,
                        'price': float(close[i]),
                        'high': float(high[i]),
                        'low': float(low[i]),
                        'volume_ratio': float(volume_ratio),
                        'body_position': float(body_position),
                        'strength': float(volume_ratio / min_ratio)
                    })

        except Exception as e:
            self.logger.debug(f"[ORDERFLOW] Bearish imbalance error: {e}")

        return imbalances

    # =========================================================================
    # CUMULATIVE VOLUME DELTA (CVD)
    # =========================================================================

    def calculate_cvd(
        self, df: pd.DataFrame, lookback: int = None
    ) -> Optional[Dict]:
        """
        Calculate Cumulative Volume Delta (CVD).
        
        CVD tracks the difference between buy and sell volume.
        Positive CVD = more buying pressure
        Negative CVD = more selling pressure
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of bars for calculation
            
        Returns:
            Dict with CVD data, or None on failure
        """
        if lookback is None:
            lookback = self.cvd_lookback

        if df is None or df.empty or len(df) < 20:
            return None

        try:
            close = df['close'].values.astype(float)
            open_ = df['open'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))
            open_ = np.nan_to_num(open_, nan=np.nanmean(open_))

            # Get volume
            volume = self._get_volume(df)
            if volume is None:
                return None

            # Estimate buy/sell volume based on candle direction
            # Bullish candle: more buy volume
            # Bearish candle: more sell volume
            direction = np.where(close >= open_, 1.0, -1.0)

            # Delta per bar (signed volume)
            delta = direction * volume

            # Cumulative delta
            cvd = np.cumsum(delta)

            # Normalize CVD
            cvd_normalized = cvd / (np.abs(cvd).max() + 1e-10) if np.abs(cvd).max() > 0 else cvd

            # Calculate CVD slope (trend)
            cvd_slope = np.gradient(cvd)

            # Detect CVD divergence
            price_change = close[-1] - close[-lookback]
            cvd_change = cvd[-1] - cvd[-lookback]

            divergence = self._detect_divergence(price_change, cvd_change)

            return {
                'cvd': cvd,
                'cvd_normalized': cvd_normalized,
                'cvd_slope': cvd_slope,
                'current_cvd': float(cvd[-1]),
                'cvd_trend': 'RISING' if cvd_slope[-1] > 0 else 'FALLING',
                'divergence': divergence,
                'delta': delta
            }

        except Exception as e:
            self.logger.error(f"[ORDERFLOW] CVD calculation error: {e}")
            return None

    # =========================================================================
    # LIQUIDITY POOL DETECTION
    # =========================================================================

    def detect_liquidity_pools(
        self, df: pd.DataFrame, lookback: int = 100, pool_threshold: float = 0.3
    ) -> List[Dict]:
        """
        Detect liquidity pools (zones where orders are likely clustered).
        
        Liquidity pools are typically found at:
          - Recent swing highs/lows
          - Round numbers
          - Previous day high/low
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of bars to analyze
            pool_threshold: Threshold for pool detection
            
        Returns:
            List of liquidity pool dicts
        """
        if df is None or df.empty or len(df) < 30:
            return []

        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            pools = []

            # Detect swing high liquidity
            swing_highs = self._find_swing_points(high, 'high', lookback)
            for idx, price in swing_highs:
                pools.append({
                    'type': 'LIQUIDITY_ABOVE',
                    'price': float(price),
                    'index': int(idx),
                    'reason': 'Swing High',
                    'strength': self._calculate_pool_strength(price, close)
                })

            # Detect swing low liquidity
            swing_lows = self._find_swing_points(low, 'low', lookback)
            for idx, price in swing_lows:
                pools.append({
                    'type': 'LIQUIDITY_BELOW',
                    'price': float(price),
                    'index': int(idx),
                    'reason': 'Swing Low',
                    'strength': self._calculate_pool_strength(price, close)
                })

            return pools

        except Exception as e:
            self.logger.error(f"[ORDERFLOW] Liquidity pool detection error: {e}")
            return []

    # =========================================================================
    # ORDER ABSORPTION DETECTION
    # =========================================================================

    def detect_absorption(
        self, df: pd.DataFrame, lookback: int = 20
    ) -> List[Dict]:
        """
        Detect order absorption (large orders absorbing market orders).
        
        Absorption occurs when:
          - Price tries to move but volume doesn't support it
          - Large passive orders absorb aggressive orders
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of bars to analyze
            
        Returns:
            List of absorption events
        """
        if df is None or df.empty or len(df) < lookback + 10:
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
            if volume is None:
                return []

            absorptions = []
            n = len(close)

            for i in range(lookback, n):
                # Calculate price change and volume ratio
                price_change = abs(close[i] - close[i-1])
                avg_range = np.mean(high[i-20:i] - low[i-20:i])
                avg_volume = np.mean(volume[i-20:i])

                if avg_range <= 0 or avg_volume <= 0:
                    continue

                # Absorption: small price change with high volume
                price_change_ratio = price_change / avg_range
                volume_ratio = volume[i] / avg_volume

                # High volume but small price movement = absorption
                if volume_ratio > 1.5 and price_change_ratio < 0.3:
                    # Determine direction
                    if close[i] > close[i-1]:
                        direction = 'BUY_ABSORPTION'
                    else:
                        direction = 'SELL_ABSORPTION'

                    absorptions.append({
                        'type': direction,
                        'index': i,
                        'price': float(close[i]),
                        'volume_ratio': float(volume_ratio),
                        'price_change_ratio': float(price_change_ratio),
                        'strength': float(volume_ratio / (price_change_ratio + 1e-10))
                    })

            return absorptions

        except Exception as e:
            self.logger.error(f"[ORDERFLOW] Absorption detection error: {e}")
            return []

    # =========================================================================
    # DELTA DIVERGENCE DETECTION
    # =========================================================================

    def detect_delta_divergence(
        self, df: pd.DataFrame, lookback: int = None
    ) -> Dict:
        """
        Detect divergence between price and CVD.
        
        Divergence types:
          - Bullish divergence: Price makes lower low, CVD makes higher low
          - Bearish divergence: Price makes higher high, CVD makes lower high
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of bars for analysis
            
        Returns:
            Dict with divergence analysis
        """
        if lookback is None:
            lookback = self.divergence_lookback

        if df is None or df.empty or len(df) < lookback + 10:
            return {
                'divergence_detected': False,
                'divergence_type': 'NONE',
                'strength': 0.0
            }

        try:
            # Calculate CVD
            cvd_result = self.calculate_cvd(df, lookback)
            if cvd_result is None:
                return {
                    'divergence_detected': False,
                    'divergence_type': 'NONE',
                    'strength': 0.0
                }

            cvd = cvd_result['cvd']
            close = df['close'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))

            # Find recent price and CVD extremes
            price_high = np.max(close[-lookback:])
            price_low = np.min(close[-lookback:])
            cvd_high = np.max(cvd[-lookback:])
            cvd_low = np.min(cvd[-lookback:])

            current_price = close[-1]
            current_cvd = cvd[-1]

            # Detect divergences
            divergence_type = 'NONE'
            strength = 0.0

            # Bullish divergence: Price lower low, CVD higher low
            if current_price < price_low * 1.01 and current_cvd > cvd_low * 1.05:
                divergence_type = 'BULLISH'
                strength = min(1.0, abs(current_cvd - cvd_low) / (abs(cvd_low) + 1e-10))

            # Bearish divergence: Price higher high, CVD lower high
            elif current_price > price_high * 0.99 and current_cvd < cvd_high * 0.95:
                divergence_type = 'BEARISH'
                strength = min(1.0, abs(current_cvd - cvd_high) / (abs(cvd_high) + 1e-10))

            return {
                'divergence_detected': divergence_type != 'NONE',
                'divergence_type': divergence_type,
                'strength': float(strength),
                'current_price': float(current_price),
                'current_cvd': float(current_cvd),
                'price_high': float(price_high),
                'price_low': float(price_low),
                'cvd_high': float(cvd_high),
                'cvd_low': float(cvd_low)
            }

        except Exception as e:
            self.logger.error(f"[ORDERFLOW] Divergence detection error: {e}")
            return {
                'divergence_detected': False,
                'divergence_type': 'NONE',
                'strength': 0.0
            }

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

    def _find_swing_points(
        self, data: np.ndarray, point_type: str, lookback: int
    ) -> List[Tuple[int, float]]:
        """Find swing points in data."""
        swings = []

        try:
            n = len(data)
            order = 3  # Bars on each side

            for i in range(order, min(n - order, n - lookback + order)):
                if point_type == 'high':
                    if data[i] == np.max(data[i-order:i+order+1]):
                        swings.append((i, float(data[i])))
                else:
                    if data[i] == np.min(data[i-order:i+order+1]):
                        swings.append((i, float(data[i])))

            # Keep only recent swings
            swings = swings[-10:]

        except Exception:
            pass

        return swings

    def _calculate_pool_strength(self, pool_price: float, close: np.ndarray) -> float:
        """Calculate strength of liquidity pool."""
        try:
            current_price = close[-1]
            distance = abs(pool_price - current_price) / current_price

            # Closer pools are more relevant
            if distance < 0.005:  # Within 0.5%
                return 1.0
            elif distance < 0.01:  # Within 1%
                return 0.8
            elif distance < 0.02:  # Within 2%
                return 0.5
            else:
                return 0.2

        except Exception:
            return 0.5

    def _detect_divergence(self, price_change: float, cvd_change: float) -> str:
        """Detect divergence type."""
        try:
            # Price up, CVD down = Bearish divergence
            if price_change > 0 and cvd_change < 0:
                return 'BEARISH'
            # Price down, CVD up = Bullish divergence
            elif price_change < 0 and cvd_change > 0:
                return 'BULLISH'
            else:
                return 'NONE'
        except Exception:
            return 'NONE'

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_orderflow_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive order flow summary.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with complete order flow analysis
        """
        result = {
            'volume_imbalances': [],
            'cvd': None,
            'liquidity_pools': [],
            'absorption': [],
            'divergence': None
        }

        if df is None or df.empty or len(df) < 30:
            return result

        try:
            # Detect volume imbalances
            result['volume_imbalances'] = self.detect_volume_imbalance(df)

            # Calculate CVD
            result['cvd'] = self.calculate_cvd(df)

            # Detect liquidity pools
            result['liquidity_pools'] = self.detect_liquidity_pools(df)

            # Detect absorption
            result['absorption'] = self.detect_absorption(df)

            # Detect divergence
            result['divergence'] = self.detect_delta_divergence(df)

            return result

        except Exception as e:
            self.logger.error(f"[ORDERFLOW] Summary error: {e}")
            return result

    def format_orderflow_log(self, summary: Dict) -> str:
        """
        Format order flow summary as concise log string.
        
        Args:
            summary: Result from get_orderflow_summary
            
        Returns:
            Formatted log string
        """
        if summary is None:
            return "[ORDERFLOW] Analysis failed"

        imbalances = len(summary.get('volume_imbalances', []))
        cvd_trend = summary.get('cvd', {}).get('cvd_trend', 'UNKNOWN') if summary.get('cvd') else 'UNKNOWN'
        pools = len(summary.get('liquidity_pools', []))
        absorption = len(summary.get('absorption', []))
        divergence = summary.get('divergence', {}).get('divergence_type', 'NONE') if summary.get('divergence') else 'NONE'

        return (
            f"[ORDERFLOW] Imbalances: {imbalances} | "
            f"CVD: {cvd_trend} | "
            f"Pools: {pools} | "
            f"Absorption: {absorption} | "
            f"Divergence: {divergence}"
        )