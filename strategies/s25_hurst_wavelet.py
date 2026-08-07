"""
S25_HurstWavelet - Hurst Exponent + Wavelet Strategy.

Trend-following strategy that uses Hurst exponent for regime
classification and wavelet analysis for multi-resolution trend detection.

Strategy Logic:
  1. Calculate Hurst exponent to classify market regime
  2. Apply wavelet decomposition for multi-resolution analysis
  3. Determine trend direction from wavelet components
  4. Generate entry signal when trend is confirmed

Hurst Exponent Definition:
  The Hurst exponent (H) measures the long-term memory of a time series:
    - H > 0.5: Trending market (persistent)
    - H = 0.5: Random walk
    - H < 0.5: Mean-reverting market (anti-persistent)
    
  For trend-following, we want H > 0.5 (trending market).

Wavelet Analysis:
  Wavelets decompose a signal into multiple frequency bands:
    - Approximation: Low-frequency component (trend)
    - Details: High-frequency components (cycles, noise)
    
  For trend detection, we use the approximation component.

Used Engines:
  - HurstWaveletEngine: Hurst and wavelet analysis
  - AdaptiveTPEngine: Dynamic TP calculation

Strategy Category: TREND
Timeframe: M15 (primary), M5 (confirmation)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from core.base_strategy import BaseStrategy
from core.hurst_wavelet_engine import HurstWaveletEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S25_HurstWavelet(BaseStrategy):
    """
    Hurst Exponent + Wavelet Strategy.
    
    This strategy uses Hurst exponent for regime classification
    and wavelet analysis for multi-resolution trend detection.
    
    Hurst Exponent Definition:
      Measures the long-term memory of a time series:
        - H > 0.5: Trending (persistent)
        - H = 0.5: Random walk
        - H < 0.5: Mean-reverting (anti-persistent)
        
    Wavelet Analysis:
      Decomposes signal into multiple frequency bands for
      multi-resolution analysis. The approximation component
      captures the trend, while details capture cycles and noise.
      
    Entry Criteria:
      - Hurst exponent > 0.55 (trending market)
      - Wavelet trend component confirms direction
      - Momentum alignment
      - Confirmation from M5 timeframe
    """

    def __init__(self):
        """Initialize S25_HurstWavelet strategy."""
        super().__init__(
            strategy_name='S25_HurstWavelet',
            strategy_category='TREND',
            timeframes=['M15', 'M5'],
            risk_per_trade_pct=0.5,
            min_rr_ratio=1.5,
            max_spread_points=30,
            trailing_enabled=True,
            partial_close_enabled=True,
            requires_dynamic_exit=False
        )

        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize engines
        self.hurst_engine = HurstWaveletEngine()
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.hurst_lookback = 100  # Lookback for Hurst calculation
        self.wavelet_levels = 3  # Number of wavelet decomposition levels
        self.min_hurst = 0.55  # Minimum Hurst for trending market
        self.max_hurst = 0.85  # Maximum Hurst (too high = unstable)

    # =========================================================================
    # MAIN ANALYSIS METHOD
    # =========================================================================

    def analyze(
        self,
        df_m15: pd.DataFrame,
        df_m5: pd.DataFrame = None,
        regime_context: Dict = None
    ) -> Dict:
        """
        Main analysis method for S25_HurstWavelet.
        
        Args:
            df_m15: M15 DataFrame
            df_m5: M5 DataFrame (optional)
            regime_context: Current regime information
            
        Returns:
            Signal dict with entry/exit information
        """
        # Default neutral signal
        default_signal = self._create_neutral_signal()

        # Validate input
        if df_m15 is None or df_m15.empty or len(df_m15) < 100:
            return default_signal

        try:
            # Check regime compatibility
            if regime_context and not self._is_regime_compatible(regime_context):
                return default_signal

            # =========================================================================
            # STEP 1: Calculate Hurst Exponent
            # =========================================================================
            close = df_m15['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            hurst_result = self._calculate_hurst(close)

            if hurst_result is None:
                return default_signal

            # =========================================================================
            # STEP 2: Apply Wavelet Decomposition
            # =========================================================================
            wavelet_result = self._apply_wavelet(close)

            if wavelet_result is None:
                return default_signal

            # =========================================================================
            # STEP 3: Classify Regime
            # =========================================================================
            regime_info = self._classify_regime(hurst_result, wavelet_result)

            if regime_info is None or not regime_info.get('is_trending', False):
                return default_signal

            # =========================================================================
            # STEP 4: Determine Direction
            # =========================================================================
            direction_info = self._determine_direction(wavelet_result, df_m15)

            if direction_info is None:
                return default_signal

            # =========================================================================
            # STEP 5: M5 Confirmation
            # =========================================================================
            if df_m5 is not None and not df_m5.empty:
                if not self._confirm_m5(df_m5, direction_info):
                    return default_signal

            # =========================================================================
            # STEP 6: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, direction_info, regime_info, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S25_HURST] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # HURST CALCULATION
    # =========================================================================

    def _calculate_hurst(self, close: np.ndarray) -> Optional[Dict]:
        """
        Calculate Hurst exponent.
        
        Args:
            close: Close price array
            
        Returns:
            Hurst result dict or None
        """
        try:
            n = len(close)
            if n < self.hurst_lookback:
                return None

            # Use recent data
            recent_close = close[-self.hurst_lookback:]

            # Calculate Hurst exponent
            hurst = self.hurst_engine.calculate_hurst_exponent(recent_close)

            if hurst is None:
                return None

            # Calculate fractal dimension
            fractal_dim = self.hurst_engine.calculate_fractal_dimension(recent_close)

            return {
                'hurst': float(hurst),
                'fractal_dimension': float(fractal_dim) if fractal_dim else None,
                'is_trending': hurst > self.min_hurst and hurst < self.max_hurst,
                'is_mean_reverting': hurst < 0.45,
                'is_random': 0.45 <= hurst <= 0.55
            }

        except Exception as e:
            self.logger.debug(f"[S25_HURST] Hurst calculation error: {e}")
            return None

    # =========================================================================
    # WAVELET DECOMPOSITION
    # =========================================================================

    def _apply_wavelet(self, close: np.ndarray) -> Optional[Dict]:
        """
        Apply wavelet decomposition.
        
        Args:
            close: Close price array
            
        Returns:
            Wavelet result dict or None
        """
        try:
            n = len(close)
            if n < 50:
                return None

            # Apply wavelet decomposition
            wavelet_result = self.hurst_engine.wavelet_decomposition(
                close, levels=self.wavelet_levels, wavelet='haar'
            )

            if wavelet_result is None:
                return None

            # Get approximation (trend) component
            approximations = wavelet_result.get('approximations', [])
            details = wavelet_result.get('details', [])

            if not approximations:
                return None

            # Last approximation = trend
            trend_component = approximations[-1]

            return {
                'approximations': approximations,
                'details': details,
                'trend_component': trend_component,
                'levels': len(approximations),
                'current_trend': float(trend_component[-1]) if len(trend_component) > 0 else 0.0
            }

        except Exception as e:
            self.logger.debug(f"[S25_HURST] Wavelet application error: {e}")
            return None

    # =========================================================================
    # REGIME CLASSIFICATION
    # =========================================================================

    def _classify_regime(self, hurst_result: Dict, wavelet_result: Dict) -> Optional[Dict]:
        """
        Classify market regime based on Hurst and wavelet.
        
        Args:
            hurst_result: Hurst calculation result
            wavelet_result: Wavelet decomposition result
            
        Returns:
            Regime info dict or None
        """
        try:
            hurst = hurst_result.get('hurst', 0.5)
            is_trending = hurst_result.get('is_trending', False)

            if not is_trending:
                return None  # Not a trending market

            # Get trend strength from wavelet
            trend_component = wavelet_result.get('trend_component', [])
            if len(trend_component) < 10:
                return None

            # Calculate trend strength
            trend_change = trend_component[-1] - trend_component[-10]
            trend_strength = abs(trend_change) / (np.std(trend_component) + 1e-10)

            return {
                'is_trending': True,
                'hurst': float(hurst),
                'regime_type': 'TRENDING',
                'trend_strength': float(trend_strength),
                'trend_change': float(trend_change)
            }

        except Exception as e:
            self.logger.debug(f"[S25_HURST] Regime classification error: {e}")
            return None

    # =========================================================================
    # DIRECTION DETERMINATION
    # =========================================================================

    def _determine_direction(self, wavelet_result: Dict, df: pd.DataFrame) -> Optional[Dict]:
        """
        Determine trend direction from wavelet components.
        
        Args:
            wavelet_result: Wavelet decomposition result
            df: DataFrame with OHLCV data
            
        Returns:
            Direction info dict or None
        """
        try:
            trend_component = wavelet_result.get('trend_component', [])
            if len(trend_component) < 10:
                return None

            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            # Calculate trend direction from wavelet
            trend_slope = trend_component[-1] - trend_component[-5]

            # Confirm with price momentum
            recent_close = close[-20:]
            price_momentum = recent_close[-1] - recent_close[0]

            # Determine direction
            if trend_slope > 0 and price_momentum > 0:
                direction = 'BUY'
                strength = min(1.0, abs(trend_slope) / 5.0 + 0.3)
            elif trend_slope < 0 and price_momentum < 0:
                direction = 'SELL'
                strength = min(1.0, abs(trend_slope) / 5.0 + 0.3)
            else:
                return None  # Trend and momentum don't align

            return {
                'direction': direction,
                'strength': float(strength),
                'trend_slope': float(trend_slope),
                'price_momentum': float(price_momentum)
            }

        except Exception as e:
            self.logger.debug(f"[S25_HURST] Direction determination error: {e}")
            return None

    # =========================================================================
    # M5 CONFIRMATION
    # =========================================================================

    def _confirm_m5(self, df_m5: pd.DataFrame, direction_info: Dict) -> bool:
        """
        Confirm signal on M5 timeframe.
        
        Args:
            df_m5: M5 DataFrame
            direction_info: Direction information
            
        Returns:
            True if confirmed
        """
        if df_m5 is None or df_m5.empty or len(df_m5) < 20:
            return True  # Skip confirmation if no M5 data

        try:
            close = df_m5['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            direction = direction_info.get('direction', 'BUY')

            # Check M5 momentum aligns with trend direction
            recent_close = close[-10:]
            momentum = recent_close[-1] - recent_close[0]

            if direction == 'BUY':
                return momentum > 0  # Bullish momentum on M5
            else:
                return momentum < 0  # Bearish momentum on M5

        except Exception:
            return True  # Skip confirmation on error

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def _generate_signal(
        self, df: pd.DataFrame, direction_info: Dict, regime_info: Dict,
        regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            direction_info: Direction information
            regime_info: Regime information
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            direction = direction_info.get('direction', 'BUY')
            strength = direction_info.get('strength', 0.5)

            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            entry_price = close[-1]

            # Calculate ATR for stop loss
            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(np.abs(high[1:] - close[:-1]),
                                       np.abs(low[1:] - close[:-1])))
            atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)

            # Calculate Stop Loss
            if direction == 'BUY':
                sl_price = entry_price - atr * 2.0
            else:  # SELL
                sl_price = entry_price + atr * 2.0

            # Validate SL
            if sl_price <= 0 or sl_price == entry_price:
                return self._create_neutral_signal()

            # Calculate Take Profit
            tp_result = self.adaptive_tp_engine.calculate_adaptive_tp(
                df, entry_price, sl_price, direction == 'BUY',
                regime_context.get('regime_name', 'UNKNOWN') if regime_context else 'UNKNOWN'
            )

            if tp_result and tp_result.get('tp_price', 0) > 0:
                tp_price = tp_result['tp_price']
            else:
                # Fallback: Fixed R:R
                risk = abs(entry_price - sl_price)
                if direction == 'BUY':
                    tp_price = entry_price + risk * 2.0
                else:
                    tp_price = entry_price - risk * 2.0

            # Calculate confidence
            hurst_bonus = (regime_info.get('hurst', 0.5) - 0.5) * 0.5  # 0-0.175 bonus
            trend_bonus = regime_info.get('trend_strength', 0.5) * 0.1
            confidence = min(1.0, 0.4 + strength * 0.3 + hurst_bonus + trend_bonus)

            # Build signal
            signal = {
                'signal': f'{direction}_MARKET',
                'meta': {
                    'strategy': self.strategy_name,
                    'strategy_category': self.strategy_category,
                    'entry_price': round(entry_price, 2),
                    'sl_price': round(sl_price, 2),
                    'tp_price': round(tp_price, 2),
                    'confidence': confidence,
                    'hurst': regime_info.get('hurst', 0.5),
                    'trend_strength': regime_info.get('trend_strength', 0.5),
                    'trend_slope': direction_info.get('trend_slope', 0),
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit
                }
            }

            self.logger.info(
                f"[S25_HURST] Signal generated: {direction} | "
                f"Hurst: {regime_info.get('hurst', 0.5):.3f} | "
                f"Trend Strength: {regime_info.get('trend_strength', 0.5):.2f} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S25_HURST] Signal generation error: {e}")
            return self._create_neutral_signal()

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _is_regime_compatible(self, regime_context: Dict) -> bool:
        """
        Check if current regime is compatible with this strategy.
        
        Args:
            regime_context: Current regime information
            
        Returns:
            True if compatible
        """
        regime_name = regime_context.get('regime_name', 'UNKNOWN')

        # TREND strategies work best in trending regimes
        compatible_regimes = [
            'HEALTHY_UPTREND', 'HEALTHY_DOWNTREND',
            'QUIET_RALLY', 'SLOW_BLEED',
            'FALSE_SIDEWAY', 'PRE_BREAKOUT'
        ]

        return regime_name in compatible_regimes

    def _create_neutral_signal(self) -> Dict:
        """Create neutral signal."""
        return {
            'signal': 'NEUTRAL',
            'meta': {
                'strategy': self.strategy_name,
                'strategy_category': self.strategy_category,
                'confidence': 0.0
            }
        }