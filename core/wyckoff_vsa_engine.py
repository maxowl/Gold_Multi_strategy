"""
Wyckoff VSA Engine.

Provides Wyckoff Method and Volume Spread Analysis (VSA):
  - Volume Spread Analysis
  - Wyckoff phase detection
  - Spring detection (bear trap)
  - Upthrust detection (bull trap)
  - Effort vs Result analysis

Used by:
  - S22_WyckoffSpring (Wyckoff Spring strategy)
  - Wyckoff method analysis
  - Smart money detection
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple


class WyckoffVSAEngine:
    """
    Wyckoff VSA Analysis engine.
    
    Features:
      - Volume Spread Analysis (VSA)
      - Wyckoff phase detection
      - Spring detection (bear trap)
      - Upthrust detection (bull trap)
      - Effort vs Result analysis
    """

    def __init__(self):
        """Initialize WyckoffVSAEngine."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # VSA parameters
        self.vsa_lookback = 50  # Lookback for VSA analysis
        self.volume_threshold = 1.5  # High volume threshold
        self.spread_threshold = 0.7  # Narrow spread threshold

        # Wyckoff parameters
        self.phase_lookback = 100  # Lookback for phase detection
        self.spring_threshold = 0.005  # Spring depth threshold (0.5%)

    # =========================================================================
    # VOLUME SPREAD ANALYSIS (VSA)
    # =========================================================================

    def analyze_vsa(self, df: pd.DataFrame, lookback: int = None) -> Dict:
        """
        Perform Volume Spread Analysis (VSA).
        
        VSA analyzes the relationship between volume and price spread
        to identify smart money activity.
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Lookback period
            
        Returns:
            Dict with VSA analysis
        """
        if lookback is None:
            lookback = self.vsa_lookback

        if df is None or df.empty or len(df) < lookback:
            return {
                'signals': [],
                'current_signal': 'NEUTRAL',
                'volume_trend': 'UNKNOWN'
            }

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

            # Get volume
            volume = self._get_volume(df)
            if volume is None:
                return {
                    'signals': [],
                    'current_signal': 'NEUTRAL',
                    'volume_trend': 'UNKNOWN'
                }

            signals = []
            n = len(close)
            start_idx = max(1, n - lookback)

            # Calculate average volume and spread
            avg_volume = np.mean(volume[-lookback:])
            avg_spread = np.mean(high[-lookback:] - low[-lookback:])

            for i in range(start_idx, n):
                signal = self._analyze_single_bar_vsa(
                    i, close, high, low, open_, volume,
                    avg_volume, avg_spread
                )

                if signal:
                    signals.append(signal)

            # Get current signal
            current_signal = signals[-1]['signal'] if signals else 'NEUTRAL'

            # Determine volume trend
            recent_volume = volume[-20:]
            if len(recent_volume) >= 20:
                volume_first_half = np.mean(recent_volume[:10])
                volume_second_half = np.mean(recent_volume[10:])
                if volume_second_half > volume_first_half * 1.2:
                    volume_trend = 'INCREASING'
                elif volume_second_half < volume_first_half * 0.8:
                    volume_trend = 'DECREASING'
                else:
                    volume_trend = 'STABLE'
            else:
                volume_trend = 'UNKNOWN'

            return {
                'signals': signals,
                'current_signal': current_signal,
                'volume_trend': volume_trend,
                'signal_count': len(signals)
            }

        except Exception as e:
            self.logger.error(f"[WYCKOFF] VSA analysis error: {e}")
            return {
                'signals': [],
                'current_signal': 'NEUTRAL',
                'volume_trend': 'UNKNOWN'
            }

    def _analyze_single_bar_vsa(
        self, i: int, close: np.ndarray, high: np.ndarray, low: np.ndarray,
        open_: np.ndarray, volume: np.ndarray, avg_volume: float, avg_spread: float
    ) -> Optional[Dict]:
        """Analyze single bar for VSA signals."""
        try:
            # Calculate bar metrics
            spread = high[i] - low[i]
            body = abs(close[i] - open_[i])
            volume_ratio = volume[i] / avg_volume if avg_volume > 0 else 1.0
            spread_ratio = spread / avg_spread if avg_spread > 0 else 1.0

            # Determine bar direction
            is_bullish = close[i] > open_[i]
            is_bearish = close[i] < open_[i]

            # Calculate close position in range
            if spread > 0:
                close_position = (close[i] - low[i]) / spread
            else:
                close_position = 0.5

            # VSA Signal Detection

            # 1. High Volume Bullish Bar (Buying Climax or Strength)
            if is_bullish and volume_ratio > self.volume_threshold:
                if close_position > 0.7:
                    return {
                        'index': i,
                        'signal': 'BUYING_STRENGTH',
                        'type': 'BULLISH',
                        'volume_ratio': float(volume_ratio),
                        'spread_ratio': float(spread_ratio),
                        'strength': float(volume_ratio / self.volume_threshold)
                    }
                else:
                    return {
                        'index': i,
                        'signal': 'BUYING_CLIMAX',
                        'type': 'BEARISH_REVERSAL',
                        'volume_ratio': float(volume_ratio),
                        'spread_ratio': float(spread_ratio),
                        'strength': float(volume_ratio / self.volume_threshold)
                    }

            # 2. High Volume Bearish Bar (Selling Climax or Weakness)
            if is_bearish and volume_ratio > self.volume_threshold:
                if close_position < 0.3:
                    return {
                        'index': i,
                        'signal': 'SELLING_STRENGTH',
                        'type': 'BEARISH',
                        'volume_ratio': float(volume_ratio),
                        'spread_ratio': float(spread_ratio),
                        'strength': float(volume_ratio / self.volume_threshold)
                    }
                else:
                    return {
                        'index': i,
                        'signal': 'SELLING_CLIMAX',
                        'type': 'BULLISH_REVERSAL',
                        'volume_ratio': float(volume_ratio),
                        'spread_ratio': float(spread_ratio),
                        'strength': float(volume_ratio / self.volume_threshold)
                    }

            # 3. Low Volume Narrow Spread (No Supply/No Demand)
            if volume_ratio < 0.7 and spread_ratio < self.spread_threshold:
                if is_bullish:
                    return {
                        'index': i,
                        'signal': 'NO_SUPPLY',
                        'type': 'BULLISH',
                        'volume_ratio': float(volume_ratio),
                        'spread_ratio': float(spread_ratio),
                        'strength': float(1.0 - volume_ratio)
                    }
                else:
                    return {
                        'index': i,
                        'signal': 'NO_DEMAND',
                        'type': 'BEARISH',
                        'volume_ratio': float(volume_ratio),
                        'spread_ratio': float(spread_ratio),
                        'strength': float(1.0 - volume_ratio)
                    }

            # 4. High Volume Narrow Spread (Stopping Volume)
            if volume_ratio > self.volume_threshold and spread_ratio < self.spread_threshold:
                return {
                    'index': i,
                    'signal': 'STOPPING_VOLUME',
                    'type': 'REVERSAL',
                    'volume_ratio': float(volume_ratio),
                    'spread_ratio': float(spread_ratio),
                    'strength': float(volume_ratio / self.volume_threshold)
                }

            return None

        except Exception:
            return None

    # =========================================================================
    # WYCKOFF PHASE DETECTION
    # =========================================================================

    def detect_wyckoff_phase(self, df: pd.DataFrame, lookback: int = None) -> Dict:
        """
        Detect current Wyckoff phase.
        
        Wyckoff Phases:
          - Phase A: Stopping the previous trend
          - Phase B: Building a cause (accumulation/distribution)
          - Phase C: Test (Spring or Upthrust)
          - Phase D: Trend within the range
          - Phase E: Trend outside the range
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Lookback period
            
        Returns:
            Dict with Wyckoff phase analysis
        """
        if lookback is None:
            lookback = self.phase_lookback

        if df is None or df.empty or len(df) < lookback:
            return {
                'phase': 'UNKNOWN',
                'phase_description': 'Insufficient data',
                'is_accumulation': False,
                'is_distribution': False
            }

        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            # Use lookback window
            recent_close = close[-lookback:]
            recent_high = high[-lookback:]
            recent_low = low[-lookback:]

            # Calculate range
            range_high = np.max(recent_high)
            range_low = np.min(recent_low)
            range_width = range_high - range_low

            if range_width <= 0:
                return {
                    'phase': 'UNKNOWN',
                    'phase_description': 'No range detected',
                    'is_accumulation': False,
                    'is_distribution': False
                }

            # Current price position in range
            current_price = close[-1]
            price_position = (current_price - range_low) / range_width

            # Calculate price trend over lookback
            price_change = (recent_close[-1] - recent_close[0]) / recent_close[0] * 100

            # Determine phase based on price position and trend
            if price_position < 0.3 and price_change > -2:
                phase = 'PHASE_C_ACCUMULATION'
                phase_description = 'Spring/Test phase in accumulation'
                is_accumulation = True
                is_distribution = False
            elif price_position > 0.7 and price_change < 2:
                phase = 'PHASE_C_DISTRIBUTION'
                phase_description = 'Upthrust/Test phase in distribution'
                is_accumulation = False
                is_distribution = True
            elif 0.3 <= price_position <= 0.7:
                phase = 'PHASE_B'
                phase_description = 'Building cause (range-bound)'
                is_accumulation = price_change < 0
                is_distribution = price_change > 0
            elif price_position < 0.3 and price_change < -5:
                phase = 'PHASE_A_DISTRIBUTION'
                phase_description = 'Stopping downtrend (potential distribution)'
                is_accumulation = False
                is_distribution = True
            elif price_position > 0.7 and price_change > 5:
                phase = 'PHASE_A_ACCUMULATION'
                phase_description = 'Stopping uptrend (potential accumulation)'
                is_accumulation = True
                is_distribution = False
            else:
                phase = 'PHASE_D'
                phase_description = 'Trend within range'
                is_accumulation = price_change < 0
                is_distribution = price_change > 0

            return {
                'phase': phase,
                'phase_description': phase_description,
                'is_accumulation': is_accumulation,
                'is_distribution': is_distribution,
                'price_position': float(price_position),
                'price_change_pct': float(price_change),
                'range_high': float(range_high),
                'range_low': float(range_low)
            }

        except Exception as e:
            self.logger.error(f"[WYCKOFF] Phase detection error: {e}")
            return {
                'phase': 'UNKNOWN',
                'phase_description': f'Error: {str(e)}',
                'is_accumulation': False,
                'is_distribution': False
            }

    # =========================================================================
    # SPRING DETECTION
    # =========================================================================

    def detect_spring(self, df: pd.DataFrame, lookback: int = 50) -> Optional[Dict]:
        """
        Detect Wyckoff Spring (bear trap).
        
        Spring occurs when:
          - Price breaks below support
          - Quickly reverses back above support
          - Indicates smart money accumulation
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Lookback period
            
        Returns:
            Dict with spring detection, or None if no spring
        """
        if df is None or df.empty or len(df) < lookback:
            return None

        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            # Find support level (recent swing low)
            recent_low = low[-lookback:]
            support_level = np.min(recent_low[:-10])  # Exclude last 10 bars

            # Check for spring in recent bars
            n = len(close)
            for i in range(n - 10, n):
                if i < 1:
                    continue

                # Check if price broke below support
                if low[i] < support_level:
                    # Check if price recovered above support
                    if close[i] > support_level or (i + 1 < n and close[i + 1] > support_level):
                        # Calculate spring depth
                        spring_depth = (support_level - low[i]) / support_level

                        if spring_depth > self.spring_threshold:
                            return {
                                'type': 'SPRING',
                                'index': i,
                                'support_level': float(support_level),
                                'spring_low': float(low[i]),
                                'spring_depth_pct': float(spring_depth * 100),
                                'recovered': close[i] > support_level,
                                'strength': float(spring_depth / self.spring_threshold)
                            }

            return None

        except Exception as e:
            self.logger.error(f"[WYCKOFF] Spring detection error: {e}")
            return None

    # =========================================================================
    # UPTHRUST DETECTION
    # =========================================================================

    def detect_upthrust(self, df: pd.DataFrame, lookback: int = 50) -> Optional[Dict]:
        """
        Detect Wyckoff Upthrust (bull trap).
        
        Upthrust occurs when:
          - Price breaks above resistance
          - Quickly reverses back below resistance
          - Indicates smart money distribution
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Lookback period
            
        Returns:
            Dict with upthrust detection, or None if no upthrust
        """
        if df is None or df.empty or len(df) < lookback:
            return None

        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            # Find resistance level (recent swing high)
            recent_high = high[-lookback:]
            resistance_level = np.max(recent_high[:-10])  # Exclude last 10 bars

            # Check for upthrust in recent bars
            n = len(close)
            for i in range(n - 10, n):
                if i < 1:
                    continue

                # Check if price broke above resistance
                if high[i] > resistance_level:
                    # Check if price recovered below resistance
                    if close[i] < resistance_level or (i + 1 < n and close[i + 1] < resistance_level):
                        # Calculate upthrust height
                        upthrust_height = (high[i] - resistance_level) / resistance_level

                        if upthrust_height > self.spring_threshold:
                            return {
                                'type': 'UPTHRUST',
                                'index': i,
                                'resistance_level': float(resistance_level),
                                'upthrust_high': float(high[i]),
                                'upthrust_height_pct': float(upthrust_height * 100),
                                'recovered': close[i] < resistance_level,
                                'strength': float(upthrust_height / self.spring_threshold)
                            }

            return None

        except Exception as e:
            self.logger.error(f"[WYCKOFF] Upthrust detection error: {e}")
            return None

    # =========================================================================
    # EFFORT VS RESULT ANALYSIS
    # =========================================================================

    def analyze_effort_result(self, df: pd.DataFrame, lookback: int = 20) -> Dict:
        """
        Analyze Effort vs Result relationship.
        
        Effort = Volume
        Result = Price movement
        
        Divergence between effort and result indicates smart money activity.
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Lookback period
            
        Returns:
            Dict with effort vs result analysis
        """
        if df is None or df.empty or len(df) < lookback:
            return {
                'effort_result': 'UNKNOWN',
                'divergence': False,
                'effort': 0.0,
                'result': 0.0
            }

        try:
            close = df['close'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))

            # Get volume
            volume = self._get_volume(df)
            if volume is None:
                return {
                    'effort_result': 'UNKNOWN',
                    'divergence': False,
                    'effort': 0.0,
                    'result': 0.0
                }

            # Use lookback window
            recent_close = close[-lookback:]
            recent_volume = volume[-lookback:]

            # Calculate effort (volume trend)
            avg_volume_first_half = np.mean(recent_volume[:lookback//2])
            avg_volume_second_half = np.mean(recent_volume[lookback//2:])

            if avg_volume_first_half > 0:
                effort = avg_volume_second_half / avg_volume_first_half
            else:
                effort = 1.0

            # Calculate result (price movement)
            price_change = abs(recent_close[-1] - recent_close[0]) / recent_close[0] * 100

            # Normalize result
            result = min(2.0, price_change / 2.0)  # Cap at 2.0

            # Determine effort vs result relationship
            if effort > 1.2 and result < 0.5:
                effort_result = 'HIGH_EFFORT_LOW_RESULT'
                divergence = True
            elif effort < 0.8 and result > 1.0:
                effort_result = 'LOW_EFFORT_HIGH_RESULT'
                divergence = True
            elif effort > 1.2 and result > 1.0:
                effort_result = 'HIGH_EFFORT_HIGH_RESULT'
                divergence = False
            elif effort < 0.8 and result < 0.5:
                effort_result = 'LOW_EFFORT_LOW_RESULT'
                divergence = False
            else:
                effort_result = 'BALANCED'
                divergence = False

            return {
                'effort_result': effort_result,
                'divergence': divergence,
                'effort': float(effort),
                'result': float(result),
                'price_change_pct': float(price_change)
            }

        except Exception as e:
            self.logger.error(f"[WYCKOFF] Effort vs Result error: {e}")
            return {
                'effort_result': 'UNKNOWN',
                'divergence': False,
                'effort': 0.0,
                'result': 0.0
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

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_wyckoff_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive Wyckoff analysis.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with complete Wyckoff analysis
        """
        result = {
            'vsa': None,
            'phase': None,
            'spring': None,
            'upthrust': None,
            'effort_result': None
        }

        if df is None or df.empty or len(df) < 50:
            return result

        try:
            # Perform VSA analysis
            result['vsa'] = self.analyze_vsa(df)

            # Detect Wyckoff phase
            result['phase'] = self.detect_wyckoff_phase(df)

            # Detect spring
            result['spring'] = self.detect_spring(df)

            # Detect upthrust
            result['upthrust'] = self.detect_upthrust(df)

            # Analyze effort vs result
            result['effort_result'] = self.analyze_effort_result(df)

            return result

        except Exception as e:
            self.logger.error(f"[WYCKOFF] Analysis error: {e}")
            return result

    def format_wyckoff_log(self, analysis_result: Dict) -> str:
        """
        Format Wyckoff analysis result as concise log string.
        
        Args:
            analysis_result: Result from get_wyckoff_analysis
            
        Returns:
            Formatted log string
        """
        if analysis_result is None:
            return "[WYCKOFF] Analysis failed"

        vsa = analysis_result.get('vsa', {})
        phase = analysis_result.get('phase', {})
        spring = analysis_result.get('spring')
        upthrust = analysis_result.get('upthrust')

        vsa_signal = vsa.get('current_signal', 'NEUTRAL') if vsa else 'NEUTRAL'
        phase_str = phase.get('phase', 'UNKNOWN') if phase else 'UNKNOWN'
        spring_str = "YES" if spring else "NO"
        upthrust_str = "YES" if upthrust else "NO"

        return (
            f"[WYCKOFF] VSA: {vsa_signal} | "
            f"Phase: {phase_str} | "
            f"Spring: {spring_str} | "
            f"Upthrust: {upthrust_str}"
        )

    def get_wyckoff_signal(self, analysis_result: Dict) -> Dict:
        """
        Get Wyckoff-based trading signal.
        
        Args:
            analysis_result: Result from get_wyckoff_analysis
            
        Returns:
            Dict with Wyckoff signal
        """
        if analysis_result is None:
            return {'signal': 'NEUTRAL', 'reason': 'No data'}

        spring = analysis_result.get('spring')
        upthrust = analysis_result.get('upthrust')
        phase = analysis_result.get('phase', {})
        effort_result = analysis_result.get('effort_result', {})

        # Spring is a strong buy signal
        if spring and spring.get('recovered', False):
            return {
                'signal': 'WYCKOFF_SPRING_BUY',
                'reason': f"Spring detected at {spring.get('spring_low', 0):.2f}, depth {spring.get('spring_depth_pct', 0):.2f}%",
                'strength': spring.get('strength', 0.5)
            }

        # Upthrust is a strong sell signal
        if upthrust and upthrust.get('recovered', False):
            return {
                'signal': 'WYCKOFF_UPTHRUST_SELL',
                'reason': f"Upthrust detected at {upthrust.get('upthrust_high', 0):.2f}, height {upthrust.get('upthrust_height_pct', 0):.2f}%",
                'strength': upthrust.get('strength', 0.5)
            }

        # Phase-based signals
        if phase.get('is_accumulation', False):
            return {
                'signal': 'ACCUMULATION_BIAS',
                'reason': phase.get('phase_description', 'Accumulation phase'),
                'strength': 0.3
            }
        elif phase.get('is_distribution', False):
            return {
                'signal': 'DISTRIBUTION_BIAS',
                'reason': phase.get('phase_description', 'Distribution phase'),
                'strength': 0.3
            }

        # Effort vs Result divergence
        if effort_result.get('divergence', False):
            if effort_result.get('effort_result') == 'HIGH_EFFORT_LOW_RESULT':
                return {
                    'signal': 'EFFORT_DIVERGENCE_BEARISH',
                    'reason': 'High effort, low result - potential distribution',
                    'strength': 0.4
                }
            elif effort_result.get('effort_result') == 'LOW_EFFORT_HIGH_RESULT':
                return {
                    'signal': 'EFFORT_DIVERGENCE_BULLISH',
                    'reason': 'Low effort, high result - potential accumulation',
                    'strength': 0.4
                }

        return {
            'signal': 'NEUTRAL',
            'reason': 'No Wyckoff signal',
            'strength': 0.0
        }