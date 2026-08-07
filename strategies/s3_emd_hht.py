"""
S3_EMD_HHT - Empirical Mode Decomposition + Hilbert-Huang Transform Strategy.

Trend-following strategy that uses EMD to decompose price into
Intrinsic Mode Functions (IMFs) and HHT to analyze instantaneous
phase and frequency.

Strategy Logic:
  1. Decompose price using EMD into multiple IMFs
  2. Apply Hilbert Transform to calculate instantaneous phase
  3. Detect dominant cycles from IMF analysis
  4. Identify trend direction from low-frequency IMFs
  5. Generate entry signal based on phase alignment

Used Engines:
  - DSPEngine: EMD and Hilbert Transform
  - AdaptiveTPEngine: Dynamic TP calculation

Strategy Category: TREND
Timeframe: M15 (primary), M5 (confirmation)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from core.base_strategy import BaseStrategy
from core.dsp_engine import DSPEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S3_EMD_HHT(BaseStrategy):
    """
    EMD + Hilbert-Huang Transform Strategy.
    
    This strategy uses advanced signal processing techniques to
    identify trend direction and cycle phase.
    
    EMD (Empirical Mode Decomposition):
      Decomposes a signal into Intrinsic Mode Functions (IMFs),
      each representing a different frequency scale.
      
    HHT (Hilbert-Huang Transform):
      Applies Hilbert Transform to IMFs to calculate instantaneous
      phase, frequency, and amplitude.
      
    Entry Criteria:
      - Dominant cycle detected
      - Phase alignment with trend direction
      - Low-frequency IMFs confirm trend
      - Momentum confirmation
    """

    def __init__(self):
        """Initialize S3_EMD_HHT strategy."""
        super().__init__(
            strategy_name='S3_EMD_HHT',
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
        self.dsp_engine = DSPEngine()
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.emd_imfs = 4  # Number of IMFs to extract
        self.cycle_period_min = 10  # Minimum cycle period
        self.cycle_period_max = 50  # Maximum cycle period
        self.phase_threshold = 0.5  # Phase alignment threshold

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
        Main analysis method for S3_EMD_HHT.
        
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
            # STEP 1: Apply EMD
            # =========================================================================
            close = df_m15['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            imfs = self.dsp_engine.empirical_mode_decomposition(close, max_imfs=self.emd_imfs)

            if imfs is None or len(imfs) < 2:
                return default_signal

            # =========================================================================
            # STEP 2: Apply Hilbert Transform
            # =========================================================================
            hilbert_result = self._apply_hilbert_transform(imfs)

            if hilbert_result is None:
                return default_signal

            # =========================================================================
            # STEP 3: Detect Dominant Cycle
            # =========================================================================
            cycle_info = self._detect_dominant_cycle(imfs, hilbert_result)

            if cycle_info is None or not cycle_info.get('cycle_detected', False):
                return default_signal

            # =========================================================================
            # STEP 4: Identify Trend
            # =========================================================================
            trend_info = self._identify_trend(imfs, close)

            if trend_info is None:
                return default_signal

            # =========================================================================
            # STEP 5: Check Phase Alignment
            # =========================================================================
            if not self._check_phase_alignment(hilbert_result, trend_info):
                return default_signal

            # =========================================================================
            # STEP 6: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, trend_info, cycle_info, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S3_EMD] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # HILBERT TRANSFORM
    # =========================================================================

    def _apply_hilbert_transform(self, imfs: List[np.ndarray]) -> Optional[Dict]:
        """
        Apply Hilbert Transform to IMFs.
        
        Args:
            imfs: List of IMF arrays
            
        Returns:
            Dict with Hilbert transform results, or None on failure
        """
        try:
            results = []

            for i, imf in enumerate(imfs):
                # Calculate instantaneous phase
                phase = self.dsp_engine.hilbert_phase(imf)

                # Calculate instantaneous frequency
                frequency = self.dsp_engine.hilbert_frequency(imf)

                if phase is not None and frequency is not None:
                    results.append({
                        'imf_index': i,
                        'imf': imf,
                        'phase': phase,
                        'frequency': frequency,
                        'current_phase': float(phase[-1]) if len(phase) > 0 else 0.0,
                        'current_frequency': float(frequency[-1]) if len(frequency) > 0 else 0.0
                    })

            if not results:
                return None

            return {
                'imfs': results,
                'num_imfs': len(results)
            }

        except Exception as e:
            self.logger.debug(f"[S3_EMD] Hilbert transform error: {e}")
            return None

    # =========================================================================
    # CYCLE DETECTION
    # =========================================================================

    def _detect_dominant_cycle(self, imfs: List[np.ndarray], hilbert_result: Dict) -> Optional[Dict]:
        """
        Detect dominant cycle from IMFs.
        
        Args:
            imfs: List of IMF arrays
            hilbert_result: Hilbert transform results
            
        Returns:
            Dict with cycle information, or None on failure
        """
        try:
            # Use the second IMF (typically represents dominant cycle)
            if len(imfs) < 2:
                return None

            dominant_imf = imfs[1]  # Second IMF

            # Calculate autocorrelation to find cycle period
            cycle_period = self._detect_cycle_period(dominant_imf)

            if cycle_period is None or cycle_period < self.cycle_period_min or cycle_period > self.cycle_period_max:
                return None

            # Get phase information
            imf_results = hilbert_result.get('imfs', [])
            if len(imf_results) < 2:
                return None

            current_phase = imf_results[1].get('current_phase', 0.0)
            current_frequency = imf_results[1].get('current_frequency', 0.0)

            return {
                'cycle_detected': True,
                'cycle_period': cycle_period,
                'current_phase': current_phase,
                'current_frequency': current_frequency,
                'dominant_imf_index': 1
            }

        except Exception as e:
            self.logger.debug(f"[S3_EMD] Cycle detection error: {e}")
            return None

    def _detect_cycle_period(self, imf: np.ndarray) -> Optional[int]:
        """Detect cycle period using autocorrelation."""
        try:
            n = len(imf)
            if n < 20:
                return None

            # Calculate autocorrelation
            imf_centered = imf - np.mean(imf)
            autocorr = np.correlate(imf_centered, imf_centered, mode='full')
            autocorr = autocorr[n-1:]  # Positive lags
            autocorr = autocorr / autocorr[0]  # Normalize

            # Find first peak after minimum lag
            min_lag = self.cycle_period_min
            max_lag = min(self.cycle_period_max, n // 2)

            best_lag = None
            best_corr = 0

            for lag in range(min_lag, max_lag):
                if autocorr[lag] > best_corr and autocorr[lag] > autocorr[lag-1] and autocorr[lag] > autocorr[lag+1]:
                    best_corr = autocorr[lag]
                    best_lag = lag

            return best_lag

        except Exception:
            return None

    # =========================================================================
    # TREND IDENTIFICATION
    # =========================================================================

    def _identify_trend(self, imfs: List[np.ndarray], close: np.ndarray) -> Optional[Dict]:
        """
        Identify trend direction from low-frequency IMFs.
        
        Args:
            imfs: List of IMF arrays
            close: Close price array
            
        Returns:
            Dict with trend information, or None on failure
        """
        try:
            # Low-frequency IMFs (last 2) represent trend
            if len(imfs) < 3:
                return None

            # Sum of low-frequency IMFs = trend
            trend_imfs = imfs[-2:]  # Last 2 IMFs
            trend = np.sum(trend_imfs, axis=0)

            # Calculate trend direction
            current_trend = trend[-1]
            prev_trend = trend[-10] if len(trend) > 10 else trend[0]

            # Trend slope
            trend_slope = current_trend - prev_trend

            # Determine direction
            if trend_slope > 0:
                direction = 'BUY'
                trend_strength = min(1.0, trend_slope / 10.0)
            elif trend_slope < 0:
                direction = 'SELL'
                trend_strength = min(1.0, abs(trend_slope) / 10.0)
            else:
                return None  # No clear trend

            # Check trend strength
            if trend_strength < 0.3:
                return None  # Weak trend

            return {
                'direction': direction,
                'trend_strength': float(trend_strength),
                'trend_slope': float(trend_slope),
                'current_trend': float(current_trend)
            }

        except Exception as e:
            self.logger.debug(f"[S3_EMD] Trend identification error: {e}")
            return None

    # =========================================================================
    # PHASE ALIGNMENT
    # =========================================================================

    def _check_phase_alignment(self, hilbert_result: Dict, trend_info: Dict) -> bool:
        """
        Check if phase is aligned with trend direction.
        
        Args:
            hilbert_result: Hilbert transform results
            trend_info: Trend information
            
        Returns:
            True if phase is aligned
        """
        try:
            imf_results = hilbert_result.get('imfs', [])
            if len(imf_results) < 2:
                return False

            # Get phase from dominant IMF
            current_phase = imf_results[1].get('current_phase', 0.0)

            # Normalize phase to [0, 2*pi]
            phase_normalized = current_phase % (2 * np.pi)

            direction = trend_info.get('direction', 'BUY')

            # For BUY: Phase should be in rising portion (0 to pi)
            # For SELL: Phase should be in falling portion (pi to 2*pi)
            if direction == 'BUY':
                # Rising phase: 0 to pi
                aligned = 0 < phase_normalized < np.pi
            else:  # SELL
                # Falling phase: pi to 2*pi
                aligned = np.pi < phase_normalized < 2 * np.pi

            return aligned

        except Exception:
            return False

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def _generate_signal(
        self, df: pd.DataFrame, trend_info: Dict, cycle_info: Dict,
        regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal based on EMD-HHT analysis.
        
        Args:
            df: DataFrame with OHLCV data
            trend_info: Trend information
            cycle_info: Cycle information
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            direction = trend_info.get('direction', 'BUY')
            trend_strength = trend_info.get('trend_strength', 0.5)

            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))
            entry_price = close[-1]

            # Calculate Stop Loss based on ATR
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            # ATR
            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(np.abs(high[1:] - close[:-1]),
                                       np.abs(low[1:] - close[:-1])))
            atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)

            # SL based on ATR
            if direction == 'BUY':
                sl_price = entry_price - atr * 1.5
            else:  # SELL
                sl_price = entry_price + atr * 1.5

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
            cycle_strength = 0.7 if cycle_info.get('cycle_detected', False) else 0.5
            confidence = min(1.0, trend_strength * 0.6 + cycle_strength * 0.4)

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
                    'trend_strength': trend_strength,
                    'cycle_period': cycle_info.get('cycle_period', 0),
                    'cycle_phase': cycle_info.get('current_phase', 0),
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit
                }
            }

            self.logger.info(
                f"[S3_EMD] Signal generated: {direction} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Cycle: {cycle_info.get('cycle_period', 0)} bars | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S3_EMD] Signal generation error: {e}")
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