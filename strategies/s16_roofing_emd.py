"""
S16_RoofingEMD - Roofing Filter + EMD Strategy.

Mean-reversion strategy that combines Ehlers Roofing filter with
Empirical Mode Decomposition (EMD) for cycle-based trading.

Strategy Logic:
  1. Apply Roofing filter to remove trend and noise
  2. Decompose filtered signal using EMD
  3. Detect dominant cycles from decomposition
  4. Generate entry signal on cycle extremes

Roofing Filter:
  The Roofing filter is a band-pass filter that removes both
  trend (low frequency) and noise (high frequency), leaving
  only the cycle component.
  
  Formula:
    High-pass removes trend
    Low-pass removes noise
    Result = Band-pass filtered signal

EMD (Empirical Mode Decomposition):
  Decomposes a signal into Intrinsic Mode Functions (IMFs),
  each representing a different frequency scale.
  
  For cycle detection:
    - IMF 1: High frequency (noise)
    - IMF 2: Dominant cycle
    - IMF 3+: Low frequency (trend)

Used Engines:
  - EhlersDSPEngine: Roofing filter
  - DSPEngine: EMD decomposition
  - AdaptiveTPEngine: Dynamic TP calculation

Strategy Category: MEAN_REVERSION
Timeframe: M15 (primary), M5 (confirmation)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from core.base_strategy import BaseStrategy
from core.dsp_ehlers_engine import EhlersDSPEngine
from core.dsp_engine import DSPEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S16_RoofingEMD(BaseStrategy):
    """
    Roofing Filter + EMD Strategy.
    
    This strategy combines Ehlers Roofing filter with EMD
    for cycle-based mean reversion trading.
    
    Roofing Filter Definition:
      A band-pass filter that removes trend (high-pass) and
      noise (low-pass), isolating the cycle component.
      
    EMD Definition:
      Empirical Mode Decomposition decomposes a signal into
      Intrinsic Mode Functions (IMFs), each representing a
      different frequency scale.
      
    Entry Criteria:
      - Roofing filter applied successfully
      - Dominant cycle detected
      - Price at cycle extreme
      - Confirmation from M5 timeframe
    """

    def __init__(self):
        """Initialize S16_RoofingEMD strategy."""
        super().__init__(
            strategy_name='S16_RoofingEMD',
            strategy_category='MEAN_REVERSION',
            timeframes=['M15', 'M5'],
            risk_per_trade_pct=0.4,
            min_rr_ratio=1.5,
            max_spread_points=30,
            trailing_enabled=False,  # No trailing for mean reversion
            partial_close_enabled=False,  # No partial close
            requires_dynamic_exit=True
        )

        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize engines
        self.ehlers_engine = EhlersDSPEngine()
        self.dsp_engine = DSPEngine()
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.roofing_hp_period = 48  # High-pass period
        self.roofing_lp_period = 10  # Low-pass period
        self.emd_imfs = 3  # Number of IMFs to extract
        self.cycle_threshold = 0.7  # Cycle strength threshold

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
        Main analysis method for S16_RoofingEMD.
        
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
            # STEP 1: Apply Roofing Filter
            # =========================================================================
            close = df_m15['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            roofing_result = self.ehlers_engine.roofing_filter(
                close, hp_period=self.roofing_hp_period, lp_period=self.roofing_lp_period
            )

            if roofing_result is None:
                return default_signal

            # =========================================================================
            # STEP 2: Apply EMD to Filtered Signal
            # =========================================================================
            imfs = self.dsp_engine.empirical_mode_decomposition(
                roofing_result, max_imfs=self.emd_imfs
            )

            if imfs is None or len(imfs) < 2:
                return default_signal

            # =========================================================================
            # STEP 3: Detect Cycle
            # =========================================================================
            cycle_info = self._detect_cycle(imfs, roofing_result)

            if cycle_info is None or not cycle_info.get('cycle_detected', False):
                return default_signal

            # =========================================================================
            # STEP 4: Detect Reversion
            # =========================================================================
            reversion = self._detect_reversion(cycle_info, roofing_result, close)

            if reversion is None or not reversion.get('signal_detected', False):
                return default_signal

            # =========================================================================
            # STEP 5: M5 Confirmation
            # =========================================================================
            if df_m5 is not None and not df_m5.empty:
                if not self._confirm_m5(df_m5, reversion):
                    return default_signal

            # =========================================================================
            # STEP 6: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, reversion, cycle_info, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S16_ROOF] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # CYCLE DETECTION
    # =========================================================================

    def _detect_cycle(self, imfs: List[np.ndarray], roofing: np.ndarray) -> Optional[Dict]:
        """
        Detect cycle from IMFs and roofing filter.
        
        Args:
            imfs: List of IMF arrays
            roofing: Roofing filtered signal
            
        Returns:
            Cycle info dict or None
        """
        try:
            # Use IMF 2 (dominant cycle)
            if len(imfs) < 2:
                return None

            dominant_imf = imfs[1]  # Second IMF is typically dominant cycle

            # Calculate autocorrelation for cycle period
            cycle_period = self._detect_cycle_period(dominant_imf)

            if cycle_period is None or cycle_period < 10 or cycle_period > 50:
                return None

            # Calculate cycle strength
            cycle_strength = np.std(dominant_imf) / (np.std(roofing) + 1e-10)

            if cycle_strength < self.cycle_threshold:
                return None

            return {
                'cycle_detected': True,
                'cycle_period': cycle_period,
                'cycle_strength': float(cycle_strength),
                'dominant_imf': dominant_imf,
                'roofing': roofing
            }

        except Exception as e:
            self.logger.debug(f"[S16_ROOF] Cycle detection error: {e}")
            return None

    def _detect_cycle_period(self, imf: np.ndarray) -> Optional[int]:
        """Detect cycle period using autocorrelation."""
        try:
            n = len(imf)
            if n < 30:
                return None

            # Calculate autocorrelation
            imf_centered = imf - np.mean(imf)
            autocorr = np.correlate(imf_centered, imf_centered, mode='full')
            autocorr = autocorr[n - 1:]  # Positive lags
            autocorr = autocorr / autocorr[0]  # Normalize

            # Find first peak after minimum lag
            min_lag = 10
            max_lag = min(50, n // 2)

            best_lag = None
            best_corr = 0

            for lag in range(min_lag, max_lag):
                if autocorr[lag] > best_corr and autocorr[lag] > autocorr[lag - 1] and autocorr[lag] > autocorr[lag + 1]:
                    best_corr = autocorr[lag]
                    best_lag = lag

            return best_lag

        except Exception:
            return None

    # =========================================================================
    # REVERSION DETECTION
    # =========================================================================

    def _detect_reversion(
        self, cycle_info: Dict, roofing: np.ndarray, close: np.ndarray
    ) -> Optional[Dict]:
        """
        Detect mean reversion signal at cycle extremes.
        
        Args:
            cycle_info: Cycle information
            roofing: Roofing filtered signal
            close: Close price array
            
        Returns:
            Reversion dict or None
        """
        try:
            dominant_imf = cycle_info.get('dominant_imf')
            if dominant_imf is None:
                return None

            current_price = close[-1]
            current_roofing = roofing[-1]
            current_imf = dominant_imf[-1]

            # Calculate IMF statistics
            imf_mean = np.mean(dominant_imf)
            imf_std = np.std(dominant_imf)

            if imf_std == 0:
                return None

            # Calculate z-score of current IMF value
            zscore = (current_imf - imf_mean) / imf_std

            # Determine direction based on z-score
            if zscore > 1.5:
                # IMF at high extreme → expect reversion down → SELL
                direction = 'SELL'
                signal_strength = min(1.0, zscore / 2.0)
            elif zscore < -1.5:
                # IMF at low extreme → expect reversion up → BUY
                direction = 'BUY'
                signal_strength = min(1.0, abs(zscore) / 2.0)
            else:
                return None  # Not at extreme

            return {
                'signal_detected': True,
                'direction': direction,
                'zscore': float(zscore),
                'signal_strength': float(signal_strength),
                'current_imf': float(current_imf),
                'imf_mean': float(imf_mean)
            }

        except Exception as e:
            self.logger.debug(f"[S16_ROOF] Reversion detection error: {e}")
            return None

    # =========================================================================
    # M5 CONFIRMATION
    # =========================================================================

    def _confirm_m5(self, df_m5: pd.DataFrame, reversion: Dict) -> bool:
        """
        Confirm signal on M5 timeframe.
        
        Args:
            df_m5: M5 DataFrame
            reversion: Reversion dict
            
        Returns:
            True if confirmed
        """
        if df_m5 is None or df_m5.empty or len(df_m5) < 20:
            return True  # Skip confirmation if no M5 data

        try:
            close = df_m5['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            direction = reversion.get('direction', 'BUY')

            # Check M5 momentum for reversal signs
            recent_close = close[-10:]
            momentum = recent_close[-1] - recent_close[0]

            if direction == 'BUY':
                # For BUY: M5 should show slowing decline or reversal
                return momentum > -0.1 * abs(recent_close[0])
            else:  # SELL
                # For SELL: M5 should show slowing rise or reversal
                return momentum < 0.1 * abs(recent_close[0])

        except Exception:
            return True  # Skip confirmation on error

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def _generate_signal(
        self, df: pd.DataFrame, reversion: Dict, cycle_info: Dict,
        regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            reversion: Reversion dict
            cycle_info: Cycle information
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            direction = reversion.get('direction', 'BUY')
            signal_strength = reversion.get('signal_strength', 0.5)

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
                    tp_price = entry_price + risk * 1.5
                else:
                    tp_price = entry_price - risk * 1.5

            # Calculate confidence
            cycle_strength_bonus = cycle_info.get('cycle_strength', 0.5) * 0.2
            confidence = min(1.0, 0.4 + signal_strength * 0.4 + cycle_strength_bonus)

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
                    'zscore': reversion.get('zscore', 0),
                    'cycle_period': cycle_info.get('cycle_period', 0),
                    'cycle_strength': cycle_info.get('cycle_strength', 0),
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit
                }
            }

            self.logger.info(
                f"[S16_ROOF] Signal generated: {direction} | "
                f"Z-score: {reversion.get('zscore', 0):.2f} | "
                f"Cycle: {cycle_info.get('cycle_period', 0)} bars | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S16_ROOF] Signal generation error: {e}")
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

        # MEAN_REVERSION strategies work best in ranging regimes
        compatible_regimes = [
            'CLASSIC_RANGE', 'TIGHT_RANGE',
            'CONSOLIDATING_BULL', 'CONSOLIDATING_BEAR',
            'OVERSOLD_BOUNCE', 'EXHAUSTED_BULL', 'EXHAUSTED_BEAR',
            'ANOMALY_BULL', 'ANOMALY_BEAR',
            'FALSE_SIDEWAY'
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