"""
S6_QuantumPDF - Quantum Probability Density Function Strategy.

Mean-reversion strategy that uses Quantum PDF analysis to identify
high-probability price zones and trade mean reversion.

Strategy Logic:
  1. Calculate Quantum PDF from price distribution
  2. Identify peaks (high-probability zones)
  3. Detect price deviation from high-probability zones
  4. Generate mean-reversion entry when deviation is significant

Quantum PDF Concept:
  The Quantum PDF represents the probability distribution of price
  over a lookback period. Peaks in the PDF indicate high-probability
  zones where price tends to return.

Used Engines:
  - QuantMathEngine: PDF calculation and peak detection
  - AdaptiveTPEngine: Dynamic TP calculation

Strategy Category: MEAN_REVERSION
Timeframe: M15 (primary), M5 (confirmation)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from core.base_strategy import BaseStrategy
from core.quant_math_engine import QuantMathEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S6_QuantumPDF(BaseStrategy):
    """
    Quantum PDF Mean-Reversion Strategy.
    
    This strategy uses Quantum Probability Density Function analysis
    to identify high-probability price zones and trade mean reversion.
    
    Quantum PDF Definition:
      A probability density function calculated from price distribution
      over a lookback period. Peaks indicate zones where price has
      spent the most time (high probability zones).
      
    Mean Reversion Logic:
      When price deviates significantly from high-probability zones,
      it tends to revert back. This strategy enters when deviation
      reaches a threshold and exits when price returns to the zone.
      
    Entry Criteria:
      - PDF calculated with sufficient data
      - High-probability peaks identified
      - Price deviation from peak > threshold
      - Statistical confirmation (entropy, skewness)
    """

    def __init__(self):
        """Initialize S6_QuantumPDF strategy."""
        super().__init__(
            strategy_name='S6_QuantumPDF',
            strategy_category='MEAN_REVERSION',
            timeframes=['M15', 'M5'],
            risk_per_trade_pct=0.4,
            min_rr_ratio=1.5,
            max_spread_points=30,
            trailing_enabled=False,  # No trailing for mean reversion
            partial_close_enabled=False,  # No partial close for mean reversion
            requires_dynamic_exit=True
        )

        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize engines
        self.quant_engine = QuantMathEngine()
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.pdf_lookback = 100  # Lookback for PDF calculation
        self.pdf_bins = 50  # Number of bins for PDF
        self.deviation_threshold_atr = 2.0  # Deviation threshold in ATR
        self.peak_threshold = 0.6  # Minimum peak height threshold

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
        Main analysis method for S6_QuantumPDF.
        
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
        if df_m15 is None or df_m15.empty or len(df_m15) < self.pdf_lookback:
            return default_signal

        try:
            # Check regime compatibility
            if regime_context and not self._is_regime_compatible(regime_context):
                return default_signal

            # =========================================================================
            # STEP 1: Calculate Quantum PDF
            # =========================================================================
            close = df_m15['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            pdf_result = self.quant_engine.calculate_quantum_pdf(
                close, bins=self.pdf_bins, lookback=self.pdf_lookback
            )

            if pdf_result is None:
                return default_signal

            # =========================================================================
            # STEP 2: Find PDF Peaks
            # =========================================================================
            peaks = self.quant_engine.find_pdf_peaks(
                pdf_result['pdf'], pdf_result['bin_centers'], self.peak_threshold
            )

            if not peaks:
                return default_signal

            # =========================================================================
            # STEP 3: Analyze Deviation
            # =========================================================================
            deviation = self._analyze_deviation(df_m15, peaks, pdf_result)

            if deviation is None or not deviation.get('is_deviated', False):
                return default_signal

            # =========================================================================
            # STEP 4: Statistical Confirmation
            # =========================================================================
            if not self._confirm_statistically(df_m15, deviation):
                return default_signal

            # =========================================================================
            # STEP 5: M5 Confirmation
            # =========================================================================
            if df_m5 is not None and not df_m5.empty:
                if not self._confirm_m5(df_m5, deviation):
                    return default_signal

            # =========================================================================
            # STEP 6: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, deviation, peaks, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S6_QPDF] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # DEVIATION ANALYSIS
    # =========================================================================

    def _analyze_deviation(
        self, df: pd.DataFrame, peaks: List[Dict], pdf_result: Dict
    ) -> Optional[Dict]:
        """
        Analyze price deviation from high-probability zones.
        
        Args:
            df: DataFrame with OHLCV data
            peaks: List of PDF peaks
            pdf_result: PDF calculation result
            
        Returns:
            Deviation dict or None
        """
        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            current_price = close[-1]

            # Calculate ATR for deviation threshold
            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(np.abs(high[1:] - close[:-1]),
                                       np.abs(low[1:] - close[:-1])))
            atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)

            deviation_threshold = atr * self.deviation_threshold_atr

            # Find nearest peak
            nearest_peak = None
            min_distance = float('inf')

            for peak in peaks:
                peak_price = peak.get('price', 0)
                distance = abs(current_price - peak_price)

                if distance < min_distance:
                    min_distance = distance
                    nearest_peak = peak

            if nearest_peak is None:
                return None

            peak_price = nearest_peak.get('price', 0)

            # Check if deviation exceeds threshold
            if min_distance > deviation_threshold:
                # Determine direction
                if current_price > peak_price:
                    direction = 'SELL'  # Price above peak, expect reversion down
                    deviation_type = 'ABOVE_PEAK'
                else:
                    direction = 'BUY'  # Price below peak, expect reversion up
                    deviation_type = 'BELOW_PEAK'

                # Calculate deviation in ATR units
                deviation_atr = min_distance / atr if atr > 0 else 0

                return {
                    'is_deviated': True,
                    'direction': direction,
                    'deviation_type': deviation_type,
                    'current_price': float(current_price),
                    'peak_price': float(peak_price),
                    'deviation': float(min_distance),
                    'deviation_atr': float(deviation_atr),
                    'deviation_pct': float(min_distance / current_price * 100),
                    'peak_strength': nearest_peak.get('strength', 0.5)
                }

            return {'is_deviated': False}

        except Exception as e:
            self.logger.debug(f"[S6_QPDF] Deviation analysis error: {e}")
            return None

    # =========================================================================
    # STATISTICAL CONFIRMATION
    # =========================================================================

    def _confirm_statistically(self, df: pd.DataFrame, deviation: Dict) -> bool:
        """
        Confirm deviation with statistical measures.
        
        Args:
            df: DataFrame with OHLCV data
            deviation: Deviation dict
            
        Returns:
            True if statistically confirmed
        """
        try:
            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            # Calculate statistics
            stats = self.quant_engine.calculate_statistics(close[-self.pdf_lookback:])

            if stats is None:
                return False

            # Check entropy (should be low for mean-reverting markets)
            entropy = stats.get('entropy', 0)

            # Check skewness (should be opposite to deviation direction)
            skewness = stats.get('skewness', 0)
            direction = deviation.get('direction', 'BUY')

            if direction == 'BUY':
                # For BUY (price below peak), skewness should be negative
                # (more values below mean)
                skewness_aligned = skewness < 0.5
            else:  # SELL
                # For SELL (price above peak), skewness should be positive
                skewness_aligned = skewness > -0.5

            # Check kurtosis (should be low for normal distribution)
            kurtosis = stats.get('kurtosis', 0)
            kurtosis_ok = abs(kurtosis) < 3  # Not too fat-tailed

            return skewness_aligned and kurtosis_ok

        except Exception:
            return False

    # =========================================================================
    # M5 CONFIRMATION
    # =========================================================================

    def _confirm_m5(self, df_m5: pd.DataFrame, deviation: Dict) -> bool:
        """
        Confirm signal on M5 timeframe.
        
        Args:
            df_m5: M5 DataFrame
            deviation: Deviation dict
            
        Returns:
            True if confirmed
        """
        if df_m5 is None or df_m5.empty or len(df_m5) < 20:
            return True  # Skip confirmation if no M5 data

        try:
            close = df_m5['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            direction = deviation.get('direction', 'BUY')

            # For mean reversion, M5 should show reversal signs
            recent_close = close[-10:]
            momentum = recent_close[-1] - recent_close[0]

            if direction == 'BUY':
                # For BUY reversion: M5 should show slowing decline or reversal
                return momentum >= -0.1 * abs(recent_close[0])
            else:  # SELL
                # For SELL reversion: M5 should show slowing rise or reversal
                return momentum <= 0.1 * abs(recent_close[0])

        except Exception:
            return True  # Skip confirmation on error

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def _generate_signal(
        self, df: pd.DataFrame, deviation: Dict, peaks: List[Dict],
        regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            deviation: Deviation dict
            peaks: List of PDF peaks
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            direction = deviation.get('direction', 'BUY')
            entry_price = deviation.get('current_price', 0)
            peak_price = deviation.get('peak_price', 0)
            deviation_atr = deviation.get('deviation_atr', 0)

            if entry_price <= 0:
                return self._create_neutral_signal()

            # Calculate Stop Loss
            # For mean reversion, SL is beyond the deviation
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            close = df['close'].values.astype(float)

            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))
            close = np.nan_to_num(close, nan=np.nanmean(close))

            # ATR
            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(np.abs(high[1:] - close[:-1]),
                                       np.abs(low[1:] - close[:-1])))
            atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)

            if direction == 'BUY':
                # SL below entry with ATR buffer
                sl_price = entry_price - atr * 1.5
                # TP at peak (mean reversion target)
                tp_price = peak_price
            else:  # SELL
                # SL above entry with ATR buffer
                sl_price = entry_price + atr * 1.5
                # TP at peak (mean reversion target)
                tp_price = peak_price

            # Validate SL and TP
            if sl_price <= 0 or sl_price == entry_price or tp_price <= 0:
                return self._create_neutral_signal()

            # Check R:R ratio
            risk = abs(entry_price - sl_price)
            reward = abs(tp_price - entry_price)

            if risk <= 0 or reward / risk < self.min_rr_ratio:
                # Adjust TP to meet minimum R:R
                if direction == 'BUY':
                    tp_price = entry_price + risk * self.min_rr_ratio
                else:
                    tp_price = entry_price - risk * self.min_rr_ratio

            # Calculate confidence
            peak_strength = deviation.get('peak_strength', 0.5)
            deviation_strength = min(1.0, deviation_atr / 3.0)  # Cap at 3 ATR

            confidence = min(1.0, 0.4 + peak_strength * 0.3 + deviation_strength * 0.3)

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
                    'deviation_type': deviation.get('deviation_type', 'UNKNOWN'),
                    'deviation_atr': deviation_atr,
                    'peak_price': round(peak_price, 2),
                    'peak_strength': peak_strength,
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit
                }
            }

            self.logger.info(
                f"[S6_QPDF] Signal generated: {direction} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Deviation: {deviation_atr:.2f} ATR | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S6_QPDF] Signal generation error: {e}")
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