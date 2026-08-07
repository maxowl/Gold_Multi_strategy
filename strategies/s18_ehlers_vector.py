"""
S18_EhlersVector - Ehlers Vector Indicators Strategy.

Mean-reversion strategy that uses Ehlers Vector indicators for
statistical analysis and mean reversion trading.

Strategy Logic:
  1. Calculate Ehlers Vector indicators
  2. Analyze statistical measures (skewness, kurtosis)
  3. Detect mean reversion opportunities
  4. Generate entry signal on statistical extremes

Ehlers Vector Concepts:
  Vector Candlestick:
    Represents price movement as a vector with magnitude and direction.
    Magnitude = Price change
    Direction = Up or down
    
  Vector Moving Average:
    A moving average that considers both magnitude and direction
    of price movement, providing smoother trend identification.
    
  Statistical Analysis:
    - Skewness: Measures asymmetry of price distribution
    - Kurtosis: Measures fat-tails of price distribution
    - Entropy: Measures randomness/predictability

Used Engines:
  - EhlersDSPEngine: Vector calculations
  - QuantMathEngine: Statistical analysis
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
from core.quant_math_engine import QuantMathEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S18_EhlersVector(BaseStrategy):
    """
    Ehlers Vector Indicators Strategy.
    
    This strategy uses Ehlers Vector indicators and statistical
    analysis for mean reversion trading.
    
    Vector Definition:
      A vector represents price movement with both magnitude
      (size of move) and direction (up or down).
      
    Statistical Extremes:
      - High skewness: Distribution is asymmetric
      - High kurtosis: Distribution has fat tails
      - Low entropy: Market is predictable
      
    Entry Criteria:
      - Vector indicators calculated
      - Statistical extremes detected
      - Mean reversion signal confirmed
      - Confirmation from M5 timeframe
    """

    def __init__(self):
        """Initialize S18_EhlersVector strategy."""
        super().__init__(
            strategy_name='S18_EhlersVector',
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
        self.quant_engine = QuantMathEngine()
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.vector_lookback = 50  # Lookback for vector calculation
        self.stats_lookback = 100  # Lookback for statistical analysis
        self.skewness_threshold = 0.5  # Skewness threshold
        self.kurtosis_threshold = 2.0  # Kurtosis threshold
        self.entropy_threshold = 3.0  # Entropy threshold

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
        Main analysis method for S18_EhlersVector.
        
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
            # STEP 1: Calculate Vector Indicators
            # =========================================================================
            close = df_m15['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            vector_result = self._calculate_vector(close)

            if vector_result is None:
                return default_signal

            # =========================================================================
            # STEP 2: Analyze Statistics
            # =========================================================================
            stats_result = self.quant_engine.calculate_statistics(
                close[-self.stats_lookback:]
            )

            if stats_result is None:
                return default_signal

            # =========================================================================
            # STEP 3: Detect Reversion
            # =========================================================================
            reversion = self._detect_reversion(vector_result, stats_result, close)

            if reversion is None or not reversion.get('signal_detected', False):
                return default_signal

            # =========================================================================
            # STEP 4: M5 Confirmation
            # =========================================================================
            if df_m5 is not None and not df_m5.empty:
                if not self._confirm_m5(df_m5, reversion):
                    return default_signal

            # =========================================================================
            # STEP 5: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, reversion, stats_result, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S18_VECTOR] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # VECTOR CALCULATION
    # =========================================================================

    def _calculate_vector(self, close: np.ndarray) -> Optional[Dict]:
        """
        Calculate Ehlers Vector indicators.
        
        Args:
            close: Close price array
            
        Returns:
            Vector result dict or None
        """
        try:
            n = len(close)
            if n < self.vector_lookback:
                return None

            # Calculate price changes (vectors)
            changes = np.diff(close)
            changes = np.insert(changes, 0, 0)

            # Calculate vector magnitude (absolute change)
            magnitude = np.abs(changes)

            # Calculate vector direction (sign of change)
            direction = np.sign(changes)

            # Calculate Vector Moving Average
            vma_period = 20
            vma = pd.Series(close).ewm(span=vma_period, adjust=False).mean().values

            # Calculate current vector statistics
            recent_magnitude = magnitude[-self.vector_lookback:]
            recent_direction = direction[-self.vector_lookback:]

            avg_magnitude = np.mean(recent_magnitude)
            current_magnitude = magnitude[-1]
            current_direction = direction[-1]

            # Calculate vector strength
            vector_strength = current_magnitude / (avg_magnitude + 1e-10)

            return {
                'magnitude': float(current_magnitude),
                'direction': float(current_direction),
                'avg_magnitude': float(avg_magnitude),
                'vector_strength': float(vector_strength),
                'vma': float(vma[-1]),
                'current_price': float(close[-1])
            }

        except Exception as e:
            self.logger.debug(f"[S18_VECTOR] Vector calculation error: {e}")
            return None

    # =========================================================================
    # REVERSION DETECTION
    # =========================================================================

    def _detect_reversion(
        self, vector_result: Dict, stats_result: Dict, close: np.ndarray
    ) -> Optional[Dict]:
        """
        Detect mean reversion signal.
        
        Args:
            vector_result: Vector calculation result
            stats_result: Statistical analysis result
            close: Close price array
            
        Returns:
            Reversion dict or None
        """
        try:
            current_price = close[-1]
            vma = vector_result.get('vma', 0)
            vector_strength = vector_result.get('vector_strength', 0)

            skewness = stats_result.get('skewness', 0)
            kurtosis = stats_result.get('kurtosis', 0)
            entropy = stats_result.get('entropy', 0)
            mean_price = stats_result.get('mean', 0)
            std_price = stats_result.get('std', 0)

            if std_price == 0:
                return None

            # Calculate z-score
            zscore = (current_price - mean_price) / std_price

            # Check statistical extremes
            skewness_extreme = abs(skewness) > self.skewness_threshold
            kurtosis_extreme = abs(kurtosis) > self.kurtosis_threshold
            entropy_low = entropy < self.entropy_threshold

            # Determine direction based on z-score and statistics
            if zscore > 1.5 and skewness > 0:
                # Price above mean with positive skew → expect reversion down
                direction = 'SELL'
                signal_strength = min(1.0, zscore / 2.0 + abs(skewness) / 2.0)
            elif zscore < -1.5 and skewness < 0:
                # Price below mean with negative skew → expect reversion up
                direction = 'BUY'
                signal_strength = min(1.0, abs(zscore) / 2.0 + abs(skewness) / 2.0)
            else:
                return None  # No clear reversion signal

            # Check vector confirmation
            # For reversion, vector should be weakening
            if vector_strength > 2.0:
                return None  # Vector too strong, trend may continue

            return {
                'signal_detected': True,
                'direction': direction,
                'zscore': float(zscore),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'entropy': float(entropy),
                'signal_strength': float(signal_strength),
                'vector_strength': float(vector_strength)
            }

        except Exception as e:
            self.logger.debug(f"[S18_VECTOR] Reversion detection error: {e}")
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
        self, df: pd.DataFrame, reversion: Dict, stats_result: Dict,
        regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            reversion: Reversion dict
            stats_result: Statistical analysis result
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

            # Calculate mean price for TP
            mean_price = stats_result.get('mean', entry_price)

            # Calculate Stop Loss
            if direction == 'BUY':
                sl_price = entry_price - atr * 1.5
                tp_price = mean_price  # TP at mean
            else:  # SELL
                sl_price = entry_price + atr * 1.5
                tp_price = mean_price  # TP at mean

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
            stats_bonus = 0.1 if reversion.get('entropy', 5) < 3.0 else 0.0
            confidence = min(1.0, 0.4 + signal_strength * 0.4 + stats_bonus)

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
                    'skewness': reversion.get('skewness', 0),
                    'kurtosis': reversion.get('kurtosis', 0),
                    'entropy': reversion.get('entropy', 0),
                    'vector_strength': reversion.get('vector_strength', 0),
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit
                }
            }

            self.logger.info(
                f"[S18_VECTOR] Signal generated: {direction} | "
                f"Z-score: {reversion.get('zscore', 0):.2f} | "
                f"Skewness: {reversion.get('skewness', 0):.2f} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S18_VECTOR] Signal generation error: {e}")
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