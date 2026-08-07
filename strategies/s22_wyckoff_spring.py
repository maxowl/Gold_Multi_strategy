"""
S22_WyckoffSpring - Wyckoff Spring Strategy.

Mean-reversion strategy that identifies Wyckoff Springs (bear traps)
and trades the reversal after the spring completes.

Strategy Logic:
  1. Detect Wyckoff phase (accumulation)
  2. Identify Spring pattern (fake breakdown)
  3. Confirm spring with VSA analysis
  4. Generate entry signal after spring completion

Wyckoff Spring Definition:
  A Spring is a "bear trap" that occurs during accumulation:
    - Price breaks below support (triggers stop losses)
    - Quickly reverses back above support
    - Indicates smart money accumulation
    
  The Spring is a key component of Wyckoff's accumulation phase,
  signaling the end of the accumulation and the beginning of markup.

Wyckoff Phases:
  Phase A: Stopping the downtrend
  Phase B: Building a cause (accumulation)
  Phase C: Test (Spring occurs here)
  Phase D: Trend within the range
  Phase E: Trend outside the range (markup)

Used Engines:
  - WyckoffVSAEngine: Spring detection and VSA analysis
  - AdaptiveTPEngine: Dynamic TP calculation

Strategy Category: MEAN_REVERSION
Timeframe: M15 (primary), M5 (confirmation)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from core.base_strategy import BaseStrategy
from core.wyckoff_vsa_engine import WyckoffVSAEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S22_WyckoffSpring(BaseStrategy):
    """
    Wyckoff Spring Strategy.
    
    This strategy identifies Wyckoff Springs (bear traps) and
    trades the reversal after the spring completes.
    
    Spring Definition:
      A Spring is a fake breakdown below support that triggers
      stop losses before reversing. It's a bear trap that
      allows smart money to accumulate at lower prices.
      
    Spring Characteristics:
      - Price breaks below support
      - Shallow breakdown (typically < 1%)
      - Quick reversal back above support
      - Volume spike during the spring
      - Low volume on the reversal
      
    Entry Criteria:
      - Spring detected
      - Spring confirmed (price back above support)
      - VSA confirms accumulation
      - Confirmation from M5 timeframe
    """

    def __init__(self):
        """Initialize S22_WyckoffSpring strategy."""
        super().__init__(
            strategy_name='S22_WyckoffSpring',
            strategy_category='MEAN_REVERSION',
            timeframes=['M15', 'M5'],
            risk_per_trade_pct=0.5,
            min_rr_ratio=1.8,
            max_spread_points=30,
            trailing_enabled=True,
            partial_close_enabled=True,
            requires_dynamic_exit=False
        )

        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize engines
        self.wyckoff_engine = WyckoffVSAEngine()
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.spring_lookback = 50  # Lookback for spring detection
        self.spring_depth_threshold = 0.005  # Maximum spring depth (0.5%)
        self.phase_lookback = 100  # Lookback for phase detection
        self.min_spring_strength = 0.5  # Minimum spring strength

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
        Main analysis method for S22_WyckoffSpring.
        
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
            # STEP 1: Detect Wyckoff Phase
            # =========================================================================
            phase_result = self.wyckoff_engine.detect_wyckoff_phase(
                df_m15, lookback=self.phase_lookback
            )

            if phase_result is None:
                return default_signal

            # Check if in accumulation phase
            if not phase_result.get('is_accumulation', False):
                return default_signal

            # =========================================================================
            # STEP 2: Detect Spring
            # =========================================================================
            spring_result = self.wyckoff_engine.detect_spring(
                df_m15, lookback=self.spring_lookback
            )

            if spring_result is None:
                return default_signal

            # =========================================================================
            # STEP 3: Confirm Spring
            # =========================================================================
            if not self._confirm_spring(df_m15, spring_result):
                return default_signal

            # =========================================================================
            # STEP 4: VSA Analysis
            # =========================================================================
            vsa_result = self.wyckoff_engine.analyze_vsa(df_m15)

            if vsa_result is None:
                return default_signal

            # Check VSA confirms accumulation
            if not self._check_vsa_accumulation(vsa_result):
                return default_signal

            # =========================================================================
            # STEP 5: M5 Confirmation
            # =========================================================================
            if df_m5 is not None and not df_m5.empty:
                if not self._confirm_m5(df_m5):
                    return default_signal

            # =========================================================================
            # STEP 6: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, spring_result, phase_result, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S22_WYCK] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # SPRING CONFIRMATION
    # =========================================================================

    def _confirm_spring(self, df: pd.DataFrame, spring_result: Dict) -> bool:
        """
        Confirm spring completion.
        
        Args:
            df: DataFrame with OHLCV data
            spring_result: Spring detection result
            
        Returns:
            True if spring is confirmed
        """
        try:
            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            current_price = close[-1]
            support_level = spring_result.get('support_level', 0)
            spring_low = spring_result.get('spring_low', 0)
            recovered = spring_result.get('recovered', False)

            # Spring must be recovered (price back above support)
            if not recovered:
                return False

            # Current price must be above support
            if current_price < support_level:
                return False

            # Spring depth must be shallow
            spring_depth = spring_result.get('spring_depth_pct', 0)
            if spring_depth > self.spring_depth_threshold * 100:
                return False

            return True

        except Exception:
            return False

    # =========================================================================
    # VSA ACCUMULATION CHECK
    # =========================================================================

    def _check_vsa_accumulation(self, vsa_result: Dict) -> bool:
        """
        Check if VSA confirms accumulation.
        
        Args:
            vsa_result: VSA analysis result
            
        Returns:
            True if VSA confirms accumulation
        """
        try:
            signals = vsa_result.get('signals', [])
            volume_trend = vsa_result.get('volume_trend', 'UNKNOWN')

            # Check for accumulation signals
            accumulation_signals = [
                'NO_SUPPLY', 'STOPPING_VOLUME', 'BUYING_STRENGTH'
            ]

            for signal in signals:
                if signal.get('signal') in accumulation_signals:
                    return True

            # Check volume trend
            if volume_trend == 'INCREASING':
                return True

            return False

        except Exception:
            return False

    # =========================================================================
    # M5 CONFIRMATION
    # =========================================================================

    def _confirm_m5(self, df_m5: pd.DataFrame) -> bool:
        """
        Confirm signal on M5 timeframe.
        
        Args:
            df_m5: M5 DataFrame
            
        Returns:
            True if confirmed
        """
        if df_m5 is None or df_m5.empty or len(df_m5) < 20:
            return True  # Skip confirmation if no M5 data

        try:
            close = df_m5['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            # Check M5 momentum is bullish
            recent_close = close[-10:]
            momentum = recent_close[-1] - recent_close[0]

            return momentum > 0  # Bullish momentum on M5

        except Exception:
            return True  # Skip confirmation on error

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def _generate_signal(
        self, df: pd.DataFrame, spring_result: Dict, phase_result: Dict,
        regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            spring_result: Spring detection result
            phase_result: Phase detection result
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            entry_price = close[-1]
            support_level = spring_result.get('support_level', 0)
            spring_low = spring_result.get('spring_low', 0)

            if entry_price <= 0 or support_level <= 0:
                return self._create_neutral_signal()

            # Calculate ATR for stop loss
            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(np.abs(high[1:] - close[:-1]),
                                       np.abs(low[1:] - close[:-1])))
            atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)

            # Calculate Stop Loss
            # SL below spring low with buffer
            sl_buffer = abs(spring_low - support_level) * 0.3
            sl_price = spring_low - sl_buffer

            # Validate SL
            if sl_price <= 0 or sl_price == entry_price:
                return self._create_neutral_signal()

            # Calculate Take Profit
            tp_result = self.adaptive_tp_engine.calculate_adaptive_tp(
                df, entry_price, sl_price, True,  # Spring is always BUY
                regime_context.get('regime_name', 'UNKNOWN') if regime_context else 'UNKNOWN'
            )

            if tp_result and tp_result.get('tp_price', 0) > 0:
                tp_price = tp_result['tp_price']
            else:
                # Fallback: Fixed R:R
                risk = abs(entry_price - sl_price)
                tp_price = entry_price + risk * 2.0

            # Calculate confidence
            spring_strength = spring_result.get('strength', 0.5)
            phase_bonus = 0.1 if phase_result.get('phase') == 'PHASE_C_ACCUMULATION' else 0.0
            confidence = min(1.0, 0.4 + spring_strength * 0.4 + phase_bonus)

            # Build signal
            signal = {
                'signal': 'BUY_MARKET',  # Spring is always BUY
                'meta': {
                    'strategy': self.strategy_name,
                    'strategy_category': self.strategy_category,
                    'entry_price': round(entry_price, 2),
                    'sl_price': round(sl_price, 2),
                    'tp_price': round(tp_price, 2),
                    'confidence': confidence,
                    'spring_depth_pct': spring_result.get('spring_depth_pct', 0),
                    'support_level': round(support_level, 2),
                    'spring_low': round(spring_low, 2),
                    'phase': phase_result.get('phase', 'UNKNOWN'),
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit
                }
            }

            self.logger.info(
                f"[S22_WYCK] Signal generated: BUY | "
                f"Spring Depth: {spring_result.get('spring_depth_pct', 0):.2f}% | "
                f"Support: {support_level:.2f} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S22_WYCK] Signal generation error: {e}")
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