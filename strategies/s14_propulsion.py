"""
S14_Propulsion - Propulsion Momentum Strategy.

Trend-following strategy that uses propulsion analysis to identify
strong directional momentum and trade trend continuation.

Strategy Logic:
  1. Calculate momentum (rate of price change)
  2. Detect acceleration (rate of momentum change)
  3. Analyze directional thrust
  4. Generate entry signal when propulsion is strong

Propulsion Concept:
  Propulsion measures the "force" behind price movement:
    - Momentum: Rate of price change
    - Acceleration: Rate of momentum change
    - Thrust: Directional force
  
  Strong propulsion = Price moving with force and sustainability
  Weak propulsion = Price moving without conviction

Used Engines:
  - PropulsionEngine: Momentum, acceleration, and thrust calculation
  - AdaptiveTPEngine: Dynamic TP calculation

Strategy Category: TREND
Timeframe: M15 (primary), M5 (confirmation)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from core.base_strategy import BaseStrategy
from core.propulsion_engine import PropulsionEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S14_Propulsion(BaseStrategy):
    """
    Propulsion Momentum Strategy.
    
    This strategy uses propulsion analysis to identify strong
    directional momentum and trade trend continuation.
    
    Propulsion Definition:
      Propulsion measures the "force" behind price movement:
        - Momentum: How fast price is moving
        - Acceleration: How momentum is changing
        - Thrust: Directional force of movement
        
    Strong Propulsion Signals:
      - High momentum + positive acceleration = Strong trend
      - Momentum increasing + thrust aligned = Trend continuation
      - Low momentum + negative acceleration = Trend weakening
      
    Entry Criteria:
      - Propulsion score above threshold
      - Momentum and acceleration aligned
      - Trend direction confirmed
      - Confirmation from M5 timeframe
    """

    def __init__(self):
        """Initialize S14_Propulsion strategy."""
        super().__init__(
            strategy_name='S14_Propulsion',
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
        self.propulsion_engine = PropulsionEngine()
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.momentum_period = 14  # Momentum lookback
        self.acceleration_period = 7  # Acceleration lookback
        self.propulsion_threshold = 50  # Minimum propulsion score (0-100)
        self.min_trend_strength = 0.3  # Minimum trend strength

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
        Main analysis method for S14_Propulsion.
        
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
        if df_m15 is None or df_m15.empty or len(df_m15) < 50:
            return default_signal

        try:
            # Check regime compatibility
            if regime_context and not self._is_regime_compatible(regime_context):
                return default_signal

            # =========================================================================
            # STEP 1: Calculate Propulsion Score
            # =========================================================================
            propulsion_result = self.propulsion_engine.calculate_propulsion_score(df_m15)

            if propulsion_result is None:
                return default_signal

            # =========================================================================
            # STEP 2: Check Propulsion Threshold
            # =========================================================================
            propulsion_score = propulsion_result.get('propulsion_score', 0)

            if propulsion_score < self.propulsion_threshold:
                return default_signal

            # =========================================================================
            # STEP 3: Analyze Components
            # =========================================================================
            momentum = propulsion_result.get('momentum', 0)
            acceleration = propulsion_result.get('acceleration', 0)
            thrust = propulsion_result.get('thrust', 0)
            trend_strength = propulsion_result.get('trend_strength', 0)
            direction = propulsion_result.get('direction', 'NEUTRAL')

            if direction == 'NEUTRAL':
                return default_signal

            # =========================================================================
            # STEP 4: Confirm Alignment
            # =========================================================================
            if not self._confirm_alignment(momentum, acceleration, thrust, direction):
                return default_signal

            # =========================================================================
            # STEP 5: M5 Confirmation
            # =========================================================================
            if df_m5 is not None and not df_m5.empty:
                if not self._confirm_m5(df_m5, direction):
                    return default_signal

            # =========================================================================
            # STEP 6: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, propulsion_result, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S14_PROP] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # ALIGNMENT CONFIRMATION
    # =========================================================================

    def _confirm_alignment(
        self, momentum: float, acceleration: float, thrust: float, direction: str
    ) -> bool:
        """
        Confirm propulsion components are aligned.
        
        Args:
            momentum: Momentum value
            acceleration: Acceleration value
            thrust: Thrust value
            direction: Direction (BUY/SELL)
            
        Returns:
            True if components are aligned
        """
        try:
            if direction == 'BUY':
                # For BUY: momentum > 0, acceleration > 0, thrust > 0
                return momentum > 0 and acceleration > 0 and thrust > 0
            else:  # SELL
                # For SELL: momentum < 0, acceleration < 0, thrust < 0
                return momentum < 0 and acceleration < 0 and thrust < 0

        except Exception:
            return False

    # =========================================================================
    # M5 CONFIRMATION
    # =========================================================================

    def _confirm_m5(self, df_m5: pd.DataFrame, direction: str) -> bool:
        """
        Confirm signal on M5 timeframe.
        
        Args:
            df_m5: M5 DataFrame
            direction: Direction (BUY/SELL)
            
        Returns:
            True if confirmed
        """
        if df_m5 is None or df_m5.empty or len(df_m5) < 20:
            return True  # Skip confirmation if no M5 data

        try:
            close = df_m5['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            # Check M5 momentum aligns with direction
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
        self, df: pd.DataFrame, propulsion_result: Dict,
        regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            propulsion_result: Propulsion calculation result
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            direction = propulsion_result.get('direction', 'BUY')
            propulsion_score = propulsion_result.get('propulsion_score', 0)
            trend_strength = propulsion_result.get('trend_strength', 0)

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
                    tp_price = entry_price + risk * 2.0
                else:
                    tp_price = entry_price - risk * 2.0

            # Calculate confidence
            score_bonus = propulsion_score / 100 * 0.3
            strength_bonus = trend_strength * 0.2
            confidence = min(1.0, 0.4 + score_bonus + strength_bonus)

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
                    'propulsion_score': propulsion_score,
                    'momentum': propulsion_result.get('momentum', 0),
                    'acceleration': propulsion_result.get('acceleration', 0),
                    'thrust': propulsion_result.get('thrust', 0),
                    'trend_strength': trend_strength,
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit
                }
            }

            self.logger.info(
                f"[S14_PROP] Signal generated: {direction} | "
                f"Propulsion: {propulsion_score:.1f} | "
                f"Momentum: {propulsion_result.get('momentum', 0):.2f} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S14_PROP] Signal generation error: {e}")
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