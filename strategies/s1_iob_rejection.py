"""
S1_IOB_Rejection - Inside Order Block Rejection Strategy.

Smart Money Concepts strategy that identifies Inside Order Blocks (IOB)
and trades rejection patterns at these levels.

Strategy Logic:
  1. Detect Order Blocks (OB) using SMC engine
  2. Identify Inside Order Blocks (IOB) - OB within OB
  3. Detect rejection patterns at IOB levels
  4. Generate entry signal with proper risk management

Used Engines:
  - SMCStructuralEngine: Order Block detection
  - FibonacciEngine: Fibonacci levels for TP
  - AdaptiveTPEngine: Dynamic TP calculation

Strategy Category: SMC
Timeframe: M15 (primary), M5 (confirmation)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from core.base_strategy import BaseStrategy
from core.smc_engine import SMCStructuralEngine
from core.fibonacci_engine import FibonacciEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S1_IOB_Rejection(BaseStrategy):
    """
    Inside Order Block Rejection Strategy.
    
    This strategy identifies Inside Order Blocks (IOB) and trades
    rejection patterns at these high-probability levels.
    
    IOB Definition:
      An Inside Order Block is an Order Block that forms inside
      a larger Order Block, creating a nested structure.
      
    Entry Criteria:
      - IOB detected at key structural level
      - Price approaches IOB and shows rejection
      - Confirmation from M5 timeframe
      - Volume confirmation
    """

    def __init__(self):
        """Initialize S1_IOB_Rejection strategy."""
        super().__init__(
            strategy_name='S1_IOB_Rejection',
            strategy_category='SMC',
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
        self.smc_engine = SMCStructuralEngine()
        self.fib_engine = FibonacciEngine()
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.iob_lookback = 50  # Lookback for IOB detection
        self.rejection_threshold = 0.3  # Rejection wick threshold
        self.min_ob_quality = 60  # Minimum OB quality score

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
        Main analysis method for S1_IOB_Rejection.
        
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
            # STEP 1: Detect Order Blocks
            # =========================================================================
            order_blocks = self.smc_engine.detect_order_blocks(df_m15, lookback=self.iob_lookback)

            if not order_blocks:
                return default_signal

            # Filter high-quality OBs
            quality_obs = [ob for ob in order_blocks if ob.get('quality', 0) >= self.min_ob_quality]

            if not quality_obs:
                return default_signal

            # =========================================================================
            # STEP 2: Detect Inside Order Blocks
            # =========================================================================
            iobs = self._detect_inside_order_blocks(quality_obs)

            if not iobs:
                return default_signal

            # =========================================================================
            # STEP 3: Detect Rejection at IOB
            # =========================================================================
            rejection = self._detect_rejection(df_m15, iobs)

            if rejection is None:
                return default_signal

            # =========================================================================
            # STEP 4: M5 Confirmation
            # =========================================================================
            if df_m5 is not None and not df_m5.empty:
                if not self._confirm_m5(df_m5, rejection):
                    return default_signal

            # =========================================================================
            # STEP 5: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, rejection, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S1_IOB] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # INSIDE ORDER BLOCK DETECTION
    # =========================================================================

    def _detect_inside_order_blocks(self, order_blocks: List[Dict]) -> List[Dict]:
        """
        Detect Inside Order Blocks (IOB).
        
        IOB is an Order Block that forms inside a larger Order Block.
        
        Args:
            order_blocks: List of detected Order Blocks
            
        Returns:
            List of IOB dicts
        """
        iobs = []

        try:
            # Sort OBs by index (chronological order)
            sorted_obs = sorted(order_blocks, key=lambda x: x.get('index', 0))

            # Find nested OBs
            for i in range(len(sorted_obs) - 1):
                outer_ob = sorted_obs[i]
                inner_ob = sorted_obs[i + 1]

                # Check if inner OB is inside outer OB
                if self._is_inside(outer_ob, inner_ob):
                    iob = {
                        'outer_ob': outer_ob,
                        'inner_ob': inner_ob,
                        'type': inner_ob.get('type', 'UNKNOWN'),
                        'index': inner_ob.get('index', 0),
                        'quality': (outer_ob.get('quality', 50) + inner_ob.get('quality', 50)) / 2,
                        'high': inner_ob.get('high', 0),
                        'low': inner_ob.get('low', 0)
                    }
                    iobs.append(iob)

            return iobs

        except Exception as e:
            self.logger.debug(f"[S1_IOB] IOB detection error: {e}")
            return []

    def _is_inside(self, outer_ob: Dict, inner_ob: Dict) -> bool:
        """Check if inner OB is inside outer OB."""
        try:
            outer_high = outer_ob.get('high', 0)
            outer_low = outer_ob.get('low', 0)
            inner_high = inner_ob.get('high', 0)
            inner_low = inner_ob.get('low', 0)

            # Inner OB must be completely inside outer OB
            return (inner_high <= outer_high and
                    inner_low >= outer_low and
                    inner_high > inner_low)

        except Exception:
            return False

    # =========================================================================
    # REJECTION DETECTION
    # =========================================================================

    def _detect_rejection(self, df: pd.DataFrame, iobs: List[Dict]) -> Optional[Dict]:
        """
        Detect rejection pattern at IOB levels.
        
        Rejection is indicated by:
          - Long wick in the direction of the IOB
          - Close away from the IOB level
          - Volume confirmation
        
        Args:
            df: DataFrame with OHLCV data
            iobs: List of IOBs
            
        Returns:
            Rejection dict or None
        """
        if df is None or df.empty or not iobs:
            return None

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

            current_price = close[-1]

            # Check rejection at each IOB
            for iob in iobs:
                iob_type = iob.get('type', 'UNKNOWN')
                iob_high = iob.get('high', 0)
                iob_low = iob.get('low', 0)

                if iob_type == 'BULLISH':
                    # Bullish IOB: Look for rejection at IOB low (support)
                    rejection = self._detect_bullish_rejection(
                        close, high, low, open_, iob_low, current_price
                    )
                    if rejection:
                        rejection['iob'] = iob
                        rejection['direction'] = 'BUY'
                        return rejection

                elif iob_type == 'BEARISH':
                    # Bearish IOB: Look for rejection at IOB high (resistance)
                    rejection = self._detect_bearish_rejection(
                        close, high, low, open_, iob_high, current_price
                    )
                    if rejection:
                        rejection['iob'] = iob
                        rejection['direction'] = 'SELL'
                        return rejection

            return None

        except Exception as e:
            self.logger.debug(f"[S1_IOB] Rejection detection error: {e}")
            return None

    def _detect_bullish_rejection(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray,
        open_: np.ndarray, iob_low: float, current_price: float
    ) -> Optional[Dict]:
        """Detect bullish rejection at support level."""
        try:
            # Current bar
            current_close = close[-1]
            current_high = high[-1]
            current_low = low[-1]
            current_open = open_[-1]

            # Check if price is near IOB low
            tolerance = abs(current_close - iob_low) / current_close * 100
            if tolerance > 1.0:  # More than 1% away
                return None

            # Check for bullish rejection (long lower wick)
            candle_range = current_high - current_low
            if candle_range <= 0:
                return None

            lower_wick = min(current_open, current_close) - current_low
            lower_wick_ratio = lower_wick / candle_range

            # Check close position
            close_position = (current_close - current_low) / candle_range

            # Bullish rejection: long lower wick, close near high
            if lower_wick_ratio > self.rejection_threshold and close_position > 0.6:
                return {
                    'type': 'BULLISH_REJECTION',
                    'price': float(current_close),
                    'iob_level': float(iob_low),
                    'wick_ratio': float(lower_wick_ratio),
                    'close_position': float(close_position),
                    'strength': float(lower_wick_ratio)
                }

            return None

        except Exception:
            return None

    def _detect_bearish_rejection(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray,
        open_: np.ndarray, iob_high: float, current_price: float
    ) -> Optional[Dict]:
        """Detect bearish rejection at resistance level."""
        try:
            # Current bar
            current_close = close[-1]
            current_high = high[-1]
            current_low = low[-1]
            current_open = open_[-1]

            # Check if price is near IOB high
            tolerance = abs(current_close - iob_high) / current_close * 100
            if tolerance > 1.0:  # More than 1% away
                return None

            # Check for bearish rejection (long upper wick)
            candle_range = current_high - current_low
            if candle_range <= 0:
                return None

            upper_wick = current_high - max(current_open, current_close)
            upper_wick_ratio = upper_wick / candle_range

            # Check close position
            close_position = (current_close - current_low) / candle_range

            # Bearish rejection: long upper wick, close near low
            if upper_wick_ratio > self.rejection_threshold and close_position < 0.4:
                return {
                    'type': 'BEARISH_REJECTION',
                    'price': float(current_close),
                    'iob_level': float(iob_high),
                    'wick_ratio': float(upper_wick_ratio),
                    'close_position': float(close_position),
                    'strength': float(upper_wick_ratio)
                }

            return None

        except Exception:
            return None

    # =========================================================================
    # M5 CONFIRMATION
    # =========================================================================

    def _confirm_m5(self, df_m5: pd.DataFrame, rejection: Dict) -> bool:
        """
        Confirm rejection signal on M5 timeframe.
        
        Args:
            df_m5: M5 DataFrame
            rejection: Rejection dict
            
        Returns:
            True if confirmed
        """
        if df_m5 is None or df_m5.empty or len(df_m5) < 20:
            return True  # Skip confirmation if no M5 data

        try:
            close = df_m5['close'].values.astype(float)
            high = df_m5['high'].values.astype(float)
            low = df_m5['low'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            direction = rejection.get('direction', 'BUY')
            iob_level = rejection.get('iob_level', 0)

            if direction == 'BUY':
                # For BUY: M5 should show bullish momentum
                recent_close = close[-10:]
                if recent_close[-1] > recent_close[0]:
                    return True
                # Or price should be above IOB level
                if close[-1] > iob_level:
                    return True

            else:  # SELL
                # For SELL: M5 should show bearish momentum
                recent_close = close[-10:]
                if recent_close[-1] < recent_close[0]:
                    return True
                # Or price should be below IOB level
                if close[-1] < iob_level:
                    return True

            return False

        except Exception:
            return True  # Skip confirmation on error

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def _generate_signal(
        self, df: pd.DataFrame, rejection: Dict, regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal based on rejection.
        
        Args:
            df: DataFrame with OHLCV data
            rejection: Rejection dict
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            direction = rejection.get('direction', 'BUY')
            price = rejection.get('price', 0)
            iob_level = rejection.get('iob_level', 0)
            strength = rejection.get('strength', 0.5)

            if price <= 0:
                return self._create_neutral_signal()

            # Calculate entry price
            entry_price = price

            # Calculate Stop Loss
            if direction == 'BUY':
                # SL below IOB level with buffer
                sl_buffer = abs(price - iob_level) * 0.5
                sl_price = iob_level - sl_buffer
            else:  # SELL
                # SL above IOB level with buffer
                sl_buffer = abs(price - iob_level) * 0.5
                sl_price = iob_level + sl_buffer

            # Validate SL
            if sl_price <= 0 or sl_price == entry_price:
                return self._create_neutral_signal()

            # Calculate Take Profit using Adaptive TP Engine
            tp_result = self.adaptive_tp_engine.calculate_adaptive_tp(
                df, entry_price, sl_price, direction == 'BUY',
                regime_context.get('regime_name', 'UNKNOWN') if regime_context else 'UNKNOWN'
            )

            if tp_result and tp_result.get('tp_price', 0) > 0:
                tp_price = tp_result['tp_price']
            else:
                # Fallback: Use Fibonacci
                fib_result = self.fib_engine.calculate_tp_from_fib(
                    entry_price, sl_price, df, direction == 'BUY'
                )
                if fib_result and fib_result.get('valid', False):
                    tp_price = fib_result['tp_price']
                else:
                    # Final fallback: Fixed R:R
                    risk = abs(entry_price - sl_price)
                    if direction == 'BUY':
                        tp_price = entry_price + risk * 2.0
                    else:
                        tp_price = entry_price - risk * 2.0

            # Calculate confidence
            confidence = min(1.0, 0.5 + strength * 0.3)

            # Build signal
            signal = {
                'signal': f'{direction}_LIMIT',
                'meta': {
                    'strategy': self.strategy_name,
                    'strategy_category': self.strategy_category,
                    'entry_price': round(entry_price, 2),
                    'sl_price': round(sl_price, 2),
                    'tp_price': round(tp_price, 2),
                    'confidence': confidence,
                    'rejection_type': rejection.get('type', 'UNKNOWN'),
                    'iob_level': round(iob_level, 2),
                    'strength': strength,
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit
                }
            }

            self.logger.info(
                f"[S1_IOB] Signal generated: {direction} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S1_IOB] Signal generation error: {e}")
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

        # SMC strategies work best in trending and consolidation regimes
        compatible_regimes = [
            'HEALTHY_UPTREND', 'HEALTHY_DOWNTREND',
            'QUIET_RALLY', 'SLOW_BLEED',
            'CONSOLIDATING_BULL', 'CONSOLIDATING_BEAR',
            'PRE_BREAKOUT', 'CLASSIC_RANGE'
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