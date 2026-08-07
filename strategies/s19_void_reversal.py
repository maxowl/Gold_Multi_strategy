"""
S19_VoidReversal - Liquidity Void Reversal Strategy.

Scalping strategy that trades reversals when price fills liquidity voids.
Liquidity voids are gaps where price moved quickly without trading activity,
and price tends to return to fill these voids before reversing.

Strategy Logic:
  1. Detect liquidity voids (gaps in price action)
  2. Monitor void fill progress
  3. Detect reversal signals at void completion
  4. Generate entry signal on reversal confirmation

Liquidity Void Definition:
  A liquidity void is a gap where price moved quickly with low volume,
  leaving an "empty" zone that price tends to fill.
  
  Bullish Void: Gap up (price jumped up)
  Bearish Void: Gap down (price dropped down)
  
  Price tends to return and fill these voids before continuing.

Reversal Logic:
  When price fills a void and shows reversal signs, it indicates
  that the void fill is complete and price may reverse.

Used Engines:
  - VoidStructuralEngine: Void detection and fill tracking
  - AdaptiveTPEngine: Dynamic TP calculation

Strategy Category: SCALP
Timeframe: M15 (primary), M5 (confirmation)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from core.base_strategy import BaseStrategy
from core.void_structural_engine import VoidStructuralEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S19_VoidReversal(BaseStrategy):
    """
    Liquidity Void Reversal Strategy.
    
    This strategy trades reversals when price fills liquidity voids.
    
    Liquidity Void Definition:
      A gap where price moved quickly with low volume, creating
      an "empty" zone that price tends to fill before reversing.
      
    Void Fill Process:
      1. Void created (gap up or down)
      2. Price returns to fill the void
      3. Void fill completes
      4. Price reverses
      
    Entry Criteria:
      - Liquidity void detected
      - Void is being filled (>50%)
      - Reversal signs detected
      - Confirmation from M5 timeframe
    """

    def __init__(self):
        """Initialize S19_VoidReversal strategy."""
        super().__init__(
            strategy_name='S19_VoidReversal',
            strategy_category='SCALP',
            timeframes=['M15', 'M5'],
            risk_per_trade_pct=0.3,
            min_rr_ratio=1.5,
            max_spread_points=25,
            trailing_enabled=True,
            partial_close_enabled=False,
            requires_dynamic_exit=True,
            friction_sensitive=True
        )

        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize engines
        self.void_engine = VoidStructuralEngine()
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.void_lookback = 50  # Lookback for void detection
        self.min_fill_percentage = 0.5  # Minimum void fill (50%)
        self.max_fill_percentage = 0.9  # Maximum void fill (90%)
        self.min_void_size_atr = 1.0  # Minimum void size in ATR

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
        Main analysis method for S19_VoidReversal.
        
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
            # STEP 1: Detect Liquidity Voids
            # =========================================================================
            voids = self.void_engine.detect_liquidity_void(df_m15, lookback=self.void_lookback)

            if not voids:
                return default_signal

            # =========================================================================
            # STEP 2: Check Void Fill Status
            # =========================================================================
            voids = self.void_engine.check_void_fill(df_m15, voids)

            # Filter voids that are being filled
            filling_voids = [
                v for v in voids
                if self.min_fill_percentage <= v.get('fill_percentage', 0) <= self.max_fill_percentage
            ]

            if not filling_voids:
                return default_signal

            # =========================================================================
            # STEP 3: Detect Reversal
            # =========================================================================
            reversal = self._detect_reversal(df_m15, filling_voids)

            if reversal is None or not reversal.get('reversal_detected', False):
                return default_signal

            # =========================================================================
            # STEP 4: M5 Confirmation
            # =========================================================================
            if df_m5 is not None and not df_m5.empty:
                if not self._confirm_m5(df_m5, reversal):
                    return default_signal

            # =========================================================================
            # STEP 5: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, reversal, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S19_VOID] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # REVERSAL DETECTION
    # =========================================================================

    def _detect_reversal(self, df: pd.DataFrame, voids: List[Dict]) -> Optional[Dict]:
        """
        Detect reversal at void fill completion.
        
        Args:
            df: DataFrame with OHLCV data
            voids: List of voids being filled
            
        Returns:
            Reversal dict or None
        """
        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            current_price = close[-1]

            # Find best void for reversal
            best_void = None
            best_score = 0.0

            for void in voids:
                void_type = void.get('type', 'UNKNOWN')
                fill_percentage = void.get('fill_percentage', 0)
                void_top = void.get('gap_top', 0)
                void_bottom = void.get('gap_bottom', 0)

                # Calculate score based on fill percentage and proximity
                fill_score = fill_percentage
                proximity_score = self._calculate_proximity(current_price, void_top, void_bottom)
                total_score = fill_score * 0.6 + proximity_score * 0.4

                if total_score > best_score:
                    best_score = total_score
                    best_void = void

            if best_void is None:
                return None

            void_type = best_void.get('type', 'UNKNOWN')
            void_top = best_void.get('gap_top', 0)
            void_bottom = best_void.get('gap_bottom', 0)

            # Determine reversal direction
            if void_type == 'BULLISH_VOID':
                # Bullish void (gap up) filled → expect reversal down → SELL
                direction = 'SELL'
                reversal_level = void_top
            else:  # BEARISH_VOID
                # Bearish void (gap down) filled → expect reversal up → BUY
                direction = 'BUY'
                reversal_level = void_bottom

            # Check for reversal signs
            reversal_confirmed = self._check_reversal_signs(df, direction)

            if not reversal_confirmed:
                return None

            return {
                'reversal_detected': True,
                'direction': direction,
                'void_type': void_type,
                'void_top': float(void_top),
                'void_bottom': float(void_bottom),
                'fill_percentage': float(best_void.get('fill_percentage', 0)),
                'reversal_level': float(reversal_level),
                'score': float(best_score)
            }

        except Exception as e:
            self.logger.debug(f"[S19_VOID] Reversal detection error: {e}")
            return None

    def _calculate_proximity(self, price: float, void_top: float, void_bottom: float) -> float:
        """Calculate how close price is to void."""
        try:
            void_mid = (void_top + void_bottom) / 2
            distance = abs(price - void_mid) / price
            return max(0, 1.0 - distance * 100)  # Closer = higher score
        except Exception:
            return 0.0

    def _check_reversal_signs(self, df: pd.DataFrame, direction: str) -> bool:
        """Check for reversal signs in recent bars."""
        try:
            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            # Check recent momentum
            recent_close = close[-5:]
            momentum = recent_close[-1] - recent_close[0]

            if direction == 'BUY':
                # For BUY: Price should be rising
                return momentum > 0
            else:  # SELL
                # For SELL: Price should be falling
                return momentum < 0

        except Exception:
            return False

    # =========================================================================
    # M5 CONFIRMATION
    # =========================================================================

    def _confirm_m5(self, df_m5: pd.DataFrame, reversal: Dict) -> bool:
        """
        Confirm signal on M5 timeframe.
        
        Args:
            df_m5: M5 DataFrame
            reversal: Reversal dict
            
        Returns:
            True if confirmed
        """
        if df_m5 is None or df_m5.empty or len(df_m5) < 20:
            return True  # Skip confirmation if no M5 data

        try:
            close = df_m5['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            direction = reversal.get('direction', 'BUY')

            # Check M5 momentum aligns with reversal direction
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
        self, df: pd.DataFrame, reversal: Dict, regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            reversal: Reversal dict
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            direction = reversal.get('direction', 'BUY')
            reversal_level = reversal.get('reversal_level', 0)

            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            entry_price = close[-1]

            if entry_price <= 0 or reversal_level <= 0:
                return self._create_neutral_signal()

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
            fill_bonus = reversal.get('fill_percentage', 0.5) * 0.2
            score_bonus = reversal.get('score', 0.5) * 0.2
            confidence = min(1.0, 0.4 + fill_bonus + score_bonus)

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
                    'void_type': reversal.get('void_type', 'UNKNOWN'),
                    'fill_percentage': reversal.get('fill_percentage', 0),
                    'reversal_level': round(reversal_level, 2),
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit,
                    'friction_sensitive': self.friction_sensitive
                }
            }

            self.logger.info(
                f"[S19_VOID] Signal generated: {direction} | "
                f"Void: {reversal.get('void_type')} | "
                f"Fill: {reversal.get('fill_percentage', 0):.1%} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S19_VOID] Signal generation error: {e}")
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

        # SCALP strategies work best in choppy and volatile regimes
        compatible_regimes = [
            'VOLATILE_CHOP', 'WHIPSAW_MARKET',
            'CLASSIC_RANGE', 'TIGHT_RANGE',
            'OVERSOLD_BOUNCE', 'EXHAUSTED_BULL', 'EXHAUSTED_BEAR',
            'PRE_BREAKOUT'
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