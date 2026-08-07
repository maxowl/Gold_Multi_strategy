"""
S5_Breaker_Void - Breaker Block + Liquidity Void Strategy.

Smart Money Concepts strategy that combines Breaker Blocks with
Liquidity Voids for high-probability reversal entries.

Strategy Logic:
  1. Detect Breaker Blocks (failed Order Blocks)
  2. Detect Liquidity Voids (price gaps)
  3. Find confluence zones where Breaker + Void overlap
  4. Generate entry signal at confluence

Smart Money Concepts:
  Breaker Block:
    An Order Block that was broken and now acts as opposite S/R.
    - Broken Bullish OB → Acts as Resistance
    - Broken Bearish OB → Acts as Support
    
  Liquidity Void:
    A gap where price moved quickly without trading activity.
    Price tends to return to fill these voids.

Used Engines:
  - SMCStructuralEngine: Order Block and Breaker detection
  - VoidStructuralEngine: Liquidity void detection
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
from core.void_structural_engine import VoidStructuralEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S5_Breaker_Void(BaseStrategy):
    """
    Breaker Block + Liquidity Void Strategy.
    
    This strategy combines Breaker Blocks with Liquidity Voids
    for high-probability reversal entries.
    
    Breaker Block Definition:
      A Breaker is an Order Block that was broken and now acts
      as the opposite type of support/resistance.
      
    Liquidity Void Definition:
      A gap where price moved quickly without sufficient trading
      volume, leaving an "empty" zone that price tends to fill.
      
    Entry Criteria:
      - Breaker Block detected
      - Liquidity Void nearby
      - Price approaching confluence zone
      - Confirmation from M5 timeframe
    """

    def __init__(self):
        """Initialize S5_Breaker_Void strategy."""
        super().__init__(
            strategy_name='S5_Breaker_Void',
            strategy_category='SMC',
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
        self.smc_engine = SMCStructuralEngine()
        self.void_engine = VoidStructuralEngine()
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.breaker_lookback = 100  # Lookback for breaker detection
        self.void_lookback = 50  # Lookback for void detection
        self.confluence_tolerance = 0.003  # Confluence tolerance (0.3%)

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
        Main analysis method for S5_Breaker_Void.
        
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
            # STEP 1: Detect Breaker Blocks
            # =========================================================================
            breakers = self.smc_engine.detect_breaker_blocks(df_m15, lookback=self.breaker_lookback)

            if not breakers:
                return default_signal

            # =========================================================================
            # STEP 2: Detect Liquidity Voids
            # =========================================================================
            voids = self.void_engine.detect_liquidity_void(df_m15, lookback=self.void_lookback)

            if not voids:
                return default_signal

            # =========================================================================
            # STEP 3: Find Confluence Zones
            # =========================================================================
            confluence = self._find_confluence(breakers, voids, df_m15)

            if confluence is None:
                return default_signal

            # =========================================================================
            # STEP 4: Check Price Approaching Zone
            # =========================================================================
            if not self._check_price_approaching(df_m15, confluence):
                return default_signal

            # =========================================================================
            # STEP 5: M5 Confirmation
            # =========================================================================
            if df_m5 is not None and not df_m5.empty:
                if not self._confirm_m5(df_m5, confluence):
                    return default_signal

            # =========================================================================
            # STEP 6: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, confluence, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S5_BV] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # CONFLUENCE DETECTION
    # =========================================================================

    def _find_confluence(
        self, breakers: List[Dict], voids: List[Dict], df: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Find confluence zones where Breaker and Void overlap.
        
        Args:
            breakers: List of Breaker Blocks
            voids: List of Liquidity Voids
            df: DataFrame with OHLCV data
            
        Returns:
            Confluence dict or None
        """
        if not breakers or not voids:
            return None

        try:
            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))
            current_price = close[-1]

            best_confluence = None
            best_score = 0.0

            # Check each breaker against each void
            for breaker in breakers:
                breaker_type = breaker.get('type', 'UNKNOWN')
                breaker_high = breaker.get('high', 0)
                breaker_low = breaker.get('low', 0)
                breaker_quality = breaker.get('quality', 50)

                for void in voids:
                    void_type = void.get('type', 'UNKNOWN')
                    void_top = void.get('gap_top', 0)
                    void_bottom = void.get('gap_bottom', 0)

                    # Check if breaker and void overlap
                    overlap = self._check_overlap(
                        breaker_high, breaker_low, void_top, void_bottom
                    )

                    if overlap:
                        # Calculate confluence score
                        score = self._calculate_confluence_score(
                            breaker_quality, void_type, breaker_type, current_price,
                            void_top, void_bottom
                        )

                        if score > best_score:
                            best_score = score

                            # Determine direction based on types
                            direction = self._determine_direction(breaker_type, void_type)

                            best_confluence = {
                                'breaker': breaker,
                                'void': void,
                                'direction': direction,
                                'zone_high': max(breaker_high, void_top),
                                'zone_low': min(breaker_low, void_bottom),
                                'score': score,
                                'overlap_pct': overlap
                            }

            return best_confluence

        except Exception as e:
            self.logger.debug(f"[S5_BV] Confluence detection error: {e}")
            return None

    def _check_overlap(
        self, breaker_high: float, breaker_low: float,
        void_top: float, void_bottom: float
    ) -> Optional[float]:
        """
        Check if breaker and void overlap.
        
        Returns:
            Overlap percentage or None if no overlap
        """
        try:
            # Calculate overlap
            overlap_top = min(breaker_high, void_top)
            overlap_bottom = max(breaker_low, void_bottom)

            if overlap_top > overlap_bottom:
                overlap_range = overlap_top - overlap_bottom
                total_range = max(breaker_high, void_top) - min(breaker_low, void_bottom)

                if total_range > 0:
                    return overlap_range / total_range

            return None

        except Exception:
            return None

    def _calculate_confluence_score(
        self, breaker_quality: float, void_type: str, breaker_type: str,
        current_price: float, void_top: float, void_bottom: float
    ) -> float:
        """Calculate confluence zone score."""
        try:
            score = breaker_quality / 100 * 0.4  # 40% weight for breaker quality

            # Void freshness bonus
            if void_type in ['BULLISH_VOID', 'BEARISH_VOID']:
                score += 0.3  # 30% weight for void type
            else:
                score += 0.15

            # Breaker type bonus
            if 'BREAKER' in breaker_type:
                score += 0.2  # 20% weight for breaker type
            else:
                score += 0.1

            # Price proximity bonus (10%)
            zone_mid = (void_top + void_bottom) / 2
            distance = abs(current_price - zone_mid) / current_price

            if distance < 0.005:  # Within 0.5%
                score += 0.1
            elif distance < 0.01:  # Within 1%
                score += 0.05

            return min(1.0, score)

        except Exception:
            return 0.5

    def _determine_direction(self, breaker_type: str, void_type: str) -> str:
        """Determine trading direction based on breaker and void types."""
        # Breaker type determines direction
        if 'BULLISH_BREAKER' in breaker_type:
            # Broken bearish OB → acts as support → BUY
            return 'BUY'
        elif 'BEARISH_BREAKER' in breaker_type:
            # Broken bullish OB → acts as resistance → SELL
            return 'SELL'
        else:
            # Use void type as fallback
            if 'BULLISH' in void_type:
                return 'BUY'
            else:
                return 'SELL'

    # =========================================================================
    # PRICE APPROACH CHECK
    # =========================================================================

    def _check_price_approaching(self, df: pd.DataFrame, confluence: Dict) -> bool:
        """
        Check if price is approaching the confluence zone.
        
        Args:
            df: DataFrame with OHLCV data
            confluence: Confluence dict
            
        Returns:
            True if price is approaching
        """
        try:
            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))
            current_price = close[-1]

            zone_high = confluence.get('zone_high', 0)
            zone_low = confluence.get('zone_low', 0)
            zone_mid = (zone_high + zone_low) / 2

            # Calculate distance to zone
            distance = abs(current_price - zone_mid) / current_price

            # Price should be within 1% of zone
            return distance < 0.01

        except Exception:
            return False

    # =========================================================================
    # M5 CONFIRMATION
    # =========================================================================

    def _confirm_m5(self, df_m5: pd.DataFrame, confluence: Dict) -> bool:
        """
        Confirm signal on M5 timeframe.
        
        Args:
            df_m5: M5 DataFrame
            confluence: Confluence dict
            
        Returns:
            True if confirmed
        """
        if df_m5 is None or df_m5.empty or len(df_m5) < 20:
            return True  # Skip confirmation if no M5 data

        try:
            close = df_m5['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            direction = confluence.get('direction', 'BUY')

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
        self, df: pd.DataFrame, confluence: Dict, regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            confluence: Confluence dict
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            direction = confluence.get('direction', 'BUY')
            zone_high = confluence.get('zone_high', 0)
            zone_low = confluence.get('zone_low', 0)

            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))
            entry_price = close[-1]

            # Calculate Stop Loss
            if direction == 'BUY':
                # SL below zone low
                sl_buffer = abs(entry_price - zone_low) * 0.2
                sl_price = zone_low - sl_buffer
            else:  # SELL
                # SL above zone high
                sl_buffer = abs(zone_high - entry_price) * 0.2
                sl_price = zone_high + sl_buffer

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
            score = confluence.get('score', 0.5)
            confidence = min(1.0, 0.3 + score * 0.7)

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
                    'breaker_type': confluence.get('breaker', {}).get('type', 'UNKNOWN'),
                    'void_type': confluence.get('void', {}).get('type', 'UNKNOWN'),
                    'confluence_score': score,
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit
                }
            }

            self.logger.info(
                f"[S5_BV] Signal generated: {direction} | "
                f"Breaker: {confluence.get('breaker', {}).get('type')} | "
                f"Void: {confluence.get('void', {}).get('type')} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S5_BV] Signal generation error: {e}")
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

        # SMC reversal strategies work best at exhaustion points
        compatible_regimes = [
            'EXHAUSTED_BULL', 'EXHAUSTED_BEAR',
            'OVERSOLD_BOUNCE', 'ANOMALY_BULL', 'ANOMALY_BEAR',
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