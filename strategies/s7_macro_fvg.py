"""
S7_MacroFVG - Macro Fair Value Gap Strategy.

Smart Money Concepts strategy that identifies Fair Value Gaps (FVG)
on macro timeframes and trades at these high-probability zones.

Strategy Logic:
  1. Detect Fair Value Gaps on M15 and H1 timeframes
  2. Identify unfilled FVGs (price hasn't returned to fill the gap)
  3. Find entry zones at FVG levels
  4. Generate entry signal when price approaches FVG

Fair Value Gap Definition:
  A Fair Value Gap is a three-candle pattern where there's an imbalance
  between buying and selling pressure, creating a "gap" in the price action.
  
  Bullish FVG: Candle 1 high < Candle 3 low (gap up)
  Bearish FVG: Candle 1 low > Candle 3 high (gap down)

Used Engines:
  - SMCStructuralEngine: FVG detection
  - AdaptiveTPEngine: Dynamic TP calculation

Strategy Category: SMC
Timeframe: M15 (primary), H1 (macro confirmation)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from core.base_strategy import BaseStrategy
from core.smc_engine import SMCStructuralEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S7_MacroFVG(BaseStrategy):
    """
    Macro Fair Value Gap Strategy.
    
    This strategy identifies Fair Value Gaps on macro timeframes
    and trades at these high-probability zones.
    
    Fair Value Gap Definition:
      A three-candle pattern where the middle candle creates an
      imbalance between candle 1 and candle 3, leaving a gap
      that price tends to fill.
      
    Macro Analysis:
      By analyzing FVGs on higher timeframes (H1), we identify
      institutional interest zones that have higher probability
      of being respected.
      
    Entry Criteria:
      - Unfilled FVG detected on M15 or H1
      - Price approaching FVG zone
      - Confirmation from M5 timeframe
      - Volume confirmation
    """

    def __init__(self):
        """Initialize S7_MacroFVG strategy."""
        super().__init__(
            strategy_name='S7_MacroFVG',
            strategy_category='SMC',
            timeframes=['M15', 'H1', 'M5'],
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
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.fvg_lookback = 50  # Lookback for FVG detection
        self.min_fvg_size_pct = 0.1  # Minimum FVG size (% of price)
        self.entry_tolerance_pct = 0.5  # Entry tolerance (% of price)

    # =========================================================================
    # MAIN ANALYSIS METHOD
    # =========================================================================

    def analyze(
        self,
        df_m15: pd.DataFrame,
        df_h1: pd.DataFrame = None,
        df_m5: pd.DataFrame = None,
        regime_context: Dict = None
    ) -> Dict:
        """
        Main analysis method for S7_MacroFVG.
        
        Args:
            df_m15: M15 DataFrame
            df_h1: H1 DataFrame (optional, for macro confirmation)
            df_m5: M5 DataFrame (optional, for entry confirmation)
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
            # STEP 1: Detect FVGs on M15
            # =========================================================================
            m15_fvgs = self.smc_engine.detect_fvg(df_m15, lookback=self.fvg_lookback)

            # =========================================================================
            # STEP 2: Detect FVGs on H1 (if available)
            # =========================================================================
            h1_fvgs = []
            if df_h1 is not None and not df_h1.empty and len(df_h1) >= 30:
                h1_fvgs = self.smc_engine.detect_fvg(df_h1, lookback=30)

            # Combine FVGs (H1 FVGs have higher priority)
            all_fvgs = self._combine_fvgs(m15_fvgs, h1_fvgs)

            if not all_fvgs:
                return default_signal

            # =========================================================================
            # STEP 3: Filter Unfilled FVGs
            # =========================================================================
            unfilled_fvgs = [fvg for fvg in all_fvgs if not fvg.get('filled', False)]

            if not unfilled_fvgs:
                return default_signal

            # =========================================================================
            # STEP 4: Find Entry Zone
            # =========================================================================
            entry_zone = self._find_entry_zone(df_m15, unfilled_fvgs)

            if entry_zone is None:
                return default_signal

            # =========================================================================
            # STEP 5: M5 Confirmation
            # =========================================================================
            if df_m5 is not None and not df_m5.empty:
                if not self._confirm_m5(df_m5, entry_zone):
                    return default_signal

            # =========================================================================
            # STEP 6: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, entry_zone, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S7_FVG] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # FVG COMBINATION
    # =========================================================================

    def _combine_fvgs(self, m15_fvgs: List[Dict], h1_fvgs: List[Dict]) -> List[Dict]:
        """
        Combine FVGs from multiple timeframes.
        
        H1 FVGs have higher priority and weight.
        
        Args:
            m15_fvgs: FVGs from M15 timeframe
            h1_fvgs: FVGs from H1 timeframe
            
        Returns:
            Combined FVG list with priority
        """
        combined = []

        # Add H1 FVGs with higher priority
        for fvg in h1_fvgs:
            fvg_copy = fvg.copy()
            fvg_copy['timeframe'] = 'H1'
            fvg_copy['priority'] = 2  # Higher priority
            fvg_copy['weight'] = 1.5  # Higher weight
            combined.append(fvg_copy)

        # Add M15 FVGs with normal priority
        for fvg in m15_fvgs:
            fvg_copy = fvg.copy()
            fvg_copy['timeframe'] = 'M15'
            fvg_copy['priority'] = 1  # Normal priority
            fvg_copy['weight'] = 1.0  # Normal weight
            combined.append(fvg_copy)

        # Sort by priority and gap size
        combined.sort(key=lambda x: (x.get('priority', 1), x.get('gap_pct', 0)), reverse=True)

        return combined

    # =========================================================================
    # ENTRY ZONE DETECTION
    # =========================================================================

    def _find_entry_zone(self, df: pd.DataFrame, fvgs: List[Dict]) -> Optional[Dict]:
        """
        Find entry zone at FVG levels.
        
        Args:
            df: DataFrame with OHLCV data
            fvgs: List of FVGs
            
        Returns:
            Entry zone dict or None
        """
        if not fvgs:
            return None

        try:
            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))
            current_price = close[-1]

            best_zone = None
            best_score = 0.0

            for fvg in fvgs:
                fvg_type = fvg.get('type', 'UNKNOWN')
                fvg_top = fvg.get('top', 0)
                fvg_bottom = fvg.get('bottom', 0)
                fvg_mid = (fvg_top + fvg_bottom) / 2

                # Check if price is approaching FVG
                distance_to_fvg = abs(current_price - fvg_mid) / current_price * 100

                if distance_to_fvg > self.entry_tolerance_pct:
                    continue  # Price too far from FVG

                # Calculate entry zone score
                score = self._calculate_zone_score(fvg, current_price)

                if score > best_score:
                    best_score = score

                    # Determine direction based on FVG type
                    if fvg_type == 'BULLISH':
                        # Bullish FVG: Price approaches from above, BUY at FVG
                        direction = 'BUY'
                        entry_price = fvg_top  # Enter at top of bullish FVG
                    else:  # BEARISH
                        # Bearish FVG: Price approaches from below, SELL at FVG
                        direction = 'SELL'
                        entry_price = fvg_bottom  # Enter at bottom of bearish FVG

                    best_zone = {
                        'fvg': fvg,
                        'direction': direction,
                        'entry_price': entry_price,
                        'fvg_top': fvg_top,
                        'fvg_bottom': fvg_bottom,
                        'fvg_mid': fvg_mid,
                        'score': score,
                        'distance_pct': distance_to_fvg
                    }

            return best_zone

        except Exception as e:
            self.logger.debug(f"[S7_FVG] Entry zone error: {e}")
            return None

    def _calculate_zone_score(self, fvg: Dict, current_price: float) -> float:
        """Calculate entry zone score."""
        try:
            score = 0.0

            # FVG size bonus (larger FVG = more significant)
            gap_pct = fvg.get('gap_pct', 0)
            if gap_pct > 0.3:
                score += 0.3
            elif gap_pct > 0.15:
                score += 0.2
            else:
                score += 0.1

            # Timeframe priority bonus
            priority = fvg.get('priority', 1)
            score += priority * 0.1

            # Freshness bonus (unfilled FVGs are more significant)
            if not fvg.get('filled', False):
                score += 0.2

            # Price proximity bonus
            fvg_mid = (fvg.get('top', 0) + fvg.get('bottom', 0)) / 2
            distance = abs(current_price - fvg_mid) / current_price

            if distance < 0.002:  # Within 0.2%
                score += 0.2
            elif distance < 0.005:  # Within 0.5%
                score += 0.1

            return min(1.0, score)

        except Exception:
            return 0.5

    # =========================================================================
    # M5 CONFIRMATION
    # =========================================================================

    def _confirm_m5(self, df_m5: pd.DataFrame, entry_zone: Dict) -> bool:
        """
        Confirm signal on M5 timeframe.
        
        Args:
            df_m5: M5 DataFrame
            entry_zone: Entry zone dict
            
        Returns:
            True if confirmed
        """
        if df_m5 is None or df_m5.empty or len(df_m5) < 20:
            return True  # Skip confirmation if no M5 data

        try:
            close = df_m5['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            direction = entry_zone.get('direction', 'BUY')

            # Check M5 momentum aligns with entry direction
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
        self, df: pd.DataFrame, entry_zone: Dict, regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            entry_zone: Entry zone dict
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            direction = entry_zone.get('direction', 'BUY')
            entry_price = entry_zone.get('entry_price', 0)
            fvg_top = entry_zone.get('fvg_top', 0)
            fvg_bottom = entry_zone.get('fvg_bottom', 0)

            if entry_price <= 0:
                return self._create_neutral_signal()

            # Calculate Stop Loss
            if direction == 'BUY':
                # SL below FVG bottom with buffer
                sl_buffer = abs(fvg_top - fvg_bottom) * 0.3
                sl_price = fvg_bottom - sl_buffer
            else:  # SELL
                # SL above FVG top with buffer
                sl_buffer = abs(fvg_top - fvg_bottom) * 0.3
                sl_price = fvg_top + sl_buffer

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
            score = entry_zone.get('score', 0.5)
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
                    'fvg_type': entry_zone.get('fvg', {}).get('type', 'UNKNOWN'),
                    'fvg_timeframe': entry_zone.get('fvg', {}).get('timeframe', 'M15'),
                    'fvg_gap_pct': entry_zone.get('fvg', {}).get('gap_pct', 0),
                    'zone_score': score,
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit
                }
            }

            self.logger.info(
                f"[S7_FVG] Signal generated: {direction} | "
                f"FVG Type: {entry_zone.get('fvg', {}).get('type')} | "
                f"Timeframe: {entry_zone.get('fvg', {}).get('timeframe')} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S7_FVG] Signal generation error: {e}")
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
            'PRE_BREAKOUT', 'CLASSIC_RANGE',
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